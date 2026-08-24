"""Metacognitive awareness — deterministic DETECT → RECORD (minimal version).

The "Task 1" slice of the agent-awareness design docs
(``hermes-agent-docs/plans/memory/agent-awareness-*.md``): the metacognitive
cycle reduced to the two steps that can run with zero LLM calls.

DETECT  Reuses the tool-call guardrail controller's warn/halt decisions
        (``agent.tool_guardrails``) — repeated identical failures, per-tool
        failure streaks, no-progress results — plus two memory-side feeds
        observed at the same choke point (``run_agent.py::
        _append_guardrail_observation``): "No entry matched" errors from the
        ``memory`` tool and streaks of empty ``memory read`` results against
        a non-empty store.
RECALL  When a stuck episode opens, one FTS query against the built-in
        memory store looks for a prior ``pattern`` entry with the same stuck
        signature; if found, a compact past-experience note is appended to
        the current tool result. Same mechanism as guardrail guidance — new
        tokens in a *new* tool-result message, never a mutation of past
        context, so the per-conversation prompt cache stays valid.
RECORD  At turn end, an episode that got an honest outcome is written back
        to the store as a ``pattern`` entry: "stuck pattern → what helped".
        Interrupted/erroring turns record nothing — experience without a
        verified outcome is not growth (design doc §2).

Everything here is counting, FTS and templates. No model calls, no new
model tools, no system-prompt changes — which is also why ``/awareness
off`` takes effect mid-session without touching the prompt cache. When
``memory.write_approval`` is on, RECORD is skipped entirely: the user
asked to approve memory writes, and that gate is respected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from utils import safe_json_loads

logger = logging.getLogger(__name__)

# Modes: "auto" = detect + recall-notes + record. "off" = fully inert.
# "deep" is a planned later phase and is rejected explicitly by set_mode.
MODES = ("off", "auto")

# Arg keys considered safe to quote in a recovery brief (paths, commands,
# queries). Everything else — file *contents*, code bodies — is never put
# into a memory record by this module.
_RECOVERY_ARG_KEYS: dict[str, tuple[str, ...]] = {
    "terminal": ("command",),
    "read_file": ("path", "file_path"),
    "write_file": ("path", "file_path"),
    "patch": ("path", "file_path"),
    "search_files": ("query", "path"),
    "web_search": ("query",),
    "web_extract": ("url",),
    "browser_navigate": ("url",),
}

_NOTE_SNIPPET_LIMIT = 240
_ERROR_SNIPPET_LIMIT = 140
_MAX_RECOVERY_ACTIONS = 3


def normalize_mode(value: Any) -> str:
    """Normalize an ``awareness.mode`` config value to "off" | "auto".

    YAML 1.1 parses bare ``off``/``on`` as booleans, so a hand-edited
    ``awareness.mode: off`` arrives here as ``False`` — without this
    normalization the controller would silently fall back to "auto" and
    ignore the user's switch-off.
    """
    if value is False:
        return "off"
    if value is True:
        return "auto"
    text = str(value or "auto").strip().lower()
    if text in ("false", "no", "0", "disabled"):
        return "off"
    if text in ("true", "yes", "1", "enabled"):
        return "auto"
    if text == "on":
        return "auto"
    return text if text in MODES else "auto"


@dataclass
class StuckEpisode:
    """One detected stuck pattern within a turn."""

    signature: str
    tool_name: str
    trigger: str  # guardrail decision code or awareness-specific code
    count: int
    error_snippet: str
    recovery_actions: list[str] = field(default_factory=list)
    note_injected: bool = False


def _error_snippet(result: Optional[str]) -> str:
    """First line / JSON error field of a tool result, whitespace-collapsed."""
    text = ""
    parsed = safe_json_loads(result or "")
    if isinstance(parsed, dict):
        text = str(parsed.get("error") or parsed.get("message") or "")
    if not text:
        stripped = (result or "").strip()
        text = stripped.splitlines()[0] if stripped else ""
    text = " ".join(text.split())
    return text[:_ERROR_SNIPPET_LIMIT]


def _recovery_brief(tool_name: str, args: Optional[Mapping[str, Any]]) -> str:
    """Safe one-line description of a successful call ("what helped").

    Quotes only whitelisted argument kinds (command/path/query/url); other
    tools degrade to the bare tool name so file contents never leak into
    memory records through this path.
    """
    brief = tool_name
    if isinstance(args, Mapping):
        for key in _RECOVERY_ARG_KEYS.get(tool_name, ()):
            val = args.get(key)
            if val:
                first_line = str(val).strip().splitlines()[0] if str(val).strip() else ""
                if first_line:
                    brief = f"{tool_name}({first_line[:80]})"
                    break
    return brief


class AwarenessController:
    """Deterministic metacognitive DETECT → RECORD over the memory store.

    One instance per agent (per session). Fed once per executed tool call
    from the single choke point both executor paths share; closed once per
    turn from ``finalize_turn``. All public methods are fail-safe by
    contract at the call sites and never raise into the agent loop.
    """

    def __init__(
        self,
        store: Any,
        config: Optional[Mapping[str, Any]] = None,
        *,
        session_id: Optional[str] = None,
        write_approval: bool = False,
    ) -> None:
        cfg = config if isinstance(config, Mapping) else {}
        self.mode = normalize_mode(cfg.get("mode", "auto"))
        self.record_enabled = bool(cfg.get("record", True))
        self.note_enabled = bool(cfg.get("note_on_detect", True))
        try:
            self.empty_streak_limit = max(2, int(cfg.get("empty_recall_streak", 3) or 3))
        except (TypeError, ValueError):
            self.empty_streak_limit = 3
        self.write_approval = bool(write_approval)
        self.session_id = session_id
        self._store = store

        # Per-turn state (reset_for_turn / finalize_turn).
        self._episodes: list[StuckEpisode] = []
        self._open_episode: Optional[StuckEpisode] = None
        self._empty_recall_streak = 0

        # Session-lifetime stats for /awareness status.
        self.episodes_detected = 0
        self.notes_injected = 0
        self.patterns_recorded = 0
        self.empty_streaks_hit = 0
        self.records_skipped = 0
        self._recorded_signatures: set[str] = set()

    # ------------------------------------------------------------------
    # Turn lifecycle
    # ------------------------------------------------------------------

    def reset_for_turn(self) -> None:
        """Drop per-turn state; runs beside the guardrail reset each turn."""
        self._episodes = []
        self._open_episode = None
        self._empty_recall_streak = 0

    def finalize_turn(self, *, exit_reason: str) -> Optional[dict[str, Any]]:
        """Close the turn: RECORD honest outcomes for detected episodes.

        Returns the last record's metadata (or ``None``). Never raises.
        """
        recorded: Optional[dict[str, Any]] = None
        try:
            if self.enabled:
                for episode in self._episodes:
                    recorded = self._record_episode(episode, exit_reason=exit_reason) or recorded
        except Exception:
            logger.debug("awareness: finalize_turn failed", exc_info=True)
        finally:
            self.reset_for_turn()
        return recorded

    # ------------------------------------------------------------------
    # DETECT feeds
    # ------------------------------------------------------------------

    def observe_tool_result(
        self,
        tool_name: str,
        args: Optional[Mapping[str, Any]],
        result: Optional[str],
        *,
        failed: bool,
        guardrail_decision: Any = None,
    ) -> Optional[str]:
        """Feed one executed tool call; return a note to append, if any.

        ``guardrail_decision`` is the guardrail controller's decision for
        this call when it signalled warn/halt (``None`` for allow).
        """
        if not self.enabled:
            return None
        note: Optional[str] = None
        try:
            triggered = False
            if guardrail_decision is not None and getattr(guardrail_decision, "action", "") == "warn":
                triggered = self._open_or_update_episode(guardrail_decision, result)
                note = self._maybe_build_note()
            if tool_name == "memory":
                note = self._observe_memory_read(args, result) or note
            if not failed and not triggered and self._open_episode is not None:
                self._record_recovery(tool_name, args)
        except Exception:
            logger.debug("awareness: observe_tool_result failed", exc_info=True)
        return note

    def observe_block(self, decision: Any) -> None:
        """Feed a guardrail *block* (before_call: loop caps / hard stop).

        The blocked call never executes and the turn is about to halt, so
        there is no note to inject — only the episode for the RECORD path.
        """
        if not self.enabled:
            return
        try:
            self._open_or_update_episode(decision, None)
        except Exception:
            logger.debug("awareness: observe_block failed", exc_info=True)

    # ------------------------------------------------------------------
    # Mode / status surface (for /awareness)
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def set_mode(self, mode: str) -> bool:
        """Switch mode live. Returns True for a known mode.

        Strict on purpose: command inputs are strings, and unknown values
        (e.g. "deep") must be rejected, not coerced to a default.
        """
        text = str(mode or "").strip().lower()
        if text == "on":
            text = "auto"
        if text not in MODES:
            return False
        self.mode = text
        return True

    def status_summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "record": self.record_enabled and not self.write_approval,
            "note_on_detect": self.note_enabled,
            "episodes_detected": self.episodes_detected,
            "notes_injected": self.notes_injected,
            "patterns_recorded": self.patterns_recorded,
            "empty_streaks_hit": self.empty_streaks_hit,
            "records_skipped": self.records_skipped,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _open_or_update_episode(self, decision: Any, result: Optional[str]) -> bool:
        """Open (or refresh) the episode for a warn/halt/block decision.

        Returns True when *this* decision belongs to the active episode —
        the caller uses it to avoid recording the triggering call itself
        as its own recovery.
        """
        signature = self._episode_signature(decision, result)
        episode = self._open_episode
        if episode is not None and episode.signature == signature:
            episode.count = max(episode.count, int(getattr(decision, "count", 0) or 0))
            return True
        episode = StuckEpisode(
            signature=signature,
            tool_name=str(getattr(decision, "tool_name", "") or "tool"),
            trigger=str(getattr(decision, "code", "") or "unknown"),
            count=int(getattr(decision, "count", 0) or 0),
            error_snippet=_error_snippet(result),
        )
        self._episodes.append(episode)
        self._open_episode = episode
        self.episodes_detected += 1
        logger.info(
            "awareness: stuck episode detected (%s, count=%d)", signature, episode.count
        )
        return True

    @staticmethod
    def _episode_signature(decision: Any, result: Optional[str]) -> str:
        tool = str(getattr(decision, "tool_name", "") or "tool")
        if tool == "memory" and "No entry matched" in (result or ""):
            return "memory:no_match"
        code = str(getattr(decision, "code", "") or "unknown")
        # Collapse warn/block/halt variants of one detector into one pattern.
        for suffix in ("_warning", "_block", "_halt"):
            if code.endswith(suffix):
                code = code[: -len(suffix)]
                break
        return f"{tool}:{code}"

    def _record_recovery(self, tool_name: str, args: Optional[Mapping[str, Any]]) -> None:
        episode = self._open_episode
        if episode is None or len(episode.recovery_actions) >= _MAX_RECOVERY_ACTIONS:
            return
        brief = _recovery_brief(tool_name, args)
        if brief not in episode.recovery_actions:
            episode.recovery_actions.append(brief)

    def _maybe_build_note(self) -> Optional[str]:
        """RECALL: past-experience note for the open episode, once per episode."""
        episode = self._open_episode
        if episode is None or not self.note_enabled or episode.note_injected:
            return None
        episode.note_injected = True
        past = self._find_past_patterns(episode.tool_name)
        if not past:
            return None
        content = " ".join(str(past[0].get("content") or "").split())
        snippet = content[:_NOTE_SNIPPET_LIMIT] + ("…" if len(content) > _NOTE_SNIPPET_LIMIT else "")
        self.notes_injected += 1
        return (
            f"[Awareness: stuck pattern '{episode.signature}' seen before. "
            f"Past experience: {snippet}]"
        )

    def _find_past_patterns(self, tool_name: str, limit: int = 3) -> list[dict[str, Any]]:
        """FTS search for prior awareness patterns about this tool."""
        try:
            res = self._store.recall(
                f"stuck pattern {tool_name}", types=("pattern",), limit=limit + 3
            )
        except Exception:
            return []
        if not isinstance(res, dict) or not res.get("success"):
            return []
        results = res.get("results") or []
        return [r for r in results if tool_name in str(r.get("content") or "")][:limit]

    def _observe_memory_read(
        self, args: Optional[Mapping[str, Any]], result: Optional[str]
    ) -> Optional[str]:
        """Track consecutive empty ``memory read`` calls against a live store."""
        if not isinstance(args, Mapping) or args.get("action") != "read":
            return None
        parsed = safe_json_loads(result or "")
        if not isinstance(parsed, dict) or not parsed.get("success"):
            return None  # failed reads neither extend nor reset the streak
        if int(parsed.get("count") or 0) > 0:
            self._empty_recall_streak = 0
            return None
        if not self._store_has_entries():
            return None  # an empty store is not a "stuck" signal
        self._empty_recall_streak += 1
        if self._empty_recall_streak == self.empty_streak_limit:
            self.empty_streaks_hit += 1
            logger.info(
                "awareness: empty memory-read streak (%d)", self._empty_recall_streak
            )
            return (
                f"[Awareness: {self._empty_recall_streak} consecutive memory reads "
                "returned nothing while the store has entries. Rephrase with short "
                "unique substrings of the stored text, or stop querying memory and "
                "proceed without it.]"
            )
        return None

    def _store_has_entries(self) -> bool:
        try:
            conn = self._store._connect()
            row = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE status IN ('active','pinned')"
            ).fetchone()
            return bool(row and row[0] > 0)
        except Exception:
            return False

    def _record_episode(
        self, episode: StuckEpisode, *, exit_reason: str
    ) -> Optional[dict[str, Any]]:
        """RECORD: write the episode's honest outcome to the store."""
        if not self.record_enabled:
            return None
        if self.write_approval:
            self.records_skipped += 1
            return None
        if episode.signature in self._recorded_signatures:
            return None

        reason = str(exit_reason or "unknown")
        attempts = "; ".join(episode.recovery_actions) or "nothing"
        if reason.startswith("text_response("):
            if episode.recovery_actions:
                outcome = "recovered"
                helped = f"What helped: {attempts}."
            else:
                outcome = "answered after warning without further tool attempts"
                helped = "What helped: reporting the blocker instead of retrying."
        elif reason.startswith("guardrail_halt"):
            outcome = "halted by tool-call guardrail"
            helped = "No recovery observed before the halt."
        elif (
            reason.startswith("budget_exhausted")
            or reason.startswith("max_iterations_reached")
            or reason.startswith("error_near_max_iterations")
        ):
            outcome = "unresolved (budget exhausted)"
            helped = f"Attempted after detection: {attempts}."
        else:
            # Interrupted / errored / mailbox exits carry no verified outcome —
            # experience without verification is not growth.
            return None

        content = (
            f"Stuck pattern: {episode.tool_name} — {episode.trigger} "
            f"({episode.count}x this turn). Error: {episode.error_snippet}. "
            f"{helped} Outcome: {outcome}."
        )
        written_by = f"awareness:{self.session_id}" if self.session_id else "awareness"
        try:
            resp = self._store.add(
                target="memory",
                content=content,
                entry_type="pattern",
                importance=0.55,
                written_by=written_by,
            )
        except Exception:
            logger.debug("awareness: record failed", exc_info=True)
            return None
        self._recorded_signatures.add(episode.signature)
        if isinstance(resp, dict) and resp.get("success"):
            self.patterns_recorded += 1
            logger.info("awareness: recorded stuck pattern %s (%s)", episode.signature, outcome)
            return {"signature": episode.signature, "outcome": outcome, "content": content}
        return None
