"""Drift Detection Hook plugin.

Detects when the agent's actual tool-call sequence drifts from the numbered
steps documented in a skill's SKILL.md, and annotates the tool result so the
existing ``background_review`` fork sees the drift and can patch the skill.

Why a hook and not a patcher
----------------------------
The background review fork (``agent/background_review.py``) already knows how
to patch skills — with guardrails the hook does not have: protected-skill
enforcement, anti-pattern filters, and a ``memory + skills`` tool whitelist.
AGENTS.md §10.7 is explicit: "do not write your own skill patcher." So this
hook's job is *detection + surfacing*, not *writing*. It annotates the tool
result; the annotation flows into ``messages_snapshot``; the review agent
reads it and decides whether to patch.

Detection model
---------------
1. On ``skill_view``: parse numbered steps (``1.``, ``2.``, ``**Step 1 —``)
   from the SKILL.md content the tool returned. Track them per session.
2. On subsequent tool calls in the same turn: advance a cursor through the
   documented steps. If the agent calls a tool that doesn't fit the next
   expected step (heuristic: the tool name isn't mentioned in the step text
   AND the step text names a different tool), flag a drift.
3. On ``transform_tool_result`` for the flagged call: append a
   ``⚠️ Drift detected`` block to the result string so it lands in the
   message history the review agent inherits.

The heuristic is deliberately conservative: false negatives (missed drift)
are acceptable; false positives (flagging correct behaviour) would train the
review agent to ignore the annotations. We only flag when the documented
step names a specific tool and the agent called a *different* one.

State is keyed by ``session_id`` (falls back to ``task_id``) and cleaned up
on ``on_session_end``. All access is lock-guarded because parallel tool
calls are possible.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger("drift_detection")

# ---------------------------------------------------------------------------
# Per-session state — lock-guarded (parallel tool calls are possible).
# ---------------------------------------------------------------------------

_lock = threading.Lock()

# session_key -> {"skill": name, "steps": [str], "cursor": int,
#                 "drifts": [{"step": int, "expected": str, "observed": str}]}
_active: Dict[str, Dict[str, Any]] = {}


def _session_key(session_id: str, task_id: str) -> str:
    return session_id or task_id or "_global"


# ---------------------------------------------------------------------------
# Step parsing — handles the three common SKILL.md formats (see research):
#   - "1. **Foo.**" numbered lists
#   - "**Step 1 — Foo.**" bold markers
#   - "## 1. Foo" numbered headers
# Returns a list of step description strings, in order.
# ---------------------------------------------------------------------------

# Matches "1. text", "  2. text", "1) text", "**1.** text"
_NUMBERED_LIST = re.compile(r"^\s*\*{0,2}(\d+)[.)]\s+\*{0,2}(.+)$", re.MULTILINE)
# Matches "**Step 1 — text**", "**Step 1: text**"
_BOLD_STEP = re.compile(r"\*{2}Step\s+(\d+)\s*[—:\-]\s*([^*]+)\*{2}", re.IGNORECASE)


def parse_numbered_steps(content: str) -> List[str]:
    """Extract numbered procedure steps from SKILL.md content.

    Returns up to 30 steps in order. Tries the bold-``Step N`` format first
    (most specific), then plain numbered lists. Steps shorter than 6 chars
    are skipped — they're usually list items inside prose, not procedure
    steps. The 6-char floor lets short-but-real verbs like "Verify" through
    while filtering numeric/fragment noise.
    """
    if not content:
        return []

    # Bold "**Step N — ...**" markers — collect and sort by N.
    bold: Dict[int, str] = {}
    for m in _BOLD_STEP.finditer(content):
        n = int(m.group(1))
        text = m.group(2).strip().rstrip(".")
        if len(text) >= 6 and n not in bold:
            bold[n] = text
    if bold:
        return [bold[k] for k in sorted(bold)][:30]

    # Plain numbered list. Keep only strictly-increasing sequence numbers
    # starting from 1 (filters out "2026. date" style false matches).
    steps: List[str] = []
    last_n = 0
    for m in _NUMBERED_LIST.finditer(content):
        try:
            n = int(m.group(1))
        except (TypeError, ValueError):
            continue
        text = m.group(2).strip().strip("*").rstrip(".")
        if len(text) < 6:
            continue
        if n == last_n + 1:
            steps.append(text)
            last_n = n
        elif n == 1 and not steps:
            # Restart at 1 if we hit a fresh list.
            steps.append(text)
            last_n = 1
    return steps[:30]


# ---------------------------------------------------------------------------
# Drift detection heuristic.
# ---------------------------------------------------------------------------

# Tools whose result we track as "doing step N". We ignore pure-info tools
# (skill_view itself, read_file, search_files) because they don't represent
# progress through a procedure — the agent reads, then acts.
_OBSERVED_TOOLS = {
    "terminal", "write_file", "patch", "skill_manage", "web_search",
    "web_extract", "browser_navigate", "browser_click", "execute_code",
    "delegate_task", "kanban_create", "kanban_complete", "memory",
}


def _tool_words(step_text: str) -> set:
    """Hermes tool names mentioned in a step's description."""
    words = set(re.findall(r"`?([a-z_]+)`?", step_text.lower()))
    return words & _OBSERVED_TOOLS


def detect_drift(
    expected_step: str,
    observed_tool: str,
    observed_args: Dict[str, Any],
) -> Optional[str]:
    """Return a drift description if the observed tool contradicts the
    expected step, else None.

    Conservative: only flags when (a) the expected step names a specific
    tool and (b) the observed tool is different AND also a tracked action
    tool. A step that names no tool is treated as "any tool OK" — too noisy
    to flag otherwise.
    """
    if not expected_step:
        return None
    expected_tools = _tool_words(expected_step)
    if not expected_tools:
        # Step doesn't pin a specific tool — can't contradict it.
        return None
    if observed_tool in expected_tools:
        return None
    if observed_tool not in _OBSERVED_TOOLS:
        # Observing a non-action tool (read/search) — not a drift signal.
        return None
    return (
        f"skill step expected `{', '.join(sorted(expected_tools))}` "
        f"but agent called `{observed_tool}`"
    )


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def _on_post_tool_call(
    tool_name: str = "",
    args: Optional[Dict[str, Any]] = None,
    result: Any = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    **_: Any,
) -> None:
    """Track skill_view loads and compare subsequent tool calls."""
    if not isinstance(args, dict):
        args = {}
    key = _session_key(session_id, task_id)

    # skill_view → capture the skill's numbered steps.
    if tool_name == "skill_view":
        name = args.get("name", "")
        content = ""
        if isinstance(result, str):
            try:
                payload = json.loads(result)
                if isinstance(payload, dict):
                    content = payload.get("content", "") or ""
                    if not name:
                        name = payload.get("name", "") or ""
            except (json.JSONDecodeError, TypeError):
                pass
        steps = parse_numbered_steps(content)
        if name and steps:
            with _lock:
                _active[key] = {
                    "skill": name,
                    "steps": steps,
                    "cursor": 0,
                    "drifts": [],
                }
            logger.debug(
                "drift_detection: tracking skill %r with %d steps (session=%s)",
                name, len(steps), key,
            )
        return

    # Other tool calls → advance cursor, check for drift.
    with _lock:
        state = _active.get(key)
        if not state or not state["steps"]:
            return
        cursor = state["cursor"]
        if cursor >= len(state["steps"]):
            return  # already past documented steps
        expected = state["steps"][cursor]
        drift = detect_drift(expected, tool_name, args)
        if drift:
            state["drifts"].append(
                {
                    "step": cursor + 1,
                    "expected": expected,
                    "observed_tool": tool_name,
                    "tool_call_id": tool_call_id,
                    "detail": drift,
                }
            )
            # Don't advance cursor on drift — the step is still "current"
            # until the agent does what it says or the turn ends.
        else:
            # Tool matches (or step names no specific tool) → advance.
            state["cursor"] = cursor + 1


def _on_transform_tool_result(
    tool_name: str = "",
    result: Any = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    **_: Any,
) -> Optional[str]:
    """Append a drift note to the tool result so background_review sees it.

    Returns the (possibly modified) result string. The note is only added
    once per drift — we mark it consumed so it doesn't duplicate if the
    result is transformed by other plugins too.
    """
    if not isinstance(result, str):
        return None
    key = _session_key(session_id, task_id)
    with _lock:
        state = _active.get(key)
        if not state or not state["drifts"]:
            return None
        # Find an unconsumed drift matching this tool_call_id (or any if
        # the id is empty — some call paths don't populate it).
        for i, d in enumerate(state["drifts"]):
            if d.get("consumed"):
                continue
            if tool_call_id and d.get("tool_call_id") and d["tool_call_id"] != tool_call_id:
                continue
            d["consumed"] = True
            note = _format_drift_note(state["skill"], d)
            return _append_note(result, note)
    return None


def _format_drift_note(skill: str, drift: Dict[str, Any]) -> str:
    return (
        f"\n\n⚠️ **Drift detected in skill `{skill}`** — documented step "
        f"{drift['step']} expected: \"{drift['expected'][:120]}\", "
        f"but the agent called `{drift['observed_tool']}`. "
        f"If the agent found a *better* way than the documented procedure, "
        f"this is a signal that the skill's SKILL.md is stale and should be "
        f"updated to match the actual effective workflow."
    )


def _append_note(result: str, note: str) -> str:
    """Append the drift note to a JSON tool-result string or plain string."""
    try:
        payload = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return result + note
    if isinstance(payload, dict):
        # Add a structured field + append to any existing message text.
        payload.setdefault("drift_detected", True)
        existing = payload.get("message", "") or payload.get("error", "") or ""
        payload["message"] = (existing + note) if existing else note.strip()
        return json.dumps(payload)
    return result + note


def _on_session_end(session_id: str = "", task_id: str = "", **_: Any) -> None:
    """Clean up per-session state."""
    key = _session_key(session_id, task_id)
    with _lock:
        _active.pop(key, None)


# ---------------------------------------------------------------------------
# Test/debug helpers (not hooks — imported by the test suite).
# ---------------------------------------------------------------------------


def _reset_for_test() -> None:
    """Clear all tracked state — for unit tests only."""
    with _lock:
        _active.clear()


def _state_for_test(session_id: str = "_test", task_id: str = "") -> Optional[dict]:
    """Read-only snapshot of tracked state for a session — for tests."""
    with _lock:
        state = _active.get(_session_key(session_id, task_id))
        return dict(state) if state else None


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — called by PluginManager on load."""
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("transform_tool_result", _on_transform_tool_result)
    ctx.register_hook("on_session_end", _on_session_end)
    logger.info("drift_detection plugin registered")
