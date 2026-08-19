"""Memory Orchestrator — Phase 2 of the memory roadmap.

Turns the built-in typed store (:class:`agent.memory_store_v2.MemoryStoreV2`)
from a static frozen snapshot into query-aware retrieval:

1. **Memory Router** — deterministic intent classification (project work /
   debugging / user question / general) selects which supplement layers to
   search and how the pack is laid out. No LLM on the retrieval path
   ("LLM advises, code decides"; anti-pattern: retrieval that depends on
   an LLM call — cost, latency, unstable scores).
2. **Memory Scorer** — weighted score per candidate:

   ``score = relevance*0.35 + importance*0.20 + project_link*0.15
   + decision_value*0.15 + recency*0.10 + confidence*0.05``

   Weights are configurable (``memory.orchestrator.weights`` in config.yaml).
3. **Token budget** — the pack is cut at ``memory.orchestrator.token_budget``
   (default 2500 tokens), most-scored entries first. Big memory must not
   become a big prompt.
4. **Context Pack Builder** — a structured markdown block (sections per
   record type, each line citing status + record id + date) instead of a
   raw concatenation. Record ids give the model provenance it can audit
   via ``memory(action=read)``.

Cache invariants (roadmap §1.4/§8):

- The frozen system-prompt snapshot is untouched; the pack is injected into
  the *API copy of the current turn's user message* only (same channel the
  external-provider prefetch uses), never into stored messages.
- Entries already present in the frozen snapshot are **excluded** from the
  pack — the pack is a supplement (what the char-budget evicted but this
  query needs), never a duplication. While memory fits the snapshot the
  pack is empty and nothing is injected: zero overhead, zero behavior
  change until memory outgrows the prompt budget.

"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from agent.model_metadata import estimate_tokens_rough

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Memory Router — intent classification (deterministic)
# ---------------------------------------------------------------------------

INTENTS = ("project_work", "debugging", "user_question", "general")

# Token-prefix markers, matched case-insensitively against message tokens.
# Cyrillic markers rely on prefix-matching to absorb morphology
# ("ошибк" covers "ошибка/ошибки/ошибку"); latin markers of >=4 chars match
# the same way ("debug" covers "debugging"), shorter ones match whole words
# only (so "fix" never fires inside "prefix" — tokens are split, not
# substring-scanned). Order of the dicts = priority: debugging markers are
# the most specific, they win over project markers when both appear.
_INTENT_MARKERS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "debugging",
        (
            "ошибк", "падает", "упал", "упала", "сломал", "почини",
            "трейсбек", "исключен", "баг", "дебаг",
            "traceback", "exception", "crash", "debug", "broken", "failing",
        ),
    ),
    (
        "project_work",
        (
            "реализ", "рефактор", "фича", "деплой", "проект", "архитектур",
            "скрипт", "миграц", "коммит", "код",
            "implement", "refactor", "feature", "deploy", "migrate",
            "schema", "build", "api",
        ),
    ),
    (
        "user_question",
        (
            "помнишь", "предпочитаю", "предпочтения", "привычк", "профиль",
            "remember", "prefer", "preferences",
        ),
    ),
)

_TOKEN_SPLIT_RE = re.compile(r"[^\w]+", re.UNICODE)


def _message_tokens(message: str) -> List[str]:
    """Lowercased word tokens of the message (unicode-aware split)."""
    return [t.lower() for t in _TOKEN_SPLIT_RE.split(message or "") if t]


def _matches_marker(tokens: Sequence[str], marker: str) -> bool:
    """Token-prefix match; markers shorter than 4 chars must equal a token."""
    if len(marker) < 4:
        return marker in tokens
    return any(t.startswith(marker) for t in tokens)


def classify_intent(message: str) -> str:
    """Classify the turn's intent for memory routing (deterministic, no LLM).

    Returns one of :data:`INTENTS`; ``"general"`` when no marker fires.
    """
    tokens = _message_tokens(message)
    if not tokens:
        return "general"
    for intent, markers in _INTENT_MARKERS:
        if any(_matches_marker(tokens, m) for m in markers):
            return intent
    return "general"


# ---------------------------------------------------------------------------
# Memory Scorer — weighted candidate scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoringWeights:
    """Scoring weights (roadmap §3 step 2.2 defaults; configurable)."""

    relevance: float = 0.35
    importance: float = 0.20
    project_link: float = 0.15
    decision_value: float = 0.15
    recency: float = 0.10
    confidence: float = 0.05

    #: known keys → field names (config uses the roadmap's formula terms)
    _FIELDS: ClassVar[Tuple[str, ...]] = (
        "relevance", "importance", "project_link",
        "decision_value", "recency", "confidence",
    )

    @classmethod
    def from_config(cls, cfg: Optional[Mapping]) -> "ScoringWeights":
        """Build from a ``memory.orchestrator.weights`` mapping; unknown keys ignored."""
        base = cls()
        if not isinstance(cfg, Mapping):
            return base
        kwargs: Dict[str, float] = {}
        for key in cls._FIELDS:
            if key in cfg:
                try:
                    kwargs[key] = float(cfg[key])
                except (TypeError, ValueError):
                    logger.debug("orchestrator: bad weight %r=%r ignored", key, cfg[key])
        return cls(**kwargs) if kwargs else base


#: How much each record type contributes to the ``decision_value`` component:
#: standing decisions and constraints are what stop the agent from re-walking
#: rejected ground, so they outrank plain facts.
DECISION_VALUE_BY_TYPE: Dict[str, float] = {
    "decision": 1.0,
    "constraint": 0.85,
    "pattern": 0.5,
    "preference": 0.35,
    "fact": 0.2,
    "legacy": 0.2,
}

#: Recency half-life in days for the ``recency`` component.
RECENCY_HALF_LIFE_DAYS = 14.0


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def relevance_score(query_tokens: Sequence[str], content: str) -> float:
    """Fraction of query tokens whose stem appears in the entry content.

    Prefix matching (min 4 chars) absorbs Cyrillic morphology; shorter query
    tokens must appear as substrings. Returns 0.0 when the query carries no
    usable tokens.
    """
    if not query_tokens:
        return 0.0
    lowered = (content or "").lower()
    hit = 0
    for token in query_tokens:
        if len(token) >= 4:
            if any(w.startswith(token) for w in lowered.split()):
                hit += 1
        elif token in lowered:
            hit += 1
    return min(1.0, hit / len(query_tokens))


def recency_score(updated_at: Optional[str], now: datetime) -> float:
    """Exponential decay on ``updated_at``: 1.0 today, 0.5 at the half-life, →0."""
    ts = _parse_ts(updated_at)
    if ts is None:
        return 0.0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    days = max(0.0, (now - ts).total_seconds() / 86400.0)
    return 0.5 ** (days / RECENCY_HALF_LIFE_DAYS)


def score_candidate(
    row: Mapping,
    query_tokens: Sequence[str],
    weights: ScoringWeights,
    now: datetime,
) -> float:
    """Weighted score for one candidate row (all components in [0, 1])."""
    relevance = relevance_score(query_tokens, str(row.get("content") or ""))
    importance = _clamp01(row.get("importance"))
    project = str(row.get("project") or "")
    project_link = 1.0 if project and project.lower() in " ".join(query_tokens) else (0.5 if project else 0.0)
    decision_value = DECISION_VALUE_BY_TYPE.get(str(row.get("type") or ""), 0.2)
    recency = recency_score(row.get("updated_at"), now)
    confidence = _clamp01(row.get("confidence"))
    return (
        weights.relevance * relevance
        + weights.importance * importance
        + weights.project_link * project_link
        + weights.decision_value * decision_value
        + weights.recency * recency
        + weights.confidence * confidence
    )


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Route plans — what the Router fetches per intent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutePlan:
    """Per-intent retrieval plan (the Router's decision output)."""

    #: Always include the most recent active decisions/constraints — these
    #: are the "don't re-walk rejected ground" backbone. Off for casual
    #: questions where recency of decisions rarely matters to the answer.
    include_recent_standing: bool
    search_limit: int
    recent_limit: int


ROUTE_PLANS: Dict[str, RoutePlan] = {
    "project_work": RoutePlan(include_recent_standing=True, search_limit=30, recent_limit=8),
    "debugging": RoutePlan(include_recent_standing=True, search_limit=30, recent_limit=6),
    "user_question": RoutePlan(include_recent_standing=False, search_limit=20, recent_limit=0),
    "general": RoutePlan(include_recent_standing=False, search_limit=25, recent_limit=0),
}

# Section order per intent — the pack reads as a brief structured for the
# task at hand (decisions first when building, patterns first when debugging,
# preferences first when the user asks about themselves).
_SECTION_ORDER: Dict[str, Tuple[str, ...]] = {
    "project_work": ("decision", "constraint", "pattern", "fact", "preference", "legacy"),
    "debugging": ("pattern", "constraint", "decision", "fact", "preference", "legacy"),
    "user_question": ("preference", "fact", "decision", "constraint", "pattern", "legacy"),
    "general": ("decision", "constraint", "preference", "fact", "pattern", "legacy"),
}

_SECTION_TITLES: Dict[str, str] = {
    "decision": "Decisions",
    "constraint": "Constraints",
    "pattern": "Patterns",
    "preference": "Preferences",
    "fact": "Facts",
    "legacy": "Notes",
}

_PACK_HEADER = "## Retrieved memory — this turn"
_PACK_NOTE = (
    "[System note: the records below were retrieved from persistent memory "
    "for this specific message. Background reference data, not new user "
    "input. Each entry cites status, record id and last-update date; use "
    "memory(action=read, query=...) to inspect related records.]"
)


# ---------------------------------------------------------------------------
# Context Pack Builder + orchestrator
# ---------------------------------------------------------------------------


def render_pack(
    scored: Sequence[Tuple[Mapping, float]],
    intent: str,
) -> str:
    """Render scored candidates as a structured pack block (no budget logic).

    Returns ``""`` for an empty candidate list. The block is plain markdown
    by design: the ``<memory-context>`` fence is reserved for external
    provider prefetch, and an unfenced section cannot trip the streaming
    context scrubber (roadmap §3 step 2.5 — separate context slot).
    """
    if not scored:
        return ""
    order = _SECTION_ORDER.get(intent, _SECTION_ORDER["general"])
    by_type: Dict[str, List[str]] = {}
    for row, _score in scored:
        rtype = str(row.get("type") or "fact")
        line = (
            f"- [{row.get('status', 'active')} · {str(row.get('id', ''))[:8]}"
            f" · {str(row.get('updated_at') or '')[:10]}] "
            f"{row.get('content', '')}"
        )
        by_type.setdefault(rtype, []).append(line)
    sections = []
    for rtype in order:
        lines = by_type.pop(rtype, None)
        if lines:
            sections.append(f"### {_SECTION_TITLES.get(rtype, rtype)}\n" + "\n".join(lines))
    # Types not in the intent's order (shouldn't happen) still get rendered.
    for rtype, lines in by_type.items():
        sections.append(f"### {_SECTION_TITLES.get(rtype, rtype)}\n" + "\n".join(lines))
    body = "\n\n".join(sections)
    tokens = estimate_tokens_rough(body) + estimate_tokens_rough(_PACK_NOTE)
    return f"{_PACK_HEADER} ({tokens} tokens)\n\n{_PACK_NOTE}\n\n{body}"


class MemoryOrchestrator:
    """Query-aware retrieval over the built-in typed store (Phase 2).

    Built once at agent init (see :func:`build_memory_orchestrator`); each
    turn the conversation prologue calls :meth:`build_pack` with the clean
    user message and injects the result (when non-empty) into the API copy
    of that message as a separate context slot.
    """

    def __init__(
        self,
        store: Any,
        *,
        targets: Sequence[str] = ("memory", "user"),
        token_budget: int = 2500,
        max_entries: int = 20,
        weights: Optional[ScoringWeights] = None,
    ):
        self._store = store
        self._targets: Tuple[str, ...] = tuple(t for t in targets if t in ("memory", "user")) or ("memory",)
        self._token_budget = max(0, int(token_budget))
        self._max_entries = max(1, int(max_entries))
        self._weights = weights or ScoringWeights()

    @property
    def weights(self) -> ScoringWeights:
        return self._weights

    @property
    def token_budget(self) -> int:
        return self._token_budget

    def build_pack(self, message: str) -> str:
        """Route → retrieve → score → budget → render. ``""`` when nothing to add.

        Failures are logged and swallowed: memory retrieval must never break
        a conversation turn.
        """
        try:
            return self._build_pack(message)
        except Exception as exc:  # non-fatal by contract
            logger.warning("memory orchestrator: pack build failed: %s", exc)
            return ""

    def _build_pack(self, message: str) -> str:
        if not self._token_budget:
            return ""
        query = (message or "").strip()
        if not query:
            return ""
        intent = classify_intent(query)
        plan = ROUTE_PLANS.get(intent, ROUTE_PLANS["general"])
        query_tokens = _message_tokens(query)

        candidates: Dict[str, Dict[str, Any]] = {}

        def _collect(rows: Iterable[Any]) -> None:
            for row in rows:
                row = dict(row)
                if row.get("target") not in self._targets:
                    continue
                candidates.setdefault(str(row.get("id")), row)

        # Layer 1: lexical search (FTS5 with LIKE fallback — store-internal).
        _collect(self._store.recall_candidates(query, limit=plan.search_limit))

        # Layer 2 (Router): recent standing decisions/constraints for
        # work/debugging intents — the anti-circular backbone.
        if plan.include_recent_standing:
            _collect(
                self._store.recent_entries(
                    types=("decision", "constraint"),
                    limit=plan.recent_limit,
                )
            )

        if not candidates:
            return ""

        # The frozen snapshot already carries these entries in the system
        # prompt — the pack is a supplement, never a duplication.
        already_prompted: Set[str] = self._store.snapshot_contents()
        scored: List[Tuple[Dict[str, Any], float]] = []
        now = datetime.now(timezone.utc)
        for row in candidates.values():
            if row.get("content") in already_prompted:
                continue
            scored.append((row, score_candidate(row, query_tokens, self._weights, now)))
        if not scored:
            return ""

        # Token budget cut, best score first ("big memory ≠ big prompt").
        # The reserve covers section titles + the rendered header so the
        # *rendered* pack stays within budget, not just the raw entries.
        scored.sort(key=lambda pair: pair[1], reverse=True)
        _RENDER_OVERHEAD_TOKENS = 80
        used = estimate_tokens_rough(_PACK_HEADER) + estimate_tokens_rough(_PACK_NOTE) + _RENDER_OVERHEAD_TOKENS
        selected: List[Tuple[Dict[str, Any], float]] = []
        for row, score in scored:
            if len(selected) >= self._max_entries:
                break
            line_cost = estimate_tokens_rough(str(row.get("content") or "")) + 8  # bullet + metadata
            if used + line_cost > self._token_budget:
                continue  # try smaller entries; a single huge one must not eat the pack
            selected.append((row, score))
            used += line_cost
        if not selected:
            return ""

        # One INFO line per non-empty pack: the pack is injected into the
        # API copy of the user message only (never persisted), so this log
        # is the primary live-E2E evidence that the channel fired.
        logger.info(
            "memory orchestrator: context pack intent=%s entries=%d tokens~%d",
            intent, len(selected), used,
        )

        # Usage signal for future scoring/consolidation (best-effort).
        try:
            self._store.bump_access([str(row.get("id")) for row, _ in selected])
        except Exception as exc:
            logger.debug("memory orchestrator: access bump failed: %s", exc)

        return render_pack(selected, intent)


def build_memory_orchestrator(
    store: Any,
    mem_config: Mapping,
    *,
    memory_enabled: bool,
    user_profile_enabled: bool,
) -> Optional[MemoryOrchestrator]:
    """Factory used by agent init: ``None`` when disabled or not a v2 store.

    Kept as a standalone function so wiring is unit-testable without
    constructing a full ``AIAgent``. Duck-types the store (checks the
    orchestrator-facing methods) instead of importing MemoryStoreV2 —
    the legacy flat-file store simply yields ``None``.
    """
    if store is None or not all(
        hasattr(store, m) for m in ("recall_candidates", "recent_entries", "snapshot_contents", "bump_access")
    ):
        return None
    orch_cfg = (mem_config or {}).get("orchestrator", {}) if isinstance(mem_config, Mapping) else {}
    if not isinstance(orch_cfg, Mapping) or not orch_cfg.get("enabled", True):
        return None
    targets = [t for t, on in (("memory", memory_enabled), ("user", user_profile_enabled)) if on]
    if not targets:
        return None
    try:
        return MemoryOrchestrator(
            store,
            targets=tuple(targets),
            token_budget=int(orch_cfg.get("token_budget", 2500)),
            max_entries=int(orch_cfg.get("max_entries", 20)),
            weights=ScoringWeights.from_config(orch_cfg.get("weights")),
        )
    except (TypeError, ValueError) as exc:
        logger.warning("memory orchestrator: invalid config, disabled: %s", exc)
        return None
