"""Turn-level usage accounting for ``run_conversation`` (zero model footprint).

One row in ``state.db.turn_usage`` per turn (a full ``run_conversation`` call:
user prompt → final response). The lifecycle mirrors the crash-resilience pair
``mark_run_active`` / ``mark_run_idle``:

* ``begin_turn_usage(agent)`` — called by the conversation loop right after the
  turn prologue (``build_turn_context``) has flushed the user message, so the
  row's ``user_message_id`` can be resolved from the session's newest user row.
  Stashes the turn number on ``agent._turn_usage_no``.
* per-API-call deltas — the existing ``queue_token_counts`` chokepoints
  (conversation_loop + codex_runtime) pass ``turn_no=`` along; the SessionDB
  writer folds them into ``turn_usage`` in the same transaction as the
  session/per-model rows. No extra write path, no extra thread.
* ``finalize_turn_usage(agent, ...)`` — called from
  ``agent/turn_finalizer.finalize_turn`` next to ``mark_run_idle``, after
  ``_persist_session`` drained the token queue: stamps ended_at / duration_ms /
  status / tool_call_count.

Everything here is best-effort by contract: usage accounting must never fail,
delay, or alter a turn. The model's context window is never touched (no tools,
no schema, no prompt changes) — this is developer telemetry, not UX.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("agent.turn_usage")


def begin_turn_usage(agent) -> None:
    """Open this turn's ``turn_usage`` row; stash its number on the agent.

    Called once per turn after the turn prologue. Failures degrade to
    ``_turn_usage_no = None`` (deltas then still create the row from the
    first API call, with started_at at that moment).
    """
    db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if db is None or not session_id:
        agent._turn_usage_no = None
        return
    try:
        agent._turn_usage_no = db.begin_turn_usage(
            session_id, model=getattr(agent, "model", None)
        )
    except Exception:
        agent._turn_usage_no = None
        logger.debug(
            "turn_usage begin failed (session=%s)", session_id, exc_info=True
        )


def current_turn_no(agent):
    """The open turn's number, or None outside a turn / when disabled."""
    return getattr(agent, "_turn_usage_no", None)


def finalize_turn_usage(agent, *, status: str, tool_call_count: int = 0) -> None:
    """Close the open turn's row and clear the turn number on the agent.

    ``status`` is one of completed / error / cancelled (turn_finalizer maps
    its own interrupted/failed/completed flags). Best-effort: never raises.
    """
    db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    turn_no = getattr(agent, "_turn_usage_no", None)
    agent._turn_usage_no = None
    if db is None or not session_id or turn_no is None:
        return
    try:
        db.finalize_turn_usage(
            session_id,
            turn_no,
            status=status,
            tool_call_count=tool_call_count,
            model=getattr(agent, "model", None),
        )
    except Exception:
        logger.debug(
            "turn_usage finalize failed (session=%s turn=%s)",
            session_id, turn_no, exc_info=True,
        )


def count_turn_tool_calls(messages) -> int:
    """Number of individual tool calls issued in this turn's transcript.

    Counts every entry of ``tool_calls`` on assistant rows (a single assistant
    message may issue several parallel calls), which is the number the
    ``tool_call_count`` column of ``turn_usage`` promises.
    """
    total = 0
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            total += len(tool_calls)
    return total
