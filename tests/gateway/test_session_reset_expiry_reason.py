"""Reset-policy leg classification for session expiry (visibility fix).

Field case 27–28.09.2026 (~04:00): the orchestrator session was finalized
by the DAILY leg of ``session_reset`` while kanban workers ran in separate
processes — context and granted approvals were dropped, and the only trace
was an INFO line. ``session_expiry_reason`` splits the legs so the watcher
can warn loudly on the wall-clock one; these tests pin the classification
the warning rides on.
"""
from __future__ import annotations

from datetime import timedelta

from gateway.config import GatewayConfig, SessionResetPolicy
from gateway.session import SessionEntry, SessionStore


def _store(tmp_path, policy: SessionResetPolicy) -> SessionStore:
    cfg = GatewayConfig()
    cfg.default_reset_policy = policy
    return SessionStore(sessions_dir=tmp_path, config=cfg)


def _entry(updated_minutes_ago: float) -> SessionEntry:
    from gateway.session import _now

    return SessionEntry(
        session_key="agent:main:telegram:dm:chat-1",
        session_id="sess-1",
        platform="telegram",
        chat_type="dm",
        created_at=_now() - timedelta(hours=48),
        updated_at=_now() - timedelta(minutes=updated_minutes_ago),
    )


def test_daily_leg_classified_by_wall_clock(tmp_path):
    """mode=both with a fresh update still expires via the daily leg — the
    exact field shape: session touched minutes before 04:00, boundary
    crossed, workers running elsewhere."""
    store = _store(
        tmp_path,
        SessionResetPolicy(mode="both", idle_minutes=1440, at_hour=4),
    )
    # updated 30 minutes ago, idle (24h) NOT reached. Whether the daily leg
    # fires depends on the wall clock relative to at_hour — inject it by
    # shifting updated_at across today's boundary.
    from gateway.session import _now

    now = _now()
    today_reset = now.replace(hour=4, minute=0, second=0, microsecond=0)
    if now.hour < 4:
        today_reset -= timedelta(days=1)
    stale = today_reset - timedelta(minutes=10)  # last touch just before 04:00
    entry = SessionEntry(
        session_key="agent:main:telegram:dm:chat-1",
        session_id="sess-1",
        platform="telegram",
        chat_type="dm",
        created_at=stale - timedelta(hours=1),
        updated_at=stale,
    )
    assert store.session_expiry_reason(entry) == "daily"


def test_idle_leg_classified_quietly(tmp_path):
    store = _store(
        tmp_path,
        SessionResetPolicy(mode="both", idle_minutes=60, at_hour=4),
    )
    # updated 2 hours ago (idle reached); today's daily boundary is AFTER
    # the update if we are past 04:00, else before — idle must win the
    # classification either way when it fires first in the method's order.
    entry = _entry(120)
    reason = store.session_expiry_reason(entry)
    # With updated 120 min ago and idle=60: idle leg definitely fires.
    assert reason == "idle"


def test_mode_none_never_expires(tmp_path):
    store = _store(tmp_path, SessionResetPolicy(mode="none"))
    assert store.session_expiry_reason(_entry(10_000)) is None
    assert store._is_session_expired(_entry(10_000)) is False


def test_default_policy_is_none_no_nightly_wipe(tmp_path):
    """The built-in default must never arm the 4am wipe — that is opt-in
    config only (upstream changed it July 2026; legacy config.yaml blocks
    like mode:both at_hour:4 are what resurrect it)."""
    cfg = GatewayConfig()
    assert cfg.default_reset_policy.mode == "none"
    store = SessionStore(sessions_dir=tmp_path, config=cfg)
    assert store.session_expiry_reason(_entry(10_000)) is None
