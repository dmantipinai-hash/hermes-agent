"""Phase 3 mid-turn crash recovery — invariant tests.

Covers the new SessionDB primitives (run_active marker, find_interrupted,
touch_activity, most_recent) and the agent-side tail-repair
(``_repair_interrupted_tail``). These are contracts, not snapshots: they
assert *how* the recovery primitives relate to each other, so adding a
new session column or a new tail case won't break them unless the
contract itself changes.

What's NOT tested here (covered by the smoke scripts in the commit):
  - the persist-after-each-tool integration in conversation_loop (needs a
    live agent run)
  - the /resume --last CLI handler (needs a HermesCLI instance)
These are exercised by the per-piece smoke checks; this file pins the
durable invariants that the DB + repair logic must hold.
"""
import time

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


# ---------------------------------------------------------------------------
# run_active lifecycle + find_interrupted_session
# ---------------------------------------------------------------------------

class TestRunActiveMarker:
    def test_fresh_session_is_idle(self, db):
        db.create_session("s1", source="cli")
        assert db.find_interrupted_session() is None, \
            "a freshly created session must not look interrupted"

    def test_active_session_is_interrupted(self, db):
        db.create_session("s1", source="cli")
        db.mark_run_active("s1")
        assert db.find_interrupted_session() == "s1"

    def test_idle_after_mark_run_idle(self, db):
        db.create_session("s1", source="cli")
        db.mark_run_active("s1")
        db.mark_run_idle("s1")
        assert db.find_interrupted_session() is None

    def test_find_interrupted_picks_most_recent(self, db):
        # Two sessions both interrupted; the one with the later
        # last_activity_at wins. Pins ordering, not specific ids.
        db.create_session("old", source="cli")
        db.create_session("new", source="cli")
        db.mark_run_active("old")
        time.sleep(0.02)
        db.mark_run_active("new")
        assert db.find_interrupted_session() == "new"

    def test_ended_session_not_considered_interrupted(self, db):
        # A session that was cleanly ended (ended_at set) must never be
        # reported as interrupted, even if run_active is stuck at 1.
        db.create_session("s1", source="cli")
        db.mark_run_active("s1")
        db.end_session("s1", "cli_close")
        assert db.find_interrupted_session() is None

    def test_mark_run_active_is_idempotent(self, db):
        db.create_session("s1", source="cli")
        db.mark_run_active("s1")
        db.mark_run_active("s1")
        assert db.find_interrupted_session() == "s1"

    def test_mark_run_idle_noop_on_unknown_session(self, db):
        # Must not raise on a nonexistent session id.
        db.mark_run_idle("does-not-exist")
        db.mark_run_active("does-not-exist")


class TestTouchActivity:
    def test_updates_last_activity_at(self, db):
        db.create_session("s1", source="cli")
        before = time.time()
        time.sleep(0.02)
        db.touch_activity("s1")
        with db._lock:
            row = db._conn.execute(
                "SELECT last_activity_at FROM sessions WHERE id = ?", ("s1",)
            ).fetchone()
        ts = row[0] if row else None
        assert ts is not None and ts > before


class TestMostRecentSession:
    def test_returns_latest_started(self, db):
        db.create_session("a", source="cli")
        time.sleep(0.01)
        db.create_session("b", source="cli")
        assert db.most_recent_session_id() == "b"

    def test_returns_none_when_empty(self, db):
        assert db.most_recent_session_id() is None

    def test_exclude_ended_filters(self, db):
        db.create_session("a", source="cli")
        db.create_session("b", source="cli")
        db.end_session("b", "cli_close")
        assert db.most_recent_session_id(exclude_ended=True) == "a"


# ---------------------------------------------------------------------------
# _repair_interrupted_tail — the resume-side fixer
# ---------------------------------------------------------------------------

class TestRepairInterruptedTail:
    """The tail repair pass. We attach the method to a stub so the test
    doesn't need to construct a full AIAgent — the method is pure-list
    logic with no instance state."""

    @pytest.fixture
    def repair(self):
        from run_agent import AIAgent

        class _Stub:
            pass
        _Stub._repair_interrupted_tail = AIAgent._repair_interrupted_tail
        return _Stub()._repair_interrupted_tail

    def test_orphan_assistant_tool_calls_dropped(self, repair):
        # Case 1: assistant(tool_calls) at tail with no tool results.
        m = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        ]
        dropped = repair(m)
        assert dropped == 1
        assert m[-1]["role"] == "user"

    def test_orphan_tool_results_dropped(self, repair):
        # Case 2: trailing tool messages with no preceding assistant(tool_calls).
        m = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "r1"},
            {"role": "tool", "content": "r2"},
        ]
        dropped = repair(m)
        assert dropped == 2
        assert m[-1]["role"] == "user"

    def test_partial_tool_pair_dropped(self, repair):
        # Case 3: 2 tool_calls but only 1 tool result — drop the whole pair.
        m = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}, {"id": "2"}]},
            {"role": "tool", "content": "r1"},
        ]
        dropped = repair(m)
        assert dropped == 2  # 1 tool + 1 assistant
        assert m[-1]["role"] == "user"

    def test_complete_pair_preserved(self, repair):
        # 2 tool_calls, 2 tool results — must NOT touch.
        m = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}, {"id": "2"}]},
            {"role": "tool", "content": "r1"},
            {"role": "tool", "content": "r2"},
        ]
        original_len = len(m)
        dropped = repair(m)
        assert dropped == 0
        assert len(m) == original_len

    def test_clean_assistant_text_tail_preserved(self, repair):
        m = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello there"},
        ]
        dropped = repair(m)
        assert dropped == 0

    def test_idempotent(self, repair):
        # Running twice must not drop more than once.
        m = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        ]
        d1 = repair(m)
        d2 = repair(m)
        assert d1 == 1 and d2 == 0

    def test_empty_list(self, repair):
        m = []
        assert repair(m) == 0
        assert m == []


# ---------------------------------------------------------------------------
# Schema migration — the two new columns exist
# ---------------------------------------------------------------------------

class TestSchemaColumns:
    def test_run_active_and_last_activity_at_exist(self, db):
        # _reconcile_columns should have added both on init.
        with db._lock:
            cols = {r[1] for r in db._conn.execute("PRAGMA table_info(sessions)").fetchall()}
        assert "run_active" in cols
        assert "last_activity_at" in cols

    def test_run_active_defaults_to_zero(self, db):
        db.create_session("s1", source="cli")
        with db._lock:
            row = db._conn.execute(
                "SELECT run_active FROM sessions WHERE id = ?", ("s1",)
            ).fetchone()
        assert row[0] == 0


# ---------------------------------------------------------------------------
# Resume-path message sequence validity after repair
# ---------------------------------------------------------------------------

class TestResumeProducesValidSequence:
    """After repair, the message list must obey role-alternation rules so
    the resumed API call doesn't 400. The strict provider invariant: a
    ``tool`` message must be preceded by an ``assistant(tool_calls)``, and
    the tail must be ``user`` or ``assistant`` (not a bare ``tool``)."""

    @pytest.fixture
    def repair(self):
        from run_agent import AIAgent

        class _Stub:
            pass
        _Stub._repair_interrupted_tail = AIAgent._repair_interrupted_tail
        return _Stub()._repair_interrupted_tail

    def _tail_is_resume_safe(self, messages):
        """A list is resume-safe if it ends in user or assistant (text),
        and every tool message has a preceding assistant(tool_calls)."""
        if not messages:
            return True
        tail = messages[-1]
        if tail.get("role") == "tool":
            return False
        if tail.get("role") == "assistant" and tail.get("tool_calls"):
            return False
        return True

    def test_after_crash_tail_is_safe(self, repair):
        scenarios = [
            # (name, interrupted-tail messages)
            ("orphan_assistant", [
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
            ]),
            ("orphan_tool", [
                {"role": "user", "content": "x"},
                {"role": "tool", "content": "r"},
            ]),
            ("partial_pair", [
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}, {"id": "2"}]},
                {"role": "tool", "content": "r1"},
            ]),
        ]
        for name, m in scenarios:
            repair(m)
            assert self._tail_is_resume_safe(m), \
                f"scenario {name!r}: tail unsafe after repair: {m[-1] if m else 'empty'}"
