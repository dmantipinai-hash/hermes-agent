"""Contracts for per-turn usage accounting (turn_usage table).

Turn-level token telemetry: ``begin_turn_usage`` opens a 'running' row per
run_conversation() turn, per-API-call deltas fold into it through
``update_token_counts(turn_no=...)`` in the same transaction as the session /
per-model ledger, and ``finalize_turn_usage`` stamps ended_at / duration_ms /
status. Behavior contracts only — no snapshots of counters.
"""
import time

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _turn_rows(db, session_id):
    with db._lock:
        rows = db._conn.execute(
            "SELECT * FROM turn_usage WHERE session_id = ? ORDER BY turn_no",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def _single(db, session_id, turn_no):
    with db._lock:
        row = db._conn.execute(
            "SELECT * FROM turn_usage WHERE session_id = ? AND turn_no = ?",
            (session_id, turn_no),
        ).fetchone()
    return dict(row) if row is not None else None


class TestBeginTurnUsage:
    def test_first_turn_is_one_and_running(self, db):
        db.create_session("s1", source="cli")
        assert db.begin_turn_usage("s1", model="m1") == 1
        row = _single(db, "s1", 1)
        assert row["status"] == "running"
        assert row["started_at"] is not None
        assert row["model"] == "m1"
        assert row["ended_at"] is None

    def test_sequential_turns_number_upward(self, db):
        db.create_session("s1", source="cli")
        assert db.begin_turn_usage("s1") == 1
        assert db.begin_turn_usage("s1") == 2
        assert db.begin_turn_usage("s1") == 3

    def test_user_message_id_resolves_to_latest_user_row(self, db):
        db.create_session("s1", source="cli")
        db.append_message("s1", "user", "prompt one")
        first_user_id = _single(db, "s1", 1)  # no turn yet
        assert first_user_id is None
        turn_no = db.begin_turn_usage("s1")
        db.finalize_turn_usage("s1", turn_no, status="completed")
        db.append_message("s1", "user", "prompt two")
        second_turn = db.begin_turn_usage("s1")
        with db._lock:
            ids = [
                r["id"]
                for r in db._conn.execute(
                    "SELECT id FROM messages WHERE session_id = 's1'"
                    " AND role = 'user' ORDER BY id"
                ).fetchall()
            ]
        assert len(ids) == 2
        row1 = _single(db, "s1", 1)
        row2 = _single(db, "s1", 2)
        # Turn 1 began when only the first user row existed.
        assert row1["user_message_id"] == ids[0]
        assert row2["user_message_id"] == ids[1]
        assert second_turn == 2

    def test_begin_closes_stale_running_rows_as_error(self, db):
        db.create_session("s1", source="cli")
        t1 = db.begin_turn_usage("s1")
        # Simulate a crash: no finalize, next turn begins.
        t2 = db.begin_turn_usage("s1")
        row1 = _single(db, "s1", t1)
        assert row1["status"] == "error"
        assert row1["ended_at"] is not None
        assert _single(db, "s1", t2)["status"] == "running"

    def test_no_session_id_is_noop(self, db):
        assert db.begin_turn_usage("") is None


class TestTurnDeltas:
    def test_deltas_accumulate_into_the_turn_row(self, db):
        db.create_session("s1", source="cli")
        t = db.begin_turn_usage("s1", model="m1")
        db.update_token_counts(
            "s1", input_tokens=1000, output_tokens=200,
            cache_read_tokens=5000, reasoning_tokens=42,
            model="m1", api_call_count=1, turn_no=t,
        )
        db.update_token_counts(
            "s1", input_tokens=300, output_tokens=50,
            cache_write_tokens=700,
            model="m1", api_call_count=1, turn_no=t,
        )
        row = _single(db, "s1", t)
        assert row["input_tokens"] == 1300
        assert row["output_tokens"] == 250
        assert row["cache_read_tokens"] == 5000
        assert row["cache_write_tokens"] == 700
        assert row["reasoning_tokens"] == 42
        assert row["api_call_count"] == 2

    def test_turn_row_matches_session_ledger(self, db):
        """The turn row is derived from the same deltas as the session row —
        their token columns must agree for a single-turn session."""
        db.create_session("s1", source="cli")
        t = db.begin_turn_usage("s1")
        db.update_token_counts(
            "s1", input_tokens=111, output_tokens=22,
            cache_read_tokens=333, model="m1", api_call_count=1, turn_no=t,
        )
        db.finalize_turn_usage("s1", t, status="completed")
        with db._lock:
            session = dict(
                db._conn.execute(
                    "SELECT input_tokens, output_tokens, cache_read_tokens,"
                    " api_call_count FROM sessions WHERE id = 's1'"
                ).fetchone()
            )
        row = _single(db, "s1", t)
        for col in ("input_tokens", "output_tokens", "cache_read_tokens",
                    "api_call_count"):
            assert row[col] == session[col]

    def test_no_turn_no_leaves_table_untouched(self, db):
        db.create_session("s1", source="cli")
        db.begin_turn_usage("s1")
        db.update_token_counts(
            "s1", input_tokens=500, model="m1", api_call_count=1,
        )
        # The turn row exists (from begin) but carries no tokens: the delta
        # without turn_no must not be attributed to it.
        row = _single(db, "s1", 1)
        assert row["input_tokens"] == 0
        assert row["api_call_count"] == 0

    def test_absolute_updates_never_touch_turns(self, db):
        """Gateway cumulative overwrites carry no turn identity — folding
        them into a turn would double-count its per-call deltas."""
        db.create_session("s1", source="cli")
        t = db.begin_turn_usage("s1")
        db.update_token_counts(
            "s1", input_tokens=100, model="m1", api_call_count=1, turn_no=t,
        )
        db.update_token_counts(
            "s1", input_tokens=999999, output_tokens=1,
            absolute=True, model="m1", api_call_count=5,
        )
        row = _single(db, "s1", t)
        assert row["input_tokens"] == 100
        assert row["api_call_count"] == 1

    def test_delta_without_begin_creates_degraded_row(self, db):
        db.create_session("s1", source="cli")
        db.update_token_counts(
            "s1", input_tokens=7, model="m1", api_call_count=1, turn_no=5,
        )
        row = _single(db, "s1", 5)
        assert row is not None
        assert row["input_tokens"] == 7
        assert row["started_at"] is not None

    def test_queued_deltas_of_different_turns_never_coalesce(self, db):
        db.create_session("s1", source="cli")
        t1 = db.begin_turn_usage("s1")
        t2 = db.begin_turn_usage("s1")
        assert (t1, t2) == (1, 2)
        # Enqueue interleaved deltas for both turns, then drain once —
        # the background writer coalesces same-route runs, and turn_no is
        # part of the route: tokens must land in their own turn's row.
        for _ in range(3):
            db.queue_token_counts(
                "s1", input_tokens=10, model="m1", api_call_count=1, turn_no=t1,
            )
            db.queue_token_counts(
                "s1", input_tokens=1000, model="m1", api_call_count=1, turn_no=t2,
            )
        assert db.flush_token_counts()
        assert _single(db, "s1", t1)["input_tokens"] == 30
        assert _single(db, "s1", t2)["input_tokens"] == 3000


class TestFinalizeTurnUsage:
    def test_finalize_stamps_duration_and_status(self, db):
        db.create_session("s1", source="cli")
        t = db.begin_turn_usage("s1", model="m1")
        time.sleep(0.02)
        db.finalize_turn_usage(
            "s1", t, status="completed", tool_call_count=4, model="m1",
        )
        row = _single(db, "s1", t)
        assert row["status"] == "completed"
        assert row["ended_at"] is not None
        assert row["duration_ms"] >= 20
        assert row["tool_call_count"] == 4

    def test_finalize_is_noop_for_missing_row(self, db):
        db.create_session("s1", source="cli")
        # Must not raise and must not create anything.
        db.finalize_turn_usage("s1", 77, status="completed")
        assert _single(db, "s1", 77) is None

    def test_finalize_does_not_reopen_on_late_delta(self, db):
        """A crash-time atexit drain can apply a delta after finalize — the
        status must survive (the row stays closed; tokens are best-effort)."""
        db.create_session("s1", source="cli")
        t = db.begin_turn_usage("s1")
        db.update_token_counts(
            "s1", input_tokens=5, model="m1", api_call_count=1, turn_no=t,
        )
        db.finalize_turn_usage("s1", t, status="completed")
        db.update_token_counts(
            "s1", input_tokens=5, model="m1", api_call_count=1, turn_no=t,
        )
        assert _single(db, "s1", t)["status"] == "completed"


class TestTurnUsageLifecycle:
    def test_two_prompts_two_rows_with_distinct_tokens(self, db):
        """The acceptance contract from the task: one session, two prompts —
        two turn rows with different token counts."""
        db.create_session("s1", source="cli")
        t1 = db.begin_turn_usage("s1", model="m1")
        db.append_message("s1", "user", "first prompt")
        db.update_token_counts(
            "s1", input_tokens=2000, output_tokens=100,
            cache_read_tokens=10000, model="m1", api_call_count=3, turn_no=t1,
        )
        db.finalize_turn_usage("s1", t1, status="completed", tool_call_count=6)

        db.append_message("s1", "user", "second prompt")
        t2 = db.begin_turn_usage("s1", model="m1")
        db.update_token_counts(
            "s1", input_tokens=50, output_tokens=10,
            model="m1", api_call_count=1, turn_no=t2,
        )
        db.finalize_turn_usage("s1", t2, status="completed", tool_call_count=0)

        rows = _turn_rows(db, "s1")
        assert len(rows) == 2
        assert rows[0]["turn_no"] == 1 and rows[1]["turn_no"] == 2
        assert rows[0]["input_tokens"] != rows[1]["input_tokens"]
        assert rows[0]["api_call_count"] != rows[1]["api_call_count"]

    def test_session_delete_cascades_turn_rows(self, db):
        db.create_session("s1", source="cli")
        t = db.begin_turn_usage("s1")
        db.finalize_turn_usage("s1", t, status="completed")
        with db._lock:
            db._conn.execute("DELETE FROM sessions WHERE id = 's1'")
            db._conn.commit()
        assert _turn_rows(db, "s1") == []
