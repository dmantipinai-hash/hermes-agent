"""Contracts for the per-turn insights view (``hermes insights --turns``).

``generate_turns`` reads ``turn_usage`` into a developer-facing report:
window filtering, per-turn token rows, totals, and cache-share math. The
default ``generate()`` report is unchanged by the turns feature (no new
keys, no behavior shift).
"""
import time

import pytest

from agent.insights import InsightsEngine
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _finish_turn(db, session_id, *, input_tokens, output_tokens=0,
                 cache_read_tokens=0, api_calls=1, tool_calls=0,
                 status="completed", age_days=0.0, duration_ms=1000):
    db.begin_turn_usage(session_id, model="m1")
    with db._lock:
        row = db._conn.execute(
            "SELECT MAX(turn_no) FROM turn_usage WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        turn_no = row[0]
        started = time.time() - age_days * 86400
        db._conn.execute(
            "UPDATE turn_usage SET started_at = ? WHERE session_id = ?"
            " AND turn_no = ?",
            (started, session_id, turn_no),
        )
        db._conn.commit()
    db.update_token_counts(
        session_id, input_tokens=input_tokens, output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens, model="m1",
        api_call_count=api_calls, turn_no=turn_no,
    )
    db.finalize_turn_usage(
        session_id, turn_no, status=status, tool_call_count=tool_calls,
    )
    if duration_ms is not None:
        with db._lock:
            db._conn.execute(
                "UPDATE turn_usage SET duration_ms = ? WHERE session_id = ?"
                " AND turn_no = ?",
                (duration_ms, session_id, turn_no),
            )
            db._conn.commit()
    return turn_no


class TestGenerateTurns:
    def test_window_filter_excludes_old_turns(self, db):
        db.create_session("s1", source="cli")
        _finish_turn(db, "s1", input_tokens=100, age_days=10)
        _finish_turn(db, "s1", input_tokens=200, age_days=0)
        engine = InsightsEngine(db)
        recent = engine.generate_turns(days=1)
        assert recent["totals"]["turns"] == 1
        assert recent["totals"]["input_tokens"] == 200
        wide = engine.generate_turns(days=30)
        assert wide["totals"]["turns"] == 2
        assert wide["totals"]["input_tokens"] == 300

    def test_totals_and_cache_share(self, db):
        db.create_session("s1", source="cli")
        _finish_turn(db, "s1", input_tokens=1000, output_tokens=100,
                     cache_read_tokens=3000, api_calls=2, tool_calls=5,
                     duration_ms=5000)
        _finish_turn(db, "s1", input_tokens=1000, output_tokens=100,
                     cache_read_tokens=1000, api_calls=1, duration_ms=7000)
        report = InsightsEngine(db).generate_turns(days=1)
        t = report["totals"]
        assert t["turns"] == 2
        assert t["sessions"] == 1
        assert t["input_tokens"] == 2000
        assert t["cache_read_tokens"] == 4000
        # cache hit = cache_read / (cache_read + input)
        assert t["cache_hit_pct"] == pytest.approx(4000 / 6000 * 100)
        assert t["api_calls"] == 3
        assert t["tool_calls"] == 5
        assert t["avg_duration_ms"] == 6000
        assert t["completed"] == 2 and t["error"] == 0

    def test_per_turn_rows_carry_identity(self, db):
        db.create_session("s1", source="cli")
        _finish_turn(db, "s1", input_tokens=10)
        _finish_turn(db, "s1", input_tokens=20)
        report = InsightsEngine(db).generate_turns(days=1)
        turns = report["turns"]
        # Newest first, and both turns identify their session + number.
        assert [t["turn_no"] for t in turns] == [2, 1]
        assert all(t["session_id"] == "s1" for t in turns)
        assert turns[0]["input_tokens"] == 20

    def test_missing_table_degrades_gracefully(self, db):
        db.create_session("s1", source="cli")
        with db._lock:
            db._conn.execute("DROP TABLE turn_usage")
            db._conn.commit()
        report = InsightsEngine(db).generate_turns(days=1)
        assert report.get("unavailable")
        assert report["turns"] == []
        text = InsightsEngine(db).format_turns_terminal(report)
        assert "unavailable" in text

    def test_default_report_shape_unchanged(self, db):
        """The turns feature must not leak into the default report."""
        db.create_session("s1", source="cli")
        _finish_turn(db, "s1", input_tokens=10)
        report = InsightsEngine(db).generate(days=1)
        assert "turns" not in report
        for key in ("overview", "models", "platforms", "tools", "skills",
                    "activity", "top_sessions", "delegation"):
            assert key in report


class TestFormatTurnsTerminal:
    def test_renders_turn_lines_and_totals(self, db):
        db.create_session("sessionxyz", source="cli")
        _finish_turn(db, "sessionxyz", input_tokens=1234, output_tokens=56,
                     cache_read_tokens=876, api_calls=2, tool_calls=3)
        engine = InsightsEngine(db)
        text = engine.format_turns_terminal(engine.generate_turns(days=1))
        assert "sessionxyz"[:8] in text
        assert "#1" in text
        assert "1,234" in text
        assert "ok 1" in text

    def test_empty_window_message(self, db):
        db.create_session("s1", source="cli")
        engine = InsightsEngine(db)
        text = engine.format_turns_terminal(engine.generate_turns(days=1))
        assert "No turns recorded" in text
