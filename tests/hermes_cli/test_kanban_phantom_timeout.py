"""Phantom ``timed_out`` after ``completed`` (BUG 19.09, t_3326ae69).

Field case: a kanban worker lands kanban_complete on its FINAL budget
iteration, then finalize_turn's budget-exhausted fallback records a
``timed_out`` seconds AFTER ``completed`` — the run stays closed (CAS in
``_end_run``, event carries run_id NULL) but the event, a fresh
consecutive_failures bump and retry_status land on the DONE task, and the
notifier later wakes the orchestrator with a false timeout.

The fix is a terminal-task guard in ``_record_task_failure``: failures are
never recorded against done/archived tasks. These tests replay the exact
entry path (``agent.turn_finalizer._record_kanban_budget_exhausted``) and
keep the legitimate running-task path as the control.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest


@pytest.fixture
def board_env(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="phantom-timeout", assignee="default")
        kb.claim_task(conn, tid)
    finally:
        conn.close()
    return tid


def _events(board_env, kind):
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        rows = conn.execute(
            "SELECT id FROM task_events WHERE task_id = ? AND kind = ? "
            "ORDER BY id",
            (board_env, kind),
        ).fetchall()
        return [int(r["id"]) for r in rows]
    finally:
        conn.close()


def _task(board_env):
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        return kb.get_task(conn, board_env)
    finally:
        conn.close()


def test_budget_exhausted_after_complete_is_silent(board_env):
    """The exact field sequence: complete on the last budget iteration,
    then finalize_turn's fallback fires — nothing may land on the done
    task (no event, no counter, no retry)."""
    from agent.turn_finalizer import _record_kanban_budget_exhausted
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        assert kb.complete_task(conn, board_env, summary="done on iter 90")
    finally:
        conn.close()
    completed_events = _events(board_env, "completed")

    # What finalize_turn does 8 seconds later in the field case:
    _record_kanban_budget_exhausted(
        board_env, 90, 90, logging.getLogger("test")
    )

    assert _events(board_env, "timed_out") == []
    assert _events(board_env, "completed") == completed_events
    task = _task(board_env)
    assert task.status == "done"
    assert task.consecutive_failures == 0

    # The completed run's outcome is untouched — no gave_up/timed_out flip.
    conn = kb.connect()
    try:
        run = kb.latest_run(conn, board_env)
    finally:
        conn.close()
    assert run.outcome == "completed"


def test_budget_exhausted_on_running_task_still_records(board_env):
    """Control: the guard must not weaken the legitimate path — a worker
    that exhausts its budget WITHOUT completing still times out, counts,
    and returns the task to its retry phase."""
    from agent.turn_finalizer import _record_kanban_budget_exhausted

    assert _task(board_env).status == "running"

    _record_kanban_budget_exhausted(
        board_env, 90, 90, logging.getLogger("test")
    )

    assert len(_events(board_env, "timed_out")) == 1
    task = _task(board_env)
    assert task.status == "ready", "retry phase must be restored"
    assert task.consecutive_failures == 1


def test_record_task_failure_ignores_archived(board_env):
    """Archived is terminal too — a late failure recording on it is the
    same noise class."""
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        kb.complete_task(conn, board_env, summary="done then archived")
        conn.execute(
            "UPDATE tasks SET status = 'archived' WHERE id = ?",
            (board_env,),
        )
        conn.commit()
        blocked = kb._record_task_failure(
            conn,
            board_env,
            error="late crash observer",
            outcome="crashed",
            release_claim=True,
            end_run=True,
        )
    finally:
        conn.close()
    assert blocked is False
    assert _events(board_env, "crashed") == []
