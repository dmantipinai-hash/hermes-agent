"""Mailbox completion gate vs. owner change (field deadlock 29.09, t_ffd496a5).

The gate demands a response attempt from the CURRENT run, but mailbox mail
is addressed per-profile. A guidance sent to the previous owner before a
review-reassign can never be delivered to or answered by the new owner's
run — counting it blocked kanban_complete forever (Forseti's run got 4×
completion_blocked_mailbox; the orchestrator had to reassign back to Tor
to release the card).

Fix: the gate is scoped to mail addressed to the current run's profile
(canonical assignee fallback for manual completions), mirroring the
recipient filter try_close_mailbox_intake already applies.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def board_env(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return kb


def _send_guidance(kb, tid, recipient, key):
    conn = kb.connect()
    try:
        kb.send_mailbox_message(
            conn,
            task_id=tid,
            actor_identity="test:orchestrator",
            actor_kind="manager",
            sender_profile="orchestrator",
            recipient_profile=recipient,
            kind="guidance",
            body=f"guidance for {recipient} ({key})",
            wake_requested=True,
            idempotency_key=key,
        )
    finally:
        conn.close()


def _run_profile(kb, tid):
    conn = kb.connect()
    try:
        row = conn.execute(
            "SELECT current_run_id, profile FROM tasks t "
            "LEFT JOIN task_runs r ON r.id = t.current_run_id "
            "WHERE t.id = ?",
            (tid,),
        ).fetchone()
        return (row["current_run_id"], row["profile"])
    finally:
        conn.close()


def test_gate_ignores_mail_addressed_to_previous_owner(board_env):
    """The exact field sequence: guidance → tor (delivered+responded in
    tor's run), review-reassign to forseti — forseti's run must complete
    without owing an answer to tor's mail."""
    kb = board_env
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="gate reassign", assignee="tor")
        kb.claim_task(conn, tid)  # run 1, profile=tor
        kb.send_mailbox_message(
            conn,
            task_id=tid,
            actor_identity="test:forseti",
            actor_kind="manager",
            sender_profile="forseti",
            recipient_profile="tor",
            kind="guidance",
            body="review guidance №16",
            wake_requested=True,
            idempotency_key="field-16",
        )
        # tor's run answered its mail (delivery attempt, responded).
        run1_id, _ = _run_profile(kb, tid)
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_mailbox_delivery_attempts "
                "(message_id, run_id, state, claim_token, lease_expires, "
                " attempt_count, claimed_at, accepted_at, included_at, "
                " responded_at, updated_at) "
                "VALUES ((SELECT id FROM task_mailbox_messages WHERE "
                "        task_id=? AND idempotency_key='field-16'), "
                "        ?, 'model_response_received', 'tok', 0, 1, 0, 0, 0, 0, 0)",
                (tid, run1_id),
            )
        # Review-reassign to forseti: close tor's run, reassign, new claim.
        kb.reassign_task(conn, tid, "forseti", reclaim_first=True)
        kb.claim_task(conn, tid)  # run 2, profile=forseti
        run2_id, run2_profile = _run_profile(kb, tid)
        assert run2_profile == "forseti" and run2_id != run1_id

        outcome = kb.complete_task(
            conn, tid, summary="accepted per verdict", expected_run_id=run2_id,
        )
        assert outcome is True, (
            "mail addressed to the previous owner must not block the new "
            "owner's run — the field deadlock"
        )
    finally:
        conn.close()


def test_gate_still_blocks_same_profile_unresponded(board_env):
    """Control: guidance addressed to THIS run's profile, unanswered —
    completion must stay blocked (each run acknowledges its mail)."""
    kb = board_env
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="same profile gate", assignee="tor")
        kb.claim_task(conn, tid)
        kb.send_mailbox_message(
            conn,
            task_id=tid,
            actor_identity="test:orchestrator",
            actor_kind="manager",
            sender_profile="orchestrator",
            recipient_profile="tor",
            kind="guidance",
            body="must answer before complete",
            wake_requested=False,
            idempotency_key="same-1",
        )
        run_id, _ = _run_profile(kb, tid)
        with pytest.raises(kb.MailboxCompletionBlockedError):
            kb.complete_task(conn, tid, summary="try", expected_run_id=run_id)
    finally:
        conn.close()


def test_manual_complete_scopes_to_current_assignee(board_env):
    """No live run (CLI completion): the reference profile is the task's
    canonical assignee — stale mail to a previous owner frees, mail to the
    current assignee still gates."""
    kb = board_env
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="manual complete", assignee="old-owner")
        _send_guidance(kb, tid, "old-owner", "manual-1")
        # Owner changed after the mail was sent; no run is active.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET assignee='new-owner' WHERE id=?", (tid,)
            )
        assert kb.complete_task(conn, tid, summary="manual ok") is True

        tid2 = kb.create_task(conn, title="manual complete 2", assignee="cur")
        _send_guidance(kb, tid2, "cur", "manual-2")
        with pytest.raises(kb.MailboxCompletionBlockedError):
            kb.complete_task(conn, tid2, summary="manual blocked")
    finally:
        conn.close()
