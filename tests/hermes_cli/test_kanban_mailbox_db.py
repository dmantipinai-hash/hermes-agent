"""Persistence contracts for the A2 active-agent mailbox."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def mailbox_db(tmp_path: Path) -> sqlite3.Connection:
    db_path = tmp_path / "kanban.db"
    conn = kb.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def _running_task(
    conn: sqlite3.Connection,
    *,
    assignee: str = "recipient",
) -> tuple[str, int]:
    task_id = kb.create_task(conn, title="mailbox target", assignee=assignee)
    assert kb.claim_task(conn, task_id, claimer="test:worker") is not None
    row = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    assert row is not None and row["current_run_id"] is not None
    return task_id, int(row["current_run_id"])


def _send(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    body: str = "please use the new constraint",
    kind: str = "guidance",
    actor_identity: str = "session:manager",
    actor_kind: str = "manager",
    sender_profile: str = "manager",
    recipient_profile: str = "recipient",
    idempotency_key: str = "message-1",
    wake_requested: bool = False,
    now: int = 1_000,
):
    return kb.send_mailbox_message(
        conn,
        task_id=task_id,
        actor_identity=actor_identity,
        actor_kind=actor_kind,
        sender_profile=sender_profile,
        recipient_profile=recipient_profile,
        kind=kind,
        body=body,
        wake_requested=wake_requested,
        idempotency_key=idempotency_key,
        now=now,
    )


def _claim_one(
    conn: sqlite3.Connection,
    task_id: str,
    run_id: int,
    *,
    recipient_profile: str = "recipient",
    now: int = 1_100,
):
    deliveries = kb.claim_mailbox_messages(
        conn,
        task_id=task_id,
        run_id=run_id,
        recipient_profile=recipient_profile,
        now=now,
    )
    assert len(deliveries) == 1
    return deliveries[0]


def _ack_deliveries(
    conn: sqlite3.Connection,
    task_id: str,
    run_id: int,
    *,
    recipient_profile: str = "recipient",
    now: int = 1_100,
):
    deliveries = kb.claim_mailbox_messages(
        conn,
        task_id=task_id,
        run_id=run_id,
        recipient_profile=recipient_profile,
        now=now,
    )
    assert deliveries
    ids = [delivery.message_id for delivery in deliveries]
    tokens = {delivery.claim_token for delivery in deliveries}
    assert len(tokens) == 1
    token = tokens.pop()
    assert kb.accept_mailbox_messages(
        conn,
        message_ids=ids,
        run_id=run_id,
        recipient_profile=recipient_profile,
        claim_token=token,
        now=now + 1,
    )
    assert kb.mark_mailbox_messages_included(
        conn,
        message_ids=ids,
        run_id=run_id,
        recipient_profile=recipient_profile,
        claim_token=token,
        now=now + 2,
    )
    assert kb.mark_mailbox_messages_responded(
        conn,
        message_ids=ids,
        run_id=run_id,
        recipient_profile=recipient_profile,
        claim_token=token,
        now=now + 3,
    )
    return deliveries


def _insert_mailbox_audit(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    run_id: int,
    message_id: int,
) -> None:
    conn.execute(
        "INSERT INTO task_mailbox_audit "
        "(task_id, run_id, message_id, actor_identity, actor_kind, "
        " recipient_profile, action, allowed, reason, created_at) "
        "VALUES (?, ?, ?, 'session:manager', 'manager', 'recipient', "
        " 'send', 1, 'allowed', 1000)",
        (task_id, run_id, message_id),
    )


def _assert_hard_deleted(conn: sqlite3.Connection, task_id: str) -> None:
    assert conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM task_mailbox_messages WHERE task_id = ?", (task_id,)
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM task_mailbox_wake_evaluations e "
        "JOIN task_mailbox_messages m ON m.id = e.message_id "
        "WHERE m.task_id = ?",
        (task_id,),
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM task_mailbox_delivery_attempts a "
        "JOIN task_runs r ON r.id = a.run_id WHERE r.task_id = ?",
        (task_id,),
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM task_mailbox_audit WHERE task_id = ?", (task_id,)
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM task_comments WHERE task_id = ?", (task_id,)
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (task_id,)
    ).fetchone()[0] == 0


# ---------------------------------------------------------------------------
# Schema and additive migration
# ---------------------------------------------------------------------------


def test_mailbox_schema_uses_rowid_tables_and_expected_indexes(mailbox_db):
    table_names = {
        "task_mailbox_messages",
        "task_mailbox_wake_evaluations",
        "task_mailbox_delivery_attempts",
        "task_mailbox_audit",
    }
    rows = mailbox_db.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type = 'table' AND name LIKE 'task_mailbox_%'"
    ).fetchall()
    assert {row["name"] for row in rows} == table_names
    for row in rows:
        normalized = " ".join(row["sql"].lower().split())
        primary_key = (
            "message_id integer primary key"
            if row["name"] == "task_mailbox_wake_evaluations"
            else "id integer primary key"
        )
        assert primary_key in normalized
        assert "without rowid" not in normalized

    indexes = {
        row["name"]
        for row in mailbox_db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        )
    }
    assert {
        "idx_mailbox_messages_recipient",
        "idx_mailbox_wake_events_message",
        "idx_mailbox_attempts_run_state",
        "idx_mailbox_attempts_lease",
    } <= indexes
    triggers = {
        row["name"]
        for row in mailbox_db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger'"
        )
    }
    assert "trg_mailbox_wake_evaluations_no_insert" in triggers


def test_mailbox_schema_migration_is_idempotent_on_reopen(tmp_path: Path):
    db_path = tmp_path / "legacy.db"
    conn = kb.connect(db_path)
    task_id, run_id = _running_task(conn)
    conn.close()

    # Recreate the pre-mailbox shape while retaining a live legacy run.
    raw = sqlite3.connect(str(db_path), isolation_level=None)
    for trigger in (
        "trg_mailbox_messages_immutable",
        "trg_mailbox_messages_no_delete",
        "trg_task_runs_close_mailbox",
    ):
        raw.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    for table in (
        "task_mailbox_delivery_attempts",
        "task_mailbox_wake_evaluations",
        "task_mailbox_messages",
        "task_mailbox_audit",
    ):
        raw.execute(f"DROP TABLE IF EXISTS {table}")
    run_columns = {
        row[1] for row in raw.execute("PRAGMA table_info(task_runs)").fetchall()
    }
    for column in ("mailbox_accepting", "mailbox_wake_pending"):
        if column in run_columns:
            raw.execute(f"ALTER TABLE task_runs DROP COLUMN {column}")
    raw.close()

    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    migrated = kb.connect(db_path)
    row = migrated.execute(
        "SELECT mailbox_accepting, mailbox_wake_pending "
        "FROM task_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    assert tuple(row) == (0, 0)
    migrated.close()

    kb.init_db(db_path)
    reopened = kb.connect(db_path)
    assert reopened.execute(
        "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (task_id,)
    ).fetchone()[0] == 1
    assert reopened.execute(
        "SELECT COUNT(*) FROM sqlite_master "
        "WHERE type = 'table' AND name LIKE 'task_mailbox_%'"
    ).fetchone()[0] == 4
    assert reopened.execute(
        "SELECT COUNT(*) FROM sqlite_master "
        "WHERE type = 'trigger' "
        "AND name = 'trg_mailbox_wake_evaluations_no_insert'"
    ).fetchone()[0] == 1
    reopened.close()


def test_mailbox_message_constraints_reject_invalid_values(mailbox_db):
    task_id = kb.create_task(mailbox_db, title="constraint target")
    comment_id = kb.add_comment(mailbox_db, task_id, "human", "linked")
    valid = (
        task_id,
        "session:1",
        "manager",
        "manager",
        "recipient",
        "guidance",
        "body",
        0,
        comment_id,
        "key",
        1,
    )
    sql = (
        "INSERT INTO task_mailbox_messages "
        "(task_id, actor_identity, actor_kind, sender_profile, "
        " recipient_profile, kind, body, wake_requested, comment_id, "
        " idempotency_key, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    invalid_rows = [
        valid[:5] + ("command",) + valid[6:],
        valid[:6] + ("",) + valid[7:],
        valid[:7] + (2,) + valid[8:],
        valid[:9] + ("",) + valid[10:],
    ]
    for values in invalid_rows:
        with pytest.raises(sqlite3.IntegrityError):
            mailbox_db.execute(sql, values)


def test_delivery_attempt_constraints_reject_invalid_state_and_duplicates(mailbox_db):
    task_id, run_id = _running_task(mailbox_db)
    message_id = _send(mailbox_db, task_id).message_id
    sql = (
        "INSERT INTO task_mailbox_delivery_attempts "
        "(message_id, run_id, state, claim_token, lease_expires, "
        " attempt_count, claimed_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    )
    with pytest.raises(sqlite3.IntegrityError):
        mailbox_db.execute(
            sql, (message_id, run_id, "queued", "token", 2_000, 1, 1_000, 1_000)
        )
    mailbox_db.execute(
        sql,
        (
            message_id,
            run_id,
            "claimed_for_run",
            "token",
            2_000,
            1,
            1_000,
            1_000,
        ),
    )
    with pytest.raises(sqlite3.IntegrityError):
        mailbox_db.execute(
            sql,
            (
                message_id,
                run_id,
                "claimed_for_run",
                "other",
                2_000,
                1,
                1_000,
                1_000,
            ),
        )


def test_mailbox_messages_are_immutable(mailbox_db):
    task_id = kb.create_task(mailbox_db, title="immutable target")
    message_id = _send(mailbox_db, task_id).message_id
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        mailbox_db.execute(
            "UPDATE task_mailbox_messages SET body = 'changed' WHERE id = ?",
            (message_id,),
        )


def test_mailbox_wake_evaluation_is_immutable_but_hard_delete_removes_it(mailbox_db):
    task_id = kb.create_task(mailbox_db, title="immutable wake evaluation")
    sent = _send(mailbox_db, task_id, wake_requested=False)
    row = mailbox_db.execute(
        "SELECT effect FROM task_mailbox_wake_evaluations WHERE message_id = ?",
        (sent.message_id,),
    ).fetchone()
    assert row["effect"] == "not_requested"

    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        mailbox_db.execute(
            "UPDATE task_mailbox_wake_evaluations SET effect = 'promoted' "
            "WHERE message_id = ?",
            (sent.message_id,),
        )
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        mailbox_db.execute(
            "DELETE FROM task_mailbox_wake_evaluations WHERE message_id = ?",
            (sent.message_id,),
        )

    assert kb.delete_task(mailbox_db, task_id) is True
    assert mailbox_db.execute(
        "SELECT COUNT(*) FROM task_mailbox_wake_evaluations WHERE message_id = ?",
        (sent.message_id,),
    ).fetchone()[0] == 0


def test_mailbox_wake_evaluation_rejects_duplicate_insert_and_replace(mailbox_db):
    task_id = kb.create_task(mailbox_db, title="immutable wake replacement")
    sent = _send(mailbox_db, task_id, wake_requested=False)
    assert mailbox_db.execute("PRAGMA recursive_triggers").fetchone()[0] == 0

    for insert_prefix in ("INSERT", "INSERT OR REPLACE"):
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            mailbox_db.execute(
                f"{insert_prefix} INTO task_mailbox_wake_evaluations "
                "(message_id, effect, created_at) VALUES (?, 'promoted', 2000)",
                (sent.message_id,),
            )
        row = mailbox_db.execute(
            "SELECT effect, created_at FROM task_mailbox_wake_evaluations "
            "WHERE message_id = ?",
            (sent.message_id,),
        ).fetchone()
        assert tuple(row) == ("not_requested", 1_000)


def test_mailbox_message_delete_is_rejected_without_orphaning_attempt(
    mailbox_db,
):
    task_id, run_id = _running_task(mailbox_db)
    sent = _send(mailbox_db, task_id)
    delivery = _claim_one(mailbox_db, task_id, run_id)
    linked_comment_id = mailbox_db.execute(
        "SELECT comment_id FROM task_mailbox_messages WHERE id = ?",
        (sent.message_id,),
    ).fetchone()[0]

    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        mailbox_db.execute(
            "DELETE FROM task_mailbox_messages WHERE id = ?", (sent.message_id,)
        )

    assert mailbox_db.execute(
        "SELECT COUNT(*) FROM task_mailbox_delivery_attempts "
        "WHERE message_id = ? AND run_id = ?",
        (delivery.message_id, run_id),
    ).fetchone()[0] == 1
    visible, cursor = kb.list_worker_comments_since(mailbox_db, task_id, 0)
    assert visible == []
    assert cursor == linked_comment_id


def test_delete_task_cascades_mailbox_rows_and_linked_comment(mailbox_db):
    task_id, run_id = _running_task(mailbox_db)
    sent = _send(mailbox_db, task_id)
    _claim_one(mailbox_db, task_id, run_id)
    _insert_mailbox_audit(
        mailbox_db,
        task_id=task_id,
        run_id=run_id,
        message_id=sent.message_id,
    )

    assert kb.delete_task(mailbox_db, task_id) is True

    _assert_hard_deleted(mailbox_db, task_id)
    assert mailbox_db.execute(
        "SELECT COUNT(*) FROM task_mailbox_delivery_attempts "
        "WHERE message_id = ?",
        (sent.message_id,),
    ).fetchone()[0] == 0
    assert mailbox_db.execute(
        "SELECT COUNT(*) FROM task_mailbox_wake_evaluations WHERE message_id = ?",
        (sent.message_id,),
    ).fetchone()[0] == 0


def test_delete_archived_task_cascades_mailbox_rows_and_linked_comment(mailbox_db):
    task_id, run_id = _running_task(mailbox_db)
    sent = _send(mailbox_db, task_id)
    _claim_one(mailbox_db, task_id, run_id)
    _insert_mailbox_audit(
        mailbox_db,
        task_id=task_id,
        run_id=run_id,
        message_id=sent.message_id,
    )
    assert kb.archive_task(mailbox_db, task_id) is True

    assert kb.delete_archived_task(mailbox_db, task_id) is True

    _assert_hard_deleted(mailbox_db, task_id)
    assert mailbox_db.execute(
        "SELECT COUNT(*) FROM task_mailbox_delivery_attempts "
        "WHERE message_id = ?",
        (sent.message_id,),
    ).fetchone()[0] == 0
    assert mailbox_db.execute(
        "SELECT COUNT(*) FROM task_mailbox_wake_evaluations WHERE message_id = ?",
        (sent.message_id,),
    ).fetchone()[0] == 0


def test_mailbox_migration_reinstalls_all_immutability_triggers(mailbox_db):
    mailbox_db.execute("DROP TRIGGER trg_mailbox_messages_immutable")
    mailbox_db.execute("DROP TRIGGER IF EXISTS trg_mailbox_messages_no_delete")
    mailbox_db.execute(
        "DROP TRIGGER IF EXISTS trg_mailbox_wake_evaluations_immutable"
    )
    mailbox_db.execute(
        "DROP TRIGGER IF EXISTS trg_mailbox_wake_evaluations_no_delete"
    )
    mailbox_db.execute(
        "DROP TRIGGER IF EXISTS trg_mailbox_wake_evaluations_no_insert"
    )

    kb._migrate_add_optional_columns(mailbox_db)

    triggers = {
        row["name"]
        for row in mailbox_db.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'trigger' AND tbl_name = 'task_mailbox_messages'"
        )
    }
    assert {
        "trg_mailbox_messages_immutable",
        "trg_mailbox_messages_no_delete",
    } <= triggers
    evaluation_triggers = {
        row["name"]
        for row in mailbox_db.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'trigger' "
            "AND tbl_name = 'task_mailbox_wake_evaluations'"
        )
    }
    assert {
        "trg_mailbox_wake_evaluations_immutable",
        "trg_mailbox_wake_evaluations_no_delete",
        "trg_mailbox_wake_evaluations_no_insert",
    } <= evaluation_triggers


def test_mailbox_audit_has_no_body_column(mailbox_db):
    columns = {
        row["name"]
        for row in mailbox_db.execute("PRAGMA table_info(task_mailbox_audit)")
    }
    assert "body" not in columns
    assert {
        "task_id",
        "run_id",
        "message_id",
        "actor_identity",
        "actor_kind",
        "recipient_profile",
        "action",
        "allowed",
        "reason",
        "created_at",
    } <= columns


# ---------------------------------------------------------------------------
# Atomic send and human/worker comment views
# ---------------------------------------------------------------------------


def test_send_mailbox_message_is_atomic_on_event_failure(mailbox_db):
    task_id = kb.create_task(mailbox_db, title="atomic target")
    mailbox_db.execute(
        "CREATE TRIGGER abort_mailbox_event BEFORE INSERT ON task_events "
        "WHEN NEW.kind = 'mailbox_message_sent' "
        "BEGIN SELECT RAISE(ABORT, 'event rejected'); END"
    )

    with pytest.raises(sqlite3.IntegrityError, match="event rejected"):
        _send(mailbox_db, task_id)

    assert mailbox_db.execute(
        "SELECT COUNT(*) FROM task_comments WHERE task_id = ?", (task_id,)
    ).fetchone()[0] == 0
    assert mailbox_db.execute(
        "SELECT COUNT(*) FROM task_mailbox_messages WHERE task_id = ?", (task_id,)
    ).fetchone()[0] == 0


def test_send_mailbox_message_is_idempotent_without_duplicate_side_effects(mailbox_db):
    task_id = kb.create_task(mailbox_db, title="idempotent target")
    first = _send(mailbox_db, task_id, body="first body")
    second = _send(mailbox_db, task_id, body="first body", now=2_000)

    assert first.created is True
    assert second.created is False
    assert second.message_id == first.message_id
    assert mailbox_db.execute(
        "SELECT COUNT(*) FROM task_comments WHERE task_id = ?", (task_id,)
    ).fetchone()[0] == 1


def test_idempotent_retry_compares_canonical_redacted_body(mailbox_db):
    task_id = kb.create_task(mailbox_db, title="canonical retry")
    first = _send(
        mailbox_db,
        task_id,
        body="OPENAI_API_KEY=sk-proj-AAAA1111TAIL",
    )
    second = _send(
        mailbox_db,
        task_id,
        body="OPENAI_API_KEY=sk-proj-BBBB2222TAIL",
        now=2_000,
    )

    assert first.redacted is True
    assert second.redacted is True
    assert second.created is False
    assert second.message_id == first.message_id
    assert mailbox_db.execute(
        "SELECT COUNT(*) FROM task_comments WHERE task_id = ?", (task_id,)
    ).fetchone()[0] == 1
    assert mailbox_db.execute(
        "SELECT COUNT(*) FROM task_mailbox_messages WHERE task_id = ?", (task_id,)
    ).fetchone()[0] == 1


@pytest.mark.parametrize(
    ("override", "conflicting_field"),
    [
        ({"recipient_profile": "other"}, "recipient_profile"),
        ({"kind": "question"}, "kind"),
        ({"body": "different durable body"}, "body"),
        ({"wake_requested": True}, "wake_requested"),
    ],
)
def test_idempotency_conflict_rejects_changed_durable_payload_without_side_effects(
    mailbox_db, override, conflicting_field
):
    task_id = kb.create_task(mailbox_db, title="conflicting retry")
    first = _send(mailbox_db, task_id, body="original durable body")
    before = {
        "comments": mailbox_db.execute(
            "SELECT COUNT(*) FROM task_comments WHERE task_id = ?", (task_id,)
        ).fetchone()[0],
        "messages": mailbox_db.execute(
            "SELECT COUNT(*) FROM task_mailbox_messages WHERE task_id = ?",
            (task_id,),
        ).fetchone()[0],
        "events": mailbox_db.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)
        ).fetchone()[0],
    }

    retry_payload = {"body": "original durable body", **override}
    with pytest.raises(
        ValueError,
        match=rf"idempotency conflict.*{conflicting_field}",
    ) as exc_info:
        _send(
            mailbox_db,
            task_id,
            now=2_000,
            **retry_payload,
        )

    assert "idempotency conflict" in str(exc_info.value)
    assert "original durable body" not in str(exc_info.value)
    assert mailbox_db.execute(
        "SELECT COUNT(*) FROM task_comments WHERE task_id = ?", (task_id,)
    ).fetchone()[0] == before["comments"]
    assert mailbox_db.execute(
        "SELECT COUNT(*) FROM task_mailbox_messages WHERE task_id = ?", (task_id,)
    ).fetchone()[0] == before["messages"]
    assert mailbox_db.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)
    ).fetchone()[0] == before["events"]
    assert kb.get_task(mailbox_db, task_id).status == "ready"


@pytest.mark.parametrize("terminal_status", ["done", "archived"])
def test_idempotent_retry_after_terminal_state_is_still_terminal_rejected(
    mailbox_db, terminal_status
):
    task_id = kb.create_task(mailbox_db, title="terminal retry")
    first = _send(mailbox_db, task_id, body="same retry body")
    mailbox_db.execute(
        "UPDATE tasks SET status = ? WHERE id = ?", (terminal_status, task_id)
    )
    before_events = mailbox_db.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)
    ).fetchone()[0]

    with pytest.raises(ValueError, match="terminal") as exc_info:
        _send(mailbox_db, task_id, body="same retry body", now=2_000)

    assert "idempotency conflict" not in str(exc_info.value)
    assert mailbox_db.execute(
        "SELECT COUNT(*) FROM task_comments WHERE task_id = ?", (task_id,)
    ).fetchone()[0] == 1
    assert mailbox_db.execute(
        "SELECT id FROM task_mailbox_messages WHERE task_id = ?", (task_id,)
    ).fetchone()[0] == first.message_id
    assert mailbox_db.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)
    ).fetchone()[0] == before_events
    assert mailbox_db.execute(
        "SELECT COUNT(*) FROM task_events "
        "WHERE task_id = ? AND kind = 'mailbox_message_sent'",
        (task_id,),
    ).fetchone()[0] == 1


def test_send_mailbox_message_idempotency_is_scoped_to_task_and_sender(mailbox_db):
    first_task = kb.create_task(mailbox_db, title="first")
    second_task = kb.create_task(mailbox_db, title="second")

    a = _send(mailbox_db, first_task)
    b = _send(mailbox_db, second_task)
    c = _send(mailbox_db, first_task, sender_profile="other")

    assert len({a.message_id, b.message_id, c.message_id}) == 3


def test_wake_keeps_running_task_with_active_listener_unchanged(mailbox_db):
    task_id, run_id = _running_task(mailbox_db)

    _send(mailbox_db, task_id, wake_requested=True)

    task = mailbox_db.execute(
        "SELECT status, current_run_id FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    run = mailbox_db.execute(
        "SELECT mailbox_accepting, mailbox_wake_pending FROM task_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    assert tuple(task) == ("running", run_id)
    assert tuple(run) == (1, 0)


@pytest.mark.parametrize(
    "expected_effect",
    [
        "not_requested",
        "dependency_blocked",
        "none_running",
        "status_ineligible",
        "promoted",
        "wake_pending",
    ],
)
def test_each_message_persists_one_immutable_wake_evaluation(
    mailbox_db, expected_effect
):
    wake_requested = expected_effect != "not_requested"
    if expected_effect == "dependency_blocked":
        parent = kb.create_task(mailbox_db, title="unresolved parent")
        task_id = kb.create_task(
            mailbox_db,
            title="dependency-blocked target",
            assignee="recipient",
            parents=[parent],
        )
    elif expected_effect in {"none_running", "wake_pending"}:
        task_id, run_id = _running_task(mailbox_db)
        if expected_effect == "wake_pending":
            mailbox_db.execute(
                "UPDATE task_runs SET mailbox_accepting = 0 WHERE id = ?",
                (run_id,),
            )
    else:
        task_id = kb.create_task(
            mailbox_db, title=f"{expected_effect} target", assignee="recipient"
        )
        if expected_effect == "status_ineligible":
            mailbox_db.execute(
                "UPDATE tasks SET status = 'scheduled' WHERE id = ?", (task_id,)
            )
        elif expected_effect == "promoted":
            mailbox_db.execute(
                "UPDATE tasks SET status = 'todo' WHERE id = ?", (task_id,)
            )

    first = _send(mailbox_db, task_id, wake_requested=wake_requested)
    evaluated = [
        event
        for event in kb.list_events(mailbox_db, task_id)
        if event.kind == "mailbox_wake_evaluated"
    ]
    before_event_count = mailbox_db.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)
    ).fetchone()[0]
    retry = _send(
        mailbox_db,
        task_id,
        wake_requested=wake_requested,
        now=2_000,
    )

    assert first.created is True
    assert first.wake_effect == expected_effect
    assert len(evaluated) == 1
    assert evaluated[0].payload == {
        "message_id": first.message_id,
        "action": expected_effect,
    }
    assert retry.created is False
    assert retry.message_id == first.message_id
    assert retry.wake_effect == expected_effect
    assert mailbox_db.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)
    ).fetchone()[0] == before_event_count


def test_new_format_retry_reads_one_indexed_evaluation_not_task_events(mailbox_db):
    task_id, _ = _running_task(mailbox_db)
    first = _send(mailbox_db, task_id, wake_requested=True)
    assert first.wake_effect == "none_running"
    mailbox_db.executemany(
        "INSERT INTO task_events (task_id, kind, payload, created_at) "
        "VALUES (?, 'mailbox_wake_evaluated', ?, 2000)",
        [
            (
                task_id,
                json.dumps(
                    {"message_id": first.message_id + offset + 1, "action": "promoted"}
                ),
            )
            for offset in range(5_000)
        ],
    )
    before_events = mailbox_db.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)
    ).fetchone()[0]
    traced = []
    mailbox_db.set_trace_callback(traced.append)
    try:
        retry = _send(mailbox_db, task_id, wake_requested=True, now=3_000)
    finally:
        mailbox_db.set_trace_callback(None)

    normalized = [" ".join(statement.lower().split()) for statement in traced]
    assert retry.created is False
    assert retry.wake_effect == "none_running"
    assert any(
        "select effect from task_mailbox_wake_evaluations where message_id =" in sql
        for sql in normalized
    )
    assert not any(" from task_events " in f" {sql} " for sql in normalized)
    assert mailbox_db.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)
    ).fetchone()[0] == before_events
    plan = mailbox_db.execute(
        "EXPLAIN QUERY PLAN SELECT effect FROM task_mailbox_wake_evaluations "
        "WHERE message_id = ?",
        (first.message_id,),
    ).fetchall()
    assert any("integer primary key" in row[3].lower() for row in plan)


def test_legacy_wake_lookup_is_exact_indexed_bounded_and_reopen_safe(tmp_path: Path):
    db_path = tmp_path / "legacy-wake.db"
    conn = kb.connect(db_path)
    task_id = kb.create_task(conn, title="legacy promoted", assignee="recipient")
    conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (task_id,))
    first = _send(conn, task_id, wake_requested=True)
    assert first.wake_effect == "promoted"
    conn.execute(
        "DELETE FROM task_events "
        "WHERE task_id = ? AND kind = 'mailbox_wake_evaluated'",
        (task_id,),
    )
    conn.execute(
        "INSERT INTO task_events (task_id, kind, payload, created_at) "
        "VALUES ('legacy-malformed', 'mailbox_wake_requested', 'not-json', 1999)"
    )
    conn.executemany(
        "INSERT INTO task_events (task_id, kind, payload, created_at) "
        "VALUES (?, 'mailbox_wake_requested', ?, 2000)",
        [
            (
                task_id,
                json.dumps(
                    {"message_id": first.message_id + offset + 1, "action": "promoted"}
                ),
            )
            for offset in range(5_000)
        ],
    )
    conn.execute("DROP INDEX IF EXISTS idx_mailbox_wake_events_message")
    conn.execute("DROP TABLE task_mailbox_wake_evaluations")
    conn.close()

    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    reopened = kb.connect(db_path)
    assert reopened.execute(
        "SELECT COUNT(*) FROM task_mailbox_wake_evaluations"
    ).fetchone()[0] == 0
    before_events = reopened.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)
    ).fetchone()[0]
    traced = []
    reopened.set_trace_callback(traced.append)
    try:
        retry = _send(reopened, task_id, wake_requested=True, now=3_000)
    finally:
        reopened.set_trace_callback(None)

    normalized = [" ".join(statement.lower().split()) for statement in traced]
    assert retry.created is False
    assert retry.wake_effect == "promoted"
    legacy_queries = [
        sql
        for sql in normalized
        if "from task_events" in sql and "json_extract" in sql
    ]
    assert len(legacy_queries) == 1
    assert "limit 1" in legacy_queries[0]
    assert reopened.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)
    ).fetchone()[0] == before_events
    plan = reopened.execute(
        "EXPLAIN QUERY PLAN SELECT payload FROM task_events "
        "WHERE kind IN ('mailbox_wake_evaluated', 'mailbox_wake_requested') "
        "AND CASE WHEN json_valid(payload) "
        "THEN CAST(json_extract(payload, '$.message_id') AS INTEGER) END = ? "
        "ORDER BY id DESC LIMIT 1",
        (first.message_id,),
    ).fetchall()
    assert any("idx_mailbox_wake_events_message" in row[3] for row in plan)
    reopened.close()

    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    with kb.connect(db_path) as idempotent_reopen:
        assert idempotent_reopen.execute(
            "SELECT COUNT(*) FROM task_mailbox_wake_evaluations"
        ).fetchone()[0] == 0


@pytest.mark.parametrize(
    ("status", "expected_status", "terminal"),
    [
        ("triage", "triage", False),
        ("todo", "todo", False),
        ("scheduled", "scheduled", False),
        ("ready", "ready", False),
        ("running", "running", False),
        ("blocked", "blocked", False),
        ("review", "review", False),
        ("done", "done", True),
        ("archived", "archived", True),
    ],
)
def test_wake_status_matrix_all_valid_statuses(
    mailbox_db, status, expected_status, terminal
):
    assert {
        "triage", "todo", "scheduled", "ready", "running",
        "blocked", "review", "done", "archived",
    } == kb.VALID_STATUSES

    if status == "todo":
        parent = kb.create_task(mailbox_db, title="unresolved parent")
        task_id = kb.create_task(
            mailbox_db,
            title="wake matrix target",
            assignee="recipient",
            parents=[parent],
        )
    else:
        task_id = kb.create_task(
            mailbox_db, title="wake matrix target", assignee="recipient"
        )
        if status == "running":
            assert kb.claim_task(
                mailbox_db, task_id, claimer="test:worker"
            ) is not None
        elif status != "ready":
            mailbox_db.execute(
                "UPDATE tasks SET status = ? WHERE id = ?", (status, task_id)
            )

    before = {
        "comments": mailbox_db.execute(
            "SELECT COUNT(*) FROM task_comments WHERE task_id = ?", (task_id,)
        ).fetchone()[0],
        "messages": mailbox_db.execute(
            "SELECT COUNT(*) FROM task_mailbox_messages WHERE task_id = ?",
            (task_id,),
        ).fetchone()[0],
        "events": mailbox_db.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)
        ).fetchone()[0],
    }

    if terminal:
        with pytest.raises(ValueError, match="terminal"):
            _send(mailbox_db, task_id, wake_requested=True)
        assert mailbox_db.execute(
            "SELECT COUNT(*) FROM task_comments WHERE task_id = ?", (task_id,)
        ).fetchone()[0] == before["comments"]
        assert mailbox_db.execute(
            "SELECT COUNT(*) FROM task_mailbox_messages WHERE task_id = ?",
            (task_id,),
        ).fetchone()[0] == before["messages"]
        assert mailbox_db.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)
        ).fetchone()[0] == before["events"]
    else:
        assert _send(mailbox_db, task_id, wake_requested=True).created is True

    assert kb.get_task(mailbox_db, task_id).status == expected_status


@pytest.mark.parametrize("parent_status", ["done", "archived"])
def test_wake_promotes_todo_when_every_parent_is_terminal(
    mailbox_db, parent_status
):
    parent = kb.create_task(mailbox_db, title="terminal parent")
    task_id = kb.create_task(
        mailbox_db,
        title="dependency-clear target",
        assignee="recipient",
        parents=[parent],
    )
    mailbox_db.execute(
        "UPDATE tasks SET status = ? WHERE id = ?", (parent_status, parent)
    )

    sent = _send(mailbox_db, task_id, wake_requested=True)

    assert sent.created is True
    assert kb.get_task(mailbox_db, task_id).status == "ready"
    wake_events = [
        event for event in kb.list_events(mailbox_db, task_id)
        if event.kind == "mailbox_wake_requested"
    ]
    assert len(wake_events) == 1
    assert wake_events[0].payload["action"] == "promoted"
    assert wake_events[0].payload["message_id"] == sent.message_id


def test_wake_keeps_todo_deferred_when_any_parent_is_unresolved(mailbox_db):
    done_parent = kb.create_task(mailbox_db, title="done parent")
    pending_parent = kb.create_task(mailbox_db, title="pending parent")
    task_id = kb.create_task(
        mailbox_db,
        title="still deferred",
        assignee="recipient",
        parents=[done_parent, pending_parent],
    )
    mailbox_db.execute(
        "UPDATE tasks SET status = 'done' WHERE id = ?", (done_parent,)
    )

    _send(mailbox_db, task_id, wake_requested=True)

    assert kb.get_task(mailbox_db, task_id).status == "todo"


def test_wake_after_intake_fence_sets_pending_on_exact_current_run(mailbox_db):
    task_id, run_id = _running_task(mailbox_db)
    mailbox_db.execute(
        "UPDATE task_runs SET mailbox_accepting = 0 WHERE id = ?", (run_id,)
    )

    sent = _send(mailbox_db, task_id, wake_requested=True)

    task = mailbox_db.execute(
        "SELECT status, current_run_id FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    run = mailbox_db.execute(
        "SELECT mailbox_accepting, mailbox_wake_pending FROM task_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    assert tuple(task) == ("running", run_id)
    assert tuple(run) == (0, 1)
    wake_events = [
        event for event in kb.list_events(mailbox_db, task_id)
        if event.kind == "mailbox_wake_requested"
    ]
    assert len(wake_events) == 1
    assert wake_events[0].run_id == run_id
    assert wake_events[0].payload == {
        "message_id": sent.message_id,
        "action": "wake_pending",
    }


def test_idempotent_wake_retry_does_not_repeat_status_or_event_changes(mailbox_db):
    task_id = kb.create_task(
        mailbox_db, title="idempotent wake", assignee="recipient"
    )
    mailbox_db.execute(
        "UPDATE tasks SET status = 'todo' WHERE id = ?", (task_id,)
    )

    first = _send(mailbox_db, task_id, wake_requested=True)
    second = _send(
        mailbox_db,
        task_id,
        wake_requested=True,
        now=2_000,
    )

    assert first.created is True
    assert second.created is False
    assert second.message_id == first.message_id
    assert kb.get_task(mailbox_db, task_id).status == "ready"
    assert mailbox_db.execute(
        "SELECT COUNT(*) FROM task_events "
        "WHERE task_id = ? AND kind = 'mailbox_wake_requested'",
        (task_id,),
    ).fetchone()[0] == 1


def test_send_mailbox_message_validates_body_by_utf8_bytes(mailbox_db):
    task_id = kb.create_task(mailbox_db, title="body limit")
    exactly_16_kib = "🚀" * 4_096
    assert len(exactly_16_kib.encode("utf-8")) == 16_384
    assert _send(mailbox_db, task_id, body=exactly_16_kib).created is True

    with pytest.raises(ValueError, match="16384"):
        _send(
            mailbox_db,
            task_id,
            body=exactly_16_kib + "x",
            idempotency_key="too-large",
        )
    with pytest.raises(ValueError, match="body"):
        _send(mailbox_db, task_id, body="   ", idempotency_key="blank")


def test_send_mailbox_message_cannot_raise_hard_body_limit(mailbox_db):
    task_id = kb.create_task(mailbox_db, title="hard body limit")

    with pytest.raises(ValueError, match="16384"):
        kb.send_mailbox_message(
            mailbox_db,
            task_id=task_id,
            actor_identity="session:manager",
            actor_kind="manager",
            sender_profile="manager",
            recipient_profile="recipient",
            kind="guidance",
            body="x" * 16_385,
            wake_requested=False,
            idempotency_key="cannot-raise-body-limit",
            max_body_bytes=50_000,
            now=1_000,
        )


def test_send_mailbox_message_force_redacts_before_any_durable_write(
    mailbox_db, monkeypatch
):
    from agent import redact

    task_id = kb.create_task(mailbox_db, title="redaction target")
    raw_token = "sk-proj-mailbox-secret-1234567890"
    raw_body = f"use credential {raw_token} for the check"
    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)

    result = _send(mailbox_db, task_id, body=raw_body)
    row = mailbox_db.execute(
        "SELECT body, comment_id FROM task_mailbox_messages WHERE id = ?",
        (result.message_id,),
    ).fetchone()
    comment_body = mailbox_db.execute(
        "SELECT body FROM task_comments WHERE id = ?", (row["comment_id"],)
    ).fetchone()[0]
    event_payloads = [
        event[0]
        for event in mailbox_db.execute(
            "SELECT COALESCE(payload, '') FROM task_events WHERE task_id = ?",
            (task_id,),
        )
    ]
    audit_rows = [tuple(row) for row in mailbox_db.execute(
        "SELECT * FROM task_mailbox_audit WHERE task_id = ?", (task_id,)
    )]

    assert result.redacted is True
    assert row["body"] == comment_body
    assert row["body"] != raw_body
    assert raw_token not in repr(result)
    assert raw_token not in row["body"]
    assert raw_token not in comment_body
    assert raw_token not in repr(event_payloads)
    assert raw_token not in repr(audit_rows)

    too_large = ("x" * 20_000) + " " + raw_token
    with pytest.raises(ValueError) as exc_info:
        _send(
            mailbox_db,
            task_id,
            body=too_large,
            idempotency_key="redacted-error",
        )
    assert raw_token not in str(exc_info.value)


def test_linked_comment_is_human_visible_but_worker_hidden_and_advances_cursor(mailbox_db):
    task_id = kb.create_task(mailbox_db, title="comment target")
    sent = _send(mailbox_db, task_id)
    linked = mailbox_db.execute(
        "SELECT comment_id FROM task_mailbox_messages WHERE id = ?",
        (sent.message_id,),
    ).fetchone()[0]

    assert [comment.id for comment in kb.list_comments(mailbox_db, task_id)] == [linked]
    assert [
        comment.id for comment in kb.list_comments_since(mailbox_db, task_id, 0)
    ] == [linked]
    visible, cursor = kb.list_worker_comments_since(mailbox_db, task_id, 0)
    assert visible == []
    assert cursor == linked

    normal_id = kb.add_comment(mailbox_db, task_id, "human", "ordinary comment")
    visible, cursor = kb.list_worker_comments_since(mailbox_db, task_id, cursor)
    assert [comment.id for comment in visible] == [normal_id]
    assert cursor == normal_id


def test_worker_comment_cursor_query_uses_task_id_id_index_without_sort(mailbox_db):
    task_id = kb.create_task(mailbox_db, title="cursor query plan")
    kb.add_comment(mailbox_db, task_id, "human", "first")
    plan_rows = mailbox_db.execute(
        "EXPLAIN QUERY PLAN "
        "SELECT c.*, m.id AS mailbox_message_id "
        "FROM task_comments c "
        "LEFT JOIN task_mailbox_messages m ON m.comment_id = c.id "
        "WHERE c.task_id = ? AND c.id > ? "
        "ORDER BY c.id ASC LIMIT ?",
        (task_id, 0, 100),
    ).fetchall()
    details = [row["detail"] for row in plan_rows]

    assert any("idx_comments_task_id" in detail for detail in details)
    assert all("TEMP B-TREE" not in detail.upper() for detail in details)


# ---------------------------------------------------------------------------
# Per-run claims, ordered transitions, token ownership, and leases
# ---------------------------------------------------------------------------


def test_task_claim_opens_mailbox_only_for_new_runs_and_run_end_closes_it(mailbox_db):
    task_id, run_id = _running_task(mailbox_db)
    row = mailbox_db.execute(
        "SELECT mailbox_accepting, mailbox_wake_pending "
        "FROM task_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    assert tuple(row) == (1, 0)

    mailbox_db.execute(
        "UPDATE task_runs SET mailbox_wake_pending = 1 WHERE id = ?", (run_id,)
    )
    assert kb.block_task(mailbox_db, task_id, expected_run_id=run_id) is True
    row = mailbox_db.execute(
        "SELECT mailbox_accepting, mailbox_wake_pending "
        "FROM task_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    assert tuple(row) == (0, 0)


def test_direct_run_ending_update_closes_mailbox_fence(mailbox_db):
    _, run_id = _running_task(mailbox_db)
    mailbox_db.execute(
        "UPDATE task_runs SET mailbox_wake_pending = 1, status = 'crashed', "
        "ended_at = 2_000 WHERE id = ?",
        (run_id,),
    )
    row = mailbox_db.execute(
        "SELECT mailbox_accepting, mailbox_wake_pending "
        "FROM task_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    assert tuple(row) == (0, 0)


def test_claim_mailbox_messages_requires_exact_current_run_and_recipient(mailbox_db):
    task_id, run_id = _running_task(mailbox_db)
    _send(mailbox_db, task_id)

    assert kb.claim_mailbox_messages(
        mailbox_db,
        task_id=task_id,
        run_id=run_id + 1,
        recipient_profile="recipient",
        now=1_100,
    ) == []
    assert kb.claim_mailbox_messages(
        mailbox_db,
        task_id=task_id,
        run_id=run_id,
        recipient_profile="someone-else",
        now=1_100,
    ) == []

    delivery = _claim_one(mailbox_db, task_id, run_id)
    assert delivery.message_id > 0
    assert delivery.run_id == run_id
    assert delivery.body == "please use the new constraint"
    assert delivery.kind == "guidance"
    assert delivery.claim_token


def test_claim_mailbox_messages_honors_row_and_cumulative_utf8_limits(mailbox_db):
    task_id, run_id = _running_task(mailbox_db)
    body = "🚀" * 4_096  # exactly 16 KiB
    for index in range(5):
        _send(
            mailbox_db,
            task_id,
            body=body,
            idempotency_key=f"large-{index}",
            now=1_000 + index,
        )

    deliveries = kb.claim_mailbox_messages(
        mailbox_db,
        task_id=task_id,
        run_id=run_id,
        recipient_profile="recipient",
        now=1_100,
        max_messages=20,
        max_batch_bytes=32_768,
    )
    assert [item.message_id for item in deliveries] == sorted(
        item.message_id for item in deliveries
    )
    assert len(deliveries) == 2
    assert sum(len(item.body.encode("utf-8")) for item in deliveries) <= 32_768


def test_claim_mailbox_messages_always_allows_first_legal_message(mailbox_db):
    task_id, run_id = _running_task(mailbox_db)
    _send(mailbox_db, task_id, body="x" * 16_384)

    deliveries = kb.claim_mailbox_messages(
        mailbox_db,
        task_id=task_id,
        run_id=run_id,
        recipient_profile="recipient",
        now=1_100,
        max_batch_bytes=1,
    )
    assert len(deliveries) == 1


def test_claim_mailbox_messages_defaults_to_twenty_rows(mailbox_db):
    task_id, run_id = _running_task(mailbox_db)
    for index in range(25):
        _send(
            mailbox_db,
            task_id,
            body="small",
            idempotency_key=f"small-{index}",
        )

    deliveries = kb.claim_mailbox_messages(
        mailbox_db,
        task_id=task_id,
        run_id=run_id,
        recipient_profile="recipient",
        now=1_100,
    )
    assert len(deliveries) == 20


def test_claim_mailbox_messages_cannot_raise_hard_row_limit(mailbox_db):
    task_id, run_id = _running_task(mailbox_db)
    for index in range(25):
        _send(
            mailbox_db,
            task_id,
            body="small",
            idempotency_key=f"hard-row-{index}",
        )

    deliveries = kb.claim_mailbox_messages(
        mailbox_db,
        task_id=task_id,
        run_id=run_id,
        recipient_profile="recipient",
        now=1_100,
        max_messages=50,
        max_batch_bytes=50_000,
    )
    assert len(deliveries) == 20


def test_claim_mailbox_messages_cannot_raise_hard_byte_limit(mailbox_db):
    task_id, run_id = _running_task(mailbox_db)
    for index in range(20):
        _send(
            mailbox_db,
            task_id,
            body="x" * 2_000,
            idempotency_key=f"hard-bytes-{index}",
        )

    deliveries = kb.claim_mailbox_messages(
        mailbox_db,
        task_id=task_id,
        run_id=run_id,
        recipient_profile="recipient",
        now=1_100,
        max_messages=50,
        max_batch_bytes=50_000,
    )
    claimed_bytes = sum(len(item.body.encode("utf-8")) for item in deliveries)
    assert claimed_bytes <= 32_768
    assert len(deliveries) == 16


def test_mailbox_transitions_enforce_legal_order_and_exact_batch(mailbox_db):
    task_id, run_id = _running_task(mailbox_db)
    for index in range(3):
        _send(mailbox_db, task_id, idempotency_key=f"message-{index}")
    deliveries = kb.claim_mailbox_messages(
        mailbox_db,
        task_id=task_id,
        run_id=run_id,
        recipient_profile="recipient",
        now=1_100,
    )
    ids = [item.message_id for item in deliveries]
    token = deliveries[0].claim_token
    assert {item.claim_token for item in deliveries} == {token}

    assert kb.mark_mailbox_messages_included(
        mailbox_db,
        message_ids=ids,
        run_id=run_id,
        recipient_profile="recipient",
        claim_token=token,
        now=1_101,
    ) is False
    assert kb.accept_mailbox_messages(
        mailbox_db,
        message_ids=[ids[0], ids[2]],
        run_id=run_id,
        recipient_profile="recipient",
        claim_token=token,
        now=1_102,
    ) is True
    states = {
        row["message_id"]: row["state"]
        for row in mailbox_db.execute(
            "SELECT message_id, state FROM task_mailbox_delivery_attempts "
            "WHERE run_id = ?",
            (run_id,),
        )
    }
    assert states == {
        ids[0]: "accepted_by_steer",
        ids[1]: "claimed_for_run",
        ids[2]: "accepted_by_steer",
    }
    assert kb.accept_mailbox_messages(
        mailbox_db,
        message_ids=[ids[1]],
        run_id=run_id,
        recipient_profile="recipient",
        claim_token=token,
        now=1_103,
    ) is True
    assert kb.mark_mailbox_messages_included(
        mailbox_db,
        message_ids=ids,
        run_id=run_id,
        recipient_profile="recipient",
        claim_token=token,
        now=1_104,
    ) is True
    assert kb.mark_mailbox_messages_responded(
        mailbox_db,
        message_ids=ids,
        run_id=run_id,
        recipient_profile="recipient",
        claim_token=token,
        now=1_105,
    ) is True
    assert kb.accept_mailbox_messages(
        mailbox_db,
        message_ids=ids,
        run_id=run_id,
        recipient_profile="recipient",
        claim_token=token,
        now=1_106,
    ) is False


def test_mailbox_transition_rejects_wrong_token(mailbox_db):
    task_id, run_id = _running_task(mailbox_db)
    _send(mailbox_db, task_id)
    delivery = _claim_one(mailbox_db, task_id, run_id)

    assert kb.accept_mailbox_messages(
        mailbox_db,
        message_ids=[delivery.message_id],
        run_id=run_id,
        recipient_profile="recipient",
        claim_token="stale-token",
        now=1_101,
    ) is False
    assert kb.accept_mailbox_messages(
        mailbox_db,
        message_ids=[delivery.message_id],
        run_id=run_id,
        recipient_profile="recipient",
        claim_token=delivery.claim_token,
        now=1_101,
    ) is True


def test_expired_same_run_claim_is_reclaimed_with_new_token_and_count(mailbox_db):
    task_id, run_id = _running_task(mailbox_db)
    _send(mailbox_db, task_id)
    first = _claim_one(mailbox_db, task_id, run_id, now=1_100)

    assert kb.claim_mailbox_messages(
        mailbox_db,
        task_id=task_id,
        run_id=run_id,
        recipient_profile="recipient",
        now=1_129,
    ) == []
    second = _claim_one(mailbox_db, task_id, run_id, now=1_131)
    assert second.claim_token != first.claim_token
    row = mailbox_db.execute(
        "SELECT attempt_count, state FROM task_mailbox_delivery_attempts "
        "WHERE message_id = ? AND run_id = ?",
        (first.message_id, run_id),
    ).fetchone()
    assert tuple(row) == (2, "claimed_for_run")
    assert kb.accept_mailbox_messages(
        mailbox_db,
        message_ids=[first.message_id],
        run_id=run_id,
        recipient_profile="recipient",
        claim_token=first.claim_token,
        now=1_132,
    ) is False


def test_lease_renewal_is_bounded_and_requires_current_token(mailbox_db):
    task_id, run_id = _running_task(mailbox_db)
    _send(mailbox_db, task_id)
    delivery = _claim_one(mailbox_db, task_id, run_id)

    assert kb.renew_mailbox_message_leases(
        mailbox_db,
        message_ids=[delivery.message_id],
        run_id=run_id,
        recipient_profile="recipient",
        claim_token="wrong",
        lease_seconds=999,
        now=1_120,
    ) is False
    assert kb.renew_mailbox_message_leases(
        mailbox_db,
        message_ids=[delivery.message_id],
        run_id=run_id,
        recipient_profile="recipient",
        claim_token=delivery.claim_token,
        lease_seconds=999,
        now=1_120,
    ) is True
    lease_expires = mailbox_db.execute(
        "SELECT lease_expires FROM task_mailbox_delivery_attempts "
        "WHERE message_id = ? AND run_id = ?",
        (delivery.message_id, run_id),
    ).fetchone()[0]
    assert lease_expires == 1_120 + kb.MAX_MAILBOX_LEASE_SECONDS


def test_completed_delivery_is_redelivered_once_to_a_new_run(mailbox_db):
    task_id, first_run = _running_task(mailbox_db)
    _send(mailbox_db, task_id)
    first = _claim_one(mailbox_db, task_id, first_run)
    assert kb.accept_mailbox_messages(
        mailbox_db,
        message_ids=[first.message_id],
        run_id=first_run,
        recipient_profile="recipient",
        claim_token=first.claim_token,
        now=1_101,
    )
    assert kb.mark_mailbox_messages_included(
        mailbox_db,
        message_ids=[first.message_id],
        run_id=first_run,
        recipient_profile="recipient",
        claim_token=first.claim_token,
        now=1_102,
    )
    assert kb.mark_mailbox_messages_responded(
        mailbox_db,
        message_ids=[first.message_id],
        run_id=first_run,
        recipient_profile="recipient",
        claim_token=first.claim_token,
        now=1_103,
    )

    assert kb.block_task(mailbox_db, task_id, expected_run_id=first_run)
    assert kb.unblock_task(mailbox_db, task_id)
    assert kb.claim_task(mailbox_db, task_id, claimer="test:replacement") is not None
    second_run = mailbox_db.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()[0]
    second = _claim_one(mailbox_db, task_id, int(second_run), now=2_000)

    assert second.message_id == first.message_id
    assert second.run_id != first.run_id
    assert second.claim_token != first.claim_token
    assert mailbox_db.execute(
        "SELECT COUNT(*) FROM task_mailbox_delivery_attempts WHERE message_id = ?",
        (first.message_id,),
    ).fetchone()[0] == 2


def test_old_run_acknowledgement_is_rejected_after_replacement_run(mailbox_db):
    task_id, first_run = _running_task(mailbox_db)
    _send(mailbox_db, task_id)
    first = _claim_one(mailbox_db, task_id, first_run)
    assert kb.block_task(mailbox_db, task_id, expected_run_id=first_run)
    assert kb.unblock_task(mailbox_db, task_id)
    assert kb.claim_task(mailbox_db, task_id, claimer="test:replacement") is not None

    assert kb.accept_mailbox_messages(
        mailbox_db,
        message_ids=[first.message_id],
        run_id=first_run,
        recipient_profile="recipient",
        claim_token=first.claim_token,
        now=1_101,
    ) is False


# ---------------------------------------------------------------------------
# Completion barrier and intake-closing fence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["guidance", "question"])
def test_completion_barrier_commits_audit_and_preserves_task_and_run(
    mailbox_db, kind
):
    task_id, run_id = _running_task(mailbox_db)
    sent = _send(mailbox_db, task_id, kind=kind)

    with pytest.raises(kb.MailboxCompletionBlockedError) as exc_info:
        kb.complete_task(
            mailbox_db,
            task_id,
            summary="must not land",
            expected_run_id=run_id,
        )

    assert exc_info.value.message_ids == [sent.message_id]
    assert exc_info.value.current_run_id == run_id
    task = mailbox_db.execute(
        "SELECT status, result, current_run_id FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    run = mailbox_db.execute(
        "SELECT status, outcome, ended_at, mailbox_accepting "
        "FROM task_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    assert tuple(task) == ("running", None, run_id)
    assert tuple(run) == ("running", None, None, 1)
    events = [
        event for event in kb.list_events(mailbox_db, task_id)
        if event.kind == "completion_blocked_mailbox"
    ]
    assert len(events) == 1
    assert events[0].run_id == run_id
    assert events[0].payload == {
        "message_ids": [sent.message_id],
        "current_run_id": run_id,
    }


def test_info_message_never_blocks_completion(mailbox_db):
    task_id, run_id = _running_task(mailbox_db)
    _send(mailbox_db, task_id, kind="info")

    assert kb.complete_task(
        mailbox_db,
        task_id,
        summary="info did not require a response",
        expected_run_id=run_id,
    ) is True
    assert kb.get_task(mailbox_db, task_id).status == "done"


def test_current_run_response_clears_completion_barrier(mailbox_db):
    task_id, run_id = _running_task(mailbox_db)
    _send(mailbox_db, task_id, kind="guidance")
    _send(
        mailbox_db,
        task_id,
        kind="question",
        idempotency_key="message-2",
    )
    _ack_deliveries(mailbox_db, task_id, run_id)

    assert kb.complete_task(
        mailbox_db,
        task_id,
        summary="responded to every blocking message",
        expected_run_id=run_id,
    ) is True


def test_old_run_response_does_not_clear_current_run_completion_barrier(mailbox_db):
    task_id, first_run_id = _running_task(mailbox_db)
    sent = _send(mailbox_db, task_id, kind="guidance")
    _ack_deliveries(mailbox_db, task_id, first_run_id)
    assert kb.block_task(
        mailbox_db, task_id, reason="replace", expected_run_id=first_run_id
    )
    assert kb.unblock_task(mailbox_db, task_id)
    assert kb.claim_task(
        mailbox_db, task_id, claimer="test:replacement"
    ) is not None
    second_run_id = kb.get_task(mailbox_db, task_id).current_run_id
    assert second_run_id is not None and second_run_id != first_run_id

    with pytest.raises(kb.MailboxCompletionBlockedError) as exc_info:
        kb.complete_task(
            mailbox_db,
            task_id,
            summary="old response is insufficient",
            expected_run_id=second_run_id,
        )

    assert exc_info.value.message_ids == [sent.message_id]
    assert exc_info.value.current_run_id == second_run_id
    assert kb.get_task(mailbox_db, task_id).status == "running"


def test_expected_run_id_none_is_not_a_completion_barrier_override(mailbox_db):
    task_id = kb.create_task(
        mailbox_db, title="manual completion", assignee="recipient"
    )
    sent = _send(mailbox_db, task_id, kind="question")

    with pytest.raises(kb.MailboxCompletionBlockedError) as exc_info:
        kb.complete_task(
            mailbox_db,
            task_id,
            summary="manual path cannot bypass",
            expected_run_id=None,
        )

    assert exc_info.value.message_ids == [sent.message_id]
    assert exc_info.value.current_run_id is None
    assert kb.get_task(mailbox_db, task_id).status == "ready"


def test_try_close_mailbox_intake_requires_every_visible_message_response(mailbox_db):
    task_id, run_id = _running_task(mailbox_db)
    first = _send(mailbox_db, task_id, kind="guidance")
    second = _send(
        mailbox_db,
        task_id,
        kind="info",
        idempotency_key="message-2",
    )

    pending = kb.try_close_mailbox_intake(
        mailbox_db,
        task_id=task_id,
        run_id=run_id,
        recipient_profile="recipient",
    )
    assert pending.closed is False
    assert pending.pending_message_ids == [first.message_id, second.message_id]
    assert mailbox_db.execute(
        "SELECT mailbox_accepting FROM task_runs WHERE id = ?", (run_id,)
    ).fetchone()[0] == 1

    _ack_deliveries(mailbox_db, task_id, run_id)
    closed = kb.try_close_mailbox_intake(
        mailbox_db,
        task_id=task_id,
        run_id=run_id,
        recipient_profile="recipient",
    )
    assert closed.closed is True
    assert closed.pending_message_ids == []
    assert mailbox_db.execute(
        "SELECT mailbox_accepting, mailbox_wake_pending "
        "FROM task_runs WHERE id = ?",
        (run_id,),
    ).fetchone()[:] == (0, 0)


@pytest.mark.parametrize(
    ("run_delta", "recipient"),
    [(1, "recipient"), (0, "other")],
)
def test_try_close_mailbox_intake_requires_exact_live_current_run_and_profile(
    mailbox_db, run_delta, recipient
):
    task_id, run_id = _running_task(mailbox_db)

    result = kb.try_close_mailbox_intake(
        mailbox_db,
        task_id=task_id,
        run_id=run_id + run_delta,
        recipient_profile=recipient,
    )

    assert result.closed is False
    assert result.pending_message_ids == []
    assert mailbox_db.execute(
        "SELECT mailbox_accepting FROM task_runs WHERE id = ?", (run_id,)
    ).fetchone()[0] == 1


def _install_pause_trigger(
    conn: sqlite3.Connection,
    *,
    sql: str,
    entered: threading.Event,
    release: threading.Event,
) -> None:
    def pause() -> int:
        entered.set()
        assert release.wait(timeout=10), "test did not release paused transaction"
        return 0

    conn.create_function("test_pause_txn", 0, pause)
    conn.execute(sql)


def _join_threads(*threads: threading.Thread) -> None:
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive(), f"thread {thread.name} did not finish"


def test_concurrent_send_commits_first_then_completion_is_blocked(tmp_path: Path):
    db_path = tmp_path / "send-first-complete.db"
    setup = kb.connect(db_path)
    task_id, run_id = _running_task(setup)
    setup.close()
    send_entered = threading.Event()
    release_send = threading.Event()
    completion_started = threading.Event()
    outcomes: dict[str, object] = {}

    def send() -> None:
        conn = kb.connect(db_path)
        try:
            _install_pause_trigger(
                conn,
                sql=(
                    "CREATE TEMP TRIGGER pause_send BEFORE INSERT ON task_events "
                    "WHEN NEW.kind = 'mailbox_message_sent' "
                    "BEGIN SELECT test_pause_txn(); END"
                ),
                entered=send_entered,
                release=release_send,
            )
            outcomes["send"] = _send(conn, task_id, wake_requested=True)
        except Exception as exc:  # pragma: no cover - asserted below
            outcomes["send_error"] = exc
        finally:
            conn.close()

    def complete() -> None:
        conn = kb.connect(db_path)
        completion_started.set()
        try:
            outcomes["complete"] = kb.complete_task(
                conn,
                task_id,
                summary="racing completion",
                expected_run_id=run_id,
            )
        except Exception as exc:
            outcomes["complete_error"] = exc
        finally:
            conn.close()

    send_thread = threading.Thread(target=send, name="send-first")
    send_thread.start()
    assert send_entered.wait(timeout=10)
    complete_thread = threading.Thread(target=complete, name="complete-second")
    complete_thread.start()
    assert completion_started.wait(timeout=10)
    release_send.set()
    _join_threads(send_thread, complete_thread)

    assert "send_error" not in outcomes
    assert isinstance(outcomes.get("complete_error"), kb.MailboxCompletionBlockedError)
    verify = kb.connect(db_path)
    try:
        task = kb.get_task(verify, task_id)
        assert task.status == "running"
        assert task.current_run_id == run_id
        run = verify.execute(
            "SELECT status, outcome, ended_at, mailbox_accepting, "
            "mailbox_wake_pending FROM task_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        assert tuple(run) == ("running", None, None, 1, 0)
        assert verify.execute(
            "SELECT COUNT(*) FROM task_comments WHERE task_id = ?", (task_id,)
        ).fetchone()[0] == 1
        assert verify.execute(
            "SELECT COUNT(*) FROM task_mailbox_messages WHERE task_id = ?",
            (task_id,),
        ).fetchone()[0] == 1
        assert verify.execute(
            "SELECT COUNT(*) FROM task_events "
            "WHERE task_id = ? AND kind = 'mailbox_message_sent'",
            (task_id,),
        ).fetchone()[0] == 1
        assert verify.execute(
            "SELECT COUNT(*) FROM task_events "
            "WHERE task_id = ? AND kind = 'mailbox_wake_requested'",
            (task_id,),
        ).fetchone()[0] == 0
        assert verify.execute(
            "SELECT COUNT(*) FROM task_events "
            "WHERE task_id = ? AND kind = 'completed'",
            (task_id,),
        ).fetchone()[0] == 0
        assert verify.execute(
            "SELECT COUNT(*) FROM task_events "
            "WHERE task_id = ? AND kind = 'completion_blocked_mailbox'",
            (task_id,),
        ).fetchone()[0] == 1
    finally:
        verify.close()


def test_concurrent_completion_commits_first_then_send_is_terminal_rejected(
    tmp_path: Path,
):
    db_path = tmp_path / "complete-first-send.db"
    setup = kb.connect(db_path)
    task_id, run_id = _running_task(setup)
    setup.close()
    completion_entered = threading.Event()
    release_completion = threading.Event()
    send_started = threading.Event()
    outcomes: dict[str, object] = {}

    def complete() -> None:
        conn = kb.connect(db_path)
        try:
            _install_pause_trigger(
                conn,
                sql=(
                    "CREATE TEMP TRIGGER pause_complete BEFORE INSERT ON task_events "
                    "WHEN NEW.kind = 'completed' "
                    "BEGIN SELECT test_pause_txn(); END"
                ),
                entered=completion_entered,
                release=release_completion,
            )
            outcomes["complete"] = kb.complete_task(
                conn,
                task_id,
                summary="wins serialization",
                expected_run_id=run_id,
            )
        except Exception as exc:  # pragma: no cover - asserted below
            outcomes["complete_error"] = exc
        finally:
            conn.close()

    def send() -> None:
        conn = kb.connect(db_path)
        send_started.set()
        try:
            outcomes["send"] = _send(conn, task_id, wake_requested=True)
        except Exception as exc:
            outcomes["send_error"] = exc
        finally:
            conn.close()

    complete_thread = threading.Thread(target=complete, name="complete-first")
    complete_thread.start()
    assert completion_entered.wait(timeout=10)
    send_thread = threading.Thread(target=send, name="send-second")
    send_thread.start()
    assert send_started.wait(timeout=10)
    release_completion.set()
    _join_threads(complete_thread, send_thread)

    assert outcomes.get("complete") is True
    assert "complete_error" not in outcomes
    assert isinstance(outcomes.get("send_error"), ValueError)
    assert "terminal" in str(outcomes["send_error"])
    verify = kb.connect(db_path)
    try:
        task = kb.get_task(verify, task_id)
        assert task.status == "done"
        assert task.current_run_id is None
        run = verify.execute(
            "SELECT status, outcome, ended_at, mailbox_accepting, "
            "mailbox_wake_pending FROM task_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        assert run[0:2] == ("done", "completed")
        assert run[2] is not None
        assert run[3:5] == (0, 0)
        assert verify.execute(
            "SELECT COUNT(*) FROM task_comments WHERE task_id = ?", (task_id,)
        ).fetchone()[0] == 0
        assert verify.execute(
            "SELECT COUNT(*) FROM task_mailbox_messages WHERE task_id = ?",
            (task_id,),
        ).fetchone()[0] == 0
        assert verify.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? "
            "AND kind IN ('mailbox_message_sent', 'mailbox_wake_requested')",
            (task_id,),
        ).fetchone()[0] == 0
        assert verify.execute(
            "SELECT COUNT(*) FROM task_events "
            "WHERE task_id = ? AND kind = 'completed'",
            (task_id,),
        ).fetchone()[0] == 1
    finally:
        verify.close()


def test_concurrent_send_commits_first_then_intake_fence_stays_open(
    tmp_path: Path,
):
    db_path = tmp_path / "send-first-fence.db"
    setup = kb.connect(db_path)
    task_id, run_id = _running_task(setup)
    setup.close()
    send_entered = threading.Event()
    release_send = threading.Event()
    fence_started = threading.Event()
    outcomes: dict[str, object] = {}

    def send() -> None:
        conn = kb.connect(db_path)
        try:
            _install_pause_trigger(
                conn,
                sql=(
                    "CREATE TEMP TRIGGER pause_send BEFORE INSERT ON task_events "
                    "WHEN NEW.kind = 'mailbox_message_sent' "
                    "BEGIN SELECT test_pause_txn(); END"
                ),
                entered=send_entered,
                release=release_send,
            )
            outcomes["send"] = _send(conn, task_id, wake_requested=True)
        finally:
            conn.close()

    def fence() -> None:
        conn = kb.connect(db_path)
        fence_started.set()
        try:
            outcomes["fence"] = kb.try_close_mailbox_intake(
                conn,
                task_id=task_id,
                run_id=run_id,
                recipient_profile="recipient",
            )
        finally:
            conn.close()

    send_thread = threading.Thread(target=send, name="send-first")
    send_thread.start()
    assert send_entered.wait(timeout=10)
    fence_thread = threading.Thread(target=fence, name="fence-second")
    fence_thread.start()
    assert fence_started.wait(timeout=10)
    release_send.set()
    _join_threads(send_thread, fence_thread)

    sent = outcomes["send"]
    fence_result = outcomes["fence"]
    assert fence_result.closed is False
    assert fence_result.pending_message_ids == [sent.message_id]
    verify = kb.connect(db_path)
    try:
        row = verify.execute(
            "SELECT mailbox_accepting, mailbox_wake_pending "
            "FROM task_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        assert tuple(row) == (1, 0)
        task = kb.get_task(verify, task_id)
        assert task.status == "running"
        assert task.current_run_id == run_id
        assert verify.execute(
            "SELECT COUNT(*) FROM task_comments WHERE task_id = ?", (task_id,)
        ).fetchone()[0] == 1
        assert verify.execute(
            "SELECT COUNT(*) FROM task_mailbox_messages WHERE task_id = ?",
            (task_id,),
        ).fetchone()[0] == 1
        assert verify.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? "
            "AND kind = 'mailbox_message_sent'",
            (task_id,),
        ).fetchone()[0] == 1
        assert verify.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? "
            "AND kind = 'mailbox_wake_requested'",
            (task_id,),
        ).fetchone()[0] == 0
    finally:
        verify.close()


def test_concurrent_intake_fence_commits_first_then_send_sets_wake_pending(
    tmp_path: Path,
):
    db_path = tmp_path / "fence-first-send.db"
    setup = kb.connect(db_path)
    task_id, run_id = _running_task(setup)
    setup.close()
    fence_entered = threading.Event()
    release_fence = threading.Event()
    send_started = threading.Event()
    outcomes: dict[str, object] = {}

    def fence() -> None:
        conn = kb.connect(db_path)
        try:
            _install_pause_trigger(
                conn,
                sql=(
                    "CREATE TEMP TRIGGER pause_fence "
                    "BEFORE UPDATE OF mailbox_accepting ON task_runs "
                    f"WHEN OLD.id = {run_id} AND NEW.mailbox_accepting = 0 "
                    "BEGIN SELECT test_pause_txn(); END"
                ),
                entered=fence_entered,
                release=release_fence,
            )
            outcomes["fence"] = kb.try_close_mailbox_intake(
                conn,
                task_id=task_id,
                run_id=run_id,
                recipient_profile="recipient",
            )
        finally:
            conn.close()

    def send() -> None:
        conn = kb.connect(db_path)
        send_started.set()
        try:
            outcomes["send"] = _send(conn, task_id, wake_requested=True)
        finally:
            conn.close()

    fence_thread = threading.Thread(target=fence, name="fence-first")
    fence_thread.start()
    assert fence_entered.wait(timeout=10)
    send_thread = threading.Thread(target=send, name="send-second")
    send_thread.start()
    assert send_started.wait(timeout=10)
    release_fence.set()
    _join_threads(fence_thread, send_thread)

    assert outcomes["fence"].closed is True
    assert outcomes["fence"].pending_message_ids == []
    assert outcomes["send"].created is True
    verify = kb.connect(db_path)
    try:
        row = verify.execute(
            "SELECT mailbox_accepting, mailbox_wake_pending "
            "FROM task_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        assert tuple(row) == (0, 1)
        task = kb.get_task(verify, task_id)
        assert task.status == "running"
        assert task.current_run_id == run_id
        assert verify.execute(
            "SELECT COUNT(*) FROM task_comments WHERE task_id = ?", (task_id,)
        ).fetchone()[0] == 1
        assert verify.execute(
            "SELECT COUNT(*) FROM task_mailbox_messages WHERE task_id = ?",
            (task_id,),
        ).fetchone()[0] == 1
        assert verify.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? "
            "AND kind = 'mailbox_message_sent'",
            (task_id,),
        ).fetchone()[0] == 1
        assert verify.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? "
            "AND kind = 'mailbox_wake_requested'",
            (task_id,),
        ).fetchone()[0] == 1
    finally:
        verify.close()
