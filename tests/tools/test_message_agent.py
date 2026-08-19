"""A2 authorization and trusted-principal contracts for ``message_agent``."""

from __future__ import annotations

import json
import os
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def board(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    db_path = kb.kanban_db_path(board="default")
    from hermes_cli import profiles

    monkeypatch.setattr(
        profiles,
        "profile_exists",
        lambda name: name in {"default", "manager", "recipient", "sender"},
    )
    conn = kb.connect(db_path)
    try:
        yield conn, db_path
    finally:
        conn.close()


def _task(conn, *, assignee="recipient", tenant=None, running=False, todo=False):
    task_id = kb.create_task(
        conn, title=f"target for {assignee}", assignee=assignee, tenant=tenant
    )
    run_id = None
    if todo:
        assert not running
        conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (task_id,))
        conn.commit()
    if running:
        assert kb.claim_task(conn, task_id, claimer="test:worker") is not None
        run_id = int(
            conn.execute(
                "SELECT current_run_id FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()[0]
        )
        conn.execute(
            "UPDATE tasks SET worker_pid = ? WHERE id = ?", (os.getpid(), task_id)
        )
        conn.execute(
            "UPDATE task_runs SET worker_pid = ? WHERE id = ?", (os.getpid(), run_id)
        )
        conn.commit()
    return task_id, run_id


def _manager(*, session_id="manager-session", tenant=None):
    from agent.mailbox_principal import MailboxPrincipal

    return MailboxPrincipal(
        kind="manager",
        actor_identity=f"cli:manager:{session_id}",
        sender_profile="manager",
        session_id=session_id,
        tenant=tenant,
        board=(os.environ.get("HERMES_KANBAN_BOARD") or "default"),
        db_path=str(
            Path(os.environ.get("HERMES_KANBAN_DB") or kb.kanban_db_path()).resolve()
        ),
    )


def _worker(db_path: Path, own_task: str, run_id: int, *, tenant=None):
    from agent.mailbox_principal import MailboxPrincipal

    return MailboxPrincipal(
        kind="worker",
        actor_identity=f"worker:sender:{own_task}:{run_id}",
        sender_profile="sender",
        session_id="worker-session",
        tenant=tenant,
        board="default",
        db_path=str(db_path.resolve()),
        task_id=own_task,
        run_id=run_id,
        worker_pid=os.getpid(),
    )


def _call(trusted_principal, task_id: str, **overrides):
    from tools import agent_manager as am

    args = {
        "agent": "recipient",
        "task_id": task_id,
        "body": "use the new constraint",
        "idempotency_key": "msg-1",
    }
    args.update(overrides)
    return json.loads(
        am._handle_message_agent(args, mailbox_principal=trusted_principal)
    )


def _audit(conn):
    return conn.execute(
        "SELECT * FROM task_mailbox_audit ORDER BY id DESC LIMIT 1"
    ).fetchone()


def test_message_agent_schema_has_no_authority_bearing_sender_fields():
    from tools import agent_manager as am

    props = am._MESSAGE_AGENT_SCHEMA["parameters"]["properties"]
    assert set(props) == {
        "agent", "task_id", "body", "idempotency_key", "kind", "wake", "board"
    }
    assert set(am._MESSAGE_AGENT_SCHEMA["parameters"]["required"]) == {
        "agent", "task_id", "body", "idempotency_key"
    }
    assert not ({"sender", "actor", "principal", "session_id", "run_id"} & set(props))


def test_message_agent_is_registered_and_exposed_by_both_toolsets(monkeypatch):
    from model_tools import get_tool_definitions
    from tools import agent_manager as am
    from tools import registry as registry_module
    from tools.registry import registry
    from toolsets import TOOLSETS

    monkeypatch.setenv("HERMES_KANBAN_TASK", "schema-worker")
    registry_module._check_fn_cache.clear()
    assert "message_agent" in TOOLSETS["agent_manager"]["tools"]
    assert "message_agent" in TOOLSETS["kanban"]["tools"]
    for toolset in ("agent_manager", "kanban"):
        names = {
            item["function"]["name"]
            for item in get_tool_definitions([toolset], quiet_mode=True)
        }
        assert "message_agent" in names


def test_principal_is_frozen_and_agent_defaults_to_none():
    from agent.mailbox_principal import MailboxPrincipal
    from run_agent import AIAgent

    principal = _manager()
    with pytest.raises(FrozenInstanceError):
        principal.kind = "worker"
    agent = object.__new__(AIAgent)
    # The initializer contract is tested without constructing a provider client.
    from agent.agent_init import initialize_mailbox_principal

    initialize_mailbox_principal(agent)
    assert agent.mailbox_principal is None
    assert isinstance(principal, MailboxPrincipal)


@pytest.mark.parametrize(
    ("kind", "wake_expected"),
    [("guidance", 1), ("question", 1), ("info", 0)],
)
def test_default_wake_depends_on_kind(board, kind, wake_expected):
    conn, _ = board
    task_id, _ = _task(conn)
    result = _call(_manager(), task_id, kind=kind)
    assert result["success"] is True
    row = conn.execute(
        "SELECT kind, wake_requested FROM task_mailbox_messages WHERE id = ?",
        (result["message_id"],),
    ).fetchone()
    assert (row["kind"], row["wake_requested"]) == (kind, wake_expected)


def test_explicit_wake_must_be_boolean(board):
    conn, _ = board
    task_id, _ = _task(conn)
    result = _call(_manager(), task_id, wake="yes")
    assert result["success"] is False
    assert conn.execute("SELECT COUNT(*) FROM task_mailbox_messages").fetchone()[0] == 0


@pytest.mark.parametrize(
    ("overrides", "reason_fragment"),
    [
        ({"wake": "yes"}, "wake"),
        ({"task_id": ""}, "required"),
        ({"body": ""}, "body"),
        ({"kind": "urgent"}, "kind"),
        ({"body": "x" * 16_385}, "size"),
    ],
)
def test_trusted_validation_denial_is_committed_once_without_side_effects(
    board, overrides, reason_fragment
):
    conn, _ = board
    task_id, _ = _task(conn, todo=True)
    raw_body = overrides.get("body", "api_key=sk-super-secret-value")
    before = {
        "messages": conn.execute(
            "SELECT COUNT(*) FROM task_mailbox_messages"
        ).fetchone()[0],
        "comments": conn.execute("SELECT COUNT(*) FROM task_comments").fetchone()[0],
        "events": conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0],
        "audits": conn.execute(
            "SELECT COUNT(*) FROM task_mailbox_audit"
        ).fetchone()[0],
        "status": kb.get_task(conn, task_id).status,
    }

    call_task_id = overrides.get("task_id", task_id)
    call_overrides = {key: value for key, value in overrides.items() if key != "task_id"}
    result = _call(
        _manager(), call_task_id, **{"body": raw_body, **call_overrides}
    )

    assert result["success"] is False
    assert result["allowed"] is False
    assert reason_fragment in result["reason"]
    assert conn.execute("SELECT COUNT(*) FROM task_mailbox_audit").fetchone()[0] == (
        before["audits"] + 1
    )
    audit = _audit(conn)
    assert audit["allowed"] == 0
    assert conn.execute("SELECT COUNT(*) FROM task_mailbox_messages").fetchone()[0] == before[
        "messages"
    ]
    assert conn.execute("SELECT COUNT(*) FROM task_comments").fetchone()[0] == before[
        "comments"
    ]
    assert conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == before[
        "events"
    ]
    assert kb.get_task(conn, task_id).status == before["status"]
    if raw_body:
        assert raw_body not in json.dumps(dict(audit))
        assert raw_body not in json.dumps(result)


def test_untrusted_invalid_input_is_audited_as_missing_principal_first(board):
    conn, _ = board
    task_id, _ = _task(conn, todo=True)
    raw_body = "api_key=sk-untrusted-invalid-secret"

    result = _call(None, task_id, body=raw_body, wake="not-a-boolean")

    assert result == {
        "success": False,
        "allowed": False,
        "reason": "missing trusted mailbox principal",
    }
    assert conn.execute("SELECT COUNT(*) FROM task_mailbox_audit").fetchone()[0] == 1
    audit = _audit(conn)
    assert audit["allowed"] == 0
    assert raw_body not in json.dumps(dict(audit))
    assert raw_body not in json.dumps(result)
    assert conn.execute("SELECT COUNT(*) FROM task_mailbox_messages").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM task_comments").fetchone()[0] == 0
    assert kb.get_task(conn, task_id).status == "todo"


def test_idempotency_conflict_commits_one_deny_audit_without_new_side_effects(board):
    conn, _ = board
    task_id, _ = _task(conn)
    first = _call(
        _manager(),
        task_id,
        body="original durable body",
        wake=False,
        idempotency_key="conflict-key",
    )
    assert first["allowed"] is True
    before = {
        "messages": conn.execute(
            "SELECT COUNT(*) FROM task_mailbox_messages"
        ).fetchone()[0],
        "comments": conn.execute("SELECT COUNT(*) FROM task_comments").fetchone()[0],
        "events": conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0],
        "audits": conn.execute(
            "SELECT COUNT(*) FROM task_mailbox_audit"
        ).fetchone()[0],
        "status": kb.get_task(conn, task_id).status,
    }
    raw_body = "api_key=sk-conflicting-secret-value"

    result = _call(
        _manager(),
        task_id,
        body=raw_body,
        wake=False,
        idempotency_key="conflict-key",
    )

    assert result["success"] is False
    assert result["allowed"] is False
    assert result["reason"] == "mailbox idempotency conflict"
    assert conn.execute("SELECT COUNT(*) FROM task_mailbox_audit").fetchone()[0] == (
        before["audits"] + 1
    )
    audit = _audit(conn)
    assert audit["allowed"] == 0
    assert conn.execute("SELECT COUNT(*) FROM task_mailbox_messages").fetchone()[0] == before[
        "messages"
    ]
    assert conn.execute("SELECT COUNT(*) FROM task_comments").fetchone()[0] == before[
        "comments"
    ]
    assert conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == before[
        "events"
    ]
    assert kb.get_task(conn, task_id).status == before["status"]
    assert raw_body not in json.dumps(dict(audit))
    assert raw_body not in json.dumps(result)


def test_manager_can_send_and_wake_dependency_clear_todo(board):
    conn, _ = board
    task_id, _ = _task(conn, todo=True)
    result = _call(_manager(), task_id)
    assert result == {
        "success": True,
        "allowed": True,
        "message_id": result["message_id"],
        "created": True,
        "stored": True,
        "redacted": False,
        "task_id": task_id,
        "agent": "recipient",
        "kind": "guidance",
        "delivery_state": "stored",
        "wake_requested": True,
        "wake_effect": "promoted",
        "task_status": "ready",
    }
    assert kb.get_task(conn, task_id).status == "ready"
    audit = _audit(conn)
    assert audit["allowed"] == 1
    assert audit["message_id"] == result["message_id"]
    assert audit["actor_identity"] == "cli:manager:manager-session"


def test_model_supplied_sender_fields_cannot_spoof_attribution(board):
    conn, _ = board
    task_id, _ = _task(conn)
    result = _call(
        _manager(),
        task_id,
        wake=False,
        sender_profile="root",
        actor_identity="admin",
        principal={"kind": "manager"},
    )
    assert result["allowed"] is True
    row = conn.execute(
        "SELECT actor_identity, sender_profile FROM task_mailbox_messages WHERE id = ?",
        (result["message_id"],),
    ).fetchone()
    assert (row["actor_identity"], row["sender_profile"]) == (
        "cli:manager:manager-session",
        "manager",
    )


def test_session_id_without_principal_is_denied_and_audited(board):
    conn, _ = board
    task_id, _ = _task(conn)
    from model_tools import handle_function_call

    result = json.loads(handle_function_call(
        "message_agent",
        {"agent": "recipient", "task_id": task_id, "body": "secretless", "idempotency_key": "x"},
        session_id="perfectly-valid-session",
        enabled_toolsets=["agent_manager"],
    ))
    assert result["success"] is False
    assert result["allowed"] is False
    assert conn.execute("SELECT COUNT(*) FROM task_mailbox_messages").fetchone()[0] == 0
    assert _audit(conn)["allowed"] == 0


def test_untrusted_board_hint_never_selects_an_audit_database(
    board,
):
    conn, default_db_path = board
    task_id, _ = _task(conn)
    other_db_path = kb.kanban_db_path(board="model-chosen-board")
    raw_body = "must-not-enter-denial-audit"

    result = _call(
        None,
        task_id,
        board="model-chosen-board",
        body=raw_body,
    )

    assert result["allowed"] is False
    assert default_db_path.exists()
    assert not other_db_path.exists()
    audit = _audit(conn)
    assert audit["allowed"] == 0
    assert raw_body not in json.dumps(dict(audit))


def test_delegate_or_aux_with_inherited_worker_env_stays_denied(board, monkeypatch):
    conn, db_path = board
    own_task, run_id = _task(conn, assignee="sender", running=True)
    target, _ = _task(conn)
    monkeypatch.setenv("HERMES_KANBAN_TASK", own_task)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.setenv("HERMES_PROFILE", "sender")

    result = _call(None, target)
    assert result["allowed"] is False
    assert conn.execute("SELECT COUNT(*) FROM task_mailbox_messages").fetchone()[0] == 0


def test_worker_can_send_on_same_board_and_tenant(board):
    conn, db_path = board
    own_task, run_id = _task(conn, assignee="sender", tenant="team-a", running=True)
    target, _ = _task(conn, tenant="team-a")
    result = _call(_worker(db_path, own_task, run_id, tenant="team-a"), target, wake=False)
    assert result["success"] is True
    row = conn.execute(
        "SELECT sender_profile, actor_kind FROM task_mailbox_messages WHERE id = ?",
        (result["message_id"],),
    ).fetchone()
    assert (row["sender_profile"], row["actor_kind"]) == ("sender", "worker")


@pytest.mark.parametrize(
    "mutation", ["wrong_run", "stale_run", "wrong_profile", "wrong_pid"]
)
def test_worker_principal_is_revalidated_at_call_time(board, mutation):
    conn, db_path = board
    own_task, run_id = _task(conn, assignee="sender", running=True)
    target, _ = _task(conn)
    principal = _worker(db_path, own_task, run_id)
    if mutation == "wrong_run":
        principal = type(principal)(**{**principal.__dict__, "run_id": run_id + 1})
    elif mutation == "stale_run":
        conn.execute("UPDATE task_runs SET ended_at = 1, status = 'ended' WHERE id = ?", (run_id,))
        conn.commit()
    elif mutation == "wrong_profile":
        conn.execute("UPDATE task_runs SET profile = 'somebody-else' WHERE id = ?", (run_id,))
        conn.commit()
    else:
        other_pid = os.getpid() + 10_000
        conn.execute("UPDATE tasks SET worker_pid = ? WHERE id = ?", (other_pid, own_task))
        conn.execute("UPDATE task_runs SET worker_pid = ? WHERE id = ?", (other_pid, run_id))
        conn.commit()
    result = _call(principal, target, wake=False)
    assert result["allowed"] is False
    assert _audit(conn)["allowed"] == 0


def test_recipient_must_equal_target_assignee(board):
    conn, _ = board
    task_id, _ = _task(conn, assignee="recipient")
    result = _call(_manager(), task_id, agent="impostor")
    assert result["allowed"] is False
    assert conn.execute("SELECT COUNT(*) FROM task_mailbox_messages").fetchone()[0] == 0


def test_recipient_profile_must_exist_before_message_or_wake(board):
    conn, _ = board
    task_id, _ = _task(conn, assignee="ghost", todo=True)
    result = _call(_manager(), task_id, agent="ghost")
    assert result["allowed"] is False
    assert "profile" in result["reason"]
    assert kb.get_task(conn, task_id).status == "todo"
    assert conn.execute("SELECT COUNT(*) FROM task_mailbox_messages").fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE kind LIKE 'mailbox_%'"
    ).fetchone()[0] == 0
    audit = _audit(conn)
    assert audit["allowed"] == 0
    assert "body" not in audit.keys()


def test_manager_without_tenant_authority_cannot_target_tenant_task(board):
    conn, _ = board
    task_id, _ = _task(conn, tenant="team-a")
    result = _call(_manager(), task_id)
    assert result["allowed"] is False
    assert "tenant" in result["reason"]


def test_worker_cannot_cross_tenant_or_board(board):
    conn, db_path = board
    own_task, run_id = _task(conn, assignee="sender", tenant="team-a", running=True)
    target, _ = _task(conn, tenant="team-b")
    principal = _worker(db_path, own_task, run_id, tenant="team-a")
    assert _call(principal, target, wake=False)["allowed"] is False
    assert _call(principal, target, board="other", wake=False)["allowed"] is False


def test_manager_principal_is_pinned_and_model_cannot_route_cross_board(board):
    conn, db_path = board
    task_id, _ = _task(conn)
    principal = _manager()
    assert principal.board == "default"
    assert principal.db_path == str(db_path.resolve())
    result = _call(principal, task_id, board="other", wake=False)
    assert result["allowed"] is False
    assert "board" in result["reason"]
    assert conn.execute("SELECT COUNT(*) FROM task_mailbox_messages").fetchone()[0] == 0
    assert _audit(conn)["allowed"] == 0


def test_receipt_reports_active_todo_dependency_and_dedup_effects(board):
    conn, _ = board

    running_id, _ = _task(conn, running=True)
    running = _call(_manager(), running_id, idempotency_key="running")
    assert running["wake_effect"] == "none_running"
    assert running["task_status"] == "running"
    assert running["delivery_state"] == "stored"

    parent_id, _ = _task(conn)
    child_id, _ = _task(conn)
    kb.link_tasks(conn, parent_id, child_id)
    blocked = _call(_manager(), child_id, idempotency_key="blocked")
    assert blocked["wake_effect"] == "dependency_blocked"
    assert blocked["task_status"] == "todo"

    todo_id, _ = _task(conn, todo=True)
    first = _call(_manager(), todo_id, idempotency_key="dedup")
    second = _call(_manager(), todo_id, idempotency_key="dedup")
    assert first["wake_effect"] == "promoted"
    assert second["wake_effect"] == "promoted"
    assert second["created"] is False
    assert second["stored"] is False
    assert second["message_id"] == first["message_id"]


def test_receipt_reports_wake_pending_after_closed_running_fence(board):
    conn, _ = board
    task_id, run_id = _task(conn, running=True)
    conn.execute(
        "UPDATE task_runs SET mailbox_accepting = 0 WHERE id = ?", (run_id,)
    )
    conn.commit()
    result = _call(_manager(), task_id, idempotency_key="wake-pending")
    assert result["wake_effect"] == "wake_pending"
    assert result["task_status"] == "running"


def test_dedup_returns_its_own_historical_wake_effect_after_later_pending(board):
    conn, _ = board
    task_id, run_id = _task(conn, running=True)
    first = _call(_manager(), task_id, idempotency_key="first-running-message")
    assert first["wake_effect"] == "none_running"

    conn.execute(
        "UPDATE task_runs SET mailbox_accepting = 0 WHERE id = ?", (run_id,)
    )
    conn.commit()
    later = _call(_manager(), task_id, idempotency_key="later-pending-message")
    assert later["wake_effect"] == "wake_pending"
    before_events = conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)
    ).fetchone()[0]

    retry = _call(_manager(), task_id, idempotency_key="first-running-message")

    assert retry["stored"] is False
    assert retry["message_id"] == first["message_id"]
    assert retry["wake_effect"] == "none_running"
    assert conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)
    ).fetchone()[0] == before_events


def test_worker_wake_default_denied_and_allowlist_enables_it(board, monkeypatch):
    conn, db_path = board
    own_task, run_id = _task(conn, assignee="sender", running=True)
    first, _ = _task(conn)
    principal = _worker(db_path, own_task, run_id)

    denied = _call(principal, first)  # guidance defaults wake=true
    assert denied["allowed"] is False
    assert "wake" in denied["reason"]

    from tools import agent_manager as am
    monkeypatch.setattr(am, "_worker_wake_profiles", lambda: {"sender"})
    allowed = _call(principal, first, idempotency_key="msg-2")
    assert allowed["allowed"] is True


def test_denial_audit_never_contains_raw_body_and_unknown_target_is_nullable(board):
    conn, _ = board
    raw = "api_key=sk-super-secret-value"
    result = _call(None, "missing-task", body=raw)
    assert result["allowed"] is False
    row = _audit(conn)
    assert row["task_id"] is None
    assert raw not in json.dumps(dict(row))
    assert raw not in json.dumps(result)


@pytest.mark.parametrize("field", ["agent", "idempotency_key"])
def test_secret_in_metadata_is_never_persisted(board, field):
    conn, _ = board
    task_id, _ = _task(conn)
    raw = "api_key=sk-super-secret-value"
    result = _call(_manager(), task_id, **{field: raw})
    assert result["allowed"] is False
    dump = "\n".join(
        str(tuple(row))
        for table in (
            "task_mailbox_messages",
            "task_mailbox_audit",
            "task_comments",
            "task_events",
        )
        for row in conn.execute(f"SELECT * FROM {table}").fetchall()
    )
    assert raw not in dump
    assert raw not in json.dumps(result)


def test_authorization_precedes_idempotency_resolution(board):
    conn, _ = board
    task_id, _ = _task(conn)
    first = _call(_manager(), task_id, wake=False)
    assert first["allowed"] is True
    denied = _call(None, task_id, wake=False)
    assert denied["allowed"] is False
    assert conn.execute("SELECT COUNT(*) FROM task_mailbox_audit").fetchone()[0] == 2


def test_allowed_audit_and_message_are_one_atomic_transaction(board):
    conn, _ = board
    task_id, _ = _task(conn)
    conn.execute(
        "CREATE TRIGGER reject_mailbox_allow_audit "
        "BEFORE INSERT ON task_mailbox_audit WHEN NEW.allowed = 1 "
        "BEGIN SELECT RAISE(ABORT, 'audit unavailable'); END"
    )
    result = _call(_manager(), task_id, wake=False)
    assert result["success"] is False
    assert conn.execute("SELECT COUNT(*) FROM task_mailbox_messages").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM task_comments").fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE kind = 'mailbox_message_sent'"
    ).fetchone()[0] == 0


def test_direct_and_tool_search_dispatch_forward_same_principal(board, monkeypatch):
    conn, _ = board
    target1, _ = _task(conn)
    target2, _ = _task(conn)
    from model_tools import handle_function_call

    principal = _manager()
    direct = json.loads(handle_function_call(
        "message_agent",
        {"agent": "recipient", "task_id": target1, "body": "one", "idempotency_key": "direct"},
        mailbox_principal=principal,
        enabled_toolsets=["agent_manager"],
    ))
    assert direct["allowed"] is True

    from tools import tool_search as ts
    monkeypatch.setattr(ts, "resolve_underlying_call", lambda args: (
        "message_agent",
        {"agent": "recipient", "task_id": target2, "body": "two", "idempotency_key": "bridge"},
        None,
    ))
    monkeypatch.setattr(ts, "scoped_deferrable_names", lambda defs: {"message_agent"})
    bridged = json.loads(handle_function_call(
        ts.TOOL_CALL_NAME, {}, mailbox_principal=principal, enabled_toolsets=["agent_manager"]
    ))
    assert bridged["allowed"] is True


def test_runtime_helper_and_both_executor_paths_forward_principal():
    """The sequential path uses invoke_tool; concurrent quiet/nonquiet share
    the two explicit registry calls in tool_executor. Assert all three trusted
    forwarding seams receive the per-agent principal.
    """
    principal = _manager()
    agent = SimpleNamespace(
        mailbox_principal=principal,
        session_id="s",
        valid_tool_names=set(),
        enabled_toolsets=["agent_manager"],
        disabled_toolsets=None,
        _memory_manager=None,
        clarify_callback=None,
    )
    from agent.agent_runtime_helpers import invoke_tool

    with patch("run_agent.handle_function_call", return_value="{}") as dispatch:
        invoke_tool(agent, "message_agent", {}, "task")
    assert dispatch.call_args.kwargs["mailbox_principal"] is principal


def test_top_level_grant_validates_worker_and_never_falls_back_to_manager(board, monkeypatch):
    conn, db_path = board
    own_task, run_id = _task(conn, assignee="sender", running=True)
    from agent.mailbox_principal import grant_top_level_mailbox_principal

    for key, value in {
        "HERMES_KANBAN_TASK": own_task,
        "HERMES_KANBAN_RUN_ID": str(run_id),
        "HERMES_KANBAN_DB": str(db_path),
        "HERMES_KANBAN_BOARD": "default",
        "HERMES_PROFILE": "sender",
    }.items():
        monkeypatch.setenv(key, value)
    agent = SimpleNamespace(mailbox_principal=None, session_id="worker-session")
    principal = grant_top_level_mailbox_principal(
        agent, platform="cli", sender_profile="manager", actor_identity="cli:manager"
    )
    assert principal.kind == "worker"
    assert principal.task_id == own_task
    assert principal.worker_pid == os.getpid()

    conn.execute("UPDATE task_runs SET profile = 'stale' WHERE id = ?", (run_id,))
    conn.commit()
    other = SimpleNamespace(mailbox_principal=None, session_id="valid-session")
    assert grant_top_level_mailbox_principal(
        other, platform="cli", sender_profile="manager", actor_identity="cli:manager"
    ) is None
    assert other.mailbox_principal is None


def test_worker_grant_retries_until_dispatcher_registers_pid(board, monkeypatch):
    conn, db_path = board
    own_task, run_id = _task(conn, assignee="sender", running=True)
    conn.execute("UPDATE tasks SET worker_pid = 0 WHERE id = ?", (own_task,))
    conn.execute("UPDATE task_runs SET worker_pid = 0 WHERE id = ?", (run_id,))
    conn.commit()
    for key, value in {
        "HERMES_KANBAN_TASK": own_task,
        "HERMES_KANBAN_RUN_ID": str(run_id),
        "HERMES_KANBAN_DB": str(db_path),
        "HERMES_KANBAN_BOARD": "default",
        "HERMES_PROFILE": "sender",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("HERMES_KANBAN_BUSY_TIMEOUT_MS", "200")

    from agent import mailbox_principal as mailbox

    clock = [0.0]
    sleeps = []
    probes = []
    real_probe = mailbox._worker_principal_from_environment

    def _record_probe(session_id):
        probe = real_probe(session_id)
        probes.append(getattr(probe, "state", None))
        return probe

    def _register_pid_after_100ms(delay):
        sleeps.append(delay)
        clock[0] += delay
        if clock[0] >= 0.15:
            conn.execute(
                "UPDATE tasks SET worker_pid = ? WHERE id = ?",
                (os.getpid(), own_task),
            )
            conn.execute(
                "UPDATE task_runs SET worker_pid = ? WHERE id = ?",
                (os.getpid(), run_id),
            )
            conn.commit()

    monkeypatch.setattr(mailbox, "_worker_principal_from_environment", _record_probe)
    with patch("time.monotonic", side_effect=lambda: clock[0]), patch(
        "time.sleep", side_effect=_register_pid_after_100ms
    ):
        agent = SimpleNamespace(mailbox_principal=None, session_id="worker-session")
        principal = mailbox.grant_top_level_mailbox_principal(
            agent,
            platform="cli",
            sender_profile="sender",
            actor_identity="worker",
        )

    assert principal is not None
    assert principal.kind == "worker"
    assert principal.worker_pid == os.getpid()
    assert probes == ["pending_registration"] * 3 + ["granted"]
    assert sleeps == [pytest.approx(0.05)] * 3
    assert clock[0] == pytest.approx(0.15)


def test_worker_registration_deadline_starts_after_first_pending_probe(
    board, monkeypatch
):
    _, db_path = board
    monkeypatch.setenv("HERMES_KANBAN_TASK", "pending-task")
    monkeypatch.setenv("HERMES_KANBAN_BUSY_TIMEOUT_MS", "200")
    from agent import mailbox_principal as mailbox

    granted = mailbox.MailboxPrincipal(
        kind="worker",
        actor_identity="worker:sender:pending-task:1",
        sender_profile="sender",
        session_id="worker-session",
        board="default",
        db_path=str(db_path),
        task_id="pending-task",
        run_id=1,
        worker_pid=os.getpid(),
    )
    responses = [
        mailbox.WorkerPrincipalProbe("pending_registration"),
        mailbox.WorkerPrincipalProbe("granted", granted),
    ]
    clock = [0.0]
    sleeps = []

    def _blocking_first_probe(_session_id):
        probe = responses.pop(0)
        if probe.state == "pending_registration":
            clock[0] += 0.3
        return probe

    def _advance(delay):
        sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setattr(
        mailbox, "_worker_principal_from_environment", _blocking_first_probe
    )
    with patch("time.monotonic", side_effect=lambda: clock[0]), patch(
        "time.sleep", side_effect=_advance
    ):
        agent = SimpleNamespace(mailbox_principal=None, session_id="worker-session")
        principal = mailbox.grant_top_level_mailbox_principal(
            agent,
            platform="cli",
            sender_profile="sender",
            actor_identity="worker",
        )

    assert principal is granted
    assert responses == []
    assert sleeps == [pytest.approx(0.05)]
    assert clock[0] == pytest.approx(0.35)


def test_manager_grant_never_polls_or_sleeps(board, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from agent import mailbox_principal as mailbox

    with patch("time.monotonic", side_effect=AssertionError("manager path polled")), patch(
        "time.sleep", side_effect=AssertionError("manager path slept")
    ):
        agent = SimpleNamespace(mailbox_principal=None, session_id="manager-session")
        principal = mailbox.grant_top_level_mailbox_principal(
            agent,
            platform="cli",
            sender_profile="manager",
            actor_identity="manager",
        )

    assert principal is not None
    assert principal.kind == "manager"


def test_mismatched_nested_worker_grant_denies_without_retry(board, monkeypatch):
    conn, db_path = board
    own_task, run_id = _task(conn, assignee="sender", running=True)
    other_pid = os.getpid() + 10_000
    conn.execute("UPDATE tasks SET worker_pid = ? WHERE id = ?", (other_pid, own_task))
    conn.execute(
        "UPDATE task_runs SET worker_pid = ? WHERE id = ?", (other_pid, run_id)
    )
    conn.commit()
    for key, value in {
        "HERMES_KANBAN_TASK": own_task,
        "HERMES_KANBAN_RUN_ID": str(run_id),
        "HERMES_KANBAN_DB": str(db_path),
        "HERMES_KANBAN_BOARD": "default",
        "HERMES_PROFILE": "sender",
    }.items():
        monkeypatch.setenv(key, value)

    from agent import mailbox_principal as mailbox

    clock = [0.0]
    sleeps = []
    probes = []
    real_probe = mailbox._worker_principal_from_environment

    def _record_probe(session_id):
        probe = real_probe(session_id)
        probes.append(getattr(probe, "state", None))
        return probe

    def _advance_clock(delay):
        sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setattr(mailbox, "_worker_principal_from_environment", _record_probe)
    with patch("time.monotonic", side_effect=lambda: clock[0]), patch(
        "time.sleep", side_effect=_advance_clock
    ):
        child = SimpleNamespace(mailbox_principal=None, session_id="nested-child")
        principal = mailbox.grant_top_level_mailbox_principal(
            child,
            platform="cli",
            sender_profile="sender",
            actor_identity="child",
        )

    assert principal is None
    assert child.mailbox_principal is None
    assert probes == ["denied"]
    assert sleeps == []
    assert clock[0] == 0


@pytest.mark.parametrize("mutation", ["task", "run", "profile"])
def test_invalid_worker_identity_denies_without_retry(board, monkeypatch, mutation):
    conn, db_path = board
    own_task, run_id = _task(conn, assignee="sender", running=True)
    worker_env = {
        "HERMES_KANBAN_TASK": own_task,
        "HERMES_KANBAN_RUN_ID": str(run_id),
        "HERMES_KANBAN_DB": str(db_path),
        "HERMES_KANBAN_BOARD": "default",
        "HERMES_PROFILE": "sender",
    }
    if mutation == "task":
        worker_env["HERMES_KANBAN_TASK"] = "missing-task"
    elif mutation == "run":
        worker_env["HERMES_KANBAN_RUN_ID"] = str(run_id + 1)
    else:
        worker_env["HERMES_PROFILE"] = "recipient"
    for key, value in worker_env.items():
        monkeypatch.setenv(key, value)

    from agent import mailbox_principal as mailbox

    sleeps = []
    probes = []
    real_probe = mailbox._worker_principal_from_environment

    def _record_probe(session_id):
        probe = real_probe(session_id)
        probes.append(getattr(probe, "state", None))
        return probe

    monkeypatch.setattr(mailbox, "_worker_principal_from_environment", _record_probe)
    with patch("time.sleep", side_effect=lambda delay: sleeps.append(delay)):
        agent = SimpleNamespace(mailbox_principal=None, session_id="invalid-worker")
        principal = mailbox.grant_top_level_mailbox_principal(
            agent,
            platform="cli",
            sender_profile="sender",
            actor_identity="invalid-worker",
        )

    assert principal is None
    assert agent.mailbox_principal is None
    assert probes == ["denied"]
    assert sleeps == []


def test_nested_child_pid_cannot_claim_root_worker_capability(board, monkeypatch):
    conn, db_path = board
    own_task, run_id = _task(conn, assignee="sender", running=True)
    other_pid = os.getpid() + 10_000
    conn.execute("UPDATE tasks SET worker_pid = ? WHERE id = ?", (other_pid, own_task))
    conn.execute("UPDATE task_runs SET worker_pid = ? WHERE id = ?", (other_pid, run_id))
    conn.commit()
    for key, value in {
        "HERMES_KANBAN_TASK": own_task,
        "HERMES_KANBAN_RUN_ID": str(run_id),
        "HERMES_KANBAN_DB": str(db_path),
        "HERMES_KANBAN_BOARD": "default",
        "HERMES_PROFILE": "sender",
    }.items():
        monkeypatch.setenv(key, value)
    from agent.mailbox_principal import grant_top_level_mailbox_principal

    child = SimpleNamespace(mailbox_principal=None, session_id="nested-child")
    assert grant_top_level_mailbox_principal(
        child, platform="cli", sender_profile="sender", actor_identity="child"
    ) is None
    assert child.mailbox_principal is None


def test_board_or_db_pin_alone_still_grants_manager(board, monkeypatch):
    _, db_path = board
    from agent.mailbox_principal import grant_top_level_mailbox_principal

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    agent = SimpleNamespace(mailbox_principal=None, session_id="manager-session")
    principal = grant_top_level_mailbox_principal(
        agent, platform="gateway", sender_profile="manager", actor_identity="gateway:x"
    )
    assert principal.kind == "manager"
    assert principal.board == "default"
    assert principal.db_path == str(db_path.resolve())


def test_task_marker_with_missing_worker_companions_never_grants_manager(monkeypatch):
    from agent.mailbox_principal import grant_top_level_mailbox_principal

    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-only")
    for key in ("HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(key, raising=False)
    agent = SimpleNamespace(mailbox_principal=None, session_id="valid-manager-session")
    assert grant_top_level_mailbox_principal(
        agent, platform="gateway", sender_profile="manager", actor_identity="gateway:x"
    ) is None
    assert agent.mailbox_principal is None


@pytest.mark.parametrize("presentation", ["full", "off"], ids=["chat-q", "chat-Q"])
def test_real_cli_init_grants_pid_bound_worker_and_visible_tool(
    board, monkeypatch, presentation
):
    conn, db_path = board
    # The dispatcher claims under ITS profile — the run row's profile column
    # is stamped at claim time, so the worker env (HERMES_PROFILE=sender)
    # must be in place before _task() claims, matching production order.
    for key, value in {
        "HERMES_KANBAN_DB": str(db_path),
        "HERMES_KANBAN_BOARD": "default",
        "HERMES_PROFILE": "sender",
    }.items():
        monkeypatch.setenv(key, value)
    own_task, run_id = _task(conn, assignee="sender", running=True)
    monkeypatch.setenv("HERMES_KANBAN_TASK", own_task)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    import cli as cli_mod
    from hermes_cli import mcp_startup
    from model_tools import get_tool_definitions

    shell = cli_mod.HermesCLI(compact=True, toolsets=["kanban"])
    shell.tool_progress_mode = presentation
    shell._session_db = object()
    shell._resumed = False
    shell.conversation_history = []
    shell._install_tool_callbacks = lambda: None
    shell._ensure_tirith_security = lambda: None
    shell._ensure_runtime_credentials = lambda: True
    monkeypatch.setattr(cli_mod, "_prepare_deferred_agent_startup", lambda: None)
    monkeypatch.setattr(mcp_startup, "wait_for_mcp_discovery", lambda timeout=0.75: None)

    def _fake_agent(*_args, **kwargs):
        names = {
            item["function"]["name"]
            for item in get_tool_definitions(
                kwargs.get("enabled_toolsets"), quiet_mode=True
            )
        }
        return SimpleNamespace(
            mailbox_principal=None,
            session_id=kwargs["session_id"],
            valid_tool_names=names,
        )

    monkeypatch.setattr(cli_mod, "AIAgent", _fake_agent)
    assert shell._init_agent() is True
    assert shell.agent.mailbox_principal.kind == "worker"
    assert shell.agent.mailbox_principal.worker_pid == os.getpid()
    assert "message_agent" in shell.agent.valid_tool_names


def test_manager_root_but_delegate_and_auxiliary_default_deny_under_worker_env(
    board, monkeypatch
):
    _, db_path = board
    from agent.agent_init import initialize_mailbox_principal
    from agent.mailbox_principal import grant_top_level_mailbox_principal
    from model_tools import get_tool_definitions

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    manager = SimpleNamespace(mailbox_principal=None, session_id="manager-root")
    grant_top_level_mailbox_principal(
        manager,
        platform="cli",
        sender_profile="manager",
        actor_identity="cli:manager:manager-root",
    )
    assert manager.mailbox_principal.kind == "manager"

    monkeypatch.setenv("HERMES_KANBAN_TASK", "inherited-task")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    for session_id in ("delegate-valid-session", "aux-valid-session"):
        child = SimpleNamespace(session_id=session_id)
        initialize_mailbox_principal(child)
        visible = {
            item["function"]["name"]
            for item in get_tool_definitions(["kanban"], quiet_mode=True)
        }
        assert "message_agent" in visible
        assert child.mailbox_principal is None


def test_mailbox_config_defaults_are_bounded_and_have_no_enabled_toggle():
    from hermes_cli.config import DEFAULT_CONFIG

    mailbox = DEFAULT_CONFIG["kanban"]["mailbox"]
    assert mailbox == {
        "poll_interval_seconds": 0.5,
        "lease_seconds": 30,
        "max_body_bytes": 16_384,
        "max_batch_messages": 20,
        "max_batch_bytes": 32_768,
        "worker_wake_profiles": [],
    }
    assert "enabled" not in mailbox
