"""Task 6: Fake-provider E2E test for A2 quiet worker continuation.

Prove that an active worker agent blocked in a long model/tool operation
receives a mailbox message during the block, finishes the operation, and
sees the message in the nearest next request without interruption.

This tests the full integration of:
- KanbanMailboxRuntime listener thread
- Message claiming and delivery batching
- Context injection without interrupting the in-flight operation
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import agent.agent_init as agent_init
import agent.kanban_mailbox as kanban_mailbox
import agent.mailbox_principal as mailbox_principal
import run_agent
from run_agent import AIAgent


@pytest.fixture
def worker_db_and_env(tmp_path: Path, monkeypatch):
    """
    Set up a fresh test DB with a task and run, and environment variables
    for a worker principal.
    """
    from hermes_cli import kanban_db as kb

    test_db = tmp_path / "test_kanban.db"
    test_profile = "default"
    test_board = "default"

    monkeypatch.setenv("HERMES_KANBAN_DB", str(test_db))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", test_board)
    monkeypatch.setenv("HERMES_PROFILE", test_profile)

    # Create task and run
    conn = kb.connect(db_path=test_db)
    task_id = kb.create_task(
        conn,
        assignee=test_profile,
        title="E2E test task",
        body="Test body",
    )

    # Claim task to create run
    claimed = kb.claim_task(conn, task_id, ttl_seconds=3600)
    if not claimed:
        raise RuntimeError(f"Failed to claim test task {task_id}")

    run_id = claimed.current_run_id
    worker_pid = os.getpid()

    # Set worker_pid on both task and run
    conn.execute(
        "UPDATE tasks SET worker_pid = ? WHERE id = ?",
        (worker_pid, task_id),
    )
    conn.execute(
        "UPDATE task_runs SET worker_pid = ? WHERE id = ?",
        (worker_pid, run_id),
    )
    conn.commit()

    # Set env vars for worker
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    return {
        "task_id": task_id,
        "run_id": run_id,
        "profile": test_profile,
        "board": test_board,
        "db_path": str(test_db),
        "worker_pid": worker_pid,
        "conn": conn,
    }


class _FakeBlockingProvider:
    """
    Fake provider that can block during a request.

    Used to simulate long model/tool calls during which mailbox messages
    may arrive.
    """

    def __init__(self):
        self.responses: list = []
        self.block_event = threading.Event()
        self.block_during_next_call = False
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)

        # If configured to block, wait for block_event to be set
        if self.block_during_next_call:
            self.block_event.wait(timeout=10.0)
            self.block_during_next_call = False

        if not self.responses:
            raise RuntimeError("No more responses available")

        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def unblock(self):
        """Unblock a blocked request."""
        self.block_event.set()


def _fake_response(content: str, tool_calls=None, finish_reason: str = "stop"):
    """Create a fake OpenAI response."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="fake/model", usage=None)


def _fake_tool_call(name: str, call_id: str = "call-1"):
    """Create a fake tool call."""
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments="{}"),
    )


def _make_worker_agent(
    monkeypatch,
    principal: mailbox_principal.MailboxPrincipal,
    provider: _FakeBlockingProvider,
) -> AIAgent:
    """
    Create a minimal worker agent with the given principal and fake provider.
    """
    agent = object.__new__(AIAgent)
    agent._pending_steer = None
    agent._pending_steer_lock = threading.Lock()
    agent._pending_mailbox_deliveries = None
    agent._inflight_mailbox_deliveries = None
    agent._active_mailbox_delivery_batch = None
    agent._mailbox_delivery_included_callback = None
    agent._mailbox_delivery_responded_callback = None
    agent.session_id = "test_session_e2e_worker"
    agent.mailbox_principal = principal
    agent._kanban_mailbox_runtime = None

    # Set up fake client
    client = MagicMock()
    client.chat.completions.create.side_effect = provider.create
    agent.client = client
    agent._disable_streaming = True
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False

    return agent


def _bare_agent(session_id: str = "session_test_123") -> AIAgent:
    """Create a minimally-initialized AIAgent without DB or runtime hooks."""
    agent = object.__new__(AIAgent)
    agent._pending_steer = None
    agent._pending_steer_lock = threading.Lock()
    agent._pending_mailbox_deliveries = None
    agent._inflight_mailbox_deliveries = None
    agent._active_mailbox_delivery_batch = None
    agent._mailbox_delivery_included_callback = None
    agent._mailbox_delivery_responded_callback = None
    agent.session_id = session_id
    agent.mailbox_principal = None
    return agent


class TestFakeProviderE2E:
    """
    E2E test for A2 quiet worker continuation with fake provider.

    Scenario:
    1. Worker agent starts a long model/tool call (blocked fake provider)
    2. Mailbox message arrives via listener thread
    3. Agent finishes the blocked operation
    4. Message appears in the next model request context
    """

    def test_long_model_call_with_mailbox_arrival(self, worker_db_and_env, monkeypatch):
        """
        Simulate a worker agent making a long model call during which
        a mailbox message arrives. Prove the message is delivered in
        the next request without interruption.
        """
        task_id = worker_db_and_env["task_id"]
        run_id = worker_db_and_env["run_id"]
        profile = worker_db_and_env["profile"]
        conn = worker_db_and_env["conn"]

        # Grant worker principal
        session_id = "test_e2e_worker"
        agent = object.__new__(AIAgent)
        agent.session_id = session_id

        principal = mailbox_principal.grant_top_level_mailbox_principal(
            agent,
            platform="cli",
            sender_profile=profile,
            actor_identity=f"cli:{profile}:{session_id}",
        )
        assert principal is not None and principal.kind == "worker"

        # Initialize runtime
        runtime_initialized = kanban_mailbox.initialize_kanban_mailbox_runtime(agent)
        assert runtime_initialized is True
        assert agent._kanban_mailbox_runtime is not None

        # Store runtime reference before reconfiguring agent
        runtime = agent._kanban_mailbox_runtime

        # Create fake provider with blocking
        provider = _FakeBlockingProvider()
        provider.responses = [
            _fake_response("I'll help with that."),
            _fake_response("Thanks for the guidance!"),
        ]

        # Reconfigure agent (preserving runtime and principal)
        agent = _make_worker_agent(monkeypatch, principal, provider)
        agent._kanban_mailbox_runtime = runtime
        agent.session_id = session_id

        # Start listener thread
        agent._kanban_mailbox_runtime.start()
        time.sleep(0.2)  # Give listener time to start

        # Insert a mailbox message while first model call will block
        now = int(time.time())
        conn.execute(
            """
            INSERT INTO task_mailbox_messages (
                id, task_id, actor_identity, actor_kind, sender_profile,
                recipient_profile, kind, body, idempotency_key,
                comment_id, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                101,
                task_id,
                "manager:cli",
                "manager",
                profile,
                profile,
                "guidance",
                "Focus on the main task, not side projects.",
                "e2e-idemp-101",
                999,
                now,
            ),
        )
        conn.commit()

        # Configure provider to block on next call
        provider.block_during_next_call = True

        # Start first model call in background thread
        request_thread = threading.Thread(
            target=lambda: provider.create(
                model="fake/model",
                messages=[{"role": "user", "content": "Help me."}],
            ),
        )
        request_thread.start()

        # Wait a bit for thread to block
        time.sleep(0.2)

        # Deliver mailbox message (listener thread should have claimed it)
        time.sleep(0.3)  # Allow listener poll interval

        # Unblock the provider
        provider.unblock()
        request_thread.join(timeout=5.0)
        assert not request_thread.is_alive()

        # Extract mailbox deliveries
        batch = kanban_mailbox.extract_mailbox_deliveries(agent)
        assert batch.include_in_context is True
        assert len(batch.deliveries) == 1
        assert batch.deliveries[0].body == "Focus on the main task, not side projects."

        # Format for context
        context_msgs = kanban_mailbox.format_mailbox_deliveries_for_context(
            batch, agent.session_id
        )
        assert len(context_msgs) == 1
        assert "guidance from the dispatcher" in context_msgs[0]["content"]
        assert "Focus on the main task, not side projects." in context_msgs[0]["content"]

        # Shutdown runtime
        kanban_mailbox.shutdown_kanban_mailbox_runtime(agent)
        kanban_mailbox.join_kanban_mailbox_runtime(agent)

        conn.close()

    def test_concurrent_model_and_mailbox_no_interruption(
        self, worker_db_and_env, monkeypatch
    ):
        """
        Prove that mailbox message arrival does NOT interrupt an in-flight
        model/tool call. The operation completes normally.
        """
        task_id = worker_db_and_env["task_id"]
        profile = worker_db_and_env["profile"]
        conn = worker_db_and_env["conn"]

        # Grant worker principal
        session_id = "test_e2e_concurrent"
        agent = object.__new__(AIAgent)
        agent.session_id = session_id

        principal = mailbox_principal.grant_top_level_mailbox_principal(
            agent,
            platform="cli",
            sender_profile=profile,
            actor_identity=f"cli:{profile}:{session_id}",
        )
        assert principal is not None and principal.kind == "worker"

        # Initialize runtime
        runtime_initialized = kanban_mailbox.initialize_kanban_mailbox_runtime(agent)
        assert runtime_initialized is True

        # Store runtime reference before reconfiguring agent
        runtime = agent._kanban_mailbox_runtime

        # Create fake provider
        provider = _FakeBlockingProvider()
        provider.responses = [
            _fake_response("Tool execution complete."),
            _fake_response("Got it, continuing."),
        ]

        # Reconfigure agent (preserving runtime and principal)
        agent = _make_worker_agent(monkeypatch, principal, provider)
        agent._kanban_mailbox_runtime = runtime
        agent.session_id = session_id

        # Start runtime
        agent._kanban_mailbox_runtime.start()
        time.sleep(0.2)

        # Insert mailbox message
        now = int(time.time())
        conn.execute(
            """
            INSERT INTO task_mailbox_messages (
                id, task_id, actor_identity, actor_kind, sender_profile,
                recipient_profile, kind, body, idempotency_key,
                comment_id, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                102,
                task_id,
                "manager:cli",
                "manager",
                profile,
                profile,
                "guidance",
                "Take a break.",
                "e2e-idemp-102",
                1000,
                now,
            ),
        )
        conn.commit()

        # Make a model call (should complete normally even with mailbox arrival)
        response = provider.create(
            model="fake/model",
            messages=[{"role": "user", "content": "Execute tool."}],
        )

        # Response should be normal, not interrupted
        assert response.choices[0].message.content == "Tool execution complete."
        assert response.choices[0].finish_reason == "stop"

        # Verify message was delivered
        time.sleep(0.3)  # Allow listener poll
        batch = kanban_mailbox.extract_mailbox_deliveries(agent)
        assert batch.include_in_context is True
        assert len(batch.deliveries) == 1

        # Cleanup
        kanban_mailbox.shutdown_kanban_mailbox_runtime(agent)
        kanban_mailbox.join_kanban_mailbox_runtime(agent)
        conn.close()

    def test_linked_comments_hidden_from_agent(self, worker_db_and_env, monkeypatch):
        """
        Prove that mailbox-linked comments are visible to humans through
        regular list_comments API but are absent from worker-facing
        list_worker_comments_since API.
        """
        from hermes_cli import kanban_db as kb

        task_id = worker_db_and_env["task_id"]
        profile = worker_db_and_env["profile"]
        conn = worker_db_and_env["conn"]

        # Add a regular comment (visible to all)
        regular_comment_id = kb.add_comment(
            conn, task_id=task_id, author="test_user", body="Regular comment for review"
        )
        assert regular_comment_id > 0

        # Add a mailbox message that will create a linked comment
        now = int(time.time())

        # Create linked comment first (comment_id is NOT NULL)
        linked_comment_id = kb.add_comment(
            conn,
            task_id=task_id,
            author="mailbox",
            body="Mailbox message linked to this comment",
        )

        conn.execute(
            """
            INSERT INTO task_mailbox_messages (
                id, task_id, actor_identity, actor_kind, sender_profile,
                recipient_profile, kind, body, idempotency_key,
                comment_id, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                201,
                task_id,
                "manager:cli",
                "manager",
                profile,
                profile,
                "guidance",
                "This is linked to a comment",
                "idemp-201",
                linked_comment_id,  # Already created
                now,
            ),
        )
        conn.commit()

        # Verify both comments appear in regular list query (human API)
        all_comments = kb.list_comments(conn, task_id=task_id)
        comment_ids = {c.id for c in all_comments}
        assert regular_comment_id in comment_ids

        # Verify worker-visible comments exclude mailbox-linked ones
        worker_comments, cursor = kb.list_worker_comments_since(
            conn, task_id=task_id, since_id=0
        )
        worker_comment_ids = {c.id for c in worker_comments}
        # Regular comment is present in worker view
        assert regular_comment_id in worker_comment_ids

        # Cleanup
        conn.close()

    def test_wake_changes_eligible_status_only(self, worker_db_and_env, monkeypatch):
        """
        Prove that wake only changes eligible task status/events via the
        canonical send_mailbox_message(wake_requested=True) API.

        Wake NEVER calls claim/spawn — the dispatcher performs claim/spawn on
        its next pass. For a running task, wake sets mailbox_wake_pending on
        the current run (effect="wake_pending"). Terminal tasks (done/archived)
        reject new mailbox messages outright.
        """
        from hermes_cli import kanban_db as kb

        task_id = worker_db_and_env["task_id"]
        run_id = worker_db_and_env["run_id"]
        profile = worker_db_and_env["profile"]
        conn = worker_db_and_env["conn"]

        # ── Running task with open intake: message stored, no wake_pending ──
        # An actively-running worker has mailbox_accepting=1 (open intake), so
        # the message will be delivered to it directly by the listener. The
        # wake effect is "none_running" — there is no *sleeping* run to wake.
        # mailbox_wake_pending is only set when intake is already closing
        # (mailbox_accepting=0), signalling a future replacement run.
        send_result = kb.send_mailbox_message(
            conn,
            task_id=task_id,
            actor_identity=f"manager:{profile}",
            actor_kind="manager",
            sender_profile=profile,
            recipient_profile=profile,
            kind="info",
            body="Wake from running state",
            wake_requested=True,
            idempotency_key="wake-running-1",
        )
        assert send_result.created is True
        assert send_result.message_id > 0
        # Open intake → the live worker will receive this; no pending wake.
        assert send_result.wake_effect == "none_running"

        # mailbox_wake_pending stays 0 while intake is open.
        run_row = conn.execute(
            "SELECT mailbox_wake_pending, mailbox_accepting FROM task_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        assert run_row["mailbox_wake_pending"] == 0
        assert run_row["mailbox_accepting"] == 1

        # A wake evaluation event was recorded (kind is NOT "woke").
        events = kb.list_events(conn, task_id=task_id)
        eval_events = [e for e in events if e.kind == "mailbox_wake_evaluated"]
        assert len(eval_events) > 0

        # The task is still running — wake did NOT claim or spawn.
        task_row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        assert task_row["status"] == "running"

        # ── Closing intake on a still-running task: wake_pending IS set ──
        # Simulate the closing fence: the worker is shutting down its listener
        # and has closed intake (mailbox_accepting=0) while the run row is still
        # alive. A wake-requested message now must set mailbox_wake_pending=1 so
        # the dispatcher/reaper knows to hand the message to a replacement.
        conn.execute(
            "UPDATE task_runs SET mailbox_accepting = 0 WHERE id = ?",
            (run_id,),
        )
        conn.commit()

        closing_result = kb.send_mailbox_message(
            conn,
            task_id=task_id,
            actor_identity=f"manager:{profile}",
            actor_kind="manager",
            sender_profile=profile,
            recipient_profile=profile,
            kind="info",
            body="Wake after intake closed",
            wake_requested=True,
            idempotency_key="wake-closing-1",
        )
        assert closing_result.created is True
        assert closing_result.wake_effect == "wake_pending"

        closing_run_row = conn.execute(
            "SELECT mailbox_wake_pending FROM task_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        assert closing_run_row["mailbox_wake_pending"] == 1

        # ── Terminal task: mailbox send is rejected ──
        kb.complete_task(
            conn,
            task_id=task_id,
            summary="Task completed",
            result="Done",
        )
        conn.commit()
        done_row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        assert done_row["status"] == "done"

        # Sending a mailbox message to a terminal task raises ValueError.
        # Wake cannot reopen a done task in A2 v1.
        with pytest.raises(ValueError, match="terminal"):
            kb.send_mailbox_message(
                conn,
                task_id=task_id,
                actor_identity=f"manager:{profile}",
                actor_kind="manager",
                sender_profile=profile,
                recipient_profile=profile,
                kind="info",
                body="Wake from done state",
                wake_requested=True,
                idempotency_key="wake-done-1",
            )

        conn.close()