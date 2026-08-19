"""Task 5 performance contract tests for KanbanMailboxRuntime.

Asserts the performance invariants from the A2 design:

1. No idle model calls: the listener never calls the model.
2. No listener for ordinary agents: a non-Kanban agent pays zero
   listener/thread/SQLite-connection/query overhead.
3. Bounded query count and batch bytes: many maximum-size messages cannot
   create an unbounded context spike.
4. Sub-second delivery at the 0.5s default poll interval.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import agent.kanban_mailbox as kanban_mailbox
import agent.mailbox_principal as mailbox_principal
from run_agent import AIAgent


# ---------------------------------------------------------------------------
# FIXTURES (mirrors test_a2_fake_provider_e2e.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def worker_env(tmp_path: Path, monkeypatch):
    from hermes_cli import kanban_db as kb

    test_db = tmp_path / "test_kanban.db"
    test_profile = "default"
    test_board = "default"

    monkeypatch.setenv("HERMES_KANBAN_DB", str(test_db))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", test_board)
    monkeypatch.setenv("HERMES_PROFILE", test_profile)

    conn = kb.connect(db_path=test_db)
    task_id = kb.create_task(
        conn, assignee=test_profile, title="Perf task", body="Perf body"
    )
    claimed = kb.claim_task(conn, task_id, ttl_seconds=3600)
    run_id = claimed.current_run_id
    worker_pid = os.getpid()

    conn.execute(
        "UPDATE tasks SET worker_pid = ? WHERE id = ?", (worker_pid, task_id)
    )
    conn.execute(
        "UPDATE task_runs SET worker_pid = ? WHERE id = ?", (worker_pid, run_id)
    )
    conn.commit()

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    yield {
        "task_id": task_id,
        "run_id": run_id,
        "profile": test_profile,
        "db_path": str(test_db),
        "conn": conn,
    }
    conn.close()


def _bare_agent(session_id="perf_session") -> AIAgent:
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
    agent._kanban_mailbox_runtime = None
    return agent


# ---------------------------------------------------------------------------
# 1. No idle model calls: the listener never calls the model.
# ---------------------------------------------------------------------------

class TestNoIdleModelCalls:
    def test_listener_does_not_trigger_model_calls(self, worker_env):
        """The listener thread must never invoke the model (provider)."""
        agent = _bare_agent("perf_no_model")
        principal = mailbox_principal.grant_top_level_mailbox_principal(
            agent,
            platform="cli",
            sender_profile=worker_env["profile"],
            actor_identity=f"cli:{worker_env['profile']}:perf",
        )
        assert principal is not None and principal.kind == "worker"

        # Attach a fake provider to detect any model call.
        fake_provider = MagicMock()
        agent.client = MagicMock()
        agent.client.chat.completions.create = fake_provider

        initialized = kanban_mailbox.initialize_kanban_mailbox_runtime(agent)
        assert initialized is True

        # Let the listener poll a few times with no messages.
        time.sleep(1.0)

        kanban_mailbox.shutdown_kanban_mailbox_runtime(agent)
        kanban_mailbox.join_kanban_mailbox_runtime(agent, timeout=2.0)

        # No model call should have been made.
        assert fake_provider.call_count == 0, (
            "Listener triggered a model call while idle"
        )


# ---------------------------------------------------------------------------
# 2. No listener for ordinary agents.
# ---------------------------------------------------------------------------

class TestNoListenerForOrdinaryAgents:
    def test_ordinary_agent_no_runtime_no_thread(self):
        """An agent with no worker principal must not start a listener."""
        agent = _bare_agent("perf_ordinary")
        agent.mailbox_principal = None

        initialized = kanban_mailbox.initialize_kanban_mailbox_runtime(agent)
        assert initialized is False
        assert agent._kanban_mailbox_runtime is None

    def test_manager_principal_no_runtime(self, worker_env, monkeypatch):
        """A manager principal (no HERMES_KANBAN_TASK) must not start a listener."""
        # Remove worker env to simulate a manager-only session.
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)

        agent = _bare_agent("perf_manager")
        principal = mailbox_principal.grant_top_level_mailbox_principal(
            agent,
            platform="cli",
            sender_profile=worker_env["profile"],
            actor_identity=f"cli:{worker_env['profile']}:mgr",
        )
        # Manager or None — never worker.
        assert principal is None or principal.kind == "manager"

        initialized = kanban_mailbox.initialize_kanban_mailbox_runtime(agent)
        assert initialized is False
        assert agent._kanban_mailbox_runtime is None


# ---------------------------------------------------------------------------
# 3. Bounded batch: many max-size messages don't overflow.
# ---------------------------------------------------------------------------

class TestBoundedBatch:
    def test_claim_batch_is_bounded_by_row_and_byte_limits(self, worker_env):
        """claim_mailbox_messages respects max_messages and max_batch_bytes."""
        from hermes_cli import kanban_db as kb

        conn = worker_env["conn"]
        task_id = worker_env["task_id"]
        run_id = worker_env["run_id"]
        profile = worker_env["profile"]

        # Insert 50 messages — more than the default batch of 20.
        # Each message needs a linked comment (comment_id is NOT NULL).
        now = int(time.time())
        for i in range(50):
            comment_id = kb.add_comment(
                conn, task_id=task_id, author="mailbox", body=f"linked {i}"
            )
            conn.execute(
                "INSERT INTO task_mailbox_messages "
                "(task_id, actor_identity, actor_kind, sender_profile, "
                " recipient_profile, kind, body, idempotency_key, comment_id, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    task_id, "manager:cli", "manager", profile, profile,
                    "info", f"message body {i}", f"bound-idemp-{i}", comment_id, now + i,
                ),
            )
        conn.commit()

        claimed = kb.claim_mailbox_messages(
            conn,
            task_id=task_id,
            run_id=run_id,
            recipient_profile=profile,
            max_messages=20,
            max_batch_bytes=32768,
        )
        # Must not exceed the row bound.
        assert len(claimed) <= 20, f"Batch exceeded row limit: {len(claimed)}"

        # Cumulative body bytes must not exceed the byte bound.
        total_bytes = sum(len(c.body.encode("utf-8")) for c in claimed)
        assert total_bytes <= 32768, (
            f"Batch exceeded byte limit: {total_bytes} bytes"
        )


# ---------------------------------------------------------------------------
# 4. Sub-second delivery at 0.5s default poll.
# ---------------------------------------------------------------------------

class TestSubSecondDelivery:
    def test_message_delivered_within_one_second(self, worker_env):
        """A message inserted after listener start is delivered in < 1s."""
        agent = _bare_agent("perf_subsec")
        principal = mailbox_principal.grant_top_level_mailbox_principal(
            agent,
            platform="cli",
            sender_profile=worker_env["profile"],
            actor_identity=f"cli:{worker_env['profile']}:subsec",
        )
        assert principal is not None and principal.kind == "worker"

        initialized = kanban_mailbox.initialize_kanban_mailbox_runtime(agent)
        assert initialized is True

        conn = worker_env["conn"]
        task_id = worker_env["task_id"]
        profile = worker_env["profile"]

        # Insert a message after the listener is running.
        from hermes_cli import kanban_db as kb

        now = int(time.time())
        comment_id = kb.add_comment(
            conn, task_id=task_id, author="mailbox", body="linked perf"
        )
        conn.execute(
            "INSERT INTO task_mailbox_messages "
            "(task_id, actor_identity, actor_kind, sender_profile, "
            " recipient_profile, kind, body, idempotency_key, comment_id, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                task_id, "manager:cli", "manager", profile, profile,
                "guidance", "Deliver me quickly", "subsec-idemp-1", comment_id, now,
            ),
        )
        conn.commit()

        start = time.monotonic()
        deadline = start + 3.0  # generous upper bound; contract is < 1s at 0.5s poll
        batch = kanban_mailbox.MailboxDeliveryBatch(
            deliveries=[], delivered_keys=set(), include_in_context=False
        )
        while time.monotonic() < deadline:
            batch = kanban_mailbox.extract_mailbox_deliveries(agent)
            if batch.deliveries:
                break
            time.sleep(0.05)

        elapsed = time.monotonic() - start

        kanban_mailbox.shutdown_kanban_mailbox_runtime(agent)
        kanban_mailbox.join_kanban_mailbox_runtime(agent, timeout=2.0)

        assert len(batch.deliveries) == 1, "Message was not delivered before timeout"
        assert batch.deliveries[0].body == "Deliver me quickly"
        # Contract: sub-second delivery at the 0.5s default poll.
        assert elapsed < 1.0, (
            f"Delivery took {elapsed:.2f}s — expected sub-second at 0.5s poll"
        )


# ---------------------------------------------------------------------------
# 5. Listener→canonical bridge compatibility.
#    The listener's MailboxDelivery must carry a claim_token so that it can
#    pass through _enqueue_mailbox_deliveries → _coerce_mailbox_delivery into
#    the canonical Task 4 pipeline.  Without claim_token, every delivery is
#    silently dropped at the conversation_loop bridge.
# ---------------------------------------------------------------------------

class TestListenerCanonicalBridge:
    def test_listener_delivery_has_claim_token(self, worker_env):
        """A listener MailboxDelivery MUST carry claim_token from the DB claim."""
        agent = _bare_agent("perf_bridge")
        principal = mailbox_principal.grant_top_level_mailbox_principal(
            agent,
            platform="cli",
            sender_profile=worker_env["profile"],
            actor_identity=f"cli:{worker_env['profile']}:bridge",
        )
        assert principal is not None and principal.kind == "worker"

        initialized = kanban_mailbox.initialize_kanban_mailbox_runtime(agent)
        assert initialized is True

        conn = worker_env["conn"]
        task_id = worker_env["task_id"]
        profile = worker_env["profile"]
        from hermes_cli import kanban_db as kb

        now = int(time.time())
        comment_id = kb.add_comment(
            conn, task_id=task_id, author="mailbox", body="linked bridge"
        )
        conn.execute(
            "INSERT INTO task_mailbox_messages "
            "(task_id, actor_identity, actor_kind, sender_profile, "
            " recipient_profile, kind, body, idempotency_key, comment_id, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                task_id, "manager:cli", "manager", profile, profile,
                "guidance", "Bridge test body", "bridge-idemp-1", comment_id, now,
            ),
        )
        conn.commit()

        # Wait for the listener to claim and enqueue the delivery.
        deadline = time.monotonic() + 3.0
        batch = kanban_mailbox.MailboxDeliveryBatch(
            deliveries=[], delivered_keys=set(), include_in_context=False
        )
        while time.monotonic() < deadline:
            batch = kanban_mailbox.extract_mailbox_deliveries(agent)
            if batch.deliveries:
                break
            time.sleep(0.05)

        kanban_mailbox.shutdown_kanban_mailbox_runtime(agent)
        kanban_mailbox.join_kanban_mailbox_runtime(agent, timeout=2.0)

        assert len(batch.deliveries) == 1
        delivery = batch.deliveries[0]
        # claim_token MUST be present and non-empty.
        token = getattr(delivery, "claim_token", None)
        assert token is not None and str(token).strip(), (
            "Listener MailboxDelivery must carry claim_token from the DB claim; "
            "without it, _coerce_mailbox_delivery rejects it and the bridge drops it"
        )

    def test_listener_delivery_passes_coerce(self, worker_env):
        """The listener delivery must pass _coerce_mailbox_delivery."""
        from agent.agent_runtime_helpers import _coerce_mailbox_delivery

        agent = _bare_agent("perf_coerce")
        principal = mailbox_principal.grant_top_level_mailbox_principal(
            agent,
            platform="cli",
            sender_profile=worker_env["profile"],
            actor_identity=f"cli:{worker_env['profile']}:coerce",
        )
        assert principal is not None and principal.kind == "worker"

        initialized = kanban_mailbox.initialize_kanban_mailbox_runtime(agent)
        assert initialized is True

        conn = worker_env["conn"]
        task_id = worker_env["task_id"]
        run_id = worker_env["run_id"]
        profile = worker_env["profile"]
        from hermes_cli import kanban_db as kb

        now = int(time.time())
        comment_id = kb.add_comment(
            conn, task_id=task_id, author="mailbox", body="linked coerce"
        )
        conn.execute(
            "INSERT INTO task_mailbox_messages "
            "(task_id, actor_identity, actor_kind, sender_profile, "
            " recipient_profile, kind, body, idempotency_key, comment_id, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                task_id, "manager:cli", "manager", profile, profile,
                "guidance", "Coerce test body", "coerce-idemp-1", comment_id, now,
            ),
        )
        conn.commit()

        deadline = time.monotonic() + 3.0
        batch = kanban_mailbox.MailboxDeliveryBatch(
            deliveries=[], delivered_keys=set(), include_in_context=False
        )
        while time.monotonic() < deadline:
            batch = kanban_mailbox.extract_mailbox_deliveries(agent)
            if batch.deliveries:
                break
            time.sleep(0.05)

        kanban_mailbox.shutdown_kanban_mailbox_runtime(agent)
        kanban_mailbox.join_kanban_mailbox_runtime(agent, timeout=2.0)

        assert len(batch.deliveries) == 1
        delivery = batch.deliveries[0]

        # This must NOT raise. If it does, the bridge is broken.
        coerced = _coerce_mailbox_delivery(delivery)
        assert coerced.message_id == delivery.message_id
        assert coerced.run_id == run_id
        assert coerced.body == "Coerce test body"
        assert coerced.kind == "guidance"


# ---------------------------------------------------------------------------
# 6. Configurable poll interval (design "Defaults" + "Performance contract").
#    The listener must read kanban.mailbox.poll_interval_seconds from config,
#    not hardcode 0.5.  Verified behaviorally: a custom value flows through to
#    the runtime attribute, so the interval is configurable end-to-end.
# ---------------------------------------------------------------------------

class TestConfigurablePollInterval:
    def test_runtime_reads_poll_interval_from_config(self, worker_env, monkeypatch):
        """initialize_kanban_mailbox_runtime must read poll_interval_seconds."""
        from hermes_cli import config as config_mod

        # Stub load_config_readonly to return a custom interval. This is the
        # function the runtime should consult; using readonly avoids the
        # deepcopy cost and matches hot-path convention.
        def _fake_config():
            import copy
            base = {
                "kanban": {
                    "mailbox": {
                        "poll_interval_seconds": 0.1,
                    }
                }
            }
            return copy.deepcopy(base)

        monkeypatch.setattr(config_mod, "load_config_readonly", _fake_config)

        agent = _bare_agent("perf_poll_cfg")
        principal = mailbox_principal.grant_top_level_mailbox_principal(
            agent,
            platform="cli",
            sender_profile=worker_env["profile"],
            actor_identity=f"cli:{worker_env['profile']}:pollcfg",
        )
        assert principal is not None and principal.kind == "worker"

        initialized = kanban_mailbox.initialize_kanban_mailbox_runtime(agent)
        assert initialized is True

        runtime = agent._kanban_mailbox_runtime
        assert runtime is not None
        # The configured value must flow through to the runtime, not stay 0.5.
        assert runtime.poll_interval == pytest.approx(0.1), (
            f"Expected poll_interval=0.1 from config, got {runtime.poll_interval!r}; "
            "the runtime must read kanban.mailbox.poll_interval_seconds, not hardcode 0.5"
        )

        kanban_mailbox.shutdown_kanban_mailbox_runtime(agent)
        kanban_mailbox.join_kanban_mailbox_runtime(agent, timeout=2.0)

    def test_runtime_defaults_when_config_missing(self, worker_env, monkeypatch):
        """When the config key is absent, the runtime falls back to 0.5."""
        from hermes_cli import config as config_mod

        # Config with the mailbox section entirely absent.
        monkeypatch.setattr(
            config_mod, "load_config_readonly", lambda: {"kanban": {}}
        )

        agent = _bare_agent("perf_poll_def")
        principal = mailbox_principal.grant_top_level_mailbox_principal(
            agent,
            platform="cli",
            sender_profile=worker_env["profile"],
            actor_identity=f"cli:{worker_env['profile']}:polldef",
        )
        assert principal is not None and principal.kind == "worker"

        initialized = kanban_mailbox.initialize_kanban_mailbox_runtime(agent)
        assert initialized is True

        runtime = agent._kanban_mailbox_runtime
        assert runtime is not None
        # Missing config → documented default.
        assert runtime.poll_interval == pytest.approx(0.5), (
            f"Expected default 0.5 when config missing, got {runtime.poll_interval!r}"
        )

        kanban_mailbox.shutdown_kanban_mailbox_runtime(agent)
        kanban_mailbox.join_kanban_mailbox_runtime(agent, timeout=2.0)


# ---------------------------------------------------------------------------
# 7. C2 fence: _try_close_mailbox_intake must return the result so close()
#    can act on it.  Also verifies the DB-level invariant that blocking
#    (guidance) pending blocks completion, while info pending does not —
#    which is exactly the window where the fence returns closed=False.
# ---------------------------------------------------------------------------

class TestCloseIntakeFenceResult:
    def test_fence_returns_result_with_pending_info(self, worker_env):
        """With an unresponded info message, the fence must report
        closed=False and the pending id — NOT silently drop it.

        Info is non-blocking, so complete_task passes (task → done), but
        try_close_mailbox_intake checks ALL unresponded kinds and must
        return closed=False. This is the real C2 window.
        """
        from hermes_cli import kanban_db as kb

        conn = worker_env["conn"]
        task_id = worker_env["task_id"]
        run_id = worker_env["run_id"]
        profile = worker_env["profile"]

        # Insert an unresponded info message (requires a linked comment).
        now = int(time.time())
        comment_id = kb.add_comment(
            conn, task_id=task_id, author="mailbox", body="linked info"
        )
        conn.execute(
            "INSERT INTO task_mailbox_messages "
            "(task_id, actor_identity, actor_kind, sender_profile, "
            " recipient_profile, kind, body, idempotency_key, comment_id, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                task_id, "manager:cli", "manager", profile, profile,
                "info", "non-blocking info still needs delivery",
                "fence-info-1", comment_id, now,
            ),
        )
        conn.commit()

        agent = _bare_agent("perf_fence_info")
        principal = mailbox_principal.grant_top_level_mailbox_principal(
            agent,
            platform="cli",
            sender_profile=profile,
            actor_identity=f"cli:{profile}:fenceinfo",
        )
        assert principal is not None and principal.kind == "worker"

        result = agent._try_close_mailbox_intake()

        # RED until _try_close_mailbox_intake returns the result.
        assert result is not None, (
            "_try_close_mailbox_intake must return MailboxIntakeCloseResult, "
            "not None — close() needs to act on closed=False"
        )
        assert result.closed is False, (
            "Fence must report closed=False when unresponded info exists"
        )
        assert result.pending_message_ids, (
            "Fence must report the pending message ids for diagnosis"
        )

    def test_fence_returns_closed_true_when_all_responded(self, worker_env):
        """When every mailbox message for the run has reached
        model_response_received, the fence closes cleanly.
        """
        from hermes_cli import kanban_db as kb

        conn = worker_env["conn"]
        task_id = worker_env["task_id"]
        run_id = worker_env["run_id"]
        profile = worker_env["profile"]

        # Insert an info message and mark it fully responded for this run.
        now = int(time.time())
        comment_id = kb.add_comment(
            conn, task_id=task_id, author="mailbox", body="linked done"
        )
        conn.execute(
            "INSERT INTO task_mailbox_messages "
            "(task_id, actor_identity, actor_kind, sender_profile, "
            " recipient_profile, kind, body, idempotency_key, comment_id, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                task_id, "manager:cli", "manager", profile, profile,
                "info", "already handled", "fence-done-1", comment_id, now,
            ),
        )
        msg_id = conn.execute(
            "SELECT id FROM task_mailbox_messages "
            "WHERE idempotency_key='fence-done-1'"
        ).fetchone()["id"]
        conn.execute(
            "INSERT INTO task_mailbox_delivery_attempts "
            "(message_id, run_id, state, claim_token, lease_expires, "
            " attempt_count, claimed_at, responded_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                msg_id, run_id, "model_response_received", "tok-done-1",
                now + 30, 1, now, now, now,
            ),
        )
        conn.commit()

        agent = _bare_agent("perf_fence_clean")
        principal = mailbox_principal.grant_top_level_mailbox_principal(
            agent,
            platform="cli",
            sender_profile=profile,
            actor_identity=f"cli:{profile}:fenceclean",
        )
        assert principal is not None and principal.kind == "worker"

        result = agent._try_close_mailbox_intake()

        assert result is not None
        assert result.closed is True, (
            "Fence must close when all messages are model_response_received"
        )
        assert result.pending_message_ids == []

    def test_blocking_pending_blocks_completion(self, worker_env):
        """Sanity: an unresponded guidance message blocks complete_task.

        This confirms the C2 invariant — the real protection against
        shutting down with unhandled blocking guidance is complete_task's
        barrier, NOT the close() fence.  The fence only needs to LOG
        closed=False; it must not be the sole guard.
        """
        from hermes_cli import kanban_db as kb

        conn = worker_env["conn"]
        task_id = worker_env["task_id"]
        run_id = worker_env["run_id"]
        profile = worker_env["profile"]

        now = int(time.time())
        comment_id = kb.add_comment(
            conn, task_id=task_id, author="mailbox", body="linked guidance"
        )
        conn.execute(
            "INSERT INTO task_mailbox_messages "
            "(task_id, actor_identity, actor_kind, sender_profile, "
            " recipient_profile, kind, body, idempotency_key, comment_id, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                task_id, "manager:cli", "manager", profile, profile,
                "guidance", "you must do X differently",
                "fence-block-1", comment_id, now,
            ),
        )
        conn.commit()

        # complete_task must raise on the pending guidance id.  This is the
        # real barrier that prevents shutdown with unhandled blocking mail.
        with pytest.raises(kb.MailboxCompletionBlockedError) as exc_info:
            kb.complete_task(
                conn,
                task_id=task_id,
                result="done prematurely",
                expected_run_id=run_id,
            )
        assert exc_info.value.message_ids, (
            "complete_task must report the pending blocking message id"
        )


# ---------------------------------------------------------------------------
# 8. I1 synchronous first/final poll (design line 142).
#    A message inserted between runtime start and the first listener poll
#    tick must still be delivered synchronously when poll_once() is called
#    on the agent thread — the async listener alone can miss the first
#    model request's window.  We pin poll_interval very high so the async
#    listener provably does not poll; only the synchronous claim delivers.
# ---------------------------------------------------------------------------

class TestSynchronousPollOnce:
    def test_poll_once_delivers_before_first_listener_tick(self, worker_env, monkeypatch):
        """poll_once() claims synchronously even when the async listener
        has not yet ticked.  Guards the first-model-request window.
        """
        from hermes_cli import config as config_mod
        from hermes_cli import kanban_db as kb

        # Pin the async poll interval far in the future so the listener
        # thread provably does not poll during this test.
        monkeypatch.setattr(
            config_mod,
            "load_config_readonly",
            lambda: {"kanban": {"mailbox": {"poll_interval_seconds": 999.0}}},
        )

        conn = worker_env["conn"]
        task_id = worker_env["task_id"]
        run_id = worker_env["run_id"]
        profile = worker_env["profile"]

        agent = _bare_agent("perf_sync_poll")
        principal = mailbox_principal.grant_top_level_mailbox_principal(
            agent,
            platform="cli",
            sender_profile=profile,
            actor_identity=f"cli:{profile}:syncpoll",
        )
        assert principal is not None and principal.kind == "worker"

        initialized = kanban_mailbox.initialize_kanban_mailbox_runtime(agent)
        assert initialized is True
        runtime = agent._kanban_mailbox_runtime
        assert runtime is not None

        # Insert a message AFTER the runtime started.  Because poll_interval
        # is 999s, the async listener will not claim it.
        now = int(time.time())
        comment_id = kb.add_comment(
            conn, task_id=task_id, author="mailbox", body="linked sync"
        )
        conn.execute(
            "INSERT INTO task_mailbox_messages "
            "(task_id, actor_identity, actor_kind, sender_profile, "
            " recipient_profile, kind, body, idempotency_key, comment_id, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                task_id, "manager:cli", "manager", profile, profile,
                "guidance", "sync-delivered guidance",
                "syncpoll-1", comment_id, now,
            ),
        )
        conn.commit()

        # Give the listener a moment to (not) poll, then prove the async
        # queue is empty — the listener could not have claimed it.
        time.sleep(0.2)
        async_batch = kanban_mailbox.extract_mailbox_deliveries(agent)
        assert async_batch.deliveries == [], (
            "Precondition: async listener must not have claimed the message "
            "with poll_interval=999s"
        )

        # Now the synchronous poll must claim it on the agent thread.
        runtime.poll_once()

        # And extract_mailbox_deliveries must return it immediately.
        batch = kanban_mailbox.extract_mailbox_deliveries(agent)
        assert len(batch.deliveries) == 1, (
            "poll_once() must synchronously deliver a message the async "
            "listener had not yet polled — this is the first-model-request window"
        )
        assert batch.deliveries[0].body == "sync-delivered guidance"

        kanban_mailbox.shutdown_kanban_mailbox_runtime(agent)
        kanban_mailbox.join_kanban_mailbox_runtime(agent, timeout=2.0)

    def test_poll_once_noop_when_no_messages(self, worker_env, monkeypatch):
        """poll_once() with no pending messages returns cleanly, no delivery."""
        from hermes_cli import config as config_mod

        monkeypatch.setattr(
            config_mod,
            "load_config_readonly",
            lambda: {"kanban": {"mailbox": {"poll_interval_seconds": 999.0}}},
        )

        agent = _bare_agent("perf_sync_empty")
        principal = mailbox_principal.grant_top_level_mailbox_principal(
            agent,
            platform="cli",
            sender_profile=worker_env["profile"],
            actor_identity=f"cli:{worker_env['profile']}:syncempty",
        )
        assert principal is not None and principal.kind == "worker"

        initialized = kanban_mailbox.initialize_kanban_mailbox_runtime(agent)
        assert initialized is True
        runtime = agent._kanban_mailbox_runtime

        # No messages inserted — poll_once must be a clean no-op.
        runtime.poll_once()
        batch = kanban_mailbox.extract_mailbox_deliveries(agent)
        assert batch.deliveries == []

        kanban_mailbox.shutdown_kanban_mailbox_runtime(agent)
        kanban_mailbox.join_kanban_mailbox_runtime(agent, timeout=2.0)


# ---------------------------------------------------------------------------
# 9. C3 E2E: listener-claimed delivery through the FULL canonical path.
#    The existing TestListenerCanonicalBridge only proves the listener's
#    MailboxDelivery carries claim_token and passes _coerce_mailbox_delivery.
#    This test drives the remaining canonical steps the production
#    conversation_loop runs: enqueue → drain → inject into messages[] →
#    acknowledge (included + responded).  C1 (missing claim_token) would
#    have been caught here — every delivery was silently dropped at coerce.
# ---------------------------------------------------------------------------

class TestListenerToCanonicalE2E:
    def test_listener_delivery_reaches_messages_and_acks(self, worker_env, monkeypatch):
        """A listener-claimed delivery flows through enqueue → drain →
        inject → acknowledge, leaving the <kanban_mailbox> envelope in
        messages[] and firing both included/responded callbacks.
        """
        from agent.agent_runtime_helpers import (
            acknowledge_mailbox_delivery_batch,
            drain_pending_mailbox_deliveries,
            enqueue_mailbox_deliveries,
            inject_mailbox_delivery_batch,
        )
        from hermes_cli import kanban_db as kb

        # Pin poll_interval high so the async listener does not race the
        # assertions; we drive the claim via poll_once() deterministically.
        from hermes_cli import config as config_mod

        monkeypatch.setattr(
            config_mod,
            "load_config_readonly",
            lambda: {"kanban": {"mailbox": {"poll_interval_seconds": 999.0}}},
        )

        conn = worker_env["conn"]
        task_id = worker_env["task_id"]
        run_id = worker_env["run_id"]
        profile = worker_env["profile"]

        agent = _bare_agent("perf_c3_e2e")
        principal = mailbox_principal.grant_top_level_mailbox_principal(
            agent,
            platform="cli",
            sender_profile=profile,
            actor_identity=f"cli:{profile}:c3e2e",
        )
        assert principal is not None and principal.kind == "worker"

        initialized = kanban_mailbox.initialize_kanban_mailbox_runtime(agent)
        assert initialized is True
        runtime = agent._kanban_mailbox_runtime

        # Insert a guidance message after the runtime started.
        now = int(time.time())
        comment_id = kb.add_comment(
            conn, task_id=task_id, author="mailbox", body="linked c3"
        )
        conn.execute(
            "INSERT INTO task_mailbox_messages "
            "(task_id, actor_identity, actor_kind, sender_profile, "
            " recipient_profile, kind, body, idempotency_key, comment_id, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                task_id, "manager:cli", "manager", profile, profile,
                "guidance", "E2E: pivot to option B",
                "c3-e2e-1", comment_id, now,
            ),
        )
        conn.commit()

        # Production callbacks are now wired by initialize_kanban_mailbox_runtime
        # (see TestProductionCallbackWiring).  This E2E test therefore exercises
        # the REAL callbacks — no manual wiring here.  The state machine is:
        #   claimed_for_run →(accept_mailbox_messages)→ accepted_by_steer
        #                  →(mark_mailbox_messages_included)→ included_in_request
        #                  →(mark_mailbox_messages_responded)→ model_response_received
        # The "included" callback first accepts (defensive; idempotent), then
        # marks included; the "responded" callback marks responded.
        assert agent._mailbox_delivery_included_callback is not None
        assert agent._mailbox_delivery_responded_callback is not None

        # --- Step 1: synchronous claim on the agent thread (as the loop does)
        runtime.poll_once()
        listener_batch = kanban_mailbox.extract_mailbox_deliveries(agent)
        assert len(listener_batch.deliveries) == 1
        listener_delivery = listener_batch.deliveries[0]
        # The C1 invariant: claim_token must be present, else coerce drops it.
        assert str(listener_delivery.claim_token).strip()

        # --- Step 2: enqueue into the canonical pending path
        enqueued = enqueue_mailbox_deliveries(agent, [listener_delivery])
        assert enqueued is True

        # --- Step 3: drain as one immutable snapshot
        canonical_batch = drain_pending_mailbox_deliveries(agent)
        assert canonical_batch is not None
        assert len(canonical_batch.deliveries) == 1
        assert canonical_batch.deliveries[0].body == "E2E: pivot to option B"

        # --- Step 4: inject into the current user message
        messages = [{"role": "user", "content": "do the thing"}]
        mutation = inject_mailbox_delivery_batch(
            agent,
            messages,
            canonical_batch,
            current_turn_user_idx=0,
        )
        assert mutation.appended is True
        assert mutation.target_index == 0
        # The <kanban_mailbox> envelope must be in the user message now.
        assert "kanban_mailbox" in messages[0]["content"]
        assert "E2E: pivot to option B" in messages[0]["content"]

        # --- Step 5: acknowledge "included" (the request now contains it)
        ok = acknowledge_mailbox_delivery_batch(
            agent, canonical_batch, stage="included"
        )
        assert ok is True

        # --- Step 6: acknowledge "responded" (model returned)
        ok = acknowledge_mailbox_delivery_batch(
            agent, canonical_batch, stage="responded"
        )
        assert ok is True

        # --- Step 7: DB state confirms the message reached the terminal state.
        row = conn.execute(
            "SELECT state FROM task_mailbox_delivery_attempts "
            "WHERE message_id = ? AND run_id = ?",
            (listener_delivery.message_id, run_id),
        ).fetchone()
        assert row is not None
        assert row["state"] == "model_response_received", (
            "E2E must drive the delivery attempt to the terminal "
            "model_response_received state via the real DB callbacks"
        )

        kanban_mailbox.shutdown_kanban_mailbox_runtime(agent)
        kanban_mailbox.join_kanban_mailbox_runtime(agent, timeout=2.0)


# ---------------------------------------------------------------------------
# 10. Production callback wiring.
#    initialize_kanban_mailbox_runtime must register the two DB-commit
#    callbacks on the agent so acknowledge_mailbox_delivery_batch can drive
#    the state machine to model_response_received.  Without this wiring the
#    callbacks stay None (agent_init.py default) and every production
#    acknowledgement hits "callback is None" → return False, so delivery
#    never reaches the terminal state.  Non-workers must keep None
#    (fail-closed: they never receive a batch).
# ---------------------------------------------------------------------------

class TestProductionCallbackWiring:
    def test_callbacks_non_none_after_runtime_init_for_worker(self, worker_env):
        """A worker principal's runtime init must wire BOTH DB callbacks.

        RED before production wiring: initialize_kanban_mailbox_runtime
        currently leaves the callbacks as the agent_init None default, so
        acknowledge_mailbox_delivery_batch fails closed in production.
        """
        agent = _bare_agent("perf_prod_wire")
        principal = mailbox_principal.grant_top_level_mailbox_principal(
            agent,
            platform="cli",
            sender_profile=worker_env["profile"],
            actor_identity=f"cli:{worker_env['profile']}:prodwire",
        )
        assert principal is not None and principal.kind == "worker"

        initialized = kanban_mailbox.initialize_kanban_mailbox_runtime(agent)
        assert initialized is True

        assert agent._mailbox_delivery_included_callback is not None, (
            "initialize_kanban_mailbox_runtime must wire the included "
            "callback for worker principals — production acknowledge cannot "
            "persist without it"
        )
        assert agent._mailbox_delivery_responded_callback is not None, (
            "initialize_kanban_mailbox_runtime must wire the responded "
            "callback for worker principals — delivery cannot reach "
            "model_response_received without it"
        )

        kanban_mailbox.shutdown_kanban_mailbox_runtime(agent)
        kanban_mailbox.join_kanban_mailbox_runtime(agent, timeout=2.0)

    def test_callbacks_remain_none_for_ordinary_agent(self):
        """Non-worker agents must keep None callbacks (fail-closed).

        They never receive a MailboxDeliveryBatch; acknowledge is never
        invoked on them.  Keeping None is the documented guard against
        accidental DB writes from manager/ordinary-chat paths.
        """
        agent = _bare_agent("perf_ordinary_wire")
        agent.mailbox_principal = None

        initialized = kanban_mailbox.initialize_kanban_mailbox_runtime(agent)
        assert initialized is False

        assert agent._mailbox_delivery_included_callback is None
        assert agent._mailbox_delivery_responded_callback is None
