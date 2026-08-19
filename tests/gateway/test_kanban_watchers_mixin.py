"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.config import Platform
from gateway.kanban_watchers import GatewayKanbanWatchersMixin
from hermes_cli import kanban_db as kb

KANBAN_METHODS = [
    "_kanban_notifier_watcher",
    "_kanban_dispatcher_watcher",
    "_kanban_advance",
    "_kanban_unsub",
    "_kanban_rewind",
    "_deliver_kanban_artifacts",
]


def test_mixin_defines_kanban_methods():
    for m in KANBAN_METHODS:
        assert hasattr(GatewayKanbanWatchersMixin, m), f"mixin missing {m}"


def test_gateway_runner_inherits_mixin():
    # Import here so a heavy gateway import only happens if the first test passed.
    from gateway.run import GatewayRunner

    assert issubclass(GatewayRunner, GatewayKanbanWatchersMixin)
    assert issubclass(GatewayRunner, GatewayAuthorizationMixin)
    assert GatewayRunner.__mro__.index(
        GatewayAuthorizationMixin
    ) < GatewayRunner.__mro__.index(GatewayKanbanWatchersMixin)
    # Each kanban method resolves to the mixin's implementation via the MRO.
    for m in KANBAN_METHODS:
        owner = next(c for c in GatewayRunner.__mro__ if m in c.__dict__)
        assert owner is GatewayKanbanWatchersMixin, (
            f"{m} resolved to {owner.__name__}, expected the mixin"
        )
    auth_owner = next(
        c for c in GatewayRunner.__mro__ if "_authorization_adapter" in c.__dict__
    )
    assert auth_owner is GatewayAuthorizationMixin


def test_watcher_loops_are_coroutines():
    # The two long-running watchers are async loops.
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._kanban_notifier_watcher)
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._kanban_dispatcher_watcher)


def test_mixin_notifier_sends_status_without_synthetic_followup(tmp_path, monkeypatch):
    """The extracted notifier must keep the live GatewayRunner contract.

    GatewayRunner owns default-profile adapters on ``self.adapters``. Its
    authorization-mixin dependency must preserve delivery through that registry
    when the extracted notifier resolves the adapter. A task ``session_id``
    must not turn that human notification into an injected agent follow-up.
    """
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        task_id = kb.create_task(
            conn,
            title="notify",
            assignee="worker",
            session_id="creator-session",
        )
        kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="chat-1",
        )
        kb.complete_task(conn, task_id, summary="done")
    finally:
        conn.close()

    class RecordingAdapter:
        def __init__(self):
            self.sent = []
            self.synthetic_events = []

        async def send(self, chat_id, text, metadata=None):
            self.sent.append((chat_id, text, metadata or {}))

        async def handle_message(self, event):
            self.synthetic_events.append(event)
            raise AssertionError("notifier must not inject a synthetic follow-up")

    from gateway.run import GatewayRunner

    adapter = RecordingAdapter()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_counts = {}

    real_sleep = asyncio.sleep

    async def one_tick(delay):
        if delay == 5:
            return
        runner._running = False
        await real_sleep(0)

    # If synthetic wake is reintroduced, this checkout's older SessionSource
    # signature would otherwise fail before ``handle_message`` and mask the
    # regression. The parity contract is that the branch is never attempted.
    from gateway import session as gateway_session

    monkeypatch.setattr(gateway_session, "SessionSource", lambda **kwargs: kwargs)
    monkeypatch.setattr(asyncio, "sleep", one_tick)
    asyncio.run(
        GatewayKanbanWatchersMixin._kanban_notifier_watcher(runner, interval=1)
    )

    assert len(adapter.sent) == 1
    assert task_id in adapter.sent[0][1]
    assert adapter.synthetic_events == []


def test_named_profile_notifier_uses_its_own_adapter_registry(tmp_path, monkeypatch):
    """A stamped subscription for this named-profile gateway uses self.adapters."""
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="notify", assignee="worker")
        kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="chat-1",
            notifier_profile="business-partner",
        )
        kb.complete_task(conn, task_id, summary="done")
    finally:
        conn.close()

    class RecordingAdapter:
        def __init__(self):
            self.sent = []

        async def send(self, chat_id, text, metadata=None):
            self.sent.append((chat_id, text, metadata or {}))

    from gateway.run import GatewayRunner

    adapter = RecordingAdapter()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_counts = {}
    runner._kanban_notifier_profile = "business-partner"

    real_sleep = asyncio.sleep

    async def one_tick(delay):
        if delay == 5:
            return
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", one_tick)
    asyncio.run(
        GatewayKanbanWatchersMixin._kanban_notifier_watcher(runner, interval=1)
    )

    assert len(adapter.sent) == 1
    assert task_id in adapter.sent[0][1]


@pytest.mark.asyncio
async def test_dispatcher_releases_lock_when_initial_sleep_is_cancelled(
    tmp_path, monkeypatch
):
    """Cancellation anywhere after lock acquisition releases it exactly once."""
    from gateway.run import GatewayRunner
    from gateway import kanban_watchers

    handle = object()
    release_calls = []
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(
        kanban_watchers,
        "_acquire_singleton_lock",
        lambda _path: (handle, "held"),
    )
    monkeypatch.setattr(
        kanban_watchers,
        "_release_singleton_lock",
        lambda released: release_calls.append(released),
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"dispatch_in_gateway": True}},
    )

    async def cancel_initial_sleep(delay):
        assert delay == 5
        raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", cancel_initial_sleep)

    with pytest.raises(asyncio.CancelledError):
        await runner._kanban_dispatcher_watcher()

    assert release_calls == [handle]
    assert runner._kanban_dispatcher_lock_handle is None


def test_singleton_dispatcher_lock_is_exclusive(tmp_path):
    """Only one holder of the dispatcher lock at a time — the backstop that
    stops concurrent dispatchers double reclaiming and corrupting shared
    kanban SQLite index pages under wal_autocheckpoint=0."""
    import os

    from gateway.kanban_watchers import _acquire_singleton_lock, _release_singleton_lock

    lock = tmp_path / "kanban" / ".dispatcher.lock"

    h1, st1 = _acquire_singleton_lock(lock)
    assert st1 == "held" and h1 is not None

    # A second acquire while the first is held must be refused, not granted.
    h2, st2 = _acquire_singleton_lock(lock)
    assert st2 == "contended" and h2 is None

    # Releasing the first lets a fresh acquire succeed (lock is reusable).
    _release_singleton_lock(h1)
    h3, st3 = _acquire_singleton_lock(lock)
    assert st3 == "held" and h3 is not None
    _release_singleton_lock(h3)
