"""Exact mailbox-delivery tests for the A2 steer boundary.

Mailbox guidance is durable state, not a second form of manual ``/steer``.
These tests pin the in-memory ownership batch and the two persistence barriers
around one real conversation-loop request made against a local fake provider.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import agent.agent_runtime_helpers as runtime_helpers
import agent.conversation_loop as conversation_loop
import hermes_cli.plugins as plugin_runtime
import run_agent
from run_agent import AIAgent


def _delivery(
    message_id: int,
    *,
    run_id: int = 41,
    token: str | None = None,
    body: str | None = None,
    kind: str = "guidance",
):
    return SimpleNamespace(
        message_id=message_id,
        run_id=run_id,
        claim_token=token or f"lease-{message_id}",
        body=body or f"mailbox body {message_id}",
        kind=kind,
    )


def _bare_agent() -> AIAgent:
    agent = object.__new__(AIAgent)
    agent._pending_steer = None
    agent._pending_steer_lock = threading.Lock()
    return agent


def _tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "test tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _tool_call(name: str, call_id: str = "call-1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments="{}"),
    )


def _response(*, content: str, tool_calls=None, finish_reason: str = "stop"):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="fake/model", usage=None)


class _FakeProvider:
    def __init__(self, responses, events: list[str] | None = None):
        self._responses = list(responses)
        self.events = events if events is not None else []
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.events.append("provider")
        self.requests.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _PayloadTooLarge(RuntimeError):
    status_code = 413
    body = {"error": "too large"}


class _ContextOverflow(RuntimeError):
    status_code = 400
    body = {"error": "Prompt exceeds max length"}


class _ContentPolicyBlocked(RuntimeError):
    status_code = 400
    body = {"error": "flagged for possible cybersecurity risk"}


def _install_lifecycle_spies(agent, monkeypatch):
    session_events: list[str] = []
    plugin_events: list[str] = []
    agent._session_db = SimpleNamespace(
        mark_run_active=lambda _session_id: session_events.append("active"),
        mark_run_idle=lambda _session_id: session_events.append("idle"),
    )
    agent._session_db_created = True
    monkeypatch.setattr(
        plugin_runtime,
        "invoke_hook",
        lambda name, **_kwargs: plugin_events.append(name) or [],
    )
    return session_events, plugin_events


def _make_agent(
    monkeypatch,
    responses,
    *,
    tool_names: tuple[str, ...] = (),
    events: list[str] | None = None,
):
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **_: _tool_defs(*tool_names))
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda *_, **__: {})
    with patch("run_agent.OpenAI"):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://fake.invalid/v1",
            model="fake/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=4,
        )

    event_log = events if events is not None else []
    provider = _FakeProvider(responses, event_log)
    client = MagicMock()
    client.chat.completions.create.side_effect = provider.create
    agent.client = client
    agent._disable_streaming = True
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.tool_delay = 0
    agent._api_max_retries = 1
    agent.valid_tool_names = set(tool_names)
    # Keep the fake provider on the native multimodal path.  The mailbox
    # tests exercise list preservation, not the separate auxiliary-vision
    # fallback used for text-only models.
    agent._model_supports_vision = lambda: True
    agent._persist_session = lambda *_, **__: None
    agent._save_trajectory = lambda *_, **__: None
    agent._cleanup_task_resources = lambda *_, **__: None
    agent._mailbox_delivery_included_callback = lambda _batch: True
    agent._mailbox_delivery_responded_callback = lambda _batch: True
    return agent, provider


def _all_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


def _request_message(request: dict, role: str, *, last: bool = False) -> dict:
    matches = [message for message in request["messages"] if message.get("role") == role]
    return matches[-1] if last else matches[0]


class TestMailboxGuidanceQueue:
    def test_recovery_is_allocation_free_for_an_agent_that_never_used_mailbox(self):
        agent = _bare_agent()

        agent._recover_inflight_mailbox_deliveries()

        assert "_pending_mailbox_deliveries" not in agent.__dict__
        assert "_inflight_mailbox_deliveries" not in agent.__dict__

    def test_complete_ordinary_turn_does_not_allocate_mailbox_maps(self, monkeypatch):
        agent, _provider = _make_agent(monkeypatch, [_response(content="done")])

        result = agent.run_conversation("ordinary request")

        assert result["final_response"] == "done"
        assert agent._pending_mailbox_deliveries is None
        assert agent._inflight_mailbox_deliveries is None

    def test_drain_is_one_ordered_immutable_batch_and_requeue_precedes_later_arrival(self):
        agent = _bare_agent()
        agent._enqueue_mailbox_deliveries([_delivery(1), _delivery(2)])

        batch = agent._drain_pending_mailbox_deliveries()

        assert tuple(item.message_id for item in batch.deliveries) == (1, 2)
        assert isinstance(batch.deliveries, tuple)
        with pytest.raises((AttributeError, TypeError)):
            batch.deliveries += (_delivery(99),)

        agent._enqueue_mailbox_deliveries([_delivery(3)])
        agent._requeue_mailbox_delivery_batch(batch)
        recovered = agent._drain_pending_mailbox_deliveries()
        assert tuple(item.message_id for item in recovered.deliveries) == (1, 2, 3)

    def test_duplicate_ref_is_not_duplicated_and_renewed_token_replaces_it(self):
        agent = _bare_agent()
        agent._enqueue_mailbox_deliveries([_delivery(1, token="old", body="only once")])
        agent._enqueue_mailbox_deliveries([_delivery(1, token="renewed", body="only once")])

        batch = agent._drain_pending_mailbox_deliveries()

        assert len(batch.deliveries) == 1
        assert batch.deliveries[0].claim_token == "renewed"
        assert batch.render().count("only once") == 1
        assert "old" not in batch.render()
        assert "renewed" not in batch.render()

    def test_stale_requeue_cannot_overwrite_a_newer_token_for_same_ref(self):
        agent = _bare_agent()
        agent._enqueue_mailbox_deliveries([_delivery(1, token="old")])
        stale_batch = agent._drain_pending_mailbox_deliveries()

        agent._enqueue_mailbox_deliveries([_delivery(1, token="new")])
        agent._requeue_mailbox_delivery_batch(stale_batch)
        recovered = agent._drain_pending_mailbox_deliveries()

        assert len(recovered.deliveries) == 1
        assert recovered.deliveries[0].claim_token == "new"

    def test_partial_renewal_keeps_retained_batch_refs_and_render_exact(self):
        agent = _bare_agent()
        agent._mailbox_delivery_included_callback = lambda _batch: True
        agent._enqueue_mailbox_deliveries(
            [
                _delivery(1, token="old-a", body="guidance A"),
                _delivery(2, token="old-b", body="guidance B"),
            ]
        )
        original = agent._drain_pending_mailbox_deliveries()
        assert runtime_helpers.acknowledge_mailbox_delivery_batch(
            agent,
            original,
            stage="included",
        )
        included = runtime_helpers.current_mailbox_delivery_batch(agent, original)
        original_envelope = included.render()
        agent._enqueue_mailbox_deliveries(
            [_delivery(1, token="new-a", body="guidance A")]
        )

        retained = runtime_helpers.reconcile_mailbox_delivery_batch_after_response_failure(
            agent,
            included,
        )

        assert [(item.message_id, item.claim_token) for item in retained.deliveries] == [
            (2, "old-b")
        ]
        assert "guidance A" not in retained.render()
        assert "guidance B" in retained.render()
        assert list(agent._inflight_mailbox_deliveries) == [(2, 41)]
        assert agent._pending_mailbox_deliveries[(1, 41)].claim_token == "new-a"

        messages = [{"role": "user", "content": f"start\n\n{original_envelope}"}]
        mutation = runtime_helpers.inject_mailbox_delivery_batch(
            agent,
            messages,
            retained,
            current_turn_user_idx=0,
        )
        assert mutation.appended is False
        assert messages[0]["content"].count("guidance B") == 1

    def test_manual_steer_remains_a_separate_slot_under_the_same_lock(self):
        agent = _bare_agent()
        assert agent.steer("manual note") is True
        agent._enqueue_mailbox_deliveries([_delivery(1, body="durable note")])

        assert agent._drain_pending_steer() == "manual note"
        batch = agent._drain_pending_mailbox_deliveries()
        assert "durable note" in batch.render()
        assert "manual note" not in batch.render()


class TestMailboxRequestInjection:
    def test_first_string_request_is_acknowledged_before_provider_and_persisted_once(self, monkeypatch):
        events: list[str] = []
        agent, provider = _make_agent(
            monkeypatch,
            [_response(content="done")],
            events=events,
        )
        delivery = _delivery(1, token="secret-claim-token", body="inspect the audit trail")
        agent._enqueue_mailbox_deliveries([delivery])
        agent._mailbox_delivery_included_callback = lambda batch: events.append("included") or True
        agent._mailbox_delivery_responded_callback = lambda batch: events.append("responded") or True

        result = agent.run_conversation("start")

        request_text = _all_text(_request_message(provider.requests[0], "user")["content"])
        stored_text = _all_text(result["messages"][0]["content"])
        assert events.index("included") < events.index("provider") < events.index("responded")
        assert "<kanban_mailbox>" in request_text
        assert "message_id=1 kind=guidance" in request_text
        assert "inspect the audit trail" in request_text
        assert "secret-claim-token" not in request_text
        assert stored_text.count("<kanban_mailbox>") == 1
        assert stored_text.count("inspect the audit trail") == 1

    def test_first_multimodal_user_content_is_preserved_and_gets_one_text_block(self, monkeypatch):
        agent, provider = _make_agent(monkeypatch, [_response(content="done")])
        agent._enqueue_mailbox_deliveries([_delivery(1, body="read the new constraint")])
        original = [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        ]

        result = agent.run_conversation(original)

        request_content = _request_message(provider.requests[0], "user")["content"]
        stored_content = result["messages"][0]["content"]
        assert request_content[:2] == original
        assert stored_content[:2] == original
        assert len(stored_content) == 3
        assert stored_content[-1]["type"] == "text"
        assert "read the new constraint" in stored_content[-1]["text"]

    @pytest.mark.parametrize("multimodal", [False, True])
    def test_later_delivery_is_appended_to_latest_tool_result(self, monkeypatch, multimodal):
        agent, provider = _make_agent(
            monkeypatch,
            [
                _response(
                    content="checking",
                    tool_calls=[_tool_call("fake_tool")],
                    finish_reason="tool_calls",
                ),
                _response(content="done"),
            ],
            tool_names=("fake_tool",),
        )

        def fake_tool(*_args, **_kwargs):
            agent._enqueue_mailbox_deliveries([_delivery(2, body="change the next step")])
            if multimodal:
                return {
                    "_multimodal": True,
                    "content": [
                        {"type": "text", "text": "tool output"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                    ],
                    "text_summary": "tool output",
                }
            return "tool output"

        monkeypatch.setattr(run_agent, "handle_function_call", fake_tool)

        result = agent.run_conversation("start")

        second_tool = _request_message(provider.requests[1], "tool", last=True)
        stored_tool = [m for m in result["messages"] if m.get("role") == "tool"][-1]
        assert "tool output" in _all_text(second_tool["content"])
        assert "change the next step" in _all_text(second_tool["content"])
        assert _all_text(stored_tool["content"]).count("change the next step") == 1
        if multimodal:
            assert isinstance(stored_tool["content"], list)
            assert stored_tool["content"][1]["type"] == "image_url"

    def test_arrival_during_include_ack_is_not_acknowledged_by_old_snapshot(self, monkeypatch):
        agent, provider = _make_agent(
            monkeypatch,
            [
                _response(
                    content="checking",
                    tool_calls=[_tool_call("fake_tool")],
                    finish_reason="tool_calls",
                ),
                _response(content="done"),
            ],
            tool_names=("fake_tool",),
        )
        agent._enqueue_mailbox_deliveries([_delivery(1, body="first body")])
        included: list[tuple[int, ...]] = []
        responded: list[tuple[int, ...]] = []

        def include(batch):
            ids = tuple(item.message_id for item in batch.deliveries)
            included.append(ids)
            if ids == (1,):
                agent._enqueue_mailbox_deliveries([_delivery(2, body="second body")])
            return True

        agent._mailbox_delivery_included_callback = include
        agent._mailbox_delivery_responded_callback = (
            lambda batch: responded.append(tuple(item.message_id for item in batch.deliveries)) or True
        )
        monkeypatch.setattr(run_agent, "handle_function_call", lambda *_args, **_kwargs: "tool output")

        agent.run_conversation("start")

        first_request = "\n".join(_all_text(m.get("content", "")) for m in provider.requests[0]["messages"])
        second_request = "\n".join(_all_text(m.get("content", "")) for m in provider.requests[1]["messages"])
        assert included == [(1,), (2,)]
        assert responded == [(1,), (2,)]
        assert "first body" in first_request
        assert "second body" not in first_request
        assert "first body" in second_request
        assert "second body" in second_request


class TestMailboxPersistenceBarriers:
    def test_compression_restart_keeps_included_batch_until_response_barrier(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(conversation_loop.time, "sleep", lambda _seconds: None)
        history = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "older"},
            {"role": "assistant", "content": "older answer"},
        ]
        events: list[str] = []
        agent, provider = _make_agent(
            monkeypatch,
            [
                _PayloadTooLarge("request too large"),
                _response(
                    content="tool",
                    tool_calls=[_tool_call("fake_tool")],
                    finish_reason="tool_calls",
                ),
                _response(content="done"),
            ],
            tool_names=("fake_tool",),
            events=events,
        )
        agent._api_max_retries = 2
        agent._compress_context = (
            lambda messages, *_args, **_kwargs: (messages[-1:], "You are helpful.")
        )
        agent._enqueue_mailbox_deliveries([_delivery(1, body="survive compression")])
        agent._mailbox_delivery_included_callback = (
            lambda _batch: events.append("included") or True
        )
        agent._mailbox_delivery_responded_callback = (
            lambda _batch: events.append("responded") or True
        )
        monkeypatch.setattr(
            run_agent,
            "handle_function_call",
            lambda *_args, **_kwargs: events.append("dispatch") or "tool output",
        )

        result = agent.run_conversation("start", conversation_history=history)

        assert result["final_response"] == "done"
        assert events.count("included") == 1
        assert events.index("responded") < events.index("dispatch")
        second_request = "\n".join(
            _all_text(message.get("content", ""))
            for message in provider.requests[1]["messages"]
        )
        assert second_request.count("survive compression") == 1
        assert agent._drain_pending_mailbox_deliveries() is None

    def test_mailbox_response_is_normalized_once_before_ack_and_reused(self, monkeypatch):
        agent, _provider = _make_agent(monkeypatch, [_response(content="done")])
        agent._enqueue_mailbox_deliveries([_delivery(1)])
        callback_stages: list[tuple[str, bool]] = []
        agent._mailbox_delivery_included_callback = (
            lambda batch: callback_stages.append(("included", batch.included)) or True
        )
        agent._mailbox_delivery_responded_callback = (
            lambda batch: callback_stages.append(("responded", batch.included)) or True
        )
        transport = agent._get_transport()
        normalize_calls: list[object] = []
        original_normalize = transport.normalize_response

        def normalize(response, **kwargs):
            normalize_calls.append(response)
            return original_normalize(response, **kwargs)

        monkeypatch.setattr(transport, "normalize_response", normalize)

        result = agent.run_conversation("start")

        assert result["final_response"] == "done"
        assert len(normalize_calls) == 1
        assert callback_stages == [("included", False), ("responded", True)]

    def test_include_callback_retries_are_bounded_before_provider(self, monkeypatch):
        events: list[str] = []
        agent, provider = _make_agent(monkeypatch, [_response(content="done")], events=events)
        agent._enqueue_mailbox_deliveries([_delivery(1)])
        attempts = iter([False, False, True])

        def include(_batch):
            events.append("included-attempt")
            return next(attempts)

        agent._mailbox_delivery_included_callback = include

        result = agent.run_conversation("start")

        assert result["final_response"] == "done"
        assert events == ["included-attempt", "included-attempt", "included-attempt", "provider"]
        assert len(provider.requests) == 1

    def test_permanent_include_failure_restores_content_makes_zero_provider_calls_and_requeues(self, monkeypatch):
        agent, provider = _make_agent(monkeypatch, [_response(content="must not run")])
        agent._enqueue_mailbox_deliveries([_delivery(1, body="must be recovered")])
        include_calls: list[object] = []
        session_events: list[str] = []
        agent._session_db = SimpleNamespace(
            mark_run_active=lambda _session_id: session_events.append("active"),
            mark_run_idle=lambda _session_id: session_events.append("idle"),
        )
        agent._session_db_created = True
        agent._mailbox_delivery_included_callback = lambda batch: include_calls.append(batch) or False

        result = agent.run_conversation("original user text")

        assert result["failed"] is True
        assert "mailbox" in result["error"].lower()
        assert provider.requests == []
        assert len(include_calls) == 3
        assert session_events == ["active", "idle"]
        assert result["messages"][0]["content"] == "original user text"
        recovered = agent._drain_pending_mailbox_deliveries()
        assert tuple(item.message_id for item in recovered.deliveries) == (1,)

    def test_response_callback_transient_failures_retry_before_post_processing(self, monkeypatch):
        events: list[str] = []
        agent, _provider = _make_agent(monkeypatch, [_response(content="done")], events=events)
        agent._enqueue_mailbox_deliveries([_delivery(1)])
        attempts = iter([False, False, True])

        def responded(_batch):
            events.append("responded-attempt")
            return next(attempts)

        agent._mailbox_delivery_responded_callback = responded

        def hook(name, **_kwargs):
            events.append(name)
            return []

        monkeypatch.setattr(plugin_runtime, "invoke_hook", hook)

        result = agent.run_conversation("start")

        assert result["final_response"] == "done"
        provider_idx = events.index("provider")
        post_idx = events.index("post_api_request")
        assert events[provider_idx + 1:post_idx] == [
            "responded-attempt",
            "responded-attempt",
            "responded-attempt",
        ]

    def test_response_ack_failure_prevents_post_plugin_validation_and_tool_dispatch(self, monkeypatch):
        events: list[str] = []
        agent, provider = _make_agent(
            monkeypatch,
            [
                _response(
                    content="tool now",
                    tool_calls=[_tool_call("fake_tool")],
                    finish_reason="tool_calls",
                )
            ],
            tool_names=("fake_tool",),
            events=events,
        )
        agent._enqueue_mailbox_deliveries([_delivery(1, body="do not lose me")])
        response_calls: list[object] = []
        agent._mailbox_delivery_responded_callback = (
            lambda batch: response_calls.append(batch) or False
        )

        def hook(name, **_kwargs):
            events.append(name)
            return []

        monkeypatch.setattr(plugin_runtime, "invoke_hook", hook)
        monkeypatch.setattr(
            run_agent,
            "handle_function_call",
            lambda *_args, **_kwargs: events.append("dispatch") or "tool output",
        )

        result = agent.run_conversation("start")

        assert result["failed"] is True
        assert "mailbox" in result["error"].lower()
        assert len(response_calls) == 3
        assert "post_api_request" not in events
        assert "dispatch" not in events
        recovered = agent._drain_pending_mailbox_deliveries()
        assert tuple(item.message_id for item in recovered.deliveries) == (1,)

    def test_renewed_token_supersedes_stale_included_batch_after_response_failure(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(runtime_helpers.time, "sleep", lambda _seconds: None)
        agent, provider = _make_agent(
            monkeypatch,
            [_response(content="first"), _response(content="recovered")],
        )
        agent._enqueue_mailbox_deliveries(
            [_delivery(1, token="old", body="same durable guidance")]
        )
        included_tokens: list[str] = []
        responded_tokens: list[str] = []

        def included(batch):
            included_tokens.append(batch.deliveries[0].claim_token)
            return True

        def responded(batch):
            token = batch.deliveries[0].claim_token
            responded_tokens.append(token)
            if token == "old":
                agent._enqueue_mailbox_deliveries(
                    [_delivery(1, token="new", body="same durable guidance")]
                )
                return False
            return True

        agent._mailbox_delivery_included_callback = included
        agent._mailbox_delivery_responded_callback = responded

        first = agent.run_conversation("start")
        second = agent.run_conversation("resume", conversation_history=first["messages"])

        assert first["failed"] is True
        assert second["final_response"] == "recovered"
        assert len(provider.requests) == 2
        assert included_tokens == ["old", "new"]
        assert responded_tokens == ["old", "old", "old", "new"]
        second_request = "\n".join(
            _all_text(message.get("content", ""))
            for message in provider.requests[1]["messages"]
        )
        assert second_request.count("same durable guidance") == 1
        assert agent._drain_pending_mailbox_deliveries() is None

    def test_success_order_is_include_provider_response_post_plugin_then_dispatch(self, monkeypatch):
        events: list[str] = []
        agent, _provider = _make_agent(
            monkeypatch,
            [
                _response(
                    content="tool now",
                    tool_calls=[_tool_call("fake_tool")],
                    finish_reason="tool_calls",
                ),
                _response(content="done"),
            ],
            tool_names=("fake_tool",),
            events=events,
        )
        agent._enqueue_mailbox_deliveries([_delivery(1)])
        agent._mailbox_delivery_included_callback = lambda _batch: events.append("included") or True
        agent._mailbox_delivery_responded_callback = lambda _batch: events.append("responded") or True

        def hook(name, **_kwargs):
            events.append(name)
            return []

        monkeypatch.setattr(plugin_runtime, "invoke_hook", hook)
        monkeypatch.setattr(
            run_agent,
            "handle_function_call",
            lambda *_args, **_kwargs: events.append("dispatch") or "tool output",
        )

        result = agent.run_conversation("start")

        assert result["final_response"] == "done"
        assert events.index("included") < events.index("provider")
        assert events.index("provider") < events.index("responded")
        assert events.index("responded") < events.index("post_api_request")
        assert events.index("post_api_request") < events.index("dispatch")

    def test_terminal_provider_failure_keeps_exact_batch_recoverable(self, monkeypatch):
        agent, provider = _make_agent(monkeypatch, [RuntimeError("terminal fake provider failure")])
        agent._enqueue_mailbox_deliveries([_delivery(7, token="exact-token", body="recover exactly")])
        session_events: list[str] = []
        plugin_events: list[str] = []
        agent._session_db = SimpleNamespace(
            mark_run_active=lambda _session_id: session_events.append("active"),
            mark_run_idle=lambda _session_id: session_events.append("idle"),
        )
        agent._session_db_created = True
        monkeypatch.setattr(
            plugin_runtime,
            "invoke_hook",
            lambda name, **_kwargs: plugin_events.append(name) or [],
        )

        result = agent.run_conversation("start")

        assert result["failed"] is True
        assert result.get("error") or result.get("final_response")
        assert len(provider.requests) == 1
        recovered = agent._drain_pending_mailbox_deliveries()
        assert len(recovered.deliveries) == 1
        assert recovered.deliveries[0].message_id == 7
        assert recovered.deliveries[0].claim_token == "exact-token"
        assert recovered.deliveries[0].body == "recover exactly"
        stored = "\n".join(_all_text(message.get("content", "")) for message in result["messages"])
        assert stored.count("recover exactly") == 1
        assert session_events == ["active", "idle"]
        assert "on_session_end" in plugin_events

    def test_terminal_provider_retry_does_not_repeat_committed_include_transition(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(runtime_helpers.time, "sleep", lambda _seconds: None)
        state = {"value": "accepted"}
        include_states: list[str] = []
        agent, provider = _make_agent(
            monkeypatch,
            [RuntimeError("terminal"), _response(content="recovered")],
        )
        agent._enqueue_mailbox_deliveries([_delivery(1, token="same")])

        def included(_batch):
            include_states.append(state["value"])
            if state["value"] != "accepted":
                return False
            state["value"] = "included"
            return True

        def responded(_batch):
            assert state["value"] == "included"
            state["value"] = "responded"
            return True

        agent._mailbox_delivery_included_callback = included
        agent._mailbox_delivery_responded_callback = responded

        first = agent.run_conversation("start")
        second = agent.run_conversation("resume", conversation_history=first["messages"])

        assert second["final_response"] == "recovered"
        assert len(provider.requests) == 2
        assert state["value"] == "responded"
        assert include_states == ["accepted"]

    def test_terminal_invalid_provider_response_uses_shared_turn_finalizer(
        self,
        monkeypatch,
    ):
        invalid_response = SimpleNamespace(
            choices=[],
            model="fake/model",
            error=None,
            message=None,
            usage=None,
        )
        agent, _provider = _make_agent(monkeypatch, [invalid_response])
        agent._enqueue_mailbox_deliveries([_delivery(1, token="invalid-response")])
        session_events: list[str] = []
        plugin_events: list[str] = []
        agent._session_db = SimpleNamespace(
            mark_run_active=lambda _session_id: session_events.append("active"),
            mark_run_idle=lambda _session_id: session_events.append("idle"),
        )
        agent._session_db_created = True
        monkeypatch.setattr(
            plugin_runtime,
            "invoke_hook",
            lambda name, **_kwargs: plugin_events.append(name) or [],
        )

        result = agent.run_conversation("start")

        assert result["failed"] is True
        assert "invalid api response" in result["error"].lower()
        assert session_events == ["active", "idle"]
        assert "on_session_end" in plugin_events
        active = agent._drain_pending_mailbox_deliveries()
        assert active.included is True
        assert active.deliveries[0].claim_token == "invalid-response"

    def test_terminal_compression_failure_uses_shared_turn_finalizer(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(conversation_loop.time, "sleep", lambda _seconds: None)
        agent, _provider = _make_agent(
            monkeypatch,
            [_PayloadTooLarge("request too large")],
        )
        agent._compress_context = (
            lambda messages, *_args, **_kwargs: (messages, "You are helpful.")
        )
        agent._enqueue_mailbox_deliveries([_delivery(1, token="compression-terminal")])
        session_events: list[str] = []
        plugin_events: list[str] = []
        agent._session_db = SimpleNamespace(
            mark_run_active=lambda _session_id: session_events.append("active"),
            mark_run_idle=lambda _session_id: session_events.append("idle"),
        )
        agent._session_db_created = True
        monkeypatch.setattr(
            plugin_runtime,
            "invoke_hook",
            lambda name, **_kwargs: plugin_events.append(name) or [],
        )

        result = agent.run_conversation("start")

        assert result["failed"] is True
        assert "payload too large" in result["error"].lower()
        assert session_events == ["active", "idle"]
        assert "on_session_end" in plugin_events
        active = agent._drain_pending_mailbox_deliveries()
        assert active.included is True
        assert active.deliveries[0].claim_token == "compression-terminal"

    def test_nous_rate_guard_failure_uses_shared_turn_finalizer(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            "agent.nous_rate_guard.nous_rate_limit_remaining",
            lambda: 60.0,
        )
        monkeypatch.setattr(
            "agent.nous_rate_guard.format_remaining",
            lambda _seconds: "1m",
        )
        agent, provider = _make_agent(monkeypatch, [_response(content="must not run")])
        agent.provider = "nous"
        agent._try_activate_fallback = lambda: False
        agent._enqueue_mailbox_deliveries([_delivery(1, token="nous-rate-guard")])
        session_events, plugin_events = _install_lifecycle_spies(agent, monkeypatch)

        result = agent.run_conversation("start")

        assert result["failed"] is True
        assert provider.requests == []
        assert session_events == ["active", "idle"]
        assert "on_session_end" in plugin_events
        active = agent._drain_pending_mailbox_deliveries()
        assert active.included is True
        assert active.deliveries[0].claim_token == "nous-rate-guard"

    def test_terminal_context_compression_failure_uses_shared_turn_finalizer(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(conversation_loop.time, "sleep", lambda _seconds: None)
        agent, _provider = _make_agent(
            monkeypatch,
            [_ContextOverflow("Prompt exceeds max length")],
        )
        agent._compress_context = (
            lambda messages, *_args, **_kwargs: (messages, "You are helpful.")
        )
        agent._enqueue_mailbox_deliveries([_delivery(1, token="context-terminal")])
        session_events, plugin_events = _install_lifecycle_spies(agent, monkeypatch)

        result = agent.run_conversation("start")

        assert result["failed"] is True
        assert "context length exceeded" in result["error"].lower()
        assert session_events == ["active", "idle"]
        assert "on_session_end" in plugin_events
        active = agent._drain_pending_mailbox_deliveries()
        assert active.included is True
        assert active.deliveries[0].claim_token == "context-terminal"

    @pytest.mark.parametrize(
        "api_error",
        [
            ValueError("bad local request"),
            _ContentPolicyBlocked("flagged for possible cybersecurity risk"),
        ],
        ids=("general", "content-policy"),
    )
    def test_nonretryable_provider_failure_uses_shared_turn_finalizer(
        self,
        monkeypatch,
        api_error,
    ):
        agent, _provider = _make_agent(monkeypatch, [api_error])
        agent._enqueue_mailbox_deliveries([_delivery(1, token="nonretryable")])
        session_events, plugin_events = _install_lifecycle_spies(agent, monkeypatch)

        result = agent.run_conversation("start")

        assert result["failed"] is True
        assert result["error"]
        assert session_events == ["active", "idle"]
        assert "on_session_end" in plugin_events
        active = agent._drain_pending_mailbox_deliveries()
        assert active.included is True
        assert active.deliveries[0].claim_token == "nonretryable"

    def test_interrupted_provider_error_uses_shared_turn_finalizer(
        self,
        monkeypatch,
    ):
        agent, provider = _make_agent(monkeypatch, [_response(content="unused")])
        agent._api_max_retries = 2
        agent._enqueue_mailbox_deliveries([_delivery(1, token="interrupted-error")])
        session_events, plugin_events = _install_lifecycle_spies(agent, monkeypatch)

        def interrupted_create(**kwargs):
            provider.events.append("provider")
            provider.requests.append(kwargs)
            agent._interrupt_requested = True
            raise RuntimeError("interrupt after provider error")

        agent.client.chat.completions.create.side_effect = interrupted_create

        result = agent.run_conversation("start")

        assert result["interrupted"] is True
        assert session_events == ["active", "idle"]
        assert "on_session_end" in plugin_events
        active = agent._drain_pending_mailbox_deliveries()
        assert active.included is True
        assert active.deliveries[0].claim_token == "interrupted-error"
