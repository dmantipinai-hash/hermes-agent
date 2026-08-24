"""E2E wiring tests for awareness — the real resolution chain, no mocks
on the memory path.

Covers what the unit tests can't: config.yaml → ``agent_init`` → a live
``AwarenessController`` over the agent's real ``MemoryStoreV2`` → the
shared choke point in ``run_agent.py`` → the RECORD call wired into
``turn_finalizer.finalize_turn``. Pattern follows
``test_skip_memory_store_65429.py`` (real AIAgent, FakeOpenAI, temp
HERMES_HOME).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from run_agent import AIAgent


class _FakeOpenAI:
    def __init__(self, **kw):
        self.api_key = kw.get("api_key", "test")
        self.base_url = kw.get("base_url", "http://test")

    def close(self):
        pass


def _make_agent(monkeypatch, tmp_path, config_text=None, enabled_toolsets=None):
    home = tmp_path / "hm"
    home.mkdir(parents=True, exist_ok=True)
    if config_text:
        (home / "config.yaml").write_text(config_text)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("run_agent.get_tool_definitions", lambda **kw: [])
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("run_agent.OpenAI", _FakeOpenAI)
    return AIAgent(
        api_key="test-key",
        base_url="http://test",
        provider="openrouter",
        api_mode="chat_completions",
        max_iterations=1,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=enabled_toolsets if enabled_toolsets is not None else ["memory"],
        session_id="e2e-sess",
    )


class TestAgentInitWiring:
    def test_real_agent_gets_controller_over_real_store(self, monkeypatch, tmp_path):
        agent = _make_agent(monkeypatch, tmp_path)
        assert agent._awareness is not None
        assert agent._awareness.mode == "auto"  # merged-default config
        assert agent._awareness._store is agent._memory_store

    def test_user_config_mode_off_reaches_controller(self, monkeypatch, tmp_path):
        agent = _make_agent(
            monkeypatch, tmp_path,
            config_text="memory:\n  memory_enabled: true\nawareness:\n  mode: off\n",
        )
        assert agent._awareness is not None
        assert agent._awareness.mode == "off"
        assert not agent._awareness.enabled

    def test_no_store_means_no_controller(self, monkeypatch, tmp_path):
        # memory disabled in config and no memory toolset → no store → no
        # awareness (every call site guards with getattr).
        agent = _make_agent(
            monkeypatch, tmp_path,
            config_text="memory:\n  memory_enabled: false\n  user_profile_enabled: false\n",
            enabled_toolsets=[],
        )
        assert agent._memory_store is None
        assert agent._awareness is None


class TestFullCycleThroughChokePoint:
    def test_detect_note_and_record_over_real_store(self, monkeypatch, tmp_path):
        agent = _make_agent(monkeypatch, tmp_path)
        # Past experience from an earlier session of the same profile.
        agent._memory_store.add(
            target="memory",
            content=(
                "Stuck pattern: search_files — same tool failure (3x). "
                "What helped: read_file(/data/idx). Outcome: recovered."
            ),
            entry_type="pattern",
            written_by="awareness:past",
        )
        # Drive the real choke point both executor paths share; the 3rd
        # distinct-arg failure trips same_tool_failure_warning and the
        # recalled experience rides along in the same tool result.
        last = ""
        for i in range(3):
            last = agent._append_guardrail_observation(
                "search_files", {"query": f"q{i}"},
                json.dumps({"error": "no match"}), failed=True,
            )
        assert "[Tool loop warning:" in last
        assert "[Awareness:" in last
        assert "read_file(/data/idx)" in last
        assert agent._awareness.episodes_detected == 1
        # The agent changes strategy: one successful different call after
        # the stuck point is the captured recovery.
        agent._append_guardrail_observation(
            "read_file", {"path": "/data/idx"}, '{"content": "ok"}', failed=False,
        )

        # RECORD via the real finalize entry point (turn_finalizer wiring).
        from agent.turn_finalizer import finalize_turn

        finalize_turn(
            _FinalizeStub(agent),
            final_response="done",
            api_call_count=4,
            interrupted=False,
            failed=False,
            messages=[{"role": "user", "content": "task"}],
            conversation_history=[],
            effective_task_id="task",
            turn_id="turn",
            user_message="task",
            original_user_message="task",
            _should_review_memory=False,
            _turn_exit_reason="text_response(stop)",
        )
        rows = list(
            agent._memory_store._connect()
            .execute("SELECT written_by, content FROM memories WHERE type='pattern'")
            .fetchall()
        )
        assert any(
            r["written_by"] == "awareness:e2e-sess"
            and "Outcome: recovered" in r["content"]
            for r in rows
        )


class _FinalizeStub:
    """Minimal agent surface for finalize_turn (pattern: _LimitAgent in
    test_turn_finalizer_iteration_limit_exit.py); awareness + token fields
    are the parts under test here."""

    def __init__(self, agent):
        self._inner = agent
        self.max_iterations = 60
        self.iteration_budget = SimpleNamespace(
            remaining=60, used=4, max_total=60
        )
        self.quiet_mode = True
        self.model = "test-model"
        self.provider = "test-provider"
        self.base_url = ""
        self.session_id = agent.session_id
        self._awareness = agent._awareness
        self._tool_guardrail_halt_decision = None
        self._interrupt_message = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = []
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        for name in (
            "session_input_tokens", "session_output_tokens",
            "session_cache_read_tokens", "session_cache_write_tokens",
            "session_reasoning_tokens", "session_prompt_tokens",
            "session_completion_tokens", "session_total_tokens",
        ):
            setattr(self, name, 0)
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.session_cost_source = "test"

    def _handle_max_iterations(self, messages, api_call_count):
        return "summary"

    def _emit_status(self, *a, **k):
        pass

    def _safe_print(self, *a, **k):
        pass

    def _save_trajectory(self, *a, **k):
        pass

    def _cleanup_task_resources(self, *a, **k):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        pass

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _format_turn_completion_explanation(self, _reason):
        return ""

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **_kwargs):
        pass
