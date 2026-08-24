"""Tests for agent.awareness — the deterministic DETECT → RECORD slice.

Behavior contracts (not snapshots):

- a guardrail warn decision opens a stuck episode keyed by tool+detector;
- a past ``pattern`` entry surfaces as an appended note, once per episode;
- a turn that ends with an honest outcome records "pattern → what helped"
  with awareness provenance; interrupted turns record nothing;
- ``memory.write_approval`` suppresses recording; ``mode=off`` is inert;
- empty memory reads against a *non-empty* store nudge after N in a row
  and reset on a hit; an empty store never nudges;
- recovery briefs quote only whitelisted args (paths/commands/queries),
  never file contents.
"""

from __future__ import annotations

import json

import pytest

import tools.memory_tool as mt
from agent.awareness import AwarenessController
from agent.memory_store_v2 import MemoryStoreV2
from agent.tool_guardrails import (
    ToolGuardrailDecision,
)


@pytest.fixture()
def mem_dir(tmp_path, monkeypatch):
    """Point the memory directory at a temp dir (canonical fixture pattern)."""
    monkeypatch.setattr(mt, "get_memory_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture()
def store(mem_dir):
    s = MemoryStoreV2()
    s.load_from_disk()
    yield s
    s.close()


def _warn(tool="search_files", code="same_tool_failure_warning", count=3):
    return ToolGuardrailDecision(
        action="warn",
        code=code,
        message="loop warning",
        tool_name=tool,
        count=count,
        signature=None,
    )


def _fail_result(msg="no match"):
    return json.dumps({"success": False, "error": msg})


def _patterns(store):
    return list(
        store._connect()
        .execute("SELECT type, written_by, content FROM memories WHERE type='pattern'")
        .fetchall()
    )


class TestDetectAndRecord:
    def test_warn_opens_episode_and_records_on_recovery(self, store):
        aw = AwarenessController(store, {}, session_id="sess-1")
        aw.observe_tool_result(
            "search_files", {"query": "a"}, _fail_result(), failed=True,
            guardrail_decision=_warn(count=3),
        )
        aw.observe_tool_result(
            "terminal", {"command": "ls -la /tmp"}, '{"stdout": "ok"}', failed=False,
        )
        record = aw.finalize_turn(exit_reason="text_response(stop)")
        assert record is not None
        assert record["signature"] == "search_files:same_tool_failure"
        assert record["outcome"] == "recovered"
        rows = _patterns(store)
        assert len(rows) == 1
        assert rows[0]["written_by"] == "awareness:sess-1"
        assert "What helped: terminal(ls -la /tmp)" in rows[0]["content"]
        assert "Outcome: recovered" in rows[0]["content"]
        assert aw.patterns_recorded == 1

    def test_note_injected_once_from_past_pattern(self, store):
        store.add(
            target="memory",
            content=(
                "Stuck pattern: search_files — repeated exact failure (2x). "
                "Error: no match. What helped: read_file(/etc/hosts). "
                "Outcome: recovered."
            ),
            entry_type="pattern",
            written_by="awareness:old-session",
        )
        aw = AwarenessController(store, {})
        note = aw.observe_tool_result(
            "search_files", {"query": "a"}, _fail_result(), failed=True,
            guardrail_decision=_warn(),
        )
        assert note is not None
        assert "[Awareness:" in note
        assert "read_file(/etc/hosts)" in note
        assert "search_files:same_tool_failure" in note
        assert aw.notes_injected == 1
        # Once per episode: the next warning of the same signature is silent.
        note2 = aw.observe_tool_result(
            "search_files", {"query": "a"}, _fail_result(), failed=True,
            guardrail_decision=_warn(count=4),
        )
        assert note2 is None

    def test_note_skipped_when_no_past_experience(self, store):
        aw = AwarenessController(store, {})
        note = aw.observe_tool_result(
            "terminal", {"command": "x"}, _fail_result(), failed=True,
            guardrail_decision=_warn(tool="terminal"),
        )
        assert note is None

    def test_cross_turn_recall_closes_the_loop(self, store):
        """The whole point of the feature: turn 1 records what helped,
        turn 2's same stuck pattern gets it injected as a note."""
        aw = AwarenessController(store, {}, session_id="s")
        # Turn 1: stuck → recover → record.
        aw.observe_tool_result(
            "search_files", {"query": "a"}, _fail_result(), failed=True,
            guardrail_decision=_warn(),
        )
        aw.observe_tool_result(
            "read_file", {"path": "/x"}, '{"content": "ok"}', failed=False,
        )
        assert aw.finalize_turn(exit_reason="text_response(stop)") is not None
        # Turn 2: the same stuck pattern resurfaces.
        aw.reset_for_turn()
        note = aw.observe_tool_result(
            "search_files", {"query": "b"}, _fail_result(), failed=True,
            guardrail_decision=_warn(),
        )
        assert note is not None
        assert "[Awareness:" in note
        assert "read_file(/x)" in note
        assert "Outcome: recovered" in note

    def test_budget_exhausted_records_unresolved(self, store):
        aw = AwarenessController(store, {})
        aw.observe_tool_result(
            "search_files", {"query": "a"}, _fail_result(), failed=True,
            guardrail_decision=_warn(),
        )
        record = aw.finalize_turn(exit_reason="budget_exhausted")
        assert record is not None
        assert "unresolved (budget exhausted)" in record["content"]

    def test_guardrail_halt_records_halted(self, store):
        aw = AwarenessController(store, {})
        aw.observe_tool_result(
            "search_files", {"query": "a"}, _fail_result(), failed=True,
            guardrail_decision=_warn(),
        )
        record = aw.finalize_turn(exit_reason="guardrail_halt")
        assert record is not None
        assert "halted by tool-call guardrail" in record["content"]

    def test_interrupted_turn_records_nothing(self, store):
        aw = AwarenessController(store, {})
        aw.observe_tool_result(
            "search_files", {"query": "a"}, _fail_result(), failed=True,
            guardrail_decision=_warn(),
        )
        assert aw.finalize_turn(exit_reason="interrupted_by_user") is None
        assert _patterns(store) == []

    def test_clean_turn_records_nothing(self, store):
        aw = AwarenessController(store, {})
        aw.observe_tool_result(
            "terminal", {"command": "ls"}, '{"stdout": "ok"}', failed=False,
        )
        assert aw.finalize_turn(exit_reason="text_response(stop)") is None
        assert _patterns(store) == []

    def test_record_suppressed_by_write_approval(self, store):
        aw = AwarenessController(store, {}, write_approval=True)
        aw.observe_tool_result(
            "search_files", {"query": "a"}, _fail_result(), failed=True,
            guardrail_decision=_warn(),
        )
        aw.observe_tool_result(
            "terminal", {"command": "ls"}, '{"stdout": "ok"}', failed=False,
        )
        assert aw.finalize_turn(exit_reason="text_response(stop)") is None
        assert _patterns(store) == []
        assert aw.records_skipped == 1

    def test_mode_off_is_inert(self, store):
        aw = AwarenessController(store, {"mode": "off"})
        assert aw.observe_tool_result(
            "search_files", {"query": "a"}, _fail_result(), failed=True,
            guardrail_decision=_warn(),
        ) is None
        assert aw.finalize_turn(exit_reason="text_response(stop)") is None
        assert _patterns(store) == []

    def test_record_deduped_per_signature_within_session(self, store):
        aw = AwarenessController(store, {})
        for _ in range(2):
            aw.observe_tool_result(
                "search_files", {"query": "a"}, _fail_result(), failed=True,
                guardrail_decision=_warn(),
            )
            aw.finalize_turn(exit_reason="text_response(stop)")
            aw.reset_for_turn()
        assert len(_patterns(store)) == 1

    def test_recovery_brief_excludes_file_contents(self, store):
        aw = AwarenessController(store, {})
        aw.observe_tool_result(
            "search_files", {"query": "a"}, _fail_result(), failed=True,
            guardrail_decision=_warn(),
        )
        secret = "SECRET-BODY-" * 50
        aw.observe_tool_result(
            "write_file", {"path": "/tmp/a.py", "content": secret}, '{"success": true}',
            failed=False,
        )
        record = aw.finalize_turn(exit_reason="text_response(stop)")
        assert record is not None
        assert "write_file(/tmp/a.py)" in record["content"]
        assert "SECRET-BODY" not in record["content"]

    def test_no_match_memory_error_gets_special_signature(self, store):
        aw = AwarenessController(store, {})
        aw.observe_tool_result(
            "memory", {"action": "replace", "old_text": "x"},
            json.dumps({"success": False, "error": "No entry matched 'x'."}),
            failed=True,
            guardrail_decision=_warn(tool="memory"),
        )
        assert aw._open_episode is not None
        assert aw._open_episode.signature == "memory:no_match"

    def test_yaml_boolean_mode_values_normalize(self, store):
        # Hand-edited `awareness.mode: off` parses as YAML 1.1 boolean False.
        assert AwarenessController(store, {"mode": False}).mode == "off"
        assert AwarenessController(store, {"mode": True}).mode == "auto"
        assert AwarenessController(store, {"mode": "off"}).mode == "off"
        assert AwarenessController(store, {"mode": "deep"}).mode == "auto"


class TestEmptyRecallStreak:
    def _read(self, aw, count, query="zzz"):
        payload = json.dumps({"success": True, "query": query, "results": [], "count": count})
        return aw.observe_tool_result(
            "memory", {"action": "read", "query": query}, payload, failed=False,
        )

    def test_streak_nudges_after_threshold(self, store):
        store.add(target="memory", content="live entry about docker", entry_type="fact")
        aw = AwarenessController(store, {})
        assert self._read(aw, 0) is None
        assert self._read(aw, 0) is None
        note = self._read(aw, 0)
        assert note is not None
        assert "consecutive memory reads" in note
        assert aw.empty_streaks_hit == 1

    def test_streak_resets_on_hit(self, store):
        store.add(target="memory", content="live entry about docker", entry_type="fact")
        aw = AwarenessController(store, {})
        self._read(aw, 0)
        self._read(aw, 0)
        assert self._read(aw, 1) is None  # a hit resets the streak
        assert self._read(aw, 0) is None
        assert self._read(aw, 0) is None
        assert aw.empty_streaks_hit == 0  # never reached the threshold again

    def test_empty_store_never_nudges(self, store):
        aw = AwarenessController(store, {})
        for _ in range(5):
            assert self._read(aw, 0) is None
        assert aw.empty_streaks_hit == 0


class TestChokePointIntegration:
    """The run_agent.py choke point both executor paths share."""

    def _agent(self, store):
        from run_agent import AIAgent
        from agent.tool_guardrails import (
            ToolCallGuardrailConfig,
            ToolCallGuardrailController,
        )

        agent = AIAgent.__new__(AIAgent)
        agent._tool_guardrails = ToolCallGuardrailController(ToolCallGuardrailConfig())
        agent._tool_guardrail_halt_decision = None
        agent._awareness = AwarenessController(store, {}, session_id="integ")
        return agent

    def test_note_appended_to_guardrail_annotated_result(self, store):
        store.add(
            target="memory",
            content=(
                "Stuck pattern: search_files — same tool failure (3x). "
                "What helped: read_file(/data/index). Outcome: recovered."
            ),
            entry_type="pattern",
            written_by="awareness:past",
        )
        agent = self._agent(store)
        # Three distinct-arg failures of one tool: the 3rd trips
        # same_tool_failure_warning (warn_after=3) and the awareness note
        # rides along in the same annotated tool result.
        for i in range(3):
            result = agent._append_guardrail_observation(
                "search_files", {"query": f"q{i}"}, _fail_result(), failed=True,
            )
        assert "[Tool loop warning:" in result
        assert "[Awareness:" in result
        assert "read_file(/data/index)" in result
        # RECORD path closes over the same episode.
        record = agent._awareness.finalize_turn(exit_reason="text_response(stop)")
        assert record is not None
        assert record["signature"] == "search_files:same_tool_failure"

    def test_block_decision_feeds_episode_without_note(self, store):
        from agent.tool_guardrails import ToolCallSignature

        agent = self._agent(store)
        decision = ToolGuardrailDecision(
            action="block",
            code="loop_web_search_cap",
            message="capped",
            tool_name="web_search",
            count=51,
            signature=ToolCallSignature.from_call("web_search", {"query": "x"}),
        )
        synthetic = agent._guardrail_block_result(decision)
        assert '"guardrail"' in synthetic
        assert agent._awareness._open_episode is not None
        assert agent._awareness._open_episode.signature == "web_search:loop_web_search_cap"
        record = agent._awareness.finalize_turn(exit_reason="guardrail_halt")
        assert record is not None
        assert "halted by tool-call guardrail" in record["content"]
