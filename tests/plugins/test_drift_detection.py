"""Tests for the drift-detection plugin.

Covers:
- Step parsing from the three common SKILL.md formats.
- Drift detection heuristic (flag contradictions, tolerate ambiguity).
- Full hook flow: skill_view capture → drifting tool call → annotation
  via transform_tool_result.
- Cleanup on session end.
- No false positives on read/search tools and multi-step sequences.

Uses only stdlib + pytest + the plugin module itself. No live agent, no
network, no plugin manager — hooks are called directly with constructed
kwargs.
"""

import json
import importlib.util
from pathlib import Path

import pytest

# The plugin lives under plugins/drift-detection/ — a hyphenated dir name
# that Python can't import as a module. Load it directly from its file path.
_PLUGIN_FILE = (
    Path(__file__).resolve().parents[2]
    / "plugins"
    / "drift-detection"
    / "__init__.py"
)
_spec = importlib.util.spec_from_file_location("drift_detection", _PLUGIN_FILE)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_on_post_tool_call = _mod._on_post_tool_call
_on_session_end = _mod._on_session_end
_on_transform_tool_result = _mod._on_transform_tool_result
_reset_for_test = _mod._reset_for_test
_state_for_test = _mod._state_for_test
detect_drift = _mod.detect_drift
parse_numbered_steps = _mod.parse_numbered_steps
register = _mod.register


@pytest.fixture(autouse=True)
def _clean_state():
    """Each test starts with no tracked sessions."""
    _reset_for_test()
    yield
    _reset_for_test()


# ---------------------------------------------------------------------------
# parse_numbered_steps
# ---------------------------------------------------------------------------


class TestParseNumberedSteps:
    def test_plain_numbered_list(self):
        content = (
            "## Procedure\n\n"
            "1. **Survey the codebase.** Use search_files to find peers.\n"
            "2. **Check the validator.** Read the schema constraints.\n"
            "3. **Write the skill.** Create SKILL.md with frontmatter.\n"
        )
        steps = parse_numbered_steps(content)
        assert len(steps) == 3
        assert "Survey the codebase" in steps[0]
        assert "Check the validator" in steps[1]
        assert "Write the skill" in steps[2]

    def test_bold_step_markers(self):
        content = (
            "## The canonical workflow\n\n"
            "**Step 1 — Capture first.** Take a snapshot.\n"
            "**Step 2 — Click by element index.** Use the accessibility tree.\n"
            "**Step 3 — Verify.** Re-snapshot after each action.\n"
        )
        steps = parse_numbered_steps(content)
        assert len(steps) == 3
        assert "Capture first" in steps[0]
        assert "Click by element index" in steps[1]
        assert "Verify" in steps[2]

    def test_numbered_headers(self):
        content = (
            "## 1. Viewing Issues\nUse search_files.\n\n"
            "## 2. Creating Issues\nUse write_file.\n\n"
        )
        # Numbered headers aren't matched by the list regex, but the
        # number-in-header form yields no list items — parser returns [].
        # This documents the (intentional) limitation: we only catch
        # numbered *list items*, not section headers.
        steps = parse_numbered_steps(content)
        # Headers "1. Viewing Issues" have len >= 8 so they'd match the
        # numbered-list regex if it were greedy enough; verify we get
        # something reasonable rather than garbage.
        assert all(len(s) >= 8 for s in steps)

    def test_short_items_filtered(self):
        content = (
            "1. ok this is long enough to be a step\n"
            "2. short\n"  # filtered (< 8 chars after stripping)
            "3. also a sufficiently long step description\n"
        )
        steps = parse_numbered_steps(content)
        # "short" is filtered; the sequence 1,3 should NOT both pass the
        # strictly-increasing check. We accept either [first] or [] but
        # never a list containing "short".
        assert "short" not in steps

    def test_empty_content(self):
        assert parse_numbered_steps("") == []
        assert parse_numbered_steps(None) == []  # type: ignore[arg-type]

    def test_no_steps_in_prose(self):
        content = "This skill helps with things. There are no numbered steps here."
        assert parse_numbered_steps(content) == []

    def test_cap_at_30(self):
        content = "\n".join(f"{i}. This is step number {i} and it is long enough." for i in range(1, 50))
        steps = parse_numbered_steps(content)
        assert len(steps) == 30


# ---------------------------------------------------------------------------
# detect_drift heuristic
# ---------------------------------------------------------------------------


class TestDetectDrift:
    def test_contradiction_flagged(self):
        drift = detect_drift(
            expected_step="Use `terminal` to run the build",
            observed_tool="write_file",
            observed_args={},
        )
        assert drift is not None
        assert "terminal" in drift
        assert "write_file" in drift

    def test_matching_tool_no_drift(self):
        drift = detect_drift(
            expected_step="Use `terminal` to run the build",
            observed_tool="terminal",
            observed_args={},
        )
        assert drift is None

    def test_step_without_tool_name_no_drift(self):
        # Step mentions no specific tool — can't contradict it.
        drift = detect_drift(
            expected_step="Review the output and decide what to do next",
            observed_tool="write_file",
            observed_args={},
        )
        assert drift is None

    def test_observed_non_action_tool_no_drift(self):
        # read_file/search_files aren't "progress" — not a drift signal.
        drift = detect_drift(
            expected_step="Use `terminal` to run the build",
            observed_tool="read_file",
            observed_args={},
        )
        assert drift is None

    def test_empty_step(self):
        assert detect_drift("", "terminal", {}) is None


# ---------------------------------------------------------------------------
# Full hook flow
# ---------------------------------------------------------------------------


def _skill_view_result(name: str, content: str) -> str:
    return json.dumps({"success": True, "name": name, "content": content})


class TestHookFlow:
    def test_skill_view_captures_steps(self):
        _on_post_tool_call(
            tool_name="skill_view",
            args={"name": "deploy"},
            result=_skill_view_result(
                "deploy",
                "1. **Build.** Use `terminal` to build the image.\n"
                "2. **Push.** Use `terminal` to push to registry.\n",
            ),
            session_id="s1",
        )
        state = _state_for_test("s1")
        assert state is not None
        assert state["skill"] == "deploy"
        assert len(state["steps"]) == 2
        assert state["cursor"] == 0

    def test_drift_on_wrong_tool(self):
        _on_post_tool_call(
            tool_name="skill_view",
            args={"name": "deploy"},
            result=_skill_view_result(
                "deploy",
                "1. **Build.** Use `terminal` to build the image.\n",
            ),
            session_id="s2",
        )
        # Agent should call terminal but calls write_file — drift.
        _on_post_tool_call(
            tool_name="write_file",
            args={},
            result="{}",
            session_id="s2",
            tool_call_id="tc1",
        )
        state = _state_for_test("s2")
        assert len(state["drifts"]) == 1
        assert state["drifts"][0]["observed_tool"] == "write_file"
        assert state["drifts"][0]["tool_call_id"] == "tc1"

    def test_annotation_added_to_result(self):
        _on_post_tool_call(
            tool_name="skill_view",
            args={"name": "deploy"},
            result=_skill_view_result(
                "deploy",
                "1. **Build.** Use `terminal` to build the image.\n",
            ),
            session_id="s3",
        )
        _on_post_tool_call(
            tool_name="write_file",
            args={},
            result="{}",
            session_id="s3",
            tool_call_id="tc2",
        )
        annotated = _on_transform_tool_result(
            tool_name="write_file",
            result=json.dumps({"success": True, "message": "wrote file"}),
            session_id="s3",
            tool_call_id="tc2",
        )
        assert annotated is not None
        payload = json.loads(annotated)
        assert payload["drift_detected"] is True
        assert "Drift detected" in payload["message"]
        assert "deploy" in payload["message"]

    def test_annotation_consumed_once(self):
        _on_post_tool_call(
            tool_name="skill_view",
            args={"name": "deploy"},
            result=_skill_view_result(
                "deploy",
                "1. **Build.** Use `terminal` to build the image.\n",
            ),
            session_id="s4",
        )
        _on_post_tool_call(
            tool_name="write_file",
            args={},
            result="{}",
            session_id="s4",
            tool_call_id="tc3",
        )
        first = _on_transform_tool_result(
            tool_name="write_file", result="{}", session_id="s4", tool_call_id="tc3",
        )
        second = _on_transform_tool_result(
            tool_name="write_file", result="{}", session_id="s4", tool_call_id="tc3",
        )
        assert first is not None
        # Already consumed — second call returns None (no re-annotation).
        assert second is None

    def test_matching_tool_advances_cursor(self):
        _on_post_tool_call(
            tool_name="skill_view",
            args={"name": "deploy"},
            result=_skill_view_result(
                "deploy",
                "1. **Build.** Use `terminal` to build the image.\n"
                "2. **Push.** Use `terminal` to push to registry.\n",
            ),
            session_id="s5",
        )
        # Correct tool for step 1 → cursor advances.
        _on_post_tool_call(
            tool_name="terminal",
            args={},
            result="{}",
            session_id="s5",
        )
        state = _state_for_test("s5")
        assert state["cursor"] == 1
        assert len(state["drifts"]) == 0

    def test_read_tools_ignored(self):
        _on_post_tool_call(
            tool_name="skill_view",
            args={"name": "deploy"},
            result=_skill_view_result(
                "deploy",
                "1. **Build.** Use `terminal` to build the image.\n",
            ),
            session_id="s6",
        )
        # read_file doesn't contradict (not an action tool) and shouldn't
        # advance the cursor either — but it must not flag a drift.
        _on_post_tool_call(
            tool_name="read_file",
            args={},
            result="{}",
            session_id="s6",
        )
        state = _state_for_test("s6")
        assert len(state["drifts"]) == 0

    def test_session_end_clears_state(self):
        _on_post_tool_call(
            tool_name="skill_view",
            args={"name": "deploy"},
            result=_skill_view_result(
                "deploy", "1. **Build.** Use `terminal`.\n",
            ),
            session_id="s7",
        )
        assert _state_for_test("s7") is not None
        _on_session_end(session_id="s7")
        assert _state_for_test("s7") is None

    def test_no_skill_loaded_no_crash(self):
        # Tool call with no prior skill_view — plugin is a no-op.
        _on_post_tool_call(
            tool_name="terminal",
            args={},
            result="{}",
            session_id="s8",
        )
        assert _state_for_test("s8") is None

    def test_skill_view_no_steps_not_tracked(self):
        # SKILL.md with no numbered steps — don't track (nothing to compare).
        _on_post_tool_call(
            tool_name="skill_view",
            args={"name": "vague"},
            result=_skill_view_result("vague", "Just some prose. No steps."),
            session_id="s9",
        )
        assert _state_for_test("s9") is None

    def test_task_id_fallback_when_no_session(self):
        _on_post_tool_call(
            tool_name="skill_view",
            args={"name": "deploy"},
            result=_skill_view_result(
                "deploy", "1. **Build.** Use `terminal`.\n",
            ),
            task_id="task-99",
        )
        state = _state_for_test("", task_id="task-99")
        assert state is not None
        assert state["skill"] == "deploy"

    def test_append_note_plain_string(self):
        # Non-JSON result — note appended as plain string.
        _on_post_tool_call(
            tool_name="skill_view",
            args={"name": "deploy"},
            result=_skill_view_result(
                "deploy", "1. **Build.** Use `terminal`.\n",
            ),
            session_id="s10",
        )
        _on_post_tool_call(
            tool_name="write_file",
            args={},
            result="plain text result",
            session_id="s10",
            tool_call_id="tc10",
        )
        annotated = _on_transform_tool_result(
            tool_name="write_file",
            result="plain text result",
            session_id="s10",
            tool_call_id="tc10",
        )
        assert annotated is not None
        assert "Drift detected" in annotated
        assert "plain text result" in annotated


# ---------------------------------------------------------------------------
# register() smoke — the plugin must expose register() for PluginManager.
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_calls_ctx(self):
        class _FakeCtx:
            def __init__(self):
                self.hooks = []

            def register_hook(self, name, cb):
                self.hooks.append((name, cb))

        ctx = _FakeCtx()
        register(ctx)
        names = [h[0] for h in ctx.hooks]
        assert "post_tool_call" in names
        assert "transform_tool_result" in names
        assert "on_session_end" in names
