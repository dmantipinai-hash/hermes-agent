"""Tests for the agent_manager backend (tools/agent_manager.py).

These are **invariant** tests, not change-detectors: they assert contracts
about how agent management behaves (file layout, isolation, guards), not
snapshots of specific role names or counts. Adding a new role or a new
tool to the toolset will not break these.

What's covered:
  - create_agent produces a profile with the right files on disk
    (config.yaml, .env, SOUL.md, profile.yaml) — written by ABSOLUTE path
  - .env is written mode 0600 (secret hygiene)
  - the ambient HERMES_HOME of the caller is NOT mutated when writing a
    child profile (the load-bearing isolation invariant)
  - set_agent_model overwrites the model section without losing other keys
  - delete_agent refuses when the agent has running kanban tasks (guard),
    and succeeds with force=True
  - delete_agent refuses on the built-in 'default' profile
  - list_agents reflects created agents with the expected fields
  - assign_task lands a real row in kanban.db with the right assignee
  - role validation: unknown role is rejected before any file is created
  - toolset wiring: every declared agent_manager tool is registered/resolvable
"""

import json
import os
import stat
from pathlib import Path

import pytest


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME + Path.home() for agent_manager tests.

    Mirrors the canonical pattern from tests/hermes_cli/test_profiles.py:
    Path.home() must be redirected because _get_profiles_root() anchors to
    Path.home() / ".hermes" / "profiles" (NOT to HERMES_HOME — that's the
    one place profile code is HOME-anchored by design, see AGENTS.md).
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return tmp_path


# ---------------------------------------------------------------------------
# create_agent — file layout + isolation
# ---------------------------------------------------------------------------

class TestCreateAgentFiles:
    def _create(self, profile_env, **overrides):
        from tools import agent_manager as am
        kwargs = dict(
            name="finance",
            role="researcher",
            description="Финансовый аналитик команды.",
            model="glm-5.1",
            provider="zai",
            base_url="https://api.z.ai/api/coding/paas/v4",
            api_key="sk-FAKE-test-key-123",
            api_key_env="GLM_API_KEY",
        )
        kwargs.update(overrides)
        return am.create_agent(**kwargs)

    def test_creates_profile_directory_with_expected_files(self, profile_env):
        result = self._create(profile_env)
        pdir = Path(result["path"])
        assert pdir.is_dir(), "profile dir must exist"
        # The four files our backend writes by absolute path:
        assert (pdir / "config.yaml").is_file()
        assert (pdir / ".env").is_file()
        assert (pdir / "SOUL.md").is_file()
        assert (pdir / "profile.yaml").is_file()

    def test_config_yaml_contains_model_section(self, profile_env):
        result = self._create(profile_env)
        cfg = (Path(result["path"]) / "config.yaml").read_text()
        assert "glm-5.1" in cfg
        assert "zai" in cfg
        assert "api.z.ai" in cfg

    def test_env_contains_api_key_under_named_var(self, profile_env):
        import yaml
        result = self._create(profile_env)
        env = (Path(result["path"]) / ".env").read_text()
        assert "GLM_API_KEY=sk-FAKE-test-key-123" in env

    def test_env_is_mode_0600(self, profile_env):
        result = self._create(profile_env)
        env_path = Path(result["path"]) / ".env"
        mode = stat.S_IMODE(env_path.stat().st_mode)
        assert mode == 0o600, f".env must be 0600, got {oct(mode)}"

    def test_soul_md_nonempty_and_mentions_role(self, profile_env):
        result = self._create(profile_env)
        soul = (Path(result["path"]) / "SOUL.md").read_text()
        assert len(soul) > 50
        # role hint from ROLE_TOOLSET_MAP['researcher'] should surface
        assert "research" in soul.lower()

    def test_profile_yaml_records_description_with_role_stamp(self, profile_env):
        import yaml
        result = self._create(profile_env)
        meta = yaml.safe_load((Path(result["path"]) / "profile.yaml").read_text())
        assert "description" in meta
        assert "researcher" in meta["description"]  # role stamped for recovery


class TestAmbientHomeIsolation:
    """The load-bearing invariant: writing a child profile must NOT mutate
    the orchestrator's own HERMES_HOME. If this breaks, an orchestrator
    running inside a live session would corrupt its own config.
    """

    def test_create_agent_does_not_change_caller_hermes_home(self, profile_env, monkeypatch):
        from tools import agent_manager as am
        caller_home = os.environ["HERMES_HOME"]
        am.create_agent(
            name="child",
            role="coder",
            model="glm-5.1",
            provider="zai",
            api_key="sk-x",
            api_key_env="GLM_API_KEY",
        )
        # HERMES_HOME must be untouched.
        assert os.environ["HERMES_HOME"] == caller_home
        # And the orchestrator's OWN config.yaml must NOT contain the child's model
        # (the whole point of the absolute-path write).
        caller_cfg = Path(caller_home) / "config.yaml"
        if caller_cfg.exists():
            assert "child" not in caller_cfg.read_text()

    def test_create_agent_does_not_write_api_key_to_caller_env(self, profile_env):
        from tools import agent_manager as am
        caller_home = os.environ["HERMES_HOME"]
        am.create_agent(
            name="child2",
            role="coder",
            api_key="sk-SECRET-must-not-leak",
            api_key_env="GLM_API_KEY",
            model="glm-5.1",
            provider="zai",
        )
        caller_env = Path(caller_home) / ".env"
        if caller_env.exists():
            assert "sk-SECRET-must-not-leak" not in caller_env.read_text()


# ---------------------------------------------------------------------------
# set_agent_model — overwrite model section, preserve everything else
# ---------------------------------------------------------------------------

class TestSetAgentModel:
    def _make(self, profile_env):
        from tools import agent_manager as am
        return am.create_agent(
            name="swap",
            role="coder",
            model="glm-5.1",
            provider="zai",
            base_url="https://api.z.ai/api/coding/paas/v4",
            api_key="sk-old",
            api_key_env="GLM_API_KEY",
        )

    def test_swaps_provider_and_model(self, profile_env):
        from tools import agent_manager as am
        self._make(profile_env)
        result = am.set_agent_model(
            "swap",
            provider="anthropic",
            model="claude-sonnet-4",
            api_key="sk-ant-new",
            api_key_env="ANTHROPIC_API_KEY",
        )
        assert result["provider"] == "anthropic"
        assert result["model"] == "claude-sonnet-4"

    def test_preserves_other_config_keys(self, profile_env):
        import yaml
        from tools import agent_manager as am
        pdir = Path(self._make(profile_env)["path"])
        # Inject an unrelated top-level key we expect to survive the swap.
        cfg_path = pdir / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())
        cfg["memory"] = {"provider": "honcho"}
        cfg_path.write_text(yaml.safe_dump(cfg))

        am.set_agent_model("swap", provider="anthropic", model="claude-x")

        after = yaml.safe_load(cfg_path.read_text())
        assert after.get("memory", {}).get("provider") == "honcho", \
            "set_agent_model must not clobber unrelated config sections"
        assert after["model"]["provider"] == "anthropic"

    def test_new_api_key_added_alongside_old(self, profile_env):
        from tools import agent_manager as am
        pdir = Path(self._make(profile_env)["path"])
        am.set_agent_model("swap", api_key="sk-ant-new", api_key_env="ANTHROPIC_API_KEY")
        env = (pdir / ".env").read_text()
        assert "GLM_API_KEY=sk-old" in env  # old preserved
        assert "ANTHROPIC_API_KEY=sk-ant-new" in env  # new added

    def test_returns_serializable_result(self, profile_env):
        """Guard against the earlier bug: 'updated' was a set (not JSON-serializable)."""
        from tools import agent_manager as am
        self._make(profile_env)
        result = am.set_agent_model("swap", model="m", provider="p")
        json.dumps(result)  # must not raise


# ---------------------------------------------------------------------------
# delete_agent — guards + force
# ---------------------------------------------------------------------------

class TestDeleteAgent:
    def test_refuses_default(self, profile_env):
        from tools import agent_manager as am
        with pytest.raises(ValueError, match="default"):
            am.delete_agent("default")

    def test_refuses_when_agent_has_running_task(self, profile_env):
        from tools import agent_manager as am
        from hermes_cli import kanban_db
        am.create_agent(name="busy", role="coder")
        # Create + force a task into 'running' for this assignee.
        am.assign_task("busy", "do something")
        conn = kanban_db.connect()
        try:
            conn.execute("UPDATE tasks SET status='running' WHERE assignee='busy'")
            conn.commit()
        finally:
            conn.close()

        result = am.delete_agent("busy")
        assert result["deleted"] is False
        assert result["active_tasks"] >= 1
        assert "force" in result["blocked_reason"].lower()

    def test_force_deletes_busy_agent(self, profile_env):
        from tools import agent_manager as am
        from hermes_cli import kanban_db
        am.create_agent(name="busy2", role="coder")
        am.assign_task("busy2", "do something")
        conn = kanban_db.connect()
        try:
            conn.execute("UPDATE tasks SET status='running' WHERE assignee='busy2'")
            conn.commit()
        finally:
            conn.close()

        result = am.delete_agent("busy2", force=True)
        assert result["deleted"] is True
        assert result["had_active_tasks"] >= 1

    def test_deletes_idle_agent(self, profile_env):
        from tools import agent_manager as am
        am.create_agent(name="idle", role="coder")
        result = am.delete_agent("idle")
        assert result["deleted"] is True


# ---------------------------------------------------------------------------
# list_agents + agent_status
# ---------------------------------------------------------------------------

class TestListAndStatus:
    def test_list_includes_created_agent(self, profile_env):
        from tools import agent_manager as am
        am.create_agent(name="visible", role="researcher", description="x")
        names = [a["name"] for a in am.list_agents()]
        assert "default" in names  # always present
        assert "visible" in names

    def test_list_entry_has_expected_fields(self, profile_env):
        from tools import agent_manager as am
        am.create_agent(name="fieldcheck", role="coder", model="glm-5.1", provider="zai")
        entry = next(a for a in am.list_agents() if a["name"] == "fieldcheck")
        for key in ("name", "model", "provider", "busy", "active_tasks", "description"):
            assert key in entry, f"missing field {key!r}"
        assert entry["model"] == "glm-5.1"
        assert entry["provider"] == "zai"
        assert entry["busy"] is False  # no tasks

    def test_status_returns_detail(self, profile_env):
        from tools import agent_manager as am
        am.create_agent(name="stat", role="reviewer")
        st = am.agent_status("stat")
        assert st["name"] == "stat"
        assert "active_tasks" in st
        assert "busy" in st


# ---------------------------------------------------------------------------
# assign_task — lands a real kanban row
# ---------------------------------------------------------------------------

class TestAssignTask:
    def test_creates_kanban_task_with_correct_assignee(self, profile_env):
        from tools import agent_manager as am
        from hermes_cli import kanban_db
        am.create_agent(name="worker", role="coder")
        result = am.assign_task("worker", "исследуй рынок X", title="market-research")
        assert "task_id" in result
        assert result["agent"] == "worker"

        conn = kanban_db.connect()
        try:
            tasks = kanban_db.list_tasks(conn, assignee="worker")
            assert len(tasks) >= 1
            titles = [(t.title if hasattr(t, "title") else t["title"]) for t in tasks]
            assert any("market-research" in t for t in titles)
        finally:
            conn.close()

    def test_refuses_unknown_agent(self, profile_env):
        from tools import agent_manager as am
        with pytest.raises(FileNotFoundError):
            am.assign_task("nonexistent-agent", "goal")


# ---------------------------------------------------------------------------
# Role validation
# ---------------------------------------------------------------------------

class TestRoleValidation:
    def test_unknown_role_rejected_before_file_creation(self, profile_env):
        from tools import agent_manager as am
        with pytest.raises(ValueError, match="Unknown role"):
            am.create_agent(name="badrole", role="nonexistent-role")
        # No profile dir should have been created.
        assert not (Path(profile_env) / ".hermes" / "profiles" / "badrole").exists()

    def test_known_role_accepted(self, profile_env):
        from tools import agent_manager as am
        from toolsets import ROLE_TOOLSET_MAP
        # Try every role in the map — invariant: all must be accepted.
        for role in ROLE_TOOLSET_MAP:
            am.create_agent(name=f"agent-{role}", role=role)


# ---------------------------------------------------------------------------
# Toolset wiring
# ---------------------------------------------------------------------------

class TestToolsetWiring:
    def test_toolset_resolves_all_registered_tools(self):
        from toolsets import TOOLSETS, resolve_toolset
        from tools.registry import registry
        import tools.agent_manager  # noqa: F401 — triggers registration

        declared = set(TOOLSETS["agent_manager"]["tools"])
        registered = set(registry.get_tool_names_for_toolset("agent_manager"))
        resolved = set(resolve_toolset("agent_manager"))

        # Every declared tool must be registered and resolvable.
        assert declared <= registered, \
            f"declared but not registered: {declared - registered}"
        assert declared <= resolved, \
            f"declared but not resolvable: {declared - resolved}"


# ---------------------------------------------------------------------------
# ask_agent — synchronous subprocess spawn (Architecture 2.0 §9.5)
# ---------------------------------------------------------------------------
#
# ask_agent spawns `hermes -p <profile> chat -Q -q <prompt>` as a subprocess
# and blocks until it answers. These tests mock subprocess.run so they never
# spawn a real process; they verify the argv, env, timeout, and error
# contracts documented in the function's docstring.
#
# Key invariants under test:
#   - argv is built from sys.executable (NOT `which hermes`) — see TD-4 fix.
#     The old path went through a shim that exec'd a foreign install,
#     silently running the child against the wrong codebase/memory.
#   - env["HERMES_HOME"] is pinned to the profile dir (isolation by
#     construction), and stale HERMES_KANBAN_* identity vars are scrubbed
#     so a child never inherits the parent's kanban-task identity.
#   - TimeoutError mentions assign_task as the async alternative.
#   - Non-zero exit / empty stdout raise RuntimeError with stderr tail.


class _FakeCompletedProcess:
    """Minimal stand-in for subprocess.CompletedProcess (the attr set ask_agent reads)."""

    def __init__(self, returncode=0, stdout="answer", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestAskAgent:
    """ask_agent: synchronous blocking question to a named profile."""

    def _make_profile(self, profile_env, name="finance"):
        from tools import agent_manager as am
        am.create_agent(name=name, role="researcher")
        return name

    def _patch_run(self, monkeypatch, captures, *, result=None, raises=None):
        """Replace subprocess.run with a capture+return stub.

        `captures` is a list that will receive the (cmd, env) the function
        built, so each test can assert on argv/env without re-reading the
        module. `result` is the CompletedProcess to return; `raises` is an
        exception to raise instead (for timeout/exit-path coverage).
        """
        def fake_run(cmd, env=None, capture_output=True, text=True,
                     timeout=None, stdin=None):
            captures["cmd"] = list(cmd)
            captures["env"] = dict(env or {})
            captures["timeout"] = timeout
            if raises is not None:
                raise raises
            return result if result is not None else _FakeCompletedProcess()

        # ask_agent imports subprocess INSIDE the function, so patch the
        # module-level attribute it resolves at call time.
        import subprocess
        monkeypatch.setattr(subprocess, "run", fake_run)
        return fake_run

    def test_unknown_agent_raises_before_spawn(self, profile_env, monkeypatch):
        # No profile created; subprocess.run must never be called.
        called = []
        monkeypatch.setattr("subprocess.run",
                            lambda *a, **kw: called.append(kw) or _FakeCompletedProcess())
        from tools import agent_manager as am
        with pytest.raises(FileNotFoundError, match="does not exist"):
            am.ask_agent("nonexistent", "hi")
        assert called == [], "subprocess.run must not be called for unknown agent"

    def test_empty_question_raises_value_error(self, profile_env, monkeypatch):
        self._make_profile(profile_env)
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: _FakeCompletedProcess())
        from tools import agent_manager as am
        with pytest.raises(ValueError, match="question is required"):
            am.ask_agent("finance", "   ")
            am.ask_agent("finance", "")

    def test_argv_uses_sys_executable_not_which_hermes(self, profile_env, monkeypatch):
        """TD-4: the child must inherit the parent's interpreter exactly.

        We do NOT assert on the literal sys.executable path (it's the test
        runner's venv), only that argv[0] is sys.executable and argv[1] is
        '-m hermes_cli.main' — the interpreter-bound form, not a bare
        'hermes' shim that could resolve to a foreign install.
        """
        import sys
        self._make_profile(profile_env)
        captures = {}
        self._patch_run(monkeypatch, captures)
        from tools import agent_manager as am
        am.ask_agent("finance", "hello")
        cmd = captures["cmd"]
        assert cmd[0] == sys.executable, \
            f"argv[0] must be sys.executable (TD-4); got {cmd[0]!r}"
        assert cmd[1] == "-m" and cmd[2] == "hermes_cli.main", \
            f"argv must be interpreter-bound; got {cmd[:3]}"
        assert "-p" in cmd and "finance" in cmd
        assert "-Q" in cmd, "-Q (machine-readable stdout) required"
        assert "-q" in cmd, "-q (single-query mode) required"
        # The prompt is the LAST -q argument value.
        qi = cmd.index("-q")
        assert cmd[qi + 1] == "hello"

    def test_env_pins_hermes_home_to_profile_dir(self, profile_env, monkeypatch):
        """The load-bearing isolation line: env["HERMES_HOME"] == profile dir."""
        from tools import agent_manager as am
        from hermes_cli import profiles as P
        self._make_profile(profile_env, name="finance")
        expected_home = str(P.get_profile_dir("finance"))
        captures = {}
        self._patch_run(monkeypatch, captures)
        am.ask_agent("finance", "hi")
        assert captures["env"]["HERMES_HOME"] == expected_home
        assert captures["env"]["HERMES_PROFILE"] == "finance"

    def test_env_scrubs_stale_kanban_identity(self, profile_env, monkeypatch):
        """A parent orchestrator that is itself a kanban worker must not leak
        its task/run/claim identity into the child."""
        from tools import agent_manager as am
        self._make_profile(profile_env)
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")
        monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "lockxyz")
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", "/tmp/parent")
        captures = {}
        self._patch_run(monkeypatch, captures)
        am.ask_agent("finance", "hi")
        env = captures["env"]
        for stale in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID",
                      "HERMES_KANBAN_CLAIM_LOCK", "HERMES_KANBAN_WORKSPACE"):
            assert stale not in env, f"{stale} must be scrubbed from child env"

    def test_context_is_appended_under_header(self, profile_env, monkeypatch):
        from tools import agent_manager as am
        self._make_profile(profile_env)
        captures = {}
        self._patch_run(monkeypatch, captures)
        am.ask_agent("finance", "what is X?", context="think about Y")
        qi = captures["cmd"].index("-q")
        prompt = captures["cmd"][qi + 1]
        assert prompt.startswith("what is X?")
        assert "## Context from the manager" in prompt
        assert "think about Y" in prompt

    def test_timeout_raises_with_assign_task_hint(self, profile_env, monkeypatch):
        import subprocess
        from tools import agent_manager as am
        self._make_profile(profile_env)
        captures = {}
        self._patch_run(
            monkeypatch, captures,
            raises=subprocess.TimeoutExpired(cmd=["x"], timeout=5),
        )
        with pytest.raises(TimeoutError, match="assign_task"):
            am.ask_agent("finance", "hi", timeout_seconds=5)
        # The cap passed to subprocess.run must be the explicit override.
        assert captures["timeout"] == 5

    def test_nonzero_exit_raises_runtime_error_with_stderr(self, profile_env, monkeypatch):
        from tools import agent_manager as am
        self._make_profile(profile_env)
        self._patch_run(
            monkeypatch, {},
            result=_FakeCompletedProcess(returncode=2, stdout="", stderr="boom! detailed"),
        )
        with pytest.raises(RuntimeError, match="exit 2"):
            am.ask_agent("finance", "hi")
        # Re-raise with the stderr tail surfaced.
        with pytest.raises(RuntimeError, match="boom! detailed"):
            am.ask_agent("finance", "hi")

    def test_empty_stdout_raises_runtime_error(self, profile_env, monkeypatch):
        from tools import agent_manager as am
        self._make_profile(profile_env)
        self._patch_run(
            monkeypatch, {},
            result=_FakeCompletedProcess(returncode=0, stdout="   \n", stderr="noise"),
        )
        with pytest.raises(RuntimeError, match="empty output"):
            am.ask_agent("finance", "hi")

    def test_success_returns_answer_and_elapsed(self, profile_env, monkeypatch):
        from tools import agent_manager as am
        self._make_profile(profile_env)
        self._patch_run(
            monkeypatch, {},
            result=_FakeCompletedProcess(returncode=0, stdout="  42 is the answer  "),
        )
        out = am.ask_agent("finance", "what?")
        assert out["agent"] == "finance"
        assert out["answer"] == "42 is the answer"  # stripped
        assert out["question"] == "what?"
        assert isinstance(out["elapsed_seconds"], float)


# ---------------------------------------------------------------------------
# ask_agent busy-registry (TD-5)
# ---------------------------------------------------------------------------
#
# ask_agent is a synchronous subprocess that doesn't write a kanban run row
# (by design, §8.3). So list_agents / agent_status — which probe the kanban
# board for busy status — reported a profile as idle while ask_agent was
# blocked on it. The fix is a process-local registry: ask_agent registers
# on entry, releases on exit (including error paths), and the status
# functions consult it.


class TestAskAgentBusyRegistry:
    """TD-5: list_agents / agent_status report busy during in-flight ask_agent."""

    def _make_profile(self, profile_env, name="finance"):
        from tools import agent_manager as am
        am.create_agent(name=name, role="researcher")
        return name

    def test_registry_empty_by_default(self):
        from tools import agent_manager as am
        am._reset_active_ask_registry_for_test()
        assert am._active_ask_count("finance") == 0

    def test_busy_set_during_in_flight_call(self, profile_env, monkeypatch):
        """While ask_agent is blocked, list_agents must report busy=True."""
        from tools import agent_manager as am
        am._reset_active_ask_registry_for_test()
        self._make_profile(profile_env)

        # Capture the moment we're inside subprocess.run (registry populated).
        seen = {}
        import subprocess

        def _slow(cmd, **kw):
            seen["during_count"] = am._active_ask_count("finance")
            return _FakeCompletedProcess(returncode=0, stdout="answer")

        monkeypatch.setattr(subprocess, "run", _slow)
        am.ask_agent("finance", "q")

        # During the call the registry had 1 entry.
        assert seen["during_count"] == 1
        # After the call returns, it's released.
        assert am._active_ask_count("finance") == 0

    def test_released_on_timeout(self, profile_env, monkeypatch):
        """TimeoutError path must still release the registry (no leak)."""
        import subprocess
        from tools import agent_manager as am
        am._reset_active_ask_registry_for_test()
        self._make_profile(profile_env)

        def _hang(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=1)

        monkeypatch.setattr(subprocess, "run", _hang)
        with pytest.raises(TimeoutError):
            am.ask_agent("finance", "q")
        assert am._active_ask_count("finance") == 0

    def test_released_on_subprocess_error(self, profile_env, monkeypatch):
        """RuntimeError path (non-zero exit) must release the registry."""
        from tools import agent_manager as am
        am._reset_active_ask_registry_for_test()
        self._make_profile(profile_env)
        import subprocess

        def _fail(cmd, **kw):
            return _FakeCompletedProcess(returncode=1, stdout="", stderr="boom")

        monkeypatch.setattr(subprocess, "run", _fail)
        with pytest.raises(RuntimeError):
            am.ask_agent("finance", "q")
        assert am._active_ask_count("finance") == 0

    def test_concurrent_calls_increment_count(self, profile_env, monkeypatch):
        """Two simultaneous ask_agent to the same profile → count = 2."""
        import threading
        from tools import agent_manager as am
        am._reset_active_ask_registry_for_test()
        self._make_profile(profile_env)

        barrier = threading.Barrier(2)
        results = []

        import subprocess

        def _synced(cmd, **kw):
            barrier.wait(timeout=5)  # both inside subprocess.run at once
            results.append(am._active_ask_count("finance"))
            return _FakeCompletedProcess(returncode=0, stdout="a")

        monkeypatch.setattr(subprocess, "run", _synced)

        t1 = threading.Thread(target=am.ask_agent, args=("finance", "q1"))
        t2 = threading.Thread(target=am.ask_agent, args=("finance", "q2"))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        # Both calls were in flight simultaneously: at least one observed
        # count == 2. (Both threads pass the barrier before either returns,
        # so the peak is 2; but the second reader may see 1 if the first
        # has already released — hence "at least one == 2", not "both".)
        assert 2 in results, f"expected peak count 2, got {results}"
        assert all(c >= 1 for c in results), f"each call saw itself in flight: {results}"
        # After both done, registry is empty.
        assert am._active_ask_count("finance") == 0


# ---------------------------------------------------------------------------
# list_comments_since — cursor-based thread read (Architecture 2.0 §9.5)
# ---------------------------------------------------------------------------
#
# The read half of the A1 mailbox. Workers call this between lifecycle
# checkpoints to see only NEW comments (id > last seen). The `id` column
# is AUTOINCREMENT, so it's strictly monotonic even when two comments land
# in the same wall-clock second — that ordering invariant is what makes the
# cursor safe (no re-delivery, no skipped messages).


class TestListCommentsSince:
    """list_comments_since: cursor-read of a task's comment thread."""

    def _make_board_with_task(self, profile_env):
        """Open a kanban board under the isolated HERMES_HOME and return (conn, task_id).

        connect() auto-runs init_db on first use, so no explicit init is
        needed — the schema is created lazily. Mirrors how TestAssignTask
        relies on assign_task → create_task to set up state.
        """
        from hermes_cli import kanban_db
        conn = kanban_db.connect()
        try:
            task_id = kanban_db.create_task(
                conn, title="thread-test", body="body", assignee=None,
            )
            return conn, task_id
        except Exception:
            conn.close()
            raise

    def test_empty_thread_returns_empty_list(self, profile_env):
        conn, task_id = self._make_board_with_task(profile_env)
        try:
            from hermes_cli import kanban_db
            assert kanban_db.list_comments_since(conn, task_id, since_id=0) == []
        finally:
            conn.close()

    def test_default_cursor_zero_returns_all(self, profile_env):
        conn, task_id = self._make_board_with_task(profile_env)
        try:
            from hermes_cli import kanban_db
            kanban_db.add_comment(conn, task_id, author="mgr", body="one")
            kanban_db.add_comment(conn, task_id, author="mgr", body="two")
            comments = kanban_db.list_comments_since(conn, task_id)  # since=0
            assert [c.body for c in comments] == ["one", "two"]
        finally:
            conn.close()

    def test_cursor_returns_only_newer(self, profile_env):
        conn, task_id = self._make_board_with_task(profile_env)
        try:
            from hermes_cli import kanban_db
            cid1 = kanban_db.add_comment(conn, task_id, author="mgr", body="first")
            cid2 = kanban_db.add_comment(conn, task_id, author="mgr", body="second")
            kanban_db.add_comment(conn, task_id, author="mgr", body="third")
            # Cursor at cid1 → only second+third.
            new = kanban_db.list_comments_since(conn, task_id, since_id=cid1)
            assert [c.body for c in new] == ["second", "third"]
            # Cursor at cid2 → only third.
            new2 = kanban_db.list_comments_since(conn, task_id, since_id=cid2)
            assert [c.body for c in new2] == ["third"]
        finally:
            conn.close()

    def test_strict_greater_than_not_inclusive(self, profile_env):
        """since_id == last comment's id must yield empty (strict >)."""
        conn, task_id = self._make_board_with_task(profile_env)
        try:
            from hermes_cli import kanban_db
            cid = kanban_db.add_comment(conn, task_id, author="mgr", body="only")
            assert kanban_db.list_comments_since(conn, task_id, since_id=cid) == []
        finally:
            conn.close()

    def test_ordering_by_id_not_timestamp(self, profile_env):
        """Two comments in the same second must still come back in id order.

        The id column is AUTOINCREMENT so it's strictly monotonic; this test
        would catch a regression that switched the ORDER BY to created_at
        (which can tie and yield non-deterministic ordering).
        """
        conn, task_id = self._make_board_with_task(profile_env)
        try:
            from hermes_cli import kanban_db
            kanban_db.add_comment(conn, task_id, author="a", body="c1")
            kanban_db.add_comment(conn, task_id, author="a", body="c2")
            kanban_db.add_comment(conn, task_id, author="a", body="c3")
            all_comments = kanban_db.list_comments_since(conn, task_id, since_id=0)
            ids = [c.id for c in all_comments]
            assert ids == sorted(ids), "comments must be ordered by id ascending"
            assert [c.body for c in all_comments] == ["c1", "c2", "c3"]
        finally:
            conn.close()

    def test_unknown_task_returns_empty(self, profile_env):
        """Unlike add_comment, list_comments_since does NOT validate task
        existence — it just SELECTs. An unknown task id yields []. This is
        the documented behaviour (the validator is add_comment itself)."""
        conn, _ = self._make_board_with_task(profile_env)
        try:
            from hermes_cli import kanban_db
            assert kanban_db.list_comments_since(conn, "t_nonexistent", since_id=0) == []
        finally:
            conn.close()
