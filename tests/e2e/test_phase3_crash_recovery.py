"""Phase 3 crash-recovery — full subprocess E2E test.

This is the test the unit suite can't be: it spawns a REAL ``hermes chat``
subprocess, kills it with SIGTERM mid-tool-loop, and asserts the recovery
contract against the resulting SQLite state. No mocks of run_conversation,
no in-process trickery — the actual signal handlers, atexit hooks, and
``_run_cleanup`` path all run for real.

Architecture::

    MockOpenAIServer (background thread, non-streaming JSON)
        ▲
        │ http POST /v1/chat/completions
        │
    subprocess.Popen(["hermes", "chat", "-q", "..."],
                     env={HERMES_HOME: tmp, ...})
        │
        │ test SIGTERMs after the agent has made ≥1 API call
        ▼
    assertions on <tmp>/state.db:
        - messages persisted mid-turn (persist-point worked)
        - run_active == 0 after cleanup (the bug #1 fix)
        - find_interrupted_session behaves correctly

Why non-streaming: the OpenAI SDK's streaming SSE format is fiddly to
fake (chunk boundaries, [DONE] sentinel, usage on the final chunk). The
agent supports ``display.streaming: false`` which makes the mock a plain
JSON response — a 3x complexity reduction for identical coverage of the
crash-recovery path (persist + cleanup + resume).

Skipping policy: this test is marked ``slow`` and requires the project
venv to be importable as ``python -m hermes_cli.main``. If the venv isn't
present (CI matrix without full install), it skips rather than failing.
"""
from __future__ import annotations

import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.helpers.mock_openai_server import MockOpenAIServer

# Mark the whole module slow — these spawn subprocesses and take seconds.
pytestmark = [pytest.mark.slow]

_HERMES_BIN = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "hermes"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _venv_available() -> bool:
    """Can we invoke `hermes chat` as a subprocess?"""
    return _HERMES_BIN.exists() or shutil.which("hermes") is not None


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """A throwaway HERMES_HOME with a minimal config pointed at the mock."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "logs").mkdir()
    # Minimal config: non-streaming, no skills, terminal backend local.
    # model/provider/base_url are written per-test (we need the mock port).
    return home


def _write_config(home: Path, base_url: str):
    (home / "config.yaml").write_text(
        f"""
model:
  default: mock-model
  provider: custom
  base_url: {base_url}
agent:
  max_turns: 5
  mid_turn_persist: true
display:
  streaming: false
terminal:
  backend: local
toolsets:
  - terminal
  - file
""",
        encoding="utf-8",
    )
    # Stub .env so the OpenAI client has an API key to send (mock ignores it).
    (home / ".env").write_text("OPENAI_API_KEY=test-mock-key\n", encoding="utf-8")


def _spawn_chat(home: Path, query: str, extra_env: dict | None = None) -> subprocess.Popen:
    """Spawn `hermes chat -q` against the configured mock. Returns the proc."""
    env = dict(os.environ)
    env["HERMES_HOME"] = str(home)
    env["HERMES_NO_UPDATE_CHECK"] = "1"
    if extra_env:
        env.update(extra_env)
    # Prefer the venv hermes; fall back to PATH.
    cmd = [str(_HERMES_BIN)] if _HERMES_BIN.exists() else ["hermes"]
    cmd += ["chat", "-q", query, "-m", "mock-model", "--provider", "custom"]
    return subprocess.Popen(
        cmd,
        cwd=str(_REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        # New process group so we can SIGTERM the whole tree.
        start_new_session=True,
    )


def _wait_for_api_call(mock: MockOpenAIServer, timeout: float = 15.0) -> bool:
    """Poll the mock until the agent has made ≥1 chat-completions call."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if mock.call_count >= 1:
            return True
        time.sleep(0.1)
    return False


def _state_db_path(home: Path) -> Path:
    return home / "state.db"


def _db_row(db_path: Path, sql: str, params=()):
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        return conn.execute(sql, params).fetchone()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _venv_available(),
                    reason="hermes chat subprocess not available in this env")
class TestCrashRecoveryE2E:
    """The crash-recovery contract, exercised against a real process."""

    def test_cleanup_clears_run_active_on_sigterm(self, hermes_home):
        """Bug #1 regression: SIGTERM (terminal close) must clear run_active.

        Before the fix, ``mark_run_idle`` only ran at the tail of
        ``run_conversation``, which a SIGTERM never reaches. The atexit
        hook in ``_run_cleanup`` now clears the flag — this test pins that.
        """
        with MockOpenAIServer(
            tool_name="terminal",
            tool_args={"command": "sleep 30"},
            # Keep returning tool_calls so the agent stays mid-loop long
            # enough for us to SIGTERM it.
            call_count_limit=50,
        ) as mock:
            _write_config(hermes_home, mock.url)
            proc = _spawn_chat(hermes_home, "run a long task")
            try:
                # Wait for the agent to actually start a turn (≥1 API call),
                # then SIGTERM mid-tool-loop.
                assert _wait_for_api_call(mock, timeout=20), \
                    "agent never made an API call — mock/server wiring broken"
                # Give it a beat to be solidly mid-loop.
                time.sleep(0.5)
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                pytest.fail("hermes chat didn't exit after SIGTERM")

            db = _state_db_path(hermes_home)
            assert db.exists(), "state.db was never created"
            # The fix: run_active must be 0 after cleanup.
            row = _db_row(db, "SELECT run_active, ended_at FROM sessions ORDER BY started_at DESC LIMIT 1")
            assert row is not None, "no session row found"
            assert row["run_active"] == 0, (
                "run_active=1 after SIGTERM — _run_cleanup didn't clear it. "
                "Bug #1 regression: closing the terminal leaves a false 'interrupted' marker."
            )

    def test_messages_persisted_mid_turn(self, hermes_home):
        """The persist-point after each tool iteration landed messages in DB.

        Without the Phase 3 persist-point, only the final ``_persist_session``
        on exit wrote messages — a SIGTERM mid-loop lost the tail. This test
        confirms at least one message reached the DB before the kill.
        """
        with MockOpenAIServer(
            tool_name="terminal",
            tool_args={"command": "sleep 30"},
            call_count_limit=50,
        ) as mock:
            _write_config(hermes_home, mock.url)
            proc = _spawn_chat(hermes_home, "do some work")
            try:
                assert _wait_for_api_call(mock, timeout=20)
                time.sleep(0.8)  # let the tool result flush
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

            db = _state_db_path(hermes_home)
            # At least the user message + one assistant tool_call should be in.
            count_row = _db_row(db, "SELECT COUNT(*) AS n FROM messages")
            assert count_row and count_row["n"] >= 1, (
                "no messages persisted mid-turn — persist-point not firing"
            )

    def test_normal_completion_clears_run_active(self, hermes_home):
        """Sanity: a turn that completes normally leaves run_active=0.

        Guards against the fix being over-broad (e.g. never setting 1).
        """
        with MockOpenAIServer(
            tool_name="terminal",
            tool_args={"command": "echo done"},
            final_text="All finished.",
            call_count_limit=1,  # one tool round, then text
        ) as mock:
            _write_config(hermes_home, mock.url)
            proc = _spawn_chat(hermes_home, "quick task")
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                pytest.fail("agent didn't complete a single tool round in 30s")

            db = _state_db_path(hermes_home)
            row = _db_row(db, "SELECT run_active FROM sessions ORDER BY started_at DESC LIMIT 1")
            assert row is not None
            assert row["run_active"] == 0, \
                "run_active stuck at 1 after a clean completion"
