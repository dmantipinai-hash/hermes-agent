"""Field-feedback fixes from the 18–19.09 kanban diagnostics (Локи).

One test per live trap that fired in production:

  П1  a card pointed at a non-existent profile is silently unspawnable —
      kanban_create now answers with assignee_warning (and still creates
      the card; profiles may legitimately be created later).
  П2  the agent surface could add a parent→child edge but not remove one —
      kanban_unlink mirrors the CLI: removes the edge, re-evaluates the
      child's promotion immediately, errors on a missing edge.
  П3  workers learned the terminal-call rule only after losing a run to the
      protocol-violation detector — build_worker_context now ends with an
      explicit Turn protocol section.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def board_env(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "orchestrator")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return {"home": home, "tmp_path": tmp_path}


def _handle(board_env, tool, args):
    from tools import kanban_tools as kt

    return json.loads(getattr(kt, f"_handle_{tool}")(args))


# ---------------------------------------------------------------------------
# П1 — assignee warning on create
# ---------------------------------------------------------------------------

def test_create_warns_when_assignee_profile_missing(board_env):
    """The loki-phantom trap: unknown assignee must warn loudly in the
    response (with the known profile list) while still creating the card —
    cards-before-profiles is a legitimate workflow."""
    out = _handle(
        board_env,
        "create",
        {"title": "phantom", "assignee": "loki-ghost", "body": "x"},
    )
    assert out["ok"] is True
    warning = out.get("assignee_warning")
    assert warning and "loki-ghost" in warning
    # The fix must teach, not just scold: list what exists. In the isolated
    # home the default profile always exists.
    assert "default" in warning
    assert "unclaimed" in warning

    # The card really landed (not blocked by the warning).
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        assert kb.get_task(conn, out["task_id"]) is not None
    finally:
        conn.close()


def test_create_no_warning_for_existing_profile(board_env):
    out = _handle(
        board_env,
        "create",
        {"title": "real", "assignee": "default"},
    )
    assert out["ok"] is True
    assert out.get("assignee_warning") is None


# ---------------------------------------------------------------------------
# П2 — kanban_unlink
# ---------------------------------------------------------------------------

def _two_linked_tasks(board_env):
    """parent (todo, never done) -> child; linking demotes child to todo."""
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="parent", assignee="default")
        child = kb.create_task(conn, title="child", assignee="default")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        child_status = kb.get_task(conn, child).status
    finally:
        conn.close()
    assert child_status == "todo", "non-done parent must gate the child"
    return parent, child


def test_unlink_removes_edge_and_repromotes_child(board_env):
    parent, child = _two_linked_tasks(board_env)
    out = _handle(
        board_env,
        "unlink",
        {"parent_id": parent, "child_id": child},
    )
    assert out["ok"] is True
    # No unfinished parents remain — the child must be immediately
    # promotion-eligible again (unlink_tasks recomputes readiness).
    assert out["child_status"] == "ready"

    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        assert kb.parent_ids(conn, child) == []
    finally:
        conn.close()


def test_unlink_missing_edge_is_a_clear_error(board_env):
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="a", assignee="default")
        b = kb.create_task(conn, title="b", assignee="default")
    finally:
        conn.close()
    out = _handle(board_env, "unlink", {"parent_id": a, "child_id": b})
    assert out.get("ok") is not True
    assert "no dependency edge" in out["error"]


def test_unlink_registered_in_kanban_toolset(board_env):
    from toolsets import TOOLSETS

    assert "kanban_unlink" in TOOLSETS["kanban"]["tools"]


# ---------------------------------------------------------------------------
# П3 — turn protocol footer in the worker context
# ---------------------------------------------------------------------------

def test_worker_context_ends_with_turn_protocol(board_env):
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="protocol", assignee="default")
        ctx = kb.build_worker_context(conn, tid)
    finally:
        conn.close()
    assert "## Turn protocol" in ctx
    assert "kanban_complete" in ctx
    assert "protocol violation" in ctx
    # The footer is the last section — a worker scanning top-down meets the
    # rule right before its first action.
    assert ctx.rstrip().rfind("## Turn protocol") > ctx.rfind("## Comment thread")
