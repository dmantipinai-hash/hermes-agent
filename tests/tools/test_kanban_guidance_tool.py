"""Ф1, инструментный уровень: kanban_comment(kind=guidance) разрешает карточку.

Дополняет tests/hermes_cli/test_kanban_guidance_unblock.py (db-уровень):
проверяем, что _handle_comment вызывает авто-переход и возвращает
``guidance_effect`` в ответе инструмента.
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
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="guidance-tool", assignee="brokk")
        kb.claim_task(conn, tid)
        assert kb.block_task(conn, tid, reason="жду решения", kind="needs_input")
    finally:
        conn.close()
    return {"home": home, "task_id": tid, "tmp_path": tmp_path}


def _handle(board_env, tool, args):
    from tools import kanban_tools as kt

    args.setdefault("task_id", board_env["task_id"])
    return json.loads(getattr(kt, f"_handle_{tool}")(args))


def _db():
    from hermes_cli import kanban_db as kb

    return kb.connect()


def test_guidance_comment_auto_unblocks_via_tool(board_env):
    out = _handle(
        board_env, "comment",
        {"body": "Решение: делаем вариант A, выделяем второй контур.", "kind": "guidance"},
    )
    assert out["ok"] is True
    assert out["guidance_effect"]["effect"] == "unblocked"

    conn = _db()
    try:
        from hermes_cli import kanban_db as kb

        assert kb.get_task(conn, board_env["task_id"]).status == "ready"
    finally:
        conn.close()


def test_info_comment_reports_no_effect(board_env):
    out = _handle(board_env, "comment", {"body": "заметка", "kind": "info"})
    assert out["ok"] is True
    assert "guidance_effect" not in out

    conn = _db()
    try:
        from hermes_cli import kanban_db as kb

        assert kb.get_task(conn, board_env["task_id"]).status == "blocked"
    finally:
        conn.close()
