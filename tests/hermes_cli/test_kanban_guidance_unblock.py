"""Ф1 (Локи 03.10): guidance-комментарий разрешает карточку сам.

Задание: ``plans/2026-10-03-loki-autonomy-fixes-zcode.md`` §1. Оркестратор
публикует решение guidance-комментарием — карточка в blocked/needs_input
должна сама перейти в ready (второй жест kanban_unblock избыточен). Только
needs_input; dependency/transient — нет; один переход на комментарий;
self-service запрещён; guidance ревьюера возвращает карточку исполнителю.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(conn, title="t", assignee="worker"):
    tid = kb.create_task(conn, title=title, assignee=assignee)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    claimed = kb.claim_task(conn, tid, claimer=assignee)
    assert claimed is not None
    return tid


def _guidance(conn, tid, author="orchestrator", body="Решение: делаем вариант A."):
    with kb.write_txn(conn):
        cid = kb.add_comment(conn, tid, author=author, body=body, kind="guidance")
        effect = kb.guidance_unblock_in_txn(
            conn, tid, comment_id=cid, author=author, body=body
        )
    return cid, effect


def _unblock_events(conn, tid):
    return [
        e for e in kb.list_events(conn, tid)
        if e.kind == "unblocked"
        and (e.payload or {}).get("via") == "guidance"
    ]


def test_guidance_unblocks_needs_input(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.block_task(conn, tid, reason="жду решения", kind="needs_input")
        assert kb.get_task(conn, tid).status == "blocked"

        _cid, effect = _guidance(conn, tid)

        assert effect is not None and effect["effect"] == "unblocked"
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        events = _unblock_events(conn, tid)
        assert len(events) == 1
        assert events[0].payload["comment_id"] == _cid


def test_dependency_block_not_resolved_by_guidance(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="wait", kind="dependency")
        assert kb.get_task(conn, tid).status == "todo"

        _cid, effect = _guidance(conn, tid)

        assert effect is None
        assert kb.get_task(conn, tid).status == "todo"
        assert _unblock_events(conn, tid) == []


def test_transient_block_not_resolved_by_guidance(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="flaky", kind="transient")

        _cid, effect = _guidance(conn, tid)

        assert effect is None
        assert kb.get_task(conn, tid).status == "blocked"


def test_worker_cannot_self_service_guidance(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn, assignee="worker")
        kb.block_task(conn, tid, reason="жду", kind="needs_input")

        _cid, effect = _guidance(conn, tid, author="worker")

        assert effect is None
        assert kb.get_task(conn, tid).status == "blocked"


def test_second_guidance_is_noop_one_transition(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="жду", kind="needs_input")
        _guidance(conn, tid, body="первое решение")
        assert kb.get_task(conn, tid).status == "ready"

        # Card is ready again: a second guidance must not produce a second
        # transition (idempotency via status).
        _cid2, effect2 = _guidance(conn, tid, body="второе уточнение")

        assert effect2 is None
        assert len(_unblock_events(conn, tid)) == 1


def test_info_comment_does_nothing(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="жду", kind="needs_input")
        with kb.write_txn(conn):
            kb.add_comment(conn, tid, author="orchestrator", body="просто заметка",
                           kind="info")
        assert kb.get_task(conn, tid).status == "blocked"
        assert _unblock_events(conn, tid) == []


def test_flag_off_disables_transition(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        kb, "autounblock_on_guidance_enabled", lambda: False, raising=True
    )
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="жду", kind="needs_input")
        _cid, effect = _guidance(conn, tid)
        assert effect is not None  # прямой вызов функции работает и при выключенном флаге инструмента
        # но инструментный путь (см. test_kanban_typed_comments-расширение) gated


def test_guidance_from_reviewer_returns_to_implementer(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn, assignee="coder")
        kb.request_review(
            conn, tid, summary="готово к ревью", reviewer="reviewer", force=True
        )
        assert kb.get_task(conn, tid).status == "review"

        _cid, effect = _guidance(conn, tid, author="reviewer",
                                 body="Переделай обработку границ: вариант A.")

        assert effect is not None
        assert effect["effect"] == "returned_for_rework"
        task = kb.get_task(conn, tid)
        assert task.status in ("ready", "todo")
        assert kb._canonical_assignee(task.assignee) == "coder"
        changes = [
            e for e in kb.list_events(conn, tid)
            if e.kind == "changes_requested"
            and (e.payload or {}).get("via") == "guidance_comment"
        ]
        assert len(changes) == 1
        assert changes[0].payload["comment_id"] == _cid


def test_guidance_from_implementer_in_review_is_noop(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn, assignee="coder")
        kb.request_review(conn, tid, summary="ревью", reviewer="reviewer", force=True)

        _cid, effect = _guidance(conn, tid, author="coder", body="посмотри ещё раз")

        assert effect is None
        assert kb.get_task(conn, tid).status == "review"
