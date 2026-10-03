"""Ф2 (Локи 03.10): дайджест блокера по событию blocker_needs_triage.

Задание: ``plans/2026-10-03-loki-autonomy-fixes-zcode.md`` §2. Воркер блокируется
с анализом (варианты A/B/C с ценами) — асинхронный обработчик собирает
структурированный дайджест, постит его kind=info комментарием и оставляет
событие blocker_triage_done (его нотификатор доставляет подписчикам с wake).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_triage as kt


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _blocked_with_options(conn, reason=None):
    tid = kb.create_task(conn, title="cutover контейнера", assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    reason = reason or (
        "Не могу выбрать схему переезда. Противоречие: нельзя одновременно "
        "перейти без простоя и не удваивать расходы.\n"
        "Вариант A — blue/green, цена ~2x токенов на 3 часа, риск падения БД.\n"
        "Вариант B — по одной ноде, цена 30 мин простоя, риск расхождения схем.\n"
        "Вариант C — отложить, цена 0, риск слива дедлайна."
    )
    assert kb.block_task(conn, tid, reason=reason, kind="needs_input")
    return tid


def _events(conn, tid, kind):
    return [e for e in kb.list_events(conn, tid) if e.kind == kind]


def test_needs_input_block_emits_triage_event(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _blocked_with_options(conn)
        assert _events(conn, tid, "blocker_needs_triage"), (
            "needs_input блок должен поставить событие blocker_needs_triage"
        )


def test_transient_and_dependency_do_not_emit(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        t_transient = kb.create_task(conn, title="t1", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (t_transient,))
        kb.claim_task(conn, t_transient, claimer="worker")
        kb.block_task(conn, t_transient, reason="r", kind="transient")

        t_dep = kb.create_task(conn, title="t2", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (t_dep,))
        kb.claim_task(conn, t_dep, claimer="worker")
        kb.block_task(conn, t_dep, reason="r", kind="dependency")

        assert _events(conn, t_transient, "blocker_needs_triage") == []
        assert _events(conn, t_dep, "blocker_needs_triage") == []


def test_triage_posts_digest_comment_with_options(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _blocked_with_options(conn)
        processed = kt.process_pending_triage(conn)
        assert len(processed) == 1
        assert processed[0]["task_id"] == tid
        assert processed[0].get("comment_id")

        comments = kb.list_comments(conn, tid)
        digest_comments = [c for c in comments if c.author == "kanban-triage"]
        assert len(digest_comments) == 1
        body = digest_comments[0].body
        assert "Вариант" in body
        assert "A" in body and "B" in body and "C" in body
        assert "Противоречие" in body
        assert "ТРИЗ-рамка" in body

        done = _events(conn, tid, "blocker_triage_done")
        assert len(done) == 1
        payload = done[0].payload or {}
        assert payload.get("comment_id") == digest_comments[0].id
        assert payload.get("digest")


def test_triage_idempotent_second_pass_silent(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _blocked_with_options(conn)
        first = kt.process_pending_triage(conn)
        assert len(first) == 1
        second = kt.process_pending_triage(conn)
        assert second == [], "повторный прогон не должен ничего добавлять"
        digest_comments = [
            c for c in kb.list_comments(conn, tid) if c.author == "kanban-triage"
        ]
        assert len(digest_comments) == 1


def test_triage_dedups_reblock_of_same_kind(kanban_home: Path) -> None:
    # Интеграция: реальный ре-блок того же рода перехватывает loop-breaker
    # (BLOCK_RECURRENCE_LIMIT=2 → triage, события не ставит) — поэтому второй
    # pending-событие моделируем напрямую и проверяем контракт дедупа:
    # свежий дайджест по карточке в 30-мин окне → skip, без нового комментария.
    with kb.connect_closing() as conn:
        tid = _blocked_with_options(conn)
        kt.process_pending_triage(conn)

        with kb.write_txn(conn):
            kb._append_event(conn, tid, "blocker_needs_triage",
                             {"reason": "снова та же дилемма"})

        processed = kt.process_pending_triage(conn)
        assert len(processed) == 1
        assert processed[0].get("skipped") == "dedup"
        digest_comments = [
            c for c in kb.list_comments(conn, tid) if c.author == "kanban-triage"
        ]
        assert len(digest_comments) == 1, "дублирующий дайджест не постится"


def test_triage_skips_when_card_no_longer_blocked(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _blocked_with_options(conn)
        kb.unblock_task(conn, tid)  # событие есть, карточка уже не ждёт
        processed = kt.process_pending_triage(conn)
        assert processed and processed[0].get("skipped") == "stale"
        assert [
            c for c in kb.list_comments(conn, tid) if c.author == "kanban-triage"
        ] == []


def test_flag_off_disables_event_and_pass(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(kb, "blocker_triage_enabled", lambda: False)
    with kb.connect_closing() as conn:
        tid = _blocked_with_options(conn)
        assert _events(conn, tid, "blocker_needs_triage") == []
        assert kt.process_pending_triage(conn) == []


def test_wake_message_rendering(kanban_home: Path) -> None:
    msg = kt.render_triage_wake_message(
        "Дайджест блокера «cutover» (авто, kanban-triage)\nСуть:\n- жду решения",
        task_id="t_abc123",
        board_tag="[bybit] ",
        assignee_tag="@worker ",
    )
    assert msg.startswith("🧭 [bybit] @worker Kanban t_abc123 blocker digest —")
    assert "Дайджест блокера" in msg
    assert "Суть" in msg


def test_extract_options_and_contradiction(kanban_home: Path) -> None:
    text = (
        "Вариант A — быстрый фикс, цена 2 часа.\n"
        "B) полный рефакторинг, $50, риск регрессии.\n"
        "Просто строка без маркера варианта."
    )
    options = kt.extract_options(text)
    assert [o["label"] for o in options] == ["A", "B"]
    assert options[1]["price"] == "$"
    assert options[1]["risk"] is True
    assert options[0]["risk"] is False

    contra = kt.detect_contradiction(
        "Нельзя одновременно держать старый формат и не тратить память на дубль."
    )
    assert contra and "Нельзя одновременно" in contra
    assert kt.detect_contradiction("Просто описание без противоречия.") is None
