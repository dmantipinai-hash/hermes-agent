"""Ф3 (Локи 03.10): чекпоинт-предохранитель.

Задание: ``plans/2026-10-03-loki-autonomy-fixes-zcode.md`` §3. Две части:
1) напоминание воркеру при ~2/3 бюджета итераций («закоммить и зафиксируй
   передатку») — в run_agent._append_guardrail_observation;
2) секция «ПЕРЕДАТКА» в контексте ретрая после незавершённого рана
   (последний комментарий + git status воркспейса + хвост лога рана).
"""

from __future__ import annotations

import subprocess
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


def _task_with_ended_run(conn, outcome, assignee="worker"):
    tid = kb.create_task(conn, title="task", assignee=assignee)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer=assignee) is not None
    kb._end_run(conn, tid, outcome=outcome, status="ready", summary="умер на финале")
    kb.add_comment(conn, tid, author="worker", body="Успел поправить миграцию,\n"
                                                    "осталось прогнать тесты.")
    return tid


def _git_workspace(conn, tid):
    ws = kb.resolve_workspace(kb.get_task(conn, tid))
    ws.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    subprocess.run(["git", "-C", str(ws), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(ws), "config", "user.name", "t"], check=True)
    (ws / "partial.txt").write_text("незакоммиченная наработка", encoding="utf-8")
    return ws


# ---------------------------------------------------------------------------
# Часть 2: ПЕРЕДАТКА в build_worker_context
# ---------------------------------------------------------------------------


def test_handoff_section_after_timed_out_run(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _task_with_ended_run(conn, outcome="timed_out")
        ws = _git_workspace(conn, tid)
        log = kb.worker_logs_dir() / f"{tid}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("шаг 1 ок\nшаг 2 ок\nфинальный прогон: начат и умер\n",
                       encoding="utf-8")

        ctx = kb.build_worker_context(conn, tid)

        assert "ПЕРЕДАТКА" in ctx
        assert "timed_out" in ctx
        assert "последний комментарий" in ctx
        assert "поправить миграцию" in ctx
        assert "partial.txt" in ctx, "git status должен показать грязный воркспейс"
        assert "финальный прогон" in ctx, "хвост лога рана должен попасть в контекст"


def test_no_handoff_section_after_completed_run(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _task_with_ended_run(conn, outcome="completed")
        _git_workspace(conn, tid)
        ctx = kb.build_worker_context(conn, tid)
        assert "ПЕРЕДАТКА" not in ctx


def test_no_handoff_section_after_blocked_run(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _task_with_ended_run(conn, outcome="blocked")
        ctx = kb.build_worker_context(conn, tid)
        assert "ПЕРЕДАТКА" not in ctx, (
            "блок уже несёт свой собственный handoff (reason в трейде)"
        )


def test_handoff_section_flag_off(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(kb, "timeout_handoff_enabled", lambda: False)
    with kb.connect_closing() as conn:
        tid = _task_with_ended_run(conn, outcome="timed_out")
        _git_workspace(conn, tid)
        ctx = kb.build_worker_context(conn, tid)
        assert "ПЕРЕДАТКА" not in ctx


# ---------------------------------------------------------------------------
# Часть 1: напоминание о бюджете в цикле агента
# ---------------------------------------------------------------------------


def _bare_agent(budget_max: int, used: int):
    from run_agent import AIAgent
    from agent.iteration_budget import IterationBudget
    from agent.tool_guardrails import (
        ToolCallGuardrailConfig,
        ToolCallGuardrailController,
    )

    agent = AIAgent.__new__(AIAgent)
    agent._tool_guardrails = ToolCallGuardrailController(ToolCallGuardrailConfig())
    agent._tool_guardrail_halt_decision = None
    agent._awareness = None
    agent.iteration_budget = IterationBudget(budget_max)
    for _ in range(used):
        agent.iteration_budget.consume()
    return agent


def test_budget_reminder_fires_at_two_thirds(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    agent = _bare_agent(9, 6)  # ровно 2/3
    result = agent._append_guardrail_observation(
        "read_file", {"path": "x"}, "ok", failed=False
    )
    assert "Бюджет итераций на исходе" in result
    assert "6/9" in result
    assert "kanban_comment" in result


def test_budget_reminder_fires_once(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    agent = _bare_agent(9, 6)
    first = agent._append_guardrail_observation(
        "read_file", {"path": "x"}, "ok", failed=False
    )
    second = agent._append_guardrail_observation(
        "read_file", {"path": "x"}, "ok", failed=False
    )
    assert "Бюджет итераций" in first
    assert "Бюджет итераций" not in second, "напоминание должно быть однократным"


def test_no_budget_reminder_below_threshold(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    agent = _bare_agent(9, 3)
    result = agent._append_guardrail_observation(
        "read_file", {"path": "x"}, "ok", failed=False
    )
    assert "Бюджет итераций" not in result


def test_no_budget_reminder_outside_kanban(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    agent = _bare_agent(9, 9)
    result = agent._append_guardrail_observation(
        "read_file", {"path": "x"}, "ok", failed=False
    )
    assert "Бюджет итераций" not in result, (
        "обычные (не-канбан) сессии не получают напоминание"
    )


def test_no_budget_reminder_when_flag_off(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    from hermes_cli import kanban_db as kb_db

    monkeypatch.setattr(kb_db, "timeout_handoff_enabled", lambda: False)
    agent = _bare_agent(9, 6)
    result = agent._append_guardrail_observation(
        "read_file", {"path": "x"}, "ok", failed=False
    )
    assert "Бюджет итераций" not in result
