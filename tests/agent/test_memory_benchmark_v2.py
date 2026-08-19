"""Behavioral benchmark for the typed memory store (Phase 1, step 1.8).

20 scenario-style tests answering the product question: "does the memory
actually change what the agent knows across sessions?" Each test is a
mini-narrative (save → reload/sessions later → recall/snapshot) rather than
an implementation detail check. Idea: Claude Opus §5.1 (handcrafted cases
beat auto-generated summaries).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tools.memory_tool as mt
from agent.memory_store_v2 import MemoryStoreV2


@pytest.fixture()
def mem_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "get_memory_dir", lambda: tmp_path)
    return tmp_path


def _fresh(mem_dir: Path, **kw) -> MemoryStoreV2:
    """New store instance = new agent session (same on-disk memory)."""
    s = MemoryStoreV2(**kw)
    s.load_from_disk()
    return s


# ---------------------------------------------------------------- scenarios 1-5:
# Decisions survive sessions and stop the agent from re-proposing rejected ideas.

def test_scenario_1_decision_survives_new_session(mem_dir):
    s1 = _fresh(mem_dir)
    s1.add("memory", "Не строить локального AI-персонажа — высокая нагрузка на ПК",
           entry_type="decision", importance=0.9)
    s1.close()
    s2 = _fresh(mem_dir)
    found = s2.recall("персонаж")
    assert found["count"] == 1 and found["results"][0]["type"] == "decision"
    s2.close()


def test_scenario_2_rejected_decision_not_in_active_recall_after_deprecate(mem_dir):
    s = _fresh(mem_dir)
    s.add("memory", "Храним всё в одной большой таблице", entry_type="decision")
    s.deprecate("memory", "большой таблице", reason="перешли на доменные таблицы")
    s.close()
    s2 = _fresh(mem_dir)
    assert s2.recall("таблице")["count"] == 0, "rejected idea must be invisible to active recall"
    s2.close()


def test_scenario_3_decision_with_reason_persists_reason(mem_dir):
    s = _fresh(mem_dir)
    s.add("memory", "Отказ от Kubernetes: дорого для одного сервера", entry_type="decision")
    s.close()
    s2 = _fresh(mem_dir)
    hit = s2.recall("Kubernetes")["results"]
    assert hit and "дорого" in hit[0]["content"]
    s2.close()


def test_scenario_4_reversal_flow_deprecate_then_add(mem_dir):
    s = _fresh(mem_dir)
    s.add("memory", "Отвечаем всегда кратко, одной строкой", entry_type="constraint")
    # User reverses the rule:
    r = s.add("memory", "Отвечаем развёрнуто с примерами", entry_type="constraint")
    assert r.get("related_active"), "reversal must surface the old rule"
    s.deprecate("memory", "одной строкой", reason="superseded by: развёрнуто с примерами")
    s.close()
    s2 = _fresh(mem_dir)
    hits = s2.recall("Отвечаем")["results"]
    assert len(hits) == 1 and "развёрнуто" in hits[0]["content"]
    s2.close()


def test_scenario_5_decision_visible_in_next_session_prompt(mem_dir):
    s1 = _fresh(mem_dir)
    s1.add("memory", "Проект памяти — приоритет №1 до конца месяца", entry_type="decision", importance=0.9)
    s1.close()
    s2 = _fresh(mem_dir)
    snap = s2.format_for_system_prompt("memory") or ""
    assert "[decision]" in snap and "приоритет" in snap
    s2.close()


# ------------------------------------------------------- scenarios 6-10: recall quality

def test_scenario_6_partial_word_recall_russian(mem_dir):
    s = _fresh(mem_dir)
    s.add("memory", "Пользователь изучает архитектуру памяти агентов")
    assert s.recall("архитектур")["count"] == 1, "morphology-safe substring recall"
    s.close()


def test_scenario_7_many_entries_limit_respected(mem_dir):
    s = _fresh(mem_dir)
    for i in range(25):
        s.add("memory", f"Запись номер {i} про инфраструктуру")
    r = s.recall("инфраструктуру", limit=5)
    assert r["count"] == 5
    s.close()


def test_scenario_8_type_filter_isolates_decisions(mem_dir):
    s = _fresh(mem_dir)
    s.add("memory", "Сервер на Debian", entry_type="fact")
    s.add("memory", "Не используем Docker в проде", entry_type="decision")
    decision_only = s.recall("Docker", types=["decision"])
    assert decision_only["count"] == 1 and "Docker" in decision_only["results"][0]["content"]
    # Facts are filtered out when only decisions are requested:
    assert s.recall("Debian", types=["decision"])["count"] == 0
    # Without a filter both are reachable:
    assert s.recall("Debian")["count"] == 1
    s.close()


def test_scenario_9_user_and_memory_stores_isolated(mem_dir):
    s = _fresh(mem_dir)
    s.add("user", "Дмитрий любит короткие ответы")
    s.add("memory", "Дмитрий работает над агентской оболочкой")
    u = s.recall("Дмитрий", target="user")
    m = s.recall("Дмитрий", target="memory")
    assert u["count"] == 1 and "короткие" in u["results"][0]["content"]
    assert m["count"] == 1 and "оболочкой" in m["results"][0]["content"]
    s.close()


def test_scenario_10_access_count_grows_with_use(mem_dir):
    s = _fresh(mem_dir)
    s.add("memory", "Часто нужная заметка о сборке проекта")
    s.recall("сборке"); s.recall("сборке"); s.recall("сборке")
    row = s._query("SELECT access_count FROM memories")[0]
    assert row["access_count"] == 3, "usage stats feed Phase-2 scoring"
    s.close()


# ---------------------------------------------------- scenarios 11-15: snapshot budget

def test_scenario_11_small_budget_keeps_top_decisions(mem_dir):
    s1 = _fresh(mem_dir, memory_char_limit=250)
    s1.add("memory", "Ключевое решение проекта " + "к" * 100, entry_type="decision", importance=0.95)
    s1.add("memory", "Мелкий факт " + "м" * 100, entry_type="fact", importance=0.1)
    s1.close()
    s2 = _fresh(mem_dir, memory_char_limit=250)
    snap = s2.format_for_system_prompt("memory") or ""
    assert "Ключевое" in snap and "Мелкий" not in snap
    s2.close()


def test_scenario_12_everything_fits_small_memory(mem_dir):
    # Snapshot is frozen at session start: write in session 1, verify the
    # NEXT session's prompt carries it (nothing trimmed — it fits).
    s1 = _fresh(mem_dir)
    s1.add("memory", "Единственная запись")
    s1.close()
    s2 = _fresh(mem_dir)
    assert "Единственная" in (s2.format_for_system_prompt("memory") or "")
    s2.close()


def test_scenario_13_user_profile_snapshot_independent(mem_dir):
    s1 = _fresh(mem_dir)
    s1.add("user", "Таймзона UTC+3", importance=0.9)
    s1.close()
    s2 = _fresh(mem_dir)
    assert "UTC+3" in (s2.format_for_system_prompt("user") or "")
    s2.close()


def test_scenario_14_snapshot_frozen_within_session(mem_dir):
    s = _fresh(mem_dir)
    before = s.format_for_system_prompt("memory")
    s.add("memory", "Появилось середине сессии")
    assert s.format_for_system_prompt("memory") == before, "prefix-cache invariant"
    s.close()


def test_scenario_15_legacy_import_reaches_prompt(mem_dir):
    (mem_dir / "MEMORY.md").write_text("Старый факт из плоского файла", encoding="utf-8")
    s = _fresh(mem_dir)
    assert "Старый факт" in (s.format_for_system_prompt("memory") or "")
    s.close()


# --------------------------------------------------- scenarios 16-20: lifecycle & audit

def test_scenario_16_deprecated_kept_for_audit(mem_dir):
    s = _fresh(mem_dir)
    s.add("memory", "Пробовали SQLite без WAL", entry_type="decision")
    s.deprecate("memory", "без WAL", reason="конкуренция тредов")
    s.close()
    s2 = _fresh(mem_dir)
    assert s2.recall("WAL", status="deprecated")["count"] == 1, "evidence never silently destroyed"
    s2.close()


def test_scenario_17_remove_is_real_delete(mem_dir):
    s = _fresh(mem_dir)
    s.add("memory", "Мусорная тестовая запись")
    s.remove("memory", "Мусорная")
    s.close()
    s2 = _fresh(mem_dir)
    assert s2.recall("Мусорная", status="deprecated")["count"] == 0
    assert s2.recall("Мусорная")["count"] == 0
    s2.close()


def test_scenario_18_memory_md_projection_matches_store(mem_dir):
    s = _fresh(mem_dir)
    s.add("memory", "Видимая запись А")
    s.add("memory", "Скрытая запись Б")
    s.deprecate("memory", "запись Б", reason="x")
    text = (mem_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert "запись А" in text and "запись Б" not in text
    s.close()


def test_scenario_19_two_sessions_writes_merge(mem_dir):
    s1 = _fresh(mem_dir)
    s1.add("memory", "Запись первой сессии", entry_type="fact")
    s1.close()
    s2 = _fresh(mem_dir)
    s2.add("memory", "Запись второй сессии", entry_type="decision")
    s2.close()
    s3 = _fresh(mem_dir)
    assert s3.recall("сессии")["count"] == 2
    s3.close()


def test_scenario_20_concurrent_bg_review_and_agent_writes(mem_dir):
    import threading
    s = _fresh(mem_dir)
    results: list = []

    def agent_writes():
        for i in range(8):
            results.append(s.add("memory", f"Agент написал {i}", entry_type="fact"))

    def review_writes():
        for i in range(8):
            results.append(s.add("memory", f"Ревью сохранил {i}", entry_type="decision"))

    t1 = threading.Thread(target=agent_writes)
    t2 = threading.Thread(target=review_writes)
    t1.start(); t2.start(); t1.join(); t2.join()
    assert all(r["success"] for r in results)
    assert s.recall("написал")["count"] + s.recall("сохранил")["count"] == 16
    s.close()
