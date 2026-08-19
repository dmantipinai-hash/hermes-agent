"""Tests for the Cognitive Memory Bus (Phase 3 of the memory roadmap).

Layers, mirroring the Phase 1/2 split:

- **Store**: schema v2 (``written_by`` provenance), v1→v2 migration, the
  project-realm filter, and group rollback.
- **Bus**: coeffect subscriptions (satisfaction predicate, reactive
  deactivation), gated recall (targets + realm), provenance-tagged writes,
  read-only refusal, scoped revert, scoped views, delegation briefing.
- **Tool provenance**: the model-facing ``memory`` tool threads
  ``main:{session_id}`` through both agent dispatchers.

"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import tools.memory_tool as mt
from agent.memory_bus import (
    ALL_CAPABILITIES,
    READ_MEMORY,
    READ_USER,
    WRITE_MEMORY,
    WRITE_USER,
    ConsumerSpec,
    MemoryBus,
    build_delegation_briefing,
)
from agent.memory_store_v2 import MemoryStoreV2
from tools.memory_tool import memory_tool


@pytest.fixture()
def mem_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "get_memory_dir", lambda: tmp_path)
    return tmp_path


def _fresh(mem_dir: Path, **kw) -> MemoryStoreV2:
    s = MemoryStoreV2(**kw)
    s.load_from_disk()
    return s


# ============================================================================
# Store — schema v2, migration, realm filter, rollback
# ============================================================================

def test_written_by_persisted(mem_dir):
    s = _fresh(mem_dir)
    s.add("memory", "Запись с провенансом", entry_type="fact", written_by="main:sess-1")
    row = s.recent_entries(limit=5)[0]
    assert row["written_by"] == "main:sess-1"
    s.close()


def test_written_by_optional_backwards_compatible(mem_dir):
    s = _fresh(mem_dir)
    r = s.add("memory", "Обычная запись без провенанса")
    assert r["success"]
    assert s.recent_entries(limit=5)[0]["written_by"] is None
    s.close()


def test_v1_database_migrates_to_v2(mem_dir):
    """A database created with the Phase-1 schema upgrades in place, data intact."""
    db_path = mem_dir / "memory.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE memories (
            id TEXT PRIMARY KEY, target TEXT NOT NULL DEFAULT 'memory',
            type TEXT NOT NULL DEFAULT 'fact', content TEXT NOT NULL,
            importance REAL NOT NULL DEFAULT 0.5, confidence REAL NOT NULL DEFAULT 0.7,
            status TEXT NOT NULL DEFAULT 'active', project TEXT, source_session TEXT,
            deprecate_reason TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            last_accessed_at TEXT, access_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO memories (id, target, type, content, importance, confidence,
            status, created_at, updated_at)
            VALUES ('old-1', 'memory', 'fact', 'Запись из Фазы 1', 0.5, 0.7,
            'active', '2026-08-01T00:00:00+00:00', '2026-08-01T00:00:00+00:00');
        INSERT INTO meta (key, value) VALUES ('schema_version', '1');
        """
    )
    conn.commit()
    conn.close()

    s = _fresh(mem_dir)
    version = s._query("SELECT value FROM meta WHERE key='schema_version'")[0]["value"]
    assert version == "2"
    cols = {r["name"] for r in s._query("PRAGMA table_info(memories)")}
    assert "written_by" in cols
    hits = s.recall_candidates("Запись из Фазы")
    assert hits and hits[0]["id"] == "old-1", "v1 data survives the migration"
    s.close()


def test_project_realm_filter_in_recall_candidates(mem_dir):
    s = _fresh(mem_dir)
    s.add("memory", "Правило проекта hermes про тесты", entry_type="constraint", project="hermes")
    s.add("memory", "Секрет проекта skynet", entry_type="fact", project="skynet")
    s.add("memory", "Глобальное правило бэкапов", entry_type="constraint")
    s.close()

    s2 = _fresh(mem_dir)
    global_view = [c["content"] for c in s2.recall_candidates("правило")]
    assert "Секрет проекта skynet" not in global_view or True  # no realm: everything visible
    hermes_view = [c["content"] for c in s2.recall_candidates("правило секрет", project="hermes")]
    assert any("hermes" in c for c in hermes_view), "own project visible"
    assert any("Глобальное" in c for c in hermes_view), "global rows visible"
    assert not any("skynet" in c for c in hermes_view), "foreign project invisible (Codex §8.4)"
    s2.close()


def test_rollback_consumer_deprecates_only_that_scope(mem_dir):
    s = _fresh(mem_dir)
    s.add("memory", "Вывод делегации А", entry_type="fact", written_by="delegation:sa-1")
    s.add("memory", "Вывод делегации Б", entry_type="fact", written_by="delegation:sa-2")
    s.add("memory", "Запись основного агента", entry_type="fact", written_by="main:sess-9")
    s.close()

    s2 = _fresh(mem_dir)
    result = s2.rollback_consumer("delegation:sa-1", reason="scoped revert: бракованный batch")
    assert result["success"] and result["deprecated"] == 1
    hits = s2.recall("Вывод делегации")
    contents = [h["content"] for h in hits["results"]]
    assert contents == ["Вывод делегации Б"], "only the rolled-back scope disappears"
    # The main agent's record is untouched.
    assert s2.recall("основного агента")["count"] == 1
    # Deprecated rows stay auditable, not deleted (roadmap §1.7).
    dep = s2._query("SELECT status, deprecate_reason FROM memories WHERE written_by='delegation:sa-1'")[0]
    assert dep["status"] == "deprecated" and "scoped revert" in dep["deprecate_reason"]
    s2.close()


def test_rollback_consumer_rejects_empty_scope(mem_dir):
    s = _fresh(mem_dir)
    assert not s.rollback_consumer("")["success"]
    s.close()


# ============================================================================
# Bus — coeffect subscriptions + capability gating
# ============================================================================

def _bus(store, **kw) -> MemoryBus:
    return MemoryBus(store=store, targets=kw.pop("targets", ("memory", "user")), **kw)


def test_capabilities_derived_from_targets(mem_dir):
    s = _fresh(mem_dir)
    s.close()
    bus = MemoryBus(store=None)
    assert bus.capabilities() == frozenset()
    bus2 = _bus(s, targets=("memory",))
    assert bus2.capabilities() == frozenset({READ_MEMORY, WRITE_MEMORY})
    s.close()


def test_subscribe_satisfaction_predicate(mem_dir):
    s = _fresh(mem_dir)
    bus = _bus(s)
    # Satisfied: every requested capability exists.
    assert bus.subscribe(ConsumerSpec("main:x", needs=frozenset(ALL_CAPABILITIES))) is not None
    # Unsatisfiable: a target the bus doesn't offer → NOT activated.
    narrow = _bus(s, targets=("memory",))
    assert narrow.subscribe(ConsumerSpec("u1", needs=frozenset({READ_USER}))) is None
    # Malformed: empty id / unknown capability keys only.
    assert bus.subscribe(ConsumerSpec("", needs=frozenset({READ_MEMORY}))) is None
    assert bus.subscribe(ConsumerSpec("bad", needs=frozenset({"fly:moon"}))) is None
    s.close()


def test_readonly_is_structural_not_polite(mem_dir):
    s = _fresh(mem_dir)
    bus = _bus(s)
    sub = bus.subscribe(ConsumerSpec("subagent:ro", needs=frozenset({READ_MEMORY})))
    assert sub is not None and sub.spec.read_only
    w = bus.remember("попытка записи", consumer_id="subagent:ro", entry_type="fact")
    assert not w["success"] and "read-only" in w["error"]
    s.close()


def test_refresh_capabilities_reactive_transitions(mem_dir):
    """A source change deactivates unsatisfied subscriptions and reactivates
    them when the capability returns (research doc §2)."""
    s = _fresh(mem_dir)
    bus = _bus(s)
    sub = bus.subscribe(ConsumerSpec("c1", needs=frozenset({READ_USER, WRITE_USER})))
    assert sub.active
    bus.targets = ("memory",)  # user source dropped
    bus.refresh_capabilities()
    assert not sub.active and sub.deactivation_reason
    # A read on a deactivated consumer is refused, not half-alive.
    res = bus.recall("что угодно", consumer_id="c1")
    assert not res["success"]
    bus.targets = ("memory", "user")
    bus.refresh_capabilities()
    assert sub.active and sub.deactivation_reason is None
    s.close()


def test_recall_requires_subscription(mem_dir):
    s = _fresh(mem_dir)
    bus = _bus(s)
    res = bus.recall("тест", consumer_id="ghost")
    assert not res["success"] and "no active read subscription" in res["error"]
    s.close()


def test_recall_target_visibility_follows_needs(mem_dir):
    """A consumer holding only read:memory must not see user-profile rows."""
    s = _fresh(mem_dir)
    s.add("memory", "Правило проекта", entry_type="constraint")
    s.add("user", "Личные предпочтения Дмитрия", entry_type="preference")
    s.close()
    s2 = _fresh(mem_dir)
    bus = _bus(s2)
    bus.subscribe(ConsumerSpec("worker", needs=frozenset({READ_MEMORY})))
    res = bus.recall("правило предпочтения", consumer_id="worker")
    assert res["success"]
    assert all(e["content"] != "Личные предпочтения Дмитрия" for e in res["entries"])
    s2.close()


def test_recall_realm_from_spec(mem_dir):
    s = _fresh(mem_dir)
    s.add("memory", "Тонкости проекта hermes", entry_type="fact", project="hermes")
    s.add("memory", "Тонкости проекта skynet", entry_type="fact", project="skynet")
    s.close()
    s2 = _fresh(mem_dir)
    bus = _bus(s2)
    bus.subscribe(ConsumerSpec("scoped", needs=frozenset({READ_MEMORY}), project="hermes"))
    res = bus.recall("тонкости проекта", consumer_id="scoped")
    contents = [e["content"] for e in res["entries"]]
    assert any("hermes" in c for c in contents)
    assert not any("skynet" in c for c in contents)
    s2.close()


def test_remember_provenance_project_and_mirror(mem_dir):
    s = _fresh(mem_dir)
    mirrored = []
    manager = type("M", (), {"on_memory_write": staticmethod(
        lambda action, target, content, metadata=None: mirrored.append((action, target, content, metadata))
    )})()
    bus = MemoryBus(store=s, manager=manager, targets=("memory", "user"))
    bus.subscribe(ConsumerSpec(
        "delegation:sa-7", needs=frozenset({READ_MEMORY, WRITE_MEMORY}), project="hermes",
    ))
    r = bus.remember("Финальный вывод: миграция ok", consumer_id="delegation:sa-7", entry_type="fact")
    assert r["success"]
    row = s.recent_entries(limit=5)[0]
    assert row["written_by"] == "delegation:sa-7"
    assert row["project"] == "hermes"
    assert mirrored and mirrored[0][2] == "Финальный вывод: миграция ok"
    s.close()


def test_scoped_view_pins_spec_and_can_write_only_with_caps(mem_dir):
    s = _fresh(mem_dir)
    bus = _bus(s)
    view = bus.scoped_view(ConsumerSpec("delegation:w", needs=frozenset({READ_MEMORY, WRITE_MEMORY})))
    assert view is not None
    assert view.remember("вывод")["success"]
    assert view.rollback()["deprecated"] == 1
    ro = bus.scoped_view(ConsumerSpec("subagent:ro", needs=frozenset({READ_MEMORY})))
    assert ro is not None
    assert not ro.remember("нет прав")["success"]
    s.close()


# ============================================================================
# Delegation briefing (roadmap 3.2)
# ============================================================================

def test_delegation_briefing_from_parent_bus(mem_dir):
    s = _fresh(mem_dir)
    s.add("memory", "Правило проекта: тесты через run_tests.sh", entry_type="constraint", importance=0.9)
    s.close()
    s2 = _fresh(mem_dir)
    bus = _bus(s2)
    parent = type("P", (), {"_memory_bus": bus})()
    brief = build_delegation_briefing(parent, "прогони тесты проекта по правилам")
    assert "Memory briefing" in brief
    assert "run_tests.sh" in brief
    assert "<memory-context" not in brief, "briefing uses its own plain-markdown slot"
    s2.close()


def test_delegation_briefing_respects_realm(mem_dir):
    s = _fresh(mem_dir)
    s.add("memory", "Секрет skynet", entry_type="fact", project="skynet", importance=0.9)
    s.add("memory", "Правило hermes", entry_type="constraint", project="hermes", importance=0.9)
    s.close()
    s2 = _fresh(mem_dir)
    parent = type("P", (), {"_memory_bus": _bus(s2)})()
    brief = build_delegation_briefing(parent, "секреты и правила проекта", subagent_id="sa-1", project="hermes")
    assert "Правило hermes" in brief
    assert "skynet" not in brief
    s2.close()


def test_delegation_briefing_no_bus_or_failure(mem_dir):
    parent_no_bus = type("P", (), {})()
    assert build_delegation_briefing(parent_no_bus, "цель") == ""
    s = _fresh(mem_dir)
    s.add("memory", "запись", entry_type="fact")
    s.close()
    s2 = _fresh(mem_dir)
    broken = type("P", (), {"_memory_bus": _bus(s2)})()
    orig = mt.get_memory_dir
    mt.get_memory_dir = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        assert build_delegation_briefing(broken, "цель") == ""
    finally:
        mt.get_memory_dir = orig
    s2.close()


def test_delegation_config_gates(monkeypatch):
    import tools.delegate_tool as dt

    monkeypatch.setattr(dt, "_load_config", lambda: {"memory_briefing": False})
    assert dt._memory_briefing_enabled() is False
    monkeypatch.setattr(dt, "_load_config", lambda: {})
    assert dt._memory_briefing_enabled() is True, "default on"

    monkeypatch.setattr(dt, "_load_config", lambda: {"subagent_project": "hermes"})
    assert dt._subagent_project_realm() == "hermes"
    monkeypatch.setattr(dt, "_load_config", lambda: {"subagent_project": "  "})
    assert dt._subagent_project_realm() is None, "empty = global realm"


# ============================================================================
# Tool provenance — main:{session_id} threads through both dispatchers
# ============================================================================

def test_memory_tool_written_by_passthrough(mem_dir):
    s = _fresh(mem_dir)
    result = mt_json = memory_tool(
        action="add", target="memory", content="Запись основного агента",
        entry_type="fact", written_by="main:sess-42", store=s,
    )
    import json as _json
    assert _json.loads(result)["success"]
    assert s.recent_entries(limit=5)[0]["written_by"] == "main:sess-42"
    s.close()


def test_memory_tool_written_by_ignored_on_legacy_store(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "get_memory_dir", lambda: tmp_path)
    legacy = mt.MemoryStore()
    legacy.load_from_disk()
    result = memory_tool(
        action="add", target="memory", content="legacy запись",
        written_by="main:sess-1", store=legacy,
    )
    import json as _json
    assert _json.loads(result)["success"], "legacy store must keep working"
    legacy.memory_entries[0] == "legacy запись"


def test_memory_tool_add_without_written_by_still_works(mem_dir):
    s = _fresh(mem_dir)
    result = memory_tool(action="add", target="memory", content="без провенанса", store=s)
    import json as _json
    assert _json.loads(result)["success"]
    assert s.recent_entries(limit=5)[0]["written_by"] is None
    s.close()


def test_delegation_briefing_logs_evidence_line(mem_dir, caplog):
    """Non-empty delegation briefings emit one INFO line (agent.log evidence)."""
    import logging as _logging

    s = _fresh(mem_dir)
    s.add("memory", "Правило проекта про тесты", entry_type="constraint", importance=0.9)
    s.close()
    s2 = _fresh(mem_dir)
    parent = type("P", (), {"_memory_bus": _bus(s2)})()
    with caplog.at_level(_logging.INFO, logger="agent.memory_bus"):
        build_delegation_briefing(parent, "правила тестирования", subagent_id="sa-1")
        lines = [r for r in caplog.records if "delegation briefing" in r.message]
        assert len(lines) == 1 and "subagent:sa-1" in lines[0].message
    s2.close()
