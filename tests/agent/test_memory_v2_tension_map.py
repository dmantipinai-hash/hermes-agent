"""Tests for the tension map / choice memory — supersede protocol (v4).

Rewritten for the 2026-08-30 graph fix: the one-call atomic
``memory(action=supersede)`` replaces the fragile add→deprecate-marker
chain; conflicting ACTIVE standing entries surface as ``possible_tensions``
WITH evidence (shared terms) and are no longer injected into the per-turn
pack; the matcher no longer proposes ready mutations for arbitrary
candidates.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

import tools.memory_tool as mt
from agent.memory_orchestrator import build_memory_orchestrator
from agent.memory_store_v2 import MemoryStoreV2


@pytest.fixture()
def mem_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "get_memory_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture()
def store(mem_dir):
    s = MemoryStoreV2()
    s.load_from_disk()
    yield s
    s.close()


def _fresh(mem_dir: Path) -> MemoryStoreV2:
    s = MemoryStoreV2()
    s.load_from_disk()
    return s


Y = "не используем Docker — тяжело для сервера, всего 2 ГБ RAM"
Z = "используем Docker в проде, подняли RAM до 8 ГБ"


class TestSupersedeProtocol:
    """The doc's live scenario, one call: refusal → supersede → trajectory."""

    def test_full_choice_trajectory_via_supersede(self, store):
        r1 = store.add("memory", Y, entry_type="decision", importance=0.8)
        assert r1["success"] and r1["id"]
        assert "related_active" not in r1  # first entry on the topic

        r2 = store.supersede(
            "memory", old_id=r1["id"], content=Z,
            entry_type="decision", importance=0.8,
            reason="сервер усилен, ограничение снято",
        )
        assert r2["success"] and r2["link_created"] is True
        assert r2["old_id"] == r1["id"] and r2["new_id"]

        # Later recall: the CURRENT decision plus its superseded predecessor
        # with dates — the trajectory, not just the final state.
        rec = store.recall("Docker сервер RAM")
        current = next(x for x in rec["results"] if Z[:40] in x["content"])
        assert current["supersedes"], "linked predecessor must ride along"
        pred = current["supersedes"][0]
        assert pred["content"][:40] in Y
        assert pred["status"] == "deprecated"
        assert pred["created_at"]

    def test_recovery_path_add_then_deprecate_by_ids(self, store):
        """The model that already added the new entry links it afterwards."""
        r_old = store.add("memory", Y, entry_type="decision", importance=0.8)
        r_new = store.add("memory", Z, entry_type="decision", importance=0.8)
        rel = r_new.get("related_active") or []
        assert rel and any("Docker" in c["content"] for c in rel)
        assert "suggested_deprecate" not in r_new

        r = store.deprecate(
            "memory", old_id=r_old["id"],
            superseded_by_id=r_new["id"], reason="смена решения",
        )
        assert r["success"] and r["link_created"] is True
        links = store._query("SELECT source_id, target_id FROM memory_links")
        assert links[0]["source_id"] == r_new["id"]
        assert links[0]["target_id"] == r_old["id"]

    def test_memory_tool_wires_supersede_action(self, store):
        from tools.memory_tool import memory_tool

        r1 = store.add("memory", Y, entry_type="decision")
        raw = memory_tool(
            action="supersede", target="memory",
            old_id=r1["id"], content=Z,
            entry_type="decision", reason="через реальный tool",
            store=store,
        )
        result = __import__("json").loads(raw)
        assert result["success"] is True
        assert result["link_created"] is True
        assert store._query("SELECT COUNT(*) AS c FROM memory_links")[0]["c"] == 1


class TestRecallPossibleTensions:
    def test_overlapping_actives_return_possible_tensions_with_evidence(self, store):
        store.add("memory", Y, entry_type="decision")
        store.add("memory", Z, entry_type="decision")
        rec = store.recall("Docker")
        tensions = rec.get("possible_tensions")
        assert tensions, "conflicting active decisions must surface as a map"
        pair = tensions[0]
        contents = " \n ".join(e["content"] for e in pair["entries"])
        assert "Docker" in contents
        assert all(e["created_at"] for e in pair["entries"])
        assert pair.get("shared_terms"), "evidence must ride along"

    def test_disjoint_standing_entries_have_no_tension(self, store):
        store.add("memory", Y, entry_type="decision")
        store.add("memory", "еженедельный отчёт шлём по пятницам", entry_type="decision")
        rec = store.recall("Docker")
        assert "possible_tensions" not in rec

    def test_facts_never_pair(self, store):
        store.add("memory", Y, entry_type="fact")
        store.add("memory", Z, entry_type="fact")
        rec = store.recall("Docker")
        assert "possible_tensions" not in rec

    def test_tension_free_recall_shape_unchanged(self, store):
        store.add("memory", "обычный факт про сеть", entry_type="fact")
        rec = store.recall("сеть")
        assert rec["success"] and rec["count"] == 1
        assert set(rec.keys()) == {"success", "query", "results", "count"}


class TestSnapshotStandingRule:
    def test_rule_teaches_supersede_when_standing_entries_exist(self, store, mem_dir):
        store.add("memory", "факт без правил", entry_type="fact")
        store.add("memory", Y, entry_type="decision")
        store.close()
        s2 = _fresh(mem_dir)
        block = s2.format_for_system_prompt("memory")
        assert block and "action=supersede" in block
        # The rule is a block suffix, NOT an entry.
        assert not any("Memory rule:" in e for e in s2.memory_entries)
        assert not any("Memory rule:" in c for c in s2.snapshot_contents())
        rec = s2.recall("memory rule")
        assert not any("Memory rule:" in (r["content"] or "") for r in rec["results"])
        s2.close()

    def test_no_rule_without_standing_entries(self, store, mem_dir):
        store.add("memory", "только факты и предпочтения", entry_type="fact")
        store.close()
        s2 = _fresh(mem_dir)
        block = s2.format_for_system_prompt("memory")
        assert block and "supersede" not in block
        s2.close()


class TestPackHasNoTensionInjection:
    """Invariant 6 of the graph TZ: an unproven matcher must not be injected
    into the per-turn prompt. Tensions live in explicit reads and reports."""

    def test_pack_carries_no_choice_tension(self, mem_dir):
        s = MemoryStoreV2(memory_char_limit=200)
        s.load_from_disk()
        # Lane-2 filler (Gate 1): standing rows keep a reserved lane, so
        # evicting the two decisions out of the snapshot requires crowding
        # their own lane — fact filler no longer displaces them.
        for i in range(6):
            s.add("memory", f"Стоящее решение-наполнитель номер {i} занимает дорожку",
                  entry_type="decision", importance=0.95)
        for i in range(6):
            s.add("memory", f"Наполнитель номер {i} занимает бюджет снапшота",
                  entry_type="fact", importance=0.95)
        s.add("memory", Y, entry_type="decision", importance=0.5)
        s.add("memory", Z, entry_type="decision", importance=0.5)
        s.close()
        s2 = MemoryStoreV2(memory_char_limit=200)
        s2.load_from_disk()
        orch = build_memory_orchestrator(s2, {}, memory_enabled=True, user_profile_enabled=False)
        pack = orch.build_pack("что там с Docker на сервере?")
        assert pack  # entries still arrive
        assert "Choice tension" not in pack
        s2.close()


class TestChoiceReport:
    def test_choice_report_changes_and_tensions(self, store):
        r1 = store.add("memory", Y, entry_type="decision", importance=0.8)
        rec = store.supersede(
            "memory", old_id=r1["id"], content=Z,
            entry_type="decision", importance=0.8, reason="усилили сервер",
        )
        assert rec["success"]

        rep = store.choice_report(days=7)
        assert len(rep["changes"]) == 1
        ch = rep["changes"][0]
        assert "8 ГБ" in ch["new_content"] and "2 ГБ" in ch["old_content"]
        assert ch["linked_at"] and ch["old_created"]
        # the linked pair is no longer an ACTIVE tension
        assert rep["possible_tensions"] == []
        assert rep["possible_tensions_total"] == 0

    def test_choice_report_possible_tensions_total_exceeds_display(self, store):
        # Enough standing decisions that pairwise candidates outrank the cap.
        topics = ["docker", "nginx", "postgres", "redis", "backup"]
        for t in topics:
            store.add("memory", f"решение: держим {t} на сервере projects",
                      entry_type="decision")
            store.add("memory", f"решение: переносим {t} на отдельный сервер",
                      entry_type="decision")
        rep = store.choice_report(days=7)
        assert rep["changes"] == []
        assert rep["possible_tensions_total"] >= 1
        assert "possible_tensions" in rep

    def test_hermes_memory_report_prints_choice_section(self, tmp_path, monkeypatch, capsys):
        hermes_home = tmp_path / ".hermes"
        (hermes_home / "memories").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        store = MemoryStoreV2()
        store.load_from_disk()
        r1 = store.add("memory", Y, entry_type="decision", importance=0.8)
        store.supersede(
            "memory", old_id=r1["id"], content=Z,
            entry_type="decision", importance=0.8, reason="усилили сервер",
        )
        store.close()

        from hermes_cli.main import cmd_memory

        cmd_memory(Namespace(memory_command="report", days=7, prune=False))
        out = capsys.readouterr().out
        assert "Choice memory" in out
        assert "decisions changed (last 7 day(s)): 1" in out
        assert "now:" in out and "was:" in out
        assert "Graph" in out  # integrity section
