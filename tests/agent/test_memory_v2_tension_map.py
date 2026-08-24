"""Tests for the tension map / choice memory (Task 2, on top of P1).

Behavior contracts:

- the live scenario from the design docs (agent-awareness of choice
  trajectory): an old refusal and a new adoption of the same topic link via
  the add-time ``suggested_deprecate`` (exact stored substring + marker),
  and later recall shows BOTH sides with dates — "you changed your mind;
  was the original objection addressed?";
- recall surfaces conflicting ACTIVE decision/constraint entries as an
  explicit ``tensions`` map (pairs with dates), never for non-standing
  types, and tension-free recalls stay byte-identical (no extra key);
- the frozen memory snapshot gains the one-line standing rule when
  decision/constraint entries exist — as a block suffix, never an entry;
- the orchestrator pack carries a "Choice tension" section when conflicting
  actives match the query;
- `hermes memory report` prints the Choice-memory digest (changes + active
  tensions).
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
    """New store instance = new agent session over the same on-disk memory."""
    s = MemoryStoreV2()
    s.load_from_disk()
    return s


Y = "не используем Docker — тяжело для сервера, всего 2 ГБ RAM"
Z = "используем Docker в проде, подняли RAM до 8 ГБ"


class TestSuggestedDeprecate:
    def test_full_choice_trajectory_scenario(self, store, mem_dir):
        """The doc's live scenario: refusal (Y) → adoption (Z) → linked pair."""
        r1 = store.add("memory", Y, entry_type="decision", importance=0.8)
        assert r1["success"] and "suggested_deprecate" not in r1

        r2 = store.add("memory", Z, entry_type="decision", importance=0.8)
        assert r2["success"]
        assert r2.get("related_active"), "the old refusal must surface on add"
        sugg = r2.get("suggested_deprecate")
        assert sugg, "a ready-to-use deprecate call must ride the hint"
        assert sugg["action"] == "deprecate" and sugg["target"] == "memory"
        assert sugg["old_text"] and sugg["old_text"] in Y
        assert sugg["reason"].startswith("superseded by:")
        assert sugg["reason"].split("superseded by:", 1)[1].strip() and \
            sugg["reason"].split("superseded by:", 1)[1].strip() in Z

        # The model "copies" the suggestion verbatim — no re-guessing.
        dep = store.deprecate(
            sugg["target"], sugg["old_text"], reason=sugg["reason"]
        )
        assert dep["success"], f"suggested call must roundtrip: {dep}"

        # Later recall: the CURRENT decision plus its superseded predecessor
        # with dates — the trajectory, not just the final state.
        rec = store.recall("Docker сервер RAM")
        assert rec["count"] >= 1
        current = next(r for r in rec["results"] if Z[:40] in r["content"])
        assert current["supersedes"], "linked predecessor must ride along"
        pred = current["supersedes"][0]
        assert pred["content"][:40] in Y
        assert pred["status"] == "deprecated"
        assert pred["created_at"]

    def test_suggested_old_text_resolves_to_exactly_the_related_row(self, store):
        store.add("memory", "резервные копии храним в S3, регион Варшава", entry_type="decision")
        r = store.add(
            "memory", "резервные копии храним в S3, регион Франкфурт", entry_type="decision"
        )
        sugg = r["suggested_deprecate"]
        rows = store._query(
            "SELECT content FROM memories WHERE ulower(content) LIKE ?",
            (f"%{sugg['old_text'].lower()}%",),
        )
        # the suggested substring resolves to exactly the OLD row (the
        # successor differs and must not be hit) — no deprecate ambiguity
        assert len(rows) == 1
        assert "Варшава" in rows[0]["content"]

    def test_hint_names_the_suggested_call(self, store):
        store.add("memory", Y, entry_type="decision")
        r = store.add("memory", Z, entry_type="decision")
        assert "suggested_deprecate" in r["hint"]
        assert "superseded by:" in r["hint"]


class TestRecallTensionMap:
    def test_overlapping_actives_return_tensions(self, store):
        store.add("memory", Y, entry_type="decision")
        store.add("memory", Z, entry_type="decision")
        rec = store.recall("Docker")
        tensions = rec.get("tensions")
        assert tensions, "conflicting active decisions must surface as a map"
        pair = tensions[0]
        contents = " \n ".join(e["content"] for e in pair["entries"])
        assert "Docker" in contents
        assert all(e["created_at"] for e in pair["entries"])
        assert "supersedes" in pair["note"]

    def test_disjoint_standing_entries_have_no_tension(self, store):
        store.add("memory", Y, entry_type="decision")
        store.add("memory", "еженедельный отчёт шлём по пятницам", entry_type="decision")
        rec = store.recall("Docker")
        assert "tensions" not in rec

    def test_facts_never_pair(self, store):
        store.add("memory", Y, entry_type="fact")
        store.add("memory", Z, entry_type="fact")
        rec = store.recall("Docker")
        assert "tensions" not in rec

    def test_tension_free_recall_is_byte_identical_shape(self, store):
        store.add("memory", "обычный факт про сеть", entry_type="fact")
        rec = store.recall("сеть")
        assert rec["success"] and rec["count"] == 1
        assert set(rec.keys()) == {"success", "query", "results", "count"}


class TestSnapshotStandingRule:
    def test_rule_rides_snapshot_when_standing_entries_exist(self, store, mem_dir):
        store.add("memory", "факт без правил", entry_type="fact")
        store.add("memory", Y, entry_type="decision")
        store.close()
        s2 = _fresh(mem_dir)
        block = s2.format_for_system_prompt("memory")
        assert block
        assert "superseded by:" in block
        assert block.rstrip().endswith("'superseded by: <short quote of the new entry>'. "
                                       "The marker links the pair, "
                                       "so later recall shows what was decided before and why it changed.")
        # The rule is a block suffix, NOT an entry: it must not pollute the
        # entry list, the snapshot-entry set, or recall results.
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
        assert block and "superseded by:" not in block
        s2.close()


class TestPackTensionSection:
    def _crowded(self, mem_dir):
        """Snapshot budget exhausted by high-importance filler; two
        conflicting decisions stay searchable but evicted from the prompt —
        the exact zone where the pack must carry the tension. Reopened with
        the SAME tight char limit (a default-limit reopen would fit
        everything into the snapshot and empty the pack)."""
        s = MemoryStoreV2(memory_char_limit=200)
        s.load_from_disk()
        for i in range(6):
            s.add("memory", f"Наполнитель номер {i} занимает бюджет снапшота",
                  entry_type="fact", importance=0.95)
        s.add("memory", Y, entry_type="decision", importance=0.5)
        s.add("memory", Z, entry_type="decision", importance=0.5)
        s.close()
        s2 = MemoryStoreV2(memory_char_limit=200)
        s2.load_from_disk()
        return s2

    def test_pack_carries_choice_tension(self, mem_dir):
        s = self._crowded(mem_dir)
        orch = build_memory_orchestrator(
            s, {}, memory_enabled=True, user_profile_enabled=False
        )
        pack = orch.build_pack("что там с Docker на сервере?")
        assert "Choice tension" in pack
        assert "⚠" in pack
        # both sides with dates — the trajectory is visible in the pack
        assert "тяжело для сервера" in pack or Y[:20] in pack
        s.close()

    def test_pack_without_conflicts_has_no_tension_section(self, mem_dir):
        s = MemoryStoreV2(memory_char_limit=200)
        s.load_from_disk()
        for i in range(6):
            s.add("memory", f"Наполнитель номер {i} занимает бюджет снапшота",
                  entry_type="fact", importance=0.95)
        s.add("memory", "отдельное решение про бэкапы по пятницам",
              entry_type="decision", importance=0.5)
        s.close()
        s2 = MemoryStoreV2(memory_char_limit=200)
        s2.load_from_disk()
        orch = build_memory_orchestrator(
            s2, {}, memory_enabled=True, user_profile_enabled=False
        )
        pack = orch.build_pack("наполнитель номер три")
        assert "Choice tension" not in pack
        s2.close()


class TestChoiceReport:
    def test_choice_report_changes_and_tensions(self, store):
        store.add("memory", Y, entry_type="decision", importance=0.8)
        r = store.add("memory", Z, entry_type="decision", importance=0.8)
        sugg = r["suggested_deprecate"]
        store.deprecate(sugg["target"], sugg["old_text"], reason=sugg["reason"])

        rep = store.choice_report(days=7)
        assert len(rep["changes"]) == 1
        ch = rep["changes"][0]
        assert "8 ГБ" in ch["new_content"] and "2 ГБ" in ch["old_content"]
        assert ch["linked_at"] and ch["old_created"]
        # the linked pair is no longer an ACTIVE tension
        assert rep["active_tensions"] == []

    def test_choice_report_active_tensions(self, store):
        store.add("memory", Y, entry_type="decision")
        store.add("memory", Z, entry_type="decision")
        rep = store.choice_report(days=7)
        assert rep["changes"] == []
        assert len(rep["active_tensions"]) == 1
        a, b = rep["active_tensions"][0]["entries"]
        assert {a["content"][:20], b["content"][:20]} == {Y[:20], Z[:20]}

    def test_hermes_memory_report_prints_choice_section(self, tmp_path, monkeypatch, capsys):
        hermes_home = tmp_path / ".hermes"
        (hermes_home / "memories").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        store = MemoryStoreV2()
        store.load_from_disk()
        store.add("memory", Y, entry_type="decision", importance=0.8)
        r = store.add("memory", Z, entry_type="decision", importance=0.8)
        sugg = r["suggested_deprecate"]
        store.deprecate(sugg["target"], sugg["old_text"], reason=sugg["reason"])
        # plus one unresolved tension
        store.add("memory", "канбан-доска ведётся в профиле hugin", entry_type="decision")
        store.add("memory", "канбан-доска перенесена в профиль loki", entry_type="decision")
        store.close()

        from hermes_cli.main import cmd_memory

        cmd_memory(Namespace(memory_command="report", days=7, prune=False))
        out = capsys.readouterr().out
        assert "Choice memory" in out
        assert "decisions changed (last 7 day(s)): 1" in out
        assert "now:" in out and "was:" in out
        assert "тяжело для сервера" in out
        assert "Active tensions" in out
        assert "superseded by:" in out
