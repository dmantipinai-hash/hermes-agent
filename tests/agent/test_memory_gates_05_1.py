"""Gate 0.5 + Gate 1 contracts — the memory repair (Н-1A scope, memory only).

Gate 0.5 (compatibility bridge): the schema marker is MONOTONIC — an older
wheel opening a future-schema database never downgrades the stored version.

Gate 1 (three separated axes):
- intrinsic importance — eviction/type defaults never silently rewrite it;
- snapshot residency — lane-structured standing brief (pinned + constraints,
  then standing decisions/preferences, then context; awareness telemetry
  excluded from the static prompt);
- query relevance — one NormalizedQuery for generation AND ranking, with a
  relevance-dominant lexicographic comparator: type/importance can only
  break ties inside a match class, never cross it.
"""
from __future__ import annotations

import sqlite3

import pytest

import tools.memory_tool as mt
from agent.memory_store_v2 import MemoryStoreV2, match_normalized
from agent.memory_orchestrator import rank_candidates


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


class TestGate05MonotonicSchemaMarker:
    def test_future_schema_marker_is_never_downgraded(self, mem_dir):
        s = MemoryStoreV2()
        s.load_from_disk()
        s.add("memory", "запись до подмены версии", entry_type="fact")
        s.close()
        conn = sqlite3.connect(mem_dir / "memory.db")
        conn.execute(
            "UPDATE meta SET value='99' WHERE key='schema_version'"
        )
        conn.commit()
        conn.close()

        older = MemoryStoreV2()
        older.load_from_disk()
        try:
            # A routine write through the older wheel must not "fix" the
            # marker — that downgrade is the rollback blocker (Gate 0.5).
            resp = older.add("memory", "запись при будущей схеме", entry_type="fact")
            assert resp["success"]
            conn = sqlite3.connect(mem_dir / "memory.db")
            marker = conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()[0]
            conn.close()
            assert marker == "99"
        finally:
            older.close()

    def test_fresh_store_writes_current_marker(self, mem_dir):
        s = MemoryStoreV2()
        s.load_from_disk()
        try:
            from agent.memory_store_v2 import SCHEMA_VERSION
            row = s._query("SELECT value FROM meta WHERE key='schema_version'")
            assert int(row[0]["value"]) == SCHEMA_VERSION
        finally:
            s.close()


class TestIntrinsicImportanceAxis:
    def test_type_default_applies_only_when_omitted(self, mem_dir):
        s = MemoryStoreV2()
        s.load_from_disk()
        try:
            r_def = s.add("memory", "Дефолтное решение о тестах", entry_type="decision")
            assert r_def["importance_applied"]["stored"] == pytest.approx(0.85)
            assert "type-default" in r_def["importance_applied"]["rule"]
            r_exp = s.add(
                "memory", "Явное решение с явной важностью", entry_type="decision",
                importance=0.3,
            )
            # An explicit value is stored as-is — no silent clamp, no report.
            assert "importance_applied" not in r_exp
            row = [
                r for r in s._query("SELECT content, importance FROM memories")
                if r["content"].startswith("Явное решение")
            ][0]
            assert row["importance"] == pytest.approx(0.3)
        finally:
            s.close()

    def test_standing_defaults_rank_above_context_defaults(self):
        assert (
            MemoryStoreV2._DEFAULT_IMPORTANCE_BY_TYPE["constraint"]
            > MemoryStoreV2._DEFAULT_IMPORTANCE_BY_TYPE["decision"]
            > MemoryStoreV2._DEFAULT_IMPORTANCE_BY_TYPE["fact"]
        )


class TestSnapshotResidencyLanes:
    def _snapshot_contents(self, s):
        return s._system_prompt_snapshot["memory"]

    def test_decision_visible_at_historical_floor_importance(self, mem_dir):
        """The live defect: 13/16 real decisions sat at 0.05 (old demotion
        floor) and never reached the prompt. Lanes admit by TYPE — a
        standing decision is visible whatever its importance number says."""
        s = MemoryStoreV2()
        s.load_from_disk()
        try:
            s.add("memory", "Решение: никогда не брать хостинг Aeza", entry_type="decision")
            s._execute_write(lambda conn: conn.execute(
                "UPDATE memories SET importance=0.05 WHERE type='decision'"
            ))
            s._refresh_live_state()
            s._rebuild_snapshot()
            assert "Aeza" in self._snapshot_contents(s)
        finally:
            s.close()

    def test_awareness_telemetry_never_in_static_snapshot(self, mem_dir):
        s = MemoryStoreV2()
        s.load_from_disk()
        try:
            s.add("memory", "Хвостовой паттерн осознанности", entry_type="pattern",
                  importance=0.55, written_by="awareness:test-session")
            s.add("memory", "Обычный факт для снапшота", entry_type="fact")
            s._rebuild_snapshot()
            snap = self._snapshot_contents(s)
            assert "Хвостовой паттерн" not in snap
            assert "Обычный факт" in snap
            # Excluded from the PROMPT ≠ excluded from memory: still recallable.
            assert any(
                "Хвостовой паттерн" in (c.get("content") or "")
                for c in s.recall_candidates("хвостовой паттерн")
            )
        finally:
            s.close()

    def test_pinned_and_constraints_precede_standing_rows(self, mem_dir):
        s = MemoryStoreV2(memory_char_limit=4000)
        s.load_from_disk()
        try:
            s.add("memory", "Контент-факт номер один", entry_type="fact")
            s.add("memory", "Стоящее решение номер два", entry_type="decision")
            s.add("memory", "Жёсткое ограничение номер три", entry_type="constraint")
            s._rebuild_snapshot()
            snap = self._snapshot_contents(s)
            pos_constraint = snap.index("ограничение")
            pos_decision = snap.index("Стоящее решение")
            pos_fact = snap.index("Контент-факт")
            assert pos_constraint < pos_decision < pos_fact
        finally:
            s.close()

    def test_lane_overflow_reported_not_hidden(self, mem_dir):
        s = MemoryStoreV2(memory_char_limit=120)
        s.load_from_disk()
        try:
            s.add("memory", "Пин один " + "а" * 60, entry_type="fact")
            s.add("memory", "Пин два " + "б" * 60, entry_type="fact")
            s._execute_write(
                lambda conn: conn.execute("UPDATE memories SET status='pinned'")
            )
            s._rebuild_snapshot()
            overflowed = s._snapshot_lane_overflow.get("memory")
            assert overflowed, "overflow must name the ids"
        finally:
            s.close()

    def test_no_row_appears_twice_in_snapshot(self, mem_dir):
        s = MemoryStoreV2(memory_char_limit=3000)
        s.load_from_disk()
        try:
            for i in range(8):
                s.add("memory", f"Запись номер {i} для проверки дублей", entry_type="fact")
            s.add("memory", "Решение поверх фактов", entry_type="decision")
            s._rebuild_snapshot()
            walk = s._budget_walk("memory")
            assert len(walk.included_contents) == len(set(walk.included_contents))
            assert len(walk.texts) == len(walk.included_contents)
        finally:
            s.close()

    def test_entries_are_bounded_snippets(self, mem_dir):
        s = MemoryStoreV2(max_entry_chars=100)
        s.load_from_disk()
        try:
            s.add("memory", "Очень длинный факт " + "х" * 500, entry_type="fact")
            walk = s._budget_walk("memory")
            assert walk.texts, "entry must still be included (bounded)"
            for text in walk.texts:
                assert len(text) <= 100, text[:120]
                assert not text.endswith("ххх"), "truncation marker required"
        finally:
            s.close()


class TestRelevanceDominantRanking:
    def _nq(self, s, q):
        return s.normalized_query(q)

    def test_zero_relevance_decision_cannot_displace_matching_fact(self, mem_dir):
        """The live displacement defect: an irrelevant decision outranked a
        topically-exact fact on type+importance weight alone."""
        s = MemoryStoreV2()
        s.load_from_disk()
        try:
            s.add("memory", "Хостеры VPS — антирекомендации: НЕ брать Aeza",
                  entry_type="decision", importance=0.9)
            s.add("memory", "Бэкап сервера: рестар weekly через launchd",
                  entry_type="fact", importance=0.4)
            nq = self._nq(s, "настройка бэкап сервера")
            cands = s.recall_candidates("настройка бэкап сервера", nq=nq)
            ranked = rank_candidates(cands, nq)
            assert ranked, "matching fact must be admitted"
            assert "Бэкап сервера" in ranked[0]["content"]
            assert not any(
                "антирекомендации" in c["content"] for c in ranked
            ), "zero-relevance decision must not enter the pack at all"
        finally:
            s.close()

    def test_match_class_phrase_beats_partial(self, mem_dir):
        s = MemoryStoreV2()
        s.load_from_disk()
        try:
            s.add("memory", "DNS через VPN: маршрутизация роутера", entry_type="fact")
            s.add("memory", "VPN-провайдеры выбирали по цене", entry_type="fact")
            nq = self._nq(s, "DNS VPN маршрутизация")
            ranked = rank_candidates(
                s.recall_candidates("DNS VPN маршрутизация", nq=nq), nq
            )
            assert ranked[0]["content"].startswith("DNS через VPN")
        finally:
            s.close()

    def test_type_tiebreak_only_inside_match_class(self):
        # Hand-built candidates with identical relevance (no bm25 field —
        # the neutral path) prove the TUPLE ORDER: same match_class/ratio →
        # type_tiebreak decides; importance only after that.
        from agent.memory_store_v2 import NormalizedQuery
        nq = NormalizedQuery(
            raw="канбан доска",
            concepts=("канб", "доск"),
            exact_terms=(),
            phrases=(),
        )
        fact = {"content": "канбан доска", "type": "fact", "importance": 0.9, "id": "b"}
        decision = {"content": "канбан доска", "type": "decision",
                    "importance": 0.2, "id": "a"}
        ranked = rank_candidates([fact, decision], nq)
        assert ranked[0]["type"] == "decision", (
            "equal relevance → the standing decision wins the tie even at "
            "lower importance"
        )
        # Importance breaks ties only WITHIN one type:
        d2 = {"content": "канбан доска", "type": "decision",
              "importance": 0.95, "id": "c"}
        assert rank_candidates([decision, d2], nq)[0]["id"] == "c"

    def test_normalized_query_alias_expansion_credits_concept(self, mem_dir):
        s = MemoryStoreV2()
        s.load_from_disk()
        try:
            s.set_alias_cache({"бэкапы": ["бэкап"], "прокси": ["vpn"]})
            s.add("memory", "бэкап сервера настроен", entry_type="fact")
            s.add("memory", "Роутер держит vpn на борту", entry_type="fact")
            nq = self._nq(s, "бэкапы сервера")
            # The alias-expanded stem found the entry in generation...
            cands = s.recall_candidates("бэкапы сервера", nq=nq)
            assert cands, "alias generation must find the entry"
            # ...and the matcher credits the concept, so ranking admits it.
            cls, ratio = match_normalized(nq, "бэкап сервера настроен")
            assert cls > 0 and ratio > 0
            # Short (3-char) aliases ride the exact-term path — the P2 §8.4
            # contract the legacy expansion had and NormalizedQuery keeps.
            nq2 = self._nq(s, "настроить прокси быстро")
            assert "vpn" in nq2.exact_terms
            cands2 = s.recall_candidates("настроить прокси быстро", nq=nq2)
            assert any("Роутер" in c["content"] for c in cands2)
        finally:
            s.close()

    def test_one_normalizer_for_generation_and_scoring(self, mem_dir):
        """The Р1 contract: a candidate FOUND by stems/aliases must never be
        scored relevance=0 by a second, different tokenizer."""
        s = MemoryStoreV2()
        s.load_from_disk()
        try:
            s.add("memory", "Дневниковые записи хранятся в MyVault", entry_type="fact")
            q = "дневниковой записью"
            nq = self._nq(s, q)
            cands = s.recall_candidates(q, nq=nq)
            assert cands, "stem generation must find the inflected entry"
            ranked = rank_candidates(cands, nq)
            assert ranked, "the found candidate must also score positively"
        finally:
            s.close()

    def test_empty_concept_query_admits_nothing(self):
        from agent.memory_store_v2 import NormalizedQuery
        nq = NormalizedQuery(raw="", concepts=(), exact_terms=(), phrases=())
        cls, ratio = match_normalized(nq, "любой контент")
        assert (cls, ratio) == (0, 0.0)
        assert rank_candidates(
            [{"content": "любой контент", "type": "fact", "importance": 0.9}], nq,
        ) == []
