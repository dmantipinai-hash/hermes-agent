"""Phase-4 retrieval & telemetry contracts (each traced to a live incident).

- Gap B′ (2026-08-19): 3-char domain terms (vpn/vps/dns) were invisible to
  the orchestrator search — `_TOKEN_MIN_STEM` dropped them before the query
  ran. Stored entries must be findable with NO alias dictionary present.
- Multi-word recall: natural-language queries ("репутация сервер для
  блокировок") share no contiguous phrase with stored entries — explicit
  recall must fall back to term-OR search before reporting empty.
- P6 telemetry: a successful write reports the PROMPT projection usage
  (what the next session's snapshot will contain), not the whole-store char
  count, and states that eviction is automatic when entries spill to cold
  storage (the 2026-08-19 incident: the model "fixed" an over-limit number
  by hard-deleting searchable entries).
- No-match errors carry ``current_entries`` — the v1 feedback contract the
  v2 mutations lost, leaving the model to re-guess ``old_text`` blind.
"""

from __future__ import annotations

import re

import pytest

import tools.memory_tool as mt
from agent.memory_store_v2 import MemoryStoreV2
from agent.memory_orchestrator import MemoryOrchestrator


@pytest.fixture()
def mem_dir(tmp_path, monkeypatch):
    """Point the memory directory at a temp dir (canonical fixture pattern)."""
    monkeypatch.setattr(mt, "get_memory_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture()
def store(mem_dir):
    s = MemoryStoreV2()
    s.load_from_disk()
    yield s
    s.close()


def _usage_chars(response: dict) -> int:
    used = re.search(r"— (\d[\d,]*)/", response["usage"])
    assert used, response["usage"]
    return int(used.group(1).replace(",", ""))


class TestShortDomainTerms:
    """Gap B′: short terms must be searchable, exactly, without aliases."""

    def test_short_term_candidates_find_stored_entry(self, store):
        store.add("memory", "Роутер CUDY TR3000: vpn через vless поднят")
        cands = store.recall_candidates("что там с vpn?")
        assert any("CUDY" in c["content"] for c in cands)

    def test_short_term_exact_not_prefix(self, store):
        # "vpn" matches the whole token only — never a prefix of "vpnhub".
        store.add("memory", "Клиент vpnhub установлен на сервере")
        assert store.recall_candidates("vpn") == []

    def test_function_words_do_not_open_search(self, store):
        store.add("memory", "Запись содержащая слово что внутри")
        assert store.recall_candidates("что или как") == []

    def test_two_char_tokens_still_ignored(self, store):
        store.add("memory", "Запись про ок и из")
        assert store.recall_candidates("ок из") == []

    def test_orchestrator_pack_finds_short_term(self, mem_dir):
        s = MemoryStoreV2()
        s.load_from_disk()
        s.add("memory", "Роутер CUDY TR3000: vpn через vless поднят")
        pack = MemoryOrchestrator(s).build_pack("что там с vpn?")
        assert "CUDY" in pack
        s.close()


class TestMultiWordRecallFallback:
    """Natural-language queries must reach entries via term-OR fallback."""

    def test_natural_language_query_finds_entry(self, store):
        store.add("memory", "Сервер Билайн душит XHTTP на WebSocket-соединениях")
        r = store.recall("репутация сервер для блокировок")
        assert r["count"] >= 1
        assert any("Билайн" in x["content"] for x in r["results"])

    def test_inflected_single_word_found_via_stem_fallback(self, store):
        store.add("memory", "Глобально зависимостей не ставим")
        assert store.recall("зависимости")["count"] == 1

    def test_exact_phrase_precision_unchanged(self, store):
        store.add("memory", "Сервер Билайн душит XHTTP")
        store.add("memory", "Другая запись про сервер Debian")
        r = store.recall("Билайн душит")
        assert r["count"] == 1
        assert "XHTTP" in r["results"][0]["content"]

    def test_results_not_duplicated(self, store):
        store.add("memory", "Роутер CUDY с vpn и сервер для туннеля")
        r = store.recall("роутер сервер туннель")
        ids = [x["id"] for x in r["results"]]
        assert len(ids) == len(set(ids))


class TestProjectionTelemetry:
    """P6: write telemetry reports the projection, not the whole store."""

    def _overfill(self, mem_dir):
        s = MemoryStoreV2(memory_char_limit=200)
        s.load_from_disk()
        s.add("memory", "Важное правило номер один " + "а" * 80, importance=0.9)
        s.add("memory", "Второй факт средней важности " + "б" * 80, importance=0.5)
        r = s.add("memory", "Третья запись хвостовая " + "в" * 80, importance=0.1)
        return s, r

    def test_over_budget_reports_projection_and_eviction(self, mem_dir):
        s, r = self._overfill(mem_dir)
        try:
            assert r["success"]
            assert r.get("evicted_to_cold", 0) >= 1
            assert "do NOT remove" in r["note"]
            assert _usage_chars(r) <= 200  # the prompt never exceeds its budget
        finally:
            s.close()

    def test_reported_usage_matches_next_session_snapshot(self, mem_dir):
        s, r = self._overfill(mem_dir)
        in_prompt = r["entry_count"] - r.get("evicted_to_cold", 0)
        s.close()
        s2 = MemoryStoreV2(memory_char_limit=200)
        s2.load_from_disk()
        try:
            assert len(s2._snapshot_entry_contents["memory"]) == in_prompt
        finally:
            s2.close()

    def test_no_eviction_keeps_plain_note(self, store):
        r = store.add("memory", "Простая запись без переполнения")
        assert "evicted_to_cold" not in r
        assert r["note"] == "Write saved. This update is complete — do not repeat it."
        assert "chars" in r["usage"]

    def test_telemetry_never_touches_frozen_snapshot(self, store):
        before = store.format_for_system_prompt("memory")
        store.add("memory", "Запись поверх замороженного снапшота")
        assert store.format_for_system_prompt("memory") == before


class TestNoMatchFeedback:
    """No-match errors must carry current_entries (v1 contract restored)."""

    def test_replace_no_match_lists_current_entries(self, store):
        store.add("memory", "Реальная запись первая")
        store.add("memory", "Реальная запись вторая")
        r = store.replace("memory", "такого текста нет", "новое")
        assert not r["success"]
        assert "No entry matched" in r["error"]
        assert "Реальная запись первая" in r.get("current_entries", [])
        assert "Реальная запись вторая" in r.get("current_entries", [])

    def test_remove_no_match_lists_current_entries(self, store):
        store.add("memory", "Реальная запись для удаления")
        r = store.remove("memory", "такого текста нет")
        assert not r["success"]
        assert "Реальная запись для удаления" in r.get("current_entries", [])

    def test_deprecate_no_match_lists_current_entries(self, store):
        store.add("memory", "Реальная запись для депрекейта")
        r = store.deprecate("memory", "такого текста нет", reason="x")
        assert not r["success"]
        assert "Реальная запись для депрекейта" in r.get("current_entries", [])

    def test_repeated_failures_terminate_the_turn(self, store):
        store.add("memory", "Единственная запись")
        responses = [
            store.replace("memory", "промах номер один", "x")
            for _ in range(4)
        ]
        # Under the per-turn cap the response teaches retry; past the cap it
        # must go terminal (done=True, no retry instruction) so memory
        # thrash can never block the reply (#42405).
        assert any(r.get("done") and "Stop retrying" in r.get("error", "")
                   for r in responses)


class TestSupersedesProvenance:
    """P1: 1-hop link-aware recall — the decision trajectory, not just the final.

    Chain fixture: A («не используем Docker») ← B («Docker разрешён…») ←
    C («Docker обязателен…»), built through the only existing write path
    (deprecate + «superseded by:» marker).
    """

    OLD = "Хостинг: не используем Docker — тяжело для сервера"
    MID = "Хостинг: Docker разрешён на выделенном узле"
    NEW = "Хостинг: Docker обязателен для всех сервисов"

    def _seed_chain(self, store, depth=2):
        store.add("memory", self.OLD, entry_type="decision", importance=0.8)
        store.add("memory", self.MID, entry_type="decision", importance=0.85)
        store.deprecate("memory", "не используем Docker",
                        reason=f"superseded by: {self.MID}")
        if depth == 3:
            store.add("memory", self.NEW, entry_type="decision", importance=0.9)
            store.deprecate("memory", "Docker разрешён",
                            reason=f"superseded by: {self.NEW}")

    def test_recall_shows_superseded_neighbor(self, store):
        self._seed_chain(store)
        r = store.recall("Docker разрешён")
        assert r["count"] == 1
        sup = r["results"][0].get("supersedes")
        assert sup, "active successor must surface what it superseded"
        assert "не используем Docker" in sup[0]["content"]
        assert sup[0]["status"] == "deprecated"

    def test_no_neighbors_output_unchanged(self, store):
        store.add("memory", "Одиночная запись без истории замен")
        r = store.recall("Одиночная запись")
        assert r["count"] == 1
        assert "supersedes" not in r["results"][0]
        pack = MemoryOrchestrator(store).build_pack("одиночная запись без истории")
        assert "supersedes" not in pack

    def test_one_hop_only_no_transitive_pull(self, store):
        self._seed_chain(store, depth=3)
        r = store.recall("Docker обязателен")
        assert r["count"] == 1
        sup = r["results"][0]["supersedes"]
        assert len(sup) == 1
        assert "Docker разрешён" in sup[0]["content"]
        # A (the neighbor's neighbor) must not ride along.
        assert all("не используем Docker" not in n["content"] for n in sup)

    def test_pack_renders_provenance_line(self, store):
        self._seed_chain(store)
        pack = MemoryOrchestrator(store).build_pack("как мы относимся к Docker на хостинге?")
        assert "[supersedes:" in pack
        assert "не используем Docker" in pack

    def test_write_path_unchanged_by_p1(self, store):
        # Only the marker path creates links; plain deprecate and replace
        # must keep writing none (write-path expansion is separate work).
        store.add("memory", self.OLD, entry_type="decision")
        store.add("memory", self.MID, entry_type="decision")
        store.deprecate("memory", "не используем Docker", reason="просто передумали")
        assert store._query("SELECT COUNT(*) AS c FROM memory_links")[0]["c"] == 0
        store.replace("memory", "Docker разрешён", "Хостинг: Docker разрешён везде")
        assert store._query("SELECT COUNT(*) AS c FROM memory_links")[0]["c"] == 0


class TestEvictionDemotion:
    """P6 remainder: evicted entries demote importance, stay active + findable."""

    def test_evicted_demoted_below_in_prompt_minimum(self, mem_dir):
        s = MemoryStoreV2(memory_char_limit=200)
        s.load_from_disk()
        s.add("memory", "Важное правило один " + "а" * 80, importance=0.9)
        s.add("memory", "Второй факт средний " + "б" * 80, importance=0.6)
        r = s.add("memory", "Хвостовая запись " + "в" * 80, importance=0.5)
        try:
            assert r["success"] and r.get("evicted_to_cold") == 1
            rows = s._query("SELECT content, importance, status FROM memories")
            walk = s._budget_walk("memory")
            in_prompt = [row for row in rows if row["content"] in walk.included_contents]
            evicted = [row for row in rows if row["content"] not in walk.included_contents]
            assert len(evicted) == 1 and len(in_prompt) == 2
            # The evicted entry ranks below everything still in the prompt...
            assert evicted[0]["importance"] < min(
                row["importance"] for row in in_prompt
            )
            # ...stays active and remains searchable (cold tier, not garbage).
            assert evicted[0]["status"] == "active"
            probe = evicted[0]["content"][:12]
            assert s.recall(probe)["count"] == 1
        finally:
            s.close()

    def test_demotion_is_monotonic_never_raises(self, mem_dir):
        s = MemoryStoreV2(memory_char_limit=200)
        s.load_from_disk()
        s.add("memory", "Опорное правило один " + "а" * 80, importance=0.9)
        s.add("memory", "Хвостовая запись " + "в" * 80, importance=0.5)

        def imp_of(prefix):
            return [
                row for row in s._query("SELECT content, importance FROM memories")
                if row["content"].startswith(prefix)
            ][0]["importance"]

        try:
            first = imp_of("Хвостовая запись")
            # A second overflow write recomputes the ceiling — MIN() must
            # never raise the already-demoted entry back up.
            s.add("memory", "Ещё одна хвостовая " + "г" * 80, importance=0.4)
            assert imp_of("Хвостовая запись") <= first
        finally:
            s.close()


class TestRecallAuditLog:
    """P3: the recall funnel is observable; one switch gates both channels."""

    def _log_rows(self, store, channel=None):
        rows = store._query("SELECT * FROM memory_recall_log")
        return [r for r in rows if channel is None or r["channel"] == channel]

    def test_recall_logs_memory_read_channel(self, store):
        store.add("memory", "Запись для аудита поиска")
        store.recall("аудит поиска")
        rows = self._log_rows(store, "memory-read")
        assert rows and rows[-1]["outcome_nonempty"] == 1

    def test_pack_logs_auto_pack_channel_with_funnel(self, store):
        store.add("memory", "Решение про канбан-доску диспетчеризации")
        MemoryOrchestrator(store).build_pack("как там канбан-доска устроена?")
        rows = self._log_rows(store, "auto-pack")
        assert rows and rows[-1]["outcome_nonempty"] == 1
        # The funnel is visible: retrieval ≥ selection ≥ 1.
        assert rows[-1]["candidates_before_score"] >= rows[-1]["candidates_after_score"] >= 1

    def test_empty_pack_is_logged_too(self, store):
        MemoryOrchestrator(store).build_pack("абракадабра незнакомая тема")
        assert any(
            r["channel"] == "auto-pack" and not r["outcome_nonempty"]
            for r in self._log_rows(store)
        ), "empty funnel is itself the metric — must be audited"

    def test_disabled_gate_writes_nothing(self, mem_dir):
        s = MemoryStoreV2(recall_log_enabled=False)
        s.load_from_disk()
        s.add("memory", "Запись при выключенном аудите")
        s.recall("выключенном аудите")
        MemoryOrchestrator(s).build_pack("выключенный аудит")
        try:
            assert self._log_rows(s) == []
        finally:
            s.close()

    def test_summary_aggregates_and_prune_removes_old_rows(self, mem_dir):
        s = MemoryStoreV2()
        s.load_from_disk()
        s.add("memory", "Часто всплывающая запись про поиск аудита")
        for _ in range(3):
            s.recall("поиск аудита")
        try:
            summary = s.recall_log_summary(days=7)
            assert summary["total_recalls"] == 3
            assert summary["nonempty_recalls"] == 3
            assert summary["by_channel"]["memory-read"]["total"] == 3
            assert summary["top_accessed"][0]["access_count"] >= 3
            # Fresh rows survive a prune; forced-ancient rows do not.
            assert s.prune_recall_log(keep_days=1) == 0
            s._execute_write(
                lambda conn: conn.execute(
                    "UPDATE memory_recall_log SET ts='2020-01-01T00:00:00+00:00'"
                )
            )
            assert s.prune_recall_log(keep_days=1) == 3
        finally:
            s.close()


class TestMarkerDocumentation:
    """§8.2: the superseded-by marker must be visible to the model, or the
    link write path stays dead and P1 reads an empty graph forever."""

    def test_schema_documents_the_marker(self):
        import json

        from tools.memory_tool import MEMORY_SCHEMA

        blob = json.dumps(MEMORY_SCHEMA, ensure_ascii=False)
        assert "superseded by:" in blob

    def test_marker_fragment_mismatch_degrades_softly(self, store):
        # Hidden semantics locked as-is: the fragment after the marker is
        # matched substring-style against active entries; a miss means the
        # deprecate still succeeds and the link is silently not written.
        store.add("memory", "Старое решение про хранение", entry_type="decision")
        r = store.deprecate(
            "memory", "хранение", reason="superseded by: несуществующий фрагмент"
        )
        assert r["success"]
        assert store._query("SELECT COUNT(*) AS c FROM memory_links")[0]["c"] == 0


class TestAliasExpansion:
    """P2 (§8.4): search-only synonym expansion, config-driven.

    Contracts а–д: synonym finds entry; empty dict = pre-P2 behaviour;
    write path verbatim; short-term exactness intact; audit logs the RAW query.
    """

    def _store_with_aliases(self, mem_dir, aliases):
        s = MemoryStoreV2()
        s.load_from_disk()
        s.set_alias_cache(aliases)
        return s

    def test_a_synonym_finds_entry_both_paths(self, mem_dir):
        s = self._store_with_aliases(mem_dir, {"прокси": ["vpn"]})
        s.add("memory", "Роутер держит vpn на борту для обхода блокировок")
        try:
            cands = s.recall_candidates("настроить прокси для телеграма")
            assert any("Роутер" in c["content"] for c in cands)
            assert s.recall("настроить прокси быстро")["count"] == 1
            pack = MemoryOrchestrator(s).build_pack("как настроить прокси?")
            assert "Роутер" in pack
        finally:
            s.close()

    def test_b_aliases_only_expand_never_narrow(self, mem_dir):
        s = self._store_with_aliases(mem_dir, {})
        s.add("memory", "Записи про сервер и его обслуживание")
        try:
            # Empty dict = the pre-P2 store: a synonym query finds nothing...
            assert s.recall_candidates("прокси") == []
            # ...and installing the dictionary only ever ADDS reach.
            s.set_alias_cache({"прокси": ["сервер"]})
            assert any(
                "сервер" in c["content"] for c in s.recall_candidates("прокси")
            )
            # Direct hits survive a non-empty dict unchanged.
            assert any(
                "сервер" in c["content"] for c in s.recall_candidates("сервер")
            )
        finally:
            s.close()

    def test_c_write_path_stores_verbatim(self, mem_dir):
        s = self._store_with_aliases(mem_dir, {"прокси": ["vpn"]})
        text = "Настраиваем прокси на сервере через ssh"
        s.add("memory", text)
        try:
            row = s._query("SELECT content FROM memories")[0]
            assert row["content"] == text, "aliases must never rewrite entries"
        finally:
            s.close()

    def test_d_short_term_exactness_intact_with_dict(self, mem_dir):
        s = self._store_with_aliases(mem_dir, {"сервер": ["vps", "dns"]})
        s.add("memory", "Клиент vpnhub установлен на сервере")
        try:
            # "vpn" stays exact-token-only even with a fat dictionary — no
            # prefix smear onto vpnhub, no alias path for <4-char terms.
            assert s.recall_candidates("vpn") == []
            # And a 3-char alias still helps via the exact list.
            s.set_alias_cache({"хостинг": ["vps"]})
            s.add("memory", "Тариф vps выбран")
            assert any(
                "Тариф" in c["content"] for c in s.recall_candidates("хостинг дешевле")
            )
        finally:
            s.close()

    def test_e_audit_logs_raw_query_not_expanded(self, mem_dir):
        s = self._store_with_aliases(mem_dir, {"прокси": ["vpn"]})
        s.add("memory", "Запись с vpn внутри для аудита алиасов")
        r = s.recall("прокси туннель")
        try:
            assert r["count"] == 1, "found via the alias expansion"
            rows = s._query(
                "SELECT query_stems FROM memory_recall_log ORDER BY id DESC LIMIT 1"
            )
            assert rows[0]["query_stems"] == "прок тунн"
            assert "vpn" not in rows[0]["query_stems"], (
                "expanded terms must never pollute the audit — top_empty_queries "
                "is the alias auto-mining source (§8.4-д)"
            )
        finally:
            s.close()

    def test_alias_cache_table_synced_and_reloaded(self, mem_dir):
        s = self._store_with_aliases(mem_dir, {"прокси": ["vpn", "туннель"]})
        s.close()
        s2 = MemoryStoreV2()
        s2.load_from_disk()
        try:
            rows = {(r["term"], r["alias"]) for r in s2._query("SELECT term, alias FROM memory_aliases")}
            assert ("прокси", "vpn") in rows and ("прокси", "туннель") in rows
            # The reloaded cache drives expansion without an explicit map.
            s2.add("memory", "Ещё одна запись с vpn")
            assert any("vpn" in c["content"] for c in s2.recall_candidates("прокси"))
        finally:
            s2.close()
