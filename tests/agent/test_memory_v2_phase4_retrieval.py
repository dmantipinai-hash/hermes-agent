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
