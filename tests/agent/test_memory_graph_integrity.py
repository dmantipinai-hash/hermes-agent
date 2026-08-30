"""Graph integrity tests — schema v4, atomic supersede, safe matcher.

RED-first fixation of the live defects (2026-08-30 memory-graph TZ):
orphaned short-ID edges no JOIN can read, no FK protection, silent
link-miss success, a case-sensitive split that crashes on 'Superseded by:',
a noisy OR-matcher that pairs unrelated topics over generic words, and a
report that hides noise scale behind a cap.
"""

from __future__ import annotations

import json
import sqlite3
import uuid

import pytest

import tools.memory_tool as mt
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


def _uuid_with_prefix(prefix8: str, tail: str) -> str:
    """A valid-length uuid-shaped id starting with ``prefix8``."""
    rest = (uuid.uuid4().hex[len(prefix8):])[: 36 - len(prefix8) - len(tail)]
    mid = f"{prefix8}{rest}"
    return f"{mid}{tail}"[:36]


def _seed_v3_links(store, edges):
    """Rewrite memory_links into the v3 shape (no FKs) with raw endpoints."""
    def _write(conn):
        conn.execute("DROP TABLE IF EXISTS memory_links")
        conn.execute(
            "CREATE TABLE memory_links ("
            " source_id TEXT NOT NULL, target_id TEXT NOT NULL,"
            " relation_type TEXT NOT NULL, created_at TEXT NOT NULL,"
            " PRIMARY KEY (source_id, target_id, relation_type))"
        )
        for src, tgt, rel in edges:
            conn.execute(
                "INSERT INTO memory_links(source_id, target_id, relation_type, created_at)"
                " VALUES (?,?,?, datetime('now'))",
                (src, tgt, rel),
            )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', '3')"
        )

    store._execute_write(_write)


class TestSchemaV4Migration:
    def test_short_id_edges_restored_to_full_uuids(self, store, mem_dir):
        full_a = _uuid_with_prefix("aaaaaaaa", "1")
        full_b = _uuid_with_prefix("bbbbbbbb", "1")
        store._execute_write(lambda conn: conn.execute(
            "INSERT INTO memories(id, target, type, content, importance, confidence,"
            " status, created_at, updated_at) VALUES (?,?,?,?,?,?, 'active', datetime('now'), datetime('now'))",
            (full_a, "memory", "fact", "запись A", 0.5, 0.7),
        ))
        store._execute_write(lambda conn: conn.execute(
            "INSERT INTO memories(id, target, type, content, importance, confidence,"
            " status, created_at, updated_at) VALUES (?,?,?,?,?,?, 'active', datetime('now'), datetime('now'))",
            (full_b, "memory", "fact", "запись B", 0.5, 0.7),
        ))
        _seed_v3_links(store, [("aaaaaaaa", "bbbbbbbb", "related")])
        store.close()

        s2 = MemoryStoreV2()
        s2.load_from_disk()
        rows = s2._query("SELECT source_id, target_id FROM memory_links")
        assert rows and rows[0]["source_id"] == full_a and rows[0]["target_id"] == full_b
        # The restored edge is now JOIN-readable.
        joined = s2._query(
            "SELECT a.content AS sc, b.content AS tc FROM memory_links l"
            " JOIN memories a ON a.id = l.source_id JOIN memories b ON b.id = l.target_id"
        )
        assert joined and joined[0]["sc"] == "запись A" and joined[0]["tc"] == "запись B"
        s2.close()

    def test_unresolvable_endpoints_reported_not_dropped_silently(self, store):
        amb1 = _uuid_with_prefix("cccccccc", "1")
        amb2 = _uuid_with_prefix("cccccccc", "2")
        for ident in (amb1, amb2):
            store._execute_write(lambda conn, i=ident: conn.execute(
                "INSERT INTO memories(id, target, type, content, importance, confidence,"
                " status, created_at, updated_at) VALUES (?,?,?,?,?,?,'active',datetime('now'),datetime('now'))",
                (i, "memory", "fact", f"дубль {i[:8]}", 0.5, 0.7),
            ))
        _seed_v3_links(store, [
            ("cccccccc", "dddddddd", "related"),   # ambiguous + missing
        ])
        store.close()

        s2 = MemoryStoreV2()
        s2.load_from_disk()
        issues = s2._query(
            "SELECT value FROM meta WHERE key='memory_links_v4_migration_issues'"
        )
        assert issues, "unresolvable endpoints must be reported in meta, not dropped silently"
        parsed = json.loads(issues[0]["value"])
        assert parsed and parsed[0]["reason"]
        assert s2._query("SELECT COUNT(*) AS c FROM memory_links")[0]["c"] == 0
        s2.close()

    def test_foreign_key_check_clean_and_self_loop_rejected(self, store):
        r1 = store.add("memory", "первая запись")
        store.add("memory", "вторая запись")
        with pytest.raises(sqlite3.IntegrityError):
            store._execute_write(lambda conn: conn.execute(
                "INSERT INTO memory_links(source_id, target_id, relation_type, created_at)"
                " VALUES (?,?, 'related', datetime('now'))",
                (r1["id"], r1["id"]),
            ))
        assert store._connect().execute("PRAGMA foreign_key_check").fetchall() == []

    def test_graph_integrity_summary_counts(self, store):
        r1 = store.add("memory", "решение хранить бэкапы в S3", entry_type="decision")
        store.add("memory", "решение писать заметки в Obsidian", entry_type="decision")
        rec = store.supersede(
            "memory", old_id=r1["id"],
            content="решение хранить бэкапы в S3 и Glacier",
            entry_type="decision", reason="двойная копия",
        )
        assert rec["success"]
        store._execute_write(lambda conn: conn.execute(
            "INSERT INTO memory_links(source_id, target_id, relation_type, created_at)"
            " VALUES (?,?, 'related', datetime('now'))",
            (r1["id"], rec["new_id"]),
        ))
        s = store.graph_integrity_summary()
        assert s["total"] == 2 and s["valid"] == 2 and s["orphan"] == 0
        assert s["supersedes"] == 1 and s["structural"] == 1
        assert s["foreign_key_violations"] == 0


class TestAtomicSupersede:
    def _seed(self, store):
        r = store.add(
            "memory", "не используем Docker — тяжело для сервера, 2 ГБ RAM",
            entry_type="decision", importance=0.8,
        )
        return r["id"]

    def test_supersede_is_atomic_new_old_link(self, store):
        old_id = self._seed(store)
        r = store.supersede(
            "memory", old_id=old_id,
            content="используем Docker в проде, RAM поднята до 8 ГБ",
            entry_type="decision", importance=0.8,
            reason="сервер усилен, ограничение снято",
        )
        assert r["success"] and r["link_created"] is True
        assert r["relation_type"] == "supersedes" and r["old_id"] == old_id and r["new_id"]
        rows = store._query(
            "SELECT id, status FROM memories WHERE id IN (?, ?)", (old_id, r["new_id"])
        )
        statuses = {row["id"]: row["status"] for row in rows}
        assert statuses[old_id] == "deprecated"
        assert statuses[r["new_id"]] == "active"
        links = store._query(
            "SELECT source_id, target_id, relation_type FROM memory_links"
        )
        assert len(links) == 1
        assert links[0]["source_id"] == r["new_id"] and links[0]["target_id"] == old_id
        # Provenance is visible to recall (1-hop predecessor with date).
        rec = store.recall("Docker сервер")
        current = next(x for x in rec["results"] if "8 ГБ" in x["content"])
        assert current["supersedes"][0]["status"] == "deprecated"

    def test_supersede_old_id_not_found_mutates_nothing(self, store):
        before = store._query("SELECT COUNT(*) AS c FROM memories")[0]["c"]
        r = store.supersede(
            "memory", old_id=str(uuid.uuid4()),
            content="новое решение без старого", entry_type="decision",
        )
        assert not r["success"] and "old_id" in r["error"].lower()
        assert store._query("SELECT COUNT(*) AS c FROM memories")[0]["c"] == before
        assert store._query("SELECT COUNT(*) AS c FROM memory_links")[0]["c"] == 0

    def test_supersede_rejects_foreign_target_and_deprecated_old(self, store):
        old_id = self._seed(store)
        r = store.supersede(
            "user", old_id=old_id, content="запись в другом таргете",
            entry_type="decision",
        )
        assert not r["success"]
        assert store._query(
            "SELECT status FROM memories WHERE id=?", (old_id,)
        )[0]["status"] == "active"
        # Deprecate the old entry, then try to supersede it again.
        store.deprecate("memory", old_id=old_id, reason="снято")
        r2 = store.supersede(
            "memory", old_id=old_id, content="повторная смена",
            entry_type="decision",
        )
        assert not r2["success"]

    def test_supersede_duplicate_content_fails_before_mutation(self, store):
        old_id = self._seed(store)
        store.add("memory", "дубликат контента", entry_type="decision")
        r = store.supersede(
            "memory", old_id=old_id, content="дубликат контента",
            entry_type="decision",
        )
        assert not r["success"]
        assert store._query(
            "SELECT status FROM memories WHERE id=?", (old_id,)
        )[0]["status"] == "active"
        assert store._query("SELECT COUNT(*) AS c FROM memory_links")[0]["c"] == 0


class TestPrefixIdResolution:
    """Live acceptance 30.08: retrieval surfaces show 8-char id prefixes,
    but the id parameters only took full UUIDs — the model missed twice and
    had to dig ids out with sqlite3. The resolver closes that gap."""

    def test_deprecate_accepts_unambiguous_prefix(self, store):
        r_old = store.add("memory", "решение про мониторинг", entry_type="decision")
        r_new = store.add("memory", "решение про observability", entry_type="decision")
        r = store.deprecate(
            "memory", old_id=r_old["id"][:8], superseded_by_id=r_new["id"][:8],
            reason="переход на новый стек",
        )
        assert r["success"] and r["link_created"] is True
        links = store._query("SELECT source_id, target_id FROM memory_links")
        assert links[0]["source_id"] == r_new["id"]
        assert links[0]["target_id"] == r_old["id"]

    def test_supersede_accepts_prefix(self, store):
        r_old = store.add("memory", "старое решение про CI", entry_type="decision")
        r = store.supersede(
            "memory", old_id=r_old["id"][:8],
            content="новое решение про CI раннеры",
            entry_type="decision", reason="переезд",
        )
        assert r["success"] and r["old_id"] == r_old["id"]

    def test_ambiguous_prefix_fails_without_mutation(self, store):
        a1 = _uuid_with_prefix("eeeeeeee", "1")
        a2 = _uuid_with_prefix("eeeeeeee", "2")
        for ident in (a1, a2):
            store._execute_write(lambda conn, i=ident: conn.execute(
                "INSERT INTO memories(id, target, type, content, importance, confidence,"
                " status, created_at, updated_at) VALUES (?,?,?,?,?,?,'active',datetime('now'),datetime('now'))",
                (i, "memory", "decision", f"решение дубль {i[:8]} сервер", 0.5, 0.7),
            ))
        r = store.supersede(
            "memory", old_id="eeeeeeee",
            content="совсем новое решение", entry_type="decision",
        )
        assert not r["success"] and "ambiguous" in r["error"]
        assert store._query("SELECT COUNT(*) AS c FROM memory_links")[0]["c"] == 0
        assert store._query(
            "SELECT COUNT(*) AS c FROM memories WHERE status='deprecated'"
        )[0]["c"] == 0

    def test_bad_length_prefix_rejected_clearly(self, store):
        r = store.deprecate("memory", old_id="abc", reason="короткий префикс")
        assert not r["success"] and "8+ char" in r["error"]


class TestWindowPairsExplainGrowth:
    """Live acceptance 30.08: tensions 47→48 after a supersede looked like a
    broken counter. It is honest — the NEW decision overlaps standing ones —
    but the report must explain the delta, not just grow."""

    def test_window_pairs_flagged_and_counted(self, store):
        store.add("memory", "решение: используем Linux на сервере projects",
                  entry_type="decision")
        store.add("memory", "решение: переносим Linux на другой сервер",
                  entry_type="decision")
        rep = store.choice_report(days=7)
        assert rep["possible_tensions_total"] >= 1
        assert rep["window_pairs"] == rep["possible_tensions_total"]  # всё свежее
        assert all(t.get("in_window") for t in rep["possible_tensions"])

    def test_old_pairs_not_flagged(self, store):
        r1 = store.add("memory", "решение: держим nginx на сервере", entry_type="decision")
        store.add("memory", "решение: переносим nginx на другой сервер",
                  entry_type="decision")
        # Age both entries beyond the window.
        store._execute_write(lambda conn: conn.execute(
            "UPDATE memories SET created_at='2026-01-01T00:00:00+00:00'"
        ))
        rep = store.choice_report(days=7)
        assert rep["possible_tensions_total"] >= 1
        assert rep["window_pairs"] == 0
        assert not any(t.get("in_window") for t in rep["possible_tensions"])


class TestDeprecateRecoveryById:
    def test_superseded_by_id_links(self, store):
        r_old = store.add("memory", "старое решение про CI", entry_type="decision")
        r_new = store.add("memory", "новое решение про CI", entry_type="decision")
        r = store.deprecate(
            "memory", old_id=r_old["id"],
            superseded_by_id=r_new["id"], reason="переезд на другой раннер",
        )
        assert r["success"] and r["link_created"] is True
        links = store._query("SELECT source_id, target_id FROM memory_links")
        assert links[0]["source_id"] == r_new["id"] and links[0]["target_id"] == r_old["id"]

    def test_invalid_superseded_by_id_leaves_old_untouched(self, store):
        r_old = store.add("memory", "активное решение", entry_type="decision")
        r = store.deprecate(
            "memory", old_id=r_old["id"],
            superseded_by_id=str(uuid.uuid4()), reason="битый successor",
        )
        assert not r["success"]
        assert store._query(
            "SELECT status FROM memories WHERE id=?", (r_old["id"],)
        )[0]["status"] == "active"
        assert store._query("SELECT COUNT(*) AS c FROM memory_links")[0]["c"] == 0

    def test_self_supersede_rejected(self, store):
        r_old = store.add("memory", "решение ссылается на себя", entry_type="decision")
        r = store.deprecate(
            "memory", old_id=r_old["id"], superseded_by_id=r_old["id"], reason="self",
        )
        assert not r["success"]

    def test_mixed_case_marker_no_crash_and_first_line_fragment(self, store):
        r_old = store.add("memory", "решение про мониторинг", entry_type="decision")
        r_new = store.add("memory", "решение про observability стек", entry_type="decision")
        r = store.deprecate(
            "memory", old_id=r_old["id"],
            reason="Superseded by: observability стек\nи ещё пояснение на второй строке",
        )
        assert r["success"] and r["link_created"] is True

    def test_marker_mismatch_is_success_with_visible_warning(self, store):
        store.add("memory", "решение про хранение", entry_type="decision")
        r = store.deprecate(
            "memory", "хранение",
            reason="superseded by: пересказ новыми словами, не подстрока",
        )
        assert r["success"]
        assert r["link_created"] is False
        assert "warning" in r and r["warning"]
        assert store._query("SELECT COUNT(*) AS c FROM memory_links")[0]["c"] == 0


class TestSafeMatcher:
    """Generic glue words must not pair unrelated decisions; real overlaps
    must survive. Synthetic corpus in the live failure's shape."""

    def _seed_corpus(self, store):
        store.add("memory", "решение: используем Linux на домашних серверах",
                  entry_type="decision")
        store.add("memory", "решение: используем велотренажёр трижды в неделю",
                  entry_type="decision")
        store.add("memory", "решение: форк развиваем поверх версии 0.16",
                  entry_type="decision")
        store.add("memory", "не используем Docker — тяжело для сервера",
                  entry_type="decision")
        store.add("memory", "используем Docker в проде после апгрейда сервера",
                  entry_type="decision")

    def test_glue_word_overlap_is_not_a_pair(self, store):
        self._seed_corpus(store)
        rep = store.choice_report(days=7)
        pair_blobs = [
            " ".join(e["content"] for e in t["entries"])
            for t in rep["possible_tensions"]
        ]
        for blob in pair_blobs:
            both = lambda a, b: a in blob and b in blob
            assert not both("велотренажёр", "Linux")
            assert not both("велотренажёр", "форк")
            assert not both("Linux", "форк")

    def test_real_overlap_still_pairs_with_evidence(self, store):
        self._seed_corpus(store)
        rep = store.choice_report(days=7)
        docker_pairs = [
            t for t in rep["possible_tensions"]
            if "Docker" in t["entries"][0]["content"]
            and "Docker" in t["entries"][1]["content"]
        ]
        assert docker_pairs, "the true conflicting pair (old vs new Docker decision) must surface"
        assert docker_pairs[0]["shared_terms"]
        assert rep["possible_tensions_total"] >= len(rep["possible_tensions"])

    def test_related_active_ranked_with_evidence_no_auto_suggestion(self, store):
        self._seed_corpus(store)
        r = store.add("memory", "используем Docker и в staging, сервер уже усилен",
                      entry_type="decision")
        assert r["success"] and r["id"]
        assert "suggested_deprecate" not in r
        rel = r.get("related_active") or []
        assert rel, "the genuinely related Docker decisions must be offered"
        top = rel[0]
        assert "Docker" in top["content"]
        assert top.get("shared_terms") and top.get("score")
        # ...and the unrelated topics are not among the candidates.
        for cand in rel:
            assert "велотренажёр" not in cand["content"]
            assert "форк" not in cand["content"]
