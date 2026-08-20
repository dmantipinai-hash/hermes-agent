"""Typed SQLite-backed memory store (Phase 1 of the memory roadmap).

``MemoryStoreV2`` extends the flat-file ``MemoryStore`` with a typed,
status-tracked SQLite backend:

- **SQLite is canonical.** All entries live in ``memories`` with type
  (fact/decision/constraint/pattern/preference/legacy), importance,
  confidence, status (active/dormant/archived/deprecated/pinned), and
  provenance columns.
- **MEMORY.md / USER.md become human-readable projections.** They are
  rewritten from the active entries after every mutation so the store
  stays git-trackable and human-auditable. The projection is *derived*:
  edits made directly to the files are overwritten on the next write.
  SQLite is the source of truth (invariant: "a summary can be rebuilt;
  evidence must not be silently destroyed").
- **Snapshot stays frozen** — captured at ``load_from_disk()`` time from
  the active entries, within the existing char limits. When the budget
  overflows, the least-important entries drop out of the *prompt*, never
  out of the store ("big memory is not a big prompt"). Mid-session writes
  never touch the snapshot (prefix-cache invariant).
- **Migration is one-shot.** On first load, existing MEMORY.md/USER.md
  ``§``-entries import as ``type=legacy, status=active`` and the fact is
  recorded in the ``meta`` table. Nothing is deleted.

Concurrency follows the SessionDB pattern (``hermes_state.py``): one shared
connection guarded by a threading lock, ``BEGIN IMMEDIATE`` transactions,
and bounded retries with jitter for "database is locked" — safe for the
background-review daemon thread and gateway threads alike.

"""

from __future__ import annotations

import json
import logging
import random
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from tools.memory_tool import (
    ENTRY_DELIMITER,
    MemoryStore,
)
from hermes_state import apply_wal_with_fallback

logger = logging.getLogger(__name__)

# Entry types understood by the store. ``legacy`` marks pre-v2 imports.
ENTRY_TYPES = ("fact", "decision", "constraint", "pattern", "preference", "legacy")
# Statuses governing retrieval/snapshot visibility.
ENTRY_STATUSES = ("active", "dormant", "archived", "deprecated", "pinned")
# Statuses that may appear in the system-prompt snapshot and projections.
_VISIBLE_STATUSES = ("active", "pinned")
# Types whose entries carry the user's standing decisions/rules — these get
# a ``[type]`` prefix in the snapshot so the model treats them as binding.
_PREFIXED_TYPES = ("decision", "constraint")

SCHEMA_VERSION = 2

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    target TEXT NOT NULL DEFAULT 'memory',
    type TEXT NOT NULL DEFAULT 'fact',
    content TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    confidence REAL NOT NULL DEFAULT 0.7,
    status TEXT NOT NULL DEFAULT 'active',
    project TEXT,
    source_session TEXT,
    deprecate_reason TEXT,
    written_by TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_accessed_at TEXT,
    access_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_memories_target_status
    ON memories(target, status);
CREATE INDEX IF NOT EXISTS idx_memories_type
    ON memories(type);

CREATE TABLE IF NOT EXISTS memory_links (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id, relation_type)
);

-- Recall-audit log (Phase-4 P3): one row per memory recall, written by the
-- orchestrator (channel='auto-pack') and the explicit read (channel=
-- 'memory-read'). Feeds `hermes memory report`; volume is tiny (hundreds of
-- rows/month). "channel" — "trigger" is a reserved word in SQLite.
CREATE TABLE IF NOT EXISTS memory_recall_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    channel TEXT NOT NULL,
    intent TEXT,
    query_stems TEXT,
    candidates_before_score INTEGER NOT NULL DEFAULT 0,
    candidates_after_score INTEGER NOT NULL DEFAULT 0,
    outcome_nonempty INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_memory_recall_log_ts ON memory_recall_log(ts);

-- Alias cache (Phase-4 P2): query-expansion synonyms. The SOURCE OF TRUTH
-- is `memory.aliases` in config.yaml; this table is a materialized cache
-- (rebuilt wholesale by set_alias_cache) kept for observability and SQL
-- access. Search-only: aliases never mutate stored entries.
CREATE TABLE IF NOT EXISTS memory_aliases (
    term TEXT NOT NULL,
    alias TEXT NOT NULL,
    PRIMARY KEY (term, alias)
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# FTS5 index over entry content, kept in sync by triggers (same shape as
# plugins/memory/holographic/store.py).
_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    content=memories,
    content_rowid=rowid
);
CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content)
    VALUES ('delete', old.rowid, old.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content)
    VALUES ('delete', old.rowid, old.content);
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;
"""

# Bounded retry for SQLite "database is locked" (SessionDB pattern — jitter
# breaks the convoy effect of the built-in busy handler).
_WRITE_MAX_RETRIES = 15
_WRITE_RETRY_BASE_DELAY = 0.020
_WRITE_RETRY_JITTER = 0.130
_LOCKED_MARKERS = ("locked", "busy")


def _now() -> str:
    """UTC timestamp in ISO-8601 (consistent, sortable, timezone-aware)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fts_quote(query: str) -> str:
    """Escape a raw query into a safe FTS5 MATCH phrase.

    Wraps the query in double quotes and doubles any embedded quotes —
    FTS5's standard escaping — so arbitrary user text is treated as a
    literal phrase instead of FTS query syntax.
    """
    escaped = query.replace('"', '""')
    return f'"{escaped}"'


# Word-split + crude stemming for orchestrator token search (recall-
# oriented). Russian/English inflection mostly lives in the word tail, so
# searching by a truncated prefix stem (FTS5 `stem*` / LIKE '%stem%')
# bridges сервера/сервером/серверу, зависимости/зависимостей and the like.
# Known limit: fleeting vowels (ветки/веток share only «вет») need a real
# stemmer — not worth the dependency at this scale.
_TOKEN_SPLIT_RE = re.compile(r"[^\w]+", re.UNICODE)
_TOKEN_SEARCH_MAX = 8
_TOKEN_MIN_STEM = 4
# Short domain terms (vpn/vps/dns/kvm/fts/мчд) are shorter than the stem
# minimum and used to be dropped outright — the orchestrator search was
# blind to them (live incident 2026-08-19: «vpn» present in stored entries,
# «что там с vpn?» recalled zero). Exactly-3-char tokens are matched EXACTLY
# (whole token), never as prefixes: a 3-char prefix is noise (сер* hits
# сервер/серия/серый). Function words of the same length are excluded so
# they cannot OR-open the search. No alias dictionary is required for this
# path to work (Gap B′ is a retrieval bug, not a synonymy gap).
_TOKEN_EXACT_LEN = 3
_EXACT_STOPWORDS = frozenset({
    "все", "всё", "где", "для", "его", "ещё", "как", "кто", "или", "над",
    "она", "они", "под", "при", "там", "тут", "чем", "что", "эта", "эти",
    "and", "are", "but", "can", "for", "had", "has", "its", "not", "off",
    "our", "out", "per", "the", "via", "was", "you",
})


def _query_exact_terms(query: str) -> List[str]:
    """Distinct lowercase 3-char tokens for exact (whole-token) matching.

    Complements :func:`_query_stems`: short domain terms stay searchable
    without stemming (which would both truncate them into noise and, before
    this, discard them entirely).
    """
    seen: set = set()
    out: List[str] = []
    for token in _TOKEN_SPLIT_RE.split(query or ""):
        token = token.lower()
        if len(token) != _TOKEN_EXACT_LEN or token in _EXACT_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= _TOKEN_SEARCH_MAX:
            break
    return out


def _query_stems(query: str) -> List[str]:
    """Distinct prefix stems of query words (len>=4, capped, ordered)."""
    seen: set = set()
    out: List[str] = []
    for token in _TOKEN_SPLIT_RE.split(query or ""):
        token = token.lower()
        if len(token) < _TOKEN_MIN_STEM:
            continue
        stem = token[: max(_TOKEN_MIN_STEM, len(token) - 3)]
        if stem in seen:
            continue
        seen.add(stem)
        out.append(stem)
        if len(out) >= _TOKEN_SEARCH_MAX:
            break
    return out


class MemoryStoreV2(MemoryStore):
    """Typed SQLite memory store; drop-in duck-type replacement for MemoryStore.

    Inherits the flat-file store's public surface (``load_from_disk``,
    ``add``/``replace``/``remove``, ``format_for_system_prompt``) so every
    existing consumer (system_prompt.py, background_review.py, turn_context,
    CLI mixins, ...) keeps working unchanged. Overrides swap the flat-file
    *source of truth* for SQLite while keeping MEMORY.md/USER.md as derived,
    human-readable projections.
    """

    def __init__(
        self,
        memory_char_limit: int = 2200,
        user_char_limit: int = 1375,
        db_path: Optional[Path] = None,
        recall_log_enabled: bool = True,
    ):
        super().__init__(memory_char_limit, user_char_limit)
        # db_path defaults next to MEMORY.md (path via the parent helper so
        # tests monkeypatching tools.memory_tool.get_memory_dir cover both).
        self._db_path = Path(db_path) if db_path else (
            self._path_for("memory").parent / "memory.db"
        )
        self._db_lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        # Recall-audit gate (memory.recall_log.enabled) — single switch for
        # both channels; the store is the one place that knows it.
        self._recall_log_enabled = bool(recall_log_enabled)
        # Query-expansion aliases (Phase-4 P2): {term: [aliases]}. In-memory
        # working copy — config (memory.aliases) is the source of truth, the
        # memory_aliases table is a cache; load_from_disk reads the cache so
        # a store built without an explicit map still expands, and
        # set_alias_cache (agent init, from config) overrides + rewrites it.
        self._alias_map: Dict[str, Tuple[str, ...]] = {}
        # Contents of the entries currently included in the frozen prompt
        # snapshot (per target, original content strings) — the Phase-2
        # orchestrator dedupes its per-turn context pack against this set.
        self._snapshot_entry_contents: Dict[str, set] = {"memory": set(), "user": set()}

    # ------------------------------------------------------------------
    # SQLite plumbing (SessionDB pattern)
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,  # one connection, many threads (lock below)
            timeout=1.0,
            isolation_level=None,    # explicit BEGIN IMMEDIATE transactions
        )
        conn.row_factory = sqlite3.Row
        apply_wal_with_fallback(conn, db_label="memory.db")
        conn.execute("PRAGMA foreign_keys=ON")
        # Unicode-aware LOWER: SQLite's built-in LOWER()/LIKE case folding
        # covers ASCII only, so Cyrillic searches like '%отвечаем%' would
        # miss 'Отвечаем'. Registered per-connection; used by ulower().
        conn.create_function(
            "ulower", 1,
            lambda s: s.lower() if isinstance(s, str) else s,
        )
        self._conn = conn
        return conn

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(_SCHEMA_SQL)
        try:
            conn.executescript(_FTS_SQL)
        except sqlite3.OperationalError as exc:
            # FTS5 unavailable in this build — search falls back to LIKE.
            logger.warning("memory.db: FTS5 unavailable (%s); recall degrades to LIKE", exc)
        # Forward migrations for existing v1 databases (fresh ones get the
        # current schema from _SCHEMA_SQL and land at SCHEMA_VERSION directly).
        stored = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        version = int(stored["value"]) if stored else SCHEMA_VERSION
        if version < 2:
            self._migrate_v1_to_v2(conn)
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )

    @staticmethod
    def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
        """v1 → v2: consumer-provenance column for scoped revert (Phase 3).

        Also rebuilds the FTS index: a pre-FTS database (or one whose FTS
        table was created after rows existed) leaves old rows invisible to
        stem search — the rebuild re-reads the content table.
        """
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(memories)")}
        if "written_by" not in cols:
            conn.execute("ALTER TABLE memories ADD COLUMN written_by TEXT")
        try:
            conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
        except sqlite3.OperationalError as exc:
            logger.debug("memory.db: FTS rebuild skipped (%s)", exc)
        logger.info("memory.db: migrated schema v1 → v2 (written_by column)")

    def _execute_write(self, fn) -> Any:
        """Run ``fn(conn)`` inside a locked BEGIN IMMEDIATE transaction.

        Retries on "database is locked" with jittered backoff so the
        background-review daemon thread and the main agent thread can
        write concurrently without long stalls.
        """
        last_exc: Optional[Exception] = None
        for _attempt in range(_WRITE_MAX_RETRIES):
            with self._db_lock:
                conn = self._connect()
                began = False
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    began = True
                    result = fn(conn)
                    conn.execute("COMMIT")
                    return result
                except sqlite3.OperationalError as exc:
                    if began:
                        try:
                            conn.execute("ROLLBACK")
                        except sqlite3.Error:
                            pass
                    if _is_locked(exc):
                        last_exc = exc
                    else:
                        raise
                except BaseException:
                    if began:
                        try:
                            conn.execute("ROLLBACK")
                        except sqlite3.Error:
                            pass
                    raise
            time.sleep(_WRITE_RETRY_BASE_DELAY + random.uniform(0.0, _WRITE_RETRY_JITTER))
        raise last_exc if last_exc else sqlite3.OperationalError(
            "memory.db write retries exhausted"
        )

    def _query(self, sql: str, params: Tuple = ()) -> List[sqlite3.Row]:
        """Read-only query under the shared lock (snapshot isolation via WAL)."""
        conn = self._connect()
        with self._db_lock:
            return list(conn.execute(sql, params).fetchall())

    def close(self) -> None:
        """Close the database connection (called on agent shutdown)."""
        with self._db_lock:
            if self._conn is not None:
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                except sqlite3.Error:
                    pass  # best-effort
                try:
                    self._conn.close()
                except sqlite3.Error:
                    pass
                self._conn = None

    # ------------------------------------------------------------------
    # Load / migration
    # ------------------------------------------------------------------

    def load_from_disk(self):
        """Initialize SQLite, run the one-shot flat-file migration, snapshot.

        Same contract as the parent: populates the frozen
        ``_system_prompt_snapshot`` (with strict threat-scan) and the live
        entry lists used by tool responses.
        """
        conn = self._connect()
        self._init_schema(conn)
        self._migrate_legacy_files()
        self._load_alias_cache()
        self._refresh_live_state()
        self._rebuild_snapshot()

    def _load_alias_cache(self) -> None:
        """Read the alias cache table into the working map (P2).

        Only used when no config-driven map has been pushed yet —
        :meth:`set_alias_cache` remains the authority once agent init runs.
        """
        try:
            rows = self._query("SELECT term, alias FROM memory_aliases")
        except sqlite3.Error:
            return
        cache: Dict[str, list] = {}
        for r in rows:
            cache.setdefault(r["term"], []).append(r["alias"])
        self._alias_map = {k: tuple(v) for k, v in cache.items()}

    def set_alias_cache(self, alias_map: Optional[Dict[str, Any]]) -> None:
        """Install config-provided aliases and rewrite the cache table (P2).

        ``memory.aliases`` in config.yaml is the source of truth
        ({term: [aliases]}); this wholesale-replaces both the in-memory map
        and the ``memory_aliases`` cache. Unknown shapes are ignored, not
        raised — a malformed config must never break memory.
        """
        clean: Dict[str, Tuple[str, ...]] = {}
        if isinstance(alias_map, dict):
            for term, aliases in alias_map.items():
                if not isinstance(term, str) or not term.strip():
                    continue
                values = [
                    str(a).strip().lower()
                    for a in (aliases if isinstance(aliases, (list, tuple)) else [aliases])
                    if isinstance(a, str) and a.strip()
                ]
                if values:
                    clean[term.strip().lower()] = tuple(dict.fromkeys(values))
        self._alias_map = clean

        def _rewrite(conn: sqlite3.Connection) -> None:
            conn.execute("DELETE FROM memory_aliases")
            conn.executemany(
                "INSERT OR REPLACE INTO memory_aliases(term, alias) VALUES (?,?)",
                [(t, a) for t, aliases in clean.items() for a in aliases],
            )

        try:
            self._execute_write(_rewrite)
        except sqlite3.Error as exc:
            logger.debug("memory v2: alias cache rewrite failed: %s", exc)

    def _migrate_legacy_files(self) -> None:
        """One-shot import of MEMORY.md/USER.md entries into SQLite."""
        migrated = self._query(
            "SELECT value FROM meta WHERE key = 'legacy_migrated'"
        )
        if migrated:
            return

        imported = {"memory": 0, "user": 0}

        def _import(conn: sqlite3.Connection) -> None:
            for target in ("memory", "user"):
                path = self._path_for(target)
                if not path.exists():
                    continue
                entries = self._read_file(path)
                for content in entries:
                    if not content.strip():
                        continue
                    dup = conn.execute(
                        "SELECT 1 FROM memories WHERE target=? AND content=? LIMIT 1",
                        (target, content),
                    ).fetchone()
                    if dup:
                        continue
                    conn.execute(
                        "INSERT INTO memories(id, target, type, content, importance,"
                        " confidence, status, created_at, updated_at)"
                        " VALUES (?,?,?,?,?,?,?,?,?)",
                        (
                            str(uuid.uuid4()), target, "legacy", content, 0.5, 0.7,
                            "active", _now(), _now(),
                        ),
                    )
                    imported[target] += 1
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('legacy_migrated', ?)",
                (_now(),),
            )

        self._execute_write(_import)
        total = imported["memory"] + imported["user"]
        if total:
            logger.info("memory.db: migrated %d legacy entries from flat files", total)

    # ------------------------------------------------------------------
    # Live-state + snapshot
    # ------------------------------------------------------------------

    def _refresh_live_state(self) -> None:
        """Sync the in-memory entry lists (tool-response state) from SQLite.

        Live lists keep only visible entries, in snapshot order, as raw
        content strings — the shape tool responses always had.
        """
        self.memory_entries = [r["content"] for r in self._visible_rows("memory")]
        self.user_entries = [r["content"] for r in self._visible_rows("user")]

    def _visible_rows(self, target: str) -> List[sqlite3.Row]:
        """Visible rows for ``target`` in snapshot priority order."""
        rows = self._query(
            "SELECT * FROM memories WHERE target=? AND status IN ('active','pinned')"
            " ORDER BY (status='pinned') DESC, importance DESC,"
            " datetime(created_at) ASC",
            (target,),
        )
        return rows

    def _rebuild_snapshot(self) -> None:
        """Rebuild the frozen system-prompt snapshot from visible entries.

        Applies the char budget by dropping the least-important entries
        from the *prompt* only (they stay in the store). Threat-scans via
        the inherited sanitizer so a poisoned DB row cannot inject into
        the system prompt.
        """
        for target in ("memory", "user"):
            walk = self._budget_walk(target)
            self._snapshot_entry_contents[target] = walk.included_contents
            sanitized = self._sanitize_entries_for_snapshot(
                walk.texts, "MEMORY.md" if target == "memory" else "USER.md"
            )
            self._system_prompt_snapshot[target] = self._render_block(target, sanitized)

    class _ProjectionWalk(NamedTuple):
        """One budget pass over the visible rows (see :meth:`_budget_walk`)."""

        texts: List[str]                      # rendered entries included in the prompt
        included_contents: set                # their original content strings
        evicted_ids: List[str]                # visible rows dropped by the budget
        used: int                             # chars the projection costs
        visible: int                          # visible rows walked
        min_included_importance: Optional[float]  # lowest importance still in-prompt

    def _budget_walk(self, target: str) -> "_ProjectionWalk":
        """Single source of the projection ordering and cost math.

        Shared by :meth:`_rebuild_snapshot` (freezes the prompt block), the
        v2 write telemetry (so the usage a memory write reports is by
        construction what the next session's snapshot will contain), and
        :meth:`_demote_evicted` (P6 ordering hygiene). Read-only: never
        mutates the frozen snapshot, so calling it mid-session cannot break
        the prompt-cache invariant.
        """
        rows = self._visible_rows(target)
        limit = self._char_limit(target)
        texts: List[str] = []
        included_contents: set = set()
        evicted_ids: List[str] = []
        used = 0
        min_importance: Optional[float] = None
        for row in rows:
            text = self._render_entry_for_prompt(row)
            cost = len(text) + (len(ENTRY_DELIMITER) if texts else 0)
            if texts and used + cost > limit:
                evicted_ids.append(row["id"])  # stays in store, drops out of prompt
                continue
            texts.append(text)
            included_contents.add(row["content"])
            used += cost
            imp = row["importance"]
            min_importance = imp if min_importance is None else min(min_importance, imp)
        return self._ProjectionWalk(
            texts, included_contents, evicted_ids, used, len(rows), min_importance,
        )

    def _demote_evicted(self, target: str) -> int:
        """Demote entries the prompt budget just evicted (P6 remainder).

        One UPDATE over the evicted ids: importance drops to at most half
        the minimum importance still inside the prompt (floor 0.05). This
        keeps future snapshot orderings stable — an evicted entry can no
        longer out-rank an in-prompt entry and flip back in when a
        same-importance entry arrives. ``MIN(importance, ceiling)`` makes
        it monotonic: already-demoted rows are never raised, and repeated
        overflows converge instead of compounding. Status, searchability
        and ``updated_at`` are untouched — this is ordering hygiene, not
        degradation. Returns the number of rows actually lowered.
        """
        walk = self._budget_walk(target)
        if not walk.evicted_ids or walk.min_included_importance is None:
            return 0
        ceiling = max(0.05, round(walk.min_included_importance / 2, 3))
        changed = {"n": 0}

        def _demote(conn: sqlite3.Connection) -> None:
            placeholders = ",".join("?" * len(walk.evicted_ids))
            cur = conn.execute(
                f"UPDATE memories SET importance = MIN(importance, ?)"
                f" WHERE id IN ({placeholders}) AND importance > ?",
                (ceiling, *walk.evicted_ids, ceiling),
            )
            changed["n"] = cur.rowcount

        try:
            self._execute_write(_demote)
        except sqlite3.Error as exc:  # best-effort — never fail a write on demotion
            logger.debug("memory v2: eviction demotion failed: %s", exc)
            return 0
        return int(changed["n"])

    @staticmethod
    def _render_entry_for_prompt(row: sqlite3.Row) -> str:
        """Render one row for the snapshot, adding [type] for binding types."""
        content = row["content"]
        if row["type"] in _PREFIXED_TYPES:
            return f"[{row['type']}] {content}"
        return content

    # ------------------------------------------------------------------
    # Projections
    # ------------------------------------------------------------------

    def _rewrite_projection(self, target: str) -> None:
        """Rewrite MEMORY.md/USER.md from the visible SQLite entries.

        The flat files are derived views: human-readable, git-trackable,
        rebuilt after every mutation. Direct external edits to them are
        intentionally overwritten (SQLite is canonical).
        """
        rows = self._visible_rows(target)
        contents = [r["content"] for r in rows]
        self._set_entries(target, contents)
        path = self._path_for(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_file(path, contents)

    # ------------------------------------------------------------------
    # Mutations (public tool surface — parent contract preserved)
    # ------------------------------------------------------------------

    def add(
        self,
        target: str,
        content: str,
        entry_type: Optional[str] = None,
        importance: Optional[float] = None,
        written_by: Optional[str] = None,
        project: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Insert a typed entry. Legacy two-arg calls keep working.

        ``written_by`` records the bus consumer that produced the entry
        (``"main:{session_id}"``, ``"cron:{job_id}"``, …) — the scoped-revert
        provenance; ``project`` binds the entry to a realm (Phase 3; scoped
        consumers see only their project's rows plus global ones).
        """
        if target not in ("memory", "user"):
            return {"success": False, "error": f"Unknown target '{target}'."}
        content = (content or "").strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}

        scan_error = self._scan_for_add(content)
        if scan_error:
            return {"success": False, "error": scan_error}

        entry_type = self._normalize_type(entry_type, target)
        importance = self._normalize_importance(importance)

        inserted_id: Optional[str] = None

        def _insert(conn: sqlite3.Connection) -> None:
            nonlocal inserted_id
            dup = conn.execute(
                "SELECT id FROM memories WHERE target=? AND content=? AND status!='deprecated' LIMIT 1",
                (target, content),
            ).fetchone()
            if dup:
                inserted_id = None
                return
            inserted_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO memories(id, target, type, content, importance, confidence,"
                " status, project, written_by, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?, 'active', ?, ?, ?, ?)",
                (inserted_id, target, entry_type, content, importance, 0.7,
                 (project or None), (written_by or None), _now(), _now()),
            )

        self._execute_write(_insert)

        if inserted_id is None:
            return self._success_response(target, "Entry already exists (no duplicate added).")

        self._rewrite_projection(target)
        self._demote_evicted(target)
        resp = self._success_response(target, f"Entry added (type={entry_type}).")

        # Contradiction hint: for standing decisions/rules, surface similar
        # active entries so the model can decide to deprecate them.
        if entry_type in ("decision", "constraint"):
            related = self._find_related_active(target, entry_type, content, exclude_id=inserted_id)
            if related:
                resp["related_active"] = related
                resp["hint"] = (
                    "Similar active entries exist. If the new entry supersedes or "
                    "contradicts one of them, call memory(action=deprecate) on it."
                )
        return resp

    def replace(
        self,
        target: str,
        old_text: str,
        new_content: str,
        entry_type: Optional[str] = None,
        importance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Update the entry whose content contains ``old_text`` (parent contract)."""
        if target not in ("memory", "user"):
            return {"success": False, "error": f"Unknown target '{target}'."}
        old_text = (old_text or "").strip()
        new_content = (new_content or "").strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            return {"success": False, "error": "new_content cannot be empty. Use 'remove' to delete entries."}

        scan_error = self._scan_for_add(new_content)
        if scan_error:
            return {"success": False, "error": scan_error}

        new_type = self._normalize_type(entry_type, target) if entry_type else None
        new_importance = self._normalize_importance(importance) if importance is not None else None

        outcome: Dict[str, Any] = {}

        def _update(conn: sqlite3.Connection) -> None:
            rows = conn.execute(
                "SELECT id, content FROM memories WHERE target=? AND ulower(content) LIKE ?"
                " AND status!='deprecated'",
                (target, f"%{old_text.lower()}%"),
            ).fetchall()
            if not rows:
                outcome["no_match"] = True
                return
            unique = {r["content"] for r in rows}
            if len(unique) > 1:
                previews = [r["content"][:80] + ("..." if len(r["content"]) > 80 else "") for r in rows]
                outcome.update({
                    "success": False,
                    "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                    "matches": previews,
                })
                return
            row_id = rows[0]["id"]
            sets = ["content=?", "updated_at=?"]
            params: List[Any] = [new_content, _now()]
            if new_type:
                sets.append("type=?")
                params.append(new_type)
            if new_importance is not None:
                sets.append("importance=?")
                params.append(new_importance)
            params.append(row_id)
            conn.execute(
                f"UPDATE memories SET {', '.join(sets)} WHERE id=?", params,
            )
            outcome.update({"success": True})

        self._execute_write(_update)
        if outcome.get("no_match"):
            return self._no_match_error("replace", target, old_text)
        if not outcome.get("success"):
            return outcome
        self._rewrite_projection(target)
        self._demote_evicted(target)
        return self._success_response(target, "Entry replaced.")

    def remove(self, target: str, old_text: str) -> Dict[str, Any]:
        """Hard-delete the entry containing ``old_text`` (parent contract).

        For standing decisions prefer ``deprecate`` — removal is for garbage.
        """
        if target not in ("memory", "user"):
            return {"success": False, "error": f"Unknown target '{target}'."}
        old_text = (old_text or "").strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}

        outcome: Dict[str, Any] = {}

        def _delete(conn: sqlite3.Connection) -> None:
            rows = conn.execute(
                "SELECT id, content FROM memories WHERE target=? AND ulower(content) LIKE ?"
                " AND status!='deprecated'",
                (target, f"%{old_text.lower()}%"),
            ).fetchall()
            if not rows:
                outcome["no_match"] = True
                return
            unique = {r["content"] for r in rows}
            if len(unique) > 1:
                previews = [r["content"][:80] + ("..." if len(r["content"]) > 80 else "") for r in rows]
                outcome.update({
                    "success": False,
                    "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                    "matches": previews,
                })
                return
            row_id = rows[0]["id"]
            conn.execute("DELETE FROM memories WHERE id=?", (row_id,))
            conn.execute("DELETE FROM memory_links WHERE source_id=? OR target_id=?", (row_id, row_id))
            outcome.update({"success": True})

        self._execute_write(_delete)
        if outcome.get("no_match"):
            return self._no_match_error("remove", target, old_text)
        if not outcome.get("success"):
            return outcome
        self._rewrite_projection(target)
        return self._success_response(target, "Entry removed.")

    # ------------------------------------------------------------------
    # v2 write telemetry + error feedback
    # ------------------------------------------------------------------

    def _success_response(self, target: str, message: str = None) -> Dict[str, Any]:
        """Success telemetry over the PROMPT projection, not the whole store.

        The inherited v1 response measured every visible entry against the
        char limit, so a store that had outgrown its prompt read
        ``"100% — 3,092/2,200 chars"`` — a number the model dutifully
        "fixed" by deleting searchable entries (live incident 2026-08-19:
        ~40 memory calls squeezing 3,722 → 2,199 chars, five entries hard-
        deleted, details surviving only in an Obsidian archive). The v2
        budget already drops the excess from the prompt automatically, so
        the honest usage is the projection's size; entries beyond it are
        reported as cold storage with an explicit do-not-remove instruction.
        """
        self._consolidation_failures = 0  # progress resets the per-turn cap (#42405)
        walk = self._budget_walk(target)
        limit = self._char_limit(target)
        pct = min(100, int((walk.used / limit) * 100)) if limit > 0 else 0
        evicted = walk.visible - len(walk.texts)
        resp = {
            "success": True,
            "done": True,
            "target": target,
            "usage": f"{pct}% — {walk.used:,}/{limit:,} chars",
            "entry_count": walk.visible,
        }
        if message:
            resp["message"] = message
        if evicted > 0:
            resp["evicted_to_cold"] = evicted
            resp["note"] = (
                f"Write saved. Prompt budget is full: {evicted} of {walk.visible} entries "
                "live in cold storage — automatically excluded from the prompt, still "
                "fully searchable via recall. Eviction is automatic; do NOT remove, "
                "shorten or archive entries to free space. This update is complete — "
                "do not repeat it."
            )
        else:
            resp["note"] = "Write saved. This update is complete — do not repeat it."
        return resp

    def _no_match_error(self, action: str, target: str, old_text: str) -> Dict[str, Any]:
        """No-match error carrying the store's current entries (v1 contract).

        The v2 mutations originally returned a bare ``No entry matched``
        dict, dropping v1's ``current_entries`` feedback — the model had to
        re-guess ``old_text`` blind and thrashed on retries (live incident
        2026-08-20: two failed replaces quoting a document's wording instead
        of the stored entry text). Restore the entries list and route through
        the per-turn consolidation-failure cap like v1.
        """
        entries = [r["content"] for r in self._visible_rows(target)]
        return self._consolidation_failure({
            "success": False,
            "error": (
                f"No entry matched '{old_text}'. Check current_entries below and "
                f"retry with a short unique substring of the exact stored text "
                f"(to {action})."
            ),
            "current_entries": entries,
        })

    # ------------------------------------------------------------------
    # New v2 operations
    # ------------------------------------------------------------------

    def deprecate(self, target: str, old_text: str, reason: str = "") -> Dict[str, Any]:
        """Mark an entry deprecated (hidden from retrieval/snapshot, kept for audit).

        Optionally links the deprecated entry to a newer one when
        ``superseded_by`` text is embedded in the reason — Phase 1 keeps it
        simple: status flip + recorded reason + ``memory_links`` row when a
        successor entry is identifiable by substring.
        """
        if target not in ("memory", "user"):
            return {"success": False, "error": f"Unknown target '{target}'."}
        old_text = (old_text or "").strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}

        outcome: Dict[str, Any] = {}

        def _dep(conn: sqlite3.Connection) -> None:
            rows = conn.execute(
                "SELECT id, content, status FROM memories WHERE target=? AND ulower(content) LIKE ?"
                " AND status IN ('active','pinned','dormant')",
                (target, f"%{old_text.lower()}%"),
            ).fetchall()
            if not rows:
                outcome["no_match"] = True
                return
            unique = {r["content"] for r in rows}
            if len(unique) > 1:
                previews = [r["content"][:80] + ("..." if len(r["content"]) > 80 else "") for r in rows]
                outcome.update({
                    "success": False,
                    "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                    "matches": previews,
                })
                return
            row = rows[0]
            conn.execute(
                "UPDATE memories SET status='deprecated', deprecate_reason=?, updated_at=?"
                " WHERE id=?",
                ((reason or "").strip() or None, _now(), row["id"]),
            )
            # Link to a successor when the reason names one via "superseded by: <text>".
            successor = None
            marker = "superseded by:"
            if marker in (reason or "").lower():
                frag = reason.split(marker, 1)[1].strip().strip('"')
                if frag:
                    hit = conn.execute(
                        "SELECT id FROM memories WHERE target=? AND ulower(content) LIKE ?"
                        " AND status!='deprecated' LIMIT 1",
                        (target, f"%{frag.lower()}%"),
                    ).fetchone()
                    if hit:
                        successor = hit["id"]
            if successor:
                conn.execute(
                    "INSERT OR REPLACE INTO memory_links(source_id, target_id, relation_type, created_at)"
                    " VALUES (?,?, 'supersedes', ?)",
                    (successor, row["id"], _now()),
                )
            outcome.update({"success": True, "deprecated_content": row["content"][:120]})

        self._execute_write(_dep)
        if outcome.get("no_match"):
            return self._no_match_error("deprecate", target, old_text)
        if not outcome.get("success"):
            return outcome
        self._rewrite_projection(target)
        resp = self._success_response(target, "Entry deprecated (kept in history, hidden from search).")
        return resp

    def recall(
        self,
        query: str,
        target: Optional[str] = None,
        types: Optional[List[str]] = None,
        status: str = "active",
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Search memories (FTS5, LIKE fallback) and bump access stats.

        Returns matching entries with their type/status metadata — this is
        the Phase-2 scoring hook (``access_count`` / ``last_accessed_at``
        feed future relevance weighting).
        """
        query = (query or "").strip()
        if not query:
            return {"success": False, "error": "query cannot be empty."}

        rows = self._search_rows(query, target, types, status, limit)
        neighbors: Dict[str, List[Dict[str, Any]]] = {}
        if rows:
            self._bump_access([r["id"] for r in rows])
            neighbors = self.supersedes_neighbors([r["id"] for r in rows])
        self.log_recall(
            "memory-read", None,
            _query_stems(query) + _query_exact_terms(query),
            len(rows), len(rows), bool(rows),
        )

        results = []
        for r in rows:
            item = {
                "id": r["id"],
                "target": r["target"],
                "type": r["type"],
                "status": r["status"],
                "importance": r["importance"],
                "content": r["content"],
            }
            # Provenance rides along only when present, so link-free stores
            # produce byte-identical output to the pre-P1 surface.
            if neighbors.get(r["id"]):
                item["supersedes"] = neighbors[r["id"]]
            results.append(item)

        return {
            "success": True,
            "query": query,
            "results": results,
            "count": len(results),
        }

    # ------------------------------------------------------------------
    # Phase-2 orchestrator API
    # ------------------------------------------------------------------

    def _alias_expand(
        self, query: str, stems: List[str], exact_terms: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Expand search terms with configured aliases (P2 — search-only).

        Rules (§8.4): aliases apply ONLY to the stemming path — query tokens
        shorter than 4 chars keep the B′ exact-match path untouched, so a
        dictionary can never degrade short-term precision. Expansion only
        ADDS OR-terms (never removes/narrows); an alias of its own may be a
        short term and then joins the exact list. Capped so a fat dictionary
        cannot blow up the FTS query.
        """
        if not self._alias_map:
            return stems, exact_terms
        out_stems = list(stems)
        out_exact = list(exact_terms)
        for token in _TOKEN_SPLIT_RE.split(query or ""):
            token = token.lower()
            if len(token) < _TOKEN_MIN_STEM:
                continue
            for alias in self._alias_map.get(token, ()):
                if len(alias) >= _TOKEN_MIN_STEM:
                    stem = alias[: max(_TOKEN_MIN_STEM, len(alias) - 3)]
                    if stem not in out_stems:
                        out_stems.append(stem)
                elif len(alias) == _TOKEN_EXACT_LEN and alias not in out_exact:
                    out_exact.append(alias)
        return (
            out_stems[: _TOKEN_SEARCH_MAX * 2],
            out_exact[:_TOKEN_SEARCH_MAX],
        )

    def recall_candidates(
        self,
        query: str,
        limit: int = 30,
        project: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Full-metadata search candidates for the memory orchestrator.

        Unlike :meth:`recall` (exact-phrase FTS with whole-substring LIKE
        fallback — precision-oriented, for the tool surface), this is
        recall-oriented: the query is term-tokenized (see :func:`_query_stems`
        and :func:`_query_exact_terms`) and terms are OR-matched — long words
        as prefix stems (so a full inflected user message still hits entries),
        short domain terms (vpn, vps, dns) as exact tokens. Ranking is
        delegated to the orchestrator's scorer; no access bumping here — the
        orchestrator bumps only entries that actually enter the context pack.

        ``project`` applies the realm filter for scoped consumers (Phase 3,
        invariant Codex §8.4: no foreign-project memory by lexical accident):
        when set, only rows bound to that project **or unbound** (global)
        pass. ``None`` sees everything.
        """
        stems = _query_stems(query)
        exact_terms = _query_exact_terms(query)
        if not stems and not exact_terms:
            return []
        stems, exact_terms = self._alias_expand(query, stems, exact_terms)
        rows: Optional[List[sqlite3.Row]] = self._fts_search_any(stems, exact_terms)
        if rows is None:
            rows = self._like_search_any(stems, exact_terms)
        seen: set = set()
        out: List[Dict[str, Any]] = []
        for r in rows:
            rid = r["id"]
            if rid in seen:
                continue
            seen.add(rid)
            if r["status"] not in ("active", "pinned"):
                continue  # deprecated/archived never reach the prompt
            if project is not None and r["project"] not in (None, project):
                continue  # realm: foreign-project rows stay invisible
            out.append(dict(r))
            if len(out) >= max(1, int(limit)):
                break
        # 1-hop supersedes provenance (P1): the recalled trajectory, not just
        # the final state — what each entry replaced, and when it was decided.
        neighbors = self.supersedes_neighbors([c["id"] for c in out])
        for cand in out:
            if neighbors.get(cand["id"]):
                cand["supersedes"] = neighbors[cand["id"]]
        return out

    def supersedes_neighbors(
        self, row_ids: Sequence[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """1-hop ``supersedes`` predecessors for the given rows (P1).

        One JOIN of ``memory_links`` restricted to the given ids — a
        neighbor's own neighbors are deliberately NOT followed (graph
        traversal is a roadmap non-goal; for <10K rows the written links are
        all a reader needs). Returns ``{row_id: [neighbor, ...]}`` with only
        the rows that actually supersede something; capped at 3 predecessors
        per row. Empty dict when the link table is empty — callers attach the
        data only when present so link-free stores render identically.
        """
        ids = [rid for rid in (row_ids or []) if rid]
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = self._query(
            "SELECT l.source_id AS sid, m.id AS nid, m.content AS ncontent,"
            " m.status AS nstatus, m.created_at AS ncreated, l.created_at AS linked"
            f" FROM memory_links l JOIN memories m ON m.id = l.target_id"
            f" WHERE l.relation_type='supersedes' AND l.source_id IN ({placeholders})"
            " ORDER BY datetime(l.created_at) DESC",
            tuple(ids),
        )
        out: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            entry = {
                "id": r["nid"],
                "short_id": r["nid"][:8],
                "content": r["ncontent"][:160],
                "status": r["nstatus"],
                "created_at": r["ncreated"],
                "linked_at": r["linked"],
            }
            bucket = out.setdefault(r["sid"], [])
            if len(bucket) < 3:
                bucket.append(entry)
        return out

    def rollback_consumer(self, written_by: str, reason: str = "") -> Dict[str, Any]:
        """Scoped revert (Phase 3): deprecate every active row from one consumer.

        The inverse in our semantics is a revision, never a deletion (roadmap
        §1.7-1.8): rows flip to ``deprecated`` with the reason recorded, stay
        auditable, and drop out of retrieval + snapshots on the next session.
        """
        written_by = (written_by or "").strip()
        if not written_by:
            return {"success": False, "error": "written_by cannot be empty."}
        affected: Dict[str, Any] = {}

        def _dep(conn: sqlite3.Connection) -> None:
            cur = conn.execute(
                "UPDATE memories SET status='deprecated', deprecate_reason=?, updated_at=?"
                " WHERE written_by=? AND status IN ('active','pinned')",
                ((reason or f"scoped revert for consumer {written_by}"), _now(), written_by),
            )
            affected["count"] = cur.rowcount

        self._execute_write(_dep)
        count = int(affected.get("count") or 0)
        if count:
            self._rewrite_projection("memory")
            self._rewrite_projection("user")
        return {"success": True, "deprecated": count, "written_by": written_by}

    def recent_entries(
        self,
        types: Optional[List[str]] = None,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        """Most recently updated visible entries, optionally type-filtered."""
        wanted = tuple(t for t in (types or []) if t in ENTRY_TYPES)
        sql = (
            "SELECT * FROM memories WHERE status IN ('active','pinned')"
            " AND target IN ('memory','user')"
        )
        params: List[Any] = []
        if wanted:
            sql += f" AND type IN ({','.join('?' * len(wanted))})"
            params.extend(wanted)
        sql += " ORDER BY datetime(updated_at) DESC LIMIT ?"
        params.append(max(1, int(limit)))
        return [dict(r) for r in self._query(sql, tuple(params))]

    def snapshot_contents(self) -> set:
        """Original content strings currently present in the frozen prompt snapshot.

        Union across targets — the orchestrator uses it to keep its per-turn
        pack a *supplement* to the snapshot, never a duplication.
        """
        union: set = set()
        for contents in self._snapshot_entry_contents.values():
            union.update(contents)
        return union

    def bump_access(self, row_ids: List[str]) -> None:
        """Record retrieval usage (public wrapper for the orchestrator)."""
        self._bump_access(row_ids)

    # ------------------------------------------------------------------
    # Recall-audit log (Phase-4 P3 — observability)
    # ------------------------------------------------------------------

    def log_recall(
        self,
        channel: str,
        intent: Optional[str],
        terms: Sequence[str],
        candidates_before_score: int,
        candidates_after_score: int,
        outcome_nonempty: bool,
    ) -> None:
        """Append one recall-audit row. Best-effort, never raises (P3).

        Called by the orchestrator (``auto-pack``) and :meth:`recall`
        (``memory-read``). No-ops when ``memory.recall_log.enabled`` is false.
        """
        if not self._recall_log_enabled:
            return

        def _append(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO memory_recall_log(ts, channel, intent, query_stems,"
                " candidates_before_score, candidates_after_score, outcome_nonempty)"
                " VALUES (?,?,?,?,?,?,?)",
                (
                    _now(), str(channel or "?"), intent or None,
                    " ".join(str(t) for t in (terms or []))[:200] or None,
                    int(candidates_before_score or 0),
                    int(candidates_after_score or 0),
                    1 if outcome_nonempty else 0,
                ),
            )

        try:
            self._execute_write(_append)
        except sqlite3.Error as exc:  # never fail a recall on its audit trail
            logger.debug("memory v2: recall log append failed: %s", exc)

    def recall_log_summary(self, days: int = 7, stale_days: int = 30) -> Dict[str, Any]:
        """Aggregate the recall log into a health digest (P3).

        The key metric is "knew but stayed silent": an empty outcome whose
        query terms recur. After the B′ fix an empty recall for a term that
        IS stored is an event worth investigating, not the norm.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max(1, int(days)))).isoformat(timespec="seconds")
        rows = self._query(
            "SELECT channel, intent, query_stems, candidates_before_score,"
            " candidates_after_score, outcome_nonempty FROM memory_recall_log"
            " WHERE ts >= ? ORDER BY ts",
            (cutoff,),
        )
        by_channel: Dict[str, Dict[str, int]] = {}
        empty_queries: Dict[str, int] = {}
        had_candidates_but_empty = 0
        for r in rows:
            bucket = by_channel.setdefault(r["channel"], {"total": 0, "nonempty": 0})
            bucket["total"] += 1
            bucket["nonempty"] += r["outcome_nonempty"]
            if not r["outcome_nonempty"]:
                if r["candidates_before_score"]:
                    had_candidates_but_empty += 1
                if r["query_stems"]:
                    key = r["query_stems"]
                    empty_queries[key] = empty_queries.get(key, 0) + 1
        total = len(rows)
        nonempty = sum(b["nonempty"] for b in by_channel.values())
        stale_cutoff = f"-{max(1, int(stale_days))} days"
        dead_weight = self._query(
            "SELECT COUNT(*) AS c FROM memories WHERE status='active'"
            " AND access_count=0 AND datetime(created_at) < datetime('now', ?)",
            (stale_cutoff,),
        )[0]["c"]
        top_accessed = [
            {"content": r["content"][:80], "access_count": r["access_count"]}
            for r in self._query(
                "SELECT content, access_count FROM memories"
                " ORDER BY access_count DESC, datetime(updated_at) DESC LIMIT 5"
            )
        ]
        return {
            "days": max(1, int(days)),
            "total_recalls": total,
            "nonempty_recalls": nonempty,
            "by_channel": by_channel,
            "empty_with_candidates": had_candidates_but_empty,
            "top_empty_queries": sorted(empty_queries.items(), key=lambda kv: -kv[1])[:5],
            "dead_weight_older_than_stale_days": dead_weight,
            "top_accessed": top_accessed,
        }

    def prune_recall_log(self, keep_days: int = 90) -> int:
        """Drop recall-audit rows older than ``keep_days``. Returns rows removed."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max(1, int(keep_days)))).isoformat(timespec="seconds")
        removed = {"n": 0}

        def _prune(conn: sqlite3.Connection) -> None:
            cur = conn.execute("DELETE FROM memory_recall_log WHERE ts < ?", (cutoff,))
            removed["n"] = cur.rowcount

        try:
            self._execute_write(_prune)
        except sqlite3.Error as exc:
            logger.debug("memory v2: recall log prune failed: %s", exc)
            return 0
        return int(removed["n"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_for_add(self, content: str) -> Optional[str]:
        """Threat-scan content before it enters the store (strict scope)."""
        from tools.memory_tool import _scan_memory_content
        return _scan_memory_content(content)

    def _normalize_type(self, entry_type: Optional[str], target: str) -> str:
        if not entry_type:
            # Sensible defaults: user store entries describe the person.
            return "preference" if target == "user" else "fact"
        entry_type = entry_type.strip().lower()
        if entry_type not in ENTRY_TYPES:
            logger.debug("memory v2: unknown type '%s' — storing as fact", entry_type)
            return "fact"
        return entry_type

    @staticmethod
    def _normalize_importance(importance: Optional[float]) -> float:
        if importance is None:
            return 0.5
        try:
            value = float(importance)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, value))

    def _search_rows(
        self,
        query: str,
        target: Optional[str],
        types: Optional[List[str]],
        status: str,
        limit: int,
    ) -> List[sqlite3.Row]:
        """FTS5-first search with LIKE fallback, post-filtered by metadata."""
        target = target if target in ("memory", "user") else None
        wanted_types = tuple(t for t in (types or []) if t in ENTRY_TYPES)
        statuses = ("pinned", "active") if status == "active" else (status,)

        rows = self._fts_search(query)
        if not rows:
            # FTS matches whole tokens only ("персонаж" ≠ "персонажа"), so a
            # zero-hit phrase query falls through to substring search.
            rows = self._like_search(query)
        if not rows:
            # Natural-language queries ("репутация сервер для блокировок")
            # rarely share one contiguous phrase with a stored entry — both
            # the phrase match and the whole-query substring miss while the
            # entries exist. Retry recall-oriented (stem/exact OR-search,
            # alias-expanded, the orchestrator's mechanics) before reporting
            # an empty recall.
            stems = _query_stems(query)
            exact_terms = _query_exact_terms(query)
            if stems or exact_terms:
                stems, exact_terms = self._alias_expand(query, stems, exact_terms)
                rows = self._fts_search_any(stems, exact_terms)
                if rows is None:
                    rows = self._like_search_any(stems, exact_terms)

        out = []
        for r in rows:
            if target and r["target"] != target:
                continue
            if wanted_types and r["type"] not in wanted_types:
                continue
            if r["status"] not in statuses:
                continue
            out.append(r)
            if len(out) >= limit:
                break
        return out

    def _fts_search(self, query: str) -> Optional[List[sqlite3.Row]]:
        """FTS5 MATCH search; ``None`` means FTS unavailable (fall back)."""
        conn = self._connect()
        try:
            with self._db_lock:
                rows = conn.execute(
                    "SELECT m.* FROM memories_fts f JOIN memories m ON m.rowid = f.rowid"
                    " WHERE memories_fts MATCH ? LIMIT 60",
                    (_fts_quote(query),),
                ).fetchall()
            return list(rows)
        except sqlite3.OperationalError as exc:
            logger.debug("memory v2: FTS search failed (%s); using LIKE fallback", exc)
            return None

    def _like_search(self, query: str) -> List[sqlite3.Row]:
        """Substring fallback when FTS5 is unavailable or matched nothing."""
        return self._query(
            "SELECT * FROM memories WHERE ulower(content) LIKE ? LIMIT 60",
            (f"%{query.lower()}%",),
        )

    def _fts_search_any(
        self, stems: List[str], exact_terms: Sequence[str] = (),
    ) -> Optional[List[sqlite3.Row]]:
        """FTS5 any-term OR search; ``None`` → FTS unavailable (fallback).

        Stems are ``\\w+``-only (see :func:`_query_stems`), so the unquoted
        ``stem*`` prefix syntax is safe — no FTS metacharacters can smuggle
        in. Exact short terms are quoted phrases (whole-token match): ``"vpn"``
        must match the token vpn, never prefix-match ``vpnhub``.
        """
        terms = [f"{s}*" for s in stems] + [_fts_quote(t) for t in exact_terms]
        if not terms:
            return None
        match = " OR ".join(terms)
        conn = self._connect()
        try:
            with self._db_lock:
                rows = conn.execute(
                    "SELECT m.* FROM memories_fts f JOIN memories m ON m.rowid = f.rowid"
                    " WHERE memories_fts MATCH ? LIMIT 120",
                    (match,),
                ).fetchall()
            return list(rows)
        except sqlite3.OperationalError as exc:
            logger.debug("memory v2: FTS any-term search failed (%s); LIKE fallback", exc)
            return None

    def _like_search_any(
        self, stems: List[str], exact_terms: Sequence[str] = (),
    ) -> List[sqlite3.Row]:
        """Substring OR fallback for term search (FTS5 unavailable/mismatch)."""
        patterns = [f"%{s.lower()}%" for s in stems]
        patterns.extend(f"%{t.lower()}%" for t in exact_terms)
        if not patterns:
            return []
        clauses = " OR ".join("ulower(content) LIKE ?" for _ in patterns)
        return self._query(
            f"SELECT * FROM memories WHERE ({clauses}) LIMIT 120", tuple(patterns),
        )

    def _find_related_active(
        self, target: str, entry_type: str, content: str, exclude_id: str,
    ) -> List[Dict[str, Any]]:
        """Find up to 3 active same-type entries similar to ``content``.

        Used for the contradiction hint on decision/constraint writes: the
        model sees potentially superseded entries and can deprecate them.
        """
        words = [w for w in content.lower().split() if len(w) > 3][:8]
        if not words:
            return []
        clauses = " OR ".join("ulower(content) LIKE ?" for _ in words)
        params = [f"%{w}%" for w in words]
        rows = self._query(
            f"SELECT id, type, status, importance, content FROM memories"
            f" WHERE target=? AND type=? AND status IN ('active','pinned')"
            f" AND id != ? AND ({clauses}) LIMIT 3",
            (target, entry_type, exclude_id, *params),
        )
        return [
            {"id": r["id"], "type": r["type"], "importance": r["importance"],
             "content": r["content"][:160]}
            for r in rows
        ]

    def _bump_access(self, row_ids: List[str]) -> None:
        """Record retrieval usage (Phase-2 scoring input)."""
        if not row_ids:
            return

        def _bump(conn: sqlite3.Connection) -> None:
            now = _now()
            for rid in row_ids:
                conn.execute(
                    "UPDATE memories SET access_count = access_count + 1,"
                    " last_accessed_at=? WHERE id=?",
                    (now, rid),
                )

        try:
            self._execute_write(_bump)
        except sqlite3.Error as exc:  # best-effort — never fail a read on stats
            logger.debug("memory v2: access bump failed: %s", exc)


def _is_locked(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _LOCKED_MARKERS)
