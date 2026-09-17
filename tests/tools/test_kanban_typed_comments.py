"""Typed kanban comments (2026-09-17 ТЗ): kind / wake / supersede / gate.

Covers the acceptance criteria of the typed-comments spec that are
machine-checkable in-process:

  A2  legacy no-parameter kanban_comment keeps working, writes no mailbox row
  A3  wake=true rides the mailbox circuit (row + wake evaluation), plain
      comments never create mailbox rows
  A4  an unanswered question blocks kanban_complete; an in_reply_to answer
      unblocks; a superseded question does not block
  A5  supersede marks the old comment, double-supersede errors, the original
      row's body/author stay untouched (audit)
  A6  read_task_thread returns kind/superseded_by/in_reply_to/wake_effect and
      the open_questions aggregate, incrementally
  A7  attribution: HERMES_PROFILE wins; without it the active-profile label
      (never the faceless "worker")
  A1* schema migration: a legacy-shape task_comments table gains the typed
      columns and existing rows read as kind='info'

Live-run acceptance (real boards, two profiles, prod-DB migration on a copy)
happens outside pytest — see docs/plans/2026-09-17-kanban-typed-comments-plan.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixture: an isolated board + a claimed (running) task, mirroring
# tests/tools/test_kanban_tools.py::worker_env.
# ---------------------------------------------------------------------------

@pytest.fixture
def board_env(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "orchestrator")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="typed-comments", assignee="brokk")
        kb.claim_task(conn, tid)
    finally:
        conn.close()
    return {"home": home, "task_id": tid, "tmp_path": tmp_path}


def _handle(board_env, tool, args):
    from tools import kanban_tools as kt

    args.setdefault("task_id", board_env["task_id"])
    return json.loads(getattr(kt, f"_handle_{tool}")(args))


def _db(board_env):
    from hermes_cli import kanban_db as kb

    return kb.connect()


def _mkprofile(board_env, name):
    (board_env["tmp_path"] / ".hermes" / "profiles" / name).mkdir(
        parents=True, exist_ok=True
    )


# ---------------------------------------------------------------------------
# A2 — legacy compatibility
# ---------------------------------------------------------------------------

def test_legacy_comment_call_unchanged(board_env):
    out = _handle(board_env, "comment", {"body": "plain old note"})
    assert out["ok"] is True
    assert out["kind"] == "info"
    assert out["wake"] == {"requested": False, "effect": None}

    conn = _db(board_env)
    try:
        from hermes_cli import kanban_db as kb

        comments = kb.list_comments(conn, board_env["task_id"])
        assert len(comments) == 1
        assert comments[0].kind == "info"
        mailbox = conn.execute(
            "SELECT COUNT(*) AS n FROM task_mailbox_messages"
        ).fetchone()["n"]
        assert mailbox == 0, "plain comment must not create a mailbox row"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# kind validation
# ---------------------------------------------------------------------------

def test_kind_validation_rules(board_env):
    bad = _handle(board_env, "comment", {"body": "x", "kind": "shout"})
    assert bad.get("ok") is not True and "kind" in bad["error"]

    woke_info = _handle(board_env, "comment", {"body": "x", "kind": "info", "wake": True})
    assert woke_info.get("ok") is not True and "info" in woke_info["error"]

    corr_no_sup = _handle(
        board_env, "comment", {"body": "x", "kind": "correction"}
    )
    assert corr_no_sup.get("ok") is not True and "supersedes" in corr_no_sup["error"]


# ---------------------------------------------------------------------------
# A4 — open questions block completion
# ---------------------------------------------------------------------------

def test_open_question_blocks_complete_until_answered(board_env):
    q = _handle(
        board_env, "comment", {"body": "which exchange?", "kind": "question"}
    )
    assert q["ok"] is True
    qid = q["comment_id"]

    blocked = _handle(board_env, "complete", {"summary": "done, honest"})
    assert blocked.get("ok") is not True
    assert f"#{qid}" in blocked["error"]

    thread = _handle(board_env, "read_thread", {})
    assert thread["open_questions"] == [qid]

    # Answer with in_reply_to → gate opens.
    ans = _handle(
        board_env,
        "comment",
        {"body": "bybit spot", "in_reply_to": qid},
    )
    assert ans["ok"] is True and ans["in_reply_to"] == qid

    thread = _handle(board_env, "read_thread", {})
    assert thread["open_questions"] == []

    done = _handle(board_env, "complete", {"summary": "done, answered"})
    assert done["ok"] is True


def test_superseded_question_does_not_block(board_env):
    q = _handle(
        board_env, "comment", {"body": "stale question?", "kind": "question"}
    )
    qid = q["comment_id"]

    withdrawn = _handle(
        board_env,
        "comment",
        {
            "body": "question withdrawn — obsolete",
            "kind": "correction",
            "supersedes": qid,
        },
    )
    assert withdrawn["ok"] is True
    assert withdrawn["superseded"] == qid

    thread = _handle(board_env, "read_thread", {})
    assert thread["open_questions"] == []

    done = _handle(board_env, "complete", {"summary": "nothing pending"})
    assert done["ok"] is True


# ---------------------------------------------------------------------------
# A5 — supersede semantics
# ---------------------------------------------------------------------------

def test_supersede_marks_and_rejects_double(board_env):
    first = _handle(
        board_env, "comment", {"body": "bank = 19.9081", "kind": "info"}
    )
    old_id = first["comment_id"]

    second = _handle(
        board_env,
        "comment",
        {
            "body": "bank = 20.1000 (updated)",
            "kind": "correction",
            "supersedes": old_id,
        },
    )
    assert second["ok"] is True

    # Double supersede → error, first replacement stays authoritative.
    third = _handle(
        board_env,
        "comment",
        {
            "body": "bank = 21.0 (also claims to replace it)",
            "kind": "correction",
            "supersedes": old_id,
        },
    )
    assert third.get("ok") is not True
    assert "already superseded" in third["error"]

    conn = _db(board_env)
    try:
        from hermes_cli import kanban_db as kb

        old = [c for c in kb.list_comments(conn, board_env["task_id"]) if c.id == old_id][0]
        # Audit: original content untouched, forward pointer set.
        assert old.body == "bank = 19.9081"
        assert old.superseded_by == second["comment_id"]
    finally:
        conn.close()

    thread = _handle(board_env, "read_thread", {})
    by_id = {c["id"]: c for c in thread["comments"]}
    assert by_id[old_id]["superseded_by"] == second["comment_id"]
    assert by_id[second["comment_id"]]["kind"] == "correction"


def test_supersede_rejects_foreign_task_comment(board_env):
    conn = _db(board_env)
    try:
        from hermes_cli import kanban_db as kb

        other = kb.create_task(conn, title="other", assignee="brokk")
        foreign = kb.add_comment(conn, other, author="someone", body="note")
    finally:
        conn.close()

    out = _handle(
        board_env,
        "comment",
        {"body": "replace?", "kind": "correction", "supersedes": foreign},
    )
    assert out.get("ok") is not True
    assert "does not exist on task" in out["error"]


# ---------------------------------------------------------------------------
# A6 — read_task_thread surface
# ---------------------------------------------------------------------------

def test_read_thread_returns_typed_fields_and_stays_incremental(board_env):
    c1 = _handle(board_env, "comment", {"body": "first", "kind": "info"})
    q = _handle(board_env, "comment", {"body": "q?", "kind": "question"})

    t1 = _handle(board_env, "read_thread", {})
    assert t1["last_comment_id"] == q["comment_id"]
    assert [c["kind"] for c in t1["comments"]] == ["info", "question"]
    assert t1["open_questions"] == [q["comment_id"]]
    for c in t1["comments"]:
        assert {"kind", "superseded_by", "in_reply_to", "wake_effect"} <= set(c)

    _handle(
        board_env,
        "comment",
        {"body": "ans", "in_reply_to": q["comment_id"]},
    )
    t2 = _handle(
        board_env, "read_thread", {"since": t1["last_comment_id"]}
    )
    assert t2["count"] == 1
    assert t2["comments"][0]["in_reply_to"] == q["comment_id"]
    assert t2["open_questions"] == []


# ---------------------------------------------------------------------------
# A3 — wake rides the mailbox circuit
# ---------------------------------------------------------------------------

def test_wake_comment_creates_mailbox_row_plain_does_not(board_env):
    _mkprofile(board_env, "brokk")

    woke = _handle(
        board_env,
        "comment",
        {"body": "course fix: use limit orders", "kind": "guidance", "wake": True},
    )
    assert woke["ok"] is True
    assert woke["wake"]["requested"] is True
    assert woke["wake"]["effect"] in {
        "promoted", "wake_pending", "none_running",
        "status_ineligible", "dependency_blocked",
    }

    conn = _db(board_env)
    try:
        row = conn.execute(
            "SELECT m.id, m.kind, m.wake_requested, e.effect "
            "FROM task_mailbox_messages m "
            "LEFT JOIN task_mailbox_wake_evaluations e ON e.message_id = m.id "
            "WHERE m.comment_id = ?",
            (woke["comment_id"],),
        ).fetchone()
        assert row is not None, "wake=true must create a mailbox row"
        assert row["kind"] == "guidance"
        assert row["wake_requested"] == 1
        assert row["effect"] == woke["wake"]["effect"]
    finally:
        conn.close()

    # Plain comment → still zero extra mailbox rows.
    _handle(board_env, "comment", {"body": "fyi only", "kind": "info"})
    conn = _db(board_env)
    try:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM task_mailbox_messages"
        ).fetchone()["n"]
        assert n == 1
    finally:
        conn.close()

    # wake_effect is visible retroactively in read_task_thread.
    thread = _handle(board_env, "read_thread", {})
    woke_comment = [
        c for c in thread["comments"] if c["id"] == woke["comment_id"]
    ][0]
    assert woke_comment["wake_effect"] == woke["wake"]["effect"]


def test_correction_maps_to_guidance_in_mailbox(board_env):
    _mkprofile(board_env, "brokk")
    old = _handle(board_env, "comment", {"body": "old fact"})
    out = _handle(
        board_env,
        "comment",
        {
            "body": "fixed fact",
            "kind": "correction",
            "supersedes": old["comment_id"],
            "wake": True,
        },
    )
    assert out["ok"] is True
    conn = _db(board_env)
    try:
        row = conn.execute(
            "SELECT kind FROM task_mailbox_messages WHERE comment_id = ?",
            (out["comment_id"],),
        ).fetchone()
        # mailbox kinds are guidance/question/info; correction rides as
        # guidance — the typed view lives on the comment row.
        assert row["kind"] == "guidance"
    finally:
        conn.close()


def test_wake_without_assignee_errors_loudly(board_env):
    conn = _db(board_env)
    try:
        from hermes_cli import kanban_db as kb

        orphan = kb.create_task(conn, title="no assignee", assignee=None)
    finally:
        conn.close()
    out = _handle(
        board_env,
        "comment",
        {"task_id": orphan, "body": "wake nobody", "kind": "guidance", "wake": True},
    )
    assert out.get("ok") is not True
    assert "no assignee" in out["error"]


# ---------------------------------------------------------------------------
# A7 — attribution
# ---------------------------------------------------------------------------

def test_attribution_prefers_env_profile(board_env, monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "loki")
    out = _handle(board_env, "comment", {"body": "from loki"})
    conn = _db(board_env)
    try:
        from hermes_cli import kanban_db as kb

        c = kb.list_comments(conn, board_env["task_id"])[-1]
        assert c.author == "loki"
    finally:
        conn.close()


def test_attribution_never_says_worker_on_default_profile(board_env, monkeypatch):
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    out = _handle(board_env, "comment", {"body": "from default profile"})
    assert out["ok"] is True
    conn = _db(board_env)
    try:
        from hermes_cli import kanban_db as kb

        c = kb.list_comments(conn, board_env["task_id"])[-1]
        assert c.author not in ("worker", ""), c.author
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# §7 — kanban_create readback hash
# ---------------------------------------------------------------------------

def test_create_returns_body_sha256_readback(board_env, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    body = "длинное кириллическое тело карточки для проверки целостности"
    out = _handle(
        board_env,
        "create",
        {"title": "hash check", "assignee": "brokk", "body": body},
    )
    assert out["ok"] is True
    assert out["body_sha256"] == hashlib.sha256(body.encode("utf-8")).hexdigest()

    conn = _db(board_env)
    try:
        from hermes_cli import kanban_db as kb

        stored = kb.get_task(conn, out["task_id"])
        assert stored.body == body
        assert (
            hashlib.sha256(stored.body.encode("utf-8")).hexdigest()
            == out["body_sha256"]
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# A8 — worker context render is typed and deterministic
# ---------------------------------------------------------------------------

def test_worker_context_renders_badges_and_open_marker(board_env, monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "loki")
    _handle(board_env, "comment", {"body": "directive", "kind": "guidance"})
    q = _handle(board_env, "comment", {"body": "why?", "kind": "question"})

    from hermes_cli import kanban_db as kb

    conn = _db(board_env)
    try:
        ctx = kb.build_worker_context(conn, board_env["task_id"])
        ctx2 = kb.build_worker_context(conn, board_env["task_id"])
        assert "[guidance]" in ctx
        assert "[question]" in ctx
        assert "❓OPEN" in ctx
        assert f"in_reply_to={q['comment_id']}" in ctx
        assert "Open questions on this task" in ctx
        # Deterministic render for the same DB snapshot (prompt-cache rule):
        # strip the relative-age stamps, which legitimately advance.
        strip_age = lambda s: "\n".join(  # noqa: E731
            line.split(", ")[0] if ", just now" in line or " ago:" in line else line
            for line in s.splitlines()
        )
        assert strip_age(ctx) == strip_age(ctx2)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# A1* — legacy schema migration
# ---------------------------------------------------------------------------

def test_legacy_db_migrates_typed_columns(board_env, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from hermes_cli import kanban_db as kb

    # Build a realistic "pre-typing" board: full current schema minus the
    # typed-comment columns (drop the covering index first — SQLite refuses
    # DROP COLUMN on indexed columns). This is exactly the shape a real
    # board has before the 2026-09-17 migration pass.
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="pre-typing", assignee="brokk")
        kb.add_comment(conn, tid, author="old-client", body="ancient note")
        conn.execute("DROP INDEX IF EXISTS idx_comments_task_kind")
        conn.execute("ALTER TABLE task_comments DROP COLUMN kind")
        conn.execute("ALTER TABLE task_comments DROP COLUMN superseded_by")
        conn.execute("ALTER TABLE task_comments DROP COLUMN in_reply_to")
        cols = {
            r["name"] for r in conn.execute("PRAGMA table_info(task_comments)")
        }
        assert not {"kind", "superseded_by", "in_reply_to"} & cols
    finally:
        conn.close()

    # Reopening the board runs the additive migration: columns return,
    # pre-existing rows read as plain info, the gate sees no questions.
    kb._INITIALIZED_PATHS.clear()
    conn = kb.connect()
    try:
        cols = {
            r["name"] for r in conn.execute("PRAGMA table_info(task_comments)")
        }
        assert {"kind", "superseded_by", "in_reply_to"} <= cols
        comments = kb.list_comments(conn, tid)
        assert comments[0].kind == "info"
        assert comments[0].superseded_by is None
        assert comments[0].in_reply_to is None
        assert kb.list_open_questions(conn, tid) == []
        # Migration is idempotent on reopen.
        kb._INITIALIZED_PATHS.clear()
        kb.init_db()
    finally:
        conn.close()
