"""Delegation-economy contracts for InsightsEngine (agent token economy).

The delegation tree is reconstructed from ``sessions.parent_session_id``
edges — delegate_task children are full sessions with their own token rows,
so the rollup must work purely from the existing ledger (no new writes,
nothing model-visible at runtime). Tokens only; no cost figures.

Locked contracts: tree rollup includes all descendants; children/main split
and share; depth across multi-level chains; roots outside the window still
resolve; corrupted lineage (cycles) degrades instead of hanging; cache-hit
comparison children vs main.
"""

from __future__ import annotations

import time

import pytest

from hermes_state import SessionDB
from agent.insights import InsightsEngine


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "economy.db")
    yield session_db
    session_db.close()


def _seed(db, sid, *, parent=None, inp=0, out=0, cache_r=0, cache_w=0,
          api=0, title=None, started_days_ago=0.0):
    db.create_session(session_id=sid, source="cli", parent_session_id=parent)
    now = time.time() - started_days_ago * 86400
    db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (now, sid))
    db.update_token_counts(
        sid, input_tokens=inp, output_tokens=out,
        cache_read_tokens=cache_r, cache_write_tokens=cache_w, api_call_count=api,
    )
    if title:
        db._conn.execute("UPDATE sessions SET title=? WHERE id=?", (title, sid))


def _tok(inp, out, cache_r=0, cache_w=0):
    return inp + out + cache_r + cache_w


def test_tree_rollup_and_children_share(db):
    _seed(db, "root", inp=1000, out=100, title="Оценить VPS для обхода")
    _seed(db, "c1", parent="root", inp=5000, out=500, cache_r=2000, api=9,
          title="Проверить хостинг Inferno")
    _seed(db, "c2", parent="c1", inp=3000, out=300, api=5,
          title="Сравнить тарифы")  # depth-2 grandchild
    _seed(db, "main", inp=7000, out=700)

    report = InsightsEngine(db).generate(days=7)
    d = report["delegation"]

    assert d["children"] == 2
    # "main" = every window session WITHOUT a parent — roots included.
    assert d["main_sessions"] == 2
    assert d["children_tokens"] == _tok(5000, 500, 2000) + _tok(3000, 300)
    assert d["main_tokens"] == _tok(1000, 100) + _tok(7000, 700)
    total = d["children_tokens"] + d["main_tokens"]
    assert d["children_share_pct"] == pytest.approx(
        d["children_tokens"] / total * 100
    )
    # Depth: grandchild sits two levels below the root.
    assert d["max_depth"] == 2
    # Tree rollup: root entry carries root + BOTH descendants.
    top = d["top_trees"][0]
    assert top["title"] == "Оценить VPS для обхода"
    assert top["children"] == 2
    assert top["tokens"] == _tok(1000, 100) + d["children_tokens"]
    assert top["api_calls"] == 9 + 5
    # Heaviest child first.
    assert d["top_children"][0]["title"] == "Проверить хостинг Inferno"


def test_root_outside_window_still_resolves(db):
    # Root started 40 days ago (outside a 7-day window); its child ran today.
    _seed(db, "oldroot", inp=100, out=10, title="Старая задача", started_days_ago=40)
    _seed(db, "freshchild", parent="oldroot", inp=2000, out=200, title="Свежий ребёнок")

    report = InsightsEngine(db).generate(days=7)
    d = report["delegation"]
    assert d["children"] == 1
    # The window's only session is a child; its tree resolves to the old
    # root and the rollup includes both rows.
    assert d["top_trees"][0]["title"] == "Старая задача"
    assert d["top_trees"][0]["tokens"] == _tok(100, 10) + _tok(2000, 200)


def test_cycle_in_lineage_degrades_not_hangs(db):
    # sessions.parent_session_id has an FK — build the cycle with a raw
    # update after both rows exist (corruption, not a legal insert).
    db.create_session(session_id="b", source="cli")
    db.create_session(session_id="a", source="cli", parent_session_id="b")
    db._conn.execute("UPDATE sessions SET parent_session_id='a' WHERE id='b'")
    db.update_token_counts("a", input_tokens=10, output_tokens=1)

    report = InsightsEngine(db).generate(days=7)  # must terminate
    assert report["delegation"]["children"] >= 1


def test_cache_hit_children_vs_main(db):
    # Children run cache-cold by definition; main reuses its prefix.
    _seed(db, "root2", inp=1000, out=100, cache_r=9000)
    _seed(db, "coldchild", parent="root2", inp=5000, out=500, cache_r=500)
    _seed(db, "warmsession", inp=1000, out=100, cache_r=9000)

    d = InsightsEngine(db).generate(days=7)["delegation"]
    assert d["cache_hit_children_pct"] == pytest.approx(500 / 5500 * 100)
    # Main side = root2 + warmsession (roots count as main sessions).
    assert d["cache_hit_main_pct"] == pytest.approx(18000 / 20000 * 100)
    assert d["cache_hit_children_pct"] < d["cache_hit_main_pct"]


def test_no_children_section_is_empty(db):
    _seed(db, "lonely", inp=100, out=10)
    d = InsightsEngine(db).generate(days=7)["delegation"]
    assert d["children"] == 0
    assert d["top_trees"][0]["children"] == 0
