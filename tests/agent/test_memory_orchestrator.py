"""Tests for the Memory Orchestrator (Phase 2 of the memory roadmap).

Three layers, mirroring the Phase-1 test split:

- **Unit**: deterministic intent routing, scoring math, config plumbing.
- **Behavioral** (store-backed): the product question — "when memory
  outgrows the system-prompt snapshot, does the turn still get the
  decisions it needs, within budget, without duplicating the prompt or
  touching the frozen snapshot?"
- **Wiring**: the agent-init factory (enable/disable, legacy store,
  targets) — unit-testable without constructing a full ``AIAgent``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import tools.memory_tool as mt
from agent.memory_orchestrator import (
    DECISION_VALUE_BY_TYPE,
    MemoryOrchestrator,
    ScoringWeights,
    build_memory_orchestrator,
    classify_intent,
    recency_score,
    relevance_score,
    render_pack,
    score_candidate,
)
from agent.memory_store_v2 import MemoryStoreV2
from agent.model_metadata import estimate_tokens_rough


@pytest.fixture()
def mem_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "get_memory_dir", lambda: tmp_path)
    return tmp_path


def _fresh(mem_dir: Path, **kw) -> MemoryStoreV2:
    """New store instance = new agent session (same on-disk memory)."""
    s = MemoryStoreV2(**kw)
    s.load_from_disk()
    return s


def _crowd(
    mem_dir: Path,
    filler: int = 6,
    char_limit: int = 200,
    user_char_limit: int = 1375,
    extra=(),
) -> MemoryStoreV2:
    """Store whose snapshot budget is exhausted by high-importance filler.

    Seed phase (session 1): fillers + ``extra`` (callable(store) or list of
    ``(target, content, type, importance)`` tuples) are written to disk.
    Then the store is reopened (session 2): the frozen snapshot is rebuilt
    from disk — fillers (importance 0.95) occupy the prompt budget, extras
    with lower importance are evicted from the *prompt* but stay searchable
    in the store. That reload matters: mid-session writes never enter the
    frozen snapshot, so seeding then reloading is the only way to exercise
    the real eviction-dedup path.
    """
    s = _fresh(mem_dir, memory_char_limit=char_limit, user_char_limit=user_char_limit)
    for i in range(filler):
        s.add(
            "memory",
            f"Наполнитель номер {i} занимает бюджет снапшота промпта целиком",
            entry_type="fact",
            importance=0.95,
        )
    for item in extra:
        target, content, etype, importance = item
        s.add(target, content, entry_type=etype, importance=importance)
    s.close()
    return _fresh(mem_dir, memory_char_limit=char_limit, user_char_limit=user_char_limit)


# ============================================================================
# Unit — Memory Router (roadmap §3 step 2.1)
# ============================================================================

def test_router_debugging_ru():
    assert classify_intent("почему падает ошибка при старте сервиса") == "debugging"


def test_router_debugging_en():
    assert classify_intent("the deploy crashed, see the traceback") == "debugging"


def test_router_project_work_ru_and_en():
    assert classify_intent("реализуй рефакторинг модуля и обнови миграции") == "project_work"
    assert classify_intent("implement the auth feature and deploy") == "project_work"


def test_router_user_question():
    assert classify_intent("помнишь, какие у меня предпочтения по редактору?") == "user_question"


def test_router_general_when_no_marker():
    assert classify_intent("привет, как сам?") == "general"
    assert classify_intent("") == "general"


def test_router_word_boundary_short_latin_marker():
    # "fix" must not fire inside "prefix" — tokens are matched, not scanned.
    assert classify_intent("add a prefix to the buffer") == "general"


def test_router_debugging_wins_over_project_when_both_present():
    assert classify_intent("реализуй фичу, но тест падает с traceback") == "debugging"


# ============================================================================
# Unit — Memory Scorer (roadmap §3 step 2.2)
# ============================================================================

def test_default_weights_are_the_roadmap_formula():
    w = ScoringWeights()
    assert (w.relevance, w.importance, w.project_link) == (0.35, 0.20, 0.15)
    assert (w.decision_value, w.recency, w.confidence) == (0.15, 0.10, 0.05)
    assert sum(vars(w).values()) == pytest.approx(1.0)


def test_weights_from_config_override_and_ignore_unknown():
    w = ScoringWeights.from_config({"relevance": 0.5, "bogus": 9})
    assert w.relevance == 0.5
    assert w.importance == 0.20  # untouched default
    assert not hasattr(w, "bogus")


def test_relevance_full_partial_none():
    assert relevance_score(["docker", "сервер"], "Docker тяжёл для сервера") == pytest.approx(1.0)
    assert relevance_score(["docker", "сервер"], "Docker тяжёл") == pytest.approx(0.5)
    assert relevance_score(["docker"], "про косметику") == 0.0
    assert relevance_score([], "что угодно") == 0.0


def test_relevance_cyrillic_morphology_prefix():
    # "ошибк" covers "ошибки/ошибка" via prefix matching.
    assert relevance_score(["ошибк"], "Исправили ошибку в миграции") == pytest.approx(1.0)


def test_recency_exponential_decay():
    now = datetime.now(timezone.utc)
    assert recency_score(now.isoformat(), now) == pytest.approx(1.0)
    half_life = recency_score((now - timedelta(days=14)).isoformat(), now)
    assert half_life == pytest.approx(0.5, abs=0.01)
    assert recency_score(None, now) == 0.0
    assert recency_score("не-дата", now) == 0.0
    assert recency_score((now - timedelta(days=200)).isoformat(), now) < 0.001


def test_decision_outranks_equal_fact():
    now = datetime.now(timezone.utc)
    base = {
        "content": "одинаковый текст про сборку",
        "importance": 0.5,
        "confidence": 0.7,
        "updated_at": now.isoformat(),
        "project": None,
    }
    decision = dict(base, type="decision")
    fact = dict(base, type="fact")
    tokens = ["сборк"]
    assert score_candidate(decision, tokens, ScoringWeights(), now) > score_candidate(
        fact, tokens, ScoringWeights(), now
    )


def test_score_bounded_zero_to_one():
    now = datetime.now(timezone.utc)
    row = {
        "content": "x y z unrelated",
        "type": "decision",
        "importance": 1.0,
        "confidence": 1.0,
        "project": "hermes",
        "updated_at": now.isoformat(),
    }
    assert 0.0 <= score_candidate(row, [], ScoringWeights(), now) <= 1.0
    # Project named in the query tokens counts higher than an unnamed one.
    assert score_candidate(row, ["hermes"], ScoringWeights(), now) > score_candidate(
        row, [], ScoringWeights(), now
    )


def test_decision_value_map_prioritizes_standing_records():
    assert DECISION_VALUE_BY_TYPE["decision"] > DECISION_VALUE_BY_TYPE["constraint"]
    assert DECISION_VALUE_BY_TYPE["constraint"] > DECISION_VALUE_BY_TYPE["fact"]


# ============================================================================
# Unit — Context Pack rendering (roadmap §3 steps 2.4/2.5)
# ============================================================================

def _row(rid="11111111-2222-3333-4444-555555555555", **kw):
    base = {
        "id": rid,
        "target": "memory",
        "type": "decision",
        "status": "active",
        "content": "Не используем Docker",
        "importance": 0.8,
        "confidence": 0.7,
        "project": None,
        "updated_at": "2026-08-14T10:00:00+00:00",
    }
    base.update(kw)
    return base


def test_render_pack_empty_returns_empty_string():
    assert render_pack([], "general") == ""


def test_render_pack_structure_ids_and_status():
    pack = render_pack([(_row(), 0.9)], "project_work")
    assert pack.startswith("## Retrieved memory")
    assert "### Decisions" in pack
    assert "[active · 11111111 · 2026-08-14] Не используем Docker" in pack


def test_render_pack_is_plain_markdown_not_memory_context_tag():
    # Roadmap §3 step 2.5: separate context slot — the <memory-context> fence
    # is reserved for external providers; a plain-markdown pack cannot trip
    # the streaming context scrubber either.
    pack = render_pack([(_row(), 0.9)], "general")
    assert "<memory-context" not in pack and "</memory" not in pack
    assert "System note" in pack


def test_render_pack_section_order_follows_intent():
    rows = [
        (_row(type="fact", content="факт один"), 0.5),
        (_row(type="preference", content="предпочтение одно"), 0.5),
        (_row(type="decision", content="решение одно"), 0.5),
    ]
    user_pack = render_pack(rows, "user_question")
    project_pack = render_pack(rows, "project_work")
    assert user_pack.index("### Preferences") < user_pack.index("### Decisions")
    assert project_pack.index("### Decisions") < project_pack.index("### Preferences")


# ============================================================================
# Store-facing orchestrator API (recall_candidates / recent_entries /
# snapshot_contents / bump_access)
# ============================================================================

def test_recall_candidates_token_or_search(mem_dir):
    s = _fresh(mem_dir)
    s.add("memory", "Не используем Docker — тяжело для сервера", entry_type="decision")
    s.add("memory", "Любим строгую типизацию в коде", entry_type="preference")
    # Multi-word message: phrase search would find nothing, token-OR must.
    hits = s.recall_candidates("давай развернём Docker в проде?")
    assert any("Docker" in h["content"] for h in hits)
    assert all(h["status"] in ("active", "pinned") for h in hits)
    assert {"id", "type", "importance", "confidence", "updated_at"} <= set(hits[0].keys())
    s.close()


def test_recall_candidates_excludes_deprecated(mem_dir):
    s = _fresh(mem_dir)
    s.add("memory", "Старый план по Kubernetes", entry_type="decision")
    s.deprecate("memory", "Kubernetes", reason="отказались")
    assert not s.recall_candidates("Kubernetes план")
    s.close()


def test_recall_candidates_stems_russian_inflection(mem_dir):
    """Query word form ≠ stored word form must still hit (сервера/сервером,
    зависимостей/зависимости) — the pack channel lives or dies on this."""
    s = _fresh(mem_dir)
    s.add("memory", "Ставим зависимости через uv, не через pip", entry_type="decision", importance=0.4)
    s.add("memory", "Бэкапы сервера делаем по субботам", entry_type="constraint", importance=0.4)
    s.close()
    s2 = _fresh(mem_dir)
    hits = s2.recall_candidates("а как правильно ставить зависимостей в новых проектах?")
    assert any("зависимости" in h["content"] for h in hits), "inflected query must find the entry"
    hits2 = s2.recall_candidates("расписание бэкапов для сервером?")
    assert any("Бэкапы" in h["content"] for h in hits2)
    s2.close()


def test_recent_entries_orders_by_updated_at_and_filters_type(mem_dir):
    s = _fresh(mem_dir)
    s.add("memory", "Решение номер один", entry_type="decision")
    s.add("memory", "Просто факт", entry_type="fact")
    recent = s.recent_entries(types=["decision", "constraint"], limit=5)
    assert recent and all(r["type"] in ("decision", "constraint") for r in recent)
    assert "Решение номер один" in [r["content"] for r in recent]
    s.close()


def test_snapshot_contents_reflects_budget_eviction(mem_dir):
    s = _crowd(
        mem_dir,
        extra=[("memory", "Не используем Docker — тяжело для сервера", "decision", 0.4)],
    )
    snap = s.snapshot_contents()
    assert "Не используем Docker — тяжело для сервера" not in snap, "evicted from prompt"
    assert any("Наполнитель" in c for c in snap), "filler is in the prompt"
    s.close()


def test_bump_access_is_public(mem_dir):
    s = _fresh(mem_dir)
    r = s.add("memory", "Что-то важное", entry_type="fact")
    s.bump_access([])  # no-op must not raise
    s.close()
    s2 = _fresh(mem_dir)
    s2.recall("важное")
    # access_count persisted across instances (bumped via recall path)
    s2.close()


# ============================================================================
# Behavioral — the per-turn context pack over a real store
# ============================================================================

def test_pack_empty_when_memory_fits_snapshot(mem_dir):
    """Small memory: everything is already in the frozen prompt → no pack.

    The pack is a supplement, never a duplication (roadmap §8 invariant 1:
    big memory ≠ big prompt; here small memory = small prompt). The reload
    matters — the entry must actually be in the load-time snapshot.
    """
    s = _fresh(mem_dir)
    s.add("memory", "Не используем Docker — тяжело для сервера", entry_type="decision")
    s.close()
    s2 = _fresh(mem_dir)
    assert "Docker" in (s2.format_for_system_prompt("memory") or "")
    orch = MemoryOrchestrator(s2)
    assert orch.build_pack("давай развернём Docker?") == ""
    s2.close()


def test_pack_empty_on_blank_query(mem_dir):
    s = _crowd(
        mem_dir,
        extra=[("memory", "Не используем Docker", "decision", 0.4)],
    )
    orch = MemoryOrchestrator(s)
    assert orch.build_pack("   ") == ""
    s.close()


def test_evicted_entry_recalled_via_inflected_query(mem_dir):
    """Live-E2E shape: users ask with arbitrary word forms ("бэкапов", not
    "бэкапы"); the pack must still surface the evicted entry."""
    s = _crowd(
        mem_dir,
        extra=[("memory", "Бэкапы сервера делаем по субботам в 3 утра", "constraint", 0.35)],
    )
    assert "Бэкапы" not in (s.format_for_system_prompt("memory") or ""), "victim evicted"
    pack = MemoryOrchestrator(s).build_pack("как у нас с расписанием бэкапов на сервере?")
    assert "Бэкапы сервера" in pack
    s.close()


def test_evicted_decision_recalled_via_pack(mem_dir):
    """The headline Phase-2 scenario: snapshot evicted it, the pack brings
    it back for exactly the turn that needs it."""
    s = _crowd(
        mem_dir,
        extra=[("memory", "Не используем Docker — тяжело для одного сервера", "decision", 0.4)],
    )
    assert "Docker" not in (s.format_for_system_prompt("memory") or ""), "must be evicted"
    orch = MemoryOrchestrator(s)
    pack = orch.build_pack("давай развернём Docker в проде, реализуй пайплайн?")
    assert "Docker" in pack
    assert "### Decisions" in pack
    # Provenance: status + record id + date on every line.
    assert "[active · " in pack
    s.close()


def test_pack_leaves_frozen_snapshot_untouched(mem_dir):
    """Prefix-cache invariant: mid-session retrieval never rewrites the prompt."""
    s = _crowd(
        mem_dir,
        extra=[("memory", "Не используем Docker — тяжело для одного сервера", "decision", 0.4)],
    )
    snap_before = s.format_for_system_prompt("memory")
    MemoryOrchestrator(s).build_pack("развернём Docker?")
    assert s.format_for_system_prompt("memory") == snap_before
    s.close()


def test_pack_never_contains_deprecated(mem_dir):
    s = _crowd(
        mem_dir,
        extra=[("memory", "Разворачиваем Kubernetes с Docker", "decision", 0.4)],
    )
    s.deprecate("memory", "Kubernetes", reason="перешли на bare metal")
    pack = MemoryOrchestrator(s).build_pack("давай Kubernetes с Docker?")
    assert "Kubernetes" not in pack
    s.close()


def test_pack_respects_token_budget(mem_dir):
    s = _crowd(
        mem_dir,
        filler=4,
        extra=[
            ("memory", f"Запись номер {i} с достаточно длинным содержимым чтобы занимать место в паке", "fact", 0.3)
            for i in range(12)
        ],
    )
    # Fixed pack overhead (header + system note + section titles) is ~155
    # rough tokens; 300 leaves room for a few entries and proves the cut.
    pack = MemoryOrchestrator(s, token_budget=300).build_pack("запись номер содержимым")
    assert pack, "pack must be non-empty"
    assert estimate_tokens_rough(pack) <= 300, "rendered pack must fit the budget"
    s.close()


def test_pack_respects_max_entries(mem_dir):
    s = _crowd(
        mem_dir,
        filler=2,
        extra=[("memory", f"Запис {i} про паковку и бюджет", "fact", 0.4) for i in range(15)],
    )
    pack = MemoryOrchestrator(s, token_budget=2500, max_entries=5).build_pack("запис паковку бюджет")
    assert pack, "pack must be non-empty"
    assert pack.count("\n- [") <= 5
    s.close()


def test_pack_bumps_access_only_for_included(mem_dir):
    s = _crowd(
        mem_dir,
        extra=[
            ("memory", "Не используем Docker — тяжело для сервера", "decision", 0.4),
            ("memory", "Совсем нерелевантная запись про варенье", "fact", 0.9),
        ],
    )
    before = {
        r["content"]: r["access_count"]
        for r in s.recent_entries(limit=50)
    }
    MemoryOrchestrator(s).build_pack("давай Docker в проде?")
    after = {
        r["content"]: r["access_count"]
        for r in s.recent_entries(limit=50)
    }
    docker = [c for c in after if "Docker" in c][0]
    varenye = [c for c in after if "варенье" in c][0]
    assert after[docker] > before[docker], "included entry gets a usage bump"
    assert after[varenye] == before[varenye], "non-included entry is untouched"
    s.close()


def test_pack_logs_nonempty_build_for_observability(mem_dir, caplog):
    """Live-E2E evidence hook: non-empty packs emit exactly one INFO line.

    The pack rides the API copy of the user message and is never persisted,
    so agent.log is the primary way to prove the channel fired on a live bot.
    """
    import logging as _logging

    s = _crowd(
        mem_dir,
        extra=[("memory", "Не используем Docker — тяжело для сервера", "decision", 0.4)],
    )
    with caplog.at_level(_logging.INFO, logger="agent.memory_orchestrator"):
        MemoryOrchestrator(s).build_pack("давай Docker в проде, реализуй пайплайн?")
        pack_logs = [r for r in caplog.records if "context pack" in r.message]
        assert len(pack_logs) == 1
        assert "intent=project_work" in pack_logs[0].message
        caplog.clear()
        # General chit-chat: no candidates → empty pack → no log line.
        MemoryOrchestrator(s).build_pack("какая завтра погода?")
        assert not [r for r in caplog.records if "context pack" in r.message]
    s.close()


def test_pack_failure_is_non_fatal(mem_dir, monkeypatch):
    s = _crowd(
        mem_dir,
        extra=[("memory", "Не используем Docker", "decision", 0.4)],
    )

    def boom(*a, **kw):
        raise RuntimeError("db exploded")

    monkeypatch.setattr(s, "recall_candidates", boom)
    assert MemoryOrchestrator(s).build_pack("Docker?") == ""
    s.close()


def test_zero_budget_disables_pack(mem_dir):
    s = _crowd(
        mem_dir,
        extra=[("memory", "Не используем Docker", "decision", 0.4)],
    )
    assert MemoryOrchestrator(s, token_budget=0).build_pack("Docker?") == ""
    s.close()


def test_user_profile_disabled_limits_pack_targets(mem_dir):
    # A user-filler crowds the user snapshot so the preference is evicted
    # from the prompt — otherwise snapshot dedup would (correctly) keep it
    # out of the pack and the target filter would be untestable.
    s = _crowd(
        mem_dir,
        user_char_limit=60,
        extra=[
            ("user", "Пользователь предпочитает строгие ответы без воды", "preference", 0.95),
            ("user", "Пользователь любит тёмную тему", "preference", 0.4),
        ],
    )
    orch = MemoryOrchestrator(s, targets=("memory",))
    assert "тёмную тему" not in orch.build_pack("какую тему интерфейса prefers?")
    orch_both = MemoryOrchestrator(s, targets=("memory", "user"))
    assert "тёмную тему" in orch_both.build_pack("какую тему интерфейса prefers?")
    s.close()


# ============================================================================
# Wiring — the agent-init factory
# ============================================================================

def test_factory_returns_none_for_legacy_store():
    # A plain object (like the legacy MemoryStore) lacks the v2 API → None.
    assert build_memory_orchestrator(object(), {}, memory_enabled=True, user_profile_enabled=True) is None


def test_factory_disabled_by_config(mem_dir):
    s = _crowd(mem_dir)
    cfg = {"orchestrator": {"enabled": False}}
    assert build_memory_orchestrator(s, cfg, memory_enabled=True, user_profile_enabled=True) is None
    s.close()


def test_factory_none_when_all_targets_disabled(mem_dir):
    s = _crowd(mem_dir)
    assert build_memory_orchestrator(s, {}, memory_enabled=False, user_profile_enabled=False) is None
    s.close()


def test_factory_wires_config_values(mem_dir):
    s = _crowd(mem_dir)
    cfg = {
        "orchestrator": {
            "token_budget": 777,
            "max_entries": 4,
            "weights": {"relevance": 0.6},
        }
    }
    orch = build_memory_orchestrator(s, cfg, memory_enabled=True, user_profile_enabled=False)
    assert orch is not None
    assert orch.token_budget == 777
    assert orch._max_entries == 4
    assert orch.weights.relevance == 0.6
    assert orch._targets == ("memory",)
    s.close()


def test_factory_bad_config_values_fall_back(mem_dir):
    s = _crowd(mem_dir)
    cfg = {"orchestrator": {"token_budget": "много", "max_entries": None}}
    orch = build_memory_orchestrator(s, cfg, memory_enabled=True, user_profile_enabled=True)
    # int("много") raises → factory disables rather than misconfiguring
    assert orch is None
    s.close()


# ============================================================================
# E2E — the pack rides the live conversation loop into the API request
# (mock OpenAI-compatible provider over localhost HTTP; pattern from
# tests/agent/test_empty_tool_name_loop_dampening.py)
# ============================================================================

import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class _MockHandler(BaseHTTPRequestHandler):
    captured_requests: list = []

    def do_POST(self):  # noqa: N8002 (http.server API)
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length).decode())
        type(self).captured_requests.append(req)
        text = "Хорошо, учту."
        is_stream = req.get("stream") is True
        if is_stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [
                {"id": "m", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
                {"id": "m", "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]},
                {"id": "m", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]
            for c in chunks:
                self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            body = json.dumps({
                "id": "m",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *a, **kw):
        pass


@pytest.fixture()
def e2e_env():
    """Mock provider + isolated HERMES_HOME with a crowded memory on disk."""
    _MockHandler.captured_requests = []
    srv = HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    test_home = tempfile.mkdtemp(prefix="hermes_e2e_orch_")
    hermes_home = os.path.join(test_home, ".hermes")
    os.makedirs(hermes_home)
    prev_home = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = hermes_home
    # Shrink the snapshot budget via config so a handful of entries crowds it.
    with open(os.path.join(hermes_home, "config.yaml"), "w", encoding="utf-8") as f:
        f.write("memory:\n  memory_char_limit: 200\n")

    for mod in list(sys.modules):
        if mod == "run_agent" or mod.startswith("agent.") or mod.startswith("tools.") or mod.startswith("hermes_"):
            del sys.modules[mod]

    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key", base_url=f"http://127.0.0.1:{port}/v1",
        provider="openai-compat", model="test-model",
        max_iterations=5, enabled_toolsets=[],
        quiet_mode=True, skip_context_files=True, skip_memory=False,
        save_trajectories=False, platform="cli",
    )

    try:
        yield agent, _MockHandler, hermes_home
    finally:
        srv.shutdown()
        shutil.rmtree(test_home, ignore_errors=True)
        if prev_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = prev_home


def _seed_crowded(hermes_home: str, filler: int = 6, extra=()):
    """Fill the on-disk store before agent init (import AFTER env is set)."""
    from agent.memory_store_v2 import MemoryStoreV2

    s = MemoryStoreV2(memory_char_limit=200, user_char_limit=1375)
    s.load_from_disk()
    for i in range(filler):
        s.add(
            "memory",
            f"Наполнитель номер {i} занимает бюджет снапшота промпта целиком",
            entry_type="fact", importance=0.95,
        )
    for target, content, etype, importance in extra:
        s.add(target, content, entry_type=etype, importance=importance)
    s.close()


def _last_user_contents(handler) -> list:
    out = []
    for req in handler.captured_requests:
        for m in req.get("messages", []):
            if m.get("role") == "user":
                out.append(str(m.get("content", "")))
    return out


def test_e2e_agent_wires_orchestrator_and_injects_pack(e2e_env):
    agent, handler, home = e2e_env
    agent._memory_orchestrator = None
    _seed_crowded(
        home,
        extra=[("memory", "Не используем Docker — тяжело для одного сервера", "decision", 0.4)],
    )
    # Reload memory exactly as a fresh session would, then rebuild the pack source.
    agent._memory_store.load_from_disk()
    from agent.memory_orchestrator import build_memory_orchestrator
    agent._memory_orchestrator = build_memory_orchestrator(
        agent._memory_store, {}, memory_enabled=True, user_profile_enabled=True,
    )
    assert agent._memory_orchestrator is not None

    agent.run_conversation("давай развернём Docker в проде?", conversation_history=[], task_id="t")

    users = _last_user_contents(handler)
    assert users, "a user message must have reached the provider"
    injected = [u for u in users if "Retrieved memory" in u]
    assert injected, "the pack must ride the API copy of the user message"
    assert "Docker" in injected[0]
    assert "<memory-context>" not in injected[0], "pack uses its own slot, not the provider fence"


def test_e2e_small_memory_sends_clean_user_message(e2e_env):
    agent, handler, home = e2e_env
    # Small memory that fits the (config-shrunk) snapshot budget entirely.
    _seed_crowded(home, filler=0, extra=[("memory", "Любим короткие ответы", "preference", 0.8)])
    agent._memory_store.load_from_disk()
    from agent.memory_orchestrator import build_memory_orchestrator
    agent._memory_orchestrator = build_memory_orchestrator(
        agent._memory_store, {}, memory_enabled=True, user_profile_enabled=True,
    )

    agent.run_conversation("какие у нас предпочтения по ответам?", conversation_history=[], task_id="t")

    users = _last_user_contents(handler)
    assert users
    assert not any("Retrieved memory" in u for u in users), "no eviction → no pack → clean message"


def test_e2e_orchestrator_attached_at_agent_init(e2e_env):
    """Default config (memory enabled, store_v2 on) wires the orchestrator."""
    agent, handler, home = e2e_env
    assert getattr(agent, "_memory_orchestrator", None) is not None
    assert agent._memory_store is not None


def test_e2e_bus_attached_at_agent_init(e2e_env):
    """Phase 3: the memory bus rides agent init alongside the orchestrator,
    with the store attached and both read targets offered."""
    agent, handler, home = e2e_env
    bus = getattr(agent, "_memory_bus", None)
    assert bus is not None
    assert bus.store is agent._memory_store
    from agent.memory_bus import READ_MEMORY, READ_USER
    assert {READ_MEMORY, READ_USER} <= bus.capabilities()
