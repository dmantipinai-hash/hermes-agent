"""Blocker triage: heuristic digest for needs_input blocks (Ф2, Локи 03.10).

Задание: ``plans/2026-10-03-loki-autonomy-fixes-zcode.md`` §2. Воркер блокируется
с готовым анализом (варианты A/B/C с ценами) — и всё замирает, пока оркестратор
не препарирует сырой трейд. Этот модуль отвечает на событие
``blocker_needs_triage`` структурированным дайджестом: комментарием
``kind=info`` в карточку + событием ``blocker_triage_done``, которое нотификатор
доставляет подписчикам с wake (короткий дайджест вместо полного трейда).

v1 — эвристики без модели. Модельный разбор с загрузкой карточек
strategy-registry (фаза 2 задания) сознательно не строится до вердикта
экоТРИЗ-бенча (§4 задания): строить канал под метод, который может быть закрыт,
— спекулятивная инфраструктура. ТРИЗ-рамка (ИКР/ресурсы/приёмы) присутствует в
дайджесте как статичный шаблон с указанием источника.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

_DIGEST_MAX_CHARS = 1200
_WAKE_DIGEST_MAX_CHARS = 400
_TRIAGE_DEDUP_WINDOW_SECONDS = 30 * 60
_PARENT_TITLES_MAX = 3

_TRIZ_FRAME = (
    "ТРИЗ-рамка (strategy-registry v0.1.0-bench1):\n"
    "- ИКР: сформулируйте «X сам устраняет вред, сохраняя пользу, без усложнения».\n"
    "- Ресурсы: время (фазы/батчи), пространство (сегменты/слои), уже существующие поля и состояния.\n"
    "- Приёмы: разделение во времени / в пространстве / по условию; заранее подложенная подушка; копия вместо оригинала."
)

# Маркеры вариантов: «Вариант A», «A)», «A:», «A —», буллет, нумерация.
_OPTION_LINE_RE = re.compile(
    r"^\s*(?:[-*•]|\d+[.)])?\s*(?:вариант[а-яё]*\s+)?([A-ZА-Я])\s*[)():.—-]\s+\S.*$",
    re.IGNORECASE,
)
_PRICE_RE = re.compile(
    r"(\$|USD|руб\.?|₽|%|час[а-яё]*|мин[а-яё]*|h\b|токен[а-яё]*|долл[а-яё]*)",
    re.IGNORECASE,
)
_RISK_RE = re.compile(
    r"(риск|опасн|потер[яи]|минус|сложн|дорог|падение|деградац|не совместим|конфликт)",
    re.IGNORECASE,
)
_CONTRADICTION_RE = re.compile(
    r"[^.!?\n]*(?:нельзя одновременно|нельзя и|либо\s+[^,.]{3,40},\s+либо|"
    r"улучш(?:ает|ение)\s+[^,.]{3,40},?\s+но\s+ухудш(?:ает|ение)|"
    r"требу(?:ется|ет)\s+и|компромисс между)[^.!?\n]*[.!?]?",
    re.IGNORECASE,
)


def extract_options(text: str) -> list[dict[str, Any]]:
    """Извлечь «варианты» из текста трейда/причины блокировки (эвристика v1).

    Вариант = строка с маркером (A) / «Вариант B» / буллет с буквой или
    нумерацией. К каждому подбираются цена (маркеры $/%/часы/токены) и риск
    (слова риска). Возвращает список {'label', 'line', 'price', 'risk'}.
    """
    options: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or len(line) < 8:
            continue
        m = _OPTION_LINE_RE.match(line)
        if not m:
            continue
        label = (m.group(1) or "").upper()
        if not label or label in seen_labels:
            continue
        seen_labels.add(label)
        price_hit = _PRICE_RE.search(line)
        options.append(
            {
                "label": label,
                "line": line[:200],
                "price": price_hit.group(0) if price_hit else None,
                "risk": bool(_RISK_RE.search(line)),
            }
        )
        if len(options) >= 5:
            break
    return options


def detect_contradiction(text: str) -> Optional[str]:
    """Найти предложение-противоречие (эвристика). None, если не распознано."""
    m = _CONTRADICTION_RE.search(text or "")
    if not m:
        return None
    sentence = m.group(0).strip()
    return sentence[:220] or None


def build_digest(
    *,
    task_title: str,
    block_reason: Optional[str],
    thread_text: str,
    parent_titles: list[str],
) -> str:
    """Собрать дайджест блокера (≤ ~1200 символов, эвристики v1)."""
    source = f"{block_reason or ''}\n{thread_text or ''}"
    sections: list[str] = [f"Дайджест блокера «{task_title[:120]}» (авто, kanban-triage)"]

    essence_lines = [
        ln.strip()
        for ln in (block_reason or "").strip().splitlines()
        if ln.strip()
    ][:3]
    if essence_lines:
        sections.append("Суть:\n" + "\n".join(f"- {ln[:200]}" for ln in essence_lines))

    contradiction = detect_contradiction(source)
    if contradiction:
        sections.append(
            f"Противоречие (распознано эвристикой): {contradiction}\n{_TRIZ_FRAME}"
        )

    options = extract_options(source)
    if options:
        lines = []
        for opt in options:
            marks = []
            if opt["price"]:
                marks.append(f"цена ~{opt['price']}")
            if opt["risk"]:
                marks.append("есть риск")
            suffix = f" ({', '.join(marks)})" if marks else ""
            lines.append(f"- {opt['line']}{suffix}")
        sections.append("Варианты из трейда:\n" + "\n".join(lines))
        recommended = next(
            (o for o in options if not o["risk"] and o["price"]), None
        )
        if recommended:
            sections.append(
                f"Эвристическая рекомендация: вариант {recommended['label']} "
                f"(единственный без явного риска с оценкой цены). Решение за "
                f"оркестратором/владельцем."
            )
    else:
        sections.append(
            "Варианты в трейде не распознаны эвристикой — прочитайте трейд "
            "карточки целиком."
        )

    if parent_titles:
        titles = parent_titles[:_PARENT_TITLES_MAX]
        sections.append("Родословная: " + "; ".join(t[:80] for t in titles))

    digest = "\n\n".join(sections)
    if len(digest) > _DIGEST_MAX_CHARS:
        digest = digest[: _DIGEST_MAX_CHARS - 1] + "…"
    return digest


def render_triage_wake_message(
    digest: str, *, task_id: str, board_tag: str, assignee_tag: str
) -> str:
    """Строка уведомления/будилки подписчиков (используется нотификатором)."""
    short = (digest or "").strip().splitlines()
    head = short[0][:160] if short else ""
    detail = next(
        (ln.strip() for ln in short[1:] if ln.strip() and not ln.startswith("-")),
        "",
    )[:_WAKE_DIGEST_MAX_CHARS]
    msg = f"🧭 {board_tag}{assignee_tag}Kanban {task_id} blocker digest — {head}"
    if detail:
        msg += f"\n{detail}"
    return msg


def process_pending_triage(conn, *, board: Optional[str] = None) -> list[dict]:
    """Ответить на все необработанные ``blocker_needs_triage`` события.

    Для каждого события без парного ``blocker_triage_done`` (связка по
    ``source_event_id``), пока карточка всё ещё ``blocked``+``needs_input``:
    строит дайджест, постит его комментарием ``kind=info`` от автора
    ``kanban-triage`` и пишет ``blocker_triage_done`` с коротким дайджестом в
    payload (его доставит нотификатор с wake). Дедуп одного рода: если по этой
    карточке уже есть свежий (окно 30 минут, без промежуточного unblock)
    дайджест — событие помечается обработанным с ``{"skipped": "dedup"}``, без
    нового комментария. Возвращает список dicts для телеметрии/тестов.
    """
    from hermes_cli import kanban_db as kb

    # Связку «событие → его blocker_triage_done» считаем в Python: не все
    # сборки SQLite гарантированно несут JSON1 (json_extract), а объём
    # событий на доске — сотни, не миллионы.
    done_by_task: dict[str, set[int]] = {}
    for d in conn.execute(
        "SELECT task_id, payload FROM task_events "
        "WHERE kind = 'blocker_triage_done'"
    ).fetchall():
        try:
            payload = json.loads(d["payload"]) if d["payload"] else {}
        except (json.JSONDecodeError, TypeError):
            payload = {}
        source_id = (
            payload.get("source_event_id") if isinstance(payload, dict) else None
        )
        if source_id is not None:
            done_by_task.setdefault(d["task_id"], set()).add(int(source_id))
    pending = [
        row
        for row in conn.execute(
            "SELECT id, task_id, created_at, payload FROM task_events "
            "WHERE kind = 'blocker_needs_triage' ORDER BY id"
        ).fetchall()
        if int(row["id"]) not in done_by_task.get(row["task_id"], ())
    ]
    processed: list[dict] = []
    for row in pending:
        task_id = row["task_id"]
        task = kb.get_task(conn, task_id)
        if task is None or task.status != "blocked" or task.block_kind != "needs_input":
            # Карточка уже не ждёт решения (unblock/архив/смена рода) —
            # пометить обработанным, дайджест не нужен.
            kb._append_event(
                conn, task_id, "blocker_triage_done",
                {"source_event_id": int(row["id"]), "skipped": "stale"},
            )
            processed.append({"task_id": task_id, "skipped": "stale"})
            continue

        # Дедуп «одного рода»: свежий дайджест по этой карточке в окне 30
        # минут — не спамить на каждый ре-блок того же рода (включая цикл
        # unblock→re-block без реального решения между ними).
        last_done = conn.execute(
            """
            SELECT d.created_at AS at FROM task_events d
             WHERE d.task_id = ? AND d.kind = 'blocker_triage_done'
             ORDER BY d.id DESC LIMIT 1
            """,
            (task_id,),
        ).fetchone()
        if last_done is not None:
            if (int(row["created_at"]) - int(last_done["at"])) < _TRIAGE_DEDUP_WINDOW_SECONDS:
                kb._append_event(
                    conn, task_id, "blocker_triage_done",
                    {
                        "source_event_id": int(row["id"]),
                        "skipped": "dedup",
                    },
                )
                processed.append({"task_id": task_id, "skipped": "dedup"})
                continue

        comments = kb.list_comments(conn, task_id)
        thread_text = "\n".join(f"{c.author}: {c.body}" for c in comments)
        try:
            event_payload = (
                json.loads(row["payload"]) if row["payload"] else {}
            )
        except (json.JSONDecodeError, TypeError):
            event_payload = {}
        block_reason = (
            event_payload.get("reason")
            if isinstance(event_payload, dict)
            else None
        )
        graph = kb.task_graph_context(conn, task_id)
        parent_titles = [p.get("title") or p.get("id", "?") for p in graph.get("parents", [])]
        digest = build_digest(
            task_title=task.title or task_id,
            block_reason=block_reason,
            thread_text=thread_text,
            parent_titles=parent_titles,
        )
        # Комментарий + событие — одной транзакцией: крах между ними не
        # должен приводить к повторному дайджесту на следующем тике.
        with kb.write_txn(conn):
            cid = kb.add_comment(
                conn,
                task_id,
                author="kanban-triage",
                body=digest,
                kind="info",
            )
            kb._append_event(
                conn, task_id, "blocker_triage_done",
                {
                    "source_event_id": int(row["id"]),
                    "comment_id": cid,
                    "digest": digest[:_WAKE_DIGEST_MAX_CHARS],
                },
            )
        processed.append(
            {"task_id": task_id, "comment_id": cid, "digest_head": digest[:120]}
        )
    return processed
