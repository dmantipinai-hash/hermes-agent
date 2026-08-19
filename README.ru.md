<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent — форк Architecture 2.0 ☤

[![Upstream](https://img.shields.io/badge/Upstream-Hermes%20Agent-FFD700?style=for-the-badge)](https://github.com/NousResearch/hermes-agent)
[![Docs](https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge)](https://hermes-agent.nousresearch.com/docs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![English version](https://img.shields.io/badge/README-English-9cf?style=for-the-badge)](README.md)

> Форк [Hermes Agent](https://github.com/NousResearch/hermes-agent) от
> [Nous Research](https://nousresearch.com), добавляющий к базовому агенту
> **когнитивную систему памяти** (типизированное долговременное хранилище,
> контекст-паки на каждый ход, шину памяти для саб-агентов и крона) и
> **мульти-агентную оркестрацию** (ролевое делегирование, персистентные
> профильные команды, координация задач через kanban-доску). Всё добавлено
> аддитивно — базовое поведение не меняется.
>
> **English README:** [README.md](README.md)

---

## Зачем этот форк

Hermes Agent — самосовершенствующийся AI-агент с циклом обучения, memory,
skills и поддержкой мессенджеров. Форк расширяет его в двух направлениях:

1. **Память, которая реально сохраняется и компонуется** — типизированный
   SQLite-store как канонический источник правды, человекочитаемые
   проекции, контекст-паки на каждый ход (intent-router + скоринг), и
   MemoryBus, через который саб-агенты и cron-задачи читают (и пишут с
   провенансом) память, не владея хранилищем.
2. **Несколько агентов-специалистов под одним оркестратором** — ролевой
   `delegate_task`, персистентные команды профилей (`/team`) и долгие
   асинхронные задачи на SQLite kanban-доске с heartbeat'ами воркеров и
   mailbox-протоколом.

Plus: **OpenAI Codex по подписке ChatGPT** — вход device-code OAuth с
обычным аккаунтом ChatGPT Plus/Pro, без API-ключа.

### Что добавлено

| Слой | Что делает | Поверхность |
|------|-----------|-------------|
| **Memory store v2** | Типизированный SQLite-store (`fact`/`decision`/`constraint`/`pattern`/`preference`), lifecycle-статусы (`active`/`deprecated`/`pinned`), FTS5-поиск, деprecation с детекцией противоречий. `MEMORY.md`/`USER.md` остаются человекочитаемыми проекциями | `memory(action=read/write/...)` |
| **Контекст-паки** | Каждый ход хранилище ищется по сообщению пользователя (intent-routing, скоринг, токен-бюджет); записи, которых ещё нет в замороженном системном промпте, инжектятся в API-копию этого сообщения | автоматом, `memory.orchestrator.*` |
| **MemoryBus** | Единый facade recall/remember для делегирования и крона. Для них read-only по построению; каждая запись несёт `written_by`-провенанс и поддерживает scoped revert | `agent/memory_bus.py` |
| **Фоновый self-review** | Фоновый обзор рассматривает обновления памяти по недавнему диалогу каждые N ходов пользователя (по умолчанию 5) | `memory.nudge_interval` |
| **Ролевое делегирование** | Саб-агенты с преднастроенными toolset'ами и системными промптами под специализацию (`researcher`, `coder`, `reviewer`, `analyst`, `writer`) | `delegate_task(role=...)` |
| **Команды профилей** | Персистентные агенты-профили (финансист, философ, продуктолог...) со своей моделью, памятью, `SOUL.md` — не исчезают после задачи | `/team create`, `ask_agent`, `assign_task` |
| **Kanban-координация** | Долгие асинхронные задачи на SQLite-доске: диспетчер спавнит воркеров, lifecycle с heartbeat, mailbox через тред комментов | `kanban_*`, `read_task_thread` |
| **Crash recovery** | Детекция краша mid-turn, ремонт транскрипта, возобновление сессии | `/resume` |
| **Codex по подписке** | OpenAI Codex Responses API через OAuth аккаунта ChatGPT (device code) — без API-ключа | `hermes setup` → OpenAI Codex |

Все фичи памяти **включены по умолчанию** и не требуют внешних сервисов —
хранилище встроено (SQLite). Внешние memory-провайдеры (mem0, honcho,
supermemory, ...) остаются доступны через `memory.provider` в `config.yaml`.

---

## Как устроена система памяти

```
ход диалога
      │
      ▼
 memory tool ──write──▶ типизированный SQLite-store (memories/memory.db, канонический)
      │                      │
      │ read                 │ проекции
      ▼                      ▼
 orchestrator ──────▶ MEMORY.md / USER.md (человекочитаемые, снапшот в промпте)
  intent router
  + скорер        ──▶ контекст-пак → инжектится в API-копию ЭТОГО хода
  + токен-бюджет
      ▲
      │ recall/remember (записи с провенансом, scoped revert)
      │
 MemoryBus ◀── саб-агенты (delegate_task) · cron-задачи (брифинги)
```

- Пока память влезает в снапшот системного промпта, контекст-пак пуст —
  нулевой оверхед, нулевое изменение поведения, пока память не перерастёт
  бюджет промпта (по умолчанию 2500 токенов, максимум 20 записей).
- Саб-агенты и cron никогда не трогают хранилище напрямую — только через
  шину, которая read-only для них по построению и метит каждую запись
  автором для scoped revert.
- Cron-задачи могут запросить memory-брифинг — шина ищет по хранилищу с
  собранным промптом задачи и инжектит релевантные записи.

## Два режима координации

```
СИНХРОННО                            АСИНХРОННО
──────────────────────                ─────────────────────
менеджер: ask_agent("finance",        менеджер: assign_task("researcher",
  "дай цифры Q3")                       "глубокий анализ рынка, 2 часа")
  → ждёт ответ (блокирующе)            → kanban-задача, диспетчер спавнит
  ← ответ строкой в контекст             воркера, менеджер свободен

                                      позже: message_agent с [question]
                                      воркер: read_task_thread на чекпоинте
                                        → kanban_comment с [answer]
                                      воркер: kanban_complete → notifier пушит
```

«Агент команды» — это профиль со своим `config.yaml` (модель/провайдер),
`.env` (ключи), `state.db` (сессии), `SOUL.md` (персона) и `skills/`,
изолированный по `HERMES_HOME` в `~/.hermes/profiles/<name>/`.

---

## Быстрый старт

### Требования

- **Python 3.11+** (3.12/3.14 тоже работают)
- **uv** — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Git**
- Кред модели: API-ключ (OpenAI, Anthropic, Z.AI/GLM, OpenRouter, ...)
  **или** подписка ChatGPT Plus/Pro (Codex, см. ниже)
- **Node.js 20+** — *опционально*, только для Ink/React TUI (`hermes --tui`)
  и browser-инструментов. Базовый CLI работает без Node.

### Установка

```bash
git clone https://github.com/dmantipinai-hash/hermes-agent.git
cd hermes-agent

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[all]"

hermes setup        # интерактивный мастер: модель, ключи, платформы
```

Или глобальная установка без активации venv:

```bash
uv tool install --from "git+https://github.com/dmantipinai-hash/hermes-agent.git" "hermes-agent[all]"
```

**Обновление:**

```bash
cd hermes-agent
git pull
source .venv/bin/activate
uv pip install -e ".[all]"      # нужно только при изменении зависимостей
```

Release-теги (`v0.16.0`, ...) отмечают стабильные точки — `git checkout
v0.16.0` для фиксированной версии.

### Подключение Codex по подписке ChatGPT

API-ключ не нужен — провайдер `openai-codex` аутентифицируется аккаунтом
ChatGPT (Plus/Pro) через device-code OAuth:

1. `hermes setup` → выбрать **OpenAI Codex**
2. Открыть напечатанный URL, войти с аккаунтом ChatGPT, подтвердить код
3. Выбрать модель Codex — готово

Повторный вход позже — `hermes auth`. Codex работает и как провайдер
делегирования: саб-агенты могут гоняться по той же подписке.

### Telegram-шлюз

```bash
hermes setup            # сохранить токен бота (Telegram BotFather)
hermes gateway run      # запустить шлюз в foreground
```

Зависимость `python-telegram-bot` доустанавливается лениво при первом
использовании. Остальные 12+ платформ (Discord, Slack, Matrix, ...) — см.
документацию upstream.

### Конфигурация toolset'ов (важно!)

Чтобы multi-agent инструменты были видны агенту, включите toolset'ы в
`~/.hermes/config.yaml`:

```yaml
toolsets:
  - hermes-cli
  - agent_manager    # ask_agent, assign_task, list_agents, /team
  - kanban           # read_task_thread, kanban_comment, lifecycle

platform_toolsets:
  cli:
    - hermes-cli
    - agent_manager
    - kanban
```

> ⚠️ **Оба ключа** должны содержать toolset — `toolsets:` гейтит отдельные
> инструменты через check_fn, `platform_toolsets:` активирует toolset целиком.

### Первая multi-agent сессия

```bash
hermes                                        # запустить CLI-оркестратор

# В сессии:
/team create finance --role researcher        # создать профиль finance
/team create writer --role writer             # создать профиль writer
/team list                                    # кто есть, кто занят

# Асинхронная долгая задача:
/team assign finance "анализ бюджета Q3"      # kanban-задача
kanban daemon --force                         # поднять диспетчер
kanban list                                   # статус задач
```

Системе памяти настройка не нужна — она включена по умолчанию. Расскажите
агенту устойчивый факт о себе, выполните `/new` и спросите о нём снова.

---

## Что проверено

Живыми end-to-end тестами (2026-08):

| Сценарий | Статус |
|---|---|
| Память переживает `/new` (решение вспомнено в свежей сессии) | ✅ |
| Recall контекст-паками по запросам с русской словоизменённостью | ✅ |
| Бот сам вычистил тестовый секрет через memory-инструмент | ✅ |
| Memory-брифинг cron-задачи (поиск по собранному промпту) | ✅ |
| `ask_agent` — синхронный round-trip между профилями | ✅ |
| `assign_task → kanban daemon → spawn → heartbeat → complete` | ✅ |
| Mailbox: `[question]` во время работы воркера → перечитывает тред → `[answer]` | ✅ |
| `message_agent` — мягкая доставка guidance активному воркеру (A2) | ✅ |
| Crash recovery: ремонт orphaned tool_calls + resume | ✅ |

```bash
scripts/run_tests.sh                            # полный suite (CI-parity)
scripts/run_tests.sh tests/agent/test_memory_bus.py
scripts/run_tests.sh tests/agent/test_memory_orchestrator.py
```

---

## Известные ограничения

Форк в активной разработке. Заметные шероховатости (известны, обходные
пути в работе):

- **Telegram:** медленное восстановление после flood-control на длинных
  ответах (работает, но не мгновенно); edge-case'ы голосовых при
  прерывании.
- **WhatsApp:** плагин платформы пока не загружается (нужен node-bridge
  слой). Остальные 12+ платформ работают.

Если вы столкнулись с поведением не из этого списка — откройте issue.

---

## Происхождение и лицензия

Это **форк** [Hermes Agent by Nous Research](https://github.com/NousResearch/hermes-agent),
ответвившийся от upstream в точке `5cc2951` (июнь 2026, линия v0.15.x).
Распространяется под той же лицензией **MIT** (см. [LICENSE](LICENSE)).

Базовый функционал (цикл агента, skills, gateway, TUI, инструменты) — работа
команды Nous Research и контрибьюторов. Система памяти v2
(store/orchestrator/bus), слои мульти-агентной оркестрации (ролевые
делегаты, команды профилей, kanban-координация, A2 mailbox, crash recovery)
и интеграция Codex-провайдера по подписке добавлены этим форком
(Dmitry Antipin).

> **О расхождении с upstream.** После архитектурных изменений (extract
> `gateway/run.py` в mixins, mailbox-подсистема в `kanban_db.py`)
> автоматический merge свежих upstream-релизов больше не имеет смысла —
> upstream развивает inline-архитектуру, этот форк — mixin-подход.
> Upstream-фиксы переносятся точечно (cherry-pick), полный sync — нет.

Базовая документация (установка, CLI, gateway, skills, memory, MCP) — на
[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/);
она применима к форку для всего, кроме описанных выше слоёв.

---

## Ресурсы

- 🏛️ [Upstream Hermes Agent](https://github.com/NousResearch/hermes-agent) — оригинал
- 💬 [Discord Nous Research](https://discord.gg/NousResearch)
- 📚 [Skills Hub](https://agentskills.io)
- 🇬🇧 [English README](README.md)

---

*Built on [Hermes Agent](https://github.com/NousResearch/hermes-agent) by
[Nous Research](https://nousresearch.com). Расширения Architecture 2.0
(система памяти v2, мульти-агентная оркестрация, Codex по подписке)
добавлены этим форком.*
