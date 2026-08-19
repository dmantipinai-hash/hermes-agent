<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent — Architecture 2.0 Fork ☤

[![Upstream](https://img.shields.io/badge/Upstream-Hermes%20Agent-FFD700?style=for-the-badge)](https://github.com/NousResearch/hermes-agent)
[![Docs](https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge)](https://hermes-agent.nousresearch.com/docs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Русская версия](https://img.shields.io/badge/README-Русский-9cf?style=for-the-badge)](README.ru.md)

> A fork of [Hermes Agent](https://github.com/NousResearch/hermes-agent) by
> [Nous Research](https://nousresearch.com) that adds a **cognitive memory
> system** (typed long-term store, per-turn context packs, a memory bus for
> sub-agents and cron) and **multi-agent orchestration** (role-based
> delegation, persistent profile teams, kanban task coordination) on top of
> the base agent. Everything is additive — base behaviour is unchanged.
>
> **Русская версия README:** [README.ru.md](README.ru.md)

---

## Why this fork

Hermes Agent is a self-improving AI agent with a learning loop, memory,
skills, and messaging-gateway support. This fork extends it in two
directions:

1. **Memory that actually persists and composes** — a typed SQLite store as
   the canonical source of truth, human-readable projections, per-turn
   context packs assembled by an intent router + scorer, and a MemoryBus
   that lets sub-agents and cron jobs recall and (provenance-tagged) write
   memories without owning the store.
2. **Several specialist agents under one orchestrator** — role-based
   `delegate_task`, persistent profile teams (`/team`), and long-running
   async work on a SQLite kanban board with worker heartbeats and a
   mailbox protocol.

Also included: **OpenAI Codex via ChatGPT subscription** — device-code OAuth
login with a regular ChatGPT Plus/Pro account, no API key.

### What's added

| Layer | What it does | Surface |
|------|--------------|---------|
| **Memory store v2** | Typed SQLite store (`fact`/`decision`/`constraint`/`pattern`/`preference`), status lifecycle (`active`/`deprecated`/`pinned`), FTS5 recall, contradiction-aware deprecation. `MEMORY.md`/`USER.md` stay as human-readable projections | `memory(action=read/write/...)` |
| **Context packs** | Each turn the store is searched with the user's message (intent-routed, scored, token-budgeted) and records not already in the frozen system prompt are injected into that message's API copy | automatic, `memory.orchestrator.*` |
| **MemoryBus** | One recall/remember facade for delegation and cron consumers. Read-only by construction for them; every write carries `written_by` provenance and supports scoped revert | `agent/memory_bus.py` |
| **Background self-review** | A background review considers memory updates from the recent conversation every N user turns (default 5) | `memory.nudge_interval` |
| **Role-based delegation** | Sub-agents with pre-configured toolsets and system prompts per specialization (`researcher`, `coder`, `reviewer`, `analyst`, `writer`) | `delegate_task(role=...)` |
| **Profile teams** | Persistent agent-profiles (finance, philosophy, product...) with their own model, memory, `SOUL.md` — they don't vanish after a task | `/team create`, `ask_agent`, `assign_task` |
| **Kanban coordination** | Long async tasks on a SQLite board: dispatcher spawns workers, lifecycle with heartbeats, mailbox messaging via comment threads | `kanban_*`, `read_task_thread` |
| **Crash recovery** | Mid-turn crash detection, transcript repair, session resume | `/resume` |
| **Codex subscription auth** | OpenAI Codex Responses API via ChatGPT account OAuth (device code) — no API key | `hermes setup` → OpenAI Codex |

All memory features are **on by default** with zero external services — the
store is embedded SQLite. Pluggable external memory providers (mem0, honcho,
supermemory, ...) remain available via `memory.provider` in `config.yaml`.

---

## How the memory system works

```
conversation turn
      │
      ▼
 memory tool ──write──▶ typed SQLite store (memories/memory.db, canonical)
      │                      │
      │ read                 │ projections
      ▼                      ▼
 orchestrator ──────▶ MEMORY.md / USER.md (human-readable, prompt snapshot)
  intent router
  + scorer        ──▶ context pack → injected into THIS turn's API copy
  + token budget
      ▲
      │ recall/remember (provenance-tagged writes, scoped revert)
      │
 MemoryBus ◀── sub-agents (delegate_task) · cron jobs (briefings)
```

- While memory fits the system-prompt snapshot, the context pack is empty —
  zero overhead, zero behaviour change until memory outgrows the prompt
  budget (default 2500 tokens, max 20 entries).
- Sub-agents and cron jobs never touch the store directly: they talk to the
  bus, which is read-only for them by construction and tags every write
  with its author for scoped revert.
- Cron jobs can request a memory briefing — the bus searches the store with
  the assembled prompt and injects relevant records.

## Two coordination modes

```
SYNC                                  ASYNC
──────────────────────                ─────────────────────
manager: ask_agent("finance",         manager: assign_task("researcher",
  "give me Q3 numbers")                 "deep market analysis, 2h")
  → blocks until reply                 → kanban task, dispatcher spawns
  ← reply lands in context               a worker, manager stays free

                                      later: message_agent with [question]
                                      worker: read_task_thread at checkpoint
                                        → kanban_comment with [answer]
                                      worker: kanban_complete → notifier push
```

A "team agent" is a profile with its own `config.yaml` (model/provider),
`.env` (keys), `state.db` (sessions), `SOUL.md` (persona) and `skills/`,
isolated by `HERMES_HOME` under `~/.hermes/profiles/<name>/`.

---

## Quick Start

### Prerequisites

- **Python 3.11+** (3.12/3.14 also work)
- **uv** — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Git**
- A model credential: API key (OpenAI, Anthropic, Z.AI/GLM, OpenRouter, ...)
  **or** a ChatGPT Plus/Pro subscription (Codex, see below)
- **Node.js 20+** — *optional*, only for the Ink/React TUI (`hermes --tui`)
  and browser tools. The base CLI works without Node.

### Install

```bash
git clone https://github.com/dmantipinai-hash/hermes-agent.git
cd hermes-agent

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[all]"

hermes setup        # interactive wizard: model, keys, platforms
```

Or as a global tool without activating a venv:

```bash
uv tool install --from "git+https://github.com/dmantipinai-hash/hermes-agent.git" "hermes-agent[all]"
```

**Windows (native, no WSL):**

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

The installer handles everything: uv, Python 3.11, Node.js, ripgrep,
ffmpeg, and a portable Git Bash (MinGit — no admin required, isolated
from any system Git install). See the upstream
[Windows guide](https://hermes-agent.nousresearch.com/docs) for details.

**Updating:**

```bash
cd hermes-agent
git pull
source .venv/bin/activate
uv pip install -e ".[all]"      # only needed when dependencies changed
```

Release tags (`v0.16.0`, ...) mark stable points — `git checkout v0.16.0`
for a pinned version.

### Connecting Codex via ChatGPT subscription

No API key needed — the `openai-codex` provider authenticates with your
ChatGPT account (Plus/Pro) via device-code OAuth:

1. `hermes setup` → choose **OpenAI Codex**
2. Open the printed URL, log in with the ChatGPT account, confirm the code
3. Pick a Codex model — done

Re-login later with `hermes auth`. Codex also works as the delegation
provider, so sub-agents can run on the same subscription.

### Telegram gateway

```bash
hermes setup            # store your bot token (Telegram BotFather)
hermes gateway run      # start the gateway in the foreground
```

The `python-telegram-bot` dependency installs lazily on first use. See the
upstream docs for the other 12+ platforms (Discord, Slack, Matrix, ...).

### Toolset configuration (important!)

For the multi-agent tools to be visible to the agent, enable the toolsets in
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

> ⚠️ **Both keys** must list the toolset — `toolsets:` gates individual
> tools via check_fn, `platform_toolsets:` activates the toolset as a whole.

### First multi-agent session

```bash
hermes                                        # start the CLI orchestrator

# In the session:
/team create finance --role researcher        # create the finance profile
/team create writer --role writer             # create the writer profile
/team list                                    # who exists, who is busy

# Async long task:
/team assign finance "analyze the Q3 budget"  # kanban task
kanban daemon --force                         # raise the dispatcher
kanban list                                   # task status
```

The memory system needs no setup — it is on by default. Try telling the
agent a durable fact about yourself, run `/new`, and ask about it again.

---

## What's verified

Live end-to-end tests (2026-08):

| Scenario | Status |
|---|---|
| Memory survives `/new` (decision recalled from a fresh session) | ✅ |
| Context pack recall with Cyrillic-inflected queries | ✅ |
| Bot self-cleaning a leaked test secret via the memory tool | ✅ |
| Cron job memory briefing (store searched with the assembled prompt) | ✅ |
| `ask_agent` synchronous round-trip between profiles | ✅ |
| `assign_task → kanban daemon → spawn → heartbeat → complete` | ✅ |
| Mailbox: `[question]` mid-flight → worker re-reads thread → `[answer]` | ✅ |
| `message_agent` soft guidance delivery to an active worker (A2) | ✅ |
| Crash recovery: orphaned tool_calls repair + resume | ✅ |

```bash
scripts/run_tests.sh                            # full suite (CI parity)
scripts/run_tests.sh tests/agent/test_memory_bus.py
scripts/run_tests.sh tests/agent/test_memory_orchestrator.py
```

---

## Known issues

The fork is under active development. Notable rough edges (documented, with
workarounds in progress):

- **Telegram:** slow recovery after flood-control on long replies (works,
  not instant); voice-message edge cases on interruption.
- **WhatsApp:** platform plugin not loading yet (needs a node-bridge layer).
  The other 12+ platforms work.

If you hit behaviour outside this list, please open an issue.

---

## Provenance & License

This is a **fork** of [Hermes Agent by Nous Research](https://github.com/NousResearch/hermes-agent),
diverged from upstream at `5cc2951` (June 2026, v0.15.x line). Distributed
under the same **MIT** license (see [LICENSE](LICENSE)).

The base functionality (agent loop, skills, gateway, TUI, tools) is the work
of the Nous Research team and contributors. The memory system v2
(store/orchestrator/bus), multi-agent orchestration layers (role delegates,
profile teams, kanban coordination, A2 mailbox, crash recovery) and the
Codex subscription provider integration were added by this fork
(Dmitry Antipin).

> **Divergence note.** After architectural changes (extracting
> `gateway/run.py` into mixins, the mailbox subsystem in `kanban_db.py`),
> automatic merges of fresh upstream releases no longer make sense —
> upstream develops the inline architecture, this fork the mixin approach.
> Upstream fixes are cherry-picked; full syncs are not.

Base documentation (install, CLI, gateway, skills, memory, MCP) lives at
[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)
and applies to this fork for everything except the layers described above.

---

## Resources

- 🏛️ [Upstream Hermes Agent](https://github.com/NousResearch/hermes-agent) — the original
- 💬 [Nous Research Discord](https://discord.gg/NousResearch)
- 📚 [Skills Hub](https://agentskills.io)
- 🇷🇺 [README на русском](README.ru.md)

---

*Built on [Hermes Agent](https://github.com/NousResearch/hermes-agent) by
[Nous Research](https://nousresearch.com). Architecture 2.0 extensions
(memory system v2, multi-agent orchestration, Codex subscription auth)
added by this fork.*
