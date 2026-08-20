---
sidebar_position: 3
title: "Persistent Memory"
description: "How Hermes Agent remembers across sessions — the typed memory store, eviction to cold storage, recall search, aliases, and health reporting"
---

# Persistent Memory

Hermes Agent has bounded, curated memory that persists across sessions: your preferences, projects, environment, and lessons it has learned. This page documents the **typed memory store (v2)** — the SQLite-backed architecture with automatic eviction, full-text recall, decision provenance, synonym aliases, and a health report.

:::note Who should read this
This page is written **for the agent first**. A freshly installed agent reads it to learn how its own memory works and what it must and must not do (see [Operating Rules for Agents](#operating-rules-for-agents)). Humans will find the configuration reference and the day-to-day commands below. If you are setting up a new install, the two sections that matter most are the [architecture overview](#how-it-works-two-tiers) and [aliases](#synonym-aliases-memoryaliases).
:::

## How It Works (Two Tiers)

Memory is a single SQLite database — `~/.hermes/memories/memory.db` — with two tiers of *visibility*, not two stores:

| Tier | What it is | Bound |
|------|-----------|-------|
| **Hot** | A frozen snapshot rendered into the system prompt at session start | `memory.memory_char_limit` (default 2,200 chars) for `memory`, `user_char_limit` (1,375) for `user` |
| **Cold** | Every record ever saved — full-text searchable via `memory(action="read")` and re-injected per turn by the orchestrator when relevant | Unbounded |

Everything lives in the same tables. When the snapshot budget overflows, the least-important entries **drop out of the prompt only** — they stay in the store, fully searchable, and are re-injected into individual turns when your message matches them. Eviction is automatic and lossless.

`MEMORY.md` and `USER.md` in the same directory are **human-readable projections** — rewritten from the database after every change so you can read and git-track your memory. They are derived views: direct edits to them are overwritten on the next write. SQLite is the source of truth.

### Tables in memory.db

| Table | Purpose |
|-------|---------|
| `memories` | Every entry: `type`, `importance`, `status` (`active`/`dormant`/`archived`/`deprecated`/`pinned`), provenance columns, `access_count` |
| `memories_fts` | FTS5 full-text index over entry content |
| `memory_links` | `supersedes` links between decisions (see [provenance](#decision-provenance-supersedes-links)) |
| `memory_recall_log` | Recall audit log feeding `hermes memory report` |
| `memory_aliases` | Cache of the synonym dictionary from `memory.aliases` in config.yaml |

### The frozen snapshot (prompt-cache invariant)

The system-prompt block is captured once at session start and never changes mid-session — this preserves the LLM's prefix cache. Writes made during a session are persisted immediately but appear in the system prompt only next session. Mid-session, the per-turn [context pack](#the-per-turn-context-pack-orchestrator) supplements the snapshot with records your current message matches.

## Operating Rules for Agents

If you are an agent reading this, these rules are your contract with the memory system. They exist because every one of them was violated by a real agent at least once, with real data loss.

1. **Overflow is not a problem. Never "clean" memory.** When a write response shows `evicted_to_cold`, the excess has moved to cold storage automatically — still active, still searchable. Do **not** remove, shorten, or archive-and-delete entries to "free space". The store is unbounded by design.
2. **The `usage` number is the prompt projection, not the store size.** `"98% — 2,167/2,200 chars"` describes what the next session's prompt will contain. A store 3× larger than the limit is healthy.
3. **A changed decision is `deprecate`, not `remove`.** Removing destroys history. Deprecate hides the old entry from retrieval but keeps it for audit — and with the `superseded by:` marker it records the lineage (below).
4. **Read before deciding.** Before saving a decision or constraint, `memory(action="read", query=...)` on the topic. If an active entry contradicts the new one, deprecate it with the marker, then add the new one.
5. **When `old_text` fails, the response lists `current_entries`.** Retry with a short unique substring of the *actual stored text* — not text you remember writing. After three consecutive failures the tool goes terminal for the turn: stop retrying and answer the user; save in a later turn.
6. **Save what will still matter next week** — decisions with reasons, constraints, preferences, environment facts. Not task progress (that's `session_search` territory), not data dumps.

## Memory Tool Actions

```python
memory(action="add", target="memory", content="...", type="decision", importance=0.8)
memory(action="replace", target="memory", old_text="unique substring", content="...", ...)
memory(action="remove", target="memory", old_text="unique substring")
memory(action="deprecate", target="memory", old_text="...", reason="...")
memory(action="read", target="memory", query="vpn сервер")
```

- **add** — insert a typed entry. Duplicate content is rejected softly. If you add a `decision`/`constraint` that resembles an active one, the response carries `related_active` + a hint — review it for conflicts (rule 4).
- **replace** — edit-in-place of the entry matched by `old_text` (substring). Preserves `type`/`importance` unless overridden. Replacing does **not** create links — it's the same entry, reworded.
- **remove** — hard delete. For garbage only.
- **deprecate** — retire a decision/constraint: hidden from retrieval and snapshot, kept for audit with the reason. **To link the successor, end `reason` with `superseded by: <exact substring of the new entry>`** — recall then shows what each decision replaced (see below).
- **read** — full-text search over the *entire* cold store, including entries not in your current prompt. This is the primary "what do I know about X" tool.

Entry types: `fact` (default), `decision` (a choice made — include the reason), `constraint` (a rule to respect), `preference`, `pattern`. `decision` and `constraint` get a `[type]` prefix in the prompt snapshot so the model treats them as binding. `importance` (0..1, default 0.5) orders survival in the snapshot budget; `pinned` status beats importance.

### Search semantics (what `read` matches)

Queries go through a recall-oriented funnel:

1. **Exact phrase** FTS match (whole tokens) — try `"Билайн душит"`.
2. **Whole-query substring** — bridges simple inflection (`персонаж` → `персонажа`).
3. **Term OR-search fallback** — for natural-language queries that share no contiguous phrase with any entry: every word ≥4 chars becomes a truncated prefix stem (`зависимости`/`зависимостей` → `завис*`), and **3-character terms match exactly** (`vpn`, `vps`, `dns`, `kvm` — whole tokens, never prefixes: `vpn` will not match `vpnhub`), plus configured [aliases](#synonym-aliases-memoryaliases).

Results carry `type`, `status`, `importance`, and — when the entry supersedes another — a `supersedes` block with the predecessor's id, date and content preview.

### Decision provenance (`supersedes` links)

When a decision changes, the *trajectory* matters as much as the final state:

```python
memory(action="add", target="memory", type="decision", importance=0.8,
       content="Хостинг: Docker обязателен для всех сервисов")
memory(action="deprecate", target="memory", old_text="не используем Docker",
       reason="пересмотрели:superseded by: Docker обязателен для всех сервисов")
```

Later, any recall that surfaces the new decision also shows:

```
[supersedes: 1f2a3b4c (2026-08-14, «Хостинг: не используем Docker — тяжело для сервера»)]
```

so the agent can say "you changed your mind — the original objection was server load; is that factor addressed?" instead of knowing only the latest state. The fragment after the marker is matched as a substring against active entries; if it matches nothing, the deprecate still succeeds and the link is simply not written (soft degradation). Links are strictly one hop — a predecessor's own predecessor does not ride along.

### No-match feedback

`replace`/`remove`/`deprecate` with an `old_text` that matches nothing return the full `current_entries` list — retry from the actual stored texts. Multiple matches return previews and ask for a more specific substring.

## The Per-Turn Context Pack (Orchestrator)

Each turn, before the model sees your message, the memory orchestrator searches the cold store with the message text (intent-routed, scored by relevance/importance/recency), drops entries already present in the frozen snapshot, and injects the rest as a compact context block attached to that one message. This is how evicted decisions resurface exactly when relevant — no explicit read needed. Packs are token-budgeted (`memory.orchestrator.token_budget`, default 2,500) and never persisted into the transcript.

## Eviction and Importance Demotion

When the snapshot budget overflows on a write:

- the overflow entries leave the **prompt only** — `status` stays `active`, search still finds them;
- each evicted entry's `importance` is lowered to at most half the minimum importance still in the prompt (floor 0.05), so future snapshot ordering is stable and an evicted entry cannot out-rank an in-prompt one. The demotion is monotonic (never raises) and does not touch `updated_at`.

The write response states this explicitly: `usage` shows the projection, `evicted_to_cold` counts the spill, and the note repeats **"eviction is automatic; do NOT remove, shorten or archive entries to free space."**

## Recall Audit and Health Report

Every recall — the orchestrator's auto-pack and your explicit reads — appends one row to `memory_recall_log` (query terms, funnel counts, whether anything reached the turn). Aggregate it:

```bash
hermes memory report             # health digest for the last 7 days
hermes memory report --days 30   # wider window
hermes memory report --prune     # also drop rows older than memory.recall_log.retain_days
```

The report shows hit-rate by channel, empty recalls that *had* candidates (scored out), the **top recurring empty-recall queries** — your best candidates for new [aliases](#synonym-aliases-memoryaliases) — dead weight (active entries never accessed in 30+ days), and the most-accessed entries. A weekly cron of `hermes memory report --prune` delivered to your platform of choice is a cheap, high-signal health loop.

Gate and retention:

```yaml
memory:
  recall_log:
    enabled: true     # one switch for both channels
    retain_days: 90
```

## Synonym Aliases (`memory.aliases`)

Lexical search cannot know that in *your* vocabulary «прокси», «vpn» and «туннель» are one topic. The alias dictionary teaches recall your synonyms — **search-only**: stored entries are never rewritten.

```yaml
# ~/.hermes/config.yaml
memory:
  aliases:
    # keys are query words of 4+ chars; values may be any length
    прокси: [vpn, vless, туннель]
    роутер: [cudy, openwrt, luci]
    сервер: [vps, kvm, хостинг]
    оркестрация: [канбан, доска, диспетчер]
    деньги: [бюджет, финмодель, цена]
    # anchor names → the topic they belong to
    миша: [flowwow]
```

Rules:

- A key expands a **query** word to also match its aliases' terms. Short query terms (3 chars: `vpn`, `vps`, `dns`) are **not** alias-expanded — they already match exactly on their own; aliases apply to the 4+-char stemming path.
- Aliases only ever **add** search terms — an empty dictionary is exactly the no-alias behavior; a wrong alias can add noise to recall, never remove results.
- Takes effect on the next session start (config is read at agent init).
- To grow the dictionary: watch `top_empty_queries` in `hermes memory report` — recurring misses are alias candidates. Hermes will suggest; **you** decide what goes into config.yaml.

## What to Save vs Skip

**Save (proactively):** user preferences and corrections → `user`; decisions with reasons and constraints → `memory` as `type=decision`/`constraint`; environment facts, project conventions, tool quirks, stable lessons → `memory`.

**Skip:** trivial/obvious facts, easily re-discovered information, raw data dumps, session-specific ephemera, task progress (use `session_search`), anything already in context files.

Compact, information-dense entries work best:

```
# Good
User runs macOS 14, Homebrew, Docker Desktop + Podman; zsh + oh-my-zsh; VS Code with Vim keys.
# Good (decision with reason)
[decision] Не используем Docker в проде — 2 ГБ RAM на VPS не тянет (2026-08).
# Bad
User has a project.
```

## Security

Entries are scanned for prompt-injection and exfiltration patterns before entering the store, and the snapshot is sanitized again before entering the system prompt — a poisoned database row cannot inject into the prompt.

## Session Search vs Memory

| Feature | Persistent Memory | Session Search |
|---------|------------------|----------------|
| Capacity | Prompt snapshot is bounded; cold store unbounded | All sessions ever |
| Speed | Instant (snapshot) / ~ms (recall, packs) | ~20ms FTS5 |
| Use case | Standing facts, decisions, preferences | "Did we discuss X last week?" |

See [Session Search](/user-guide/sessions#session-search-tool).

## Learning Journey (`/journey`)

The learning journey is a timeline view of everything Hermes has learned — saved skills and memory entries plotted over time (oldest at top, newest at bottom), with a playable "constellation" scrubber that replays the build-up. The same graph data drives three surfaces:

- **Classic CLI / standalone** — `hermes journey` (aliases: `hermes learning`, `hermes memory-graph`) renders the timeline in the terminal. Flags: `--play` animates the build-up (`--fps` to tune it), `--width`/`--height` override the render size, `--no-color` disables color, and `--json` dumps the raw graph payload.
- **TUI** — `/journey` (aliases: `/learning`, `/memory-graph`) opens the timeline as an overlay.
- **Desktop app** — `/journey` opens the Star Map / memory-graph panel, an interactive visual of the same nodes.

Beyond viewing, the journey is also where you **prune and correct** what Hermes has learned:

| Command | What it does |
|---------|--------------|
| `hermes journey list` | List node ids — skill names and `memory:<source>:<index>` ids for memory chunks. |
| `hermes journey delete <node> [-y]` | Delete a node. Skills are **archived** (restorable), memory chunks are removed. `-y` skips the confirmation. |
| `hermes journey edit <node>` | Open the node's content (a skill's `SKILL.md` or the memory chunk) in `$EDITOR`. |

## Configuration Reference

```yaml
# ~/.hermes/config.yaml
memory:
  memory_enabled: true            # memory tool + snapshot injection
  user_profile_enabled: true      # the 'user' target
  store_v2: true                  # typed SQLite store (false = legacy flat files)
  memory_char_limit: 2200         # hot-tier budget, memory target (~800 tokens)
  user_char_limit: 1375           # hot-tier budget, user target (~500 tokens)
  write_approval: false           # true = stage every write for your approval
  aliases: {}                     # synonym dictionary (see above)
  recall_log:
    enabled: true                 # recall audit log
    retain_days: 90
  orchestrator:                   # per-turn context packs
    enabled: true
    token_budget: 2500
    max_entries: 20
```

Other memory commands:

```bash
hermes memory report [--days N] [--prune]   # health digest (see above)
hermes memory reset [--target memory|user]  # erase the built-in stores
hermes memory setup / status / off          # external providers
```

## Controlling memory writes (`write_approval`)

By default the agent saves memory freely — including from the background self-improvement review that runs after a turn. If you'd rather approve saves first, set `memory.write_approval: true`. It's a simple on/off gate applied to **both** foreground turns and the background review:

| `write_approval` | Behaviour |
|------------------|-----------|
| `false` (default) | Write freely — the gate is off (the pre-gate behaviour). |
| `true` | Require approval before anything is saved. In the interactive CLI, foreground writes prompt you inline (entries are small enough to read in full). Everywhere else — messaging platforms, scripts, and the background self-improvement review — writes are **staged** for review with `/memory pending`. |

> To turn memory off entirely (not just gate it), set `memory_enabled: false`.

Review staged writes from the CLI or any messaging platform:

```
/memory pending             # list staged memory writes (auto ones tagged [auto])
/memory approve <id>        # apply one (or 'all')
/memory reject <id>         # drop one (or 'all')
/memory approval on         # turn the gate on (or 'off') and persist it
```

This is the answer to "the agent saved a wrong assumption about me": set `write_approval: true`, and every save — especially the unprompted background ones — waits for your yes/no before it ever enters your profile.

## Background review notifications (`display.memory_notifications`)

After a turn, the background self-improvement review may quietly save a memory or update a skill. By default it surfaces a short `💾 Memory updated` line in chat so you know it happened. Control how chatty that is:

```yaml
display:
  memory_notifications: on    # off | on (default) | verbose
```

| Value | Behaviour |
|-------|-----------|
| `off` | No chat notification. The review still runs and still writes — you just don't see a line for it. |
| `on` (default) | Generic line, e.g. `💾 Memory updated`, `💾 Skill 'foo' patched`. |
| `verbose` | Includes a compact preview of what changed, e.g. `💾 Memory ➕ User prefers terse replies` or a `"old" → "new"` skill diff snippet. |

> This only governs the **gateway** chat notification. The review itself, and writes to your memory/skill stores, are unaffected. Set it per-platform via `display.platforms.<platform>.memory_notifications`.

## Running the review on a cheaper model (`auxiliary.background_review`)

The review runs on your **main chat model** by default, replaying the conversation — which is already warm in the prompt cache, so it's cheap cache reads. On an expensive main model you can run the review on a cheaper model instead:

```yaml
auxiliary:
  background_review:
    provider: openrouter
    model: google/gemini-3-flash-preview   # auto (default) = main chat model
```

When you point it at a model **different** from your main one, the review runs there for substantially lower cost (~3–5× in benchmarks), replaying a compact **digest** of the conversation (recent turns verbatim + a summary of older ones). Capture holds: in testing, memory capture was identical and skill capture near-identical to the main-model review.

### Disabling automatic reviews (`enabled`)

```yaml
auxiliary:
  background_review:
    enabled: true              # false = skip automatic post-turn forks (manual /refine still works)
```

Fork usage is persisted in `session_model_usage` with `task='background_review'` and a completion line is written to `agent.log`.

## Controlling skill writes (`skills.write_approval`)

Skills use the same on/off gate, but the review UX differs because a `SKILL.md` is far too large to read in a chat bubble:

```yaml
skills:
  write_approval: false     # false = write freely (default) | true = require approval
```

When `write_approval: true`, skill writes (create / edit / patch / write_file / delete) always **stage** regardless of origin:

```
/skills pending             # list staged skill writes + a one-line gist each
/skills diff <id>           # full unified diff (best viewed in CLI or dashboard)
/skills approve <id>        # apply it (or 'all')
/skills reject <id>         # drop it (or 'all')
/skills approval on         # turn the gate on (or 'off') and persist it
```

Full details in [Gating agent skill writes](/user-guide/features/skills#gating-agent-skill-writes-skillswrite_approval).

## External Memory Providers

For deeper, persistent memory that goes beyond the built-in store, Hermes ships with external memory provider plugins — including Honcho, OpenViking, Mem0, Hindsight, Holographic, RetainDB, and ByteRover.

External providers run **alongside** built-in memory (never replacing it) and add capabilities like knowledge graphs, semantic search, automatic fact extraction, and cross-session user modeling.

```bash
hermes memory setup      # pick a provider and configure it
hermes memory status     # check what's active
```

See the [Memory Providers](./memory-providers.md) guide for full details on each provider, setup instructions, and comparison.

:::caution One agent per Hermes home
Don't point two agent processes at the same Hermes home directory. Memory writes are automatic and load back into the system prompt at session start, so two writers sharing one home will compound each other's entries into state neither of them (nor you) authored. Memory is scoped per [profile](/user-guide/profiles) by design — give a second agent its own profile, and if they need shared memory, use an [external memory provider](/user-guide/features/memory-providers) instead.
:::
