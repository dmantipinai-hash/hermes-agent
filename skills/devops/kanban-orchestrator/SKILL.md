---
name: kanban-orchestrator
description: Kanban routing playbook. Decompose, delegate, verify.
version: 3.1.0
author: Dmitry + Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [kanban, multi-agent, orchestration, routing, codex, pitfalls]
    related_skills: [codex, hermes-agent]
---

# Kanban Orchestrator — Decomposition Playbook

> The **core worker lifecycle** (including the `kanban_create` fan-out pattern and the "decompose, don't execute" rule) is auto-injected into every kanban process via the `KANBAN_GUIDANCE` system-prompt block. This skill is the deeper playbook when you're an orchestrator profile whose whole job is routing. It absorbs the former `kanban-worker` and `kanban-codex-lane` skills.

## Profiles are user-configured — not a fixed roster

Hermes setups vary widely. Some users run a single profile that does everything; some run a small fleet (`docker-worker`, `cron-worker`); some run a curated specialist team they've named themselves. There is **no default specialist roster** — the orchestrator skill does not know what profiles exist on this machine.

Before fanning out, you must ground the decomposition in the profiles that actually exist. The dispatcher silently fails to spawn unknown assignee names — it doesn't autocorrect, doesn't suggest, doesn't fall back. So a card assigned to `researcher` on a setup that only has `docker-worker` just sits in `ready` forever.

**Step 0: discover available profiles before planning.**

> If no profiles exist yet (or you need to create new specialists for the task),
> see `references/multi-agent-profile-setup.md` for the full profile creation
> workflow — create, configure model, set personality, wire up gateway, verify.

Use one of these:

- `hermes profile list` — prints the table of profiles configured on this machine. Run it through your terminal tool if you have one; otherwise ask the user.
- `kanban_list(assignee="<some-name>")` — sanity-check a single name. Returns an empty list (rather than an error) for an unknown assignee, so this only confirms a name you're already considering.
- **Just ask the user.** "What profiles do you have set up?" is a fine first turn when the goal needs more than one specialist.

Cache the result in your working memory for the rest of the conversation. Re-asking every turn wastes a tool call.

## Cross-profile delegation for context management

When your current profile runs out of context window (40+ minute session, "max tokens exceeded", or sluggish responses), delegate to a secondary profile with larger context capacity. This is a common pattern: small-context daily-use profile (fast, cheap) → large-context reserve profile (slow, expensive, but handles big jobs).

**Setup checklist before delegating:**

1. **Verify the target profile exists and has enough context:** Run `hermes profile list` to confirm the profile and its model context length.
2. **Check MCP configuration:** Secondary profiles often lack MCP servers (web search, etc.) that the default profile has. Copy the `mcp_servers` block from `~/.hermes/config.yaml` to `~/.hermes/profiles/<name>/config.yaml` if needed.

   ⚠️ **CRITICAL — YAML duplicate key pitfall:** If the profile's config.yaml already has an `mcp_servers:` block (e.g. for cua-driver), do NOT append a second `mcp_servers:` block. YAML silently uses the LAST occurrence of a duplicate top-level key — the first block (your web-search-prime) gets completely shadowed and never loads. The worker falls back to browser/computer_use, which is 10x slower and more expensive. **Always merge new servers into the existing single `mcp_servers:` block.**

   How to verify before dispatching:
   ```bash
   # Count mcp_servers blocks — must be exactly 1
   grep -c "^mcp_servers:" ~/.hermes/profiles/<name>/config.yaml
   ```

3. **Resolve gateway conflicts:** If both profiles use the same Telegram bot token, the second profile's gateway will crash. Fix by appending `TELEGRAM_BOT_TOKEN=` (empty override) to the secondary profile's `.env` file — the dispatcher handles Kanban work; Telegram integration isn't needed for delegated tasks.
4. **Start or RESTART the target gateway:** If the target profile's gateway isn't running (`hermes profile list` shows "stopped"), start it. If you just edited its `config.yaml` (e.g. added MCP servers), you MUST restart the gateway for changes to take effect — the running process keeps the old config.
   ```bash
   # Find and kill the gateway PID — systemd auto-restarts with new config
   GATEWAY_PID=$(ps aux | grep "profile <name>" | grep gateway | grep -v grep | awk '{print $2}')
   kill $GATEWAY_PID
   # Verify restart within ~5s
   sleep 5 && ps aux | grep "profile <name>" | grep gateway | grep -v grep
   ```
5. **Verify MCP loaded:** After the gateway restarts, check the first heartbeat note. If it mentions "Safari", "browser", or "computer_use" when you expected MCP web search — the MCP server didn't load. Re-check for duplicate `mcp_servers:` keys.

**Workspace sharing for delegated tasks:**

When delegating to a profile that needs access to files, use `dir:/absolute/path/to/workspace` in the Kanban task. This points the worker to an existing directory containing all context files, templates, and output locations. Example:

```python
kanban_create(
    title="Build prototype",
    assignee="<large-context-profile>",  # secondary profile with a big context window
    body="Read TASK-brief.md in workspace root. Output to website/ folder.",
    workspace="dir:/home/<you>/projects/<project>",
    goal_mode=True,
    goal_max_turns=25,
)
```

The target profile doesn't need Telegram for Kanban work — the dispatcher picks up the task from the SQLite database. The primary gateway notifies on completion.

**Brief file pattern (recommended for complex tasks):**

For multi-stage or research-heavy tasks, don't cram everything into the card body. Write a detailed `TASK-<name>.md` file in the shared workspace root and point the card to it:

```
card body: "ПОЛНЫЙ БРИФ: прочитай TASK-research.md в корне workspace.
            ЭТАП 1: ... ЭТАП 2: ... ЭТАП 3: ...
            Результаты: file1.md + file2.md + file3.md."
```

Benefits: the brief file survives context resets, can be versioned, and the worker reads it in full with its large context window. The card body stays a concise routing instruction.

**Dispatching with a non-default model (no gateway restart needed):**

When the user says "launch <profile> with DeepSeek" or wants a specific model for a research task, you can patch the profile's `config.yaml` directly — the kanban dispatcher spawns a **fresh worker process per task**, which reads config at launch time. Unlike gateway-level model changes (which require a `/restart`), kanban tasks pick up the new model automatically on the next dispatch tick (~60s).

Workflow:
1. Patch `~/.hermes/profiles/<name>/config.yaml` → change `model.default`, `model.provider`, `model.context_length` (see `hermes-provider-switching` skill)
2. Create the kanban task with `goal_mode=True`, `workspace_kind="dir"`, `workspace_path=<project>` (see "Goal-mode cards" section)
3. Old model stays in `fallback_providers` for rollback — no key deletion needed
4. The dispatcher picks up the task within ~60s and the worker process starts with the new model

This is safe: the profile's **gateway** keeps running on whatever model it booted with, but each **kanban worker** is a fresh process that reads config at spawn time.

**MCP self-test pattern:**

When a task requires MCP tools (web search, database, etc.), include a self-diagnostic instruction in the card body:

```
"If MCP web-search-prime не работает — заблокируй задачу с причиной."
```

The worker either uses the MCP tool or blocks the task with a clear reason — no silent failures. Check for `blocked` status before assuming the task is running.

**When to delegate vs. just continue:**

- Delegate: Session running 30+ minutes, context window exhausted, multi-step research/analysis/implementation task, need web search or file access that requires MCP.
- Continue: Quick one-off task (5-10 minutes), simple response, no new files needed, model still responsive.

**User delegation calibration (user direction, 2026-06):**

The threshold for delegation should be **low, not high**: "if the task is small (a few edits, ~10 tokens) — do it yourself. If it's labor-intensive — delegate via Kanban. **Don't hesitate, it's cheap.**"

Decision heuristic:
- **Token-cost check, not time check:** if the task is "a few patches" → do it inline. If it's "multi-file generation, research + synthesis, or anything that'll eat context" → delegate.
- **Cost is NOT a blocker:** the secondary profile (~$0.01/task) is negligible. Do not spend turns deliberating whether delegation is "worth it" — if the task is non-trivial, delegate.
- **Context exhaustion is preventable:** the previous session hung for 40 minutes because the orchestrator tried to do everything in one context window. Delegate *before* the window fills, not after. See `hermes-agent` → `references/model-selection-policy.md` for the profile roster and cost details.

**Enforcing TDD discipline in delegated development tasks (2026-08-08):**

When delegating a multi-task development effort (e.g., "implement Tasks 5-7 from handoff doc") to a sub-agent via background `chat -q`, the agent will **jump ahead to the next task before finishing the current one** — writing untested code, noting "tests don't pass yet," and moving on. This produces piles of broken code.

The fix is in the **prompt construction**, not the tooling. Every delegation prompt for development work must include explicit gates:

```
ПРАВИЛО TDD (строго):
- Таск считается ЗАКОНЧЕННЫМ только когда ВСЕ его тесты ЗЕЛЁНЫЕ.
- НЕ переходи к Task N+1 пока Task N полностью не пройден (0 падающих тестов).
- Если тест падает — чини реализацию или тесты, запускай снова, пока все не пройдут.
- НЕ плоди код вперёд. Пустой/непротестированный код не нужен.
```

Also include: "Работай до конца. НЕ останавливайся для подтверждений между этапами." — otherwise the agent will pause after each step asking for permission, wasting turns.

**IMPORTANT — prompt gates alone are insufficient for multi-task TDD work via `chat -q`.** Even with the rules above in the prompt, Kimi K3 (and likely other models) still jumped ahead after a single `chat -q` invocation, wrote untested code for Task 6, and stopped to "report." The user had to manually re-launch 3 times before switching to `kanban create --goal`. For multi-task development work, **always use goal-mode kanban dispatch** — see "The background `chat -q` dispatch pattern" section above for the decision rule.

**Model selection for delegated coding work (2026-08-08, reinforced):**

Not all models are equal for delegated development, even when they're fast and produce polished-looking output. Observed reliability tiers for multi-task TDD coding:

- **GLM-5.2 / DeepSeek V4 Pro** — methodical, reads specs, writes tests that verify real behavior, fixes failing tests before moving on. Good for architecture-sensitive work.
- **Kimi K3** — fast (6-14s/call), great prompt-cache hit rate (70-98%), but produces **artifacts that look complete without being correct**: tests that pass but don't test real behavior, code that doesn't match specs, import path mismatches between test and implementation files. In a live A2 mailbox implementation, Kimi K3's entire output had to be rewritten by GLM-5.2. Useful for prototyping, boilerplate, straightforward refactors, AND **deep research tasks** (large context: 256K, up to 250 sub-agents on Moonshot API). For research — literature review, solution comparison, architecture landscape mapping — Kimi K3 is strong: it holds many documents in context simultaneously and can synthesize across them. **Do NOT use for agent architecture implementation, TDD workflows, or any task where "looks done" ≠ "is done."** Research output is text that the orchestrator can review; code output must actually run.

Also: Kimi K3 has aggressive rate limits (500K TPM, 1.5M TPD on Moonshot API). Multiple concurrent sessions or gateway + one-shot workers can hit the ceiling fast, triggering auto-fallback. Check the provider's rate limits before launching parallel workers on the same key.

**The background `chat -q` dispatch pattern** — ONE-SHOT only, NOT for multi-task work:

When you need a single-shot delegation, use a background terminal process:

```bash
cd /path/to/project && \
HERMES_HOME=~/.hermes/profiles/<profile> \
/path/to/venv/bin/python -m hermes_cli.main \
  -p <profile> --yolo chat -q "$(cat /tmp/prompt.txt)" 2>&1
```

Run with `terminal(background=true, notify_on_complete=true)` — you get auto-notified on completion or crash.

**⚠️ CRITICAL limitation — do NOT use `chat -q` for multi-task work.** Even with "don't stop for confirmations" and explicit TDD gates in the prompt, `chat -q` agents:
1. Treat each invocation as a separate session — no continuity between tasks
2. Stop after each logical unit to "report and wait for confirmation"
3. Jump ahead to the next task before finishing the current one (broken code piles up)
4. You end up manually re-launching for each continuation — wasting turns

**For multi-task work, use `kanban create --goal --goal-max-turns N` instead** (see "Goal-mode cards" section below). The goal-loop solves all of the above:
- Judge checks after each turn against acceptance criteria — worker can't fake completion
- Worker keeps going in the **same session** (full context retained)
- Only stops when judge agrees the work is complete (or budget exhausts → blocked for human review)
- Protocol compliance enforced — worker must call `kanban_complete` or task stays open

**Decision rule:**
- Single self-contained prompt, no board tracking needed → `chat -q`
- Anything with "Tasks 5-7" or multiple sequential steps → `kanban create --goal`

**Pitfall:** Don't assume secondary profiles have the same MCP servers as default. Always check `config.yaml` before delegating tasks that require internet access, database queries, or other MCP capabilities.

**Pitfall — protocol violation on `chat -q` tasks that were also kanban-dispatched:** When a kanban task (runs #28, #29) is dispatched to a worker and the worker exits cleanly (rc=0) without calling `kanban_complete` or `kanban_block`, the dispatcher flags it as a **protocol violation** → after `max-retries` consecutive violations the task goes to `blocked`. This happens when a worker finishes its response and simply ends its turn. **Fix:** create a NEW task (don't reuse the blocked one) with `--goal --goal-max-turns N` flags. The goal-loop judge prevents premature exit — the worker keeps running until the judge agrees it's done.

**CLI recipe — creating a goal-mode task via terminal** (when you can't use `kanban_create` tool):

```bash
cd /path/to/project && \
HERMES_HOME=~/.hermes/profiles/<name> \
/path/to/venv/bin/python -m hermes_cli.main \
  -p <name> kanban create \
  --assignee <assignee> \
  --workspace "dir:/path/to/project" \
  --max-retries 3 \
  --max-runtime 2h \
  --goal \
  --goal-max-turns 30 \
  --created-by <your-name> \
  --body "$(cat /tmp/task-body.txt)" \
  "Task title here" \
  --json
```

**CRITICAL — `delegate_task` subagents ≠ Kanban delegation to a dedicated profile (2026-07-02):**

When the user asks for deep research with large context and names a specific profile (or says "kanban") — **create Kanban tasks assigned to that profile**, do NOT substitute with `delegate_task`. The two are fundamentally different:

- `delegate_task` spawns subagents on the **same profile, same model, same context window** (the orchestrator's own model, limited context). Subagents also time out at 600s and cannot access other profiles' MCP servers.
- Kanban delegation dispatches to **the named profile** (large context, its own MCP servers). The task survives context resets and has several times the reasoning budget.

Field case (2026-07-02, video-editing research): two `delegate_task` subagents ran; one timed out at 600s with 28 API calls wasted. The kanban route to a 1M-context profile was the right tool. The delegation calibration section already says "don't hesitate, it's cheap" — the missing piece was: **when the user explicitly names a profile, route through Kanban, not through `delegate_task`.**

**Note on A2A (upstream v0.16+):** Hermes now has an Agent-to-Agent protocol (SSE streaming, push notifications, anti-loop, `orchestrate` primitive). This is a third coordination channel alongside delegate_task (sync) and Kanban (async). It's a plugin and may need enablement. Before building custom inter-agent communication, check if A2A already covers the use case. See `hermes-agent-development` → `references/upstream-v0.16-v0.20-changes.md`.

## When to use the board (vs. just doing the work)

Create Kanban tasks when any of these are true:

1. **Multiple specialists are needed.** Research + analysis + writing is three profiles.
2. **The work should survive a crash or restart.** Long-running, recurring, or important.
3. **The user might want to interject.** Human-in-the-loop at any step.
4. **Multiple subtasks can run in parallel.** Fan-out for speed.
5. **Review / iteration is expected.** A reviewer profile loops on drafter output.
6. **The audit trail matters.** Board rows persist in SQLite forever.

If *none* of those apply — it's a small one-shot reasoning task — use `delegate_task` instead or answer the user directly.

## The anti-temptation rules

Your job description says "route, don't execute." The rules that enforce that:

- **Do not execute the work yourself.** Your restricted toolset usually doesn't even include terminal/file/code/web for implementation. If you find yourself "just fixing this quickly" — stop and create a task for the right specialist.
- **For any concrete task, create a Kanban task and assign it.** Every single time.
- **Split multi-lane requests before creating cards.** A user prompt can contain several independent workstreams. Extract those lanes first, then create one card per lane instead of bundling unrelated work into a single implementer card.
- **Run independent lanes in parallel.** If two cards do not need each other's output, leave them unlinked so the dispatcher can fan them out. Link only true data dependencies.
- **Never create dependent work as independent ready cards.** If a card must wait for another card, pass `parents=[...]` in the original `kanban_create` call. Do not create it first and link it later, and do not rely on prose like "wait for T1" inside the body.
- **If no specialist fits the available profiles, ask the user which profile to create or which existing profile to use.** Do not invent profile names; the dispatcher will silently drop unknown assignees.
- **Decompose, route, and summarize — that's the whole job.**

## Decomposition playbook

### Step 1 — Understand the goal

Ask clarifying questions if the goal is ambiguous. Cheap to ask; expensive to spawn the wrong fleet.

### Step 2 — Sketch the task graph

Before creating anything, draft the graph out loud (in your response to the user). Treat every concrete workstream as a candidate card:

1. Extract the lanes from the request.
2. Map each lane to one of the profiles you discovered in Step 0. If a lane doesn't fit any existing profile, ask the user which to use or create.
3. Decide whether each lane is independent or gated by another lane.
4. Create independent lanes as parallel cards with no parent links.
5. Create synthesis/review/integration cards with parent links to the lanes they depend on. A child created with unfinished parents starts in `todo`; the dispatcher promotes it to `ready` only after every parent is done.

Examples of prompts that should fan out (using placeholder profile names — substitute whatever exists on the user's setup):

- "Build an app" → one card to a design-oriented profile for product/UI direction, one or two cards to engineering profiles for implementation, plus a later integration/review card if the user has a reviewer profile.
- "Fix blockers and check model variants" → one implementation card for the blocker fixes plus one discovery/research card for config/source verification. A final reviewer card can depend on both.
- "Research docs and implement" → a docs-research card can run in parallel with a codebase-discovery card; implementation waits only if it truly needs those findings.
- "Analyze this screenshot and find the related code" → one card to a vision-capable profile for the visual analysis while another searches the codebase.

Words like "also," "finally," or "and" do not automatically imply a dependency. They often mean "make sure this is covered before reporting back." Only link tasks when one card cannot start until another card's output exists.

Show the graph to the user before creating cards. Let them correct it — including which actual profile name should own each lane.

### Step 3 — Create tasks and link

Use the profile names from Step 0. The example below uses placeholders `<profile-A>`, `<profile-B>`, `<profile-C>` — replace them with what the user actually has.

```python
t1 = kanban_create(
    title="research: Postgres cost vs current",
    assignee="<profile-A>",  # whichever profile handles research on this setup
    body="Compare estimated infrastructure costs, migration costs, and ongoing ops costs over a 3-year window. Sources: AWS/GCP pricing, team time estimates, current Postgres bills from peers.",
    tenant=os.environ.get("HERMES_TENANT"),
)["task_id"]

t2 = kanban_create(
    title="research: Postgres performance vs current",
    assignee="<profile-A>",  # same profile, run in parallel
    body="Compare query latency, throughput, and scaling characteristics at our expected data volume (~500GB, 10k QPS peak). Sources: benchmark papers, public case studies, pgbench results if easy.",
)["task_id"]

t3 = kanban_create(
    title="synthesize migration recommendation",
    assignee="<profile-B>",  # whichever profile does synthesis/analysis
    body="Read the findings from T1 (cost) and T2 (performance). Produce a 1-page recommendation with explicit trade-offs and a go/no-go call.",
    parents=[t1, t2],
)["task_id"]

t4 = kanban_create(
    title="draft decision memo",
    assignee="<profile-C>",  # whichever profile drafts user-facing prose
    body="Turn the analyst's recommendation into a 2-page memo for the CTO. Match the tone of previous decision memos in the team's knowledge base.",
    parents=[t3],
)["task_id"]
```

`parents=[...]` gates promotion — children stay in `todo` until every parent reaches `done`, then auto-promote to `ready`. No manual coordination needed; the dispatcher and dependency engine handle it.

If the task graph has dependencies, create the parent cards first, capture their returned ids, and include those ids in the child card's `parents` list during the child `kanban_create` call. Avoid creating all cards in parallel and linking them afterward; that creates a window where the dispatcher can claim a child before its inputs exist.

### Step 4 — Complete your own task

If you were spawned as a task yourself (e.g. a planner profile was assigned `T0: "investigate Postgres migration"`), mark it done with a summary of what you created:

```python
kanban_complete(
    summary="decomposed into T1-T4: 2 research lanes in parallel, 1 synthesis on their outputs, 1 prose draft on the recommendation",
    metadata={
        "task_graph": {
            "T1": {"assignee": "<profile-A>", "parents": []},
            "T2": {"assignee": "<profile-A>", "parents": []},
            "T3": {"assignee": "<profile-B>", "parents": ["T1", "T2"]},
            "T4": {"assignee": "<profile-C>", "parents": ["T3"]},
        },
    },
)
```

### Step 5 — Report back to the user

Tell them what you created in plain prose, naming the actual profiles you used:

> I've queued 4 tasks:
> - **T1** (`<profile-A>`): cost comparison
> - **T2** (`<profile-A>`): performance comparison, in parallel with T1
> - **T3** (`<profile-B>`): synthesizes T1 + T2 into a recommendation
> - **T4** (`<profile-C>`): turns T3 into a CTO memo
>
> The dispatcher will pick up T1 and T2 now. T3 starts when both finish. You'll get a gateway ping when T4 completes. Use the dashboard or `hermes kanban tail <id>` to follow along.

## Common patterns

**Fan-out + fan-in (research → synthesize):** N research-style cards with no parents, one synthesis card with all of them as parents.

**Parallel implementation + validation:** one implementer card makes the change while one explorer/researcher card verifies config, docs, or source mapping. A reviewer card can depend on both. Do not make the implementer own unrelated verification just because the user mentioned both in one sentence.

**Pipeline with gates:** `planner → implementer → reviewer`. Each stage's `parents=[previous_task]`. Reviewer blocks or completes; if reviewer blocks, the operator unblocks with feedback and respawns.

**Same-profile queue:** N tasks, all assigned to the same profile, no dependencies between them. Dispatcher serializes — that profile processes them in priority order, accumulating experience in its own memory.

**Human-in-the-loop:** Any task can `kanban_block()` to wait for input. Dispatcher respawns after `/unblock`. The comment thread carries the full context.

**Multi-model research iteration (proven Aug 2026):** Run the same research task with different models to get complementary results — the second model gets a larger context window and explicit instructions to build on (not repeat) the first model's findings.

Workflow:
1. **Run iteration 1** with the cheaper/faster model (e.g., DeepSeek, 64K context). Create a kanban task, let it complete.
2. **Rename the output** to preserve it: `Отчет.md` → `Отчет (DeepSeek).md`. This prevents overwrite and signals lineage.
3. **Switch the profile model** to the stronger research model (e.g., Kimi K3, 256K context). Patch `~/.hermes/profiles/<name>/config.yaml` — the dispatcher spawns a fresh worker that reads the new config at launch.
4. **Create iteration 2 task** with explicit framing: *"Read the previous report (Отчет DeepSeek.md). Find NEW solutions and patterns the first iteration missed. Do NOT repeat findings that are already covered."*
5. **Set up completion notification** via cron polling (see `references/kanban-completion-notification.md`).

Key detail: the second task must **explicitly name the previous output file** and say "extend, don't repeat." Without this, the new model wastes context re-deriving what the first already found. The 256K context window lets Kimi K3 hold both the previous report AND new research simultaneously.

Model pairing strategy: **DeepSeek (fast, cheap, 64K) → Kimi K3 (slow, expensive, 256K)**. DeepSeek for breadth; Kimi K3 for depth. This mirrors the general principle: cheap model first, expensive model second — the expensive model benefits from the cheap model's output as input.

## Completion notification for async tasks

Kanban workers run in separate processes — the orchestrator can't use `notify_on_complete` to know when they finish. Use a **cron polling job** that checks the task's `status` in the SQLite DB every 15 minutes and delivers a message when `done` or `blocked`. Marker-file deduplication prevents re-delivery.

See `references/kanban-completion-notification.md` for the full script template and `no_agent=True` cron semantics.

## Orchestrator-side monitoring discipline (anti-pattern: sleep-loop waiting)

Proven failure (2026-08-25, a channel-research task): the orchestrator waited for a web-research worker with repeated `sleep 300` terminal calls — 20+ minutes of dead turns while the worker fought search-engine captchas it was never told to give up on. Total runtime ~45 min when the core analysis was done at ~25 min.

Rules for any fan-out with async workers:

1. **Set `max_runtime_seconds` on every research card** (20–30 min is enough for read-and-writeup tasks). The dispatcher then re-queues or kills the runaway itself.
2. **Fail-fast clause in every card body that needs network/search:** «Если поиск недоступен (капчи, блокировки, таймауты) — максимум 2 попытки на источник, затем заблокируй задачу с причиной. НЕ мучайся часами.» Combine with the MCP self-test pattern above.
3. **Steer, don't wait.** At the 15-minute mark with no artifact: `message_agent(guidance)` — «статус? если застрял на блокировках — зафиксируй что нашёл и блокируй задачу». At 25 min: stop/reclaim.
4. **Never sleep-loop.** Between status checks, do useful orchestrator work (verify artifacts already delivered, read samples, prep synthesis). If there is nothing to do, end the turn and let the completion notification (cron poller) wake you.
5. **Timing rule of thumb:** if a phase's remaining value is only «wait for agent X», the whole phase should cost zero orchestrator turns.

## Pitfalls

**Inventing profile names that don't exist.** The dispatcher silently fails to spawn unknown assignees — the card just sits in `ready` forever. Always assign to a profile from your Step 0 discovery; ask the user if you're unsure. Sub-case (proven 2026-09): assigning to your OWN persona name (`loki`) when the system profile is `default` — the phantom card sits in `ready`, then gets PROMOTED again later when its parents complete, resurfacing as clutter. If you create a phantom: create a corrected card, leave a comment on the phantom, archive it via `hermes kanban archive <id>`. Note `kanban_create` has `idempotency_key` for retry-safe replacements.

**Semantic parent/child deadlock (proven 2026-09).** Do NOT create a source card as a CHILD of the consumer card while also instructing the consumer to include the child's output as an input. The board is acyclic mechanically, but the cycle lives in the semantics: child waits for parent's completion, parent waits for child's result. Correct shape: parallel input cards (no links) + a separate joiner card with `parents=[all inputs]`. Never make «ребёнок = источник входа родителя». If the deadlock already exists: remove the wrong edge with `kanban_unlink(parent_id=..., child_id=...)` — the child's promotion eligibility is re-evaluated immediately. Alternatively resolve semantically: complete the parent with the inputs it has, let the child promote and run, then the orchestrator collects the final synthesis.

**Skills are isolated per profile (proven 2026-09).** A project skill loaded in your profile is INVISIBLE to kanban workers — profiles have their own skills/ directories. A card cannot say «см. скилл X». Fix in card bodies: point to concrete FILE PATHS in the project folder (e.g. `~/Desktop/<project>/04-механика.md`) instead of skill names, and instruct the worker to `ls *.md` the project folder first — project docs move faster than your skill snapshot.

**Bundling independent lanes into one card.** If the user asks for two independent outcomes, create two cards. Example: "fix blockers and check model variants" is not one fixer task; create a fixer/engineer card for the fixes and an explorer/researcher card for the variant check, then optionally gate review on both.

**Over-linking because of wording.** "Finally check X" may still be parallel with implementation if X is static config, docs, or source discovery. Link it after implementation only when the check depends on the implementation result.

**Forgetting dependency links.** If the task graph says `research -> implement -> review`, do not create all tasks as independent ready cards. Use parent links so implement/review cannot run before their inputs exist.

**Reassignment vs. new task.** If a reviewer blocks with "needs changes," create a NEW task linked from the reviewer's task — don't re-run the same task with a stern look. The new task is assigned to the original implementer profile.

**Argument order for links.** `kanban_link(parent_id=..., child_id=...)` — parent first. Mixing them up demotes the wrong task to `todo`.

**Don't pre-create the whole graph if the shape depends on intermediate findings.** If T3's structure depends on what T1 and T2 find, let T3 exist as a "synthesize findings" task whose own first step is to read parent handoffs and plan the rest. Orchestrators can spawn orchestrators.

**Tenant inheritance.** If `HERMES_TENANT` is set in your env, pass `tenant=os.environ.get("HERMES_TENANT")` on every `kanban_create` call so child tasks stay in the same namespace.

## Goal-mode cards (persistent workers)

By default a dispatched worker gets **one shot** at its card: it does its work, calls `kanban_complete`/`kanban_block`, and exits. For open-ended cards where one turn rarely finishes the job, pass `goal_mode=True` to wrap that worker in a Ralph-style goal loop — the same engine behind the `/goal` slash command:

```python
kanban_create(
    title="Translate the full docs site to French",
    body="Acceptance: every page translated, no English left, links intact.",
    assignee="<translator-profile>",
    goal_mode=True,        # judge re-checks the card after each turn
    goal_max_turns=15,     # optional budget (default 20)
)["task_id"]
```

How it behaves:
- After each worker turn, an auxiliary judge evaluates the worker's response against the card's **title + body** (treated as the acceptance criteria).
- Not done + budget remains → the worker keeps going **in the same session** (full context retained — not a fresh respawn).
- Worker calls `kanban_complete`/`kanban_block` itself → loop stops, normal lifecycle.
- Budget exhausted without completion → the card is **blocked** for human review (sticky), never a silent exit.

When to use it: long, multi-step, or "keep going until X is true" cards. When NOT to: cheap one-shot cards (translation of a single string, a quick lookup) — the judge overhead isn't worth it, and the dispatcher's existing retry/circuit-breaker already handles transient worker failures.

Write the body as **explicit acceptance criteria** — the judge is only as good as the goal text. "Translate the README" is weaker than "Translate every section of the README to French; no English sentences remain."

## Recovering stuck workers

When a worker profile keeps crashing, hallucinating, or getting blocked by its own mistakes (usually: wrong model, missing skill, broken credential), the kanban dashboard flags the task with a ⚠ badge and opens a **Recovery** section in the drawer. Three primary actions:

1. **Reclaim** (or `hermes kanban reclaim <task_id>`) — abort the running worker immediately and reset the task to `ready`. The existing claim TTL is ~15 min; this is the fast path out.
2. **Reassign** (or `hermes kanban reassign <task_id> <new-profile> --reclaim`) — switch the task to a different profile (one that exists on this setup) and let the dispatcher pick it up with a fresh worker.
3. **Change profile model** — the dashboard prints a copy-paste hint for `hermes -p <profile> model` since profile config lives on disk; edit it in a terminal, then Reclaim to retry with the new model.

Hallucination warnings appear on tasks where a worker's `kanban_complete(created_cards=[...])` claim included card ids that don't exist or weren't created by the worker's profile (the gate blocks the completion), or where the free-form summary references `t_<hex>` ids that don't resolve (advisory prose scan, non-blocking). Both produce audit events that persist even after recovery actions — the trail stays for debugging.

## Worker Pitfalls & Edge Cases

> Condensed from the former `kanban-worker` skill. The core lifecycle (orient → work → heartbeat → block/complete) is auto-injected via `KANBAN_GUIDANCE`; this section covers the deeper operational detail.

### Workspace kinds

| Kind | What it is | How to work |
|---|---|---|
| `scratch` | Fresh tmp dir, yours alone | Read/write freely; GC'd on archive. |
| `dir:<path>` | Shared persistent directory | Other runs will read what you write. Path is absolute. |
| `worktree` | Git worktree at the resolved path | If `.git` missing, run `git worktree add <path> ${HERMES_KANBAN_BRANCH:-wt/$HERMES_KANBAN_TASK}` from main repo, then cd and work normally. Commit here. |

### Claiming cards you created

Only pass ids you captured from successful `kanban_create` return values in `created_cards`. The kernel verifies each id exists and was created by your profile — phantom ids block completion and are permanently logged. Never invent ids from prose, paste ids from earlier runs, or claim cards another worker created.

### Block reasons that get answered fast

Bad: `"stuck"`. Good: one sentence naming the specific decision needed. Put longer context in a `kanban_comment` first; the block reason is what appears in the dashboard / gateway notifier.

```python
kanban_comment(task_id=..., body="Full context: ...")
kanban_block(reason="Rate limit key choice: IP (simple, NAT-unsafe) or user_id (requires auth, skips anonymous endpoints)?")
```

### Heartbeats worth sending

Good: `"epoch 12/50, loss 0.31"`, `"scanned 1.2M/2.4M rows"`. Bad: `"still working"`, empty notes, sub-second intervals. Every few minutes max; skip entirely for tasks under ~2 minutes.

### Retry scenarios

If `kanban_show` returns closed prior runs, you're a retry. Diagnose by `outcome`:

- `timed_out` — chunk the work or shorten it.
- `crashed` — OOM or segfault; reduce memory footprint.
- `spawn_failed` — profile config issue (missing credential, bad PATH); block for human input.
- `reclaimed` — operator archived the task; check status carefully.
- `blocked` — a previous attempt blocked; read the unblock comment in the thread.

### Worker DO NOT rules

- Call `delegate_task` as a substitute for `kanban_create`. `delegate_task` is for short reasoning subtasks inside your run; `kanban_create` is for cross-agent handoffs that outlive one API loop.
- Call `clarify` to ask the human — you are headless; use `kanban_comment` + `kanban_block(reason=...)` instead.
- Modify files outside `$HERMES_KANBAN_WORKSPACE` unless the task body says to.
- Create follow-up tasks assigned to yourself — assign to the right specialist.
- Complete a task you didn't actually finish. Block it instead.

### Worker pitfalls

- **Task state can change between dispatch and startup.** Always `kanban_show` first. If `blocked` or `archived`, stop.
- **Workspace may have stale artifacts.** Especially `dir:` and `worktree` workspaces. Read the comment thread for context.
- **Don't rely on the CLI in containerized backends.** `hermes kanban <verb>` from terminal fails when the CLI isn't installed. Use the `kanban_*` tools instead.

### Review-required pattern

For code-changing tasks, block instead of complete with `reason` prefixed `review-required: `. Drop structured metadata (changed files, test counts, diff/PR url) into a comment first, since `kanban_block` only carries the human-readable reason. Use `kanban_complete` only for genuinely terminal tasks (one-line typo fixes, docs changes, research where the writeup is the artifact).

### CLI fallback (for scripting / operators)

- `kanban_show` ↔ `hermes kanban show <id> --json`
- `kanban_complete` ↔ `hermes kanban complete <id> --summary "..." --metadata '{...}'`
- `kanban_block` ↔ `hermes kanban block <id> "reason"`
- `kanban_create` ↔ `hermes kanban create "title" --assignee <profile> [--parent <id>]`

## Codex CLI Integration Lane

> Condensed from the former `kanban-codex-lane` skill. Template file: `templates/codex-lane-prompt.md`.

### When to use

Use the Codex lane when: task is coding/refactor/docs/test/migration with clear acceptance criteria; a bounded diff can be evaluated in one run; repo can be isolated in a worktree/branch; Hermes can run the tests after Codex exits.

Do not use when: task needs human judgment not in the body; worker lacks repo access or Codex auth; change touches secrets/credentials/production; a small direct edit is faster; task is research-only; worker would mark Done based only on Codex self-report.

### Ownership rules

1. **Hermes owns Kanban lifecycle.** Codex must never call `kanban_complete`, `kanban_block`, `kanban_create`, gateway messaging, or any Hermes board CLI.
2. **Hermes owns final acceptance.** Treat Codex commits/diffs as untrusted patches until reviewed and verified.
3. **Hermes owns test execution.** Codex may run tests (advisory); repeat required verification from Hermes with the canonical wrapper.
4. **Hermes owns safety.** If Codex changes safety boundaries, risk gates, live trading behavior, or secrets handling, reject even if tests pass.
5. **Hermes owns cleanup.** Kill stuck Codex processes and remove temporary worktrees.

### Mode selection

- `codex exec --full-auto` for bounded one-shot edits (preferred).
- Codex `/goal` for broader multi-step work needing durable objective tracking.
- Never use `--yolo` for safety-sensitive repos.

### Prompt construction

Use `templates/codex-lane-prompt.md` as the base template — keep the structure and fill in YOUR repo's safety invariants. Every prompt must include: task_id + title + acceptance criteria; repo/worktree/branch/scope; ownership statement; required output (summary, files changed, commits, tests, risks); prohibited actions; verification commands for both Codex and Hermes.

Example safety constraints for a paper-trading repo (replace with your own invariants): live-SIM is paper-only; never place market orders; no fake fills/PnL; no risk-gate weakening; no secrets access.

### Worktree isolation

Never run Codex in a shared dirty checkout. Use a branch named `codex/<safe-task-id>/<timestamp>` in a separate worktree under `/tmp/`. Clean up after reconciliation unless needed as a review artifact.

### Monitoring and kill behavior

Start long lanes in background with PTY + `notify_on_complete`. Send `kanban_heartbeat` every few minutes. Kill conditions: no useful output for remaining budget; Codex requests secrets/credentials; Codex modifies files outside worktree; Codex starts unrelated rewrites; near worker timeout with no safe partial artifact.

### Reconciliation checklist

Before accepting any Codex result: verify `git status --short` shows only expected files; review `git diff`; confirm no secrets/unrelated data; verify safety constraints preserved; confirm commits are small enough to cherry-pick; Hermes ran canonical tests independently; accepted commits applied to Hermes-owned workspace.

Acceptance outcomes: `accepted`, `partial`, `rejected`, `timed_out`. Record all in `metadata.codex_lane`.

### Codex lane pitfalls

1. Treating Codex self-report as verification — always inspect diff and rerun tests.
2. Running Codex in the user's dirty main checkout — always isolate.
3. Letting Codex own Kanban state — Hermes writes board state.
4. Forgetting repo safety invariants in the prompt — missing safety text is a setup failure.
5. Using `/goal` for quick edits — prefer `codex exec`.
6. Killing a stuck lane without recording why — `rejected_reason` must explain.
7. Accepting broad unrelated cleanup because tests pass — cherry-pick only scoped changes.
