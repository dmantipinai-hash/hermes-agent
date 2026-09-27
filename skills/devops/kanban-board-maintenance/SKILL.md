---
name: kanban-board-maintenance
description: Clean the Kanban board when purging test or dead tasks.
version: 1.0.0
author: Dmitry + Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [kanban, maintenance, cleanup, sqlite]
    related_skills: [kanban-orchestrator]
---

# Kanban Board Maintenance

Recurring housekeeping class: test cards, dead experiment tails, and stale archived entries accumulate on the board and bury real work.

## Key fact: deletion is CLI-only and two-phase

The `kanban_*` tools have **no archive or delete action**. Only the CLI can remove tasks, and it requires two passes:

1. `hermes kanban archive t_xxx t_yyy ...` — moves tasks to `archived` (scratch workspaces are GC'd automatically at this point).
2. `hermes kanban archive --rm t_xxx t_yyy ...` — permanent delete. **Only works on tasks already in `archived` status**; you cannot `--rm` a live task directly.

## Full cleanup procedure

1. **Inventory** — list the suspects before acting:
   - `kanban_list(status="blocked")` — experiment tails, abandoned work
   - `kanban_list(status="ready")` — old test cards never claimed
   - `kanban_list(include_archived=true)` — stale archive leftovers
2. **Confirm each card is really dead.** A `blocked` card may hold an unanswered `[question]` in its thread — `kanban_show(task_id=...)` when in doubt. Test cards created by `user` in a testing spree (titles like `тест`, `Test task`, `test 4`) are safe calls when the user confirms they were tests.
3. **Archive → purge** (commands above). Batch all ids in one call each.
4. **Sweep the existing archive too** — old test entries linger there:
   ```bash
   sqlite3 ~/.hermes/kanban.db "SELECT id, title, assignee FROM tasks WHERE status='archived';"
   ```
5. **Verify** the end state:
   ```bash
   sqlite3 ~/.hermes/kanban.db "SELECT status, COUNT(*) FROM tasks GROUP BY status;"
   ```

## Keep vs purge

**Keep `done` tasks with real history** — research reports, project deliverables, completed smoke tests that documented a working method. Task rows are the audit trail (observability value).

**Purge:** duplicate/typo test cards, blocked cards from abandoned experiments where the thread shows no pending question, archived cards with test titles and no deliverable.

## `archive --rm` vs `gc` — different tools

- `archive --rm` deletes task rows permanently.
- `hermes kanban gc [--event-retention-days N] [--log-retention-days N]` trims task_events and worker logs for terminal tasks (default 30 days). It never deletes task rows. Use it for event-log bloat, not for removing cards.

## Pitfalls

- **Don't `--rm` without archiving first** — the purge silently targets only already-archived ids.
- **Don't purge on suspicion alone** — one `kanban_show` per ambiguous card is cheap; recovering a purged card is impossible.
- **User confirmation for bulk purge** — when the user says "they were test tasks", batch-delete confidently; when they just say "clean up", show the inventory and confirm the purge list first.

## Proven

2026-08-19: 17 stale tasks purged in one pass (9 blocked A2-experiment tails + 5 ready June test cards + 3 archived leftovers). Board went to 22 done / 0 everything else. Workspaces of deleted scratch tasks cleaned automatically.
