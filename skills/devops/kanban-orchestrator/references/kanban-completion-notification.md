# Kanban Task Completion Notification (cron polling)

## Problem

Kanban tasks run in **separate worker processes** spawned by the dispatcher. The orchestrator profile cannot use `terminal(background=true, notify_on_complete=true)` because it doesn't own the worker process. The gateway notifier *may* fire for the task's assigned profile, but the **requesting user is on the orchestrator's chat**, not the worker's.

## Solution: cron job that polls task status

Create a `no_agent=True` cron job (script-only, no LLM call) that checks the kanban SQLite DB directly. It fires every 15 minutes, checks `status`, and delivers a message to the user's chat only when the task is `done` or `blocked`.

### Marker-file deduplication

Without deduplication, the cron fires every 15 min and re-delivers the same "done" message. Use a `/tmp/.<task_id>_notified` marker file:

```python
#!/usr/bin/env python3
import sqlite3, sys, os

DB = "/Users/<user>/.hermes/kanban.db"
TASK_ID = "t_XXXXXXXX"
MARKER = f"/tmp/.{TASK_ID}_notified"

# Already notified? silent
if os.path.exists(MARKER):
    sys.exit(0)

try:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT status, summary FROM tasks WHERE id = ?", (TASK_ID,)
    ).fetchone()
    conn.close()

    if not row:
        sys.exit(0)  # task not found, silent

    status = row["status"]
    if status == "done":
        summary = row["summary"] or "(no summary)"
        if len(summary) > 500:
            summary = summary[:500] + "…"
        print(f"✅ Task {TASK_ID} complete!\n\n{summary}")
        open(MARKER, "w").close()  # prevent re-delivery
    elif status == "blocked":
        print(f"⚠️ Task {TASK_ID} blocked — check kanban board.")
        open(MARKER, "w").close()
    else:
        sys.exit(0)  # still running, silent — empty stdout = no delivery
except Exception:
    sys.exit(0)  # silent on errors
```

### Key semantics of `no_agent=True` cron jobs

- **Empty stdout** → **SILENT** — nothing sent to user. This is correct for "still running" polls.
- **Non-empty stdout** → delivered verbatim as message to the chat
- **Non-zero exit** → error alert sent (broken watchdog can't fail silently)
- **Marker file** prevents re-delivery after first notification

### Cron job parameters

```python
cronjob(
    action="create",
    name="Task <name> — completion poll",
    no_agent=True,              # script-only, no LLM
    schedule="15m",             # poll interval
    repeat=16,                  # 16 × 15 min = 4 hours coverage (match max_runtime)
    deliver="origin",           # deliver to the chat that created the job
    script="<python script above>"
)
```

### Cleanup

The marker file in `/tmp/` is ephemeral (cleared on reboot). The cron job self-disables after `repeat` count. No manual cleanup needed.

If the user wants to **stop** monitoring early (e.g., they check the board themselves):

```python
cronjob(action="list")  # find job_id
cronjob(action="remove", job_id="...")
```

### When to use

- Long-running kanban tasks (>10 min) where the user wants async notification
- Tasks assigned to secondary profiles (e.g., a research profile) where the orchestrator's chat ≠ worker's chat
- Multiple parallel tasks where you want per-task completion alerts

### When NOT to use

- Short tasks (<5 min) — just `kanban_show` and report inline
- Tasks where you're already using `terminal(notify_on_complete=True)` for the same process
