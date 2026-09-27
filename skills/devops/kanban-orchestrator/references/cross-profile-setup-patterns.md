# Cross-Profile Setup Patterns

## Context: When context window runs out

Your daily profile (small context, fast, cheap) hits the limit. You need to delegate to a reserve profile (large context, slow, expensive).

## Pattern 1: MCP server replication

Secondary profiles often don't inherit MCP from default. Check and copy:

```bash
# Check default profile MCP config
cat ~/.hermes/config.yaml | grep -A 20 "mcp_servers:"

# Check if secondary profile has MCP
cat ~/.hermes/profiles/<profile-name>/config.yaml | grep -A 20 "mcp_servers:"

# CRITICAL: count mcp_servers blocks — must be exactly 1
grep -c "^mcp_servers:" ~/.hermes/profiles/<profile-name>/config.yaml
# If >1, the LAST block wins and all others are silently ignored!
```

⚠️ **YAML duplicate key pitfall:** If the profile already has an `mcp_servers:` block (e.g. for cua-driver), do NOT append a second one. Merge into the existing block. A second `mcp_servers:` at any point in the file completely shadows the first — no error, no warning. The worker silently falls back to browser/computer_use instead of using the MCP API.

Correct Z.AI web-search-prime config (HTTP endpoint format, per https://docs.z.ai/devpack/mcp/search-mcp-server):

```yaml
mcp_servers:
  web-search-prime:
    url: https://api.z.ai/api/mcp/web_search_prime/mcp
    headers:
      Authorization: "Bearer <your-api-key>"
    timeout: 180
  cua-driver:                    # example of a second server in SAME block
    command: /Users/<user>/.local/bin/cua-driver
    args:
    - mcp
```

## Pattern 2: Gateway startup and restart

Secondary profiles need their gateway running:

```bash
# Check status
hermes profile list

# Start if stopped
hermes -p <profile-name> gateway
```

**After editing config.yaml (e.g. adding MCP servers), you MUST restart the gateway** — the running process keeps the old config in memory:

```bash
# Find and kill the gateway PID — systemd auto-restarts with new config
GATEWAY_PID=$(ps aux | grep "profile <name>" | grep gateway | grep -v grep | awk '{print $2}')
kill $GATEWAY_PID
# Verify restart within ~5s
sleep 5 && ps aux | grep "profile <name>" | grep gateway | grep -v grep
```

## Pattern 3: Telegram token conflict

If both profiles use the same Telegram bot token, the second gateway crashes with:

```
Error: Telegram bot token already in use by PID 12345
```

**Fix:** Empty the token in the secondary profile's `.env` (last value wins):

```bash
echo "TELEGRAM_BOT_TOKEN=" >> ~/.hermes/profiles/<profile-name>/.env
```

Kanban tasks don't need Telegram — the dispatcher works fine without it. The primary gateway notifies on completion.

## Pattern 4: Directory workspace sharing

When delegating tasks that need file access:

```bash
hermes kanban create "Task Title" \
  --assignee <secondary-profile> \
  --workspace "dir:/absolute/path/to/project" \
  --goal \
  --goal-max-turns 25 \
  --max-runtime 45m \
  --created-by "orchestrator-name"
```

The workspace parameter points to an existing directory containing:
- Context files (brief, research, config)
- Templates (HTML, CSS, boilerplate)
- Output destination (e.g., `website/` folder)

The secondary profile reads/writes directly in this shared workspace.

## Pattern 5: Brief file pattern for complex tasks

Don't cram multi-stage tasks into the card body. Write a detailed `TASK-<name>.md` in the shared workspace root:

```
card body: "ПОЛНЫЙ БРИФ: прочитай TASK-research.md в корне workspace.
            ЭТАП 1: ... ЭТАП 2: ... ЭТАП 3: ...
            Результаты: file1.md + file2.md + file3.md.
            Если MCP web-search-prime не работает — заблокируй задачу с причиной."
```

The brief file survives context resets and can be versioned. The card body stays a concise routing instruction.

## Pattern 6: MCP self-test in card body

When a task requires MCP tools, include a self-diagnostic instruction:

```
"If MCP web-search-prime не работает — заблокируй задачу с причиной."
```

The worker either uses the MCP tool or blocks the task with a clear reason — no silent failures. Check for `blocked` status before assuming the task is running.

## Verification checklist

Before delegating:

- [ ] Target profile exists: `hermes profile list`
- [ ] Target model has sufficient context: check `config.yaml` → `context_length`
- [ ] MCP servers configured: `config.yaml` → exactly 1 `mcp_servers:` block with all servers inside
- [ ] Gateway restarted after config changes: kill PID → systemd auto-restart
- [ ] No Telegram conflicts: override `TELEGRAM_BOT_TOKEN` in secondary `.env`

After delegation:

- [ ] Task status shows "running" (not "ready"): `hermes kanban show <task_id>`
- [ ] Heartbeats arriving every 1-5 minutes
- [ ] Heartbeat notes don't mention "Safari" or "browser" when MCP search was expected (if they do → MCP didn't load, check duplicate keys)
- [ ] Worker has file access: output files appear in workspace

## Real-world example (2026-06-14 session)

Profile "default" (glm-5.1, 200K context) was about to hit limit after 40 minutes on a multi-stage research + website build task.

Delegated to "<research-profile>" (deepseek-v4-pro, 1M context):

1. `hermes profile list` → confirmed <research-profile> exists with deepseek-v4-pro
2. Checked <research-profile> config.yaml → no `mcp_servers` block initially
3. Patched <research-profile> config.yaml → added `mcp_servers.web-search-prime` (HTTP endpoint format)
4. Started <research-profile> gateway → crashed (Telegram bot token conflict with default profile)
5. Fixed: `echo "TELEGRAM_BOT_TOKEN=" >> ~/.hermes/profiles/<research-profile>/.env`
6. Restarted <research-profile> gateway → success
7. Created Kanban task with `--workspace "dir:/home/<you>/projects/<project>" --goal --goal-max-turns 25`
8. Worker completed in 6 minutes — 1232 lines of code generated

**Second task (research + financial model):**
1. Added the MCP config to <research-profile> — but a duplicate `mcp_servers:` block at the bottom of config.yaml (for cua-driver) silently overwrote the web-search-prime block
2. Worker fell back to Safari/computer_use for web search (10x slower)
3. Diagnosed via heartbeat note: "Поиск 1 завершён через Z.ai Safari" — the word "Safari" revealed MCP wasn't loaded
4. Fix: merged both servers into single `mcp_servers:` block, killed gateway PID, auto-restarted
5. Reclaimed task, new run used MCP correctly

**Lesson:** `grep -c "^mcp_servers:" config.yaml` must return 1. If it returns 2+, merge immediately.
