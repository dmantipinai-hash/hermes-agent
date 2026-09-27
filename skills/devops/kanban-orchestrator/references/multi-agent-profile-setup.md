# Multi-Agent Profile Setup

How to create and configure Hermes profiles for a multi-agent team.
The orchestrator skill assumes profiles already exist (Step 0: discover).
This reference covers everything *before* that — bringing profiles into existence.

## Profile = independent agent on the same machine

Each profile lives at `~/.hermes/profiles/<name>/` with its own:
- `config.yaml` — model, provider, toolsets, all settings
- `.env` — API keys, bot tokens
- `SOUL.md` — personality, name, character
- Memory, sessions, skills, cron — all separate

Profiles become shell commands automatically: `hermes profile create diamond` →
`diamond chat`, `diamond gateway start`, `diamond config set ...`.

## Creation workflow

### 1. Create the profile

```bash
# Blank (bundled skills only, no config)
hermes profile create diamond

# Clone config + .env + SOUL.md + skills from current profile
hermes profile create diamond --clone

# Clone from a specific profile
hermes profile create diamond --clone-from default

# Full snapshot (config, memory, skills, cron — NOT sessions/history)
hermes profile create diamond --clone-all

# With role description for kanban routing
hermes profile create diamond --description "Code review and refactoring specialist"
```

### 2. Configure the model

```bash
diamond config set model.default deepseek/deepseek-chat-v3-0324
# or interactive picker:
diamond model
```

Or edit `~/.hermes/profiles/diamond/config.yaml` directly.

### 3. Set personality

Write a SOUL.md for the new agent:
```bash
# Either clone brought your SOUL — edit it:
nano ~/.hermes/profiles/diamond/SOUL.md
# Or write fresh:
cat > ~/.hermes/profiles/diamond/SOUL.md << 'EOF'
# Diamond
You are Diamond, a focused code-review specialist...
EOF
```

### 4. Configure API keys

```bash
diamond setup    # interactive wizard
# or edit directly:
nano ~/.hermes/profiles/diamond/.env
```

Each profile has independent `.env` — separate API keys, separate billing.

### 5. Set up gateway (Telegram/Discord/etc.)

Each profile needs its **own bot token**. Two profiles cannot share one token —
Hermes enforces token locks and blocks the second gateway with a clear error.

```bash
# Create a new bot via @BotFather in Telegram, then:
diamond gateway setup    # configure platform tokens
diamond gateway start    # start gateway as foreground process
diamond gateway install  # install as persistent launchd/systemd service
```

Each profile gets its own service name (e.g. `ai.hermes.gateway.diamond`).

### 6. Verify

```bash
diamond doctor            # health check
diamond profile           # show profile info
hermes profile list       # see all profiles with status
```

## Key facts

- **Profiles are NOT sandboxes.** A profile does not limit filesystem access.
  All profiles share the same user-level filesystem on the local terminal backend.
  Set `terminal.cwd` in the profile's config for a default working directory.
- **Token locks.** Telegram, Discord, Slack, WhatsApp, Signal — if two profiles
  accidentally use the same bot token, the second gateway is blocked immediately.
- **Cost isolation.** Each profile with its own model/API key = separate billing.
  Cloning `--clone` copies `.env` including API keys — fine for same provider,
  edit `.env` if the new profile should use different credentials.
- **Skill sync.** `hermes update` syncs bundled skills to all profiles.
  User-modified skills are never overwritten.
- **Sticky default.** `hermes profile use diamond` makes plain `hermes` commands
  target diamond. `hermes profile use default` to switch back.

## Inter-agent communication via Kanban

Profiles don't chat with each other directly. The Kanban board is the
communication channel — it acts as a shared task queue / inbox:

```bash
hermes kanban init                          # create board (once)
hermes kanban create "Analyze X" \
  --assignee diamond                        # assign task to Diamond
hermes kanban list                          # see all tasks
hermes kanban show <task-id>               # read results
```

The dispatcher (runs inside the gateway by default) automatically:
- Reclaims stale worker claims
- Promotes ready tasks
- Spawns the assigned profile as a worker process
- Delivers results back to the board

This is the inbox protocol: one agent creates a task → another picks it up →
result returns to the board. No direct agent-to-agent messaging needed.

## Multi-agent team pattern

For a permanent-agent setup (Loki + Diamond + others):

| Building block | Hermes feature |
|---|---|
| Permanent agent with SOUL | Profile + SOUL.md |
| Separate memory per agent | Profile's own memory store |
| Shared task board | Kanban (`hermes kanban`) |
| Inbox protocol | Kanban create → assign → dispatch |
| Heartbeat | Cron jobs + kanban heartbeat events |
| Separate Telegram per agent | Per-profile gateway + bot token |
| Different models per agent | Per-profile config.yaml model setting |

Create the team:
```bash
hermes profile create diamond --description "Code review specialist"
hermes profile create researcher --description "Research and analysis"
# Configure each: model, SOUL, API keys, gateway
hermes kanban init
# Now the orchestrator can assign tasks to the new profile and researcher via the board
```
