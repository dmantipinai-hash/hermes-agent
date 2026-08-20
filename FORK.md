# Installing and updating THIS fork

This repository is a fork of [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)
carrying extra features (Architecture 2.0: memory v2 with FTS5, multi-agent
`/team` orchestration, kanban mailbox, Codex subscription auth, and more).
Everything below exists because **the obvious install/update paths silently
give you upstream code instead of this fork**. Read this before installing;
the upstream README instructions do not all apply here.

## TL;DR — the three safe paths

**1. Global tool from this fork's git URL (recommended for users):**

```bash
uv tool install --from "git+https://github.com/dmantipinai-hash/hermes-agent.git" "hermes-agent[all]"
hermes setup
```

Update later by re-running the same command with `--force` (pins to the
current tip of the fork's `main`):

```bash
uv tool install --force --from "git+https://github.com/dmantipinai-hash/hermes-agent.git" "hermes-agent[all]"
```

**2. Dev checkout (editable venv):**

```bash
git clone https://github.com/dmantipinai-hash/hermes-agent.git
cd hermes-agent
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[all]"
```

Update with `git pull` (re-run the editable install only when dependencies
changed). Fine for development — but do **not** point an always-on gateway
at a checkout you actively merge or rebase: mid-merge breakage flows
straight into the running agent.

**3. Pinned wheel snapshot (most stable — what the maintainer uses):**

```bash
git clone https://github.com/dmantipinai-hash/hermes-agent.git
cd hermes-agent && git log --oneline -1        # note the commit
HERMES_NIX_BUILD=1 uv build --wheel -o /tmp/hermes-dist
uv tool install --force 'hermes-agent[all] @ file:///tmp/hermes-dist/hermes_agent-<VERSION>-py3-none-any.whl'
```

A snapshot install never changes until you explicitly rebuild it. To move
to a newer state: `git pull`, re-run the two commands. `HERMES_NIX_BUILD=1`
is required — upstream deliberately fails wheel builds without it.

## Traps — do NOT do these

- **Do not run `hermes update`.** For pip / uv-tool / pipx installs it
  upgrades the `hermes-agent` package **from PyPI**, which is the upstream
  release — it silently replaces this fork and drops every fork feature.
  Its archive fallback also downloads from NousResearch directly. Update
  only via the commands above.
- **Do not use the upstream installers.** `scripts/install.sh`,
  `scripts/install.ps1`, `scripts/install.cmd`, and
  `https://hermes-agent.nousresearch.com/install.ps1` hardcode
  `NousResearch/hermes-agent` — they install upstream, not this fork.
- **Do not mix installs.** Running upstream hermes against a `~/.hermes`
  produced by this fork (memory v2 database, fork config keys) is
  untested and can lose or misread data. Back up `~/.hermes` before
  switching between fork and upstream in either direction.
- **Do not use an editable install as your daily driver's runtime.**
  See path 2 above.

## Notes

- The version string (e.g. `0.20.4`) is inherited from upstream at merge
  time. It identifies the merge base, not "upstream 0.20.4 code".
- User data lives in `~/.hermes/` (config.yaml, .env secrets, skills,
  memories, sessions). Profiles are isolated under
  `~/.hermes/profiles/<name>/`.
- After updating a global install that runs as a service, restart the
  service (e.g. `launchctl kickstart -k gui/$(id -u)/ai.hermes.gateway`).
  Expect a "Gateway shutting down" notification to the home channel —
  it is sent on every clean restart, active tasks or not.
