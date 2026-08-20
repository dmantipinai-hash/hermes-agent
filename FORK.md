# This fork: install, update, and the version story

This repository is the **dmantipinai-hash fork** of
[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent).
It carries **Architecture 2.0** — the memory system (v2, with FTS5 search),
the fast message-delivery pipeline, and the multi-agent orchestration
(`/team`, kanban board, agent mailbox) — which does **not exist in any
upstream release**.

## Version lineage — read this before touching versions

- The architecture was built on the **v0.16.0** line of this fork and is
  carried forward **only on this fork's `main`**.
- Upstream releases **0.17 / 0.18 / 0.19 / 0.20** (the PyPI package, the
  NousResearch installers, GitHub NousResearch archives) do **not** contain
  the architecture. Installing or "updating" to them loses the memory
  system, the message-delivery pipeline, and the orchestration. There is no
  migration path back except reinstalling the fork.
- The fork's version string (e.g. `0.20.4`) is inherited from the upstream
  base the maintainer merged underneath the architecture. It identifies the
  merge base, **not** "upstream 0.20.4 code". Identify the fork by this
  repository, never by the version number.
- Upstream merges into the fork are deliberate, manually-resolved events by
  the maintainer. They are never automated and never happen through
  `hermes update`.
- **`hermes update` is disabled in this fork.** All of its automatic
  channels resolve to upstream (the PyPI package, the GitHub archive ZIP
  fallback, the upstream-remote sync) and would silently replace the fork.
  Running it prints the fork's update command instead.

## Install — one command per platform

**Windows 10/11 (PowerShell):**

```powershell
iex (irm https://raw.githubusercontent.com/dmantipinai-hash/hermes-agent/main/scripts/install.ps1)
```

**macOS / Linux:**

```bash
curl -fsSL https://raw.githubusercontent.com/dmantipinai-hash/hermes-agent/main/scripts/install.sh | bash
```

**Any OS, if you already have [uv](https://docs.astral.sh/uv/):**

```bash
uv tool install --from "git+https://github.com/dmantipinai-hash/hermes-agent.git" "hermes-agent[all]"
```

The installers clone this repository (not NousResearch) and provision uv,
Python and Node as needed. After installing, run `hermes setup` to pick a
model and keys.

## Update — one command

The fork updates **from this repository only**. When the maintainer pushes
improvements to the fork's `main`:

```bash
uv tool install --force --from "git+https://github.com/dmantipinai-hash/hermes-agent.git" "hermes-agent[all]"
```

If you installed via the platform installer (which leaves a git checkout),
`hermes update` still works for you — it pulls this fork's repository. The
uv-tool command above covers every install type regardless.

## Traps — do NOT do these

- **Do not install or update from PyPI.** The `hermes-agent` PyPI package is
  the upstream release without the architecture. `pip install --upgrade
  hermes-agent` and `uv tool upgrade hermes-agent` both resolve to it.
- **Do not use upstream installers.** Anything pointing at
  `NousResearch/hermes-agent` or `hermes-agent.nousresearch.com` installs
  upstream. Use the commands above, which point at this fork.
- **Do not mix installs.** Running upstream hermes against a `~/.hermes`
  produced by this fork (memory v2 database, fork config keys) is untested
  and can lose or misread data. Back up `~/.hermes` before switching in
  either direction.
- **Do not use an editable dev checkout as your always-on gateway's
  runtime.** A checkout that is mid-merge or mid-rebase feeds breakage
  straight into the running agent.

## Advanced paths

**Pinned wheel snapshot** (most stable — what the maintainer uses; the
install never changes until you rebuild it):

```bash
git clone https://github.com/dmantipinai-hash/hermes-agent.git
cd hermes-agent && git log --oneline -1        # note the commit
HERMES_NIX_BUILD=1 uv build --wheel -o /tmp/hermes-dist
uv tool install --force 'hermes-agent[all] @ file:///tmp/hermes-dist/hermes_agent-<VERSION>-py3-none-any.whl'
```

`HERMES_NIX_BUILD=1` is required — upstream deliberately fails wheel builds
without it.

**Development checkout (editable venv):**

```bash
git clone https://github.com/dmantipinai-hash/hermes-agent.git
cd hermes-agent
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[all]"
```

## Notes

- User data lives in `~/.hermes/` (config.yaml, .env secrets, skills,
  memories, sessions). Profiles are isolated under `~/.hermes/profiles/<name>/`.
- After updating a global install that runs as a service, restart the
  service (e.g. `launchctl kickstart -k gui/$(id -u)/ai.hermes.gateway`).
  Expect a "Gateway shutting down" notification to the home channel —
  it is sent on every clean restart, active tasks or not.
