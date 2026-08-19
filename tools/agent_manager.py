"""Agent Manager — orchestrate a team of specialized agent profiles.

A "team" is a set of Hermes profiles, each treated as a persistent,
specialized agent (e.g. finance, marketing, product) with its own
model / provider / API key / memory / SOUL.md persona. This module is
the backend the ``agent_manager`` toolset and the ``/team`` slash
command call into. It does NOT itself talk to the model — it manages
profile state and routes work onto the kanban board (the existing
cross-profile coordination primitive).

Design rules (load-bearing — read before editing):

1. **All writes go by absolute path.** We deliberately bypass
   ``hermes_cli.config.set_config_value`` / ``save_env_value`` because
   those operate on the *ambient* ``HERMES_HOME`` of the orchestrator
   process and would corrupt its session if we pointed them at a child
   profile. Instead we read/write ``<profile_dir>/config.yaml`` and
   ``<profile_dir>/.env`` directly. This is the single biggest
   correctness invariant in this module.

2. **Profiles, not delegate_task, are the persistence unit.** A
   subagent's context dies when ``delegate_task`` returns; a profile's
   ``state.db`` / ``memories/`` survive across runs. The async path
   (``assign_task``) uses kanban so each piece of work lands on a real
   profile subprocess; the sync path is plain ``delegate_task`` with a
   role, documented in the tool descriptions.

3. **Direct imports, no shelling out.** Following ``tools/kanban_tools.py``
   we import ``hermes_cli.profiles`` / ``hermes_cli.kanban_db`` directly
   (lazily, inside functions) so we reach the right DB regardless of the
   configured terminal backend and avoid shell-quoting footguns.

4. **JSON-string returns.** Every tool handler returns a JSON string —
   the registry's contract.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-process registry of active synchronous ask_agent calls.
#
# ask_agent spawns a blocking subprocess that does NOT write a kanban run row
# (by design — B-branch, see ARCHITECTURE-2.0.md §8.3). So the kanban-based
# busy-probe in list_agents never sees it. Without this registry, list_agents
# reports a profile as idle while an ask_agent to it is mid-flight, which
# misled a manager into double-calling (TD-5).
#
# Process-local: only the orchestrator process sees its own ask_agent calls.
# Two orchestrators on the same board won't share this state — acceptable,
# the kanban busy-probe covers cross-process workers; this covers the
# in-process gap.
# ---------------------------------------------------------------------------
import threading as _threading

_ask_lock = _threading.Lock()
# profile canonical name -> number of in-flight ask_agent calls
_active_ask_calls: Dict[str, int] = {}


def _register_active_ask(profile: str) -> None:
    with _ask_lock:
        _active_ask_calls[profile] = _active_ask_calls.get(profile, 0) + 1


def _release_active_ask(profile: str) -> None:
    with _ask_lock:
        n = _active_ask_calls.get(profile, 0)
        if n <= 1:
            _active_ask_calls.pop(profile, None)
        else:
            _active_ask_calls[profile] = n - 1


def _active_ask_count(profile: str) -> int:
    """Number of in-flight ask_agent calls to this profile (process-local)."""
    with _ask_lock:
        return _active_ask_calls.get(profile, 0)


def _reset_active_ask_registry_for_test() -> None:
    """Clear the registry — unit tests only."""
    with _ask_lock:
        _active_ask_calls.clear()


# ===========================================================================
# Lazy-import helpers (keep this module importable in contexts that never
# touch profiles/kanban — mirrors tools/kanban_tools.py:_connect).
# ===========================================================================

def _profiles():
    from hermes_cli import profiles
    return profiles


def _kanban():
    from hermes_cli import kanban_db
    return kanban_db


def _role_map():
    """ROLE_TOOLSET_MAP from toolsets — the Phase-1 specialization catalog."""
    from toolsets import ROLE_TOOLSET_MAP
    return ROLE_TOOLSET_MAP


# ===========================================================================
# THE helper: write a profile's config.yaml + .env + SOUL.md by absolute path.
# This is the load-bearing piece — never replace with set_config_value.
# ===========================================================================

def _atomic_write(path: Path, data: str, mode: int = 0o644) -> None:
    """Write ``data`` to ``path`` atomically (tmp + rename), then chmod."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def _deep_merge(dst: dict, src: dict) -> dict:
    """In-place deep merge of ``src`` into ``dst``; returns ``dst``."""
    for k, v in src.items():
        if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def _write_profile_config(
    profile_dir: Path,
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_mode: Optional[str] = None,
) -> None:
    """Merge model/provider/base_url into ``<profile_dir>/config.yaml``.

    Reads the existing YAML (or starts from ``{}``), deep-merges the
    ``model:`` dict section, and writes back atomically. Never touches
    any other top-level section. Safe on a profile that has no
    config.yaml yet (one gets created).
    """
    import yaml

    cfg_path = profile_dir / "config.yaml"
    cfg: dict = {}
    if cfg_path.exists():
        try:
            from hermes_cli.config import read_user_config_raw

            loaded = read_user_config_raw(cfg_path)
            if isinstance(loaded, dict):
                cfg = loaded
        except Exception:
            logger.warning("agent_manager: corrupt config.yaml at %s, rebuilding", cfg_path)
            cfg = {}

    # Build the model-section overlay. Existing keys are preserved.
    model_section = cfg.get("model")
    if not isinstance(model_section, dict):
        # Was a bare string (back-compat form) or absent — normalize to dict.
        model_section = {"default": model_section} if isinstance(model_section, str) else {}
    if model is not None:
        model_section["default"] = model
    if provider is not None:
        model_section["provider"] = provider
    if base_url is not None:
        model_section["base_url"] = base_url
    if api_mode is not None:
        model_section["api_mode"] = api_mode
    if model_section:
        cfg["model"] = model_section

    _atomic_write(cfg_path, yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))


def _write_profile_env(profile_dir: Path, key: str, value: str) -> None:
    """Write a single KEY=value line into ``<profile_dir>/.env`` (mode 0600).

    Preserves existing entries; updates if the key already exists.
    """
    env_path = profile_dir / ".env"
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    found = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.split("=", 1)[0].strip() == key:
            lines[i] = f"{key}={value}"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}")
    _atomic_write(env_path, "\n".join(lines) + "\n", mode=0o600)


def _write_profile_soul(profile_dir: Path, persona: str) -> None:
    """Write the profile's SOUL.md (the system-prompt identity slot)."""
    _atomic_write(profile_dir / "SOUL.md", persona.strip() + "\n")


# ===========================================================================
# Persona generation — turns a role + description into a SOUL.md.
# ===========================================================================

def _build_persona(role: Optional[str], description: Optional[str]) -> str:
    """Compose a SOUL.md persona for a new agent from role + free-form description."""
    role_map = _role_map()
    parts: list[str] = []

    if role and role in role_map:
        hint = role_map[role].get("prompt_hint", "").strip()
        role_desc = role_map[role].get("description", "").strip()
        parts.append(f"# {role.capitalize()} Agent")
        if role_desc:
            parts.append(role_desc)
        if hint:
            parts.append("")
            parts.append(hint)
    else:
        parts.append("# Specialized Agent")

    if description:
        parts.append("")
        parts.append("## Identity")
        parts.append(description.strip())

    parts.append("")
    parts.append(
        "You are a member of a coordinated agent team. Stay within your "
        "specialty. When a task falls outside your role, say so rather "
        "than improvising."
    )
    return "\n".join(parts)


# ===========================================================================
# Public backend API (called by tool handlers + /team slash command)
# ===========================================================================

def list_agents() -> List[Dict[str, Any]]:
    """Return all profiles with live busy-status.

    "busy" = has at least one kanban task in ``running`` status assigned
    to it (the dispatcher's own definition, kanban_db.py:5873). Falls
    back to ``gateway_running`` if the kanban DB is unreachable.
    """
    profiles = _profiles()
    agents: list[dict] = []

    # Count running tasks per assignee, best-effort.
    running_counts: Dict[str, int] = {}
    try:
        kanban = _kanban()
        conn = kanban.connect()
        try:
            rows = conn.execute(
                "SELECT assignee, COUNT(*) AS n FROM tasks "
                "WHERE status = 'running' AND assignee IS NOT NULL "
                "GROUP BY assignee"
            ).fetchall()
            running_counts = {r[0]: r[1] for r in rows if r[0]}
        finally:
            conn.close()
    except Exception as e:
        logger.debug("agent_manager: kanban busy-probe failed (%s); using gateway flag", e)

    for info in profiles.list_profiles():
        name = info.name
        active = running_counts.get(name, 0)
        active_asks = _active_ask_count(name)
        agents.append({
            "name": name,
            "is_default": info.is_default,
            "role": _role_from_profile(info),
            "model": info.model,
            "provider": info.provider,
            # busy = kanban task running OR gateway running OR a synchronous
            # ask_agent to this profile is currently in flight (process-local,
            # TD-5). Without the ask check, list_agents reported a profile as
            # idle while ask_agent was blocked on its subprocess.
            "busy": active > 0 or bool(info.gateway_running) or active_asks > 0,
            "active_tasks": active,
            "active_asks": active_asks,
            "gateway_running": info.gateway_running,
            "description": info.description or "",
            "path": str(info.path),
        })
    return agents


def _role_from_profile(info) -> Optional[str]:
    """Best-effort: infer a profile's role from its description/SOUL.md."""
    # Stored in profile.yaml description by create_agent (prefixed "role: <r> — ...").
    desc = (info.description or "").strip()
    if desc.lower().startswith("role:"):
        try:
            return desc.split("—", 1)[0].split(":", 1)[1].strip().split()[0].lower()
        except Exception:
            return None
    return None


def create_agent(
    *,
    name: str,
    role: Optional[str] = None,
    description: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    api_key_env: Optional[str] = None,
    api_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Create + configure a specialized agent profile.

    Args:
        name: profile name (lowercase, [a-z0-9_-]).
        role: one of ROLE_TOOLSET_MAP keys; seeds toolset, SOUL.md, description.
        description: free-form identity; if omitted, derived from the role.
        model / provider / base_url / api_mode: written into the profile's
            ``model:`` config section.
        api_key: the secret value (e.g. "sk-..."). Requires ``api_key_env``
            so we know which env var name to write it under (e.g. GLM_API_KEY).
        api_key_env: the env var NAME (not value), e.g. ``"GLM_API_KEY"``.
    """
    profiles = _profiles()
    role_map = _role_map()

    # Validate role eagerly so we fail before creating files.
    if role is not None and role not in role_map:
        raise ValueError(
            f"Unknown role {role!r}. Valid: {sorted(role_map.keys())}"
        )

    # Build a description from role if not supplied.
    if description is None and role:
        description = role_map[role].get("description", "")
    # Stamp the role into the description so _role_from_profile can recover it.
    meta_description = description or ""
    if role:
        meta_description = f"role: {role} — {meta_description}".strip(" —")

    profile_dir = profiles.create_profile(
        name,
        description=meta_description,
        no_alias=True,
        no_skills=True,
    )

    # Configure model / provider / key — all by absolute path.
    configured_any = False
    if model or provider or base_url or api_mode:
        _write_profile_config(
            profile_dir,
            model=model,
            provider=provider,
            base_url=base_url,
            api_mode=api_mode,
        )
        configured_any = True
    if api_key:
        if not api_key_env:
            raise ValueError(
                "api_key requires api_key_env (the env var NAME, e.g. 'GLM_API_KEY')"
            )
        _write_profile_env(profile_dir, api_key_env, api_key)
        configured_any = True

    # Write the SOUL.md persona from role + description.
    _write_profile_soul(profile_dir, _build_persona(role, description))

    return {
        "name": name,
        "role": role,
        "description": description or "",
        "model": model,
        "provider": provider,
        "base_url": base_url,
        "path": str(profile_dir),
        "configured": configured_any,
    }


def set_agent_model(
    name: str,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    api_key_env: Optional[str] = None,
    api_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Reconfigure an agent's model/provider/key WITHOUT recreating it.

    This is the "I don't like this provider, swap it" path. Reads the
    existing profile, overwrites only the model section / the named env
    key. Nothing else about the profile changes (memories, SOUL.md,
    sessions all preserved).
    """
    profiles = _profiles()
    canon = profiles.normalize_profile_name(name)
    profile_dir = profiles.get_profile_dir(canon)
    if not profile_dir.is_dir():
        raise FileNotFoundError(f"Agent/profile {name!r} does not exist at {profile_dir}")

    if model or provider or base_url or api_mode:
        _write_profile_config(
            profile_dir,
            model=model,
            provider=provider,
            base_url=base_url,
            api_mode=api_mode,
        )
    if api_key:
        if not api_key_env:
            raise ValueError("api_key requires api_key_env (the env var NAME)")
        _write_profile_env(profile_dir, api_key_env, api_key)

    # Read back the resulting model/provider for the response.
    m, p = profiles._read_config_model(profile_dir)
    updated = sorted(
        k for k, v in {
            "model": model, "provider": provider,
            "base_url": base_url, "api_mode": api_mode,
            api_key_env or "api_key": api_key,
        }.items() if v is not None
    )
    return {
        "name": canon,
        "model": m,
        "provider": p,
        "updated": updated,
    }


def delete_agent(name: str, *, force: bool = False) -> Dict[str, Any]:
    """Delete an agent profile. Refuses if it has running kanban tasks.

    The built-in ``profiles.delete_profile`` doesn't check kanban state;
    we add the guard here so an orchestrator can't silently delete a
    busy agent mid-task.
    """
    profiles = _profiles()
    canon = profiles.normalize_profile_name(name)

    if canon == "default":
        raise ValueError("Cannot delete the built-in 'default' profile")

    # Guard: count running tasks for this assignee.
    active = _count_active_tasks(canon)
    if active > 0 and not force:
        return {
            "name": canon,
            "deleted": False,
            "blocked_reason": f"agent has {active} active kanban task(s); pass force=True to override",
            "active_tasks": active,
        }

    removed = profiles.delete_profile(canon, yes=True)
    return {
        "name": canon,
        "deleted": True,
        "path": str(removed),
        "had_active_tasks": active,
    }


def _count_active_tasks(assignee: str) -> int:
    """Count running/ready tasks assigned to ``assignee`` (0 if DB unreachable)."""
    try:
        kanban = _kanban()
        conn = kanban.connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE assignee = ? AND status IN ('running','ready')",
                (kanban._canonical_assignee(assignee),),
            ).fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()
    except Exception as e:
        logger.debug("agent_manager: kanban active-count failed (%s)", e)
        return 0


def agent_status(name: str) -> Dict[str, Any]:
    """Detailed status for one agent: config, active tasks, recent activity."""
    profiles = _profiles()
    canon = profiles.normalize_profile_name(name)
    profile_dir = profiles.get_profile_dir(canon)
    if not profile_dir.is_dir():
        raise FileNotFoundError(f"Agent/profile {name!r} does not exist at {profile_dir}")

    info = None
    for p in profiles.list_profiles():
        if p.name == canon:
            info = p
            break

    active_tasks: List[Dict[str, Any]] = []
    try:
        kanban = _kanban()
        conn = kanban.connect()
        try:
            tasks = kanban.list_tasks(conn, assignee=canon, status="running")
            active_tasks = [
                {"id": t.id if hasattr(t, "id") else t.get("id"),
                 "title": t.title if hasattr(t, "title") else t.get("title")}
                for t in tasks[:5]
            ]
        finally:
            conn.close()
    except Exception as e:
        logger.debug("agent_manager: status kanban probe failed (%s)", e)

    active_asks = _active_ask_count(canon)

    return {
        "name": canon,
        "role": _role_from_profile(info) if info else None,
        "model": info.model if info else None,
        "provider": info.provider if info else None,
        "gateway_running": info.gateway_running if info else False,
        # busy includes in-flight ask_agent calls (TD-5), same as list_agents.
        "busy": len(active_tasks) > 0 or (info.gateway_running if info else False) or active_asks > 0,
        "active_tasks": active_tasks,
        "active_asks": active_asks,
        "description": info.description if info else "",
        "path": str(profile_dir),
    }


def assign_task(
    agent: str,
    goal: str,
    *,
    parent_task_id: Optional[str] = None,
    priority: int = 0,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a kanban task assigned to ``agent`` — the async coordination path.

    The kanban dispatcher will spawn ``hermes -p <agent>`` to work on it.
    Use ``parent_task_id`` to gate this task on another's completion
    (kanban promotes ``todo→ready`` only when all parents complete).
    """
    profiles = _profiles()
    kanban = _kanban()

    canon = profiles.normalize_profile_name(agent)
    if not profiles.get_profile_dir(canon).is_dir():
        raise FileNotFoundError(
            f"Agent {agent!r} does not exist. Create it first with create_agent."
        )

    conn = kanban.connect()
    try:
        task_id = kanban.create_task(
            conn,
            title=title or goal[:120],
            body=goal,
            assignee=canon,
            parents=(parent_task_id,) if parent_task_id else (),
            priority=priority,
            # initial_status defaults to 'running' in create_task, but
            # kanban's recompute_ready (called inside create_task) will
            # promote a task with unfinished parents down to 'todo', and
            # a parentless task up to 'ready'. We don't force the value —
            # let kanban derive the correct lifecycle status.
        )
    finally:
        conn.close()

    return {
        "task_id": task_id,
        "agent": canon,
        "title": title or goal[:120],
        "parent_task_id": parent_task_id,
        "note": (
            "Task is on the kanban board. Ensure the dispatcher is running "
            "(hermes kanban daemon, or kanban.dispatch_in_gateway: true which "
            "is the default) so hermes -p <agent> gets spawned to work on it."
        ),
    }


# ===========================================================================
# ask_agent — Phase 1 of ARCHITECTURE-2.0 (B-subprocess).
#
# Synchronous "manager asks a named profile a question" path. Spawns
# ``hermes -p <agent> chat -Q -q "<question>"`` as a real subprocess so the
# child runs under the profile's own HERMES_HOME (config.yaml, .env keys,
# state.db memory, SOUL.md persona) — isolation by construction.
#
# WHY subprocess and not in-process AIAgent (design record, §8.3 of
# ARCHITECTURE-2.0.md):
#   * agent/agent_init.py resolves config + memory from the AMBIENT
#     process HERMES_HOME (get_hermes_home() at init: line ~988,
#     _load_agent_config() at ~1046). Building a "profile child" in the
#     orchestrator's process would silently load the ORCHESTRATOR's
#     MEMORY.md/config under the child's name — cross-profile data
#     poisoning (see hermes_constants.get_hermes_home docstring +
#     issue #18594: "subprocess spawners are expected to propagate
#     HERMES_HOME explicitly").
#   * The kanban dispatcher already crossed this bridge the same way
#     (kanban_db._default_spawn → resolve_profile_env → env injection).
#     We mirror that proven pattern, not invent a second one.
#
# Role vs profile — source-of-truth rule (§8.4): for a NAMED profile the
# profile's own config.yaml decides toolsets/model/persona; ``role`` presets
# are only for ephemeral delegate_task children. ask_agent therefore takes
# NO role/toolsets parameters at all — the profile is the truth.
# ===========================================================================

ASK_AGENT_DEFAULT_TIMEOUT = 600  # seconds; mirrors delegation.child_timeout_seconds default


def _ask_agent_timeout() -> int:
    """Read agent_manager.ask_timeout_seconds from config (default 600s)."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config().get("agent_manager", {}) or {}
        val = int(cfg.get("ask_timeout_seconds", ASK_AGENT_DEFAULT_TIMEOUT))
        return max(30, min(val, 7200))  # clamp [30s, 2h]
    except Exception:
        return ASK_AGENT_DEFAULT_TIMEOUT


def ask_agent(
    agent: str,
    question: str,
    *,
    context: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """Synchronously ask a named agent profile a question; return its answer.

    Blocking call: spawns ``hermes -p <agent> chat -Q -q <prompt>`` and
    waits. The child runs with the profile's own HERMES_HOME so its
    config/keys/memory/persona apply (isolation by construction — see
    module comment above). stdout is machine-readable thanks to ``-Q``
    (banner/spinner/tool previews suppressed; resume noise goes to
    stderr, see cli.py issue #11793).

    Raises FileNotFoundError for unknown agents, TimeoutError on cap.
    """
    import subprocess
    import time as _time

    profiles = _profiles()
    canon = profiles.normalize_profile_name(agent)
    profile_dir = profiles.get_profile_dir(canon)
    if not profile_dir.is_dir():
        raise FileNotFoundError(
            f"Agent {agent!r} does not exist. Create it first with create_agent."
        )
    if not question or not question.strip():
        raise ValueError("question is required")

    # Spawn the child through the SAME interpreter + codebase as the parent
    # orchestrator. Do NOT use kanban_db._resolve_hermes_argv() here — that
    # helper resolves the *system* hermes (HERMES_BIN → `which hermes` →
    # module fallback), which on a dev checkout points at a different install
    # (e.g. ~/.local/bin/hermes → ~/.hermes/hermes-agent), silently running
    # the child against a foreign codebase/memory. For ask_agent the child
    # must inherit the parent's process identity exactly: same sys.executable,
    # same hermes_cli module. The kanban dispatcher is different — it runs as
    # a long-lived daemon and *should* find the operator's hermes; ask_agent
    # is a synchronous in-tree call. See ARCHITECTURE-2.0.md bug TD-4.
    import sys as _sys
    argv0 = [_sys.executable, "-m", "hermes_cli.main"]

    prompt = question.strip()
    if context and context.strip():
        prompt = f"{prompt}\n\n## Context from the manager\n{context.strip()}"

    env = dict(os.environ)
    # THE load-bearing line: pin the child to the profile's home so
    # config.yaml/.env/state.db resolve to the PROFILE, not the ambient
    # orchestrator home. Mirrors kanban_db._default_spawn.
    env["HERMES_HOME"] = str(profile_dir)
    env["HERMES_PROFILE"] = canon
    # Defensive: the child must never inherit a kanban-task identity from
    # an orchestrator that happens to be a kanban worker itself.
    for stale in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID",
                  "HERMES_KANBAN_CLAIM_LOCK", "HERMES_KANBAN_WORKSPACE"):
        env.pop(stale, None)

    cmd = [
        *argv0,
        "-p", canon,
        "chat",
        "-Q",          # machine-readable stdout (no banner/spinner/previews)
        "-q", prompt,  # single-query mode: answer and exit
    ]

    cap = timeout_seconds or _ask_agent_timeout()
    started = _time.monotonic()
    # Register the in-flight call so list_agents reports this profile as
    # busy while we're blocked in subprocess.run (TD-5). Released in the
    # finally below — covers success, TimeoutError, and RuntimeError paths.
    _register_active_ask(canon)
    try:
        proc = subprocess.run(  # noqa: S603 — argv is a fixed list built above
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=cap,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(
            f"ask_agent({canon}) exceeded {cap}s. For long work use "
            f"assign_task (async kanban path) instead of a synchronous ask."
        )
    finally:
        _release_active_ask(canon)
    elapsed = round(_time.monotonic() - started, 1)

    answer = (proc.stdout or "").strip()
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip()[-800:]
        raise RuntimeError(
            f"ask_agent({canon}) subprocess failed (exit {proc.returncode}). "
            f"stderr tail: {tail or '(empty)'}"
        )
    if not answer:
        tail = (proc.stderr or "").strip()[-400:]
        raise RuntimeError(
            f"ask_agent({canon}) returned empty output. stderr tail: {tail or '(empty)'}"
        )

    return {
        "agent": canon,
        "question": question.strip(),
        "answer": answer,
        "elapsed_seconds": elapsed,
    }


# ===========================================================================
# Tool handlers (registry.register targets). Each returns a JSON string.
# ===========================================================================

def _ok(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _err(msg: str, **extra) -> str:
    return json.dumps({"error": str(msg), **extra}, ensure_ascii=False, default=str)


def _handle_list_agents(args: dict, **kw) -> str:
    agents = list_agents()
    return _ok({"agents": agents, "count": len(agents)})


def _handle_create_agent(args: dict, **kw) -> str:
    try:
        result = create_agent(
            name=args.get("name", "").strip(),
            role=args.get("role"),
            description=args.get("description"),
            model=args.get("model"),
            provider=args.get("provider"),
            base_url=args.get("base_url"),
            api_key=args.get("api_key"),
            api_key_env=args.get("api_key_env"),
            api_mode=args.get("api_mode"),
        )
        return _ok({"created": result})
    except Exception as e:
        logger.exception("create_agent failed")
        return _err(str(e))


def _handle_set_agent_model(args: dict, **kw) -> str:
    try:
        result = set_agent_model(
            args.get("name", "").strip(),
            provider=args.get("provider"),
            model=args.get("model"),
            base_url=args.get("base_url"),
            api_key=args.get("api_key"),
            api_key_env=args.get("api_key_env"),
            api_mode=args.get("api_mode"),
        )
        return _ok({"updated": result})
    except Exception as e:
        logger.exception("set_agent_model failed")
        return _err(str(e))


def _handle_delete_agent(args: dict, **kw) -> str:
    try:
        result = delete_agent(args.get("name", "").strip(), force=bool(args.get("force", False)))
        return _ok(result)
    except Exception as e:
        logger.exception("delete_agent failed")
        return _err(str(e))


def _handle_agent_status(args: dict, **kw) -> str:
    try:
        return _ok(agent_status(args.get("name", "").strip()))
    except Exception as e:
        logger.exception("agent_status failed")
        return _err(str(e))


def _handle_assign_task(args: dict, **kw) -> str:
    try:
        result = assign_task(
            args.get("agent", "").strip(),
            args.get("goal", "").strip(),
            parent_task_id=args.get("parent_task_id"),
            priority=int(args.get("priority", 0) or 0),
            title=args.get("title"),
        )
        return _ok(result)
    except Exception as e:
        logger.exception("assign_task failed")
        return _err(str(e))


def _handle_ask_agent(args: dict, **kw) -> str:
    try:
        timeout_raw = args.get("timeout_seconds")
        result = ask_agent(
            args.get("agent", "").strip(),
            args.get("question", "").strip(),
            context=args.get("context"),
            timeout_seconds=int(timeout_raw) if timeout_raw else None,
        )
        return _ok(result)
    except TimeoutError as e:
        return _err(str(e), kind="timeout")
    except Exception as e:
        logger.exception("ask_agent failed")
        return _err(str(e))


def _worker_wake_profiles() -> set[str]:
    """Configured worker profiles allowed to request a Kanban wake."""
    try:
        from hermes_cli.config import load_config

        values = ((load_config().get("kanban") or {}).get("mailbox") or {}).get(
            "worker_wake_profiles", []
        )
        if not isinstance(values, list):
            return set()
        return {str(value).strip() for value in values if str(value).strip()}
    except Exception:
        return set()


def _mailbox_denied(reason: str) -> str:
    return json.dumps(
        {"success": False, "allowed": False, "reason": reason},
        ensure_ascii=False,
    )


def _mailbox_expected_send_denial(exc: ValueError) -> Optional[str]:
    """Map known low-level validation failures to body-free policy reasons."""
    message = str(exc)
    if message.startswith("mailbox body exceeds "):
        return "mailbox body exceeds configured size limit"
    if message.startswith("mailbox idempotency conflict:"):
        return "mailbox idempotency conflict"
    return None


def _handle_message_agent(args: dict, **kw) -> str:
    """Authorize and persist a soft message in one SQLite write txn."""
    from agent.mailbox_principal import MailboxPrincipal

    principal = kw.get("mailbox_principal")
    task_id = str(args.get("task_id") or "").strip()
    recipient = str(args.get("agent") or "").strip()
    body = args.get("body")
    idempotency_key = str(args.get("idempotency_key") or "").strip()
    kind = str(args.get("kind") or "guidance").strip().lower()
    board_arg = args.get("board")
    board = str(board_arg).strip().lower() if board_arg is not None else None
    wake_raw = args.get("wake")

    try:
        from hermes_cli.config import load_config

        mailbox_cfg = ((load_config().get("kanban") or {}).get("mailbox") or {})
    except Exception:
        mailbox_cfg = {}
    max_body_bytes = int(mailbox_cfg.get("max_body_bytes", 16_384))
    worker_wake_profiles = (
        _worker_wake_profiles()
        if isinstance(principal, MailboxPrincipal) and principal.kind == "worker"
        else set()
    )

    kb = _kanban()
    conn = None
    try:
        trusted_principal = isinstance(principal, MailboxPrincipal)
        principal_board = None
        requested_board = None
        board_error = None
        if trusted_principal:
            # Both root managers and dispatcher workers are permanently pinned
            # at grant time.  The model's board argument is only a routing hint;
            # it can never select a different authority database.
            try:
                principal_board = kb._normalize_board_slug(principal.board)
                if not principal_board or not principal.db_path:
                    raise ValueError("principal is missing its board or DB pin")
                if board is not None:
                    requested_board = kb._normalize_board_slug(board)
            except (TypeError, ValueError):
                board_error = "invalid or missing trusted board pin"
            conn = (
                kb.connect(db_path=Path(str(principal.db_path)))
                if principal.db_path
                else kb.connect()
            )
        else:
            # An untrusted caller has no authority regardless of this routing
            # hint.  Never let model input select even the denial-audit DB.
            conn = kb.connect()

        actor_identity = (
            principal.actor_identity
            if isinstance(principal, MailboxPrincipal)
            else "untrusted"
        )
        actor_kind = principal.kind if isinstance(principal, MailboxPrincipal) else "none"
        run_id = (
            principal.run_id
            if isinstance(principal, MailboxPrincipal) and principal.kind == "worker"
            else None
        )

        with kb.write_txn(conn):
            from agent.redact import redact_sensitive_text

            safe_recipient = redact_sensitive_text(recipient, force=True)
            safe_idempotency_key = redact_sensitive_text(idempotency_key, force=True)
            recipient_contains_secret = safe_recipient != recipient
            idempotency_contains_secret = safe_idempotency_key != idempotency_key
            denial_reason = None
            target = None
            audited_task_id = None
            supplied_recipient = None
            wake_requested = False

            # Authenticate the dispatch capability before interpreting model
            # input.  An untrusted malformed call stays default-denied instead
            # of learning which validation branch it reached.
            if not trusted_principal:
                denial_reason = "missing trusted mailbox principal"
            elif board_error is not None:
                denial_reason = board_error
            elif requested_board is not None and requested_board != principal_board:
                denial_reason = f"{principal.kind} cannot cross board boundary"
            elif principal.kind == "worker":
                current_pid = os.getpid()
                if principal.worker_pid != current_pid:
                    denial_reason = "worker principal is not owned by this process"
                else:
                    owner = conn.execute(
                        """
                        SELECT t.tenant
                          FROM tasks t
                          JOIN task_runs r ON r.id = t.current_run_id
                         WHERE t.id = ? AND t.status = 'running'
                           AND t.assignee = ?
                           AND t.current_run_id = ? AND r.id = ?
                           AND r.task_id = t.id AND r.profile = ?
                           AND r.status = 'running' AND r.ended_at IS NULL
                           AND t.worker_pid = ? AND r.worker_pid = ?
                        """,
                        (
                            principal.task_id,
                            principal.sender_profile,
                            principal.run_id,
                            principal.run_id,
                            principal.sender_profile,
                            current_pid,
                            current_pid,
                        ),
                    ).fetchone()
                    if owner is None or owner["tenant"] != principal.tenant:
                        denial_reason = "worker principal is not the exact live current run"
            elif principal.kind != "manager":
                denial_reason = "unsupported mailbox principal kind"

            if denial_reason is None:
                if wake_raw is None:
                    wake_requested = kind in {"guidance", "question"}
                elif isinstance(wake_raw, bool):
                    wake_requested = wake_raw
                else:
                    denial_reason = "wake must be a boolean"

            if denial_reason is None:
                if not task_id or not recipient or not idempotency_key:
                    denial_reason = "agent, task_id, and idempotency_key are required"
                elif not isinstance(body, str) or not body.strip():
                    denial_reason = "body must be non-empty text"
                elif kind not in {"guidance", "question", "info"}:
                    denial_reason = "kind must be guidance, question, or info"
                elif recipient_contains_secret:
                    denial_reason = "recipient profile contains sensitive data"
                elif idempotency_contains_secret:
                    denial_reason = "idempotency key contains sensitive data"

            if denial_reason is None:
                try:
                    from hermes_cli import profiles

                    supplied_recipient = profiles.normalize_profile_name(safe_recipient)
                    profiles.validate_profile_name(supplied_recipient)
                    if not profiles.profile_exists(supplied_recipient):
                        denial_reason = "recipient profile does not exist"
                except Exception:
                    denial_reason = "recipient profile is invalid"

            if denial_reason is None:
                target = conn.execute(
                    "SELECT id, status, assignee, tenant FROM tasks WHERE id = ?",
                    (task_id,),
                ).fetchone()
                audited_task_id = task_id if target is not None else None
                if target is None:
                    denial_reason = "unknown target task"
                elif principal.kind == "manager":
                    if principal.tenant is None and target["tenant"] is not None:
                        denial_reason = "manager has no authority for tenant-scoped target"
                    elif principal.tenant is not None and target["tenant"] != principal.tenant:
                        denial_reason = "manager cannot cross tenant boundary"
                elif target["tenant"] != principal.tenant:
                    denial_reason = "worker cannot cross tenant boundary"
                elif (
                    wake_requested
                    and principal.sender_profile not in worker_wake_profiles
                ):
                    denial_reason = "worker profile is not allowed to request wake"

            if denial_reason is None and target is not None:
                expected_recipient = kb._canonical_assignee(target["assignee"])
                if not expected_recipient or supplied_recipient != expected_recipient:
                    denial_reason = "recipient does not match task assignee"
                elif target["status"] in {"done", "archived"}:
                    denial_reason = "target task is terminal"

            sent = None
            if denial_reason is None:
                conn.execute("SAVEPOINT message_agent_send")
                try:
                    sent = kb._send_mailbox_message_in_txn(
                        conn,
                        task_id=task_id,
                        actor_identity=principal.actor_identity,
                        actor_kind=principal.kind,
                        sender_profile=principal.sender_profile,
                        recipient_profile=supplied_recipient,
                        kind=kind,
                        body=body,
                        wake_requested=wake_requested,
                        idempotency_key=idempotency_key,
                        max_body_bytes=max_body_bytes,
                    )
                except ValueError as exc:
                    safe_reason = _mailbox_expected_send_denial(exc)
                    conn.execute("ROLLBACK TO SAVEPOINT message_agent_send")
                    conn.execute("RELEASE SAVEPOINT message_agent_send")
                    if safe_reason is None:
                        raise
                    denial_reason = safe_reason
                else:
                    conn.execute("RELEASE SAVEPOINT message_agent_send")

            if denial_reason is not None:
                kb._append_mailbox_audit_in_txn(
                    conn,
                    task_id=audited_task_id,
                    run_id=run_id,
                    message_id=None,
                    actor_identity=actor_identity,
                    actor_kind=actor_kind,
                    recipient_profile=(
                        None if recipient_contains_secret else safe_recipient or None
                    ),
                    action="message_agent.send",
                    allowed=False,
                    reason=denial_reason,
                )
                result = _mailbox_denied(denial_reason)
            else:
                kb._append_mailbox_audit_in_txn(
                    conn,
                    task_id=task_id,
                    run_id=run_id,
                    message_id=sent.message_id,
                    actor_identity=principal.actor_identity,
                    actor_kind=principal.kind,
                    recipient_profile=supplied_recipient,
                    action="message_agent.send",
                    allowed=True,
                    reason="authorized",
                )
                result = json.dumps(
                    {
                        "success": True,
                        "allowed": True,
                        "message_id": sent.message_id,
                        "created": sent.created,
                        "stored": sent.created,
                        "redacted": sent.redacted,
                        "task_id": task_id,
                        "agent": supplied_recipient,
                        "kind": kind,
                        "delivery_state": sent.delivery_state,
                        "wake_requested": wake_requested,
                        "wake_effect": sent.wake_effect,
                        "task_status": sent.task_status,
                    },
                    ensure_ascii=False,
                )
        return result
    except Exception as exc:
        logger.warning("message_agent failed: %s", type(exc).__name__)
        return _mailbox_denied("mailbox send failed")
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


# ===========================================================================
# /team slash command — shared text formatter for CLI + gateway + Telegram.
# Mirrors hermes_cli.kanban.run_slash: takes the text after "/team " and
# returns a single formatted string.
# ===========================================================================

def _format_agents_table(agents: list) -> str:
    """Render a compact table of agents for terminal/messaging output."""
    if not agents:
        return "(._.) No agent profiles found. Create one with: /team create <name>"
    lines = ["👥 Agent team", ""]
    for a in agents:
        status_emoji = "🟢" if not a["busy"] else "🔴"
        role = a.get("role") or "—"
        model = a.get("model") or "—"
        provider = a.get("provider") or "—"
        active = a.get("active_tasks", 0)
        busy_note = f" ({active} task{'s' if active != 1 else ''})" if active else ""
        desc = a.get("description") or ""
        # Strip the role stamp from the displayed description.
        if desc.lower().startswith("role:"):
            desc = desc.split("—", 1)[-1].strip() if "—" in desc else ""
        desc_short = (desc[:48] + "…") if len(desc) > 48 else desc
        marker = " [default]" if a.get("is_default") else ""
        lines.append(
            f"{status_emoji} {a['name']}{marker} · {role} · {model}/{provider}{busy_note}"
        )
        if desc_short:
            lines.append(f"     {desc_short}")
    return "\n".join(lines)


def run_team_slash(text: str) -> str:
    """Handle ``/team <subcommand> [args]`` — returns a formatted string.

    This is the shared entry point called by both cli.py (terminal) and
    gateway/run.py (Telegram/Slack/etc). Keeps parsing logic in one place.
    """
    import shlex

    text = (text or "").strip()
    # Strip leading slash and command name if echoed back.
    if text.startswith("/"):
        text = text.lstrip("/")
    if text.lower().startswith("team"):
        text = text[4:].lstrip()

    try:
        parts = shlex.split(text)
    except ValueError:
        parts = text.split()

    if not parts or parts[0] in ("-h", "--help", "help"):
        return _team_help()

    cmd = parts[0].lower()
    rest = parts[1:]

    try:
        if cmd in ("list", "ls"):
            return _format_agents_table(list_agents())

        if cmd == "status":
            if not rest:
                return "Usage: /team status <name>"
            return _format_status(rest[0])

        if cmd == "create":
            return _team_create(rest)

        if cmd == "delete":
            return _team_delete(rest)

        if cmd in ("models", "model"):
            return _team_models(rest)

        if cmd == "assign":
            return _team_assign(rest)

        return f"Unknown /team subcommand: {cmd}\n\n" + _team_help()
    except Exception as exc:
        logger.exception("/team command failed")
        return f"(._.) /team error: {exc}"


def _team_help() -> str:
    return (
        "👥 /team — manage your team of specialized agent profiles\n\n"
        "Subcommands:\n"
        "  /team list                          — show all agents + busy status\n"
        "  /team status <name>                 — detail on one agent\n"
        "  /team create <name> [options]       — create a new agent\n"
        "  /team models <name> [options]       — change an agent's provider/model/key\n"
        "  /team assign <agent> <goal...>      — assign a kanban task (async)\n"
        "  /team delete <name> [--force]       — remove an agent profile\n\n"
        "create/models options:\n"
        "  --role <r>         researcher | coder | reviewer | analyst | writer\n"
        "  --model <m>        e.g. glm-5.1\n"
        "  --provider <p>     e.g. zai, anthropic, openai\n"
        "  --base-url <url>\n"
        "  --api-key <key>    secret (NEVER logged)\n"
        "  --api-key-env <e>  var name, e.g. GLM_API_KEY\n"
        "  --description <d>  free-form identity\n\n"
        "Agents are persistent profiles — they keep their memory and context "
        "across tasks, unlike delegate_task subagents."
    )


def _team_create(args: list) -> str:
    """Parse /team create flags and call create_agent."""
    name, opts = _parse_flags(args, ["name"])
    required = ["name"]
    missing = [k for k in required if not opts.get(k)]
    if missing:
        return "Usage: /team create <name> [--role R] [--model M] [--provider P] " \
               "[--base-url U] [--api-key K] [--api-key-env E] [--description D]"

    result = create_agent(
        name=opts["name"],
        role=opts.get("role"),
        description=opts.get("description"),
        model=opts.get("model"),
        provider=opts.get("provider"),
        base_url=opts.get("base-url"),
        api_key=opts.get("api-key"),
        api_key_env=opts.get("api-key-env"),
    )
    role_str = f" (role: {result['role']})" if result.get("role") else ""
    model_str = f" · {result['model']}/{result['provider']}" if result.get("model") else ""
    return f"✨ Created agent '{result['name']}'{role_str}{model_str}\n   path: {result['path']}"


def _team_models(args: list) -> str:
    name, opts = _parse_flags(args, ["name"])
    if not opts.get("name"):
        return "Usage: /team models <name> [--provider P] [--model M] " \
               "[--base-url U] [--api-key K] [--api-key-env E]"
    result = set_agent_model(
        opts["name"],
        provider=opts.get("provider"),
        model=opts.get("model"),
        base_url=opts.get("base-url"),
        api_key=opts.get("api-key"),
        api_key_env=opts.get("api-key-env"),
    )
    upd = ", ".join(result["updated"]) if result["updated"] else "nothing"
    return f"🔄 Updated '{result['name']}': now {result['model']}/{result['provider']} (changed: {upd})"


def _team_delete(args: list) -> str:
    name, opts = _parse_flags(args, ["name"])
    if not opts.get("name"):
        return "Usage: /team delete <name> [--force]"
    force = "--force" in args or opts.get("force") in (True, "true", "True")
    result = delete_agent(opts["name"], force=force)
    if not result["deleted"]:
        return f"⚠️ Refused: {result['blocked_reason']}"
    note = " (had active tasks)" if result.get("had_active_tasks") else ""
    return f"🗑️ Deleted agent '{result['name']}'{note}"


def _team_assign(args: list) -> str:
    if len(args) < 2:
        return "Usage: /team assign <agent> <goal...> [--title T] [--parent ID]"
    agent = args[0]
    # Collect positional words after agent until a flag, then parse flags.
    goal_words = []
    i = 1
    while i < len(args) and not args[i].startswith("--"):
        goal_words.append(args[i])
        i += 1
    flag_args = args[i:]
    _, opts = _parse_flags(flag_args, [])
    goal = " ".join(goal_words)
    if not goal:
        return "Usage: /team assign <agent> <goal...> [--title T] [--parent ID]"
    result = assign_task(
        agent, goal,
        title=opts.get("title"),
        parent_task_id=opts.get("parent"),
    )
    return (
        f"📌 Assigned task {result['task_id']} to '{result['agent']}'\n"
        f"   title: {result['title']}\n"
        f"   The kanban dispatcher will spawn `hermes -p {result['agent']}` "
        f"to work on it. Ensure the dispatcher is running "
        f"(kanban.dispatch_in_gateway: true is the default)."
    )


def _format_status(name: str) -> str:
    st = agent_status(name)
    status_emoji = "🔴 busy" if st["busy"] else "🟢 idle"
    lines = [
        f"📊 Agent '{st['name']}' — {status_emoji}",
        f"   role: {st.get('role') or '—'}",
        f"   model: {st.get('model') or '—'} / {st.get('provider') or '—'}",
        f"   gateway: {'running' if st.get('gateway_running') else 'off'}",
    ]
    if st.get("active_tasks"):
        lines.append("   active tasks:")
        for t in st["active_tasks"][:5]:
            tid = t.get("id", "?")
            title = (t.get("title") or "")[:60]
            lines.append(f"     • {tid}: {title}")
    if st.get("description"):
        lines.append(f"   {st['description']}")
    return "\n".join(lines)


def _parse_flags(args: list, positionals: list):
    """Tiny --flag value parser. Returns (positional_values, flag_dict).

    positionals is a list of positional-names to consume in order from the
    front of args (before any --flag). Boolean flags (--force) get True.
    """
    opts: dict = {}
    pos_values: dict = {}
    i = 0
    # Positionals come first, before any --flag.
    pi = 0
    while i < len(args) and pi < len(positionals) and not args[i].startswith("--"):
        pos_values[positionals[pi]] = args[i]
        i += 1
        pi += 1
    # Flags.
    while i < len(args):
        tok = args[i]
        if tok == "--force":
            opts["force"] = True
            i += 1
            continue
        if tok.startswith("--") and i + 1 < len(args) and not args[i + 1].startswith("--"):
            opts[tok[2:]] = args[i + 1]
            i += 2
        elif tok.startswith("--"):
            opts[tok[2:]] = True
            i += 1
        else:
            i += 1
    opts.update(pos_values)
    return None, opts


# ===========================================================================
# Tool schemas
# ===========================================================================

_LIST_AGENTS_SCHEMA = {
    "name": "list_agents",
    "description": (
        "List all agent profiles in the team with live busy status. Use "
        "this FIRST to see who exists and who's available before assigning "
        "work. Returns each agent's name, role, model, provider, and "
        "whether it's currently busy (active kanban tasks or running gateway)."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

_CREATE_AGENT_SCHEMA = {
    "name": "create_agent",
    "description": (
        "Create a new specialized agent profile for the team. This is the "
        "PRIMARY tool for adding a new agent — call it DIRECTLY; do NOT "
        "shell out, do NOT read .env manually, do NOT use execute_code. "
        "Each agent is a persistent Hermes profile with its own "
        "model/provider/api key/memory/persona — it does NOT disappear "
        "after a task (unlike delegate_task subagents). Optionally seeds "
        "a role from the catalog (researcher/coder/reviewer/analyst/writer). "
        "For synchronous short chains use delegate_task instead; create_agent "
        "is for agents you'll reuse across many tasks.\n\n"
        "IMPORTANT about API keys: pass the key value DIRECTLY in the "
        "`api_key` parameter along with its env-var NAME in `api_key_env` "
        "(e.g. api_key='sk-...', api_key_env='GLM_API_KEY'). This tool "
        "writes the secret into the new profile's .env securely (mode 0600) "
        "by absolute path. Do NOT read the orchestrator's .env, do NOT "
        "copy .env with terminal, do NOT clone an existing profile just "
        "to inherit a key — pass api_key + api_key_env and this tool "
        "handles the rest."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Profile name: lowercase, [a-z0-9_-], not 'default'.",
                "pattern": "^[a-z0-9][a-z0-9_-]{0,63}$",
            },
            "role": {
                "type": "string",
                "description": "Optional specialization from ROLE_TOOLSET_MAP (researcher, coder, reviewer, analyst, writer). Seeds toolset + persona.",
            },
            "description": {"type": "string", "description": "Free-form identity; defaults to the role's description."},
            "model": {"type": "string", "description": "Model id, e.g. 'glm-5.1'."},
            "provider": {"type": "string", "description": "Provider name, e.g. 'zai', 'anthropic', 'openai'."},
            "base_url": {"type": "string", "description": "Provider API base URL."},
            "api_mode": {"type": "string", "description": "API mode if non-default (chat_completions, codex_responses, ...)."},
            "api_key": {"type": "string", "description": "Secret API key value — pass DIRECTLY here, do NOT read .env manually. NEVER logged."},
            "api_key_env": {
                "type": "string",
                "description": "Env var NAME to store api_key under (e.g. 'GLM_API_KEY', 'ANTHROPIC_API_KEY'). Required if api_key is set.",
            },
        },
        "required": ["name"],
    },
}

_SET_AGENT_MODEL_SCHEMA = {
    "name": "set_agent_model",
    "description": (
        "Change an existing agent's model/provider/api key WITHOUT recreating "
        "it. Memories, sessions, and persona are preserved. Use this when a "
        "provider isn't working well and you want to swap it. Call this tool "
        "DIRECTLY with the api_key parameter — do NOT read .env manually, "
        "do NOT use terminal/execute_code to rewrite the key."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Existing agent profile name."},
            "provider": {"type": "string"},
            "model": {"type": "string"},
            "base_url": {"type": "string"},
            "api_mode": {"type": "string"},
            "api_key": {"type": "string", "description": "New secret value — pass DIRECTLY here, do NOT read .env manually. NEVER logged."},
            "api_key_env": {"type": "string", "description": "Env var NAME to store under. Required if api_key is set."},
        },
        "required": ["name"],
    },
}

_DELETE_AGENT_SCHEMA = {
    "name": "delete_agent",
    "description": (
        "Delete an agent profile. Refuses if the agent has running kanban "
        "tasks unless force=true. This is destructive — the profile's "
        "memories, sessions, and config are removed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "force": {"type": "boolean", "description": "Delete even if the agent has active tasks.", "default": False},
        },
        "required": ["name"],
    },
}

_AGENT_STATUS_SCHEMA = {
    "name": "agent_status",
    "description": (
        "Detailed status for one agent: model, provider, active tasks, "
        "busy flag, recent activity. Deeper than list_agents."
    ),
    "parameters": {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
}

_ASK_AGENT_SCHEMA = {
    "name": "ask_agent",
    "description": (
        "Synchronously ask a named agent profile a question and get its "
        "answer back in THIS turn (BLOCKING). The agent runs as its own "
        "process with its own config/model/api-key/memory/persona — the "
        "profile's config.yaml is the source of truth for its toolsets "
        "and behavior (no role override here; roles are only for "
        "ephemeral delegate_task children). Use ask_agent for dialogue "
        "and dependent steps ('get finance's numbers, then have writer "
        "format them'). For long or parallel work use assign_task "
        "(async kanban) instead — ask_agent blocks you until the agent "
        "answers or the timeout hits."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "agent": {
                "type": "string",
                "description": "Target agent profile name (must exist — see list_agents).",
            },
            "question": {
                "type": "string",
                "description": "The question or instruction for the agent.",
            },
            "context": {
                "type": "string",
                "description": (
                    "Optional context to pass along (prior findings, "
                    "constraints, data). The agent does NOT see your "
                    "conversation — include everything it needs."
                ),
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Max seconds to wait (default from agent_manager.ask_timeout_seconds, 600).",
            },
        },
        "required": ["agent", "question"],
    },
}

_ASSIGN_TASK_SCHEMA = {
    "name": "assign_task",
    "description": (
        "Assign a goal to an agent via the kanban board (ASYNC coordination). "
        "The kanban dispatcher will spawn 'hermes -p <agent>' to work on it. "
        "Use parent_task_id to chain: the task stays 'todo' until its parent "
        "completes, then auto-promotes to 'ready'. For synchronous immediate "
        "results within one turn, use delegate_task(role=...) instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "agent": {"type": "string", "description": "Target agent profile name."},
            "goal": {"type": "string", "description": "The task goal/prompt for the agent."},
            "title": {"type": "string", "description": "Short title; defaults to truncated goal."},
            "parent_task_id": {"type": "string", "description": "Optional: wait for this task to complete first."},
            "priority": {"type": "integer", "description": "Higher = sooner.", "default": 0},
        },
        "required": ["agent", "goal"],
    },
}

_MESSAGE_AGENT_SCHEMA = {
    "name": "message_agent",
    "description": (
        "Soft-deliver a durable message to an assigned agent's Kanban task. "
        "Guidance and questions request wake by default; info does not."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "agent": {"type": "string", "description": "Assigned recipient profile."},
            "task_id": {"type": "string", "description": "Target Kanban task id."},
            "body": {"type": "string", "description": "Message text."},
            "idempotency_key": {
                "type": "string",
                "description": "Stable unique key for safe retries.",
            },
            "kind": {
                "type": "string",
                "enum": ["guidance", "question", "info"],
                "default": "guidance",
            },
            "wake": {
                "type": "boolean",
                "description": (
                    "Override wake request. Defaults true for guidance/question "
                    "and false for info."
                ),
            },
            "board": {"type": "string", "description": "Optional board slug."},
        },
        "required": ["agent", "task_id", "body", "idempotency_key"],
    },
}


# ===========================================================================
# Availability check (model the kanban pattern).
# ===========================================================================

def _check_agent_manager_mode() -> bool:
    """Gate the toolset: true if the active profile opted into agent_manager.

    Mirrors tools/kanban_tools.py:_check_kanban_mode — checks the active
    profile's config for the agent_manager toolset. This keeps the schema
    footprint zero for profiles that aren't orchestrators.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        toolsets = cfg.get("toolsets") or []
        if isinstance(toolsets, str):
            toolsets = [toolsets]
        if "agent_manager" in toolsets:
            return True
        # Also accept if explicitly enabled via tools config.
        enabled = (cfg.get("tools", {}) or {}).get("enabled") or []
        if "agent_manager" in enabled:
            return True
    except Exception:
        pass
    return False


def _check_message_agent_mode() -> bool:
    """Visible to dispatcher workers and explicitly configured coordinators."""
    if (os.environ.get("HERMES_KANBAN_TASK") or "").strip():
        return True
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        configured = cfg.get("toolsets") or []
        if isinstance(configured, str):
            configured = [configured]
        enabled = ((cfg.get("tools") or {}).get("enabled") or [])
        return bool({"agent_manager", "kanban"} & (set(configured) | set(enabled)))
    except Exception:
        return False


# ===========================================================================
# Registration (auto-discovered by model_tools.discover_builtin_tools).
# ===========================================================================

from tools.registry import registry  # noqa: E402

registry.register(
    name="list_agents",
    toolset="agent_manager",
    schema=_LIST_AGENTS_SCHEMA,
    handler=_handle_list_agents,
    check_fn=_check_agent_manager_mode,
    emoji="👥",
)
registry.register(
    name="create_agent",
    toolset="agent_manager",
    schema=_CREATE_AGENT_SCHEMA,
    handler=_handle_create_agent,
    check_fn=_check_agent_manager_mode,
    emoji="✨",
)
registry.register(
    name="set_agent_model",
    toolset="agent_manager",
    schema=_SET_AGENT_MODEL_SCHEMA,
    handler=_handle_set_agent_model,
    check_fn=_check_agent_manager_mode,
    emoji="🔄",
)
registry.register(
    name="delete_agent",
    toolset="agent_manager",
    schema=_DELETE_AGENT_SCHEMA,
    handler=_handle_delete_agent,
    check_fn=_check_agent_manager_mode,
    emoji="🗑️",
)
registry.register(
    name="agent_status",
    toolset="agent_manager",
    schema=_AGENT_STATUS_SCHEMA,
    handler=_handle_agent_status,
    check_fn=_check_agent_manager_mode,
    emoji="📊",
)
registry.register(
    name="assign_task",
    toolset="agent_manager",
    schema=_ASSIGN_TASK_SCHEMA,
    handler=_handle_assign_task,
    check_fn=_check_agent_manager_mode,
    emoji="📌",
)
registry.register(
    name="ask_agent",
    toolset="agent_manager",
    schema=_ASK_AGENT_SCHEMA,
    handler=_handle_ask_agent,
    check_fn=_check_agent_manager_mode,
    emoji="🗣️",
)
registry.register(
    name="message_agent",
    toolset="agent_manager",
    schema=_MESSAGE_AGENT_SCHEMA,
    handler=_handle_message_agent,
    check_fn=_check_message_agent_mode,
    emoji="✉️",
)
