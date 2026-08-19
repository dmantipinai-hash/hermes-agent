"""Trusted, per-agent authorization identity for the A2 mailbox."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


@dataclass(frozen=True)
class MailboxPrincipal:
    """Immutable capability created only by a trusted root entrypoint."""

    kind: Literal["manager", "worker"]
    actor_identity: str
    sender_profile: str
    session_id: str
    tenant: Optional[str] = None
    board: Optional[str] = None
    db_path: Optional[str] = None
    task_id: Optional[str] = None
    run_id: Optional[int] = None
    worker_pid: Optional[int] = None


@dataclass(frozen=True)
class WorkerPrincipalProbe:
    """Result of validating dispatcher-owned worker startup state."""

    state: Literal["granted", "pending_registration", "denied"]
    principal: Optional[MailboxPrincipal] = None


_WORKER_ENV_KEYS = (
    "HERMES_KANBAN_TASK",
    "HERMES_KANBAN_RUN_ID",
    "HERMES_KANBAN_DB",
    "HERMES_KANBAN_BOARD",
)

_WORKER_STARTUP_GRANT_POLL_SECONDS = 0.05
_WORKER_STARTUP_GRANT_CUSHION_SECONDS = 0.05


def _worker_principal_from_environment(session_id: str) -> WorkerPrincipalProbe:
    """Probe the exact dispatcher-owned live run and PID registration."""
    values = {key: (os.environ.get(key) or "").strip() for key in _WORKER_ENV_KEYS}
    if not all(values.values()):
        return WorkerPrincipalProbe("denied")
    profile = (os.environ.get("HERMES_PROFILE") or "").strip()
    if not profile:
        return WorkerPrincipalProbe("denied")
    try:
        run_id = int(values["HERMES_KANBAN_RUN_ID"])
    except (TypeError, ValueError):
        return WorkerPrincipalProbe("denied")
    if run_id <= 0:
        return WorkerPrincipalProbe("denied")

    from hermes_cli import kanban_db as kb

    db_path = Path(values["HERMES_KANBAN_DB"]).expanduser().resolve()
    worker_pid = os.getpid()
    conn = None
    try:
        conn = kb.connect(db_path=db_path)
        row = conn.execute(
            """
            SELECT t.tenant, t.worker_pid AS task_worker_pid,
                   r.worker_pid AS run_worker_pid
              FROM tasks t
              JOIN task_runs r ON r.id = t.current_run_id
             WHERE t.id = ? AND t.status = 'running' AND t.assignee = ?
               AND t.current_run_id = ? AND r.id = ?
               AND r.task_id = t.id AND r.profile = ?
               AND r.status = 'running' AND r.ended_at IS NULL
            """,
            (
                values["HERMES_KANBAN_TASK"],
                profile,
                run_id,
                run_id,
                profile,
            ),
        ).fetchone()
        if row is None:
            return WorkerPrincipalProbe("denied")
        task_pid = row["task_worker_pid"]
        run_pid = row["run_worker_pid"]
        task_unregistered = task_pid is None or int(task_pid) == 0
        run_unregistered = run_pid is None or int(run_pid) == 0
        if task_unregistered and run_unregistered:
            return WorkerPrincipalProbe("pending_registration")
        if task_pid != worker_pid or run_pid != worker_pid:
            return WorkerPrincipalProbe("denied")
        task_id = values["HERMES_KANBAN_TASK"]
        return WorkerPrincipalProbe(
            "granted",
            MailboxPrincipal(
                kind="worker",
                actor_identity=f"worker:{profile}:{task_id}:{run_id}",
                sender_profile=profile,
                session_id=session_id,
                tenant=row["tenant"],
                board=values["HERMES_KANBAN_BOARD"],
                db_path=str(db_path),
                task_id=task_id,
                run_id=run_id,
                worker_pid=worker_pid,
            ),
        )
    except Exception:
        return WorkerPrincipalProbe("denied")
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _worker_principal_after_startup(
    session_id: str,
    *,
    monotonic=None,
    sleep=None,
) -> Optional[MailboxPrincipal]:
    """Wait only for exact-run PID registration, never identity mismatch.

    The scheduled-poll deadline mirrors SQLite's configured busy timeout.
    A connect/query can itself consume that timeout, so this does not claim a
    tighter wall-clock bound for time spent blocked inside SQLite.
    """
    from hermes_cli import kanban_db as kb

    monotonic_fn = monotonic or time.monotonic
    sleep_fn = sleep or time.sleep
    registration_window = (
        kb._resolve_busy_timeout_ms() / 1000.0
        + _WORKER_STARTUP_GRANT_CUSHION_SECONDS
    )
    deadline = None
    while True:
        probe = _worker_principal_from_environment(session_id)
        if probe.state == "granted":
            return probe.principal
        if probe.state == "denied":
            return None
        if deadline is None:
            deadline = monotonic_fn() + registration_window
        remaining = deadline - monotonic_fn()
        if remaining <= 0:
            return None
        sleep_fn(min(_WORKER_STARTUP_GRANT_POLL_SECONDS, remaining))


def grant_top_level_mailbox_principal(
    agent,
    *,
    platform: str,
    sender_profile: str,
    actor_identity: str,
) -> Optional[MailboxPrincipal]:
    """Grant authority to one root CLI/gateway agent, never its children."""
    if getattr(agent, "mailbox_principal", None) is not None:
        return agent.mailbox_principal
    session_id = str(getattr(agent, "session_id", "") or "").strip()
    if not session_id:
        return None

    # Board/DB pinning is also valid for an ordinary manager.  Only the task
    # marker declares a dispatcher worker; once present, every companion field
    # must validate or the grant fails closed.
    worker_marked = bool((os.environ.get("HERMES_KANBAN_TASK") or "").strip())
    if worker_marked:
        principal = _worker_principal_after_startup(session_id)
        if principal is None:  # never turn a malformed worker into a manager
            return None
    else:
        from hermes_cli import kanban_db as kb

        profile = str(sender_profile or "").strip() or "default"
        identity = str(actor_identity or "").strip() or f"{platform}:{profile}:{session_id}"
        board = kb._normalize_board_slug(kb.get_current_board()) or kb.DEFAULT_BOARD
        db_path = str(kb.kanban_db_path(board=board).expanduser().resolve())
        principal = MailboxPrincipal(
            kind="manager",
            actor_identity=identity,
            sender_profile=profile,
            session_id=session_id,
            board=board,
            db_path=db_path,
        )
    agent.mailbox_principal = principal
    return principal
