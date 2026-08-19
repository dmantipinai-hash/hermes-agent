"""KanbanMailboxRuntime: quiet worker continuation and message delivery.

Background listener for Kanban worker agents. Manages message arrival,
delivery batching, and wake continuation with model request interception.

Owned by AIAgent._kanban_mailbox_runtime, created only for worker principals.
"""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agent.mailbox_principal import MailboxPrincipal


logger = logging.getLogger("run_agent")


@dataclass
class MailboxDelivery:
    """A single mailbox message ready for delivery into agent context."""

    message_id: int
    task_id: str
    run_id: int
    claim_token: str
    kind: str  # 'guidance' | 'question' | 'info'
    body: str
    actor_identity: str
    sender_profile: str
    comment_id: int


@dataclass
class MailboxDeliveryBatch:
    """Batch of mailbox deliveries for a single agent turn."""

    deliveries: List[MailboxDelivery]
    # De-duplication: (message_id, run_id) tuples we've already delivered.
    delivered_keys: set[Tuple[int, int]]
    # If True, model should receive a synthetic message like:
    #   "You received a new message from the dispatcher: [guidance body]"
    include_in_context: bool


class KanbanMailboxRuntime:
    """
    Background runtime for Kanban worker mailbox listening.

    - Exactly one instance per worker agent.
    - Listener thread polls task_mailbox_messages for arrivals.
    - Claims are fenced with DB uniqueness + lease tokens.
    - In-memory queue batches deliveries into context or side-channel.
    - Idempotent: multiple init/shutdown calls are safe.
    """

    def __init__(self, principal: MailboxPrincipal, poll_interval: float = 0.5):
        if principal.kind != "worker":
            raise ValueError(
                f"KanbanMailboxRuntime requires worker principal, got {principal.kind}"
            )
        if not principal.task_id or not principal.run_id or not principal.db_path:
            raise ValueError("Worker principal missing task_id, run_id, or db_path")

        self.principal = principal
        self.db_path: str = principal.db_path
        self.task_id: str = principal.task_id
        self.run_id: int = principal.run_id
        self.recipient_profile: str = principal.sender_profile

        # Poll interval is configurable via kanban.mailbox.poll_interval_seconds
        # in config.yaml (design "Defaults" + "Performance contract").  The
        # initializer below reads the config and passes it in; 0.5 is the
        # documented fallback when the key is absent.
        self.poll_interval: float = poll_interval

        # Threading state
        self._listener_thread: Optional[threading.Thread] = None
        self._shutdown_requested: threading.Event = threading.Event()
        self._listener_ready: threading.Event = threading.Event()

        # DB connection (owned by listener thread)
        self._db_conn: Optional[sqlite3.Connection] = None

        # In-memory delivery queue (thread-safe)
        self._delivery_queue: queue.Queue[Optional[MailboxDelivery]] = queue.Queue()

        # Tracking what we've delivered (de-duplication)
        self._delivered_keys: set[Tuple[int, int]] = set()
        self._delivered_lock = threading.Lock()

        # Lease renewal: claim_token and message_ids for inflight messages.
        # Populated on claim, used by periodic renewal.  The DB layer's
        # renew_mailbox_message_leases already skips messages that have reached
        # model_response_received, so stale entries here are harmless.
        self._claim_token: Optional[str] = None
        self._inflight_message_ids: set = set()
        self._inflight_lock = threading.Lock()
        self._last_renewal_at: float = 0.0

        # Agent callback for batch extraction
        self._extract_batch_callback = None

        logger.info(
            f"KanbanMailboxRuntime initialized: task={self.task_id}, run={self.run_id}"
        )

    def start(self) -> None:
        """Start the listener thread. Idempotent."""
        if self._listener_thread is not None and self._listener_thread.is_alive():
            logger.debug("Listener thread already running, skipping start")
            return

        self._shutdown_requested.clear()
        self._listener_ready.clear()

        self._listener_thread = threading.Thread(
            target=self._listener_loop,
            name=f"kanban-listener-{self.task_id}-{self.run_id}",
            daemon=True,
        )
        self._listener_thread.start()

        # Wait for listener to be ready (or timeout)
        self._listener_ready.wait(timeout=5.0)
        if not self._listener_ready.is_set():
            logger.warning("Listener thread did not signal ready within 5s")

    def shutdown(self) -> None:
        """Request listener shutdown. Idempotent."""
        if self._shutdown_requested.is_set():
            logger.debug("Shutdown already requested")
            return

        self._shutdown_requested.set()
        logger.debug("Shutdown requested for listener")

    def join(self, timeout: Optional[float] = 10.0) -> None:
        """Wait for listener thread to exit. Idempotent."""
        if self._listener_thread is None:
            return

        if not self._listener_thread.is_alive():
            logger.debug("Listener thread already stopped")
            return

        self._listener_thread.join(timeout=timeout)
        if self._listener_thread.is_alive():
            logger.warning(
                f"Listener thread did not exit within {timeout}s timeout"
            )
        else:
            logger.debug("Listener thread joined successfully")

    def extract_batch(self) -> MailboxDeliveryBatch:
        """
        Extract all pending deliveries from the in-memory queue.

        Called from agent thread during turn processing. Returns a batch
        with context inclusion flag and delivery list.

        De-duplication: only returns messages with (message_id, run_id)
        not yet seen.
        """
        new_deliveries = []
        batch_delivered_keys = set()

        # Drain queue non-blocking
        while not self._delivery_queue.empty():
            delivery = self._delivery_queue.get_nowait()
            if delivery is None:  # Sentinel for shutdown
                self._delivery_queue.put(None)  # Put back for other consumers
                break

            key = (delivery.message_id, delivery.run_id)
            with self._delivered_lock:
                if key in self._delivered_keys:
                    # Already delivered, skip
                    continue
                self._delivered_keys.add(key)
                batch_delivered_keys.add(key)
                new_deliveries.append(delivery)

        return MailboxDeliveryBatch(
            deliveries=new_deliveries,
            delivered_keys=batch_delivered_keys,
            include_in_context=bool(new_deliveries),
        )

    def poll_once(self) -> None:
        """Synchronously claim a bounded mailbox batch on the agent thread.

        Design contract (design doc line 142): a synchronous poll before
        the first model request and at the final response boundary closes
        the window where a message arrives between ``runtime.start()`` and
        the first async listener poll tick.  Without it, a worker that
        makes exactly one model request can answer without seeing guidance
        that landed in that window.

        Safe to call concurrently with the listener loop: ``claim_mailbox_messages``
        runs inside a ``BEGIN IMMEDIATE`` write txn with a UNIQUE(message_id,
        run_id) constraint, so the listener and this call cannot double-claim.
        Any messages claimed here are enqueued into the same in-memory
        ``_delivery_queue`` the listener feeds, so ``extract_batch`` and its
        ``_delivered_keys`` de-dup handle them uniformly.

        Uses its OWN SQLite connection (opened and closed per call) — never
        the listener thread's connection — so no connection crosses threads
        (invariant 4).  Best-effort: failures are logged and never raised.
        """
        from pathlib import Path

        try:
            from hermes_cli import kanban_db as kb

            conn = kb.connect(db_path=Path(self.db_path))
            try:
                claimed = kb.claim_mailbox_messages(
                    conn,
                    task_id=self.task_id,
                    run_id=self.run_id,
                    recipient_profile=self.recipient_profile,
                )
            finally:
                conn.close()
        except Exception:
            logger.debug("poll_once claim failed", exc_info=True)
            return

        if not claimed:
            return

        # Track claim token + message ids for lease renewal, same as the
        # async loop.  Messages claimed here are indistinguishable from
        # listener-claimed ones downstream.
        with self._inflight_lock:
            self._claim_token = claimed[0].claim_token
            for msg in claimed:
                self._inflight_message_ids.add(msg.message_id)

        # Fetch full message rows and enqueue deliveries.  We re-open a
        # short-lived connection for the reads to keep the claim txn small.
        try:
            from hermes_cli import kanban_db as kb

            conn = kb.connect(db_path=Path(self.db_path))
            try:
                for msg in claimed:
                    row = conn.execute(
                        "SELECT * FROM task_mailbox_messages WHERE id = ?",
                        (msg.message_id,),
                    ).fetchone()
                    if row:
                        self._delivery_queue.put(
                            MailboxDelivery(
                                message_id=row["id"],
                                task_id=row["task_id"],
                                run_id=self.run_id,
                                claim_token=msg.claim_token,
                                kind=row["kind"],
                                body=row["body"],
                                actor_identity=row["actor_identity"],
                                sender_profile=row["sender_profile"],
                                comment_id=row["comment_id"],
                            )
                        )
            finally:
                conn.close()
        except Exception:
            logger.debug("poll_once enqueue failed", exc_info=True)

    def _listener_loop(self) -> None:
        """Main listener loop: poll DB, claim messages, enqueue deliveries."""
        from pathlib import Path

        from hermes_cli import kanban_db as kb

        try:
            # Open DB connection for this thread
            self._db_conn = kb.connect(db_path=Path(self.db_path))

            # Enable WAL for concurrent access
            self._db_conn.execute("PRAGMA journal_mode=WAL")

            self._listener_ready.set()
            logger.debug("Listener thread ready")

            poll_interval = self.poll_interval
            renewal_interval = 5.0  # seconds; less frequent than poll

            while not self._shutdown_requested.is_set():
                try:
                    # Claim available messages
                    claimed = kb.claim_mailbox_messages(
                        self._db_conn,
                        task_id=self.task_id,
                        run_id=self.run_id,
                        recipient_profile=self.recipient_profile,
                    )

                    if claimed:
                        logger.debug(
                            f"Claimed {len(claimed)} message(s) for delivery"
                        )
                        # Track claim token + message ids for lease renewal.
                        # claim_mailbox_messages returns all deliveries with
                        # the same claim_token for this batch.
                        with self._inflight_lock:
                            self._claim_token = claimed[0].claim_token
                            for msg in claimed:
                                self._inflight_message_ids.add(msg.message_id)

                        for msg in claimed:
                            # Get full message details from DB
                            msg_row = self._db_conn.execute(
                                "SELECT * FROM task_mailbox_messages WHERE id = ?",
                                (msg.message_id,),
                            ).fetchone()

                            if msg_row:
                                delivery = MailboxDelivery(
                                    message_id=msg_row["id"],
                                    task_id=msg_row["task_id"],
                                    run_id=self.run_id,
                                    claim_token=msg.claim_token,
                                    kind=msg_row["kind"],
                                    body=msg_row["body"],
                                    actor_identity=msg_row["actor_identity"],
                                    sender_profile=msg_row["sender_profile"],
                                    comment_id=msg_row["comment_id"],
                                )
                                self._delivery_queue.put(delivery)

                    # Periodic lease renewal for inflight messages that have
                    # not yet reached model_response_received.  The DB layer
                    # skips already-responded messages, so stale ids are safe.
                    now = time.time()
                    if (
                        now - self._last_renewal_at >= renewal_interval
                        and self._inflight_message_ids
                        and self._claim_token
                    ):
                        self._renew_leases(kb)

                    # Sleep before next poll
                    self._shutdown_requested.wait(timeout=poll_interval)

                except Exception as e:
                    # Log error but continue loop
                    logger.error(f"Error in listener loop: {e}", exc_info=True)
                    time.sleep(poll_interval)

        except Exception as e:
            logger.error(
                f"Fatal error in listener thread: {e}",
                exc_info=True,
            )
        finally:
            # Cleanup DB connection
            if self._db_conn is not None:
                try:
                    self._db_conn.close()
                    logger.debug("Listener thread closed DB connection")
                except Exception:
                    pass

            # Signal queue is closed
            try:
                self._delivery_queue.put(None)
            except Exception:
                pass

            logger.debug("Listener thread exited")

    def _renew_leases(self, kb) -> None:
        """Renew leases for inflight messages. Best-effort, never raises."""
        try:
            with self._inflight_lock:
                message_ids = list(self._inflight_message_ids)
                token = self._claim_token
            if not message_ids or not token:
                return
            renewed = kb.renew_mailbox_message_leases(
                self._db_conn,
                message_ids=message_ids,
                run_id=self.run_id,
                recipient_profile=self.recipient_profile,
                claim_token=token,
            )
            self._last_renewal_at = time.time()
            if renewed:
                logger.debug(
                    f"Renewed lease for {len(message_ids)} inflight message(s)"
                )
        except Exception:
            logger.debug("Lease renewal failed", exc_info=True)


def _mailbox_batch_args(batch, principal) -> tuple:
    """Extract the shared kwargs for the three DB state-machine transitions.

    All three (accept / mark_included / mark_responded) share one signature;
    every delivery in a single claim batch shares one claim_token.
    """
    ids = [delivery.message_id for delivery in batch.deliveries]
    token = str(batch.deliveries[0].claim_token)
    return (
        ids,
        int(principal.run_id),
        principal.sender_profile,
        token,
    )


def _build_mailbox_included_callback(principal: MailboxPrincipal):
    """Build the "included" persistence barrier callback for one worker.

    Called by acknowledge_mailbox_delivery_batch(stage="included") right after
    the canonical request messages contain the batch and before network I/O
    (design "Steer integration").  Must transition every message in the batch
    to ``included_in_request`` and return True, or return False so the caller
    retries / fails closed.

    The state machine requires ``claimed_for_run → accepted_by_steer →
    included_in_request`` in strict order.  ``accept_mailbox_messages`` is
    called defensively first (idempotent: a no-op if already accepted, which
    can happen if a prior attempt already advanced the row) so
    ``mark_mailbox_messages_included`` finds the expected
    ``accepted_by_steer`` state.
    """
    db_path = principal.db_path

    def _included(batch) -> bool:
        from pathlib import Path

        from hermes_cli import kanban_db as kb

        ids, run_id, recipient_profile, token = _mailbox_batch_args(batch, principal)
        try:
            conn = kb.connect(db_path=Path(db_path))
            try:
                # Defensive accept (idempotent guard by expected_state).  The
                # canonical pipeline does not accept elsewhere, so this is the
                # sole producer of accepted_by_steer for production batches.
                kb.accept_mailbox_messages(
                    conn,
                    message_ids=ids,
                    run_id=run_id,
                    recipient_profile=recipient_profile,
                    claim_token=token,
                )
                ok = kb.mark_mailbox_messages_included(
                    conn,
                    message_ids=ids,
                    run_id=run_id,
                    recipient_profile=recipient_profile,
                    claim_token=token,
                )
                conn.commit()
                return bool(ok)
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("mailbox included callback failed: %s", exc, exc_info=True)
            return False

    return _included


def _build_mailbox_responded_callback(principal: MailboxPrincipal):
    """Build the "responded" persistence barrier callback for one worker.

    Called by acknowledge_mailbox_delivery_batch(stage="responded") right after
    a successful model response, before tool validation/dispatch (design "Steer
    integration").  Transitions the batch to the terminal
    ``model_response_received`` and returns True, or returns False so the turn
    aborts before dispatching any returned tool (fail-closed for side effects).
    """
    db_path = principal.db_path

    def _responded(batch) -> bool:
        from pathlib import Path

        from hermes_cli import kanban_db as kb

        ids, run_id, recipient_profile, token = _mailbox_batch_args(batch, principal)
        try:
            conn = kb.connect(db_path=Path(db_path))
            try:
                ok = kb.mark_mailbox_messages_responded(
                    conn,
                    message_ids=ids,
                    run_id=run_id,
                    recipient_profile=recipient_profile,
                    claim_token=token,
                )
                conn.commit()
                return bool(ok)
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("mailbox responded callback failed: %s", exc, exc_info=True)
            return False

    return _responded


def initialize_kanban_mailbox_runtime(agent) -> bool:
    """
    Initialize KanbanMailboxRuntime if agent has a worker principal.

    Returns True if runtime was created/already exists, False otherwise.
    Sets agent._kanban_mailbox_runtime.

    This is called during agent initialization, after mailbox principal
    is granted. Idempotent: multiple calls are safe.
    """
    # Skip if runtime already exists
    if getattr(agent, "_kanban_mailbox_runtime", None) is not None:
        return True

    # Skip if no worker principal
    principal = getattr(agent, "mailbox_principal", None)
    if principal is None or principal.kind != "worker":
        return False

    try:
        # Read the configurable poll interval from config.yaml
        # (kanban.mailbox.poll_interval_seconds).  Missing key → 0.5 default.
        poll_interval = 0.5
        try:
            from hermes_cli.config import load_config_readonly

            cfg = load_config_readonly() or {}
            mailbox_cfg = (cfg.get("kanban") or {}).get("mailbox") or {}
            raw = mailbox_cfg.get("poll_interval_seconds")
            if raw is not None:
                poll_interval = float(raw)
        except Exception:
            # Config read must never block mailbox init; fall back to default.
            pass

        runtime = KanbanMailboxRuntime(principal, poll_interval=poll_interval)

        # Wire the two DB-commit callbacks so acknowledge_mailbox_delivery_batch
        # can drive the state machine to model_response_received in production.
        # Without this the callbacks stay None (agent_init default) and every
        # acknowledgement hits "callback is None" → return False.  The closures
        # capture the immutable worker principal, which carries run_id,
        # recipient_profile, db_path, and task_id.  Each callback opens its own
        # short-lived kb.connect() (invariant 4: never shares the listener
        # thread's connection), commits, and closes.
        agent._mailbox_delivery_included_callback = _build_mailbox_included_callback(principal)
        agent._mailbox_delivery_responded_callback = _build_mailbox_responded_callback(principal)

        runtime.start()
        agent._kanban_mailbox_runtime = runtime
        logger.info(
            f"KanbanMailboxRuntime started for agent {agent.session_id}: "
            f"task={principal.task_id}, run={principal.run_id}"
        )
        return True
    except Exception as e:
        logger.error(
            f"Failed to initialize KanbanMailboxRuntime: {e}",
            exc_info=True,
        )
        return False


def extract_mailbox_deliveries(agent) -> MailboxDeliveryBatch:
    """
    Extract pending mailbox deliveries from agent's runtime.

    Returns empty batch if no runtime or no deliveries.
    Called during agent turn processing.
    """
    runtime = getattr(agent, "_kanban_mailbox_runtime", None)
    if runtime is None:
        return MailboxDeliveryBatch(deliveries=[], delivered_keys=set(), include_in_context=False)

    return runtime.extract_batch()


def shutdown_kanban_mailbox_runtime(agent) -> None:
    """
    Shutdown agent's KanbanMailboxRuntime if it exists.

    Idempotent: safe to call multiple times.
    Called during agent cleanup.
    """
    runtime = getattr(agent, "_kanban_mailbox_runtime", None)
    if runtime is None:
        return

    try:
        runtime.shutdown()
        logger.debug(f"Shutdown requested for runtime of agent {agent.session_id}")
    except Exception as e:
        logger.error(f"Error shutting down KanbanMailboxRuntime: {e}", exc_info=True)


def join_kanban_mailbox_runtime(agent, timeout: Optional[float] = 10.0) -> None:
    """
    Wait for agent's KanbanMailboxRuntime to exit.

    Idempotent: safe to call multiple times.
    """
    runtime = getattr(agent, "_kanban_mailbox_runtime", None)
    if runtime is None:
        return

    try:
        runtime.join(timeout=timeout)
        logger.debug(f"Joined runtime of agent {agent.session_id}")
    except Exception as e:
        logger.error(f"Error joining KanbanMailboxRuntime: {e}", exc_info=True)


def format_mailbox_deliveries_for_context(
    batch: MailboxDeliveryBatch,
    agent_session_id: str,
) -> List[Dict[str, Any]]:
    """
    Format mailbox deliveries for inclusion in agent context.

    Returns a list of message dicts ready to append to conversation.
    Each is structured like:
    {
        "role": "user",
        "content": "You received a new message from the dispatcher...",
    }

    If batch.include_in_context is False or no deliveries, returns [].
    """
    if not batch.include_in_context or not batch.deliveries:
        return []

    messages = []
    for delivery in batch.deliveries:
        kind_label = {
            "guidance": "guidance",
            "question": "question",
            "info": "information",
        }.get(delivery.kind, "message")

        content = f"You received a new {kind_label} from the dispatcher:\n\n{delivery.body}"

        messages.append({
            "role": "user",
            "content": content,
        })

    return messages