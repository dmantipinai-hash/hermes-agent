#!/usr/bin/env python3
"""Phase 3 — Crash Recovery integration test.

Runs ALL crash-recovery scenarios in sequence against a temporary SessionDB
and the real _repair_interrupted_tail logic. This is the bridge between
unit-tests (which check invariants) and manual E2E testing (which needs a
live agent + API key).

Usage:
    cd ~/Desktop/hermes-agent
    source .venv/bin/activate
    python scripts/test_phase3_crash_recovery.py

Exit code 0 = all scenarios passed, 1 = at least one failed.
"""
import sys
import os
import time
import json
import tempfile
import traceback
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hermes_state import SessionDB
from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_repair():
    """Get the bound _repair_interrupted_tail method without full AIAgent init."""
    class _Stub:
        pass
    _Stub._repair_interrupted_tail = AIAgent._repair_interrupted_tail
    return _Stub()._repair_interrupted_tail


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

passed = 0
failed = 0


def ok(name, detail=""):
    global passed
    passed += 1
    print(f"  {GREEN}✅ {name}{RESET}" + (f" — {detail}" if detail else ""))


def fail(name, detail=""):
    global failed
    failed += 1
    print(f"  {RED}❌ {name}{RESET}" + (f" — {detail}" if detail else ""))


def section(title):
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN} {title}{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")


# ---------------------------------------------------------------------------
# Scenario 1: Crash detection lifecycle
# ---------------------------------------------------------------------------

def test_crash_detection():
    section("1. Crash Detection Lifecycle")
    with tempfile.TemporaryDirectory() as td:
        db = SessionDB(Path(td) / "state.db")

        # Fresh session — not interrupted
        db.create_session("s1", source="test")
        if db.find_interrupted_session() is None:
            ok("fresh session not flagged as interrupted")
        else:
            fail("fresh session flagged as interrupted")

        # Simulate crash: mark active, don't mark idle
        db.mark_run_active("s1")
        result = db.find_interrupted_session()
        if result == "s1":
            ok("active session detected as interrupted")
        else:
            fail("active session not detected", f"got {result!r}")

        # Clean exit clears the flag
        db.mark_run_idle("s1")
        if db.find_interrupted_session() is None:
            ok("idle after mark_run_idle")
        else:
            fail("still interrupted after mark_run_idle")

        # Ended session never counts as interrupted
        db.mark_run_active("s1")
        db.end_session("s1", "test_end")
        if db.find_interrupted_session() is None:
            ok("ended session excluded from interrupted")
        else:
            fail("ended session still flagged")


# ---------------------------------------------------------------------------
# Scenario 2: Multiple interrupted sessions — most recent wins
# ---------------------------------------------------------------------------

def test_most_recent_interrupted():
    section("2. Multiple Interrupted — Most Recent Wins")
    with tempfile.TemporaryDirectory() as td:
        db = SessionDB(Path(td) / "state.db")

        db.create_session("old", source="test")
        db.mark_run_active("old")
        time.sleep(0.05)

        db.create_session("new", source="test")
        db.mark_run_active("new")

        winner = db.find_interrupted_session()
        if winner == "new":
            ok("most recent interrupted session selected")
        else:
            fail("wrong session selected", f"expected 'new', got {winner!r}")


# ---------------------------------------------------------------------------
# Scenario 3: Activity tracking timestamp
# ---------------------------------------------------------------------------

def test_activity_tracking():
    section("3. Activity Tracking (touch_activity)")
    with tempfile.TemporaryDirectory() as td:
        db = SessionDB(Path(td) / "state.db")
        db.create_session("s1", source="test")

        before = time.time()
        time.sleep(0.05)
        db.touch_activity("s1")

        with db._lock:
            row = db._conn.execute(
                "SELECT last_activity_at FROM sessions WHERE id = ?", ("s1",)
            ).fetchone()

        if row and row[0] and row[0] > before:
            ok(f"last_activity_at updated ({row[0]:.3f} > {before:.3f})")
        else:
            fail("last_activity_at not updated", f"row={row}")


# ---------------------------------------------------------------------------
# Scenario 4: Transcript repair — all broken-tail cases
# ---------------------------------------------------------------------------

def test_transcript_repair():
    section("4. Transcript Repair (_repair_interrupted_tail)")
    repair = make_repair()

    # Case A: orphaned assistant(tool_calls) — no tool results followed
    m = [
        {"role": "user", "content": "write hello.py"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "write_file", "arguments": "{}"}}]},
    ]
    dropped = repair(m)
    if dropped == 1 and m[-1]["role"] == "user":
        ok("orphaned assistant(tool_calls) dropped", f"removed {dropped}")
    else:
        fail("orphaned assistant not handled", f"dropped={dropped}, tail={m[-1]['role'] if m else 'empty'}")

    # Case B: orphaned tool results — no preceding assistant(tool_calls)
    m = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "content": "stale result 1", "tool_call_id": "x"},
        {"role": "tool", "content": "stale result 2", "tool_call_id": "y"},
    ]
    dropped = repair(m)
    if dropped == 2 and m[-1]["role"] == "user":
        ok("orphaned tool results dropped", f"removed {dropped}")
    else:
        fail("orphaned tool results not handled", f"dropped={dropped}")

    # Case C: partial pair — 2 tool_calls, 1 tool result
    m = [
        {"role": "user", "content": "do two things"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "f1", "arguments": "{}"}},
            {"id": "2", "type": "function", "function": {"name": "f2", "arguments": "{}"}},
        ]},
        {"role": "tool", "content": "result 1", "tool_call_id": "1"},
    ]
    dropped = repair(m)
    if dropped == 2 and m[-1]["role"] == "user":
        ok("partial tool pair dropped", f"removed {dropped} (1 assistant + 1 tool)")
    else:
        fail("partial pair not handled", f"dropped={dropped}")

    # Case D: complete pair — must be preserved
    m = [
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f1", "arguments": "{}"}}]},
        {"role": "tool", "content": "done", "tool_call_id": "1"},
        {"role": "assistant", "content": "All done!"},
    ]
    orig_len = len(m)
    dropped = repair(m)
    if dropped == 0 and len(m) == orig_len:
        ok("complete tool pair preserved", "0 dropped")
    else:
        fail("complete pair modified", f"dropped={dropped}, was {orig_len} now {len(m)}")

    # Case E: idempotent — running twice doesn't drop more
    m = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
    ]
    d1 = repair(m)
    d2 = repair(m)
    if d1 == 1 and d2 == 0:
        ok("repair is idempotent", f"pass1={d1}, pass2={d2}")
    else:
        fail("repair not idempotent", f"pass1={d1}, pass2={d2}")

    # Case F: clean tail — no change
    m = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    dropped = repair(m)
    if dropped == 0:
        ok("clean assistant text tail preserved")
    else:
        fail("clean tail modified", f"dropped={dropped}")


# ---------------------------------------------------------------------------
# Scenario 5: Resume produces API-safe message sequence
# ---------------------------------------------------------------------------

def test_resume_safe_sequence():
    section("5. Resume — API-Safe Message Sequence")
    repair = make_repair()

    scenarios = [
        ("orphan_assistant_tail", [
            {"role": "user", "content": "do stuff"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        ]),
        ("orphan_tool_tail", [
            {"role": "user", "content": "x"},
            {"role": "tool", "content": "orphan", "tool_call_id": "z"},
        ]),
        ("partial_pair_mid", [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "1", "type": "function", "function": {"name": "f1", "arguments": "{}"}},
                {"id": "2", "type": "function", "function": {"name": "f2", "arguments": "{}"}},
            ]},
            {"role": "tool", "content": "r1", "tool_call_id": "1"},
            # missing tool result for id "2"
        ]),
    ]

    for name, messages in scenarios:
        m_copy = [dict(msg) for msg in messages]
        repair(m_copy)

        # After repair, tail must be user or assistant(text) — NOT tool, NOT assistant(tool_calls)
        if not m_copy:
            fail(f"{name}: repair emptied everything")
            continue

        tail = m_copy[-1]
        tail_role = tail.get("role")
        has_tool_calls = tail.get("tool_calls") is not None

        if tail_role == "tool":
            fail(f"{name}: tail is tool (would 400 on resume)")
        elif tail_role == "assistant" and has_tool_calls:
            fail(f"{name}: tail is assistant(tool_calls) (would 400 on resume)")
        elif tail_role in ("user", "assistant"):
            ok(f"{name}: tail is {tail_role} (resume-safe)")
        else:
            fail(f"{name}: unexpected tail role {tail_role!r}")


# ---------------------------------------------------------------------------
# Scenario 6: Schema migration — columns exist on fresh DB
# ---------------------------------------------------------------------------

def test_schema():
    section("6. Schema Migration (run_active, last_activity_at)")
    with tempfile.TemporaryDirectory() as td:
        db = SessionDB(Path(td) / "state.db")

        with db._lock:
            cols = {r[1] for r in db._conn.execute("PRAGMA table_info(sessions)").fetchall()}

        if "run_active" in cols:
            ok("run_active column exists")
        else:
            fail("run_active column missing")

        if "last_activity_at" in cols:
            ok("last_activity_at column exists")
        else:
            fail("last_activity_at column missing")

        # Default value check
        db.create_session("s1", source="test")
        with db._lock:
            row = db._conn.execute(
                "SELECT run_active FROM sessions WHERE id = ?", ("s1",)
            ).fetchone()
        if row and row[0] == 0:
            ok("run_active defaults to 0")
        else:
            fail("run_active default wrong", f"got {row[0] if row else 'NULL'}")


# ---------------------------------------------------------------------------
# Scenario 7: Full crash → resume simulation (DB-level, no API)
# ---------------------------------------------------------------------------

def test_full_crash_resume_cycle():
    section("7. Full Crash → Resume Cycle (DB + Repair)")
    with tempfile.TemporaryDirectory() as td:
        db = SessionDB(Path(td) / "state.db")
        repair = make_repair()

        # 1. Create session, start a turn
        db.create_session("crash-test", source="test")
        db.mark_run_active("crash-test")
        db.touch_activity("crash-test")
        ok("session created and marked active")

        # 2. Simulate broken transcript (agent crashed mid-tool-execution)
        broken_messages = [
            {"role": "user", "content": "delegate a complex task"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "delegate_task", "arguments": '{"goal": "do research"}'}},
            ]},
            # CRASH HERE — tool result was never written
        ]
        ok("simulated crash: orphaned assistant(tool_calls) at tail")

        # 3. Detect interrupted session
        interrupted = db.find_interrupted_session()
        if interrupted == "crash-test":
            ok("crash detected via find_interrupted_session()")
        else:
            fail("crash not detected", f"got {interrupted!r}")
            return

        # 4. Repair the transcript
        repaired = [dict(m) for m in broken_messages]
        dropped = repair(repaired)
        if dropped >= 1 and repaired[-1]["role"] == "user":
            ok(f"transcript repaired ({dropped} messages removed)")
        else:
            fail("repair failed", f"dropped={dropped}, tail={repaired[-1].get('role') if repaired else 'empty'}")
            return

        # 5. Mark idle (simulating successful resume)
        db.mark_run_idle("crash-test")
        if db.find_interrupted_session() is None:
            ok("session cleared after resume")
        else:
            fail("session still flagged after resume")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"{CYAN}╔══════════════════════════════════════════════════════════╗{RESET}")
    print(f"{CYAN}║  Phase 3: Crash Recovery — Integration Test Suite        ║{RESET}")
    print(f"{CYAN}╚══════════════════════════════════════════════════════════╝{RESET}")

    tests = [
        test_crash_detection,
        test_most_recent_interrupted,
        test_activity_tracking,
        test_transcript_repair,
        test_resume_safe_sequence,
        test_schema,
        test_full_crash_resume_cycle,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            fail(f"{test.__name__} raised", str(e))
            traceback.print_exc()

    print(f"\n{CYAN}{'='*60}{RESET}")
    total = passed + failed
    if failed == 0:
        print(f"{GREEN}  ALL {total} SCENARIOS PASSED{RESET}")
    else:
        print(f"{RED}  {failed}/{total} SCENARIOS FAILED{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
