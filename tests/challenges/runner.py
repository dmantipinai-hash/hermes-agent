"""Challenge-based testing runner — Stage 1 (skeleton).

Implements the design from CHALLENGE-TESTING.md. This is Stage 1: the
runner parses YAML challenge files, walks their steps, and emits a
structured report. The step-execution engine is a stub that records the
declared steps but does NOT yet drive the real agent (that's Stage 2 —
requires a mock LLM provider). Even as a stub, the runner is CI-ready:
``python tests/challenges/runner.py`` returns 0 on a clean parse of all
challenges, 1 on any YAML/format error.

Run::

    python tests/challenges/runner.py                       # all challenges
    python tests/challenges/runner.py --category delegation # filter
    python tests/challenges/runner.py --list                # just list, don't run

Why YAML and not pytest: challenges declare intent (steps + expectations)
separately from execution. The same YAML will run against Stage 2's
in-process engine and Stage 3's live-LLM engine without edits. pytest
would couple declaration to a single execution model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import yaml

CHALLENGES_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class ChallengeResult:
    name: str
    category: str
    status: str  # "pass", "fail", "flaky", "error", "skipped"
    duration_s: float
    attempts: int
    passes: int
    detail: str = ""

    def icon(self) -> str:
        return {"pass": "✅", "fail": "❌", "flaky": "⚠️",
                "error": "💥", "skipped": "⏭️"}.get(self.status, "?")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_challenges(category: Optional[str] = None) -> list[dict]:
    """Load all *.yaml challenge files, optionally filtered by category."""
    files = sorted(CHALLENGES_DIR.glob("**/*.yaml"))
    challenges = []
    for f in files:
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
        except Exception as e:
            challenges.append({
                "_file": str(f),
                "_error": f"YAML parse error: {e}",
                "name": f.stem,
                "category": "unknown",
            })
            continue
        if not isinstance(data, dict):
            continue
        data.setdefault("_file", str(f))
        if category and data.get("category") != category:
            continue
        challenges.append(data)
    return challenges


def validate_challenge(ch: dict) -> Optional[str]:
    """Structural check — returns an error message or None if valid."""
    for key in ("name", "category", "steps"):
        if key not in ch:
            return f"missing required field: {key!r}"
    if not isinstance(ch["steps"], list) or not ch["steps"]:
        return "'steps' must be a non-empty list"
    for i, step in enumerate(ch["steps"]):
        if not isinstance(step, dict) or "action" not in step:
            return f"step {i}: missing 'action'"
    return None


# ---------------------------------------------------------------------------
# Execution (Stage 1 stub)
# ---------------------------------------------------------------------------

def _execute_steps(steps: list[dict]) -> tuple[bool, str]:
    """Stage 1 stub: validate steps structurally, don't drive the agent.

    Returns (ok, detail). Stage 2 will replace this with a real engine
    that calls delegate_task / agent_manager against a mock LLM provider.
    For now, any structurally-valid set of steps 'passes' — the point is
    to prove the YAML→runner→report pipeline works end-to-end.
    """
    for i, step in enumerate(steps):
        if "action" not in step:
            return False, f"step {i}: missing 'action'"
        if "expect" in step and not isinstance(step["expect"], dict):
            return False, f"step {i}: 'expect' must be a dict"
    return True, f"{len(steps)} steps parsed (Stage 1 stub — no execution)"


def run_challenge(ch: dict) -> ChallengeResult:
    """Run one challenge ``attempts`` times, aggregate pass/fail/flaky."""
    if "_error" in ch:
        return ChallengeResult(
            name=ch.get("name", "?"), category=ch.get("category", "?"),
            status="error", duration_s=0.0, attempts=0, passes=0,
            detail=ch["_error"],
        )
    err = validate_challenge(ch)
    if err:
        return ChallengeResult(
            name=ch.get("name", "?"), category=ch.get("category", "?"),
            status="error", duration_s=0.0, attempts=0, passes=0, detail=err,
        )

    attempts = int(ch.get("attempts", 1))
    passes = 0
    last_detail = ""
    t0 = time.time()
    for _ in range(attempts):
        ok, detail = _execute_steps(ch["steps"])
        if ok:
            passes += 1
        else:
            last_detail = detail
    duration = time.time() - t0

    if passes == attempts:
        status = "pass"
    elif passes == 0:
        status = "fail"
    else:
        status = "flaky"
    return ChallengeResult(
        name=ch["name"], category=ch["category"], status=status,
        duration_s=duration, attempts=attempts, passes=passes,
        detail=last_detail,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Hermes challenge runner (Stage 1)")
    ap.add_argument("--category", help="filter by category (delegation, crash_recovery, team)")
    ap.add_argument("--list", action="store_true", help="list challenges and exit")
    ap.add_argument("--json", help="write JSON report to this path")
    args = ap.parse_args()

    challenges = load_challenges(category=args.category)
    if not challenges:
        print(f"No challenges found in {CHALLENGES_DIR}")
        return 0

    if args.list:
        for ch in challenges:
            mark = "⚠️ " if "_error" in ch else "   "
            print(f"{mark}{ch.get('category','?'):>16} / {ch.get('name','?')}")
        print(f"\n{len(challenges)} challenge(s)")
        return 0

    results = [run_challenge(ch) for ch in challenges]

    print("=" * 64)
    for r in results:
        print(f"  {r.icon()} {r.name}  [{r.category}]  ({r.passes}/{r.attempts})")
        if r.detail and r.status in ("fail", "error"):
            print(f"      └─ {r.detail[:200]}")
    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status in ("fail", "error"))
    flaky = sum(1 for r in results if r.status == "flaky")
    print("=" * 64)
    print(f"  Total: {len(results)} | Pass: {passed} | Flaky: {flaky} | Fail: {failed}")
    print(f"  ({sum(r.duration_s for r in results):.1f}s)")

    if args.json:
        Path(args.json).write_text(
            json.dumps([asdict(r) for r in results], indent=2, default=str),
            encoding="utf-8",
        )
    # Exit 1 if any challenge failed OR errored (structural issues count).
    return 1 if (failed or any(r.status == "error" for r in results)) else 0


if __name__ == "__main__":
    sys.exit(main())
