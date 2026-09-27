---
name: memory-store-maintenance
description: "Safe Hermes memory edits. Trigger: replace/supersede fail."
author: Dmitry + Hermes Agent
license: MIT
---

# Memory Store Maintenance

Class of task: editing/compacting the agent's own persistent memory without burning retries or corrupting history. Full triggers: `replace` fails "no entry matched", `supersede` fails "matches no entry", `old_text` rejected though the fact was just recalled from a system-prompt/retrieval block, store at ~98% of char budget.

## Architecture (the real trap)

- Two tiers: **hot** (prompt snapshot, ~2,200 chars, auto-eviction) + **cold** (SQLite, unbounded, searchable). This split is BENIGN — eviction never blocks edits; never delete entries to "free space".
- Two TARGETS are the trap: `memory` (agent's notes) and `user` (profile). The retrieval pack crosses BOTH stores, and before 31.08.2026 pack lines carried no store label — so an entry recalled from `user` looked like a memory note, and mutations were addressed to the wrong store. The lookups then honestly searched the wrong table and returned "no entry matched" / "matches no entry". This, not hot/cold, caused the 30–31.08 four-in-a-row loops (confirmed by ZCode commit 182f633759, which explicitly refutes the hot/cold theory).

## Fix in place (182f633759, 2026-08-31) — verified live ×3 by Loki

- `supersede`/`deprecate` with an exact `old_id` (or unique prefix) resolves ACROSS stores: an id belonging to the other target operates on its actual store, response carries a "target corrected" note. Verified: id from read (target=user), call with target=memory → superseded correctly, first try.
- `old_text` no-match on replace/remove now probes the other store and says so ("entry exists in target user — retry with target=user"), turning the loop into one corrective retry.
- Pack lines tag user-store entries with `· user` — check the tag before mutating.

## Decision table

| Situation | Working action |
|---|---|
| Full id known — from a live `read` this session, or `new_id` returned by the store itself | `supersede` with `old_id` = FULL uuid — works from any store/tier, first try; wrong `target` is auto-corrected |
| Id taken from a retrieval-pack line | NOT safe alone (incident 2026-09-20: supersede hit the WRONG entry with a plausible id). First `read` the topic and confirm this id sits next to the entry you mean — truncated previews look alike. After supersede, check `deprecated_content` in the response: it must match what you replaced |
| Only text known, replace failed with "no entry matched" | Read the error: it now names the other store if the text lives there — retry with that `target` |
| Entry appears in `current_entries` of the error dump | `replace` with a SHORT unique substring of the REAL stored text (paraphrase rejected — quirk 25.08) |
| Store at ~98% of budget | Plan as a SINGLE `replace` of one hot entry folding new facts in. NEVER `add` first then `replace`: a failed replace after a successful add overflows the budget |

## Pitfalls

- **"Not found" ≠ wrong architecture.** When a lookup honestly returns no-match for something you can SEE, first suspect a wrong store/scope/addressing — not a fancy two-layer theory. The 31.08 incident: an elegant "hot-showcase vs cold-store" theory was disproven by `git show` of the fix commit; the real cause was cross-target addressing. Verify the mechanism against code before enshrining a theory in a skill.
- Loop discipline still applies AFTER a corrective retry: if the error's own hint (other-store note / current_entries) fails too, stop after 2 attempts and use `add` — the loop-warning system counts these.
- `replace` without `old_text` errors immediately (old_id alone is not enough for it) — supply both when possible.
- Never record a replacement in any external archive unless it actually executed (archive truthfulness).
- Eviction is automatic and benign — never delete/shorten entries to "free space".

## Verification

Success responses carry `usage`, the new entry `id`, and `link_created` — note the id in session output. Post-fix, a later session can supersede by that id from ANY tier or store (hot or cold, memory or user) — no "while still hot" window anymore. Chain check: re-`read` the topic afterwards; the deprecated entry should appear in the `supersedes` block with full provenance.

## References

- `references/incident-2026-08-31-cross-target.md` — full live-verification protocol for the cross-target id resolution fix (commit 182f633759): three staged checks, wrong-store supersede test, error-message contracts.
