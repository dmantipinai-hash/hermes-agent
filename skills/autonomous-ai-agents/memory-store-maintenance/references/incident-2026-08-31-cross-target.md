# Incident 2026-08-31: cross-target id resolution ("read sees it, write can't")

Fix: commit `182f633759` in ~/Desktop/hermes-agent — `fix(memory): cross-target id resolution`.
Symptom class: `replace` → "No entry matched", `supersede` → "matches no entry", for an entry that is visibly recalled in the same turn.

## Root cause (confirmed against code, not theory)

- Memory tool has TWO targets: `memory` (agent notes) and `user` (profile). The retrieval pack mixes BOTH stores into one list.
- Before the fix, pack lines carried no store label. An entry living in `user` read like an ordinary memory note, so mutations were addressed to `target=memory` — which honestly searched the wrong table and returned no-match.
- The agent's contemporaneous "hot-showcase vs cold-store" theory was WRONG: hot/cold tiering never blocked edits. Disproven by reading the fix commit. Lesson recorded in SKILL.md Pitfalls: verify mechanism against code before enshrining a theory.

## What the fix changed

1. `_resolve_entry_id` resolves ids ACROSS stores: exact `old_id` (or unique prefix) belonging to the other target operates on the entry's actual store; response carries `"target corrected"` note; supersede puts both sides in the old entry's store.
2. `old_text` no-match on replace/remove probes the other store and says so: "An entry matching this text exists in target user — retry with target=user".
3. Pack lines tag user-store entries with `· user`.
4. Tests: 254 memory tests green incl. 4 new cross-target contracts; "foreign target is an error" contract deliberately superseded by "correct with a note".

## Live verification protocol (run by Loki 31.08, all green)

1. **Wrong-store supersede (the incident scenario, deliberately miscalled):**
   - `read` target=user by topic keywords → get entry id (e.g. `558384d7-…`, the user-profile "Фокус/карьерный вектор" entry)
   - single `supersede` with `target=memory`, `old_id` = that uuid
   - EXPECT: success, `"target corrected: the old entry lives in 'user', not 'memory'"`, `link_created: true`, both sides in user store.
2. **Cold-tier supersede via read id:** recall a pre-session entry by refined query (first query may miss — refine BEFORE writing), supersede by full uuid from the read result. EXPECT: first-try success.
3. **Chain check:** re-`read` the topic; the new entry shows a `supersedes` block with the deprecated old id, content head, `linked_at` timestamp. Full provenance visible.

## Error-message contracts (post-fix)

- no-match + text exists in other store → error names the store and dumps that store's current_entries: one corrective retry, not a loop.
- `replace` without `old_text` → immediate error regardless of old_id (still true).
- Short 8-char id prefixes: unique ones resolve; ambiguous ones fail — prefer FULL uuid from read.

## Sequencing rule for curators

When a memory-mutation error appears, read the ERROR BODY first: it now contains the corrective hint (other-store name or current_entries). Only if the hint itself fails after 1 retry, fall back to `add` (loop discipline: 2 attempts max).