"""Schema-shape tests for the built-in memory tool.

The memory tool previously used ``allOf: [{if: ..., then: {required: ...}}]``
at the top level of ``parameters`` to hint per-action required fields.  That
form was:

  1. Ignored by every provider (Chat Completions doesn't honour ``if/then``
     on function schemas), so it never actually enforced anything.
  2. **Rejected outright by strict backends** — OpenAI's Codex endpoint
     (``chatgpt.com/backend-api/codex``, gpt-5.x) returns
     ``Invalid schema for function 'memory': schema must have type 'object'
     and not have 'oneOf'/'anyOf'/'allOf'/'enum'/'not' at the top level``.

We now rely on the runtime handler (``memory_tool()`` in ``tools/memory_tool.py``)
to validate required fields per action and return actionable error messages.
These tests guard the schema against regressing back to a shape strict
backends reject.
"""

import json

from tools.memory_tool import MEMORY_SCHEMA


_FORBIDDEN_TOP_LEVEL_KEYS = ("allOf", "anyOf", "oneOf", "enum", "not")


def test_memory_schema_has_no_forbidden_top_level_combinators():
    """OpenAI's Codex backend rejects these at the top level of parameters."""
    params = MEMORY_SCHEMA["parameters"]
    for key in _FORBIDDEN_TOP_LEVEL_KEYS:
        assert key not in params, (
            f"top-level {key!r} in memory tool parameters will break the "
            "Codex backend (chatgpt.com/backend-api/codex). Per-action "
            "required-field checks belong in the runtime handler, not the schema."
        )


def test_memory_schema_is_well_formed():
    params = MEMORY_SCHEMA["parameters"]
    assert params["type"] == "object"
    assert params["required"] == ["action", "target"]
    # Nested ``enum`` on property values is fine — only top-level is forbidden.
    # v2 store adds deprecate/read actions plus typed-entry parameters.
    assert params["properties"]["action"]["enum"] == [
        "add", "replace", "remove", "deprecate", "read",
    ]
    assert params["properties"]["target"]["enum"] == ["memory", "user"]
    assert params["properties"]["type"]["enum"] == [
        "fact", "decision", "constraint", "pattern", "preference",
    ]


def test_memory_schema_is_json_serializable():
    json.dumps(MEMORY_SCHEMA)


def test_memory_schema_describes_both_memory_tiers():
    """The agent's self-model contract: the schema must tell the model that
    memory is two-tier (hot prompt snapshot + cold full store), that `read`
    searches the cold store beyond the visible prompt, and that evicted
    records resurface per turn. Without this the agent describes its memory
    as "just the 2200-char budget" (live-observed on hemdal, 2026-08-17)."""
    desc = MEMORY_SCHEMA["description"]
    assert "ARCHITECTURE (two tiers)" in desc
    assert "hot tier" in desc and "cold tier" in desc
    assert "ENTIRE cold store" in desc
    assert "re-injected automatically" in desc
