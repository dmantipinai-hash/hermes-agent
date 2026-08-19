"""Tests for role-based agent profiles (delegate_task specialization roles).

These are **invariant** tests, not change-detectors. We assert contracts
about how specialization roles relate to the rest of the system, not
specific role names or counts. If a PR adds a new role, these tests
keep passing; if a PR breaks a structural contract, they fail.

Contracts covered:
  - ROLE_TOOLSET_MAP structure (every role has the expected keys)
  - every role's toolsets reference real, registered toolsets
  - toolsets are deduplicated and non-empty
  - prompt_hint is non-empty (it's what shapes child behavior)
  - _normalize_role accepts all specialization roles + leaf/orchestrator,
    and silently coerces unknowns to 'leaf'
  - the delegate_task schema enum stays in sync with ROLE_TOOLSET_MAP
    (no drift between the enum and the runtime map)
"""

import pytest

from toolsets import ROLE_TOOLSET_MAP, TOOLSETS, VALID_ROLES
from tools.delegate_tool import _ALL_ROLE_NAMES, _normalize_role


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------

class TestRoleMapStructure:
    """ROLE_TOOLSET_MAP must be a well-formed map of role definitions."""

    def test_map_is_nonempty_dict(self):
        assert isinstance(ROLE_TOOLSET_MAP, dict)
        assert len(ROLE_TOOLSET_MAP) >= 1

    @pytest.mark.parametrize("role_name", list(ROLE_TOOLSET_MAP.keys()))
    def test_every_role_has_required_keys(self, role_name):
        cfg = ROLE_TOOLSET_MAP[role_name]
        assert set(cfg.keys()) == {"description", "toolsets", "prompt_hint"}, (
            f"role {role_name!r} has unexpected keys: {set(cfg.keys())}"
        )

    @pytest.mark.parametrize("role_name", list(ROLE_TOOLSET_MAP.keys()))
    def test_every_role_toolsets_is_nonempty_list(self, role_name):
        ts = ROLE_TOOLSET_MAP[role_name]["toolsets"]
        assert isinstance(ts, list) and ts, f"{role_name}: toolsets must be a non-empty list"

    @pytest.mark.parametrize("role_name", list(ROLE_TOOLSET_MAP.keys()))
    def test_every_role_toolsets_have_no_duplicates(self, role_name):
        ts = ROLE_TOOLSET_MAP[role_name]["toolsets"]
        assert len(ts) == len(set(ts)), f"{role_name}: duplicate toolsets in {ts}"

    @pytest.mark.parametrize("role_name", list(ROLE_TOOLSET_MAP.keys()))
    def test_every_role_prompt_hint_is_nonempty(self, role_name):
        hint = ROLE_TOOLSET_MAP[role_name]["prompt_hint"]
        assert isinstance(hint, str) and hint.strip(), (
            f"{role_name}: prompt_hint must be a non-empty string (it shapes child behavior)"
        )


class TestRoleToolsetsResolveToRealToolsets:
    """A role's toolsets must all be real keys in TOOLSETS.

    A role pointing at a nonexistent toolset would silently give the child
    an empty capability — the worst kind of failure (looks configured,
    isn't).
    """

    @pytest.mark.parametrize("role_name", list(ROLE_TOOLSET_MAP.keys()))
    def test_role_toolsets_exist_in_toolsets(self, role_name):
        ts = ROLE_TOOLSET_MAP[role_name]["toolsets"]
        unknown = [t for t in ts if t not in TOOLSETS]
        assert not unknown, f"role {role_name!r} references unknown toolsets: {unknown}"


class TestValidRolesDerivedFromMap:
    """VALID_ROLES must stay in sync with ROLE_TOOLSET_MAP keys."""

    def test_valid_roles_equals_map_keys(self):
        assert VALID_ROLES == set(ROLE_TOOLSET_MAP.keys()), (
            "VALID_ROLES drifted from ROLE_TOOLSET_MAP — re-derive, don't hardcode"
        )


# ---------------------------------------------------------------------------
# Role normalization (delegate_tool._normalize_role)
# ---------------------------------------------------------------------------

class TestNormalizeRole:
    """_normalize_role must accept every specialization role + leaf/orchestrator,
    and silently coerce unknowns to 'leaf' (matches Hermes' silent-degrade pattern).
    """

    @pytest.mark.parametrize("role_name", list(ROLE_TOOLSET_MAP.keys()))
    def test_specialization_roles_normalize_to_themselves(self, role_name):
        assert _normalize_role(role_name) == role_name

    def test_specialization_roles_are_case_insensitive(self):
        # 'Researcher' / 'CODER' should resolve just like lowercase
        for role_name in ROLE_TOOLSET_MAP:
            assert _normalize_role(role_name.upper()) == role_name
            assert _normalize_role(role_name.capitalize()) == role_name

    def test_leaf_and_orchestrator_pass_through(self):
        assert _normalize_role("leaf") == "leaf"
        assert _normalize_role("orchestrator") == "orchestrator"

    def test_none_and_empty_coerce_to_leaf(self):
        assert _normalize_role(None) == "leaf"
        assert _normalize_role("") == "leaf"
        assert _normalize_role("   ") == "leaf"

    def test_unknown_role_coerces_to_leaf(self, caplog):
        # Silent-degrade: unknown roles must not raise, just warn + coerce.
        assert _normalize_role("bogus_role_xyz") == "leaf"
        assert _normalize_role("admin") == "leaf"


# ---------------------------------------------------------------------------
# Schema / runtime sync
# ---------------------------------------------------------------------------

class TestSchemaEnumInSync:
    """The delegate_task schema enum (_ALL_ROLE_NAMES) must include every
    specialization role so the model can actually request them. Drift here
    means a role exists in the runtime map but is unreachable from the API.
    """

    def test_enum_contains_all_specialization_roles(self):
        for role_name in ROLE_TOOLSET_MAP:
            assert role_name in _ALL_ROLE_NAMES, (
                f"role {role_name!r} is in ROLE_TOOLSET_MAP but missing from "
                f"_ALL_ROLE_NAMES — the model can't request it"
            )

    def test_enum_contains_leaf_and_orchestrator(self):
        assert "leaf" in _ALL_ROLE_NAMES
        assert "orchestrator" in _ALL_ROLE_NAMES

    def test_enum_has_no_duplicates(self):
        assert len(_ALL_ROLE_NAMES) == len(set(_ALL_ROLE_NAMES)), (
            f"_ALL_ROLE_NAMES has duplicates: {_ALL_ROLE_NAMES}"
        )
