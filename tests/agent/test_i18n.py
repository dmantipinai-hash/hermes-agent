"""Tests for agent.i18n -- catalog parity, fallback, language resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent import i18n


LOCALES_DIR = Path(__file__).resolve().parents[2] / "locales"


def _load_raw(lang: str) -> dict:
    with (LOCALES_DIR / f"{lang}.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _flatten(d, prefix="") -> dict:
    flat = {}
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten(v, key))
        else:
            flat[key] = v
    return flat


# ---------------------------------------------------------------------------
# Catalog completeness -- this is the key invariant test.  If someone adds a
# new key to en.yaml they MUST add it to every other locale, else runtime
# falls back to English for those users and defeats the feature.
# ---------------------------------------------------------------------------



@pytest.mark.parametrize("lang", [l for l in i18n.SUPPORTED_LANGUAGES if l != "en"])
def test_catalog_keys_match_english(lang: str):
    """Every non-English catalog must have exactly the same key set as English."""
    en_keys = set(_flatten(_load_raw("en")).keys())
    lang_keys = set(_flatten(_load_raw(lang)).keys())
    missing = en_keys - lang_keys
    extra = lang_keys - en_keys
    assert not missing, f"{lang}.yaml missing keys: {sorted(missing)}"
    assert not extra, f"{lang}.yaml has keys not in en.yaml: {sorted(extra)}"


@pytest.mark.parametrize("lang", list(i18n.SUPPORTED_LANGUAGES))
def test_catalog_placeholders_match_english(lang: str):
    """Every translated value must use the same {placeholder} tokens as English.

    A mistranslated placeholder (e.g. ``{description}`` typoed as ``{descricao}``)
    would either raise KeyError at runtime or silently drop the interpolated
    value.  Pin parity at the test layer.
    """
    import re
    placeholder_re = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
    en_flat = _flatten(_load_raw("en"))
    lang_flat = _flatten(_load_raw(lang))
    for key, en_value in en_flat.items():
        en_placeholders = set(placeholder_re.findall(en_value))
        lang_value = lang_flat.get(key, "")
        lang_placeholders = set(placeholder_re.findall(lang_value))
        assert en_placeholders == lang_placeholders, (
            f"{lang}.yaml key={key!r}: placeholders {lang_placeholders} "
            f"don't match English {en_placeholders}"
        )


# ---------------------------------------------------------------------------
# Language resolution
# ---------------------------------------------------------------------------











def test_default_when_nothing_set(monkeypatch):
    """With no env var and no config override, falls back to English."""
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    # Force config lookup to return None -- patch the cached reader.
    i18n.reset_language_cache()
    monkeypatch.setattr(i18n, "_config_language_cached", lambda: None)
    assert i18n.get_language() == "en"


# ---------------------------------------------------------------------------
# t() semantics
# ---------------------------------------------------------------------------







def test_t_missing_key_in_non_english_falls_back_to_english(tmp_path, monkeypatch):
    """If a key exists in English but not in the target locale, fall back."""
    # Stand up a fake incomplete locale under a temp locales dir.
    fake_locales = tmp_path / "locales"
    fake_locales.mkdir()
    (fake_locales / "en.yaml").write_text("foo: English Foo\n", encoding="utf-8")
    (fake_locales / "zh.yaml").write_text("# intentionally empty\n", encoding="utf-8")
    monkeypatch.setattr(i18n, "_locales_dir", lambda: fake_locales)
    i18n.reset_language_cache()
    try:
        assert i18n.t("foo", lang="zh") == "English Foo"
    finally:
        # Clear the cache on teardown so subsequent tests don't see the
        # fake "foo: English Foo" catalog instead of the real locales/*.yaml.
        i18n.reset_language_cache()




# ---------------------------------------------------------------------------
# _locales_dir resolution ladder -- regression for #23943 / #27632 / #35374.
# Sealed installs (Nix store venv, pip wheel) have no source tree next to
# agent/, so _locales_dir must resolve via env override or the data scheme.
# ---------------------------------------------------------------------------



def test_locales_dir_env_override_ignored_when_missing(tmp_path, monkeypatch):
    """A bogus HERMES_BUNDLED_LOCALES falls through to source/wheel resolution
    instead of returning a path that doesn't exist."""
    monkeypatch.setenv("HERMES_BUNDLED_LOCALES", str(tmp_path / "does-not-exist"))
    result = i18n._locales_dir()
    assert result != tmp_path / "does-not-exist"
    # In a source checkout this is the repo-root locales dir.
    assert result.name == "locales"


# ---------------------------------------------------------------------------
# Wheel-packaging contract (BUG-kanban-wake-empty-payload, 2026-09-17).
#
# agent/i18n.py resolves catalogs as <parent-of-agent>/locales. In a wheel
# install that is site-packages/locales — a top-level data dir that setuptools
# only ships when it is declared as a package. When the declaration (or the
# __init__.py marker) is lost, wheels install with NO catalogs: t() then
# returns raw dotted keys, and the kanban notifier's wake notes arrive as
# `gateway.kanban.wake.message ...` with the task id, title, status and
# handoff payload silently dropped — the woken agent cannot tell what woke
# it. These tests pin the two halves of that contract.
# ---------------------------------------------------------------------------


def test_locales_dir_is_a_declared_package():
    """The catalog dir must carry the packaging marker.

    Deleting the marker re-breaks wheel builds (catalogs vanish) while every
    source-checkout test keeps passing — exactly how the bug shipped.
    """
    assert (LOCALES_DIR / "__init__.py").is_file()


def test_kanban_wake_keys_render_with_payload():
    """Every wake-note key the kanban notifier references must resolve and
    interpolate its payload — the woken agent's ONLY cue about what happened
    (raw-key output means an information-free wake turn)."""
    cases = {
        "gateway.kanban.wake.message": dict(
            task_id="t_deadbeef",
            status="completed",
            title="Fix the thing",
            assignee="brokk",
            board="main",
        ),
        "gateway.kanban.wake.handoff": dict(summary="did the work"),
    }
    plain_keys = [
        "gateway.kanban.wake.guidance",
        "gateway.kanban.wake.completed",
        "gateway.kanban.wake.gave_up",
        "gateway.kanban.wake.crashed",
        "gateway.kanban.wake.timed_out",
        "gateway.kanban.wake.blocked",
        "gateway.kanban.wake.status_joiner",
        "gateway.kanban.wake.status_default",
    ]
    rendered = i18n.t("gateway.kanban.wake.message", **cases["gateway.kanban.wake.message"])
    # Resolved template, not a bare key, and the payload survived formatting.
    assert "t_deadbeef" in rendered
    assert "Fix the thing" in rendered
    assert rendered != "gateway.kanban.wake.message"
    handoff = i18n.t("gateway.kanban.wake.handoff", summary="did the work")
    assert "did the work" in handoff
    assert handoff != "gateway.kanban.wake.handoff"
    for key in plain_keys:
        value = i18n.t(key)
        assert value and value != key, f"missing translation for {key}"


