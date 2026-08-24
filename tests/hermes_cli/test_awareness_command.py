"""Tests for the `/awareness` slash command (CLI surface).

Asserts the toggle persists ``awareness.mode`` to the user's config.yaml,
flips the live controller when one exists, and that ``status`` reflects
both. Follows the `/timestamps` command-test pattern.
"""

from __future__ import annotations

import yaml

from agent.awareness import AwarenessController
from hermes_cli.cli_commands_mixin import CLICommandsMixin


class _Stub(CLICommandsMixin):
    def __init__(self, awareness=None):
        self.agent = type("A", (), {"_awareness": awareness})() if awareness else None


def _seed(tmp_path, monkeypatch, mode="auto"):
    hh = tmp_path / ".hermes"
    hh.mkdir()
    (hh / "config.yaml").write_text(f"awareness:\n  mode: {mode}\n")
    monkeypatch.setenv("HERMES_HOME", str(hh))
    import cli

    monkeypatch.setattr(cli, "_hermes_home", hh, raising=False)
    return hh


def test_awareness_off_persists_and_flips_live_controller(tmp_path, monkeypatch):
    hh = _seed(tmp_path, monkeypatch)
    live = AwarenessController(None, {})
    s = _Stub(awareness=live)
    s._handle_awareness_command("/awareness off")
    # ruamel may keep the plain (unquoted) scalar style, which YAML 1.1
    # readers parse as boolean False — the contract is what a freshly
    # constructed controller reads, not the raw scalar text.
    from agent.awareness import normalize_mode
    raw = yaml.safe_load((hh / "config.yaml").read_text())["awareness"]["mode"]
    assert normalize_mode(raw) == "off"
    assert AwarenessController(None, {"mode": raw}).mode == "off"
    assert live.mode == "off"


def test_awareness_on_persists_and_flips_live_controller(tmp_path, monkeypatch):
    hh = _seed(tmp_path, monkeypatch, mode="off")
    live = AwarenessController(None, {"mode": "off"})
    s = _Stub(awareness=live)
    s._handle_awareness_command("/awareness on")
    assert yaml.safe_load((hh / "config.yaml").read_text())["awareness"]["mode"] == "auto"
    assert live.mode == "auto"


def test_awareness_status_reports_live_counters(tmp_path, monkeypatch, capsys):
    _seed(tmp_path, monkeypatch)
    live = AwarenessController(None, {})
    live.episodes_detected = 2
    live.patterns_recorded = 1
    s = _Stub(awareness=live)
    s._handle_awareness_command("/awareness")
    out = capsys.readouterr().out
    assert "Awareness: auto" in out
    assert "episodes detected: 2" in out.lower()
    assert "patterns recorded: 1" in out.lower()


def test_awareness_status_without_agent_reads_config(tmp_path, monkeypatch, capsys):
    _seed(tmp_path, monkeypatch, mode="off")
    s = _Stub(awareness=None)
    s._handle_awareness_command("/awareness status")
    out = capsys.readouterr().out
    assert "Awareness: off (config)" in out


def test_awareness_deep_is_honest_about_missing_phase(tmp_path, monkeypatch, capsys):
    _seed(tmp_path, monkeypatch)
    s = _Stub(awareness=None)
    s._handle_awareness_command("/awareness deep")
    assert "not implemented" in capsys.readouterr().out


def test_awareness_unknown_arg_prints_usage(tmp_path, monkeypatch, capsys):
    _seed(tmp_path, monkeypatch)
    s = _Stub(awareness=None)
    s._handle_awareness_command("/awareness bogus")
    assert "Use: status, on (auto), off" in capsys.readouterr().out
