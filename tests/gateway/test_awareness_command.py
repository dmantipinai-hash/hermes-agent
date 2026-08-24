"""Tests for the gateway `/awareness` handler.

Asserts the mode toggle persists to config.yaml and evicts the cached
agent (so the next message picks it up), and that ``status`` reads the
live controller from the session's cached agent when present.
"""

from __future__ import annotations

from types import SimpleNamespace

import yaml

import pytest

from agent.awareness import AwarenessController
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_source(),
        message_id="m1",
        internal=True,
    )


def _make_runner(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    hh = tmp_path / ".hermes"
    hh.mkdir()
    (hh / "config.yaml").write_text("awareness:\n  mode: auto\n")
    monkeypatch.setattr("gateway.run._hermes_home", hh)

    runner = object.__new__(GatewayRunner)
    runner._session_key_for_source = lambda _source: "telegram:u1:c1"
    runner._agent_cache = {}
    evicted = []
    runner._evict_cached_agent = lambda key: evicted.append(key)
    return runner, hh, evicted


@pytest.mark.asyncio
async def test_awareness_off_persists_and_evicts_cached_agent(tmp_path, monkeypatch):
    runner, hh, evicted = _make_runner(tmp_path, monkeypatch)
    out = await runner._handle_awareness_command(_event("/awareness off"))
    assert "Awareness: off" in out
    assert yaml.safe_load((hh / "config.yaml").read_text())["awareness"]["mode"] == "off"
    assert evicted == ["telegram:u1:c1"]


@pytest.mark.asyncio
async def test_awareness_status_reads_live_controller(tmp_path, monkeypatch):
    runner, hh, _ = _make_runner(tmp_path, monkeypatch)
    live = AwarenessController(None, {})
    live.episodes_detected = 3
    live.notes_injected = 2
    runner._agent_cache["telegram:u1:c1"] = SimpleNamespace(_awareness=live)
    out = await runner._handle_awareness_command(_event("/awareness"))
    assert "Awareness: auto" in out
    assert "episodes detected: 3" in out.lower()
    assert "notes injected: 2" in out.lower()


@pytest.mark.asyncio
async def test_awareness_unknown_arg_prints_usage(tmp_path, monkeypatch):
    runner, _hh, _ = _make_runner(tmp_path, monkeypatch)
    out = await runner._handle_awareness_command(_event("/awareness bogus"))
    assert "Use: status, on (auto), off" in out
