"""Tests for the `hermes memory report` CLI command (Phase-4 P3).

Covers: report renders the audit digest against an isolated HERMES_HOME,
--prune drops rows past the retain window, and an empty log degrades to
zeros instead of crashing.
"""

from __future__ import annotations

from argparse import Namespace

import pytest


@pytest.fixture()
def memory_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    (hermes_home / "memories").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def _run_report(days=7, prune=False):
    from hermes_cli.main import cmd_memory

    args = Namespace(memory_command="report", days=days, prune=prune)
    cmd_memory(args)


def test_report_shows_digest(memory_env, capsys):
    from agent.memory_store_v2 import MemoryStoreV2

    store = MemoryStoreV2()
    store.load_from_disk()
    store.add("memory", "Решение держать VPN на своём сервере")
    store.recall("VPN сервер")
    store.close()

    _run_report(days=7)
    out = capsys.readouterr().out
    assert "Memory v2 health" in out
    assert "memory-read" in out
    assert "Recall events: 1" in out


def test_report_with_prune_removes_old_rows(memory_env, capsys):
    from agent.memory_store_v2 import MemoryStoreV2

    store = MemoryStoreV2()
    store.load_from_disk()
    store.add("memory", "Старая запись для аудита")
    store.recall("старая запись")
    store._execute_write(
        lambda conn: conn.execute(
            "UPDATE memory_recall_log SET ts='2020-01-01T00:00:00+00:00'"
        )
    )
    store.close()

    _run_report(days=7, prune=True)
    out = capsys.readouterr().out
    assert "Pruned 1 audit row(s)" in out


def test_report_on_empty_log_degrades_gracefully(memory_env, capsys):
    _run_report(days=7)
    out = capsys.readouterr().out
    assert "Memory v2 health" in out
    assert "Recall events: 0" in out
