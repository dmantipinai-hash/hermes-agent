"""Fork guard contracts: `hermes update` must never pull upstream.

This tree is the dmantipinai-hash fork (Architecture 2.0 — memory v2,
message delivery, multi-agent orchestration), which no upstream release
contains. Every automatic update channel in ``hermes_cli/update_cmd.py``
resolves to the upstream NousResearch release:

- ``_cmd_update_pip`` upgrades the bare ``hermes-agent`` name from PyPI;
- ``_update_via_zip`` downloads the NousResearch GitHub archive;
- ``_sync_with_upstream_if_needed`` offers an upstream remote and ff-pulls
  from it (the exact mechanism that once erased a prior upstream sync).

``FORK_REPO_URL`` in ``hermes_cli/__init__.py`` marks this build as the
fork; the channels must refuse (or no-op) while it is set. The tests that
exercise the underlying mechanics with the marker neutralized live next to
them (``_upstream_mechanics_mode`` fixtures).
"""

import subprocess
from pathlib import Path

import pytest


def test_fork_marker_declares_the_fork_repo():
    from hermes_cli import FORK_REPO_URL

    assert FORK_REPO_URL == "https://github.com/dmantipinai-hash/hermes-agent.git"


def test_pip_update_refuses_and_prints_the_fork_command(capsys):
    """The PyPI path must exit before touching pip/uv and say how to update."""
    from hermes_cli.update_cmd import _cmd_update_pip

    with pytest.raises(SystemExit) as excinfo:
        _cmd_update_pip(args=None)

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "dmantipinai-hash/hermes-agent" in out
    assert "uv tool install" in out


def test_zip_update_refuses_before_any_download(capsys, monkeypatch):
    """The archive-ZIP fallback must exit before urlretrieve is even imported."""
    from hermes_cli import update_cmd

    def _no_subprocess(*args, **kwargs):
        raise AssertionError("fork zip update must not spawn any subprocess")

    monkeypatch.setattr(subprocess, "run", _no_subprocess)

    with pytest.raises(SystemExit) as excinfo:
        update_cmd._update_via_zip(args=None)

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "dmantipinai-hash/hermes-agent" in out


def test_upstream_sync_is_a_silent_noop_for_the_fork(monkeypatch, capsys, tmp_path):
    """No prompt, no remote add, no fetch — the fork takes upstream changes
    only via deliberate maintainer-run merges, never automatically after an
    update."""
    from hermes_cli import update_cmd

    def _no_subprocess(*args, **kwargs):
        raise AssertionError("fork build must not touch git for upstream sync")

    monkeypatch.setattr(subprocess, "run", _no_subprocess)

    # Must not raise and must not block on stdin (input() would under pytest)
    update_cmd._sync_with_upstream_if_needed(["git"], Path(tmp_path))

    out = capsys.readouterr().out
    assert "NousResearch" not in out


def test_banner_does_not_count_fork_installs_against_pypi(monkeypatch, tmp_path):
    """A fork pip/uv-tool install must never report 'behind PyPI' — PyPI
    carries only the upstream release, and the nag pushes users into the
    very channel the guard refuses. The PyPI probe must not even run."""
    from hermes_cli import banner
    import hermes_cli.config as config

    def _no_pypi_probe():
        raise AssertionError("fork builds must not probe PyPI for the behind count")
        # pragma: no cover

    monkeypatch.setattr(banner, "check_via_pypi", _no_pypi_probe)
    monkeypatch.delenv("HERMES_REVISION", raising=False)
    # Non-docker/apt install; and no git checkout anywhere (the banner
    # resolves the repo dir from its own __file__, then from HERMES_HOME —
    # point both at a site-packages-looking tmp path without .git).
    monkeypatch.setattr(config, "detect_install_method", lambda *_a, **_k: "pip")
    monkeypatch.setattr(
        banner, "__file__", str(tmp_path / "site-packages" / "banner.py")
    )

    assert banner.check_for_updates() is None
