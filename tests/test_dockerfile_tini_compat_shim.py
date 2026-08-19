"""Regression test for #34192 — Dockerfile must keep the tini compat shim
for orchestration templates that still reference /usr/bin/tini.

This is a documentation-as-test guard: removing the shim is a real
choice, but it should be done deliberately (e.g. once Hostinger's
'Hermes WebUI' catalog updates to /init) and not by accident.

The mechanism evolved past the original `ln -sf /init` symlink: a plain
symlink forwarded tini CLI flags into s6 rc.init and boot-looped
`restart: unless-stopped` deploys, so the compat path is now a shim
script (docker/tini-shim.sh) that strips the tini flag surface and
exec's /init + main-wrapper. ENTRYPOINT is the dispatcher, not /init
directly — the dispatcher keeps docker's `--init`-style contracts and
forwards to s6.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_tini_compat_shim_present():
    df = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY --chmod=0755 docker/tini-shim.sh /usr/bin/tini" in df, (
        "Dockerfile must keep the tini compat shim (#34192). Removing it "
        "breaks orchestration templates that still pin /usr/bin/tini as the "
        "entrypoint (Hostinger 'Hermes WebUI' catalog as of v0.14.x)."
    )
    assert (ROOT / "docker" / "tini-shim.sh").exists(), (
        "the shim script referenced by the Dockerfile must ship in docker/"
    )


def test_tini_compat_comment_explains_why():
    df = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "#34192" in df, (
        "The compat shim needs its rationale comment: it is deliberately "
        "load-bearing until external catalogs move off /usr/bin/tini."
    )


def test_entrypoint_is_dispatcher():
    df = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert 'ENTRYPOINT [ "/opt/hermes/docker/entrypoint-dispatch.sh" ]' in df, (
        "Dockerfile ENTRYPOINT must remain the dispatcher (which exec's "
        "s6-overlay's /init); the tini shim is only for external wrappers "
        "that haven't been updated yet."
    )
