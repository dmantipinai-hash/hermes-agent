from pathlib import Path
from types import SimpleNamespace

from gateway.config import GatewayConfig, load_gateway_config
from gateway.run import GatewayRunner


def test_stt_echo_transcripts_defaults_off_for_the_fork():
    """Fork contract: the transcript echo is opt-in.

    Before the upstream merge, a voice message's transcript was agent-internal
    only — the user's chat never saw it. The upstream echo feature (🎙️
    "<transcript>" on voice interrupts) shipped default-ON; the fork restores
    the quiet pre-merge default. `stt.echo_transcripts: true` opts back in.
    """
    cfg = GatewayConfig.from_dict({})

    assert cfg.stt_enabled is True
    assert cfg.stt_echo_transcripts is False
    assert cfg.to_dict()["stt_echo_transcripts"] is False


def test_stt_echo_transcripts_can_be_opted_in():
    cfg = GatewayConfig.from_dict({"stt": {"echo_transcripts": True}})

    assert cfg.stt_echo_transcripts is True


def test_top_level_stt_echo_transcripts_takes_precedence():
    cfg = GatewayConfig.from_dict({
        "stt_echo_transcripts": False,
        "stt": {"echo_transcripts": True},
    })

    assert cfg.stt_echo_transcripts is False
