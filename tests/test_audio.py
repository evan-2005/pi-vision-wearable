"""Where speech is sent, decided afresh for every utterance."""

import subprocess

import pytest

from fypvision import audio

MAC = "98:80:BB:80:4E:CA"


def _connected(monkeypatch, mac):
    monkeypatch.setattr(audio, "detect_headset_mac", lambda: mac)


def test_connected_headset_is_used(monkeypatch):
    _connected(monkeypatch, MAC)

    assert audio.output_device() == f"plug:'bluealsa:DEV={MAC},PROFILE=a2dp'"


def test_nowhere_to_play_when_no_headset_is_connected(monkeypatch):
    _connected(monkeypatch, None)

    assert audio.output_device() is None


def test_headset_that_connects_later_is_picked_up(monkeypatch):
    _connected(monkeypatch, None)
    before = audio.output_device()
    _connected(monkeypatch, MAC)

    assert before is None
    assert MAC in audio.output_device()


def test_configured_mac_is_only_a_fallback(monkeypatch):
    _connected(monkeypatch, None)
    assert "AA:AA" in audio.output_device(fallback_mac="AA:AA")

    _connected(monkeypatch, MAC)
    assert MAC in audio.output_device(fallback_mac="AA:AA")


@pytest.mark.parametrize("override", ["default", "none", "", "DEFAULT"])
def test_override_to_onboard_output_ignores_bluetooth(monkeypatch, override):
    _connected(monkeypatch, MAC)

    assert audio.output_device(override) == audio.ONBOARD_DEVICE


def test_other_override_is_used_verbatim(monkeypatch):
    _connected(monkeypatch, MAC)

    assert audio.output_device("null") == "null"


def test_detect_reads_the_first_connected_device(monkeypatch):
    listing = f"Device {MAC} SOUNDPEATS RunFree Lite2\n"
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=listing))

    assert audio.detect_headset_mac() == MAC


def test_detect_survives_a_missing_bluetoothctl(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("bluetoothctl")
    monkeypatch.setattr(subprocess, "run", missing)

    assert audio.detect_headset_mac() is None
