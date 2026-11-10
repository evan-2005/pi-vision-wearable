"""When listening starts and stops, driven by a fake microphone stream."""

import io

import numpy as np

from fypvision import voice

CHUNK = int(voice.SAMPLE_RATE * voice.CHUNK_S)
CHUNK_BYTES = CHUNK * 2


def _stream(*segments) -> io.BytesIO:
    """A capture pipe holding (seconds, amplitude) stretches of audio."""
    parts = [np.full(int(round(seconds / voice.CHUNK_S)) * CHUNK, level, "<i2")
             for seconds, level in segments]
    return io.BytesIO(np.concatenate(parts).tobytes())


def _capture(stream, max_seconds=7.0):
    chunks, reason = voice._capture_utterance(stream, CHUNK_BYTES, max_seconds)
    return sum(c.size for c in chunks) / voice.SAMPLE_RATE, reason


def test_link_is_live_once_samples_stop_being_zero():
    assert voice._wait_for_link(_stream((1.0, 0), (0.5, 40)), CHUNK_BYTES)


def test_link_that_closes_without_audio_is_reported():
    assert not voice._wait_for_link(_stream((0.3, 0)), CHUNK_BYTES)


def test_listening_stops_shortly_after_the_speaker_does():
    seconds, reason = _capture(_stream((0.5, 40), (1.5, 9000), (5.0, 40)))

    assert reason == "end of speech"
    assert 2.0 + voice.TRAILING_SILENCE_S - 0.2 <= seconds \
        <= 2.0 + voice.TRAILING_SILENCE_S + 0.2


def test_a_pause_inside_a_sentence_does_not_end_it():
    pause = voice.TRAILING_SILENCE_S / 2
    seconds, reason = _capture(
        _stream((1.0, 9000), (pause, 40), (1.0, 9000), (5.0, 40)))

    assert reason == "end of speech"
    assert seconds > 2.0 + pause


def test_silence_gives_up_early():
    seconds, reason = _capture(_stream((10.0, 40)))

    assert reason == "no speech"
    assert abs(seconds - voice.NO_SPEECH_TIMEOUT_S) < 0.2


def test_nonstop_talking_is_cut_at_the_limit():
    seconds, reason = _capture(_stream((20.0, 9000)), max_seconds=3.0)

    assert reason == "time limit"
    assert abs(seconds - 3.0) < 0.2


def test_capture_pipe_closing_mid_sentence_keeps_what_was_heard():
    seconds, reason = _capture(_stream((0.6, 9000)))

    assert reason == "capture ended"
    assert abs(seconds - 0.6) < 0.01
