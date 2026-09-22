"""Listen through the headset mic and work out what was asked.

The headset is the only microphone available, so listening means opening an
HFP/SCO link. That link and A2DP music playback do not coexist comfortably:
holding both is what made this headset stutter before. So recording and
speaking are strictly sequential here - the mic is opened, read, and closed
again before anything is spoken, with a short settle pause to let the
headset switch profiles back.

Recognition is Vosk, running fully offline against a small English model,
which keeps the device usable with no network. Its vocabulary is limited
and it will mishear; parse_intent() therefore matches on keywords rather
than exact phrases, and returns ("unknown", "") when it is not confident,
so the caller can say "I did not catch that" instead of acting on noise.
"""

import json
import os
import subprocess
import time
import wave
from pathlib import Path

import numpy as np

MODEL_DIR = Path(os.environ.get("FYP_MODEL_DIR", str(Path.home() / "fyp" / "models")))
VOSK_MODEL = MODEL_DIR / "vosk-model-small-en-us-0.15"

CAPTURE_WAV = Path("/tmp/fyp_listen.wav")
LISTEN_SECONDS = float(os.environ.get("FYP_LISTEN_SECONDS", "5"))
SAMPLE_RATE = 16000          # HFP/mSBC rate, and what the Vosk model expects

# After closing the SCO link the headset needs a moment to fall back to A2DP.
# Speaking immediately clips the first word.
PROFILE_SETTLE_S = 0.6

_model = None

# Everything the wearer can usefully say. Vosk accepts a grammar and will then
# only decode these words, which is far more accurate than open dictation on
# the small model - free-form, "what is in front of me" came back as "i'm not
# me". "[unk]" lets it report that it heard something outside the grammar
# instead of forcing a wrong match.
_COMMAND_PHRASES = [
    "what is in front of me", "what is in front", "what do you see",
    "what is there", "describe the scene", "describe", "what is this",
    "read this", "read the text", "read the sign", "what does it say",
    "what is written",
    "what note is this", "how much money is this", "what currency is this",
    "how much is this note",
    "how does he look", "how does she look", "how do they look",
    "what emotion", "what expression",
    "find", "where is", "look for", "locate",
]


# Words absent from the US English model's dictionary. Keeping them in the
# grammar only produces "Ignoring word missing in vocabulary" warnings; they
# stay in finder's synonym table, which matches on text, not audio.
_NOT_IN_VOCABULARY = {"handphone"}


def _grammar_phrases():
    """Command phrases plus every object name the finder can resolve."""
    from fypvision import finder
    objects = sorted((set(finder._SYNONYMS.keys())
                      | set(finder._SYNONYMS.values())) - _NOT_IN_VOCABULARY)
    return _COMMAND_PHRASES + objects + ["the", "a", "my", "[unk]"]


def available() -> bool:
    return VOSK_MODEL.exists()


def _load_model():
    """Load the Vosk model once; returns None if unavailable."""
    global _model
    if _model is not None:
        return _model
    if not available():
        print(f"[VOICE] Vosk model not found at {VOSK_MODEL}")
        return None
    try:
        import vosk
        vosk.SetLogLevel(-1)          # the C library is very chatty otherwise
        _model = vosk.Model(str(VOSK_MODEL))
        return _model
    except Exception as e:
        print(f"[VOICE] could not load Vosk: {e}")
        return None


def _load_recogniser():
    """Back-compat warm-up hook: pre-loads the model."""
    return _load_model()


def _decode(pcm_bytes, grammar: bool):
    """Run one recognition pass. Recognisers are stateful, so build a fresh
    one per utterance rather than reusing and resetting."""
    model = _load_model()
    if model is None:
        return ""
    import vosk
    if grammar:
        recogniser = vosk.KaldiRecognizer(
            model, SAMPLE_RATE, json.dumps(_grammar_phrases()))
    else:
        recogniser = vosk.KaldiRecognizer(model, SAMPLE_RATE)
    for i in range(0, len(pcm_bytes), 8000):
        recogniser.AcceptWaveform(pcm_bytes[i:i + 8000])
    text = json.loads(recogniser.FinalResult()).get("text", "").strip()
    # "[unk]" tokens are Vosk saying "outside the grammar" - not words.
    return " ".join(w for w in text.split() if w != "[unk]").strip()


def record(mac: str, seconds: float = LISTEN_SECONDS):
    """Capture from the headset mic over HFP. Returns a path, or None."""
    device = f"bluealsa:DEV={mac},PROFILE=sco"
    # arecord's -d takes whole seconds only: "7.0" is rejected outright.
    duration = max(1, int(round(float(seconds))))
    cmd = ["arecord", "-q", "-D", device, "-f", "S16_LE",
           "-r", str(SAMPLE_RATE), "-c", "1", "-d", str(duration),
           str(CAPTURE_WAV)]
    try:
        result = subprocess.run(cmd, capture_output=True,
                                timeout=duration + 15)
        if result.returncode != 0:
            notes = [ln for ln in result.stderr.decode(errors="replace").splitlines()
                     if ln.strip() and "] D: " not in ln]
            print(f"[VOICE] arecord exited {result.returncode}: "
                  f"{' | '.join(notes)[-300:]}")
            return None
        return CAPTURE_WAV if CAPTURE_WAV.exists() else None
    except Exception as e:
        print(f"[VOICE] recording failed: {e}")
        return None
    finally:
        # Always pause, even on failure: the SCO link may still be tearing
        # down, and the caller's next action is usually to speak.
        time.sleep(PROFILE_SETTLE_S)


# The headset mic arrives very quiet over HFP - measured peaks around 8% of
# full scale with the Bluetooth gain already at its maximum of 15. Vosk
# performs poorly on signal that weak, so the capture is normalised first.
TARGET_PEAK_FRACTION = 0.70
MAX_GAIN = 25.0          # beyond this we would just be amplifying hiss
SILENCE_PEAK = 120       # peak below this is background noise, not speech


def _normalise_audio(samples):
    """Scale a mono int16 array up towards full scale. Returns (array, gain)."""
    peak = int(np.abs(samples).max()) if samples.size else 0
    if peak < SILENCE_PEAK:
        return samples, 0.0          # nothing worth amplifying
    gain = min((TARGET_PEAK_FRACTION * 32767.0) / peak, MAX_GAIN)
    if gain <= 1.0:
        return samples, 1.0
    boosted = np.clip(samples.astype(np.float32) * gain, -32768, 32767)
    return boosted.astype(np.int16), gain


def transcribe(wav_path: Path) -> str:
    """Offline speech-to-text. Returns "" if nothing was recognised.

    Tries the constrained grammar first, since this is command-and-control
    with a small fixed vocabulary. Falls back to open dictation so that a
    phrase nobody anticipated still produces something to log.
    """
    if wav_path is None or _load_model() is None:
        return ""
    try:
        with wave.open(str(wav_path), "rb") as w:
            if w.getnchannels() != 1 or w.getframerate() != SAMPLE_RATE:
                print(f"[VOICE] unexpected capture format: "
                      f"{w.getframerate()} Hz, {w.getnchannels()} ch")
            raw = w.readframes(w.getnframes())

        samples = np.frombuffer(raw, dtype="<i2")
        peak_before = int(np.abs(samples).max()) if samples.size else 0
        samples, gain = _normalise_audio(samples)
        print(f"[VOICE] capture peak {peak_before}, gain x{gain:.1f}")
        if gain == 0.0:
            print("[VOICE] capture is silence; not transcribing")
            return ""

        data = samples.tobytes()
        text = _decode(data, grammar=True)
        if text:
            print(f"[VOICE] heard (grammar): {text!r}")
            return text

        text = _decode(data, grammar=False)
        print(f"[VOICE] heard (free): {text!r}")
        return text
    except Exception as e:
        print(f"[VOICE] transcription failed: {e}")
        return ""


def listen(mac: str, seconds: float = LISTEN_SECONDS) -> str:
    """Record from the headset and return what was said (may be "")."""
    return transcribe(record(mac, seconds))


# --- intent routing -------------------------------------------------------
# Keyword sets rather than exact phrases: the small Vosk model drops and
# swaps words constantly, so "what is in front of me" often arrives as
# "what in front me". Order matters - the first intent that matches wins,
# so the more specific ones are listed first.
_INTENTS = [
    ("find",     ("find", "where is", "wheres", "where are", "locate",
                  "look for", "looking for")),
    ("currency", ("money", "note", "cash", "ringgit", "dollar", "currency",
                  "denomination", "how much")),
    ("emotion",  ("emotion", "feeling", "mood", "expression", "face",
                  "he look", "she look", "they look", "look happy",
                  "look sad", "looks like")),
    ("read",     ("read", "text", "written", "writing", "sign", "label",
                  "say", "says", "saying")),
    ("detect",   ("in front", "what do you see", "whats there",
                  "what is there", "around me", "describe", "what is this",
                  "whats this", "see")),
]


def _normalise(text: str) -> str:
    """Lowercase, drop apostrophes, collapse spaces.

    Vosk emits unpunctuated lowercase text, so "what's there" arrives as
    "whats there". Normalising both sides keeps the keyword lists readable.
    """
    return " ".join(text.lower().replace("'", "").split())


def parse_intent(text: str):
    """(intent, argument) for a transcript. Intent is "unknown" if unclear.

    The argument carries the original text so "find my bottle" can have its
    target resolved downstream by finder.normalise_target().
    """
    if not text:
        return "unknown", ""
    lowered = _normalise(text)
    for intent, keywords in _INTENTS:
        if any(k in lowered for k in keywords):
            return intent, text
    return "unknown", text
