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
BEEP_WAV = Path("/tmp/fyp_beep.wav")
# Longest the microphone stays open after the beep. Listening normally ends
# sooner, as soon as the wearer stops talking.
LISTEN_SECONDS = float(os.environ.get("FYP_LISTEN_SECONDS", "7"))
SAMPLE_RATE = 16000          # HFP/mSBC rate, and what the Vosk model expects

# The SCO link takes about a second to carry audio after it is opened, and
# until then the capture is pure zeros. Anything said in that second is lost,
# which is why commands used to arrive with their first words missing. So the
# mic is opened first, and only once real samples are flowing does a beep -
# played down the same SCO link - tell the wearer to speak.
LINK_TIMEOUT_S = 4.0         # give up if no audio arrives in this long
BEEP_HZ, BEEP_S, BEEP_LEVEL = 880, 0.18, 0.35
AFTER_BEEP_S = 0.15          # skip the beep's own tail in the capture
CHUNK_S = 0.1                # how often the capture is examined
# End of speech: a chunk counts as speech when its peak clears SPEECH_PEAK
# (the open mic idles around 30-50, speech peaks in the thousands), and
# listening stops after TRAILING_SILENCE_S without any.
SPEECH_PEAK = 600
TRAILING_SILENCE_S = 1.0
NO_SPEECH_TIMEOUT_S = 4.0    # nobody said anything: stop waiting

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
    "what emotion", "what expression", "is he happy", "is she happy",
    "are they happy", "who is there", "is anyone there",
    "find", "where is", "look for", "locate", "how far is", "how far away is",
]


# Words absent from the US English model's dictionary. Keeping them in the
# grammar only produces "Ignoring word missing in vocabulary" warnings; they
# stay in finder's synonym table, which matches on text, not audio.
_NOT_IN_VOCABULARY = {"handphone"}

# Things people ask for that the detector has no class for. They are in the
# grammar so that they are heard as themselves and turned down ("I cannot
# look for that"); left out, "keys" comes back as "keyboard" and "door" as
# "dog", and the wearer is sent after the wrong thing.
_UNFINDABLE = ["keys", "key", "door", "wallet", "glasses", "shoes", "pen",
               "card", "charger", "headphones", "window", "stairs", "exit"]

# In grammar mode Vosk must answer with words from the grammar, so speech
# from outside it is bent onto whatever sounds nearest. Its confidence gives
# that away: words really spoken score 1.00, forced ones well under 0.85.
# A grammar result with any weaker word is not trusted. The little words
# below are exempt - they are slurred in natural speech and carry no meaning.
GRAMMAR_MIN_WORD_CONF = 0.85
_FILLER_WORDS = frozenset({"the", "a", "my", "is", "it", "of", "me", "this"})


def _grammar_phrases():
    """Command phrases plus every object name the finder can resolve."""
    from fypvision import finder
    objects = sorted((set(finder._SYNONYMS.keys())
                      | set(finder._SYNONYMS.values())) - _NOT_IN_VOCABULARY)
    return (_COMMAND_PHRASES + objects + _UNFINDABLE
            + ["the", "a", "my", "[unk]"])


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


def _decode(pcm_bytes, grammar: bool) -> tuple:
    """Run one recognition pass; return (text, weakest word confidence).

    Recognisers are stateful, so a fresh one is built per utterance rather
    than reused and reset.
    """
    model = _load_model()
    if model is None:
        return "", 0.0
    import vosk
    if grammar:
        recogniser = vosk.KaldiRecognizer(
            model, SAMPLE_RATE, json.dumps(_grammar_phrases()))
    else:
        recogniser = vosk.KaldiRecognizer(model, SAMPLE_RATE)
    recogniser.SetWords(True)
    for i in range(0, len(pcm_bytes), 8000):
        recogniser.AcceptWaveform(pcm_bytes[i:i + 8000])
    result = json.loads(recogniser.FinalResult())
    # "[unk]" tokens are Vosk saying "outside the grammar" - not words.
    words = [w for w in result.get("result", []) if w["word"] != "[unk]"]
    heard_unknown = len(words) < len(result.get("result", []))
    weakest = min((w["conf"] for w in words
                   if w["word"] not in _FILLER_WORDS), default=1.0)
    text = " ".join(w["word"] for w in words)
    # Part of the sentence fell outside the grammar: what is left is a
    # fragment, not a command.
    return text, (0.0 if heard_unknown else weakest)


def _ensure_beep() -> Path:
    """Write the cue tone once, in the SCO link's own format."""
    if not BEEP_WAV.exists():
        t = np.arange(int(SAMPLE_RATE * BEEP_S)) / SAMPLE_RATE
        fade = np.minimum(1.0, np.minimum(t, BEEP_S - t) / 0.01)
        tone = np.sin(2 * np.pi * BEEP_HZ * t) * fade * BEEP_LEVEL * 32767
        with wave.open(str(BEEP_WAV), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(SAMPLE_RATE)
            out.writeframes(tone.astype("<i2").tobytes())
    return BEEP_WAV


def _read_chunk(stream, n_bytes: int) -> np.ndarray:
    """Next block of samples from the capture pipe (empty at end of stream)."""
    data = stream.read(n_bytes)
    return np.frombuffer(data[:len(data) // 2 * 2], dtype="<i2")


def _wait_for_link(stream, chunk_bytes: int) -> bool:
    """Discard capture until real audio arrives. False if it never does."""
    deadline = time.time() + LINK_TIMEOUT_S
    while time.time() < deadline:
        chunk = _read_chunk(stream, chunk_bytes)
        if chunk.size == 0:
            return False             # arecord exited
        if np.any(chunk):
            return True
    return False


def _capture_utterance(stream, chunk_bytes: int, max_seconds: float) -> tuple:
    """Collect samples until the wearer stops talking.

    Returns (samples, why it stopped), for the log.
    """
    chunks, spoke, quiet_s, elapsed = [], False, 0.0, 0.0
    while elapsed < max_seconds:
        chunk = _read_chunk(stream, chunk_bytes)
        if chunk.size == 0:
            return chunks, "capture ended"
        chunks.append(chunk)
        elapsed += CHUNK_S
        if int(np.abs(chunk.astype(np.int32)).max()) >= SPEECH_PEAK:
            spoke, quiet_s = True, 0.0
        else:
            quiet_s += CHUNK_S
        if spoke and quiet_s >= TRAILING_SILENCE_S:
            return chunks, "end of speech"
        if not spoke and elapsed >= NO_SPEECH_TIMEOUT_S:
            return chunks, "no speech"
    return chunks, "time limit"


def record(mac: str, seconds: float = LISTEN_SECONDS):
    """Capture one spoken command from the headset mic. Returns a WAV path,
    or None if the microphone could not be opened.

    Opens the mic, waits for the link to carry audio, beeps, then records
    until the wearer stops talking (or `seconds` at most).
    """
    device = f"bluealsa:DEV={mac},PROFILE=sco"
    chunk_bytes = int(SAMPLE_RATE * CHUNK_S) * 2
    capture = None
    try:
        capture = subprocess.Popen(
            ["arecord", "-q", "-D", device, "-f", "S16_LE",
             "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not _wait_for_link(capture.stdout, chunk_bytes):
            print("[VOICE] microphone link never produced audio")
            return None

        beep = subprocess.run(["aplay", "-q", "-D", device,
                               str(_ensure_beep())],
                              capture_output=True, timeout=5)
        if beep.returncode != 0:
            print("[VOICE] could not play the listening beep")
        # Drop what was captured while the beep played: it is the beep.
        _read_chunk(capture.stdout,
                    int(SAMPLE_RATE * (BEEP_S + AFTER_BEEP_S)) * 2)
        capture.stdout.read1(1 << 20)

        chunks, reason = _capture_utterance(capture.stdout, chunk_bytes,
                                            float(seconds))
        samples = (np.concatenate(chunks) if chunks
                   else np.zeros(0, dtype="<i2"))
        print(f"[VOICE] recorded {samples.size / SAMPLE_RATE:.1f}s "
              f"({reason})")
        with wave.open(str(CAPTURE_WAV), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(SAMPLE_RATE)
            out.writeframes(samples.astype("<i2").tobytes())
        return CAPTURE_WAV
    except Exception as e:
        print(f"[VOICE] recording failed: {e}")
        return None
    finally:
        if capture is not None:
            capture.terminate()
            try:
                _, err = capture.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                capture.kill()
                _, err = capture.communicate()
            notes = [ln for ln in err.decode(errors="replace").splitlines()
                     if ln.strip() and "] D: " not in ln
                     and "Aborted by signal" not in ln
                     and "Interrupted system call" not in ln]
            if notes:
                print(f"[VOICE] arecord: {' | '.join(notes)[-300:]}")
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
    with a small fixed vocabulary, and keeps that answer only if every word
    was heard confidently. Otherwise falls back to open dictation, which is
    less accurate on real commands but does not invent one from other speech.
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
        text, weakest = _decode(data, grammar=True)
        if text and weakest >= GRAMMAR_MIN_WORD_CONF:
            print(f"[VOICE] heard (grammar): {text!r}")
            return text
        if text:
            print(f"[VOICE] not trusting grammar match {text!r} "
                  f"(weakest word {weakest:.2f})")

        text, _ = _decode(data, grammar=False)
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
                  "look for", "looking for", "how far", "anyone",
                  "who is there")),
    ("currency", ("money", "note", "cash", "ringgit", "dollar", "currency",
                  "denomination", "how much")),
    ("emotion",  ("emotion", "feeling", "mood", "expression", "face",
                  "he look", "she look", "they look", "look happy",
                  "look sad", "looks like", "happy")),
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
