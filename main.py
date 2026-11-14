#!/usr/bin/env python3
"""
FYP Assistive Vision Wearable - Main Pipeline
==============================================
Single press   -> scene detection (YOLOv8n/NCNN) -> spoken summary with
                  the nearest object's bearing and distance
                  (falls back to OCR when nothing is recognised)
Double press   -> OCR (Tesseract) -> spoken text
Long press     -> voice command through the headset microphone
Hold 5 seconds -> say "Shutting down" and power the Pi off cleanly

Additional modes, reachable by voice or from the keyboard:
    detection with distance and bearing for the nearest object
    find one named object ("where is my bottle")
    facial expression of whoever is in view
    banknote denomination, read off the note
    voice commands through the headset microphone (offline, Vosk)

Every trigger also writes an input/annotated image pair to ~/fyp/reports
for the write-up; set FYP_REPORT=0 to turn that off.

Test mode (no GPIO button needed):
    ENTER          = detection
    'l' + ENTER    = read text (OCR)
    'e' + ENTER    = emotion
    'c' + ENTER    = currency note
    'v' + ENTER    = voice command (ask through the headset)
    'f bottle'     = find a named object
    'q' + ENTER    = quit

Run:
    cd ~/fyp && source venv/bin/activate && python3 main.py

Configuration is via the constants below, each overridable by an
environment variable (useful for the systemd service later):

    FYP_USE_GPIO=1        use the GPIO button instead of the keyboard
    FYP_BUTTON_ACTIVE=    low/high/auto: pin level when the button is pressed
    FYP_LONG_PRESS=voice  what a long press does: voice or read
    FYP_AUDIO_DEVICE=...  override the aplay output device
    FYP_HEADSET_MAC=...   headset to try when none is detected (optional)
    FYP_REPORT=0          stop writing report images
    FYP_REPORT_DIR=...    where report images go (default ~/fyp/reports)
    FYP_CAMERA_FOCAL_PX=  measured focal length, for accurate distances
    FYP_CAMERA_ROTATION=  0/90/180/270, how far to turn frames upright
    FYP_CURRENCY=MYR      which currency to recognise
    FYP_LISTEN_SECONDS=5  how long the microphone stays open
"""

import os
import re
import subprocess
import sys
import threading
import time
import wave
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import controls
from ultralytics import YOLO

from fypvision import distance as dist
from fypvision import (audio, button, currency, emotion, finder, ocr,
                       reporting, voice)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
FYP_DIR      = Path.home() / "fyp"
MODEL_PATH   = FYP_DIR / "yolov8n_ncnn_model"
SPEECH_WAV   = Path("/tmp/speech.wav")
# Piper writes SPEECH_WAV; to_a2dp_format() rewrites it here for playback.
PLAYBACK_WAV = Path("/tmp/speech_playback.wav")
CAPTURE_JPG  = Path("/tmp/capture.jpg")

# Voice model: look in voices/ first, fall back to the fyp dir itself
# (piper.download_voices drops files into the current directory).
_VOICE_CANDIDATES = [
    FYP_DIR / "voices" / "en_US-lessac-medium.onnx",
    FYP_DIR / "en_US-lessac-medium.onnx",
]
VOICE_PATH = next((p for p in _VOICE_CANDIDATES if p.exists()),
                  _VOICE_CANDIDATES[0])

# Piper is installed into the venv's bin/, which is NOT on systemd's default
# PATH - so calling it by bare name works in an activated shell and fails
# silently under the service. Resolve it next to the running interpreter.
_PIPER_IN_VENV = Path(sys.executable).parent / "piper"
PIPER_BIN = str(_PIPER_IN_VENV) if _PIPER_IN_VENV.exists() else "piper"

DETECT_CONF        = 0.40    # min confidence for detections to be spoken
# Save input/annotated images for the write-up. Off costs nothing at
# runtime; on, every trigger leaves a timestamped set in ~/fyp/reports.
SAVE_REPORTS       = os.environ.get("FYP_REPORT", "1") != "0"
MAX_SPOKEN_CLASSES = 4       # don't overwhelm the listener

# The camera sits upside-down in the wearable's housing, so every frame is
# turned the right way up at capture. Without this the detector sees inverted
# people and furniture, and "to your left" comes out as "to your right".
_VALID_ROTATIONS = (0, 90, 180, 270)
_ROTATION_SETTING = os.environ.get("FYP_CAMERA_ROTATION", "180")
if _ROTATION_SETTING in {str(r) for r in _VALID_ROTATIONS}:
    CAMERA_ROTATION = int(_ROTATION_SETTING)
else:
    print(f"[CAMERA] ignoring FYP_CAMERA_ROTATION={_ROTATION_SETTING!r}; "
          f"must be one of {_VALID_ROTATIONS}. Using 180.")
    CAMERA_ROTATION = 180

USE_GPIO_BUTTON = os.environ.get("FYP_USE_GPIO", "0") == "1"
BUTTON_GPIO     = 17
# Level of the pin while the button is pressed. "auto" reads the idle level
# at start-up, which covers both a bare switch to ground and the 3-pin button
# modules (some pull their output high when pressed, some low).
BUTTON_ACTIVE   = os.environ.get("FYP_BUTTON_ACTIVE", "auto").lower()
if BUTTON_ACTIVE not in ("auto", "low", "high"):
    print(f"[GPIO] ignoring FYP_BUTTON_ACTIVE={BUTTON_ACTIVE!r}; using auto")
    BUTTON_ACTIVE = "auto"
# What a long press does: "voice" opens the microphone, "read" reads text
# (for a headset with no usable microphone).
LONG_PRESS_ACTION = os.environ.get("FYP_LONG_PRESS", "voice").lower()

# Speech goes to whichever Bluetooth headset is connected at the moment it is
# spoken (see fypvision/audio.py) - the Pi usually boots before the headset is
# switched on, so nothing is decided at start-up.
# FYP_AUDIO_DEVICE overrides that; "default" forces the onboard 3.5mm/HDMI
# output for bench tests. FYP_HEADSET_MAC is an optional address to try when
# no connected headset is detected.
# Speech is converted to 44.1 kHz stereo before playback - Piper's native
# 22.05 kHz mono triggers a BlueALSA mono-SBC bug; see to_a2dp_format().
_AUDIO_OVERRIDE = os.environ.get("FYP_AUDIO_DEVICE")
HEADSET_MAC = os.environ.get("FYP_HEADSET_MAC", "")

HEADSET_POLL_S   = 2.0   # how often to check whether the headset is there
# A2DP is not playable the instant the headset reports "connected".
HEADSET_SETTLE_S = 3.0

# Target playback format. 44.1 kHz stereo is the universally safe A2DP/SBC
# configuration; see to_a2dp_format() for why mono is not an option.
A2DP_RATE     = 44100
A2DP_CHANNELS = 2

# ----------------------------------------------------------------------------
# Speech
# ----------------------------------------------------------------------------
def synthesize(text: str) -> bool:
    """Text -> WAV via Piper. Returns True on success."""
    try:
        result = subprocess.run(
            # Long-form flags: matches the CLI validated on piper-tts 1.4.2
            [PIPER_BIN, "--model", str(VOICE_PATH),
             "--output_file", str(SPEECH_WAV)],
            input=text.encode(),
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"[TTS] piper exited {result.returncode}: "
                  f"{result.stderr.decode(errors='replace')[-300:]}")
        return result.returncode == 0 and SPEECH_WAV.exists()
    except Exception as e:
        print(f"[TTS] synthesis failed: {e}")
        return False


def to_a2dp_format(src: Path, dst: Path) -> Path:
    """Rewrite a WAV as 44.1 kHz stereo. Returns the path to play.

    Mono SBC is broken in bluez-alsa 4.3.1: a 44.1 kHz mono stream reaches the
    headset garbled, a 48 kHz mono stream initialises the encoder as
    "0 Hz ... => 0 bps", and playback can hang outright. A stereo stream of the
    same bitrate plays cleanly, so we never hand BlueALSA a mono PCM.

    Falls back to the original file if anything here fails - degraded audio
    beats no audio on an assistive device.
    """
    try:
        with wave.open(str(src), "rb") as w:
            if w.getsampwidth() != 2:
                return src                      # not S16; leave it alone
            rate, channels = w.getframerate(), w.getnchannels()
            if rate == A2DP_RATE and channels == A2DP_CHANNELS:
                return src
            raw = w.readframes(w.getnframes())

        samples = np.frombuffer(raw, dtype="<i2").reshape(-1, channels)
        if samples.size == 0:
            return src

        if rate != A2DP_RATE:
            n_out = int(round(samples.shape[0] * A2DP_RATE / rate))
            src_ix = np.arange(samples.shape[0])
            dst_ix = np.linspace(0, samples.shape[0] - 1, n_out)
            samples = np.stack(
                [np.interp(dst_ix, src_ix, samples[:, c])
                 for c in range(channels)], axis=1).astype(np.int16)

        if samples.shape[1] == 1:
            samples = np.repeat(samples, A2DP_CHANNELS, axis=1)
        elif samples.shape[1] > A2DP_CHANNELS:
            samples = samples[:, :A2DP_CHANNELS]

        with wave.open(str(dst), "wb") as out:
            out.setnchannels(A2DP_CHANNELS)
            out.setsampwidth(2)
            out.setframerate(A2DP_RATE)
            out.writeframes(samples.tobytes())
        return dst
    except Exception as e:
        print(f"[AUDIO] format conversion failed ({e}); playing original")
        return src


def play_wav(path: Path) -> bool:
    """Play a WAV on the current output device. Returns True on success."""
    device = audio.output_device(_AUDIO_OVERRIDE, HEADSET_MAC)
    if device is None:
        print("[AUDIO] no headset connected; not spoken")
        return False
    # No -q: aplay's rate/format warnings are the only signal that a stream is
    # being played at the wrong rate, and suppressing them hid a real bug.
    cmd = ["aplay"]
    if device:
        cmd += ["-D", device]
    cmd.append(str(path))
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        # The BlueALSA ALSA plugin is built with debug logging; drop those
        # lines so a real warning is not buried in per-open chatter.
        notes = [ln for ln in result.stderr.decode(errors="replace").splitlines()
                 if ln.strip() and "] D: " not in ln]
        if result.returncode != 0:
            print(f"[AUDIO] aplay exited {result.returncode}: "
                  f"{' | '.join(notes)[-300:]}")
        else:
            for ln in notes:
                print(f"[AUDIO] {ln}")
        return result.returncode == 0
    except Exception as e:
        print(f"[AUDIO] playback failed: {e}")
        return False


def speak(text: str) -> None:
    """Say text out loud (and always print it, for headless debugging)."""
    print(f"[SPEAK] {text}")
    if synthesize(text):
        play_wav(to_a2dp_format(SPEECH_WAV, PLAYBACK_WAV))


# ----------------------------------------------------------------------------
# Vision - detection
# ----------------------------------------------------------------------------
def extract_detections(results, frame_shape) -> list:
    """YOLO results -> plain dicts, one per confident detection.

    Everything downstream (phrasing, finding, distance, report drawing)
    works off this list rather than the ultralytics objects, so those parts
    stay testable without a model loaded.
    """
    height, width = frame_shape[:2]
    names = results[0].names
    detections = []
    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < DETECT_CONF:
            continue
        label = names[int(box.cls[0])]
        coords = tuple(float(v) for v in box.xyxy[0])
        estimate = dist.estimate_distance(label, coords, width, height)
        detections.append({
            "label": label,
            "conf": conf,
            "box": coords,
            "distance_m": estimate.metres if estimate else None,
            # True when the object runs out of frame: it is no farther
            # than distance_m, and probably nearer.
            "distance_is_max": bool(estimate and estimate.is_upper_bound),
        })
    return detections


def build_detection_phrase(detections, frame_width: int) -> str:
    """Turn detections into a short spoken phrase, with a distance cue.

    The listing answers "what is around me"; the trailing sentence gives the
    nearest object a bearing and range, which is the part a wearer can act
    on. Objects with no distance estimate are still listed, just not ranged.
    """
    if not detections:
        return "Nothing recognized ahead."

    counts = Counter(d["label"] for d in detections)
    parts = [f"{n} {cls if n == 1 else finder.plural(cls)}"
             for cls, n in counts.most_common(MAX_SPOKEN_CLASSES)]

    if len(parts) == 1:
        listing = parts[0]
    else:
        listing = ", ".join(parts[:-1]) + " and " + parts[-1]
    phrase = listing.capitalize() + " ahead."

    ranged = [d for d in detections if d.get("distance_m") is not None]
    if ranged:
        nearest = min(ranged, key=lambda d: d["distance_m"])
        x1, _, x2, _ = nearest["box"]
        phrase += (f" Nearest {nearest['label']} "
                   f"{dist.describe_direction((x1 + x2) / 2.0, frame_width)}, "
                   + dist.describe_distance(nearest["distance_m"],
                                            nearest["distance_is_max"])
                   + ".")
    return phrase


def run_detection(frame_bgr, model):
    """Returns (spoken phrase, detections)."""
    t0 = time.time()
    results = model.predict(frame_bgr, imgsz=640, verbose=False)
    print(f"[DETECT] inference {(time.time() - t0) * 1000:.0f} ms")
    detections = extract_detections(results, frame_bgr.shape)
    return build_detection_phrase(detections, frame_bgr.shape[1]), detections


# ----------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------
def init_camera() -> Picamera2:
    picam2 = Picamera2()
    config = picam2.create_still_configuration(
        main={"size": (2304, 1296)},   # good for both OCR and detection
        buffer_count=2,
    )
    picam2.configure(config)
    picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
    picam2.start()
    time.sleep(1.0)  # let AE/AF settle after start
    return picam2


def capture_frame(picam2) -> "np.ndarray":
    """Grab one frame as BGR (OpenCV order), turned the right way up."""
    rgb = picam2.capture_array("main")
    bgr = ocr.rotate_image(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           CAMERA_ROTATION)
    cv2.imwrite(str(CAPTURE_JPG), bgr)   # keep last capture for debugging
    return bgr


# ----------------------------------------------------------------------------
# Triggers
# ----------------------------------------------------------------------------
_busy = threading.Lock()
_shutting_down = threading.Event()


def guarded(fn, *args) -> None:
    """
    Run a handler unless one is already running. gpiozero fires callbacks on
    a background thread, so without this a second button press mid-OCR would
    re-enter the camera/TTS pipeline. Extra presses are simply ignored.
    Any handler crash is reported by voice so a headless user isn't left
    wondering, then logged to the console.
    """
    if _shutting_down.is_set() or not _busy.acquire(blocking=False):
        print("[BUSY] ignoring trigger while previous task is running")
        return
    try:
        fn(*args)
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        try:
            speak("Sorry, something went wrong.")
        except Exception:
            pass  # audio itself is broken; nothing more we can do here
    finally:
        _busy.release()


# A real word, for the purposes of "did OCR read anything meaningful".
_WORD_RE = re.compile(r"[A-Za-z]{3,}")
MIN_FALLBACK_WORDS = 2


def looks_like_words(text: str) -> bool:
    """True if `text` reads as language rather than OCR noise.

    Pointed at a blank wall, Tesseract happily returns things like
    ", ` - as 4 et * a P @ waa ig a". Speaking that aloud is worse than
    saying nothing, so the detection fallback requires a couple of genuine
    words before it will read anything out. The explicit "read text" mode
    does not use this gate: there the wearer asked for whatever is there.
    """
    return len(_WORD_RE.findall(text or "")) >= MIN_FALLBACK_WORDS


def save_report(frame, detections, phrase, mode):
    """Persist the input/annotated pair, if reporting is enabled."""
    if not SAVE_REPORTS:
        return
    paths = reporting.save_report(frame, detections, phrase, mode)
    if paths:
        print(f"[REPORT] {paths['sidebyside']}")


def handle_short_press(picam2, model):
    """Describe the scene; if nothing is recognised, try reading text.

    A blank detection result is often not an empty scene but a sign, menu or
    label the wearer is pointed at, which the object model has no class for.
    Falling through to OCR turns a useless "nothing recognized" into the
    answer they were actually after.
    """
    speak("Detecting.")
    frame = capture_frame(picam2)
    phrase, detections = run_detection(frame, model)

    if not detections:
        print("[DETECT] nothing recognised; falling back to OCR")
        text = ocr.run_ocr(frame)
        if text != ocr.NO_TEXT_PHRASE and looks_like_words(text):
            phrase = f"No objects recognized, but I can read: {text}"
            save_report(frame, detections, phrase, "detect-ocr-fallback")
            speak(phrase)
            return

    save_report(frame, detections, phrase, "detect")
    speak(phrase)


def handle_find_object(picam2, model, target_phrase: str):
    """Look for one named object and report bearing and distance."""
    target = finder.normalise_target(target_phrase)
    if not target:
        # Not echoed back: over voice, target_phrase is the whole sentence.
        print(f"[FIND] no detectable object in {target_phrase!r}")
        speak("I cannot look for that. Try a bottle, a chair or a person.")
        return

    speak(f"Looking for {target}.")
    frame = capture_frame(picam2)
    _, detections = run_detection(frame, model)
    phrase = finder.find_in_detections(detections, target, frame.shape[1],
                                       dist.describe_direction,
                                       dist.describe_distance)
    save_report(frame, [d for d in detections if d["label"] == target],
                phrase, "find")
    speak(phrase)


def handle_long_press(picam2):
    speak("Reading text.")
    # Continuous AF is usually still hunting at trigger time, so force one
    # focus cycle and wait for it to lock before grabbing the frame. Best
    # effort only - never let a focus hiccup abort the read.
    try:
        picam2.autofocus_cycle()
    except Exception as e:
        print(f"[OCR] autofocus_cycle skipped: {e}")
    frame = capture_frame(picam2)
    text = ocr.run_ocr(frame)
    save_report(frame, [], text, "ocr")
    speak(text)


def handle_emotion(picam2):
    """Report the expression of whoever is in view."""
    speak("Looking at faces.")
    frame = capture_frame(picam2)
    faces = emotion.detect_faces(frame)
    phrase = emotion.describe(faces)
    save_report(frame, faces, phrase, "emotion")
    speak(phrase)


def handle_currency(picam2):
    """Read a banknote's denomination.

    Notes are printed matter, so this reuses the OCR pipeline rather than a
    dedicated detector. It reports the value only - it cannot judge whether
    a note is genuine.
    """
    speak("Reading the note.")
    try:
        picam2.autofocus_cycle()
    except Exception as e:
        print(f"[CURRENCY] autofocus_cycle skipped: {e}")
    frame = capture_frame(picam2)
    text = ocr.run_ocr_sparse(frame)
    phrase = currency.identify("" if text == ocr.NO_TEXT_PHRASE else text)
    save_report(frame, [], phrase, "currency")
    speak(phrase)


def dispatch_intent(intent: str, said: str, picam2, model):
    """Run whatever the wearer asked for."""
    if intent == "detect":
        handle_short_press(picam2, model)
    elif intent == "read":
        handle_long_press(picam2)
    elif intent == "find":
        handle_find_object(picam2, model, said)
    elif intent == "emotion":
        handle_emotion(picam2)
    elif intent == "currency":
        handle_currency(picam2)
    else:
        speak("I did not catch that. Try asking what is in front of me.")


def handle_voice(picam2, model):
    """Listen through the headset, then act on what was asked.

    Recording holds an HFP/SCO link to the headset, which cannot coexist
    with A2DP playback, so speak() is never called while the mic is open -
    voice.record() closes the link and settles before returning. There is
    no spoken "listening" prompt for the same reason: the cue to talk is a
    beep that voice.record() plays once the mic is really live.
    """
    if not voice.available():
        speak("Voice input is not set up.")
        return

    mac = audio.detect_headset_mac() or HEADSET_MAC
    if not mac:
        print("[VOICE] no headset connected; cannot listen")
        return
    said = voice.listen(mac)
    if _shutting_down.is_set():
        return      # the hold that opened the mic turned into a shutdown
    if not said:
        speak("I did not hear anything.")
        return

    intent, argument = voice.parse_intent(said)
    print(f"[VOICE] intent={intent} from {said!r}")
    dispatch_intent(intent, argument, picam2, model)


def handle_shutdown():
    """Power the Pi off cleanly; pulling the plug risks the SD card.

    Reached by holding the button well past a long press, so the voice
    handler that the same hold started is usually still listening: wait for
    it rather than talk over the open microphone.
    """
    _shutting_down.set()
    with _busy:
        speak("Shutting down.")
        try:
            # Allowed without a password by the sudoers rule install.sh adds.
            result = subprocess.run(["sudo", "-n", "systemctl", "poweroff"],
                                    capture_output=True, timeout=30, text=True)
            failure = result.stderr.strip() if result.returncode else ""
        except (OSError, subprocess.SubprocessError) as e:
            failure = str(e)
        if failure:
            print(f"[SHUTDOWN] poweroff failed: {failure}")
            _shutting_down.clear()
            speak("Sorry, I could not shut down.")


def announce_headset():
    """Say "Ready." each time an output device appears.

    The wearer has no screen: this is how they learn the device has booted
    and can be heard, and that a headset which dropped out is back.
    """
    was_available = False
    while True:
        available = audio.output_device(_AUDIO_OVERRIDE,
                                        HEADSET_MAC) is not None
        if available and not was_available:
            if _AUDIO_OVERRIDE is None:
                time.sleep(HEADSET_SETTLE_S)
            with _busy:
                speak("Ready.")
        was_available = available
        time.sleep(HEADSET_POLL_S)


def keyboard_loop(picam2, model):
    print()
    print("=== TEST MODE ===")
    print("  ENTER      detection (falls back to OCR if nothing recognised)")
    print("  l          read text (OCR)")
    print("  e          emotion")
    print("  c          currency note")
    print("  v          voice command, asked through the headset")
    print("  f <thing>  find an object, e.g. 'f bottle'")
    print("  q          quit")
    print()
    while True:
        try:
            cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        lowered = cmd.lower()
        if lowered == "q":
            break
        elif lowered == "l":
            guarded(handle_long_press, picam2)
        elif lowered == "e":
            guarded(handle_emotion, picam2)
        elif lowered == "c":
            guarded(handle_currency, picam2)
        elif lowered == "v":
            guarded(handle_voice, picam2, model)
        elif lowered.startswith("f ") and len(cmd) > 2:
            guarded(handle_find_object, picam2, model, cmd[2:])
        else:
            guarded(handle_short_press, picam2, model)


def gpio_loop(picam2, model):
    from signal import pause

    if LONG_PRESS_ACTION == "read":
        def on_long():
            guarded(handle_long_press, picam2)
    else:
        def on_long():
            guarded(handle_voice, picam2, model)

    classifier = button.PressClassifier(
        on_single=lambda: guarded(handle_short_press, picam2, model),
        on_double=lambda: guarded(handle_long_press, picam2),
        on_long=on_long,
        on_shutdown=handle_shutdown)
    # Keep the reference: gpiozero stops watching a Button once it is
    # garbage-collected.
    watched = button.watch(BUTTON_GPIO, classifier, BUTTON_ACTIVE)
    print(f"[GPIO] press = detect, double press = read text, "
          f"hold {button.HOLD_TIME_S}s = {LONG_PRESS_ACTION}, "
          f"hold {button.SHUTDOWN_HOLD_S}s = power off")
    try:
        pause()
    finally:
        watched.close()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    # --- preflight ---
    problems = []
    if not MODEL_PATH.exists():
        problems.append(f"model missing: {MODEL_PATH}")
    if not VOICE_PATH.exists():
        problems.append(f"voice missing: {VOICE_PATH} "
                        f"(run: python -m piper.download_voices "
                        f"en_US-lessac-medium)")
    if problems:
        for p in problems:
            print(f"[PREFLIGHT] {p}")
        sys.exit(1)

    print("[INIT] loading YOLO NCNN model...")
    model = YOLO(str(MODEL_PATH), task="detect")

    print("[INIT] starting camera...")
    picam2 = init_camera()

    # warm up inference once so first real trigger is fast
    print("[INIT] warmup inference...")
    model.predict(capture_frame(picam2), imgsz=640, verbose=False)

    threading.Thread(target=announce_headset, daemon=True,
                     name="announce-headset").start()

    try:
        if USE_GPIO_BUTTON:
            gpio_loop(picam2, model)
        else:
            keyboard_loop(picam2, model)
    finally:
        picam2.stop()
        print("[EXIT] camera stopped.")


if __name__ == "__main__":
    main()