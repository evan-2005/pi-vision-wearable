#!/usr/bin/env python3
"""
FYP Assistive Vision Wearable - Main Pipeline
==============================================
Short trigger  -> scene detection (YOLOv8n/NCNN) -> spoken summary
                  (falls back to OCR when nothing is recognised)
Long trigger   -> OCR (Tesseract) -> spoken text

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
    FYP_AUDIO_DEVICE=...  override the aplay output device
    FYP_HEADSET_MAC=...   override the Bluetooth headset MAC
    FYP_REPORT=0          stop writing report images
    FYP_REPORT_DIR=...    where report images go (default ~/fyp/reports)
    FYP_CAMERA_FOCAL_PX=  measured focal length, for accurate distances
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
import pytesseract
from picamera2 import Picamera2
from libcamera import controls
from ultralytics import YOLO

from fypvision import distance as dist
from fypvision import currency, emotion, finder, reporting, voice

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

# OCR tunables
OCR_PSM            = 6        # tesseract page segmentation mode (assume a block)
OCR_MIN_CONF       = 40       # drop words tesseract is less than this % sure of
# Orientations to try when reading text. We OCR each and keep the one that
# reads most confidently, instead of trusting Tesseract's OSD (which kept
# confidently flipping upright signs 180 degrees). Add 90/270 if the camera
# can ever be mounted sideways; for a fixed chest mount 0 vs 180 is enough.
OCR_ORIENTATIONS   = (0, 180)
# Spoken when OCR finds nothing. Shared so the detection fallback can
# tell "read something" from "read nothing" without matching a literal.
NO_TEXT_PHRASE     = "No readable text found."

USE_GPIO_BUTTON = os.environ.get("FYP_USE_GPIO", "0") == "1"
BUTTON_GPIO     = 17
HOLD_TIME_S     = 1.0        # press longer than this = long press (OCR)

# Bluetooth headset fallback MAC (Nothing Ear (a)); used only when auto-detect
# can't find a connected headset. Override with FYP_HEADSET_MAC.
HEADSET_MAC = os.environ.get("FYP_HEADSET_MAC", "3C:B0:ED:B6:18:A9")

# Audio output device for aplay, resolved at startup by resolve_audio_device().
# Speech is converted to 44.1 kHz stereo before playback - Piper's native
# 22.05 kHz mono triggers a BlueALSA mono-SBC bug; see to_a2dp_format().
# Priority: FYP_AUDIO_DEVICE override > whichever headset is connected
# (auto-detected) > HEADSET_MAC fallback. This lets the user swap between
# paired earphones - whichever is powered on and connected gets used.
# Set FYP_AUDIO_DEVICE=default to force the onboard 3.5mm/HDMI output.
_AUDIO_OVERRIDE = os.environ.get("FYP_AUDIO_DEVICE")
AUDIO_DEVICE = None   # resolved in main() before any speech

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
    """Play a WAV on AUDIO_DEVICE. Returns True on success."""
    # No -q: aplay's rate/format warnings are the only signal that a stream is
    # being played at the wrong rate, and suppressing them hid a real bug.
    cmd = ["aplay"]
    if AUDIO_DEVICE:
        cmd += ["-D", AUDIO_DEVICE]
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
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        detections.append({
            "label": label,
            "conf": conf,
            "box": (x1, y1, x2, y2),
            "distance_m": dist.estimate_distance_m(label, y2 - y1,
                                                   width, height),
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
    parts = []
    for cls, n in counts.most_common(MAX_SPOKEN_CLASSES):
        label = cls if n == 1 else cls + "s"   # naive plural, fine for COCO
        parts.append(f"{n} {label}")

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
                   f"{dist.describe_distance(nearest['distance_m'])}.")
    return phrase


def run_detection(frame_bgr, model):
    """Returns (spoken phrase, detections)."""
    t0 = time.time()
    results = model.predict(frame_bgr, imgsz=640, verbose=False)
    print(f"[DETECT] inference {(time.time() - t0) * 1000:.0f} ms")
    detections = extract_detections(results, frame_bgr.shape)
    return build_detection_phrase(detections, frame_bgr.shape[1]), detections


# ----------------------------------------------------------------------------
# Vision - OCR
# ----------------------------------------------------------------------------
def preprocess_for_ocr(img_bgr):
    """Grayscale -> upscale -> local contrast. Returns image for tesseract."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess_otsu(img_bgr):
    """Fallback strategy: grayscale -> upscale -> Otsu threshold.

    CLAHE wins on most real-world captures, but on clean high-contrast pages
    a global Otsu binarization can recover text CLAHE's local contrast noise
    smears out. Used only as a second pass when CLAHE comes back near-empty.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


_ROTATIONS = {
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def rotate_image(img_bgr, angle):
    """Rotate by a multiple of 90 degrees (0 returns the image unchanged)."""
    if angle == 0:
        return img_bgr
    return cv2.rotate(img_bgr, _ROTATIONS[angle])


def clean_ocr_text(text: str) -> str:
    """Strip OCR junk so TTS doesn't read stray symbols."""
    text = re.sub(r"[|{}\[\]<>_~^`]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _alnum_count(text: str) -> int:
    """How many letters/digits survived - our 'is this real text?' signal."""
    return sum(ch.isalnum() for ch in text)


def ocr_scored(processed):
    """OCR one preprocessed image; return (confidence_score, text).

    Uses image_to_data so we can drop low-confidence fragments (conf <
    OCR_MIN_CONF) before they reach TTS. Surviving words are rejoined in
    reading order (the order image_to_data returns them in). The score is the
    summed confidence of the surviving words - upright text produces many
    confident words and scores high, so it lets us pick the right orientation.
    """
    data = pytesseract.image_to_data(
        processed,
        config=f"--psm {OCR_PSM}",
        output_type=pytesseract.Output.DICT,
    )
    words, confs = [], []
    for word, conf in zip(data["text"], data["conf"]):
        try:
            c = float(conf)
        except (TypeError, ValueError):
            continue          # tesseract uses "-1"/"" for non-text boxes
        if c >= OCR_MIN_CONF and word.strip():
            words.append(word)
            confs.append(c)
    return sum(confs), clean_ocr_text(" ".join(words))


def _best_over_orientations(frame_bgr, preprocess, label, start_score, start):
    """Run `preprocess` at each candidate orientation; keep the top scorer."""
    best_score, best_text, best_pass = start_score, start[0], start[1]
    for angle in OCR_ORIENTATIONS:
        score, text = ocr_scored(preprocess(rotate_image(frame_bgr, angle)))
        print(f"[OCR] {angle:>3} deg / {label}: score={score:.0f} "
              f"text={text[:40]!r}")
        if score > best_score:
            best_score, best_text, best_pass = score, text, f"{angle}/{label}"
    return best_score, best_text, best_pass


def run_ocr(frame_bgr) -> str:
    # Primary strategy: CLAHE at each orientation, keep the most confident read.
    # This replaces Tesseract OSD, which kept flipping upright signs 180 deg.
    score, text, winning_pass = _best_over_orientations(
        frame_bgr, preprocess_for_ocr, "CLAHE", -1.0, ("", "none"))

    # Fallback: if CLAHE read almost nothing, retry each orientation with a
    # plain Otsu threshold (recovers clean high-contrast pages CLAHE smears).
    if _alnum_count(text) < 3:
        print("[OCR] CLAHE passes near-empty; retrying with Otsu threshold")
        score, text, winning_pass = _best_over_orientations(
            frame_bgr, preprocess_otsu, "Otsu", score, (text, winning_pass))

    if not text:
        return NO_TEXT_PHRASE
    print(f"[OCR] winning pass: {winning_pass}")
    print(f"[OCR] raw text: {text!r}")
    return text


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
    """Grab one frame as BGR (OpenCV order)."""
    rgb = picam2.capture_array("main")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(CAPTURE_JPG), bgr)   # keep last capture for debugging
    return bgr


# ----------------------------------------------------------------------------
# Triggers
# ----------------------------------------------------------------------------
_busy = threading.Lock()


def guarded(fn, *args) -> None:
    """
    Run a handler unless one is already running. gpiozero fires callbacks on
    a background thread, so without this a second button press mid-OCR would
    re-enter the camera/TTS pipeline. Extra presses are simply ignored.
    Any handler crash is reported by voice so a headless user isn't left
    wondering, then logged to the console.
    """
    if not _busy.acquire(blocking=False):
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
        text = run_ocr(frame)
        if text and text != NO_TEXT_PHRASE and looks_like_words(text):
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
        speak(f"I cannot look for {target_phrase}.")
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
    text = run_ocr(frame)
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
    text = run_ocr(frame)
    phrase = currency.identify("" if text == NO_TEXT_PHRASE else text)
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
    voice.record() closes the link and settles before returning.
    """
    if not voice.available():
        speak("Voice input is not set up.")
        return

    mac = detect_headset_mac() or HEADSET_MAC
    speak("Listening.")
    said = voice.listen(mac)
    if not said:
        speak("I did not hear anything.")
        return

    intent, argument = voice.parse_intent(said)
    print(f"[VOICE] intent={intent} from {said!r}")
    dispatch_intent(intent, argument, picam2, model)


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
    from gpiozero import Button
    from signal import pause

    button = Button(BUTTON_GPIO, pull_up=True, hold_time=HOLD_TIME_S)
    state = {"held": False}

    def on_held():
        state["held"] = True
        guarded(handle_long_press, picam2)

    def on_released():
        if state["held"]:
            state["held"] = False    # long press already handled
        else:
            guarded(handle_short_press, picam2, model)

    button.when_held = on_held
    button.when_released = on_released
    print(f"[GPIO] listening on GPIO{BUTTON_GPIO} "
          f"(short press = detect, hold {HOLD_TIME_S}s = OCR)")
    pause()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def bluetooth_connected(mac: str) -> bool:
    """True if bluetoothctl reports the headset as connected."""
    try:
        result = subprocess.run(
            ["bluetoothctl", "info", mac],
            capture_output=True, timeout=10, text=True,
        )
        return "Connected: yes" in result.stdout
    except Exception:
        return False


def detect_headset_mac() -> "str | None":
    """MAC of a currently-connected Bluetooth device, or None.

    Lets the user swap between paired headsets without editing config:
    whichever one is powered on and connected is the one we use. On a
    wearable the only paired device is the headset, so "connected" == it.
    """
    try:
        result = subprocess.run(
            ["bluetoothctl", "devices", "Connected"],
            capture_output=True, timeout=10, text=True,
        )
        # Lines look like: "Device 3C:B0:ED:B6:18:A9 Nothing Ear (a)"
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "Device":
                return parts[1]
    except Exception:
        pass  # old bluez without the "Connected" filter, or bluetoothctl absent
    return None


def bluealsa_device(mac: str) -> str:
    """ALSA device string for playing to a headset over BlueALSA A2DP.

    The inner PCM name is single-quoted because ALSA splits a device string
    on commas: bare `plug:bluealsa:DEV=...,PROFILE=a2dp` is read as two
    arguments to `plug` and fails with "Unknown parameter". Quoted, it opens
    fine, and `plug` absorbs any residual rate/format conversion.
    """
    return f"plug:'bluealsa:DEV={mac},PROFILE=a2dp'"


def resolve_audio_device() -> "str | None":
    """Pick the ALSA device speech plays through, in priority order.

    1. FYP_AUDIO_DEVICE override ("default"/"none"/"" -> onboard 3.5mm/HDMI).
    2. Whichever Bluetooth headset is currently connected (auto-detected).
    3. HEADSET_MAC fallback, so it still tries if detection is unavailable.
    """
    if _AUDIO_OVERRIDE is not None:
        if _AUDIO_OVERRIDE.lower() in ("default", "none", ""):
            return None
        return _AUDIO_OVERRIDE
    mac = detect_headset_mac()
    if mac:
        print(f"[AUDIO] using connected headset {mac}")
        return bluealsa_device(mac)
    print(f"[AUDIO] no connected headset detected; "
          f"falling back to {HEADSET_MAC}")
    return bluealsa_device(HEADSET_MAC)


def main():
    # Resolve the audio sink before anything tries to speak.
    global AUDIO_DEVICE
    AUDIO_DEVICE = resolve_audio_device()

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

    # Warn (don't abort) if the Bluetooth sink isn't up: audio would go
    # nowhere, but keyboard-mode bench testing may deliberately use 3.5mm.
    if AUDIO_DEVICE and "bluealsa" in AUDIO_DEVICE:
        mac_match = re.search(r"DEV=([0-9A-Fa-f:]+)", AUDIO_DEVICE)
        target_mac = mac_match.group(1) if mac_match else HEADSET_MAC
        if not bluetooth_connected(target_mac):
            print(f"[PREFLIGHT] WARNING: headset {target_mac} not connected;"
                  " speech will fail until it reconnects.")

    print("[INIT] loading YOLO NCNN model...")
    model = YOLO(str(MODEL_PATH), task="detect")

    print("[INIT] starting camera...")
    picam2 = init_camera()

    # warm up inference once so first real trigger is fast
    print("[INIT] warmup inference...")
    model.predict(capture_frame(picam2), imgsz=640, verbose=False)

    speak("Ready.")

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