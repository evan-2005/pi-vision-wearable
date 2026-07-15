#!/usr/bin/env python3
"""
FYP Assistive Vision Wearable - Main Pipeline
==============================================
Short trigger  -> scene detection (YOLOv8n/NCNN) -> spoken summary
Long trigger   -> OCR (Tesseract) -> spoken text

Test mode (no GPIO button needed):
    press ENTER            = short press (detection)
    type 'l' then ENTER    = long press (OCR)
    type 'q' then ENTER    = quit

Run:
    cd ~/fyp && source venv/bin/activate && python3 main.py

Configuration is via the constants below, each overridable by an
environment variable (useful for the systemd service later):

    FYP_USE_GPIO=1        use the GPIO button instead of the keyboard
    FYP_AUDIO_DEVICE=...  override the aplay output device
    FYP_HEADSET_MAC=...   override the Bluetooth headset MAC
"""

import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from picamera2 import Picamera2
from libcamera import controls
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
FYP_DIR      = Path.home() / "fyp"
MODEL_PATH   = FYP_DIR / "yolov8n_ncnn_model"
SPEECH_WAV   = Path("/tmp/speech.wav")
CAPTURE_JPG  = Path("/tmp/capture.jpg")

# Voice model: look in voices/ first, fall back to the fyp dir itself
# (piper.download_voices drops files into the current directory).
_VOICE_CANDIDATES = [
    FYP_DIR / "voices" / "en_US-lessac-medium.onnx",
    FYP_DIR / "en_US-lessac-medium.onnx",
]
VOICE_PATH = next((p for p in _VOICE_CANDIDATES if p.exists()),
                  _VOICE_CANDIDATES[0])

DETECT_CONF        = 0.40    # min confidence for detections to be spoken
MAX_SPOKEN_CLASSES = 4       # don't overwhelm the listener

# OCR tunables
OCR_PSM            = 6        # tesseract page segmentation mode (assume a block)
OCR_MIN_CONF       = 40       # drop words tesseract is less than this % sure of
OSD_MIN_CONFIDENCE = 2.0      # ignore orientation guesses weaker than this

USE_GPIO_BUTTON = os.environ.get("FYP_USE_GPIO", "0") == "1"
BUTTON_GPIO     = 17
HOLD_TIME_S     = 1.0        # press longer than this = long press (OCR)

# Bluetooth headset (Nothing Ear (a)); validated 2026-07-15.
HEADSET_MAC = os.environ.get("FYP_HEADSET_MAC", "3C:B0:ED:B6:18:A9")

# Audio output device for aplay.
# The plug: wrapper resamples Piper's 22.05 kHz output to the 48 kHz
# A2DP link. If your buds played fine WITHOUT plug:, drop the wrapper.
# Set FYP_AUDIO_DEVICE=default to use the 3.5mm/HDMI output for bench tests.
AUDIO_DEVICE = os.environ.get(
    "FYP_AUDIO_DEVICE",
    f"plug:bluealsa:DEV={HEADSET_MAC},PROFILE=a2dp",
)
if AUDIO_DEVICE.lower() in ("default", "none", ""):
    AUDIO_DEVICE = None

# ----------------------------------------------------------------------------
# Speech
# ----------------------------------------------------------------------------
def synthesize(text: str) -> bool:
    """Text -> WAV via Piper. Returns True on success."""
    try:
        result = subprocess.run(
            # Long-form flags: matches the CLI validated on piper-tts 1.4.2
            ["piper", "--model", str(VOICE_PATH),
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


def play_wav(path: Path) -> bool:
    """Play a WAV on AUDIO_DEVICE. Returns True on success."""
    cmd = ["aplay", "-q"]
    if AUDIO_DEVICE:
        cmd += ["-D", AUDIO_DEVICE]
    cmd.append(str(path))
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            print(f"[AUDIO] aplay exited {result.returncode}: "
                  f"{result.stderr.decode(errors='replace')[-300:]}")
        return result.returncode == 0
    except Exception as e:
        print(f"[AUDIO] playback failed: {e}")
        return False


def speak(text: str) -> None:
    """Say text out loud (and always print it, for headless debugging)."""
    print(f"[SPEAK] {text}")
    if synthesize(text):
        play_wav(SPEECH_WAV)


# ----------------------------------------------------------------------------
# Vision - detection
# ----------------------------------------------------------------------------
def build_detection_phrase(results) -> str:
    """Turn YOLO results into a short spoken phrase."""
    names = results[0].names
    counts = Counter()
    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf >= DETECT_CONF:
            counts[names[int(box.cls[0])]] += 1

    if not counts:
        return "Nothing recognized ahead."

    parts = []
    for cls, n in counts.most_common(MAX_SPOKEN_CLASSES):
        label = cls if n == 1 else cls + "s"   # naive plural, fine for COCO
        parts.append(f"{n} {label}")

    if len(parts) == 1:
        listing = parts[0]
    else:
        listing = ", ".join(parts[:-1]) + " and " + parts[-1]
    return listing.capitalize() + " ahead."


def run_detection(frame_bgr, model) -> str:
    t0 = time.time()
    results = model.predict(frame_bgr, imgsz=640, verbose=False)
    dt = (time.time() - t0) * 1000
    print(f"[DETECT] inference {dt:.0f} ms")
    return build_detection_phrase(results)


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


def auto_rotate(img_bgr):
    """Use tesseract orientation detection; rotate image upright if needed."""
    try:
        osd = pytesseract.image_to_osd(img_bgr)
        angle = int(re.search(r"Rotate: (\d+)", osd).group(1))
        conf_match = re.search(r"Orientation confidence: ([\d.]+)", osd)
        confidence = float(conf_match.group(1)) if conf_match else 0.0
    except Exception:
        return img_bgr  # OSD needs enough text; fall back to as-is
    # OSD misfires on sparse text (e.g. reading "2019" as an upside-down
    # "6LOZ"), so only trust a rotation the detector is actually sure of.
    if confidence < OSD_MIN_CONFIDENCE:
        print(f"[OCR] OSD confidence {confidence:.1f} < {OSD_MIN_CONFIDENCE}; "
              f"skipping rotation")
        return img_bgr
    rotations = {
        90:  cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    if angle in rotations:
        print(f"[OCR] auto-rotating {angle} degrees")
        return cv2.rotate(img_bgr, rotations[angle])
    return img_bgr


def clean_ocr_text(text: str) -> str:
    """Strip OCR junk so TTS doesn't read stray symbols."""
    text = re.sub(r"[|{}\[\]<>_~^`]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _alnum_count(text: str) -> int:
    """How many letters/digits survived - our 'is this real text?' signal."""
    return sum(ch.isalnum() for ch in text)


def ocr_confident_words(processed) -> str:
    """OCR one preprocessed image, keeping only words tesseract is sure of.

    Uses image_to_data so we can drop low-confidence fragments (conf <
    OCR_MIN_CONF) before they reach TTS. Surviving words are rejoined in
    reading order, which is the order image_to_data already returns them in.
    """
    data = pytesseract.image_to_data(
        processed,
        config=f"--psm {OCR_PSM}",
        output_type=pytesseract.Output.DICT,
    )
    words = []
    for word, conf in zip(data["text"], data["conf"]):
        try:
            score = float(conf)
        except (TypeError, ValueError):
            continue          # tesseract uses "-1"/"" for non-text boxes
        if score >= OCR_MIN_CONF and word.strip():
            words.append(word)
    return clean_ocr_text(" ".join(words))


def run_ocr(frame_bgr) -> str:
    frame_bgr = auto_rotate(frame_bgr)

    # Primary pass: CLAHE local contrast (best on most real-world captures).
    text = ocr_confident_words(preprocess_for_ocr(frame_bgr))
    winning_pass = "CLAHE"

    # Fallback: if CLAHE gave us almost nothing, retry once with plain Otsu.
    if _alnum_count(text) < 3:
        print("[OCR] CLAHE pass near-empty; retrying with Otsu threshold")
        otsu_text = ocr_confident_words(preprocess_otsu(frame_bgr))
        if _alnum_count(otsu_text) > _alnum_count(text):
            text, winning_pass = otsu_text, "Otsu"

    if not text:
        return "No readable text found."
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


def handle_short_press(picam2, model):
    speak("Detecting.")
    frame = capture_frame(picam2)
    phrase = run_detection(frame, model)
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
    speak(text)


def keyboard_loop(picam2, model):
    print("\n=== TEST MODE ===")
    print("ENTER = detection | 'l'+ENTER = OCR | 'q'+ENTER = quit\n")
    while True:
        try:
            cmd = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd == "q":
            break
        elif cmd == "l":
            guarded(handle_long_press, picam2)
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

    # Warn (don't abort) if the Bluetooth sink isn't up: audio would go
    # nowhere, but keyboard-mode bench testing may deliberately use 3.5mm.
    if AUDIO_DEVICE and "bluealsa" in AUDIO_DEVICE:
        if not bluetooth_connected(HEADSET_MAC):
            print(f"[PREFLIGHT] WARNING: headset {HEADSET_MAC} not connected;"
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