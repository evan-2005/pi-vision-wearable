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
# Orientations to try when reading text. We OCR each and keep the one that
# reads most confidently, instead of trusting Tesseract's OSD (which kept
# confidently flipping upright signs 180 degrees). Add 90/270 if the camera
# can ever be mounted sideways; for a fixed chest mount 0 vs 180 is enough.
OCR_ORIENTATIONS   = (0, 180)

USE_GPIO_BUTTON = os.environ.get("FYP_USE_GPIO", "0") == "1"
BUTTON_GPIO     = 17
HOLD_TIME_S     = 1.0        # press longer than this = long press (OCR)

# Bluetooth headset fallback MAC (Nothing Ear (a)); used only when auto-detect
# can't find a connected headset. Override with FYP_HEADSET_MAC.
HEADSET_MAC = os.environ.get("FYP_HEADSET_MAC", "3C:B0:ED:B6:18:A9")

# Audio output device for aplay, resolved at startup by resolve_audio_device().
# Speech plays through the bluealsa PCM directly (no plug: wrapper - see
# bluealsa_device()). Piper's 22.05 kHz mono output plays as-is over A2DP.
# Priority: FYP_AUDIO_DEVICE override > whichever headset is connected
# (auto-detected) > HEADSET_MAC fallback. This lets the user swap between
# paired earphones - whichever is powered on and connected gets used.
# Set FYP_AUDIO_DEVICE=default to force the onboard 3.5mm/HDMI output.
_AUDIO_OVERRIDE = os.environ.get("FYP_AUDIO_DEVICE")
AUDIO_DEVICE = None   # resolved in main() before any speech

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

    No `plug:` wrapper: the bluealsa PCM plays Piper's 22.05 kHz mono output
    directly (verified with `aplay -D bluealsa:DEV=...`). Prefixing `plug:`
    breaks parsing on this bluez-alsa build - ALSA reads the comma-separated
    args as the plug plugin's own parameters and fails with "Unknown PCM".
    """
    return f"bluealsa:DEV={mac},PROFILE=a2dp"


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