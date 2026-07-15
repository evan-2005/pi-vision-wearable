# Pi Vision Wearable

An offline, wearable assistive-vision aid for people who are blind or have low
vision, built on a Raspberry Pi 5. One button does two jobs:

- **Short press** — describes what's in front of you ("2 people and a chair ahead").
- **Long press** — reads printed text out loud (signs, labels, a page of a book).

Everything runs on the device. There's no internet dependency, no cloud service,
and no image ever leaves the Pi. That was a hard requirement for me: the thing a
user points at the world all day should not be streaming that world to someone
else's server.

---

## How it works

The whole pipeline is one Python process driven by a single GPIO button (with a
keyboard fallback for bench testing).

```
                 ┌──────────────┐
   button ──────▶│  main.py     │
                 │  (dispatcher)│
                 └──────┬───────┘
        short press     │      long press (hold ≥ 1s)
        ┌───────────────┴────────────────┐
        ▼                                 ▼
  capture frame                     autofocus lock
  YOLOv8n (NCNN)                    capture frame
  build phrase                     preprocess → Tesseract
        │                                 │
        └──────────────┬──────────────────┘
                       ▼
              Piper TTS → BlueALSA → Bluetooth headset
```

**Scene detection** uses YOLOv8n exported to NCNN, which runs at roughly 85 ms
per frame on the Pi 5 CPU. Detections above a confidence threshold are counted
by class and turned into a short spoken phrase, capped at a few classes so it
doesn't ramble.

**Text reading (OCR)** is the harder half. A naive `image_to_string` call gives
garbage on real-world captures, so the OCR path does several things:

- **Autofocus lock before capture.** Continuous autofocus is usually still
  hunting at the moment you press the button, so the read triggers one focus
  cycle and waits for it to settle before grabbing the frame. This alone fixed
  most of the "blurry capture" failures.
- **Orientation gating.** Tesseract's orientation detector (OSD) is happy to
  rotate a perfectly upright image 180° when there's very little text on the
  page (it once read "2019" as an upside-down "6LOZ"). The rotation is now only
  applied when OSD reports enough confidence to be trusted.
- **Two preprocessing strategies.** The primary pass upscales, then applies
  CLAHE local contrast, which wins on most real-world captures. If that comes
  back nearly empty, it retries once with a plain Otsu threshold, which recovers
  clean high-contrast pages that CLAHE's local noise smears out. The winning
  pass is logged.
- **Confidence filtering.** Instead of dumping every character Tesseract emits,
  the reader uses `image_to_data` and drops individual words below a confidence
  threshold, so stray junk fragments never get spoken.

**Speech** is Piper (the `en_US-lessac-medium` voice) rendered to a WAV and
played through `aplay` over a BlueALSA A2DP link to a Bluetooth headset. Every
spoken line is also printed to the console, which makes headless debugging over
SSH painless.

---

## Hardware

| Part | Notes |
|------|-------|
| Raspberry Pi 5 | Runs the whole pipeline on CPU; no accelerator needed |
| Camera Module 3 | Autofocus is used directly by the OCR path |
| Momentary push button | Wired GPIO17 → GND, internal pull-up |
| Bluetooth headset | A2DP; bone-conduction works well so ambient hearing stays open |
| Power bank | Any USB-C PD bank that can feed a Pi 5 |

Capture resolution is 2304×1296, chosen as a compromise that's sharp enough for
OCR while staying fast enough for detection.

## Software

- Raspberry Pi OS Bookworm (64-bit)
- Python 3.13
- `picamera2` (from apt — this is why the venv is created with
  `--system-site-packages`)
- `ultralytics` + `ncnn` for detection
- `pytesseract` + the `tesseract-ocr` apt package for OCR
- `piper-tts` for speech
- `bluez` + `bluez-alsa-utils` for the audio link

---

## Repository layout

```
.
├── main.py              # the wearable: button → detect / OCR → speech
├── preview_capture.py   # dev tool: live MJPEG preview + click-to-OCR in a browser
├── install.sh           # one-command Pi setup (apt, venv, models, systemd unit)
├── .gitignore
└── README.md
```

`preview_capture.py` is a small development aid, not part of the wearable. It
serves a live camera preview on `http://<pi-ip>:8000` with a CAPTURE button that
runs OCR on the shot — handy for aiming the camera and eyeballing what Tesseract
actually sees while tuning.

---

## Installation

On a freshly flashed Pi with network access:

```bash
git clone https://github.com/evan-2005/pi-vision-wearable.git ~/fyp-src
cd ~/fyp-src
bash install.sh
```

`install.sh` is idempotent — re-running it skips whatever already exists. It:

1. Installs the apt packages (Tesseract, BlueZ/BlueALSA, camera libs, ALSA).
2. Creates `~/fyp` and a `--system-site-packages` virtual environment.
3. Installs the Python dependencies.
4. Downloads YOLOv8n and exports it to NCNN (runs on-device, a few minutes).
5. Downloads the Piper voice model.
6. Copies `main.py` into `~/fyp`.
7. Installs a `systemd` service, left **disabled** until the button is wired.

### Pair the Bluetooth headset (one-time)

```bash
bluetoothctl
power on
scan on            # note your headset's MAC, then:
scan off
pair <MAC>
trust <MAC>        # required for auto-reconnect after reboot
connect <MAC>
exit
```

Put the MAC in the service unit (`FYP_HEADSET_MAC`) or edit the default in
`main.py`.

---

## Usage

### Bench test (no button needed)

`main.py` has a keyboard mode so you can test everything before the GPIO button
is wired:

```bash
cd ~/fyp && source venv/bin/activate && python3 main.py
```

```
ENTER          → scene detection
l  then ENTER  → OCR / read text
q  then ENTER  → quit
```

For bench testing over the 3.5 mm / HDMI output instead of Bluetooth:

```bash
FYP_AUDIO_DEVICE=default python3 main.py
```

### On the wearable

With `FYP_USE_GPIO=1`, a short press runs detection and holding the button for
more than a second runs OCR.

### Run as a service

Once the button is wired (GPIO17 ↔ GND) and tested:

```bash
sudo systemctl enable --now fyp.service
journalctl -u fyp.service -f      # live logs
```

---

## Configuration

Most behaviour is controlled by constants at the top of `main.py`, each
overridable by an environment variable so the systemd unit can set them without
editing code.

| Variable | Default | Purpose |
|----------|---------|---------|
| `FYP_USE_GPIO` | `0` | `1` = GPIO button, `0` = keyboard test mode |
| `FYP_AUDIO_DEVICE` | BlueALSA A2DP | `default` routes to 3.5 mm/HDMI for bench tests |
| `FYP_HEADSET_MAC` | your paired MAC | Bluetooth headset address |

OCR tuning constants (in `main.py`):

| Constant | Default | Meaning |
|----------|---------|---------|
| `OCR_PSM` | `6` | Tesseract page-segmentation mode (assume a uniform block) |
| `OCR_MIN_CONF` | `40` | Drop words Tesseract is less than this % sure of |
| `OSD_MIN_CONFIDENCE` | `2.0` | Ignore orientation guesses weaker than this |

If OCR does poorly on non-block layouts (scattered signage, single lines),
`OCR_PSM = 3` (auto) or `11` (sparse text) is the first knob to try.

---

## Troubleshooting

- **No speech, but console prints the text** — the headset isn't connected.
  `bluetoothctl info <MAC>` should say `Connected: yes`. `main.py` prints a
  preflight warning when the sink is down.
- **OCR is blurry or empty** — check `/tmp/capture.jpg` (the last capture is
  always saved there). If it's out of focus, the autofocus cycle didn't settle;
  if it's dark, add light. The console logs which preprocessing pass won and the
  raw recognized text.
- **`picamera2` import fails inside the venv** — the venv must be created with
  `--system-site-packages`; `install.sh` does this, but a hand-rolled venv
  won't have it.

---

## Limitations & roadmap

- English only for now (single Piper voice, Latin-script OCR).
- OCR assumes reasonably flat, reasonably lit text — it's an assistive reader,
  not a document scanner.
- Detection vocabulary is the COCO class set that ships with YOLOv8n.

Planned next steps: barcode/currency modes, a distance cue for detected
obstacles, and a lighter wake-word alternative to the physical button.

---

## Acknowledgements

Built as a final-year capstone project. Uses YOLOv8 (Ultralytics), Tesseract
OCR, Piper TTS, and the Raspberry Pi `picamera2` stack.
