# Pi Vision Wearable

An offline, wearable assistive-vision aid for people who are blind or have low
vision, built on a Raspberry Pi 5. One button does three jobs:

- **Press** — describes what's in front of you, with the nearest object's
  direction and distance ("2 people and 1 chair ahead. Nearest person to your
  left, about 2 metres.").
- **Double press** — reads printed text out loud (signs, labels, a page of a book).
- **Hold** — listens for a spoken command through the headset (speak after
  the beep; it stops listening when you stop talking): "where is my
  bottle", "how does she look", "what note is this", "read this".
- **Keep holding for 5 seconds** — says "Shutting down" and powers the Pi off
  safely. Give it about ten seconds before unplugging.

Power it on, switch the headset on, and it says "Ready" when it can be heard.

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
          press         │      double press
        ┌───────────────┴────────────────┐
        ▼                                 ▼
  capture frame                     autofocus lock
  YOLOv8n (NCNN)                    capture frame
  build phrase                     find text blocks → Tesseract
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
- **Find the text first.** Handing Tesseract a whole camera frame took 20-80
  seconds and returned noise: most of a scene is not text, and Tesseract
  cannot read a line tilted more than about ten degrees. `fypvision/textregions.py`
  looks for text-line-shaped blobs, groups them into blocks, and returns each
  block cropped, rotated level and scaled to a letter size Tesseract likes.
  Each block is then read on its own (`fypvision/ocr.py`). A page or sign
  now reads in 2-8 seconds, level or tilted up to 45°.
- **Upright frames.** The camera sits upside-down in the housing, so every
  frame is rotated 180° at capture (`FYP_CAMERA_ROTATION`). If nothing reads
  the right way up, the text-like blocks are read once more upside-down, for
  a page held the wrong way round. Tesseract's own orientation detector is
  not used - it once read "2019" as an upside-down "6LOZ".
- **Confidence filtering.** The reader uses `image_to_data` and drops words
  below a confidence threshold and stray single characters, and throws a
  whole block away when most of what Tesseract saw in it was rejected, so
  texture that merely looks like lettering is never spoken.
- **Time limits.** One Tesseract call is capped at `OCR_TIMEOUT_S` and a
  whole read at `OCR_BUDGET_S`, so a cluttered scene with no text in it
  still answers promptly.

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
| Push button | Bare switch (GPIO17 → GND) or a 3-pin button module; see *Wiring the button* |
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
├── fypvision/           # feature modules
│   ├── textregions.py   #   finds blocks of text and levels them
│   ├── ocr.py           #   reads text blocks and banknotes with Tesseract
│   ├── audio.py         #   picks the connected headset for each utterance
│   ├── button.py        #   single / double / long press and the power-off hold
│   └── ...              #   distance, finder, emotion, currency, voice, reporting
├── tests/               # pytest unit tests (run without a camera)
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
7. Installs a `systemd` service, left **disabled** until the button is wired,
   and a sudoers rule that lets it power the Pi off.

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

There is no MAC to configure: speech goes to whichever paired headset is
connected at that moment, so the headset can be switched on after the Pi, or
swapped for another paired one.

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
e  then ENTER  → facial expression
c  then ENTER  → banknote
v  then ENTER  → voice command through the headset
f bottle       → find a named object
q  then ENTER  → quit
```

For bench testing over the 3.5 mm / HDMI output instead of Bluetooth:

```bash
FYP_AUDIO_DEVICE=default python3 main.py
```

### On the wearable

With `FYP_USE_GPIO=1` the button takes over from the keyboard: press to detect,
double press to read text, hold for 0.8 s to give a voice command. A press is
acted on 0.4 s after release, the time it takes to be sure it is not the first
half of a double press. Set `FYP_LONG_PRESS=read` to make holding read text
instead, for a headset with no usable microphone. Holding on for 5 s in
total powers the Pi off (the voice prompt it opened on the way is dropped).

### Wiring the button

A bare push button goes between GPIO17 (pin 11) and any GND pin.

A 3-pin button module (GND / VCC / OUT, sometimes labelled G / V / S) is wired:

| Module pin | Pi pin |
|------------|--------|
| GND | GND (pin 9) |
| VCC (middle) | **3.3 V (pin 1)** — never 5 V: the output would put 5 V on a GPIO pin, which damages the Pi |
| OUT / S | GPIO17 (pin 11) |

Some modules pull their output high when pressed and some low. `main.py` works
out which at start-up from the pin's resting level, so **do not hold the button
while the program starts**. To force it, set `FYP_BUTTON_ACTIVE=low` or `high`.

### Run as a service

Once the button is wired (GPIO17 ↔ GND) and tested:

```bash
sudo systemctl enable --now fyp.service
journalctl -u fyp.service -f      # live logs
```

From then on the wearable starts by itself on every power-on. The service
turns report images off (`FYP_REPORT=0`) so daily use does not fill the SD
card; run `main.py` by hand when you want them for the write-up.

---

## Configuration

Most behaviour is controlled by constants at the top of `main.py`, each
overridable by an environment variable so the systemd unit can set them without
editing code.

| Variable | Default | Purpose |
|----------|---------|---------|
| `FYP_USE_GPIO` | `0` | `1` = GPIO button, `0` = keyboard test mode |
| `FYP_BUTTON_ACTIVE` | `auto` | Pin level when pressed: `low`, `high`, or detect at start-up |
| `FYP_LONG_PRESS` | `voice` | What holding the button does: `voice` or `read` |
| `FYP_CAMERA_FOCAL_PX` | from the lens (982) | Measured focal length in pixels, for more accurate distances |
| `FYP_AUDIO_DEVICE` | BlueALSA A2DP | `default` routes to 3.5 mm/HDMI for bench tests |
| `FYP_HEADSET_MAC` | none | Headset to try when no connected one is detected; normally not needed |
| `FYP_REPORT` | `1` | `0` stops saving report images (the service sets this) |
| `FYP_REPORT_DIR` | `~/fyp/reports` | Where report images go |
| `FYP_CURRENCY` | `MYR` | Which currency to recognise |
| `FYP_LISTEN_SECONDS` | `5` | Longest the microphone stays open |
| `FYP_CAMERA_ROTATION` | `180` | Degrees to turn each frame so it is upright (`0` if the camera is mounted the right way up) |

OCR tuning constants (in `fypvision/ocr.py`):

| Constant | Default | Meaning |
|----------|---------|---------|
| `OCR_PSM` | `6` | Tesseract page-segmentation mode for each text block |
| `OCR_MIN_CONF` | `60` | Drop words Tesseract is less than this % sure of |
| `OCR_MIN_KEPT` | `0.5` | Discard a block if fewer than this share of its words survive |
| `OCR_TIMEOUT_S` | `8` | Cap on one Tesseract call |
| `OCR_BUDGET_S` | `12` | Cap on a whole read |

How text blocks are found (line sizes, tilt limit, target letter height) is
tuned in `fypvision/textregions.py`.

To run the unit tests (no camera needed): `python -m pytest tests`.

### Distance accuracy and calibration

Distance comes from how large a recognised object looks, given a typical real
size for its class (`fypvision/distance.py`). Expect roughly ±20%: people and
chairs come in different sizes. An object that runs out of the frame cannot be
measured, so it is reported as a limit instead ("within about 2 metres").

To remove the lens part of the error, measure the focal length once: stand an
object of known height `H` metres upright in the middle of the view at a known
distance `D` metres, run a detection, read its box height `h` in pixels off the
report image (at the full 2304-pixel width), and set

    FYP_CAMERA_FOCAL_PX = h × D / H

---

## Troubleshooting

- **No speech, but console prints the text** — the headset isn't connected.
  `bluetoothctl devices Connected` should list it; the log shows
  `[AUDIO] no headset connected; not spoken`. It says "Ready" by itself once
  the headset is back.
- **Holding the button does not power off** — the log shows
  `[SHUTDOWN] poweroff failed`; re-run `install.sh`, which adds the sudoers
  rule that allows it.
- **OCR is blurry or empty** — check `/tmp/capture.jpg` (the last capture is
  always saved there). If it's out of focus, the autofocus cycle didn't settle;
  if it's dark, add light. The console logs how many text blocks were found,
  how long the read took and the raw recognized text.
- **Left and right are swapped, or nothing is detected** — the frames are
  upside-down. Look at `/tmp/capture.jpg`; if it is inverted, set
  `FYP_CAMERA_ROTATION` to match how the camera is mounted.
- **`picamera2` import fails inside the venv** — the venv must be created with
  `--system-site-packages`; `install.sh` does this, but a hand-rolled venv
  won't have it.

---

## Limitations & roadmap

- English only for now (single Piper voice, Latin-script OCR).
- OCR assumes reasonably flat, reasonably lit text — it's an assistive reader,
  not a document scanner. Text seen at a steep angle (perspective, not just
  tilt), text tilted past 45°, and a lone one- or two-character sign such as a
  bus number are not read.
- Detection vocabulary is the COCO class set that ships with YOLOv8n.

- Voice commands are matched against a fixed list of phrases; anything else
  is answered with "I did not catch that" rather than guessed at.
- Expression reading is a hint, not a fact: it needs a face at least 40 pixels
  wide, roughly facing the camera.

Planned next steps: a barcode mode and a lighter wake-word alternative to the
physical button.

---

## Acknowledgements

Built as a final-year capstone project. Uses YOLOv8 (Ultralytics), Tesseract
OCR, Piper TTS, and the Raspberry Pi `picamera2` stack.
