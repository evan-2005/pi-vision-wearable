# Plan: add an Intel RealSense D435i on a 5 V / 3 A power bank

Status: **plan only, nothing implemented.** Branch `feat/realsense-d435i`.

## Goal

Use the D435i's depth sensor to give real measured distances, keep every
existing function (detect, find, read text, banknote, emotion, voice, button,
power-off), and stay inside what a 5 V / 3 A (15 W) power bank can deliver
without resets or hardware stress.

## Decision needed first: add the D435i, or replace the Camera Module 3?

**Recommended: add it. Keep the Camera Module 3 for pictures, use the D435i
for depth only.**

| | Keep Camera Module 3 + D435i depth (recommended) | D435i only |
|---|---|---|
| Text reading, banknotes | Unchanged: 12 MP with autofocus | Worse: the D435i colour camera is 1080p fixed-focus, so small print is soft |
| Detection, emotion, find | Unchanged | Works, needs re-tuning for the new lens |
| Distance | Measured by depth | Measured by depth, boxes line up exactly |
| Code change | Small: one new module, `distance` gains a measured path | Large: new capture path for every mode |
| Extra power | Depth stream only, on demand | Depth + colour streams |
| Cost | One-off alignment between the two cameras | Loses autofocus for OCR |

The rest of this plan assumes the recommended option.

## Power budget

Estimates; step 1 replaces them with measurements on this Pi.

| Consumer | Typical | Peak |
|---|---|---|
| Pi 5, service idle, Wi-Fi + Bluetooth | 3-4 W | - |
| Pi 5 during YOLO / Tesseract (4 cores) | - | 7-8 W |
| Camera Module 3 | 1 W | 1 W |
| D435i, depth stream with laser projector | 2 W (one user measured 440 mA) | up to about 3.5 W (700 mA) |
| **Worst case if everything overlaps** | | **about 12.5 W of 15 W** |

Two limits matter, and the second is the tighter one:

1. **Total: 15 W.** There is about 2.5 W of headroom at worst case, less if the
   bank sags under load or the cable is thin.
2. **USB ports: 600 mA total.** On a 5 V / 3 A supply the Pi 5 caps its USB
   ports at 600 mA unless `usb_max_current_enable=1` is set, which raises the
   cap to 1.6 A. The D435i can peak above 600 mA with the projector on.

The D435i figures are not from the official datasheet table (the search did
not surface it); treat them as provisional until measured.

## How it stays inside the budget

Applied in this order; stop as soon as the measurements say it is enough.

1. **Depth on demand, not continuous.** Start the depth stream on a button
   press, take a few frames, stop it. The camera idles at roughly 0.5 W the
   rest of the time. Cost: about 1 s extra per press.
2. **Never overlap the two peaks.** Grab the depth frame and stop the stream
   *before* running YOLO, so the camera peak and the CPU peak do not coincide.
3. **Smallest useful depth mode.** 424x240 or 480x270 at 6 fps, colour and
   IMU streams off.
4. **Lower the laser power.** Reduce the projector from its default until
   depth on a plain wall starts to fail, then back off one step. This keeps
   the camera under the 600 mA USB cap without touching `config.txt`.
5. **Cap inference threads.** Run YOLO/NCNN on 2-3 cores instead of 4. No
   root needed, lowers the CPU peak; detection gets a little slower.
6. **Downclock (only if 1-5 are not enough).** `arm_freq=1800` in
   `/boot/firmware/config.txt` (stock is 2400). Needs sudo and a reboot.
7. **Raise the USB cap (last resort).** `usb_max_current_enable=1`. Only
   after measurements show the total stays under about 13 W, because this
   flag removes a protection rather than adding power.

## Will anything be damaged?

No realistic path to electrical damage:

- The Pi limits current on its USB ports, and the power bank has its own
  over-current cut-off. Overdrawing causes a voltage sag or shutdown, not heat
  damage.
- The real risks are **a reset mid-write corrupting the SD card** and **the
  camera dropping off USB**. The measures above and the checks below target
  those.

Rules to keep it that way:

- Plug the D435i straight into a **blue USB 3 port** with the short cable it
  shipped with. No unpowered hubs or long cables.
- **Do not** power the camera from a second power-bank output through a
  Y-cable or a hub that back-feeds the Pi. If a powered hub is ever needed, it
  must be one that does not back-power its upstream port.
- Keep using a short, thick USB-C cable from the bank to the Pi; cable drop is
  the usual cause of under-voltage on a "3 A" bank.
- Plug and unplug the camera with the Pi powered off while testing.

## Software changes

| File | Change |
|---|---|
| `fypvision/depth.py` (new) | Open D435i, grab a median of a few depth frames, close. `available()` returns False when the camera or library is missing, like `voice.available()`. |
| `fypvision/distance.py` | New measured path: median depth inside the central part of a detection box. Falls back to today's size-based estimate when depth is missing or invalid (too near, glass, sunlight). |
| `main.py` | Take the depth frame next to the picture in detect and find; pass it to `extract_detections`. Read text, banknote, emotion, voice, button, power-off untouched. |
| `install.sh` | Install `pyrealsense2`, add the RealSense udev rule, optional `config.txt` lines from the power steps. |
| `tests/` | Depth-to-distance and fallback tests with synthetic depth arrays (no camera needed). |
| `README.md` | Hardware, power rules, calibration. |

Everything keeps working with the D435i unplugged: distance silently uses
the current estimate.

**Aligning the two cameras.** The depth image and the Camera Module 3 picture
come from different lenses (depth is about 87 degrees wide, the picture about
66), and the Camera Module 3 is mounted upside-down. A box is mapped into the
depth image by field-of-view ratio plus a fixed offset, measured once with a
calibration shot of one object at a known distance. Sampling the median of
the box centre makes a few pixels of error harmless.

## Steps, in order

1. **Baseline measurement, no D435i.** Log `vcgencmd pmic_read_adc` (input
   voltage and rail currents) and `vcgencmd get_throttled` during idle,
   detection and text reading. This gives the Pi's real draw on this bank.
2. **Install `pyrealsense2`.** Try the piwheels build first (it lists Python
   3.13 on Trixie); PyPI's own ARM wheels stop at Python 3.12. If that fails,
   build librealsense from source with `-DFORCE_RSUSB_BACKEND=ON` (about an
   hour on the Pi). Needs sudo for the udev rule.
3. **Plug in the D435i and measure.** Same logging as step 1 with the depth
   stream idle, streaming, and streaming during YOLO. Check `dmesg` for USB
   over-current or disconnects.
4. **Apply power measures 1-4, re-measure.** Add 5-7 only if needed.
5. **Write `depth.py` and the `distance.py` change, with tests first.**
6. **Calibrate alignment** and compare measured distance with a tape measure
   at 0.5, 1, 2 and 4 m.
7. **Soak test on the power bank:** 30 minutes of a press every 20 s,
   alternating detect and read text. Pass = `get_throttled` stays `0x0`, input
   voltage stays above 4.8 V, no USB disconnects, service never restarts.
8. **Deploy, README, commit.**

## Pass criteria

- `vcgencmd get_throttled` is `0x0` after the soak test.
- 5 V input never below 4.8 V under load.
- No USB disconnect or over-current lines in `dmesg`.
- Distance within 10 % of a tape measure from 0.5 to 4 m.
- All existing tests still pass; the device behaves as today with the D435i
  unplugged.

## Risks and unknowns

- **`pyrealsense2` on Pi 5 / Python 3.13 is unconfirmed.** One report says a
  D435f worked on a Pi 4 but not a Pi 5. Step 2 finds out before any code is
  written.
- **D435i peak current is provisional** until measured in step 3.
- **Depth has blind spots:** nearer than about 0.3 m, glass, mirrors, strong
  sunlight. The size-based estimate stays as the fallback.
- **Mounting.** The D435i is 90 mm wide and about 72 g; it has to sit rigidly
  next to the Camera Module 3 or the alignment drifts.
- **Battery life** drops by the camera's average draw; on-demand depth keeps
  that small.

## Sources

- [Raspberry Pi: USB power delivery on Raspberry Pi 5 (white paper)](https://pip-assets.raspberrypi.com/categories/685-app-notes-guides-whitepapers/documents/RP-009856-WP-1-USB%20Power%20delivery%20on%20Raspberry%20Pi%205.pdf)
- [RealSense D400 Series datasheet, March 2026](https://www.realsenseai.com/wp-content/uploads/2026/03/RealSense-D400-Series-Datasheet-Mar-2026.pdf)
- [Intel community: D435 power management (440 mA measurement)](https://community.intel.com/t5/Items-with-no-label/D435-Power-Management/m-p/565219)
- [piwheels: pyrealsense2](https://www.piwheels.org/project/pyrealsense2)
- [RealSense support: Raspberry Pi OS support](https://support.realsenseai.com/hc/en-us/community/posts/25181882603155/comments/25185545664915)
