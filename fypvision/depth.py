"""Measured distance from an Intel RealSense D435i depth camera.

The size-based estimate in distance.py guesses range from how big a known
object looks. A depth camera measures it, for any object, so when one is
plugged in its reading replaces the guess; when it is not (or the reading is
a hole: too near, glass, sunlight) the guess is used exactly as before.

The depth image is read straight from the camera's video node with v4l2-ctl
rather than through pyrealsense2, which has no build for the Pi's Python.
The stream is opened only for the moment of a capture, so the camera's
projector and sensors are idle the rest of the time.

The two cameras have different lenses and sit side by side, so a box in the
picture is carried into the depth image by angle, plus a fixed pixel offset
found once by calibration (see the README). Only the middle of the box is
sampled, which makes a few pixels of misalignment harmless.
"""

import math
import os
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from fypvision import distance as dist

WIDTH, HEIGHT = 848, 480        # the mode with the depth sensor's full view
SKIP_FRAMES = 8                 # let exposure settle after the stream opens
FRAMES = 3                      # median of these removes flicker
GRAB_TIMEOUT_S = 6.0
MIN_MM, MAX_MM = 200, 10000     # outside this the D435i reading is noise
CENTRE_FRACTION = 0.4           # sample this much of the box, about its centre
MIN_VALID_FRACTION = 0.3        # fewer good pixels than this is a hole


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        print(f"[DEPTH] ignoring bad {name}={os.environ[name]!r}")
        return default


DEPTH_HFOV_DEG = _env_float("FYP_DEPTH_HFOV", 87.0)
# Where the picture's centre lands in the depth image, in depth pixels from
# its centre. Zero suits cameras mounted close together and parallel.
OFFSET_X = _env_float("FYP_DEPTH_OFFSET_X", 0.0)
OFFSET_Y = _env_float("FYP_DEPTH_OFFSET_Y", 0.0)
ROTATION = int(_env_float("FYP_DEPTH_ROTATION", 0))   # 180 if mounted upside-down


def depth_focal_px(depth_width_px: int) -> float:
    return (depth_width_px / 2.0) / math.tan(math.radians(DEPTH_HFOV_DEG / 2.0))


def find_node() -> Optional[str]:
    """The RealSense video node that serves 16-bit depth, or None."""
    for name_file in sorted(Path("/sys/class/video4linux").glob("video*/name")):
        try:
            if "RealSense" not in name_file.read_text():
                continue
            node = f"/dev/{name_file.parent.name}"
            formats = subprocess.run(["v4l2-ctl", "-d", node, "--list-formats"],
                                     capture_output=True, text=True, timeout=3)
        except (OSError, subprocess.SubprocessError):
            continue
        if "Z16" in formats.stdout:
            return node
    return None


def available() -> bool:
    return find_node() is not None


def _capture(node: str) -> bytes:
    done = subprocess.run(
        ["v4l2-ctl", "-d", node,
         f"--set-fmt-video=width={WIDTH},height={HEIGHT},pixelformat=Z16 ",
         "--stream-mmap", f"--stream-skip={SKIP_FRAMES}",
         f"--stream-count={FRAMES}", "--stream-to=-"],
        capture_output=True, timeout=GRAB_TIMEOUT_S)
    return done.stdout


def grab() -> Optional[np.ndarray]:
    """One depth image in millimetres (0 = no reading), or None.

    None means "no depth this time" for any reason: camera unplugged, busy,
    or dropped off USB. Callers carry on with the size-based estimate.
    """
    node = find_node()
    if node is None:
        return None
    try:
        raw = _capture(node)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"[DEPTH] capture failed: {type(e).__name__}: {e}")
        return None
    if len(raw) != FRAMES * WIDTH * HEIGHT * 2:
        print(f"[DEPTH] short read from {node}: {len(raw)} bytes")
        return None
    frames = np.frombuffer(raw, dtype="<u2").reshape(FRAMES, HEIGHT, WIDTH)
    image = np.median(frames, axis=0).astype(np.uint16)
    return np.rot90(image, 2) if ROTATION == 180 else image


def measure(depth_mm: np.ndarray, box, frame_width_px: int,
            frame_height_px: int) -> Optional[float]:
    """Straight-line range in metres to whatever fills the middle of `box`.

    `box` is (x1, y1, x2, y2) in picture pixels. None if that part of the
    scene is outside the depth camera's view or has no dependable reading.
    """
    depth_h, depth_w = depth_mm.shape
    focal_px = dist.focal_length_px(frame_width_px)
    scale = depth_focal_px(depth_w) / focal_px

    x1, y1, x2, y2 = box
    dx = (x1 + x2) / 2.0 - frame_width_px / 2.0      # box centre, off the axis
    dy = (y1 + y2) / 2.0 - frame_height_px / 2.0
    cx = depth_w / 2.0 + dx * scale + OFFSET_X
    cy = depth_h / 2.0 + dy * scale + OFFSET_Y
    if not (0 <= cx < depth_w and 0 <= cy < depth_h):
        return None

    half_w = max(2.0, (x2 - x1) * scale * CENTRE_FRACTION / 2.0)
    half_h = max(2.0, (y2 - y1) * scale * CENTRE_FRACTION / 2.0)
    patch = depth_mm[max(0, int(cy - half_h)):int(cy + half_h) + 1,
                     max(0, int(cx - half_w)):int(cx + half_w) + 1]
    good = patch[(patch >= MIN_MM) & (patch <= MAX_MM)]
    if good.size < MIN_VALID_FRACTION * patch.size:
        return None

    # The camera reports depth along its axis; the wearer wants range.
    off_axis = math.sqrt(1.0 + (dx * dx + dy * dy) / (focal_px * focal_px))
    return float(np.median(good)) / 1000.0 * off_axis


def range_for(depth_mm: Optional[np.ndarray], label: str, box,
              frame_width_px: int, frame_height_px: int) -> Optional[dist.Estimate]:
    """Measured range when the depth camera has one, else the size estimate."""
    if depth_mm is not None:
        metres = measure(depth_mm, box, frame_width_px, frame_height_px)
        if metres is not None:
            return dist.Estimate(metres)
    return dist.estimate_distance(label, box, frame_width_px, frame_height_px)


if __name__ == "__main__":
    # Bench check: python -m fypvision.depth
    image = grab()
    if image is None:
        print("No depth image: is the D435i plugged in?")
    else:
        centre = (WIDTH * 0.45, HEIGHT * 0.45, WIDTH * 0.55, HEIGHT * 0.55)
        valid = 100.0 * np.count_nonzero(image) / image.size
        print(f"{valid:.0f}% of the depth image has a reading; straight ahead: "
              f"{dist.describe_distance(measure(image, centre, WIDTH, HEIGHT)) or 'no reading'}")
