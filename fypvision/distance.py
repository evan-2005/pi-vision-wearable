"""Monocular distance estimation from bounding-box size.

With one fixed camera and no depth sensor, the only cue available is how
large a known object appears. For an object of real size S metres whose box
is s pixels across, the pinhole model gives its depth along the camera axis:

    depth = (S * focal_length_px) / s

and its straight-line range is that depth divided by the cosine of the angle
it sits off-axis, which matters on a wide lens (up to +40% at the frame
edge).

What limits accuracy, roughly in order:
  - the assumed real size. People are 1.5-1.9 m and may be sitting; a
    "chair" may be a stool. Expect +/-20% from this alone.
  - a box cut off by the frame edge. The visible part is smaller than the
    object, so the plain formula reads too far. Here the other dimension is
    used when it is whole, and otherwise the result is flagged as an upper
    bound ("within about 2 metres") instead of being passed off as a
    measurement.
  - the focal length. The default comes from the lens datasheet; measuring
    it (see focal_length_px) removes a few percent of systematic error.

It is good enough to say "about 3 metres", never good enough for navigation.
Callers should treat a None result as "cannot tell" and say nothing.
"""

import math
import os
from typing import NamedTuple, Optional

# Camera Module 3 Wide (imx708_wide): 2.75 mm lens over a sensor 6.45 mm wide
# (4608 px of 1.4 um). The ratio gives the focal length in pixels for any
# full-width capture mode, and is more accurate at the image centre than
# working back from the quoted field of view.
LENS_FOCAL_MM = 2.75
SENSOR_WIDTH_MM = 6.45

# Typical real-world (height, width) in metres for COCO classes a wearer
# plausibly meets indoors or on a pavement. Width is None where it depends
# too much on which way the object is facing (a car side-on vs head-on) to
# be worth using. Anything absent simply gets no distance.
KNOWN_SIZES_M = {
    "person": (1.65, 0.45), "bicycle": (1.05, None), "car": (1.50, None),
    "motorcycle": (1.10, None), "bus": (3.00, None), "truck": (3.00, None),
    "train": (3.50, None),
    "bench": (0.85, None), "chair": (0.90, 0.48), "couch": (0.85, None),
    "bed": (0.60, None), "dining table": (0.75, None),
    "toilet": (0.75, 0.40), "tv": (0.60, None), "laptop": (0.23, 0.34),
    "refrigerator": (1.70, 0.65), "oven": (0.90, 0.60),
    "microwave": (0.30, 0.48), "sink": (0.25, None),
    "potted plant": (0.50, None), "backpack": (0.45, 0.30),
    "suitcase": (0.60, 0.40), "umbrella": (0.90, None),
    "bottle": (0.24, 0.07), "cup": (0.10, 0.08), "book": (0.23, 0.16),
    "cell phone": (0.15, 0.07), "keyboard": (0.14, 0.44),
    "mouse": (0.04, 0.07), "remote": (0.18, 0.05), "clock": (0.30, 0.30),
    "dog": (0.50, None), "cat": (0.28, None), "bird": (0.18, None),
    "traffic light": (0.90, 0.30), "stop sign": (0.75, 0.75),
    "fire hydrant": (0.75, 0.30),
}

# Below this fraction of the frame a box is too few pixels for the estimate
# to mean anything.
MIN_BOX_FRACTION = 0.03
# A box edge this close to the frame edge (as a fraction of the frame) means
# the object carries on out of view, so that dimension is not its full size.
EDGE_MARGIN_FRACTION = 0.01

# A measured focal length always beats one derived from a datasheet.
# Set FYP_CAMERA_FOCAL_PX after calibrating: photograph an object of known
# height H metres, upright and near the middle of the frame, at a known
# distance D metres, note its pixel height h, then
#     focal_px = h * D / H
# FYP_CAMERA_HFOV (degrees) is the older, coarser override and is used only
# if no focal length is given.
_FOCAL_PX_OVERRIDE = os.environ.get("FYP_CAMERA_FOCAL_PX")
_HFOV_OVERRIDE = os.environ.get("FYP_CAMERA_HFOV")


class Estimate(NamedTuple):
    """A distance in metres. `is_upper_bound` marks an object that runs out
    of the frame: it is no farther than `metres`, and probably nearer."""
    metres: float
    is_upper_bound: bool = False


def _positive_float(raw: Optional[str], name: str) -> Optional[float]:
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        value = 0.0
    if value <= 0:
        print(f"[DISTANCE] ignoring bad {name}={raw!r}")
        return None
    return value


def focal_length_px(frame_width_px: int) -> float:
    """Focal length in pixels: measured if calibrated, else from the lens.

    NOTE: this is tied to the wearable's own camera. Estimates for images
    from any other source (stock photos, phone pictures) will be wrong,
    because their focal length differs.
    """
    measured = _positive_float(_FOCAL_PX_OVERRIDE, "FYP_CAMERA_FOCAL_PX")
    if measured:
        return measured
    hfov = _positive_float(_HFOV_OVERRIDE, "FYP_CAMERA_HFOV")
    if hfov and hfov < 180:
        return (frame_width_px / 2.0) / math.tan(math.radians(hfov / 2.0))
    return frame_width_px * LENS_FOCAL_MM / SENSOR_WIDTH_MM


def _off_axis_factor(box, frame_width_px: int, frame_height_px: int,
                     focal_px: float) -> float:
    """Ratio of straight-line range to depth for the centre of `box`."""
    x1, y1, x2, y2 = box
    dx = (x1 + x2) / 2.0 - frame_width_px / 2.0
    dy = (y1 + y2) / 2.0 - frame_height_px / 2.0
    return math.sqrt(1.0 + (dx * dx + dy * dy) / (focal_px * focal_px))


def estimate_distance(label: str, box, frame_width_px: int,
                      frame_height_px: int) -> Optional[Estimate]:
    """Range to a detected object, or None if it cannot be estimated.

    `box` is (x1, y1, x2, y2) in pixels. The object's height is used when
    the box shows all of it; if the box is cut off top or bottom, its width
    is used instead where the class has a dependable one; failing that the
    height still bounds how far away it can be.
    """
    sizes = KNOWN_SIZES_M.get(label)
    if sizes is None or frame_width_px <= 0 or frame_height_px <= 0:
        return None
    real_height, real_width = sizes
    x1, y1, x2, y2 = box
    box_width, box_height = x2 - x1, y2 - y1
    if box_width <= 0 or box_height <= 0:
        return None

    margin_x = EDGE_MARGIN_FRACTION * frame_width_px
    margin_y = EDGE_MARGIN_FRACTION * frame_height_px
    height_cut = y1 <= margin_y or y2 >= frame_height_px - margin_y
    width_cut = x1 <= margin_x or x2 >= frame_width_px - margin_x

    focal_px = focal_length_px(frame_width_px)
    range_factor = _off_axis_factor(box, frame_width_px, frame_height_px,
                                    focal_px)
    if not height_cut:
        if box_height < MIN_BOX_FRACTION * frame_height_px:
            return None          # too few pixels to trust
        return Estimate(real_height * focal_px / box_height * range_factor)
    if real_width is not None and not width_cut:
        if box_width < MIN_BOX_FRACTION * frame_width_px:
            return None
        return Estimate(real_width * focal_px / box_width * range_factor)
    # Cut off in every direction we could measure: the visible part is
    # smaller than the object, so this is the farthest it can be.
    return Estimate(real_height * focal_px / box_height * range_factor,
                    is_upper_bound=True)


def describe_distance(metres: Optional[float],
                      is_upper_bound: bool = False) -> str:
    """Phrase a distance the way a person would say it aloud.

    Deliberately coarse: the underlying estimate is not precise enough to
    justify decimals. Half-metre steps are used up close, where the
    difference between one metre and two matters most to someone walking.
    """
    if metres is None:
        return ""
    if metres < 0.75:
        return "very close" if is_upper_bound else "less than a metre"
    prefix = "within about" if is_upper_bound else "about"
    if metres < 1.25:
        return f"{prefix} a metre"
    if metres < 3.25:
        halves = round(metres * 2) / 2.0
        if halves == int(halves):
            return f"{prefix} {int(halves)} metres"
        return f"{prefix} {int(halves)} and a half metres"
    if metres < 10.0:
        return f"{prefix} {round(metres)} metres"
    return "within 10 metres or so" if is_upper_bound else "more than 10 metres"


def describe_direction(box_centre_x: float, frame_width_px: int) -> str:
    """Rough bearing, so the wearer knows which way to turn."""
    if frame_width_px <= 0:
        return "ahead"
    position = box_centre_x / frame_width_px
    if position < 0.33:
        return "to your left"
    if position > 0.67:
        return "to your right"
    return "straight ahead"
