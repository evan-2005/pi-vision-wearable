"""Monocular distance estimation from bounding-box size.

With one fixed camera and no depth sensor, the only cue available is how
large a known object appears. For an object of real height H metres whose
box is h pixels tall, the pinhole model gives:

    distance = (H * focal_length_px) / h

Accuracy is roughly +/-25% for upright, fully-visible objects, and degrades
badly when the object is cut off by the frame edge or seen at an angle. It
is good enough to say "about 3 metres", never good enough for navigation.
Callers should treat a None result as "cannot tell" and say nothing.
"""

import math
import os

# Camera Module 3 Wide (imx708_wide) horizontal field of view, degrees.
# Override with FYP_CAMERA_HFOV if a different lens is fitted.
CAMERA_HFOV_DEG = float(os.environ.get("FYP_CAMERA_HFOV", "102"))

# Typical real-world heights in metres, for COCO classes a wearer plausibly
# meets indoors or on a pavement. Anything absent simply gets no distance.
KNOWN_HEIGHTS_M = {
    "person": 1.70, "bicycle": 1.10, "car": 1.50, "motorcycle": 1.10,
    "bus": 3.00, "truck": 3.00, "train": 3.50,
    "bench": 0.90, "chair": 0.90, "couch": 0.80, "bed": 0.60,
    "dining table": 0.75, "toilet": 0.75, "tv": 0.60, "laptop": 0.25,
    "refrigerator": 1.70, "oven": 0.90, "microwave": 0.30, "sink": 0.90,
    "door": 2.00, "potted plant": 0.50, "backpack": 0.45, "suitcase": 0.55,
    "bottle": 0.25, "cup": 0.10, "book": 0.24, "cell phone": 0.15,
    "dog": 0.50, "cat": 0.30, "bird": 0.20,
    "traffic light": 0.80, "stop sign": 0.75, "fire hydrant": 0.75,
}

# Below this fraction of the frame height a box is too few pixels for the
# estimate to mean anything; above 0.95 the object is almost certainly
# cut off by the frame edge, so its true height is unknown.
MIN_BOX_FRACTION = 0.03
MAX_BOX_FRACTION = 0.95


# A measured focal length always beats one derived from a datasheet FOV.
# Set FYP_CAMERA_FOCAL_PX after calibrating: photograph an object of known
# height H metres at a known distance D metres, note its pixel height h, then
#     focal_px = h * D / H
# Leave unset to fall back on the nominal lens FOV above.
_FOCAL_PX_OVERRIDE = os.environ.get("FYP_CAMERA_FOCAL_PX")


def focal_length_px(frame_width_px: int) -> float:
    """Focal length in pixels: measured if calibrated, else from the FOV.

    NOTE: this is tied to the wearable's own camera. Estimates for images
    from any other source (stock photos, phone pictures) will be wrong,
    because their focal length differs.
    """
    if _FOCAL_PX_OVERRIDE:
        try:
            return float(_FOCAL_PX_OVERRIDE)
        except ValueError:
            print(f"[DISTANCE] ignoring bad FYP_CAMERA_FOCAL_PX="
                  f"{_FOCAL_PX_OVERRIDE!r}; using lens FOV instead")
    half_fov_rad = math.radians(CAMERA_HFOV_DEG / 2.0)
    return (frame_width_px / 2.0) / math.tan(half_fov_rad)


def estimate_distance_m(label: str, box_height_px: float,
                        frame_width_px: int, frame_height_px: int):
    """Distance to a detected object in metres, or None if not estimable."""
    real_height = KNOWN_HEIGHTS_M.get(label)
    if real_height is None or box_height_px <= 0 or frame_height_px <= 0:
        return None

    fraction = box_height_px / frame_height_px
    if not (MIN_BOX_FRACTION <= fraction <= MAX_BOX_FRACTION):
        return None      # too small to trust, or clipped by the frame edge

    return (real_height * focal_length_px(frame_width_px)) / box_height_px


def describe_distance(metres) -> str:
    """Phrase a distance the way a person would say it aloud.

    Deliberately coarse: the underlying estimate is not precise enough to
    justify decimals, and "about 3 metres" is easier to act on than "3.4".
    """
    if metres is None:
        return ""
    if metres < 1.0:
        return "less than a metre"
    if metres < 1.5:
        return "about a metre"
    if metres < 10.0:
        return f"about {round(metres)} metres"
    return "more than 10 metres"


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
