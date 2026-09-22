"""Save the input frame and an annotated result image, for the write-up.

Nothing here is needed for the wearable to work - it exists so every trigger
leaves behind evidence of what the camera saw and what the system concluded,
which is what a report needs. Saving is best-effort: a failure here must
never stop the device from speaking, so callers get None rather than an
exception.

Files land in FYP_REPORT_DIR (default ~/fyp/reports) as a timestamped set:

    20260922-151203-detect-input.jpg      what the camera saw
    20260922-151203-detect-output.jpg     boxes, labels, distances, phrase
    20260922-151203-detect-sidebyside.jpg both, captioned, for pasting in
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np

REPORT_DIR = Path(os.environ.get(
    "FYP_REPORT_DIR", str(Path.home() / "fyp" / "reports")))

# Reports are for a document, not for the device: shrink so a session's worth
# of captures does not fill the SD card.
MAX_REPORT_WIDTH = 1280

BOX_COLOUR = (0, 220, 0)
TEXT_COLOUR = (255, 255, 255)
BANNER_COLOUR = (32, 32, 32)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _scaled(img):
    """Downscale for report use, preserving aspect ratio."""
    h, w = img.shape[:2]
    if w <= MAX_REPORT_WIDTH:
        return img.copy()
    scale = MAX_REPORT_WIDTH / w
    return cv2.resize(img, (MAX_REPORT_WIDTH, int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def _banner(img, text: str, height: int = 64):
    """Caption strip under an image, wrapped to the image width."""
    w = img.shape[1]
    strip = np.full((height, w, 3), BANNER_COLOUR, np.uint8)
    if not text:
        return strip
    scale, thick = 0.6, 2
    # crude wrap: break on width, good enough for one or two short lines
    max_chars = max(10, int(w / (scale * 19)))
    lines = [text[i:i + max_chars] for i in range(0, len(text), max_chars)][:2]
    for i, line in enumerate(lines):
        cv2.putText(strip, line, (12, 26 + i * 26), FONT, scale,
                    TEXT_COLOUR, thick, cv2.LINE_AA)
    return strip


def annotate(frame_bgr, detections, phrase: str = ""):
    """Draw boxes, labels and distances onto a copy of the frame."""
    img = _scaled(frame_bgr)
    scale_x = img.shape[1] / frame_bgr.shape[1]
    scale_y = img.shape[0] / frame_bgr.shape[0]

    for det in detections or []:
        x1, y1, x2, y2 = (int(v * s) for v, s in
                          zip(det["box"], (scale_x, scale_y, scale_x, scale_y)))
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOUR, 2)

        label = f"{det['label']} {det.get('conf', 0):.2f}"
        if det.get("distance_m"):
            label += f"  ~{det['distance_m']:.1f}m"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 8)),
                      (x1 + tw + 8, y1), BOX_COLOUR, -1)
        cv2.putText(img, label, (x1 + 4, max(th, y1 - 5)), FONT, 0.55,
                    (0, 0, 0), 2, cv2.LINE_AA)

    return np.vstack([img, _banner(img, phrase)]) if phrase else img


def save_report(frame_bgr, detections, phrase: str, mode: str = "detect"):
    """Write input / output / side-by-side images. Returns paths, or None."""
    try:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        # Millisecond suffix: two triggers inside the same second would
        # otherwise share a filename and silently overwrite each other.
        stem = (f"{time.strftime('%Y%m%d-%H%M%S')}"
                f"{int((time.time() % 1) * 1000):03d}-{mode}")

        raw = _scaled(frame_bgr)
        out = annotate(frame_bgr, detections, phrase)

        paths = {
            "input": REPORT_DIR / f"{stem}-input.jpg",
            "output": REPORT_DIR / f"{stem}-output.jpg",
            "sidebyside": REPORT_DIR / f"{stem}-sidebyside.jpg",
        }
        cv2.imwrite(str(paths["input"]), raw)
        cv2.imwrite(str(paths["output"]), out)

        # Pad the shorter image so the two can sit side by side.
        left = np.vstack([raw, _banner(raw, "INPUT - what the camera saw")])
        right = out
        height = max(left.shape[0], right.shape[0])
        def pad(img):
            if img.shape[0] == height:
                return img
            fill = np.full((height - img.shape[0], img.shape[1], 3),
                           BANNER_COLOUR, np.uint8)
            return np.vstack([img, fill])
        divider = np.full((height, 6, 3), (90, 90, 90), np.uint8)
        cv2.imwrite(str(paths["sidebyside"]),
                    np.hstack([pad(left), divider, pad(right)]))

        return {k: str(v) for k, v in paths.items()}
    except Exception as e:
        print(f"[REPORT] could not save report images: {e}")
        return None
