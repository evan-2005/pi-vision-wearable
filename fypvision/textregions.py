"""Find where the text is in a frame and hand each block back upright.

Tesseract reads a tidy, level page well and almost nothing else. Given a
whole camera frame it has two problems:

  - most of the frame is not text, and it spends seconds inventing "words"
    out of carpet, skin and wall texture;
  - a line tilted more than about ten degrees is not read at all, and a
    body-worn camera is rarely level with the sign it is pointed at.

So before any OCR this module looks for text-line-shaped blobs (dense edges,
much longer than tall), groups neighbouring lines into blocks, and returns
each block as its own small image: cropped, rotated level and scaled so the
letters are a size Tesseract likes. Detection is plain morphology on a
downscaled frame and costs tens of milliseconds.

Limits: text tilted more than MAX_SKEW_DEG is ignored (it could as well be
vertical clutter), and upside-down text comes back upside-down - the caller
decides between 0 and 180 degrees by reading both.
"""

from typing import NamedTuple

import cv2
import numpy as np

# Frame widths the search runs at. The fine pass finds ordinary print; the
# coarse pass finds sign lettering too large for the fine pass to see as a
# line (its letters do not merge into one blob).
DETECT_WIDTHS = (1152, 384)
# A text line, measured at detection scale, in pixels.
MIN_LINE_LENGTH = 30
MIN_LINE_HEIGHT = 6
MAX_LINE_HEIGHT = 80
MIN_LINE_ASPECT = 1.3          # length / height
MIN_LINE_FILL = 0.45           # blob area / its bounding rectangle
# Closing with a wide, one-pixel-high kernel joins letters into words and
# words into lines without also fusing one line to the next.
CLOSE_KERNEL = (13, 1)
MAX_SKEW_DEG = 45.0
# Lines this many text-heights apart (vertically) belong to one block.
BLOCK_LINE_GAP = 1.1
# Members of a block must agree on tilt to within this, in degrees.
BLOCK_ANGLE_TOLERANCE = 12.0
# Blocks are scaled so a text line is about this tall for Tesseract.
TARGET_LINE_HEIGHT = 40
MIN_SCALE, MAX_SCALE = 0.25, 2.0
MAX_BLOCKS = 8                 # bounds the time spent on a cluttered frame
MAX_BLOCK_SIDE = 2400          # px, longest side of a returned crop
PROBE_SIDE = 1200              # px, longest side of the measuring crop
# A row of a levelled crop holds text if at least this fraction of it is
# edge pixels (an absolute floor: relative to the busiest row, the border of
# a sign outvotes its lettering), and a run of such rows this tall is a line.
ROW_INK_FRACTION = 0.02
MIN_TEXT_ROW_PX = 5
# Fewest edge crossings along a row for a blob to count as lettering.
MIN_STROKE_CHANGES = 8
GROUP_CANVAS_WIDTH = 576       # blocks are grouped on a mask this wide
ROW_BAND_FRACTION = 0.08       # blocks within this much height share a row


class Line(NamedTuple):
    """One text-line-shaped blob, in full-frame pixels."""
    cx: float
    cy: float
    length: float
    height: float
    angle: float               # degrees, cv2.minAreaRect convention


class Block(NamedTuple):
    """A block of text, cropped level and scaled for OCR."""
    image: np.ndarray          # grayscale
    cx: float                  # centre in the original frame
    cy: float
    angle: float               # tilt that was removed, degrees
    line_height: float         # text height in the original frame, px


def _lines_at_width(gray: np.ndarray, width: int) -> list:
    """Text-line blobs found with the frame shrunk to `width` pixels."""
    scale = min(1.0, width / gray.shape[1])
    small = cv2.resize(gray, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    joined = cv2.morphologyEx(
        _edge_map(small).astype(np.uint8) * 255, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, CLOSE_KERNEL))
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        (cx, cy), (length, height), angle = cv2.minAreaRect(contour)
        if length < height:
            length, height, angle = height, length, angle + 90.0
        angle = (angle + 90.0) % 180.0 - 90.0
        if not _is_line_shaped(length, height, angle):
            continue
        if cv2.contourArea(contour) < MIN_LINE_FILL * length * height:
            continue
        lines.append(Line(cx / scale, cy / scale, length / scale,
                          height / scale, angle))
    return lines


def _is_line_shaped(length: float, height: float, angle: float) -> bool:
    return (length >= MIN_LINE_LENGTH
            and MIN_LINE_HEIGHT <= height <= MAX_LINE_HEIGHT
            and length >= MIN_LINE_ASPECT * height
            and abs(angle) <= MAX_SKEW_DEG)


def find_lines(gray: np.ndarray) -> list:
    """Every text-line blob in the frame, from all detection scales."""
    lines = []
    for width in DETECT_WIDTHS:
        lines.extend(_lines_at_width(gray, width))
    return lines


def _weighted_median(values, weights) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(np.asarray(weights, dtype=float)[order])
    return float(np.asarray(values, dtype=float)[order][
        np.searchsorted(cumulative, cumulative[-1] / 2.0)])


def _corners(line: Line, pad_length: float, pad_height: float) -> np.ndarray:
    """Corner points of a line's rectangle, grown by the given padding."""
    return cv2.boxPoints(((line.cx, line.cy),
                          (line.length + pad_length, line.height + pad_height),
                          line.angle))


def _group(lines: list, frame_shape) -> list:
    """Split lines into blocks: lists of lines that sit close together."""
    scale = GROUP_CANVAS_WIDTH / frame_shape[1]
    canvas = np.zeros((int(frame_shape[0] * scale) + 1,
                       GROUP_CANVAS_WIDTH + 1), np.uint8)
    for line in lines:
        grown = _corners(line, line.height,
                         line.height * BLOCK_LINE_GAP) * scale
        cv2.fillPoly(canvas, [np.round(grown).astype(np.int32)], 255)
    _, labels = cv2.connectedComponents(canvas)
    groups = {}
    for line in lines:
        y = min(int(line.cy * scale), canvas.shape[0] - 1)
        x = min(int(line.cx * scale), canvas.shape[1] - 1)
        groups.setdefault(int(labels[y, x]), []).append(line)
    groups.pop(0, None)        # a centre that fell on background
    return list(groups.values())


def _level_matrix(points: np.ndarray, angle: float, scale: float,
                  margin: float):
    """Affine transform that levels `points` and fits them, plus a margin,
    into an image starting at the origin. Returns (matrix, (width, height))."""
    centre = points.mean(axis=0)
    matrix = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])),
                                     angle, scale)
    turned = points @ matrix[:, :2].T + matrix[:, 2]
    x0, y0 = turned.min(axis=0) - margin * scale
    x1, y1 = turned.max(axis=0) + margin * scale
    matrix[:, 2] -= (x0, y0)
    return matrix, (int(round(x1 - x0)), int(round(y1 - y0)))


def _warp(gray: np.ndarray, matrix: np.ndarray, size, scale: float):
    return cv2.warpAffine(
        gray, matrix, size,
        flags=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE)


def _edge_map(gray: np.ndarray) -> np.ndarray:
    """Boolean map of strong edges. Letters are dense in them whichever way
    round the text is (dark on light or light on dark)."""
    edges = cv2.morphologyEx(
        gray, cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    _, binary = cv2.threshold(edges, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary > 0


def _text_rows(edges: np.ndarray) -> list:
    """(start, end) row spans that hold text in a levelled crop's edge map.

    Rows crossing letters are full of edges and the gaps between lines have
    none, so the spans come straight off the row-by-row edge count.
    """
    busy = np.concatenate(([False], edges.mean(axis=1) > ROW_INK_FRACTION,
                           [False]))
    changes = np.flatnonzero(busy[1:] != busy[:-1])
    return [(int(a), int(b)) for a, b in zip(changes[::2], changes[1::2])
            if b - a >= MIN_TEXT_ROW_PX]


def _stroke_count(edges: np.ndarray, rows: list) -> int:
    """Most edge crossings along any sampled row of the tallest text line.

    A row through a word crosses a stroke every few pixels; a row through a
    shadow, a cable or a pair of LEDs crosses two or three. That separates
    text from the line-shaped clutter the blob search also picks up.
    """
    start, end = max(rows, key=lambda span: span[1] - span[0])
    best = 0
    for fraction in (0.25, 0.4, 0.5, 0.6, 0.75):
        row = edges[start + int((end - start) * fraction)]
        best = max(best, int(np.count_nonzero(row[1:] != row[:-1])))
    return best


def _extract(gray: np.ndarray, members: list) -> "Block | None":
    """Crop one group of lines out of the frame, level and scaled.

    Returns None when the levelled crop turns out not to look like text.
    """
    lengths = [m.length for m in members]
    angle = _weighted_median([m.angle for m in members], lengths)
    agreeing = [m for m in members
                if abs(m.angle - angle) <= BLOCK_ANGLE_TOLERANCE]
    points = np.vstack([_corners(m, 0.0, 0.0) for m in agreeing])
    margin = 0.5 * min(m.height for m in agreeing)

    # First pass at (up to) native size, only to measure the text: blob
    # heights are unreliable - a whole paragraph can come back as one blob.
    extent = float(np.ptp(points, axis=0).max()) + 2 * margin
    probe_scale = min(1.0, PROBE_SIDE / extent)
    matrix, size = _level_matrix(points, angle, probe_scale, margin)
    if min(size) < 8:
        return None
    edges = _edge_map(_warp(gray, matrix, size, probe_scale))
    rows = _text_rows(edges)
    if not rows or _stroke_count(edges, rows) < MIN_STROKE_CHANGES:
        return None
    heights = [end - start for start, end in rows]
    line_height = _weighted_median(heights, heights) / probe_scale

    scale = float(np.clip(TARGET_LINE_HEIGHT / line_height,
                          MIN_SCALE, MAX_SCALE))
    scale = min(scale, MAX_BLOCK_SIDE / extent)
    matrix, size = _level_matrix(points, angle, scale, margin)
    if min(size) < 8:
        return None
    centre = points.mean(axis=0)
    return Block(_warp(gray, matrix, size, scale), float(centre[0]),
                 float(centre[1]), angle, line_height)


def _reading_order(blocks: list, frame_height: int) -> list:
    """Top to bottom, then left to right within a row of blocks."""
    band = max(1.0, ROW_BAND_FRACTION * frame_height)
    return sorted(blocks, key=lambda b: (int(b.cy // band), b.cx))


def find_blocks(gray: np.ndarray) -> list:
    """Text blocks in `gray`, each level and scaled, in reading order.

    Returns an empty list when nothing in the frame looks like text.
    """
    groups = _group(find_lines(gray), gray.shape)
    # Most text first, so that if there are too many candidates the ones
    # dropped are the specks of clutter rather than the paragraph.
    groups.sort(key=lambda g: -sum(m.length * m.height for m in g))
    blocks = [_extract(gray, g) for g in groups[:MAX_BLOCKS]]
    return _reading_order([b for b in blocks if b is not None],
                          gray.shape[0])
