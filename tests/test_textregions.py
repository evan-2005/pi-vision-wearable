"""Text-block finding, on synthetic frames (no camera or Tesseract needed)."""

import cv2
import numpy as np
import pytest

from fypvision import textregions

FRAME_SIZE = (1296, 2304)          # rows, cols - the wearable's capture size
PAPER, INK = 215, 25
LINES = ("NOTICE TO ALL RESIDENTS", "THE WATER SUPPLY WILL BE",
         "TURNED OFF ON TUESDAY")


def _frame_with_text(angle: float, lines=LINES, font_scale: float = 2.0):
    """A plain frame with `lines` drawn level, then turned by `angle`."""
    frame = np.full(FRAME_SIZE, PAPER, np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (600, 500 + int(i * 45 * font_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, INK,
                    int(2 * font_scale), cv2.LINE_AA)
    centre = (FRAME_SIZE[1] / 2, FRAME_SIZE[0] / 2)
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
    return cv2.warpAffine(frame, matrix, FRAME_SIZE[::-1],
                          borderValue=PAPER)


def _text_row_spans(block_image) -> list:
    return textregions._text_rows(textregions._edge_map(block_image))


def test_blank_frame_has_no_blocks():
    frame = np.full(FRAME_SIZE, PAPER, np.uint8)

    assert textregions.find_blocks(frame) == []


def test_long_straight_edge_is_not_mistaken_for_text():
    frame = np.full(FRAME_SIZE, PAPER, np.uint8)
    cv2.line(frame, (300, 640), (1700, 660), INK, 14)

    assert textregions.find_blocks(frame) == []


def test_level_paragraph_comes_back_as_one_block():
    blocks = textregions.find_blocks(_frame_with_text(0))

    assert len(blocks) == 1
    assert abs(blocks[0].angle) < 2


@pytest.mark.parametrize("tilt", [-25, -10, 8, 20])
def test_tilted_text_is_returned_level(tilt):
    blocks = textregions.find_blocks(_frame_with_text(tilt))

    assert len(blocks) == 1
    # Levelled text separates cleanly into its three lines; text that is
    # still tilted smears into one band.
    assert len(_text_row_spans(blocks[0].image)) == len(LINES)


@pytest.mark.parametrize("font_scale", [1.0, 2.0, 4.0])
def test_block_is_scaled_to_the_target_line_height(font_scale):
    blocks = textregions.find_blocks(
        _frame_with_text(0, font_scale=font_scale))

    assert len(blocks) == 1
    heights = [end - start for start, end in _text_row_spans(blocks[0].image)]
    assert textregions.TARGET_LINE_HEIGHT * 0.6 \
        <= np.median(heights) <= textregions.TARGET_LINE_HEIGHT * 1.4


def test_light_text_on_dark_background_is_found():
    frame = 255 - _frame_with_text(5)

    assert len(textregions.find_blocks(frame)) == 1


def test_blocks_come_back_in_reading_order():
    frame = np.full(FRAME_SIZE, PAPER, np.uint8)
    for x, y in ((1500, 1000), (300, 250), (1500, 250)):
        cv2.putText(frame, "KEEP CLEAR", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, INK, 4, cv2.LINE_AA)

    centres = [(round(b.cx, -2), round(b.cy, -2))
               for b in textregions.find_blocks(frame)]

    assert len(centres) == 3
    assert centres == sorted(centres, key=lambda c: (c[1], c[0]))


def test_weighted_median_follows_the_weight():
    assert textregions._weighted_median([10, 20, 300], [5, 1, 1]) == 10
    assert textregions._weighted_median([10, 20, 300], [1, 1, 9]) == 300
