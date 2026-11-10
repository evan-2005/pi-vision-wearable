"""Scoring and filtering of OCR output (Tesseract itself is not run)."""

import numpy as np
import pytest

pytest.importorskip("pytesseract")
from fypvision import ocr  # noqa: E402


def _words(text: str, conf: float) -> list:
    return [(word, conf) for word in text.split()]


def test_confident_words_are_kept_in_order():
    score, text = ocr.score_words(_words("PUSH TO OPEN", 96))

    assert text == "PUSH TO OPEN"
    assert score > 0


def test_low_confidence_words_are_dropped():
    found = _words("Room 204", 95) + [("sagn", 31)]

    assert ocr.score_words(found)[1] == "Room 204"


def test_stray_letters_and_symbols_are_dropped():
    found = _words("EMERGENCY EXIT", 95) + [("f", 70), ("|", 86), ("=", 90)]

    assert ocr.score_words(found)[1] == "EMERGENCY EXIT"


def test_real_one_letter_words_survive():
    assert ocr.score_words(_words("I saw a bus", 92))[1] == "I saw a bus"


def test_read_is_discarded_when_most_of_it_was_noise():
    found = _words("in the app", 93) + _words("ee oS mee nee Ae ae Ne Oh", 20)

    assert ocr.score_words(found) == (0.0, "")


def test_read_with_no_content_is_discarded():
    assert ocr.score_words(_words("a 4 I 2", 90)) == (0.0, "")


def test_nothing_found_scores_zero():
    assert ocr.score_words([]) == (0.0, "")


def test_a_number_counts_as_content():
    assert ocr.score_words([("47", 95)])[1] == "47"


def test_clean_read_outscores_a_noisy_one():
    clean, _ = ocr.score_words(_words("KEEP CLEAR", 95))
    noisy, _ = ocr.score_words(_words("KEEP CLEAR", 95) + _words("xq zv", 10))

    assert clean > noisy > 0


def test_clean_ocr_text_strips_symbols_and_collapses_spaces():
    assert ocr.clean_ocr_text(" EXIT |  [here] ~ ") == "EXIT here"


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_rotate_image_turns_clockwise(angle):
    img = np.zeros((2, 3), np.uint8)
    img[0, 0] = 255                       # marker in the top-left corner

    turned = ocr.rotate_image(img, angle)

    corners = {0: (0, 0), 90: (0, -1), 180: (-1, -1), 270: (-1, 0)}
    assert turned[corners[angle]] == 255


@pytest.mark.parametrize("word, conf, expected", [
    ("50", 95, ["50"]),
    ("50", 85, []),                       # numbers need the higher bar
    ("MALAYSIA", 60, ["MALAYSIA"]),
    ("RM50", 95, []),                     # mixed token: neither rule applies
    ("1234", 96, []),                     # serial-number length
    ("the", 96, []),                      # too short to be an issuer word
])
def test_meaningful_tokens_for_banknotes(word, conf, expected):
    assert ocr._meaningful_tokens(word, conf) == expected
