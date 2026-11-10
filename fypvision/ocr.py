"""Read text out of a camera frame with Tesseract.

Two readers live here:

  run_ocr         signs, labels and pages - whatever text is in view
  run_ocr_sparse  the few scattered tokens on a banknote, for currency.py

Both return a plain string, or NO_TEXT_PHRASE when nothing could be read.
Neither touches the camera or the speaker, so they can be run on a saved
image for testing.
"""

import re
import time

import cv2
import numpy as np
import pytesseract

from fypvision import textregions

OCR_PSM            = 6        # tesseract page segmentation mode (assume a block)
# Drop words Tesseract is less than this % sure of. Measured on this camera:
# words it really read score 85-96, words it made up from texture below 60.
OCR_MIN_CONF       = 60
# A read is discarded unless at least this share of the words Tesseract saw
# survived the confidence floor: a few confident words in a sea of rejects
# are coincidence, not text.
OCR_MIN_KEPT       = 0.5
# Hard cap on one Tesseract call. A real block of text reads in 1-4 s; on
# grainy texture Tesseract can grind for minutes over thousands of noise
# "words", and it is better to give up.
OCR_TIMEOUT_S      = 8
# Cap on a whole read, so a cluttered scene with no text in it still answers
# promptly. Checked between blocks, so the true worst case is this plus one
# OCR_TIMEOUT_S.
OCR_BUDGET_S       = 12
# If nothing reads upright, the blocks are tried upside-down (a page held the
# wrong way up). Only blocks where Tesseract saw at least this many glyph
# groups are worth that second read; the rest were not text either way up.
OCR_RETRY_MIN_BOXES = 3
# Sparse-text pass (currency): sizes to read at, and the confidence floors a
# number or word must clear to be trusted (see _meaningful_tokens).
OCR_SPARSE_TARGET_SIZES = (1600, 800)   # longest side, px, of each read
OCR_DIGIT_MIN_CONF  = 90
OCR_WORD_MIN_CONF   = 50
# Banknote localisation: how colourful a pixel must be to count as note (white
# walls sit near 30, notes 50+), the width the search runs at, and the
# smallest blob, as a fraction of the frame, still treated as a note.
NOTE_MIN_SATURATION = 50
NOTE_SEARCH_WIDTH   = 640
NOTE_MIN_AREA_FRAC  = 0.02
# Spoken when OCR finds nothing. Shared so callers can tell "read something"
# from "read nothing" without matching a literal.
NO_TEXT_PHRASE     = "No readable text found."

_ROTATIONS = {
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def rotate_image(img, angle: int):
    """Rotate by a multiple of 90 degrees (0 returns the image unchanged)."""
    if angle == 0:
        return img
    return cv2.rotate(img, _ROTATIONS[angle])


def _ocr_gray(img_bgr):
    """Grayscale with sensor grain removed.

    The median blur matters more than it looks: grain turns into speckle
    that Tesseract 'reads' as thousands of tiny words, slowly.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.medianBlur(gray, 3)


def clean_ocr_text(text: str) -> str:
    """Strip OCR junk so TTS doesn't read stray symbols."""
    text = re.sub(r"[|{}\[\]<>_~^`]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# What counts as content when scoring a read: a word of 3+ letters or a
# number of 2+ digits. Shorter fragments are what noise mostly looks like.
_CONTENT_RE = re.compile(r"[A-Za-z]{3,}|[0-9]{2,}")
_ONE_LETTER_WORDS = frozenset("aAI")


def _is_speakable(word: str) -> bool:
    """False for tokens that are OCR debris: bare punctuation, or a lone
    letter that is not a word in its own right ("f", "z")."""
    alnum = [ch for ch in word if ch.isalnum()]
    if not alnum:
        return False
    if len(alnum) == 1 and alnum[0].isalpha():
        return alnum[0] in _ONE_LETTER_WORDS
    return True


def _tesseract_words(processed, psm: int) -> list:
    """(word, confidence) for everything Tesseract found, in reading order.

    Returns an empty list if Tesseract times out or fails: a read that
    cannot finish is treated as "nothing here" rather than crashing the
    trigger.
    """
    try:
        data = pytesseract.image_to_data(
            processed,
            config=f"--psm {psm}",
            output_type=pytesseract.Output.DICT,
            timeout=OCR_TIMEOUT_S,
        )
    except RuntimeError as e:   # timeout, or tesseract itself failing
        print(f"[OCR] tesseract gave up (limit {OCR_TIMEOUT_S}s): {e}")
        return []
    found = []
    for word, conf in zip(data["text"], data["conf"]):
        try:
            c = float(conf)
        except (TypeError, ValueError):
            continue
        if c >= 0 and word.strip():     # conf -1 marks layout boxes, not words
            found.append((word, c))
    return found


def score_words(found: list) -> tuple:
    """Turn Tesseract's (word, confidence) pairs into (score, text).

    Words under OCR_MIN_CONF and debris tokens are dropped before they can
    reach TTS, and the whole read is discarded - (0.0, "") - when it has no
    real content or when most of what Tesseract saw had to be dropped.
    Otherwise score = mean confidence x content words x (share kept)^2, so a
    clean read of a sign beats a long stretch of half-recognised noise.
    """
    kept = [(w, c) for w, c in found
            if c >= OCR_MIN_CONF and _is_speakable(w)]
    if not kept:
        return 0.0, ""
    text = clean_ocr_text(" ".join(w for w, _ in kept))
    content = len(_CONTENT_RE.findall(text))
    # Share of the word-like tokens that survived. Bare symbols are left out
    # of the count: a sign's border routinely adds a "|" or "=" or two, and
    # that must not sink a one-word read like "EXIT".
    wordlike = sum(1 for w, _ in found if any(ch.isalnum() for ch in w))
    share_kept = len(kept) / wordlike
    if content == 0 or share_kept < OCR_MIN_KEPT:
        return 0.0, ""
    mean_conf = sum(c for _, c in kept) / len(kept)
    return mean_conf * content * share_kept ** 2, text


def _read_blocks(images: list, deadline: float) -> list:
    """OCR each image in turn; return [(text, boxes Tesseract saw), ...].

    Stops early, leaving the remaining images unread, once `deadline`
    (a time.time() value) has passed.
    """
    reads = []
    for image in images:
        if time.time() > deadline:
            print(f"[OCR] out of time; {len(images) - len(reads)} "
                  f"block(s) left unread")
            break
        found = _tesseract_words(image, OCR_PSM)
        reads.append((score_words(found)[1], len(found)))
    return reads


def run_ocr(frame_bgr) -> str:
    """Read the text in a frame; NO_TEXT_PHRASE if there is none.

    The frame is searched for blocks of text and each is read on its own,
    levelled and at a sensible size (see fypvision.textregions). Giving
    Tesseract the whole scene instead took 20-80 s and returned noise.

    If nothing reads the right way up, the blocks that looked like text are
    read again upside-down, for a page or packet held the wrong way round.
    """
    t0 = time.time()
    deadline = t0 + OCR_BUDGET_S
    blocks = textregions.find_blocks(_ocr_gray(frame_bgr))
    upright = _read_blocks([b.image for b in blocks], deadline)
    text, orientation = " ".join(t for t, _ in upright if t), 0

    if not text:
        # Turning the frame over reverses the reading order of its blocks.
        candidates = [rotate_image(block.image, 180)
                      for block, (_, boxes) in zip(blocks, upright)
                      if boxes >= OCR_RETRY_MIN_BOXES]
        flipped = _read_blocks(candidates[::-1], deadline)
        text, orientation = " ".join(t for t, _ in flipped if t), 180

    print(f"[OCR] {len(blocks)} block(s), {time.time() - t0:.1f}s, "
          f"read at {orientation} deg: {text!r}")
    return text or NO_TEXT_PHRASE


def find_note_region(frame_bgr):
    """Crop of the most banknote-like coloured blob in the frame, or None.

    Notes are printed in saturated colour, while the usual backdrop (white
    walls, black clothing) is not, and skin hues are masked out. Picking the
    largest such blob and cropping to it keeps Tesseract from inventing
    "words" out of the shirt, wall and hand texture around the note.
    """
    scale = NOTE_SEARCH_WIDTH / frame_bgr.shape[1]
    small = cv2.resize(frame_bgr, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    hue, sat, val = cv2.split(
        cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2HSV))
    skin = (hue <= 22) & (sat < 170)
    mask = (((sat > NOTE_MIN_SATURATION) & (val > 60)) & ~skin
            ).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < NOTE_MIN_AREA_FRAC * mask.size:
        return None
    x, y, w, h = cv2.boundingRect(biggest)
    pad = int(0.08 * max(w, h))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(mask.shape[1], x + w + pad), min(mask.shape[0], y + h + pad)
    inv = 1.0 / scale
    return frame_bgr[int(y0 * inv):int(y1 * inv), int(x0 * inv):int(x1 * inv)]


_SPARSE_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _meaningful_tokens(word: str, conf: float) -> list:
    """Keep only tokens a banknote matcher can use, and only if Tesseract is
    sure of them: short numbers (a denomination) and words of 4+ letters
    (issuer / currency names). Numbers need a higher bar because a wrong one
    is spoken as a wrong amount - real numerals read at ~95 here, while
    texture noise and misreads (a "50" read as "20") sit at or below 85."""
    kept = []
    for token in _SPARSE_TOKEN_RE.findall(word):
        if token.isdigit() and len(token) <= 3 and conf >= OCR_DIGIT_MIN_CONF:
            kept.append(token)
        elif token.isalpha() and len(token) >= 4 and conf >= OCR_WORD_MIN_CONF:
            kept.append(token)
    return kept


def _sparse_read(gray, angles) -> list:
    """Sparse-text OCR of `gray` at each angle and scale; returns tokens."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    tokens = []
    for angle in angles:
        rotated = rotate_image(gray, angle)
        for target in OCR_SPARSE_TARGET_SIZES:
            factor = target / max(rotated.shape)
            img = cv2.resize(rotated, None, fx=factor, fy=factor,
                             interpolation=cv2.INTER_AREA if factor < 1
                             else cv2.INTER_CUBIC)
            found = []
            for word, conf in _tesseract_words(clahe.apply(img), psm=11):
                found += _meaningful_tokens(word, conf)
            print(f"[OCR] sparse {angle:>3} deg @{target}px: {found}")
            tokens.extend(found)
    return tokens


def run_ocr_sparse(frame_bgr) -> str:
    """Read scattered text such as a banknote held in view, in any rotation.

    --psm 6 assumes one tidy block of text and returns noise on a busy scene,
    so this uses --psm 11 (sparse text) on a crop of the note where one can
    be found, falling back to the whole frame. The note is read upright and
    upside-down first (people hold notes either way up); the sideways
    rotations are only tried if that yields no number. Each crop is read at
    two sizes because a large numeral reads best small while fine print needs
    the detail. Tokens from every pass are pooled for the caller to
    keyword-match.
    """
    region = find_note_region(frame_bgr)
    source = frame_bgr if region is None else region
    print("[OCR] note region: " + (
        "none, using whole frame" if region is None
        else f"found {source.shape[1]}x{source.shape[0]}"))
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

    tokens = _sparse_read(gray, (0, 180))
    if not any(t.isdigit() for t in tokens):
        tokens += _sparse_read(gray, (90, 270))
    if not tokens and region is not None:
        tokens = _sparse_read(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY),
                              (0, 180))
    return " ".join(tokens) if tokens else NO_TEXT_PHRASE
