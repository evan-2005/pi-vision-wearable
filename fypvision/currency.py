"""Identify a banknote's denomination by reading it.

YOLO's COCO classes contain no banknotes, and training a detector needs a
labelled dataset. Notes are, however, covered in large high-contrast
numerals and issuer text - exactly what the OCR pipeline already handles.
So this reads the note rather than recognising it as an image.

That choice has real limits, and callers should present results as "looks
like" rather than a certainty:
  - a worn, folded or badly lit note may not read at all
  - it verifies denomination, NOT authenticity; it cannot spot a forgery
  - a photo or drawing of a note reads exactly like a real one

Defaults to Malaysian ringgit. Set FYP_CURRENCY to another configured code.
"""

import os
import re

# Denominations, plus the words that identify the issuer. Requiring an
# issuer word stops random numbers in the scene reading as money.
CURRENCIES = {
    "MYR": {
        "name": "ringgit",
        "denominations": [1, 5, 10, 20, 50, 100],
        "keywords": ["ringgit", "malaysia", "bank negara", "negara"],
        "symbols": ["rm"],
    },
    "SGD": {
        "name": "dollars",
        "denominations": [2, 5, 10, 50, 100],
        "keywords": ["singapore", "monetary authority"],
        "symbols": ["sgd", "$"],
    },
    "USD": {
        "name": "dollars",
        "denominations": [1, 2, 5, 10, 20, 50, 100],
        "keywords": ["united states", "federal reserve", "america"],
        "symbols": ["usd", "$"],
    },
    "EUR": {
        "name": "euro",
        "denominations": [5, 10, 20, 50, 100, 200],
        "keywords": ["euro", " euro ", "bce", "ecb"],
        "symbols": ["eur"],
    },
}

ACTIVE = os.environ.get("FYP_CURRENCY", "MYR").upper()

# OCR routinely confuses these inside numerals; normalise before matching.
_DIGIT_FIXES = {"o": "0", "O": "0", "l": "1", "I": "1", "S": "5", "B": "8"}


def _denomination_candidates(text: str, config) -> set:
    """Numbers in `text` that could plausibly be a printed denomination.

    Digit corrections are applied per token and only kept when the token
    turns out to be entirely numeric. Applying them blindly to the whole
    string rewrites ordinary words - "SERIAL" becomes "5ERIAL" and the note
    reads as a 5.
    """
    values = set()
    alpha_symbols = tuple(s for s in config["symbols"] if s.isalpha())

    for raw in re.split(r"[^0-9A-Za-z$]+", text):
        token = raw.strip("$")
        if not token:
            continue
        # "RM10" / "rm1O": drop the currency prefix before reading the value.
        low = token.lower()
        for sym in alpha_symbols:
            if low.startswith(sym) and len(token) > len(sym):
                token = token[len(sym):]
                break
        if not any(ch.isdigit() for ch in token):
            continue
        fixed = "".join(_DIGIT_FIXES.get(ch, ch) for ch in token)
        # Cap at 3 digits so serial numbers and years cannot masquerade as
        # a denomination, and "1000" never reads as "100".
        if fixed.isdigit() and len(fixed) <= 3:
            values.add(int(fixed))
    return values


def identify(ocr_text: str, currency: str = None) -> str:
    """Describe the note in `ocr_text`, or say it could not be read."""
    config = CURRENCIES.get((currency or ACTIVE).upper())
    if config is None:
        return "Currency recognition is not configured."
    if not ocr_text:
        return "I cannot read the note."

    lowered = ocr_text.lower()
    has_issuer = any(k in lowered for k in config["keywords"])
    has_symbol = any(sym in lowered for sym in config["symbols"])

    found = _denomination_candidates(ocr_text, config)
    candidates = sorted(found & set(config["denominations"]), reverse=True)

    if not candidates:
        if has_issuer or has_symbol:
            return (f"That looks like a {config['name']} note, "
                    f"but I cannot read the value.")
        return "I cannot tell if that is a banknote."

    value = candidates[0]
    phrase = f"{value} {config['name']}"
    if has_issuer or has_symbol:
        return f"This looks like a {phrase} note."
    # A bare number with no issuer text is weak evidence - hedge harder.
    return f"I can see {value}. It may be a {phrase} note, but I am not sure."
