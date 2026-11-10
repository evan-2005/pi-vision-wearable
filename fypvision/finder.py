"""Locate one named object and say where it is.

Detection mode answers "what is around me". This answers "where is my X" -
the wearer names a thing, and the system reports bearing and rough distance
for the single best match rather than listing the whole scene.

The target usually arrives from speech recognition, so it is messy: "my
phone", "the bottle", "a chair". normalise_target() maps that onto a COCO
class the model can actually detect, and returns None when it cannot - so
the caller can say "I cannot look for that" instead of silently finding
nothing.
"""

# Words a wearer is likely to say, mapped to the COCO class YOLO knows.
# Only classes the model can detect belong here; anything else would make
# the device promise a search it cannot perform.
_SYNONYMS = {
    "phone": "cell phone", "mobile": "cell phone", "smartphone": "cell phone",
    "handphone": "cell phone", "telephone": "cell phone",
    "bottle": "bottle", "water": "bottle", "flask": "bottle",
    "cup": "cup", "mug": "cup", "glass": "cup",
    "chair": "chair", "seat": "chair", "stool": "chair",
    "table": "dining table", "desk": "dining table",
    "sofa": "couch", "settee": "couch", "couch": "couch",
    "bag": "backpack", "backpack": "backpack", "rucksack": "backpack",
    "luggage": "suitcase", "suitcase": "suitcase",
    "laptop": "laptop", "computer": "laptop", "notebook": "laptop",
    "keyboard": "keyboard", "mouse": "mouse", "remote": "remote",
    "tv": "tv", "television": "tv", "monitor": "tv", "screen": "tv",
    "book": "book", "clock": "clock", "scissors": "scissors",
    "door": "door", "plant": "potted plant",
    "person": "person", "people": "person", "someone": "person",
    "somebody": "person", "man": "person", "woman": "person",
    "dog": "dog", "cat": "cat", "bird": "bird",
    "car": "car", "bus": "bus", "bicycle": "bicycle", "bike": "bicycle",
    "motorcycle": "motorcycle", "truck": "truck", "train": "train",
    "bench": "bench", "bed": "bed", "toilet": "toilet", "sink": "sink",
    "fridge": "refrigerator", "refrigerator": "refrigerator",
    "microwave": "microwave", "oven": "oven", "umbrella": "umbrella",
    "keys": "keys", "wallet": "wallet",
}

# Filler words to strip before matching, so "where is the bottle" works.
_STOPWORDS = {"a", "an", "the", "my", "your", "our", "is", "are", "where",
              "find", "locate", "look", "for", "me", "please", "can", "you",
              "s", "of", "some", "any", "there"}


def normalise_target(spoken: str):
    """Map a spoken phrase onto a detectable class name, or None."""
    if not spoken:
        return None
    words = [w.strip(".,?!'\"").lower() for w in spoken.split()]
    words = [w for w in words if w and w not in _STOPWORDS]

    # Try longest phrases first so "cell phone" beats "phone".
    for size in (2, 1):
        for i in range(len(words) - size + 1):
            key = " ".join(words[i:i + size])
            if key in _SYNONYMS:
                return _SYNONYMS[key]
    return None


def find_in_detections(detections, target: str, frame_width: int,
                       describe_direction, describe_distance) -> str:
    """Phrase describing where `target` is, or that it was not found.

    The two describe_* callables are injected rather than imported so this
    module stays independent of the distance model.
    """
    if not target:
        return "I did not catch what to look for."

    matches = [d for d in (detections or []) if d["label"] == target]
    if not matches:
        return f"No {target} in view."

    # The nearest instance is the useful one to a wearer; fall back to the
    # most confident when no distance could be estimated.
    def proximity(det):
        metres = det.get("distance_m")
        return (0, metres) if metres is not None else (1, -det.get("conf", 0.0))

    best = min(matches, key=proximity)
    x1, _, x2, _ = best["box"]
    bearing = describe_direction((x1 + x2) / 2.0, frame_width)
    distance = describe_distance(best.get("distance_m"))

    where = bearing + (f", {distance}" if distance else "")
    if len(matches) > 1:
        return f"{len(matches)} {target}s in view. Nearest is {where}."
    return f"{target.capitalize()} {where}."
