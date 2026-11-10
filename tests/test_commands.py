"""What a spoken sentence is taken to mean, and how results are phrased."""

import pytest

from fypvision import emotion, finder, voice


@pytest.mark.parametrize("said, intent", [
    ("what is in front of me", "detect"),
    ("what do you see", "detect"),
    ("describe the scene", "detect"),
    ("read this", "read"),
    ("what does it say", "read"),
    ("what note is this", "currency"),
    ("how much money is this", "currency"),
    ("how does she look", "emotion"),
    ("is he happy", "emotion"),
    ("find my phone", "find"),
    ("where's my bottle", "find"),
    ("how far is the chair", "find"),
    ("is anyone there", "find"),
    ("hello how are you today", "unknown"),
    ("", "unknown"),
])
def test_parse_intent(said, intent):
    assert voice.parse_intent(said)[0] == intent


@pytest.mark.parametrize("said, target", [
    ("find my phone", "cell phone"),
    ("where is the water bottle", "bottle"),
    ("how far is the chair", "chair"),
    ("is anyone there", "person"),
    ("look for a cup", "cup"),
    ("find my keys", None),              # not something the model can detect
    ("where is the door", None),
    ("", None),
])
def test_normalise_target(said, target):
    assert finder.normalise_target(said) == target


def test_every_findable_target_is_a_real_detector_class():
    coco = {
        "person", "bicycle", "car", "motorcycle", "bus", "train", "truck",
        "bird", "cat", "dog", "bench", "backpack", "umbrella", "suitcase",
        "bottle", "cup", "chair", "couch", "potted plant", "bed",
        "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "sink",
        "refrigerator", "book", "clock", "scissors",
    }

    assert set(finder._SYNONYMS.values()) <= coco


def _detection(label, x1, x2, metres, is_max=False):
    return {"label": label, "conf": 0.8, "box": (x1, 100, x2, 500),
            "distance_m": metres, "distance_is_max": is_max}


def _find(detections, target):
    from fypvision import distance
    return finder.find_in_detections(
        detections, target, 2304, distance.describe_direction,
        distance.describe_distance)


def test_find_reports_bearing_and_distance():
    phrase = _find([_detection("bottle", 1900, 2100, 1.4)], "bottle")

    assert phrase == "Bottle to your right, about 1 and a half metres."


def test_find_picks_the_nearest_of_several_and_pluralises():
    phrase = _find([_detection("person", 100, 300, 4.0),
                    _detection("person", 1000, 1300, 1.0, is_max=True)],
                   "person")

    assert phrase == ("2 people in view. Nearest is straight ahead, "
                      "within about a metre.")


def test_find_says_so_when_the_object_is_absent():
    assert _find([_detection("cup", 0, 100, 1.0)], "bottle") \
        == "No bottle in view."


def _face(label, conf, size):
    return {"box": (0, 0, size, size), "label": label, "conf": conf,
            "face_score": 0.9}


def test_describe_no_faces():
    assert emotion.describe([]) == "I cannot see anyone's face."


def test_describe_one_face():
    assert emotion.describe([_face("happy", 0.8, 200)]) \
        == "Someone who looks happy."


def test_describe_unsure_and_distant_faces():
    phrase = emotion.describe([_face("sad", 0.2, 300), _face(None, 0.0, 30)])

    assert phrase == ("2 faces: someone whose expression I cannot read "
                      "and someone too far away to read.")


def test_describe_more_faces_than_are_spoken():
    faces = [_face("neutral", 0.9, 100 + i) for i in range(5)]

    assert emotion.describe(faces).startswith("5 faces. The nearest are ")
