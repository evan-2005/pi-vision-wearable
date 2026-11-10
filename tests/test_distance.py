"""Distance estimation from box size."""

import pytest

from fypvision import distance as dist

W, H = 2304, 1296
FOCAL = dist.focal_length_px(W)


def _centred_box(width: float, height: float) -> tuple:
    return (W / 2 - width / 2, H / 2 - height / 2,
            W / 2 + width / 2, H / 2 + height / 2)


def _person_box_height(metres: float) -> float:
    return dist.KNOWN_SIZES_M["person"][0] * FOCAL / metres


def test_default_focal_length_comes_from_the_lens():
    assert FOCAL == pytest.approx(982, abs=2)


@pytest.mark.parametrize("metres", [1.5, 3.0, 6.0])
def test_whole_centred_person_is_ranged_from_height(metres):
    box = _centred_box(200, _person_box_height(metres))

    estimate = dist.estimate_distance("person", box, W, H)

    assert estimate.metres == pytest.approx(metres, rel=0.01)
    assert not estimate.is_upper_bound


def test_unknown_class_has_no_distance():
    assert dist.estimate_distance("giraffe", _centred_box(200, 400), W, H) is None


def test_tiny_box_has_no_distance():
    assert dist.estimate_distance("person", _centred_box(8, 20), W, H) is None


def test_person_cut_off_at_the_bottom_is_ranged_from_width():
    shoulder_px = dist.KNOWN_SIZES_M["person"][1] * FOCAL / 1.0   # 1 m away
    box = (W / 2 - shoulder_px / 2, 300, W / 2 + shoulder_px / 2, H)

    estimate = dist.estimate_distance("person", box, W, H)

    assert not estimate.is_upper_bound
    assert estimate.metres == pytest.approx(1.0, rel=0.1)


def test_object_cut_off_on_every_side_is_only_an_upper_bound():
    estimate = dist.estimate_distance("person", (0, 0, 900, H), W, H)

    assert estimate.is_upper_bound
    assert estimate.metres < 2.0


def test_class_without_a_dependable_width_becomes_a_bound_when_cut():
    estimate = dist.estimate_distance("car", (600, 700, 1500, H), W, H)

    assert estimate.is_upper_bound


def test_off_axis_object_is_farther_than_its_depth():
    height = _person_box_height(3.0)
    centred = dist.estimate_distance("person", _centred_box(200, height), W, H)
    at_edge = dist.estimate_distance(
        "person", (100, H / 2 - height / 2, 300, H / 2 + height / 2), W, H)

    assert at_edge.metres > centred.metres * 1.2


@pytest.mark.parametrize("metres, bound, phrase", [
    (None, False, ""),
    (0.5, False, "less than a metre"),
    (0.5, True, "very close"),
    (1.1, False, "about a metre"),
    (1.4, False, "about 1 and a half metres"),
    (2.1, False, "about 2 metres"),
    (2.6, True, "within about 2 and a half metres"),
    (4.4, False, "about 4 metres"),
    (25.0, False, "more than 10 metres"),
])
def test_describe_distance(metres, bound, phrase):
    assert dist.describe_distance(metres, bound) == phrase


@pytest.mark.parametrize("x, phrase", [
    (100, "to your left"), (1152, "straight ahead"), (2200, "to your right"),
])
def test_describe_direction(x, phrase):
    assert dist.describe_direction(x, W) == phrase
