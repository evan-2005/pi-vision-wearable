"""Measured distance from a depth image, with synthetic depth arrays."""

import numpy as np
import pytest

from fypvision import depth
from fypvision import distance as dist

W, H = 2304, 1296              # the wearable's picture
DW, DH = depth.WIDTH, depth.HEIGHT


def _flat(mm: int) -> np.ndarray:
    return np.full((DH, DW), mm, dtype=np.uint16)


def _centred_box(width: float, height: float) -> tuple:
    return (W / 2 - width / 2, H / 2 - height / 2,
            W / 2 + width / 2, H / 2 + height / 2)


def test_centred_box_reads_the_depth_behind_it():
    metres = depth.measure(_flat(1500), _centred_box(300, 300), W, H)

    assert metres == pytest.approx(1.5, rel=0.01)


def test_off_centre_box_is_mapped_into_the_depth_image():
    # A near object 300 px right of the picture centre; the wall is at 3 m.
    scale = depth.depth_focal_px(DW) / dist.focal_length_px(W)
    scene = _flat(3000)
    cx = int(DW / 2 + 300 * scale)
    scene[DH // 2 - 30:DH // 2 + 30, cx - 30:cx + 30] = 800
    box = (W / 2 + 200, H / 2 - 100, W / 2 + 400, H / 2 + 100)

    metres = depth.measure(scene, box, W, H)

    assert metres == pytest.approx(0.8, rel=0.06)   # 0.8 m deep, slightly off-axis


def test_holes_in_the_depth_image_give_no_measurement():
    assert depth.measure(_flat(0), _centred_box(300, 300), W, H) is None


def test_box_outside_the_depth_cameras_view_gives_no_measurement():
    # The picture is wider than the depth image; its far edge has no depth.
    assert depth.measure(_flat(1500), (W - 60, 600, W - 10, 700), W, H) is None


def test_measured_depth_beats_the_size_estimate():
    person_box = _centred_box(200, 600)              # size alone says about 2.7 m

    estimate = depth.range_for(_flat(1200), "person", person_box, W, H)

    assert estimate.metres == pytest.approx(1.2, rel=0.01)
    assert not estimate.is_upper_bound


def test_depth_ranges_objects_the_size_estimate_cannot():
    assert dist.estimate_distance("giraffe", _centred_box(300, 300), W, H) is None

    assert depth.range_for(_flat(2000), "giraffe", _centred_box(300, 300), W, H).metres \
        == pytest.approx(2.0, rel=0.01)


@pytest.mark.parametrize("depth_mm", [None, _flat(0)])
def test_falls_back_to_the_size_estimate_without_usable_depth(depth_mm):
    box = _centred_box(200, 600)

    assert depth.range_for(depth_mm, "person", box, W, H) \
        == dist.estimate_distance("person", box, W, H)


def test_grab_returns_none_when_no_depth_camera_is_plugged_in(monkeypatch):
    monkeypatch.setattr(depth, "find_node", lambda: None)

    assert depth.grab() is None


def test_grab_returns_none_when_the_camera_sends_a_short_read(monkeypatch):
    monkeypatch.setattr(depth, "find_node", lambda: "/dev/video8")
    monkeypatch.setattr(depth, "_capture", lambda node: b"\x00" * 100)

    assert depth.grab() is None


def test_grab_takes_the_median_of_the_frames(monkeypatch):
    frames = np.stack([_flat(1000), _flat(1010), _flat(5000)])   # one outlier frame
    monkeypatch.setattr(depth, "find_node", lambda: "/dev/video8")
    monkeypatch.setattr(depth, "_capture", lambda node: frames.astype("<u2").tobytes())

    assert int(depth.grab()[0, 0]) == 1010
