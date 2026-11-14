"""Single / double / long press classification (no GPIO involved)."""

import threading
import time

import pytest

from fypvision.button import PressClassifier

WINDOW_S = 0.08


class Recorder:
    """Collects the gestures a classifier fires, across its threads."""

    def __init__(self):
        self.gestures = []
        self._lock = threading.Lock()
        self.classifier = PressClassifier(
            on_single=lambda: self._add("single"),
            on_double=lambda: self._add("double"),
            on_long=lambda: self._add("long"),
            double_press_s=WINDOW_S,
            on_shutdown=lambda: self._add("shutdown"),
            shutdown_after_held_s=WINDOW_S)

    def _add(self, gesture):
        with self._lock:
            self.gestures.append(gesture)

    def settle(self):
        time.sleep(WINDOW_S * 3)
        return list(self.gestures)


@pytest.fixture
def recorder():
    return Recorder()


def test_one_press_is_a_single_after_the_window(recorder):
    recorder.classifier.released()

    assert recorder.gestures == []          # still might become a double
    assert recorder.settle() == ["single"]


def test_two_quick_presses_are_one_double(recorder):
    recorder.classifier.released()
    recorder.classifier.released()

    assert recorder.settle() == ["double"]


def test_two_slow_presses_are_two_singles(recorder):
    recorder.classifier.released()
    time.sleep(WINDOW_S * 2)
    recorder.classifier.released()

    assert recorder.settle() == ["single", "single"]


def test_hold_is_a_long_press_and_its_release_is_ignored(recorder):
    recorder.classifier.held()
    recorder.classifier.released()

    assert recorder.settle() == ["long"]


def test_holding_on_past_a_long_press_shuts_down(recorder):
    recorder.classifier.held()

    assert recorder.settle() == ["long", "shutdown"]


def test_releasing_a_long_press_in_time_does_not_shut_down(recorder):
    recorder.classifier.held()
    recorder.classifier.released()

    assert "shutdown" not in recorder.settle()


def test_press_then_hold_is_just_a_long_press(recorder):
    recorder.classifier.released()
    recorder.classifier.held()
    recorder.classifier.released()

    assert recorder.settle() == ["long"]


def test_press_after_a_long_press_works_normally(recorder):
    recorder.classifier.held()
    recorder.classifier.released()
    recorder.classifier.released()

    assert recorder.settle() == ["long", "single"]


def test_a_slow_action_does_not_block_the_next_gesture():
    release = threading.Event()
    seen = []
    classifier = PressClassifier(
        on_single=lambda: seen.append("single"),
        on_double=lambda: seen.append("double"),
        on_long=lambda: (release.wait(2), seen.append("long")),
        double_press_s=WINDOW_S)

    classifier.held()
    classifier.released()
    classifier.released()
    time.sleep(WINDOW_S * 3)
    first = list(seen)
    release.set()

    assert first == ["single"]
