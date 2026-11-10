"""Turn one push button into three gestures: single, double and long press.

The wearable has a single button, so the gesture is the whole interface:

    single press   the everyday action (describe the scene)
    double press   a second action that must work without speech (read text)
    long press     open the microphone for a spoken command

A single press cannot be acted on the instant the button comes up, because
it may be the first half of a double press; it fires once DOUBLE_PRESS_S has
passed with no second press. That delay is the price of the third gesture.

PressClassifier holds the timing logic and knows nothing about GPIO, so it
can be tested by calling held() and released() directly. watch() wires it to
a real pin.
"""

import threading
import time
from typing import Callable

DOUBLE_PRESS_S = 0.40     # max gap between the two presses of a double press
HOLD_TIME_S = 0.80        # holding this long is a long press
BOUNCE_TIME_S = 0.03      # contact chatter shorter than this is ignored
_PROBE_SETTLE_S = 0.05


class PressClassifier:
    """Calls one of three callbacks per gesture, each on its own thread.

    The callbacks may take seconds (they capture, think and speak), so they
    never run on the caller's thread: the GPIO library must stay free to
    notice the next press.
    """

    def __init__(self, on_single: Callable[[], None],
                 on_double: Callable[[], None],
                 on_long: Callable[[], None],
                 double_press_s: float = DOUBLE_PRESS_S):
        self._actions = {"single": on_single, "double": on_double,
                         "long": on_long}
        self._double_press_s = double_press_s
        self._lock = threading.Lock()
        self._pending = None          # timer waiting to confirm a single press
        self._long_in_progress = False

    def _fire(self, gesture: str) -> None:
        threading.Thread(target=self._actions[gesture], daemon=True,
                         name=f"button-{gesture}").start()

    def _single_confirmed(self) -> None:
        with self._lock:
            self._pending = None
        self._fire("single")

    def held(self) -> None:
        """The button has been down for the hold time."""
        with self._lock:
            self._long_in_progress = True
            if self._pending is not None:      # press-then-hold: just a hold
                self._pending.cancel()
                self._pending = None
        self._fire("long")

    def released(self) -> None:
        """The button has come up."""
        with self._lock:
            if self._long_in_progress:
                self._long_in_progress = False     # end of a long press
                return
            if self._pending is not None:
                self._pending.cancel()
                self._pending = None
                gesture = "double"
            else:
                self._pending = threading.Timer(self._double_press_s,
                                                self._single_confirmed)
                self._pending.daemon = True
                self._pending.start()
                return
        self._fire(gesture)


def _idle_level_is_high(pin: int) -> bool:
    """Read the pin with the Pi's weak pull-up on while nobody is pressing.

    A bare switch to ground, or a module that pulls its output up, idles
    high. A module that idles low holds the line down against the pull-up
    (its own resistor is much stronger), so it reads low.
    """
    from gpiozero import InputDevice
    probe = InputDevice(pin, pull_up=True)
    try:
        time.sleep(_PROBE_SETTLE_S)
        return bool(probe.pin.state)
    finally:
        probe.close()


def watch(pin: int, classifier: PressClassifier, active: str = "auto"):
    """Attach `classifier` to a GPIO pin. Returns the gpiozero Button, which
    the caller must keep a reference to.

    `active` is the level the pin goes to when pressed: "low", "high", or
    "auto" to work it out from the idle level at start-up (so the button
    must not be held while the program starts).
    """
    from gpiozero import Button
    if active == "auto":
        active = "low" if _idle_level_is_high(pin) else "high"
    button = Button(pin, pull_up=(active == "low"), hold_time=HOLD_TIME_S,
                    bounce_time=BOUNCE_TIME_S)
    button.when_held = classifier.held
    button.when_released = classifier.released
    print(f"[GPIO] button on GPIO{pin}, pressed = {active}")
    return button
