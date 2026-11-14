"""Which ALSA device speech goes to, asked afresh every time.

On the wearable the Pi boots first and the headset is switched on later, and
a headset can drop and come back during the day. So the output device is
never cached: each utterance looks up whichever headset is connected now.
"""

import subprocess
from typing import Optional

# Values of FYP_AUDIO_DEVICE that mean "the Pi's own 3.5mm/HDMI output".
_ONBOARD = ("default", "none", "")
ONBOARD_DEVICE = ""


def detect_headset_mac() -> Optional[str]:
    """MAC of a currently-connected Bluetooth device, or None.

    On a wearable the only paired device is the headset, so "connected"
    means it. This is what lets the wearer swap between paired headsets
    without touching any configuration.
    """
    try:
        result = subprocess.run(
            ["bluetoothctl", "devices", "Connected"],
            capture_output=True, timeout=10, text=True,
        )
    except (OSError, subprocess.SubprocessError) as e:
        print(f"[AUDIO] bluetoothctl failed: {e}")
        return None
    # Lines look like: "Device 98:80:BB:80:4E:CA SOUNDPEATS RunFree Lite2"
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "Device":
            return parts[1]
    return None


def bluealsa_device(mac: str) -> str:
    """ALSA device string for playing to a headset over BlueALSA A2DP.

    The inner PCM name is single-quoted because ALSA splits a device string
    on commas: bare `plug:bluealsa:DEV=...,PROFILE=a2dp` is read as two
    arguments to `plug` and fails with "Unknown parameter". Quoted, it opens
    fine, and `plug` absorbs any residual rate/format conversion.
    """
    return f"plug:'bluealsa:DEV={mac},PROFILE=a2dp'"


def output_device(override: Optional[str] = None,
                  fallback_mac: str = "") -> Optional[str]:
    """The device to play through right now.

    Returns ONBOARD_DEVICE ("") for the Pi's own output, a BlueALSA device
    string for a headset, or None when there is nowhere to play.

    1. `override` (FYP_AUDIO_DEVICE): "default"/"none"/"" -> onboard output,
       anything else is used as the device string as-is.
    2. Whichever Bluetooth headset is connected at this moment.
    3. `fallback_mac` (FYP_HEADSET_MAC), if one was configured.
    """
    if override is not None:
        return ONBOARD_DEVICE if override.lower() in _ONBOARD else override
    mac = detect_headset_mac() or fallback_mac
    return bluealsa_device(mac) if mac else None
