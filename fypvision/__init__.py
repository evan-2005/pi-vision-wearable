"""Feature modules for the assistive vision wearable.

Every module here is self-contained and returns plain data or text: none of
them import main.py, and none of them speak. main.py stays the only place
that drives audio, so these stay unit-testable without a camera or a headset.
"""
