"""
Check of the lighting advice in the calibration (chordar/placement.py: lighting_check).
Run from the repository root:  python validation/lighting_test.py   Exits with status 1 on any mismatch.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import app_smoke_test as smoke
from chordar import config
from chordar.calibration import calibration_step
from chordar.hands import FingertipTracker
from chordar.markers import MarkerTracker
from chordar.placement import lighting_check


def main():
    failures = []

    def check(label, ok):
        print(("ok    " if ok else "FAIL  ") + label)
        if not ok:
            failures.append(label)

    check("a normal room is fine", lighting_check(120, 115)["ok"])
    check("a dark room is flagged", not lighting_check(40)["ok"] and "dark" in lighting_check(40)["hint"])
    check("a washed-out picture is flagged", not lighting_check(230)["ok"])
    check("a neck much darker than the room (window behind) is flagged as backlit",
          "Backlit" in lighting_check(140, 60)["hint"])
    check("an unknown neck does not trigger the backlit advice", lighting_check(140, None)["ok"])

    config.NECK_CORNER.clear()
    config.NECK_CORNER.update(smoke.MARKER_CORNERS)
    markers, hands = MarkerTracker(), FingertipTracker()
    frame = smoke.synthetic_frame()
    dark = (frame * 0.15).astype(np.uint8)
    for _ in range(3):
        image, _ = calibration_step(frame, markers, hands, {})
        image_dark, _ = calibration_step(dark, markers, hands, {})
    check("the calibration screen draws with a normal and with a dark frame", image.shape == frame.shape == image_dark.shape)
    hands.close()

    print()
    if failures:
        print("FAIL:", *failures, sep="\n  ")
        sys.exit(1)
    print("PASS: lighting advice matches dark, bright and backlit pictures")


if __name__ == "__main__":
    main()
