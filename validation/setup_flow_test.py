"""
Check of the photo set-up that replaces the calibration (chordar/reference.py: capture_reference) and of the app that
uses it, with the camera, the window and the mouse replaced by stand-ins and a synthetic neck with fret wires as the photo.
Run from the repository root:  python validation/setup_flow_test.py   Exits with status 1 on any mismatch.
"""
import os
import sys
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import frets_test as ft
import chordar.app as app_module
import chordar.reference as ref_module
from chordar import placement
from chordar.app import ChordARApp
from chordar.markerless import MarklessTracker


def main():
    failures = []

    def check(label, ok):
        print(("ok    " if ok else "FAIL  ") + label)
        if not ok:
            failures.append(label)

    photo = ft.synthetic_neck()                      # a raw frame: the neck with wires, nut on the right
    width = photo.shape[1]
    # the four clicks (on the mirrored view), 14 px off the nut and the 12th fret as a hurried user would click
    off = ft.clicked(-14, 6)
    clicks = [(width - x, y) for x, y in off]
    # a textured scene around the synthetic neck gives the matcher something to hold: add random blobs outside it
    rng = np.random.default_rng(5)
    for _ in range(250):
        cv2.circle(photo, (int(rng.integers(0, width)), int(rng.integers(0, 330))), int(rng.integers(3, 12)),
                   tuple(int(c) for c in rng.integers(30, 230, 3)), -1)

    class Camera:
        def read(self):
            return True, photo.copy()

        def release(self):
            pass

    snapped_bounds = None
    with mock.patch.object(ref_module, "take_photo", return_value=photo), \
            mock.patch.object(ref_module, "click_corners", side_effect=lambda frame: ref_module_click(frame, clicks)), \
            mock.patch.object(ref_module.cv2, "namedWindow"), mock.patch.object(ref_module.cv2, "destroyWindow"), \
            mock.patch.object(ref_module.cv2, "waitKey", return_value=255):
        tracker = ref_module.capture_reference(Camera())
    check("the set-up returns a marker-free tracker", isinstance(tracker, MarklessTracker))
    corners = tracker.reference.corners
    expected_raw = np.array(ft.TRUE, np.float32)
    check(f"the corners were snapped onto the wires (max {np.abs(corners - expected_raw).max():.1f} px from the true ones, "
          f"the clicks were 14 px off)", np.abs(corners - expected_raw).max() < 3.0)
    check("the measured fret spacing is kept in the reference", tracker.reference.fret_bounds is not None)
    mid = np.array([(width - x, y) for x, y in ft.TRUE]).mean(axis=0)
    expected_center = (mid[0] / width, mid[1] / photo.shape[0])
    check(f"the playing monitor now expects the neck where the photo was taken (center {placement.IDEAL_CENTER[0]:.3f}, "
          f"{placement.IDEAL_CENTER[1]:.3f} vs {expected_center[0]:.3f}, {expected_center[1]:.3f})",
          abs(placement.IDEAL_CENTER[0] - expected_center[0]) < 0.01 and abs(placement.IDEAL_CENTER[1] - expected_center[1]) < 0.01)

    # the app: no photo given -> the set-up runs; ESC at the photo -> the app quits cleanly
    with mock.patch.object(app_module, "capture_reference", return_value=None), mock.patch.object(app_module, "open_window"), \
            mock.patch.object(app_module.cv2, "destroyAllWindows"):
        app = ChordARApp(Camera(), markerless=True)
        check("a marker-free app has no tracker until the photo is taken", app.markers is None)
        app.run()
        check("cancelling the photo ends the app without a tracker", app.markers is None)
    with mock.patch.object(app_module, "capture_reference", return_value=tracker):
        app = ChordARApp(Camera(), markerless=True)
        check("the photo set-up gives the app its tracker", app._set_up() and app.markers is tracker)
        app.hands.close()

    print()
    if failures:
        print("FAIL:", *failures, sep="\n  ")
        sys.exit(1)
    print("PASS: the photo set-up replaces the calibration")


def ref_module_click(frame, clicks):
    """Stand-in for the click screen: the user's four clicks, snapped to the wires like the real screen does."""
    view = cv2.flip(frame, 1)
    snapped, info = ref_module.snap_corners(view, np.array(clicks, np.float32))
    return [tuple(map(float, c)) for c in snapped], (info["fret_bounds"] if info else None)


if __name__ == "__main__":
    main()
