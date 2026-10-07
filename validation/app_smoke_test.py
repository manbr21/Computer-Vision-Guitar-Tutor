"""
End-to-end smoke test of the chordar package without a camera.

It draws four real ArUco markers on a synthetic frame around a neck whose corners are known, then runs the
detection, the per-frame processing, one calibration step and the key handling, checking that the detected
neck sits where it was drawn and that nothing raises.

Run from the repository root:  python validation/app_smoke_test.py
Exits with status 1 on any mismatch.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chordar import config
from chordar.app import ChordARApp
from chordar.calibration import calibration_step

W, H, MARKER, MARGIN = 1280, 720, 100, 20

# The neck as the display shows it (mirrored): nut on the left. ID -> (neck corner, marker corner index)
# chosen so that each marker, drawn unrotated, touches its neck corner with the corner facing the neck.
NECK_IN_DISPLAY = {"TL": (300, 300), "TR": (300, 420), "BR": (1000, 430), "BL": (1000, 300)}
MARKER_CORNERS = {0: ("TL", 3), 1: ("TR", 0), 2: ("BR", 1), 3: ("BL", 2)}
# where each marker's reference corner lands in the camera frame (x mirrored), and which way the marker extends
PLACEMENT = {0: ((W - 300, 300), (0, -1)), 1: ((W - 300, 420), (0, 0)),
             2: ((W - 1000, 430), (-1, 0)), 3: ((W - 1000, 300), (-1, -1))}


def synthetic_frame():
    """A camera frame with the four markers drawn (unmirrored, as the camera delivers it)."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    frame = np.full((H, W, 3), 200, np.uint8)
    for marker_id, ((cx, cy), (dx, dy)) in PLACEMENT.items():
        x0, y0 = cx + dx * MARKER, cy + dy * MARKER  # top-left of the marker
        image = cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER)
        cv2.rectangle(frame, (x0 - MARGIN, y0 - MARGIN), (x0 + MARKER + MARGIN, y0 + MARKER + MARGIN),
                      (255, 255, 255), -1)
        frame[y0:y0 + MARKER, x0:x0 + MARKER] = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return frame


class FakeCapture:
    def release(self):
        pass


def main():
    failures = []

    def check(label, ok):
        print(("ok    " if ok else "FAIL  ") + label)
        if not ok:
            failures.append(label)

    config.NECK_CORNER.clear()
    config.NECK_CORNER.update({marker_id: value for marker_id, value in MARKER_CORNERS.items()})

    frame = synthetic_frame()
    app = ChordARApp(FakeCapture())

    display, neck = app.markers.track(frame)
    check("the four markers are detected and a neck is built", neck is not None)
    if neck is not None:
        error = max(np.hypot(*(np.array(neck.corners[i]) - NECK_IN_DISPLAY[name]))
                    for i, name in enumerate(("TL", "TR", "BR", "BL")))
        check(f"detected neck corners are within 3 px of the drawn ones (max error {error:.1f} px)", error < 3)

    for _ in range(3):  # several frames, as in the live loop (smoothing, pose, monitor)
        shown = app.process_frame(frame, now=0.0)
    check("process_frame returns an image of the frame size", shown.shape == frame.shape)
    check("the overlay changed the image", not np.array_equal(shown, cv2.flip(frame, 1)))

    screen, ready = calibration_step(frame, app.markers, app.hands, {})
    check("calibration_step returns an image and is not ready without a hand", screen.shape == frame.shape and not ready)

    check("a chord key selects the chord", app.handle_key(ord("8")) and app.chord == "G")
    app.handle_key(ord("p"))
    app.handle_key(ord("x"))
    check("P toggles the parallax fix and X swaps the pose", not app.parallax_on and app.pose_variant == 1)
    check("ESC asks to quit", app.handle_key(27) is False)
    app.process_frame(frame, now=0.0)  # still works with the alternative settings and chord

    blank = np.full((H, W, 3), 200, np.uint8)
    shown = app.process_frame(blank, now=0.0)
    check("a frame without markers is handled (no neck, no crash)", shown.shape == blank.shape)

    app.hands.close()
    print()
    if failures:
        print("FAIL:", *failures, sep="\n  ")
        sys.exit(1)
    print("PASS: the package runs end to end on a synthetic frame")


if __name__ == "__main__":
    main()
