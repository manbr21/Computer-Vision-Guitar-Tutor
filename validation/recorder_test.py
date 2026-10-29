"""
Check of the session recorder (chordar/recorder.py): the video and the ground-truth corners it writes.
Run from the repository root:  python validation/recorder_test.py   Exits with status 1 on any mismatch.
"""
import csv
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app_smoke_test as smoke
from chordar import config
from chordar.markers import MarkerTracker
from chordar.recorder import CORNER_NAMES, SessionRecorder


def main():
    failures = []

    def check(label, ok):
        print(("ok    " if ok else "FAIL  ") + label)
        if not ok:
            failures.append(label)

    config.NECK_CORNER.clear()
    config.NECK_CORNER.update(smoke.MARKER_CORNERS)
    tracker = MarkerTracker()
    frame = smoke.synthetic_frame()
    blank = np.full_like(frame, 200)

    with tempfile.TemporaryDirectory() as folder:
        recorder = SessionRecorder(folder)
        for i in range(4):
            tracker.track(frame)
            recorder.write(frame, tracker, now=10.0 + i / 30)
        tracker.track(blank)
        recorder.write(blank, tracker, now=10.2)
        recorder.close()

        rows = list(csv.DictReader(open(os.path.join(recorder.path, "corners.csv"))))
        check("one row per frame", len(rows) == 5)
        check("all four markers seen: the row is complete, and not for the blank frame",
              [r["all_markers"] for r in rows] == ["1", "1", "1", "1", "0"])
        # the neck as drawn on the mirrored display, converted back to raw camera pixels
        errors = []
        for name in CORNER_NAMES:
            x, y = smoke.NECK_IN_DISPLAY[name]
            errors.append(np.hypot(float(rows[0][name + "x"]) - (smoke.W - x), float(rows[0][name + "y"]) - y))
        check(f"the recorded corners are in raw camera pixels, within 3 px of the drawn ones (max {max(errors):.1f} px)",
              max(errors) < 3)
        check("an incomplete frame has empty corner cells", rows[4]["TLx"] == "")
        capture = cv2.VideoCapture(os.path.join(recorder.path, "video.mp4"))
        ok, first = capture.read()
        count = 1 + sum(1 for _ in iter(lambda: capture.read()[0], False))
        capture.release()
        check(f"the video has the same frames, un-mirrored and without overlay ({count} frames)",
              ok and count == 5 and first.shape == frame.shape and float(np.abs(first.astype(int) - frame).mean()) < 6)

    print()
    if failures:
        print("FAIL:", *failures, sep="\n  ")
        sys.exit(1)
    print("PASS: the recorder writes the raw video and the ground-truth corners")


if __name__ == "__main__":
    main()
