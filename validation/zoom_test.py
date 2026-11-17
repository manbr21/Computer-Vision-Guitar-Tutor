"""
Check of the zoomed view (chordar/zoom.py, key Z): same size as the frame, the first frets and the hand below the neck inside it,
the picture enlarged, and the tracker's limit on how far the live fret fit may take the nut.
Run from the repository root:  python validation/zoom_test.py   Exits with status 1 on any mismatch.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chordar import markerless
from chordar.geometry import Fretboard
from chordar.zoom import NeckZoom


def main():
    failures = []

    def check(label, ok):
        print(("ok    " if ok else "FAIL  ") + label)
        if not ok:
            failures.append(label)

    neck = Fretboard((300, 300), (300, 380), (900, 390), (900, 300))
    zoom = NeckZoom()
    x0, y0, x1, y1 = zoom.box(neck, (1280, 720))
    check("the crop has the proportions of the frame", abs((x1 - x0) / (y1 - y0) - 1280 / 720) < 1e-6)
    check("the crop is inside the frame", x0 >= 0 and y0 >= 0 and x1 <= 1280 and y1 <= 720)
    check("the nut, fret 7 and the fingers below the neck are inside it",
          all(x0 <= x <= x1 and y0 <= y <= y1 for x, y in (neck.to_image(0.5, 0.0), neck.to_image(0.5, 0.6), neck.to_image(2.0, 0.2))))
    check("the picture is enlarged at least 1.5x", (1280 / (x1 - x0)) >= 1.5)
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[300:320, 300:900] = 255
    out = zoom.apply(frame, neck)
    check("the zoomed image has the size of the frame", out.shape == frame.shape)
    check("the marked strip is thicker in it", int((out[:, 640, 0] > 128).sum()) > 20 * 1.5)
    check("a neck at the border of the frame does not push the crop out of it",
          all(v >= 0 for v in NeckZoom().box(Fretboard((20, 20), (20, 100), (620, 110), (620, 20)), (1280, 720))[:2]))
    check("the live fret fit may take the nut only a few percent from the followed corners", 0 < markerless.FRET_MAX_NUT_SHIFT <= 0.03)
    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("\nPASS: the zoomed view keeps the neck and the hand and the fret fit stays near the nut")


if __name__ == "__main__":
    main()
