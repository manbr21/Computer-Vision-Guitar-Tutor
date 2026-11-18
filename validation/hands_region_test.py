"""
Check of the fretting-hand region (chordar/hands.py: fretting_hand_region): the part of the frame MediaPipe looks at covers the
first frets and the fingers around the neck and leaves the strumming hand, over the body, out.
Run from the repository root:  python validation/hands_region_test.py   Exits with status 1 on any mismatch.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chordar.geometry import Fretboard
from chordar.hands import fretting_hand_region


def main():
    failures = []

    def check(label, ok):
        print(("ok    " if ok else "FAIL  ") + label)
        if not ok:
            failures.append(label)

    # the neck as the mirrored display shows it: nut on the left at x=300, fret 12 at x=900
    neck = Fretboard((300, 300), (300, 380), (900, 390), (900, 300))
    x0, y0, x1, y1 = fretting_hand_region(neck, (1280, 720))
    inside = lambda x, y: x0 <= x <= x1 and y0 <= y <= y1
    check("the first frets are inside", all(inside(*neck.to_image(0.5, t)) for t in (0.0, 0.1, 0.3, 0.5, 0.7)))
    check("the fingers below the neck are inside (two neck widths under it)", inside(*neck.to_image(2.0, 0.15)))
    check("the body end of the neck, where the strumming hand is, is outside", not inside(*neck.to_image(0.5, 1.2)) and not inside(*neck.to_image(0.5, 1.6)))
    cx0, cy0, cx1, cy1 = fretting_hand_region(neck, (1280, 720), 7)
    check("with a capo on fret 7 the region moves along the neck and takes the hand at frets 8-10",
          cx0 > x0 + 100 and cx0 <= neck.to_image(0.5, neck.fret_bounds[7])[0] and cx1 >= neck.to_image(2.0, neck.fret_bounds[10])[0])
    check(f"the region is a fraction of the frame ({(x1 - x0) * (y1 - y0) / (1280 * 720):.0%})", (x1 - x0) * (y1 - y0) < 0.5 * 1280 * 720)

    near_edge = Fretboard((20, 20), (20, 100), (600, 110), (600, 20))
    x0, y0, x1, y1 = fretting_hand_region(near_edge, (1280, 720))
    check("it stays inside the frame near the edges", x0 >= 0 and y0 >= 0 and x1 <= 1280 and y1 <= 720)

    print()
    if failures:
        print("FAIL:", *failures, sep="\n  ")
        sys.exit(1)
    print("PASS: the hand region covers the fretting hand and leaves the strumming hand out")


if __name__ == "__main__":
    main()
