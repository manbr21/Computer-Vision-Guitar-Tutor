"""
Check of the fret-wire snapping (chordar/frets.py): clicks that are a few pixels off the nut and the 12th fret are
moved onto the real wires. A synthetic tapered neck with wires at the equal-temperament positions stands in for a photo.
Run from the repository root:  python validation/frets_test.py   Exits with status 1 on any mismatch.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chordar.frets import snap_corners
from chordar.geometry import FRET_BOUNDS, Fretboard

W, H = 1280, 720
TRUE = np.array([(1100, 400), (1100, 482), (500, 500), (500, 395)], np.float32)  # TL TR BR BL: tapered, slightly tilted


def foreshortened(b, k):
    """Where a wire at the ideal position b (0 = nut, 1 = fret 12) appears when the neck is foreshortened by k."""
    return b * (1 + k) / (1 + k * b)


def synthetic_neck(seed=0, hand=False, k=0.0):
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 90, np.uint8)
    fb = Fretboard(*[tuple(map(float, c)) for c in TRUE])
    board = np.array([fb.to_image(-0.08, -0.12), fb.to_image(1.08, -0.12), fb.to_image(1.08, 1.4), fb.to_image(-0.08, 1.4)], np.int32)
    cv2.fillConvexPoly(img, board, (35, 30, 28))                       # dark fretboard (beyond fret 12 too)
    cv2.line(img, fb.to_image(-0.08, -0.02), fb.to_image(1.08, -0.02), (230, 230, 230), 8)   # the nut bar, just before t = 0
    for n in range(1, 16):                                              # fret wires (also past 12)
        t = foreshortened(2 * (1 - 2 ** (-n / 12)), k)
        cv2.line(img, fb.to_image(-0.05, t), fb.to_image(1.05, t), (200, 200, 205), 2)
    for s in range(6):
        cv2.line(img, fb.to_image(s / 5, -0.1), fb.to_image(s / 5, 1.5), (190, 190, 190), 1)
    if hand:
        cv2.ellipse(img, fb.to_image(0.6, 0.45), (28, 60), 20, 0, 360, (110, 150, 205), -1)
    return cv2.add(img, rng.normal(0, 4, img.shape).clip(0, 255).astype(np.uint8))


def clicked(shift_nut_px, shift_far_px):
    """The true corners with the nut and the far end moved along the neck, as imprecise clicks would."""
    tl, tr, br, bl = TRUE
    axis = ((bl - tl) + (br - tr)) / 2
    axis = axis / np.linalg.norm(axis)
    return np.array([tl + axis * shift_nut_px, tr + axis * shift_nut_px, br + axis * shift_far_px, bl + axis * shift_far_px], np.float32)


def main():
    failures = []

    def check(label, ok):
        print(("ok    " if ok else "FAIL  ") + label)
        if not ok:
            failures.append(label)

    img = synthetic_neck()
    for label, nut, far in (("clicks exactly on the nut and fret 12", 0, 0), ("nut clicked 18 px too early", -18, 0),
                            ("nut 12 px late and fret 12 8 px early", 12, 8), ("both ends 15 px off", -15, -15)):
        start = clicked(nut, far)
        snapped, info = snap_corners(img, start)
        error = float(np.abs(snapped - TRUE).max())
        check(f"{label}: snapped to within 3 px of the true corners (max {error:.1f} px, was {np.abs(start - TRUE).max():.0f})",
              info is not None and error < 3.0)

    snapped, info = snap_corners(synthetic_neck(hand=True), clicked(-14, 6))
    check(f"a hand over part of the neck does not break the fit (max {np.abs(snapped - TRUE).max():.1f} px)",
          info is not None and np.abs(snapped - TRUE).max() < 4.0)

    for k in (0.08, -0.08):
        snapped, info = snap_corners(synthetic_neck(k=k), clicked(-12, 5))
        expected = np.array([foreshortened(b, k) for b in FRET_BOUNDS])
        got = np.array(info["fret_bounds"])
        length = float(np.linalg.norm(TRUE[0] - TRUE[3]))
        check(f"a neck foreshortened by k={k:+.2f}: the measured wire positions are within 2 px of the truth "
              f"(max {np.abs(got - expected).max() * length:.1f} px; the ideal formula would be off by "
              f"{np.abs(np.array(FRET_BOUNDS) - expected).max() * length:.0f} px)",
              np.abs(got - expected).max() * length < 2.0 and np.abs(snapped - TRUE).max() < 3.0)

    blank = np.full((H, W, 3), 90, np.uint8)
    snapped, info = snap_corners(blank, clicked(-10, 0))
    check("a photo with no wires leaves the clicks untouched", info is None and np.allclose(snapped, clicked(-10, 0)))

    print()
    if failures:
        print("FAIL:", *failures, sep="\n  ")
        sys.exit(1)
    print("PASS: the clicked corners are snapped onto the fret wires")


if __name__ == "__main__":
    main()
