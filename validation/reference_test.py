"""
Check of the marker-free tracker and its reference (chordar/markerless.py, chordar/reference.py) on a synthetic,
textured "guitar": no camera and no personal photo needed.
Run from the repository root:  python validation/reference_test.py   Exits with status 1 on any mismatch.
"""
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chordar.markerless import MarklessTracker, NeckReference
from chordar.reference import reference_from_clicks

W, H = 1280, 720
TRUE_DISPLAY = [(300, 300), (300, 420), (1000, 430), (1000, 300)]  # the neck as the mirrored view shows it


def textured_scene(seed=3):
    """Gray scene with random blobs and lines (something for the features to hold on to)."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 120, np.uint8)
    for _ in range(400):
        center = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        cv2.circle(img, center, int(rng.integers(3, 14)), tuple(int(c) for c in rng.integers(20, 235, 3)), -1)
    for _ in range(150):
        a = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        b = (a[0] + int(rng.integers(-60, 60)), a[1] + int(rng.integers(-60, 60)))
        cv2.line(img, a, b, tuple(int(c) for c in rng.integers(20, 235, 3)), 2)
    return cv2.GaussianBlur(img, (3, 3), 0)


def main():
    failures = []

    def check(label, ok):
        print(("ok    " if ok else "FAIL  ") + label)
        if not ok:
            failures.append(label)

    scene = textured_scene()
    reference = reference_from_clicks(scene, TRUE_DISPLAY)
    expected_raw = np.array([(W - x, y) for x, y in TRUE_DISPLAY], np.float32)
    check("clicks on the mirrored view become raw camera corners", np.allclose(reference.corners, expected_raw))

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "ref.npz")
        reference.save(path)
        loaded = NeckReference.load(path)
        check("a saved reference loads back", np.allclose(loaded.corners, reference.corners) and loaded.image.shape == scene.shape)

    tracker = MarklessTracker(reference)
    found, inliers = tracker.estimate(scene)
    check(f"the same view is found with no shift ({inliers} inliers, error {np.abs(found - expected_raw).max():.2f} px)",
          found is not None and np.abs(found - expected_raw).max() < 1.5)

    rng = np.random.default_rng(1)
    worst, lost = 0.0, 0
    for _ in range(30):
        angle, scale = np.radians(rng.uniform(-10, 10)), rng.uniform(0.9, 1.1)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), np.degrees(angle), scale)
        M[:, 2] += rng.uniform(-40, 40, 2)
        frame = cv2.warpAffine(scene, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
        frame = np.clip(frame.astype(np.float32) * rng.uniform(0.7, 1.2) + rng.normal(0, 4, frame.shape), 0, 255).astype(np.uint8)
        truth = cv2.transform(expected_raw.reshape(-1, 1, 2), M).reshape(4, 2)
        got, _ = tracker.estimate(frame)
        if got is None:
            lost += 1
        else:
            worst = max(worst, float(np.abs(got - truth).max()))
    check(f"a moved, rotated, rescaled and noisy view is followed (lost {lost}/30, worst error {worst:.1f} px)", lost == 0 and worst < 4)

    blank = np.full((H, W, 3), 120, np.uint8)
    got, _ = tracker.estimate(blank)
    check("a frame without the neck gives no corners", got is None)

    display, neck = tracker.track(scene)
    check("track() returns the mirrored frame and a Fretboard whose corners are the display ones",
          neck is not None and np.allclose(np.array(neck.corners), np.array(TRUE_DISPLAY, float), atol=1.5)
          and np.array_equal(display, cv2.flip(scene, 1)))
    for _ in range(12):
        _, neck = tracker.track(blank)
    check("a lost neck is held for a few frames, then dropped", neck is None)

    try:
        MarklessTracker(reference_from_clicks(blank, TRUE_DISPLAY))
        check("a photo without features is refused", False)
    except ValueError:
        check("a photo without features is refused", True)

    print()
    if failures:
        print("FAIL:", *failures, sep="\n  ")
        sys.exit(1)
    print("PASS: the marker-free tracker follows a synthetic neck and its reference round-trips")


if __name__ == "__main__":
    main()
