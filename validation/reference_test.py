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
from chordar.calibration import ghost_point
from chordar.placement import IDEAL_CENTER
from chordar.reference import GUIDE_ORIGIN, GUIDE_SIZE, draw_click_guide, photo_hint, reference_from_clicks

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

    # a slow sequence with sensor noise: followed from frame to frame, the corners must be steady and right
    sequence = MarklessTracker(reference)
    errors, steps, previous = [], [], None
    for i in range(40):
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 3 * np.sin(i / 7), 1 + 0.03 * np.sin(i / 9))
        M[:, 2] += (12 * np.sin(i / 5), 8 * np.cos(i / 6))
        frame = cv2.warpAffine(scene, M, (W, H), borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        frame = np.clip(frame + np.random.default_rng(i).normal(0, 6, frame.shape), 0, 255).astype(np.uint8)
        truth = cv2.transform(expected_raw.reshape(-1, 1, 2), M).reshape(4, 2)
        got = sequence.update(frame)
        if got is None:
            errors.append(99.0)
            continue
        errors.append(float(np.mean(np.linalg.norm(got - truth, axis=1))))
        if previous is not None:
            steps.append(float(np.mean(np.linalg.norm((got - previous[0]) - (truth - previous[1]), axis=1))))
        previous = (got, truth)
    check(f"a noisy slow sequence is tracked (median error {np.median(errors):.1f} px, worst {max(errors):.1f} px)",
          np.median(errors) < 3 and max(errors) < 8)
    check(f"... and the corners are steady: the step error is {np.median(steps):.2f} px per frame", np.median(steps) < 1.0)

    # shape check: a stretched or sheared neck is a bad homography and must be refused
    ref = np.array(expected_raw, np.float32)
    tl, tr, br, bl = ref  # raw corners (mirrored layout: the nut is on the right of the raw frame)

    def stretched(k):  # move the fret-12 corners away from the nut by factor k along the neck
        return np.array([tl, tr, tl + (bl - tl) * k, tr + (br - tr) * k], np.float32)[[0, 1, 3, 2]]

    work = lambda q: q * tracker._scale  # _plausible works at the matching size
    check("the reference shape itself is plausible", tracker._plausible(work(ref)))
    check("a neck 10% longer is plausible", tracker._plausible(work(stretched(1.1))))
    check("a neck twice as long is refused", not tracker._plausible(work(stretched(2.0))))
    sheared = ref + np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.float32)
    sheared[2:, 1] += np.linalg.norm(tl - bl)  # the far edge slides sideways by the neck length: 45 degrees off square
    check("a sheared neck is refused", not tracker._plausible(work(sheared)))

    # advice on the reference photo: the same checks as the calibration, on the clicked corners
    ideal = [ghost_point(u, t, (W, H)) for u, t in ((0, 0), (1, 0), (1, 1), (0, 1))]
    cx, cy = IDEAL_CENTER[0] * W, IDEAL_CENTER[1] * H
    check("a photo with the neck on the calibration target gets no advice", photo_hint(ideal, (W, H)) is None)
    comfortable = [(cx + (x - cx) * 0.9, cy + (y - cy) * 0.9) for x, y in ideal]
    check("a neck at 90% of the target size (a comfortable, longer distance) gets no distance advice",
          photo_hint(comfortable, (W, H)) is None)
    tiny = [(cx + (x - cx) * 0.4, cy + (y - cy) * 0.4) for x, y in ideal]
    check(f"a neck at 40% of the target size is told to come closer ({photo_hint(tiny, (W, H))})",
          photo_hint(tiny, (W, H)) == "Move the guitar closer to the camera")

    # the guide that shows which corner to click: the little neck changes with the step and pulses
    def guide(step, now):
        img = np.full((H, W, 3), 100, np.uint8)
        draw_click_guide(img, step, now)
        x0, y0 = GUIDE_ORIGIN
        return img, img[y0 - 14:y0 + GUIDE_SIZE[1] + 14, x0:x0 + GUIDE_SIZE[0]]
    full0, mini0 = guide(0, 0.0)
    _, mini1 = guide(1, 0.0)
    _, mini0_later = guide(0, 0.26)
    check("the guide draws a little neck", float(mini0.std()) > 10)
    check("it shows a different corner for each step", not np.array_equal(mini0, mini1))
    check("and pulses (the ring grows and shrinks over time)", not np.array_equal(mini0, mini0_later))
    outside = np.ones(full0.shape[:2], bool)
    x0, y0 = GUIDE_ORIGIN
    outside[y0 - 16:y0 + GUIDE_SIZE[1] + 16, x0:x0 + GUIDE_SIZE[0] + 2] = False
    check("nothing is drawn on the photo itself, only in the little neck", bool((full0[outside] == 100).all()))
    _, done = guide(4, 0.0)
    check("after the four clicks the guide has no pulsing ring", not np.array_equal(mini0, done))

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
