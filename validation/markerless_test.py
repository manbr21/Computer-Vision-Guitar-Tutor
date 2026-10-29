"""
Marker-free tracking (chordar/markerless.py) against the ArUco markers, on one photo with random changes.

Take a real photo of the guitar WITH the four markers glued on. The markers give the true neck corners; they are
then painted out, the cleaned photo becomes the reference, and test frames are made from it by random shifts,
rotations, scale and perspective, brightness/contrast/noise/blur changes and skin-colored patches over the neck.
The marker-free tracker must find the corners again; the ArUco tracker is run, for context, on the same changes
applied to the original photo (markers visible).

This is an optimistic test: the test frames come from the same photo as the reference, so the camera noise and the
lighting of a real session are only imitated. A recorded session (python -m chordar --record, then
validation/evaluate_recording.py) is the real test.

Usage (the photo is personal and not part of the repository):
    python validation/markerless_test.py --photo path/to/photo.jpg [--frames 100] [--seed 0]
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chordar.geometry import Fretboard
from chordar.markers import MarkerTracker
from chordar.markerless import MarklessTracker, NeckReference
from chordar.recorder import CORNER_NAMES

STRING_PROBES = [(s, f) for s in range(6) for f in (1, 2, 3, 4)]  # where a fingertip would press


def truth_corners(photo):
    """The neck corners (TL TR BR BL, raw pixels) and the marker squares found by the ArUco tracker on the photo."""
    tracker = MarkerTracker()
    for _ in range(5):
        _, neck = tracker.track(photo)
    if neck is None:
        raise SystemExit("The photo needs all four ArUco markers visible (IDs 0-3)")
    width = photo.shape[1]
    corners = np.array([[width - x, y] for x, y in neck.corners], dtype=np.float32)
    return corners, tracker


def marker_squares(photo):
    """Mask (255) of the ArUco markers in the photo, grown to cover their white margin."""
    from cv2 import aruco
    detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_4X4_1000), aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY))
    mask = np.zeros(photo.shape[:2], np.uint8)
    for c in corners:
        c = c.reshape(4, 2)
        center = c.mean(axis=0)
        cv2.fillConvexPoly(mask, (center + (c - center) * 1.6).astype(np.int32), 255)
    return mask


def random_homography(rng, size):
    w, h = size
    angle = np.radians(rng.uniform(-12, 12))
    scale = rng.uniform(0.88, 1.12)
    shift = np.array([rng.uniform(-0.07, 0.07) * w, rng.uniform(-0.07, 0.07) * h])
    c = np.array([w / 2, h / 2])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) * scale
    affine = np.eye(3)
    affine[:2, :2] = rot
    affine[:2, 2] = c - rot @ c + shift
    persp = np.eye(3)
    persp[2, 0], persp[2, 1] = rng.uniform(-1.5e-4, 1.5e-4), rng.uniform(-1.5e-4, 1.5e-4)
    return persp @ affine


def photometric(img, rng):
    out = img.astype(np.float32) / 255
    out = np.clip(out, 0, 1) ** rng.uniform(0.7, 1.4)
    out = out * rng.uniform(0.6, 1.3) + rng.uniform(-0.1, 0.1)
    out += rng.normal(0, rng.uniform(0.01, 0.03), out.shape)
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    if rng.random() < 0.4:
        out = cv2.GaussianBlur(out, (5, 5), 0)
    return out


def stress(img, kind, rng):
    """Harder changes than the basic ones: dim and noisy light, motion blur, an uneven shadow, heavy compression."""
    out = img.copy()
    if kind == "dark+noise":
        out = (out.astype(np.float32) * rng.uniform(0.2, 0.4) + rng.normal(0, rng.uniform(6, 12), out.shape))
        out = np.clip(out, 0, 255).astype(np.uint8)
    elif kind == "motion blur":
        size = int(rng.integers(9, 21))
        kernel = np.zeros((size, size), np.float32)
        kernel[size // 2, :] = 1.0 / size
        angle = rng.uniform(0, 180)
        M = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (size, size))
        out = cv2.filter2D(out, -1, kernel / max(kernel.sum(), 1e-6))
    elif kind == "uneven light":
        h, w = out.shape[:2]
        gradient = np.linspace(rng.uniform(0.15, 0.5), 1.4, w)[None, :, None]
        out = np.clip(out.astype(np.float32) * (gradient if rng.random() < 0.5 else gradient[:, ::-1]), 0, 255).astype(np.uint8)
    elif kind == "heavy jpeg":
        small = cv2.resize(out, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 12])
        out = cv2.resize(cv2.imdecode(buf, cv2.IMREAD_COLOR), (out.shape[1], out.shape[0]))
    return out


def occlude(img, corners, rng, fraction):
    """Skin-colored blobs over the neck quad, covering about `fraction` of its area."""
    out = img.copy()
    area = abs(cv2.contourArea(corners.astype(np.float32)))
    covered, tries = 0.0, 0
    while covered < fraction * area and tries < 30:
        tries += 1
        t = rng.uniform(0.1, 0.9)
        edge = corners[0] * (1 - t) + corners[3] * t
        p = edge + (corners[1] - corners[0]) * rng.uniform(0.0, 1.0) * 0.5
        axes = (int(rng.uniform(0.04, 0.12) * np.linalg.norm(corners[1] - corners[0])),
                int(rng.uniform(0.06, 0.2) * np.linalg.norm(corners[3] - corners[0])))
        cv2.ellipse(out, (int(p[0]), int(p[1])), axes, rng.uniform(0, 180), 0, 360, (110, 150, 205), -1)
        covered += np.pi * axes[0] * axes[1]
    return out


def string_fret_accuracy(true_corners, found_corners, size):
    """Share of probe points (6 strings x frets 1-4) that the found neck reads as the right (string, fret)."""
    w, h = size
    true_neck = Fretboard(*[tuple(p) for p in true_corners])
    found_neck = Fretboard(*[tuple(p) for p in found_corners])
    right_s = right_f = 0
    for s, f in STRING_PROBES:
        x, y = true_neck.target(s, f)
        hit = found_neck.locate(x, y, parallax=False)
        right_s += hit is not None and hit[0] == s
        right_f += hit is not None and hit[1] == f
    return right_s / len(STRING_PROBES), right_f / len(STRING_PROBES)


def run(photo_path, frames, seed):
    rng = np.random.default_rng(seed)
    photo = cv2.imread(photo_path)
    if photo is None:
        raise SystemExit(f"Cannot read {photo_path}")
    height, width = photo.shape[:2]
    truth, _ = truth_corners(photo)
    squares = marker_squares(photo)
    clean = cv2.inpaint(photo, squares, 7, cv2.INPAINT_TELEA)
    reference = NeckReference(clean, truth, ignore_mask=squares)
    tracker = MarklessTracker(reference)
    neck_len = float(np.linalg.norm(truth[1] - truth[2]) + np.linalg.norm(truth[0] - truth[3])) / 2
    neck_len = max(neck_len, float(np.linalg.norm(truth[0] - truth[3])))
    print(f"photo {width}x{height}, neck {neck_len:.0f} px long, {len(tracker._ref_points)} reference features\n")

    rows = []
    conditions = (("clean", 0.0, None), ("25% covered", 0.25, None), ("dark+noise", 0.0, "dark+noise"),
                  ("motion blur", 0.0, "motion blur"), ("uneven light", 0.0, "uneven light"),
                  ("heavy jpeg", 0.0, "heavy jpeg"), ("blur + 25% cov.", 0.25, "motion blur"))
    for label, occlusion, kind in conditions:
        errors, s_acc, f_acc, times, aruco_found = [], [], [], [], 0
        found_n = 0
        for _ in range(frames):
            H = random_homography(rng, (width, height))
            gt = cv2.perspectiveTransform(truth.reshape(-1, 1, 2), H).reshape(4, 2)
            frame = photometric(cv2.warpPerspective(clean, H, (width, height), borderMode=cv2.BORDER_REPLICATE), rng)
            with_markers = photometric(cv2.warpPerspective(photo, H, (width, height), borderMode=cv2.BORDER_REPLICATE), rng)
            if kind:
                state = rng.bit_generator.state
                frame = stress(frame, kind, rng)
                rng.bit_generator.state = state
                with_markers = stress(with_markers, kind, rng)
            if occlusion:
                state = rng.bit_generator.state
                frame = occlude(frame, gt, rng, occlusion)
                rng.bit_generator.state = state
                with_markers = occlude(with_markers, gt, rng, occlusion)
            start = time.perf_counter()
            found, _ = tracker.estimate(frame)
            times.append((time.perf_counter() - start) * 1000)
            if found is not None:
                found_n += 1
                errors.append(float(np.mean(np.linalg.norm(found - gt, axis=1))))
                s, f = string_fret_accuracy(gt, found, (width, height))
                s_acc.append(s), f_acc.append(f)
            aruco = MarkerTracker()
            if aruco.track(with_markers)[1] is not None and len(aruco.fresh_corners) == 4:
                aruco_found += 1
        good = sum(e < 0.03 * neck_len for e in errors)
        rows.append((label, found_n / frames, good / frames, float(np.median(errors)) if errors else float("nan"),
                     float(np.mean(s_acc)) if s_acc else 0.0, float(np.mean(f_acc)) if f_acc else 0.0,
                     float(np.mean(times)), aruco_found / frames))

    print(f"{'condition':<16}{'found':>7}{'<3% err':>9}{'median px':>11}{'string':>8}{'fret':>7}{'ms/frame':>10}{'ArUco found':>13}")
    for label, found, good, med, s, f, ms, ar in rows:
        print(f"{label:<16}{found:>7.0%}{good:>9.0%}{med:>11.1f}{s:>8.0%}{f:>7.0%}{ms:>10.0f}{ar:>13.0%}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo", default=os.environ.get("CHORDAR_TEST_PHOTO"))
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.photo:
        print("SKIP: pass --photo (a photo of the guitar with the four markers) or set CHORDAR_TEST_PHOTO")
        sys.exit(0)
    run(args.photo, args.frames, args.seed)
