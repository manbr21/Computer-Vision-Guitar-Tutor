"""
Synthetic check of the fingertip parallax correction in map_fret_board.Fretboard.

A virtual pinhole camera looks at a virtual guitar neck from a chosen angle. Fingertips are
placed FINGER_HEIGHT_MM above the center of each string/fret, projected to pixels with noise,
and located with Fretboard.locate. Compares the lookup with the correction off, with the
estimated camera pose (pose A) and with the alternative pose solution (pose B, the X key).

Run from the repository root:  python validation/synthetic_parallax_test.py
Exits with status 1 if the correction stops recovering the right string.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import map_fret_board as mfb
from map_fret_board import FRET_BOUNDS, NUM_STRINGS, Fretboard

W, H = 1280, 720
ASSUMED = dict(w0=mfb.QUAD_NUT_WIDTH_MM, w12=mfb.QUAD_FRET12_WIDTH_MM, L=mfb.NECK_LENGTH_MM,
               h=mfb.FINGER_HEIGHT_MM, hfov=mfb.CAMERA_HFOV_DEG)
FRETS_TESTED = range(1, 5)
CONDITIONS = ("off", "pose A", "pose B")


def camera_matrix(hfov):
    f = (W / 2) / np.tan(np.radians(hfov) / 2)
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])


def neck_corners_3d(p):
    return np.float64([[0, -p["w0"] / 2, 0], [0, p["w0"] / 2, 0], [p["L"], p["w12"] / 2, 0], [p["L"], -p["w12"] / 2, 0]])


def make_camera(across, along, distance, length, offset_px=(0.0, 0.0), hfov=ASSUMED["hfov"]):
    """World->camera pose; offset_px turns the camera so the neck center lands that far from the image center."""
    center = np.array([length / 2, 0, 0.0])
    toward = np.array([np.tan(np.radians(along)), np.tan(np.radians(across)), -1.0])
    toward /= np.linalg.norm(toward)
    position = center + distance * toward
    z_axis = -toward
    x_axis = np.array([1.0, 0, 0]) - z_axis[0] * z_axis
    x_axis /= np.linalg.norm(x_axis)
    R = np.vstack([x_axis, np.cross(z_axis, x_axis), z_axis])
    if tuple(offset_px) != (0.0, 0.0):
        focal = camera_matrix(hfov)[0, 0]
        a, b = np.arctan2(offset_px[0], focal), -np.arctan2(offset_px[1], focal)
        Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
        R = Ry @ Rx @ R
    return R, -R @ position


def project(points, R, t, K):
    rvec, _ = cv2.Rodrigues(R)
    return cv2.projectPoints(np.float64(points).reshape(-1, 1, 3), rvec, t, K, None)[0].reshape(-1, 2)


def fingertip_3d(string_idx, fret, true):
    u = mfb.string_u(string_idx)
    t = (FRET_BOUNDS[fret - 1] + FRET_BOUNDS[fret]) / 2
    width = true["w0"] + (true["w12"] - true["w0"]) * t
    return np.array([t * true["L"], (u - 0.5) * width, -true["h"]])


def experiment(across, along=0.0, distance=400.0, true=None, frames=120, corner_sd=0.7, tip_sd=2.0, seed=1,
               offset_px=(0.0, 0.0)):
    """Percent of (string, fret) lookups right, per condition: [string, fret, both]."""
    true = {**ASSUMED, **(true or {})}
    rng = np.random.default_rng(seed)
    K_true = camera_matrix(true["hfov"])
    R, t = make_camera(across, along, distance, true["L"], offset_px, true["hfov"])
    clean_corners = project(neck_corners_3d(true), R, t, K_true)
    hits = {c: np.zeros(3) for c in CONDITIONS}
    total = 0
    for _ in range(frames):
        corners = clean_corners + rng.normal(0, corner_sd, clean_corners.shape)
        necks = {"off": Fretboard(*corners, image_size=(W, H), pose_variant=0),
                 "pose A": Fretboard(*corners, image_size=(W, H), pose_variant=0),
                 "pose B": Fretboard(*corners, image_size=(W, H), pose_variant=1)}
        for s in range(NUM_STRINGS):
            for f in FRETS_TESTED:
                px = project(fingertip_3d(s, f, true), R, t, K_true)[0] + rng.normal(0, tip_sd, 2)
                total += 1
                for name, neck in necks.items():
                    got = neck.locate(px[0], px[1], parallax=(name != "off"))
                    gs, gf = got if got is not None else (None, None)
                    hits[name] += np.array([gs == s, gf == f, gs == s and gf == f])
    return {name: h / total * 100 for name, h in hits.items()}


def show(label, r):
    cols = "   ".join(f"{c}: {r[c][0]:5.1f}/{r[c][1]:5.1f}" for c in CONDITIONS)
    print(f"{label:<26}{cols}")


def main():
    failures = []

    def check(label, ok):
        if not ok:
            failures.append(label)

    print("Columns: string% / fret% right for each condition (correction off, pose A, pose B)\n")
    print("== Tilt across the neck (camera at 400 mm) ==")
    for a in (0, 10, 20, 30, 40, 50):
        r = experiment(a); show(f"across {a} deg", r)
        if 20 <= a <= 40:
            check(f"across {a}: pose A string >= 95%", r["pose A"][0] >= 95)

    print("\n== Combined tilt (across, along) ==")
    print("Moderate (what the calibration allows): string and fret must stay right")
    for a, b in ((20, 5), (20, 10)):
        r = experiment(a, b); show(f"across {a}, along {b}", r)
        check(f"across {a}, along {b}: pose A both >= 95%", r["pose A"][2] >= 95)
    print("Strong (known limit: the drawn grid does not model perspective along the neck, so frets drift)")
    for a, b in ((20, 20), (30, 30), (0, 30)):
        r = experiment(a, b); show(f"across {a}, along {b}", r)
        check(f"across {a}, along {b}: pose A string >= 95%", r["pose A"][0] >= 95)

    print("\n== Neck off-center in the image (the camera turned away from it) ==")
    print("Moderate (what the calibration allows): string and fret must stay right")
    for across, offset, label in ((20, (-128, 0), "10% left, tilt 20"), (20, (128, 0), "10% right, tilt 20"),
                                  (20, (0, 108), "15% lower, tilt 20")):
        r = experiment(across, offset_px=offset); show(label, r)
        check(f"off-center {label}: pose A both >= 95%", r["pose A"][2] >= 95)
    print("Strong (known limit, same reason as above): only the string is required")
    for across, offset, label in ((0, (-320, 0), "25% left, no tilt"), (0, (320, 0), "25% right, no tilt"),
                                  (20, (-320, 0), "25% left, tilt 20"), (30, (0, 150), "20% lower, tilt 30"),
                                  (20, (-400, 150), "far corner, tilt 20")):
        r = experiment(across, offset_px=offset); show(label, r)
        check(f"off-center {label}: pose A string >= 95%", r["pose A"][0] >= 95)

    print("\n== Wrong webcam field of view at 30 deg (assumed %.0f) ==" % ASSUMED["hfov"])
    for fov in (50, 65, 85):
        r = experiment(30, true=dict(hfov=fov)); show(f"true hfov {fov}", r)
        check(f"hfov {fov}: pose A string >= 95%", r["pose A"][0] >= 95)

    print("\n== Wrong guitar numbers / finger height at 30 deg ==")
    for label, true in (("quad 42/52 mm, L=320", dict(w0=42, w12=52, L=320)), ("quad 40/48 mm", dict(w0=40, w12=48)),
                        ("finger h=6 mm", dict(h=6)), ("finger h=14 mm", dict(h=14))):
        r = experiment(30, true=true); show(label, r)
        check(f"{label}: pose A string >= 90%", r["pose A"][0] >= 90)

    wide = dict(w0=ASSUMED["w0"] * 1.18, w12=ASSUMED["w12"] * 1.18)
    print("\n== Quad 18% wider than the code assumes (e.g. markers on the neck edges with the wrong preset) ==")
    print("With little tilt the frets must not be affected: they are anchored to the drawn grid")
    for a in (0, 10):
        r = experiment(a, true=wide); show(f"widths +18%, across {a}", r)
        check(f"widths +18%, across {a}: pose A both >= 95%", r["pose A"][2] >= 95)

    print("\n== Wrong quad proportions at 30 deg (informational, no pass/fail) ==")
    print("The tilt estimate relies on the quad width and length set in map_fret_board.py: measure them.")
    for label, true in (("widths +9%", dict(w0=ASSUMED["w0"] * 1.09, w12=ASSUMED["w12"] * 1.09)),
                        ("widths +18%", wide),
                        ("length -8%", dict(L=ASSUMED["L"] * 0.92)),
                        ("length +8%", dict(L=ASSUMED["L"] * 1.08))):
        show(label, experiment(30, true=true))

    print()
    if failures:
        print("FAIL:", *failures, sep="\n  ")
        sys.exit(1)
    print("PASS: the correction recovers the right string across the tested conditions")


if __name__ == "__main__":
    main()
