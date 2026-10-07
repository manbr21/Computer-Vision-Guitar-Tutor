"""
Synthetic check of the calibration alignment metrics and hints (calibration.py).

A virtual camera projects the neck onto the image in the ideal spot and in deliberately wrong ones
(too close, too far, shifted, rotated, tilted); the test checks that the measured values and the
hints say the right thing.

Run from the repository root:  python validation/calibration_alignment_test.py
Exits with status 1 on any mismatch.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibration as cal
import synthetic_parallax_test as syn
from map_fret_board import Fretboard


def neck_corners(across=0.0, along=0.0, distance=cal.IDEAL_DISTANCE_MM, shift=(0.0, 0.0), rotate_deg=0.0):
    """Marker corners (pixels) of a neck seen by the virtual camera, then moved/rotated in the image."""
    true = dict(syn.ASSUMED)
    R, t = syn.make_camera(across, along, distance, true["L"])
    pixels = syn.project(syn.neck_corners_3d(true), R, t, syn.camera_matrix(true["hfov"]))
    center = np.array([syn.W / 2, syn.H / 2])
    a = math.radians(rotate_deg)
    rotation = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    pixels = (pixels - center) @ rotation.T + center
    to_ideal = np.array([(cal.IDEAL_CENTER[0] - 0.5) * syn.W, (cal.IDEAL_CENTER[1] - 0.5) * syn.H])
    return pixels + to_ideal + np.array(shift)


def run(**kwargs):
    neck = Fretboard(*neck_corners(**kwargs), image_size=(syn.W, syn.H))
    metrics = cal.alignment_metrics(neck, (syn.W, syn.H))
    return metrics, {c["label"]: c for c in cal.evaluate(metrics)}


def main():
    failures = []

    def check(label, ok):
        print(("ok    " if ok else "FAIL  ") + label)
        if not ok:
            failures.append(label)

    m, c = run()
    check("ideal placement: every check passes", all(x["ok"] for x in c.values()))
    # the target neck sits below the image center, so the camera sees it ~2 deg off-axis even when frontal
    check("ideal placement: scale ~1, offsets ~0, level, almost no tilt",
          abs(m["scale"] - 1) < 0.02 and abs(m["dx"]) < 0.01 and abs(m["dy"]) < 0.01
          and abs(m["roll"]) < 1 and m["tilt"] < 5)

    m, c = run(distance=300)
    check("too close: distance fails, asks to move farther", not c["Distance"]["ok"] and "farther" in c["Distance"]["hint"])
    m, c = run(distance=600)
    check("too far: distance fails, asks to move closer", not c["Distance"]["ok"] and "closer" in c["Distance"]["hint"])

    m, c = run(shift=(0.15 * syn.W, 0))
    check("shifted right: horizontal fails, asks to move left", not c["Horizontal"]["ok"] and "left" in c["Horizontal"]["hint"])
    m, c = run(shift=(-0.15 * syn.W, 0))
    check("shifted left: horizontal fails, asks to move right", not c["Horizontal"]["ok"] and "right" in c["Horizontal"]["hint"])

    m, c = run(shift=(0, 0.2 * syn.H))
    check("too low: vertical fails, asks to move up", not c["Vertical"]["ok"] and "up" in c["Vertical"]["hint"])
    m, c = run(shift=(0, -0.2 * syn.H))
    check("too high: vertical fails, asks to move down", not c["Vertical"]["ok"] and "down" in c["Vertical"]["hint"])

    for angle in (20, -20):
        m, c = run(rotate_deg=angle)
        check(f"rotated {angle} deg: level fails, roll ~{angle}", not c["Level"]["ok"] and abs(m["roll"] - angle) < 2)

    m, c = run(across=45)
    check("camera 45 deg across: camera angle fails", not c["Camera angle"]["ok"] and "across" in c["Camera angle"]["hint"])
    m, c = run(along=45)
    check("camera 45 deg along: camera angle fails", not c["Camera angle"]["ok"] and "along" in c["Camera angle"]["hint"])

    # --- placement monitor during play, with a simulated clock at 10 frames per second ---
    size = (syn.W, syn.H)

    def feed(monitor, start, seconds, **placement):
        neck = Fretboard(*neck_corners(**placement), image_size=size)
        hint = None
        for i in range(int(seconds * 10)):
            hint = monitor.update(neck, size, start + i / 10)
        return hint, start + seconds

    monitor = cal.PlacementMonitor()
    hint, now = feed(monitor, 0, 5)
    check("monitor: quiet while the guitar is well placed", hint is None)
    hint, now = feed(monitor, now, 1.0, shift=(0.3 * syn.W, 0))
    check("monitor: no warning before the delay", hint is None)
    hint, now = feed(monitor, now, 2.0, shift=(0.3 * syn.W, 0))
    check("monitor: warns after the delay, with the hint", hint is not None and "left" in hint)
    hint, now = feed(monitor, now, 1.0)
    check("monitor: warning clears once the placement is fixed", hint is None)
    hint, now = feed(monitor, now, 5, shift=(0.09 * syn.W, 0))
    check("monitor: a small drift inside the play tolerance stays quiet", hint is None)
    strict = {x["label"]: x for x in cal.evaluate(cal.alignment_metrics(
        Fretboard(*neck_corners(shift=(0.09 * syn.W, 0)), image_size=size), size))}
    check("... although the calibration tolerance would reject that drift", not strict["Horizontal"]["ok"])
    check("monitor: a lost neck gives no hint (the marker message covers it)", monitor.update(None, size, now + 10) is None)

    print()
    if failures:
        print("FAIL:", *failures, sep="\n  ")
        sys.exit(1)
    print("PASS: calibration metrics, hints and the placement monitor match the simulated placements")


if __name__ == "__main__":
    main()
