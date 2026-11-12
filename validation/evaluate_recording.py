"""
Evaluates the marker-free tracker on a recorded session (python -m chordar --record FOLDER), with the ArUco corners
of that session as the ground truth.

The markers are glued on during the recording; they are painted out of the reference and ignored in every frame
(masked) so the marker-free tracker cannot use them.

    python validation/evaluate_recording.py sessions/A --save-reference references/from_A.npz   # reference from session A
    python validation/evaluate_recording.py sessions/B --reference references/from_A.npz        # test another session with it
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from markerless_test import marker_squares, string_fret_accuracy  # noqa: E402
from chordar.markerless import MarklessTracker, NeckReference  # noqa: E402
from chordar.recorder import CORNER_NAMES  # noqa: E402


def load_session(folder):
    rows = list(csv.DictReader(open(os.path.join(folder, "corners.csv"))))
    truth = {}
    for row in rows:
        if row["all_markers"] == "1":
            truth[int(row["frame"])] = np.array([[float(row[n + "x"]), float(row[n + "y"])] for n in CORNER_NAMES], np.float32)
    return truth, len(rows)


def read_frame(folder, index):
    capture = cv2.VideoCapture(os.path.join(folder, "video.mp4"))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    parser.add_argument("--reference", help="a reference .npz made elsewhere (default: built from a frame of this session)")
    parser.add_argument("--save-reference", metavar="FILE", help="save the reference built from this session, to evaluate other sessions with it")
    parser.add_argument("--reference-frame", type=int, default=None, help="frame of the session to use as the reference photo")
    args = parser.parse_args()

    truth, total = load_session(args.session)
    if not truth:
        sys.exit("No frame of this session had all four markers detected")
    if args.reference:
        reference = NeckReference.load(args.reference)
    else:
        index = args.reference_frame if args.reference_frame is not None else sorted(truth)[len(truth) // 4]
        frame = read_frame(args.session, index)
        squares = marker_squares(frame)
        reference = NeckReference(cv2.inpaint(frame, squares, 7, cv2.INPAINT_TELEA), truth[index], ignore_mask=squares)
        print(f"reference built from frame {index}")
        truth.pop(index)
        if args.save_reference:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_reference)), exist_ok=True)
            reference.save(args.save_reference)
            print(f"reference saved to {args.save_reference}")
    tracker = MarklessTracker(reference)

    capture = cv2.VideoCapture(os.path.join(args.session, "video.mp4"))
    errors, s_acc, f_acc, times, evaluated, found_n = [], [], [], [], 0, 0
    est_steps, gt_steps, previous = [], [], None   # frame-to-frame movement of the corners, for consecutive frames
    index = -1
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        index += 1
        # every frame goes through the tracker, in order, as in the app; only frames with ground truth are scored
        allowed = 255 - marker_squares(frame)
        start = time.perf_counter()
        found = tracker.update(frame, allowed)
        times.append((time.perf_counter() - start) * 1000)
        if index not in truth:
            previous = None
            continue
        evaluated += 1
        if found is None:
            previous = None
            continue
        found_n += 1
        errors.append(float(np.mean(np.linalg.norm(found - truth[index], axis=1))))
        s, f = string_fret_accuracy(truth[index], found, frame.shape[1::-1])
        s_acc.append(s), f_acc.append(f)
        if previous is not None and previous[0] == index - 1:
            est_steps.append(float(np.mean(np.linalg.norm(found - previous[1], axis=1))))
            gt_steps.append(float(np.mean(np.linalg.norm(truth[index] - previous[2], axis=1))))
        previous = (index, found, truth[index])
    capture.release()

    neck_len = float(np.mean([np.linalg.norm(t[0] - t[3]) for t in truth.values()]))
    print(f"{total} frames, {evaluated} with ground truth (all four markers seen), neck about {neck_len:.0f} px long")
    print(f"found:            {found_n / evaluated:.0%}")
    if errors:
        print(f"corner error:     median {np.median(errors):.1f} px, 90th percentile {np.percentile(errors, 90):.1f} px"
              f" ({100 * np.median(errors) / neck_len:.1f}% of the neck length)")
        print(f"within 3% length: {np.mean([e < 0.03 * neck_len for e in errors]):.0%} of the found frames")
        print(f"string read right: {np.mean(s_acc):.0%}   fret read right: {np.mean(f_acc):.0%}   (probe points 6 strings x frets 1-4)")
    if est_steps:
        print(f"jitter (px/frame): estimate {np.median(est_steps):.2f}, markers {np.median(gt_steps):.2f}; "
              f"90th percentile estimate {np.percentile(est_steps, 90):.1f}, markers {np.percentile(gt_steps, 90):.1f}")
    print(f"time per frame:   {np.mean(times):.0f} ms")


if __name__ == "__main__":
    main()
