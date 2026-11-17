"""
Analyses a recorded session (python -m chordar --record FOLDER) the way the app saw it: the neck tracker and the fretting hand
are replayed on the raw video with the photo the session was started with, and for every chord the notes the app would have
shown as covered are counted, second by second after the key press, with the finger that belongs on each note.

    python validation/analyze_session.py sessions/<date_time> references/<its photo>.npz [--save out.json]

The photo of a session is references/last.npz right after it (the app overwrites it at every start: copy it).
"""
import argparse
import json
import os
import pickle
import sys
from itertools import groupby

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chordar import config
from chordar.chords import STRING_NAMES, fretted_notes
from chordar.geometry import Fretboard
from chordar.hands import fretting_hand_region
from chordar.markerless import MarklessTracker, NeckReference
from chordar.scoring import NoteStabilizer, common_shift, covered_notes, finger_offsets

NAMES = {1: "index", 2: "middle", 3: "ring", 4: "pinky"}
SETTLE_SECONDS = 4   # the fingers need about this long to get into a chord after the key press


def replay(session, reference_path):
    """Per frame: chord shown, neck (corners, fret bounds), and the fingertip/last-joint landmarks of the fretting hand."""
    import mediapipe as mp
    chords = [r["chord"] for r in __import__("csv").DictReader(open(os.path.join(session, "corners.csv")))]
    tracker = MarklessTracker(NeckReference.load(reference_path))
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5)
    tip_ids, dip_ids = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}, {"index": 7, "middle": 11, "ring": 15, "pinky": 19}
    capture = cv2.VideoCapture(os.path.join(session, "video.mp4"))
    frames, i = [], 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        display, neck = tracker.track(frame)
        record = {"i": i, "chord": chords[i] if i < len(chords) else None, "neck": None, "lm": None,
                  "size": (frame.shape[1], frame.shape[0])}
        if neck is not None:
            record["neck"] = ([tuple(map(float, c)) for c in neck.corners], list(neck.fret_bounds))
            h, w = display.shape[:2]
            x0, y0, x1, y1 = fretting_hand_region(neck, (w, h))
            crop = display[y0:y1, x0:x1]
            result = hands.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if result.multi_hand_landmarks:
                hand, (ch, cw) = result.multi_hand_landmarks[0], crop.shape[:2]
                point = lambda k: (x0 + hand.landmark[k].x * cw, y0 + hand.landmark[k].y * ch)
                record["lm"] = ({n: point(k) for n, k in tip_ids.items()}, {n: point(k) for n, k in dip_ids.items()})
        frames.append(record)
        i += 1
    hands.close()
    return frames


def neck_of(d):
    return Fretboard(*d["neck"][0], image_size=d["size"], fret_bounds=d["neck"][1])


def touch_points(d):
    back = config.CONTACT_BACK
    return {n: tuple(np.array(d["lm"][0][n]) + back * (np.array(d["lm"][1][n]) - np.array(d["lm"][0][n]))) for n in d["lm"][0]}


def analyse(frames):
    report = {"chords": {}}
    necks = [d for d in frames if d["neck"]]
    lengths = np.array([np.linalg.norm(((np.array(d["neck"][0][3]) - d["neck"][0][0]) + (np.array(d["neck"][0][2]) - d["neck"][0][1])) / 2)
                        for d in necks])
    centers = np.array([np.mean(d["neck"][0], axis=0) for d in necks])
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    angles = np.median([neck_of(d).camera_angles() for d in necks[::15]], axis=0)
    report["neck"] = {"frames_with_neck": len(necks), "frames": len(frames), "length_px_median": float(np.median(lengths)),
                      "length_px_sd": float(lengths.std()), "center_y_fraction": float(np.median(centers[:, 1]) / frames[0]["size"][1]),
                      "center_step_px_median": float(np.median(steps)), "jumps_over_15px": int((steps > 15).sum()),
                      "camera_angle_total_across_along_deg": [round(float(a), 1) for a in angles]}
    for chord, group in groupby(enumerate(frames), key=lambda x: x[1]["chord"]):
        group = [d for _, d in group]
        if len(group) < 30:
            continue
        notes = fretted_notes(chord)
        stabilizer, per_second, steady_notes, steady_whole, offsets, shifts, first_full = NoteStabilizer(), [], [], [], [], [], None
        settle = SETTLE_SECONDS if len(group) >= 6 * 30 else max(len(group) // 30 - 2, 0)   # short holds: the last seconds count
        for s in range(len(group) // 30):
            values = []
            for d in group[s * 30:(s + 1) * 30]:
                if not d["lm"] or not d["neck"]:
                    stabilizer.reset()
                    continue
                neck, points = neck_of(d), touch_points(d)
                covered = covered_notes(notes, points, neck, False)
                shown = stabilizer.update(covered)
                values.append(np.mean(shown))
                if all(shown) and first_full is None:
                    first_full = d["i"] - group[0]["i"]
                if s >= settle:
                    steady_notes.append(covered)
                    offs = finger_offsets(notes, points, neck, False)
                    if len(offs) == len(notes):
                        shift = common_shift(offs)
                        shifts.append(shift[0])
                        offsets.append([offs[i][0] - shift[0] for i in range(len(notes))])
            per_second.append(float(np.mean(values) * 100) if values else float("nan"))
        steady = np.array(steady_notes, dtype=float)
        report["chords"][chord] = {
            "seconds": len(group) / 30, "per_second": per_second, "peak_second_pct": float(np.nanmax(per_second)) if per_second else None,
            "seconds_to_whole_chord": None if first_full is None else round(first_full / 30, 1), "steady_from_second": settle,
            "steady_notes_pct": float(steady.mean() * 100) if len(steady) else None,
            "steady_whole_chord_pct": float(steady.min(axis=1).mean() * 100) if len(steady) else None,
            "per_note_steady_pct": {f"{STRING_NAMES[s]}{f}/{NAMES[fin]}": float(c * 100) for (s, f, fin), c in zip(notes, steady.mean(axis=0))} if len(steady) else {},
            "per_note_offset_after_shift": {f"{STRING_NAMES[s]}{f}/{NAMES[fin]}": [round(float(np.mean(np.array(offsets)[:, i])), 2), round(float(np.std(np.array(offsets)[:, i])), 2)]
                                            for i, (s, f, fin) in enumerate(notes)} if offsets else {},
            "shared_shift_mean": float(np.mean(shifts)) if shifts else None}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    parser.add_argument("reference")
    parser.add_argument("--save", help="write the report as JSON")
    parser.add_argument("--cache", help="keep the replayed frames in this pickle (and reuse it)")
    args = parser.parse_args()
    if args.cache and os.path.exists(args.cache):
        frames = pickle.load(open(args.cache, "rb"))
    else:
        frames = replay(args.session, args.reference)
        if args.cache:
            pickle.dump(frames, open(args.cache, "wb"))
    report = analyse(frames)
    n = report["neck"]
    print(f"{args.session}: neck found in {n['frames_with_neck']}/{n['frames']} frames, length {n['length_px_median']:.0f} px (sd {n['length_px_sd']:.0f}), "
          f"center y {n['center_y_fraction']:.2f}, steps >15 px: {n['jumps_over_15px']}, camera angle total/across/along {n['camera_angle_total_across_along_deg']} deg")
    print("\nPercent of the notes shown covered, second by second after the key press")
    for chord, c in report["chords"].items():
        if c["steady_notes_pct"] is None:
            print(f"  {chord:3s} " + " ".join(f"{x:4.0f}" for x in c["per_second"][:12]) + "   | no hand in the steady part")
            continue
        print(f"  {chord:3s} " + " ".join(f"{x:4.0f}" for x in c["per_second"][:12]) + f"   | from second {c['steady_from_second'] + 1}: notes {c['steady_notes_pct']:.0f}%, whole chord {c['steady_whole_chord_pct']:.0f}%, whole chord first reached at {c['seconds_to_whole_chord']} s")
        print("       per note: " + "  ".join(f"{k} {v:.0f}%" for k, v in c["per_note_steady_pct"].items()) + f"   | shared shift {c['shared_shift_mean']:+.2f}")
    steady = [c["steady_notes_pct"] for c in report["chords"].values() if c["steady_notes_pct"] is not None]
    whole = [c["steady_whole_chord_pct"] for c in report["chords"].values() if c["steady_whole_chord_pct"] is not None]
    print(f"\nMEAN steady: notes {np.mean(steady):.0f}%, whole chord {np.mean(whole):.0f}%")
    if args.save:
        json.dump(report, open(args.save, "w"), indent=1)


if __name__ == "__main__":
    main()
