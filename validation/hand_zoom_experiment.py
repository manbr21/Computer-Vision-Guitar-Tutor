"""
Does enlarging the hand region before MediaPipe read the fingers better? Replays a recorded session (the neck from an analyze_session cache,
the raw video) at several enlargement factors and compares, on the held parts of the chords, the share of notes covered and the frame to frame
jitter of the fingertips.

    python validation/hand_zoom_experiment.py sessions/<s> /tmp/cache/<s>.pkl [1 1.5 2]
"""
import os
import pickle
import sys
from itertools import groupby

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_session import neck_of, touch_points
from chordar.chords import fretted_notes, reading_bias
from chordar.hands import fretting_hand_region
from chordar.scoring import covered_notes, finger_offsets, common_shift

TIPS, DIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}, {"index": 7, "middle": 11, "ring": 15, "pinky": 19}


def run(session, frames, zoom):
    import mediapipe as mp
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    capture = cv2.VideoCapture(os.path.join(session, "video.mp4"))
    out = []
    for d in frames:
        ok, frame = capture.read()
        if not ok:
            break
        landmarks = None
        if d["neck"]:
            display = cv2.flip(frame, 1)
            h, w = display.shape[:2]
            x0, y0, x1, y1 = fretting_hand_region(neck_of(d), (w, h))
            crop = display[y0:y1, x0:x1]
            if zoom != 1:
                crop = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
            result = hands.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                ch, cw = (y1 - y0), (x1 - x0)
                point = lambda k: (x0 + hand.landmark[k].x * cw, y0 + hand.landmark[k].y * ch)
                landmarks = ({n: point(k) for n, k in TIPS.items()}, {n: point(k) for n, k in DIPS.items()})
        out.append(dict(d, lm=landmarks))
    hands.close()
    return out


def held(frames):
    for chord, group in groupby(frames, key=lambda d: d["chord"]):
        group = list(group)
        if len(group) >= 60 and fretted_notes(chord):
            yield chord, group[int(len(group) * 0.6):len(group) - 30]


def evaluate(frames):
    hits, whole, jitter, total, found = [], [], [], 0, 0
    for chord, group in held(frames):
        notes, bias, last = fretted_notes(chord), reading_bias(chord), None
        for d in group:
            total += 1
            if not d["lm"] or not d["neck"]:
                last = None
                continue
            found += 1
            points = touch_points(d)
            covered = covered_notes(notes, points, neck_of(d), False, bias)
            hits.append(np.mean(covered))
            whole.append(all(covered))
            tips = np.array([points[n] for n in sorted(points)])
            if last is not None:
                jitter.append(float(np.median(np.linalg.norm(tips - last, axis=1))))
            last = tips
    return found / total * 100, np.mean(hits) * 100, np.mean(whole) * 100, np.median(jitter)


if __name__ == "__main__":
    session, cache = sys.argv[1], sys.argv[2]
    zooms = [float(z) for z in sys.argv[3:]] or [1, 1.5, 2]
    base = pickle.load(open(cache, "rb"))
    print("zoom  hand found  notes covered  whole chord  median tip jitter (px/frame)")
    for zoom in zooms:
        f, n, w, j = evaluate(run(session, base, zoom))
        print(f"{zoom:4.1f}  {f:8.0f}%  {n:11.0f}%  {w:10.0f}%  {j:8.2f}", flush=True)
