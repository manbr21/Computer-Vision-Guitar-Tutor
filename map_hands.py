import math

import cv2
import mediapipe as mp
from collections import deque
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,          # both hands may be in frame; the one over the neck is picked below
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}

# History buffer for smoothing (5 frames of memory)
finger_history = {name: deque(maxlen=5) for name in FINGER_TIPS.keys()}

# A new hand position farther than this from the smoothed one (in pixels) restarts the smoothing,
# instead of averaging the old and new hands together.
JUMP_PX = 200


def _fingertips_px(hand_landmarks, w, h):
    return {name: (int(hand_landmarks.landmark[i].x * w), int(hand_landmarks.landmark[i].y * h))
            for name, i in FINGER_TIPS.items()}


def _mean_point(points):
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


def get_fingertip_positions(frame, anchor=None):
    """
    Smoothed fingertip pixels (index, middle, ring, pinky) of one hand, in the mirrored frame.
    With two hands in view and an anchor (a pixel over the neck, near the first frets), the hand
    closest to the anchor is used: that is the fretting hand, not the strumming one.
    """
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        for history in finger_history.values():
            history.clear()
        return frame, {}, []

    h, w, _ = frame.shape
    candidates = [(lm, _fingertips_px(lm, w, h)) for lm in results.multi_hand_landmarks]
    if anchor is not None and len(candidates) > 1:
        candidates.sort(key=lambda c: math.dist(_mean_point(list(c[1].values())), anchor))
    hand_landmarks, raw_tips = candidates[0]

    previous = [p for history in finger_history.values() for p in history]
    if previous and math.dist(_mean_point(previous), _mean_point(list(raw_tips.values()))) > JUMP_PX:
        for history in finger_history.values():
            history.clear()

    tips = {}
    for name, (x, y) in raw_tips.items():
        finger_history[name].append((x, y))

        # Average over history
        avg_x = int(np.mean([p[0] for p in finger_history[name]]))
        avg_y = int(np.mean([p[1] for p in finger_history[name]]))
        tips[name] = (avg_x, avg_y)

    return frame, tips, [hand_landmarks]
