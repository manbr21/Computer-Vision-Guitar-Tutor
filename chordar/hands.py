"""Fretting-hand tracking with MediaPipe Hands: smoothed fingertip pixels."""
import math
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}  # MediaPipe landmark ids
SMOOTH_FRAMES = 5
JUMP_PX = 200  # a position farther than this from the smoothed one restarts the smoothing


def _mean_point(points):
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


class FingertipTracker:
    """Tracks the fingertips of the hand over the neck and smooths them over a few frames."""

    def __init__(self):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # both hands may be in frame; the one over the neck is picked in track()
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._history = {name: deque(maxlen=SMOOTH_FRAMES) for name in FINGER_TIPS}

    def close(self):
        self._hands.close()

    def track(self, frame, anchor=None):
        """
        Fingertip pixels {finger: (x, y)} in the mirrored frame, or {} without a hand. With two hands in view
        and an anchor (a pixel over the first frets), the hand closest to it is used: the fretting hand, not
        the strumming one.
        """
        mirrored = cv2.flip(frame, 1)
        results = self._hands.process(cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            self._reset()
            return {}

        height, width = mirrored.shape[:2]
        hands = [{name: (int(hand.landmark[i].x * width), int(hand.landmark[i].y * height))
                  for name, i in FINGER_TIPS.items()} for hand in results.multi_hand_landmarks]
        if anchor is not None and len(hands) > 1:
            hands.sort(key=lambda tips: math.dist(_mean_point(list(tips.values())), anchor))
        raw = hands[0]

        previous = [p for history in self._history.values() for p in history]
        if previous and math.dist(_mean_point(previous), _mean_point(list(raw.values()))) > JUMP_PX:
            self._reset()  # a different hand, or a jump: do not average it with the old position

        tips = {}
        for name, point in raw.items():
            self._history[name].append(point)
            tips[name] = tuple(int(v) for v in np.mean(self._history[name], axis=0))
        return tips

    def _reset(self):
        for history in self._history.values():
            history.clear()
