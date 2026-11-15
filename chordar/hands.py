"""Fretting-hand tracking with MediaPipe Hands: smoothed touch points of the fingers."""
import math
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from chordar import config

FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}  # MediaPipe landmark ids
FINGER_DIPS = {"index": 7, "middle": 11, "ring": 15, "pinky": 19}   # the last joint of each finger
SMOOTH_FRAMES = 5
JUMP_PX = 200  # a position farther than this from the smoothed one restarts the smoothing


def _mean_point(points):
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


def fretting_hand_region(neck, size):
    """
    (x0, y0, x1, y1) in the mirrored frame: where the fretting hand is, around the first frets and below and above the neck
    (a few neck widths either way, from a little before the nut to well past fret 7). The strumming hand is outside it, so
    it is never seen; MediaPipe also looks at fewer pixels.
    """
    width, height = size
    corners = [neck.to_image(u, t) for u in (-2.5, 3.5) for t in (-0.3, 0.85)]
    xs, ys = [p[0] for p in corners], [p[1] for p in corners]
    return max(int(min(xs)), 0), max(int(min(ys)), 0), min(int(max(xs)), width), min(int(max(ys)), height)


class FingertipTracker:
    """Tracks the fingertips of the hand over the neck and smooths them over a few frames."""

    def __init__(self):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_HANDS,  # with 2, the hand over the neck is picked in track()
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._history = {name: deque(maxlen=SMOOTH_FRAMES) for name in FINGER_TIPS}

    def close(self):
        self._hands.close()

    def track(self, frame, anchor=None, region=None):
        """
        Touch points {finger: (x, y)} in the mirrored frame, or {} without a hand. With `region` (x0, y0, x1, y1), only that
        part of the frame is looked at (see fretting_hand_region). With several hands in view and an anchor (a pixel over
        the first frets), the hand closest to it is used: the fretting hand, not the strumming one.
        """
        mirrored = cv2.flip(frame, 1)
        x0, y0 = 0, 0
        if region is not None and region[2] - region[0] > 64 and region[3] - region[1] > 64:
            x0, y0, x1, y1 = region
            mirrored = mirrored[y0:y1, x0:x1]
        results = self._hands.process(cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            self._reset()
            return {}

        height, width = mirrored.shape[:2]
        hands = [{name: self._touch_point(hand, name, width, height, x0, y0) for name in FINGER_TIPS}
                 for hand in results.multi_hand_landmarks]
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

    @staticmethod
    def _touch_point(hand, name, width, height, x0=0, y0=0):
        """
        Where the finger presses: the pad is behind the fingertip landmark, so the point is moved from the tip toward
        the last joint by config.CONTACT_BACK of the way (the tip alone read about one string off for the index).
        """
        tip, joint = hand.landmark[FINGER_TIPS[name]], hand.landmark[FINGER_DIPS[name]]
        x = tip.x + config.CONTACT_BACK * (joint.x - tip.x)
        y = tip.y + config.CONTACT_BACK * (joint.y - tip.y)
        return (int(x * width) + x0, int(y * height) + y0)

    def _reset(self):
        for history in self._history.values():
            history.clear()
