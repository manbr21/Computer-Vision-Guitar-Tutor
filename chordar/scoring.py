"""
Chord accuracy: which of the notes a chord needs are covered by the finger that belongs on each of them.

A finger is read relative to its note (how many string gaps and fret spaces away its touch point is). Part of that
reading depends on where the camera is and how the hand is seen (the height of the guitar against the camera, the pose
of the hand) and shifts all the fingers the same way; so one shared shift, within limits, is taken out before the
fingers are compared with their notes. What is left is what each finger got wrong, and it is judged one finger per note.
"""
import numpy as np

from chordar import config

FINGER_NAMES = {1: "index", 2: "middle", 3: "ring", 4: "pinky"}


def finger_offsets(notes, fingertips, neck, parallax=True):
    """
    {note index: (strings, fret spaces)}: where the finger that belongs on each (string_idx, fret, finger) note is,
    relative to the note (+ strings = toward the high e, + fret spaces = toward the body; 0 = centre of the fret space).
    Notes whose finger is not tracked are left out.
    """
    lo, hi = config.STRING_U_RANGE
    offsets = {}
    for i, (string_idx, fret, finger) in enumerate(notes):
        point = fingertips.get(FINGER_NAMES.get(finger))
        if point is None:
            continue
        u, t = neck.to_neck_corrected(point[0], point[1], parallax)
        near_wire, far_wire = neck.fret_bounds[fret - 1], neck.fret_bounds[fret]
        offsets[i] = ((u - lo) / (hi - lo) * (config.NUM_STRINGS - 1) - string_idx,
                      (t - (near_wire + far_wire) / 2) / (far_wire - near_wire))
    return offsets


def common_shift(offsets):
    """
    The shift shared by the fingers, limited to COMMON_SHIFT strings and COMMON_FRET_SHIFT fret spaces (0 with a
    single finger; at most 0.4 strings with two, which could otherwise hide one finger's error).
    """
    if len(offsets) < 2:
        return np.zeros(2)
    mean = np.mean(list(offsets.values()), axis=0)
    string_limit = config.COMMON_SHIFT if len(offsets) >= 3 else min(config.COMMON_SHIFT, 0.4)
    return np.clip(mean, [-string_limit, -config.COMMON_FRET_SHIFT], [string_limit, config.COMMON_FRET_SHIFT])


def covered_notes(notes, fingertips, neck, parallax=True):
    """
    One bool per (string_idx, fret, finger) note: is it covered by its own finger? After the shared shift, the finger
    must be within STRING_TOLERANCE string gaps of the string and in the fret space (or FRET_BEHIND of it past the wire).
    """
    offsets = finger_offsets(notes, fingertips, neck, parallax)
    shift = common_shift(offsets)
    done = [False] * len(notes)
    for i, (strings, spaces) in offsets.items():
        strings, spaces = strings - shift[0], spaces - shift[1]
        done[i] = abs(strings) <= config.STRING_TOLERANCE and -0.5 <= spaces < 0.5 + config.FRET_BEHIND
    return done


def chord_accuracy(notes, fingertips, neck, parallax=True):
    """Percent (0-100) of the notes covered (see covered_notes), or None when the chord has no fretted notes."""
    if not notes:
        return None
    return int(100 * sum(covered_notes(notes, fingertips, neck, parallax)) / len(notes))


class NoteStabilizer:
    """
    Turns the per-frame covered/uncovered flags of the notes into steady ones: a note is shown as covered only after
    `on_frames` frames in a row and stays covered until it has been missed `off_frames` frames in a row. A finger
    that is really on its note passes within a tenth of a second, while a finger a string off, which the tolerance
    lets through in a third of the frames, almost never has a long enough run; and the notes no longer flicker.
    """

    def __init__(self, on_frames=None, off_frames=None):
        self.on_frames = config.NOTE_ON_FRAMES if on_frames is None else on_frames
        self.off_frames = config.NOTE_OFF_FRAMES if off_frames is None else off_frames
        self.reset()

    def reset(self):
        self._shown = []
        self._run = []

    def update(self, covered):
        """covered: one bool per note for this frame. Returns the steady flags."""
        if len(covered) != len(self._shown):   # a different chord: start again
            self._shown, self._run = [False] * len(covered), [0] * len(covered)
        for i, now in enumerate(covered):
            same = now == self._shown[i]
            self._run[i] = 0 if same else self._run[i] + 1
            if not same and self._run[i] >= (self.on_frames if now else self.off_frames):
                self._shown[i], self._run[i] = now, 0
        return list(self._shown)
