"""Chord accuracy: which of the notes a chord needs are covered by a finger, one finger per note."""
from chordar import config


def _reach(neck, point, note, parallax):
    """How far a touch point is from a note if it can cover it (within the string and fret tolerances), else None."""
    string_idx, fret, _ = note
    u, t = neck.to_neck_corrected(point[0], point[1], parallax)
    lo, hi = config.STRING_U_RANGE
    string_gap = abs((u - lo) / (hi - lo) * (config.NUM_STRINGS - 1) - string_idx)
    near_wire, far_wire = neck.fret_bounds[fret - 1], neck.fret_bounds[fret]
    space = far_wire - near_wire
    if string_gap > config.STRING_TOLERANCE or not near_wire <= t < far_wire + config.FRET_BEHIND * space:
        return None
    return string_gap + 0.5 * abs(t - (near_wire + far_wire) / 2) / space


def assign_fingers(notes, fingertips, neck, parallax=True):
    """
    {note index: finger name} for the notes that are covered. A finger covers a note when its touch point is on the
    note's string (within STRING_TOLERANCE string gaps) and in its fret space (or just behind the wire), and a finger
    covers one note only: the closest pairs are matched first. Which finger it is is not checked.
    """
    pairs = []
    for name, point in fingertips.items():
        for i, note in enumerate(notes):
            distance = _reach(neck, point, note, parallax)
            if distance is not None:
                pairs.append((distance, name, i))
    assigned = {}
    for _, name, i in sorted(pairs):
        if i not in assigned and name not in assigned.values():
            assigned[i] = name
    return assigned


def covered_notes(notes, fingertips, neck, parallax=True):
    """One bool per (string_idx, fret, finger) note: is it covered by a finger? (see assign_fingers)"""
    assigned = assign_fingers(notes, fingertips, neck, parallax)
    return [i in assigned for i in range(len(notes))]


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
