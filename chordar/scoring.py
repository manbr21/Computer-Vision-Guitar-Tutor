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


def covered_notes(notes, fingertips, neck, parallax=True):
    """
    One bool per (string_idx, fret, finger) note: is it covered by a finger? A finger covers a note when its touch
    point is on the note's string (within STRING_TOLERANCE string gaps) and in its fret space (or just behind the
    wire), and a finger covers one note only: the closest pairs are matched first. Which finger it is is not checked.
    """
    pairs = []
    for name, point in fingertips.items():
        for i, note in enumerate(notes):
            distance = _reach(neck, point, note, parallax)
            if distance is not None:
                pairs.append((distance, name, i))
    done, used = [False] * len(notes), set()
    for _, name, i in sorted(pairs):
        if not done[i] and name not in used:
            done[i] = True
            used.add(name)
    return done


def chord_accuracy(notes, fingertips, neck, parallax=True):
    """Percent (0-100) of the notes covered (see covered_notes), or None when the chord has no fretted notes."""
    if not notes:
        return None
    return int(100 * sum(covered_notes(notes, fingertips, neck, parallax)) / len(notes))
