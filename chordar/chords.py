"""Chord library: which fret and finger each string takes."""

STRING_NAMES = ["E", "A", "D", "G", "B", "e"]  # index 0 = lowest (6th) string

# frets: fret per string (None = muted, 0 = open); fingers: finger that presses it (1 = index ... 4 = pinky).
CHORD_LIBRARY = {
    "C": {"frets": [None, 3, 2, 0, 1, 0], "fingers": [None, 3, 2, None, 1, None], "name": "C Major"},
    "D": {"frets": [None, None, 0, 2, 3, 2], "fingers": [None, None, None, 1, 3, 2], "name": "D Major"},
    "E": {"frets": [0, 2, 2, 1, 0, 0], "fingers": [None, 2, 3, 1, None, None], "name": "E Major"},
    "G": {"frets": [3, 2, 0, 0, 0, 3], "fingers": [3, 2, None, None, None, 4], "name": "G Major"},
    "A": {"frets": [None, 0, 2, 2, 2, 0], "fingers": [None, None, 1, 2, 3, None], "name": "A Major"},
    "Em": {"frets": [0, 2, 2, 0, 0, 0], "fingers": [None, 2, 3, None, None, None], "name": "E Minor"},
    "Am": {"frets": [None, 0, 2, 2, 1, 0], "fingers": [None, None, 2, 3, 1, None], "name": "A Minor"},
    "Dm": {"frets": [None, None, 0, 2, 3, 1], "fingers": [None, None, None, 2, 4, 1], "name": "D Minor"},
}


MAX_CAPO = 7  # highest capo fret offered; with the highest chord note (fret 3) it stays on the 12-fret neck


def fretted_notes(chord, capo=0):
    """
    [(string_idx, fret, finger)] for the notes of a chord that a finger presses. With a capo on fret N the
    whole shape moves up N frets: the capo plays the open strings, so they need no finger, and muted
    strings stay muted.
    """
    info = CHORD_LIBRARY[chord]
    return [(string_idx, fret + capo, finger)
            for string_idx, (fret, finger) in enumerate(zip(info["frets"], info["fingers"]))
            if fret is not None and fret > 0 and finger]
