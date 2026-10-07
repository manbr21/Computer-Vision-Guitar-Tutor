"""Chord accuracy: how many of the notes a chord needs are covered by a fingertip."""


def chord_accuracy(notes, fingertips, neck, parallax=True):
    """
    Percent (0-100) of the (string_idx, fret, finger) notes covered by some fingertip, or None when the chord
    has no fretted notes. The string and the fret must both match; which finger does it is not checked.
    """
    if not notes:
        return None
    covered = {neck.locate(x, y, parallax) for (x, y) in fingertips.values()}
    return int(100 * sum(1 for string_idx, fret, _ in notes if (string_idx, fret) in covered) / len(notes))
