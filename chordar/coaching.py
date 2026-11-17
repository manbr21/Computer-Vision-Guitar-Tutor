"""Turns the finger positions into what a teacher would say: which notes are done and what to fix first."""
from dataclasses import dataclass
from typing import Optional

from chordar.chords import STRING_NAMES
from chordar.config import NUM_STRINGS, STRING_TOLERANCE
from chordar.scoring import FINGER_NAMES, common_shift, covered_notes, finger_offsets


@dataclass
class Coaching:
    done: list                       # one bool per note, in the order of `notes`
    message: str                     # the one thing to tell the user
    fix: Optional[tuple] = None      # (note, fingertip pixel) of the finger to guide with an arrow, or None
    perfect: bool = False


def _plural(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def _move(strings, spaces):
    """Words for a finger that is `strings` string gaps and `spaces` fret spaces away from its note (screen directions)."""
    ds, df = int(round(strings)), int(round(spaces))
    parts = []
    if ds:
        parts.append(f"{_plural(abs(ds), 'string')} {'down' if ds < 0 else 'up'}")
    if df:
        parts.append(f"{_plural(abs(df), 'fret')} {'right' if df < 0 else 'left'}")
    if not parts:   # just outside the tolerance: say which way
        if abs(strings) >= abs(spaces):
            parts.append("a little " + ("down" if strings < 0 else "up"))
        else:
            parts.append("a little " + ("right" if spaces < 0 else "left"))
    return ", ".join(parts)


def coach(notes, fingertips, neck, parallax=True, stabilizer=None, bias=None):
    """
    notes: [(string_idx, fret, finger)] the chord needs; each note belongs to one finger. fingertips: name -> pixel.
    Returns a Coaching: which notes are covered by their own finger (steadied by `stabilizer` if given), and one message
    about the first note, in finger order, that is not covered. Screen directions are used: the nut is on the left
    and the low E on top.
    """
    offsets = finger_offsets(notes, fingertips, neck, parallax, bias)
    done = covered_notes(notes, fingertips, neck, parallax, bias)
    if stabilizer is not None:
        done = stabilizer.update(done)
    if not notes:
        return Coaching(done, "Open strings: just strum", perfect=True)
    if all(done):
        return Coaching(done, "Perfect!", perfect=True)
    if not fingertips:
        return Coaching(done, "Place your fretting hand on the neck")

    shift = common_shift(offsets)
    for i in sorted((i for i in range(len(notes)) if not done[i]), key=lambda i: notes[i][2]):
        s, f, finger = notes[i]
        where = f"string {NUM_STRINGS - s} ({STRING_NAMES[s]}), fret {f}"
        if i not in offsets:   # that finger is not in view
            return Coaching(done, f"Finger {finger}: put it on {where}")
        pixel = fingertips[FINGER_NAMES[finger]]
        strings, spaces = offsets[i][0] - shift[0], offsets[i][1] - shift[1]
        if neck.locate(pixel[0], pixel[1], parallax) is None:
            return Coaching(done, f"Finger {finger}: bring it onto the neck, {where}", fix=(notes[i], pixel))
        if abs(strings) <= STRING_TOLERANCE and -0.5 <= spaces < 0.75:   # covered, only held back by the steadying
            continue
        return Coaching(done, f"Finger {finger}: {_move(strings, spaces)}", fix=(notes[i], pixel))
    return Coaching(done, "Almost: hold the fingers still")
