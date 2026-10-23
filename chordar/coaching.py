"""Turns the finger positions into what a teacher would say: which notes are done and what to fix first."""
from dataclasses import dataclass
from typing import Optional

from chordar.chords import STRING_NAMES
from chordar.config import NUM_STRINGS

FINGER_NAMES = {1: "index", 2: "middle", 3: "ring", 4: "pinky"}  # chord-library finger number -> tracker name


@dataclass
class Coaching:
    done: list                       # one bool per note, in the order of `notes`
    message: str                     # the one thing to tell the user
    fix: Optional[tuple] = None      # (note, fingertip pixel) of the finger to guide with an arrow, or None
    perfect: bool = False


def _plural(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def _move(ds, df):
    """Words for a finger that is ds strings and df frets away from its target (screen directions)."""
    parts = []
    if ds:
        parts.append(f"{_plural(abs(ds), 'string')} {'down' if ds < 0 else 'up'}")
    if df:
        parts.append(f"{_plural(abs(df), 'fret')} {'right' if df < 0 else 'left'}")
    return ", ".join(parts)


def coach(notes, fingertips, neck, parallax=True):
    """
    notes: [(string_idx, fret, finger)] the chord needs. fingertips: name -> pixel. Returns a Coaching: which
    notes are covered (same rule as the score), and one message about the first finger (in finger order)
    that is not where it should be. Screen directions are used: the nut is on the left and the low E on top.
    """
    located = {name: neck.locate(x, y, parallax) for name, (x, y) in fingertips.items()}
    covered = set(located.values())
    done = [(s, f) in covered for s, f, _ in notes]
    if not notes:
        return Coaching(done, "Open strings: just strum", perfect=True)
    if all(done):
        return Coaching(done, "Perfect!", perfect=True)
    if not fingertips:
        return Coaching(done, "Place your fretting hand on the neck")

    for (s, f, finger), ok in sorted(zip(notes, done), key=lambda item: item[0][2]):
        if ok:
            continue
        name = FINGER_NAMES.get(finger)
        label = f"Finger {finger}"
        where = f"string {NUM_STRINGS - s} ({STRING_NAMES[s]}), fret {f}"
        if name not in fingertips:
            return Coaching(done, f"{label}: put it on {where}")
        hit = located[name]
        pixel = fingertips[name]
        if hit is None:
            return Coaching(done, f"{label}: bring it onto the neck, {where}", fix=((s, f, finger), pixel))
        return Coaching(done, f"{label}: {_move(hit[0] - s, hit[1] - f)}", fix=((s, f, finger), pixel))
    return Coaching(done, "Perfect!", perfect=True)
