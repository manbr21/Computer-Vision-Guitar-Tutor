"""Turns the finger positions into what a teacher would say: which notes are done and what to fix first."""
import itertools
import math
from dataclasses import dataclass
from typing import Optional

from chordar.chords import STRING_NAMES
from chordar.config import NUM_STRINGS
from chordar.scoring import assign_fingers

FINGER_NAMES = {1: "index", 2: "middle", 3: "ring", 4: "pinky"}  # chord-library finger number -> tracker name
FINGER_NUMBERS = {name: number for number, name in FINGER_NAMES.items()}


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


def _closest_pairing(missing, free, notes, neck):
    """{note index: finger name}: the free fingers matched to the missing notes so that the total distance is smallest."""
    count = min(len(missing), len(free))
    if count == 0:
        return {}
    best, best_cost = {}, math.inf
    for chosen_notes in itertools.permutations(missing, count):
        for chosen_fingers in itertools.permutations(free, count):
            cost = sum(math.dist(free[name], neck.target(notes[i][0], notes[i][1]))
                       for i, name in zip(chosen_notes, chosen_fingers))
            if cost < best_cost:
                best, best_cost = dict(zip(chosen_notes, chosen_fingers)), cost
    return best


def coach(notes, fingertips, neck, parallax=True, stabilizer=None):
    """
    notes: [(string_idx, fret, finger)] the chord needs. fingertips: name -> pixel. Returns a Coaching: which
    notes are covered (same rule as the score, steadied by `stabilizer` if given), and one message about the first
    note (in the library's finger order) that is not covered. The finger to move is the one the free fingers' best
    pairing with the missing notes gives (smallest total distance), whichever finger the chord library suggests:
    people finger chords differently. Screen directions are used:
    the nut is on the left and the low E on top.
    """
    assigned = assign_fingers(notes, fingertips, neck, parallax)
    done = [i in assigned for i in range(len(notes))]
    if stabilizer is not None:
        done = stabilizer.update(done)
    if not notes:
        return Coaching(done, "Open strings: just strum", perfect=True)
    if all(done):
        return Coaching(done, "Perfect!", perfect=True)
    if not fingertips:
        return Coaching(done, "Place your fretting hand on the neck")

    free = {name: point for name, point in fingertips.items() if name not in assigned.values()}
    missing = sorted((i for i in range(len(notes)) if not done[i]), key=lambda i: notes[i][2])
    pairing = _closest_pairing(missing, free, notes, neck)   # which free finger goes to which missing note
    for i in missing:
        s, f, library_finger = notes[i]
        where = f"string {NUM_STRINGS - s} ({STRING_NAMES[s]}), fret {f}"
        if i not in pairing:
            return Coaching(done, f"Finger {library_finger}: put it on {where}")
        pixel, number = free[pairing[i]], FINGER_NUMBERS[pairing[i]]
        hit = neck.locate(pixel[0], pixel[1], parallax)
        if hit is None:
            return Coaching(done, f"Finger {number}: bring it onto the neck, {where}", fix=(notes[i], pixel))
        return Coaching(done, f"Finger {number}: {_move(hit[0] - s, hit[1] - f)}", fix=(notes[i], pixel))
    return Coaching(done, "Perfect!", perfect=True)
