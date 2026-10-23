"""
Check of the finger coaching (chordar/coaching.py): which notes are done and what the message says.
Run from the repository root:  python validation/coaching_test.py   Exits with status 1 on any mismatch.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chordar.chords import fretted_notes
from chordar.coaching import coach
from chordar.geometry import Fretboard


def main():
    failures = []

    def check(label, ok):
        print(("ok    " if ok else "FAIL  ") + label)
        if not ok:
            failures.append(label)

    # nut on the left, low E on top (u grows downward on the screen), frets grow to the right
    neck = Fretboard((100, 100), (100, 200), (900, 200), (900, 100))
    notes = fretted_notes("C")  # [(string 1, fret 3, finger 3), (2, 2, 2), (4, 1, 1)]
    names = {1: "index", 2: "middle", 3: "ring"}

    def tips(**moves):
        """Fingertips on the targets, with `name=(dstring, dfret)` offsets (target string/fret + offset)."""
        out = {}
        for s, f, finger in notes:
            ds, df = moves.get(names[finger], (0, 0))
            out[names[finger]] = neck.target(s + ds, f + df)
        return out

    result = coach(notes, tips(), neck, parallax=False)
    check("all fingers on target: perfect", result.perfect and all(result.done) and result.message == "Perfect!")

    result = coach(notes, tips(middle=(0, 1)), neck, parallax=False)
    check(f"middle one fret too far right: '{result.message}'", result.message == "Finger 2: 1 fret left" and result.done == [True, False, True])
    check("... and the arrow points from that fingertip to its note", result.fix[0] == notes[1])

    result = coach(notes, tips(ring=(1, 0)), neck, parallax=False)
    check(f"ring one string too low on the screen: '{result.message}'", result.message == "Finger 3: 1 string up")

    result = coach(notes, tips(ring=(-1, -1)), neck, parallax=False)
    check(f"ring one string up and one fret left: '{result.message}'", result.message == "Finger 3: 1 string down, 1 fret right")

    result = coach(notes, tips(ring=(0, 2), index=(0, 1)), neck, parallax=False)
    check(f"two wrong fingers: the lower finger number comes first: '{result.message}'", result.message.startswith("Finger 1"))

    result = coach(notes, {}, neck, parallax=False)
    check("no hand: asks to place it", result.message == "Place your fretting hand on the neck" and not result.perfect)

    partial = {k: v for k, v in tips().items() if k != "ring"}
    result = coach(notes, partial, neck, parallax=False)
    check(f"a finger missing from the hand: '{result.message}'", result.message == "Finger 3: put it on string 5 (A), fret 3")

    off = tips()
    off["middle"] = (5, 5)  # off the neck
    result = coach(notes, off, neck, parallax=False)
    check(f"a finger off the neck: '{result.message}'", result.message.startswith("Finger 2: bring it onto the neck"))

    result = coach(fretted_notes("Em", 0)[:0], {}, neck, parallax=False)
    check("a chord with no fretted notes: nothing to fix", result.perfect)

    shifted = fretted_notes("C", 2)
    on_shape = {names[fin]: neck.target(s, f) for s, f, fin in shifted}
    result = coach(shifted, on_shape, neck, parallax=False)
    check("with a capo the shifted notes are what counts", result.perfect)
    check("... and the unshifted shape is corrected by 2 frets",
          coach(fretted_notes("C", 2), {names[fin]: neck.target(s, f) for s, f, fin in fretted_notes("C")},
                neck, parallax=False).message.endswith("2 frets right"))

    print()
    if failures:
        print("FAIL:", *failures, sep="\n  ")
        sys.exit(1)
    print("PASS: the coaching names the right finger and the right correction")


if __name__ == "__main__":
    main()
