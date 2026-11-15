"""
Check of the scoring tolerances (chordar/scoring.py): a finger a little off its string or just behind the wire still
covers the note; one finger covers one note only; a finger a whole string off does not.
Run from the repository root:  python validation/scoring_test.py   Exits with status 1 on any mismatch.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chordar import config
from chordar.chords import fretted_notes
from chordar.geometry import Fretboard
from chordar.scoring import chord_accuracy, covered_notes


def main():
    failures = []

    def check(label, ok):
        print(("ok    " if ok else "FAIL  ") + label)
        if not ok:
            failures.append(label)

    neck = Fretboard((100, 100), (100, 200), (900, 200), (900, 100))   # strings 20 px apart, nut on the left
    gap = 20.0
    notes = fretted_notes("Am")          # (string 2 fret 2, finger 2), (3, 2, finger 3), (4, 1, finger 1)
    names = {1: "index", 2: "middle", 3: "ring", 4: "pinky"}

    def tips(dstring=None, dx=None):
        """Every finger on its target, with `finger: offset` shifts in strings (dstring) or pixels along the neck (dx)."""
        out = {}
        for s, f, finger in notes:
            x, y = neck.target(s, f)
            out[names[finger]] = (x + (dx or {}).get(names[finger], 0), y + gap * (dstring or {}).get(names[finger], 0))
        return out

    check("fingers on the targets score 100%", chord_accuracy(notes, tips(), neck, False) == 100)
    check("a finger 0.5 string off still covers its note", chord_accuracy(notes, tips({"index": -0.5}), neck, False) == 100)
    check("a finger 0.9 string off does not (66%)", chord_accuracy(notes, tips({"index": -0.9}), neck, False) == 66)
    check("a finger 1 string off, as seen from the old tip reading, does not either", chord_accuracy(notes, tips({"index": -1.0}), neck, False) == 66)

    x_wire = neck.to_image(0.5, neck.fret_bounds[2])[0]
    space_px = (neck.fret_bounds[2] - neck.fret_bounds[1]) * 800
    behind = tips(dx={"middle": (x_wire - neck.target(2, 2)[0]) + 0.15 * space_px})       # 15% of a fret past the wire
    check("a finger just behind the fret wire still covers the fret", chord_accuracy(notes, behind, neck, False) == 100)
    far = tips(dx={"middle": (x_wire - neck.target(2, 2)[0]) + 0.6 * space_px})            # well into the next fret
    check("a finger well into the next fret does not", chord_accuracy(notes, far, neck, False) == 66)

    # one finger between the D and G strings at fret 2 is within tolerance of both notes: it may cover only one
    between = {"index": neck.target(4, 1), "middle": (neck.target(2, 2)[0], neck.target(2, 2)[1] + 0.5 * gap)}
    done = covered_notes(notes, between, neck, False)
    check(f"one finger between two notes covers one of them only ({done})", sum(done) == 2 and done[2])

    # the steadying of the covered flags
    from chordar.scoring import NoteStabilizer
    st = NoteStabilizer(3, 4)
    seq = [[True]] * 2 + [[False]] + [[True]] * 3
    shown = [st.update(x)[0] for x in seq]
    check(f"a note turns green only after 3 covered frames in a row ({shown})", shown == [False, False, False, False, False, True])
    shown = [st.update([False])[0] for _ in range(4)]
    check(f"... and turns off only after 4 missed frames in a row ({shown})", shown == [True, True, True, False])
    st = NoteStabilizer(3, 4)
    flicker = [st.update([i % 2 == 0])[0] for i in range(20)]
    check("a note that flickers every frame never turns green", not any(flicker))
    st.update([True, True, True])
    check("a different chord (other number of notes) starts again", st.update([True, True]) == [False, False])

    check("the tolerances are the configured ones", config.STRING_TOLERANCE >= 0.5 and config.FRET_BEHIND >= 0)

    print()
    if failures:
        print("FAIL:", *failures, sep="\n  ")
        sys.exit(1)
    print("PASS: the scoring tolerates small offsets and matches one finger to one note")


if __name__ == "__main__":
    main()
