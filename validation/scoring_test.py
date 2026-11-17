"""
Check of the scoring (chordar/scoring.py): each note is covered by its own finger, read after taking out the shift that
all the fingers share (camera height, angle of the hand), with tolerances on the string and the fret; and the steadying of the
covered flags. Run from the repository root:  python validation/scoring_test.py   Exits with status 1 on any mismatch.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chordar import config
from chordar.chords import fretted_notes
from chordar.geometry import Fretboard
from chordar.scoring import NoteStabilizer, chord_accuracy, covered_notes


def main():
    failures = []

    def check(label, ok):
        print(("ok    " if ok else "FAIL  ") + label)
        if not ok:
            failures.append(label)

    neck = Fretboard((100, 100), (100, 200), (900, 200), (900, 100))   # strings 20 px apart, nut on the left
    gap = 20.0
    notes = fretted_notes("Am")          # (string 2 fret 2, middle), (3, 2, ring), (4, 1, index)
    names = {1: "index", 2: "middle", 3: "ring", 4: "pinky"}

    def tips(dstring=None, dx=None, all_strings=0.0, swap=False):
        """Every finger on its note, with per-finger shifts in strings (dstring) or pixels along the neck (dx)."""
        out = {}
        for s, f, finger in notes:
            x, y = neck.target(s, f)
            name = names[finger]
            out[name] = (x + (dx or {}).get(name, 0), y + gap * ((dstring or {}).get(name, 0) + all_strings))
        if swap:
            out["middle"], out["ring"] = out["ring"], out["middle"]
        return out

    def score(t):
        return chord_accuracy(notes, t, neck, False)

    check("fingers on their notes score 100%", score(tips()) == 100)
    check("a finger 0.4 string off still covers its note", score(tips({"index": -0.4})) == 100)
    check("a finger 0.9 string off does not (66%)", score(tips({"index": -0.9})) == 66)
    check("a finger a whole string off does not (66%)", score(tips({"index": -1.0})) == 66)

    space_px = (neck.fret_bounds[2] - neck.fret_bounds[1]) * 800
    check("a finger a whole fret off does not (66%): nothing along the neck is shared", score(tips(dx={"middle": space_px})) == 66)
    x_wire = neck.to_image(0.5, neck.fret_bounds[2])[0] - neck.target(2, 2)[0]
    check("a finger just behind the fret wire still covers the fret", score(tips(dx={"middle": x_wire + 0.15 * space_px})) == 100)

    check("the shared shift is taken out: all the fingers 0.4 string off (another camera height) still score 100%",
          score(tips(all_strings=0.4)) == 100)
    check("... up to a limit: all the fingers 1.2 strings off do not", score(tips(all_strings=1.2)) == 0)
    check("each finger belongs on its own note: two fingers swapped score 33%", score(tips(swap=True)) == 33)

    missing = {k: v for k, v in tips().items() if k != "ring"}
    check("a finger that is not in view leaves its note uncovered", covered_notes(notes, missing, neck, False) == [True, False, True])

    # two fingers: the shared shift is limited to 0.4 so that it cannot hide an error
    em = fretted_notes("Em")
    two = {names[finger]: neck.target(s, f) for s, f, finger in em}
    check("a two-note chord scores 100% on target", chord_accuracy(em, two, neck, False) == 100)
    moved = dict(two)
    moved["index"] = (two["index"][0], two["index"][1] + gap)
    check("a two-note chord with one finger a whole string off does not score 100%", chord_accuracy(em, moved, neck, False) < 100)

    # what is drawn is what is scored: a finger that counts is drawn within the tolerance of its note, a finger that does not is not
    from chordar.scoring import corrected_points
    shifted = tips(all_strings=0.4)                          # every finger read 0.4 string off, as with another camera height
    drawn = corrected_points(notes, shifted, neck, False)
    on_note = all(abs(drawn[names[finger]][1] - neck.target(s, f)[1]) < 0.1 * gap for s, f, finger in notes)
    check("with the shared shift taken out the drawn fingers are on their notes", on_note)
    off_one = corrected_points(notes, tips({"index": -0.9}), neck, False)
    check("a finger that is really off is drawn off (more than the tolerance away from its note)",
          abs(off_one["index"][1] - neck.target(4, 1)[1]) > 0.5 * gap)

    # the reading bias: fingers the tracker reads 0.6 of a string off still cover its note once that is taken out
    d_notes = fretted_notes("D")                      # G2 index (string 3), B3 ring (4), e2 middle (5)
    d_tips = {names[finger]: neck.target(s, f) for s, f, finger in d_notes}
    d_tips["middle"] = (d_tips["middle"][0], d_tips["middle"][1] - 0.62 * gap)         # read 0.6 string toward the bass
    d_tips["index"] = (d_tips["index"][0], d_tips["index"][1] + 0.56 * gap)            # and the index 0.55 of a string low
    check("without the bias a finger read that far off is not covered", covered_notes(d_notes, d_tips, neck, False) != [True, True, True])
    from chordar.chords import reading_bias
    check("with the bias of the D chord all its notes are covered", covered_notes(d_notes, d_tips, neck, False, reading_bias("D")) == [True, True, True])
    check("a chord without a bias is read as it is", reading_bias("Am") == {})

    # the steadying of the covered flags
    st = NoteStabilizer(3, 4)
    shown = [st.update(x)[0] for x in [[True]] * 2 + [[False]] + [[True]] * 3]
    check(f"a note turns green only after 3 covered frames in a row ({shown})", shown == [False, False, False, False, False, True])
    shown = [st.update([False])[0] for _ in range(4)]
    check(f"... and turns off only after 4 missed frames in a row ({shown})", shown == [True, True, True, False])
    st = NoteStabilizer(3, 4)
    check("a note that flickers every frame never turns green", not any(st.update([i % 2 == 0])[0] for i in range(20)))
    st.update([True, True, True])
    check("a different chord (other number of notes) starts again", st.update([True, True]) == [False, False])

    print()
    if failures:
        print("FAIL:", *failures, sep="\n  ")
        sys.exit(1)
    print("PASS: each finger is judged on its own note, with the shared shift taken out")


if __name__ == "__main__":
    main()
