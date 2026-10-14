"""
Check of the virtual capo (chordar/chords.py, scoring and the keys of the app).

With a capo on fret N every fretted note of a chord moves up N frets, the capo plays the open strings and
the score expects the shifted frets. Run from the repository root:  python validation/capo_test.py
Exits with status 1 on any mismatch.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app_smoke_test as smoke
from chordar import config
from chordar.app import ChordARApp
from chordar.chords import CHORD_LIBRARY, MAX_CAPO, fretted_notes
from chordar.geometry import FRET_BOUNDS, Fretboard
from chordar.scoring import chord_accuracy


def main():
    failures = []

    def check(label, ok):
        print(("ok    " if ok else "FAIL  ") + label)
        if not ok:
            failures.append(label)

    # --- the notes of a chord ---
    check("no capo: C is the usual shape", fretted_notes("C") == [(1, 3, 3), (2, 2, 2), (4, 1, 1)])
    check("capo 2: every fret of C moves up 2, same strings and fingers",
          fretted_notes("C", 2) == [(1, 5, 3), (2, 4, 2), (4, 3, 1)])
    shifted_ok = all(
        [(s, f - capo, fin) for s, f, fin in fretted_notes(chord, capo)] == fretted_notes(chord)
        and max([f for _, f, _ in fretted_notes(chord, capo)], default=0) <= config.NUM_FRETS
        for chord in CHORD_LIBRARY for capo in range(MAX_CAPO + 1))
    check(f"every chord shifts rigidly and stays on the neck for capo 0-{MAX_CAPO}", shifted_ok)
    check("open and muted strings need no finger with a capo", all(
        len(fretted_notes(chord, 3)) == len(fretted_notes(chord)) for chord in CHORD_LIBRARY))

    # --- the score ---
    neck = Fretboard((100, 100), (100, 200), (900, 200), (900, 100))  # a plain quad, no camera pose needed
    notes_capo2, notes_open = fretted_notes("C", 2), fretted_notes("C")
    fingertips = {name: neck.target(s, fret) for name, (s, fret, _) in zip(("index", "middle", "ring"), notes_capo2)}
    check("fingers on the capo-2 shape score 100% against capo 2",
          chord_accuracy(notes_capo2, fingertips, neck, parallax=False) == 100)
    check("the same fingers score 0% against the no-capo shape",
          chord_accuracy(notes_open, fingertips, neck, parallax=False) == 0)
    fingertips["ring"] = neck.target(1, 6)  # one finger a fret too high
    check("one wrong finger out of three scores 66%", chord_accuracy(notes_capo2, fingertips, neck, parallax=False) == 66)

    # --- the keys and the drawing ---
    config.NECK_CORNER.clear()
    config.NECK_CORNER.update(smoke.MARKER_CORNERS)
    app = ChordARApp(smoke.FakeCapture())
    for _ in range(MAX_CAPO + 3):
        app.handle_key(ord("]"))
    check("] raises the capo up to the limit", app.capo == MAX_CAPO)
    app.handle_key(ord("["))
    check("[ lowers it", app.capo == MAX_CAPO - 1)
    app.handle_key(ord("0"))
    app.handle_key(ord("["))
    check("0 removes the capo and [ stops at 0", app.capo == 0)

    frame = smoke.synthetic_frame()
    for _ in range(3):
        plain = app.process_frame(frame, now=0.0)
    app.capo = 2
    for _ in range(3):
        with_capo = app.process_frame(frame, now=0.0)
    check("the capo changes the picture", not np.array_equal(plain, with_capo))

    _, neck_in_frame = app.markers.track(frame)
    gap = FRET_BOUNDS[2] - FRET_BOUNDS[1]
    x, y = neck_in_frame.to_image(0.5, FRET_BOUNDS[2] - 0.25 * gap)  # middle of the bar, just behind wire 2
    blue, _, red = (int(c) for c in with_capo[y, x])
    check(f"the capo bar is orange just behind the wire of fret 2 (pixel BGR {tuple(with_capo[y, x])})",
          red > 170 and blue < 100)
    x, y = neck_in_frame.to_image(0.5, FRET_BOUNDS[5] - 0.25 * gap)
    blue, _, red = (int(c) for c in with_capo[y, x])
    check("and not at fret 5", not (red > 170 and blue < 100))

    app.hands.close()
    print()
    if failures:
        print("FAIL:", *failures, sep="\n  ")
        sys.exit(1)
    print("PASS: the virtual capo shifts, draws and scores as expected")


if __name__ == "__main__":
    main()
