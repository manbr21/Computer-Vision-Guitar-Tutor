"""
Check of the set-up screens and the key hints (chordar/ui.py, reference.py, app.py): every screen draws on a frame of any size without
errors, the instructions are on it, the key bar is always on the playing screen and the help panel stays above the neck.
Run from the repository root:  python validation/ui_test.py   Exits with status 1 on any mismatch.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chordar.app import CHORD_NAMES, HELP_SECTIONS, KEY_BAR, ChordARApp
from chordar.chords import MAX_CAPO
from chordar.reference import click_screen, photo_screen


def main():
    failures = []

    def check(label, ok):
        print(("ok    " if ok else "FAIL  ") + label)
        if not ok:
            failures.append(label)

    for size in ((1280, 720), (960, 540), (1920, 1080)):
        w, h = size
        base = np.full((h, w, 3), 90, np.uint8)
        shot = base.copy()
        photo_screen(shot, "Dim light")
        check(f"{w}x{h}: the photo screen draws the header, the tips and the keys", float(np.abs(shot.astype(int) - 90).mean()) > 1)
        clicks = [(300, 400), (300, 480), (900, 490), (900, 410)]
        for n in (0, 2):
            shown = click_screen(base, clicks[:n])
            check(f"{w}x{h}: the click screen with {n} clicks draws", shown.shape == base.shape and not np.array_equal(shown, base))
        shown = click_screen(base, clicks, clicks, None, "Move the guitar closer to the camera")
        check(f"{w}x{h}: the check screen draws (no fret wires found)", not np.array_equal(shown, base))

    check("the capo goes up to fret 4 only (the fingers are read poorly above it)", MAX_CAPO == 4)
    check("the help lists the keys of every chord", len(CHORD_NAMES) == 8)
    listed = {key for _, items in HELP_SECTIONS for key, _ in items}
    check("the help lists the capo, zoom, neck lines, auto-advance, new photo and quit keys", {"[", "]", "0", "Z", "O", "A", "C", "H", "ESC"} <= listed)
    check("the key bar names the chord, capo, zoom and help keys", [k for k, _ in KEY_BAR] == ["1-8", "[ ]", "Z", "H"])

    app = ChordARApp(None)   # the default (marker) tracker finds nothing in a plain frame: only the screen text is drawn
    frame = np.full((720, 1280, 3), 90, np.uint8)
    plain = app.process_frame(frame, 0.0)
    check("the playing screen has the key bar at the bottom left", float(np.abs(plain[670:710, 12:400].astype(int) - 90).mean()) > 2)
    app.show_help = True
    helped = app.process_frame(frame, 0.0)
    check("the help panel opens at the bottom left, over the key bar (nothing changes above 60% of the height)",
          np.array_equal(helped[:430], plain[:430]) and float(np.abs(helped[480:710, 12:1000].astype(int) - plain[480:710, 12:1000]).mean()) > 5)
    from chordar.overlay import draw_arrow
    a0, a1 = np.full((200, 400, 3), 90, np.uint8), np.full((200, 400, 3), 90, np.uint8)
    draw_arrow(a0, (50, 100), (300, 100), 10, now=0.0)
    draw_arrow(a1, (50, 100), (300, 100), 10, now=0.05)
    check("the arrow is animated (its dashes move with time)", not np.array_equal(a0, a1))
    check("... and still points at the note (the head is near the end)", float(np.abs(a0[90:110, 280:292].astype(int) - 90).mean()) > 20)
    from chordar import config
    from chordar.window import scale_up
    check("the windows are bigger than the picture by config.WINDOW_SCALE, keeping its proportions",
          config.WINDOW_SCALE > 1.0 and scale_up(np.zeros((720, 1280, 3), np.uint8)).shape[:2] == (round(720 * config.WINDOW_SCALE), round(1280 * config.WINDOW_SCALE)))
    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("\nPASS: the set-up screens, the key bar and the help draw and keep clear of the neck")


if __name__ == "__main__":
    main()
