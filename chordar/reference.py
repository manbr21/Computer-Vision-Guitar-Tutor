"""
Makes the reference photo for marker-free tracking: a photo of the guitar and the four corners of the neck.

    python -m chordar.reference --out references/mine.npz              # photo from the camera (SPACE to take it)
    python -m chordar.reference --image photo.jpg --out references/mine.npz

Then play with:  python -m chordar --reference references/mine.npz
"""
import argparse
import math
import os
import sys
import time

import cv2
import numpy as np

from chordar import config
from chordar.calibration import draw_ghost, ghost_point
from chordar.geometry import Fretboard
from chordar.frets import snap_corners
from chordar.markerless import MarklessTracker, NeckReference
from chordar import ui
from chordar.overlay import draw_neck
from chordar.window import WINDOW, open_window, show, to_picture
from chordar.placement import (MIN_STRING_GAP, alignment_metrics, evaluate, lighting_check, set_reference_center,
                               set_reference_distance)

MIN_VISIBLE_FRETS = 10   # fewer fret wires than this found in the photo: the light is too poor to see the frets
_EDGE = "string" if config.MARKER_PLACEMENT == "outer_strings" else "edge of the fretboard"
# in the order TL, TR, BR, BL, as the neck looks on the (mirrored) screen: nut on the left, low E string on top
PROMPTS = (f"1/4  NUT, at the low E {_EDGE} (top)",
           f"2/4  NUT, at the high e {_EDGE} (bottom)",
           f"3/4  FRET 12 wire, at the high e {_EDGE} (bottom)",
           f"4/4  FRET 12 wire, at the low E {_EDGE} (top)")


def reference_from_clicks(frame, clicks, fret_bounds=None):
    """
    NeckReference from a raw camera frame and the four corners on its mirrored view (x, y in view pixels), and
    optionally the fret positions measured in the photo (see chordar.frets).
    """
    width = frame.shape[1]
    corners = [(width - x, y) for x, y in clicks]  # back to raw camera pixels
    return NeckReference(frame, corners, fret_bounds=fret_bounds)


def photo_hint(clicks, size):
    """
    Advice on how good the reference photo is, from the four clicked corners (on the mirrored view) and the image
    size: the first calibration check that fails (position, level, camera angle), or None if all pass. The distance
    is only questioned when the neck is too small to tell the strings apart (or implausibly large): a comfortable
    playing distance is kept, though the fingers project more precisely the closer the guitar is.
    """
    neck = Fretboard(*[tuple(map(float, c)) for c in clicks], image_size=size)
    metrics = alignment_metrics(neck, size)
    distance_matters = metrics["string_gap"] < MIN_STRING_GAP or not 0.55 <= metrics["scale"] <= 1.6
    return next((c["hint"] for c in evaluate(metrics) if not c["ok"] and (distance_matters or c["label"] != "Distance")), None)


def _text(img, text, y, color=(255, 255, 255)):
    ui.text(img, text, (20, y), 0.7, color, 2)


GUIDE_ORIGIN, GUIDE_SIZE = (20, 112), (500, 200)   # the panel with the big little neck that shows the corner to click: top-left, width x height
# the corners in the order they are clicked, in the neck's own coordinates (u across: 0 low E / top, 1 high e / bottom; t along: 0 nut / left)
CORNER_ORDER = ((0, 0), (1, 0), (1, 1), (0, 1))
CORNER_TEXT = ("the nut, at the thickest string (top)", "the nut, at the thinnest string (bottom)",
               "fret 12 wire, at the thinnest string (bottom)", "fret 12 wire, at the thickest string (top)")


def draw_click_guide(img, step, now=None):
    """
    A see-through neck in a panel at the top-left (nut on the left, low E on top, like the photo, with the position dots: one at
    frets 3, 5, 7, 9 and two at 12, which is how the 12th fret is found) and the corner to click next marked by a pulsing
    ring, the ones already clicked in green. Nothing is marked on the photo itself: a guess of where the corner should be
    on the guitar was often wrong at fret 12, and the user clicks what the photo shows.
    """
    now = time.time() if now is None else now
    pulse = 0.5 + 0.5 * math.sin(now * 6)
    x0, y0 = GUIDE_ORIGIN
    w, h = GUIDE_SIZE
    ui.panel(img, x0, y0, x0 + w, y0 + h, alpha=0.74, radius=12, border=ui.ACCENT_DIM)
    ui.text(img, "WHERE TO CLICK", (x0 + 16, y0 + 26), 0.5, ui.ACCENT, 1, shadow=False)
    ui.progress(img, min(step, 4), 4, x0 + w - 4 * 34 + 12, y0 + 22)
    nx0, nx1, ny0, ny1 = x0 + 84, x0 + w - 52, y0 + 62, y0 + 62 + 76

    def spot(u, t):
        return int(nx0 + t * (nx1 - nx0)), int(ny0 + u * (ny1 - ny0))

    outline = np.array([spot(0, 0), spot(1, 0), spot(1, 1), spot(0, 1)], np.int32)
    layer = img.copy()
    cv2.fillPoly(layer, [outline], (75, 90, 120))
    cv2.addWeighted(layer, 0.45, img, 0.55, 0, dst=img)
    for s in range(6):
        cv2.line(img, spot(s / 5, 0), spot(s / 5, 1), (215, 215, 215), 1, cv2.LINE_AA)
    for n in range(1, 13):
        t = 2 * (1 - 2 ** (-n / 12))
        cv2.line(img, spot(0, t), spot(1, t), ui.WARN if n == 12 else (125, 125, 125), 3 if n == 12 else 1, cv2.LINE_AA)
    for n in (3, 5, 7, 9, 12):
        t_mid = (2 * (1 - 2 ** (-(n - 1) / 12)) + 2 * (1 - 2 ** (-n / 12))) / 2
        for u in ((0.5,) if n != 12 else (0.3, 0.7)):
            cv2.circle(img, spot(u, t_mid), 4, ui.ACCENT_DIM, -1, cv2.LINE_AA)
    cv2.line(img, spot(0, 0), spot(1, 0), (255, 255, 255), 5, cv2.LINE_AA)       # the nut
    cv2.polylines(img, [outline], True, (255, 255, 255), 1, cv2.LINE_AA)
    ui.text(img, "NUT", (nx0 - 14, ny1 + 30), 0.5, (255, 255, 255), 1, shadow=False)
    ui.text(img, "FRET 12", (nx1 - 36, ny1 + 30), 0.5, ui.WARN, 1, shadow=False)
    ui.text(img, "low E", (x0 + 14, ny0 + 5), 0.42, ui.MUTED, 1, shadow=False)
    ui.text(img, "high e", (x0 + 12, ny1 + 4), 0.42, ui.MUTED, 1, shadow=False)
    for i, (u, t) in enumerate(CORNER_ORDER):
        centre = spot(u, t)
        if i < step:
            cv2.circle(img, centre, 8, ui.GOOD, -1, cv2.LINE_AA)
            ui.text(img, str(i + 1), (centre[0] - 4, centre[1] + 5), 0.45, (30, 30, 30), 1, shadow=False)
        elif i == step:
            cv2.circle(img, centre, int(11 + 8 * pulse), ui.ACCENT, 3, cv2.LINE_AA)
            cv2.circle(img, centre, 9, ui.ACCENT, -1, cv2.LINE_AA)
            ui.text(img, str(i + 1), (centre[0] - 4, centre[1] + 5), 0.5, (30, 30, 30), 1, shadow=False)
        else:
            cv2.circle(img, centre, 8, (150, 150, 150), 1, cv2.LINE_AA)
            ui.text(img, str(i + 1), (centre[0] - 4, centre[1] + 5), 0.45, (170, 170, 170), 1, shadow=False)
    if step >= 4:
        ui.text(img, "All four corners marked", (x0 + 16, y0 + h - 14), 0.5, ui.GOOD, 1, shadow=False)
    else:
        ui.text(img, "The ringed corner is the one to click on the photo", (x0 + 16, y0 + h - 14), 0.45, ui.MUTED, 1, shadow=False)


def photo_screen(view, light_hint=None):
    """The 'take the photo' screen drawn on a mirrored camera view: the target outline, the instructions and the keys."""
    draw_ghost(view, False)
    h, w = view.shape[:2]
    ui.header(view, "STEP 1 OF 2", "Take the reference photo", "Hold the guitar as you will play it, with the neck inside the outline")
    ui.panel(view, 20, h - 150, 660, h - 62, alpha=0.7, radius=12)
    for i, (label, line) in enumerate((("NOTEBOOK", "base at thigh height; tilt the screen until the neck fits the outline"),
                                       ("LIGHT", "on the frets, so the wires can be seen"),
                                       ("HAND", "fretting hand off the neck"))):
        ui.text(view, label, (36, h - 122 + 24 * i), 0.5, ui.ACCENT, 1, shadow=False)
        ui.text(view, line, (130, h - 122 + 24 * i), 0.5, ui.TEXT, 1, shadow=False)
    if light_hint:
        ui.pill(view, light_hint, 20, h - 198, ui.WARN, 0.6)
    ui.key_row(view, [("SPACE", "Take the photo"), ("ESC", "Cancel")], 20, h - 46)


def take_photo(capture):
    """Shows the mirrored camera; SPACE takes the photo, ESC cancels. Returns the raw frame or None."""
    while True:
        ok, frame = capture.read()
        if not ok:
            return None
        view = cv2.flip(frame, 1)
        light = lighting_check(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
        photo_screen(view, None if light["ok"] else light["hint"])
        show(WINDOW, view)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            return frame
        if key == 27:
            return None


def click_screen(view, clicks, chosen=None, info=None, hint=None, warned=False, use_snap=True, bounds=None, now=None):
    """
    The 'mark the corners' screen drawn on a copy of the photo: the clicks, then (with four) the fitted neck, what was found
    and the keys. Returns the image to show.
    """
    shown = view.copy()
    h, w = shown.shape[:2]
    for i, (x, y) in enumerate(clicks):
        cv2.circle(shown, (x, y), 5, (0, 0, 0), -1, cv2.LINE_AA)       # small, in the colour of the example clicks, with a dark rim to see it on the wood
        cv2.circle(shown, (x, y), 3, ui.GOOD, -1, cv2.LINE_AA)
        ui.text(shown, str(i + 1), (x + 8, y - 8), 0.6, ui.GOOD, 2)
    if len(clicks) < 4:
        ui.header(shown, "STEP 2 OF 2", "Mark the four corners of the neck", f"Click {len(clicks) + 1} of 4: {CORNER_TEXT[len(clicks)]}")
        draw_click_guide(shown, len(clicks), now)
        ui.key_row(shown, [("CLICK", "Mark the corner"), ("R", "Start again"), ("T", "New photo"), ("ESC", "Cancel")], 20, h - 46)
        return shown
    draw_neck(shown, Fretboard(*[tuple(map(float, c)) for c in chosen], fret_bounds=bounds))
    ui.header(shown, "STEP 2 OF 2", "Check the neck", "The grid should sit on the frets and the strings of the photo" if hint is None
              else "The grid should sit on the frets and the strings; the photo has a problem")
    y = 112
    if hint is None:
        y = ui.pill(shown, "Good photo", 20, y, ui.GOOD) and y + 44
    else:
        text = f"{hint}. ENTER again to save anyway" if warned else f"Not ideal: {hint}"
        ui.pill(shown, text, 20, y, ui.WARN)
        y += 44
    seen = 0 if info is None else info["matched"]
    if info is None:
        ui.pill(shown, "No fret wires found: your clicks are used as they are", 20, y, ui.WARN, 0.55)
    elif seen < MIN_VISIBLE_FRETS:
        ui.pill(shown, f"Only {seen} of 12 frets are visible: more light on the neck, then T for a new photo", 20, y, ui.WARN, 0.55)
    elif use_snap:
        ui.pill(shown, f"Frets fitted to the real wires (nut {info['nut_shift_px']:+.0f} px, off by {info['rms_px']:.1f} px)", 20, y, ui.ACCENT, 0.55)
    else:
        ui.pill(shown, "Using your clicks as they are", 20, y, ui.ACCENT, 0.55)
    keys = [("ENTER", "Save"), ("R", "Click again"), ("U", "Snap to wires on/off"), ("T", "New photo"), ("ESC", "Cancel")]
    ui.panel(shown, 12, h - 58, 12 + ui.row_width(keys) + 24, h - 8, alpha=0.7, radius=10)
    ui.key_row(shown, keys, 24, h - 48)
    return shown


def click_corners(frame):
    """Asks for the four corners on the mirrored photo. Returns the clicks, or None if cancelled."""
    view = cv2.flip(frame, 1)
    clicks = []
    warned = False
    snapped = None     # (corners, info) once four clicks are made
    use_snap = True

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            px, py = to_picture(x, y)   # the window shows the picture scaled, with bars
            clicks.append((int(round(px)), int(round(py))))

    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        chosen, hint, bounds, info = clicks, None, None, None
        if len(clicks) == 4:
            if snapped is None:
                snapped = snap_corners(view, np.array(clicks, np.float32))
            info = snapped[1]
            if use_snap and info is not None:
                chosen = [tuple(map(float, c)) for c in snapped[0]]
                bounds = info["fret_bounds"]
            hint = photo_hint(chosen, (view.shape[1], view.shape[0]))
        shown = click_screen(view, clicks, chosen, info, hint, warned, use_snap, bounds)
        show(WINDOW, shown)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            return None
        if key in (ord("r"), ord("R")):
            clicks.clear()
            warned = False
            snapped = None
        if key in (ord("u"), ord("U")):
            use_snap = not use_snap
        if key in (ord("t"), ord("T")):
            return "retake"
        if key == 13 and len(clicks) == 4:
            if hint is None or warned:
                return chosen, (snapped[1]["fret_bounds"] if use_snap and snapped[1] is not None else None)
            warned = True
        elif key == 13:
            warned = False


def adopt_placement(corners, size):
    """
    Makes the neck of the reference photo (corners on the mirrored view) the reference placement of the playing
    monitor, so that the app only warns when the guitar is moved away from where the photo was taken.
    """
    neck = Fretboard(*[tuple(map(float, c)) for c in corners], image_size=size)
    set_reference_center(neck, size)
    if neck.string_gap_px() >= MIN_STRING_GAP:   # a comfortable distance is kept as long as the strings are apart
        set_reference_distance(neck, size)


def capture_reference(capture, save_to=None):
    """
    The whole set-up in one go: a photo of the guitar held in the target outline, the four clicks on the neck (snapped
    to the real fret wires) and the advice on the photo. Returns a MarklessTracker built on the new reference, or
    None if the user cancelled. The reference is also saved to `save_to`, if given.
    """
    open_window(WINDOW)
    try:
        while True:
            frame = take_photo(capture)
            if frame is None:
                return None
            result = click_corners(frame)
            if result is None:
                return None
            if result == "retake":
                continue
            corners, fret_bounds = result
            reference = reference_from_clicks(frame, corners, fret_bounds)
            try:
                tracker = MarklessTracker(reference)
            except ValueError as error:   # too few features on the neck: show why and take another photo
                shown = cv2.flip(frame, 1)
                _text(shown, str(error), 30, (0, 165, 255))
                _text(shown, "Taking a new photo...", 62)
                show(WINDOW, shown)
                cv2.waitKey(2500)
                continue
            adopt_placement(corners, (frame.shape[1], frame.shape[0]))
            if save_to:
                try:
                    os.makedirs(os.path.dirname(os.path.abspath(save_to)), exist_ok=True)
                    reference.save(save_to)
                except OSError:
                    pass
            return tracker
    finally:
        pass   # the window stays: the app goes on in the same one


def main():
    parser = argparse.ArgumentParser(prog="chordar.reference", description="Make the reference photo for marker-free tracking.")
    parser.add_argument("--out", required=True, help="where to save the reference (.npz)")
    parser.add_argument("--snap", metavar="FILE", help="do not take a photo: snap the corners of an existing reference to its fret wires")
    parser.add_argument("--image", help="use this photo instead of the camera")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    if args.snap:
        reference = NeckReference.load(args.snap)
        corners, info = snap_corners(reference.image, reference.corners)
        if info is None:
            sys.exit("No fret wires found in that photo; the reference is unchanged")
        NeckReference(reference.image, corners, reference.ignore_mask, info["fret_bounds"]).save(args.out)
        print(f"Saved {args.out}: nut moved {info['nut_shift_px']:+.1f} px along the neck, length {info['length_change']:+.1%}, "
              f"{info['matched']} of 12 frets matched to wires, {info['rms_px']:.1f} px off on average"
              + ("" if info["fret_bounds"] else " (the fret spacing looked implausible: the ideal formula is kept)"))
        return
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"Cannot read {args.image}")
    capture = None if args.image else cv2.VideoCapture(args.camera)
    if capture is not None and not capture.isOpened():
        sys.exit(f"Could not open camera {args.camera}")
    open_window(WINDOW)
    while True:
        if capture is not None:
            frame = take_photo(capture)
            if frame is None:
                sys.exit("Cancelled")
        clicks = click_corners(frame)
        if clicks == "retake" and capture is not None:
            continue
        break
    if capture is not None:
        capture.release()
    cv2.destroyAllWindows()
    if clicks is None or clicks == "retake":
        sys.exit("Cancelled")
    corners, fret_bounds = clicks
    reference = reference_from_clicks(frame, corners, fret_bounds)
    try:
        tracker = MarklessTracker(reference)  # fails early if the photo has too few features on the neck
    except ValueError as error:
        sys.exit(str(error))
    reference.save(args.out)
    print(f"Saved {args.out} ({len(tracker._ref_points)} features on the neck region)")


if __name__ == "__main__":
    main()
