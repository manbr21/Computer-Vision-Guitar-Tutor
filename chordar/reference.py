"""
Makes the reference photo for marker-free tracking: a photo of the guitar and the four corners of the neck.

    python -m chordar.reference --out references/mine.npz              # photo from the camera (SPACE to take it)
    python -m chordar.reference --image photo.jpg --out references/mine.npz

Then play with:  python -m chordar --reference references/mine.npz
"""
import argparse
import os
import sys

import cv2
import numpy as np

from chordar import config
from chordar.calibration import draw_ghost
from chordar.geometry import Fretboard
from chordar.frets import snap_corners
from chordar.markerless import MarklessTracker, NeckReference
from chordar.overlay import draw_neck
from chordar.placement import (MIN_STRING_GAP, alignment_metrics, evaluate, lighting_check, set_reference_center,
                               set_reference_distance)

WINDOW = "ChordAR reference"
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
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def take_photo(capture):
    """Shows the mirrored camera; SPACE takes the photo, ESC cancels. Returns the raw frame or None."""
    while True:
        ok, frame = capture.read()
        if not ok:
            return None
        view = cv2.flip(frame, 1)
        draw_ghost(view, False)
        _text(view, "Put the neck inside the outline, as you will play it, hand off the neck", 30)
        _text(view, "SPACE: take the photo   ESC: cancel", 62)
        light = lighting_check(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
        if not light["ok"]:
            _text(view, light["hint"], 94, (0, 165, 255))
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            return frame
        if key == 27:
            return None


def click_corners(frame):
    """Asks for the four corners on the mirrored photo. Returns the clicks, or None if cancelled."""
    view = cv2.flip(frame, 1)
    clicks = []
    warned = False
    snapped = None     # (corners, info) once four clicks are made
    use_snap = True

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))

    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        shown = view.copy()
        for i, (x, y) in enumerate(clicks):
            cv2.circle(shown, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(shown, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        chosen, hint = clicks, None
        if len(clicks) == 4:
            if snapped is None:
                snapped = snap_corners(view, np.array(clicks, np.float32))
            if use_snap and snapped[1] is not None:
                chosen = [tuple(map(float, c)) for c in snapped[0]]
            hint = photo_hint(chosen, (view.shape[1], view.shape[0]))
            bounds = snapped[1]["fret_bounds"] if use_snap and snapped[1] is not None else None
            draw_neck(shown, Fretboard(*[tuple(map(float, c)) for c in chosen], fret_bounds=bounds))
            info = snapped[1]
            if info is None or info["matched"] < MIN_VISIBLE_FRETS:
                seen = 0 if info is None else info["matched"]
                _text(shown, f"Only {seen} of 12 frets are visible: more light on the neck, then T for a new photo", 126, (0, 165, 255))
            if info is None:
                _text(shown, "No fret wires found: using your clicks as they are", 94, (0, 165, 255))
            elif use_snap:
                _text(shown, f"Frets fitted to the real wires (nut {info['nut_shift_px']:+.0f} px, off by {info['rms_px']:.1f} px). U: use my clicks", 94, (255, 255, 0))
            else:
                _text(shown, "Using your clicks. U: snap to the real fret wires", 94, (255, 255, 0))
            if hint is None:
                _text(shown, "Good photo. ENTER: save, R: click again, T: new photo, ESC: cancel", 30, (0, 255, 0))
            elif warned:
                _text(shown, f"{hint}. ENTER again: save anyway, T: new photo", 30, (0, 165, 255))
            else:
                _text(shown, f"Not ideal: {hint}", 30, (0, 165, 255))
                _text(shown, "T: new photo, R: click again, ENTER: save anyway", 62, (0, 165, 255))
        else:
            _text(shown, "Click: " + PROMPTS[len(clicks)], 30, (0, 255, 255))
        cv2.imshow(WINDOW, shown)
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
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
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
                cv2.imshow(WINDOW, shown)
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
        cv2.destroyWindow(WINDOW)
        cv2.waitKey(1)


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
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
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
