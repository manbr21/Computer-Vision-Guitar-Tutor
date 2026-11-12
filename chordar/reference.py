"""
Makes the reference photo for marker-free tracking: a photo of the guitar and the four corners of the neck.

    python -m chordar.reference --out references/mine.npz              # photo from the camera (SPACE to take it)
    python -m chordar.reference --image photo.jpg --out references/mine.npz

Then play with:  python -m chordar --reference references/mine.npz
"""
import argparse
import sys

import cv2
import numpy as np

from chordar import config
from chordar.calibration import draw_ghost
from chordar.geometry import Fretboard
from chordar.markerless import MarklessTracker, NeckReference
from chordar.placement import alignment_metrics, evaluate

WINDOW = "ChordAR reference"
_EDGE = "string" if config.MARKER_PLACEMENT == "outer_strings" else "edge of the fretboard"
# in the order TL, TR, BR, BL, as the neck looks on the (mirrored) screen: nut on the left, low E string on top
PROMPTS = (f"1/4  NUT, at the low E {_EDGE} (top)",
           f"2/4  NUT, at the high e {_EDGE} (bottom)",
           f"3/4  FRET 12 wire, at the high e {_EDGE} (bottom)",
           f"4/4  FRET 12 wire, at the low E {_EDGE} (top)")


def reference_from_clicks(frame, clicks):
    """NeckReference from a raw camera frame and the four clicks made on its mirrored view (x, y in view pixels)."""
    width = frame.shape[1]
    corners = [(width - x, y) for x, y in clicks]  # back to raw camera pixels
    return NeckReference(frame, corners)


def photo_hint(clicks, size):
    """
    Advice on how good the reference photo is, from the four clicked corners (on the mirrored view) and the image
    size: the first calibration check that fails (distance, position, level, camera angle), or None if all pass.
    A reference taken at the calibration distance makes the fingers project more precisely.
    """
    neck = Fretboard(*[tuple(map(float, c)) for c in clicks], image_size=size)
    return next((c["hint"] for c in evaluate(alignment_metrics(neck, size)) if not c["ok"]), None)


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

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))

    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        shown = view.copy()
        for i, (x, y) in enumerate(clicks):
            cv2.circle(shown, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(shown, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        hint = photo_hint(clicks, (view.shape[1], view.shape[0])) if len(clicks) == 4 else None
        if len(clicks) == 4:
            cv2.polylines(shown, [np.array(clicks, np.int32)], True, (0, 255, 0) if hint is None else (0, 165, 255), 2)
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
        if key in (ord("t"), ord("T")):
            return "retake"
        if key == 13 and len(clicks) == 4:
            if hint is None or warned:
                return clicks
            warned = True
        elif key == 13:
            warned = False


def main():
    parser = argparse.ArgumentParser(prog="chordar.reference", description="Make the reference photo for marker-free tracking.")
    parser.add_argument("--out", required=True, help="where to save the reference (.npz)")
    parser.add_argument("--image", help="use this photo instead of the camera")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

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
        if clicks != "retake" or capture is None:
            break
    if capture is not None:
        capture.release()
    cv2.destroyAllWindows()
    if clicks is None or clicks == "retake":
        sys.exit("Cancelled")
    reference = reference_from_clicks(frame, clicks)
    try:
        tracker = MarklessTracker(reference)  # fails early if the photo has too few features on the neck
    except ValueError as error:
        sys.exit(str(error))
    reference.save(args.out)
    print(f"Saved {args.out} ({len(tracker._ref_points)} features on the neck region)")


if __name__ == "__main__":
    main()
