"""Guided setup screen: a target neck is drawn on the video and the user moves the guitar onto it."""
import math

import cv2
import numpy as np

from chordar import config
from chordar.geometry import FRET_BOUNDS, outline_u_range, string_u
from chordar.overlay import draw_neck, put_text_bg
from chordar.placement import (IDEAL_CENTER, alignment_metrics, evaluate, pixels_per_mm, smooth)

STABLE_FRAMES_REQUIRED = 20  # ~a second of good, uninterrupted framing before auto-advancing

GOOD_COLOR, WARN_COLOR, BAD_COLOR = (0, 255, 0), (0, 165, 255), (0, 0, 255)

FRET_INLAYS = (3, 5, 7, 9)   # single position dots; fret 12 gets two
STRING_COLORS = [(70, 150, 200)] * 3 + [(225, 225, 225)] * 3  # wound bass strings in bronze, nylon trebles in white
STRING_THICKNESS = [3, 3, 2, 2, 1, 1]


def ghost_point(u, t, image_size):
    """Pixel of neck coordinates (u, t) on the target neck."""
    width, height = image_size
    scale = pixels_per_mm(width)
    width_mm = config.QUAD_NUT_WIDTH_MM + (config.QUAD_FRET12_WIDTH_MM - config.QUAD_NUT_WIDTH_MM) * t
    x = IDEAL_CENTER[0] * width + (t - 0.5) * config.NECK_LENGTH_MM * scale
    y = IDEAL_CENTER[1] * height + (u - 0.5) * width_mm * scale
    return int(round(x)), int(round(y))


def draw_ghost(display, aligned):
    """Draws the target neck like a real fretboard (wood, nut, frets, inlays, strings) where the guitar should be."""
    size = (display.shape[1], display.shape[0])
    accent = GOOD_COLOR if aligned else (235, 235, 235)
    u_min, u_max = outline_u_range()

    def point(u, t):
        return ghost_point(u, t, size)

    outline = np.array([point(u_min, 0), point(u_max, 0), point(u_max, 1), point(u_min, 1)], dtype=np.int32)
    shade = display.copy()
    cv2.fillPoly(shade, [outline], (30, 50, 90))  # dark rosewood, see-through
    cv2.addWeighted(shade, 0.5, display, 0.5, 0, dst=display)

    for n in FRET_INLAYS + (12,):
        t_mid = (FRET_BOUNDS[n - 1] + FRET_BOUNDS[n]) / 2
        radius = max(int(0.3 * math.dist(point(string_u(1), t_mid), point(string_u(0), t_mid))), 3)
        for u in ((0.5,) if n != 12 else (0.3, 0.7)):
            cv2.circle(display, point(u, t_mid), radius, (205, 215, 220), -1)

    for n in range(1, config.NUM_FRETS + 1):
        cv2.line(display, point(u_min, FRET_BOUNDS[n]), point(u_max, FRET_BOUNDS[n]), (200, 200, 200), 1)
    cv2.line(display, point(u_min, 0), point(u_max, 0), (225, 235, 240), 4)  # nut
    for s in range(config.NUM_STRINGS):
        cv2.line(display, point(string_u(s), 0), point(string_u(s), 1), STRING_COLORS[s], STRING_THICKNESS[s])

    cv2.polylines(display, [outline], True, accent, 2)
    label_x, label_y = outline[0]
    put_text_bg(display, "Place the neck inside this outline", (int(label_x), max(int(label_y) - 12, 20)),
                0.5, accent, thickness=1)


def calibration_step(frame, markers, hands, history):
    """
    Processes one camera frame: detects the neck and hand, draws the target neck, the message and the
    checklist. Returns (display image, ready), where ready means the neck is aligned and a hand is detected.
    history carries the smoothing state between frames.
    """
    display, neck = markers.track(frame)
    size = (display.shape[1], display.shape[0])
    hand_ok = bool(hands.track(frame, neck.to_image(0.5, 0.2) if neck is not None else None))

    checks = []
    if neck is not None:
        draw_neck(display, neck)
        checks = evaluate(smooth(history, alignment_metrics(neck, size)))
    else:
        history.clear()
    geometry_ok = neck is not None and all(c["ok"] for c in checks)

    if neck is None:
        message, color = "Move the guitar so all 4 ArUco markers (IDs 0-3) are visible", BAD_COLOR
    elif not geometry_ok:
        message, color = next(c["hint"] for c in checks if not c["ok"]), WARN_COLOR
    elif not hand_ok:
        message, color = "Neck aligned - place your fretting hand on the neck", WARN_COLOR
    else:
        message, color = "Good to go!", GOOD_COLOR

    draw_ghost(display, geometry_ok)
    put_text_bg(display, "CALIBRATION", (20, 30), 0.9, (255, 255, 255))
    put_text_bg(display, message, (20, 65), 0.7, color)
    rows = [(c["label"], c["ok"], c["detail"]) for c in checks]
    rows.append(("Hand", hand_ok, "detected" if hand_ok else "not detected"))
    for i, (label, ok, detail) in enumerate(rows):
        put_text_bg(display, f"{label:<13}{'OK' if ok else '--'}  {detail}", (20, 130 + 26 * i), 0.5,
                    GOOD_COLOR if ok else WARN_COLOR, thickness=1)
    return display, geometry_ok and hand_ok


def run_calibration(capture, markers, hands, window):
    """
    Shows the calibration screen until everything looks good for STABLE_FRAMES_REQUIRED consecutive frames, or
    until ENTER (start anyway). Returns False if the user quits with ESC or the camera stops, True otherwise.
    """
    stable_frames = 0
    history = {}

    while True:
        ok, frame = capture.read()
        if not ok:
            return False

        display, ready = calibration_step(frame, markers, hands, history)
        stable_frames = stable_frames + 1 if ready else 0
        put_text_bg(display, f"Stable: {min(stable_frames, STABLE_FRAMES_REQUIRED)}/{STABLE_FRAMES_REQUIRED}",
                    (20, 95), 0.6, (255, 255, 255), thickness=1)
        put_text_bg(display, "Press ENTER to start anyway, ESC to quit", (20, display.shape[0] - 20),
                    0.5, (255, 255, 255), thickness=1)
        cv2.imshow(window, display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:    # ESC
            return False
        if key == 13 or stable_frames >= STABLE_FRAMES_REQUIRED:  # ENTER skips the wait
            return True
