import math
from collections import deque

import cv2
import numpy as np

import map_fret_board as mfb
from map_fret_board import map_guitar
from map_hands import get_fingertip_positions
from ui_utils import put_text_bg

MIN_STRING_GAP = 15          # px between adjacent strings; below this, string discrimination gets unreliable
MAX_TILT_DEG = 40            # camera angle away from the fretboard normal beyond which detection degrades
STABLE_FRAMES_REQUIRED = 20  # ~a second of good, uninterrupted framing before auto-advancing

# Where the neck should sit in the (mirrored) image: seen by a camera looking straight at it from
# IDEAL_DISTANCE_MM, centered horizontally and a bit below the middle, nut on the left.
IDEAL_DISTANCE_MM = 400.0
IDEAL_CENTER = (0.5, 0.55)
SCALE_TOL = 0.15             # allowed relative error of the neck length vs. the target
OFFSET_TOL_X = 0.07          # allowed horizontal offset, as a fraction of the frame width
OFFSET_TOL_Y = 0.10          # allowed vertical offset, as a fraction of the frame height
ROLL_TOL_DEG = 12            # allowed rotation of the neck axis away from horizontal
SMOOTH_FRAMES = 5
PLAY_TOLERANCE_SCALE = 1.5   # during play the tolerances are 50% looser than in calibration
PLAY_WARNING_DELAY_S = 2.0   # seconds out of tolerance before the warning appears

GOOD_COLOR = (0, 255, 0)
WARN_COLOR = (0, 165, 255)
BAD_COLOR = (0, 0, 255)


def _pixels_per_mm(image_width):
    focal = (image_width / 2) / math.tan(math.radians(mfb.CAMERA_HFOV_DEG) / 2)
    return focal / IDEAL_DISTANCE_MM


def ghost_point(u, t, image_size):
    """Pixel of neck coordinates (u, t) on the target neck."""
    w, h = image_size
    k = _pixels_per_mm(w)
    width_mm = mfb.QUAD_NUT_WIDTH_MM + (mfb.QUAD_FRET12_WIDTH_MM - mfb.QUAD_NUT_WIDTH_MM) * t
    x = IDEAL_CENTER[0] * w + (t - 0.5) * mfb.NECK_LENGTH_MM * k
    y = IDEAL_CENTER[1] * h + (u - 0.5) * width_mm * k
    return int(round(x)), int(round(y))


def alignment_metrics(neck, image_size):
    """How the detected neck differs from the target one: size, offsets, rotation and camera angle."""
    w, h = image_size
    nut = np.array(neck.to_image(0.5, 0.0), dtype=float)
    far = np.array(neck.to_image(0.5, 1.0), dtype=float)
    center = (nut + far) / 2
    roll = math.degrees(math.atan2(far[1] - nut[1], far[0] - nut[0]))
    metrics = {
        "scale": float(np.linalg.norm(far - nut) / (mfb.NECK_LENGTH_MM * _pixels_per_mm(w))),
        "dx": float((center[0] - IDEAL_CENTER[0] * w) / w),
        "dy": float((center[1] - IDEAL_CENTER[1] * h) / h),
        "roll": (roll + 90) % 180 - 90,  # the same whichever side the nut is on
        "string_gap": neck.string_gap_px(),
    }
    angles = neck.camera_angles()
    if angles is not None:
        metrics.update(tilt=angles[0], across=angles[1], along=angles[2])
    return metrics


def evaluate(m, tolerance_scale=1.0):
    """
    Checks in priority order: list of dicts with label, ok, detail and the hint to show when not ok.
    tolerance_scale loosens every tolerance (1.0 for calibration, more during play).
    """
    checks = []
    scale_tol, tol_x, tol_y = SCALE_TOL * tolerance_scale, OFFSET_TOL_X * tolerance_scale, OFFSET_TOL_Y * tolerance_scale
    roll_tol, max_tilt = ROLL_TOL_DEG * tolerance_scale, MAX_TILT_DEG * tolerance_scale

    too_far = m["scale"] > 1 + scale_tol
    too_close = m["scale"] < 1 - scale_tol or m["string_gap"] < MIN_STRING_GAP
    checks.append(dict(
        label="Distance", ok=not too_far and not too_close, detail=f"neck at {m['scale'] * 100:.0f}% of target size",
        hint="Move the guitar farther from the camera" if too_far else "Move the guitar closer to the camera"))

    checks.append(dict(
        label="Horizontal", ok=abs(m["dx"]) <= tol_x, detail=f"offset {m['dx'] * 100:+.0f}%",
        hint="Move the guitar to the left of the screen" if m["dx"] > 0 else "Move the guitar to the right of the screen"))

    checks.append(dict(
        label="Vertical", ok=abs(m["dy"]) <= tol_y, detail=f"offset {m['dy'] * 100:+.0f}%",
        hint="Move the guitar up on the screen" if m["dy"] > 0 else "Move the guitar down on the screen"))

    checks.append(dict(
        label="Level", ok=abs(m["roll"]) <= roll_tol, detail=f"rotated {m['roll']:+.0f} deg",
        hint=f"Level the neck: it is rotated {m['roll']:.0f} deg"))

    if "tilt" in m:
        if m["across"] >= m["along"]:
            tilt_hint = (f"Camera tilted {m['across']:.0f} deg across the neck (max {max_tilt:.0f}) "
                         "- change the screen tilt or the guitar height")
        else:
            tilt_hint = (f"Neck angled {m['along']:.0f} deg along its length (max {max_tilt:.0f}) "
                         "- turn the guitar to face the screen")
        checks.append(dict(label="Camera angle", ok=m["tilt"] <= max_tilt,
                           detail=f"{m['tilt']:.0f} deg (across {m['across']:.0f}, along {m['along']:.0f})", hint=tilt_hint))
    return checks


def _smooth(history, metrics):
    for key, value in metrics.items():
        history.setdefault(key, deque(maxlen=SMOOTH_FRAMES)).append(value)
    return {key: float(np.mean(values)) for key, values in history.items() if key in metrics}


class PlacementMonitor:
    """
    Watches the neck placement during play. update() returns the hint of the first failing check
    once the neck has stayed out of the (loosened) tolerance for PLAY_WARNING_DELAY_S seconds, else
    None. A neck that is not detected is left to the marker message in main.py.
    """

    def __init__(self, tolerance_scale=PLAY_TOLERANCE_SCALE, delay_s=PLAY_WARNING_DELAY_S):
        self._tolerance_scale = tolerance_scale
        self._delay_s = delay_s
        self.reset()

    def reset(self):
        self._history = {}
        self._out_since = None

    def update(self, neck, image_size, now):
        if neck is None:
            self.reset()
            return None
        checks = evaluate(_smooth(self._history, alignment_metrics(neck, image_size)), self._tolerance_scale)
        failing = next((c for c in checks if not c["ok"]), None)
        if failing is None:
            self._out_since = None
            return None
        if self._out_since is None:
            self._out_since = now
        return failing["hint"] if now - self._out_since >= self._delay_s else None


FRET_INLAYS = (3, 5, 7, 9)   # single position dots; fret 12 gets two
STRING_COLORS = [(70, 150, 200)] * 3 + [(225, 225, 225)] * 3  # wound bass strings in bronze, nylon trebles in white
STRING_THICKNESS = [3, 3, 2, 2, 1, 1]


def _blend_polygon(img, polygon, color, alpha):
    shade = img.copy()
    cv2.fillPoly(shade, [np.array(polygon, dtype=np.int32)], color)
    cv2.addWeighted(shade, alpha, img, 1 - alpha, 0, dst=img)


def draw_ghost(display, aligned):
    """Draws the target neck like a real fretboard (wood, nut, frets, inlays, strings) where the guitar should be."""
    size = (display.shape[1], display.shape[0])
    accent = GOOD_COLOR if aligned else (235, 235, 235)
    u_min, u_max = mfb.outline_u_range()

    def point(u, t):
        return ghost_point(u, t, size)

    outline = [point(u_min, 0), point(u_max, 0), point(u_max, 1), point(u_min, 1)]
    _blend_polygon(display, outline, (30, 50, 90), 0.5)  # dark rosewood, see-through

    bounds = mfb.FRET_BOUNDS
    for n in FRET_INLAYS + (12,):
        t_mid = (bounds[n - 1] + bounds[n]) / 2
        radius = max(int(0.3 * math.dist(point(mfb.string_u(1), t_mid), point(mfb.string_u(0), t_mid))), 3)
        for u in ((0.5,) if n != 12 else (0.3, 0.7)):
            cv2.circle(display, point(u, t_mid), radius, (205, 215, 220), -1)

    for n in range(1, mfb.NUM_FRETS + 1):
        cv2.line(display, point(u_min, bounds[n]), point(u_max, bounds[n]), (200, 200, 200), 1)
    cv2.line(display, point(u_min, 0), point(u_max, 0), (225, 235, 240), 4)  # nut
    for s in range(mfb.NUM_STRINGS):
        u = mfb.string_u(s)
        cv2.line(display, point(u, 0), point(u, 1), STRING_COLORS[s], STRING_THICKNESS[s])

    cv2.polylines(display, [np.array(outline, dtype=np.int32)], True, accent, 2)
    label_x, label_y = outline[0]
    put_text_bg(display, "Place the neck inside this outline", (label_x, max(label_y - 12, 20)), 0.5, accent, thickness=1)


def calibration_step(frame, history):
    """
    Processes one camera frame: detects the neck and hand, draws the target neck, the message and
    the checklist. Returns (display image, ready), where ready means the neck is aligned and a hand
    is detected. history carries the smoothing state between frames.
    """
    display, neck = map_guitar(frame)
    size = (display.shape[1], display.shape[0])
    anchor = neck.to_image(0.5, 0.2) if neck is not None else None
    _, fingertips, _ = get_fingertip_positions(frame.copy(), anchor)
    hand_ok = bool(fingertips)

    checks = []
    if neck is not None:
        checks = evaluate(_smooth(history, alignment_metrics(neck, size)))
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


def run_calibration(cap):
    """
    Guided setup screen shown before the main tracking loop. A target neck is drawn on the
    video; the user moves the guitar (or the camera) until the detected neck matches it. Each
    frame checks distance, horizontal and vertical position, level and camera angle, plus the
    ArUco markers and the fretting hand, and shows one actionable message with a checklist.

    Advances automatically once everything looks good for STABLE_FRAMES_REQUIRED
    consecutive frames, or immediately on ENTER (manual skip). Returns False if
    the user quits (ESC) during calibration, True otherwise.
    """
    stable_frames = 0
    history = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            return False

        display, ready = calibration_step(frame, history)
        stable_frames = stable_frames + 1 if ready else 0

        put_text_bg(display, f"Stable: {min(stable_frames, STABLE_FRAMES_REQUIRED)}/{STABLE_FRAMES_REQUIRED}",
                    (20, 95), 0.6, (255, 255, 255), thickness=1)
        put_text_bg(display, "Press ENTER to start anyway, ESC to quit", (20, display.shape[0] - 20),
                    0.5, (255, 255, 255), thickness=1)

        cv2.imshow("Hand + Guitar Tracking", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            return False
        if key == 13:  # ENTER - skip calibration and start anyway
            return True

        if stable_frames >= STABLE_FRAMES_REQUIRED:
            return True
