"""How well the neck is placed in front of the camera: metrics, checks and a monitor for use during play."""
import math
from collections import deque

import numpy as np

from chordar import config

MIN_STRING_GAP = 15          # px between adjacent strings; below this, string discrimination gets unreliable
MAX_TILT_DEG = 40            # camera angle away from the fretboard normal beyond which detection degrades

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


def pixels_per_mm(image_width):
    """Scale of the target neck: pixels per mm for a camera looking straight at it from IDEAL_DISTANCE_MM."""
    focal = (image_width / 2) / math.tan(math.radians(config.CAMERA_HFOV_DEG) / 2)
    return focal / IDEAL_DISTANCE_MM


def alignment_metrics(neck, image_size):
    """How the detected neck differs from the target one: size, offsets, rotation and camera angle."""
    width, height = image_size
    nut = np.array(neck.to_image(0.5, 0.0), dtype=float)
    far = np.array(neck.to_image(0.5, 1.0), dtype=float)
    center = (nut + far) / 2
    roll = math.degrees(math.atan2(far[1] - nut[1], far[0] - nut[0]))
    metrics = {
        "scale": float(np.linalg.norm(far - nut) / (config.NECK_LENGTH_MM * pixels_per_mm(width))),
        "dx": float((center[0] - IDEAL_CENTER[0] * width) / width),
        "dy": float((center[1] - IDEAL_CENTER[1] * height) / height),
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
    scale_tol, tol_x, tol_y = (tol * tolerance_scale for tol in (SCALE_TOL, OFFSET_TOL_X, OFFSET_TOL_Y))
    roll_tol, max_tilt = ROLL_TOL_DEG * tolerance_scale, MAX_TILT_DEG * tolerance_scale

    too_far = m["scale"] > 1 + scale_tol
    too_close = m["scale"] < 1 - scale_tol or m["string_gap"] < MIN_STRING_GAP
    checks = [
        dict(label="Distance", ok=not too_far and not too_close,
             detail=f"neck at {m['scale'] * 100:.0f}% of target size",
             hint="Move the guitar farther from the camera" if too_far else "Move the guitar closer to the camera"),
        dict(label="Horizontal", ok=abs(m["dx"]) <= tol_x, detail=f"offset {m['dx'] * 100:+.0f}%",
             hint="Move the guitar to the left of the screen" if m["dx"] > 0
             else "Move the guitar to the right of the screen"),
        dict(label="Vertical", ok=abs(m["dy"]) <= tol_y, detail=f"offset {m['dy'] * 100:+.0f}%",
             hint="Move the guitar up on the screen" if m["dy"] > 0 else "Move the guitar down on the screen"),
        dict(label="Level", ok=abs(m["roll"]) <= roll_tol, detail=f"rotated {m['roll']:+.0f} deg",
             hint=f"Level the neck: it is rotated {m['roll']:.0f} deg"),
    ]
    if "tilt" in m:
        if m["across"] >= m["along"]:
            hint = (f"Camera tilted {m['across']:.0f} deg across the neck (max {max_tilt:.0f}) "
                    "- change the screen tilt or the guitar height")
        else:
            hint = (f"Neck angled {m['along']:.0f} deg along its length (max {max_tilt:.0f}) "
                    "- turn the guitar to face the screen")
        checks.append(dict(label="Camera angle", ok=m["tilt"] <= max_tilt,
                           detail=f"{m['tilt']:.0f} deg (across {m['across']:.0f}, along {m['along']:.0f})", hint=hint))
    return checks


def smooth(history, metrics):
    """Averages each metric over the last SMOOTH_FRAMES frames; history carries the state between calls."""
    for key, value in metrics.items():
        history.setdefault(key, deque(maxlen=SMOOTH_FRAMES)).append(value)
    return {key: float(np.mean(values)) for key, values in history.items() if key in metrics}


class PlacementMonitor:
    """
    Watches the neck placement during play. update() returns the hint of the first failing check once the
    neck has stayed out of the (loosened) tolerance for PLAY_WARNING_DELAY_S seconds, else None. A neck that
    is not detected is left to the marker message of the app.
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
        checks = evaluate(smooth(self._history, alignment_metrics(neck, image_size)), self._tolerance_scale)
        failing = next((c for c in checks if not c["ok"]), None)
        if failing is None:
            self._out_since = None
            return None
        if self._out_since is None:
            self._out_since = now
        return failing["hint"] if now - self._out_since >= self._delay_s else None
