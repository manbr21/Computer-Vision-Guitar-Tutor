"""Mapping between image pixels and the guitar neck (string lane x position along the neck)."""
import numpy as np

from chordar import config
from chordar.config import NUM_FRETS, NUM_STRINGS
from chordar.pose import camera_angles, estimate_pose, parallax_across_mm


def _fret_bounds():
    """Position of each fret wire along the neck as a fraction of nut -> fret 12 (0.0 .. ~1.0)."""
    bounds = [0.0]
    fraction = 0.0
    for _ in range(NUM_FRETS):
        fraction += (1 - fraction) / 17.817  # the "rule of 18"; the distance to fret n is 1 - 2^(-n/12) of the scale
        bounds.append(fraction * 2)          # the quad ends at fret 12, half of the scale
    return bounds


FRET_BOUNDS = _fret_bounds()


def string_u(string_idx):
    """Position across the quad (u) of a string; string_idx 0 = low E."""
    lo, hi = config.STRING_U_RANGE
    return lo + string_idx * (hi - lo) / (NUM_STRINGS - 1)


def outline_u_range():
    """(u_min, u_max) of the neck outline: the outer strings plus half a string gap on each side."""
    lo, hi = config.STRING_U_RANGE
    half_gap = (hi - lo) / (NUM_STRINGS - 1) / 2
    return lo - half_gap, hi + half_gap


class Fretboard:
    """
    Maps between image pixels and neck coordinates (u across the neck, 0 at the ID0/TL side; t along the
    neck, 0 at the nut and 1 at fret 12) by interpolating linearly between the four marker corners.
    Targets and fingertips go through the same mapping, so a tilted neck does not skew the string/fret lookup.

    A homography was tried and rejected: a guitar neck is tapered (wider at fret 12), and a homography reads
    that taper as perspective, flattening the fret spacing so the drawn frets drift away from the real ones.

    With image_size given, the camera pose relative to the fretboard plane is also estimated so fingertips
    can be shifted across the neck to the board point under them, which removes the parallax caused by their
    height above the board.
    """

    def __init__(self, TL, TR, BR, BL, image_size=None, pose_variant=0):
        self._TL, self._TR, self._BR, self._BL = (np.array(p, dtype=np.float64) for p in (TL, TR, BR, BL))
        self._pose = estimate_pose(self.corners, image_size, pose_variant) if image_size is not None else None

    @property
    def corners(self):
        """The four marker corners used, in pixels: TL, TR, BR, BL."""
        return self._TL, self._TR, self._BR, self._BL

    # --- neck <-> image ---------------------------------------------------------------------------------

    def _point(self, u, t):
        nut = self._TL + u * (self._TR - self._TL)
        far = self._BL + u * (self._BR - self._BL)
        return nut + t * (far - nut)

    def to_image(self, u, t):
        x, y = self._point(u, t)
        return int(round(x)), int(round(y))

    def to_neck(self, x, y):
        """Inverse of the interpolation, by Newton iteration (a few steps for a near-rectangular quad)."""
        target = np.array([x, y], dtype=np.float64)
        u, t = 0.5, 0.5
        for _ in range(10):
            d_u = (1 - t) * (self._TR - self._TL) + t * (self._BR - self._BL)
            d_t = (self._BL + u * (self._BR - self._BL)) - (self._TL + u * (self._TR - self._TL))
            step = np.linalg.solve(np.column_stack([d_u, d_t]), target - self._point(u, t))
            u += step[0]
            t += step[1]
            if np.abs(step).max() < 1e-6:
                break
        return float(u), float(t)

    def target(self, string_idx, fret):
        """Pixel to press: middle of the fret space on the string (string_idx 0 = low E)."""
        fret = min(max(fret, 1), NUM_FRETS)
        return self.to_image(string_u(string_idx), (FRET_BOUNDS[fret - 1] + FRET_BOUNDS[fret]) / 2)

    # --- fingertips -------------------------------------------------------------------------------------

    def to_neck_corrected(self, x, y, parallax=True):
        """
        to_neck for a fingertip. With parallax=True the string coordinate is shifted to the board point under
        the fingertip. Only the across-neck shift is applied: the pose comes from assumed neck proportions, and
        a wrong proportion shows up as a fake tilt along the neck, which pushed fingers across fret wires. The
        fret coordinate stays anchored to the marker grid that is drawn.
        """
        u, t = self.to_neck(x, y)
        if parallax and self._pose is not None:
            shift = parallax_across_mm(self._pose, x, y)
            if shift is not None:
                width = config.QUAD_NUT_WIDTH_MM + (config.QUAD_FRET12_WIDTH_MM - config.QUAD_NUT_WIDTH_MM) * t
                u += shift / width
        return u, t

    def contact_pixel(self, x, y):
        """Pixel where the board point under a fingertip appears on the drawn grid, or None without a pose."""
        if self._pose is None or parallax_across_mm(self._pose, x, y) is None:
            return None
        return self.to_image(*self.to_neck_corrected(x, y))

    def locate(self, x, y, parallax=True):
        """
        (string_idx, fret) under a pixel, or None. The string is the nearest one (each owns a lane half a
        string gap to either side); any point between the two wires of a fret counts as that fret.
        """
        u, t = self.to_neck_corrected(x, y, parallax)
        lo, hi = config.STRING_U_RANGE
        position = (u - lo) / (hi - lo) * (NUM_STRINGS - 1)  # 0 at the first string, 5 at the last
        if not (-0.5 <= position < NUM_STRINGS - 0.5 and 0 <= t < FRET_BOUNDS[-1]):
            return None
        return int(np.floor(position + 0.5)), int(np.searchsorted(FRET_BOUNDS, t, side="right"))

    # --- measurements -----------------------------------------------------------------------------------

    def camera_angles(self):
        """(total, across, along) camera tilt in degrees, or None without a pose."""
        return camera_angles(self._pose) if self._pose is not None else None

    def width_px(self):
        """Quad width in pixels at mid-neck (between the marker corners)."""
        x0, y0 = self.to_image(0, 0.5)
        x1, y1 = self.to_image(1, 0.5)
        return float(np.hypot(x1 - x0, y1 - y0))

    def string_gap_px(self):
        """Distance in pixels between adjacent strings at mid-neck."""
        lo, hi = config.STRING_U_RANGE
        return self.width_px() * (hi - lo) / (NUM_STRINGS - 1)
