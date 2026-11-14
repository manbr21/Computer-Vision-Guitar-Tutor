"""
Snaps the clicked neck corners to the real fret wires of the reference photo.

The neck quad is defined by the nut and the 12th fret wire; where exactly the user clicks (the outer or the inner edge
of the nut, a few pixels off the wire) shifts the whole drawn fret grid. The fret wires are bright thin lines across
the neck, so they can be found in the photo and the nut position and the neck length fitted to the equal-temperament
formula, which fixes the click.
"""
import cv2
import numpy as np

from chordar.geometry import FRET_BOUNDS

AXIS_MARGIN = 0.18      # the neck is looked at from this far before the nut to this far after fret 12 (fractions of its length)
AXIS_SAMPLES = 1200     # samples along the neck in the straightened view
ACROSS_SAMPLES = 160
ACROSS_RANGE = (0.2, 0.8)   # the strings area used for the profile (avoids the neck edges and the hand shadows)
TOLERANCE = 0.012       # a wire matches a predicted fret within this fraction of the neck length
MAX_NUT_SHIFT = 0.07    # the nut may move by this fraction of the neck length
MAX_LENGTH_CHANGE = 0.10
MAX_PERSPECTIVE = 0.3   # |k|: how much the scale along the neck may change from end to end
MIN_MATCHED = 7         # frets that must be matched for the fit to be trusted
N_FRETS = 12


def straighten(image, corners):
    """
    The neck as a rectangle: x along the neck (t from -AXIS_MARGIN to 1 + AXIS_MARGIN), y across (u from 0 to 1).
    Bilinear between the corners, like the fretboard model: a homography would read the taper of the neck as
    perspective and stretch the positions along it.
    """
    tl, tr, br, bl = (np.asarray(c, np.float32) for c in corners)
    t = np.linspace(-AXIS_MARGIN, 1 + AXIS_MARGIN, AXIS_SAMPLES, dtype=np.float32)[None, :, None]
    u = np.linspace(0, 1, ACROSS_SAMPLES, dtype=np.float32)[:, None, None]
    points = (1 - t) * ((1 - u) * tl + u * tr) + t * ((1 - u) * bl + u * br)
    return cv2.remap(image, points[..., 0], points[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def wire_profile(image, corners):
    """Brightness of the thin lines across the neck along it: (t positions, profile with the local level removed)."""
    flat = straighten(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image, corners).astype(np.float32)
    rows = flat[int(ACROSS_RANGE[0] * ACROSS_SAMPLES):int(ACROSS_RANGE[1] * ACROSS_SAMPLES)]
    profile = rows.mean(axis=0)
    window = max(int(0.03 * AXIS_SAMPLES / (1 + 2 * AXIS_MARGIN)), 5) | 1
    profile = np.convolve(profile, np.ones(3) / 3, mode="same") - np.convolve(profile, np.ones(window) / window, mode="same")
    t = np.linspace(-AXIS_MARGIN, 1 + AXIS_MARGIN, AXIS_SAMPLES)
    return t, profile


def find_wires(t, profile, min_height=4.0):
    """t positions of the bright thin lines (local maxima above min_height)."""
    peaks = [i for i in range(2, len(profile) - 2)
             if profile[i] >= profile[i - 1] and profile[i] > profile[i + 1] and profile[i] >= min_height
             and profile[i] >= profile[i - 2] and profile[i] >= profile[i + 2]]
    kept = []
    for i in sorted(peaks, key=lambda p: -profile[p]):   # strongest first, drop neighbors closer than 1.5% of the length
        if all(abs(t[i] - t[j]) > 0.015 for j in kept):
            kept.append(i)
    return np.array(sorted(t[i] for i in kept))


def fit_frets(wires):
    """
    Fits wire positions to t_n = (t0 + s b_n) / (1 + k b_n), b_n = FRET_BOUNDS[n] (n = 1..12): the equal-temperament
    positions seen with a one-dimensional perspective (k, the foreshortening along the neck when the camera is not
    square to it). Returns (t0, s, k, matched, rms) with t in units of the clicked neck length, or None.
    """
    if len(wires) < MIN_MATCHED:
        return None
    bounds = np.array(FRET_BOUNDS[1:N_FRETS + 1])
    best = None
    for t0 in np.arange(-MAX_NUT_SHIFT, MAX_NUT_SHIFT + 1e-9, 0.002):   # coarse search without perspective
        for s in np.arange(1 - MAX_LENGTH_CHANGE, 1 + MAX_LENGTH_CHANGE + 1e-9, 0.004):
            predicted = t0 + s * bounds
            gaps = np.abs(wires[None, :] - predicted[:, None]).min(axis=1)
            score = np.sum(np.clip(1 - gaps / TOLERANCE, 0, 1))
            if best is None or score > best[0]:
                best = (score, t0, s)
    _, t0, s = best
    k = 0.0
    for _ in range(4):  # least squares on the wires matched to each fret; t_n (1 + k b_n) = t0 + s b_n is linear in (t0, s, k)
        predicted = (t0 + s * bounds) / (1 + k * bounds)
        nearest = np.array([wires[np.argmin(np.abs(wires - p))] for p in predicted])
        ok = np.abs(nearest - predicted) < 1.5 * TOLERANCE
        if ok.sum() < MIN_MATCHED:
            return None
        A = np.vstack([np.ones(ok.sum()), bounds[ok], -nearest[ok] * bounds[ok]]).T
        (t0, s, k), *_ = np.linalg.lstsq(A, nearest[ok], rcond=None)
    predicted = (t0 + s * bounds) / (1 + k * bounds)
    nearest = np.array([wires[np.argmin(np.abs(wires - p))] for p in predicted])
    ok = np.abs(nearest - predicted) < 1.5 * TOLERANCE
    if ok.sum() < MIN_MATCHED or not -MAX_PERSPECTIVE <= k <= MAX_PERSPECTIVE:
        return None
    rms = float(np.sqrt(np.mean((nearest[ok] - predicted[ok]) ** 2)))
    return float(t0), float(s), float(k), int(ok.sum()), rms


def snap_corners(image, corners):
    """
    The corners moved along the neck so that the nut and the 12th fret sit on the fitted wires, and what was found.
    Returns (corners, info) with info None when the wires were not found (corners unchanged). info has nut_shift_px,
    length_change, matched, rms_px and fret_bounds: the 13 positions of the nut and the wires along the new neck
    (0 to 1), which follow the real wires of this photo (perspective included) instead of the ideal formula.
    """
    corners = np.asarray(corners, np.float32)
    t, profile = wire_profile(image, corners)
    fit = fit_frets(find_wires(t, profile))
    if fit is None:
        return corners, None
    t0, s, k, matched, rms = fit
    b = np.array(FRET_BOUNDS)                       # 13 values, 0 .. 1
    wire_t = (t0 + s * b) / (1 + k * b)             # where the nut and the 12 wires are, in old t units
    t_nut, t_12 = wire_t[0], wire_t[-1]
    tl, tr, br, bl = corners
    snapped = np.array([tl + (bl - tl) * t_nut, tr + (br - tr) * t_nut,
                        tr + (br - tr) * t_12, tl + (bl - tl) * t_12], np.float32)
    length = float(np.linalg.norm(((bl - tl) + (br - tr)) / 2))
    bounds = (wire_t - t_nut) / (t_12 - t_nut)
    sane = bool(np.all(np.diff(bounds) > 0.02))
    return snapped, {"nut_shift_px": float(t_nut * length), "length_change": float(t_12 - t_nut - 1), "matched": matched,
                     "rms_px": rms * length, "perspective": k, "fret_bounds": bounds.tolist() if sane else None}
