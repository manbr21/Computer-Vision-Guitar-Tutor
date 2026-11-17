"""
Marker-free neck tracking from a reference photo.

The user takes one photo of the guitar and marks the four corners of the neck (see chordar/reference.py).
Each frame, ORB features of the photo's neck region are matched to the frame and a RANSAC homography maps the
four reference corners into the frame; those corners feed the same Fretboard the ArUco markers do.
"""
import logging
import cv2
import numpy as np

from chordar.frets import snap_corners
from chordar.geometry import Fretboard

log = logging.getLogger(__name__)

WORK_WIDTH = 640        # frames are matched at this width (the features are found much faster than at full size)
N_FEATURES = 3000
FAST_THRESHOLD = 10      # ORB corner threshold; lower finds features on low-contrast surfaces such as a dark fretboard
RATIO = 0.75            # Lowe ratio test on the two nearest descriptors
RANSAC_PX = 3.0         # reprojection tolerance, in work-size pixels
MIN_INLIERS = 12
REGION_MARGIN = 0.5     # the reference features come from the neck quad grown by this fraction of its length (headstock, body)
REGION_ACROSS = 0.5     # ... and by this fraction of its width on each side (less keeps the background out)
MIN_AREA, MAX_AREA = 0.3, 3.0   # accepted area of the found quad relative to the reference one
ASPECT_RANGE = (0.75, 1.33)     # accepted length/width relative to the reference's
TAPER_RANGE = (0.6, 1.6)
SIDES_RANGE = (0.7, 1.43)       # top edge / bottom edge
MAX_SKEW_COS = 0.57             # |cos| of the angle between the nut and the neck axis (55 degrees from square)
MATCH_EVERY = 3         # the reference is matched every this many frames while tracking (the flow covers the rest)
SEARCH_MARGIN = 0.4     # while tracking, only look for the reference near the followed position (grown by this much)
FULL_SEARCH_EVERY = 6   # ... except every this many matches, which look at the whole frame so a wrong lock can be corrected
FRET_REFIT_EVERY = 3     # the fret wires are looked for in the live frame every this many frames
FRET_SMOOTH = 0.25       # weight of each new fit in the running fret positions
FRET_MAX_RMS_PX = 3.0    # a fit that misses the wires by more than this is ignored
FRET_FAILURES_BEFORE_WARNING = 7   # fits in a row (one every 3 frames, about 0.7 s) without finding the wires
FRET_MAX_JUMP = 0.05     # ... and so is one that moves the nut or fret 12 by more than this fraction of the neck at once
FRET_MAX_NUT_SHIFT = 0.012  # ... and one that puts the nut further than this fraction of the neck from the followed corners: the corners
                            # are tracked on the whole neck and stay on the nut, while fingers over the first frets drag a fit away
FRET_MAX_12_SHIFT = 0.03    # (same for fret 12)
HOLD_FRAMES = 8         # a neck that cannot be found keeps its last position this many frames
FLOW_POINTS = 300       # corner points followed from one frame to the next (optical flow)
FLOW_RANSAC_PX = 2.0
FLOW_MIN_INLIERS = 10
FLOW_REGION_MARGIN = 0.2
MAX_STEP_FRACTION = 0.12   # the neck cannot move more than this fraction of its length between two frames
MAX_STEP_SCALE = 1.12      # ... nor change its length by more than this factor
MATCH_GAP_FRAMES = 3       # a reference match comes this many frames after the previous one, so it may have moved this much more
REF_WEIGHT = 0.07       # how much of each new reference match is blended into the followed position
AGREE_FRACTION = 0.04   # reference and flow agree if the corners are within this fraction of the neck size
REANCHOR_INLIERS = 25   # a reference match this strong replaces the followed position when they disagree
REANCHOR_FRAMES = 5     # ... or after disagreeing this many frames in a row
REANCHOR_BLEND = 0.3    # share of the reference match in a re-anchoring (1 = jump to it, less = move there gradually)
HARD_REANCHOR = 2.0     # ... but a gap this many times the agreement tolerance means the followed position is lost: jump


def _gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _shape(quad):
    """(length/width, far width/nut width, top edge/bottom edge, |cos| of the nut against the neck axis) of a quad."""
    tl, tr, br, bl = quad
    nut, far = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    top, bottom = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
    if min(nut, far, top, bottom) < 1e-3:
        return None
    axis = (bl - tl) + (br - tr)
    cos = abs(float(np.dot(tr - tl, axis)) / (nut * np.linalg.norm(axis) + 1e-9))
    return (top + bottom) / (nut + far), far / nut, top / bottom, cos


def quad_region_mask(shape, corners, margin=REGION_MARGIN, across=None):
    """
    Mask (255 inside) of the quad grown by `margin` of its length along the neck on both ends and by `across`
    (default: the same fraction) of its width on both sides, so the headstock and the body are included while
    the background beside the neck can be left out.
    """
    corners = np.asarray(corners, dtype=np.float32)
    across = margin if across is None else across
    tl, tr, br, bl = corners
    along_vec = ((bl - tl) + (br - tr)) / 2          # nut -> fret 12
    across_vec = ((tr - tl) + (br - bl)) / 2         # low E side -> high e side
    center = corners.mean(axis=0)
    half_long = along_vec / 2 * (1 + 2 * margin)
    half_wide = across_vec / 2 * (1 + 2 * across)
    grown = np.array([center - half_long - half_wide, center - half_long + half_wide,
                      center + half_long + half_wide, center + half_long - half_wide], np.float32)
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, grown.astype(np.int32), 255)
    return mask


class NeckReference:
    """The reference photo and the neck corners in it (TL, TR, BR, BL in raw camera pixels, as the camera sees it)."""

    def __init__(self, image, corners, ignore_mask=None, fret_bounds=None):
        self.image = image
        self.corners = np.asarray(corners, dtype=np.float32)
        self.ignore_mask = ignore_mask  # 255 where the photo must not be used (for example a ArUco marker)
        self.fret_bounds = None if fret_bounds is None else [float(v) for v in fret_bounds]  # the wires measured in the photo

    def save(self, path):
        extra = {} if self.ignore_mask is None else {"ignore_mask": self.ignore_mask}
        if self.fret_bounds is not None:
            extra["fret_bounds"] = np.array(self.fret_bounds)
        np.savez_compressed(path, image=self.image, corners=self.corners, **extra)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        return cls(data["image"], data["corners"], data["ignore_mask"] if "ignore_mask" in data else None,
                   data["fret_bounds"] if "fret_bounds" in data else None)


class MarklessTracker:
    """Finds the neck corners in each frame from a NeckReference. Same track() interface as MarkerTracker."""

    def __init__(self, reference, refine_frets=True):
        """refine_frets: also fit the nut, the 12th fret and the fret spacing to the wires seen in the live frames."""
        self.reference = reference
        self.refine_frets = refine_frets
        self._orb = cv2.ORB_create(N_FEATURES, fastThreshold=FAST_THRESHOLD)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        height, width = reference.image.shape[:2]
        self._scale = WORK_WIDTH / width
        small = cv2.resize(_gray(reference.image), None, fx=self._scale, fy=self._scale, interpolation=cv2.INTER_AREA)
        self._ref_corners = reference.corners * self._scale
        mask = quad_region_mask(small.shape, self._ref_corners, REGION_MARGIN, REGION_ACROSS)
        if reference.ignore_mask is not None:
            blocked = cv2.resize(reference.ignore_mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask[blocked > 0] = 0
        keypoints, self._ref_desc = self._orb.detectAndCompute(small, mask)
        if self._ref_desc is None or len(keypoints) < MIN_INLIERS:
            raise ValueError("The reference photo has too few features in the neck region: use a sharper, closer photo")
        self._ref_points = np.float32([k.pt for k in keypoints])
        self._ref_area = abs(cv2.contourArea(self._ref_corners.astype(np.float32)))
        self._ref_shape = _shape(self._ref_corners.astype(np.float32))
        self._corners = None        # current position of the neck corners (work-size pixels)
        self._prev_small = None     # previous frame (work size, gray) for the optical flow
        self._disagree = 0
        self._pending = None         # the last reference match that disagreed with the followed position
        self._missed = 0
        self._frame_index = 0
        self._matches_done = 0
        self._params = None          # the last fit's (t0, s, k), the starting point of the next one
        self._fit = None             # running (t_nut, t_12, fret bounds) fitted to the wires of the live frames
        self.refined = None          # (corners, fret bounds) with the live fit applied; None until a fit succeeded
        self._fit_failures = 0       # fret fits in a row that found too few wires
        self.inliers = 0       # inliers of the last successful match, for debugging
        self.last_matched = self.last_followed = None
        self.fresh_corners = {}  # kept for the recorder; this tracker has no per-frame ground truth

    # --- estimation ---------------------------------------------------------------------------------------

    def _work_image(self, frame, allowed_mask):
        small = cv2.resize(_gray(frame), None, fx=self._scale, fy=self._scale, interpolation=cv2.INTER_AREA)
        mask = None
        if allowed_mask is not None:
            mask = cv2.resize(allowed_mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        return small, mask

    def _match_reference(self, small, mask):
        """Corners (work-size pixels) from matching the reference photo, and the number of RANSAC inliers."""
        keypoints, desc = self._orb.detectAndCompute(small, mask)
        if desc is None or len(keypoints) < MIN_INLIERS:
            return None, 0
        pairs = self._matcher.knnMatch(self._ref_desc, desc, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < RATIO * n.distance]
        if len(good) < MIN_INLIERS:
            return None, len(good)
        src = self._ref_points[[m.queryIdx for m in good]].reshape(-1, 1, 2)
        dst = np.float32([keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        homography, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
        if homography is None:
            return None, 0
        inliers = int(inlier_mask.sum())
        if inliers < MIN_INLIERS:
            return None, inliers
        found = cv2.perspectiveTransform(self._ref_corners.reshape(-1, 1, 2), homography).reshape(4, 2)
        if not self._plausible(found):
            return None, inliers
        return found, inliers

    def _follow(self, small, mask):
        """The current corners moved by the motion between the previous frame and this one (optical flow)."""
        if self._prev_small is None or self._corners is None:
            return None
        region = quad_region_mask(small.shape, self._corners, FLOW_REGION_MARGIN)
        prev_mask = region if mask is None else cv2.bitwise_and(region, mask)
        points = cv2.goodFeaturesToTrack(self._prev_small, FLOW_POINTS, 0.01, 5, mask=prev_mask)
        if points is None or len(points) < FLOW_MIN_INLIERS:
            return None
        moved, status, _ = cv2.calcOpticalFlowPyrLK(self._prev_small, small, points, None, winSize=(21, 21), maxLevel=3)
        ok = status.ravel() == 1
        if ok.sum() < FLOW_MIN_INLIERS:
            return None
        homography, inlier_mask = cv2.findHomography(points[ok], moved[ok], cv2.RANSAC, FLOW_RANSAC_PX)
        if homography is None or int(inlier_mask.sum()) < FLOW_MIN_INLIERS:
            return None
        moved_corners = cv2.perspectiveTransform(self._corners.reshape(-1, 1, 2).astype(np.float32), homography).reshape(4, 2)
        return moved_corners if self._plausible(moved_corners) and self._possible_motion(moved_corners) else None

    def _possible_motion(self, moved, frames=1):
        """A single frame cannot move or rescale the neck by much: a bigger jump is a flow fitted to the wrong points."""
        before, length = self._corners, float(np.linalg.norm(((self._corners[3] - self._corners[0]) + (self._corners[2] - self._corners[1])) / 2))
        after = float(np.linalg.norm(((moved[3] - moved[0]) + (moved[2] - moved[1])) / 2))
        shift = float(np.linalg.norm(moved.mean(axis=0) - before.mean(axis=0)))
        scale = MAX_STEP_SCALE ** frames
        return shift <= MAX_STEP_FRACTION * frames * length and 1 / scale <= after / length <= scale

    def estimate(self, frame, allowed_mask=None):
        """
        Stateless: the neck corners (4x2 float array, TL TR BR BL, raw pixels of `frame`) and the number of RANSAC
        inliers from matching the reference photo, or (None, inliers). allowed_mask: 255 where the frame may be used.
        """
        small, mask = self._work_image(frame, allowed_mask)
        found, inliers = self._match_reference(small, mask)
        return (None if found is None else found / self._scale), inliers

    def _plausible(self, quad):
        """
        A convex quad of a sensible size and shape: the proportions of the reference (length against width,
        taper, parallel sides, a nut that is square to the neck). A homography fitted to a few wrong matches
        usually stretches or shears the neck, and this rejects it.
        """
        quad = quad.astype(np.float32)
        if not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
            return False
        area = abs(cv2.contourArea(quad))
        if not MIN_AREA * self._ref_area <= area <= MAX_AREA * self._ref_area:
            return False
        shape = _shape(quad)
        if shape is None:
            return False
        aspect, taper, sides, squareness = shape
        ref_aspect, ref_taper, _, _ = self._ref_shape
        return (ASPECT_RANGE[0] <= aspect / ref_aspect <= ASPECT_RANGE[1]
                and TAPER_RANGE[0] <= taper / ref_taper <= TAPER_RANGE[1]
                and SIDES_RANGE[0] <= sides <= SIDES_RANGE[1]
                and squareness <= MAX_SKEW_COS)

    @property
    def frets_visible(self):
        """False when the last fret fits (about 2 s of them) could not find the wires: the light is not enough to see the frets."""
        return self._fit_failures < FRET_FAILURES_BEFORE_WARNING

    @staticmethod
    def _fit_ok(info):
        return info is not None and info["fret_bounds"] is not None and info["rms_px"] <= FRET_MAX_RMS_PX

    def _refine(self, frame, corners):
        """
        Fits the nut, the 12th fret and the fret spacing to the wires seen in this frame (every few frames) and keeps a
        running, outlier-gated average. The spacing seen by the camera changes with the distance and the angle to the
        neck, so the spacing measured in the reference photo is right only for the pose of the photo.
        """
        if self._frame_index % FRET_REFIT_EVERY == 0:
            snapped, info = snap_corners(frame, corners, self._params)
            if not self._fit_ok(info) and self._params is not None:
                snapped, info = snap_corners(frame, corners)   # the quick fit failed or slipped: search again
            self._fit_failures = 0 if self._fit_ok(info) else self._fit_failures + 1
            if self._fit_ok(info):
                length = float(np.linalg.norm(((corners[3] - corners[0]) + (corners[2] - corners[1])) / 2))
                t_nut = info["nut_shift_px"] / length
                t_12 = t_nut + 1 + info["length_change"]
                bounds = np.array(info["fret_bounds"])
                if abs(t_nut) > FRET_MAX_NUT_SHIFT or abs(t_12 - 1) > FRET_MAX_12_SHIFT:
                    pass   # the wires of this frame were mixed up with fingers or shadows: keep the running fit
                elif self._fit is None:
                    self._fit, self._params = (t_nut, t_12, bounds), info["params"]
                elif abs(t_nut - self._fit[0]) <= FRET_MAX_JUMP and abs(t_12 - self._fit[1]) <= FRET_MAX_JUMP:
                    w = FRET_SMOOTH
                    self._fit = ((1 - w) * self._fit[0] + w * t_nut, (1 - w) * self._fit[1] + w * t_12,
                                 (1 - w) * self._fit[2] + w * bounds)
                    self._params = info["params"]
        if self._fit is not None:
            t_nut, t_12, bounds = self._fit
            tl, tr, br, bl = corners
            moved = np.array([tl + (bl - tl) * t_nut, tr + (br - tr) * t_nut,
                              tr + (br - tr) * t_12, tl + (bl - tl) * t_12], np.float32)
            self.refined = (moved, bounds.tolist())

    # --- tracker interface ----------------------------------------------------------------------------------

    def update(self, frame, allowed_mask=None):
        """
        Processes the next frame of a sequence and returns the neck corners to use (4x2, raw pixels) or None.
        The corners are followed from frame to frame by optical flow (smooth, but it drifts) and pulled toward
        the match with the reference photo (jumpy, but it does not drift): a complementary filter. A neck that is
        lost keeps its last position for HOLD_FRAMES frames.
        """
        small, mask = self._work_image(frame, allowed_mask)
        due = (self._corners is None or self._disagree > 0 or self._missed > 0
               or self._frame_index % MATCH_EVERY == 0)
        matched = None
        if due:
            search = mask
            self._matches_done += 1
            if self._corners is not None and self._matches_done % FULL_SEARCH_EVERY != 0:
                near = quad_region_mask(small.shape, self._corners, SEARCH_MARGIN, SEARCH_MARGIN)
                search = near if mask is None else cv2.bitwise_and(near, mask)
            matched, self.inliers = self._match_reference(small, search)
        self._frame_index += 1
        followed = self._follow(small, mask)
        self.last_matched, self.last_followed = matched, followed   # for diagnostics

        if matched is not None and self._corners is not None and self._missed == 0 \
                and not self._possible_motion(matched, MATCH_GAP_FRAMES):
            matched = None   # a match that moved or rescaled the neck more than it can in a few frames is a wrong one

        if matched is not None and followed is not None:
            size = float(np.linalg.norm(self._ref_corners[0] - self._ref_corners[2]))
            gap = float(np.mean(np.linalg.norm(matched - followed, axis=1)))
            if gap <= AGREE_FRACTION * size:
                new = REF_WEIGHT * matched + (1 - REF_WEIGHT) * followed
                self._disagree, self._pending = 0, None
            else:
                # the match and the followed position disagree: only a second match that agrees with the first
                # one (or a very long disagreement) is believed, so that one wrong match is not followed
                confirmed = self._pending is not None and float(
                    np.mean(np.linalg.norm(matched - self._pending, axis=1))) <= AGREE_FRACTION * size
                self._disagree += 1
                self._pending = matched
                if confirmed or self._disagree >= REANCHOR_FRAMES:
                    blend = 1.0 if gap > HARD_REANCHOR * AGREE_FRACTION * size else REANCHOR_BLEND
                    new = blend * matched + (1 - blend) * followed
                    self._disagree, self._pending = 0, None
                else:
                    new = followed
        elif matched is not None:
            new, self._disagree, self._pending = matched, 0, None
        else:
            new = followed

        if new is not None:
            self._corners, self._missed = new, 0
        else:
            self._missed += 1
            if self._missed > HOLD_FRAMES:
                self._corners = None
        self._prev_small = small
        if self._corners is None:
            self._fit, self.refined, self._params = None, None, None
            return None
        raw = self._corners / self._scale
        if self.refine_frets:
            self._refine(frame, raw)
        return raw

    def track(self, frame, pose_variant=0, draw=True):
        """Returns (display, neck) like MarkerTracker.track: the mirrored frame and the Fretboard (or None)."""
        height, width = frame.shape[:2]
        display = cv2.flip(frame, 1)
        smooth = self.update(frame)
        if smooth is None:
            return display, None
        bounds = self.reference.fret_bounds
        if self.refined is not None:
            smooth, bounds = self.refined
        mirrored = [(width - x, y) for x, y in smooth]  # the display is mirrored
        neck = Fretboard(*mirrored, image_size=(width, height), pose_variant=pose_variant,
                         fret_bounds=bounds)
        return display, neck
