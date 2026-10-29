"""
Marker-free neck tracking from a reference photo.

The user takes one photo of the guitar and marks the four corners of the neck (see chordar/reference.py).
Each frame, ORB features of the photo's neck region are matched to the frame and a RANSAC homography maps the
four reference corners into the frame; those corners feed the same Fretboard the ArUco markers do.
"""
import logging
from collections import deque

import cv2
import numpy as np

from chordar.geometry import Fretboard

log = logging.getLogger(__name__)

WORK_WIDTH = 640        # frames are matched at this width (the features are found much faster than at full size)
N_FEATURES = 1500
RATIO = 0.75            # Lowe ratio test on the two nearest descriptors
RANSAC_PX = 3.0         # reprojection tolerance, in work-size pixels
MIN_INLIERS = 12
REGION_MARGIN = 0.5     # the reference features come from the neck quad grown by this fraction of its size
MIN_AREA, MAX_AREA = 0.3, 3.0   # accepted area of the found quad relative to the reference one
SMOOTH_FRAMES = 3
HOLD_FRAMES = 8         # a neck that cannot be found keeps its last position this many frames


def _gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def quad_region_mask(shape, corners, margin=REGION_MARGIN):
    """Mask (255 inside) of the quad grown by `margin` of its size around its center."""
    corners = np.asarray(corners, dtype=np.float32)
    center = corners.mean(axis=0)
    grown = center + (corners - center) * (1 + 2 * margin)
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(grown.astype(np.int32)), 255)
    return mask


class NeckReference:
    """The reference photo and the neck corners in it (TL, TR, BR, BL in raw camera pixels, as the camera sees it)."""

    def __init__(self, image, corners, ignore_mask=None):
        self.image = image
        self.corners = np.asarray(corners, dtype=np.float32)
        self.ignore_mask = ignore_mask  # 255 where the photo must not be used (for example a ArUco marker)

    def save(self, path):
        extra = {} if self.ignore_mask is None else {"ignore_mask": self.ignore_mask}
        np.savez_compressed(path, image=self.image, corners=self.corners, **extra)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        return cls(data["image"], data["corners"], data["ignore_mask"] if "ignore_mask" in data else None)


class MarklessTracker:
    """Finds the neck corners in each frame from a NeckReference. Same track() interface as MarkerTracker."""

    def __init__(self, reference):
        self.reference = reference
        self._orb = cv2.ORB_create(N_FEATURES)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        height, width = reference.image.shape[:2]
        self._scale = WORK_WIDTH / width
        small = cv2.resize(_gray(reference.image), None, fx=self._scale, fy=self._scale, interpolation=cv2.INTER_AREA)
        self._ref_corners = reference.corners * self._scale
        mask = quad_region_mask(small.shape, self._ref_corners)
        if reference.ignore_mask is not None:
            blocked = cv2.resize(reference.ignore_mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask[blocked > 0] = 0
        keypoints, self._ref_desc = self._orb.detectAndCompute(small, mask)
        if self._ref_desc is None or len(keypoints) < MIN_INLIERS:
            raise ValueError("The reference photo has too few features in the neck region: use a sharper, closer photo")
        self._ref_points = np.float32([k.pt for k in keypoints])
        self._ref_area = abs(cv2.contourArea(self._ref_corners.astype(np.float32)))
        self._history = deque(maxlen=SMOOTH_FRAMES)
        self._missed = 0
        self.inliers = 0       # inliers of the last successful match, for debugging
        self.fresh_corners = {}  # kept for the recorder; this tracker has no per-frame ground truth

    # --- estimation ---------------------------------------------------------------------------------------

    def estimate(self, frame, allowed_mask=None):
        """
        The neck corners (4x2 float array, TL TR BR BL, raw pixels of `frame`) and the number of RANSAC inliers,
        or (None, inliers) when the neck was not found. allowed_mask: 255 where the frame may be used.
        """
        small = cv2.resize(_gray(frame), None, fx=self._scale, fy=self._scale, interpolation=cv2.INTER_AREA)
        mask = None
        if allowed_mask is not None:
            mask = cv2.resize(allowed_mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
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
        return found / self._scale, inliers

    def _plausible(self, quad):
        """A convex quad of a sensible size; anything else is a bad homography."""
        quad = quad.astype(np.float32)
        if not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
            return False
        area = abs(cv2.contourArea(quad))
        return MIN_AREA * self._ref_area <= area <= MAX_AREA * self._ref_area

    # --- tracker interface ----------------------------------------------------------------------------------

    def track(self, frame, pose_variant=0, draw=True):
        """Returns (display, neck) like MarkerTracker.track: the mirrored frame and the Fretboard (or None)."""
        height, width = frame.shape[:2]
        display = cv2.flip(frame, 1)
        corners, self.inliers = self.estimate(frame)
        if corners is not None:
            self._history.append(corners)
            self._missed = 0
        else:
            self._missed += 1
            if self._missed > HOLD_FRAMES:
                self._history.clear()
        if not self._history:
            return display, None
        smooth = np.mean(self._history, axis=0)
        mirrored = [(width - x, y) for x, y in smooth]  # the display is mirrored
        neck = Fretboard(*mirrored, image_size=(width, height), pose_variant=pose_variant)
        return display, neck
