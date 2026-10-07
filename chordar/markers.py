"""ArUco detection: finds the four neck markers and builds the Fretboard from their corners."""
import logging
from collections import deque

import cv2
import cv2.aruco as aruco
import numpy as np

from chordar import config
from chordar.geometry import Fretboard

log = logging.getLogger(__name__)

SMOOTH_FRAMES = 5    # marker corners are averaged over this many frames
LOG_EVERY_FRAMES = 30


class MarkerTracker:
    """Detects the markers in each frame and keeps their smoothed corners."""

    def __init__(self):
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
        self._detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
        self._history = {marker_id: deque(maxlen=SMOOTH_FRAMES) for marker_id in config.NECK_CORNER}
        self._last_seen = {}
        self._frame_count = 0

    def track(self, frame, pose_variant=0):
        """
        Returns (display, neck): the mirrored frame with the detected markers drawn, and the Fretboard
        (None until all four corners are known; a marker that is briefly lost keeps its last position).
        """
        height, width = frame.shape[:2]
        corners, ids, rejected = self._detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        self._log_detections(ids, rejected)

        display = cv2.flip(frame, 1)
        quad = self._quad_corners(display, corners, ids, width) if ids is not None else {}
        if len(quad) < 4:
            return display, None
        neck = Fretboard(quad["TL"], quad["TR"], quad["BR"], quad["BL"],
                         image_size=(width, height), pose_variant=pose_variant)
        return display, neck

    def _log_detections(self, ids, rejected):
        """
        Every LOG_EVERY_FRAMES frames, log what OpenCV sees: the decoded IDs and how many square-shaped
        candidates were rejected (a high count usually means bad contrast or lighting). Enable with --debug.
        """
        self._frame_count += 1
        if self._frame_count % LOG_EVERY_FRAMES == 0:
            found = ids.flatten().tolist() if ids is not None else []
            log.debug("Detected IDs: %s | rejected candidates: %d", found, len(rejected))

    def _quad_corners(self, display, corners, ids, width):
        """Neck corner name -> pixel, from the markers seen now and, for the missing ones, their last position."""
        quad = {}
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id not in config.NECK_CORNER:
                continue
            mirrored = corners[i].copy()
            mirrored[0, :, 0] = width - corners[i][0, :, 0]  # the display is mirrored

            self._history[marker_id].append(mirrored[0])
            smoothed = np.mean(self._history[marker_id], axis=0)
            self._last_seen[marker_id] = smoothed

            aruco.drawDetectedMarkers(display, [mirrored], ids[i])
            cv2.putText(display, f"ID:{marker_id}", tuple(smoothed[0].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            name, corner_index = config.NECK_CORNER[marker_id]
            quad[name] = smoothed[corner_index]

        for marker_id, (name, corner_index) in config.NECK_CORNER.items():
            if name not in quad and marker_id in self._last_seen:
                quad[name] = self._last_seen[marker_id][corner_index]
        return quad
