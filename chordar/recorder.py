"""Records a play session (raw camera video + the neck corners the tracker found) to evaluate other trackers offline."""
import csv
import logging
import os
import time

import cv2

log = logging.getLogger(__name__)

CORNER_NAMES = ("TL", "TR", "BR", "BL")


class SessionRecorder:
    """
    Writes <folder>/<timestamp>/video.mp4 (the camera frames as delivered, no overlay, not mirrored) and
    corners.csv with one row per frame: the four neck corners in raw camera pixels as detected by the ArUco
    markers in that very frame (empty unless all four were seen), the ground truth for a marker-free tracker.
    """

    def __init__(self, folder, fps=30.0):
        self.path = os.path.join(folder, time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.path, exist_ok=True)
        self._fps = fps
        self._writer = None
        self._file = open(os.path.join(self.path, "corners.csv"), "w", newline="")
        self._csv = csv.writer(self._file)
        self._csv.writerow(["frame", "time_s", "all_markers"] + [f"{n}{axis}" for n in CORNER_NAMES for axis in "xy"]
                           + ["chord", "capo", "score"])
        self._frames = 0
        self._start = None
        log.info("Recording the session to %s", self.path)

    def write(self, frame, tracker, now, info=None):
        """
        frame: the raw camera frame. tracker: the tracker that just processed it. info: what the app showed for this
        frame, {"chord": name, "capo": fret, "score": percent or None}, so a recording can be analysed chord by chord.
        """
        info = info or {}
        height, width = frame.shape[:2]
        if self._writer is None:
            self._writer = cv2.VideoWriter(os.path.join(self.path, "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                           self._fps, (width, height))
            self._start = now
        self._writer.write(frame)
        fresh = tracker.fresh_corners
        complete = all(name in fresh for name in CORNER_NAMES)
        row = [self._frames, f"{now - self._start:.3f}", int(complete)]
        for name in CORNER_NAMES:  # the tracker works on the mirrored image: undo the mirroring
            row += [f"{width - fresh[name][0]:.2f}", f"{fresh[name][1]:.2f}"] if complete else ["", ""]
        row += [info.get("chord", ""), info.get("capo", ""), "" if info.get("score") is None else info["score"]]
        self._csv.writerow(row)
        self._frames += 1

    def close(self):
        if self._writer is not None:
            self._writer.release()
        self._file.close()
        log.info("Recorded %d frames to %s", self._frames, self.path)
