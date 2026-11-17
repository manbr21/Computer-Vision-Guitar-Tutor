"""A zoomed view of the neck and the hand (display only: the neck and the fingers are always found in the whole frame)."""
import cv2
import numpy as np

SMOOTH = 0.2                  # weight of each new crop in the running one, so the view does not shake with the tracker
ALONG = (-0.1, 0.65)          # the part of the neck shown: from just before the nut to past fret 7
ACROSS = (-1.8, 2.8)          # ... and its width in neck widths (the hand hangs below the neck)


class NeckZoom:
    """Crops the frame around the first frets and the fretting hand and scales it back to the size of the frame."""

    def __init__(self):
        self._box = None   # running (x0, y0, x1, y1)

    def reset(self):
        self._box = None

    def box(self, neck, size, capo=0):
        """
        The crop (x0, y0, x1, y1) inside a frame of `size`, with the frame's proportions, around the neck part and hand.
        With a capo the part shown starts at the capo: the fingers play from there on, and the nut is out of the view.
        """
        width, height = size
        start = neck.fret_bounds[min(capo, len(neck.fret_bounds) - 1)] if capo else 0.0
        corners = np.array([neck.to_image(u, start + t) for u in ACROSS for t in ALONG], float)
        center = (corners.min(axis=0) + corners.max(axis=0)) / 2
        w, h = np.ptp(corners[:, 0]), np.ptp(corners[:, 1])
        w = max(w, h * width / height)                  # the frame's proportions
        w = min(max(w, width * 0.35), width)            # never closer than about 2.8x, nor further than the frame
        h = w * height / width
        x0 = float(np.clip(center[0] - w / 2, 0, width - w))
        y0 = float(np.clip(center[1] - h / 2, 0, height - h))
        return np.array([x0, y0, x0 + w, y0 + h])

    def apply(self, image, neck, capo=0):
        """The zoomed image (same size as `image`)."""
        height, width = image.shape[:2]
        target = self.box(neck, (width, height), capo)
        self._box = target if self._box is None else (1 - SMOOTH) * self._box + SMOOTH * target
        x0, y0, x1, y1 = (int(round(v)) for v in self._box)
        x1, y1 = max(x1, x0 + 2), max(y1, y0 + 2)
        return cv2.resize(image[y0:y1, x0:x1], (width, height), interpolation=cv2.INTER_LINEAR)
