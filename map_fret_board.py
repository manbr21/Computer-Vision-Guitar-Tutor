import cv2
import cv2.aruco as aruco
import numpy as np
from collections import deque

# ArUco setup
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

valid_ids = {0, 1, 2, 3}

# marker ID -> (neck corner it defines, index of the marker corner that sits on that neck corner).
# The index depends on how each sticker is rotated when taped; if you re-tape one rotated,
# change its index here.
NECK_CORNER = {
    0: ("TL", 1),  # nut, top edge
    1: ("TR", 0),  # nut, bottom edge
    2: ("BR", 3),  # fret 12, bottom edge
    3: ("BL", 2),  # fret 12, top edge
}
history = {i: deque(maxlen=5) for i in valid_ids}
last_seen = {}
_debug_frame_count = 0

# Standard tuning (low E to high E)
string_labels = ["E", "A", "D", "G", "B", "E"]

NUM_STRINGS = 6
NUM_FRETS = 12


def _fret_bounds():
    """Position of each fret wire along the neck as a fraction of nut->fret 12 (0.0 .. ~1.0)."""
    bounds = [0.0]
    prev_frac = 0.0
    for _ in range(NUM_FRETS):
        prev_frac = prev_frac + (1 - prev_frac) / 17.817
        bounds.append(prev_frac * 2)
    return bounds


FRET_BOUNDS = _fret_bounds()


class Fretboard:
    """
    Maps between image pixels and neck coordinates (u across the neck, 0 at the
    ID0/TL side; t along the neck, 0 at the nut and 1 at fret 12) by interpolating
    linearly between the 4 marker corners. Targets and fingertips go through the
    same mapping, so a tilted neck no longer skews the string/fret lookup.

    A homography was tried here and rejected: a guitar neck is tapered (wider at
    fret 12), and a homography reads that taper as perspective, flattening the fret
    spacing so the drawn frets drift away from the real ones.
    """

    def __init__(self, TL, TR, BR, BL):
        self._TL, self._TR, self._BR, self._BL = (np.array(p, dtype=np.float64) for p in (TL, TR, BR, BL))

    def _point(self, u, t):
        nut = self._TL + u * (self._TR - self._TL)
        far = self._BL + u * (self._BR - self._BL)
        return nut + t * (far - nut)

    def to_image(self, u, t):
        x, y = self._point(u, t)
        return int(round(x)), int(round(y))

    def to_neck(self, x, y):
        """Inverse of the interpolation, by Newton iteration (converges in a few steps for a near-rectangular quad)."""
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
        """Pixel position to press: middle of the fret space on the string's lane (string_idx 0 = low E)."""
        fret = min(max(fret, 1), NUM_FRETS)
        u = (string_idx + 0.5) / NUM_STRINGS
        t = (FRET_BOUNDS[fret - 1] + FRET_BOUNDS[fret]) / 2
        return self.to_image(u, t)

    def locate(self, x, y):
        """(string_idx, fret) under a pixel, or None. Any point between the two wires of a fret counts as that fret."""
        u, t = self.to_neck(x, y)
        if not (0 <= u < 1 and 0 <= t < FRET_BOUNDS[-1]):
            return None
        fret = int(np.searchsorted(FRET_BOUNDS, t, side='right'))
        return int(u * NUM_STRINGS), fret

    def width_px(self):
        """Neck width in pixels at mid-neck."""
        x0, y0 = self.to_image(0, 0.5)
        x1, y1 = self.to_image(1, 0.5)
        return float(np.hypot(x1 - x0, y1 - y0))

def map_guitar(frame):
    """Process a frame, detect ArUco fretboard, draw frets + strings,
    return annotated display + Fretboard (None until all 4 markers are known)."""
    global _debug_frame_count
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    # Log every ~30 frames (not every frame, to avoid flooding the terminal) what
    # OpenCV is actually seeing: how many marker candidates were found, how many
    # were successfully decoded, and which IDs. "rejected" are square-shaped
    # candidates that looked like a marker but didn't match the 4x4_1000
    # dictionary - a high rejected count usually points to bad contrast/lighting.
    _debug_frame_count += 1
    if _debug_frame_count % 30 == 0:
        found_ids = ids.flatten().tolist() if ids is not None else []
        print(f"[ArUco debug] Detected IDs: {found_ids} | rejected candidates: {len(rejected)}")

    display = cv2.flip(frame, 1)
    quad_points = {}

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in valid_ids:
                flipped_corners = corners[i].copy()
                flipped_corners[0,:,0] = w - corners[i][0,:,0]

                history[marker_id].append(flipped_corners[0])
                avg_c = np.mean(history[marker_id], axis=0)
                last_seen[marker_id] = avg_c
                c = avg_c

                aruco.drawDetectedMarkers(display, [flipped_corners], ids[i])
                top_left = tuple(c[0].astype(int))
                cv2.putText(display, f"ID:{marker_id}", top_left,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                corner_name, corner_idx = NECK_CORNER[marker_id]
                quad_points[corner_name] = c[corner_idx]

        # reuse last seen markers if missing
        for mid in valid_ids:
            if mid not in quad_points and mid in last_seen:
                corner_name, corner_idx = NECK_CORNER[mid]
                quad_points[corner_name] = last_seen[mid][corner_idx]

    neck = None

    if len(quad_points) == 4:
        pts = np.array([quad_points["TL"], quad_points["TR"],
                        quad_points["BR"], quad_points["BL"]], dtype=np.int32)
        cv2.polylines(display, [pts], True, (0,0,255), 3)
        for pt in pts:
            cv2.circle(display, tuple(int(v) for v in pt), 5, (255, 0, 255), -1)

        neck = Fretboard(quad_points["TL"], quad_points["TR"], quad_points["BR"], quad_points["BL"])

        for n in range(1, NUM_FRETS + 1):
            wire_left = neck.to_image(0, FRET_BOUNDS[n])
            wire_right = neck.to_image(1, FRET_BOUNDS[n])
            cv2.line(display, wire_left, wire_right, (0, 255, 255), 2)
            cv2.putText(display, f"{n}", (wire_left[0] + 5, wire_left[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Strings sit in the middle of 6 equal lanes, not on the neck edges.
        for s in range(NUM_STRINGS):
            u = (s + 0.5) / NUM_STRINGS
            cv2.line(display, neck.to_image(u, 0), neck.to_image(u, 1), (155, 255, 0), 2)

    return display, neck