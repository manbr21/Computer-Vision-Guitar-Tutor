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

# Where the marker corners sit across the neck. Layout and model dimensions must match the real mounting:
#   "outer_strings": each corner touches the outer string (low E / high e); the red outline is drawn half a
#                    string gap outside. Distances between corners are ~44 mm (nut) and ~50 mm (fret 12).
#   "neck_edges":    each corner touches the edge of the fretboard; strings are inset. ~52 mm and ~60 mm.
MARKER_PLACEMENT = "outer_strings"

# Position of the outer strings across the quad (0 = ID0 side, 1 = ID1 side).
STRING_U_RANGE = (0.0, 1.0) if MARKER_PLACEMENT == "outer_strings" else (1 / 12, 11 / 12)

# Geometry and camera used to estimate where the camera is relative to the fretboard (tilt readout and
# fingertip parallax correction). The QUAD_* and NECK_LENGTH numbers matter: a >10% error in their
# proportions skews the estimated tilt and the correction. Measure them on the guitar with a ruler.
QUAD_NUT_WIDTH_MM = 44.0 if MARKER_PLACEMENT == "outer_strings" else 52.0     # corner to corner at the nut
QUAD_FRET12_WIDTH_MM = 50.0 if MARKER_PLACEMENT == "outer_strings" else 60.0  # corner to corner at fret 12
NECK_LENGTH_MM = 325.0        # nut line -> fret 12 line, between the corners (half of a 650 mm scale)
FINGER_HEIGHT_MM = 10.0       # fingertip height above the fretboard surface when pressing
CAMERA_HFOV_DEG = 65.0        # horizontal field of view of the webcam (laptop webcams: ~60-78)
MAX_PARALLAX_MM = 25.0        # ignore corrections larger than this (degenerate pose)


def string_u(string_idx):
    """Position across the quad (u) of a string; string_idx 0 = low E."""
    lo, hi = STRING_U_RANGE
    return lo + string_idx * (hi - lo) / (NUM_STRINGS - 1)


def outline_u_range():
    """(u_min, u_max) of the neck outline: the outer strings plus half a string gap on each side."""
    lo, hi = STRING_U_RANGE
    half_gap = (hi - lo) / (NUM_STRINGS - 1) / 2
    return lo - half_gap, hi + half_gap


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

    With image_size given, the camera pose relative to the fretboard plane is also
    estimated (planar PnP) so fingertips can be shifted across the neck to the board
    point under them, which removes the parallax caused by their height above the board.
    """

    def __init__(self, TL, TR, BR, BL, image_size=None, pose_variant=0):
        self._TL, self._TR, self._BR, self._BL = (np.array(p, dtype=np.float64) for p in (TL, TR, BR, BL))
        self._pose = self._estimate_pose(image_size, pose_variant) if image_size is not None else None

    def _estimate_pose(self, image_size, pose_variant):
        """
        Camera pose from the 4 neck corners. A planar view has two near-equivalent
        solutions (tilted one way or the other); pose_variant 0 takes the lower
        reprojection error, 1 the alternative.
        """
        w, h = image_size
        f = (w / 2) / np.tan(np.radians(CAMERA_HFOV_DEG) / 2)
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
        half_nut, half_12 = QUAD_NUT_WIDTH_MM / 2, QUAD_FRET12_WIDTH_MM / 2
        # X along the neck (nut -> fret 12), Y across (TL side -> TR side), Z normal to the fretboard
        model = np.float64([[0, -half_nut, 0], [0, half_nut, 0],
                            [NECK_LENGTH_MM, half_12, 0], [NECK_LENGTH_MM, -half_12, 0]])
        image = np.float64([self._TL, self._TR, self._BR, self._BL])
        try:
            count, rvecs, tvecs, errors = cv2.solvePnPGeneric(model, image, K, None, flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            return None
        if not count:
            return None
        order = np.argsort(np.asarray(errors).ravel())
        pick = order[min(pose_variant, len(order) - 1)]
        R, _ = cv2.Rodrigues(rvecs[pick])
        return {"K_inv": np.linalg.inv(K), "R": R, "cam": -R.T @ tvecs[pick].reshape(3)}

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
        u = string_u(string_idx)
        t = (FRET_BOUNDS[fret - 1] + FRET_BOUNDS[fret]) / 2
        return self.to_image(u, t)

    def _parallax_across_mm(self, x, y):
        """
        Distance in mm, across the neck, between the board point under pixel (x, y) and the board
        point right under a fingertip seen there FINGER_HEIGHT_MM above the board: the pixel's ray
        hits the board farther away than the fingertip really is. None without a pose or with a
        degenerate ray.
        """
        if self._pose is None:
            return None
        cam = self._pose["cam"]
        ray = self._pose["R"].T @ (self._pose["K_inv"] @ np.array([x, y, 1.0]))
        if abs(ray[2]) < 1e-9:
            return None
        s_board = -cam[2] / ray[2]
        s_finger = (-FINGER_HEIGHT_MM - cam[2]) / ray[2]
        if s_board <= 0 or s_finger <= 0:
            return None
        shift = (s_finger - s_board) * ray[:2]
        if np.hypot(shift[0], shift[1]) > MAX_PARALLAX_MM:
            return None
        return float(shift[1])

    def to_neck_corrected(self, x, y, parallax=True):
        """
        to_neck for a fingertip. With parallax=True the string coordinate is shifted to the board point
        under the fingertip. Only the across-neck shift is applied: the pose comes from assumed neck
        proportions, and a wrong proportion shows up as a fake tilt along the neck, which pushed fingers
        across fret wires. The fret coordinate stays anchored to the marker grid that is drawn.
        """
        u, t = self.to_neck(x, y)
        if parallax:
            shift = self._parallax_across_mm(x, y)
            if shift is not None:
                u += shift / (QUAD_NUT_WIDTH_MM + (QUAD_FRET12_WIDTH_MM - QUAD_NUT_WIDTH_MM) * t)
        return u, t

    def contact_pixel(self, x, y):
        """Pixel where the board point under a fingertip appears in the image (on the drawn grid), or None."""
        if self._parallax_across_mm(x, y) is None:
            return None
        return self.to_image(*self.to_neck_corrected(x, y))

    def locate(self, x, y, parallax=True):
        """
        (string_idx, fret) under a pixel, or None. The string is the nearest one (each owns a lane
        half a string gap to either side); any point between the two wires of a fret counts as that fret.
        """
        u, t = self.to_neck_corrected(x, y, parallax)
        lo, hi = STRING_U_RANGE
        position = (u - lo) / (hi - lo) * (NUM_STRINGS - 1)  # 0 at the first string, 5 at the last
        if not (-0.5 <= position < NUM_STRINGS - 0.5 and 0 <= t < FRET_BOUNDS[-1]):
            return None
        fret = int(np.searchsorted(FRET_BOUNDS, t, side='right'))
        return int(np.floor(position + 0.5)), fret

    def camera_angles(self):
        """
        (total, across, along) in degrees: how far the camera is from looking straight down the
        fretboard normal, split into tilt across the neck and along it. Magnitudes only - a single
        planar view can't tell the tilt direction reliably. None without a pose.
        """
        if self._pose is None:
            return None
        to_camera = self._pose["cam"] - np.array([NECK_LENGTH_MM / 2, 0.0, 0.0])
        depth = abs(to_camera[2])
        total = np.degrees(np.arccos(depth / np.linalg.norm(to_camera)))
        across = np.degrees(np.arctan2(abs(to_camera[1]), depth))
        along = np.degrees(np.arctan2(abs(to_camera[0]), depth))
        return float(total), float(across), float(along)

    def width_px(self):
        """Quad width in pixels at mid-neck (between the marker corners)."""
        x0, y0 = self.to_image(0, 0.5)
        x1, y1 = self.to_image(1, 0.5)
        return float(np.hypot(x1 - x0, y1 - y0))

    def string_gap_px(self):
        """Distance in pixels between adjacent strings at mid-neck."""
        lo, hi = STRING_U_RANGE
        return self.width_px() * (hi - lo) / (NUM_STRINGS - 1)

    def outline_u(self):
        return outline_u_range()

def map_guitar(frame, pose_variant=0):
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
        neck = Fretboard(quad_points["TL"], quad_points["TR"], quad_points["BR"], quad_points["BL"],
                         image_size=(w, h), pose_variant=pose_variant)

        # Red outline of the neck: the marker corners are the outer strings, so it sits a small
        # fixed distance outside them. The magenta dots mark the marker corners actually in use.
        u_min, u_max = neck.outline_u()
        outline = np.array([neck.to_image(u_min, 0), neck.to_image(u_max, 0),
                            neck.to_image(u_max, 1), neck.to_image(u_min, 1)], dtype=np.int32)
        cv2.polylines(display, [outline], True, (0, 0, 255), 3)
        for corner in ("TL", "TR", "BR", "BL"):
            cv2.circle(display, tuple(int(v) for v in quad_points[corner]), 5, (255, 0, 255), -1)

        for n in range(1, NUM_FRETS + 1):
            wire_left = neck.to_image(u_min, FRET_BOUNDS[n])
            wire_right = neck.to_image(u_max, FRET_BOUNDS[n])
            cv2.line(display, wire_left, wire_right, (0, 255, 255), 2)
            cv2.putText(display, f"{n}", (wire_left[0] + 5, wire_left[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        for s in range(NUM_STRINGS):
            u = string_u(s)
            cv2.line(display, neck.to_image(u, 0), neck.to_image(u, 1), (155, 255, 0), 2)

    return display, neck