"""Camera pose relative to the fretboard plane, estimated from the four neck corners."""
from dataclasses import dataclass

import cv2
import numpy as np

from chordar import config


@dataclass(frozen=True, eq=False)
class CameraPose:
    """Camera in fretboard coordinates: X along the neck (nut -> fret 12), Y across, Z normal to the board."""
    k_inv: np.ndarray      # inverse camera matrix
    rotation: np.ndarray   # fretboard -> camera
    position: np.ndarray   # camera position (mm)


def estimate_pose(corners, image_size, variant=0):
    """
    Planar PnP on the corners (TL, TR, BR, BL, in pixels). A planar view has two near-equivalent
    solutions (tilted one way or the other); variant 0 takes the lower reprojection error, 1 the
    alternative. None if the solver fails.
    """
    width, height = image_size
    focal = (width / 2) / np.tan(np.radians(config.CAMERA_HFOV_DEG) / 2)
    camera_matrix = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    half_nut, half_12 = config.QUAD_NUT_WIDTH_MM / 2, config.QUAD_FRET12_WIDTH_MM / 2
    length = config.NECK_LENGTH_MM
    model = np.float64([[0, -half_nut, 0], [0, half_nut, 0], [length, half_12, 0], [length, -half_12, 0]])
    try:
        count, rvecs, tvecs, errors = cv2.solvePnPGeneric(
            model, np.float64(corners), camera_matrix, None, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return None
    if not count:
        return None
    order = np.argsort(np.asarray(errors).ravel())
    pick = order[min(variant, len(order) - 1)]
    rotation, _ = cv2.Rodrigues(rvecs[pick])
    return CameraPose(np.linalg.inv(camera_matrix), rotation, -rotation.T @ tvecs[pick].reshape(3))


def parallax_across_mm(pose, x, y):
    """
    Distance in mm, across the neck, between the board point under pixel (x, y) and the board point right
    under a fingertip seen there FINGER_HEIGHT_MM above the board: the pixel's ray hits the board farther
    away than the fingertip really is. None with a degenerate ray.
    """
    ray = pose.rotation.T @ (pose.k_inv @ np.array([x, y, 1.0]))
    if abs(ray[2]) < 1e-9:
        return None
    s_board = -pose.position[2] / ray[2]
    s_finger = (-config.FINGER_HEIGHT_MM - pose.position[2]) / ray[2]
    if s_board <= 0 or s_finger <= 0:
        return None
    shift = (s_finger - s_board) * ray[:2]
    if np.hypot(shift[0], shift[1]) > config.MAX_PARALLAX_MM:
        return None
    return float(shift[1])


ABOVE_EDGE_RATIO = 1.004   # the low E edge of the neck must look this much longer than the high e edge to call the camera "above"


def signed_across_tilt(corners, image_size):
    """
    Degrees the camera is off the fretboard normal across the neck, signed: + when the camera is on the high e side
    (below the neck on the mirrored screen, the low E on top), - when it is on the low E side (above the neck). A planar
    view cannot tell the sign by itself (the two PnP solutions are mirror images); the side of the neck that looks longer
    can, because the edge closer to the camera is longer. The sign flips on noise near a frontal view, so the camera is
    called "above" only when the low E edge is clearly the longer one (ABOVE_EDGE_RATIO); the size comes from the pose.
    None if the pose cannot be estimated.
    """
    tl, tr, br, bl = (np.asarray(c, dtype=np.float64) for c in corners)
    pose = estimate_pose(corners, image_size, 0)
    if pose is None:
        return None
    to_camera = pose.position - np.array([config.NECK_LENGTH_MM / 2, 0.0, 0.0])
    size = float(np.degrees(np.arctan2(abs(to_camera[1]), abs(to_camera[2]))))
    above = np.linalg.norm(bl - tl) / np.linalg.norm(br - tr) >= ABOVE_EDGE_RATIO
    return -size if above else size


def camera_angles(pose):
    """
    (total, across, along) in degrees: how far the camera is from looking straight down the fretboard
    normal, split into tilt across the neck and along it. Magnitudes only: a single planar view can't
    tell the tilt direction reliably.
    """
    to_camera = pose.position - np.array([config.NECK_LENGTH_MM / 2, 0.0, 0.0])
    depth = abs(to_camera[2])
    total = np.degrees(np.arccos(depth / np.linalg.norm(to_camera)))
    across = np.degrees(np.arctan2(abs(to_camera[1]), depth))
    along = np.degrees(np.arctan2(abs(to_camera[0]), depth))
    return float(total), float(across), float(along)
