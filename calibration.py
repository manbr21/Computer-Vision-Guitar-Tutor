import cv2

from map_fret_board import map_guitar, NUM_STRINGS
from map_hands import get_fingertip_positions

MIN_STRING_GAP = 15          # px between adjacent strings; below this, string discrimination gets unreliable
STABLE_FRAMES_REQUIRED = 20  # ~a second of good, uninterrupted framing before auto-advancing

GOOD_COLOR = (0, 255, 0)
WARN_COLOR = (0, 165, 255)
BAD_COLOR = (0, 0, 255)


def run_calibration(cap):
    """
    Guided setup screen shown before the main tracking loop. Every frame, checks
    whether the 4 ArUco markers are visible, whether the neck is close enough to
    the camera for reliable string discrimination, and whether a hand is being
    tracked - then shows a single actionable message instead of the user having
    to guess why detection looks off once the real lesson starts.

    Advances automatically once everything looks good for STABLE_FRAMES_REQUIRED
    consecutive frames, or immediately on ENTER (manual skip). Returns False if
    the user quits (ESC) during calibration, True otherwise.
    """
    stable_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            return False

        display, neck = map_guitar(frame)
        _, fingertips, _ = get_fingertip_positions(frame.copy())

        markers_ok = neck is not None
        string_gap = neck.width_px() / NUM_STRINGS if markers_ok else 0
        distance_ok = markers_ok and string_gap >= MIN_STRING_GAP
        hand_ok = bool(fingertips)

        if not markers_ok:
            message = "Move the guitar so all 4 ArUco markers (IDs 0-3) are visible"
            color = BAD_COLOR
        elif not distance_ok:
            message = f"Move the camera/guitar closer (string gap: {string_gap:.0f}px, need >= {MIN_STRING_GAP}px)"
            color = WARN_COLOR
        elif not hand_ok:
            message = "Good distance - place your fretting hand on the neck"
            color = WARN_COLOR
        else:
            message = "Good to go!"
            color = GOOD_COLOR

        if markers_ok and distance_ok and hand_ok:
            stable_frames += 1
        else:
            stable_frames = 0

        cv2.putText(display, "CALIBRATION", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(display, message, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display, f"Stable: {min(stable_frames, STABLE_FRAMES_REQUIRED)}/{STABLE_FRAMES_REQUIRED}",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, "Press ENTER to start anyway, ESC to quit", (20, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Hand + Guitar Tracking", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            return False
        if key == 13:  # ENTER - skip calibration and start anyway
            return True

        if stable_frames >= STABLE_FRAMES_REQUIRED:
            return True
