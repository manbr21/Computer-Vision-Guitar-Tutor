import cv2
import mediapipe as mp
import math
import time
from map_hands import get_fingertip_positions
from map_fret_board import map_guitar
from graphics_code import draw_chord_diagram, draw_chord_instructions
import graphics_code

from match_chord import match_chord
from calibration import run_calibration, PlacementMonitor
from ui_utils import put_text_bg

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

thumb_indices = {(0,1), (1,2), (2,3), (3,4)}
custom_connections = [conn for conn in mp_hands.HAND_CONNECTIONS if conn not in thumb_indices]


cap = cv2.VideoCapture(0)

if not run_calibration(cap):
    cap.release()
    cv2.destroyAllWindows()
    exit()

current_chord = "C"  # <-- set this dynamically if needed


# Utility: compute accuracy of observed fingertips vs expected chord points
def compute_chord_accuracy(expected_positions, observed_points, max_distance=80):
    """
    expected_positions: dict mapping (s_idx, fret) -> (x_dst, y_dst)
    observed_points: list of (bx, by) observed in same dst coords
    max_distance: distance considered full-miss (>= max_distance -> 0 score for that string)

    Returns: (percent_score, details)
    details: list of (s_idx, expected_fret, matched_bool, distance)
    """
    # Build per-string expected (pick fretted positions only)
    per_string_expected = {}
    for (s_idx, fret), pos in expected_positions.items():
        if fret is not None and fret > 0:
            # prefer higher fret if duplicates (shouldn't happen normally)
            per_string_expected[s_idx] = (fret, pos)

    details = []
    correct = 0
    total = 0

    for s_idx, (fret_needed, pos) in per_string_expected.items():
        total += 1
        best_d = float('inf')
        for (bx, by) in observed_points:
            d = math.hypot(bx - pos[0], by - pos[1])
            if d < best_d:
                best_d = d

        matched = best_d <= max_distance
        if matched:
            correct += 1
        details.append((s_idx, fret_needed, matched, best_d if best_d != float('inf') else None))

    pct = int(100 * correct / total) if total else 0
    return pct, details


def compute_accuracy_from_lists(expected_list, observed_list, max_distance=80):
    """Compute percent of expected points that have an observed point within max_distance."""
    correct = 0
    total = len(expected_list)
    details = []
    for ex in expected_list:
        best = float('inf')
        for ob in observed_list:
            d = math.hypot(ex[0]-ob[0], ex[1]-ob[1])
            if d < best:
                best = d
        matched = best <= max_distance
        if matched:
            correct += 1
        details.append((ex, best, matched))
    pct = int(100 * correct / total) if total else 0
    return pct, details


def compute_accuracy_discrete(expected_string_frets, fingertips, neck, parallax=True):
    """
    Percent of expected (string, fret) pairs covered by a fingertip. Each fingertip
    is mapped into neck coordinates, so the string comes from its position across
    the neck and the fret from its position along it. Horizontal tolerance is the
    whole fret space (between the two wires), not just its center; vertically the
    finger must be inside the string's lane.
    """
    if not expected_string_frets:
        return 0

    observed = {neck.locate(x, y, parallax) for (x, y) in fingertips.values()}
    correct = sum(1 for e in expected_string_frets if e in observed)
    return int(100 * correct / len(expected_string_frets))


parallax_on = True   # P toggles the fingertip parallax correction
pose_variant = 0     # X swaps between the two possible camera-pose solutions
placement_monitor = PlacementMonitor()  # warns when the guitar drifts out of the calibrated position

while True:
    ret, frame = cap.read()
    if not ret:
        break

    loop_start = time.time()

    raw_frame = frame.copy()
    display, neck = map_guitar(frame, pose_variant)
    # the fretting hand is the one over the first frets, not the strumming hand
    hand_anchor = neck.to_image(0.5, 0.2) if neck is not None else None
    _, fingertips, landmarks_list = get_fingertip_positions(raw_frame, hand_anchor)

    if neck is None:
        put_text_bg(display, "ArUco markers not detected (IDs 0-3)", (20, 90), 0.6, (0, 0, 255))
    else:
        status = f"Parallax fix: {'ON' if parallax_on else 'OFF'} (pose {'AB'[pose_variant]})"
        angles = neck.camera_angles()
        if angles is not None:
            status += f" | camera angle {angles[0]:.0f} deg"
        put_text_bg(display, status, (20, 120), 0.5, (255, 255, 255), thickness=1)

    drift_hint = placement_monitor.update(neck, (display.shape[1], display.shape[0]), loop_start)
    if drift_hint is not None:
        put_text_bg(display, f"{drift_hint} - press C to recalibrate", (20, 150), 0.55, (0, 165, 255), thickness=1)

    if neck is not None:
        # Draw fingertip positions, each labeled with the string and fret it was assigned to
        for name, (x, y) in fingertips.items():
            hit = neck.locate(x, y, parallax_on)

            cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
            label_at = (x, y)
            if parallax_on:
                contact = neck.contact_pixel(x, y)
                if contact is not None:
                    cv2.circle(display, contact, 7, (255, 255, 0), 2)
                    label_at = contact
            if hit is not None:
                string_idx, fret_idx = hit
                put_text_bg(display, f"{graphics_code.STRING_NAMES[string_idx]}{fret_idx}",
                            (label_at[0] + 10, label_at[1] - 10), 0.5, (255, 255, 255), thickness=1)

    # ===============================================================
    # DRAW CURRENT CHORD ON FRETBOARD (yellow overlay)
    # ===============================================================
    expected_string_frets = []

    if current_chord and neck is not None:
        for string_idx, fret in enumerate(graphics_code.CHORD_LIBRARY[current_chord]['frets']):
            finger_num = graphics_code.CHORD_LIBRARY[current_chord]['fingers'][string_idx]
            if fret is not None and fret > 0 and finger_num:
                x, y = neck.target(string_idx, fret)

                # Draw yellow filled circle + white outline
                cv2.circle(display, (x, y), 8, (0, 255, 255), -1)
                cv2.circle(display, (x, y), 8, (255, 255, 255), 2)
                cv2.putText(display, str(finger_num), (x-7, y+7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                expected_string_frets.append((string_idx, fret))

    if expected_string_frets:
        pct = compute_accuracy_discrete(expected_string_frets, fingertips, neck, parallax_on)
        put_text_bg(display, f"Accuracy: {pct}%", (20, 60), 0.8, (0,255,0) if pct==100 else (0,165,255))
    else:
        # no expected points (open chord/no fretted notes) - show N/A
        put_text_bg(display, "Accuracy: N/A", (20, 60), 0.8, (200,200,200))


    # Draw chord diagram on the frame, with the finger instructions in a box below it
    diagram_x, diagram_y, diagram_w, diagram_h = display.shape[1] - 220, 20, 200, 190
    draw_chord_diagram(
        display,
        x=diagram_x,
        y=diagram_y,
        width=diagram_w,
        height=diagram_h,
        current_chord=current_chord
    )
    draw_chord_instructions(
        display,
        x=diagram_x,
        y=diagram_y + diagram_h + 10,
        width=diagram_w,
        current_chord=current_chord
    )

    put_text_bg(display, f"Current Chord: {current_chord}", (20, 30), 0.7, (255, 255, 255))

    put_text_bg(display, "Press 1-8 to change chords, P parallax fix, X swap pose, C recalibrate, ESC to exit",
                (20, display.shape[0] - 20), 0.5, (255, 255, 255), thickness=1)

    fps = 1.0 / (time.time() - loop_start) if time.time() != loop_start else 0.0
    put_text_bg(display, f"FPS: {fps:.1f}", (display.shape[1] - 130, display.shape[0] - 20), 0.7, (0, 255, 0))

    cv2.imshow("Hand + Guitar Tracking", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('1'):
        current_chord = "A"
    elif key == ord('2'):
        current_chord = "Am"
    elif key == ord('3'):
        current_chord = "C"
    elif key == ord('4'):
        current_chord = "D"
    elif key == ord('5'):
        current_chord = "Dm"
    elif key == ord('6'):
        current_chord = "E"
    elif key == ord('7'):
        current_chord = "Em"
    elif key == ord('8'):
        current_chord = "G"
    elif key == ord('p'):
        parallax_on = not parallax_on
    elif key == ord('x'):
        pose_variant = 1 - pose_variant
    elif key == ord('c'):
        run_calibration(cap)  # ESC there just comes back to playing
        placement_monitor.reset()


cap.release()
cv2.destroyAllWindows()
