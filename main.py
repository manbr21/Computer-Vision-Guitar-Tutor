import cv2
import mediapipe as mp
import math
import time
from map_hands import get_fingertip_positions
from map_fret_board import map_guitar
from graphics_code import draw_chord_diagram
import graphics_code

from match_chord import match_chord
from calibration import run_calibration

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

thumb_indices = {(0,1), (1,2), (2,3), (3,4)}
custom_connections = [conn for conn in mp_hands.HAND_CONNECTIONS if conn not in thumb_indices]

string_labels = ["E", "A", "D", "G", "B", "E"]

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


def compute_accuracy_discrete(expected_string_frets, fingertips, neck):
    """
    Percent of expected (string, fret) pairs covered by a fingertip. Each fingertip
    is mapped into neck coordinates, so the string comes from its position across
    the neck and the fret from its position along it. Horizontal tolerance is the
    whole fret space (between the two wires), not just its center; vertically the
    finger must be inside the string's lane.
    """
    if not expected_string_frets:
        return 0

    observed = {neck.locate(x, y) for (x, y) in fingertips.values()}
    correct = sum(1 for e in expected_string_frets if e in observed)
    return int(100 * correct / len(expected_string_frets))


while True:
    ret, frame = cap.read()
    if not ret:
        break

    loop_start = time.time()

    raw_frame = frame.copy()
    display, neck = map_guitar(frame)
    _, fingertips, landmarks_list = get_fingertip_positions(raw_frame)

    if neck is None:
        cv2.putText(display, "ArUco markers not detected (IDs 0-3)", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if neck is not None:
        # Draw fingertip positions
        for name, (x, y) in fingertips.items():
            hit = neck.locate(x, y)

            if hit is not None:
                string_idx, fret_idx = hit
                print(f"{name}: String {string_labels[string_idx]}, Fret {fret_idx}")
            else:
                print(f"{name}: Not on string")

            cv2.circle(display, (x, y), 5, (0, 255, 0), -1)

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
        pct = compute_accuracy_discrete(expected_string_frets, fingertips, neck)
        cv2.putText(display, f"Accuracy: {pct}%", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if pct==100 else (0,165,255), 2)
    else:
        # no expected points (open chord/no fretted notes) - show N/A
        cv2.putText(display, f"Accuracy: N/A", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)


    # Draw chord diagram on the frame
    draw_chord_diagram(
        display,
        x=20,
        y=display.shape[0]-220,
        width=200,
        height=180,
        show_instructions=True,
        current_chord=current_chord
    )

    # Display current chord
    cv2.putText(display, f"Current Chord: {current_chord}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display instructions
    cv2.putText(display, "Press 1-8 to change chords, ESC to exit", (20, display.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    fps = 1.0 / (time.time() - loop_start) if time.time() != loop_start else 0.0
    cv2.putText(display, f"FPS: {fps:.1f}", (display.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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


cap.release()
cv2.destroyAllWindows()
