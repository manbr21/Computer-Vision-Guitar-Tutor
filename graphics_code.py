import cv2

# ============================================================================
# CHORD LIBRARY - Data structure for all chords
# ============================================================================
CHORD_LIBRARY = {
    "C": {"frets": [None, 3, 2, 0, 1, 0], "fingers": [None, 3, 2, None, 1, None], "name": "C Major"},
    "D": {"frets": [None, None, 0, 2, 3, 2], "fingers": [None, None, None, 1, 3, 2], "name": "D Major"},
    "E": {"frets": [0, 2, 2, 1, 0, 0], "fingers": [None, 2, 3, 1, None, None], "name": "E Major"},
    "G": {"frets": [3, 2, 0, 0, 0, 3], "fingers": [3, 2, None, None, None, 4], "name": "G Major"},
    "A": {"frets": [None, 0, 2, 2, 2, 0], "fingers": [None, None, 1, 2, 3, None], "name": "A Major"},
    "Em": {"frets": [0, 2, 2, 0, 0, 0], "fingers": [None, 2, 3, None, None, None], "name": "E Minor"},
    "Am": {"frets": [None, 0, 2, 2, 1, 0], "fingers": [None, None, 2, 3, 1, None], "name": "A Minor"},
    "Dm": {"frets": [None, None, 0, 2, 3, 1], "fingers": [None, None, None, 2, 4, 1], "name": "D Minor"},
}

STRING_NAMES = ["E", "A", "D", "G", "B", "e"]  # index 0 = lowest (6th) string
NUM_DIAGRAM_FRETS = 4

# ============================================================================
# DRAW CHORD DIAGRAM - Draws the chord picture in a panel
# ============================================================================
def draw_chord_diagram(frame, x, y, width, height, current_chord=None):
    if not current_chord:
        return
    
    chord_info = CHORD_LIBRARY[current_chord]
    
    # Draw background panel
    cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 40), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
    
    # Draw chord name
    cv2.putText(frame, chord_info['name'], (x + 10, y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Calculate diagram dimensions: strings are rows (low E on top, same order as the
    # neck on screen), frets are columns, nut on the left
    diagram_x = x + 48
    diagram_y = y + 58
    diagram_w = width - 60
    diagram_h = height - 75

    string_spacing = diagram_h // 5  # 6 strings
    for i in range(6):
        sy = diagram_y + i * string_spacing
        cv2.line(frame, (diagram_x, sy), (diagram_x + diagram_w, sy), (200, 200, 200), 1)
        cv2.putText(frame, STRING_NAMES[i], (x + 10, sy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    fret_spacing = diagram_w // NUM_DIAGRAM_FRETS
    for i in range(NUM_DIAGRAM_FRETS + 1):
        fx = diagram_x + i * fret_spacing
        thickness = 3 if i == 0 else 1  # Nut is thicker
        cv2.line(frame, (fx, diagram_y), (fx, diagram_y + 5 * string_spacing), (200, 200, 200), thickness)
    for n in range(1, NUM_DIAGRAM_FRETS + 1):
        cx = diagram_x + int((n - 0.5) * fret_spacing)
        cv2.putText(frame, str(n), (cx - 4, diagram_y - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Draw finger positions on diagram
    marker_x = diagram_x - 12
    for string_idx, fret in enumerate(chord_info['frets']):
        sy = diagram_y + string_idx * string_spacing

        if fret is None:
            cv2.putText(frame, "X", (marker_x - 5, sy + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2)
        elif fret == 0:
            cv2.circle(frame, (marker_x, sy), 5, (100, 255, 100), 2)
        else:
            fx = diagram_x + (fret - 0.5) * fret_spacing
            cv2.circle(frame, (int(fx), int(sy)), 8, (0, 255, 255), -1)
            cv2.circle(frame, (int(fx), int(sy)), 8, (255, 255, 255), 1)
            finger_num = chord_info['fingers'][string_idx]
            if finger_num:
                cv2.putText(frame, str(finger_num), (int(fx) - 4, int(sy) + 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)


def draw_chord_instructions(frame, x, y, width, current_chord=None):
    """Text list of where each finger goes, in its own dark box right-aligned to x + width."""
    if not current_chord:
        return

    chord_info = CHORD_LIBRARY[current_chord]
    instructions = []
    for string_idx, fret in enumerate(chord_info['frets']):
        finger = chord_info['fingers'][string_idx]
        if fret is not None and fret > 0 and finger:
            instructions.append(f"Finger {finger} on string {6 - string_idx} ({STRING_NAMES[string_idx]}) fret {fret}")
    if not instructions:
        instructions = ["Open strings or muted"]

    font, scale, thickness, line_h, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, 22, 10
    text_w = max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in instructions)
    box_w = max(width, text_w + 2 * pad)
    box_h = len(instructions) * line_h + pad
    x0 = x + width - box_w

    cv2.rectangle(frame, (x0, y), (x0 + box_w, y + box_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (x0, y), (x0 + box_w, y + box_h), (0, 255, 0), 2)
    for i, line in enumerate(instructions):
        cv2.putText(frame, line, (x0 + pad, y + pad + 10 + i * line_h),
                    font, scale, (255, 255, 255), thickness)
