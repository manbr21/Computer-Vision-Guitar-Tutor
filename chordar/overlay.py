"""Everything drawn over the video: text, neck grid, chord targets, fingertip labels and the chord diagram."""
import cv2
import numpy as np

from chordar.chords import CHORD_LIBRARY, STRING_NAMES, fretted_notes
from chordar.config import NUM_FRETS, NUM_STRINGS
from chordar.geometry import FRET_BOUNDS, outline_u_range, string_u

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_FILL, PANEL_BORDER = (40, 40, 40), (0, 255, 0)
DIAGRAM_FRETS = 4  # frets shown in the chord diagram


def put_text_bg(img, text, org, scale, color, thickness=2, pad=6):
    """cv2.putText over a darkened box so the text stays readable on any background."""
    (w, h), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = org
    x0, y0 = max(x - pad, 0), max(y - h - pad, 0)
    x1, y1 = min(x + w + pad, img.shape[1]), min(y + baseline + pad, img.shape[0])
    img[y0:y1, x0:x1] = cv2.convertScaleAbs(img[y0:y1, x0:x1], alpha=0.35)
    cv2.putText(img, text, org, FONT, scale, color, thickness)


# --- on the neck ------------------------------------------------------------------------------------------

def draw_neck(display, neck):
    """Red outline, numbered fret wires and strings; magenta dots mark the marker corners in use."""
    u_min, u_max = outline_u_range()
    outline = np.array([neck.to_image(u_min, 0), neck.to_image(u_max, 0),
                        neck.to_image(u_max, 1), neck.to_image(u_min, 1)], dtype=np.int32)
    cv2.polylines(display, [outline], True, (0, 0, 255), 3)
    for corner in neck.corners:
        cv2.circle(display, tuple(int(v) for v in corner), 5, (255, 0, 255), -1)

    for n in range(1, NUM_FRETS + 1):
        wire_left = neck.to_image(u_min, FRET_BOUNDS[n])
        cv2.line(display, wire_left, neck.to_image(u_max, FRET_BOUNDS[n]), (0, 255, 255), 2)
        cv2.putText(display, str(n), (wire_left[0] + 5, wire_left[1] - 5), FONT, 0.5, (0, 255, 255), 1)

    for s in range(NUM_STRINGS):
        cv2.line(display, neck.to_image(string_u(s), 0), neck.to_image(string_u(s), 1), (155, 255, 0), 2)


def draw_chord_targets(display, neck, chord):
    """Yellow dots, numbered with the finger, where each fretted note of the chord goes."""
    for string_idx, fret, finger in fretted_notes(chord):
        x, y = neck.target(string_idx, fret)
        cv2.circle(display, (x, y), 8, (0, 255, 255), -1)
        cv2.circle(display, (x, y), 8, (255, 255, 255), 2)
        cv2.putText(display, str(finger), (x - 7, y + 7), FONT, 0.7, (0, 0, 0), 2)


def draw_fingertips(display, neck, fingertips, parallax):
    """
    A green dot per fingertip, a cyan ring at its estimated contact point (with the parallax correction)
    and a label with the string and fret it was assigned to, for example D2.
    """
    for x, y in fingertips.values():
        cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
        label_at = (x, y)
        contact = neck.contact_pixel(x, y) if parallax else None
        if contact is not None:
            cv2.circle(display, contact, 7, (255, 255, 0), 2)
            label_at = contact
        hit = neck.locate(x, y, parallax)
        if hit is not None:
            put_text_bg(display, f"{STRING_NAMES[hit[0]]}{hit[1]}", (label_at[0] + 10, label_at[1] - 10),
                        0.5, (255, 255, 255), thickness=1)


# --- chord diagram ----------------------------------------------------------------------------------------

def draw_chord_diagram(frame, x, y, width, height, chord):
    """Chord chart in a panel: strings are rows (low E on top, like the neck on screen), frets are columns."""
    info = CHORD_LIBRARY[chord]
    cv2.rectangle(frame, (x, y), (x + width, y + height), PANEL_FILL, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), PANEL_BORDER, 2)
    cv2.putText(frame, info["name"], (x + 10, y + 25), FONT, 0.7, (255, 255, 255), 2)

    left, top = x + 48, y + 58
    fret_w = (width - 60) // DIAGRAM_FRETS
    string_gap = (height - 75) // (NUM_STRINGS - 1)
    right, bottom = left + fret_w * DIAGRAM_FRETS, top + string_gap * (NUM_STRINGS - 1)

    for s in range(NUM_STRINGS):
        cv2.line(frame, (left, top + s * string_gap), (right, top + s * string_gap), (200, 200, 200), 1)
        cv2.putText(frame, STRING_NAMES[s], (x + 10, top + s * string_gap + 5), FONT, 0.45, (200, 200, 200), 1)
    for n in range(DIAGRAM_FRETS + 1):  # the nut (n = 0) is thicker
        cv2.line(frame, (left + n * fret_w, top), (left + n * fret_w, bottom), (200, 200, 200), 3 if n == 0 else 1)
    for n in range(1, DIAGRAM_FRETS + 1):
        cv2.putText(frame, str(n), (left + int((n - 0.5) * fret_w) - 4, top - 14), FONT, 0.45, (255, 255, 255), 1)

    for s, (fret, finger) in enumerate(zip(info["frets"], info["fingers"])):
        row_y = top + s * string_gap
        if fret is None:  # muted
            cv2.putText(frame, "X", (left - 17, row_y + 5), FONT, 0.5, (255, 100, 100), 2)
        elif fret == 0:   # open
            cv2.circle(frame, (left - 12, row_y), 5, (100, 255, 100), 2)
        else:
            cx = int(left + (fret - 0.5) * fret_w)
            cv2.circle(frame, (cx, row_y), 8, (0, 255, 255), -1)
            cv2.circle(frame, (cx, row_y), 8, (255, 255, 255), 1)
            if finger:
                cv2.putText(frame, str(finger), (cx - 4, row_y + 4), FONT, 0.4, (0, 0, 0), 1)


def draw_chord_instructions(frame, x, y, width, chord):
    """Where each finger goes, in its own dark box right-aligned to x + width."""
    lines = [f"Finger {finger} on string {NUM_STRINGS - s} ({STRING_NAMES[s]}) fret {fret}"
             for s, fret, finger in fretted_notes(chord)] or ["Open strings or muted"]

    scale, thickness, line_h, pad = 0.5, 1, 22, 10
    box_w = max(width, max(cv2.getTextSize(line, FONT, scale, thickness)[0][0] for line in lines) + 2 * pad)
    box_h = len(lines) * line_h + pad
    left = x + width - box_w

    cv2.rectangle(frame, (left, y), (left + box_w, y + box_h), PANEL_FILL, -1)
    cv2.rectangle(frame, (left, y), (left + box_w, y + box_h), PANEL_BORDER, 2)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (left + pad, y + pad + 10 + i * line_h), FONT, scale, (255, 255, 255), thickness)
