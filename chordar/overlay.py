"""Everything drawn over the video: text, neck grid, chord targets, fingertip labels and the chord diagram."""
import cv2
import numpy as np

from chordar.chords import CHORD_LIBRARY, STRING_NAMES
from chordar.config import NUM_FRETS, NUM_STRINGS
from chordar.geometry import FRET_BOUNDS, outline_u_range, string_u

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_FILL, PANEL_BORDER = (40, 40, 40), (0, 255, 0)
DIAGRAM_FRETS = 4  # frets shown in the chord diagram
CAPO_OVERHANG = 0.06  # how far the capo bar sticks out past the neck outline, in quad widths
ORANGE = (0, 140, 255)


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


def draw_capo(display, neck, capo):
    """A capo bar across all the strings, just behind the wire of fret `capo`."""
    u_min, u_max = outline_u_range()
    u_left, u_right = u_min - CAPO_OVERHANG, u_max + CAPO_OVERHANG
    gap = FRET_BOUNDS[capo] - FRET_BOUNDS[capo - 1]
    t_nut_side, t_fret_side = FRET_BOUNDS[capo] - 0.4 * gap, FRET_BOUNDS[capo] - 0.1 * gap
    bar = np.array([neck.to_image(u_left, t_nut_side), neck.to_image(u_right, t_nut_side),
                    neck.to_image(u_right, t_fret_side), neck.to_image(u_left, t_fret_side)], dtype=np.int32)
    shade = display.copy()
    cv2.fillPoly(shade, [bar], ORANGE)
    cv2.addWeighted(shade, 0.7, display, 0.3, 0, dst=display)
    cv2.polylines(display, [bar], True, (255, 255, 255), 2)


def draw_chord_targets(display, neck, notes, done=None):
    """Yellow dots, numbered with the finger, where each fretted note goes; a note that is covered turns green with a check."""
    for i, (string_idx, fret, finger) in enumerate(notes):
        x, y = neck.target(string_idx, fret)
        if done is not None and done[i]:
            cv2.circle(display, (x, y), 10, (0, 200, 0), -1)
            cv2.circle(display, (x, y), 10, (255, 255, 255), 2)
            cv2.polylines(display, [np.array([(x - 5, y), (x - 1, y + 4), (x + 6, y - 5)], np.int32)], False,
                          (255, 255, 255), 2)
        else:
            cv2.circle(display, (x, y), 10, (0, 255, 255), -1)
            cv2.circle(display, (x, y), 10, (255, 255, 255), 2)
            cv2.putText(display, str(finger), (x - 7, y + 7), FONT, 0.7, (0, 0, 0), 2)


def draw_arrow(display, start, end):
    """Arrow from a fingertip to the note it should cover."""
    if np.hypot(end[0] - start[0], end[1] - start[1]) > 12:
        cv2.arrowedLine(display, tuple(int(v) for v in start), tuple(int(v) for v in end), (0, 165, 255), 3,
                        tipLength=0.25)


def draw_fingertips(display, neck, fingertips, parallax, debug=False):
    """
    A light-blue dot per fingertip. With debug, also a cyan ring at its estimated contact point (with the parallax
    correction) and a label with the string and fret it was assigned to, for example D2.
    """
    for x, y in fingertips.values():
        cv2.circle(display, (x, y), 7, (0, 0, 0), -1)
        cv2.circle(display, (x, y), 5, (255, 200, 60), -1)  # light blue: stands out from the green "done" notes
        if not debug:
            continue
        label_at = (x, y)
        contact = neck.contact_pixel(x, y) if parallax else None
        if contact is not None:
            cv2.circle(display, contact, 7, (255, 255, 0), 2)
            label_at = contact
        hit = neck.locate(x, y, parallax)
        if hit is not None:
            put_text_bg(display, f"{STRING_NAMES[hit[0]]}{hit[1]}", (label_at[0] + 10, label_at[1] - 10),
                        0.5, (255, 255, 255), thickness=1)


# --- heads-up display -------------------------------------------------------------------------------------

def draw_gauge(display, center, radius, percent):
    """A ring that fills with the accuracy (grey if unknown), red -> orange -> green, the number inside."""
    cv2.circle(display, center, radius, (50, 50, 50), -1)
    cv2.circle(display, center, radius, (90, 90, 90), 6)
    if percent is None:
        put_text_bg(display, "--", (center[0] - 14, center[1] + 7), 0.7, (200, 200, 200), thickness=2, pad=0)
        return
    color = (0, 200, 0) if percent == 100 else (0, 165, 255) if percent >= 50 else (60, 60, 230)
    if percent > 0:
        cv2.ellipse(display, center, (radius, radius), 0, -90, -90 + 360 * percent / 100, color, 6, cv2.LINE_AA)
    text = f"{percent}"
    (w, h), _ = cv2.getTextSize(text, FONT, 0.8, 2)
    cv2.putText(display, text, (center[0] - w // 2, center[1] + h // 2), FONT, 0.8, (255, 255, 255), 2)


def draw_banner(display, text, color, y_from_bottom=60):
    """One short message, large, centered near the bottom of the image."""
    scale = 0.9
    (w, _), _ = cv2.getTextSize(text, FONT, scale, 2)
    put_text_bg(display, text, ((display.shape[1] - w) // 2, display.shape[0] - y_from_bottom), scale, color, pad=10)


# --- chord diagram ----------------------------------------------------------------------------------------

def draw_chord_diagram(frame, x, y, width, height, chord, capo=0):
    """
    Chord chart in a panel: strings are rows (low E on top, like the neck on screen), frets are columns. With
    a capo the thick line is the capo, the columns are numbered with the real frets and the open strings are
    the ones the capo plays.
    """
    info = CHORD_LIBRARY[chord]
    cv2.rectangle(frame, (x, y), (x + width, y + height), PANEL_FILL, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), PANEL_BORDER, 2)
    cv2.putText(frame, info["name"], (x + 10, y + 25), FONT, 0.7, (255, 255, 255), 2)
    if capo:
        cv2.putText(frame, f"capo {capo}", (x + width - 70, y + 25), FONT, 0.5, ORANGE, 1)

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
        cv2.putText(frame, str(capo + n), (left + int((n - 0.5) * fret_w) - 4, top - 14), FONT, 0.45, (255, 255, 255), 1)

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
