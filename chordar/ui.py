"""The look of the app's screens: a dark translucent panel, one light aqua accent, key caps for the commands, step headers."""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
ACCENT = (200, 230, 110)       # light aqua-green (BGR)
ACCENT_DIM = (150, 175, 90)
TEXT = (245, 245, 245)
MUTED = (190, 190, 190)
GOOD, WARN, BAD = (110, 220, 110), (60, 170, 255), (80, 80, 255)
PANEL = (34, 30, 28)


def panel(img, x0, y0, x1, y1, alpha=0.66, radius=10, color=PANEL, border=None):
    """A translucent rounded panel."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(int(x0), 0), max(int(y0), 0), min(int(x1), w - 1), min(int(y1), h - 1)
    if x1 - x0 < 2 * radius or y1 - y0 < 2 * radius:
        return
    layer = img.copy()
    cv2.rectangle(layer, (x0 + radius, y0), (x1 - radius, y1), color, -1)
    cv2.rectangle(layer, (x0, y0 + radius), (x1, y1 - radius), color, -1)
    for cx, cy in ((x0 + radius, y0 + radius), (x1 - radius, y0 + radius), (x0 + radius, y1 - radius), (x1 - radius, y1 - radius)):
        cv2.circle(layer, (cx, cy), radius, color, -1)
    cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, dst=img)
    if border is not None:
        r = radius
        for (cx, cy), start in (((x0 + r, y0 + r), 180), ((x1 - r, y0 + r), 270), ((x1 - r, y1 - r), 0), ((x0 + r, y1 - r), 90)):
            cv2.ellipse(img, (cx, cy), (r, r), 0, start, start + 90, border, 1, cv2.LINE_AA)
        cv2.line(img, (x0 + r, y0), (x1 - r, y0), border, 1, cv2.LINE_AA)
        cv2.line(img, (x0 + r, y1), (x1 - r, y1), border, 1, cv2.LINE_AA)
        cv2.line(img, (x0, y0 + r), (x0, y1 - r), border, 1, cv2.LINE_AA)
        cv2.line(img, (x1, y0 + r), (x1, y1 - r), border, 1, cv2.LINE_AA)


def text(img, label, org, scale=0.6, color=TEXT, thickness=1, shadow=True):
    """Text with a thin dark outline (readable over any picture). Returns its width."""
    if shadow:
        cv2.putText(img, label, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, label, org, FONT, scale, color, thickness, cv2.LINE_AA)
    return cv2.getTextSize(label, FONT, scale, thickness)[0][0]


def width_of(label, scale=0.6, thickness=1):
    return cv2.getTextSize(label, FONT, scale, thickness)[0][0]


def key_cap(img, key, x, y, scale=0.55):
    """A key cap ('SPACE', 'H', '1-8') with its top-left at (x, y); returns the x where it ends."""
    w = max(width_of(key, scale, 1) + 16, 28)
    h = int(30 * scale / 0.55)
    panel(img, x, y, x + w, y + h, alpha=0.9, radius=5, color=(75, 70, 66), border=(150, 150, 150))
    text(img, key, (x + (w - width_of(key, scale, 1)) // 2, y + h - 9), scale, TEXT, 1, shadow=False)
    return x + w


def key_row(img, items, x, y, scale=0.55, gap=18):
    """Key caps with their meaning: items = [(key, label), ...]. Returns the x where the row ends."""
    for key, label in items:
        x = key_cap(img, key, x, y, scale) + 8
        x += text(img, label, (x, y + int(30 * scale / 0.55) - 9), scale, TEXT, 1) + gap
    return x - gap


def row_width(items, scale=0.55, gap=18):
    total = 0
    for key, label in items:
        total += max(width_of(key, scale, 1) + 16, 28) + 8 + width_of(label, scale, 1) + gap
    return total - gap


def pill(img, label, x, y, color, scale=0.6):
    """A short colored message with its top-left at (x, y); returns the x where it ends."""
    w = width_of(label, scale, 1) + 44
    h = int(34 * scale / 0.6)
    panel(img, x, y, x + w, y + h, alpha=0.85, radius=h // 2, color=(40, 36, 34), border=color)
    cv2.circle(img, (x + 14, y + h // 2), 5, color, -1, cv2.LINE_AA)
    text(img, label, (x + 26, y + h - 10), scale, TEXT, 1, shadow=False)
    return x + w


def header(img, step, title, subtitle=None):
    """The top bar of a set-up screen: 'STEP 1 OF 2', the title and one line of explanation. Returns its height."""
    w = img.shape[1]
    height = 96 if subtitle else 70
    panel(img, 0, -20, w, height, alpha=0.78, radius=1)
    cv2.rectangle(img, (0, height - 3), (w, height), ACCENT, -1)
    text(img, step, (24, 26), 0.5, ACCENT, 1, shadow=False)
    text(img, title, (24, 56), 0.95, TEXT, 2, shadow=False)
    if subtitle:
        text(img, subtitle, (24, 84), 0.6, MUTED, 1, shadow=False)
    return height


def progress(img, done, total, x, y, labels=None):
    """Numbered dots: the ones before `done` filled, the one at `done` ringed (the current step)."""
    for i in range(total):
        cx = x + i * 34
        if i < done:
            cv2.circle(img, (cx, y), 12, ACCENT, -1, cv2.LINE_AA)
            text(img, str(i + 1), (cx - 5, y + 5), 0.5, (30, 30, 30), 1, shadow=False)
        elif i == done:
            cv2.circle(img, (cx, y), 12, ACCENT, 2, cv2.LINE_AA)
            text(img, str(i + 1), (cx - 5, y + 5), 0.5, ACCENT, 1, shadow=False)
        else:
            cv2.circle(img, (cx, y), 12, (130, 130, 130), 1, cv2.LINE_AA)
            text(img, str(i + 1), (cx - 5, y + 5), 0.5, (150, 150, 150), 1, shadow=False)
