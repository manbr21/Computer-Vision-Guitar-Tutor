"""The display window: full screen by default, with the image scaled to fit without stretching."""
import cv2
import numpy as np

from chordar import config

_fullscreen = {}


def open_window(name):
    """Creates the window, full screen if config.FULLSCREEN."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    set_fullscreen(name, config.FULLSCREEN)


def set_fullscreen(name, on):
    _fullscreen[name] = on
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL)


def toggle_fullscreen(name):
    set_fullscreen(name, not _fullscreen.get(name, False))


def fit(image, width, height):
    """Scales the image to the largest size that fits in width x height, centered on black bars."""
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = max(int(w * scale), 1), max(int(h * scale), 1)
    canvas = np.zeros((height, width, 3), np.uint8)
    x, y = (width - new_w) // 2, (height - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return canvas


def show(name, image):
    """Shows the image in the window, keeping its aspect ratio when the window is larger than the image."""
    try:
        _, _, width, height = cv2.getWindowImageRect(name)
    except cv2.error:
        width = height = 0
    if width > 0 and height > 0 and (width, height) != (image.shape[1], image.shape[0]):
        image = fit(image, width, height)
    cv2.imshow(name, image)
