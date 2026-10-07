import cv2


def put_text_bg(img, text, org, scale, color, thickness=2, pad=6):
    """cv2.putText over a darkened box so the text stays readable on any background."""
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    x0, y0 = max(x - pad, 0), max(y - h - pad, 0)
    x1, y1 = min(x + w + pad, img.shape[1]), min(y + baseline + pad, img.shape[0])
    img[y0:y1, x0:x1] = cv2.convertScaleAbs(img[y0:y1, x0:x1], alpha=0.35)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
