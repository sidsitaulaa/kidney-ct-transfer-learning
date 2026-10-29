"""Image preprocessing shared by every model: crop to the body and pad to a square.

CT exports here are 512x512 or larger non-square frames with wide black margins and the
scanner table. Cropping to the body and padding (instead of stretching) to a square puts
more of each network's fixed input resolution on anatomy and keeps proportions intact.
"""

from __future__ import annotations

import numpy as np

BODY_THRESHOLD = 20      # grey level separating body from air/background (0-255)
MIN_LINE_FRACTION = 0.05  # a row/column belongs to the body if >5% of it is above the threshold
MARGIN = 0.03            # keep a small border around the body (fraction of the crop size)


def body_crop_square(image: np.ndarray) -> np.ndarray:
    """Crop an HxWx3 uint8 image to the body's bounding box (plus margin) and zero-pad it to a square."""
    grey = image.mean(axis=2) if image.ndim == 3 else image
    mask = grey > BODY_THRESHOLD
    rows = np.flatnonzero(mask.mean(axis=1) > MIN_LINE_FRACTION)
    cols = np.flatnonzero(mask.mean(axis=0) > MIN_LINE_FRACTION)
    if rows.size == 0 or cols.size == 0:  # blank image: leave it alone
        top, bottom, left, right = 0, image.shape[0], 0, image.shape[1]
    else:
        top, bottom, left, right = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        pad_r = int(MARGIN * (bottom - top))
        pad_c = int(MARGIN * (right - left))
        top, bottom = max(0, top - pad_r), min(image.shape[0], bottom + pad_r)
        left, right = max(0, left - pad_c), min(image.shape[1], right + pad_c)
    crop = image[top:bottom, left:right]
    h, w = crop.shape[:2]
    side = max(h, w)
    out = np.zeros((side, side) + crop.shape[2:], dtype=crop.dtype)
    out[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = crop
    return out
