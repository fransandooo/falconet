"""Training targets: from boxes to the maps the network should output."""

from __future__ import annotations

import math

import numpy as np

from .model import STRIDE


def gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> float:
    """CenterNet's radius: the largest centre shift that keeps IoU >= min_overlap.

    Solves the three quadratic cases of CornerNet's `gaussian_radius` (boxes
    in output-grid units) and returns the smallest root.
    """
    a1, b1 = 1.0, height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1**2 - 4 * a1 * c1)) / 2

    a2, b2 = 4.0, 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(b2**2 - 4 * a2 * c2)) / 2

    a3, b3 = 4 * min_overlap, -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(b3**2 - 4 * a3 * c3)) / 2
    return min(r1, r2, r3)


def encode(boxes: np.ndarray, image_h: int, image_w: int) -> dict[str, np.ndarray]:
    """Encodes `boxes` ([N, 4] as x, y, w, h in pixels) as dense target maps.

    Returns (grid = image / STRIDE):
    - heatmap [1, gh, gw]: Gaussian bumps, exactly 1.0 at each centre cell,
    - mask    [1, gh, gw]: 1.0 at centre cells,
    - log_size [2, gh, gw]: log(w), log(h) at centre cells,
    - offset  [2, gh, gw]: sub-cell centre position at centre cells.
    """
    gh, gw = image_h // STRIDE, image_w // STRIDE
    heatmap = np.zeros((1, gh, gw), np.float32)
    mask = np.zeros((1, gh, gw), np.float32)
    log_size = np.zeros((2, gh, gw), np.float32)
    offset = np.zeros((2, gh, gw), np.float32)
    ys, xs = np.mgrid[0:gh, 0:gw]

    for x, y, w, h in boxes:
        if w <= 0 or h <= 0:
            continue
        cx, cy = (x + w / 2) / STRIDE, (y + h / 2) / STRIDE
        col, row = min(int(cx), gw - 1), min(int(cy), gh - 1)
        radius = max(0.0, gaussian_radius(h / STRIDE, w / STRIDE))
        sigma = (2 * int(radius) + 1) / 6
        bump = np.exp(-((xs - col) ** 2 + (ys - row) ** 2) / (2 * sigma**2))
        heatmap[0] = np.maximum(heatmap[0], bump)
        heatmap[0, row, col] = 1.0
        mask[0, row, col] = 1.0
        log_size[:, row, col] = (math.log(w), math.log(h))
        offset[:, row, col] = (cx - col, cy - row)
    return {"heatmap": heatmap, "mask": mask, "log_size": log_size, "offset": offset}
