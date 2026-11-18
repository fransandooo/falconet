"""Tests for the Anti-UAV410 loader that don't need the dataset."""

from __future__ import annotations

import numpy as np

from falconet_ref.antiuav import augment, frame_label


def test_labels_follow_the_toolkit_box_files():
    assert frame_label([]) is None
    assert frame_label([0, 0, 0, 0]) is None
    assert frame_label([596, 337, 44, 72]) == (596.0, 337.0, 44.0, 72.0)  # border drone, exist = 0
    assert frame_label([210, 300, 0, 9]) == "invalid"


def test_augment_keeps_the_box_on_the_drone():
    # A bright 6 x 4 drone on a dark sky: after any augmentation the box must
    # still cover the brightest pixels (or be dropped when it leaves the image).
    rng = np.random.default_rng(0)
    for _ in range(200):
        img = np.full((512, 640), 0.2, np.float32)
        x, y = int(rng.integers(0, 630)), int(rng.integers(0, 505))
        img[y : y + 4, x : x + 6] = 0.9
        out, boxes = augment(img, np.array([[x, y, 6, 4]], np.float32), rng)
        assert out.shape == (512, 640) and out.dtype == np.float32
        assert 0.0 <= out.min() and out.max() <= 1.0
        if len(boxes) == 0:
            continue
        bx, by, bw, bh = boxes[0]
        assert bw > 0 and bh > 0
        cy, cx = np.unravel_index(np.argmax(out), out.shape)
        assert bx - 1.5 <= cx <= bx + bw + 1.5 and by - 1.5 <= cy <= by + bh + 1.5
