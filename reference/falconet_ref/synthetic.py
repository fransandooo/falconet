"""Synthetic thermal-like scenes for smoke-testing the training code.

A stand-in for Anti-UAV410 until real data is available. Scenes have a smooth
sky with a temperature gradient, cloud-like blobs (distractors), sensor noise,
and usually one drone: a small blob that can be hotter or colder than the sky.
Every scene is generated from its index, so nothing is stored.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .targets import encode


def scene(index: int, seed: int, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (image [H, W] in [0, 1], boxes [N, 4] as x, y, w, h)."""
    rng = np.random.default_rng([seed, index])
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    img = rng.uniform(0.2, 0.6) + rng.uniform(-0.2, 0.2) * (xs / width) + rng.uniform(-0.2, 0.2) * (ys / height)

    for _ in range(rng.integers(0, 4)):  # clouds: wide, soft, low contrast
        cx, cy, s = rng.uniform(0, width), rng.uniform(0, height), rng.uniform(8, 40)
        img += rng.uniform(-0.15, 0.25) * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * s**2))

    boxes = np.zeros((0, 4), np.float32)
    if rng.random() > 0.1:  # 10% of frames have no drone, like occlusions
        side = float(np.exp(rng.uniform(np.log(4), np.log(60))))
        w, h = side * rng.uniform(1.0, 1.6), side  # drones are wider than tall
        x, y = rng.uniform(0, width - w), rng.uniform(0, height - h)
        sign = 1.0 if rng.random() < 0.8 else -1.0  # usually hotter than the sky
        contrast = sign * rng.uniform(0.15, 0.5)
        inside = (xs >= x) & (xs < x + w) & (ys >= y) & (ys < y + h)
        img = img + contrast * inside
        boxes = np.array([[x, y, w, h]], np.float32)

    img = img + rng.normal(0, 0.03, img.shape)
    return np.clip(img, 0, 1).astype(np.float32), boxes


class SyntheticScenes(Dataset):
    def __init__(self, length: int, seed: int, height: int = 256, width: int = 256) -> None:
        self.length, self.seed, self.height, self.width = length, seed, height, width

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        img, boxes = scene(index, self.seed, self.height, self.width)
        targets = {k: torch.from_numpy(v) for k, v in encode(boxes, self.height, self.width).items()}
        return torch.from_numpy(img)[None], targets, boxes
