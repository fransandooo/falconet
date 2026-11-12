"""From network outputs to scored boxes."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .model import STRIDE


@torch.no_grad()
def decode(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor], top_k: int = 10
) -> list[list[tuple[float, float, float, float, float]]]:
    """Per image, up to `top_k` detections as (x, y, w, h, score), best first.

    Peaks are local maxima of the heatmap in a 3x3 window (CenterNet's
    max-pool "NMS"), so neighbouring cells of one object don't produce
    duplicate boxes.
    """
    heat_logits, log_size, offset_logits = outputs
    heat = torch.sigmoid(heat_logits)
    peaks = heat * (heat == F.max_pool2d(heat, 3, stride=1, padding=1))
    b, _, gh, gw = heat.shape
    scores, idx = peaks.view(b, -1).topk(min(top_k, gh * gw))
    rows, cols = idx // gw, idx % gw
    offset = torch.sigmoid(offset_logits)

    results = []
    for i in range(b):
        dets = []
        for s, r, c in zip(scores[i].tolist(), rows[i].tolist(), cols[i].tolist()):
            if s <= 0:
                continue
            w, h = torch.exp(log_size[i, :, r, c]).tolist()
            cx = (c + float(offset[i, 0, r, c])) * STRIDE
            cy = (r + float(offset[i, 1, r, c])) * STRIDE
            dets.append((cx - w / 2, cy - h / 2, w, h, s))
        results.append(dets)
    return results
