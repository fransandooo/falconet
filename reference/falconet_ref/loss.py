"""CenterNet loss: penalty-reduced focal loss + L1 regressions at centres.

The same formulation as the Rust toy detector (playground/toy-detector/src/loss.rs),
with size regressed in log space so tiny and large drones weigh the same.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

ALPHA, BETA = 2.0, 4.0


def detection_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor], targets: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    heat_logits, log_size, offset_logits = outputs
    gt, mask = targets["heatmap"], targets["mask"]
    num_pos = mask.sum().clamp(min=1.0)  # frames without a drone have no positives

    p = torch.sigmoid(heat_logits)
    log_p, log_not_p = F.logsigmoid(heat_logits), F.logsigmoid(-heat_logits)
    pos = (1 - p) ** ALPHA * log_p * mask
    neg = (1 - gt) ** BETA * p**ALPHA * log_not_p * (1 - mask)
    heatmap = -(pos.sum() + neg.sum()) / num_pos

    size = ((log_size - targets["log_size"]).abs() * mask).sum() / num_pos
    offset = ((torch.sigmoid(offset_logits) - targets["offset"]).abs() * mask).sum() / num_pos
    total = heatmap + size + offset
    return {"total": total, "heatmap": heatmap, "size": size, "offset": offset}
