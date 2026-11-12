"""Train the reference detector.

    uv run python -m falconet_ref.train --dataset synthetic --epochs 10
    uv run python -m falconet_ref.train --help

Writes `config.json`, `history.csv` and `model.pt` to `--out`. The Anti-UAV410
loader is added once the dataset is available (the frame layout inside the
archive still has to be verified).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .decode import decode
from .loss import detection_loss
from .model import CenterNetLite, count_parameters
from .synthetic import SyntheticScenes


def collate(batch):
    images = torch.stack([b[0] for b in batch])
    targets = {k: torch.stack([b[1][k] for b in batch]) for k in batch[0][1]}
    boxes = [b[2] for b in batch]
    return images, targets, boxes


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = min(ax + aw, bx + bw) - max(ax, bx)
    ih = min(ay + ah, by + bh) - max(ay, by)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter)


@torch.no_grad()
def validate(model, loader, device, score_threshold: float = 0.3) -> dict[str, float]:
    """Single-drone metrics on the top detection of each frame."""
    model.eval()
    hits = misses = present = absent_ok = absent = 0
    ious = []
    for images, _, boxes in loader:
        for dets, gt in zip(decode(model(images.to(device)), top_k=1), boxes):
            best = dets[0] if dets and dets[0][4] >= score_threshold else None
            if len(gt) == 0:
                absent += 1
                absent_ok += best is None
                continue
            present += 1
            v = iou(best[:4], gt[0]) if best else 0.0
            ious.append(v)
            hits += v >= 0.5
            misses += best is None
    model.train()
    return {
        "recall_iou50": hits / max(present, 1),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "miss_rate": misses / max(present, 1),
        "absent_correct": absent_ok / max(absent, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["synthetic"], default="synthetic")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("../runs/reference/synthetic"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = SyntheticScenes(args.train_size, seed=args.seed)
    val_set = SyntheticScenes(args.val_size, seed=args.seed + 1)
    train_loader = DataLoader(
        train_set, args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate, persistent_workers=True
    )
    val_loader = DataLoader(val_set, 64, num_workers=args.workers, collate_fn=collate)

    model = CenterNetLite().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(train_loader)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))

    args.out.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "out": str(args.out), "params": count_parameters(model), "device": str(device)}
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"CenterNetLite: {config['params']:,} parameters on {device}")

    history = []
    for epoch in range(1, args.epochs + 1):
        start, losses = time.time(), []
        for images, targets, _ in train_loader:
            images = images.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
            loss = detection_loss(model(images), targets)
            optimizer.zero_grad(set_to_none=True)
            loss["total"].backward()
            optimizer.step()
            schedule.step()
            losses.append(loss["total"].item())
        metrics = validate(model, val_loader, device)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **metrics, "seconds": time.time() - start}
        history.append(row)
        print(
            f"epoch {epoch:>3} | loss {row['loss']:.4f} | R@0.5 {metrics['recall_iou50']:.3f} "
            f"| IoU {metrics['mean_iou']:.3f} | no-drone frames right {metrics['absent_correct']:.3f} "
            f"| {row['seconds']:.1f} s"
        )

    with (args.out / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    torch.save(model.state_dict(), args.out / "model.pt")
    print(f"saved {args.out / 'model.pt'}")


if __name__ == "__main__":
    main()
