"""Anti-UAV410 frames as a PyTorch dataset.

Layout after `scripts/download_antiuav410.sh` (verified on the full archive):

    $FALCONET_DATA_DIR/antiuav410/{train,val,test}/<sequence>/
        000001.jpg ... NNNNNN.jpg   640 x 512, grey stored as RGB (channels differ <= 5/255)
        IR_label.json               {"exist": [0|1, ...], "gt_rect": [[x, y, w, h] | [], ...]}

Two labels per frame, because `exist` and the box disagree on 37 frames
(drones cut by the image border: `exist = 0`, yet `gt_rect` has a box):
- `box`, for training and COCO metrics: a box with positive width and height
  in `gt_rect`, whatever `exist` says. This matches the toolkit's box files
  (`annos/<split>/<seq>.txt`) on all 438,397 frames;
- `exist`, for state accuracy: the official evaluator scores frames with
  `exist = 0` as "no drone" (an empty answer gets 1) even when they have a box.
Degenerate boxes (w or h = 0 with a non-zero position, 5 frames) are dropped:
their label is unknown, not "no drone".
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .targets import encode

HEIGHT, WIDTH = 512, 640


def data_root() -> Path:
    return Path(os.environ.get("FALCONET_DATA_DIR", Path.home() / "datasets")) / "antiuav410"


@dataclass(frozen=True)
class Frame:
    path: Path
    sequence: str
    index: int  # 0-based frame number inside the sequence
    box: tuple[float, float, float, float] | None  # x, y, w, h in pixels; None = no drone
    exist: bool = True  # official `exist` flag (what state accuracy scores)


def frame_label(rect: list[float]) -> tuple[float, float, float, float] | None | str:
    """`gt_rect` entry -> box, None (no drone) or "invalid" (degenerate box)."""
    if len(rect) != 4 or not any(rect):
        return None
    x, y, w, h = map(float, rect)
    if w > 0 and h > 0:
        return (x, y, w, h)
    return "invalid"


def load_frames(root: Path, split: str, every: int = 1) -> list[Frame]:
    """Every `every`-th labelled frame of `split`, sequences in name order."""
    frames = []
    for seq in sorted(p for p in (root / split).iterdir() if p.is_dir()):
        labels = json.loads((seq / "IR_label.json").read_text())
        rects, exist = labels["gt_rect"], labels["exist"]
        for i in range(0, len(rects), every):
            label = frame_label(rects[i])
            if label == "invalid":
                continue
            frames.append(Frame(seq / f"{i + 1:06d}.jpg", seq.name, i, label, bool(exist[i])))
    return frames


def read_grey(path: Path) -> np.ndarray:
    """[H, W] float32 in [0, 1]."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


def augment(img: np.ndarray, boxes: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Flip, scale jitter (zoom in/out around a random point) and gain/bias.

    Scale jitter changes apparent drone size, which matters because sizes in
    Anti-UAV410 span 2 to 300+ px. The output keeps the input size.
    """
    h, w = img.shape
    if rng.random() < 0.5:
        img = img[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0] = w - boxes[:, 0] - boxes[:, 2]

    if rng.random() < 0.5:
        s = float(np.exp(rng.uniform(np.log(0.75), np.log(1.33))))
        sh, sw = round(h * s), round(w * s)
        scaled = np.asarray(Image.fromarray(img).resize((sw, sh), Image.BILINEAR))
        # Place the scaled image on a canvas of the original size: crop when
        # zooming in, pad with the mean grey when zooming out.
        dx = int(rng.integers(min(0, w - sw), max(0, w - sw) + 1))
        dy = int(rng.integers(min(0, h - sh), max(0, h - sh) + 1))
        canvas = np.full((h, w), float(img.mean()), np.float32)
        src_x0, src_y0 = max(0, -dx), max(0, -dy)
        dst_x0, dst_y0 = max(0, dx), max(0, dy)
        cw, ch = min(sw - src_x0, w - dst_x0), min(sh - src_y0, h - dst_y0)
        canvas[dst_y0 : dst_y0 + ch, dst_x0 : dst_x0 + cw] = scaled[src_y0 : src_y0 + ch, src_x0 : src_x0 + cw]
        img = canvas

        kept = []
        for x, y, bw, bh in boxes * s:
            x, y = x + dx, y + dy
            cx, cy = x + bw / 2, y + bh / 2
            if not (0 <= cx < w and 0 <= cy < h):
                continue  # centre left the image: no drone to find
            x0, y0 = max(x, 0.0), max(y, 0.0)
            x1, y1 = min(x + bw, float(w)), min(y + bh, float(h))
            kept.append((x0, y0, x1 - x0, y1 - y0))
        boxes = np.array(kept, np.float32).reshape(-1, 4)

    gain, bias = rng.uniform(0.75, 1.25), rng.uniform(-0.1, 0.1)
    img = np.clip((img - 0.5) * gain + 0.5 + bias, 0.0, 1.0)
    return np.ascontiguousarray(img, dtype=np.float32), boxes


class AntiUAVFrames(Dataset):
    """Items are (image [1, H, W], target maps, boxes [N, 4], frame index)."""

    def __init__(self, frames: list[Frame], train: bool = False, seed: int = 0) -> None:
        self.frames, self.train, self.seed = frames, train, seed

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int):
        f = self.frames[i]
        img = read_grey(f.path)
        if img.shape != (HEIGHT, WIDTH):
            raise ValueError(f"{f.path}: expected {WIDTH}x{HEIGHT}, got {img.shape[1]}x{img.shape[0]}")
        boxes = np.array([f.box] if f.box else [], np.float32).reshape(-1, 4)
        if self.train:
            # Fresh randomness per draw (worker seeds differ), reproducible per worker.
            rng = np.random.default_rng([self.seed, i, torch.randint(1 << 30, ()).item()])
            img, boxes = augment(img, boxes, rng)
        targets = {k: torch.from_numpy(v) for k, v in encode(boxes, HEIGHT, WIDTH).items()}
        return torch.from_numpy(img)[None], targets, boxes, i
