"""Train the reference detector on Anti-UAV410 (thermal infrared, single drone).

    # MLflow server first (from the repo root): mise run mlflow
    uv run python -m falconet_ref.train_antiuav --epochs 30 --run-name baseline
    uv run python -m falconet_ref.train_antiuav --resume ../runs/reference/antiuav410/baseline/last.pt

What happens:
- Each epoch draws `--samples-per-epoch` random frames from the 200 train
  sequences (consecutive frames are near-duplicates, so a full pass over
  214k frames per epoch would mostly repeat itself).
- Validation runs on every `--val-every`-th frame of the 90 val sequences:
  COCO metrics (pycocotools: mAP, AP50, AP75, AP small/medium/large, AR@100)
  and single-drone metrics on the top detection (recall@0.5, mean IoU,
  "no drone" frames answered correctly, and `det_sa`: the official state
  accuracy computed per frame on the raw detector, without a tracker).
- Every epoch writes `last.pt` (resumable) and keeps the best model by AP50
  in `best.pt`; fixed validation frames are rendered with their heatmaps
  (MLflow image slider + `viz/epoch_NNN.{png,npz}` for videos).
- At the end, the best model is evaluated on the whole val split and its
  outputs are written in the official toolkit format (`final/val_sa/`) and as
  COCO files (`final/coco_*.json`) for the Rust evaluators, then exported to
  ONNX.

Everything is logged to MLflow (`MLFLOW_TRACKING_URI`, default
http://127.0.0.1:5000) under the experiment `reference-antiuav410`, and to the
run directory, so nothing is lost if the machine is switched off.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import torch
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, RandomSampler

from .antiuav import HEIGHT, WIDTH, AntiUAVFrames, Frame, data_root, load_frames
from .decode import decode
from .export_onnx import export, max_abs_difference
from .loss import detection_loss
from .model import STRIDE, CenterNetLite, count_parameters

EXPERIMENT = "reference-antiuav410"
SCORE_THRESHOLD = 0.3  # top detection below this = "no drone" (single-target metrics)
COCO_SMALL, COCO_MEDIUM = 32**2, 96**2


def collate(batch):
    images = torch.stack([b[0] for b in batch])
    targets = {k: torch.stack([b[1][k] for b in batch]) for k in batch[0][1]}
    return images, targets, [b[2] for b in batch], [b[3] for b in batch]


def iou(a, b) -> float:
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    iw = min(ax + aw, bx + bw) - max(ax, bx)
    ih = min(ay + ah, by + bh) - max(ay, by)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter)


# ---------------------------------------------------------------- evaluation


@torch.no_grad()
def predict(model, loader, device, top_k: int = 20, keep_heatmaps: set[int] | None = None):
    """Runs the model over `loader`.

    Returns per-frame detections (list of (x, y, w, h, score), best first),
    the mean validation loss, and the heatmaps (probabilities) of the frame
    indices in `keep_heatmaps`.
    """
    model.eval()
    detections: dict[int, list] = {}
    heatmaps: dict[int, np.ndarray] = {}
    losses = []
    for images, targets, _, idx in loader:
        images = images.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(images)
        out = tuple(t.float() for t in out)
        losses.append(detection_loss(out, targets)["total"].item())
        for i, dets in zip(idx, decode(out, top_k=top_k)):
            detections[i] = dets
        if keep_heatmaps:
            for j, i in enumerate(idx):
                if i in keep_heatmaps:
                    heatmaps[i] = torch.sigmoid(out[0][j, 0]).cpu().numpy()
    model.train()
    return detections, float(np.mean(losses)), heatmaps


def coco_files(frames: list[Frame], detections: dict[int, list]) -> tuple[dict, list]:
    """COCO ground truth + results for one class ("drone"), image id = frame index."""
    images, annotations, results = [], [], []
    for i, f in enumerate(frames):
        images.append({"id": i, "width": WIDTH, "height": HEIGHT, "file_name": f"{f.sequence}/{f.index + 1:06d}.jpg"})
        if f.box:
            x, y, w, h = f.box
            annotations.append(
                {"id": len(annotations) + 1, "image_id": i, "category_id": 1, "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0}
            )
        for x, y, w, h, s in detections.get(i, []):
            results.append({"image_id": i, "category_id": 1, "bbox": [x, y, w, h], "score": s})
    gt = {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "drone"}]}
    return gt, results


def coco_metrics(gt: dict, results: list) -> dict[str, float]:
    names = ["map", "ap50", "ap75", "ap_small", "ap_medium", "ap_large", "ar1", "ar10", "ar100", "ar_small", "ar_medium", "ar_large"]
    if not results:
        return {n: 0.0 for n in names}
    with contextlib.redirect_stdout(io.StringIO()):  # pycocotools prints its tables
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(results)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    # -1 means "no ground truth in this size range": not a score, leave it out.
    return {n: float(v) for n, v in zip(names, ev.stats) if v >= 0}


def single_target_metrics(frames: list[Frame], detections: dict[int, list]) -> dict[str, float]:
    """Metrics on the top detection, thresholded: what a single-drone tracker sees.

    `det_sa` is the Anti-UAV410 state accuracy per frame (IoU when a drone is
    visible, 1 / 0 for a correct / wrong answer on "no drone" frames),
    averaged per sequence, then over sequences, as the official evaluator does.
    """
    per_seq = defaultdict(list)
    hits = present = absent_ok = absent = 0
    ious = []
    by_size = defaultdict(lambda: [0, 0])  # size bin -> [hits, present]
    for i, f in enumerate(frames):
        dets = detections.get(i, [])
        best = dets[0] if dets and dets[0][4] >= SCORE_THRESHOLD else None
        if f.box is None:
            absent += 1
            absent_ok += best is None
            per_seq[f.sequence].append(1.0 if best is None else 0.0)
            continue
        present += 1
        v = iou(best, f.box) if best else 0.0
        ious.append(v)
        hits += v >= 0.5
        per_seq[f.sequence].append(v)
        area = f.box[2] * f.box[3]
        size = "small" if area < COCO_SMALL else "medium" if area < COCO_MEDIUM else "large"
        by_size[size][0] += v >= 0.5
        by_size[size][1] += 1
    metrics = {
        "recall_iou50": hits / max(present, 1),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "absent_correct": absent_ok / max(absent, 1),
        "det_sa": float(np.mean([np.mean(v) for v in per_seq.values()])),
    }
    for size, (h, n) in by_size.items():
        metrics[f"recall_iou50_{size}"] = h / max(n, 1)
    return metrics


# ------------------------------------------------------------- visualisation

GT_COLOUR, PRED_COLOUR = (27, 175, 122), (235, 104, 52)  # repo palette: green = truth, orange = model


def pick_viz_frames(frames: list[Frame], n: int = 8) -> list[int]:
    """Fixed frames to watch across epochs: drones of increasing size + one empty sky."""
    present = [i for i, f in enumerate(frames) if f.box]
    absent = [i for i, f in enumerate(frames) if f.box is None]
    by_area = sorted(present, key=lambda i: frames[i].box[2] * frames[i].box[3])
    picks = [by_area[int(q * (len(by_area) - 1))] for q in np.linspace(0.02, 0.98, n - 1 if absent else n)]
    if absent:
        picks.append(absent[len(absent) // 2])
    return picks


def heat_colour(h: np.ndarray) -> np.ndarray:
    """Probability map -> RGB (black -> orange -> white)."""
    h = np.clip(h, 0, 1)[..., None]
    return (np.concatenate([h * 2, h * 1.2, np.clip(h * 2 - 1, 0, 1)], -1).clip(0, 1) * 255).astype(np.uint8)


def render_panel(frames: list[Frame], picks: list[int], detections: dict, heatmaps: dict, epoch: int) -> Image.Image:
    """One row per frame: full frame | zoom around the drone | heatmap zoom."""
    tile = 200
    rows = []
    for i in picks:
        f = frames[i]
        img = Image.open(f.path).convert("RGB")
        best = detections.get(i, [])[:1]
        best = [d for d in best if d[4] >= SCORE_THRESHOLD]
        cx, cy = (f.box[0] + f.box[2] / 2, f.box[1] + f.box[3] / 2) if f.box else (WIDTH / 2, HEIGHT / 2)
        half = max(32.0, 1.5 * max(f.box[2], f.box[3])) if f.box else 64.0
        crop = (cx - half, cy - half, cx + half, cy + half)

        full = img.copy()
        d = ImageDraw.Draw(full)
        for b, colour in [(f.box, GT_COLOUR)] + [(p[:4], PRED_COLOUR) for p in best]:
            if b:
                d.rectangle([b[0], b[1], b[0] + b[2], b[1] + b[3]], outline=colour, width=3)
        full = full.resize((250, tile))

        zoom = img.crop(tuple(int(v) for v in crop)).resize((tile, tile), Image.NEAREST)
        d = ImageDraw.Draw(zoom)
        k = tile / (2 * half)
        for b, colour in [(f.box, GT_COLOUR)] + [(p[:4], PRED_COLOUR) for p in best]:
            if b:
                d.rectangle([(b[0] - crop[0]) * k, (b[1] - crop[1]) * k, (b[0] + b[2] - crop[0]) * k, (b[1] + b[3] - crop[1]) * k], outline=colour, width=2)

        heat = Image.fromarray(heat_colour(heatmaps[i])).resize((WIDTH, HEIGHT), Image.NEAREST)
        heat = heat.crop(tuple(int(v) for v in crop)).resize((tile, tile), Image.NEAREST)
        row = Image.new("RGB", (250 + 2 * tile + 8, tile), (16, 16, 16))
        row.paste(full, (0, 0))
        row.paste(zoom, (254, 0))
        row.paste(heat, (258 + tile, 0))
        label = f"{f.sequence} #{f.index + 1}" + (f"  score {best[0][4]:.2f}" if best else "")
        d = ImageDraw.Draw(row)
        d.rectangle(d.textbbox((4, 4), label), fill=(0, 0, 0))  # frames carry burnt-in sensor text
        d.text((4, 4), label, fill=(255, 255, 255))
        rows.append(row)
    panel = Image.new("RGB", (rows[0].width, len(rows) * (tile + 4) + 20), (16, 16, 16))
    ImageDraw.Draw(panel).text((4, 4), f"epoch {epoch}   green = ground truth, orange = model (score >= {SCORE_THRESHOLD})", fill=(255, 255, 255))
    for r, row in enumerate(rows):
        panel.paste(row, (0, 20 + r * (tile + 4)))
    return panel


# ------------------------------------------------------------------ training


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"]).returncode != 0
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def lr_at(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def final_evaluation(model, args, device, run_dir: Path) -> dict[str, float]:
    """Best model on the whole val split; files for the Rust evaluators; ONNX."""
    best = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    frames = load_frames(args.data, "val", every=args.final_every)
    loader = DataLoader(AntiUAVFrames(frames), 32, num_workers=args.workers, collate_fn=collate)
    detections, val_loss, _ = predict(model, loader, device)
    out = run_dir / "final"
    (out / "val_sa").mkdir(parents=True, exist_ok=True)

    gt, results = coco_files(frames, detections)
    (out / "coco_gt_val.json").write_text(json.dumps(gt))
    (out / "coco_dt_val.json").write_text(json.dumps(results))
    metrics = {**coco_metrics(gt, results), **single_target_metrics(frames, detections), "loss": val_loss}

    # Toolkit format: {"res": [[x, y, w, h] | [], ...]} per sequence, top detection only.
    per_seq = defaultdict(list)
    for i, f in enumerate(frames):
        dets = detections.get(i, [])
        per_seq[f.sequence].append([round(v, 2) for v in dets[0][:4]] if dets and dets[0][4] >= SCORE_THRESHOLD else [])
    for seq, res in per_seq.items():
        (out / "val_sa" / f"{seq}.txt").write_text(json.dumps({"res": res}))

    model.cpu()
    export(model, out / "centernet_lite.onnx")
    metrics["onnx_max_abs_diff"] = max_abs_difference(model, out / "centernet_lite.onnx")
    model.to(device)
    (out / "metrics_val_full.json").write_text(json.dumps({"epoch": best["epoch"], **metrics}, indent=2))
    return {f"final_val/{k}": v for k, v in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=data_root())
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--samples-per-epoch", type=int, default=40_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--val-every", type=int, default=20, help="validate on every N-th val frame")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50, help="steps between train-loss points")
    parser.add_argument("--deadline", type=str, help="ISO time; stop cleanly after the epoch that passes it")
    parser.add_argument("--max-steps", type=int, help="smoke tests: stop each epoch after N steps")
    parser.add_argument("--run-name", type=str, default=datetime.now().strftime("%Y%m%d-%H%M"))
    parser.add_argument("--out", type=Path, default=Path("../runs/reference/antiuav410"))
    parser.add_argument("--resume", type=Path, help="last.pt of an interrupted run")
    parser.add_argument("--final-every", type=int, default=1, help="final evaluation on every N-th val frame (1 = all)")
    parser.add_argument("--skip-final", action="store_true", help="smoke tests: no full-val evaluation / export")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.resume, map_location=device, weights_only=False) if args.resume else None
    if ckpt:
        saved = ckpt["args"]
        for k in ("epochs", "samples_per_epoch", "batch_size", "lr", "weight_decay", "warmup_steps", "val_every", "seed", "run_name", "out"):
            setattr(args, k, type(getattr(args, k))(saved[k]) if k != "out" else Path(saved[k]))
    run_dir = args.out / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "viz").mkdir(exist_ok=True)

    train_frames = load_frames(args.data, "train")
    val_frames = load_frames(args.data, "val", every=args.val_every)
    train_set = AntiUAVFrames(train_frames, train=True, seed=args.seed)
    val_set = AntiUAVFrames(val_frames)
    sampler_rng = torch.Generator()
    sampler = RandomSampler(train_set, num_samples=args.samples_per_epoch, generator=sampler_rng)
    train_loader = DataLoader(
        train_set, args.batch_size, sampler=sampler, num_workers=args.workers, collate_fn=collate,
        persistent_workers=True, pin_memory=True, drop_last=True,
    )  # fmt: skip
    val_loader = DataLoader(val_set, 32, num_workers=args.workers, collate_fn=collate, persistent_workers=True)
    viz_frames = pick_viz_frames(val_frames)

    model = CenterNetLite().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = args.samples_per_epoch // args.batch_size
    total_steps = args.epochs * steps_per_epoch
    start_epoch, step, best_ap50 = 1, 0, -1.0
    if ckpt:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, step, best_ap50 = ckpt["epoch"] + 1, ckpt["step"], ckpt["best_ap50"]

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(EXPERIMENT)
    run = mlflow.start_run(run_id=ckpt["mlflow_run_id"] if ckpt else None, run_name=None if ckpt else args.run_name)
    if ckpt:
        mlflow.set_tag("resumed_from_epoch", str(ckpt["epoch"]))
    else:
        mlflow.log_params({
            **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k not in ("resume",)},
            "params": count_parameters(model), "stride": STRIDE, "input": f"{WIDTH}x{HEIGHT} grey",
            "train_frames": len(train_frames), "val_frames": len(val_frames), "steps_per_epoch": steps_per_epoch,
            "score_threshold": SCORE_THRESHOLD, "precision": "bf16 autocast" if device.type == "cuda" else "fp32",
        })  # fmt: skip
        mlflow.set_tags({
            "git_commit": git_commit(), "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
            "torch": torch.__version__, "dataset": "Anti-UAV410", "model": "CenterNetLite", "run_dir": str(run_dir.resolve()),
        })  # fmt: skip
        (run_dir / "config.json").write_text(json.dumps({**vars(args), "data": str(args.data), "out": str(args.out), "resume": None}, indent=2, default=str))
    print(f"MLflow run {run.info.run_id} | {len(train_frames)} train / {len(val_frames)} val frames | {steps_per_epoch} steps per epoch", flush=True)

    deadline = datetime.fromisoformat(args.deadline) if args.deadline else None
    history_path = run_dir / "history.csv"
    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        sampler_rng.manual_seed(args.seed * 1000 + epoch)  # same frames per epoch after a resume
        t0, seen, running = time.time(), 0, defaultdict(list)
        for b, (images, targets, _, _) in enumerate(train_loader):
            if args.max_steps and b >= args.max_steps:
                break
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step, total_steps, args.warmup_steps, args.lr)
            images = images.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(images)
            loss = detection_loss(tuple(t.float() for t in out), targets)
            if not torch.isfinite(loss["total"]):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} step {step}: {loss}")
            optimizer.zero_grad(set_to_none=True)
            loss["total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            step += 1
            seen += images.shape[0]
            for k, v in loss.items():
                running[k].append(v.item())
            running["grad_norm"].append(grad_norm.item())
            if step % args.log_every == 0:
                mlflow.log_metrics(
                    {**{f"train/{k}": float(np.mean(v[-args.log_every :])) for k, v in running.items()},
                     "train/lr": optimizer.param_groups[0]["lr"], "train/images_per_s": seen / (time.time() - t0)},
                    step=step,
                )  # fmt: skip
        train_seconds = time.time() - t0

        detections, val_loss, heatmaps = predict(model, val_loader, device, keep_heatmaps=set(viz_frames))
        gt, results = coco_files(val_frames, detections)
        val = {"loss": val_loss, **coco_metrics(gt, results), **single_target_metrics(val_frames, detections)}
        row = {
            "epoch": epoch, "step": step, "train_loss": float(np.mean(running["total"])),
            **{f"val_{k}": v for k, v in val.items()}, "train_seconds": train_seconds, "val_seconds": time.time() - t0 - train_seconds,
        }  # fmt: skip
        mlflow.log_metrics({f"val/{k}": v for k, v in val.items()} | {
            "epoch/train_loss": row["train_loss"], "epoch/train_seconds": train_seconds, "epoch/val_seconds": row["val_seconds"],
            "epoch/gpu_max_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0,
        }, step=epoch)  # fmt: skip

        panel = render_panel(val_frames, viz_frames, detections, heatmaps, epoch)
        panel.save(run_dir / "viz" / f"epoch_{epoch:03d}.png")
        np.savez_compressed(
            run_dir / "viz" / f"epoch_{epoch:03d}.npz",
            frames=np.array([str(val_frames[i].path) for i in viz_frames]),
            boxes=np.array([val_frames[i].box or (0, 0, 0, 0) for i in viz_frames], np.float32),
            heatmaps=np.stack([heatmaps[i] for i in viz_frames]).astype(np.float16),
            detections=np.array([(detections[i][:1] or [(0, 0, 0, 0, 0)])[0] for i in viz_frames], np.float32),
        )  # fmt: skip
        mlflow.log_image(panel, key="val_predictions", step=epoch)

        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "step": step,
            "best_ap50": max(best_ap50, val.get("ap50", 0.0)), "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "mlflow_run_id": run.info.run_id, "val": val,
        }  # fmt: skip
        torch.save(state, run_dir / "last.pt.tmp")
        os.replace(run_dir / "last.pt.tmp", run_dir / "last.pt")  # never leave a half-written checkpoint
        if val.get("ap50", 0.0) > best_ap50:
            best_ap50 = val["ap50"]
            torch.save(state, run_dir / "best.pt")
            mlflow.log_metric("val/best_ap50", best_ap50, step=epoch)
            mlflow.set_tag("best_epoch", str(epoch))

        new_file = not history_path.exists()
        with history_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row), extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        print(
            f"epoch {epoch:>3}/{args.epochs} | loss {row['train_loss']:.3f} / val {val_loss:.3f} | "
            f"AP50 {val.get('ap50', 0):.3f} mAP {val.get('map', 0):.3f} AP_s {val.get('ap_small', 0):.3f} | "
            f"det_sa {val['det_sa']:.3f} | {seen / train_seconds:.0f} img/s | {train_seconds + row['val_seconds']:.0f} s",
            flush=True,
        )
        if deadline and datetime.now() >= deadline and epoch < args.epochs:
            mlflow.set_tag("stopped_early", f"deadline {args.deadline} after epoch {epoch}")
            print(f"deadline reached after epoch {epoch}; stopping", flush=True)
            break

    mlflow.log_artifact(str(history_path))
    if not args.skip_final:
        final = final_evaluation(model, args, device, run_dir)
        mlflow.log_metrics(final)
        mlflow.log_artifacts(str(run_dir / "final"), artifact_path="final")
        mlflow.log_artifact(str(run_dir / "best.pt"), artifact_path="weights")
        print(json.dumps(final, indent=2), flush=True)
    mlflow.end_run()


if __name__ == "__main__":
    sys.exit(main())
