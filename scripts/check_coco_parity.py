# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.3.4", "pycocotools==2.0.11"]
# ///
"""Check falconet's Rust detection metrics against pycocotools' COCOeval.

Builds a synthetic single-class dataset with objects of every COCO size
(small/medium/large), detections with jitter, misses, duplicates and false
alarms, then compares the COCO summary from pycocotools with `coco_eval`.

    uv run scripts/check_coco_parity.py [--images 400] [--seed 0]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

TOLERANCE = 1e-4


def synthetic(n_images: int, rng: np.random.Generator) -> tuple[dict, list]:
    images, annotations, results = [], [], []
    for image_id in range(1, n_images + 1):
        images.append({"id": image_id, "width": 640, "height": 512})
        for _ in range(rng.integers(0, 4)):
            side = float(np.exp(rng.uniform(np.log(4), np.log(160))))  # 4..160 px, log-uniform
            w, h = side * rng.uniform(0.7, 1.4), side * rng.uniform(0.7, 1.4)
            x, y = rng.uniform(0, 640 - w), rng.uniform(0, 512 - h)
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            if rng.random() < 0.85:  # detected, with box jitter
                jx, jy = rng.normal(0, 0.1 * w), rng.normal(0, 0.1 * h)
                sw, sh = np.exp(rng.normal(0, 0.1, 2))
                results.append(
                    {
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [x + jx, y + jy, w * sw, h * sh],
                        "score": float(rng.uniform(0.3, 1.0)),
                    }
                )
                if rng.random() < 0.1:  # duplicate detection
                    results.append({**results[-1], "score": float(rng.uniform(0.05, 0.6))})
        for _ in range(rng.poisson(0.7)):  # false alarms
            side = float(np.exp(rng.uniform(np.log(4), np.log(100))))
            results.append(
                {
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [rng.uniform(0, 600), rng.uniform(0, 480), side, side],
                    "score": float(rng.uniform(0.05, 0.9)),
                }
            )
    gt = {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "drone"}]}
    return gt, results


def pycocotools_summary(gt_path: Path, dt_path: Path) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(str(dt_path))
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    absent = lambda v: None if v < 0 else float(v)  # noqa: E731
    return {
        "map": float(s[0]),
        "ap50": float(s[1]),
        "ap75": float(s[2]),
        "ap_small": absent(s[3]),
        "ap_medium": absent(s[4]),
        "ap_large": absent(s[5]),
        "recall": float(s[8]),  # AR@100
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    gt, results = synthetic(args.images, np.random.default_rng(args.seed))
    with tempfile.TemporaryDirectory() as tmp:
        gt_path, dt_path = Path(tmp) / "gt.json", Path(tmp) / "dt.json"
        gt_path.write_text(json.dumps(gt))
        dt_path.write_text(json.dumps(results))
        reference = pycocotools_summary(gt_path, dt_path)
        out = subprocess.run(
            [
                "cargo", "run", "--release", "--quiet", "-p", "falconet-data", "--bin", "coco_eval", "--",
                "--gt", str(gt_path), "--dt", str(dt_path),
            ],
            capture_output=True, text=True, check=True,
        )
    ours = json.loads(out.stdout)

    print(f"{args.images} images, {len(gt['annotations'])} objects, {len(results)} detections (seed {args.seed})")
    ok = True
    for key, ref in reference.items():
        val = ours[key]
        if ref is None or val is None:
            same = ref is None and val is None
            print(f"  {key:<10} pycocotools {ref!s:>8}  rust {val!s:>8}  {'OK' if same else 'MISMATCH'}")
            ok &= same
            continue
        diff = abs(ref - val)
        print(f"  {key:<10} pycocotools {ref:.6f}  rust {val:.6f}  diff {diff:.1e}")
        ok &= diff <= TOLERANCE
    print("  PARITY OK" if ok else "  PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
