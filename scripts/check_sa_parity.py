# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.3.4"]
# ///
"""Check falconet's Rust state-accuracy (SA) metric against the official code.

Generates deterministic synthetic tracker outputs for every sequence of a split
(jittered boxes, misses, false alarms, "no target" answers), scores them with
the official `eval()` and with `antiuav410 sa` in Rust, then compares the
per-sequence scores.

The official functions are embedded verbatim below instead of importing the
toolkit, so no downloaded code is executed. At runtime the script checks (by
hashing text, not running it) that the toolkit's Evaluation_for_SA.py still
contains exactly this code.

    uv run scripts/check_sa_parity.py [--toolkit ~/datasets/antiuav410-toolkit] [--split test]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

TOLERANCE = 1e-5


# ---------------------------------------------------------------------------
# Verbatim copy of iou(), not_exist() and eval() from Evaluation_for_SA.py in
# github.com/HwangBo94/Anti-UAV410 (commit 8a8eb04). Do not edit.
# ---------------------------------------------------------------------------
OFFICIAL_SOURCE_SHA256 = "d50212b53860192031860503b5d2fda9b17e494bfcf53306ae3fc2da577f2f80"

def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x,y,w,h.
        bbox2 (numpy.array, list of floats): bounding box in format x,y,w,h.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, w1_1, h1_1) = bbox1
    (x0_2, y0_2, w1_2, h1_2) = bbox2
    x1_1 = x0_1 + w1_1
    x1_2 = x0_2 + w1_2
    y1_1 = y0_1 + h1_1
    y1_2 = y0_2 + h1_2
    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union


def not_exist(pred):

    if len(pred) == 1 or len(pred) == 0:
        return 1.0
    else:
        return 0.0



def eval(out_res, label_res):

    measure_per_frame = []

    for _pred, _gt, _exist in zip(out_res, label_res['gt_rect'], label_res['exist']):

        if not _exist:
            measure_per_frame.append(not_exist(_pred))
        else:

            if len(_gt)<4 or sum(_gt)==0:
                continue

            if len(_pred)==4:
                measure_per_frame.append(iou(_pred, _gt))
            else:
                measure_per_frame.append(0.0)

            # try:
            #     measure_per_frame.append(iou(_pred, _gt))
            # except:
            #     measure_per_frame.append(0)


        # measure_per_frame.append(not_exist(_pred) if not _exist else iou(_pred, _gt))


    return np.mean(measure_per_frame)
# --------------------------------------------------------------- end of copy


def check_official_source(toolkit: Path) -> None:
    """Fails if the toolkit's code differs from the embedded copy (text only)."""
    text = (toolkit / "Evaluation_for_SA.py").read_text()
    start = text.index("def iou(bbox1, bbox2):")
    end = text.index("def main():")
    digest = hashlib.sha256(text[start:end].rstrip("\n").encode() + b"\n").hexdigest()
    if digest != OFFICIAL_SOURCE_SHA256:
        raise SystemExit("the toolkit's Evaluation_for_SA.py changed; re-copy iou/not_exist/eval")


def read_boxes(path: Path) -> list[list[float]]:
    return [[float(v) for v in line.split(",")] for line in path.read_text().splitlines() if line.strip()]


def synthetic_predictions(gt: list[list[float]], rng: np.random.Generator) -> list[list[float]]:
    preds: list[list[float]] = []
    last = [300.0, 250.0, 30.0, 20.0]
    for box in gt:
        visible = any(v != 0 for v in box)
        r = rng.random()
        if visible:
            last = box
            if r < 0.08:
                preds.append([])  # missed: "no target"
            elif r < 0.11:
                preds.append([float(rng.uniform(0, 600)), float(rng.uniform(0, 480)), 20.0, 15.0])  # wrong place
            else:
                x, y, w, h = box
                preds.append(
                    [
                        x + float(rng.normal(0, 0.15 * w)),
                        y + float(rng.normal(0, 0.15 * h)),
                        w * float(np.exp(rng.normal(0, 0.1))),
                        h * float(np.exp(rng.normal(0, 0.1))),
                    ]
                )
        else:
            # Drone not visible: usually say so, sometimes keep reporting the last box.
            preds.append([] if r < 0.6 else list(last))
    return preds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    data_dir = Path(os.environ.get("FALCONET_DATA_DIR", Path.home() / "datasets"))
    parser.add_argument("--toolkit", type=Path, default=data_dir / "antiuav410-toolkit")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    check_official_source(args.toolkit)
    official_eval = eval  # the embedded official function (shadows the builtin, as in the toolkit)
    rng = np.random.default_rng(args.seed)
    official: dict[str, float] = {}

    with tempfile.TemporaryDirectory() as tmp:
        for path in sorted((args.toolkit / "annos" / args.split).glob("*.txt")):
            gt = read_boxes(path)
            preds = synthetic_predictions(gt, rng)
            label_res = {"exist": [int(any(v != 0 for v in b)) for b in gt], "gt_rect": gt}
            official[path.stem] = float(official_eval(preds, label_res))
            (Path(tmp) / path.name).write_text(json.dumps({"res": preds}))

        result = subprocess.run(
            [
                "cargo", "run", "--release", "--quiet", "-p", "falconet-data", "--bin", "antiuav410", "--",
                "sa", "--predictions", tmp, "--split", args.split,
                "--toolkit", str(args.toolkit), "--per-sequence",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

    rust = {r["sequence"]: r["sa"] for r in map(json.loads, result.stdout.splitlines())}
    if set(rust) != set(official):
        print(f"sequence sets differ: {sorted(set(rust) ^ set(official))[:5]}")
        return 1
    diffs = np.array([abs(rust[s] - official[s]) for s in official])
    msa_official = float(np.mean(list(official.values())))
    msa_rust = float(np.mean(list(rust.values())))
    print(f"{args.split}: {len(official)} sequences")
    print(f"  mSA official = {msa_official:.6f}, rust = {msa_rust:.6f}")
    print(f"  max |per-sequence difference| = {diffs.max():.2e} (tolerance {TOLERANCE:.0e})")
    ok = diffs.max() <= TOLERANCE
    print("  PARITY OK" if ok else "  PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
