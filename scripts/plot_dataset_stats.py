# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.7", "numpy==2.3.4"]
# ///
"""Plots for the Anti-UAV410 dataset statistics.

Reads the CSV/JSON written by
    cargo run --release -p falconet-data --bin antiuav410 -- stats
and renders PNG figures (light theme) for the docs and the video.

    uv run scripts/plot_dataset_stats.py [--stats runs/data-stats/antiuav410] [--out media/phase-1]

The numbers come from Rust; this script only draws.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

# Validated categorical slots 1-3 (all-pairs CVD-safe) and the sequential blue ramp.
SPLITS = ["train", "val", "test"]
SPLIT_COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BLUES = LinearSegmentedColormap.from_list(
    "blues",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
SIZE_EDGES = [10, 30, 50]
SIZE_NAMES = ["tiny", "small", "medium", "normal"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "legend.frameon": False,
        "legend.labelcolor": TEXT,
        "font.size": 10,
        "lines.linewidth": 2,
    }
)


def load_boxes(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Per split: arrays of x, y, w, h for visible frames, plus sequence ids."""
    rows: dict[str, list[tuple]] = {s: [] for s in SPLITS}
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["label"] == "present":
                rows[r["split"]].append(
                    (r["sequence"], int(r["frame"]), float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"]))
                )
            else:
                rows[r["split"]].append((r["sequence"], int(r["frame"]), np.nan, np.nan, np.nan, np.nan))
    out = {}
    for split, items in rows.items():
        seq, frame, x, y, w, h = zip(*items)
        out[split] = {
            "seq": np.array(seq),
            "frame": np.array(frame),
            "x": np.array(x),
            "y": np.array(y),
            "w": np.array(w),
            "h": np.array(h),
        }
    return out


def visible(d: dict[str, np.ndarray]) -> np.ndarray:
    return ~np.isnan(d["w"])


def save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.savefig(out / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / name}")


def plot_size(boxes, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.logspace(np.log10(2), np.log10(400), 60)
    for split in SPLITS:
        d = boxes[split]
        diag = np.hypot(d["w"], d["h"])[visible(d)]
        hist, edges = np.histogram(diag, bins=bins)
        share = 100 * hist / hist.sum()
        ax.step(edges[:-1], share, where="post", color=SPLIT_COLORS[split], label=split)
    ymax = ax.get_ylim()[1]
    for edge in SIZE_EDGES:
        ax.axvline(edge, color=TEXT_2, linewidth=1, linestyle=(0, (3, 3)))
    centers = [5, 17, 39, 110]
    for name, cx in zip(SIZE_NAMES, centers):
        ax.text(cx, ymax * 0.97, name, ha="center", va="top", color=TEXT_2, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks([3, 5, 10, 20, 30, 50, 100, 200, 400], ["3", "5", "10", "20", "30", "50", "100", "200", "400"])
    ax.set_xlabel("box diagonal (px, log scale)")
    ax.set_ylabel("share of visible frames (%)")
    ax.set_title("How big is the drone? Box diagonal per split")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.88))
    save(fig, out, "antiuav410-box-size.png")


def plot_width_height(boxes, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    w = np.concatenate([boxes[s]["w"][visible(boxes[s])] for s in SPLITS])
    h = np.concatenate([boxes[s]["h"][visible(boxes[s])] for s in SPLITS])
    bins = np.logspace(np.log10(1.5), np.log10(400), 70)
    _, _, _, image = ax.hist2d(w, h, bins=[bins, bins], cmap=BLUES, norm=LogNorm(), cmin=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [2, 5, 10, 20, 50, 100, 200]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.plot([1.5, 400], [1.5, 400], color=TEXT_2, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(120, 260, "w = h", color=TEXT_2, fontsize=9, ha="right")
    ax.set_xlabel("box width (px)")
    ax.set_ylabel("box height (px)")
    ax.set_title("Box width vs height (all splits)")
    ax.grid(False)
    bar = fig.colorbar(image, ax=ax, pad=0.02)
    bar.set_label("frames", color=TEXT_2)
    bar.outline.set_visible(False)
    save(fig, out, "antiuav410-box-width-height.png")


def plot_centres(boxes, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5))
    cx = np.concatenate([(boxes[s]["x"] + boxes[s]["w"] / 2)[visible(boxes[s])] for s in SPLITS])
    cy = np.concatenate([(boxes[s]["y"] + boxes[s]["h"] / 2)[visible(boxes[s])] for s in SPLITS])
    width = max(640, int(np.ceil(cx.max() / 64) * 64))
    height = max(512, int(np.ceil(cy.max() / 64) * 64))
    _, _, _, image = ax.hist2d(cx, cy, bins=[64, 52], range=[[0, width], [0, height]], cmap=BLUES, cmin=1)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # image coordinates: y grows downwards
    ax.set_aspect("equal")
    ax.set_xlabel("box centre x (px)")
    ax.set_ylabel("box centre y (px)")
    ax.set_title("Where in the frame is the drone?")
    ax.grid(False)
    bar = fig.colorbar(image, ax=ax, pad=0.02, shrink=0.8)
    bar.set_label("frames", color=TEXT_2)
    bar.outline.set_visible(False)
    save(fig, out, "antiuav410-box-centres.png")


def consecutive_pairs(d: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Relative centre motion and IoU between consecutive visible frames."""
    same_seq = d["seq"][1:] == d["seq"][:-1]
    ok = same_seq & visible(d)[1:] & visible(d)[:-1]
    x0, y0, w0, h0 = (d[k][:-1][ok] for k in "xywh")
    x1, y1, w1, h1 = (d[k][1:][ok] for k in "xywh")
    shift = np.hypot((x1 + w1 / 2) - (x0 + w0 / 2), (y1 + h1 / 2) - (y0 + h0 / 2))
    rel = shift / np.hypot(w1, h1)
    iw = np.clip(np.minimum(x0 + w0, x1 + w1) - np.maximum(x0, x1), 0, None)
    ih = np.clip(np.minimum(y0 + h0, y1 + h1) - np.maximum(y0, y1), 0, None)
    inter = iw * ih
    iou = inter / (w0 * h0 + w1 * h1 - inter)
    return rel, iou


def plot_motion(boxes, out: Path) -> None:
    rel, iou = zip(*(consecutive_pairs(boxes[s]) for s in SPLITS))
    rel, iou = np.concatenate(rel), np.concatenate(iou)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))

    bins = np.linspace(0, 1.5, 61)
    a.hist(np.clip(rel, 0, 1.5), bins=bins, color=SPLIT_COLORS["train"], edgecolor=SURFACE, linewidth=0.5)
    clipped = 100 * np.mean(rel >= 1.5)
    a.annotate(
        f"≥ 1.5 ({clipped:.1f}%)",
        xy=(1.49, a.get_ylim()[1] * 0.03),
        xytext=(1.2, a.get_ylim()[1] * 0.25),
        color=TEXT_2,
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": TEXT_2, "linewidth": 0.8},
    )
    a.set_xlabel("centre shift between consecutive frames / box diagonal")
    a.set_ylabel("frame pairs")
    a.set_title("Frame-to-frame motion, relative to box size")

    b.hist(iou, bins=np.linspace(0, 1, 51), color=SPLIT_COLORS["train"], edgecolor=SURFACE, linewidth=0.5)
    b.axvline(0.3, color=TEXT_2, linewidth=1, linestyle=(0, (3, 3)))
    below = 100 * np.mean(iou < 0.3)
    b.text(0.31, b.get_ylim()[1] * 0.9, f"IoU < 0.3: {below:.1f}% of pairs", color=TEXT_2, fontsize=9)
    b.set_xlabel("IoU between consecutive ground-truth boxes")
    b.set_ylabel("frame pairs")
    b.set_title("Overlap between consecutive frames")
    fig.tight_layout()
    save(fig, out, "antiuav410-motion.png")


def plot_attributes(stats, out: Path) -> None:
    per_split = {s["split"]: s for s in stats if s["split"] is not None}
    names = [name for name, _ in per_split["train"]["attribute_counts"]]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(names))
    height = 0.26
    for i, split in enumerate(SPLITS):
        s = per_split[split]
        share = [100 * count / s["sequences"] for _, count in s["attribute_counts"]]
        ax.barh(
            y + (i - 1) * height,
            share,
            height=height - 0.04,
            color=SPLIT_COLORS[split],
            label=f"{split} ({s['sequences']} videos)",
        )
    labels = {
        "TC": "thermal crossover",
        "OV": "out of view",
        "SV": "scale variation",
        "FM": "fast motion",
        "OC": "occlusion",
        "DBC": "dynamic background clutter",
        "TS": "tiny size",
        "SS": "small size",
        "MS": "medium size",
        "NS": "normal size",
    }
    ax.set_yticks(y, [f"{labels[n]} ({n})" for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("share of videos with the attribute (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.06), ncol=3, borderaxespad=0)
    ax.set_title("Challenge attributes per split", pad=34)
    save(fig, out, "antiuav410-attributes.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, default=Path("runs/data-stats/antiuav410"))
    parser.add_argument("--out", type=Path, default=Path("media/phase-1"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    boxes = load_boxes(args.stats / "boxes.csv")
    stats = json.loads((args.stats / "stats.json").read_text())
    plot_size(boxes, args.out)
    plot_width_height(boxes, args.out)
    plot_centres(boxes, args.out)
    plot_motion(boxes, args.out)
    plot_attributes(stats, args.out)


if __name__ == "__main__":
    main()
