# reference/: PyTorch reference detector

The detector is trained here in PyTorch, exported to ONNX, and then run in Rust (phase 1 task 4). Later it is re-implemented and trained in Burn (phase 2). Python is used **only** for this reference and for comparisons.

## Architecture: CenterNet-lite

An anchor-free, single-class detector (option (a) of the phase 1 plan): **1.39 M parameters**.

| Part | Details |
|---|---|
| Input | grayscale `[B, 1, H, W]`, raw pixels in `[0, 1]`, H and W divisible by 32 (Anti-UAV410: 512 × 640). Normalisation is inside the model |
| Backbone | stem to stride 4 (two stride-2 convs + residual block), then stride 8 / 16 / 32 stages with residual blocks |
| Neck | FPN-lite: 1×1 laterals to 64 channels, nearest ×2 upsample + add, back to **stride 4** (160 × 128 output for a 640 × 512 frame), because half the drones are under 32 × 32 px |
| Heads | heatmap (1 ch, focal loss, prior bias −2.19), log-size (2 ch, L1: log w, log h so tiny and large drones weigh the same), sub-cell offset (2 ch, L1) |
| Decoding | 3 × 3 max-pool peak finding, top-k, `exp` of log-size |

The exported ONNX graph uses only **Conv, Relu, Add, Resize (nearest), Sub, Div** (batch norm is folded into the convolutions), which keeps the Rust side simple.

## Usage

```bash
cd reference
uv sync                                   # PyTorch 2.15 + CUDA 13.4 wheels, in reference/.venv
uv run pytest                             # unit tests (CPU, seconds)

# smoke test on synthetic thermal-like scenes (~90 s on the RTX 3050 Ti)
uv run python -m falconet_ref.train --dataset synthetic --epochs 8

# export + check against PyTorch with ONNX Runtime
uv run python -m falconet_ref.export_onnx --weights ../runs/reference/synthetic/model.pt
```

Smoke-test result (synthetic 256 × 256 scenes, 8 epochs, not a real benchmark): recall@IoU0.5 98.9%, mean IoU 0.87, frames without a drone answered correctly 100%. ONNX export: max |ONNX Runtime − PyTorch| = 9.5 × 10⁻⁶.

## Training on Anti-UAV410

```bash
mise run mlflow                           # from the repo root, in another terminal
cd reference
uv run python -m falconet_ref.train_antiuav --epochs 50 --samples-per-epoch 40000 --workers 6 --run-name baseline-e50
uv run python -m falconet_ref.train_antiuav --resume ../runs/reference/antiuav410/baseline-e50/last.pt   # after a crash
```

- **Data loader** (`falconet_ref/antiuav.py`): it reads `$FALCONET_DATA_DIR/antiuav410`, as grey frames.
  - It keeps both the box (training and COCO metrics) and the official `exist` flag (state accuracy); they differ on 37 border frames.
  - Augmentation: flip, scale jitter, gain/bias.
- **During training:** each epoch samples random frames, then validates on every 20th val frame (COCO metrics via pycocotools, single-drone metrics, detector-only SA). It saves `last.pt` and `best.pt` (by AP50) and renders prediction/heatmap panels for MLflow's image slider (`viz/*.png`, `*.npz` for videos).
- **At the end:** the best model is evaluated on the full val split. The results are written in the toolkit format (`final/val_sa/`, score with `antiuav410 sa --split val`) and as COCO files (`final/coco_*.json`, score with `coco_eval`), then exported to ONNX.

First run (2026-11-18, 4.6 h on the RTX 3050 Ti): full val **AP50 0.841, mAP 0.460, AP_small 0.308, detector-only mSA 0.705**. See `docs/experiments/2026-11-18-reference-antiuav410-baseline.md`.

## Disk use

`reference/.venv` is ~5.4 GB, mostly CUDA libraries bundled with PyTorch. It is gitignored; remove it with `rm -rf reference/.venv` (and `uv cache clean` for the cache).
