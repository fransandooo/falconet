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

## Still to do (needs the Anti-UAV410 frames)

- Anti-UAV410 dataset class (frame layout inside the archive still to be verified), frame subsampling, augmentations that break the "drone in the centre" shortcut.
- **Real training, launched by Fran** (overnight). The command will be added here, logging to MLflow.
- Validation mAP / AP_small with `coco_eval`, and SA through the tracker.

## Disk use

`reference/.venv` is ~5.4 GB, mostly CUDA libraries bundled with PyTorch. It is gitignored; remove it with `rm -rf reference/.venv` (and `uv cache clean` for the cache).
