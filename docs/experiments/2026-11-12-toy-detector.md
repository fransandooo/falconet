# Toy detector: fixed vs fresh training scenes (exercise 3d)

- **Date:** 2026-11-12
- **Commit:** parent `d4c82ae` + uncommitted 3d changes, committed right after this log (`git.dirty=true` in MLflow)
- **Config:** `TrainConfig` defaults: 30 epochs, batch 64, 8,192 training scenes per epoch, 1,024 validation scenes (fixed), Adam with peak lr 2e-3 and cosine decay, 4 loader workers. Scenes use `SceneConfig::default()`: 64×64, boxes 3–14 px, noise σ 0.18, contrast 0.15–0.5, up to 3 distractor blobs with amplitude ≤ 0.45
- **Seed:** 42
- **Hardware:** Intel Core i7-12700H, RTX 3050 Ti Laptop 4 GB, driver 615.78.08, CUDA 13.4
- **Software:** rustc 1.99.1, Burn 0.22.1, release build
- **Commands:**
  ```bash
  cargo run --release -p playground-toy-detector -- --device cuda --gif media/phase-0/toy-detector-learning.gif
  cargo run --release -p playground-toy-detector -- --device cuda --fixed-train-set
  ```
- **Tracking:** MLflow experiment `toy-detector`, runs `cuda` and `cuda-fixed-set`, with per-epoch prediction images

## Results

Validation metrics after epoch 30 (one run each):

| Training data | Train loss | Mean IoU | Recall@0.5 | Recall@0.5 small (≤ 6 px) | Centre error |
|---|---|---|---|---|---|
| Same scenes every epoch | 0.054 | 0.757 | 88.7% | 65.6% | 3.07 px |
| Fresh scenes every epoch | 0.325 | **0.830** | **94.2%** | **77.1%** | **1.87 px** |

Epoch time: ~3.7 s (train + validation + rendering); the first epoch takes 7–9 s because of kernel compilation.

A repeat of the fixed-set configuration gave IoU 0.752, recall 88.0% and small-box recall 59.4%. The aggregate metrics vary by about ±0.005 IoU; small-box recall varies more (±3 points), since there are only 96 small boxes in the validation set.

Earlier variants, for context:
- **Without distractors** (noise 0.12, contrast 0.25–0.6): 99.6% recall after a few epochs, small boxes 95.8%. Too easy to be informative.
- **A new DataLoader per epoch:** CUDA out of memory at epoch ~18; see notes.

## Notes

- **Overfitting with a fixed set.** Training loss falls 6× below the fresh-data run while every validation metric is worse. Small-box recall peaks around epoch 10 and then declines. Fresh data is the clean fix here. On real data the equivalent tool is augmentation.
- **Small boxes lag** the overall recall by 17 points even with fresh data. Part of this is the metric: IoU ≥ 0.5 on a 3×3 box tolerates less than a pixel of error. Phase 1 should report centre-distance-based metrics alongside IoU-based ones for tiny objects.
- **GPU memory leak from per-epoch DataLoaders (Burn 0.22.1, CUDA).** GPU memory measured with `nvidia-smi` grew 318 → 636 → 830 → 1,048 → 1,192 MiB over five short epochs when a fresh loader (4 workers) was built every epoch. With one loader iterated for the whole run it plateaued at ~806 MiB. This points to per-thread device memory not being released when loader worker threads exit. A minimal reproduction (~8 MiB per spawned worker thread; flat with 0 workers, with one reused loader, or with `[streaming] max_streams = 1`) was reported as [tracel-ai/burn#6505](https://github.com/tracel-ai/burn/issues/6505).
