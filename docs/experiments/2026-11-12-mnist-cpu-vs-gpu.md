# MNIST: CPU vs GPU epoch times (exercise 3c)

- **Date:** 2026-11-12
- **Commit:** parent `bb7acf6` + the uncommitted 3c changes, which are committed right after this log (git tag `git.dirty=true` in MLflow)
- **Config:** `TrainingConfig` defaults: Adam, lr 1e-3, batch 128, 3 epochs, 4 data-loader workers, train 55,000 / valid 5,000 (taken from the MNIST train split), test 10,000
- **Seed:** 42 (GPU training is still not bit-reproducible; see notes)
- **Hardware:** Intel Core i7-12700H, 30 GB RAM; NVIDIA GeForce RTX 3050 Ti Laptop (4 GB), driver 615.78.08, CUDA 13.4; on AC power
- **Software:** rustc 1.99.1, Burn 0.22.1 (CubeCL 0.11.4), release build
- **Command:**
  ```bash
  for model in mlp cnn; do for dev in cuda wgpu flex; do
    cargo run --release -p playground-mnist -- --model $model --device $dev --epochs 3
  done; done
  ```
- **Tracking:** MLflow experiment `mnist`, local server

## Results

Training time per epoch in seconds (55,000 images). Steady-state throughput is from epoch 3. Each configuration was run **once**, so these are indicative numbers, not a benchmark with p50/p95.

| Model (params) | Backend | Epoch 1 | Epoch 2 | Epoch 3 | Steady img/s | Total wall [s] | Test acc. |
|---|---|---|---|---|---|---|---|
| MLP (235k) | cuda | 54.7 | 1.9 | 1.15 | 47,900 | 68.8 | 97.25% |
| MLP (235k) | wgpu (Vulkan) | 30.8 | 3.0 | 3.1 | 17,500 | 46.7 | 97.11% |
| MLP (235k) | flex (CPU) | 2.8 | 2.8 | 3.6 | 15,200 | **9.5** | 97.52% |
| CNN (225k) | cuda | 16.4 | 5.5 | 5.3 | **10,400** | 29.0 | 98.76% |
| CNN (225k) | wgpu (Vulkan) | 30.8 | 7.6 | 7.2 | 7,700 | 47.3 | 98.65% |
| CNN (225k) | flex (CPU) | 63.3 | 66.5 | 68.8 | 800 | 203.4 | 98.81% |

## Notes

- **CNN:** the GPU pays off. In steady state CUDA is ~13× faster than the CPU and wgpu ~9.6×. CUDA is ~1.35× faster than wgpu on the same card.
- **MLP:** a 235k-parameter MLP is too small to keep the GPU busy. Steady-state CUDA is only ~3× faster than the CPU, and wgpu is about as fast as the CPU. Once the first-epoch cost is counted, **the CPU finishes 3 epochs 7× sooner** (9.5 s vs 68.8 s).
- **The first epoch on GPU is dominated by kernel compilation and autotuning** (16–55 s). The MLP's kernels were new to the driver's JIT cache that day, while the CNN's had been compiled by earlier runs. That would explain why MLP/cuda's first epoch (55 s) is much slower than CNN/cuda's (16 s), but it is a hypothesis, not yet verified. Either way, startup cost has to be measured separately in every later benchmark.
- **CPU epochs get slower** (63 → 69 s) while all cores stay at 100%, probably thermal throttling on the laptop. Long CPU benchmarks need cool-down or a fixed clock.
- **Accuracy is not reproducible on GPU despite the fixed seed.** Two identical CNN/cuda runs earlier the same day gave 98.89% and 99.24% test accuracy. Likely causes: non-deterministic GPU reductions/atomics, autotune picking different kernels, and data-loader worker ordering. From now on, comparisons between models report mean ± std over several runs.
- All configurations reach the expected accuracy for 3 epochs (MLP ~97%, CNN ~99%), so the training loop, losses, metrics and evaluation are behaving correctly.
