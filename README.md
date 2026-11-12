# falconet

**A counter-UAV perception engine in Rust: detecting and tracking tiny drones in thermal video.**

[![CI](https://github.com/fransandooo/falconet/actions/workflows/ci.yml/badge.svg)](https://github.com/fransandooo/falconet/actions/workflows/ci.yml)
![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)

> 🚧 **Work in progress.** Phase 1 (data + end-to-end pipeline) is underway. Nothing here is usable yet.

A *falconet* is a small falcon: a small model hunting small targets.

## What this is

falconet is an end-to-end drone detection and tracking system for thermal (IR) video, written in Rust:

- **Detector** trained on public anti-UAV datasets, implemented and trained in [Burn](https://burn.dev).
- **Tracker**: a from-scratch ByteTrack implementation (Kalman filter, Hungarian assignment, track management).
- **Optimized**: INT8 quantization and hand-written GPU kernels with [CubeCL](https://github.com/tracel-ai/cubecl).
- **Benchmarked honestly** against PyTorch and ONNX Runtime on the same hardware (p50/p95/p99).
- **Runs in the browser** through WASM + WebGPU.

## Roadmap

| Phase | Topic | Status |
|---|---|---|
| 0 | Rust + Burn fundamentals, dev environment | ✅ done |
| 1 | End-to-end Rust pipeline with an imported model | 🟡 in progress |
| 2 | Training in Rust + training visualizer | ⚪ planned |
| 3 | INT8 quantization and pipeline optimization | ⚪ planned |
| 4 | Custom GPU kernels with CubeCL | ⚪ planned |
| 5 | Generalization study, web demo, technical report | ⚪ planned |

Progress reports for each milestone live in [`docs/checkpoints/`](docs/checkpoints/).

## Repository layout

```
falconet/
├── crates/        # the product: falconet-* crates
├── playground/    # phase 0 learning exercises, not part of the product
├── reference/     # PyTorch / ONNX Runtime baselines, for comparison only
├── scripts/       # dataset download and conversion
├── configs/       # versioned experiment configs
├── docs/          # checkpoints, experiment logs, technical report
└── media/         # figures, GIFs and clips
```

Directories appear as the phases that need them begin.

## Development setup

Requirements: Linux, [rustup](https://rustup.rs), and optionally [mise](https://mise.jdx.dev) for the extra dev tools. GPU backends need a recent NVIDIA driver (CUDA) and/or Vulkan.

```bash
git clone git@github.com:fransandooo/falconet.git && cd falconet

# rustup reads rust-toolchain.toml and installs the pinned toolchain automatically
cargo build

# optional: cargo-nextest and bacon, scoped to this repo
mise install

# run fmt + clippy + tests before every push (same checks as CI)
git config core.hooksPath .githooks

# check that Burn sees your GPU (flex CPU vs wgpu vs cuda)
cargo run --release -p playground-gpu-check
```

Everyday commands:

```bash
cargo fmt --all                                        # format
cargo clippy --workspace --all-targets -- -D warnings  # lint
cargo nextest run --workspace                          # test
bacon clippy                                           # re-run clippy on every save
```

### Experiment tracking

Training runs are tracked in a local [MLflow](https://mlflow.org) server: hyperparameters, live metric curves, timings, the git commit and artifacts. Rust talks to it over its REST API ([`crates/falconet-mlflow`](crates/falconet-mlflow/)).

```bash
mise run mlflow        # starts http://127.0.0.1:5000 (data in runs/mlflow/, gitignored)
```

Inside the repo, mise sets `MLFLOW_TRACKING_URI`, so training binaries log to the server automatically when it is running and train without it when it isn't.

## Datasets

No data is stored in this repository. Datasets live under `$FALCONET_DATA_DIR` (default `~/datasets`) and are fetched with the scripts in `scripts/`.

| Dataset | Use | Notes |
|---|---|---|
| [Anti-UAV410](docs/datasets/antiuav410.md) | main: train / val / test | thermal IR, 410 videos, 438k frames; `scripts/download_antiuav410.sh` |

Anti-UAV410 has no stated licence; it is used here for research only and is not redistributed. Please cite:

> B. Huang, J. Li, J. Chen, G. Wang, J. Zhao, T. Xu. *Anti-UAV410: A Thermal Infrared Benchmark and Customized Scheme for Tracking Drones in the Wild.* IEEE T-PAMI, 2023.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this work, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
