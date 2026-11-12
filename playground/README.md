# Playground (phase 0)

Small, self-contained learning exercises. **Not part of the product**: nothing in `crates/` depends on them.

| Crate | Exercise | Topic |
|---|---|---|
| [`gpu-check`](gpu-check/) | task 1 | Environment smoke test: the same matmul on Burn's `flex` (CPU), `wgpu` and `cuda` backends |
| [`annotations`](annotations/) | 3a | Ownership and borrowing: a toy annotation parser |
| [`trackers`](trackers/) | 3b | Traits and generics: a `Tracker` trait with two implementations |
| [`mnist`](mnist/) | 3c | Burn tensors and autodiff; an MLP and a CNN trained on CPU and GPU, tracked in MLflow |
| [`toy-detector`](toy-detector/) | 3d | A tiny CenterNet-style detector on synthetic scenes, hand-written training loop, per-epoch frames + GIF |

Each exercise has a README with a reading order, the concepts it covers and a few "try this" modifications.

```bash
cargo run -p playground-annotations
cargo run -p playground-trackers
cargo run --release -p playground-mnist -- --model cnn --device cuda
cargo run --release -p playground-toy-detector
cargo run --release -p playground-gpu-check   # needs an NVIDIA GPU
bacon test                                    # re-run tests on every save
```
