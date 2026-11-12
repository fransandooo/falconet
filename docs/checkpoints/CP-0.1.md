## Checkpoint 0.1 — Environment + basic Rust

*Date: 2026-11-12 · Branch: `phase-0-fundamentals`*

**What was done:**
- Dev environment on Arch/Omarchy with nothing installed system-wide: Rust pinned to 1.99.1 via `rust-toolchain.toml` (rustup, including rust-analyzer, clippy, rustfmt and rust-src); `cargo-nextest` and `bacon` scoped to the repo via `mise.toml`. CUDA 13.4, the NVIDIA 615 driver and Vulkan were already present.
- Repository `fransandooo/falconet`: Cargo workspace (edition 2024, resolver 3), dual MIT OR Apache-2.0 licence, README skeleton, `.gitignore` for data and weights, experiment-log template, media conventions.
- Quality gates: GitHub Actions CI (fmt, clippy `-D warnings`, nextest, doctests) on PRs and on `main`, plus a local `pre-push` hook with the same checks (about 2 s with a warm cache).
- `playground/gpu-check`: the same matmul on Burn's `flex` (CPU), `wgpu` (Vulkan) and `cuda` backends, checked against the CPU result.
- Exercise 3a, `playground/annotations`: a parser for a toy Anti-UAV-style annotation format (ownership, borrowing, lifetimes, `Option`/`Result`, iterators).
- Exercise 3b, `playground/trackers`: a `Tracker` trait with naive and greedy-IoU implementations (traits, default methods, generics, `dyn` dispatch, closures).

**Results:**

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop, 4 GB, driver 615.78.08, CUDA 13.4 |
| Burn / CubeCL | 0.22.1 / 0.11.4 |
| matmul 1024² f32, `flex` CPU | 12–16 ms, 130–180 GFLOP/s |
| matmul 1024² f32, `wgpu` Vulkan | 1.6–1.7 ms, 1.2–1.4 TFLOP/s; max relative error vs CPU 1.6e-6 |
| matmul 1024² f32, `cuda` | 0.93 ms, 2.3 TFLOP/s; max relative error vs CPU 2.7e-4 (TF32) |
| First GPU call (kernel JIT + autotune) | 5.7 s wgpu / 7.0 s cuda cold; 0.13–0.27 s once the NVIDIA driver cache is warm |
| Tests | 26 unit tests + 1 doctest, all passing; clippy clean |
| Cold build of Burn (`flex` + `wgpu` + `cuda`, release) | ≈ 7 min locally (16 threads) |
| CI, first (cold) run | clippy 5.5 min; test step > 10 min (compiles Burn from scratch) |
| Disk | `target/` 15 GB; toolchain 0.7 GB; mise tools 35 MB |

These are smoke-test numbers (single run, no p50/p95), not benchmarks.

Tracker demo (`cargo run -p playground-trackers`, synthetic scene with 2 drones, a 2-frame occlusion and 1 false positive; ideal = 3 IDs):

| Tracker | Distinct IDs |
|---|---|
| naive (new ID per detection) | 59 |
| greedy IoU (IoU ≥ 0.3, max_missed = 2) | 4 |
| greedy IoU (IoU ≥ 0.1, max_missed = 5) | 3 |

**What they mean:**
- Both GPU paths work, and the GPU runs about 10× (Vulkan) to 18× (CUDA) faster than the CPU on a mid-size matmul. CUDA is about 1.8× faster than wgpu/Vulkan on the same card, which is the gap expected between tensor-core and portable kernels.
- The tracker demo already shows why ByteTrack needs a motion model: without one, the tracker gives drone B a new ID after a two-frame occlusion.

**Comparison with reference/target:** Task 1 is met: Burn sees the NVIDIA GPU through both `cuda` and `wgpu`. Task 2 is met. Exercises 3a and 3b are done. One deliberate deviation: `bacon` replaces `cargo-watch`, which has been unmaintained since 2024.

**Problems and limitations found:**
1. **Burn 0.22 changed its core API.** `Tensor<B, D>` is now `Tensor<D>`, the backend is chosen at runtime through `Device` (`Device::cuda(0)`, `Device::wgpu(..)`, `Device::flex()`), and `ndarray` is deprecated in favour of `flex`. Most tutorials, blog posts and LLM knowledge describe the old API. Model code gets simpler (no backend generic everywhere), but we have to check examples against 0.22.
2. **CUDA computes f32 matmuls in TF32.** Burn's autotuner promotes f32 to TF32 on tensor cores (`burn-cubecl/src/kernel/matmul/tune/base.rs`), which gives a relative error around 1e-4 instead of 1e-6. This matters for CP 1.3 (output parity with PyTorch). Either we find a way to force full f32, or we document the tolerance explicitly.
3. **4 GB of VRAM** limits batch size and resolution for local training. Heavy runs stay on rented GPUs, as planned.
4. **Cold-start latency is several seconds** (JIT compilation and autotuning). It matters for the "startup time" metric and for the web demo.
5. **Build cost:** Burn with three backends is heavy (7 min cold locally, 15 GB of `target/`, slow cold CI). Mitigations so far: dependencies built with `opt-level = 3` in dev, rust-cache in CI, and `gpu-check` excluded from the CI test step.

**What I learned / Rust concepts that came up:** Cargo workspaces and path dependencies; `rust-toolchain.toml`; ownership, moves, `Copy`; shared vs exclusive borrows; `&str` vs `String`; lifetimes (`Header<'a>`); `Option` and `Result` with `?`; enums as tagged unions; `impl Display` and `Error`; iterator chains; closures; generics with trait bounds; traits with default methods; static (monomorphized) vs dynamic (`dyn`, vtable) dispatch; `?Sized`; `#[cfg(test)]` unit tests and doctests.

**Decisions I need from you:**
1. *Debug info for dependencies.* Option (a): add `debug = false` for dependencies in the dev profile, which shrinks `target/` a lot; the only cost is that you can't step into Burn's internals with a debugger. Option (b): leave it as is. **Recommendation: (a).**
2. *When to merge `phase-0-fundamentals` into `main`.* Option (a): after CP 0.2, as one PR for the whole phase. Option (b): now. **Recommendation: (a)**, one branch per phase as the plan says.
3. *"Try this" exercises.* Do you want to do some of them before Burn? **Recommendation:** at least 3a.1 (read the compiler errors), 3a.3 (longest gap) and 3b.2 (constant-velocity prediction).

**Proposed next step:** exercise 3c, an MNIST MLP/CNN with Burn's `Learner` and training dashboard (`train`, `tui` and `vision` features), measuring CPU (`flex`) vs GPU (`cuda`, `wgpu`) epoch times. Then 3d, the toy detector, and CP 0.2.
