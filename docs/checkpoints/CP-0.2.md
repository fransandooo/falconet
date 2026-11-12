## Checkpoint 0.2 — Burn running on GPU

*Date: 2026-11-12 · Branch: `phase-0-toy-detector` (phase 0 part 1 merged in PR #1)*

**What was done:**
- **3c, `playground/mnist`:** a tensor/autodiff tour, then an MLP and a CNN trained with Burn's `Learner` and TUI dashboard on `flex` (CPU), `wgpu` (Vulkan) and `cuda`, with a custom epoch timer (`TrainingProgressLogger`) and a hand-written test loop.
- **Experiment tracking, `crates/falconet-mlflow`:** MLflow REST client and Burn metric logger, plus step-indexed image logging compatible with MLflow's image viewer. `mise run mlflow` starts a local server.
- **3d, `playground/toy-detector`:** a CenterNet-style detector (heatmap + size + offset heads, focal loss) on synthetic scenes with distractors, trained with a hand-written loop. The predictions on 4 fixed scenes are rendered every epoch to PNG, MLflow and a GIF.
- Two experiment logs: `docs/experiments/2026-11-12-mnist-cpu-vs-gpu.md` and `2026-11-12-toy-detector.md`.

**Results:**

MNIST, steady-state training throughput (55k images, batch 128, epoch 3, one run each):

| Model | cuda | wgpu | flex (CPU) | First GPU epoch | Test acc. |
|---|---|---|---|---|---|
| CNN (225k params) | **10,400 img/s** (5.3 s/epoch) | 7,700 | 800 | 16–31 s | 98.7–98.8% |
| MLP (235k params) | 47,900 img/s | 17,500 | 15,200 | 31–55 s | 97.1–97.5% |

Toy detector, CUDA, 30 epochs (~3.7 s/epoch), validation on 1,024 fixed scenes:

| Training data | Mean IoU | Recall@0.5 | Recall@0.5, boxes ≤ 6 px | Centre error |
|---|---|---|---|---|
| Same scenes every epoch | 0.757 | 88.7% | 65.6% | 3.07 px |
| Fresh scenes every epoch | **0.830** | **94.2%** | **77.1%** | **1.87 px** |

Media: `media/phase-0/toy-detector-learning.gif`. The MNIST dashboard screenshot is still pending (it needs Fran's terminal).

**What they mean:**
- Training on the laptop GPU works on both GPU backends. For a CNN, CUDA is ~13× the CPU and ~1.35× wgpu.
- For tiny models the GPU barely pays off, and the first epoch (kernel JIT + autotune, 16–55 s) can dominate short runs. Startup cost must be measured separately in every benchmark.
- The toy detector learns, and it already shows the project's core difficulty: small-box recall lags by ~17 points, partly because IoU ≥ 0.5 is extremely strict at 3 px.
- Overfitting appears immediately with a fixed training set. On real data, augmentation will play the role that fresh synthetic data plays here.

**Comparison with reference/target:** Deliverables met: MNIST and the toy detector trained on GPU, CPU vs GPU epoch times measured, visual material produced (GIF; screenshot pending). Beyond plan: MLflow tracking, and live per-epoch prediction images, which the plan expected only in phase 2.

**Burn ergonomics, impressions so far:**
- 👍 0.22's device-based dispatch means one model codebase for CPU, CUDA and Vulkan, without generics everywhere. `#[derive(Module/Config)]` is pleasant. The `Learner` + TUI gets you training in minutes. The hand-written loop is explicit and clean (`optimizer.step` consumes and returns the model). Tensor rank is checked at compile time.
- 👎 **API churn.** 0.22 removed the backend generic and `AutodiffModule`, deprecated `ndarray`, changed `Dataset::get` to return `Result`, and added trait methods. Most docs, tutorials and LLM knowledge are outdated; the official `examples/` at the release tag are the reliable reference.
- 👎 **Heavy builds.** A cold release build with three backends takes 7–9 min; adding a dependency to a crate means 4–5 min.
- 👎 **Performance gotchas.** CUDA matmuls run in TF32 by default (CP 0.1). GPU training is not bit-reproducible despite fixed seeds. The TUI dashboard needs a real TTY.

**Problems and limitations found:**
1. **GPU memory leak when creating a DataLoader per epoch** (CUDA, 4 workers): about +200 MB per epoch, OOM at epoch ~18 on the 4 GB card. Fixed by building loaders once. *Update: Fran approved reporting it; filed as [tracel-ai/burn#6505](https://github.com/tracel-ai/burn/issues/6505) with a minimal reproduction.*
2. **Non-determinism on GPU.** Identical MNIST runs gave 98.89% and 99.24%. From now on, model comparisons need several runs (mean ± std).
3. **4 GB of VRAM** was enough here, but it is the main constraint for the real detector at full resolution.
4. **MLflow's image slider is not verified in the browser.** It relies on MLflow 3's undocumented artifact naming convention, copied from the official Python client. Artifacts and tags are verified via the API, but nobody has looked at the UI yet.

**What I learned / Rust concepts that came up:** procedural macros (`derive(Module, Config, Parser)`); `macro_rules!` and the orphan rule; associated types in trait bounds (`TrainStep<Input = MnistBatch>`); `Arc<Mutex<T>>` across threads; `Drop`/RAII for cleanup; `serde` and `thiserror`; type aliases; closures as parameters (`impl FnMut`); `Iterator::by_ref()`; collecting `Result`s; `wrapping_*` arithmetic.

**Decisions I need from you:**
1. *Report the DataLoader GPU-memory leak upstream* (a GitHub issue on `tracel-ai/burn` from your account, with a minimal reproduction). It's public and goes out under your name, so it needs your OK. **Recommendation: yes.** It's useful to the project and a good visible contribution.
2. *Where datasets live.* Option (a): `data/` inside the repo, gitignored. Option (b): a separate directory such as `~/datasets/`, referenced by an environment variable. **Recommendation: (b)**, so `cargo clean` or a fresh clone never touches tens of GB. I'll confirm Anti-UAV410's real size and licence first.

**Proposed next step:** Phase 1, tasks 1–2: verify Anti-UAV410's official source, licence, size, annotation format and evaluation metric (and tell you if it needs registration). Then the download/checksum script, the `falconet-data` parser and the dataset statistics with plots. That leads to CP 1.1.
