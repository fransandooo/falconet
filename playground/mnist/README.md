# Exercise 3c — Burn fundamentals on MNIST

Tensors and autodiff, then an MLP and a CNN trained with Burn's `Learner` on CPU and GPU, with live metrics in MLflow.

```bash
# 1. Tensor basics tour (shapes, broadcasting, matmul, autodiff check)
cargo run --release -p playground-mnist --bin tensors

# 2. (optional) start the tracking server in another terminal: http://127.0.0.1:5000
mise run mlflow

# 3. Train. In a terminal you get Burn's TUI dashboard (press `q` to stop early)
cargo run --release -p playground-mnist -- --model cnn --device cuda
cargo run --release -p playground-mnist -- --model mlp --device flex --epochs 1
cargo run --release -p playground-mnist -- --help
```

MNIST (~55 MB) is downloaded on first use to `~/.cache/burn-dataset/mnist`. Each run writes `config.json`, `epoch_times.csv`, the metric logs and the weights (`model.bpk`) to `runs/mnist/<model>-<device>/` (gitignored). If `MLFLOW_TRACKING_URI` is set (mise sets it inside the repo) and the server is up, the run also appears in MLflow; otherwise training continues without it.

## Reading order

1. [`src/bin/tensors.rs`](src/bin/tensors.rs): tensors, broadcasting, reductions, autodiff.
2. [`src/data.rs`](src/data.rs): the `Batcher` that turns dataset items into batched tensors.
3. [`src/model.rs`](src/model.rs): `#[derive(Module)]`, the MLP, the CNN, and `macro_rules!` to implement Burn's step traits.
4. [`src/training.rs`](src/training.rs): `SupervisedTraining` + `Learner`, a generic `train<M>` function with associated-type bounds, and a hand-written test loop.
5. [`src/timing.rs`](src/timing.rs): a custom `TrainingProgressLogger` shared across threads with `Arc<Mutex<..>>`.
6. [`../../crates/falconet-mlflow`](../../crates/falconet-mlflow/): the MLflow client and Burn metric logger.

## Concepts

| Concept | Where | C / Python analogy |
|---|---|---|
| `Tensor<D>` / `Tensor<D, Int>`: rank in the type, backend in the `Device` | everywhere | a NumPy array whose `ndim` is checked at compile time |
| Tensor ops consume `self`; `clone()` is a cheap handle copy | `tensors.rs` | Python references: no data copied |
| Autodiff as a device decorator: `device.autodiff()` | `main.rs`, `tensors.rs` | `requires_grad=True` + `loss.backward()` in PyTorch |
| Procedural macros: `#[derive(Module)]`, `#[derive(Config)]`, `#[derive(Parser)]` | `model.rs`, `training.rs`, `main.rs` | code generated at compile time instead of runtime introspection |
| `macro_rules!` and the orphan rule | `model.rs` | a `#define` that works on syntax, not text |
| Associated types in bounds: `TrainStep<Input = MnistBatch>` | `training.rs` | "this generic only accepts models whose input is X" |
| `Arc<Mutex<T>>`: shared ownership + locking across threads | `timing.rs`, `falconet-mlflow` | refcounted struct + `pthread_mutex`, but the data is inside the lock |
| Collecting `Result`s: `iter.collect::<Result<Vec<_>, _>>()` | `training.rs` | stop at the first error |
| `Drop` (RAII) for cleanup | `falconet-mlflow/src/run.rs` | a destructor; Python's `__exit__` |

## Try this

1. Train the MLP and the CNN on `cuda` and compare them side by side in MLflow (select both runs → *Compare*). Which one overfits?
2. Change the learning rate (`TrainingConfig::learning_rate`, or add a `--lr` flag with clap) and compare three runs in MLflow.
3. Add a `BatchNorm` after `fc1` in the CNN. Does validation accuracy move?
