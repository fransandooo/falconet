//! Train an MLP or a CNN on MNIST with Burn.
//!
//! ```text
//! cargo run --release -p playground-mnist -- --model cnn --device cuda
//! cargo run --release -p playground-mnist -- --model mlp --device flex --train-items 10000
//! cargo run --release -p playground-mnist -- --help
//! ```
//!
//! In a real terminal Burn shows its TUI dashboard (press `q` to stop early);
//! when the output is piped it falls back to plain text.

use std::path::Path;

use burn::tensor::{Device, DeviceKind};
use clap::{Parser, ValueEnum};
use falconet_mlflow::{MlflowClient, Run, RunStatus, git_tags};
use playground_mnist::model::{Cnn, Mlp};
use playground_mnist::training::{self, RunSummary, TrainingConfig};

/// Command-line arguments. `#[derive(Parser)]` (from `clap`) generates the
/// parser, `--help` text and error messages from this struct and its doc comments.
#[derive(Parser, Debug)]
#[command(about = "Train an MLP or a CNN on MNIST with Burn")]
struct Args {
    /// Model architecture.
    #[arg(long, value_enum, default_value_t = ModelKind::Cnn)]
    model: ModelKind,
    /// Burn backend to train on.
    #[arg(long, value_enum, default_value_t = Backend::Cuda)]
    device: Backend,
    #[arg(long, default_value_t = 3)]
    epochs: usize,
    #[arg(long, default_value_t = 128)]
    batch_size: usize,
    /// Number of training images to use (max 55000).
    #[arg(long, default_value_t = 55_000)]
    train_items: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// MLflow tracking server. Inside this repo mise sets MLFLOW_TRACKING_URI;
    /// if the server isn't running, training continues without tracking.
    #[arg(long, env = "MLFLOW_TRACKING_URI")]
    mlflow_uri: Option<String>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ModelKind {
    Mlp,
    Cnn,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Backend {
    Cuda,
    Wgpu,
    Flex,
}

impl Backend {
    fn name(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Wgpu => "wgpu",
            Self::Flex => "flex",
        }
    }

    fn device(self) -> Device {
        match self {
            Self::Cuda => Device::cuda(0),
            Self::Wgpu => Device::wgpu(DeviceKind::DiscreteGpu(0)),
            Self::Flex => Device::flex(),
        }
    }
}

impl ModelKind {
    fn name(self) -> &'static str {
        match self {
            Self::Mlp => "mlp",
            Self::Cnn => "cnn",
        }
    }
}

fn main() {
    let args = Args::parse();

    let config = TrainingConfig::new(args.model.name().into(), args.device.name().into())
        .with_num_epochs(args.epochs)
        .with_batch_size(args.batch_size)
        .with_train_items(args.train_items)
        .with_seed(args.seed);

    // `runs/` at the repository root (gitignored), whatever the current directory.
    // `ancestors()` walks up the path: playground/mnist → playground → repo root.
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crate lives two levels below the repo root");
    let artifact_dir = repo_root
        .join("runs/mnist")
        .join(format!("{}-{}", config.model, config.device));

    let device = args.device.device();
    device.seed(config.seed);
    // Wrapping the device with autodiff makes every tensor created on it
    // record the operations needed for `backward()`.
    let train_device = device.autodiff();

    let tracking = args
        .mlflow_uri
        .as_deref()
        .and_then(|uri| start_tracking(uri, &config));

    // Each arm calls `training::train` with a different concrete model type:
    // the compiler generates one version of `train` for `Mlp` and one for `Cnn`.
    let run = tracking.as_ref();
    let summary = match args.model {
        ModelKind::Mlp => training::train(Mlp::new(&train_device), &config, &artifact_dir, run),
        ModelKind::Cnn => training::train(Cnn::new(&train_device), &config, &artifact_dir, run),
    };

    print_summary(&config, &summary);
    if let Some(run) = tracking {
        match run.finish(RunStatus::Finished) {
            Ok(()) => println!("mlflow run:      {}", run.ui_url()),
            Err(err) => eprintln!("warning: could not close the MLflow run: {err}"),
        }
    }
}

/// Opens an MLflow run, or returns `None` (with a warning) if the server is down.
fn start_tracking(uri: &str, config: &TrainingConfig) -> Option<Run> {
    let client = MlflowClient::new(uri);
    let run_name = format!("{}-{}", config.model, config.device);
    match Run::start(&client, "mnist", &run_name, &git_tags()) {
        Ok(run) => Some(run),
        Err(err) => {
            eprintln!("warning: MLflow tracking disabled ({err}); start it with `mise run mlflow`");
            None
        }
    }
}

fn print_summary(config: &TrainingConfig, summary: &RunSummary) {
    println!(
        "\n{} on {}: {} epochs, batch {}, {} training images",
        config.model, config.device, config.num_epochs, config.batch_size, config.train_items
    );
    println!(
        "{:>5} {:>10} {:>10} {:>12}",
        "epoch", "train [s]", "valid [s]", "train img/s"
    );
    for e in &summary.epochs {
        println!(
            "{:>5} {:>10.2} {:>10.2} {:>12.0}",
            e.epoch,
            e.train.as_secs_f64(),
            e.valid.as_secs_f64(),
            e.train_items_per_sec()
        );
    }
    println!("total wall time: {:.1} s", summary.total.as_secs_f64());
    println!("test accuracy:   {:.2}%", summary.test_accuracy * 100.0);
    println!("artifacts:       {}", summary.artifact_dir.display());
}
