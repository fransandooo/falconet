//! Train the toy detector and watch it learn.
//!
//! ```text
//! cargo run --release -p playground-toy-detector
//! cargo run --release -p playground-toy-detector -- --epochs 10 --device wgpu
//! cargo run --release -p playground-toy-detector -- --gif media/phase-0/toy-detector-learning.gif
//! ```
//!
//! Every epoch it renders the predictions on 4 fixed validation scenes. The
//! frames go to `runs/toy-detector/<device>/frames/`, to MLflow (image viewer
//! with a step slider, live while training) and, at the end, into a GIF.

use std::path::{Path, PathBuf};

use burn::tensor::{Device, DeviceKind};
use clap::{Parser, ValueEnum};
use falconet_mlflow::{MlflowClient, Run, RunStatus, git_tags};
use playground_toy_detector::train::{self, TrainConfig};
use playground_toy_detector::viz;

#[derive(Parser, Debug)]
#[command(about = "Train a tiny CenterNet-style detector on synthetic scenes")]
struct Args {
    #[arg(long, value_enum, default_value_t = Backend::Cuda)]
    device: Backend,
    #[arg(long, default_value_t = 30)]
    epochs: usize,
    #[arg(long, default_value_t = 64)]
    batch_size: usize,
    #[arg(long, default_value_t = 2e-3)]
    lr: f64,
    #[arg(long, default_value_t = 8192)]
    train_scenes: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Reuse the same training scenes every epoch (to see overfitting).
    #[arg(long)]
    fixed_train_set: bool,
    /// Also copy the learning GIF here (e.g. into `media/`).
    #[arg(long)]
    gif: Option<PathBuf>,
    /// MLflow tracking server (mise sets MLFLOW_TRACKING_URI inside the repo).
    #[arg(long, env = "MLFLOW_TRACKING_URI")]
    mlflow_uri: Option<String>,
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

fn main() {
    let args = Args::parse();
    let config = TrainConfig::new(args.device.name().into())
        .with_epochs(args.epochs)
        .with_batch_size(args.batch_size)
        .with_learning_rate(args.lr)
        .with_train_scenes(args.train_scenes)
        .with_seed(args.seed)
        .with_fixed_train_set(args.fixed_train_set);

    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crate lives two levels below the repo root");
    let run_name = if config.fixed_train_set {
        format!("{}-fixed-set", config.device)
    } else {
        config.device.clone()
    };
    let out_dir = repo_root.join("runs/toy-detector").join(&run_name);
    let frames_dir = out_dir.join("frames");
    std::fs::create_dir_all(&frames_dir).expect("frames directory should be creatable");

    let tracking = args.mlflow_uri.as_deref().and_then(|uri| {
        let client = MlflowClient::new(uri);
        Run::start(&client, "toy-detector", &run_name, &git_tags())
            .inspect_err(|err| {
                eprintln!(
                    "warning: MLflow tracking disabled ({err}); start it with `mise run mlflow`"
                )
            })
            .ok()
    });
    if let Some(run) = &tracking {
        println!(
            "mlflow run: {}  (images tab updates every epoch)",
            run.ui_url()
        );
    }

    // Smaller frames for the GIF so it stays a reasonable size.
    let mut gif_frames = Vec::new();
    let on_frame = |epoch: usize, frame: &image::RgbImage| {
        let png = viz::encode_png(frame);
        let path = frames_dir.join(format!("epoch_{epoch:03}.png"));
        std::fs::write(&path, &png).expect("frame should be writable");
        if let Some(run) = &tracking {
            let thumb = viz::encode_webp_thumbnail(frame, 512);
            if let Err(err) = run.log_image("valid/predictions", epoch as i64, &png, &thumb) {
                eprintln!("warning: could not upload frame {epoch}: {err}");
            }
        }
        gif_frames.push(image::imageops::resize(
            frame,
            frame.width() * 3 / 4,
            frame.height() * 3 / 4,
            image::imageops::FilterType::Nearest,
        ));
    };

    let history = train::train(
        &config,
        &args.device.device(),
        &out_dir,
        tracking.as_ref(),
        on_frame,
    );

    let gif_path = out_dir.join("learning.gif");
    viz::write_gif(&gif_frames, &gif_path, 250).expect("GIF should be writable");
    if let Some(dest) = &args.gif {
        std::fs::copy(&gif_path, dest).expect("GIF should be copyable");
        println!("gif copied to {}", dest.display());
    }

    if let Some(last) = history.last() {
        println!(
            "\nfinal: IoU {:.3}, recall@0.5 {:.1}% (small boxes {:.1}%), centre error {:.2} px",
            last.valid.mean_iou,
            last.valid.recall_iou50 * 100.0,
            last.valid.recall_iou50_small * 100.0,
            last.valid.center_error_px
        );
    }
    println!("outputs: {}", out_dir.display());

    if let Some(run) = tracking {
        for file in [
            out_dir.join("config.json"),
            out_dir.join("history.csv"),
            gif_path,
        ] {
            if let Err(err) = run.log_artifact(&file) {
                eprintln!("warning: could not upload {}: {err}", file.display());
            }
        }
        if let Err(err) = run.finish(RunStatus::Finished) {
            eprintln!("warning: could not close the MLflow run: {err}");
        }
    }
}
