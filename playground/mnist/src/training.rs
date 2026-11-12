//! Training with Burn's `Learner`, and a hand-written evaluation loop.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::data::dataset::transform::PartialDataset;
use burn::data::dataset::vision::{MnistDataset, MnistItem};
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::train::logger::FileMetricLogger;
use burn::train::metric::{AccuracyMetric, LossMetric};
use burn::train::{
    ClassificationOutput, InferenceStep, Learner, LearnerModel, SupervisedTraining, TrainStep,
};
use falconet_mlflow::{MlflowMetricLogger, Run};

use crate::data::{MnistBatch, MnistBatcher};
use crate::model::Classifier;
use crate::timing::{EpochTimer, EpochTiming};

/// MNIST ships 60k training images; we keep the last 5k for validation so
/// that the official 10k test split is only used once, at the very end.
const TRAIN_SPLIT_END: usize = 55_000;
const MNIST_TRAIN_LEN: usize = 60_000;

/// Hyperparameters, saved next to the results for reproducibility.
///
/// `#[derive(Config)]` (Burn) generates a constructor taking the fields without
/// a default (`TrainingConfig::new(model)`), `with_*` setters for the others,
/// and JSON `save`/`load`.
#[derive(Config, Debug)]
pub struct TrainingConfig {
    /// "mlp" or "cnn".
    pub model: String,
    /// Backend name, for the record ("cuda", "wgpu", "flex").
    pub device: String,
    #[config(default = 3)]
    pub num_epochs: usize,
    #[config(default = 128)]
    pub batch_size: usize,
    /// How many of the 55k training images to use (smaller = faster runs).
    #[config(default = 55_000)]
    pub train_items: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-3)]
    pub learning_rate: f64,
}

/// What a training run produced.
#[derive(Debug)]
pub struct RunSummary {
    pub epochs: Vec<EpochTiming>,
    pub total: Duration,
    pub test_accuracy: f32,
    pub artifact_dir: PathBuf,
}

/// Trains `model` on MNIST and evaluates it on the test split.
///
/// Generic over the model type. Inside a generic function the compiler only
/// knows what the bounds say, so they spell out everything we rely on:
/// - `LearnerModel`: something Burn's `Learner` can train,
/// - `Classifier`: our own trait, used by the test loop,
/// - `TrainStep<Input = MnistBatch, ..>`: fixes the step's *associated types*.
///   Without it, `M`'s input type could be anything, and our MNIST data
///   loaders and classification metrics would not be known to fit.
///
/// `tracking` is an optional MLflow run: `Some(run)` streams metrics to it,
/// `None` trains without tracking. `Option` instead of a "null run" object.
pub fn train<M>(
    model: M,
    config: &TrainingConfig,
    artifact_dir: &Path,
    tracking: Option<&Run>,
) -> RunSummary
where
    M: LearnerModel
        + Classifier
        + TrainStep<Input = MnistBatch, Output = ClassificationOutput>
        + InferenceStep<Input = MnistBatch, Output = ClassificationOutput>,
{
    fs::create_dir_all(artifact_dir).expect("artifact directory should be creatable");
    config
        .save(artifact_dir.join("config.json"))
        .expect("config should be saved");

    if let Some(run) = tracking {
        log_params(run, config, model.num_params());
    }

    // `Arc` lets both partial datasets share one copy of the 60k images.
    let mnist_train = Arc::new(MnistDataset::train());
    let train_end = config.train_items.min(TRAIN_SPLIT_END);
    let train_set = PartialDataset::new(mnist_train.clone(), 0, train_end);
    let valid_set = PartialDataset::new(mnist_train, TRAIN_SPLIT_END, MNIST_TRAIN_LEN);

    let loader_train = DataLoaderBuilder::new(MnistBatcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_set);
    let loader_valid = DataLoaderBuilder::new(MnistBatcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(valid_set);

    let timer = EpochTimer::default();
    let mut training = SupervisedTraining::new(artifact_dir, loader_train, loader_valid)
        .metrics((AccuracyMetric::new(), LossMetric::new()))
        .with_progress_logger(timer.clone())
        .num_epochs(config.num_epochs)
        .summary();
    if let Some(run) = tracking {
        // Registering any logger replaces Burn's default, so the file logger
        // is added back explicitly, first: Burn reads metrics back from it.
        training = training
            .with_metric_logger(FileMetricLogger::new(artifact_dir))
            .with_metric_logger(MlflowMetricLogger::new(run.clone()));
    }

    let start = Instant::now();
    let result = training.launch(Learner::new(
        model,
        AdamConfig::new().init(),
        config.learning_rate,
    ));
    let total = start.elapsed();

    let test_accuracy = test_accuracy(&result.model);

    result
        .model
        .into_record()
        .save(artifact_dir.join("model"))
        .expect("trained model should be saved");

    let epochs = timer.timings();
    let csv_path = artifact_dir.join("epoch_times.csv");
    write_timings_csv(&epochs, &csv_path);

    if let Some(run) = tracking {
        // Percent, like Burn's own accuracy metric.
        run.log_metric("test/accuracy", f64::from(test_accuracy) * 100.0, 0);
        run.log_metric("time/total_s", total.as_secs_f64(), 0);
        for e in &epochs {
            let step = e.epoch as i64;
            run.log_metric("time/train_epoch_s", e.train.as_secs_f64(), step);
            run.log_metric("time/valid_epoch_s", e.valid.as_secs_f64(), step);
            run.log_metric("time/train_images_per_s", e.train_items_per_sec(), step);
        }
        for file in [artifact_dir.join("config.json"), csv_path] {
            if let Err(err) = run.log_artifact(&file) {
                eprintln!("warning: could not upload {}: {err}", file.display());
            }
        }
    }

    RunSummary {
        epochs,
        total,
        test_accuracy,
        artifact_dir: artifact_dir.to_path_buf(),
    }
}

/// Plain inference loop over the 10k test images, without the `Learner`:
/// this is what running the model looks like outside of training.
pub fn test_accuracy<M: Classifier + Module>(model: &M) -> f32 {
    let device = model.devices()[0].clone();
    // Reading an item can fail, so the iterator yields `Result`s. Collecting
    // an iterator of `Result<T, E>` into `Result<Vec<T>, E>` stops at the
    // first error: one `expect` instead of one per item.
    let items: Vec<MnistItem> = MnistDataset::test()
        .iter()
        .collect::<Result<_, _>>()
        .expect("MNIST test split should be readable");

    let mut correct = 0_i64;
    for chunk in items.chunks(1_000) {
        let batch = MnistBatcher.batch(chunk.to_vec(), &device);
        // argmax over the class dimension → predicted class per image.
        let predicted = model.logits(batch.images).argmax(1).flatten::<1>(0, 1);
        let hits: i64 = predicted.equal(batch.targets).int().sum().into_scalar();
        correct += hits;
    }
    correct as f32 / items.len() as f32
}

fn log_params(run: &Run, config: &TrainingConfig, num_params: usize) {
    let params = [
        ("model", config.model.clone()),
        ("device", config.device.clone()),
        ("num_epochs", config.num_epochs.to_string()),
        ("batch_size", config.batch_size.to_string()),
        ("train_items", config.train_items.to_string()),
        ("learning_rate", config.learning_rate.to_string()),
        ("seed", config.seed.to_string()),
        ("num_workers", config.num_workers.to_string()),
        ("optimizer", "adam".to_owned()),
        ("num_params", num_params.to_string()),
    ];
    if let Err(err) = run.log_params(&params) {
        eprintln!("warning: could not log params to MLflow: {err}");
    }
}

fn write_timings_csv(epochs: &[EpochTiming], path: &Path) {
    let mut csv = String::from("epoch,train_s,valid_s,train_items,train_items_per_s\n");
    for e in epochs {
        csv += &format!(
            "{},{:.3},{:.3},{},{:.1}\n",
            e.epoch,
            e.train.as_secs_f64(),
            e.valid.as_secs_f64(),
            e.train_items,
            e.train_items_per_sec()
        );
    }
    fs::write(path, csv).expect("timings CSV should be writable");
}
