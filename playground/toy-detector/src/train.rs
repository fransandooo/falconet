//! A hand-written training loop.
//!
//! Exercise 3c used Burn's `Learner`, which hides the loop. Here we write it
//! out, because the real project needs full control: custom losses, custom
//! metrics, and rendering the model's predictions every epoch.
//!
//! ```text
//! for epoch:
//!     for batch:  forward → loss → backward → optimizer step   (train)
//!     for batch:  forward → decode → metrics                    (valid)
//!     render showcase frame, log everything
//! ```

use std::f64::consts::PI;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::optim::{AdamConfig, GradientsParams};
use burn::prelude::*;
use falconet_mlflow::Run;
use image::RgbImage;

use crate::data::{DetectionBatcher, SceneDataset};
use crate::decode::{Metrics, MetricsAccumulator, decode};
use crate::loss::detection_loss;
use crate::model::ToyDetector;
use crate::scene::{Scene, SceneConfig};
use crate::viz;

/// Hyperparameters, saved as `config.json` next to the results.
#[derive(Config, Debug)]
pub struct TrainConfig {
    #[config(default = 30)]
    pub epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    /// Training scenes per epoch.
    #[config(default = 8192)]
    pub train_scenes: usize,
    /// `false` (default): brand-new scenes every epoch ("infinite data").
    /// `true`: the same scenes every epoch, which overfits (see README).
    #[config(default = false)]
    pub fixed_train_set: bool,
    #[config(default = 1024)]
    pub valid_scenes: usize,
    /// Peak learning rate; it then follows a cosine decay to zero.
    #[config(default = 2e-3)]
    pub learning_rate: f64,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 4)]
    pub num_workers: usize,
    /// Backend name, for the record.
    pub device: String,
}

/// Per-epoch record (also written to `history.csv`).
#[derive(Debug, Clone, Copy)]
pub struct EpochRecord {
    pub epoch: usize,
    pub learning_rate: f64,
    pub train_loss: f32,
    pub valid: Metrics,
    pub seconds: f32,
}

/// Number of scenes in the showcase frames.
const SHOWCASE: usize = 4;

/// Trains the toy detector and returns the per-epoch history.
///
/// `on_frame(epoch, frame)` is called with the rendered showcase frame after
/// every epoch (and once before training, epoch 0). Taking a closure keeps
/// this function independent of where frames end up (disk, MLflow, a GIF).
pub fn train(
    config: &TrainConfig,
    device: &Device,
    out_dir: &Path,
    tracking: Option<&Run>,
    mut on_frame: impl FnMut(usize, &RgbImage),
) -> Vec<EpochRecord> {
    fs::create_dir_all(out_dir).expect("output directory should be creatable");
    config
        .save(out_dir.join("config.json"))
        .expect("config should be saved");

    let scene_config = SceneConfig::default();
    let batcher = DetectionBatcher::new(&scene_config);
    // Different base seeds → the training and validation scenes never overlap.
    let valid_set = SceneDataset::new(scene_config, config.seed + 1_000_003, config.valid_scenes);
    let showcase = pick_showcase(&valid_set);

    let train_device = device.clone().autodiff();
    train_device.seed(config.seed);

    // The loaders create batches directly on the device that will consume
    // them: the autodiff device for training, the plain one for validation.
    //
    // ONE training loader for the whole run, iterated continuously: epoch `e`
    // takes the next `batches_per_epoch` batches from the same stream. (A first
    // version built a new loader every epoch; with CUDA each loader's worker
    // threads left ~200 MB of GPU memory behind, and the 4 GB card ran out
    // around epoch 18. See the README.)
    let total_items = config.train_scenes * config.epochs;
    let distinct = if config.fixed_train_set {
        config.train_scenes // the same scenes every epoch → overfitting
    } else {
        total_items // brand-new scenes every epoch
    };
    let train_loader = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .set_device(train_device.clone())
        .build(SceneDataset::repeating(
            scene_config,
            config.seed,
            total_items,
            distinct,
        ));
    let batches_per_epoch = config.train_scenes.div_ceil(config.batch_size);
    let mut train_batches = train_loader.iter();
    let valid_loader = DataLoaderBuilder::new(batcher.clone())
        .batch_size(256)
        .num_workers(config.num_workers)
        .set_device(device.clone())
        .build(valid_set);

    // `mut`: both are replaced on every optimizer step (see below).
    let mut model = ToyDetector::new(&train_device);
    let mut optimizer = AdamConfig::new().init();

    if let Some(run) = tracking {
        log_params(run, config, model.num_params());
    }

    let render = |model: &ToyDetector, progress: f32| {
        let batch = batcher.batch(showcase.clone(), device);
        let predictions = decode(model.forward(batch.images));
        viz::render_frame(
            &showcase,
            &predictions,
            scene_config.image_size,
            4,
            progress,
        )
    };
    // `valid()` returns a copy of the model without autodiff, for inference.
    on_frame(0, &render(&model.valid(), 0.0));

    let mut history = Vec::with_capacity(config.epochs);
    for epoch in 1..=config.epochs {
        let start = Instant::now();
        let lr = cosine_lr(config.learning_rate, epoch, config.epochs);

        // ---- training ----
        let (mut loss_sum, mut batches) = (0.0_f32, 0_usize);
        let mut part_sums = [0.0_f32; 3];
        // `by_ref()` borrows the iterator instead of consuming it, so the next
        // epoch continues where this one stopped.
        for batch in train_batches.by_ref().take(batches_per_epoch) {
            let batch = batch.expect("synthetic scenes cannot fail to load");
            let loss = detection_loss(model.forward(batch.images.clone()), &batch);

            loss_sum += loss.total.clone().into_scalar::<f32>();
            part_sums[0] += loss.heatmap.into_scalar::<f32>();
            part_sums[1] += loss.size.into_scalar::<f32>();
            part_sums[2] += loss.offset.into_scalar::<f32>();
            batches += 1;

            // backward() → gradients for every tensor in the graph;
            // from_grads keeps the ones belonging to the model's parameters.
            let grads = GradientsParams::from_grads(loss.total.backward(), &model);
            // The optimizer takes the model *by value* and returns the updated
            // one: no hidden in-place mutation, ownership makes the update explicit.
            model = optimizer.step(lr, model, grads);
        }
        let train_loss = loss_sum / batches as f32;

        // ---- validation ----
        let valid_model = model.valid();
        let mut metrics = MetricsAccumulator::default();
        for batch in valid_loader.iter() {
            let batch = batch.expect("synthetic scenes cannot fail to load");
            let predictions = decode(valid_model.forward(batch.images));
            for (truth, pred) in batch.boxes.iter().zip(&predictions) {
                metrics.add(truth, &pred.bbox);
            }
        }
        let valid = metrics.finish();
        let seconds = start.elapsed().as_secs_f32();

        let record = EpochRecord {
            epoch,
            learning_rate: lr,
            train_loss,
            valid,
            seconds,
        };
        print_epoch(&record);
        if let Some(run) = tracking {
            let parts = part_sums.map(|s| s / batches as f32);
            log_epoch(run, &record, parts);
        }
        history.push(record);

        on_frame(
            epoch,
            &render(&valid_model, epoch as f32 / config.epochs as f32),
        );
    }

    write_history_csv(&history, &out_dir.join("history.csv"));
    model
        .valid()
        .into_record()
        .save(out_dir.join("model"))
        .expect("model should be saved");
    history
}

/// Cosine learning-rate decay: `peak` at epoch 1, close to 0 at the last epoch.
pub fn cosine_lr(peak: f64, epoch: usize, epochs: usize) -> f64 {
    let t = (epoch - 1) as f64 / epochs.max(1) as f64;
    peak * 0.5 * (1.0 + (PI * t).cos())
}

/// Picks the showcase scenes from the first validation scenes: the smallest
/// box, the largest one and two in between, so the frames show both easy
/// and hard cases.
fn pick_showcase(valid_set: &SceneDataset) -> Vec<Scene> {
    let mut candidates: Vec<Scene> = (0..64)
        .map(|i| valid_set.get(i).expect("valid index"))
        .collect();
    candidates.sort_by(|a, b| a.bbox.area().total_cmp(&b.bbox.area()));
    let last = candidates.len() - 1;
    (0..SHOWCASE)
        .map(|k| candidates[k * last / (SHOWCASE - 1)].clone())
        .collect()
}

fn print_epoch(r: &EpochRecord) {
    println!(
        "epoch {:>3} | lr {:.2e} | loss {:.4} | IoU {:.3} | R@0.5 {:.1}% (small {:.1}%) | centre err {:.2} px | {:.1} s",
        r.epoch,
        r.learning_rate,
        r.train_loss,
        r.valid.mean_iou,
        r.valid.recall_iou50 * 100.0,
        r.valid.recall_iou50_small * 100.0,
        r.valid.center_error_px,
        r.seconds,
    );
}

fn log_params(run: &Run, config: &TrainConfig, num_params: usize) {
    let params = [
        ("epochs", config.epochs.to_string()),
        ("batch_size", config.batch_size.to_string()),
        ("train_scenes", config.train_scenes.to_string()),
        ("fixed_train_set", config.fixed_train_set.to_string()),
        ("valid_scenes", config.valid_scenes.to_string()),
        ("learning_rate", config.learning_rate.to_string()),
        ("lr_schedule", "cosine".to_owned()),
        ("optimizer", "adam".to_owned()),
        ("seed", config.seed.to_string()),
        ("device", config.device.clone()),
        ("num_params", num_params.to_string()),
    ];
    if let Err(err) = run.log_params(&params) {
        eprintln!("warning: could not log params to MLflow: {err}");
    }
}

fn log_epoch(run: &Run, r: &EpochRecord, [heatmap, size, offset]: [f32; 3]) {
    let step = r.epoch as i64;
    let metrics = [
        ("train/loss", r.train_loss),
        ("train/heatmap_loss", heatmap),
        ("train/size_loss", size),
        ("train/offset_loss", offset),
        ("valid/mean_iou", r.valid.mean_iou),
        ("valid/recall_iou50", r.valid.recall_iou50),
        ("valid/recall_iou50_small", r.valid.recall_iou50_small),
        ("valid/center_error_px", r.valid.center_error_px),
        ("time/epoch_s", r.seconds),
    ];
    for (key, value) in metrics {
        run.log_metric(key, f64::from(value), step);
    }
    run.log_metric("train/learning_rate", r.learning_rate, step);
    run.flush(); // update the MLflow UI right away
}

fn write_history_csv(history: &[EpochRecord], path: &PathBuf) {
    let mut csv = String::from(
        "epoch,learning_rate,train_loss,mean_iou,recall_iou50,recall_iou50_small,center_error_px,seconds\n",
    );
    for r in history {
        csv += &format!(
            "{},{:.6e},{:.5},{:.4},{:.4},{:.4},{:.3},{:.2}\n",
            r.epoch,
            r.learning_rate,
            r.train_loss,
            r.valid.mean_iou,
            r.valid.recall_iou50,
            r.valid.recall_iou50_small,
            r.valid.center_error_px,
            r.seconds
        );
    }
    fs::write(path, csv).expect("history CSV should be writable");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_schedule_starts_at_peak_and_decays() {
        assert_eq!(cosine_lr(1.0, 1, 10), 1.0);
        assert!((cosine_lr(1.0, 6, 10) - 0.5).abs() < 1e-9);
        assert!(cosine_lr(1.0, 10, 10) < 0.03);
    }

    #[test]
    fn a_few_steps_reduce_the_loss() {
        // A smoke test of the whole training step on the CPU: if gradients
        // or the optimizer were wired wrong, the loss would not move.
        let device = Device::flex().autodiff();
        device.seed(0);
        let cfg = SceneConfig::default();
        let batcher = DetectionBatcher::new(&cfg);
        let scenes: Vec<Scene> = (0..16).map(|i| cfg.generate(i)).collect();
        let batch = batcher.batch(scenes, &device);

        let mut model = ToyDetector::new(&device);
        let mut optimizer = AdamConfig::new().init();
        let loss_of = |m: &ToyDetector| -> f32 {
            detection_loss(m.forward(batch.images.clone()), &batch)
                .total
                .into_scalar()
        };
        let before = loss_of(&model);
        for _ in 0..20 {
            let loss = detection_loss(model.forward(batch.images.clone()), &batch);
            let grads = GradientsParams::from_grads(loss.total.backward(), &model);
            model = optimizer.step(5e-3, model, grads);
        }
        let after = loss_of(&model);
        assert!(after < before * 0.7, "before {before}, after {after}");
    }
}
