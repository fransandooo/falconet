//! CenterNet loss: penalty-reduced focal loss on the heatmap + L1 on size and offset.

use burn::prelude::*;
use burn::tensor::activation::{log_sigmoid, sigmoid};

use crate::data::DetectionBatch;
use crate::model::DetectorOutput;

/// Focal loss exponents from the CornerNet/CenterNet papers.
const ALPHA: f32 = 2.0;
const BETA: f32 = 4.0;
/// Size is in grid cells (values up to ~3.5) and converges more easily than
/// the heatmap, so it gets a smaller weight, as in CenterNet.
const SIZE_WEIGHT: f32 = 0.1;

/// The loss and its parts (each a 1-element tensor). The parts are only used
/// for logging; `total` is what we call `backward()` on.
#[derive(Debug, Clone)]
pub struct LossParts {
    pub total: Tensor<1>,
    pub heatmap: Tensor<1>,
    pub size: Tensor<1>,
    pub offset: Tensor<1>,
}

/// Computes the loss of a batch.
///
/// Why a *focal* loss? Each 16x16 heatmap has 1 positive cell and 255
/// negatives. Plain binary cross-entropy would be dominated by the easy
/// negatives. The focal term `(1 - p)^α` / `p^α` down-weights cells the model
/// already gets right, and `(1 - y)^β` further reduces the penalty for cells
/// right next to the centre (they are "almost" correct).
pub fn detection_loss(out: DetectorOutput, batch: &DetectionBatch) -> LossParts {
    let num_objects = batch.boxes.len() as f32;
    let target = batch.heatmap.clone();
    let positive = batch.center_mask.clone();
    let negative = positive.clone().neg().add_scalar(1.0);

    let logits = out.heatmap_logits;
    let p = sigmoid(logits.clone());
    // log(p) and log(1 - p) computed from the logits for numerical stability:
    // log(sigmoid(x)) and log(sigmoid(-x)) never take the log of 0.
    let log_p = log_sigmoid(logits.clone());
    let log_not_p = log_sigmoid(logits.neg());

    let pos_term = p.clone().neg().add_scalar(1.0).powf_scalar(ALPHA) * log_p * positive.clone();
    let neg_term = target.neg().add_scalar(1.0).powf_scalar(BETA)
        * p.powf_scalar(ALPHA)
        * log_not_p
        * negative;
    let heatmap = (pos_term.sum() + neg_term.sum()).neg() / num_objects;

    // L1 only at the centre cell: the [b, 1, g, g] mask broadcasts over the
    // two channels of size/offset.
    let size = ((out.size - batch.size.clone()).abs() * positive.clone()).sum() / num_objects;
    let offset =
        ((sigmoid(out.offset_logits) - batch.offset.clone()).abs() * positive).sum() / num_objects;

    LossParts {
        total: heatmap.clone() + size.clone() * SIZE_WEIGHT + offset.clone(),
        heatmap,
        size,
        offset,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DetectionBatcher;
    use crate::scene::SceneConfig;
    use burn::data::dataloader::batcher::Batcher;

    fn batch(device: &Device) -> DetectionBatch {
        let cfg = SceneConfig::default();
        let scenes = (0..2).map(|i| cfg.generate(i)).collect();
        DetectionBatcher::new(&cfg).batch(scenes, device)
    }

    /// Builds network outputs that match the targets almost perfectly.
    fn perfect_output(batch: &DetectionBatch) -> DetectorOutput {
        // Large positive logit at the centre, large negative elsewhere.
        let logits = batch.center_mask.clone() * 20.0 - 10.0;
        // logit(offset), clamped away from 0/1 so it stays finite.
        let off = batch.offset.clone().clamp(1e-4, 1.0 - 1e-4);
        let offset_logits = (off.clone() / off.neg().add_scalar(1.0)).log();
        DetectorOutput {
            heatmap_logits: logits,
            size: batch.size.clone(),
            offset_logits,
        }
    }

    #[test]
    fn a_perfect_prediction_has_near_zero_loss() {
        let device = Device::flex();
        let batch = batch(&device);
        let loss: f32 = detection_loss(perfect_output(&batch), &batch)
            .total
            .into_scalar();
        assert!(loss < 1e-3, "{loss}");
    }

    #[test]
    fn an_uninformed_prediction_has_a_much_larger_loss() {
        let device = Device::flex();
        let batch = batch(&device);
        let zeros = || Tensor::<4>::zeros(batch.size.dims(), &device);
        let out = DetectorOutput {
            heatmap_logits: Tensor::zeros(batch.heatmap.dims(), &device),
            size: zeros(),
            offset_logits: zeros(),
        };
        let loss: f32 = detection_loss(out, &batch).total.into_scalar();
        assert!(loss > 1.0, "{loss}");
    }
}
