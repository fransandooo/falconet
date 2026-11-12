//! From network outputs to boxes, and how good those boxes are.

use burn::prelude::*;
use burn::tensor::activation::sigmoid;
use playground_annotations::BBox;

use crate::STRIDE;
use crate::model::DetectorOutput;

/// Boxes whose longer side is at most this many pixels count as "small".
/// The real dataset is full of these; tracking their recall separately is a
/// preview of the AP_small metric in the plan.
pub const SMALL_MAX_SIDE: f32 = 6.0;

/// The model's answer for one image.
#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    pub bbox: BBox,
    /// Heatmap probability at the chosen peak.
    pub score: f32,
    /// The full heatmap (probabilities, `grid * grid`), for visualisation.
    pub heatmap: Vec<f32>,
}

/// Decodes one prediction per image: there is exactly one object per scene,
/// so we take the heatmap's global maximum. (The real detector will need
/// local-maximum search + a score threshold to handle 0..N objects.)
pub fn decode(out: DetectorOutput) -> Vec<Prediction> {
    let [batch, _, grid, _] = out.heatmap_logits.dims();
    let cells = grid * grid;
    // One device → host copy per output, then plain Rust on slices.
    let to_vec = |t: Tensor<4>| -> Vec<f32> { t.into_data().try_to_vec().expect("f32 tensor") };
    let heat = to_vec(sigmoid(out.heatmap_logits));
    let size = to_vec(out.size);
    let offset = to_vec(sigmoid(out.offset_logits));

    (0..batch)
        .map(|b| {
            let heatmap = heat[b * cells..(b + 1) * cells].to_vec();
            let (peak, score) = heatmap
                .iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .expect("heatmap is not empty");
            let (row, col) = (peak / grid, peak % grid);
            // Channel c of image b lives at offset (b * 2 + c) * cells.
            let at = |data: &[f32], channel: usize| data[(b * 2 + channel) * cells + peak];

            let stride = STRIDE as f32;
            let cx = (col as f32 + at(&offset, 0)) * stride;
            let cy = (row as f32 + at(&offset, 1)) * stride;
            let w = (at(&size, 0) * stride).max(0.0);
            let h = (at(&size, 1) * stride).max(0.0);
            Prediction {
                bbox: BBox::new(cx - w / 2.0, cy - h / 2.0, w, h),
                score,
                heatmap,
            }
        })
        .collect()
}

/// Validation metrics over many (ground truth, prediction) pairs.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Metrics {
    pub count: usize,
    pub mean_iou: f32,
    /// Fraction of images whose predicted box has IoU ≥ 0.5 with the truth.
    pub recall_iou50: f32,
    pub count_small: usize,
    /// The same, only over small boxes.
    pub recall_iou50_small: f32,
    /// Mean distance between predicted and true centres, in pixels.
    pub center_error_px: f32,
}

/// Accumulates metrics batch by batch (so we never hold all predictions).
#[derive(Debug, Default)]
pub struct MetricsAccumulator {
    count: usize,
    iou_sum: f32,
    hits: usize,
    count_small: usize,
    hits_small: usize,
    center_error_sum: f32,
}

impl MetricsAccumulator {
    pub fn add(&mut self, truth: &BBox, predicted: &BBox) {
        let iou = truth.iou(predicted);
        let hit = iou >= 0.5;
        let (tx, ty) = truth.center();
        let (px, py) = predicted.center();

        self.count += 1;
        self.iou_sum += iou;
        self.hits += usize::from(hit);
        self.center_error_sum += ((tx - px).powi(2) + (ty - py).powi(2)).sqrt();
        if truth.w.max(truth.h) <= SMALL_MAX_SIDE {
            self.count_small += 1;
            self.hits_small += usize::from(hit);
        }
    }

    pub fn finish(&self) -> Metrics {
        let ratio = |a: usize, b: usize| if b == 0 { 0.0 } else { a as f32 / b as f32 };
        let mean = |sum: f32| {
            if self.count == 0 {
                0.0
            } else {
                sum / self.count as f32
            }
        };
        Metrics {
            count: self.count,
            mean_iou: mean(self.iou_sum),
            recall_iou50: ratio(self.hits, self.count),
            count_small: self.count_small,
            recall_iou50_small: ratio(self.hits_small, self.count_small),
            center_error_px: mean(self.center_error_sum),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::encode_targets;

    /// Feeds perfect targets through `decode` as if they were network outputs.
    fn decode_targets(bbox: BBox) -> Prediction {
        let device = Device::flex();
        let g = 16;
        let t = encode_targets(&bbox, g);
        let tensor = |v: Vec<f32>, c: usize| {
            Tensor::<4>::from_data(TensorData::new(v, [1, c, g, g]), &device)
        };
        // Inverse sigmoid so that `decode`'s sigmoid gives back the targets.
        let logit = |t: Tensor<4>| {
            let t = t.clamp(1e-6, 1.0 - 1e-6);
            (t.clone() / t.neg().add_scalar(1.0)).log()
        };
        let out = DetectorOutput {
            heatmap_logits: logit(tensor(t.heatmap, 1)),
            size: tensor(t.size, 2),
            offset_logits: logit(tensor(t.offset, 2)),
        };
        decode(out).remove(0)
    }

    #[test]
    fn decoding_perfect_targets_recovers_the_box() {
        let truth = BBox::new(21.0, 37.0, 7.0, 4.0);
        let pred = decode_targets(truth);
        assert!(truth.iou(&pred.bbox) > 0.99, "{:?}", pred.bbox);
        assert!(pred.score > 0.99);
    }

    #[test]
    fn metrics_count_hits_and_small_boxes() {
        let mut acc = MetricsAccumulator::default();
        let small = BBox::new(0.0, 0.0, 4.0, 4.0);
        let large = BBox::new(10.0, 10.0, 10.0, 10.0);
        acc.add(&small, &small); // hit, small
        acc.add(&large, &BBox::new(40.0, 40.0, 10.0, 10.0)); // miss, large
        let m = acc.finish();
        assert_eq!(m.count, 2);
        assert_eq!(m.recall_iou50, 0.5);
        assert_eq!((m.count_small, m.recall_iou50_small), (1, 1.0));
        assert_eq!(m.mean_iou, 0.5);
    }
}
