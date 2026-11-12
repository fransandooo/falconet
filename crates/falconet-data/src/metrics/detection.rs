//! COCO-style detection metrics for a single class (drone).
//!
//! A port of `pycocotools`' `COCOeval` for bounding boxes (`evaluateImg` +
//! `accumulate`), checked against it by `scripts/check_coco_parity.py`:
//!
//! 1. per image, detections are sorted by score and the top `max_dets` kept;
//! 2. for each IoU threshold, each detection (best score first) is matched to
//!    the unmatched ground-truth box it overlaps most, if IoU ≥ threshold;
//! 3. for an area range (e.g. "small" = area ≤ 32²), ground truth outside the
//!    range is ignored, as are detections matched to it and unmatched
//!    detections outside the range;
//! 4. all detections of all images are ranked by score, giving a
//!    precision/recall curve; precision is made monotonic and sampled at 101
//!    recall points (0, 0.01, …, 1). AP is their mean.
//!
//! mAP@[.5:.95] averages AP over the IoU thresholds 0.50, 0.55, …, 0.95.

use falconet_core::{BBox, Detection};
use serde::Serialize;

/// The ten COCO IoU thresholds: 0.50, 0.55, …, 0.95.
pub fn coco_iou_thresholds() -> [f32; 10] {
    std::array::from_fn(|i| 0.5 + 0.05 * i as f32)
}

/// An object-size range by box area (pixels²), inclusive at both ends like COCO.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct AreaRange {
    pub name: &'static str,
    pub min: f32,
    pub max: f32,
}

pub const AREA_ALL: AreaRange = AreaRange {
    name: "all",
    min: 0.0,
    max: 1e10,
};
pub const AREA_SMALL: AreaRange = AreaRange {
    name: "small",
    min: 0.0,
    max: 32.0 * 32.0,
};
pub const AREA_MEDIUM: AreaRange = AreaRange {
    name: "medium",
    min: 32.0 * 32.0,
    max: 96.0 * 96.0,
};
pub const AREA_LARGE: AreaRange = AreaRange {
    name: "large",
    min: 96.0 * 96.0,
    max: 1e10,
};

/// Ground truth and detections of one image (frame).
#[derive(Debug, Clone, Default)]
pub struct ImageEval {
    pub ground_truth: Vec<BBox>,
    pub detections: Vec<Detection>,
}

/// The standard COCO summary, for one class.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct DetectionSummary {
    /// mAP@[.5:.95], all sizes.
    pub map: f32,
    /// AP at IoU 0.5.
    pub ap50: f32,
    /// AP at IoU 0.75.
    pub ap75: f32,
    /// mAP@[.5:.95] restricted to small / medium / large objects (`None` if
    /// there is no ground truth of that size).
    pub ap_small: Option<f32>,
    pub ap_medium: Option<f32>,
    pub ap_large: Option<f32>,
    /// Maximum recall (all sizes), averaged over the IoU thresholds.
    pub recall: f32,
}

/// Per-detection outcome for one image, IoU threshold and area range.
struct ImageMatches {
    /// (score, matched, ignored) for each kept detection, best score first.
    detections: Vec<(f32, bool, bool)>,
    /// Number of ground-truth boxes that count (not ignored).
    num_gt: usize,
}

fn evaluate_image(
    image: &ImageEval,
    iou_threshold: f32,
    area: AreaRange,
    max_dets: usize,
) -> ImageMatches {
    let in_range = |b: &BBox| b.area() >= area.min && b.area() <= area.max;

    // Ground truth: counted boxes first, then ignored ones (stable order).
    let mut gt: Vec<(BBox, bool)> = image
        .ground_truth
        .iter()
        .map(|b| (*b, !in_range(b)))
        .collect();
    gt.sort_by_key(|&(_, ignored)| ignored);

    // Detections: best score first (stable), at most `max_dets`.
    let mut dets: Vec<Detection> = image.detections.clone();
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    dets.truncate(max_dets);

    let threshold = iou_threshold.min(1.0 - 1e-10);
    let mut gt_matched = vec![false; gt.len()];
    let mut out = Vec::with_capacity(dets.len());
    for det in &dets {
        let mut best_iou = threshold;
        let mut best: Option<usize> = None;
        for (g, &(gt_box, ignored)) in gt.iter().enumerate() {
            if gt_matched[g] {
                continue;
            }
            // Once a counted match exists, stop at the ignored part of the list.
            if let Some(m) = best
                && !gt[m].1
                && ignored
            {
                break;
            }
            let iou = det.bbox.iou(&gt_box);
            if iou < best_iou {
                continue;
            }
            best_iou = iou;
            best = Some(g);
        }
        let (matched, ignored) = match best {
            Some(g) => {
                gt_matched[g] = true;
                (true, gt[g].1)
            }
            None => (false, !in_range(&det.bbox)),
        };
        out.push((det.score, matched, ignored));
    }
    ImageMatches {
        detections: out,
        num_gt: gt.iter().filter(|&&(_, ignored)| !ignored).count(),
    }
}

/// Precision/recall over all images for one IoU threshold and area range.
/// Returns `(AP, max recall)`, or `None` if there is no ground truth to find.
fn accumulate(
    images: &[ImageEval],
    iou_threshold: f32,
    area: AreaRange,
    max_dets: usize,
) -> Option<(f32, f32)> {
    let per_image: Vec<ImageMatches> = images
        .iter()
        .map(|im| evaluate_image(im, iou_threshold, area, max_dets))
        .collect();
    let num_gt: usize = per_image.iter().map(|m| m.num_gt).sum();
    if num_gt == 0 {
        return None;
    }

    // All detections, ranked by score (stable across images, like mergesort).
    let mut ranked: Vec<(f32, bool, bool)> = per_image
        .iter()
        .flat_map(|m| m.detections.iter().copied())
        .collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));

    let (mut tp, mut fp) = (0.0_f64, 0.0_f64);
    let mut recall = Vec::new();
    let mut precision = Vec::new();
    for &(_, matched, ignored) in ranked.iter().filter(|d| !d.2) {
        debug_assert!(!ignored);
        if matched {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        recall.push(tp / num_gt as f64);
        precision.push(tp / (tp + fp + f64::EPSILON));
    }
    let max_recall = recall.last().copied().unwrap_or(0.0) as f32;

    // Make precision monotonically decreasing from the right.
    for i in (1..precision.len()).rev() {
        if precision[i] > precision[i - 1] {
            precision[i - 1] = precision[i];
        }
    }
    // Sample at 101 recall points: first index with recall >= r.
    let mut sum = 0.0_f64;
    for k in 0..=100 {
        let r = k as f64 / 100.0;
        let idx = recall.partition_point(|&rc| rc < r);
        if idx < precision.len() {
            sum += precision[idx];
        }
    }
    Some(((sum / 101.0) as f32, max_recall))
}

/// mAP@[.5:.95] for one area range, plus mean max-recall. `None` if no ground truth.
fn mean_over_thresholds(
    images: &[ImageEval],
    area: AreaRange,
    max_dets: usize,
) -> Option<(f32, f32)> {
    let results: Vec<(f32, f32)> = coco_iou_thresholds()
        .iter()
        .filter_map(|&t| accumulate(images, t, area, max_dets))
        .collect();
    if results.is_empty() {
        return None;
    }
    let n = results.len() as f32;
    Some((
        results.iter().map(|r| r.0).sum::<f32>() / n,
        results.iter().map(|r| r.1).sum::<f32>() / n,
    ))
}

/// The COCO summary (with COCO's default of 100 detections per image).
pub fn evaluate(images: &[ImageEval]) -> DetectionSummary {
    const MAX_DETS: usize = 100;
    let (map, recall) = mean_over_thresholds(images, AREA_ALL, MAX_DETS).unwrap_or((0.0, 0.0));
    let ap_at = |t: f32| accumulate(images, t, AREA_ALL, MAX_DETS).map_or(0.0, |r| r.0);
    let ap_of = |area| mean_over_thresholds(images, area, MAX_DETS).map(|r| r.0);
    DetectionSummary {
        map,
        ap50: ap_at(0.5),
        ap75: ap_at(0.75),
        ap_small: ap_of(AREA_SMALL),
        ap_medium: ap_of(AREA_MEDIUM),
        ap_large: ap_of(AREA_LARGE),
        recall,
    }
}

/// Mean number of false positives per image: detections with
/// `score >= min_score` that match no ground truth at `iou_threshold`.
pub fn false_positives_per_image(images: &[ImageEval], min_score: f32, iou_threshold: f32) -> f32 {
    if images.is_empty() {
        return 0.0;
    }
    let fps: usize = images
        .iter()
        .map(|im| {
            let kept = ImageEval {
                ground_truth: im.ground_truth.clone(),
                detections: im
                    .detections
                    .iter()
                    .copied()
                    .filter(|d| d.score >= min_score)
                    .collect(),
            };
            evaluate_image(&kept, iou_threshold, AREA_ALL, usize::MAX)
                .detections
                .iter()
                .filter(|&&(_, matched, _)| !matched)
                .count()
        })
        .sum();
    fps as f32 / images.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x: f32, y: f32, w: f32, h: f32, score: f32) -> Detection {
        Detection::new(BBox::new(x, y, w, h), score)
    }

    #[test]
    fn perfect_detections_score_one() {
        let b = BBox::new(10.0, 10.0, 20.0, 20.0);
        let images = vec![
            ImageEval {
                ground_truth: vec![b],
                detections: vec![Detection::new(b, 0.9)],
            };
            5
        ];
        let s = evaluate(&images);
        assert_eq!((s.map, s.ap50, s.recall), (1.0, 1.0, 1.0));
        assert_eq!(s.ap_small, Some(1.0)); // 20×20 = 400 px² ≤ 32²
        assert_eq!((s.ap_medium, s.ap_large), (None, None));
    }

    #[test]
    fn a_false_positive_ranked_first_halves_early_precision() {
        // Two images, one object each; a confident false positive in image 1.
        let gt = BBox::new(0.0, 0.0, 10.0, 10.0);
        let images = vec![
            ImageEval {
                ground_truth: vec![gt],
                detections: vec![det(50.0, 50.0, 10.0, 10.0, 0.99), Detection::new(gt, 0.8)],
            },
            ImageEval {
                ground_truth: vec![gt],
                detections: vec![Detection::new(gt, 0.7)],
            },
        ];
        // Ranked: FP, TP, TP → precision 0, 1/2, 2/3; recall 0, 1/2, 1.
        // Monotonic precision: 2/3 everywhere → AP = 2/3.
        let ap = accumulate(&images, 0.5, AREA_ALL, 100).unwrap().0;
        assert!((ap - 2.0 / 3.0).abs() < 1e-4, "{ap}");
        assert_eq!(false_positives_per_image(&images, 0.5, 0.5), 0.5);
    }

    #[test]
    fn loose_boxes_count_at_iou_50_but_not_at_75() {
        let gt = BBox::new(0.0, 0.0, 10.0, 10.0);
        let loose = det(2.0, 0.0, 10.0, 10.0, 0.9); // IoU 80/120 = 0.667
        let images = vec![ImageEval {
            ground_truth: vec![gt],
            detections: vec![loose],
        }];
        let s = evaluate(&images);
        assert_eq!(s.ap50, 1.0);
        assert_eq!(s.ap75, 0.0);
        assert!((s.map - 0.4).abs() < 1e-6, "{}", s.map); // 4 of 10 thresholds ≤ 0.667
    }

    #[test]
    fn missed_objects_cap_recall() {
        let gt = BBox::new(0.0, 0.0, 10.0, 10.0);
        let images = vec![
            ImageEval {
                ground_truth: vec![gt],
                detections: vec![Detection::new(gt, 0.9)],
            },
            ImageEval {
                ground_truth: vec![gt],
                detections: vec![],
            },
        ];
        let (ap, recall) = accumulate(&images, 0.5, AREA_ALL, 100).unwrap();
        assert_eq!(recall, 0.5);
        assert!((ap - 51.0 / 101.0).abs() < 1e-6, "{ap}"); // precision 1 up to recall 0.5
    }
}
