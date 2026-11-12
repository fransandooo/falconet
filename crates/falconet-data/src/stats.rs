//! Dataset statistics: what makes Anti-UAV410 hard, in numbers.

use serde::Serialize;

use crate::antiuav410::{Attribute, FrameLabel, SequenceAnnotation, Split};

/// Distribution summary of a set of values.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Default)]
pub struct Summary {
    pub count: usize,
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub p05: f32,
    pub p25: f32,
    pub median: f32,
    pub p75: f32,
    pub p95: f32,
    pub max: f32,
}

impl Summary {
    /// Summarises `values` (sorted in place). Empty input gives all zeros.
    pub fn of(values: &mut [f32]) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        values.sort_by(f32::total_cmp);
        let n = values.len();
        let mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
        let var = values
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / n as f64;
        // Nearest-rank percentile on the sorted values.
        let pct = |p: f64| values[((p * (n - 1) as f64).round() as usize).min(n - 1)];
        Self {
            count: n,
            mean: mean as f32,
            std: var.sqrt() as f32,
            min: values[0],
            p05: pct(0.05),
            p25: pct(0.25),
            median: pct(0.5),
            p75: pct(0.75),
            p95: pct(0.95),
            max: values[n - 1],
        }
    }
}

/// Box-size bins by diagonal length (px), matching the size scale used by
/// Anti-UAV410's tiny/small/medium/normal attributes.
pub const SIZE_BIN_EDGES: [f32; 3] = [10.0, 30.0, 50.0];
pub const SIZE_BIN_NAMES: [&str; 4] = ["diag<=10", "10<diag<=30", "30<diag<=50", "diag>50"];

/// COCO's "small object" threshold (area < 32²), used by the AP_small metric.
pub const COCO_SMALL_AREA: f32 = 32.0 * 32.0;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SplitStats {
    /// `None` = all splits together.
    pub split: Option<Split>,
    pub sequences: usize,
    pub frames: usize,
    pub present_frames: usize,
    pub absent_frames: usize,
    pub unlabelled_frames: usize,
    pub sequence_length: Summary,
    /// Fraction of frames without a visible drone, per sequence.
    pub absent_fraction_per_sequence: Summary,
    /// Lengths (in frames) of runs of consecutive absent frames.
    pub absence_run_length: Summary,
    pub box_width: Summary,
    pub box_height: Summary,
    pub box_area: Summary,
    pub box_diagonal: Summary,
    pub aspect_ratio: Summary,
    pub size_bins: [usize; 4],
    pub coco_small_boxes: usize,
    /// Centre displacement between consecutive visible frames, px/frame.
    pub center_motion_px: Summary,
    /// The same displacement divided by the box diagonal.
    pub center_motion_rel: Summary,
    /// IoU between the boxes of consecutive visible frames.
    pub consecutive_iou: Summary,
    /// Fraction of consecutive visible pairs whose boxes overlap with IoU < 0.3,
    /// a typical IoU-matching threshold for trackers.
    pub consecutive_iou_below_0_3: f32,
    /// Number of sequences that have each attribute, in toolkit order.
    pub attribute_counts: Vec<(String, usize)>,
}

/// Computes statistics over the sequences of `split` (`None` = all of them).
pub fn split_stats(sequences: &[SequenceAnnotation], split: Option<Split>) -> SplitStats {
    let selected: Vec<&SequenceAnnotation> = sequences
        .iter()
        .filter(|s| split.is_none_or(|sp| s.split == sp))
        .collect();

    let (mut widths, mut heights, mut areas, mut diagonals, mut aspects) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut motion_px, mut motion_rel, mut consecutive_iou) = (Vec::new(), Vec::new(), Vec::new());
    let (mut lengths, mut absent_fractions, mut absence_runs) =
        (Vec::new(), Vec::new(), Vec::new());
    let (mut present, mut absent, mut unlabelled) = (0, 0, 0);
    let mut size_bins = [0_usize; 4];
    let mut coco_small = 0;

    for seq in &selected {
        lengths.push(seq.frames.len() as f32);
        let mut seq_absent = 0;
        let mut run = 0;
        // The previous frame's box, if that frame had one.
        let mut previous = None;

        for label in &seq.frames {
            match label {
                FrameLabel::Present(b) => {
                    present += 1;
                    widths.push(b.w);
                    heights.push(b.h);
                    areas.push(b.area());
                    diagonals.push(b.diagonal());
                    aspects.push(b.w / b.h);
                    let bin = SIZE_BIN_EDGES.iter().filter(|&&e| b.diagonal() > e).count();
                    size_bins[bin] += 1;
                    if b.area() < COCO_SMALL_AREA {
                        coco_small += 1;
                    }
                    if let Some(prev) = previous {
                        let (px, py) = b.center();
                        let (qx, qy) = crate::BBox::center(&prev);
                        let d = (px - qx).hypot(py - qy);
                        motion_px.push(d);
                        motion_rel.push(d / b.diagonal());
                        consecutive_iou.push(b.iou(&prev));
                    }
                    if run > 0 {
                        absence_runs.push(run as f32);
                        run = 0;
                    }
                    previous = Some(*b);
                }
                FrameLabel::Absent => {
                    absent += 1;
                    seq_absent += 1;
                    run += 1;
                    previous = None;
                }
                FrameLabel::Unlabelled => {
                    unlabelled += 1;
                    previous = None;
                }
            }
        }
        if run > 0 {
            absence_runs.push(run as f32);
        }
        absent_fractions.push(seq_absent as f32 / seq.frames.len().max(1) as f32);
    }

    let iou_below = consecutive_iou.iter().filter(|&&v| v < 0.3).count() as f32
        / consecutive_iou.len().max(1) as f32;
    let attribute_counts = Attribute::ALL
        .iter()
        .map(|&a| {
            let count = selected.iter().filter(|s| s.attributes.contains(a)).count();
            (a.short().to_owned(), count)
        })
        .collect();

    SplitStats {
        split,
        sequences: selected.len(),
        frames: present + absent + unlabelled,
        present_frames: present,
        absent_frames: absent,
        unlabelled_frames: unlabelled,
        sequence_length: Summary::of(&mut lengths),
        absent_fraction_per_sequence: Summary::of(&mut absent_fractions),
        absence_run_length: Summary::of(&mut absence_runs),
        box_width: Summary::of(&mut widths),
        box_height: Summary::of(&mut heights),
        box_area: Summary::of(&mut areas),
        box_diagonal: Summary::of(&mut diagonals),
        aspect_ratio: Summary::of(&mut aspects),
        size_bins,
        coco_small_boxes: coco_small,
        center_motion_px: Summary::of(&mut motion_px),
        center_motion_rel: Summary::of(&mut motion_rel),
        consecutive_iou: Summary::of(&mut consecutive_iou),
        consecutive_iou_below_0_3: iou_below,
        attribute_counts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BBox;
    use crate::antiuav410::Attributes;

    #[test]
    fn summary_of_known_values() {
        let s = Summary::of(&mut [4.0, 1.0, 3.0, 2.0, 5.0]);
        assert_eq!((s.count, s.min, s.median, s.max), (5, 1.0, 3.0, 5.0));
        assert_eq!(s.mean, 3.0);
        assert!((s.std - 2.0_f32.sqrt()).abs() < 1e-6);
        assert_eq!(Summary::of(&mut []), Summary::default());
    }

    #[test]
    fn stats_of_a_tiny_synthetic_dataset() {
        let p = |x: f32| FrameLabel::Present(BBox::new(x, 0.0, 6.0, 8.0)); // diagonal 10
        let seq = SequenceAnnotation {
            name: "s".into(),
            split: Split::Train,
            frames: vec![
                p(0.0),
                p(3.0),
                FrameLabel::Absent,
                FrameLabel::Absent,
                p(100.0),
            ],
            attributes: Attributes::default(),
        };
        let s = split_stats(&[seq], Some(Split::Train));
        assert_eq!((s.frames, s.present_frames, s.absent_frames), (5, 3, 2));
        assert_eq!(s.size_bins, [3, 0, 0, 0]); // diagonal exactly 10 → first bin
        assert_eq!(s.coco_small_boxes, 3);
        assert_eq!(s.absence_run_length.max, 2.0);
        // Only (frame 0 → 1) is a consecutive visible pair: shift 3 px on a 6-px-wide box.
        assert_eq!(s.center_motion_px.count, 1);
        assert_eq!(s.center_motion_px.max, 3.0);
        assert!((s.consecutive_iou.max - 24.0 / 72.0).abs() < 1e-6);
        assert_eq!(s.absent_fraction_per_sequence.max, 0.4);
    }

    #[test]
    fn other_splits_are_filtered_out() {
        let seq = SequenceAnnotation {
            name: "v".into(),
            split: Split::Val,
            frames: vec![FrameLabel::Absent],
            attributes: Attributes::default(),
        };
        // `from_ref` views one value as a 1-element slice, without cloning it.
        assert_eq!(
            split_stats(std::slice::from_ref(&seq), Some(Split::Train)).sequences,
            0
        );
        assert_eq!(split_stats(&[seq], None).sequences, 1);
    }
}
