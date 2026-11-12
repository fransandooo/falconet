//! Anti-UAV410's official tracking metric: **state accuracy** (SA).
//!
//! A port of `eval()` in the toolkit's `Evaluation_for_SA.py`
//! (github.com/HwangBo94/Anti-UAV410, commit `8a8eb04`), checked against it by
//! `scripts/check_sa_parity.py`. Per frame:
//!
//! | ground truth | prediction | score |
//! |---|---|---|
//! | drone visible | a box | IoU(prediction, truth) |
//! | drone visible | "no target" | 0 |
//! | drone not visible | "no target" | 1 |
//! | drone not visible | a box | 0 |
//! | visible, degenerate box (w or h = 0) | anything | 0 |
//! | visible but no box at all | anything | skipped |
//!
//! SA of a sequence = mean over its scored frames; the benchmark score (mSA)
//! = mean of the per-sequence SAs.

use crate::antiuav410::FrameLabel;
use crate::bbox::BBox;

/// Score of one frame, or `None` if the official code skips it.
pub fn frame_score(label: &FrameLabel, prediction: Option<&BBox>) -> Option<f32> {
    match (label, prediction) {
        (FrameLabel::Present(truth), Some(pred)) => Some(pred.iou(truth)),
        (FrameLabel::Present(_), None) => Some(0.0),
        (FrameLabel::Absent, None) => Some(1.0),
        (FrameLabel::Absent, Some(_)) => Some(0.0),
        // IoU with a zero-area box is always 0, predicted box or not.
        (FrameLabel::Invalid(_), _) => Some(0.0),
        (FrameLabel::Unlabelled, _) => None,
    }
}

/// State accuracy of one sequence.
///
/// `predictions[i]` is the tracker's output for frame `i`: `Some(box)` or
/// `None` for "no target". Like the official code (which `zip`s the two
/// lists), extra frames on either side are ignored. Returns `None` if no frame
/// can be scored.
pub fn state_accuracy(labels: &[FrameLabel], predictions: &[Option<BBox>]) -> Option<f32> {
    let scores: Vec<f32> = labels
        .iter()
        .zip(predictions)
        .filter_map(|(label, pred)| frame_score(label, pred.as_ref()))
        .collect();
    (!scores.is_empty()).then(|| scores.iter().sum::<f32>() / scores.len() as f32)
}

/// Mean state accuracy over sequences (the benchmark's headline number).
/// Sequences with no scorable frame are left out.
pub fn mean_state_accuracy<'a, I>(per_sequence: I) -> Option<f32>
where
    I: IntoIterator<Item = (&'a [FrameLabel], &'a [Option<BBox>])>,
{
    let sas: Vec<f32> = per_sequence
        .into_iter()
        .filter_map(|(labels, preds)| state_accuracy(labels, preds))
        .collect();
    (!sas.is_empty()).then(|| sas.iter().sum::<f32>() / sas.len() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    const B: BBox = BBox {
        x: 10.0,
        y: 10.0,
        w: 4.0,
        h: 4.0,
    };

    #[test]
    fn perfect_tracker_scores_one() {
        let labels = [
            FrameLabel::Present(B),
            FrameLabel::Absent,
            FrameLabel::Present(B),
        ];
        let preds = [Some(B), None, Some(B)];
        assert_eq!(state_accuracy(&labels, &preds), Some(1.0));
    }

    #[test]
    fn every_case_of_the_table() {
        let half = BBox::new(12.0, 10.0, 4.0, 4.0); // IoU 1/3 with B
        let labels = [
            FrameLabel::Present(B),                              // box → IoU 1/3
            FrameLabel::Present(B),                              // no target → 0
            FrameLabel::Absent,                                  // no target → 1
            FrameLabel::Absent,                                  // box → 0
            FrameLabel::Unlabelled,                              // skipped
            FrameLabel::Invalid(BBox::new(18.0, 0.0, 1.0, 0.0)), // → 0
        ];
        let preds = [Some(half), None, None, Some(B), Some(B), Some(B)];
        let sa = state_accuracy(&labels, &preds).unwrap();
        assert!((sa - (1.0 / 3.0 + 1.0) / 5.0).abs() < 1e-6, "{sa}");
    }

    #[test]
    fn extra_frames_are_ignored_like_zip() {
        let labels = [FrameLabel::Absent, FrameLabel::Absent];
        assert_eq!(state_accuracy(&labels, &[None]), Some(1.0));
        assert_eq!(state_accuracy(&labels[..1], &[None, Some(B)]), Some(1.0));
        assert_eq!(state_accuracy(&[FrameLabel::Unlabelled], &[None]), None);
    }

    #[test]
    fn mean_over_sequences() {
        let a = [FrameLabel::Absent];
        let b = [FrameLabel::Present(B)];
        let (pa, pb) = ([None], [None]); // SA 1.0 and 0.0
        let msa = mean_state_accuracy([(&a[..], &pa[..]), (&b[..], &pb[..])]);
        assert_eq!(msa, Some(0.5));
    }
}
