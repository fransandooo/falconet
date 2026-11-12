//! Cost matrices between tracks and detections.

use falconet_core::{BBox, Detection};

/// `1 - IoU` for every (track, detection) pair, row-major `tracks × detections`.
///
/// Deviation from the official ByteTrack: it computes IoU with
/// `cython_bbox`, which uses the legacy "+1 pixel" area convention
/// (`(x2 - x1 + 1) * (y2 - y1 + 1)`). That inflates the overlap of tiny boxes
/// noticeably, e.g. two 4×4 boxes shifted by 2 px get IoU 0.43 instead of 0.33,
/// and our drones are often that small. We use the continuous IoU, the same
/// as the Anti-UAV410 metric.
pub fn iou_distance(tracks: &[BBox], detections: &[BBox]) -> Vec<f32> {
    tracks
        .iter()
        .flat_map(|t| detections.iter().map(move |d| 1.0 - t.iou(d)))
        .collect()
}

/// ByteTrack's `fuse_score`: similarity = IoU × detection score, so a
/// confident detection wins a tie against a doubtful one. Row-major in place.
pub fn fuse_score(distance: &mut [f32], detections: &[Detection]) {
    if detections.is_empty() {
        return;
    }
    for (k, d) in distance.iter_mut().enumerate() {
        let score = detections[k % detections.len()].score;
        *d = 1.0 - (1.0 - *d) * score;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_matrix_layout_and_values() {
        let a = BBox::new(0.0, 0.0, 2.0, 2.0);
        let b = BBox::new(1.0, 0.0, 2.0, 2.0);
        let far = BBox::new(10.0, 10.0, 2.0, 2.0);
        let d = iou_distance(&[a, far], &[a, b]);
        assert_eq!(d.len(), 4);
        assert_eq!(d[0], 0.0); // a vs a
        assert!((d[1] - 2.0 / 3.0).abs() < 1e-6); // a vs b: IoU 1/3
        assert_eq!((d[2], d[3]), (1.0, 1.0)); // far vs anything
    }

    #[test]
    fn fusing_scores_penalises_doubtful_detections() {
        let b = BBox::new(0.0, 0.0, 2.0, 2.0);
        let dets = [Detection::new(b, 1.0), Detection::new(b, 0.5)];
        let mut d = vec![0.0, 0.0]; // perfect overlap with both
        fuse_score(&mut d, &dets);
        assert_eq!(d, vec![0.0, 0.5]);
    }
}
