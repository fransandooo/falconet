//! Cross-check against `trackforge`, an independent Rust ByteTrack.
//!
//! The two implementations differ by design in a few places: trackforge uses
//! `score >= thresh`, has no 0.1 low cutoff, no score fusion, and confirms new
//! tracks immediately. So the scenes below avoid those cases: every object is
//! present from frame 1, scores are high, and occlusions are shorter than the
//! track buffer. On such scenes the Kalman filter, the assignment and the
//! lost/re-found logic must agree frame by frame.
//!
//! Remaining differences, handled explicitly: trackforge reports a new track
//! on the frame it is born (ByteTrack confirms it one frame later), and after
//! a match it reports the *raw detection*
//! box, while the official ByteTrack (and we) report the Kalman-filtered box.
//! Association uses the Kalman prediction in both, so identities must match
//! exactly; boxes are only required to be close.

use std::collections::HashMap;

use falconet_track::{BBox, ByteTracker, ByteTrackerConfig, Detection, Tracker};
use trackforge::trackers::byte_track::{ByteTrack, ByteTrackParams};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next()
    }
}

/// One random scene: per frame, the list of detections.
fn scene(rng: &mut Lcg, frames: usize) -> Vec<Vec<Detection>> {
    let n = 1 + (rng.next() * 4.0) as usize;
    struct Obj {
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        vx: f32,
        vy: f32,
        hidden: std::ops::Range<usize>,
    }
    let objects: Vec<Obj> = (0..n)
        .map(|k| {
            let start = 5 + (rng.next() * 60.0) as usize;
            let len = (rng.next() * 12.0) as usize;
            Obj {
                // Spread objects apart so association has a unique optimum.
                x: 60.0 + 140.0 * k as f32 + rng.range(-10.0, 10.0),
                y: rng.range(60.0, 400.0),
                w: rng.range(8.0, 40.0),
                h: rng.range(6.0, 30.0),
                vx: rng.range(-3.0, 3.0),
                vy: rng.range(-3.0, 3.0),
                hidden: start..start + len,
            }
        })
        .collect();
    (0..frames)
        .map(|f| {
            objects
                .iter()
                .filter(|o| !o.hidden.contains(&f))
                .map(|o| {
                    let t = f as f32;
                    let jitter = |r: &mut Lcg| r.range(-1.0, 1.0);
                    let b = BBox::new(
                        o.x + o.vx * t + jitter(rng),
                        o.y + o.vy * t + jitter(rng),
                        o.w + jitter(rng) * 0.5,
                        o.h + jitter(rng) * 0.5,
                    );
                    Detection::new(b, rng.range(0.7, 1.0))
                })
                .collect()
        })
        .collect()
}

#[test]
fn agrees_with_trackforge_on_random_scenes() {
    let mut rng = Lcg(2026);
    let (mut frames_checked, mut tracks_checked, mut worst_iou) = (0, 0, 1.0_f32);
    let mut iou_sum = 0.0_f64;
    let mut skipped_newborn = 0;
    let mut worst_shift = 0.0_f32;

    for scene_index in 0..200 {
        let frames = scene(&mut rng, 100);
        let mut ours = ByteTracker::new(ByteTrackerConfig {
            fuse_score: false, // trackforge's first stage uses plain IoU
            ..Default::default()
        });
        let mut theirs = ByteTrack::from_params(ByteTrackParams::default());
        // our id → their id, must stay consistent over the whole scene.
        let mut id_map: HashMap<u64, u64> = HashMap::new();

        for (f, dets) in frames.iter().enumerate() {
            let a = ours.update(dets);
            let input = dets
                .iter()
                .map(|d| ([d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h], d.score, 0))
                .collect();
            // trackforge reports a track the frame it is born; ByteTrack waits one
            // frame to confirm it (except on frame 1). Hide that difference.
            let frame_id = f + 1;
            let (b, born_now): (Vec<_>, Vec<_>) = theirs
                .update(input)
                .into_iter()
                .partition(|t| frame_id == 1 || t.start_frame < frame_id);
            skipped_newborn += born_now.len();
            assert_eq!(
                a.len(),
                b.len(),
                "scene {scene_index}, frame {f}: {} vs {} tracks",
                a.len(),
                b.len()
            );
            for track in &a {
                let (best, iou) = b
                    .iter()
                    .map(|t| {
                        let [x, y, w, h] = t.tlwh;
                        (t.track_id, track.bbox.iou(&BBox::new(x, y, w, h)))
                    })
                    .max_by(|p, q| p.1.total_cmp(&q.1))
                    .expect("same number of tracks, so not empty");
                worst_iou = worst_iou.min(iou);
                // IoU is a poor closeness test for 5-px-tall boxes; compare centres.
                let other = b.iter().find(|t| t.track_id == best).expect("best exists");
                let (cx, cy) = track.bbox.center();
                let [x, y, w, h] = other.tlwh;
                let shift = (cx - (x + w / 2.0)).hypot(cy - (y + h / 2.0));
                worst_shift = worst_shift.max(shift);
                assert!(
                    shift <= 3.0,
                    "scene {scene_index}, frame {f}: centres {shift:.2} px apart"
                );
                iou_sum += f64::from(iou);
                let mapped = *id_map.entry(track.id).or_insert(best);
                assert_eq!(
                    mapped, best,
                    "scene {scene_index}, frame {f}: identity mismatch"
                );
                tracks_checked += 1;
            }
            frames_checked += 1;
        }
    }
    println!(
        "{frames_checked} frames, {tracks_checked} track boxes: identities identical; \
         filtered-vs-raw box IoU mean {:.4}, worst {worst_iou:.4}, worst centre shift {worst_shift:.2} px; \
         {skipped_newborn} newborn trackforge tracks ignored",
        iou_sum / tracks_checked as f64
    );
}
