//! # Exercise 3b — traits and generics
//!
//! A [`Tracker`] trait with two implementations:
//! - [`NaiveTracker`]: no memory; every detection gets a new ID (a bad baseline),
//! - [`GreedyIouTracker`]: matches detections to the previous frame's tracks by
//!   IoU. It is the core idea of SORT/ByteTrack, without the motion model and
//!   the optimal assignment that phase 1 adds.
//!
//! [`evaluate`] works with *any* tracker, either through generics (static
//! dispatch) or through trait objects (`dyn Tracker`, dynamic dispatch).

pub mod greedy;
pub mod naive;
pub mod synthetic;

use std::collections::HashSet;

pub use greedy::GreedyIouTracker;
pub use naive::NaiveTracker;
pub use playground_annotations::BBox;

/// One detector output: a box and a confidence score in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    pub bbox: BBox,
    pub score: f32,
}

/// A tracked object in one frame. The tracker's job is to keep `id` the same
/// for the same physical object across frames.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Track {
    pub id: u32,
    pub bbox: BBox,
}

/// Anything that turns per-frame detections into tracks with persistent IDs.
///
/// A *trait* is a set of methods that a type promises to implement: similar
/// to an interface in Java or Go, or to a C struct of function pointers that
/// each implementation fills in. ByteTrack (phase 1) will simply be one more
/// implementation of a trait like this one.
pub trait Tracker {
    /// Human-readable name, for reports.
    fn name(&self) -> &str;

    /// Feeds the detections of one frame; returns the tracks seen in that frame.
    fn update(&mut self, detections: &[Detection]) -> Vec<Track>;

    /// Forgets all state, ready for a new video.
    fn reset(&mut self);

    /// Runs the tracker over a whole sequence (one `Vec` of detections per frame).
    ///
    /// This is a *default method*: every implementation gets it for free, built
    /// on top of the required methods `reset` and `update`, and may override it.
    fn run(&mut self, frames: &[Vec<Detection>]) -> Vec<Vec<Track>> {
        self.reset();
        frames
            .iter()
            .map(|detections| self.update(detections))
            .collect()
    }
}

/// Summary of one tracker run.
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    pub tracker: String,
    /// Number of different IDs handed out. Lower is better, down to the
    /// number of real objects (more means the tracker "forgot" objects).
    pub distinct_ids: usize,
    /// Total number of (frame, track) boxes produced.
    pub track_boxes: usize,
}

/// Runs any tracker on a sequence and summarises the result.
///
/// `T: Tracker` is a *trait bound*: this function accepts every type that
/// implements `Tracker`. The compiler generates one specialised copy per
/// concrete type (*monomorphization*, like C++ templates), so the calls to
/// `update` are direct and can be inlined: this is *static dispatch*.
///
/// `?Sized` relaxes the implicit "size known at compile time" requirement, so
/// `T` may also be `dyn Tracker` (a trait object). That lets the same function
/// work with `&mut dyn Tracker` too.
pub fn evaluate<T: Tracker + ?Sized>(tracker: &mut T, frames: &[Vec<Detection>]) -> Report {
    let history = tracker.run(frames);
    // `flatten` turns the per-frame `Vec<Vec<Track>>` into one stream of tracks.
    let ids: HashSet<u32> = history.iter().flatten().map(|track| track.id).collect();
    Report {
        tracker: tracker.name().to_owned(),
        distinct_ids: ids.len(),
        track_boxes: history.iter().map(Vec::len).sum(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::{Trajectory, generate};

    fn one_slow_drone(n_frames: usize, hidden: impl Fn(usize) -> bool) -> Vec<Vec<Detection>> {
        let trajectory = Trajectory {
            start: BBox::new(100.0, 100.0, 8.0, 6.0),
            velocity: (0.5, 0.0),
        };
        generate(&[trajectory], n_frames, |_, frame| hidden(frame))
    }

    #[test]
    fn naive_tracker_gives_every_detection_a_new_id() {
        let frames = one_slow_drone(10, |_| false);
        let report = evaluate(&mut NaiveTracker::default(), &frames);
        assert_eq!(report.distinct_ids, 10);
    }

    #[test]
    fn greedy_tracker_keeps_the_id_of_a_slow_drone() {
        let frames = one_slow_drone(30, |_| false);
        let report = evaluate(&mut GreedyIouTracker::default(), &frames);
        assert_eq!(report.distinct_ids, 1);
        assert_eq!(report.track_boxes, 30);
    }

    #[test]
    fn greedy_tracker_survives_a_short_occlusion() {
        let frames = one_slow_drone(20, |frame| frame == 5);
        let report = evaluate(&mut GreedyIouTracker::default(), &frames);
        assert_eq!(report.distinct_ids, 1);
    }

    #[test]
    fn greedy_tracker_forgets_after_max_missed_frames() {
        // Hidden for 4 frames, more than `max_missed = 2`: the track is deleted.
        let frames = one_slow_drone(20, |frame| (5..9).contains(&frame));
        let report = evaluate(&mut GreedyIouTracker::new(0.3, 2), &frames);
        assert_eq!(report.distinct_ids, 2);
    }

    #[test]
    fn run_resets_state_between_sequences() {
        let frames = one_slow_drone(5, |_| false);
        let mut tracker = NaiveTracker::default();
        let first = tracker.run(&frames);
        let second = tracker.run(&frames);
        assert_eq!(first, second);
    }

    #[test]
    fn demo_scene_exposes_the_missing_motion_model() {
        // Drone B moves fast and is occluded for 2 frames; without predicting
        // its motion the greedy tracker can't re-associate it and opens a new ID.
        // Expected: A (1) + B before/after occlusion (2) + false positive (1).
        let frames = synthetic::two_drones_scene();
        let report = evaluate(&mut GreedyIouTracker::default(), &frames);
        assert_eq!(report.distinct_ids, 4);
    }

    #[test]
    fn trackers_work_as_trait_objects() {
        let frames = one_slow_drone(10, |_| false);
        let mut trackers: Vec<Box<dyn Tracker>> = vec![
            Box::new(NaiveTracker::default()),
            Box::new(GreedyIouTracker::default()),
        ];
        let ids: Vec<usize> = trackers
            .iter_mut()
            .map(|tracker| evaluate(tracker.as_mut(), &frames).distinct_ids)
            .collect();
        assert_eq!(ids, [10, 1]);
    }
}
