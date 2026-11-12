//! # falconet-track
//!
//! A from-scratch implementation of **ByteTrack** (Zhang et al., "ByteTrack:
//! Multi-Object Tracking by Associating Every Detection Box", ECCV 2022),
//! following the official implementation (`yolox/tracker/byte_tracker.py`).
//!
//! The pieces, bottom-up:
//! - [`kalman`]: constant-velocity Kalman filter over `(cx, cy, aspect, height)`,
//! - [`assignment`]: optimal one-to-one matching (Hungarian algorithm) with a
//!   cost limit,
//! - [`matching`]: IoU cost matrices between tracks and detections,
//! - [`byte_tracker`]: the tracker itself, with its two-stage association
//!   (confident detections first, then low-score ones) and track life cycle.
//!
//! Everything here is plain Rust on the CPU: a tracker update costs
//! microseconds, far below the detector.

pub mod assignment;
pub mod byte_tracker;
pub mod kalman;
pub mod matching;

pub use byte_tracker::{ByteTracker, ByteTrackerConfig, Track, TrackState};
pub use falconet_core::{BBox, Detection};

/// Anything that turns per-frame detections into tracks with persistent IDs.
///
/// The product version of the trait from playground exercise 3b.
pub trait Tracker {
    /// Feeds the detections of one frame; returns the tracks to report for it.
    fn update(&mut self, detections: &[Detection]) -> Vec<Track>;
    /// Forgets all state, ready for a new video.
    fn reset(&mut self);
}
