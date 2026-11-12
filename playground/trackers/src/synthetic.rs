//! Deterministic synthetic detections for testing trackers.

use crate::{BBox, Detection};

/// An object moving in a straight line at constant velocity (px/frame).
#[derive(Debug, Clone, Copy)]
pub struct Trajectory {
    pub start: BBox,
    pub velocity: (f32, f32),
}

impl Trajectory {
    /// The object's box at a given frame.
    pub fn at(&self, frame: usize) -> BBox {
        let t = frame as f32;
        self.start
            .translated(self.velocity.0 * t, self.velocity.1 * t)
    }
}

/// Builds `n_frames` frames of perfect detections (score 0.9) for the given
/// trajectories. `hidden(object_index, frame)` returns `true` when the
/// detector misses that object in that frame (occlusion, low contrast...).
///
/// `impl Fn(usize, usize) -> bool` accepts any closure with that signature.
/// It is shorthand for a generic parameter `F: Fn(usize, usize) -> bool`.
pub fn generate(
    trajectories: &[Trajectory],
    n_frames: usize,
    hidden: impl Fn(usize, usize) -> bool,
) -> Vec<Vec<Detection>> {
    (0..n_frames)
        .map(|frame| {
            trajectories
                .iter()
                .enumerate()
                .filter(|&(object, _)| !hidden(object, frame))
                .map(|(_, trajectory)| Detection {
                    bbox: trajectory.at(frame),
                    score: 0.9,
                })
                .collect()
        })
        .collect()
}

/// The demo scene (30 frames):
/// - drone A: small (8x6 px), moving right at 2 px/frame, always detected,
/// - drone B: 10x10 px, moving diagonally at 1.5 px/frame on each axis,
///   missed by the detector in frames 10 and 11,
/// - one low-confidence false positive in frame 15.
pub fn two_drones_scene() -> Vec<Vec<Detection>> {
    let drones = [
        Trajectory {
            start: BBox::new(100.0, 100.0, 8.0, 6.0),
            velocity: (2.0, 0.0),
        },
        Trajectory {
            start: BBox::new(300.0, 200.0, 10.0, 10.0),
            velocity: (-1.5, 1.5),
        },
    ];
    let mut frames = generate(&drones, 30, |object, frame| {
        object == 1 && (10..12).contains(&frame)
    });
    frames[15].push(Detection {
        bbox: BBox::new(500.0, 400.0, 6.0, 6.0),
        score: 0.4,
    });
    frames
}
