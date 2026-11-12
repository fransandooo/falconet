//! The simplest possible "tracker": no memory at all.

use crate::{Detection, Track, Tracker};

/// Gives every detection a brand-new ID, every frame.
///
/// A deliberately bad baseline: it shows what tracking adds on top of
/// detection. `#[derive(Default)]` generates `NaiveTracker::default()`, which
/// sets every field to its type's default (`next_id = 0`).
#[derive(Debug, Default)]
pub struct NaiveTracker {
    next_id: u32, // private: only code in this module can touch it
}

// `impl Trait for Type` is where a type fulfils the trait's contract. The
// compiler checks that every required method is present with the right signature.
impl Tracker for NaiveTracker {
    fn name(&self) -> &str {
        "naive (new ID per detection)"
    }

    fn update(&mut self, detections: &[Detection]) -> Vec<Track> {
        detections
            .iter()
            .map(|detection| {
                let id = self.next_id;
                self.next_id += 1;
                Track {
                    id,
                    bbox: detection.bbox,
                }
            })
            .collect()
    }

    fn reset(&mut self) {
        // `*self` is the value behind the `&mut` reference: overwrite it whole.
        *self = Self::default();
    }
}
