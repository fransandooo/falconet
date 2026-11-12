//! A greedy IoU tracker: the core idea behind SORT and ByteTrack, without
//! the motion model and the optimal assignment.

use crate::{BBox, Detection, Track, Tracker};

/// Internal per-track state. Not `pub`: it is an implementation detail that
/// other crates can't see or depend on.
#[derive(Debug, Clone)]
struct ActiveTrack {
    id: u32,
    bbox: BBox,
    /// Consecutive frames without a matching detection.
    missed: u32,
}

/// Matches each detection to the previous-frame track it overlaps most.
///
/// Every frame:
/// 1. detections are visited from most to least confident,
/// 2. each one takes the not-yet-matched track with the highest IoU, if that
///    IoU is at least `iou_threshold`,
/// 3. unmatched detections start new tracks,
/// 4. tracks unmatched for more than `max_missed` frames are deleted.
///
/// Known weaknesses, all fixed by ByteTrack in phase 1:
/// - no motion prediction: a fast or briefly occluded drone gets a new ID,
/// - greedy matching is not globally optimal (Hungarian assignment is),
/// - low-confidence detections are treated like high-confidence ones.
#[derive(Debug)]
pub struct GreedyIouTracker {
    /// Built once in `new`; `name()` lends it out as a `&str`.
    name: String,
    iou_threshold: f32,
    max_missed: u32,
    tracks: Vec<ActiveTrack>,
    next_id: u32,
}

impl GreedyIouTracker {
    /// An *associated function* (no `self`): called as `GreedyIouTracker::new(..)`.
    /// Rust has no constructors; `new` is just a naming convention.
    pub fn new(iou_threshold: f32, max_missed: u32) -> Self {
        Self {
            name: format!("greedy IoU (iou>={iou_threshold}, max_missed={max_missed})"),
            iou_threshold,
            max_missed,
            tracks: Vec::new(),
            next_id: 0,
        }
    }
}

// `Default` written by hand, because the sensible defaults aren't zeros.
impl Default for GreedyIouTracker {
    fn default() -> Self {
        Self::new(0.3, 2)
    }
}

impl Tracker for GreedyIouTracker {
    // The returned `&str` borrows from `self.name`: it is valid as long as
    // the tracker is. (The naive tracker returns a `&'static str` literal,
    // which also fits this signature because it lives forever.)
    fn name(&self) -> &str {
        &self.name
    }

    fn update(&mut self, detections: &[Detection]) -> Vec<Track> {
        // Sort *indices* by descending score instead of sorting the slice:
        // the caller's data is borrowed immutably, so we can't reorder it.
        // `total_cmp` gives a total order on floats (NaN included).
        let mut order: Vec<usize> = (0..detections.len()).collect();
        order.sort_by(|&a, &b| detections[b].score.total_cmp(&detections[a].score));

        let mut matched = vec![false; self.tracks.len()];
        let mut new_tracks = Vec::new();
        let mut output = Vec::with_capacity(detections.len());

        for &index in &order {
            let detection = &detections[index];

            // The best unmatched track above the IoU threshold, if any.
            let best = self
                .tracks
                .iter()
                .enumerate()
                .filter(|&(t, _)| !matched[t])
                .map(|(t, track)| (t, track.bbox.iou(&detection.bbox)))
                .filter(|&(_, iou)| iou >= self.iou_threshold)
                .max_by(|a, b| a.1.total_cmp(&b.1));

            match best {
                Some((t, _)) => {
                    matched[t] = true;
                    let track = &mut self.tracks[t];
                    track.bbox = detection.bbox;
                    track.missed = 0;
                    output.push(Track {
                        id: track.id,
                        bbox: track.bbox,
                    });
                }
                None => {
                    let id = self.next_id;
                    self.next_id += 1;
                    // Pushed into `self.tracks` only after the loop, so that a
                    // new track can't be matched again in this same frame.
                    new_tracks.push(ActiveTrack {
                        id,
                        bbox: detection.bbox,
                        missed: 0,
                    });
                    output.push(Track {
                        id,
                        bbox: detection.bbox,
                    });
                }
            }
        }

        // Age the tracks that found no detection, then drop the stale ones.
        for (track, &was_matched) in self.tracks.iter_mut().zip(&matched) {
            if !was_matched {
                track.missed += 1;
            }
        }
        let max_missed = self.max_missed;
        self.tracks.retain(|track| track.missed <= max_missed);
        self.tracks.extend(new_tracks); // moves the new tracks in

        output
    }

    fn reset(&mut self) {
        self.tracks.clear();
        self.next_id = 0;
    }
}
