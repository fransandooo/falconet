//! The ByteTrack tracker.
//!
//! Follows `BYTETracker.update` from the official implementation step by step
//! (`yolox/tracker/byte_tracker.py`), with Rust ownership in place of Python's
//! shared object lists. Per frame:
//!
//! 1. Split detections into **high** (`score > track_thresh`) and **low**
//!    (`0.1 < score < track_thresh`) confidence.
//! 2. Predict every confirmed and lost track forward with its Kalman filter.
//! 3. **First association**: tracks ↔ high detections, by IoU (× score).
//! 4. **Second association**: still-unmatched *tracked* tracks ↔ low
//!    detections. This is ByteTrack's key idea: a blurred or half-occluded
//!    drone produces a low-score box, and instead of discarding it, it keeps
//!    an existing track alive. Low boxes never start new tracks, so they can't
//!    add false positives.
//! 5. Unconfirmed tracks (born last frame) ↔ remaining high detections;
//!    unmatched ones are dropped, which filters one-frame false positives.
//! 6. Remaining high detections (`score ≥ track_thresh + 0.1`) start new tracks.
//! 7. Tracks lost for more than `track_buffer` frames are removed.

use falconet_core::{BBox, Detection};

use crate::Tracker;
use crate::assignment::linear_assignment;
use crate::kalman::{KalmanFilter, KalmanState};
use crate::matching::{fuse_score, iou_distance};

/// Tuning knobs, with the official defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ByteTrackerConfig {
    /// Detections above this score are "high"; between 0.1 and this, "low".
    pub track_thresh: f32,
    /// Maximum `1 - IoU×score` cost for the first association.
    pub match_thresh: f32,
    /// Frames a lost track is kept before being removed (at 30 fps).
    pub track_buffer: u32,
    pub frame_rate: u32,
    /// Multiply IoU by detection score in the first association (official
    /// default; disabled for crowded scenes in their MOT20 setting).
    pub fuse_score: bool,
}

impl Default for ByteTrackerConfig {
    fn default() -> Self {
        Self {
            track_thresh: 0.5,
            match_thresh: 0.8,
            track_buffer: 30,
            frame_rate: 30,
            fuse_score: true,
        }
    }
}

/// Life-cycle state of a track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackState {
    Tracked,
    Lost,
    Removed,
}

/// A track as reported to the caller (a snapshot, not the internal state).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Track {
    pub id: u64,
    /// The Kalman-filtered box (smoother than the raw detection).
    pub bbox: BBox,
    /// Score of the last detection associated with this track.
    pub score: f32,
    /// Frame (1-based) on which the track started.
    pub start_frame: u64,
    /// Consecutive frames this track has been matched since (re)activation.
    pub tracklet_len: u32,
}

/// Internal track state (ByteTrack's `STrack`).
#[derive(Debug, Clone)]
struct STrack {
    id: u64,
    kalman: KalmanState,
    score: f32,
    state: TrackState,
    /// Confirmed: matched on at least two consecutive frames (or born on frame 1).
    is_activated: bool,
    /// Last frame this track was matched.
    frame_id: u64,
    start_frame: u64,
    tracklet_len: u32,
}

impl STrack {
    fn bbox(&self) -> BBox {
        BBox::from_xyah(self.kalman.xyah())
    }

    fn snapshot(&self) -> Track {
        Track {
            id: self.id,
            bbox: self.bbox(),
            score: self.score,
            start_frame: self.start_frame,
            tracklet_len: self.tracklet_len,
        }
    }

    /// Matched again while being tracked.
    fn update(&mut self, kf: &KalmanFilter, det: &Detection, frame: u64) {
        self.frame_id = frame;
        self.tracklet_len += 1;
        self.kalman = kf.update(&self.kalman, det.bbox.to_xyah());
        self.state = TrackState::Tracked;
        self.is_activated = true;
        self.score = det.score;
    }

    /// Matched after having been lost.
    fn re_activate(&mut self, kf: &KalmanFilter, det: &Detection, frame: u64) {
        self.kalman = kf.update(&self.kalman, det.bbox.to_xyah());
        self.tracklet_len = 0;
        self.state = TrackState::Tracked;
        self.is_activated = true;
        self.frame_id = frame;
        self.score = det.score;
    }

    /// `update` or `re_activate` depending on the current state.
    fn absorb(&mut self, kf: &KalmanFilter, det: &Detection, frame: u64) {
        if self.state == TrackState::Tracked {
            self.update(kf, det, frame);
        } else {
            self.re_activate(kf, det, frame);
        }
    }
}

/// Where a track in the association pool lives: an index into the tracked or
/// the lost list. Indices instead of references let us mutate tracks during
/// association without fighting the borrow checker over two lists.
#[derive(Debug, Clone, Copy)]
enum Slot {
    Tracked(usize),
    Lost(usize),
}

/// The ByteTrack multi-object tracker.
#[derive(Debug, Clone)]
pub struct ByteTracker {
    config: ByteTrackerConfig,
    kf: KalmanFilter,
    tracked: Vec<STrack>,
    lost: Vec<STrack>,
    frame_id: u64,
    next_id: u64,
}

impl ByteTracker {
    pub fn new(config: ByteTrackerConfig) -> Self {
        Self {
            config,
            kf: KalmanFilter::default(),
            tracked: Vec::new(),
            lost: Vec::new(),
            frame_id: 0,
            next_id: 1,
        }
    }

    pub fn config(&self) -> &ByteTrackerConfig {
        &self.config
    }

    /// Frames a lost track survives, scaled by frame rate like the original.
    fn max_time_lost(&self) -> u64 {
        (f64::from(self.config.frame_rate) / 30.0 * f64::from(self.config.track_buffer)) as u64
    }

    fn slot_mut(&mut self, slot: Slot) -> &mut STrack {
        match slot {
            Slot::Tracked(i) => &mut self.tracked[i],
            Slot::Lost(i) => &mut self.lost[i],
        }
    }

    fn slot(&self, slot: Slot) -> &STrack {
        match slot {
            Slot::Tracked(i) => &self.tracked[i],
            Slot::Lost(i) => &self.lost[i],
        }
    }

    /// Runs one frame of tracking and returns the confirmed tracks.
    pub fn step(&mut self, detections: &[Detection]) -> Vec<Track> {
        self.frame_id += 1;
        let frame = self.frame_id;
        let thresh = self.config.track_thresh;

        // 1. High / low split (strict comparisons, as in the original: a score
        //    exactly equal to `track_thresh` falls in neither group).
        let high: Vec<Detection> = detections
            .iter()
            .copied()
            .filter(|d| d.score > thresh)
            .collect();
        let low: Vec<Detection> = detections
            .iter()
            .copied()
            .filter(|d| d.score > 0.1 && d.score < thresh)
            .collect();

        // 2. Pool = confirmed tracked + lost; unconfirmed are handled in step 5.
        let unconfirmed: Vec<usize> = (0..self.tracked.len())
            .filter(|&i| !self.tracked[i].is_activated)
            .collect();
        let pool: Vec<Slot> = (0..self.tracked.len())
            .filter(|&i| self.tracked[i].is_activated)
            .map(Slot::Tracked)
            .chain((0..self.lost.len()).map(Slot::Lost))
            .collect();
        for &slot in &pool {
            let kf = self.kf.clone();
            let track = self.slot_mut(slot);
            // Lost tracks don't keep growing/shrinking: zero the height velocity.
            if track.state != TrackState::Tracked {
                track.kalman.mean[7] = 0.0;
            }
            track.kalman = kf.predict(&track.kalman);
        }

        // 3. First association: pool ↔ high detections.
        let pool_boxes: Vec<BBox> = pool.iter().map(|&s| self.slot(s).bbox()).collect();
        let high_boxes: Vec<BBox> = high.iter().map(|d| d.bbox).collect();
        let mut cost = iou_distance(&pool_boxes, &high_boxes);
        if self.config.fuse_score {
            fuse_score(&mut cost, &high);
        }
        let first = linear_assignment(&cost, pool.len(), high.len(), self.config.match_thresh);
        for &(p, d) in &first.matches {
            let kf = self.kf.clone();
            self.slot_mut(pool[p]).absorb(&kf, &high[d], frame);
        }

        // 4. Second association: unmatched *tracked* tracks ↔ low detections.
        let remaining: Vec<Slot> = first
            .unmatched_rows
            .iter()
            .map(|&p| pool[p])
            .filter(|&s| self.slot(s).state == TrackState::Tracked)
            .collect();
        let remaining_boxes: Vec<BBox> = remaining.iter().map(|&s| self.slot(s).bbox()).collect();
        let low_boxes: Vec<BBox> = low.iter().map(|d| d.bbox).collect();
        let cost = iou_distance(&remaining_boxes, &low_boxes);
        let second = linear_assignment(&cost, remaining.len(), low.len(), 0.5);
        for &(r, d) in &second.matches {
            let kf = self.kf.clone();
            self.slot_mut(remaining[r]).absorb(&kf, &low[d], frame);
        }
        for &r in &second.unmatched_rows {
            self.slot_mut(remaining[r]).state = TrackState::Lost;
        }

        // 5. Unconfirmed tracks ↔ high detections left over from step 3.
        let left: Vec<Detection> = first.unmatched_cols.iter().map(|&d| high[d]).collect();
        let unconfirmed_boxes: Vec<BBox> = unconfirmed
            .iter()
            .map(|&i| self.tracked[i].bbox())
            .collect();
        let left_boxes: Vec<BBox> = left.iter().map(|d| d.bbox).collect();
        let mut cost = iou_distance(&unconfirmed_boxes, &left_boxes);
        if self.config.fuse_score {
            fuse_score(&mut cost, &left);
        }
        let third = linear_assignment(&cost, unconfirmed.len(), left.len(), 0.7);
        for &(u, d) in &third.matches {
            let kf = self.kf.clone();
            self.tracked[unconfirmed[u]].update(&kf, &left[d], frame);
        }
        for &u in &third.unmatched_rows {
            self.tracked[unconfirmed[u]].state = TrackState::Removed;
        }

        // 6. New tracks from confident leftovers.
        let mut new_tracks = Vec::new();
        for &d in &third.unmatched_cols {
            let det = left[d];
            if det.score < thresh + 0.1 {
                continue;
            }
            new_tracks.push(STrack {
                id: self.next_id,
                kalman: self.kf.initiate(det.bbox.to_xyah()),
                score: det.score,
                state: TrackState::Tracked,
                // Only tracks born on the very first frame are confirmed at once.
                is_activated: frame == 1,
                frame_id: frame,
                start_frame: frame,
                tracklet_len: 0,
            });
            self.next_id += 1;
        }

        // 7. Rebuild the lists in the original's order: surviving tracked
        //    tracks, then new tracks, then re-found lost tracks.
        let max_lost = self.max_time_lost();
        let refound: Vec<STrack> = first
            .matches
            .iter()
            .filter_map(|&(p, _)| match pool[p] {
                Slot::Lost(i) => Some(self.lost[i].clone()),
                Slot::Tracked(_) => None,
            })
            .collect();
        let (still_tracked, newly_lost): (Vec<STrack>, Vec<STrack>) =
            std::mem::take(&mut self.tracked)
                .into_iter()
                .filter(|t| t.state != TrackState::Removed)
                .partition(|t| t.state == TrackState::Tracked);
        let old_lost: Vec<STrack> = std::mem::take(&mut self.lost)
            .into_iter()
            .filter(|t| t.state == TrackState::Lost && frame - t.frame_id <= max_lost)
            .collect();

        let mut tracked = still_tracked;
        tracked.extend(new_tracks);
        tracked.extend(refound);
        let mut lost = old_lost;
        lost.extend(newly_lost);
        let (tracked, lost) = remove_duplicates(tracked, lost);
        self.tracked = tracked;
        self.lost = lost;

        self.tracked
            .iter()
            .filter(|t| t.is_activated)
            .map(STrack::snapshot)
            .collect()
    }

    /// Tracks currently lost but not yet removed (useful for debugging).
    pub fn lost_tracks(&self) -> Vec<Track> {
        self.lost.iter().map(STrack::snapshot).collect()
    }
}

/// If a tracked and a lost track overlap almost perfectly (IoU > 0.85), they
/// are the same object: keep the one that has existed longer.
fn remove_duplicates(tracked: Vec<STrack>, lost: Vec<STrack>) -> (Vec<STrack>, Vec<STrack>) {
    let tracked_boxes: Vec<BBox> = tracked.iter().map(STrack::bbox).collect();
    let lost_boxes: Vec<BBox> = lost.iter().map(STrack::bbox).collect();
    let distance = iou_distance(&tracked_boxes, &lost_boxes);
    let (mut drop_tracked, mut drop_lost) = (vec![false; tracked.len()], vec![false; lost.len()]);
    for (p, t) in tracked.iter().enumerate() {
        for (q, l) in lost.iter().enumerate() {
            if distance[p * lost.len() + q] < 0.15 {
                let age_t = t.frame_id - t.start_frame;
                let age_l = l.frame_id - l.start_frame;
                if age_t > age_l {
                    drop_lost[q] = true;
                } else {
                    drop_tracked[p] = true;
                }
            }
        }
    }
    let keep = |v: Vec<STrack>, drop: &[bool]| {
        v.into_iter()
            .zip(drop)
            .filter(|(_, drop)| !**drop)
            .map(|(t, _)| t)
            .collect::<Vec<_>>()
    };
    (keep(tracked, &drop_tracked), keep(lost, &drop_lost))
}

impl Default for ByteTracker {
    fn default() -> Self {
        Self::new(ByteTrackerConfig::default())
    }
}

impl Tracker for ByteTracker {
    fn update(&mut self, detections: &[Detection]) -> Vec<Track> {
        self.step(detections)
    }

    fn reset(&mut self) {
        *self = Self::new(self.config);
    }
}
