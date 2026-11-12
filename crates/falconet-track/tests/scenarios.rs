//! Behavioural tests of ByteTrack on synthetic sequences.
//!
//! Each test builds a small "video" of detections and checks one property of
//! the tracker: identity preservation, occlusion handling, the low-score
//! second association, false-positive filtering.

use std::collections::{BTreeSet, HashMap};

use falconet_track::{BBox, ByteTracker, ByteTrackerConfig, Detection, Track, Tracker};

/// An object moving at constant velocity (px/frame).
#[derive(Clone, Copy)]
struct Object {
    start: BBox,
    velocity: (f32, f32),
}

impl Object {
    fn at(&self, frame: usize) -> BBox {
        let t = frame as f32;
        BBox::new(
            self.start.x + self.velocity.0 * t,
            self.start.y + self.velocity.1 * t,
            self.start.w,
            self.start.h,
        )
    }
}

/// Runs the tracker; `detect(object, frame)` returns the score of that
/// object's detection in that frame, or `None` if it is missed.
fn run(
    tracker: &mut ByteTracker,
    objects: &[Object],
    frames: usize,
    detect: impl Fn(usize, usize) -> Option<f32>,
) -> Vec<Vec<Track>> {
    (0..frames)
        .map(|f| {
            let dets: Vec<Detection> = objects
                .iter()
                .enumerate()
                .filter_map(|(i, o)| detect(i, f).map(|s| Detection::new(o.at(f), s)))
                .collect();
            tracker.update(&dets)
        })
        .collect()
}

fn ids(history: &[Vec<Track>]) -> BTreeSet<u64> {
    history.iter().flatten().map(|t| t.id).collect()
}

/// Which object each reported track box belongs to (nearest centre).
fn identity_map(history: &[Vec<Track>], objects: &[Object]) -> HashMap<u64, BTreeSet<usize>> {
    let mut map: HashMap<u64, BTreeSet<usize>> = HashMap::new();
    for (f, tracks) in history.iter().enumerate() {
        for t in tracks {
            let (tx, ty) = t.bbox.center();
            let nearest = (0..objects.len())
                .min_by(|&a, &b| {
                    let d = |i: usize| {
                        let (cx, cy) = objects[i].at(f).center();
                        (cx - tx).hypot(cy - ty)
                    };
                    d(a).total_cmp(&d(b))
                })
                .unwrap();
            map.entry(t.id).or_default().insert(nearest);
        }
    }
    map
}

fn drone(x: f32, y: f32, vx: f32, vy: f32) -> Object {
    Object {
        start: BBox::new(x, y, 12.0, 8.0),
        velocity: (vx, vy),
    }
}

#[test]
fn a_steadily_moving_drone_keeps_one_id() {
    let mut tracker = ByteTracker::default();
    let history = run(&mut tracker, &[drone(50.0, 60.0, 3.0, 1.0)], 100, |_, _| {
        Some(0.9)
    });
    assert_eq!(ids(&history), BTreeSet::from([1]));
    assert!(history.iter().all(|f| f.len() == 1));
    // The Kalman-filtered box stays on the object.
    let last = history.last().unwrap()[0].bbox;
    assert!(last.iou(&drone(50.0, 60.0, 3.0, 1.0).at(99)) > 0.9);
}

#[test]
fn kalman_prediction_bridges_an_occlusion_that_broke_the_greedy_tracker() {
    // Exercise 3b's greedy IoU tracker lost this drone after a 2-frame gap,
    // because it matched against the last seen box. ByteTrack predicts where
    // the drone went, so even a 10-frame gap keeps the ID.
    let fast = drone(300.0, 200.0, -1.5, 1.5);
    let mut tracker = ByteTracker::default();
    let history = run(&mut tracker, &[fast], 60, |_, f| {
        (!(20..30).contains(&f)).then_some(0.9)
    });
    assert_eq!(ids(&history), BTreeSet::from([1]));
    // Nothing is reported while the drone is hidden...
    assert!(history[20..30].iter().all(Vec::is_empty));
    // ...and it is picked up again right away.
    assert_eq!(history[30].len(), 1);
}

#[test]
fn an_occlusion_longer_than_the_buffer_starts_a_new_track() {
    let config = ByteTrackerConfig {
        track_buffer: 30,
        ..Default::default()
    };
    let mut tracker = ByteTracker::new(config);
    let history = run(
        &mut tracker,
        &[drone(100.0, 100.0, 1.0, 0.0)],
        100,
        |_, f| (!(20..60).contains(&f)).then_some(0.9),
    );
    assert_eq!(ids(&history).len(), 2);
}

#[test]
fn low_score_detections_keep_an_existing_track_alive() {
    // Scores dip to 0.3 (below track_thresh 0.5) for 15 frames, like a drone
    // passing in front of a warm cloud. The second association keeps the track.
    let mut tracker = ByteTracker::default();
    let history = run(&mut tracker, &[drone(80.0, 80.0, 2.0, 0.5)], 60, |_, f| {
        Some(if (20..35).contains(&f) { 0.3 } else { 0.9 })
    });
    assert_eq!(ids(&history), BTreeSet::from([1]));
    assert!(history[20..35].iter().all(|f| f.len() == 1));
}

#[test]
fn low_score_detections_alone_never_start_a_track() {
    let mut tracker = ByteTracker::default();
    let history = run(&mut tracker, &[drone(80.0, 80.0, 2.0, 0.5)], 30, |_, _| {
        Some(0.3)
    });
    assert!(ids(&history).is_empty());
}

#[test]
fn a_one_frame_false_positive_is_never_reported() {
    // After frame 1, a new track must be confirmed on the next frame.
    let objects = [drone(50.0, 50.0, 1.0, 0.0), drone(400.0, 300.0, 0.0, 0.0)];
    let mut tracker = ByteTracker::default();
    let history = run(&mut tracker, &objects, 30, |i, f| match i {
        0 => Some(0.9),
        _ => (f == 10).then_some(0.95), // confident, but only once
    });
    assert_eq!(ids(&history), BTreeSet::from([1]));
}

#[test]
fn a_new_object_is_reported_from_its_second_frame() {
    let objects = [drone(50.0, 50.0, 1.0, 0.0), drone(300.0, 100.0, -1.0, 0.0)];
    let mut tracker = ByteTracker::default();
    let history = run(&mut tracker, &objects, 20, |i, f| {
        (i == 0 || f >= 5).then_some(0.9)
    });
    assert_eq!(history[5].len(), 1, "unconfirmed on its first frame");
    assert_eq!(history[6].len(), 2, "confirmed on its second frame");
}

#[test]
fn two_drones_crossing_keep_their_identities() {
    // Paths cross with a vertical offset: boxes overlap (IoU ≈ 0.25) at the crossing.
    let objects = [
        drone(100.0, 100.0, 4.0, 0.0),
        drone(300.0, 105.0, -4.0, 0.0),
    ];
    let mut tracker = ByteTracker::default();
    let history = run(&mut tracker, &objects, 50, |_, _| Some(0.9));
    let map = identity_map(&history, &objects);
    assert_eq!(map.len(), 2, "{map:?}");
    assert!(
        map.values().all(|objs| objs.len() == 1),
        "identity switch: {map:?}"
    );
}

#[test]
fn the_exercise_3b_scene_gets_exactly_two_ids() {
    // Exercise 3b's scene: drone A, drone B occluded on frames 10–11, and a
    // low-confidence false positive on frame 15. Greedy IoU: 4 IDs. ByteTrack: 2.
    let a = Object {
        start: BBox::new(100.0, 100.0, 8.0, 6.0),
        velocity: (2.0, 0.0),
    };
    let b = Object {
        start: BBox::new(300.0, 200.0, 10.0, 10.0),
        velocity: (-1.5, 1.5),
    };
    let fp = Object {
        start: BBox::new(500.0, 400.0, 6.0, 6.0),
        velocity: (0.0, 0.0),
    };
    let mut tracker = ByteTracker::default();
    let history = run(&mut tracker, &[a, b, fp], 30, |i, f| match i {
        0 => Some(0.9),
        1 => (!(10..12).contains(&f)).then_some(0.9),
        _ => (f == 15).then_some(0.4),
    });
    assert_eq!(ids(&history).len(), 2);
}

#[test]
fn reset_restarts_the_id_counter() {
    let mut tracker = ByteTracker::default();
    let objects = [drone(10.0, 10.0, 1.0, 1.0)];
    run(&mut tracker, &objects, 5, |_, _| Some(0.9));
    tracker.reset();
    let history = run(&mut tracker, &objects, 5, |_, _| Some(0.9));
    assert_eq!(ids(&history), BTreeSet::from([1]));
}
