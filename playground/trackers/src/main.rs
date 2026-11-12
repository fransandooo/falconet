//! Demo for exercise 3b: run both trackers on the synthetic two-drone scene.
//!
//! ```text
//! cargo run -p playground-trackers
//! ```

use playground_trackers::{GreedyIouTracker, NaiveTracker, Tracker, evaluate, synthetic};

fn main() {
    let frames = synthetic::two_drones_scene();
    let detections: usize = frames.iter().map(Vec::len).sum();
    println!(
        "{} frames, {detections} detections (2 real drones + 1 false positive)\n",
        frames.len()
    );

    // Static dispatch: the compiler knows the concrete type, `evaluate` is
    // specialised for `GreedyIouTracker`.
    let mut greedy = GreedyIouTracker::default();
    let report = evaluate(&mut greedy, &frames);
    println!(
        "static dispatch -> {}: {} IDs\n",
        report.tracker, report.distinct_ids
    );

    // Dynamic dispatch: different concrete types in one `Vec`. A
    // `Box<dyn Tracker>` is a "fat pointer" (data pointer + vtable pointer),
    // and method calls are looked up in the vtable at runtime — the same thing
    // you'd build by hand in C with a struct of function pointers.
    let mut trackers: Vec<Box<dyn Tracker>> = vec![
        Box::new(NaiveTracker::default()),
        Box::new(GreedyIouTracker::default()),
        Box::new(GreedyIouTracker::new(0.1, 5)), // looser matching, longer memory
    ];

    println!(
        "{:<40} {:>12} {:>12}",
        "tracker", "distinct IDs", "track boxes"
    );
    for tracker in &mut trackers {
        let report = evaluate(tracker.as_mut(), &frames);
        println!(
            "{:<40} {:>12} {:>12}",
            report.tracker, report.distinct_ids, report.track_boxes
        );
    }

    println!(
        "\nideal: 3 IDs (2 drones + the false positive, which a tracker alone can't reject).\n\
         The greedy tracker matches against the *last seen* box. Drone B moves while\n\
         occluded, so when it reappears it no longer overlaps enough and gets a new ID.\n\
         Predicting motion with a Kalman filter (ByteTrack, phase 1) fixes exactly this."
    );
}
