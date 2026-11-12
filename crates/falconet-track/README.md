# falconet-track

ByteTrack (Zhang et al., ECCV 2022), implemented from scratch in Rust following the official `BYTETracker.update`.

```rust
use falconet_track::{BBox, ByteTracker, Detection, Tracker};

let mut tracker = ByteTracker::default();
for frame_detections in video {
    let tracks = tracker.update(&frame_detections); // confirmed tracks, with stable ids
}
```

## How it works

| Module | Piece | Notes |
|---|---|---|
| `kalman` | Constant-velocity Kalman filter over `(cx, cy, w/h, h)` | Noise proportional to box height (1/20 position, 1/160 velocity); Cholesky solve instead of a matrix inverse; `nalgebra` fixed-size matrices |
| `assignment` | Optimal assignment with a cost limit | Hungarian algorithm on the `lapjv`-style extended matrix (dummy rows/columns at `limit / 2`); checked against brute force |
| `matching` | IoU distance, score fusion | |
| `byte_tracker` | Two-stage association + track life cycle | High detections first (cost limit 0.8), then low ones (0.1–0.5) for still-tracked tracks (0.5), then unconfirmed tracks (0.7); new tracks need confirmation on the next frame; lost tracks are kept for 30 frames |

## Validation

- **Behaviour tests** (`tests/scenarios.rs`): identity kept through constant motion, a 10-frame occlusion (bridged by the Kalman prediction), low-score dips (second association), and two crossing drones. Low-score-only objects never start tracks; one-frame false positives are never reported; a gap longer than the buffer gives a new id. On exercise 3b's scene, ByteTrack gives **2 ids** where the greedy IoU tracker gave 4.
- **Cross-check against [`trackforge`](https://crates.io/crates/trackforge)** (`tests/vs_trackforge.rs`), an independent Rust ByteTrack, on 200 random scenes of 1–4 drones (20,000 frames, 48,379 track boxes): **identical identities on every frame**, box centres within 1.33 px. The remaining differences are documented design choices of trackforge: it reports raw detection boxes instead of Kalman-filtered ones, and it confirms new tracks immediately.

## Deviations from the official implementation

- **Continuous IoU.** The official code uses `cython_bbox`, with the legacy "+1 pixel" box area. That inflates the overlap of tiny boxes, e.g. two 4×4 boxes shifted by 2 px get IoU 0.43 instead of 0.33. We use the continuous IoU, as the Anti-UAV410 metric does.
- **Per-tracker id counter** instead of a global class counter, so `reset()` restarts ids at 1 for each video.

## Known limitation worth watching

A track starts with zero velocity. A **small, fast** object can therefore move far enough between its first two frames that the predicted box no longer overlaps the detection: the track is lost, and a new one starts. In the cross-check this happened to a 19 × 5 px box moving 4 px/frame vertically. Anti-UAV410 has many boxes that small, so once real detections exist we should measure how often it happens. Possible mitigations: a centre-distance fallback for unmatched tracks, a lower `match_thresh` for young tracks, or OC-SORT's observation-centric re-update.
