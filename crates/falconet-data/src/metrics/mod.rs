//! Evaluation metrics.
//!
//! - [`state_accuracy`]: Anti-UAV410's official tracking metric (SA),
//! - [`detection`]: COCO-style detection metrics (mAP, AP per size, FP/frame).

pub mod detection;
pub mod state_accuracy;

pub use state_accuracy::{frame_score, mean_state_accuracy, state_accuracy};
