//! # falconet-core
//!
//! Small types shared by every falconet crate, with no heavy dependencies:
//! the data, model, tracking and inference crates all speak in [`BBox`] and
//! [`Detection`].

pub mod bbox;

pub use bbox::BBox;

use serde::{Deserialize, Serialize};

/// One detector output: a box and a confidence score in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    pub bbox: BBox,
    pub score: f32,
}

impl Detection {
    pub fn new(bbox: BBox, score: f32) -> Self {
        Self { bbox, score }
    }
}
