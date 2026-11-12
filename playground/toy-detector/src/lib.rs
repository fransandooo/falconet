//! # Exercise 3d — a toy detector
//!
//! The real problem in miniature: find one small bright object in a noisy
//! image and predict its bounding box.
//!
//! The model is a tiny **CenterNet**-style detector (anchor-free): instead of
//! regressing four numbers directly, it predicts on a grid 4× coarser than the
//! image:
//! - a **heatmap** whose peak marks the object's centre cell,
//! - the object's **size** (w, h) at that cell,
//! - a sub-cell **offset**, to recover the centre precisely.
//!
//! This is the same family of design the plan recommends for phase 1, so the
//! loss, target encoding and decoding here are a first draft of the real ones.
//!
//! Modules:
//! - [`rng`] and [`scene`]: deterministic synthetic scenes,
//! - [`data`]: `Dataset` + `Batcher`, including the target encoding,
//! - [`model`]: the network,
//! - [`loss`]: focal loss on the heatmap + L1 on size/offset,
//! - [`decode`]: from network outputs to boxes, plus evaluation metrics,
//! - [`viz`]: rendering predictions and heatmaps to images and a GIF,
//! - [`train`]: a hand-written training loop (no `Learner`).

pub mod data;
pub mod decode;
pub mod loss;
pub mod model;
pub mod rng;
pub mod scene;
pub mod train;
pub mod viz;

/// The output grid is this many times coarser than the input image.
pub const STRIDE: usize = 4;
