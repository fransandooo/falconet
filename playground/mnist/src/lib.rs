//! # Exercise 3c — Burn fundamentals on MNIST
//!
//! - [`data`]: turning dataset items into batched tensors (`Batcher`),
//! - [`model`]: an MLP and a small CNN, both `#[derive(Module)]`,
//! - [`training`]: the `Learner` + dashboard, plus a hand-written test loop,
//! - [`timing`]: a custom `TrainingProgressLogger` that measures epoch times.
//!
//! The tensor basics (shapes, broadcasting, autodiff) are in `src/bin/tensors.rs`.

pub mod data;
pub mod model;
pub mod timing;
pub mod training;
