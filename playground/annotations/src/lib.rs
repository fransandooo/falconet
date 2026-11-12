//! # Exercise 3a — ownership and borrowing
//!
//! A parser for a toy annotation format that mimics the Anti-UAV datasets:
//! every frame says whether the drone is visible and, if so, its bounding box.
//!
//! The code is organised around the three ways a function can receive data:
//!
//! | Signature            | Meaning                         | C analogy                      |
//! |----------------------|---------------------------------|--------------------------------|
//! | `fn f(x: &T)`        | shared borrow, read-only        | `const T *x`                   |
//! | `fn f(x: &mut T)`    | exclusive borrow, read-write    | `T *x`, but nobody else may alias it |
//! | `fn f(x: T)`         | ownership moves into `f`        | `f` now owns (and frees) `x`   |
//!
//! Modules:
//! - [`bbox`]: the `BBox` type (a `Copy` type, like a plain C struct),
//! - [`parser`]: text → data, with borrowed (`Header<'a>`) and owned (`Sequence`) results,
//! - [`stats`]: one function per way of passing a `Sequence`.

pub mod bbox;
pub mod parser;
pub mod stats;

// Re-exports: users can write `playground_annotations::BBox` instead of
// `playground_annotations::bbox::BBox`.
pub use bbox::BBox;
pub use parser::{FrameAnnotation, Header, ParseError, Sequence, parse_header, parse_sequence};
pub use stats::{SequenceStats, clip_to_image, compute_stats, into_visible};
