//! # falconet-data
//!
//! Everything about the datasets, independent of any ML framework:
//! - [`bbox`]: the bounding-box type used across falconet,
//! - [`antiuav410`]: parsers and an index for the Anti-UAV410 benchmark,
//! - [`stats`]: dataset statistics (box sizes, absence, motion, attributes),
//! - [`metrics`]: the official Anti-UAV410 tracking metric (state accuracy).
//!
//! Datasets are never stored in the repository. They live under
//! `$FALCONET_DATA_DIR` (default `~/datasets`); see `scripts/` for downloads.

pub mod antiuav410;
pub mod bbox;
pub mod metrics;
pub mod stats;

pub use bbox::BBox;

use std::path::PathBuf;

/// Root directory for datasets: `$FALCONET_DATA_DIR`, or `~/datasets`.
pub fn data_dir() -> PathBuf {
    std::env::var_os("FALCONET_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var_os("HOME").unwrap_or_default();
            PathBuf::from(home).join("datasets")
        })
}
