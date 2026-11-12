//! # falconet-mlflow
//!
//! A small client for the [MLflow](https://mlflow.org) tracking server's REST
//! API, plus a Burn [`MetricLogger`](burn::train::logger::MetricLogger) that
//! streams training metrics to it.
//!
//! MLflow only ships Python, Java and R clients, but its tracking server speaks
//! plain JSON over HTTP, so a few endpoints are all we need:
//! experiments, runs, `log-batch` and artifact upload.
//!
//! ```no_run
//! use falconet_mlflow::{MlflowClient, Run, RunStatus};
//!
//! # fn main() -> Result<(), falconet_mlflow::MlflowError> {
//! let client = MlflowClient::new("http://127.0.0.1:5000");
//! let run = Run::start(&client, "mnist", "cnn-cuda", &[])?;
//! run.log_params(&[("learning_rate", "0.001"), ("batch_size", "128")])?;
//! run.log_metric("valid/accuracy", 0.98, 3);
//! run.finish(RunStatus::Finished)?;
//! # Ok(())
//! # }
//! ```
//!
//! Design choice: **tracking must never break training.** [`Run::start`]
//! reports a missing server up front, but once training is running, failures
//! to send metrics are logged as warnings and otherwise ignored.

pub mod client;
pub mod logger;
pub mod run;

pub use client::{MlflowClient, MlflowError};
pub use logger::MlflowMetricLogger;
pub use run::{Run, RunStatus, git_tags};
