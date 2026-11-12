//! A run handle with buffered metric logging.

use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex, MutexGuard};

pub use crate::client::RunStatus;
use crate::client::{
    KeyValue, MAX_METRICS_PER_BATCH, MAX_PARAMS_PER_BATCH, Metric, MlflowClient, MlflowError,
    RunInfo, now_millis,
};

/// An MLflow run in progress.
///
/// Cheap to clone: clones share the same run and the same metric buffer
/// (`Arc<Mutex<..>>`), so the training code can keep one handle while Burn's
/// logger owns another.
///
/// Metrics are buffered and sent in batches, either when the buffer is full,
/// at the end of every epoch (the logger calls [`Run::flush`]) or in
/// [`Run::finish`]. One HTTP request per batch of 1000 points instead of one
/// per point.
#[derive(Debug, Clone)]
pub struct Run {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    client: MlflowClient,
    info: RunInfo,
    state: Mutex<State>,
}

#[derive(Debug, Default)]
struct State {
    pending: Vec<Metric>,
    finished: bool,
    warned: bool,
}

impl Run {
    /// Creates a run in `experiment` (created if it doesn't exist yet).
    pub fn start(
        client: &MlflowClient,
        experiment: &str,
        run_name: &str,
        tags: &[(String, String)],
    ) -> Result<Self, MlflowError> {
        let experiment_id = client.get_or_create_experiment(experiment)?;
        let tags: Vec<KeyValue> = tags.iter().map(|(k, v)| KeyValue::new(k, v)).collect();
        let info = client.create_run(&experiment_id, run_name, &tags)?;
        Ok(Self {
            inner: Arc::new(Inner {
                client: client.clone(),
                info,
                state: Mutex::new(State::default()),
            }),
        })
    }

    pub fn id(&self) -> &str {
        &self.inner.info.run_id
    }

    /// Link to this run in the MLflow web UI.
    pub fn ui_url(&self) -> String {
        format!(
            "{}/#/experiments/{}/runs/{}",
            self.inner.client.base_url(),
            self.inner.info.experiment_id,
            self.inner.info.run_id
        )
    }

    /// Logs hyperparameters. Params are write-once in MLflow: logging the same
    /// key twice with a different value is an error.
    pub fn log_params<K, V>(&self, params: &[(K, V)]) -> Result<(), MlflowError>
    where
        K: AsRef<str>,
        V: ToString,
    {
        let params: Vec<KeyValue> = params
            .iter()
            .map(|(k, v)| KeyValue::new(k.as_ref(), v.to_string()))
            .collect();
        for chunk in params.chunks(MAX_PARAMS_PER_BATCH) {
            self.inner.client.log_batch(self.id(), &[], chunk)?;
        }
        Ok(())
    }

    /// Queues one metric point; it is sent on the next flush.
    pub fn log_metric(&self, key: &str, value: f64, step: i64) {
        // MLflow rejects NaN/inf; skip them instead of failing the whole batch.
        if !value.is_finite() {
            return;
        }
        let full = {
            let mut state = self.lock();
            state.pending.push(Metric {
                key: key.to_owned(),
                value,
                timestamp: now_millis(),
                step,
            });
            state.pending.len() >= MAX_METRICS_PER_BATCH
        }; // the lock is released here, at the end of the block
        if full {
            self.flush();
        }
    }

    /// Sends all queued metrics. Errors are logged, not returned: a flaky
    /// tracking server must not stop a training run.
    pub fn flush(&self) {
        let pending = std::mem::take(&mut self.lock().pending);
        for chunk in pending.chunks(MAX_METRICS_PER_BATCH) {
            if let Err(err) = self.inner.client.log_batch(self.id(), chunk, &[]) {
                self.warn_once(&err);
            }
        }
    }

    /// Uploads a local file as a run artifact.
    pub fn log_artifact(&self, path: &Path) -> Result<(), MlflowError> {
        let contents = std::fs::read(path).map_err(|source| MlflowError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "artifact".into());
        self.inner
            .client
            .upload_artifact(&self.inner.info, &name, &contents)
    }

    /// Flushes pending metrics and closes the run with `status`.
    pub fn finish(&self, status: RunStatus) -> Result<(), MlflowError> {
        self.flush();
        self.lock().finished = true;
        self.inner.client.update_run(self.id(), status)
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.inner.state.lock().expect("run state mutex poisoned")
    }

    fn warn_once(&self, err: &MlflowError) {
        let mut state = self.lock();
        if !state.warned {
            state.warned = true;
            log::warn!("MLflow logging failed, continuing without it: {err}");
        }
    }
}

/// `Drop` runs automatically when the last `Arc` clone goes away (RAII, the
/// same mechanism that frees a `String`). If the program forgot to call
/// `finish` (e.g. it returned early with an error) the run is marked as
/// killed instead of staying "running" forever in the UI.
impl Drop for Inner {
    fn drop(&mut self) {
        let state = self.state.get_mut().expect("run state mutex poisoned");
        if !state.finished {
            let pending = std::mem::take(&mut state.pending);
            let _ = self.client.log_batch(&self.info.run_id, &pending, &[]);
            let _ = self.client.update_run(&self.info.run_id, RunStatus::Killed);
        }
    }
}

/// Standard MLflow tags describing the code version, so every run records the
/// exact commit it was produced with (and whether the tree had local changes).
pub fn git_tags() -> Vec<(String, String)> {
    let git = |args: &[&str]| -> Option<String> {
        let output = Command::new("git").args(args).output().ok()?;
        output
            .status
            .success()
            .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
    };

    let mut tags = Vec::new();
    if let Some(commit) = git(&["rev-parse", "HEAD"]) {
        tags.push(("mlflow.source.git.commit".to_owned(), commit));
    }
    if let Some(branch) = git(&["rev-parse", "--abbrev-ref", "HEAD"]) {
        tags.push(("mlflow.source.git.branch".to_owned(), branch));
    }
    if let Some(status) = git(&["status", "--porcelain"]) {
        tags.push(("git.dirty".to_owned(), (!status.is_empty()).to_string()));
    }
    tags
}
