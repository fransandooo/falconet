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
    tagged_images: bool,
}

/// `images/<key>+step+<step>+timestamp+<ms>+<id>`, the artifact naming scheme
/// MLflow 3's UI parses for step-indexed images (`/` in keys becomes `~`).
fn image_artifact_stem(key: &str, step: i64, timestamp: i64, id: &str) -> String {
    let key: String = key
        .chars()
        .map(|c| match c {
            '/' => '~',
            c if c.is_ascii_alphanumeric() || "_-. ".contains(c) => c,
            _ => '_',
        })
        .collect();
    format!("images/{key}+step+{step}+timestamp+{timestamp}+{id}")
}

/// A unique-enough id for artifact names, without pulling in a UUID crate:
/// nanosecond clock + a per-process counter.
fn unique_suffix() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{nanos:x}{:04x}", COUNTER.fetch_add(1, Ordering::Relaxed))
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

    /// Logs an image for a given step, so the MLflow UI shows it in the run's
    /// image viewer with a step slider (e.g. one prediction grid per epoch).
    ///
    /// MLflow has no REST endpoint for this. Its Python client simply uploads
    /// the PNG plus a small WebP thumbnail with a naming convention the UI
    /// recognises, and sets the `mlflow.loggedImages` tag. We do the same.
    /// `png` and `webp_thumbnail` are already-encoded image files.
    pub fn log_image(
        &self,
        key: &str,
        step: i64,
        png: &[u8],
        webp_thumbnail: &[u8],
    ) -> Result<(), MlflowError> {
        let stem = image_artifact_stem(key, step, now_millis(), &unique_suffix());
        let client = &self.inner.client;
        client.upload_artifact(&self.inner.info, &format!("{stem}.png"), png)?;
        client.upload_artifact(
            &self.inner.info,
            &format!("{stem}+compressed.webp"),
            webp_thumbnail,
        )?;
        let mut state = self.lock();
        if !state.tagged_images {
            state.tagged_images = true;
            drop(state); // release the lock before the network call
            client.set_tag(self.id(), "mlflow.loggedImages", "True")?;
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_artifact_names_follow_the_mlflow_convention() {
        let stem = image_artifact_stem("valid/predictions", 3, 1700, "abc");
        assert_eq!(stem, "images/valid~predictions+step+3+timestamp+1700+abc");
    }

    #[test]
    fn unsafe_characters_in_image_keys_are_replaced() {
        let stem = image_artifact_stem("a:b", 0, 0, "x");
        assert_eq!(stem, "images/a_b+step+0+timestamp+0+x");
    }

    #[test]
    fn unique_suffixes_differ() {
        assert_ne!(unique_suffix(), unique_suffix());
    }
}
