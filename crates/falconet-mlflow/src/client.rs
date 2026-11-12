//! Typed wrappers around the MLflow tracking REST API (`/api/2.0/mlflow/...`).
//!
//! Request and response bodies are plain structs; `serde` derives the JSON
//! conversion, like `dataclasses` + `json` in Python, but generated at compile
//! time and checked against the struct's types.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use ureq::Agent;

/// A type alias: a shorter name for a long type, like `typedef` in C.
type Response = ureq::http::Response<ureq::Body>;

/// Everything that can go wrong talking to MLflow.
///
/// `thiserror` derives `Display` and `std::error::Error` from the `#[error]`
/// attributes: the same thing we wrote by hand for `ParseError` in exercise 3a.
#[derive(Debug, thiserror::Error)]
pub enum MlflowError {
    /// The server could not be reached (not running, wrong URL, timeout...).
    #[error("cannot reach the MLflow server at {url}: {source}")]
    Unreachable { url: String, source: ureq::Error },
    /// The server answered with an error status.
    #[error("MLflow returned HTTP {status} for {endpoint}: {body}")]
    Api {
        endpoint: String,
        status: u16,
        body: String,
    },
    /// The response body was not the JSON we expected.
    #[error("unexpected response from {endpoint}: {source}")]
    Decode {
        endpoint: String,
        source: ureq::Error,
    },
    /// A local file (artifact) could not be read.
    #[error("cannot read artifact {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
}

/// A key/value pair, the shape MLflow uses for params and tags.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeyValue {
    pub key: String,
    pub value: String,
}

impl KeyValue {
    pub fn new(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: value.into(),
        }
    }
}

/// One metric data point.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Metric {
    pub key: String,
    pub value: f64,
    /// Milliseconds since the Unix epoch.
    pub timestamp: i64,
    pub step: i64,
}

/// Status of a run, as MLflow names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RunStatus {
    Running,
    Finished,
    Failed,
    Killed,
}

/// The run fields we care about from `runs/create`.
#[derive(Debug, Clone, Deserialize)]
pub struct RunInfo {
    pub run_id: String,
    pub experiment_id: String,
    pub artifact_uri: String,
}

/// Blocking HTTP client for one MLflow tracking server.
///
/// Blocking is fine here: Burn calls metric loggers from its own background
/// thread, and the server is on localhost.
#[derive(Debug, Clone)]
pub struct MlflowClient {
    base_url: String,
    agent: Agent,
}

// MLflow's limits for a single `log-batch` request.
pub(crate) const MAX_METRICS_PER_BATCH: usize = 1000;
pub(crate) const MAX_PARAMS_PER_BATCH: usize = 100;

impl MlflowClient {
    /// `base_url` is the tracking URI, e.g. `http://127.0.0.1:5000`.
    pub fn new(base_url: impl Into<String>) -> Self {
        let agent = Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(10)))
            // We want to read MLflow's error message on 4xx/5xx, so status
            // codes are not turned into `Err` automatically.
            .http_status_as_error(false)
            .build()
            .into();
        Self {
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            agent,
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Returns the id of the experiment called `name`, creating it if needed.
    pub fn get_or_create_experiment(&self, name: &str) -> Result<String, MlflowError> {
        #[derive(Deserialize)]
        struct Experiment {
            experiment_id: String,
        }
        #[derive(Deserialize)]
        struct GetResponse {
            experiment: Experiment,
        }
        #[derive(Deserialize)]
        struct CreateResponse {
            experiment_id: String,
        }

        let endpoint = "experiments/get-by-name";
        let response = self
            .agent
            .get(self.url(endpoint))
            .query("experiment_name", name)
            .call();
        match self.decode::<GetResponse>(endpoint, response) {
            Ok(found) => Ok(found.experiment.experiment_id),
            // 404 = no experiment with that name yet: create it.
            Err(MlflowError::Api { status: 404, .. }) => {
                let created: CreateResponse =
                    self.post("experiments/create", &serde_json::json!({ "name": name }))?;
                Ok(created.experiment_id)
            }
            Err(other) => Err(other),
        }
    }

    /// Creates a run inside an experiment.
    pub fn create_run(
        &self,
        experiment_id: &str,
        run_name: &str,
        tags: &[KeyValue],
    ) -> Result<RunInfo, MlflowError> {
        #[derive(Deserialize)]
        struct Run {
            info: RunInfo,
        }
        #[derive(Deserialize)]
        struct CreateResponse {
            run: Run,
        }

        let body = serde_json::json!({
            "experiment_id": experiment_id,
            "run_name": run_name,
            "start_time": now_millis(),
            "tags": tags,
        });
        let response: CreateResponse = self.post("runs/create", &body)?;
        Ok(response.run.info)
    }

    /// Logs metrics and params in one request. Callers must respect MLflow's
    /// per-request limits (1000 metrics, 100 params); [`crate::Run`] does.
    pub fn log_batch(
        &self,
        run_id: &str,
        metrics: &[Metric],
        params: &[KeyValue],
    ) -> Result<(), MlflowError> {
        let body = serde_json::json!({
            "run_id": run_id,
            "metrics": metrics,
            "params": params,
        });
        self.post::<serde_json::Value>("runs/log-batch", &body)
            .map(|_| ())
    }

    /// Sets (or overwrites) a tag on a run.
    pub fn set_tag(&self, run_id: &str, key: &str, value: &str) -> Result<(), MlflowError> {
        let body = serde_json::json!({ "run_id": run_id, "key": key, "value": value });
        self.post::<serde_json::Value>("runs/set-tag", &body)
            .map(|_| ())
    }

    /// Marks a run as finished, failed or killed.
    pub fn update_run(&self, run_id: &str, status: RunStatus) -> Result<(), MlflowError> {
        let body = serde_json::json!({
            "run_id": run_id,
            "status": status,
            "end_time": now_millis(),
        });
        self.post::<serde_json::Value>("runs/update", &body)
            .map(|_| ())
    }

    /// Uploads a file to the run's artifact store.
    ///
    /// Works when the server proxies artifacts (`mlflow server` does by
    /// default), i.e. when the run's artifact URI is `mlflow-artifacts:/...`.
    pub fn upload_artifact(
        &self,
        run: &RunInfo,
        file_name: &str,
        contents: &[u8],
    ) -> Result<(), MlflowError> {
        let Some(path) = run.artifact_uri.strip_prefix("mlflow-artifacts:") else {
            return Err(MlflowError::Api {
                endpoint: "mlflow-artifacts".into(),
                status: 0,
                body: format!(
                    "artifact URI {} is not served by the tracking server",
                    run.artifact_uri
                ),
            });
        };
        let url = format!(
            "{}/api/2.0/mlflow-artifacts/artifacts/{}/{file_name}",
            self.base_url,
            path.trim_matches('/')
        );
        let response = self
            .agent
            .put(&url)
            .header("Content-Type", "application/octet-stream")
            .send(contents);
        // This endpoint answers 200 with an empty body: check the status only.
        self.check_status("mlflow-artifacts", response).map(|_| ())
    }

    fn url(&self, endpoint: &str) -> String {
        format!("{}/api/2.0/mlflow/{endpoint}", self.base_url)
    }

    /// POSTs a JSON body and decodes the JSON response into `T`.
    ///
    /// `T: DeserializeOwned` = "any type serde can build from JSON without
    /// borrowing from the input buffer".
    fn post<T: DeserializeOwned>(
        &self,
        endpoint: &str,
        body: &serde_json::Value,
    ) -> Result<T, MlflowError> {
        let response = self.agent.post(self.url(endpoint)).send_json(body);
        self.decode(endpoint, response)
    }

    /// Turns transport errors and non-2xx statuses into [`MlflowError`]s.
    fn check_status(
        &self,
        endpoint: &str,
        response: Result<Response, ureq::Error>,
    ) -> Result<Response, MlflowError> {
        let mut response = response.map_err(|source| MlflowError::Unreachable {
            url: self.base_url.clone(),
            source,
        })?;
        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            let body = response.body_mut().read_to_string().unwrap_or_default();
            return Err(MlflowError::Api {
                endpoint: endpoint.to_owned(),
                status,
                body,
            });
        }
        Ok(response)
    }

    /// [`Self::check_status`], then parses the JSON body into `T`.
    fn decode<T: DeserializeOwned>(
        &self,
        endpoint: &str,
        response: Result<Response, ureq::Error>,
    ) -> Result<T, MlflowError> {
        let mut response = self.check_status(endpoint, response)?;
        response
            .body_mut()
            .read_json()
            .map_err(|source| MlflowError::Decode {
                endpoint: endpoint.to_owned(),
                source,
            })
    }
}

/// Milliseconds since the Unix epoch, MLflow's timestamp unit.
pub(crate) fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_status_serializes_like_mlflow_expects() {
        let json = serde_json::to_string(&RunStatus::Finished).unwrap();
        assert_eq!(json, "\"FINISHED\"");
    }

    #[test]
    fn metric_serializes_with_mlflow_field_names() {
        let metric = Metric {
            key: "train/loss".into(),
            value: 0.5,
            timestamp: 1,
            step: 2,
        };
        let json = serde_json::to_value(&metric).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"key": "train/loss", "value": 0.5, "timestamp": 1, "step": 2})
        );
    }

    #[test]
    fn trailing_slash_in_base_url_is_ignored() {
        let client = MlflowClient::new("http://localhost:5000/");
        assert_eq!(
            client.url("runs/create"),
            "http://localhost:5000/api/2.0/mlflow/runs/create"
        );
    }

    #[test]
    fn unreachable_server_is_reported_as_such() {
        // Port 9 ("discard") is essentially never open on a dev machine.
        let client = MlflowClient::new("http://127.0.0.1:9");
        let err = client.get_or_create_experiment("x").unwrap_err();
        assert!(matches!(err, MlflowError::Unreachable { .. }), "{err}");
    }
}
