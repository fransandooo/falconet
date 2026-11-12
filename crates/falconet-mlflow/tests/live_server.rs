//! End-to-end test against a real MLflow server.
//!
//! Ignored by default (CI has no server). Run it with a server up:
//!
//! ```text
//! mise run mlflow &
//! cargo nextest run -p falconet-mlflow --run-ignored only
//! ```

use falconet_mlflow::{MlflowClient, Run, RunStatus, git_tags};

#[test]
#[ignore = "needs a running MLflow server (mise run mlflow)"]
fn full_run_lifecycle() {
    let uri = std::env::var("MLFLOW_TRACKING_URI").unwrap_or("http://127.0.0.1:5000".into());
    let client = MlflowClient::new(uri);

    let run = Run::start(&client, "falconet-tests", "lifecycle", &git_tags()).expect("start");
    run.log_params(&[("learning_rate", 0.001), ("batch_size", 128.0)])
        .expect("params");
    for step in 0..1500 {
        // More points than one batch holds: exercises the automatic flush.
        run.log_metric("loss", 1.0 / (step as f64 + 1.0), step);
    }
    run.log_metric("not_a_number", f64::NAN, 0); // silently skipped

    let artifact = std::env::temp_dir().join("falconet-mlflow-test.txt");
    std::fs::write(&artifact, "hello").unwrap();
    run.log_artifact(&artifact).expect("artifact upload");

    run.finish(RunStatus::Finished).expect("finish");
    println!("run: {}", run.ui_url());
}
