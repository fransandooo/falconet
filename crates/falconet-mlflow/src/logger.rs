//! Bridge between Burn's training metrics and an MLflow [`Run`].

use std::collections::HashMap;

use burn::train::logger::MetricLogger;
use burn::train::metric::store::{EpochSummary, MetricsUpdate, Split};
use burn::train::metric::{MetricDefinition, MetricId, NumericEntry};

use crate::run::Run;

/// A Burn [`MetricLogger`] that sends every numeric metric to MLflow.
///
/// Naming scheme in MLflow:
/// - `train/loss`, `valid/accuracy`, ...: one point per epoch (`step` = epoch),
///   using the value Burn computes once at the end of each epoch,
/// - `train/loss_batch`, ...: one point per training batch (`step` = global
///   batch index), for fine-grained curves.
///
/// Register it *after* the default `FileMetricLogger`: Burn reads metrics back
/// (for early stopping and checkpoint selection) from the first logger that
/// can, and this one is write-only.
pub struct MlflowMetricLogger {
    run: Run,
    names: HashMap<MetricId, String>,
    train_batches: i64,
}

impl MlflowMetricLogger {
    pub fn new(run: Run) -> Self {
        Self {
            run,
            names: HashMap::new(),
            train_batches: 0,
        }
    }
}

/// `"Accuracy"` → `"accuracy"`, `"Learning Rate"` → `"learning_rate"`.
fn metric_key(name: &str) -> String {
    name.trim().to_lowercase().replace([' ', '-'], "_")
}

impl MetricLogger for MlflowMetricLogger {
    fn log(&mut self, update: MetricsUpdate, epoch: usize, split: &Split) {
        let is_train = matches!(split, Split::Train);
        let mut counted_batch = false;

        for numeric in &update.entries_numeric {
            let Some(name) = self.names.get(&numeric.entry.metric_id) else {
                continue;
            };
            let Some(entry) = &numeric.numeric_entry else {
                continue;
            };
            let key = format!("{split}/{name}");
            match entry {
                // End-of-epoch value: one point per epoch, both splits.
                NumericEntry::Final(value) => self.run.log_metric(&key, *value, epoch as i64),
                // Per-batch value: only for training, to keep the UI readable.
                other if is_train => {
                    if !counted_batch {
                        self.train_batches += 1;
                        counted_batch = true;
                    }
                    let step = self.train_batches;
                    self.run
                        .log_metric(&format!("{key}_batch"), other.current(), step);
                }
                _ => {}
            }
        }
    }

    fn read_numeric(
        &mut self,
        _name: &str,
        _epoch: usize,
        _split: &Split,
    ) -> Result<Vec<NumericEntry>, String> {
        Err("the MLflow logger is write-only".to_owned())
    }

    fn log_metric_definition(&mut self, definition: MetricDefinition) {
        self.names
            .insert(definition.metric_id.clone(), metric_key(&definition.name));
    }

    fn log_epoch_summary(&mut self, _summary: EpochSummary) {
        // Send this epoch's points now, so the MLflow UI updates live.
        self.run.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_names_become_lowercase_keys() {
        assert_eq!(metric_key("Accuracy"), "accuracy");
        assert_eq!(metric_key("Learning Rate"), "learning_rate");
        assert_eq!(metric_key("CPU-Usage"), "cpu_usage");
    }
}
