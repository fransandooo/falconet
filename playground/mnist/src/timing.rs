//! Measuring epoch times by plugging into Burn's training events.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use burn::train::logger::TrainingProgressLogger;

/// Wall-clock time of one epoch, split into training and validation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochTiming {
    pub epoch: usize,
    pub train: Duration,
    pub valid: Duration,
    pub train_items: usize,
}

impl EpochTiming {
    /// Training throughput in images per second.
    pub fn train_items_per_sec(&self) -> f64 {
        self.train_items as f64 / self.train.as_secs_f64()
    }
}

#[derive(Debug, Default)]
struct TimerState {
    split_start: Option<Instant>,
    split_is_train: bool,
    split_items: usize,
    train: Duration,
    valid: Duration,
    train_items: usize,
    epochs: Vec<EpochTiming>,
}

/// A [`TrainingProgressLogger`] that records how long each split of each epoch takes.
///
/// Burn takes ownership of the logger (it is moved into a `Box` and driven
/// from a background thread), but we still want to read the timings after
/// training. Hence `Arc<Mutex<..>>`:
/// - `Arc` = *atomically reference-counted* pointer: shared ownership, freed
///   when the last clone is dropped. Like a refcounted struct in C.
/// - `Mutex` = only one thread at a time may access the data inside. Unlike
///   `pthread_mutex_t`, the data lives *inside* the mutex, so you can't touch
///   it without locking.
///
/// Cloning an `EpochTimer` clones the `Arc` (a refcount increment), so both
/// clones see the same state.
#[derive(Debug, Clone, Default)]
pub struct EpochTimer {
    state: Arc<Mutex<TimerState>>,
}

impl EpochTimer {
    /// The timings recorded so far.
    pub fn timings(&self) -> Vec<EpochTiming> {
        self.lock().epochs.clone()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, TimerState> {
        // `lock` fails only if another thread panicked while holding the lock.
        self.state.lock().expect("timer mutex poisoned")
    }
}

impl TrainingProgressLogger for EpochTimer {
    fn start(
        &mut self,
        _epochs: usize,
        _start: usize,
        _items: Option<usize>,
        _label: Option<&str>,
    ) {
    }

    fn start_split(&mut self, split: &str, total_items: usize) {
        let mut state = self.lock();
        state.split_start = Some(Instant::now());
        state.split_is_train = split == "train";
        state.split_items = total_items;
    }

    fn update_split(&mut self, _items_processed: usize) {}

    fn end_split(&mut self) {
        let mut state = self.lock();
        if let Some(start) = state.split_start.take() {
            let elapsed = start.elapsed();
            if state.split_is_train {
                state.train = elapsed;
                state.train_items = state.split_items;
            } else {
                state.valid = elapsed;
            }
        }
    }

    fn update_epoch(&mut self, epoch: usize) {
        let mut state = self.lock();
        let timing = EpochTiming {
            epoch,
            train: state.train,
            valid: state.valid,
            train_items: state.train_items,
        };
        state.epochs.push(timing);
    }

    fn end(&mut self) {}

    fn log_event_training(&mut self, _event: String) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_one_timing_per_epoch_and_shares_state_between_clones() {
        let timer = EpochTimer::default();
        let mut handle = timer.clone(); // what Burn would own
        for epoch in 1..=2 {
            handle.start_split("train", 100);
            handle.end_split();
            handle.start_split("valid", 10);
            handle.end_split();
            handle.update_epoch(epoch);
        }
        let timings = timer.timings();
        assert_eq!(timings.len(), 2);
        assert_eq!(timings[1].epoch, 2);
        assert_eq!(timings[0].train_items, 100);
    }
}
