//! Constant-velocity Kalman filter for bounding boxes.
//!
//! A port of ByteTrack's `kalman_filter.py` (itself from DeepSORT). The state
//! is 8-dimensional: `(cx, cy, a, h)` (box centre, aspect ratio w/h, height)
//! and their velocities. Each frame:
//!
//! - **predict**: move the box by its velocity and grow the uncertainty,
//! - **update**: blend the prediction with the new detection, weighting each
//!   by how certain it is (the Kalman gain).
//!
//! Noise levels are proportional to the box height, so a 10-px drone and a
//! 100-px drone get the same *relative* treatment.
//!
//! The matrices use `nalgebra`'s statically sized types: `SMatrix<f32, 8, 8>`
//! has its dimensions in the type (const generics), so multiplying an 8×8 by
//! a 4×4 is a compile error, not a runtime crash.

use nalgebra::{SMatrix, SVector};

pub type StateVector = SVector<f32, 8>;
pub type StateCovariance = SMatrix<f32, 8, 8>;
pub type Measurement = SVector<f32, 4>;
type MeasurementCovariance = SMatrix<f32, 4, 4>;
type MeasurementMatrix = SMatrix<f32, 4, 8>;

/// Position noise, relative to box height (ByteTrack: 1/20).
const STD_WEIGHT_POSITION: f32 = 1.0 / 20.0;
/// Velocity noise, relative to box height (ByteTrack: 1/160).
const STD_WEIGHT_VELOCITY: f32 = 1.0 / 160.0;

/// A Gaussian belief about a box: mean and covariance.
#[derive(Debug, Clone, PartialEq)]
pub struct KalmanState {
    pub mean: StateVector,
    pub covariance: StateCovariance,
}

impl KalmanState {
    /// The current box estimate as `[cx, cy, a, h]`.
    pub fn xyah(&self) -> [f32; 4] {
        [self.mean[0], self.mean[1], self.mean[2], self.mean[3]]
    }
}

/// The filter itself is stateless: it holds the constant model matrices,
/// while each track owns its own [`KalmanState`].
#[derive(Debug, Clone)]
pub struct KalmanFilter {
    /// State transition `F`: position += velocity (dt = 1 frame).
    motion: StateCovariance,
    /// Measurement matrix `H`: we observe the position part only.
    observation: MeasurementMatrix,
}

impl Default for KalmanFilter {
    fn default() -> Self {
        let mut motion = StateCovariance::identity();
        for i in 0..4 {
            motion[(i, 4 + i)] = 1.0;
        }
        let mut observation = MeasurementMatrix::zeros();
        for i in 0..4 {
            observation[(i, i)] = 1.0;
        }
        Self {
            motion,
            observation,
        }
    }
}

/// Diagonal matrix from standard deviations (variances = std²).
fn diag_from_std<const N: usize>(std: [f32; N]) -> SMatrix<f32, N, N> {
    SMatrix::from_diagonal(&SVector::from(std.map(|s| s * s)))
}

impl KalmanFilter {
    /// Starts a track from its first detection: known position, unknown
    /// velocity (zero mean, large variance).
    pub fn initiate(&self, xyah: [f32; 4]) -> KalmanState {
        let mut mean = StateVector::zeros();
        mean.fixed_rows_mut::<4>(0).copy_from_slice(&xyah);
        let h = xyah[3];
        let (p, v) = (STD_WEIGHT_POSITION * h, STD_WEIGHT_VELOCITY * h);
        let covariance = diag_from_std([
            2.0 * p,
            2.0 * p,
            1e-2,
            2.0 * p,
            10.0 * v,
            10.0 * v,
            1e-5,
            10.0 * v,
        ]);
        KalmanState { mean, covariance }
    }

    /// One step forward in time: `x' = F x`, `P' = F P Fᵀ + Q`.
    pub fn predict(&self, state: &KalmanState) -> KalmanState {
        let h = state.mean[3];
        let (p, v) = (STD_WEIGHT_POSITION * h, STD_WEIGHT_VELOCITY * h);
        let process_noise = diag_from_std([p, p, 1e-2, p, v, v, 1e-5, v]);
        KalmanState {
            mean: self.motion * state.mean,
            covariance: self.motion * state.covariance * self.motion.transpose() + process_noise,
        }
    }

    /// The state's belief expressed in measurement space (`H x`, `H P Hᵀ + R`).
    fn project(&self, state: &KalmanState) -> (Measurement, MeasurementCovariance) {
        let h = state.mean[3];
        let p = STD_WEIGHT_POSITION * h;
        let measurement_noise = diag_from_std([p, p, 1e-1, p]);
        (
            self.observation * state.mean,
            self.observation * state.covariance * self.observation.transpose() + measurement_noise,
        )
    }

    /// Corrects the state with a detection `[cx, cy, a, h]`.
    ///
    /// `K = P Hᵀ S⁻¹` is computed by solving with the Cholesky factor of the
    /// innovation covariance `S` (symmetric positive definite) instead of
    /// inverting it, which is cheaper and numerically safer.
    pub fn update(&self, state: &KalmanState, xyah: [f32; 4]) -> KalmanState {
        let (projected_mean, projected_cov) = self.project(state);
        let pht = state.covariance * self.observation.transpose(); // 8×4
        let gain = match projected_cov.cholesky() {
            // S Kᵀ = (P Hᵀ)ᵀ  →  Kᵀ = S⁻¹ (P Hᵀ)ᵀ
            Some(chol) => chol.solve(&pht.transpose()).transpose(),
            // S is a covariance plus positive noise, so this should not happen;
            // fall back to the explicit inverse rather than panicking.
            None => {
                pht * projected_cov
                    .try_inverse()
                    .unwrap_or_else(MeasurementCovariance::zeros)
            }
        };
        let innovation = Measurement::from(xyah) - projected_mean;
        KalmanState {
            mean: state.mean + gain * innovation,
            covariance: state.covariance - gain * projected_cov * gain.transpose(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_moves_the_box_by_its_velocity() {
        let kf = KalmanFilter::default();
        let mut state = kf.initiate([100.0, 50.0, 1.5, 20.0]);
        state.mean[4] = 3.0; // vx
        state.mean[5] = -1.0; // vy
        let next = kf.predict(&state);
        assert_eq!(&next.xyah()[..2], &[103.0, 49.0]);
        // Uncertainty grows when we only predict.
        assert!(next.covariance[(0, 0)] > state.covariance[(0, 0)]);
    }

    #[test]
    fn velocity_is_learned_from_a_moving_box() {
        let kf = KalmanFilter::default();
        let mut state = kf.initiate([0.0, 0.0, 1.0, 20.0]);
        for t in 1..=30 {
            state = kf.predict(&state);
            state = kf.update(&state, [2.0 * t as f32, t as f32, 1.0, 20.0]);
        }
        assert!((state.mean[4] - 2.0).abs() < 0.1, "vx = {}", state.mean[4]);
        assert!((state.mean[5] - 1.0).abs() < 0.1, "vy = {}", state.mean[5]);
        // Predicting ahead now extrapolates the motion.
        let ahead = kf.predict(&state);
        assert!((ahead.mean[0] - 62.0).abs() < 0.5, "cx = {}", ahead.mean[0]);
    }

    #[test]
    fn update_pulls_the_estimate_towards_the_measurement() {
        let kf = KalmanFilter::default();
        let state = kf.predict(&kf.initiate([10.0, 10.0, 1.0, 20.0]));
        let updated = kf.update(&state, [14.0, 10.0, 1.0, 20.0]);
        let cx = updated.mean[0];
        assert!(cx > 10.0 && cx < 14.0, "{cx}");
        // ...and makes it more certain.
        assert!(updated.covariance[(0, 0)] < state.covariance[(0, 0)]);
    }

    #[test]
    fn covariance_stays_symmetric() {
        let kf = KalmanFilter::default();
        let mut state = kf.initiate([5.0, 5.0, 0.5, 30.0]);
        for t in 0..20 {
            state = kf.update(&kf.predict(&state), [5.0 + t as f32, 5.0, 0.5, 30.0]);
        }
        let asym = (state.covariance - state.covariance.transpose())
            .abs()
            .max();
        assert!(asym < 1e-3, "{asym}");
    }
}
