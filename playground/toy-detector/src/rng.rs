//! A tiny deterministic random number generator.
//!
//! Synthetic data must be reproducible: scene `i` must look the same in every
//! run and on every machine. A seeded PRNG gives exactly that. We write our own
//! instead of adding a dependency because SplitMix64 is ~10 lines.

/// SplitMix64 (Steele, Lea & Flood, 2014): 8 bytes of state, good statistical
/// quality for simulations, **not** for cryptography.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next 64 random bits. `wrapping_*` makes overflow explicit: in Rust,
    /// integer overflow panics in debug builds instead of being silent like in C.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`, using the top 24 bits (an `f32` mantissa's worth).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in `[lo, hi)`.
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.uniform()
    }

    /// Standard normal sample (Box–Muller transform).
    pub fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(f32::MIN_POSITIVE); // avoid ln(0)
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_sequence() {
        let mut a = SplitMix64::new(7);
        let mut b = SplitMix64::new(7);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn uniform_and_normal_have_the_right_moments() {
        let mut rng = SplitMix64::new(1);
        let n = 100_000;
        let mean_u = (0..n).map(|_| rng.uniform()).sum::<f32>() / n as f32;
        let normals: Vec<f32> = (0..n).map(|_| rng.normal()).collect();
        let mean_n = normals.iter().sum::<f32>() / n as f32;
        let var_n = normals.iter().map(|x| (x - mean_n).powi(2)).sum::<f32>() / n as f32;
        assert!((mean_u - 0.5).abs() < 0.01, "{mean_u}");
        assert!(mean_n.abs() < 0.02, "{mean_n}");
        assert!((var_n - 1.0).abs() < 0.03, "{var_n}");
    }
}
