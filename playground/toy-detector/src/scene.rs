//! Synthetic scenes: one bright box on a noisy, uneven background, with distractors.
//!
//! A crude stand-in for a drone in a thermal image: a small warm blob over a
//! background with a temperature gradient and sensor noise.

use playground_annotations::BBox;

use crate::rng::SplitMix64;

/// Knobs of the scene generator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneConfig {
    /// Images are `image_size` x `image_size` pixels.
    pub image_size: usize,
    /// Box sides are drawn uniformly from `[min_side, max_side]` pixels.
    pub min_side: f32,
    pub max_side: f32,
    /// Standard deviation of the per-pixel Gaussian noise.
    pub noise_std: f32,
    /// The box is brighter than the background by `[min_contrast, max_contrast]`.
    pub min_contrast: f32,
    pub max_contrast: f32,
    /// Up to this many soft Gaussian blobs ("clouds", "birds") that are *not*
    /// the target. Some are brighter than the box, so the model has to learn
    /// the target's shape, not just "find the brightest spot".
    pub max_distractors: usize,
    pub max_distractor_amplitude: f32,
}

impl Default for SceneConfig {
    fn default() -> Self {
        Self {
            image_size: 64,
            min_side: 3.0,
            max_side: 14.0,
            noise_std: 0.18,
            min_contrast: 0.15,
            max_contrast: 0.5,
            max_distractors: 3,
            max_distractor_amplitude: 0.45,
        }
    }
}

/// One generated image and its ground-truth box.
#[derive(Debug, Clone, PartialEq)]
pub struct Scene {
    /// Row-major grayscale pixels in `[0, 1]`, `image_size * image_size` of them.
    pub pixels: Vec<f32>,
    pub bbox: BBox,
}

impl SceneConfig {
    /// Generates the scene for `seed`. Same seed → same scene, always.
    pub fn generate(&self, seed: u64) -> Scene {
        let mut rng = SplitMix64::new(seed);
        let n = self.image_size;
        let size = n as f32;

        // Integer-aligned box, fully inside the image.
        let w = rng.range(self.min_side, self.max_side + 1.0).floor();
        let h = rng.range(self.min_side, self.max_side + 1.0).floor();
        let x = rng.range(0.0, size - w + 1.0).floor();
        let y = rng.range(0.0, size - h + 1.0).floor();
        let bbox = BBox::new(x, y, w, h);

        // Background: a base level plus a linear gradient (like uneven sky temperature).
        let base = rng.range(0.15, 0.45);
        let grad_x = rng.range(-0.25, 0.25) / size;
        let grad_y = rng.range(-0.25, 0.25) / size;
        let contrast = rng.range(self.min_contrast, self.max_contrast);

        // Distractors: (centre x, centre y, sigma, amplitude).
        let n_distractors = (rng.uniform() * (self.max_distractors + 1) as f32) as usize;
        let distractors: Vec<(f32, f32, f32, f32)> = (0..n_distractors)
            .map(|_| {
                (
                    rng.range(0.0, size),
                    rng.range(0.0, size),
                    rng.range(1.5, 4.0),
                    rng.range(0.1, self.max_distractor_amplitude),
                )
            })
            .collect();

        let mut pixels = Vec::with_capacity(n * n);
        for row in 0..n {
            for col in 0..n {
                let (r, c) = (row as f32, col as f32);
                let mut value = base
                    + grad_x * (c - size / 2.0)
                    + grad_y * (r - size / 2.0)
                    + self.noise_std * rng.normal();
                for &(dx, dy, sigma, amplitude) in &distractors {
                    let d2 = (c - dx).powi(2) + (r - dy).powi(2);
                    value += amplitude * (-d2 / (2.0 * sigma * sigma)).exp();
                }
                if c >= x && c < x + w && r >= y && r < y + h {
                    value += contrast;
                }
                pixels.push(value.clamp(0.0, 1.0));
            }
        }
        Scene { pixels, bbox }
    }
}

/// Spreads dataset indices over the seed space so neighbouring scenes are
/// unrelated, and keeps different splits (different `base`) disjoint.
pub fn scene_seed(base: u64, index: usize) -> u64 {
    SplitMix64::new(base ^ (index as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)).next_u64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scenes_are_deterministic() {
        let cfg = SceneConfig::default();
        assert_eq!(cfg.generate(42), cfg.generate(42));
        assert_ne!(cfg.generate(42), cfg.generate(43));
    }

    #[test]
    fn boxes_stay_inside_the_image_and_within_size_limits() {
        let cfg = SceneConfig::default();
        for seed in 0..500 {
            let scene = cfg.generate(scene_seed(1, seed));
            let b = scene.bbox;
            assert!(b.x >= 0.0 && b.y >= 0.0, "{b:?}");
            assert!(b.right() <= 64.0 && b.bottom() <= 64.0, "{b:?}");
            assert!(
                (3.0..=14.0).contains(&b.w) && (3.0..=14.0).contains(&b.h),
                "{b:?}"
            );
            assert_eq!(scene.pixels.len(), 64 * 64);
        }
    }

    #[test]
    fn the_box_is_brighter_than_its_surroundings_on_average() {
        let cfg = SceneConfig::default();
        let scene = cfg.generate(5);
        let b = scene.bbox;
        let (mut inside, mut n_in, mut outside, mut n_out) = (0.0, 0, 0.0, 0);
        for (i, &p) in scene.pixels.iter().enumerate() {
            let (r, c) = ((i / 64) as f32, (i % 64) as f32);
            if c >= b.x && c < b.right() && r >= b.y && r < b.bottom() {
                inside += p;
                n_in += 1;
            } else {
                outside += p;
                n_out += 1;
            }
        }
        assert!(inside / n_in as f32 > outside / n_out as f32 + 0.1);
    }
}
