//! Dataset, batching and **target encoding**: turning a box into what the
//! network is trained to output.

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::{Dataset, DatasetError};
use burn::prelude::*;
use playground_annotations::BBox;

use crate::STRIDE;
use crate::scene::{Scene, SceneConfig, scene_seed};

/// An infinite family of scenes, of which we use the first `len`.
///
/// Nothing is stored: `get(i)` regenerates scene `i` from its seed. That is
/// why the dataset can be cloned into every data-loader worker for free.
#[derive(Debug, Clone)]
pub struct SceneDataset {
    config: SceneConfig,
    base_seed: u64,
    len: usize,
    /// Number of *different* scenes; indices wrap around after that.
    distinct: usize,
}

impl SceneDataset {
    /// `len` scenes, all different.
    pub fn new(config: SceneConfig, base_seed: u64, len: usize) -> Self {
        Self::repeating(config, base_seed, len, len)
    }

    /// `len` items cycling through only `distinct` different scenes.
    pub fn repeating(config: SceneConfig, base_seed: u64, len: usize, distinct: usize) -> Self {
        Self {
            config,
            base_seed,
            len,
            distinct: distinct.max(1),
        }
    }
}

impl Dataset<Scene> for SceneDataset {
    fn get(&self, index: usize) -> Result<Scene, DatasetError> {
        if index >= self.len {
            return Err(DatasetError::new(std::io::Error::other(format!(
                "scene index {index} out of range ({})",
                self.len
            ))));
        }
        let scene_index = index % self.distinct;
        Ok(self
            .config
            .generate(scene_seed(self.base_seed, scene_index)))
    }

    fn len(&self) -> usize {
        self.len
    }
}

/// Training targets for one box, on a `grid` x `grid` output map.
///
/// All maps are row-major `grid * grid` (`size` and `offset` have 2 channels).
#[derive(Debug, Clone, PartialEq)]
pub struct Targets {
    /// Gaussian bump around the centre cell, exactly 1.0 at the centre.
    pub heatmap: Vec<f32>,
    /// 1.0 at the centre cell, 0.0 elsewhere: where size/offset are supervised.
    pub center_mask: Vec<f32>,
    /// Channel 0 = width, channel 1 = height, in grid cells, at the centre cell.
    pub size: Vec<f32>,
    /// Channel 0 = x, channel 1 = y sub-cell offset in `[0, 1)`, at the centre cell.
    pub offset: Vec<f32>,
}

/// Encodes a ground-truth box as CenterNet-style targets.
pub fn encode_targets(bbox: &BBox, grid: usize) -> Targets {
    let stride = STRIDE as f32;
    let (cx, cy) = bbox.center();
    let (gx, gy) = (cx / stride, cy / stride); // centre in grid units
    let col = (gx.floor() as usize).min(grid - 1);
    let row = (gy.floor() as usize).min(grid - 1);

    // The bump's width grows with the box: being one cell off matters less
    // for a big box than for a tiny one.
    let mean_side_cells = (bbox.w + bbox.h) / 2.0 / stride;
    let sigma = (mean_side_cells / 3.0).max(0.5);

    let cells = grid * grid;
    let mut heatmap = vec![0.0; cells];
    for r in 0..grid {
        for c in 0..grid {
            let d2 = (c as f32 - col as f32).powi(2) + (r as f32 - row as f32).powi(2);
            heatmap[r * grid + c] = (-d2 / (2.0 * sigma * sigma)).exp();
        }
    }

    let center = row * grid + col;
    let mut center_mask = vec![0.0; cells];
    center_mask[center] = 1.0;

    let mut size = vec![0.0; 2 * cells];
    size[center] = bbox.w / stride;
    size[cells + center] = bbox.h / stride;

    let mut offset = vec![0.0; 2 * cells];
    offset[center] = gx - col as f32;
    offset[cells + center] = gy - row as f32;

    Targets {
        heatmap,
        center_mask,
        size,
        offset,
    }
}

/// A batch of scenes and their encoded targets.
#[derive(Debug, Clone)]
pub struct DetectionBatch {
    /// `[batch, 1, image_size, image_size]`
    pub images: Tensor<4>,
    /// `[batch, 1, grid, grid]`
    pub heatmap: Tensor<4>,
    /// `[batch, 1, grid, grid]`
    pub center_mask: Tensor<4>,
    /// `[batch, 2, grid, grid]`
    pub size: Tensor<4>,
    /// `[batch, 2, grid, grid]`
    pub offset: Tensor<4>,
    /// Ground truth in pixels, kept on the CPU for metrics and drawing.
    pub boxes: Vec<BBox>,
}

#[derive(Debug, Clone)]
pub struct DetectionBatcher {
    image_size: usize,
}

impl DetectionBatcher {
    pub fn new(config: &SceneConfig) -> Self {
        Self {
            image_size: config.image_size,
        }
    }

    pub fn grid(&self) -> usize {
        self.image_size / STRIDE
    }
}

impl Batcher<Scene, DetectionBatch> for DetectionBatcher {
    fn batch(&self, scenes: Vec<Scene>, device: &Device) -> DetectionBatch {
        let (b, s, g) = (scenes.len(), self.image_size, self.grid());

        let mut pixels = Vec::with_capacity(b * s * s);
        let (mut heat, mut mask, mut size, mut offset) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for scene in &scenes {
            pixels.extend_from_slice(&scene.pixels);
            let t = encode_targets(&scene.bbox, g);
            heat.extend(t.heatmap);
            mask.extend(t.center_mask);
            size.extend(t.size);
            offset.extend(t.offset);
        }

        // A small helper closure: host Vec → device tensor of a given shape.
        let upload = |data: Vec<f32>, channels: usize, side: usize| {
            Tensor::<4>::from_data(TensorData::new(data, [b, channels, side, side]), device)
        };
        DetectionBatch {
            images: upload(pixels, 1, s),
            heatmap: upload(heat, 1, g),
            center_mask: upload(mask, 1, g),
            size: upload(size, 2, g),
            offset: upload(offset, 2, g),
            boxes: scenes.iter().map(|scene| scene.bbox).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn targets_peak_at_the_centre_cell() {
        // Centre (10, 6) px → grid (2.5, 1.5) → cell (col 2, row 1), offset (0.5, 0.5).
        let t = encode_targets(&BBox::new(6.0, 2.0, 8.0, 8.0), 16);
        let center = 16 + 2;
        assert_eq!(t.heatmap[center], 1.0);
        assert_eq!(t.center_mask.iter().sum::<f32>(), 1.0);
        assert_eq!(t.center_mask[center], 1.0);
        assert_eq!((t.size[center], t.size[256 + center]), (2.0, 2.0));
        assert_eq!((t.offset[center], t.offset[256 + center]), (0.5, 0.5));
        assert!(t.heatmap.iter().all(|&v| v <= 1.0));
    }

    #[test]
    fn dataset_regenerates_the_same_scene_for_the_same_index() {
        let ds = SceneDataset::new(SceneConfig::default(), 3, 10);
        assert_eq!(ds.get(4).unwrap(), ds.get(4).unwrap());
        assert!(ds.get(10).is_err());
    }

    #[test]
    fn repeating_dataset_cycles_through_its_scenes() {
        let ds = SceneDataset::repeating(SceneConfig::default(), 3, 10, 4);
        assert_eq!(ds.len(), 10);
        assert_eq!(ds.get(1).unwrap(), ds.get(5).unwrap());
        assert_ne!(ds.get(1).unwrap(), ds.get(2).unwrap());
    }

    #[test]
    fn batches_have_the_expected_shapes() {
        let cfg = SceneConfig::default();
        let scenes: Vec<Scene> = (0..3).map(|i| cfg.generate(i)).collect();
        let batch = DetectionBatcher::new(&cfg).batch(scenes, &Device::flex());
        assert_eq!(batch.images.dims(), [3, 1, 64, 64]);
        assert_eq!(batch.heatmap.dims(), [3, 1, 16, 16]);
        assert_eq!(batch.size.dims(), [3, 2, 16, 16]);
        assert_eq!(batch.boxes.len(), 3);
    }
}
