//! From dataset items to batched tensors.

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::vision::MnistItem;
use burn::prelude::*;

/// Side length of an MNIST image.
pub const IMAGE_SIZE: usize = 28;
/// Mean and standard deviation of MNIST pixels in `[0, 1]` (the usual values
/// from the PyTorch MNIST example), used to normalise inputs.
const PIXEL_MEAN: f32 = 0.1307;
const PIXEL_STD: f32 = 0.3081;

/// A batch ready for the model.
///
/// `Tensor<3>` is a float tensor of rank 3: `[batch, height, width]`.
/// `Tensor<1, Int>` is an integer tensor of rank 1: one class label per image.
/// The rank is part of the type, so mixing up dimensions is a compile error.
#[derive(Clone, Debug)]
pub struct MnistBatch {
    pub images: Tensor<3>,
    pub targets: Tensor<1, Int>,
}

/// Turns a `Vec` of dataset items into one [`MnistBatch`].
///
/// The data loader calls `batch` from its worker threads, which is why
/// `Batcher` implementations must be `Clone + Send + Sync`. A unit struct
/// (no fields) trivially is.
#[derive(Clone, Debug, Default)]
pub struct MnistBatcher;

impl Batcher<MnistItem, MnistBatch> for MnistBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &Device) -> MnistBatch {
        let n = items.len();

        // Normalise on the CPU and upload the whole batch in one go: one
        // host→GPU copy per batch instead of one tiny copy per image.
        let pixels: Vec<f32> = items
            .iter()
            .flat_map(|item| item.image.iter().flatten())
            .map(|&p| (p / 255.0 - PIXEL_MEAN) / PIXEL_STD)
            .collect();
        let labels: Vec<i64> = items.iter().map(|item| i64::from(item.label)).collect();

        MnistBatch {
            images: Tensor::from_data(TensorData::new(pixels, [n, IMAGE_SIZE, IMAGE_SIZE]), device),
            targets: Tensor::from_data(TensorData::new(labels, [n]), device),
        }
    }
}
