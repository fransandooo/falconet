//! The network: a small fully-convolutional backbone and three 1x1 heads.

use burn::module::Param;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, PaddingConfig2d, Relu};
use burn::prelude::*;

/// Initial heatmap bias: `sigmoid(-2.19) ≈ 0.1`. Starting from "almost no
/// object anywhere" keeps the huge number of background cells from swamping
/// the first gradient steps (trick from the RetinaNet/CenterNet papers).
const HEATMAP_PRIOR_BIAS: f32 = -2.19;

/// 3x3 convolution → batch norm → ReLU.
#[derive(Module, Debug)]
pub struct ConvBnRelu {
    conv: Conv2d,
    norm: BatchNorm,
    activation: Relu,
}

impl ConvBnRelu {
    pub fn new(channels_in: usize, channels_out: usize, stride: usize, device: &Device) -> Self {
        Self {
            conv: Conv2dConfig::new([channels_in, channels_out], [3, 3])
                .with_stride([stride, stride])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            norm: BatchNormConfig::new(channels_out).init(device),
            activation: Relu::new(),
        }
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        self.activation
            .forward(self.norm.forward(self.conv.forward(x)))
    }
}

/// Raw network outputs, all on the `[batch, _, grid, grid]` output grid.
#[derive(Debug, Clone)]
pub struct DetectorOutput {
    /// Heatmap logits (apply sigmoid for probabilities). `[b, 1, g, g]`
    pub heatmap_logits: Tensor<4>,
    /// Box width and height in grid cells. `[b, 2, g, g]`
    pub size: Tensor<4>,
    /// Sub-cell offset logits (apply sigmoid for `[0, 1]`). `[b, 2, g, g]`
    pub offset_logits: Tensor<4>,
}

/// `[b, 1, 64, 64]` → stem → ↓2 → ↓2 → two more blocks → heads on `[b, _, 16, 16]`.
///
/// Receptive field: 3 → 5 → 9 → 17 → 25 px, enough to see the largest (14 px)
/// boxes plus some context around them.
#[derive(Module, Debug)]
pub struct ToyDetector {
    stem: ConvBnRelu,
    down1: ConvBnRelu,
    down2: ConvBnRelu,
    block1: ConvBnRelu,
    block2: ConvBnRelu,
    heatmap_head: Conv2d,
    size_head: Conv2d,
    offset_head: Conv2d,
}

impl ToyDetector {
    pub fn new(device: &Device) -> Self {
        let head = |channels_out| Conv2dConfig::new([64, channels_out], [1, 1]).init(device);

        let mut heatmap_head = head(1);
        // Fields of Burn's `Conv2d` are public, so we can replace the bias
        // parameter after initialisation.
        heatmap_head.bias = Some(Param::from_tensor(Tensor::full(
            [1],
            HEATMAP_PRIOR_BIAS,
            device,
        )));

        Self {
            stem: ConvBnRelu::new(1, 16, 1, device),
            down1: ConvBnRelu::new(16, 32, 2, device),
            down2: ConvBnRelu::new(32, 64, 2, device),
            block1: ConvBnRelu::new(64, 64, 1, device),
            block2: ConvBnRelu::new(64, 64, 1, device),
            heatmap_head,
            size_head: head(2),
            offset_head: head(2),
        }
    }

    pub fn forward(&self, images: Tensor<4>) -> DetectorOutput {
        let x = self.stem.forward(images);
        let x = self.down1.forward(x);
        let x = self.down2.forward(x);
        let x = self.block1.forward(x);
        let features = self.block2.forward(x);
        DetectorOutput {
            heatmap_logits: self.heatmap_head.forward(features.clone()),
            size: self.size_head.forward(features.clone()),
            offset_logits: self.offset_head.forward(features),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heads_live_on_a_grid_four_times_coarser() {
        let device = Device::flex();
        let out = ToyDetector::new(&device).forward(Tensor::zeros([2, 1, 64, 64], &device));
        assert_eq!(out.heatmap_logits.dims(), [2, 1, 16, 16]);
        assert_eq!(out.size.dims(), [2, 2, 16, 16]);
        assert_eq!(out.offset_logits.dims(), [2, 2, 16, 16]);
    }
}
