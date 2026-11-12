//! Two classifiers: an MLP (dense layers only) and a small CNN.

use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{
    BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
};
use burn::prelude::*;
use burn::train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep};

use crate::data::{IMAGE_SIZE, MnistBatch};

const NUM_CLASSES: usize = 10;

/// What both models have in common: images in, one score (logit) per class out.
///
/// The training code only needs this, so it works with any classifier.
pub trait Classifier {
    /// `[batch, 28, 28]` → `[batch, 10]`
    fn logits(&self, images: Tensor<3>) -> Tensor<2>;
}

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------

/// Multi-layer perceptron: 784 → 256 → 128 → 10.
///
/// `#[derive(Module)]` is a *procedural macro*: at compile time it generates
/// the code that walks the struct's fields to find parameters (for the
/// optimizer), move them between devices, save/load them, and print the model.
/// In PyTorch the same happens at runtime by inspecting `nn.Module` attributes.
#[derive(Module, Debug)]
pub struct Mlp {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
    dropout: Dropout,
    activation: Relu,
}

impl Mlp {
    pub fn new(device: &Device) -> Self {
        Self {
            fc1: LinearConfig::new(IMAGE_SIZE * IMAGE_SIZE, 256).init(device),
            fc2: LinearConfig::new(256, 128).init(device),
            fc3: LinearConfig::new(128, NUM_CLASSES).init(device),
            dropout: DropoutConfig::new(0.2).init(),
            activation: Relu::new(),
        }
    }
}

impl Classifier for Mlp {
    fn logits(&self, images: Tensor<3>) -> Tensor<2> {
        let [batch, height, width] = images.dims();
        // Flatten each image into a vector: [batch, 784].
        let x = images.reshape([batch, height * width]);
        let x = self
            .dropout
            .forward(self.activation.forward(self.fc1.forward(x)));
        let x = self
            .dropout
            .forward(self.activation.forward(self.fc2.forward(x)));
        self.fc3.forward(x)
    }
}

// ---------------------------------------------------------------------------
// CNN
// ---------------------------------------------------------------------------

/// conv → batch norm → ReLU → 2x2 max-pool. The basic CNN building block, and
/// the same kind of block the real detector will be made of.
#[derive(Module, Debug)]
pub struct ConvBlock {
    conv: Conv2d,
    norm: BatchNorm,
    pool: MaxPool2d,
    activation: Relu,
}

impl ConvBlock {
    pub fn new(in_channels: usize, out_channels: usize, device: &Device) -> Self {
        Self {
            conv: Conv2dConfig::new([in_channels, out_channels], [3, 3])
                .with_padding(PaddingConfig2d::Valid)
                .init(device),
            norm: BatchNormConfig::new(out_channels).init(device),
            pool: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            activation: Relu::new(),
        }
    }

    /// `[batch, in, h, w]` → `[batch, out, (h - 2) / 2, (w - 2) / 2]`
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        self.pool.forward(
            self.activation
                .forward(self.norm.forward(self.conv.forward(x))),
        )
    }
}

/// Two conv blocks and two dense layers:
/// `[b,1,28,28]` → `[b,32,13,13]` → `[b,64,5,5]` → 128 → 10.
#[derive(Module, Debug)]
pub struct Cnn {
    block1: ConvBlock,
    block2: ConvBlock,
    fc1: Linear,
    fc2: Linear,
    dropout: Dropout,
    activation: Relu,
}

impl Cnn {
    pub fn new(device: &Device) -> Self {
        Self {
            block1: ConvBlock::new(1, 32, device),
            block2: ConvBlock::new(32, 64, device),
            fc1: LinearConfig::new(64 * 5 * 5, 128).init(device),
            fc2: LinearConfig::new(128, NUM_CLASSES).init(device),
            dropout: DropoutConfig::new(0.25).init(),
            activation: Relu::new(),
        }
    }
}

impl Classifier for Cnn {
    fn logits(&self, images: Tensor<3>) -> Tensor<2> {
        let [batch, height, width] = images.dims();
        // Conv layers expect a channel dimension: [batch, channels=1, h, w].
        let x = images.reshape([batch, 1, height, width]);
        let x = self.block2.forward(self.block1.forward(x));
        let [batch, channels, h, w] = x.dims();
        let x = x.reshape([batch, channels * h * w]);
        let x = self
            .dropout
            .forward(self.activation.forward(self.fc1.forward(x)));
        self.fc2.forward(x)
    }
}

// ---------------------------------------------------------------------------
// Training/validation steps
// ---------------------------------------------------------------------------

/// Loss + outputs in the format Burn's classification metrics expect.
fn classification_output(logits: Tensor<2>, targets: Tensor<1, Int>) -> ClassificationOutput {
    let loss = CrossEntropyLossConfig::new()
        .init(&logits.device())
        .forward(logits.clone(), targets.clone());
    ClassificationOutput::new(loss, logits, targets)
}

/// Implements Burn's `TrainStep` and `InferenceStep` for a [`Classifier`].
///
/// Why a macro and not `impl<T: Classifier> TrainStep for T`? Rust's *orphan
/// rule*: you may only implement a trait for a type if the trait or the type
/// belongs to your crate. `TrainStep` is Burn's, and a blanket `T` could be
/// anyone's type, so that impl is rejected (error E0210). `macro_rules!`
/// sidesteps it by writing out one concrete impl per model at compile time,
/// like a C preprocessor macro, but operating on syntax trees instead of text.
macro_rules! impl_classifier_steps {
    ($model:ty) => {
        impl TrainStep for $model {
            type Input = MnistBatch;
            type Output = ClassificationOutput;

            fn step(&self, batch: MnistBatch) -> TrainOutput<ClassificationOutput> {
                let output = classification_output(self.logits(batch.images), batch.targets);
                // `backward` runs autodiff from the loss and returns the gradients.
                TrainOutput::new(self, output.loss.backward(), output)
            }
        }

        impl InferenceStep for $model {
            type Input = MnistBatch;
            type Output = ClassificationOutput;

            fn step(&self, batch: MnistBatch) -> ClassificationOutput {
                classification_output(self.logits(batch.images), batch.targets)
            }
        }
    };
}

impl_classifier_steps!(Mlp);
impl_classifier_steps!(Cnn);

#[cfg(test)]
mod tests {
    use super::*;

    fn logits_shape(model: &impl Classifier, device: &Device) -> [usize; 2] {
        let images = Tensor::<3>::zeros([4, IMAGE_SIZE, IMAGE_SIZE], device);
        model.logits(images).dims()
    }

    #[test]
    fn both_models_output_one_logit_per_class() {
        let device = Device::flex();
        assert_eq!(logits_shape(&Mlp::new(&device), &device), [4, NUM_CLASSES]);
        assert_eq!(logits_shape(&Cnn::new(&device), &device), [4, NUM_CLASSES]);
    }

    #[test]
    fn the_cnn_has_more_structure_but_fewer_parameters() {
        let device = Device::flex();
        let mlp = Mlp::new(&device).num_params();
        let cnn = Cnn::new(&device).num_params();
        // MLP: 784*256 + 256 + 256*128 + 128 + 128*10 + 10 = 235_146
        assert_eq!(mlp, 235_146);
        assert!(cnn < mlp, "cnn {cnn} vs mlp {mlp}");
    }
}
