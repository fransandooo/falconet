//! Sanity check for the development environment.
//!
//! Runs the same matrix multiplication on three Burn backends:
//! - `flex`: pure-Rust CPU backend (the numerical reference),
//! - `wgpu`: portable GPU backend (Vulkan on Linux, WebGPU in the browser),
//! - `cuda`: NVIDIA-only GPU backend.
//!
//! For each backend it checks that the result matches the CPU reference and
//! reports a rough throughput number. This is a smoke test, not a benchmark.
//!
//! Note: since Burn 0.22 a `Tensor` is no longer generic over the backend. The
//! backend travels inside the `Device` value and is chosen at runtime.

use std::time::Instant;

use burn::tensor::{Device, DeviceKind, Tensor, TensorData};

/// Matrix side length used for the check.
const N: usize = 1024;
/// Timed iterations per GPU backend (after warm-up).
const GPU_ITERS: usize = 20;
/// Timed iterations on the CPU (much slower).
const CPU_ITERS: usize = 3;
/// Accepted relative error vs the CPU reference (loose enough for TF32).
const REL_TOLERANCE: f32 = 5e-3;

fn main() {
    // Same input data for every backend, generated deterministically on the host.
    let a = pseudo_random_matrix(N, 1);
    let b = pseudo_random_matrix(N, 2);

    println!("matmul {N}x{N} f32\n");

    let reference = run("flex (CPU)", &Device::flex(), &a, &b, CPU_ITERS);
    let wgpu = run(
        "wgpu (Vulkan)",
        &Device::wgpu(DeviceKind::DiscreteGpu(0)),
        &a,
        &b,
        GPU_ITERS,
    );
    let cuda = run("cuda", &Device::cuda(0), &a, &b, GPU_ITERS);

    // Errors are relative to the largest reference value. On NVIDIA tensor
    // cores Burn's matmul autotuner may compute f32 matmuls in TF32 (10-bit
    // mantissa), so ~1e-3 relative error is expected there; plain f32 gives ~1e-6.
    let scale = reference.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    println!();
    for (name, result) in [("wgpu", &wgpu), ("cuda", &cuda)] {
        let rel = max_abs_diff(&reference, result) / scale;
        let status = if rel < REL_TOLERANCE {
            "OK"
        } else {
            "MISMATCH"
        };
        println!("{name:>5} vs flex: max relative error = {rel:.2e}  [{status}]");
    }
}

/// Runs `a @ b` on `device`, prints timing, and returns the result on the host.
fn run(label: &str, device: &Device, a: &[f32], b: &[f32], iters: usize) -> Vec<f32> {
    let lhs = Tensor::<2>::from_data(TensorData::new(a.to_vec(), [N, N]), device);
    let rhs = Tensor::<2>::from_data(TensorData::new(b.to_vec(), [N, N]), device);

    // Warm-up: the first call compiles/autotunes GPU kernels and is much slower.
    let start = Instant::now();
    let out = lhs.clone().matmul(rhs.clone());
    device.sync().expect("device sync failed");
    let first_ms = start.elapsed().as_secs_f64() * 1e3;

    // GPU work is asynchronous: without `sync` we would only time the enqueue.
    let start = Instant::now();
    for _ in 0..iters {
        let _ = lhs.clone().matmul(rhs.clone());
    }
    device.sync().expect("device sync failed");
    let avg_ms = start.elapsed().as_secs_f64() * 1e3 / iters as f64;

    let gflops = 2.0 * (N as f64).powi(3) / (avg_ms * 1e-3) / 1e9;
    println!(
        "{label:<14} first call {first_ms:>8.1} ms | avg {avg_ms:>7.2} ms | {gflops:>7.1} GFLOP/s"
    );

    out.into_data()
        .try_to_vec::<f32>()
        .expect("matmul output should be f32")
}

/// Deterministic values in [-1, 1) from a tiny LCG, so no `rand` dependency.
fn pseudo_random_matrix(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n * n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn max_abs_diff(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}
