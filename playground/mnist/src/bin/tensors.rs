//! A guided tour of Burn tensors: creation, shapes, broadcasting, reductions,
//! matmul, ownership, autodiff and getting data back to the CPU.
//!
//! ```text
//! cargo run --release -p playground-mnist --bin tensors
//! ```

use burn::prelude::*;

fn main() {
    // `Device::default()` picks the first enabled backend in priority order
    // (GPUs before CPUs), unless the `BURN_DEVICE` environment variable says otherwise.
    let device = Device::default();
    println!("device: {device:?}");

    section("1. Creation and shapes");
    let a = Tensor::<2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    println!("a = {a}");
    println!("a.dims() = {:?}", a.dims()); // [2, 3]
    let ids = Tensor::<1, Int>::arange(0..6, &device).reshape([2, 3]);
    println!("arange(0..6).reshape([2, 3]) = {ids}");
    let zeros = Tensor::<3>::zeros([2, 1, 4], &device);
    println!("zeros([2, 1, 4]).dims() = {:?}", zeros.dims());

    section("2. Element-wise ops and broadcasting");
    // Ops on tensors take `self` by value (they consume the tensor), so we
    // `clone()` to keep using `a`. Cloning is cheap: it copies a handle and
    // bumps a reference count; the data on the GPU is not duplicated.
    let doubled = a.clone() * 2.0;
    println!("a * 2 = {doubled}");
    // A [1, 3] row is broadcast over both rows of the [2, 3] tensor, like NumPy.
    let row = Tensor::<2>::from_data([[10.0, 20.0, 30.0]], &device);
    println!("a + [[10, 20, 30]] = {}", a.clone() + row);

    section("3. Reductions");
    println!("a.sum() = {}", a.clone().sum()); // rank 1, one element
    // `mean_dim(1)` keeps the reduced dimension with size 1: [2, 3] → [2, 1].
    println!("a.mean_dim(1) = {}", a.clone().mean_dim(1));
    println!("a.argmax(1) = {}", a.clone().argmax(1));

    section("4. Matrix multiplication");
    // [2, 3] @ [3, 2] → [2, 2]
    let gram = a.clone().matmul(a.clone().transpose());
    println!("a @ aᵀ = {gram}");

    section("5. Autodiff");
    // On an autodiff device, operations are recorded so gradients can be computed.
    let ad = device.clone().autodiff();
    let x = Tensor::<1>::from_data([1.0, 2.0, 3.0], &ad).require_grad();
    // y = Σ x², so dy/dx = 2x = [2, 4, 6].
    let y = (x.clone() * x.clone()).sum();
    let grads = y.backward();
    let dx = x.grad(&grads).expect("x requires grad");
    println!("y = Σ x² = {y}");
    println!("dy/dx = {dx}  (expected 2x = [2, 4, 6])");

    section("6. Back to the CPU");
    let values: Vec<f32> = dx.into_data().try_to_vec().expect("f32 tensor");
    println!("as Vec<f32>: {values:?}");
    assert_eq!(values, vec![2.0, 4.0, 6.0]);
    println!("\ngradient check passed ✔");
}

fn section(title: &str) {
    println!("\n── {title} ──");
}
