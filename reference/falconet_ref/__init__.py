"""PyTorch reference detector for falconet (CenterNet-style, single class).

Trained here, exported to ONNX, then run (and later re-implemented) in Rust.
Everything is kept to simple ops (conv, batch norm, ReLU, nearest upsampling,
add) so that the Burn port in phase 1 task 4 stays straightforward.
"""
