"""Export the reference detector to ONNX and check it against PyTorch.

    uv run python -m falconet_ref.export_onnx --weights ../runs/reference/synthetic/model.pt

The ONNX graph takes `image` [1, 1, 512, 640] (raw pixels in [0, 1]) and
returns `heatmap`, `log_size`, `offset` at stride 4. ONNX Runtime's outputs
are compared with PyTorch's on random inputs before declaring success.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .model import CenterNetLite


def export(model: CenterNetLite, path: Path, height: int = 512, width: int = 640, opset: int = 17) -> None:
    model.eval()
    dummy = torch.rand(1, 1, height, width)
    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        input_names=["image"],
        output_names=["heatmap", "log_size", "offset"],
        opset_version=opset,
        dynamo=True,
        external_data=False,
    )
    onnx.checker.check_model(onnx.load(str(path)))


def max_abs_difference(model: CenterNetLite, path: Path, height: int = 512, width: int = 640, n: int = 3) -> float:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    worst = 0.0
    rng = np.random.default_rng(0)
    model.eval()
    for _ in range(n):
        x = rng.random((1, 1, height, width), dtype=np.float32)
        with torch.no_grad():
            ref = [t.numpy() for t in model(torch.from_numpy(x))]
        got = session.run(None, {"image": x})
        worst = max(worst, *(float(np.abs(a - b).max()) for a, b in zip(ref, got)))
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, help="state dict from train.py (random init if omitted)")
    parser.add_argument("--out", type=Path, default=Path("../runs/reference/centernet_lite.onnx"))
    args = parser.parse_args()

    model = CenterNetLite()
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    export(model, args.out)
    ops = sorted({node.op_type for node in onnx.load(str(args.out)).graph.node})
    diff = max_abs_difference(model, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    print(f"ONNX ops: {', '.join(ops)}")
    print(f"max |ONNX Runtime - PyTorch| = {diff:.2e}")


if __name__ == "__main__":
    main()
