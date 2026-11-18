"""Unit tests for the reference detector (CPU, fast)."""

from __future__ import annotations

import math

import numpy as np
import torch

from falconet_ref.decode import decode
from falconet_ref.loss import detection_loss
from falconet_ref.model import STRIDE, CenterNetLite
from falconet_ref.synthetic import scene
from falconet_ref.targets import encode, gaussian_radius


def test_outputs_live_on_a_stride_4_grid():
    model = CenterNetLite().eval()
    heat, log_size, offset = model(torch.rand(2, 1, 512, 640))
    assert heat.shape == (2, 1, 128, 160)
    assert log_size.shape == offset.shape == (2, 2, 128, 160)


def test_targets_peak_at_the_centre_cell():
    t = encode(np.array([[100.0, 40.0, 20.0, 12.0]]), 128, 160)
    # centre (110, 46) px -> grid (27.5, 11.5) -> cell (col 27, row 11)
    assert t["heatmap"][0, 11, 27] == 1.0
    assert t["mask"].sum() == 1.0
    assert np.allclose(t["log_size"][:, 11, 27], [math.log(20), math.log(12)])
    assert np.allclose(t["offset"][:, 11, 27], [0.5, 0.5])


def test_frames_without_objects_have_empty_targets():
    t = encode(np.zeros((0, 4)), 64, 64)
    assert t["heatmap"].sum() == 0 and t["mask"].sum() == 0


def test_gaussian_radius_grows_with_box_size():
    assert gaussian_radius(2, 2) < gaussian_radius(10, 10) < gaussian_radius(40, 40)


def to_outputs(t: dict[str, np.ndarray]):
    """Perfect network outputs for the given targets (logits)."""
    heat = torch.from_numpy(t["heatmap"])[None].clamp(1e-6, 1 - 1e-6)
    off = torch.from_numpy(t["offset"])[None].clamp(1e-6, 1 - 1e-6)
    return torch.logit(heat), torch.from_numpy(t["log_size"])[None], torch.logit(off)


def test_decoding_perfect_outputs_recovers_the_box():
    box = np.array([[37.0, 21.0, 9.0, 6.0]])
    (dets,) = decode(to_outputs(encode(box, 64, 64)), top_k=1)
    x, y, w, h, score = dets[0]
    assert np.allclose([x, y, w, h], box[0], atol=1e-3)
    assert score > 0.99


def test_loss_is_small_for_perfect_outputs_and_decreases_when_training():
    img, boxes = scene(0, seed=1, height=64, width=64)
    t = {k: torch.from_numpy(v)[None] for k, v in encode(boxes, 64, 64).items()}
    perfect = detection_loss(to_outputs({k: v[0].numpy() for k, v in t.items()}), t)["total"]
    assert float(perfect) < 0.05

    torch.manual_seed(0)
    model = CenterNetLite()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.from_numpy(img)[None, None]
    first = detection_loss(model(x), t)["total"].item()
    for _ in range(30):
        loss = detection_loss(model(x), t)["total"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.5 * first


def test_stride_constant_matches_the_model():
    assert STRIDE == 4
