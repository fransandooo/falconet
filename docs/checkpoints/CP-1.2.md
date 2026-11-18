## Checkpoint 1.2 — Reference detector

*Date: 2026-11-18 · Merged: PR #7 (model), PR #8 (Anti-UAV410 training) · Status: awaiting Fran's approval*

**Chosen architecture:** `CenterNetLite`, option (a) of the plan, written from scratch and trained from scratch, with no pretrained weights.
- **Size:** 1.39 M parameters.
- **Structure:** an anchor-free heatmap + size + offset design (CenterNet, 2019) on a small residual backbone with an FPN-lite neck.
- **Resolution:** output at stride 4. Details in `reference/README.md`.
- **Export:** the model exports to ONNX with six basic operators.

**Training:**
- **Run:** one overnight run (`baseline-e50`) on Anti-UAV410 train: 50 epochs × 40k random frames, bf16, on the laptop's RTX 3050 Ti.
- **Tracking:** everything logged to MLflow (experiment `reference-antiuav410`), with per-epoch prediction panels.
- **Full log:** `docs/experiments/2026-11-18-reference-antiuav410-baseline.md`.

**Results** (best epoch 32, **full val split**, 94,711 frames; test untouched):

| Metric | Value |
|---|---|
| AP50 | **0.841** |
| mAP@[.5:.95] | 0.460 |
| AP small / medium / large | **0.308** / 0.572 / 0.681 |
| mSA, detector only (no tracker) | **0.705** |
| Recall@0.5 of the top box, small / medium / large | 0.67 / 0.98 / 0.99 |
| False positives per frame (score ≥ 0.5) | 0.047 |

Three independent evaluators agree to 10⁻⁶:
- pycocotools and the Rust `coco_eval`;
- the official SA code and the Rust `antiuav410 sa`.

**Training cost:**
- **Time:** 4.6 h on a laptop GPU (~134 img/s, ~57 W GPU board power), about 0.26 kWh for the GPU alone, i.e. a few euro cents. No cloud.
- **Shorter schedule:** 97% of the final AP50 is reached after the first epoch (5 min), and the curve is flat from epoch ~5. A 15-epoch schedule would cost ~1.4 h for the same quality.

**What it means:**
- **Viable baseline.** The detector finds medium and large drones almost always. Small drones are the weak spot: recall 67%, and their boxes are loose (AP75 ≈ mAP). This is the plan's "drones too small" risk, but mild: AP_small is 0.31, not near zero.
- **Overfitting.** Mild, from epoch ~5: the val loss rises while AP stays flat. The checkpoint was selected on val AP50, so it isn't affected.
- **Tracker headroom.** The detector alone already gives mSA 0.705. ByteTrack should add continuity and a better "drone lost" decision; that is what CP 1.4 will measure.

**Decisions I need from you:**
1. *Accept this model as the phase 1 reference.* Alternatives: retrain with a shorter schedule and translation augmentation (one night), or try a stride-2 head for small drones (more VRAM, slower).
   **Recommendation: accept it now** and move on to task 4 (running it in Rust). The improvements are cheaper to try in phase 2, when training moves to Burn, and this model is the bar to beat.
2. *Weights.* They are derived from Anti-UAV410, so per the licence note they stay local (gitignored, in `runs/`) and are not published. **Recommendation: keep it that way.**

**Proposed next step:** Phase 1 task 4: load the ONNX model in Rust (Burn import, or a hand-written Burn model with the weights loaded) and check that its outputs match PyTorch on real frames. That leads to CP 1.3.
