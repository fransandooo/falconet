# Reference detector on Anti-UAV410: first baseline

- **Date:** 2026-11-18, 01:41 to 06:15 (4 h 34 min, including the final evaluation)
- **Commit:** `74e2117` (clean working tree; MLflow tag `git_commit`)
- **Model:** `CenterNetLite`, 1,394,805 parameters, trained from scratch (no pretrained weights). Input 640×512 grey, output at stride 4 (`reference/falconet_ref/model.py`)
- **Config:** 50 epochs × 40,000 frames sampled at random from the 200 train sequences (2,500 steps of batch 16 per epoch, 2 M samples ≈ 9.3 passes over the 214k train frames). AdamW lr 1.5e-3 with weight decay 1e-4, 500 warmup steps then cosine decay to 0, gradient clipping at 10, bf16 autocast. Augmentation: horizontal flip, scale jitter ×0.75–1.33, gain 0.75–1.25 and bias ±0.1
- **Seed:** 42 (GPU training is not bit-for-bit deterministic)
- **Hardware:** Intel Core i7-12700H, RTX 3050 Ti Laptop 4 GB (driver 615.78.08), 32 GB RAM
- **Software:** Python 3.12, PyTorch 2.15.0+cu134, pycocotools 2.0.11, MLflow 3.18.0
- **Command** (from `reference/`, MLflow server started with `mise run mlflow`):
  ```bash
  uv run python -m falconet_ref.train_antiuav --epochs 50 --samples-per-epoch 40000 --workers 6 \
      --deadline 2026-11-18T07:30 --run-name baseline-e50
  ```
- **Data:** Anti-UAV410, archive SHA-256 `339e0e56…e055`. Validated after every epoch on every 20th frame of the 90 val sequences (4,746 frames); the final evaluation of the best checkpoint uses all 94,711 val frames. **The test split was not touched.**
- **Tracking:** MLflow experiment `reference-antiuav410`, run `baseline-e50` (`712d51922cc643dc918796ea7e86be57`). Local outputs, gitignored, in `runs/reference/antiuav410/baseline-e50/`:
  - `best.pt`, `last.pt`, `history.csv`;
  - `viz/epoch_NNN.{png,npz}` and `viz/learning.gif`;
  - `final/`: toolkit-format predictions, COCO files, ONNX model, metrics.

## Results

Best checkpoint (epoch 32, selected by AP50 on the val subset), evaluated on the **full val split**:

| Metric | Value |
|---|---|
| AP50 | **0.841** |
| mAP@[.5:.95] | 0.460 |
| AP75 | 0.458 |
| AP small / medium / large | **0.308** / 0.572 / 0.681 |
| AR@100 (small / medium / large) | 0.537 (0.413 / 0.634 / 0.687) |
| False positives per frame at score ≥ 0.5 | 0.047 |
| **mSA, detector only** (top box if score ≥ 0.3, no tracker) | **0.7048** |
| Recall@IoU 0.5 of the top box: all / small / medium / large | 0.845 / 0.669 / 0.984 / 0.993 |
| Mean IoU of the top box on visible drones | 0.668 |
| "No drone" frames answered correctly | 78.3% |

The numbers agree to 10⁻⁶ across three evaluators:
- pycocotools and the Rust `coco_eval`;
- the official `eval()` and the Rust `antiuav410 sa` (`scripts/check_sa_parity.py --predictions …/final/val_sa`).

The ONNX export matches PyTorch to a maximum absolute difference of 3.4×10⁻⁵.

Training curves, validation subset, selected epochs:

| Epoch | Train loss | Val loss | AP50 | mAP | AP small | det_sa | No-drone correct |
|---|---|---|---|---|---|---|---|
| 1 | 2.160 | 1.653 | 0.820 | 0.374 | 0.243 | 0.647 | 65.5% |
| 5 | 1.218 | **1.533** | 0.839 | 0.432 | 0.277 | 0.689 | 61.9% |
| 10 | 1.077 | 1.577 | 0.839 | 0.430 | 0.286 | 0.691 | 76.1% |
| 20 | 0.964 | 1.624 | 0.839 | 0.442 | 0.297 | 0.698 | 75.2% |
| **32** | 0.895 | 1.720 | **0.848** | **0.463** | **0.310** | **0.706** | 79.6% |
| 50 | 0.848 | 1.825 | 0.836 | 0.459 | 0.302 | 0.703 | 67.3% |

Cost:
- **Time:** ~298 s of training + ~21 s of validation per epoch, at ~134 img/s with the GPU at 99%. The data loader was not the bottleneck. The full run took 4.6 h.
- **Hardware load:** GPU board power ~57 W at 62–67 °C.
- **Energy:** about 0.26 kWh for the GPU alone, i.e. a few euro cents. The whole laptop's draw was not measured.

## Notes

1. **Most of the learning happens in the first epoch.** AP50 is 0.82 after 40k frames and only reaches 0.85 at its best. Consecutive frames are nearly identical, so 214k frames hold far less information than the number suggests, and 50 epochs is more than this model needs.
2. **Mild overfitting from epoch ~5.** The train loss keeps falling (1.22 → 0.85), but the val loss rises (1.53 → 1.83) while AP stays flat. The detection metrics hold up because ranking matters more than calibration. Still, a shorter schedule (~15 epochs) or stronger augmentation would give the same model in a third of the time.
3. **Small drones are the weakness.**
   - Recall@0.5 of the top box is 98–99% for medium and large drones but 67% for small ones. AP small (0.31) is half of AP large (0.68).
   - The localisation of the small ones is also loose: AP75 ≈ mAP ≈ 0.46. An IoU of 0.75 needs a ~1–2 px error on a 15 px box.
   - This is the risk the plan anticipates ("drones too small → very low mAP"), in a mild form.
4. **"No drone" frames: 78% right**, so about 1 in 5 empty frames gets a false alarm above the 0.3 threshold. The threshold was not tuned. Absences are only 2.3% of frames, so they matter little for the mean, but a tracker's lost state should help here.
5. **The detector alone gives mSA 0.705 on val**, a frame-by-frame baseline with no temporal information. ByteTrack can add continuity (fill short misses, suppress single-frame false alarms) and a calibrated "lost" state. That comparison is the next measurement.
6. **Label subtlety, found during this run.** In `IR_label.json`, 37 frames have `exist = 0` but a box: drones cut by the image border.
   - The official evaluator scores those frames as "no drone".
   - The toolkit's box files can't express that combination, so our Rust `sa` (which read them) scored those frames differently. The SA parity check had derived `exist` from the boxes and missed it.
   - **Fixed:** `antiuav410 sa` now reads `IR_label.json` when the archive is extracted, and the parity check uses the official labels and also accepts real predictions. It passes on train, val and test, max difference 7×10⁻⁷.
   - Training keeps the box: the drone is partly visible. The impact on this run is negligible: 4 val frames, mSA 0.70482 → 0.70476.
7. **Sensor overlays.** Many videos carry burnt-in on-screen text and a crosshair (azimuth/elevation readouts, timestamps). Whether false positives concentrate on these overlays has not been checked yet; it's a cheap analysis on `final/coco_dt_val.json`.

## Next

- Phase 1 task 4: import the ONNX model in Rust (Burn) and check that its outputs match PyTorch.
- Evaluate detector + ByteTrack on val (mSA, with and without the tracker); only then touch the test split.
- Optional retrain with a shorter schedule and translation augmentation (the camera follows the drone, so drones cluster at the image centre), if CP 1.2 asks for it.
