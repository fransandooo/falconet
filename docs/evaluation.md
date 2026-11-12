# Evaluation

All metrics are implemented in Rust (`crates/falconet-data/src/metrics/`) and checked against the reference implementations.

## Tracking: state accuracy (SA)

Anti-UAV410's official metric; see [the dataset page](datasets/antiuav410.md#official-metric-state-accuracy-sa) for its definition.

```bash
cargo run --release -p falconet-data --bin antiuav410 -- sa --predictions <dir> --split test
```

Parity: `uv run scripts/check_sa_parity.py --split {train,val,test}` compares against the official `eval()` (embedded verbatim, hash-checked against the toolkit): max per-sequence difference 7×10⁻⁷.

## Detection: COCO metrics

| Metric | Meaning |
|---|---|
| mAP | mean AP over IoU thresholds 0.50, 0.55, …, 0.95 (COCO's primary metric) |
| AP50 / AP75 | AP at a single IoU threshold |
| AP_small / medium / large | mAP restricted to boxes of area ≤ 32², 32²–96², > 96² px² |
| AR@100 | max recall with up to 100 detections per image, averaged over thresholds |
| FP per image | detections with score ≥ 0.5 matching no object at IoU 0.5 |

AP_small is the one to watch: about half of Anti-UAV410's boxes are COCO-small.

```bash
cargo run --release -p falconet-data --bin coco_eval -- --gt annotations.json --dt results.json
```

Parity: `uv run scripts/check_coco_parity.py --seed {0,1,2}` builds a synthetic dataset (objects of every size, jittered detections, misses, duplicates, false alarms; ~600 objects / ~850 detections per seed) and compares with `pycocotools` 2.0.11's `COCOeval`. All seven numbers agree, typically within 10⁻⁸ and at worst 8×10⁻⁶ (an f32 vs f64 borderline case), against a tolerance of 10⁻⁴.

Implementation notes that the port has to reproduce:
- detections are sorted by score per image (top 100), then ranked globally with a stable sort;
- a detection matches the unmatched ground truth it overlaps most (IoU ≥ threshold). Ground truth outside the area range is "ignored": matching it neither helps nor hurts, and unmatched detections outside the range are ignored too;
- precision is made monotonic and sampled at 101 recall points; a size range with no ground truth gives no AP (pycocotools prints −1).
