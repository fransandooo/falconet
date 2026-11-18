## Checkpoint 1.1 — Data understood

*Date: 2026-11-18 · Merged: PR #3 (annotations, stats, download script), PR #4 (SA metric), PR #8 (frame loader) · Status: awaiting Fran's approval*

**What was done:**
- **Source, licence and download.**
  - Official source verified: toolkit plus a Google Drive archive.
  - No licence file: we treat the data as research use only, never redistribute it or derived weights, and cite the paper.
  - The archive is pinned by size and SHA-256 in `scripts/download_antiuav410.sh`, and extracted to `~/datasets/antiuav410` (200 / 90 / 120 sequences).
- **Parsers** for both label sources (toolkit box files and `IR_label.json`), attributes and tracker results (`crates/falconet-data`).
- **Statistics and plots** (`antiuav410 stats`, `scripts/plot_dataset_stats.py`): box sizes, aspect ratios, centre positions, motion, absences, attributes. See `docs/datasets/antiuav410.md` and `media/phase-1/`.
- **Official metric (SA)** in Rust, checked against the official code: max difference 7×10⁻⁷ on all splits.
- **Frame-level checks:**
  - every frame is 640×512, grey stored as 3-channel JPEG (channels differ by ≤ 5/255);
  - frame and label counts match in all 410 sequences;
  - `IR_label.json` and the toolkit boxes agree on 438,360 of 438,397 frames.

**What makes it hard:**
1. **Small drones.** 50.6% of boxes are COCO-small (< 32²), and 29% have a diagonal of 10–30 px. Truly tiny boxes (≤ 10 px) are rare (2%).
2. **Thermal crossover** in ~60–67% of videos: the drone's temperature matches the background's.
3. **Cluttered backgrounds** (trees, buildings, hills) and **burnt-in sensor overlays** (text, crosshairs) in many sequences.
4. **The camera follows the drone**, so box centres cluster in the middle of the frame. That is a shortcut a detector can learn.
5. **Motion with a heavy tail.** The median shift is 2.6 px per frame, but ~2% of consecutive pairs don't overlap at all.
6. **"No drone" frames** are rare (2.3%) but scored by SA, so the system needs a calibrated way to say "nothing here".

**Surprises in the labels:**
- 37 frames have `exist = 0` and a box (drones cut by the image border); the official evaluator scores them as "no drone".
- 5 frames have degenerate boxes (zero width or height), which the official evaluator scores as 0.
- 8.5% of consecutive pairs have identical boxes (static drones or copied labels).
- Our tools now reproduce all of these exactly. The first one was only found once the frames were in, and was fixed in the SA tool and the parity check (see the experiment log of 2026-11-18).

**Implications for the model (applied in the reference detector):**
- **Head:** a stride-4 output head (160×128 cells), not stride 16.
- **Size regression:** box size in log space, so small and large drones weigh the same.
- **Metrics:** AP_small is the metric to watch, and SA needs a score threshold plus the tracker's lost state.
- **Augmentation:** horizontal flip, scale jitter and gain/bias are in. Translation augmentation against the centre shortcut is the next candidate.
- **Data volume:** consecutive frames are near-duplicates, so the training loop samples random frames per epoch instead of full passes.

**Still open:** a visual check of the IoU = 0 jumps and the identical-box runs (annotation errors or real motion?). It doesn't block phase 1.

**Decision I need from you:**
- *Showing dataset frames in public media* (README, the Manim video, the learning GIF). Anti-UAV410 has no licence, so publishing frames is a grey area.
- Option (a): keep frames out of the repo; public media use only plots, synthetic scenes and our own footage.
- Option (b): use a few frames with attribution, as papers do.
- **Recommendation: (a) for the repo, (b) for the video**, with a credit to the paper. Until you decide, the learning GIF stays local in `runs/`.
