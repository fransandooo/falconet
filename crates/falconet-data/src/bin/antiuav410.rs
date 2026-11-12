//! Anti-UAV410 dataset tool.
//!
//! ```text
//! cargo run --release -p falconet-data --bin antiuav410 -- stats
//! cargo run --release -p falconet-data --bin antiuav410 -- stats --toolkit <dir> --out <dir>
//! cargo run --release -p falconet-data --bin antiuav410 -- sa --predictions <dir> --split test
//! ```
//!
//! - `stats` reads the official toolkit annotations and writes `stats.json`
//!   (summary per split) and `boxes.csv` (every frame) for plotting with
//!   `scripts/plot_dataset_stats.py`.
//! - `sa` scores tracker results (one `<sequence>.txt` per video, in the
//!   official `{"res": [...]}` JSON format) with the official state accuracy.

use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use falconet_data::antiuav410::{
    FrameLabel, SequenceAnnotation, Split, load_toolkit_annotations, parse_predictions_json,
};
use falconet_data::data_dir;
use falconet_data::metrics::state_accuracy;
use falconet_data::stats::{SIZE_BIN_NAMES, SplitStats, split_stats};

#[derive(Parser)]
#[command(about = "Anti-UAV410 dataset tool")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compute dataset statistics from the toolkit annotations.
    Stats {
        /// Checkout of github.com/HwangBo94/Anti-UAV410 [default: $FALCONET_DATA_DIR/antiuav410-toolkit]
        #[arg(long)]
        toolkit: Option<PathBuf>,
        /// Output directory for stats.json and boxes.csv.
        #[arg(long, default_value = "runs/data-stats/antiuav410")]
        out: PathBuf,
    },
    /// Score tracker results with the official state accuracy (SA).
    Sa {
        /// Directory with one `<sequence>.txt` per video, official JSON format.
        #[arg(long)]
        predictions: PathBuf,
        #[arg(long, value_parser = parse_split, default_value = "test")]
        split: Split,
        /// Same as for `stats`.
        #[arg(long)]
        toolkit: Option<PathBuf>,
        /// Also print per-sequence scores as JSON lines.
        #[arg(long)]
        per_sequence: bool,
    },
}

fn parse_split(s: &str) -> Result<Split, String> {
    Split::ALL
        .into_iter()
        .find(|split| split.as_str() == s)
        .ok_or_else(|| format!("unknown split {s:?} (train, val, test)"))
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let default_toolkit = || data_dir().join("antiuav410-toolkit");
    let result = match cli.command {
        Command::Stats { toolkit, out } => stats(&toolkit.unwrap_or_else(default_toolkit), &out),
        Command::Sa {
            predictions,
            split,
            toolkit,
            per_sequence,
        } => sa(
            &toolkit.unwrap_or_else(default_toolkit),
            &predictions,
            split,
            per_sequence,
        ),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn stats(toolkit: &Path, out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let sequences = load_toolkit_annotations(toolkit)?;
    let all: Vec<SplitStats> = Split::ALL
        .into_iter()
        .map(Some)
        .chain([None])
        .map(|split| split_stats(&sequences, split))
        .collect();

    print_table(&all);

    fs::create_dir_all(out)?;
    fs::write(out.join("stats.json"), serde_json::to_string_pretty(&all)?)?;
    write_boxes_csv(&sequences, &out.join("boxes.csv"))?;
    println!("\nwrote {}/stats.json and boxes.csv", out.display());
    Ok(())
}

fn sa(
    toolkit: &Path,
    predictions: &Path,
    split: Split,
    per_sequence: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let sequences = load_toolkit_annotations(toolkit)?;
    let mut scores = Vec::new();
    let mut missing = 0;
    for seq in sequences.iter().filter(|s| s.split == split) {
        let path = predictions.join(format!("{}.txt", seq.name));
        let Ok(text) = fs::read_to_string(&path) else {
            missing += 1;
            continue;
        };
        let preds = parse_predictions_json(&text, &path)?;
        if let Some(score) = state_accuracy(&seq.frames, &preds) {
            if per_sequence {
                println!(
                    "{}",
                    serde_json::json!({ "sequence": seq.name, "sa": score })
                );
            }
            scores.push(score);
        }
    }
    if scores.is_empty() {
        return Err(format!("no prediction files found in {}", predictions.display()).into());
    }
    let msa = scores.iter().sum::<f32>() / scores.len() as f32;
    eprintln!(
        "{split}: mSA = {msa:.4} over {} sequences ({missing} without predictions)",
        scores.len()
    );
    Ok(())
}

fn print_table(all: &[SplitStats]) {
    let name = |s: &SplitStats| s.split.map_or("all", Split::as_str);
    let pct = |a: usize, b: usize| 100.0 * a as f32 / b.max(1) as f32;
    println!(
        "{:<6} {:>5} {:>8} {:>8} {:>10} {:>10} {:>11} {:>10} {:>11}",
        "split",
        "seqs",
        "frames",
        "absent",
        "diag p50",
        "diag p05",
        "COCO small",
        "motion p95",
        "IoU<0.3"
    );
    for s in all {
        println!(
            "{:<6} {:>5} {:>8} {:>7.1}% {:>8.1}px {:>8.1}px {:>10.1}% {:>7.1}px {:>10.1}%",
            name(s),
            s.sequences,
            s.frames,
            pct(s.absent_frames, s.frames),
            s.box_diagonal.median,
            s.box_diagonal.p05,
            pct(s.coco_small_boxes, s.present_frames),
            s.center_motion_px.p95,
            s.consecutive_iou_below_0_3 * 100.0,
        );
    }
    let total = all.last().expect("the 'all' row is always present");
    println!("\nsize bins (all splits, by box diagonal):");
    for (bin, count) in SIZE_BIN_NAMES.iter().zip(total.size_bins) {
        println!(
            "  {bin:<12} {:>7} ({:.1}%)",
            count,
            pct(count, total.present_frames)
        );
    }
}

/// One row per frame: enough to redo any plot without re-parsing.
fn write_boxes_csv(sequences: &[SequenceAnnotation], path: &Path) -> std::io::Result<()> {
    let mut out = BufWriter::new(fs::File::create(path)?);
    writeln!(out, "split,sequence,frame,label,x,y,w,h")?;
    for seq in sequences {
        for (i, label) in seq.frames.iter().enumerate() {
            let prefix = format!("{},{},{}", seq.split, seq.name, i);
            match label {
                FrameLabel::Present(b) => {
                    writeln!(out, "{prefix},present,{},{},{},{}", b.x, b.y, b.w, b.h)?
                }
                FrameLabel::Absent => writeln!(out, "{prefix},absent,,,,")?,
                FrameLabel::Unlabelled => writeln!(out, "{prefix},unlabelled,,,,")?,
                FrameLabel::Invalid(b) => {
                    writeln!(out, "{prefix},invalid,{},{},{},{}", b.x, b.y, b.w, b.h)?
                }
            }
        }
    }
    out.flush()
}
