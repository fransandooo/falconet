//! Anti-UAV410 dataset tool.
//!
//! ```text
//! cargo run --release -p falconet-data --bin antiuav410 -- stats
//! cargo run --release -p falconet-data --bin antiuav410 -- stats --toolkit <dir> --out <dir>
//! ```
//!
//! `stats` reads the official toolkit annotations and writes `stats.json`
//! (summary per split) and `boxes.csv` (every frame) for plotting with
//! `scripts/plot_dataset_stats.py`.

use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use falconet_data::antiuav410::{FrameLabel, SequenceAnnotation, Split, load_toolkit_annotations};
use falconet_data::data_dir;
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
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Stats { toolkit, out } => {
            let toolkit = toolkit.unwrap_or_else(|| data_dir().join("antiuav410-toolkit"));
            stats(&toolkit, &out)
        }
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
            }
        }
    }
    out.flush()
}
