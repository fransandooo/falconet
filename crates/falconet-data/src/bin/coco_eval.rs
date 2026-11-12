//! COCO-format detection evaluation (single class).
//!
//! ```text
//! cargo run --release -p falconet-data --bin coco_eval -- --gt annotations.json --dt results.json
//! ```
//!
//! `annotations.json` is a COCO ground-truth file (`images`, `annotations` with
//! `bbox: [x, y, w, h]`); `results.json` is a COCO results list
//! (`[{"image_id", "bbox", "score"}, ...]`). Prints the summary as JSON.
//! Ground-truth area is taken as `w * h`, which is what COCO uses for boxes
//! without segmentation masks.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use falconet_core::{BBox, Detection};
use falconet_data::metrics::detection::{ImageEval, evaluate, false_positives_per_image};
use serde::Deserialize;

#[derive(Parser)]
#[command(about = "COCO-style detection metrics for one class")]
struct Args {
    #[arg(long)]
    gt: PathBuf,
    #[arg(long)]
    dt: PathBuf,
}

#[derive(Deserialize)]
struct GroundTruthFile {
    images: Vec<Image>,
    annotations: Vec<Annotation>,
}

#[derive(Deserialize)]
struct Image {
    id: u64,
}

#[derive(Deserialize)]
struct Annotation {
    image_id: u64,
    bbox: [f32; 4],
}

#[derive(Deserialize)]
struct Result {
    image_id: u64,
    bbox: [f32; 4],
    score: f32,
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &Args) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let gt: GroundTruthFile = serde_json::from_str(&fs::read_to_string(&args.gt)?)?;
    let dt: Vec<Result> = serde_json::from_str(&fs::read_to_string(&args.dt)?)?;

    // BTreeMap keeps images sorted by id, which fixes the order in which
    // detections of equal score are ranked (COCO sorts image ids too).
    let mut images: BTreeMap<u64, ImageEval> = gt
        .images
        .iter()
        .map(|im| (im.id, ImageEval::default()))
        .collect();
    let to_box = |[x, y, w, h]: [f32; 4]| BBox::new(x, y, w, h);
    for a in &gt.annotations {
        images
            .entry(a.image_id)
            .or_default()
            .ground_truth
            .push(to_box(a.bbox));
    }
    for d in &dt {
        if let Some(image) = images.get_mut(&d.image_id) {
            image
                .detections
                .push(Detection::new(to_box(d.bbox), d.score));
        }
    }
    let images: Vec<ImageEval> = images.into_values().collect();

    let summary = evaluate(&images);
    let fppi = false_positives_per_image(&images, 0.5, 0.5);
    let mut json = serde_json::to_value(summary)?;
    json["fp_per_image_at_score_0_5"] = fppi.into();
    println!("{}", serde_json::to_string_pretty(&json)?);
    Ok(())
}
