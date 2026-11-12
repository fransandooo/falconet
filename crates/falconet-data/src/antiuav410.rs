//! The Anti-UAV410 thermal-infrared drone tracking benchmark.
//!
//! Huang et al., "Anti-UAV410: A Thermal Infrared Benchmark and Customized
//! Scheme for Tracking Drones in the Wild", T-PAMI 2023.
//! Toolkit: <https://github.com/HwangBo94/Anti-UAV410>.
//!
//! 410 single-drone thermal videos (200 train / 90 val / 120 test). Every frame
//! is labelled either with the drone's box or as "not visible" (occluded or
//! out of view). Annotations come in two formats, both parsed here:
//!
//! - **toolkit text files** (`annos/<split>/<seq>.txt`, one `x,y,w,h` line per
//!   frame, `0,0,0,0` = not visible) plus per-video attributes
//!   (`annos/<split>/att/<seq>.txt`, ten 0/1 flags);
//! - **`IR_label.json`** next to each sequence's frames in the dataset archive:
//!   `{"exist": [1, 0, ...], "gt_rect": [[x, y, w, h], [], ...]}`.

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::bbox::BBox;

/// Dataset split.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Train,
    Val,
    Test,
}

impl Split {
    pub const ALL: [Split; 3] = [Split::Train, Split::Val, Split::Test];

    pub fn as_str(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Val => "val",
            Split::Test => "test",
        }
    }
}

impl fmt::Display for Split {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Per-video challenge attributes, in the order of the official toolkit
/// (`toolkit_matlab/run_tracker_performance_evaluation.m`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Attribute {
    ThermalCrossover,
    OutOfView,
    ScaleVariation,
    FastMotion,
    Occlusion,
    DynamicBackgroundClutter,
    TinySize,
    SmallSize,
    MediumSize,
    NormalSize,
}

impl Attribute {
    pub const ALL: [Attribute; 10] = [
        Attribute::ThermalCrossover,
        Attribute::OutOfView,
        Attribute::ScaleVariation,
        Attribute::FastMotion,
        Attribute::Occlusion,
        Attribute::DynamicBackgroundClutter,
        Attribute::TinySize,
        Attribute::SmallSize,
        Attribute::MediumSize,
        Attribute::NormalSize,
    ];

    /// The toolkit's abbreviation (TC, OV, ...).
    pub fn short(self) -> &'static str {
        match self {
            Attribute::ThermalCrossover => "TC",
            Attribute::OutOfView => "OV",
            Attribute::ScaleVariation => "SV",
            Attribute::FastMotion => "FM",
            Attribute::Occlusion => "OC",
            Attribute::DynamicBackgroundClutter => "DBC",
            Attribute::TinySize => "TS",
            Attribute::SmallSize => "SS",
            Attribute::MediumSize => "MS",
            Attribute::NormalSize => "NS",
        }
    }
}

/// A set of [`Attribute`]s stored as a 10-bit mask: one bit per attribute,
/// the way you'd store flags in C, behind a type-safe API.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Attributes(u16);

impl Attributes {
    pub fn contains(self, attribute: Attribute) -> bool {
        self.0 & (1 << attribute as u16) != 0
    }

    pub fn insert(&mut self, attribute: Attribute) {
        self.0 |= 1 << attribute as u16;
    }

    /// The attributes present, in toolkit order.
    pub fn iter(self) -> impl Iterator<Item = Attribute> {
        Attribute::ALL
            .into_iter()
            .filter(move |&a| self.contains(a))
    }
}

/// The label of one frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FrameLabel {
    /// The drone is visible at this box.
    Present(BBox),
    /// The drone is occluded or out of view.
    Absent,
    /// Marked as present but without a box at all (`[]`, or all zeros in the
    /// JSON format). The official evaluation skips such frames; we keep them
    /// visible instead of silently dropping them.
    Unlabelled,
    /// Marked as present with a degenerate box (zero width or height, e.g.
    /// `18,0,1,0`): 5 of the 438k frames. The official evaluation still scores
    /// these frames, and always with 0; statistics leave them out.
    Invalid(BBox),
}

impl FrameLabel {
    pub fn bbox(&self) -> Option<BBox> {
        match self {
            FrameLabel::Present(b) => Some(*b),
            _ => None,
        }
    }
}

/// Annotations of one video.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceAnnotation {
    pub name: String,
    pub split: Split,
    pub frames: Vec<FrameLabel>,
    pub attributes: Attributes,
}

#[derive(Debug, thiserror::Error)]
pub enum AnnotationError {
    #[error("cannot read {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{path}:{line}: {message}")]
    Parse {
        path: PathBuf,
        line: usize,
        message: String,
    },
    #[error("{path}: invalid IR_label.json: {source}")]
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("{path}: `exist` has {exist} entries but `gt_rect` has {rects}")]
    LengthMismatch {
        path: PathBuf,
        exist: usize,
        rects: usize,
    },
}

/// Parses a toolkit annotation file: one `x,y,w,h` line per frame.
/// `0,0,0,0` marks a frame where the drone is not visible.
pub fn parse_box_lines(text: &str, path: &Path) -> Result<Vec<FrameLabel>, AnnotationError> {
    let parse_error = |line: usize, message: String| AnnotationError::Parse {
        path: path.to_owned(),
        line,
        message,
    };

    text.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(i, line)| {
            let values: Vec<f32> = line
                .split(',')
                .map(|v| v.trim().parse::<f32>())
                .collect::<Result<_, _>>()
                .map_err(|e| parse_error(i + 1, format!("{e} in {line:?}")))?;
            let [x, y, w, h] = values[..] else {
                return Err(parse_error(
                    i + 1,
                    format!("expected 4 values, got {line:?}"),
                ));
            };
            Ok(box_label(x, y, w, h))
        })
        .collect()
}

/// `0,0,0,0` means "not visible" in the text format; any other box with a
/// zero width or height is a degenerate annotation.
fn box_label(x: f32, y: f32, w: f32, h: f32) -> FrameLabel {
    if [x, y, w, h] == [0.0; 4] {
        FrameLabel::Absent
    } else if w > 0.0 && h > 0.0 {
        FrameLabel::Present(BBox::new(x, y, w, h))
    } else {
        FrameLabel::Invalid(BBox::new(x, y, w, h))
    }
}

/// Parses a toolkit attribute file: ten comma-separated 0/1 flags.
pub fn parse_attributes(text: &str, path: &Path) -> Result<Attributes, AnnotationError> {
    let flags: Vec<&str> = text.trim().split(',').map(str::trim).collect();
    if flags.len() != Attribute::ALL.len() {
        return Err(AnnotationError::Parse {
            path: path.to_owned(),
            line: 1,
            message: format!(
                "expected {} flags, got {}",
                Attribute::ALL.len(),
                flags.len()
            ),
        });
    }
    let mut attributes = Attributes::default();
    for (flag, attribute) in flags.iter().zip(Attribute::ALL) {
        match *flag {
            "1" => attributes.insert(attribute),
            "0" => {}
            other => {
                return Err(AnnotationError::Parse {
                    path: path.to_owned(),
                    line: 1,
                    message: format!("attribute flag must be 0 or 1, got {other:?}"),
                });
            }
        }
    }
    Ok(attributes)
}

/// The JSON layout of `IR_label.json`.
#[derive(Deserialize)]
struct IrLabelJson {
    exist: Vec<u8>,
    gt_rect: Vec<Vec<f32>>,
}

/// Parses an `IR_label.json` file.
pub fn parse_ir_label_json(text: &str, path: &Path) -> Result<Vec<FrameLabel>, AnnotationError> {
    let json: IrLabelJson = serde_json::from_str(text).map_err(|source| AnnotationError::Json {
        path: path.to_owned(),
        source,
    })?;
    if json.exist.len() != json.gt_rect.len() {
        return Err(AnnotationError::LengthMismatch {
            path: path.to_owned(),
            exist: json.exist.len(),
            rects: json.gt_rect.len(),
        });
    }
    Ok(json
        .exist
        .iter()
        .zip(&json.gt_rect)
        // A tuple can't hold an unsized `[f32]`, so match on the slice
        // reference `&[f32]`: `&[x, y, w, h]` only matches exactly 4 elements.
        // The checks mirror the official evaluation: `len(gt) < 4 or sum(gt) == 0`
        // is skipped; any other 4-number box is scored.
        .map(|(&exist, rect)| match (exist, rect.as_slice()) {
            (0, _) => FrameLabel::Absent,
            (_, &[x, y, w, h]) if x + y + w + h != 0.0 => box_label(x, y, w, h),
            _ => FrameLabel::Unlabelled,
        })
        .collect())
}

/// Tracker results in the official toolkit's JSON format:
/// `{"res": [[x, y, w, h], [], ...]}`, one entry per frame.
///
/// Mirrors how `Evaluation_for_SA.py` interprets an entry: 4 numbers = a box,
/// 0 or 1 numbers = "no target". Any other length is malformed: the official
/// code scores it 0 on every frame, which a zero-size box reproduces.
pub fn parse_predictions_json(
    text: &str,
    path: &Path,
) -> Result<Vec<Option<BBox>>, AnnotationError> {
    #[derive(Deserialize)]
    struct Results {
        res: Vec<Vec<f32>>,
    }
    let results: Results = serde_json::from_str(text).map_err(|source| AnnotationError::Json {
        path: path.to_owned(),
        source,
    })?;
    Ok(results
        .res
        .iter()
        .map(|entry| match entry.as_slice() {
            &[x, y, w, h] => Some(BBox::new(x, y, w, h)),
            [] | [_] => None,
            _ => Some(BBox::new(0.0, 0.0, 0.0, 0.0)),
        })
        .collect())
}

fn read(path: &Path) -> Result<String, AnnotationError> {
    fs::read_to_string(path).map_err(|source| AnnotationError::Io {
        path: path.to_owned(),
        source,
    })
}

/// Loads every sequence's annotations from a checkout of the official
/// toolkit (`<root>/annos/{train,val,test}/*.txt` + `att/`), sorted by split
/// and name.
pub fn load_toolkit_annotations(root: &Path) -> Result<Vec<SequenceAnnotation>, AnnotationError> {
    let mut sequences = Vec::new();
    for split in Split::ALL {
        let dir = root.join("annos").join(split.as_str());
        let entries = fs::read_dir(&dir).map_err(|source| AnnotationError::Io {
            path: dir.clone(),
            source,
        })?;
        let mut files: Vec<PathBuf> = entries
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "txt"))
            .collect();
        files.sort();

        for path in files {
            let name = path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let frames = parse_box_lines(&read(&path)?, &path)?;
            let att_path = dir.join("att").join(format!("{name}.txt"));
            let attributes = parse_attributes(&read(&att_path)?, &att_path)?;
            sequences.push(SequenceAnnotation {
                name,
                split,
                frames,
                attributes,
            });
        }
    }
    Ok(sequences)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p() -> &'static Path {
        Path::new("test.txt")
    }

    #[test]
    fn box_lines_with_absent_frames() {
        let labels =
            parse_box_lines("206,314,8,8\n0,0,0,0\n\n207.5,312,8,9\n18,0,1,0\n", p()).unwrap();
        assert_eq!(
            labels,
            vec![
                FrameLabel::Present(BBox::new(206.0, 314.0, 8.0, 8.0)),
                FrameLabel::Absent,
                FrameLabel::Present(BBox::new(207.5, 312.0, 8.0, 9.0)),
                FrameLabel::Invalid(BBox::new(18.0, 0.0, 1.0, 0.0)),
            ]
        );
    }

    #[test]
    fn malformed_box_lines_report_the_line() {
        let err = parse_box_lines("1,2,3,4\n1,2,3\n", p()).unwrap_err();
        assert!(
            matches!(err, AnnotationError::Parse { line: 2, .. }),
            "{err}"
        );
        let err = parse_box_lines("1,2,x,4\n", p()).unwrap_err();
        assert!(
            matches!(err, AnnotationError::Parse { line: 1, .. }),
            "{err}"
        );
    }

    #[test]
    fn attributes_parse_in_toolkit_order() {
        let att = parse_attributes("0,1,0,1,0,1,1,1,1,0\n", p()).unwrap();
        let present: Vec<_> = att.iter().map(Attribute::short).collect();
        assert_eq!(present, ["OV", "FM", "DBC", "TS", "SS", "MS"]);
        assert!(parse_attributes("0,1", p()).is_err());
        assert!(parse_attributes("0,1,0,1,0,1,1,1,1,2", p()).is_err());
    }

    #[test]
    fn ir_label_json_handles_absent_and_broken_frames() {
        let json =
            r#"{"exist": [1, 0, 1, 1], "gt_rect": [[10, 20, 5, 4], [], [0, 0, 0, 0], [1, 2]]}"#;
        let labels = parse_ir_label_json(json, p()).unwrap();
        assert_eq!(
            labels,
            vec![
                FrameLabel::Present(BBox::new(10.0, 20.0, 5.0, 4.0)),
                FrameLabel::Absent,
                FrameLabel::Unlabelled,
                FrameLabel::Unlabelled,
            ]
        );
    }

    #[test]
    fn predictions_json_follows_the_official_interpretation() {
        let json = r#"{"res": [[1, 2, 3, 4], [], [0], [1, 2]]}"#;
        let preds = parse_predictions_json(json, p()).unwrap();
        assert_eq!(preds[0], Some(BBox::new(1.0, 2.0, 3.0, 4.0)));
        assert_eq!((preds[1], preds[2]), (None, None));
        assert_eq!(preds[3], Some(BBox::new(0.0, 0.0, 0.0, 0.0)));
    }

    #[test]
    fn ir_label_json_length_mismatch_is_an_error() {
        let json = r#"{"exist": [1, 0], "gt_rect": [[10, 20, 5, 4]]}"#;
        assert!(matches!(
            parse_ir_label_json(json, p()),
            Err(AnnotationError::LengthMismatch {
                exist: 2,
                rects: 1,
                ..
            })
        ));
    }
}
