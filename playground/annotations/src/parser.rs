//! Parser for the toy annotation format.
//!
//! ```text
//! # name: toy-sequence-01
//! # size: 640x512
//! # columns: frame exist x y w h
//! 0 1 312.0 240.5 6.0 4.0
//! 1 0 0 0 0 0
//! ```
//!
//! Lines starting with `#` are comments; `# key: value` comments form the
//! header. Every other non-empty line is one frame: `exist` is `1` if the drone
//! is visible (then `x y w h` is its box) and `0` otherwise.

use std::error::Error;
use std::fmt;
use std::str::FromStr;

use crate::bbox::BBox;

/// Number of whitespace-separated fields in a frame line.
const FIELDS_PER_LINE: usize = 6;

/// Header fields that **borrow** from the input text instead of copying it.
///
/// `'a` is a *lifetime parameter*. It reads: "`name` points into some string
/// that lives at least as long as `'a`". The compiler uses it to reject any
/// code where a `Header` outlives the text it was parsed from — the classic C
/// bug of returning a `char *` into a buffer that is freed later.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Header<'a> {
    pub name: &'a str,
    pub width: u32,
    pub height: u32,
}

/// The annotation of one frame. `bbox` is `None` when the drone is not visible.
///
/// In C this would typically be an `int exist` flag next to a box full of
/// garbage values. With `Option<BBox>` it is impossible to read the box
/// without first checking that it exists.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameAnnotation {
    pub frame: u32,
    pub bbox: Option<BBox>,
}

/// A fully parsed sequence that **owns** all its data (`String`, `Vec`).
///
/// It does not depend on the input text any more, so it can be stored,
/// returned from functions or sent to another thread freely. When a
/// `Sequence` goes out of scope, its `String` and `Vec` are freed
/// automatically (no `free()`, no garbage collector: this is called RAII).
#[derive(Debug, Clone, PartialEq)]
pub struct Sequence {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub frames: Vec<FrameAnnotation>,
}

/// Everything that can go wrong while parsing. Line numbers are 1-based.
///
/// Rust enums are *tagged unions*: each variant can carry different data, and
/// `match` forces you to handle every variant.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    /// A required `# key: value` header line is missing.
    MissingHeader(&'static str),
    /// A frame line does not have exactly 6 fields.
    WrongFieldCount { line: usize, found: usize },
    /// A field could not be parsed or has an invalid value.
    InvalidField {
        line: usize,
        field: &'static str,
        value: String,
    },
    /// Frame numbers must be strictly increasing.
    FramesOutOfOrder {
        line: usize,
        previous: u32,
        found: u32,
    },
}

// `Display` is the trait behind `{}` formatting: the human-readable message.
impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingHeader(key) => write!(f, "missing header line `# {key}: ...`"),
            Self::WrongFieldCount { line, found } => {
                write!(
                    f,
                    "line {line}: expected {FIELDS_PER_LINE} fields, found {found}"
                )
            }
            Self::InvalidField { line, field, value } => {
                write!(
                    f,
                    "line {line}: invalid value {value:?} for field `{field}`"
                )
            }
            Self::FramesOutOfOrder {
                line,
                previous,
                found,
            } => write!(f, "line {line}: frame {found} comes after frame {previous}"),
        }
    }
}

// Implementing `Error` lets `ParseError` work with `?` in functions returning
// `Box<dyn Error>` (see `main.rs`). The default methods are enough here.
impl Error for ParseError {}

/// Parses only the header, without allocating any string.
///
/// The return type `Header<'_>` ties the output lifetime to `input`: the
/// header can only be used while `input` is alive.
///
/// `Result<T, E>` is either `Ok(T)` or `Err(E)`. It replaces C's "return -1
/// and set errno": the error is part of the type and can't be ignored silently.
pub fn parse_header(input: &str) -> Result<Header<'_>, ParseError> {
    let mut name = None;
    let mut size = None;

    for (index, line) in input.lines().enumerate() {
        // `let ... else`: destructure or bail out of this iteration.
        let Some(comment) = line.strip_prefix('#') else {
            continue;
        };
        let Some((key, value)) = comment.split_once(':') else {
            continue;
        };
        let value = value.trim(); // still a `&str` slice into `input`: no copy

        match key.trim() {
            "name" => name = Some(value),
            // `?` means: if this is an `Err`, return it from `parse_header` now;
            // if it is `Ok(v)`, unwrap it to `v` and continue.
            "size" => size = Some(parse_size(value, index + 1)?),
            _ => {} // other keys (e.g. `columns`) are ignored
        }
    }

    // `ok_or` turns an `Option` into a `Result`, so `?` can propagate it.
    let name = name.ok_or(ParseError::MissingHeader("name"))?;
    let (width, height) = size.ok_or(ParseError::MissingHeader("size"))?;
    Ok(Header {
        name,
        width,
        height,
    })
}

/// Parses the whole file into an owned [`Sequence`].
///
/// # Example
///
/// ```
/// use playground_annotations::parse_sequence;
///
/// let text = "# name: demo\n# size: 640x512\n0 1 10 20 6 4\n1 0 0 0 0 0\n";
/// let seq = parse_sequence(text).unwrap();
/// assert_eq!(seq.name, "demo");
/// assert_eq!(seq.frames.len(), 2);
/// assert!(seq.frames[1].bbox.is_none());
/// ```
pub fn parse_sequence(input: &str) -> Result<Sequence, ParseError> {
    let header = parse_header(input)?;
    let mut frames = Vec::new();
    let mut previous: Option<u32> = None;

    for (index, line) in input.lines().enumerate() {
        let line_no = index + 1;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let annotation = parse_frame_line(line, line_no)?;

        // A "let chain": the `if` runs only if the pattern matches *and* the
        // condition holds.
        if let Some(prev) = previous
            && annotation.frame <= prev
        {
            return Err(ParseError::FramesOutOfOrder {
                line: line_no,
                previous: prev,
                found: annotation.frame,
            });
        }
        previous = Some(annotation.frame);
        frames.push(annotation);
    }

    Ok(Sequence {
        // The one place where borrowed text becomes owned: `to_owned` copies
        // the `&str` slice into a new heap-allocated `String`.
        name: header.name.to_owned(),
        width: header.width,
        height: header.height,
        frames,
    })
}

/// Parses `"640x512"` into `(640, 512)`.
fn parse_size(value: &str, line: usize) -> Result<(u32, u32), ParseError> {
    // A closure (anonymous function) that builds the error. It borrows `value`
    // and `line` from the surrounding scope.
    let invalid = || ParseError::InvalidField {
        line,
        field: "size",
        value: value.to_owned(),
    };
    let (w, h) = value.split_once('x').ok_or_else(invalid)?;
    let width = w.trim().parse().map_err(|_| invalid())?;
    let height = h.trim().parse().map_err(|_| invalid())?;
    Ok((width, height))
}

fn parse_frame_line(line: &str, line_no: usize) -> Result<FrameAnnotation, ParseError> {
    // `fields` is a `Vec` of `&str` slices that point into `line`: the text
    // itself is never copied, only (pointer, length) pairs are stored.
    let fields: Vec<&str> = line.split_whitespace().collect();
    if fields.len() != FIELDS_PER_LINE {
        return Err(ParseError::WrongFieldCount {
            line: line_no,
            found: fields.len(),
        });
    }

    // Type inference picks `T` for `parse_field` from the annotated variable.
    let frame: u32 = parse_field(fields[0], line_no, "frame")?;
    let exist: u8 = parse_field(fields[1], line_no, "exist")?;

    let bbox = match exist {
        0 => None,
        1 => Some(BBox::new(
            parse_field(fields[2], line_no, "x")?,
            parse_field(fields[3], line_no, "y")?,
            parse_positive(fields[4], line_no, "w")?,
            parse_positive(fields[5], line_no, "h")?,
        )),
        _ => {
            return Err(ParseError::InvalidField {
                line: line_no,
                field: "exist",
                value: fields[1].to_owned(),
            });
        }
    };

    Ok(FrameAnnotation { frame, bbox })
}

/// Parses any type that implements `FromStr` (`u8`, `u32`, `f32`, ...).
///
/// `<T: FromStr>` is a *generic* parameter with a *trait bound*: a sneak peek
/// at exercise 3b. One function, many concrete types, all checked at compile time.
fn parse_field<T: FromStr>(raw: &str, line: usize, field: &'static str) -> Result<T, ParseError> {
    raw.parse().map_err(|_| ParseError::InvalidField {
        line,
        field,
        value: raw.to_owned(),
    })
}

/// Like `parse_field::<f32>`, but rejects zero, negative and NaN sizes.
fn parse_positive(raw: &str, line: usize, field: &'static str) -> Result<f32, ParseError> {
    let value: f32 = parse_field(raw, line, field)?;
    if value.is_nan() || value <= 0.0 {
        return Err(ParseError::InvalidField {
            line,
            field,
            value: raw.to_owned(),
        });
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &str = "\
# name: test-seq
# size: 640x512
0 1 10.0 20.0 6.0 4.0
1 0 0 0 0 0
2 1 12.0 21.0 6.0 4.0
";

    #[test]
    fn header_borrows_the_name_from_the_input() {
        let header = parse_header(VALID).unwrap();
        assert_eq!(header.name, "test-seq");
        assert_eq!((header.width, header.height), (640, 512));
        // The name is a slice *inside* VALID, not a copy: same memory address.
        assert!(
            VALID
                .as_bytes()
                .as_ptr_range()
                .contains(&header.name.as_ptr())
        );
    }

    #[test]
    fn parses_visible_and_missing_frames() {
        let seq = parse_sequence(VALID).unwrap();
        assert_eq!(seq.name, "test-seq");
        assert_eq!(seq.frames.len(), 3);
        assert_eq!(seq.frames[0].bbox, Some(BBox::new(10.0, 20.0, 6.0, 4.0)));
        assert_eq!(seq.frames[1].bbox, None);
    }

    #[test]
    fn missing_size_header_is_an_error() {
        let err = parse_sequence("# name: x\n0 0 0 0 0 0\n").unwrap_err();
        assert_eq!(err, ParseError::MissingHeader("size"));
    }

    #[test]
    fn malformed_size_reports_its_line() {
        let err = parse_header("# name: x\n# size: 640 by 512\n").unwrap_err();
        assert!(matches!(
            err,
            ParseError::InvalidField {
                line: 2,
                field: "size",
                ..
            }
        ));
    }

    #[test]
    fn wrong_field_count_reports_the_line() {
        let text = "# name: x\n# size: 10x10\n0 1 1 2 3\n";
        let err = parse_sequence(text).unwrap_err();
        assert_eq!(err, ParseError::WrongFieldCount { line: 3, found: 5 });
    }

    #[test]
    fn exist_must_be_zero_or_one() {
        let text = "# name: x\n# size: 10x10\n0 2 1 1 1 1\n";
        let err = parse_sequence(text).unwrap_err();
        assert!(matches!(
            err,
            ParseError::InvalidField { field: "exist", .. }
        ));
    }

    #[test]
    fn negative_or_nan_sizes_are_rejected() {
        for bad in ["-1", "0", "NaN"] {
            let text = format!("# name: x\n# size: 10x10\n0 1 1 1 {bad} 1\n");
            let err = parse_sequence(&text).unwrap_err();
            assert!(
                matches!(err, ParseError::InvalidField { field: "w", .. }),
                "{bad}"
            );
        }
    }

    #[test]
    fn frames_must_be_strictly_increasing() {
        let text = "# name: x\n# size: 10x10\n3 0 0 0 0 0\n3 0 0 0 0 0\n";
        let err = parse_sequence(text).unwrap_err();
        assert_eq!(
            err,
            ParseError::FramesOutOfOrder {
                line: 4,
                previous: 3,
                found: 3
            }
        );
    }

    #[test]
    fn error_messages_are_human_readable() {
        let err = ParseError::WrongFieldCount { line: 7, found: 2 };
        assert_eq!(err.to_string(), "line 7: expected 6 fields, found 2");
    }
}
