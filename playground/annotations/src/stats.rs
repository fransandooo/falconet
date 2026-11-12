//! Functions over a parsed [`Sequence`], one per way of passing data in Rust:
//! shared borrow (`&`), exclusive borrow (`&mut`) and move (by value).

use crate::bbox::BBox;
use crate::parser::Sequence;

/// Summary statistics of a sequence. Areas are in pixels².
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SequenceStats {
    pub frames: usize,
    pub visible: usize,
    pub min_area: Option<f32>,
    pub max_area: Option<f32>,
    pub mean_area: Option<f32>,
}

impl SequenceStats {
    /// Fraction of frames in which the drone is visible.
    pub fn visible_fraction(&self) -> f32 {
        if self.frames == 0 {
            0.0
        } else {
            self.visible as f32 / self.frames as f32
        }
    }
}

/// **Shared borrow** (`&Sequence`): read-only access. Any number of shared
/// borrows may exist at the same time, and the caller keeps ownership.
pub fn compute_stats(seq: &Sequence) -> SequenceStats {
    // An iterator chain. Nothing runs until `collect` pulls the values through:
    // `iter` yields `&FrameAnnotation`, `filter_map` keeps the `Some` boxes,
    // `map` turns each box into its area. Compiles to a plain loop.
    let areas: Vec<f32> = seq
        .frames
        .iter()
        .filter_map(|frame| frame.bbox)
        .map(|bbox| bbox.area())
        .collect();

    let visible = areas.len();
    SequenceStats {
        frames: seq.frames.len(),
        visible,
        // `reduce` returns `None` for an empty iterator, which is exactly what
        // "no visible frames, no min area" should be.
        min_area: areas.iter().copied().reduce(f32::min),
        max_area: areas.iter().copied().reduce(f32::max),
        // `bool::then` builds `Some(value)` only when the condition is true.
        mean_area: (visible > 0).then(|| areas.iter().sum::<f32>() / visible as f32),
    }
}

/// **Exclusive borrow** (`&mut Sequence`): read-write access. While it lasts,
/// nobody else can read or write `seq`. That single rule is what lets the
/// compiler rule out data races and iterator invalidation (modifying a vector
/// while looping over it) before the program ever runs.
///
/// Clips every box to the image bounds and returns how many boxes changed.
pub fn clip_to_image(seq: &mut Sequence) -> usize {
    let (width, height) = (seq.width as f32, seq.height as f32);
    let mut changed = 0;

    // `&mut seq.frames` iterates with `frame: &mut FrameAnnotation`.
    for frame in &mut seq.frames {
        if let Some(bbox) = frame.bbox {
            let clipped = bbox.clipped(width, height);
            if clipped != Some(bbox) {
                frame.bbox = clipped; // writing through the mutable reference
                changed += 1;
            }
        }
    }
    changed
}

/// **Move** (`Sequence` by value): the function takes ownership. The caller's
/// variable becomes unusable afterwards (the compiler enforces it), and the
/// elements are moved out of the old `Vec` without cloning anything. When this
/// function returns, whatever is left of `seq` (its `name`) is freed.
///
/// Returns `(frame, box)` for every frame where the drone is visible.
pub fn into_visible(seq: Sequence) -> Vec<(u32, BBox)> {
    seq.frames
        .into_iter() // consumes the Vec: yields owned `FrameAnnotation`s
        .filter_map(|frame| frame.bbox.map(|bbox| (frame.frame, bbox)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_sequence;

    const TEXT: &str = "\
# name: stats
# size: 100x100
0 1 0 0 2 2
1 0 0 0 0 0
2 1 95 95 10 10
3 0 0 0 0 0
";

    #[test]
    fn stats_of_a_small_sequence() {
        let seq = parse_sequence(TEXT).unwrap();
        let stats = compute_stats(&seq);
        assert_eq!(stats.frames, 4);
        assert_eq!(stats.visible, 2);
        assert_eq!(stats.min_area, Some(4.0));
        assert_eq!(stats.max_area, Some(100.0));
        assert_eq!(stats.mean_area, Some(52.0));
        assert_eq!(stats.visible_fraction(), 0.5);
        // `seq` was only borrowed, so it is still ours to use.
        assert_eq!(seq.name, "stats");
    }

    #[test]
    fn stats_without_visible_frames_have_no_areas() {
        let seq = parse_sequence("# name: e\n# size: 1x1\n0 0 0 0 0 0\n").unwrap();
        let stats = compute_stats(&seq);
        assert_eq!(stats.mean_area, None);
        assert_eq!(stats.visible_fraction(), 0.0);
    }

    #[test]
    fn clipping_modifies_the_sequence_in_place() {
        let mut seq = parse_sequence(TEXT).unwrap();
        assert_eq!(clip_to_image(&mut seq), 1);
        assert_eq!(seq.frames[2].bbox, Some(BBox::new(95.0, 95.0, 5.0, 5.0)));
    }

    #[test]
    fn into_visible_keeps_only_frames_with_a_drone() {
        let seq = parse_sequence(TEXT).unwrap();
        let visible = into_visible(seq); // `seq` is moved here
        assert_eq!(visible.len(), 2);
        assert_eq!(visible[1].0, 2);
    }
}
