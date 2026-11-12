//! Bounding boxes: the most basic type of the whole project.

/// Axis-aligned bounding box in pixel coordinates: top-left corner plus size.
///
/// `#[derive(...)]` asks the compiler to generate trait implementations:
/// - `Debug`: printable with `{:?}` (logs, test failure messages),
/// - `Clone` + `Copy`: assigning or passing a box copies its 16 bytes, exactly
///   like a C struct, and the original stays usable. Only types made of `Copy`
///   parts can be `Copy`; anything owning heap memory (`String`, `Vec`) cannot,
///   because two owners would both try to free the same buffer,
/// - `PartialEq`: comparable with `==`. "Partial" because `f32::NAN != f32::NAN`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl BBox {
    /// Creates a box. `Self` is an alias for the type being implemented (`BBox`).
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        // Field init shorthand: `x` means `x: x`.
        Self { x, y, w, h }
    }

    /// `&self` borrows the box read-only, like a `const BBox *self` in C.
    pub fn area(&self) -> f32 {
        self.w * self.h
    }

    /// Returns a tuple: Rust's lightweight way to return several values.
    pub fn center(&self) -> (f32, f32) {
        (self.x + self.w / 2.0, self.y + self.h / 2.0)
    }

    pub fn right(&self) -> f32 {
        self.x + self.w
    }

    pub fn bottom(&self) -> f32 {
        self.y + self.h
    }

    /// Intersection over union, in `[0, 1]`: the standard overlap measure used
    /// by detection metrics (mAP) and by trackers to match boxes across frames.
    pub fn iou(&self, other: &BBox) -> f32 {
        let inter_w = (self.right().min(other.right()) - self.x.max(other.x)).max(0.0);
        let inter_h = (self.bottom().min(other.bottom()) - self.y.max(other.y)).max(0.0);
        let inter = inter_w * inter_h;
        let union = self.area() + other.area() - inter;
        // `if` is an expression in Rust: both branches produce the return value.
        if union > 0.0 { inter / union } else { 0.0 }
    }

    /// Takes `self` *by value*. Because `BBox` is `Copy`, the caller keeps its
    /// own copy, so this reads like a pure function:
    /// `let b2 = b.translated(1.0, 0.0);` leaves `b` untouched.
    pub fn translated(self, dx: f32, dy: f32) -> Self {
        // Struct update syntax: take `w` and `h` from `self`.
        Self {
            x: self.x + dx,
            y: self.y + dy,
            ..self
        }
    }

    /// Clips the box to a `width` x `height` image.
    ///
    /// Returns `Option<BBox>`: `Some(box)` if part of it is inside the image,
    /// `None` if nothing is left. Rust has no null pointers; "maybe a value" is
    /// spelled `Option`, and the compiler forces callers to handle `None`.
    pub fn clipped(self, width: f32, height: f32) -> Option<Self> {
        let x0 = self.x.clamp(0.0, width);
        let y0 = self.y.clamp(0.0, height);
        let x1 = self.right().clamp(0.0, width);
        let y1 = self.bottom().clamp(0.0, height);
        if x1 > x0 && y1 > y0 {
            Some(Self::new(x0, y0, x1 - x0, y1 - y0))
        } else {
            None
        }
    }
}

// Unit tests live next to the code, in a module compiled only for `cargo test`.
#[cfg(test)]
mod tests {
    use super::*; // bring everything from the parent module into scope

    #[test]
    fn iou_of_identical_boxes_is_one() {
        let b = BBox::new(10.0, 10.0, 4.0, 4.0);
        assert_eq!(b.iou(&b), 1.0);
    }

    #[test]
    fn iou_of_disjoint_boxes_is_zero() {
        let a = BBox::new(0.0, 0.0, 2.0, 2.0);
        let b = BBox::new(5.0, 5.0, 2.0, 2.0);
        assert_eq!(a.iou(&b), 0.0);
    }

    #[test]
    fn iou_of_half_shifted_boxes_is_one_third() {
        // Overlap 1x2 = 2, union 4 + 4 - 2 = 6.
        let a = BBox::new(0.0, 0.0, 2.0, 2.0);
        let b = BBox::new(1.0, 0.0, 2.0, 2.0);
        assert!((a.iou(&b) - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn translated_leaves_the_original_untouched() {
        let a = BBox::new(1.0, 2.0, 3.0, 4.0);
        let b = a.translated(10.0, 0.0);
        // `a` is still usable: it was copied, not moved.
        assert_eq!(a.x, 1.0);
        assert_eq!(b, BBox::new(11.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn clipping_keeps_the_part_inside_the_image() {
        let b = BBox::new(636.0, 510.0, 10.0, 8.0);
        assert_eq!(
            b.clipped(640.0, 512.0),
            Some(BBox::new(636.0, 510.0, 4.0, 2.0))
        );
    }

    #[test]
    fn clipping_a_box_outside_the_image_gives_none() {
        let b = BBox::new(641.0, 513.0, 10.0, 8.0);
        assert_eq!(b.clipped(640.0, 512.0), None);
    }
}
