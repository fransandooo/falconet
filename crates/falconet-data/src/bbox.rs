//! Axis-aligned bounding boxes in pixel coordinates.

use serde::{Deserialize, Serialize};

/// Top-left corner plus size, in pixels: the `x, y, w, h` format used by the
/// Anti-UAV annotations and by its official evaluation code.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl BBox {
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    pub fn area(&self) -> f32 {
        self.w * self.h
    }

    pub fn center(&self) -> (f32, f32) {
        (self.x + self.w / 2.0, self.y + self.h / 2.0)
    }

    pub fn right(&self) -> f32 {
        self.x + self.w
    }

    pub fn bottom(&self) -> f32 {
        self.y + self.h
    }

    /// Length of the box diagonal: Anti-UAV410's size measure.
    pub fn diagonal(&self) -> f32 {
        self.w.hypot(self.h)
    }

    /// Intersection over union, matching the official evaluation code
    /// (no overlap → 0; boxes are treated as continuous rectangles).
    pub fn iou(&self, other: &BBox) -> f32 {
        let iw = self.right().min(other.right()) - self.x.max(other.x);
        let ih = self.bottom().min(other.bottom()) - self.y.max(other.y);
        if iw <= 0.0 || ih <= 0.0 {
            return 0.0;
        }
        let inter = iw * ih;
        inter / (self.area() + other.area() - inter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iou_matches_hand_computation() {
        let a = BBox::new(0.0, 0.0, 2.0, 2.0);
        let b = BBox::new(1.0, 0.0, 2.0, 2.0);
        assert!((a.iou(&b) - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(a.iou(&BBox::new(5.0, 5.0, 1.0, 1.0)), 0.0);
        // Touching edges: zero-area overlap counts as no overlap.
        assert_eq!(a.iou(&BBox::new(2.0, 0.0, 2.0, 2.0)), 0.0);
    }

    #[test]
    fn diagonal_of_a_3_4_box_is_5() {
        assert_eq!(BBox::new(0.0, 0.0, 3.0, 4.0).diagonal(), 5.0);
    }
}
