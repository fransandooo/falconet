//! Rendering predictions to images: the "watch the model learn" frames.
//!
//! Layout of one frame, for N showcase scenes:
//!
//! ```text
//! ┌────────┬────────┬─────┐
//! │ scene 1│ scene 2│ ... │  input image, green = truth, red = prediction
//! ├────────┼────────┼─────┤
//! │ heat 1 │ heat 2 │ ... │  predicted heatmap ("where is the centre?")
//! └────────┴────────┴─────┘
//! ▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░   progress bar: epoch / total
//! ```

use std::io::Cursor;
use std::path::Path;

use image::codecs::gif::{GifEncoder, Repeat};
use image::{Delay, Frame, ImageFormat, Rgb, RgbImage, RgbaImage, imageops};
use playground_annotations::BBox;

use crate::STRIDE;
use crate::decode::Prediction;
use crate::scene::Scene;

const GREEN: Rgb<u8> = Rgb([40, 220, 90]);
const RED: Rgb<u8> = Rgb([255, 60, 60]);
const GAP: u32 = 4;
const BAR_HEIGHT: u32 = 10;

/// Draws one frame. `scale` enlarges every pixel (64 px images are tiny),
/// `progress` in `[0, 1]` fills the bar at the bottom.
pub fn render_frame(
    scenes: &[Scene],
    predictions: &[Prediction],
    image_size: usize,
    scale: u32,
    progress: f32,
) -> RgbImage {
    let tile = image_size as u32 * scale;
    let n = scenes.len() as u32;
    let width = n * tile + (n + 1) * GAP;
    let height = 2 * tile + 3 * GAP + BAR_HEIGHT + GAP;
    let mut frame = RgbImage::from_pixel(width, height, Rgb([24, 24, 28]));

    // `zip` walks both slices in lock step, like a two-index for loop in C.
    for (i, (scene, pred)) in scenes.iter().zip(predictions).enumerate() {
        let x0 = GAP + i as u32 * (tile + GAP);
        draw_scene(&mut frame, scene, image_size, scale, x0, GAP);
        draw_heatmap(
            &mut frame,
            &pred.heatmap,
            image_size,
            scale,
            x0,
            2 * GAP + tile,
        );

        draw_box(&mut frame, &scene.bbox, scale, x0, GAP, GREEN);
        draw_box(&mut frame, &pred.bbox, scale, x0, GAP, RED);
        draw_box(&mut frame, &scene.bbox, scale, x0, 2 * GAP + tile, GREEN);
    }

    // Progress bar.
    let bar_y = 2 * tile + 3 * GAP;
    let filled = ((width - 2 * GAP) as f32 * progress.clamp(0.0, 1.0)) as u32;
    for x in GAP..width - GAP {
        let color = if x - GAP < filled {
            Rgb([230, 230, 230])
        } else {
            Rgb([60, 60, 66])
        };
        for y in bar_y..bar_y + BAR_HEIGHT {
            frame.put_pixel(x, y, color);
        }
    }
    frame
}

fn draw_scene(frame: &mut RgbImage, scene: &Scene, size: usize, scale: u32, x0: u32, y0: u32) {
    for (i, &p) in scene.pixels.iter().enumerate() {
        let v = (p * 255.0) as u8;
        fill_block(
            frame,
            x0,
            y0,
            (i % size) as u32,
            (i / size) as u32,
            scale,
            Rgb([v, v, v]),
        );
    }
}

fn draw_heatmap(frame: &mut RgbImage, heat: &[f32], size: usize, scale: u32, x0: u32, y0: u32) {
    let grid = size / STRIDE;
    let cell = STRIDE as u32 * scale;
    for (i, &v) in heat.iter().enumerate() {
        fill_block(
            frame,
            x0,
            y0,
            (i % grid) as u32,
            (i / grid) as u32,
            cell,
            hot(v),
        );
    }
}

/// Fills the `block x block` square at grid position (`col`, `row`).
fn fill_block(frame: &mut RgbImage, x0: u32, y0: u32, col: u32, row: u32, block: u32, c: Rgb<u8>) {
    for dy in 0..block {
        for dx in 0..block {
            frame.put_pixel(x0 + col * block + dx, y0 + row * block + dy, c);
        }
    }
}

/// 2-px rectangle outline, clipped to the tile.
fn draw_box(frame: &mut RgbImage, b: &BBox, scale: u32, x0: u32, y0: u32, color: Rgb<u8>) {
    let tile = (frame.height() - 3 * GAP - BAR_HEIGHT - GAP) / 2;
    let s = scale as f32;
    let clamp = |v: f32| v.clamp(0.0, (tile - 1) as f32) as u32;
    let (left, top) = (clamp(b.x * s), clamp(b.y * s));
    let (right, bottom) = (clamp(b.right() * s), clamp(b.bottom() * s));
    for t in 0..2 {
        for x in left..=right {
            frame.put_pixel(x0 + x, y0 + (top + t).min(bottom), color);
            frame.put_pixel(x0 + x, y0 + bottom.saturating_sub(t).max(top), color);
        }
        for y in top..=bottom {
            frame.put_pixel(x0 + (left + t).min(right), y0 + y, color);
            frame.put_pixel(x0 + right.saturating_sub(t).max(left), y0 + y, color);
        }
    }
}

/// "Hot" colormap: black → red → yellow → white.
fn hot(v: f32) -> Rgb<u8> {
    let v = v.clamp(0.0, 1.0);
    let channel = |x: f32| (x.clamp(0.0, 1.0) * 255.0) as u8;
    Rgb([
        channel(3.0 * v),
        channel(3.0 * v - 1.0),
        channel(3.0 * v - 2.0),
    ])
}

/// PNG bytes of a frame (what gets uploaded to MLflow).
pub fn encode_png(frame: &RgbImage) -> Vec<u8> {
    encode(frame, ImageFormat::Png)
}

/// A smaller WebP copy: MLflow's UI shows these in its image grid.
pub fn encode_webp_thumbnail(frame: &RgbImage, max_width: u32) -> Vec<u8> {
    let scale = (max_width as f32 / frame.width() as f32).min(1.0);
    let h = (frame.height() as f32 * scale) as u32;
    let small = imageops::resize(frame, max_width, h, imageops::FilterType::Triangle);
    encode(&small, ImageFormat::WebP)
}

fn encode(frame: &RgbImage, format: ImageFormat) -> Vec<u8> {
    let mut bytes = Cursor::new(Vec::new());
    frame
        .write_to(&mut bytes, format)
        .expect("encoding to memory cannot fail");
    bytes.into_inner()
}

/// Writes an endlessly looping GIF; the last frame is held longer.
pub fn write_gif(frames: &[RgbImage], path: &Path, frame_ms: u32) -> image::ImageResult<()> {
    let file = std::fs::File::create(path)?;
    let mut encoder = GifEncoder::new_with_speed(file, 10);
    encoder.set_repeat(Repeat::Infinite)?;
    for (i, frame) in frames.iter().enumerate() {
        let ms = if i + 1 == frames.len() {
            2_000
        } else {
            frame_ms
        };
        let rgba: RgbaImage = image::DynamicImage::ImageRgb8(frame.clone()).to_rgba8();
        encoder.encode_frame(Frame::from_parts(
            rgba,
            0,
            0,
            Delay::from_numer_denom_ms(ms, 1),
        ))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::SceneConfig;

    #[test]
    fn frame_has_the_expected_size_and_draws_boxes() {
        let cfg = SceneConfig::default();
        let scenes: Vec<Scene> = (0..3).map(|i| cfg.generate(i)).collect();
        let preds: Vec<Prediction> = scenes
            .iter()
            .map(|s| Prediction {
                bbox: s.bbox,
                score: 1.0,
                heatmap: vec![0.5; 256],
            })
            .collect();
        let frame = render_frame(&scenes, &preds, 64, 2, 0.5);
        assert_eq!(frame.width(), 3 * 128 + 4 * GAP);
        // The prediction equals the truth, so red is drawn over green at the box corner.
        let b = scenes[0].bbox;
        let corner = frame.get_pixel(GAP + (b.x * 2.0) as u32, GAP + (b.y * 2.0) as u32);
        assert_eq!(*corner, RED);
        assert!(!encode_png(&frame).is_empty());
        assert!(!encode_webp_thumbnail(&frame, 200).is_empty());
    }

    #[test]
    fn hot_colormap_endpoints() {
        assert_eq!(hot(0.0), Rgb([0, 0, 0]));
        assert_eq!(hot(1.0), Rgb([255, 255, 255]));
    }
}
