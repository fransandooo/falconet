//! Demo for exercise 3a: parse a toy annotation file and print statistics.
//!
//! ```text
//! cargo run -p playground-annotations                 # bundled example
//! cargo run -p playground-annotations -- <file.txt>   # your own file
//! ```

use std::env;
use std::error::Error;
use std::fs;

use playground_annotations::{
    clip_to_image, compute_stats, into_visible, parse_header, parse_sequence,
};

/// `concat!` and `env!` run at compile time: this becomes an absolute path to
/// the example file inside this crate.
const DEFAULT_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/toy_sequence.txt");

// `main` can return a `Result`. With `Box<dyn Error>` as the error type, `?`
// accepts any error (I/O errors, our `ParseError`, ...). If an error reaches
// here, Rust prints it and exits with status code 1.
fn main() -> Result<(), Box<dyn Error>> {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_PATH.to_owned());

    // `text` OWNS the file contents: a heap-allocated `String`. It is freed
    // automatically when `main` ends, because `text` goes out of scope.
    let text = fs::read_to_string(&path)?;

    // --- 1. Borrowing --------------------------------------------------------
    // `&text` lends the string to `parse_header`. The returned `header` holds
    // `&str` slices pointing *into* `text`: nothing is copied.
    let header = parse_header(&text)?;

    // drop(text);
    // ^ Uncomment to free `text` while `header` still points into it:
    //   error[E0505]: cannot move out of `text` because it is borrowed
    //   In C this would compile and later read freed memory.

    println!(
        "sequence '{}' ({}x{} px)",
        header.name, header.width, header.height
    );

    // --- 2. Ownership --------------------------------------------------------
    // `seq` owns its own copy of everything it needs. It no longer depends on
    // `text`, which we could drop here without problems.
    let mut seq = parse_sequence(&text)?;

    let stats = compute_stats(&seq); // shared borrow: `seq` stays ours
    println!(
        "{} frames, drone visible in {} ({:.0}%)",
        stats.frames,
        stats.visible,
        stats.visible_fraction() * 100.0
    );
    if let (Some(min), Some(mean), Some(max)) = (stats.min_area, stats.mean_area, stats.max_area) {
        println!("box area px²: min {min:.0}, mean {mean:.1}, max {max:.0}");
        println!(
            "the mean box covers {:.4}% of the image",
            mean / (header.width * header.height) as f32 * 100.0
        );
    }

    // --- 3. Exclusive (mutable) borrow ----------------------------------------
    let changed = clip_to_image(&mut seq);
    println!("clipped {changed} box(es) to the image bounds");

    // --- 4. Move ---------------------------------------------------------------
    // `seq` is moved into `into_visible`, which now owns (and will free) it.
    let visible = into_visible(seq);

    // println!("{}", seq.name);
    // ^ Uncomment to use `seq` after the move:
    //   error[E0382]: borrow of moved value: `seq`

    println!(
        "{} frames with a visible drone after clipping",
        visible.len()
    );
    if let Some((frame, bbox)) = visible.last() {
        println!("last visible box: frame {frame}, {bbox:?}");
    }

    Ok(())
}
