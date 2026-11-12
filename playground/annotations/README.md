# Exercise 3a — ownership and borrowing

A parser for a toy annotation format modelled on the Anti-UAV datasets (one line per frame: is the drone visible, and where).

```bash
cargo run -p playground-annotations                  # bundled example
cargo run -p playground-annotations -- my_file.txt   # your own file
cargo nextest run -p playground-annotations          # tests
```

## Reading order

1. [`src/bbox.rs`](src/bbox.rs): structs, `impl` blocks, `Copy` types, `&self` vs `self`, `Option`.
2. [`src/parser.rs`](src/parser.rs): `&str` vs `String`, lifetimes (`Header<'a>`), enums as tagged unions, `Result` and `?`, closures, a first generic function.
3. [`src/stats.rs`](src/stats.rs): one function per way of passing data (`&`, `&mut`, by value), iterator chains.
4. [`src/main.rs`](src/main.rs): everything together, plus two commented-out lines that the borrow checker rejects.

## Concepts

| Concept | Where | C / Python analogy |
|---|---|---|
| Ownership: every value has exactly one owner and is freed when the owner goes out of scope | `Sequence`, `main.rs` | `malloc`/`free` done by the compiler; no GC like Python |
| Move: passing by value transfers ownership; the old variable can't be used | `into_visible(seq)` | like handing over a pointer and promising never to touch it again, but checked |
| `Copy` types are duplicated instead of moved | `BBox` | plain C struct assignment |
| Shared borrow `&T`: many readers at once | `compute_stats(&seq)` | `const T *` |
| Exclusive borrow `&mut T`: one writer, no readers | `clip_to_image(&mut seq)` | `T *` + a guarantee that it isn't aliased |
| `&str` (borrowed slice) vs `String` (owned buffer) | `Header` vs `Sequence` | `const char *` + length vs a `char *` you own |
| Lifetimes `'a`: a borrow can't outlive what it points to | `Header<'a>`, `parse_header` | prevents returning pointers into freed buffers |
| `Option<T>` instead of null or magic flags | `FrameAnnotation::bbox` | `None` in Python, but the compiler makes you check it |
| `Result<T, E>` and `?` | the whole parser | errno / exceptions, but explicit in the type |

## Try this

1. **Read the compiler errors.** Uncomment the `drop(text)` line in `main.rs`, run `cargo build -p playground-annotations`, and read the error. Then put the comment back and do the same with the `println!("{}", seq.name)` line. Rust's error messages explain a lot; get used to reading them in full.
2. **Add a header field.** Support an optional `# fps: 30` line: add `fps: Option<u32>` to `Header` and `Sequence`, and a test.
3. **Longest gap.** Write `pub fn longest_gap(seq: &Sequence) -> usize` in `stats.rs`: the longest run of consecutive frames without a drone (2 for the example file). Write the test first.
4. **No allocation.** `compute_stats` collects the areas into a `Vec` and then walks it three times. Rewrite it as a single pass using `fold` (or a plain `for` loop) without allocating.
5. **Borrow-checker puzzle.** In `clip_to_image`, try moving `let (width, height) = ...` inside the loop as `seq.width as f32`. Does it compile? Why? (Hint: which part of `seq` is borrowed mutably?)
