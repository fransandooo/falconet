# Exercise 3b — traits and generics

A `Tracker` trait with two implementations, evaluated on a synthetic scene with two small drones, an occlusion and a false positive.

```bash
cargo run -p playground-trackers
cargo nextest run -p playground-trackers
```

Output (the number to look at is *distinct IDs*; the ideal is 3):

```
tracker                                  distinct IDs  track boxes
naive (new ID per detection)                       59           59
greedy IoU (iou>=0.3, max_missed=2)                 4           59
greedy IoU (iou>=0.1, max_missed=5)                 3           59
```

The greedy tracker loses drone B after a two-frame occlusion. It compares new detections with the *last seen* box, and B has moved too far by the time it reappears. Lowering the IoU threshold hides the problem in this scene, but in a real video it would make the tracker match unrelated boxes. The real fix is to predict motion, which ByteTrack does with a Kalman filter in phase 1.

## Reading order

1. [`src/lib.rs`](src/lib.rs): the `Tracker` trait (required and default methods) and the generic `evaluate` function.
2. [`src/naive.rs`](src/naive.rs): the smallest possible `impl Tracker for ...`.
3. [`src/greedy.rs`](src/greedy.rs): a real implementation with private state.
4. [`src/synthetic.rs`](src/synthetic.rs): closures as parameters (`impl Fn`).
5. [`src/main.rs`](src/main.rs): static vs dynamic dispatch side by side.

## Concepts

| Concept | Where | C / Python analogy |
|---|---|---|
| Trait = a set of methods a type promises to provide | `trait Tracker` | an abstract base class / interface; a C struct of function pointers |
| Default methods | `Tracker::run` | a non-abstract method of a base class |
| Generics with trait bounds, static dispatch | `evaluate<T: Tracker + ?Sized>` | C++ templates: one compiled copy per type, calls can be inlined |
| Trait objects, dynamic dispatch | `Vec<Box<dyn Tracker>>` | a pointer to a struct plus a pointer to its vtable |
| `#[derive(Default)]` vs a hand-written `impl Default` | `NaiveTracker` vs `GreedyIouTracker` | a default constructor |
| Visibility: fields and types are private by default | `ActiveTrack`, `next_id` | `static` in a C file; Python's `_private`, but enforced |
| Path dependencies between workspace crates | `Cargo.toml` | linking against another library of the same project |
| Closures as arguments (`impl Fn(..) -> ..`) | `synthetic::generate` | a function pointer plus captured context; a Python lambda |

## Try this

1. **A third tracker.** Implement `CenterDistanceTracker`: like the greedy one, but it matches by the distance between box centres (a match if the distance is under `max_distance` px) instead of IoU. Is it better or worse for drone B? Add it to the `Vec` in `main.rs`; nothing else needs to change. That is the point of the trait.
2. **Constant-velocity prediction.** In `GreedyIouTracker`, store each track's last velocity (the difference between its last two boxes). Before matching, move every box by its velocity times `missed + 1`. With the default parameters, drone B should now keep its ID, giving 3 distinct IDs. This is a hand-made Kalman filter without the uncertainty part.
3. **Score threshold.** Add a `min_score` parameter so that detections below it can't *start* new tracks. What happens to the false positive? (This is half of ByteTrack's idea.)
4. **Static vs dynamic.** Remove `?Sized` from `evaluate` and see which call fails to compile, and why.
