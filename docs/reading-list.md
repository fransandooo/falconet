# Phase 0 reading list

About 1–2 h per week, matched to the exercises in [`playground/`](../playground/). "Skim" means read the examples and skip the prose you already know from C/Python.

Sources:
- *The Rust Programming Language* ("the Book"): <https://doc.rust-lang.org/book/> (or offline with `rustup doc --book`).
- *The Burn Book*: <https://burn.dev/books/burn/>. **Warning:** Burn 0.22 changed the core API (tensors are no longer generic over a `Backend`, and the backend is picked through `Device`). Prefer examples written for 0.22; older blog posts and tutorials won't compile as-is.

## Week 1 — ownership (exercise 3a)

| Read | Why |
|---|---|
| Book ch. 3 — *Common Programming Concepts* (skim) | syntax, mutability, expressions vs statements |
| Book ch. 4 — *Understanding Ownership* (**carefully**) | moves, borrows, slices: the foundation of everything |
| Book ch. 5 — *Structs* (skim) | `impl` blocks, methods, `&self` |

## Week 2 — enums, errors, generics, traits (exercises 3a + 3b)

| Read | Why |
|---|---|
| Book ch. 6 — *Enums and Pattern Matching* | `Option`, `match`, `if let` |
| Book ch. 9 — *Error Handling* | `Result`, `?`, when to `panic!` |
| Book ch. 10 — *Generic Types, Traits, and Lifetimes* (**carefully**) | trait bounds, default methods, lifetime annotations |
| Book ch. 18.2 — *Using Trait Objects…* | `dyn Trait` and dynamic dispatch |

## Week 3 — idiomatic code and Burn (before exercise 3c)

| Read | Why |
|---|---|
| Book ch. 13 — *Iterators and Closures* | the iterator chains used everywhere in the playground |
| Book ch. 7 and 14.3 — *Modules* and *Cargo Workspaces* (skim) | how this repository is organised |
| Book ch. 11 — *Writing Automated Tests* (skim) | unit tests, `#[cfg(test)]`, doc tests |
| Burn Book — *Getting Started* and the *Basic Workflow* guide (model → data → training → inference) | the structure we'll copy for MNIST and the toy detector |

## Later, when the topic comes up

- Book ch. 15 — *Smart Pointers* (`Box`, `Rc`, `RefCell`): before writing the data pipeline (phase 2).
- Book ch. 16 — *Fearless Concurrency*, and the `rayon` docs: phase 3 optimisation.
- *The Rustonomicon* (unsafe Rust): only if phase 4 needs it.
