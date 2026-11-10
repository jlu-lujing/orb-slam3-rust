# Contributing

## Before you open a pull request

The repository is set up for three checks, and CI runs them:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
```

`AGENTS.md` documents the style conventions used throughout (edition 2024,
`f32` for vision math, `tracing` instead of `println!`, doc comments on public
items, no panics in library code where an error can be returned).

## Keeping parity with the C++ implementation

This crate is a port, so behaviour changes are judged against the original.
When you touch tracking, mapping or the optimizer:

1. Cite the C++ function you are matching, and keep the port readable next to it.
2. If you intentionally deviate, say why in a comment at the call site and in
   `PARITY_AUDIT.md`, with the measurement that justifies it.
3. Trajectory numbers are noisy. `PARITY_AUDIT.md` explains how to run the
   comparison (`trajectory_eval`, `--max-diff 0.02`, medians of at least three
   runs); a single run is not evidence.

New tests belong in an inline `#[cfg(test)]` module for unit tests or in
`tests/` for integration tests. Numerical code is expected to come with a
finite-difference or fixture comparison.

## License

By contributing you agree that your changes are released under the GPLv3, the
same license as ORB-SLAM3 itself.
