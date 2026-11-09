# AGENTS.md — orb-slam3-rs

A Rust (edition 2024) port of ORB-SLAM3. Single-crate project with one library
(`src/lib.rs`) and one binary (`src/bin/stereo_inertial_euroc.rs`).

---

## Build & Run Commands

```bash
# Build the library and binary (debug)
cargo build

# Build in release mode
cargo build --release

# Run the stereo-inertial EuRoC binary
cargo run --bin stereo_inertial_euroc -- \
    --vocabulary vocabulary/orbvoc.txt \
    --settings config/euroc.yaml \
    --sequence recording/euroc/mav0
```

---

## Lint & Format Commands

No `rustfmt.toml` or `clippy.toml` exist; default settings apply.

```bash
# Format the entire codebase (must pass with no diffs)
cargo fmt

# Check formatting without modifying files
cargo fmt -- --check

# Run Clippy (use for all new code)
cargo clippy

# Clippy in release profile
cargo clippy --release

# Clippy with all warnings treated as errors (CI-style check)
cargo clippy -- -D warnings
```

---

## Test Commands

No tests exist yet. When tests are added, use:

```bash
# Run all tests
cargo test

# Run a single test by name (substring match)
cargo test <test_name>

# Run tests in a specific module
cargo test camera_models::pinhole

# Run tests with output shown
cargo test -- --nocapture

# Run a single integration test file
cargo test --test <integration_test_filename>
```

When writing new tests, place unit tests in an inline `#[cfg(test)]` module at the
bottom of the relevant source file. Integration tests go in a top-level `tests/`
directory.

---

## Project Structure

```
src/
  lib.rs                         # Public module re-exports
  settings.rs                    # YAML config parsing (SettingsError)
  system.rs                      # Top-level System struct (Sensor, SystemError)
  two_view_reconstruction.rs     # Homography/fundamental matrix estimation
  camera_models/
    mod.rs                       # GeometricCamera trait, CameraType enum
    pinhole.rs                   # PinholeCamera implementation
    kannala_brandt8.rs           # Stub (not yet implemented)
  bin/
    stereo_inertial_euroc.rs     # Binary entry point (clap CLI, tracing init)
config/
  euroc.yaml                     # EuRoC dataset camera/IMU configuration
vocabulary/
  orbvoc.txt.tar.gz              # Compressed ORB vocabulary (tracked by git)
```

---

## Code Style Guidelines

### General

- **Rust edition 2024** — use current idioms; `use<>` precise lifetime capturing,
  `impl Trait` in return position, etc.
- Run `cargo fmt` before committing. The `.vscode/settings.json` enables
  format-on-save; all agents should treat `rustfmt` output as canonical.
- Fix all `cargo clippy` warnings before opening a PR.
- Prefer idiomatic slices over references to owned collections:
  `&[T]` instead of `&Vec<T>`, `&str` instead of `&String`.

### Naming Conventions

| Item | Convention | Example |
|---|---|---|
| Types, traits, enums | `PascalCase` | `TwoViewReconstruction`, `CameraType` |
| Enum variants | `PascalCase` | `CameraType::PinHole`, `Sensor::StereoInertial` |
| Functions & methods | `snake_case` | `find_homography`, `compute_f21` |
| Fields & variables | `snake_case` | `max_iterations`, `need_to_undistort` |
| Constants & statics | `SCREAMING_SNAKE_CASE` | `TH_SCORE`, `MIN_PARALLAX` |
| Modules | `snake_case` | `camera_models`, `two_view_reconstruction` |

### Imports

Group `use` statements in this order, separated by blank lines:

1. `std::` imports
2. External crate imports (alphabetical by crate name)
3. `crate::` / `super::` / `self::` imports

Use brace grouping for multiple items from the same path:

```rust
use std::path::PathBuf;
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};

use nalgebra::{Matrix3, SMatrix, SVD};
use opencv::{core::FileStorage, prelude::*};
use tracing::warn;

use crate::settings::Settings;
```

Wildcard imports (`use module::*`) are acceptable only when importing a prelude
(e.g., `use opencv::prelude::*`).

### Error Handling

- Define custom error enums with `#[derive(Debug)]`.
- Return `Result<T, CustomError>` from all fallible public functions.
- Use the `?` operator for propagation; map errors explicitly:
  ```rust
  .map_err(|e| SystemError::InvalidSettings(e))?
  ```
- Reserve `unwrap()` / `expect("reason")` only for values that are **provably**
  infallible (e.g., regex compilation from a literal, mutex lock in single-thread).
  Always add a comment explaining why it cannot fail.
- Avoid `panic!` in library code; prefer returning an `Err(...)`.
- Result-returning `main()` is the preferred pattern:
  ```rust
  fn main() -> Result<(), RuntimeError> { ... }
  ```

### Types & Math

- Use `nalgebra` types for linear algebra: `Matrix3<f32>`, `SMatrix<f32, 3, 4>`,
  `Vector3<f32>`, `SVD`, etc.
- Use `opencv` types for image data and YAML config I/O.
- Prefer `f32` for performance-critical vision math (matching the OpenCV/ORB-SLAM3
  convention). Use `f64` only when numerical precision demands it.
- Use turbofish syntax when type inference is ambiguous:
  `SMatrix::<f32, 3, 4>::zeros()`.

### Structs & Visibility

- All items intended as part of the public API must be `pub` and re-exported
  through `lib.rs`.
- Internal helpers (free functions, private structs) should have no visibility
  modifier (crate-private by default).
- Use named fields; avoid tuple structs except for simple newtype wrappers.
- Derive common traits where applicable: `#[derive(Debug, Clone, Copy)]`.

### Comments & Documentation

- Add `///` doc comments to all public types, traits, and functions.
- Use `//!` module-level doc comments in each `mod.rs` and `lib.rs`.
- Inline `//` comments should explain *why*, not *what*.
- Mark incomplete code with `// TODO:` followed by a brief description.
- Mathematical notation in comments is encouraged (Unicode: `Σ`, `ᵀ`, etc.).

### Concurrency

- Use `Arc<T>` to share immutable data across threads.
- Use `std::thread::spawn` for coarse-grained parallelism (no `async`/`tokio`).
- Atomic types (`AtomicU64`, `Ordering::SeqCst`) are used for global ID counters.

### Logging

- Use the `tracing` crate exclusively for structured logging.
- Initialize with `tracing_subscriber::fmt::init()` in binary entry points only.
- Use appropriate macros: `tracing::info!`, `tracing::warn!`, `tracing::debug!`,
  `tracing::error!`.
- Never use `println!` / `eprintln!` in library code.

---

## Dependencies & Features

- `clap` with `features = ["derive"]` for CLI argument parsing.
- `nalgebra` for linear algebra.
- `opencv` for image processing and YAML I/O.
- `rand` for random sampling (RANSAC).
- `tracing` + `tracing-subscriber` for logging.

---

## Key Invariants

- The ORB vocabulary file (`vocabulary/orbvoc.txt`) is **not** tracked by git. There is a build script to handle it.
- Recording data (`recording/`) is gitignored; download the EuRoC dataset
  separately.
