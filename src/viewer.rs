//! Viewer-facing, read-only SLAM snapshots.
//!
//! The SLAM core intentionally does not depend on a GUI toolkit. Desktop
//! frontends consume immutable snapshots and send control commands through
//! `System`, keeping map locks and OpenCV handles off the GUI thread.

/// Placeholder for the viewer handle stored by `System` and `Atlas`.
///
/// The concrete snapshot transport is added with the desktop frontend.
#[derive(Debug, Default)]
pub struct Viewer {}
