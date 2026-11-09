//! Viewer-facing, read-only SLAM snapshots.
//!
//! The SLAM core intentionally does not depend on a GUI toolkit. Desktop
//! frontends consume immutable snapshots and send control commands through
//! `System`, keeping map locks and OpenCV handles off the GUI thread.

use nalgebra::{Isometry3, Vector3};
use opencv::core::{Mat, MatTraitConst};
use opencv::prelude::MatTraitConstManual;

use crate::tracking::TrackingState;

/// Placeholder for the viewer handle stored by `System` and `Atlas`.
///
/// The concrete snapshot transport is added with the desktop frontend.
#[derive(Debug, Default)]
pub struct Viewer {}

#[derive(Debug, Clone)]
pub struct ViewerImage {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}

impl ViewerImage {
    pub fn from_mat(image: &Mat) -> Option<Self> {
        if image.empty() {
            return None;
        }
        let width = image.cols() as usize;
        let height = image.rows() as usize;
        let channels = image.channels() as usize;
        let stride = image.step1(0).ok()?;
        let bytes = image.data_bytes().ok()?;
        if width == 0 || height == 0 {
            return None;
        }

        let mut rgba = vec![0u8; width * height * 4];
        for y in 0..height {
            let row = &bytes[y * stride..];
            for x in 0..width {
                let target = (y * width + x) * 4;
                let source = x * channels;
                match channels {
                    1 => {
                        let gray = row[source];
                        rgba[target] = gray;
                        rgba[target + 1] = gray;
                        rgba[target + 2] = gray;
                        rgba[target + 3] = 255;
                    }
                    3 => {
                        rgba[target] = row[source + 2];
                        rgba[target + 1] = row[source + 1];
                        rgba[target + 2] = row[source];
                        rgba[target + 3] = 255;
                    }
                    4 => {
                        rgba[target] = row[source + 2];
                        rgba[target + 1] = row[source + 1];
                        rgba[target + 2] = row[source];
                        rgba[target + 3] = row[source + 3];
                    }
                    _ => return None,
                }
            }
        }
        Some(Self {
            width,
            height,
            rgba,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ViewerKeyFrame {
    pub id: u64,
    pub pose: Isometry3<f32>,
    pub parent: Option<u64>,
    pub connections: Vec<u64>,
    pub loop_edges: Vec<u64>,
    pub next_keyframe: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
pub struct ViewerFeature {
    pub x: f32,
    pub y: f32,
    pub has_map_point: bool,
    pub outlier: bool,
    pub right: bool,
}

#[derive(Debug, Clone)]
pub struct ViewerSnapshot {
    pub state: TrackingState,
    pub image: Option<ViewerImage>,
    pub image_right: Option<ViewerImage>,
    pub current_pose: Option<Isometry3<f32>>,
    pub map_points: Vec<Vector3<f32>>,
    pub keyframes: Vec<ViewerKeyFrame>,
    pub features: Vec<ViewerFeature>,
    pub feature_count: usize,
    pub inlier_count: usize,
    pub map_point_count: usize,
    pub map_changed: bool,
    pub imu_initialized: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::make_image;

    #[test]
    fn gray_image_converts_to_rgba_texture() {
        let image = ViewerImage::from_mat(&make_image()).expect("convert image");
        assert_eq!(image.width, 752);
        assert_eq!(image.height, 480);
        assert_eq!(image.rgba.len(), 752 * 480 * 4);
        assert!(image.rgba.chunks_exact(4).all(|pixel| pixel[3] == 255));
    }
}
