// Shared fixtures for the inline parity tests in `frame.rs`, `map_point.rs` and `key_frame.rs`.

#![allow(clippy::excessive_precision)]

use std::sync::Arc;

use nalgebra::{Isometry3, Matrix3, Translation3, UnitQuaternion, Vector3};
use opencv::core::{CV_8UC1, KeyPoint, Mat, MatTrait, MatTraitConst, Point2f, Scalar};

use crate::camera_models::GeometricCamera;
use crate::camera_models::pinhole::Pinhole;
use crate::frame::{Frame, FrameConstants};
use crate::imu_types::Calib;
use crate::orb_extractor::OrbExtractor;
use crate::orb_vocabulary::OrbVocabulary;

// Canonical calibration (EuRoC cam0)
pub(crate) const FX: f32 = 458.654;
pub(crate) const FY: f32 = 457.296;
pub(crate) const CX: f32 = 367.215;
pub(crate) const CY: f32 = 248.375;
pub(crate) const K1: f32 = -0.28340811;
pub(crate) const K2: f32 = 0.07395907;
pub(crate) const P1: f32 = 0.00019359;
pub(crate) const P2: f32 = 1.76187114e-05;
pub(crate) const IMG_W: i32 = 752;
pub(crate) const IMG_H: i32 = 480;
pub(crate) const BF: f32 = 47.0 * 0.01 * FX;
pub(crate) const TH_DEPTH: f32 = 35.0;

// Float comparison

pub(crate) fn approx(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps + eps * b.abs()
}
pub(crate) fn assert_vec(got: Vector3<f32>, exp: [f32; 3], eps: f32) {
    for i in 0..3 {
        assert!(
            approx(got[i], exp[i], eps),
            "component {i}: got {}, expected {}",
            got[i],
            exp[i]
        );
    }
}
pub(crate) fn assert_mat3(got: Matrix3<f32>, exp: [f32; 9], eps: f32) {
    for r in 0..3 {
        for c in 0..3 {
            let e = exp[r * 3 + c];
            assert!(
                approx(got[(r, c)], e, eps),
                "({r},{c}): got {}, expected {e}",
                got[(r, c)]
            );
        }
    }
}

// Builders

pub(crate) fn make_image() -> Mat {
    let mut img = Mat::new_rows_cols_with_default(IMG_H, IMG_W, CV_8UC1, Scalar::all(0.0)).unwrap();
    for y in 0..IMG_H {
        for x in 0..IMG_W {
            let a = (x / 16) & 1;
            let b = (y / 16) & 1;
            let v: u8 = if (a ^ b) != 0 { 220 } else { 30 };
            *img.at_2d_mut::<u8>(y, x).unwrap() = v;
        }
    }
    img
}

pub(crate) fn k_mat() -> Mat {
    Mat::from_slice_2d(&[[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]]).unwrap()
}
pub(crate) fn dist_mat() -> Mat {
    Mat::from_slice(&[K1, K2, P1, P2])
        .unwrap()
        .try_clone()
        .unwrap()
}
pub(crate) fn constants() -> Arc<FrameConstants> {
    Arc::new(FrameConstants::new(k_mat(), dist_mat(), IMG_W, IMG_H).unwrap())
}
pub(crate) fn camera() -> Arc<dyn GeometricCamera> {
    Arc::new(Pinhole::with_params(vec![FX, FY, CX, CY]))
}
pub(crate) fn extractor() -> Arc<OrbExtractor> {
    Arc::new(OrbExtractor::new(1000, 1.2, 8, 20, 7))
}
pub(crate) fn dummy_voc() -> Arc<OrbVocabulary> {
    // Minimal valid DBoW2 header, no nodes -> empty vocabulary.
    Arc::new(OrbVocabulary::load_from_reader("1 1 0 0\n".as_bytes()).unwrap())
}

/// Tcw = exp([0.1, 0.2, 0.3]), t = (1, 2, 3).
pub(crate) fn make_pose() -> Isometry3<f32> {
    let rot = UnitQuaternion::from_scaled_axis(Vector3::new(0.1, 0.2, 0.3));
    Isometry3::from_parts(Translation3::new(1.0, 2.0, 3.0), rot)
}
/// IMU calibration with a non-trivial Tbc, matching the C++ tests.
pub(crate) fn make_calib() -> Calib {
    let rot = UnitQuaternion::from_scaled_axis(Vector3::new(0.01, -0.02, 0.03));
    let tbc = Isometry3::from_parts(Translation3::new(0.05, -0.02, 0.01), rot);
    Calib::from_params(tbc, 1.0, 1.0, 1.0, 1.0)
}

/// Deterministic pseudo-random texture: distinct ORB descriptors (a
/// checkerboard is too repetitive and fails ratio tests on self-matches).
pub(crate) fn make_noise_image() -> Mat {
    let mut img = Mat::new_rows_cols_with_default(IMG_H, IMG_W, CV_8UC1, Scalar::all(0.0)).unwrap();
    for y in 0..IMG_H {
        for x in 0..IMG_W {
            let idx = (y * IMG_W + x) as u32;
            let v = idx.wrapping_mul(1103515245).wrapping_add(12345) >> 16;
            *img.at_2d_mut::<u8>(y, x).unwrap() = (v & 0xFF) as u8;
        }
    }
    img
}

pub(crate) fn build_frame_from(img: &Mat) -> Frame {
    Frame::from_monocular(
        img,
        0.0,
        extractor(),
        dummy_voc(),
        constants(),
        BF,
        TH_DEPTH,
        camera(),
        None,
        Arc::new(make_calib()),
    )
    .unwrap()
}

pub(crate) fn build_frame() -> Frame {
    build_frame_from(&make_image())
}

pub(crate) fn keypoint(x: f32, y: f32) -> KeyPoint {
    KeyPoint::new_point_def(Point2f::new(x, y), 1.0).unwrap()
}
