//! Robust perspective-n-point for ORB-SLAM relocalization.
//!
//! ORB-SLAM3 uses MLPnP directly on bearing vectors. This module provides the
//! same useful interface over OpenCV's EPnP-RANSAC by first converting every
//! observation through the camera model to normalized image coordinates.

use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
use opencv::calib3d::{SOLVEPNP_EPNP, rodrigues, solve_pnp_ransac as cv_solve_pnp_ransac};
use opencv::core::{CV_32F, Mat, MatTraitConstManual, Point2f, Point3f};
use opencv::prelude::*;

/// One 3D↔bearing pair. `uv` uses normalized coordinates `(x/z, y/z)`.
#[derive(Debug, Clone, Copy)]
pub struct PnpCorrespondence {
    pub world: Point3f,
    pub bearing: Point2f,
}

/// Result of [`solve_pnp_ransac`].
#[derive(Debug, Clone)]
pub struct PnpSolution {
    /// World-to-camera pose.
    pub tcw: Isometry3<f32>,
    /// Input indices retained by RANSAC.
    pub inliers: Vec<usize>,
}

/// Solve world-to-camera pose from normalized image coordinates.
///
/// At least six correspondences are required, matching ORB-SLAM3's MLPnP
/// configuration.
pub fn solve_pnp_ransac(correspondences: &[PnpCorrespondence]) -> Option<PnpSolution> {
    if correspondences.len() < 6 {
        return None;
    }

    let object_vec: Vec<_> = correspondences
        .iter()
        .map(|correspondence| correspondence.world)
        .collect();
    let image_vec: Vec<_> = correspondences
        .iter()
        .map(|correspondence| correspondence.bearing)
        .collect();
    let object_points = Mat::from_slice(&object_vec).ok()?;
    let image_points = Mat::from_slice(&image_vec).ok()?;
    let camera_matrix = Mat::eye(3, 3, CV_32F).ok()?;
    let dist_coeffs = Mat::default();
    let mut rvec = Mat::default();
    let mut tvec = Mat::default();
    let mut inliers = Mat::default();
    let solved = cv_solve_pnp_ransac(
        &object_points,
        &image_points,
        &camera_matrix,
        &dist_coeffs,
        &mut rvec,
        &mut tvec,
        false,
        300,
        0.02,
        0.99,
        &mut inliers,
        SOLVEPNP_EPNP,
    )
    .ok()?;
    if !solved {
        return None;
    }

    let mut rotation = Mat::default();
    rodrigues(&rvec, &mut rotation, &mut Mat::default()).ok()?;
    let rotation = convert_f32(&rotation)?;
    let translation = convert_f32(&tvec)?;
    let rotation = match rotation.typ() {
        opencv::core::CV_32FC1 => {
            let values = rotation
                .reshape(1, 1)
                .ok()?
                .data_typed::<f32>()
                .ok()?
                .to_vec();
            nalgebra::Matrix3::from_row_slice(&values)
        }
        _ => return None,
    };
    let translation = translation
        .reshape(1, 1)
        .ok()?
        .data_typed::<f32>()
        .ok()?
        .to_vec();
    if translation.len() < 3 {
        return None;
    }
    let translation = Vector3::new(translation[0], translation[1], translation[2]);

    let inlier_indices: Vec<usize> = inliers
        .data_typed::<i32>()
        .ok()?
        .iter()
        .filter_map(|&index| usize::try_from(index).ok())
        .collect();
    if inlier_indices.len() < 6 {
        return None;
    }

    let tcw = Isometry3::from_parts(
        Translation3::from(translation),
        UnitQuaternion::from_matrix(&rotation),
    );
    Some(PnpSolution {
        tcw,
        inliers: inlier_indices,
    })
}

fn convert_f32(input: &Mat) -> Option<Mat> {
    let mut output = Mat::default();
    input.convert_to(&mut output, CV_32F, 1.0, 0.0).ok()?;
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pnp_recovers_known_world_to_camera_pose() {
        let expected = Isometry3::from_parts(
            Translation3::new(0.4, -0.2, 0.1),
            UnitQuaternion::from_euler_angles(0.05, -0.03, 0.08),
        );
        let world_points = [
            Point3f::new(-1.0, -0.5, 4.0),
            Point3f::new(1.0, -0.5, 4.0),
            Point3f::new(1.0, 0.5, 4.0),
            Point3f::new(-1.0, 0.5, 4.0),
            Point3f::new(-0.8, -0.4, 5.0),
            Point3f::new(0.8, -0.4, 5.0),
            Point3f::new(0.8, 0.4, 5.0),
            Point3f::new(-0.8, 0.4, 5.0),
        ];
        let correspondences: Vec<_> = world_points
            .iter()
            .map(|&world| {
                let p = expected * nalgebra::Point3::new(world.x, world.y, world.z);
                PnpCorrespondence {
                    world,
                    bearing: Point2f::new(p.x / p.z, p.y / p.z),
                }
            })
            .collect();

        let solution = solve_pnp_ransac(&correspondences).expect("PnP solution");
        let delta = solution.tcw.inverse() * expected;
        assert!(delta.rotation.angle().abs() < 1e-3);
        assert!(delta.translation.vector.norm() < 1e-3);
    }
}
