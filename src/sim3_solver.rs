//! Closed-form similarity estimation with RANSAC for ORB-SLAM3 loop closing.
//!
//! This mirrors `Sim3Solver.cc`: three camera-space correspondences are used to
//! estimate `T12`, and candidates are accepted by two-way reprojection error in
//! the source and destination cameras.

use std::sync::Arc;

use nalgebra::{Matrix3, Matrix4, SymmetricEigen, UnitQuaternion, Vector3};
use opencv::core::{Point2f, Point3f};
use opencv::prelude::KeyPointTraitConst;
use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::camera_models::GeometricCamera;
use crate::g2o_core::Sim3;
use crate::key_frame::KeyFrame;
use crate::map_point::MapPoint;

/// Result of one or more [`Sim3Solver`] RANSAC iterations.
#[derive(Debug, Clone)]
pub struct Sim3Solution {
    /// Similarity that maps camera-2 coordinates into camera 1.
    pub transform: Sim3,
    /// Inliers indexed by the full keyframe-1 feature array.
    pub inliers: Vec<bool>,
    /// Number of accepted correspondences.
    pub num_inliers: usize,
    /// No additional iteration can improve the result.
    pub no_more: bool,
    /// An estimate exceeded the configured minimum inlier count.
    pub converged: bool,
}

/// RANSAC solver for the 3D-to-3D correspondences used by loop detection.
pub struct Sim3Solver {
    points1_camera: Vec<Vector3<f32>>,
    points2_camera: Vec<Vector3<f32>>,
    indices1: Vec<usize>,
    max_errors1: Vec<f32>,
    max_errors2: Vec<f32>,
    projections1: Vec<Point2f>,
    projections2: Vec<Point2f>,
    camera1: Arc<dyn GeometricCamera>,
    camera2: Arc<dyn GeometricCamera>,
    fix_scale: bool,
    all_indices: Vec<usize>,
    n1: usize,
    iterations: usize,
    ransac_probability: f64,
    min_inliers: usize,
    max_iterations: usize,
    best_inliers: Vec<bool>,
    best_inliers_count: usize,
    best_transform: Option<Sim3>,
    rng: StdRng,
}

impl Sim3Solver {
    /// Build correspondences from matched MapPoints in two keyframes.
    ///
    /// `matched_keyframes` is optionally parallel to `matches12`, as in the
    /// upstream constructor. Missing entries fall back to `kf2`.
    pub fn new(
        kf1: &Arc<KeyFrame>,
        kf2: &Arc<KeyFrame>,
        matches12: &[Option<Arc<MapPoint>>],
        fix_scale: bool,
        matched_keyframes: Option<&[Option<Arc<KeyFrame>>]>,
    ) -> Self {
        let map_points1 = kf1.get_map_point_matches();
        let rcw1 = kf1.get_rotation();
        let tcw1 = kf1.get_translation();
        let rcw2 = kf2.get_rotation();
        let tcw2 = kf2.get_translation();

        let mut points1_camera = Vec::new();
        let mut points2_camera = Vec::new();
        let mut indices1 = Vec::new();
        let mut max_errors1 = Vec::new();
        let mut max_errors2 = Vec::new();

        for (index1, matched) in matches12.iter().enumerate() {
            let Some(point2) = matched.as_ref() else {
                continue;
            };
            let Some(point1) = map_points1.get(index1).and_then(Clone::clone) else {
                continue;
            };
            if point1.is_bad() || point2.is_bad() {
                continue;
            }

            let matched_kf = matched_keyframes
                .and_then(|keyframes| keyframes.get(index1))
                .and_then(Clone::clone)
                .unwrap_or_else(|| kf2.clone());
            let (point_index1, _) = point1.get_index_in_keyframe(kf1);
            let (point_index2, _) = point2.get_index_in_keyframe(&matched_kf);
            if point_index1 < 0 || point_index2 < 0 {
                continue;
            }

            let Some(keypoint1) = kf1.keys_un.get(point_index1 as usize) else {
                continue;
            };
            let Some(keypoint2) = matched_kf.keys_un.get(point_index2 as usize) else {
                continue;
            };
            let Some(sigma1) = kf1.level_sigma2.get(keypoint1.octave() as usize).copied() else {
                continue;
            };
            let Some(sigma2) = matched_kf
                .level_sigma2
                .get(keypoint2.octave() as usize)
                .copied()
            else {
                continue;
            };

            max_errors1.push(9.210 * sigma1);
            max_errors2.push(9.210 * sigma2);
            indices1.push(index1);
            points1_camera.push(rcw1 * point1.get_world_pos() + tcw1);
            points2_camera.push(rcw2 * point2.get_world_pos() + tcw2);
        }

        let projections1 = project_points(&points1_camera, kf1.camera.as_ref());
        let projections2 = project_points(&points2_camera, kf2.camera.as_ref());
        let count = points1_camera.len();
        let mut solver = Self {
            points1_camera,
            points2_camera,
            indices1,
            max_errors1,
            max_errors2,
            projections1,
            projections2,
            camera1: kf1.camera.clone(),
            camera2: kf2.camera.clone(),
            fix_scale,
            all_indices: (0..count).collect(),
            n1: matches12.len(),
            iterations: 0,
            ransac_probability: 0.99,
            min_inliers: 6,
            max_iterations: 300,
            best_inliers: vec![false; matches12.len()],
            best_inliers_count: 0,
            best_transform: None,
            // A fixed seed makes failures reproducible while retaining the
            // upstream three-point sampling behaviour.
            rng: StdRng::seed_from_u64(0),
        };
        solver.set_ransac_parameters(0.99, 6, 300);
        solver
    }

    /// Configure RANSAC confidence, minimum inliers and iteration cap.
    pub fn set_ransac_parameters(
        &mut self,
        probability: f64,
        min_inliers: usize,
        max_iterations: usize,
    ) {
        self.ransac_probability = probability;
        self.min_inliers = min_inliers;
        self.max_iterations = max_iterations;
        self.iterations = 0;

        let n = self.all_indices.len();
        if n == 0 {
            self.max_iterations = 1;
            return;
        }
        let epsilon = min_inliers as f64 / n as f64;
        let required = if min_inliers == n {
            1
        } else if epsilon <= 0.0 || epsilon >= 1.0 {
            max_iterations
        } else {
            let denominator = 1.0 - epsilon.powi(3);
            if denominator <= 0.0 {
                max_iterations
            } else {
                ((1.0 - probability).ln() / denominator.ln()).ceil() as usize
            }
        };
        self.max_iterations = required.clamp(1, max_iterations);
    }

    /// Run at most `iteration_budget` inner RANSAC iterations.
    pub fn iterate(&mut self, iteration_budget: usize) -> Sim3Solution {
        if self.all_indices.len() < self.min_inliers {
            return self.solution(false, true);
        }

        let mut current_iterations = 0;
        while self.iterations < self.max_iterations && current_iterations < iteration_budget {
            current_iterations += 1;
            self.iterations += 1;

            let mut available = self.all_indices.clone();
            let mut sample_indices = [0usize; 3];
            for sample_index in &mut sample_indices {
                let random_index = self.rng.random_range(0..available.len());
                *sample_index = available.swap_remove(random_index);
            }

            let p1 = Matrix3::from_columns(
                &sample_indices.map(|index| self.points1_camera[index].cast::<f64>()),
            );
            let p2 = Matrix3::from_columns(
                &sample_indices.map(|index| self.points2_camera[index].cast::<f64>()),
            );
            let Some(current_transform) = estimate_sim3(&p1, &p2, self.fix_scale) else {
                continue;
            };
            let (inliers, inliers_count) = self.check_inliers(&current_transform);

            if inliers_count >= self.best_inliers_count {
                self.best_inliers.clone_from(&inliers);
                self.best_inliers_count = inliers_count;
                self.best_transform = Some(current_transform);

                // Upstream uses a strict comparison so exactly the minimum
                // number of inliers continues the search.
                if inliers_count > self.min_inliers {
                    return self.solution(true, self.iterations >= self.max_iterations);
                }
            }
        }

        self.solution(false, self.iterations >= self.max_iterations)
    }

    /// Execute all configured RANSAC iterations.
    pub fn find(&mut self) -> Sim3Solution {
        let remaining = self.max_iterations.saturating_sub(self.iterations);
        self.iterate(remaining)
    }

    fn check_inliers(&self, transform: &Sim3) -> (Vec<bool>, usize) {
        let inverse = transform.inverse();
        let projected2_in_1 = project_points_with_sim3(
            &self.points2_camera,
            &sim3_to_f32(transform),
            self.camera1.as_ref(),
        );
        let projected1_in_2 = project_points_with_sim3(
            &self.points1_camera,
            &sim3_to_f32(&inverse),
            self.camera2.as_ref(),
        );

        let mut inliers = vec![false; self.n1];
        let mut count = 0;
        for index in 0..self.points1_camera.len() {
            let dx1 = self.projections1[index].x - projected2_in_1[index].x;
            let dy1 = self.projections1[index].y - projected2_in_1[index].y;
            let error1 = dx1 * dx1 + dy1 * dy1;
            let dx2 = projected1_in_2[index].x - self.projections2[index].x;
            let dy2 = projected1_in_2[index].y - self.projections2[index].y;
            let error2 = dx2 * dx2 + dy2 * dy2;
            if error1 < self.max_errors1[index] && error2 < self.max_errors2[index] {
                inliers[self.indices1[index]] = true;
                count += 1;
            }
        }
        (inliers, count)
    }

    fn solution(&self, converged: bool, no_more: bool) -> Sim3Solution {
        Sim3Solution {
            transform: self.best_transform.unwrap_or_else(Sim3::identity),
            inliers: self.best_inliers.clone(),
            num_inliers: self.best_inliers_count,
            no_more,
            converged,
        }
    }
}

fn estimate_sim3(p1: &Matrix3<f64>, p2: &Matrix3<f64>, fix_scale: bool) -> Option<Sim3> {
    let o1 = p1.column_sum() / 3.0;
    let o2 = p2.column_sum() / 3.0;
    let pr1 = p1 - Matrix3::from_columns(&[o1, o1, o1]);
    let pr2 = p2 - Matrix3::from_columns(&[o2, o2, o2]);
    let m = pr2 * pr1.transpose();

    let n = Matrix4::new(
        m[(0, 0)] + m[(1, 1)] + m[(2, 2)],
        m[(1, 2)] - m[(2, 1)],
        m[(2, 0)] - m[(0, 2)],
        m[(0, 1)] - m[(1, 0)],
        m[(1, 2)] - m[(2, 1)],
        m[(0, 0)] - m[(1, 1)] - m[(2, 2)],
        m[(0, 1)] + m[(1, 0)],
        m[(2, 0)] + m[(0, 2)],
        m[(2, 0)] - m[(0, 2)],
        m[(0, 1)] + m[(1, 0)],
        -m[(0, 0)] + m[(1, 1)] - m[(2, 2)],
        m[(1, 2)] + m[(2, 1)],
        m[(0, 1)] - m[(1, 0)],
        m[(2, 0)] + m[(0, 2)],
        m[(1, 2)] + m[(2, 1)],
        -m[(0, 0)] - m[(1, 1)] + m[(2, 2)],
    );
    let eigen = SymmetricEigen::new(n);
    let (max_index, _) = eigen
        .eigenvalues
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))?;
    let mut quaternion = eigen.eigenvectors.column(max_index).into_owned();
    // Eigenvectors are defined up to sign. Keep the scalar part on the
    // canonical hemisphere before converting the imaginary part to an axis.
    if quaternion[0] < 0.0 {
        quaternion = -quaternion;
    }
    let rotation = UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
        quaternion[0],
        quaternion[1],
        quaternion[2],
        quaternion[3],
    ));

    let rotation_matrix = rotation.to_rotation_matrix().into_inner();
    let p3 = rotation_matrix * pr2;
    let scale = if fix_scale {
        1.0
    } else {
        let numerator: f64 = pr1
            .iter()
            .zip(p3.iter())
            .map(|(left, right)| left * right)
            .sum();
        let denominator: f64 = p3.iter().map(|value| value * value).sum();
        if denominator <= f64::EPSILON {
            return None;
        }
        numerator / denominator
    };
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let translation = o1 - scale * (rotation_matrix * o2);
    Some(Sim3::new(rotation, translation, scale))
}

fn project_points(points_camera: &[Vector3<f32>], camera: &dyn GeometricCamera) -> Vec<Point2f> {
    points_camera
        .iter()
        .map(|point| camera.project(&Point3f::new(point.x, point.y, point.z)))
        .collect()
}

fn project_points_with_sim3(
    points_camera: &[Vector3<f32>],
    transform: &Sim3F32,
    camera: &dyn GeometricCamera,
) -> Vec<Point2f> {
    points_camera
        .iter()
        .map(|point| {
            let transformed = transform.map(point);
            camera.project(&Point3f::new(transformed.x, transformed.y, transformed.z))
        })
        .collect()
}

fn sim3_to_f32(transform: &Sim3) -> Sim3F32 {
    Sim3F32 {
        rotation: transform.rotation().cast::<f32>(),
        translation: transform.translation().cast::<f32>(),
        scale: transform.scale() as f32,
    }
}

struct Sim3F32 {
    rotation: UnitQuaternion<f32>,
    translation: Vector3<f32>,
    scale: f32,
}

impl Sim3F32 {
    fn map(&self, point: &Vector3<f32>) -> Vector3<f32> {
        self.scale * (self.rotation * point) + self.translation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key_frame::KeyFrame;
    use crate::map::Map;
    use crate::test_helpers::{build_frame, keypoint};
    use nalgebra::{Isometry3, Translation3};

    #[test]
    fn closed_form_recovers_known_similarity() {
        let expected_rotation = UnitQuaternion::from_scaled_axis(Vector3::new(0.15, -0.07, 0.11));
        let expected_translation = Vector3::new(0.3, -0.2, 1.4);
        let expected_scale = 1.7;
        let p2 = Matrix3::new(
            -1.0, 1.2, 0.3, //
            -0.5, 0.4, 0.8, //
            4.0, 4.5, 5.2,
        );
        let p1 = expected_scale * (expected_rotation.to_rotation_matrix().into_inner() * p2)
            + Matrix3::from_columns(&[
                expected_translation,
                expected_translation,
                expected_translation,
            ]);

        let estimate = estimate_sim3(&p1, &p2, false).expect("similarity");
        assert!(estimate.rotation().angle_to(&expected_rotation) < 1e-10);
        assert!((estimate.translation() - expected_translation).norm() < 1e-10);
        assert!((estimate.scale() - expected_scale).abs() < 1e-10);
    }

    #[test]
    fn fixed_scale_solver_keeps_metric_scale() {
        let rotation = UnitQuaternion::from_scaled_axis(Vector3::new(-0.04, 0.02, 0.08));
        let translation = Vector3::new(-0.5, 0.1, 0.7);
        let p2 = Matrix3::new(
            -0.8, 0.9, 0.1, //
            -0.2, 0.6, -0.7, //
            3.5, 4.2, 5.1,
        );
        let p1 = rotation.to_rotation_matrix().into_inner() * p2
            + Matrix3::from_columns(&[translation, translation, translation]);

        let estimate = estimate_sim3(&p1, &p2, true).expect("rigid transform");
        assert!(estimate.rotation().angle_to(&rotation) < 1e-10);
        assert!((estimate.translation() - translation).norm() < 1e-10);
        assert_eq!(estimate.scale(), 1.0);
    }

    #[test]
    fn ransac_solver_recovers_keyframe_similarity_and_rejects_outliers() {
        let map = Arc::new(Map::new());
        let pose1 = Isometry3::from_parts(
            Translation3::new(0.1, -0.2, 0.3),
            UnitQuaternion::from_scaled_axis(Vector3::new(0.02, -0.01, 0.04)),
        );
        let pose2 = Isometry3::from_parts(
            Translation3::new(-0.4, 0.2, 0.1),
            UnitQuaternion::from_scaled_axis(Vector3::new(-0.03, 0.05, -0.02)),
        );
        let mut frame1 = build_frame();
        let mut frame2 = build_frame();
        frame1.set_pose(pose1);
        frame2.set_pose(pose2);
        let mut keyframe1 = KeyFrame::from_frame(&frame1, map.clone());
        let mut keyframe2 = KeyFrame::from_frame(&frame2, map.clone());

        let intended_rotation = UnitQuaternion::from_scaled_axis(Vector3::new(0.03, -0.02, 0.06));
        let intended_translation = Vector3::new(0.1, 0.05, -0.08);
        let intended = Sim3::new(intended_rotation, intended_translation, 1.0);
        let points2_camera = [
            Vector3::new(-0.8f32, -0.4, 4.0),
            Vector3::new(0.7, -0.3, 4.2),
            Vector3::new(0.1, 0.6, 5.0),
            Vector3::new(-0.5, 0.5, 4.7),
            Vector3::new(0.9, 0.1, 5.3),
            Vector3::new(-0.2, -0.7, 3.8),
            Vector3::new(0.4, 0.8, 4.9),
            Vector3::new(-0.7, 0.2, 5.1),
            Vector3::new(0.3, -0.8, 4.1),
            Vector3::new(0.6, 0.4, 5.4),
        ];
        let mut matches = vec![None; Arc::get_mut(&mut keyframe1).expect("unique keyframe").n];

        let camera_pairs: Vec<_> = points2_camera
            .into_iter()
            .enumerate()
            .map(|(index, point2_camera)| {
                let point1_camera = if index < 8 {
                    intended.map(&point2_camera.cast::<f64>()).cast::<f32>()
                } else {
                    point2_camera + Vector3::new(20.0, -12.0, 3.0)
                };
                (point1_camera, point2_camera)
            })
            .collect();
        for (index, (point1_camera, point2_camera)) in camera_pairs.iter().enumerate() {
            let projected1 = keyframe1.camera.project(&Point3f::new(
                point1_camera.x,
                point1_camera.y,
                point1_camera.z,
            ));
            let projected2 = keyframe2.camera.project(&Point3f::new(
                point2_camera.x,
                point2_camera.y,
                point2_camera.z,
            ));
            Arc::get_mut(&mut keyframe1)
                .expect("unique keyframe")
                .keys_un[index] = keypoint(projected1.x, projected1.y);
            Arc::get_mut(&mut keyframe2)
                .expect("unique keyframe")
                .keys_un[index] = keypoint(projected2.x, projected2.y);
        }

        for (index, (point1_camera, point2_camera)) in camera_pairs.into_iter().enumerate() {
            let world1 = pose1.inverse() * nalgebra::Point3::from(point1_camera);
            let world2 = pose2.inverse() * nalgebra::Point3::from(point2_camera);
            let point1 = MapPoint::from_pos_ref(&world1.coords, keyframe1.clone(), map.clone());
            let point2 = MapPoint::from_pos_ref(&world2.coords, keyframe2.clone(), map.clone());
            point1.add_observation(&keyframe1, index as i32);
            point2.add_observation(&keyframe2, index as i32);
            keyframe1.add_map_point(point1, index);
            keyframe2.add_map_point(point2.clone(), index);
            matches[index] = Some(point2);
        }

        let mut solver = Sim3Solver::new(&keyframe1, &keyframe2, &matches, true, None);
        let solution = solver.find();
        assert!(solution.converged);
        assert_eq!(solution.num_inliers, 8);
        assert!(solution.transform.rotation().angle_to(&intended_rotation) < 1e-5);
        assert!((solution.transform.translation() - intended_translation).norm() < 1e-5);
        assert!((solution.transform.scale() - 1.0).abs() < 1e-12);
    }
}
