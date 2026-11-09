use crate::camera_models::{GeometricCamera, Type, next_geometric_camera_id};
use crate::two_view_reconstruction::{ReconstructResult, TwoViewReconstruction};

use nalgebra::{Matrix2x3, Matrix3, Point2, Point3, Vector3};
use opencv::core::{KeyPointTraitConst, Mat, Point2f, Point3f};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pinhole {
    parameters: Vec<f32>,
    id: u64,
    camera_type: Type,
    #[serde(skip)]
    tvr: Option<TwoViewReconstruction>,
}

impl GeometricCamera for Pinhole {
    fn new() -> Self
    where
        Self: Sized,
    {
        Pinhole {
            parameters: Vec::with_capacity(4),
            id: next_geometric_camera_id(),
            camera_type: Type::Pinhole,
            tvr: None,
        }
    }
    fn with_params(params: Vec<f32>) -> Self
    where
        Self: Sized,
    {
        if params.len() != 4 {
            panic!("Invalid GeometricCamera params size");
        }
        Pinhole {
            parameters: params,
            id: next_geometric_camera_id(),
            camera_type: Type::Pinhole,
            tvr: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn project(&self, p3d: &Point3f) -> Point2f {
        Point2f::new(
            self.parameters[0] * p3d.x / p3d.z + self.parameters[2],
            self.parameters[1] * p3d.y / p3d.z + self.parameters[3],
        )
    }
    fn project_n(&self, p3d: &Point3<f32>) -> Point2<f32> {
        Point2::new(
            self.parameters[0] * p3d[0] / p3d[2] + self.parameters[2],
            self.parameters[1] * p3d[1] / p3d[2] + self.parameters[3],
        )
    }
    fn project_n_d(&self, p3d: &Point3<f64>) -> Point2<f64> {
        Point2::new(
            self.parameters[0] as f64 * p3d[0] / p3d[2] + self.parameters[2] as f64,
            self.parameters[1] as f64 * p3d[1] / p3d[2] + self.parameters[3] as f64,
        )
    }
    fn project_mat(&self, p3d: &Point3f) -> Point2<f32> {
        let point = self.project(p3d);
        Point2::new(point.x, point.y)
    }

    fn uncertainty(&self, _p2d: &nalgebra::Matrix2x1<f64>) -> f32 {
        return 1.0;
    }

    fn unproject(&self, p2d: &Point2f) -> Point3f {
        Point3f::new(
            (p2d.x - self.parameters[2]) / self.parameters[0],
            (p2d.y - self.parameters[3]) / self.parameters[1],
            1.,
        )
    }
    fn unproject_n(&self, p2d: &Point2f) -> Vector3<f32> {
        Vector3::new(
            (p2d.x - self.parameters[2]) / self.parameters[0],
            (p2d.y - self.parameters[3]) / self.parameters[1],
            1.,
        )
    }

    fn project_jac(&self, p3d: &Point3<f64>) -> nalgebra::Matrix2x3<f64> {
        Matrix2x3::new(
            self.parameters[0] as f64 / p3d[2],
            0.,
            -self.parameters[0] as f64 * p3d[0] / (p3d[2] * p3d[2]),
            0.,
            self.parameters[1] as f64 / p3d[2],
            -self.parameters[1] as f64 * p3d[1] / (p3d[2] * p3d[2]),
        )
    }

    fn reconstruct_with_two_views(
        &mut self,
        keys1: &Vec<opencv::core::KeyPoint>,
        keys2: &Vec<opencv::core::KeyPoint>,
        matches: &Vec<Option<usize>>,
    ) -> Option<ReconstructResult> {
        if self.tvr.is_none() {
            let k = self.to_k_n();
            self.tvr = Some(TwoViewReconstruction::new(k));
        }
        if let Some(tvr) = &mut self.tvr {
            tvr.reconstruct(&keys1, &keys2, matches)
        } else {
            None
        }
    }

    fn to_k(&self) -> Mat {
        let row0 = [self.parameters[0], 0., self.parameters[2]];
        let row1 = [0., self.parameters[1], self.parameters[3]];
        let row2 = [0., 0., 1.];
        Mat::from_slice_2d(&[&row0, &row1, &row2]).expect("Failed to create matrix")
    }
    fn to_k_n(&self) -> nalgebra::Matrix3<f32> {
        Matrix3::new(
            self.parameters[0],
            0.,
            self.parameters[2],
            0.,
            self.parameters[1],
            self.parameters[3],
            0.,
            0.,
            1.,
        )
    }

    fn epipolar_constrain(
        &self,
        other_camera: &dyn GeometricCamera,
        kp1: &opencv::core::KeyPoint,
        kp2: &opencv::core::KeyPoint,
        r12: &Matrix3<f32>,
        t12: &Point3<f32>,
        _sigma_level: f32,
        unc: f32,
    ) -> bool {
        // Compute fundamental Matrix
        let t12x = t12.coords.cross_matrix();
        let k1 = self.to_k_n();
        let k2 = other_camera.to_k_n();
        let f12 = k1.transpose().try_inverse().unwrap() * t12x * r12 * k2.try_inverse().unwrap();

        // Epipolar line in second line
        let a = kp1.pt().x * f12[(0, 0)] + kp1.pt().y * f12[(1, 0)] + f12[(2, 0)];
        let b = kp1.pt().x * f12[(0, 1)] + kp1.pt().y * f12[(1, 1)] + f12[(2, 1)];
        let c = kp1.pt().x * f12[(0, 2)] + kp1.pt().y * f12[(1, 2)] + f12[(2, 2)];

        let num = a * kp2.pt().x + b * kp2.pt().y + c;
        let den = a * a + b * b;
        if den == 0. {
            return false;
        }
        let dsqr = num * num / den;

        dsqr < 3.84 * unc
    }

    fn get_parameter(&self, i: usize) -> f32 {
        *self.parameters.get(i).expect("Unknwown param")
    }
    fn set_parameter(&mut self, p: f32, i: usize) {
        self.parameters[i] = p;
    }

    fn size(&self) -> usize {
        self.parameters.len()
    }

    fn match_and_triangulate(
        &self,
        _kp1: &opencv::core::KeyPoint,
        _kp2: &opencv::core::KeyPoint,
        _other: &dyn GeometricCamera,
        _tcw1: &nalgebra::Isometry3<f32>,
        _tcw2: &nalgebra::Isometry3<f32>,
        _sigma_level1: f32,
        _sigma_level2: f32,
    ) -> Option<Vector3<f32>> {
        None
    }

    fn get_id(&self) -> u64 {
        self.id
    }

    fn get_type(&self) -> Type {
        self.camera_type
    }

    fn is_equal(&self, other: &Arc<dyn GeometricCamera>) -> bool {
        let Some(other) = other.as_any().downcast_ref::<Pinhole>() else {
            return false;
        };
        if self.size() != other.size() {
            return false;
        }
        let mut is_same = true;
        for i in 0..self.size() {
            if (self.parameters[i] - other.parameters[i]).abs() > 1e-6 {
                is_same = false;
                break;
            }
        }
        is_same
    }
}

impl Pinhole {
    fn from_pinhole(other: &Self) -> Self {
        Pinhole {
            parameters: other.parameters.clone(),
            id: next_geometric_camera_id(),
            camera_type: Type::Pinhole,
            tvr: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Rotation3, Vector3};
    use opencv::core::KeyPoint;
    use opencv::prelude::*;

    const FX: f32 = 517.3;
    const FY: f32 = 516.5;
    const CX: f32 = 318.6;
    const CY: f32 = 255.3;

    fn make_cam() -> Pinhole {
        Pinhole::with_params(vec![FX, FY, CX, CY])
    }

    fn make_kp(x: f32, y: f32) -> KeyPoint {
        KeyPoint::new_point_def(Point2f::new(x, y), 1.0).unwrap()
    }

    fn assert_approx(a: f32, b: f32, eps: f32) {
        assert!(
            (a - b).abs() < eps,
            "expected {b}, got {a} (diff={}, eps={eps})",
            (a - b).abs()
        );
    }

    fn assert_approx64(a: f64, b: f64, eps: f64) {
        assert!(
            (a - b).abs() < eps,
            "expected {b}, got {a} (diff={}, eps={eps})",
            (a - b).abs()
        );
    }

    #[test]
    fn to_k_returns_correct_intrinsic_matrix() {
        let cam = make_cam();
        let k = cam.to_k();
        assert_approx(*k.at_2d::<f32>(0, 0).unwrap(), FX, 1e-6);
        assert_approx(*k.at_2d::<f32>(1, 1).unwrap(), FY, 1e-6);
        assert_approx(*k.at_2d::<f32>(0, 2).unwrap(), CX, 1e-6);
        assert_approx(*k.at_2d::<f32>(1, 2).unwrap(), CY, 1e-6);
        assert_approx(*k.at_2d::<f32>(2, 2).unwrap(), 1.0, 1e-6);
        assert_approx(*k.at_2d::<f32>(0, 1).unwrap(), 0.0, 1e-6);
        assert_approx(*k.at_2d::<f32>(1, 0).unwrap(), 0.0, 1e-6);
        assert_approx(*k.at_2d::<f32>(2, 0).unwrap(), 0.0, 1e-6);
        assert_approx(*k.at_2d::<f32>(2, 1).unwrap(), 0.0, 1e-6);
    }

    #[test]
    fn to_k_n_returns_correct_intrinsic_matrix() {
        let cam = make_cam();
        let k = cam.to_k_n();
        assert_approx(k[(0, 0)], FX, 1e-6);
        assert_approx(k[(1, 1)], FY, 1e-6);
        assert_approx(k[(0, 2)], CX, 1e-6);
        assert_approx(k[(1, 2)], CY, 1e-6);
        assert_approx(k[(2, 2)], 1.0, 1e-6);
        assert_approx(k[(0, 1)], 0.0, 1e-6);
        assert_approx(k[(1, 0)], 0.0, 1e-6);
    }

    #[test]
    fn project_point3f() {
        let cam = make_cam();
        let p = Point3f::new(1.0, 2.0, 5.0);
        let uv = cam.project(&p);
        assert_approx(uv.x, FX * 1.0 / 5.0 + CX, 1e-4);
        assert_approx(uv.y, FY * 2.0 / 5.0 + CY, 1e-4);
    }

    #[test]
    fn project_n_d_point3_f64() {
        let cam = make_cam();
        let p = Point3::new(1.0_f64, 2.0, 5.0);
        let uv = cam.project_n_d(&p);
        assert_approx64(uv[0], FX as f64 * 1.0 / 5.0 + CX as f64, 1e-4);
        assert_approx64(uv[1], FY as f64 * 2.0 / 5.0 + CY as f64, 1e-4);
    }

    #[test]
    fn project_n_point3_f32() {
        let cam = make_cam();
        let p = Point3::new(1.0_f32, 2.0, 5.0);
        let uv = cam.project_n(&p);
        assert_approx(uv[0], FX * 1.0 / 5.0 + CX, 1e-4);
        assert_approx(uv[1], FY * 2.0 / 5.0 + CY, 1e-4);
    }

    #[test]
    fn project_mat_matches_project() {
        let cam = make_cam();
        let p = Point3f::new(3.0, -1.0, 4.0);
        let uv_cv = cam.project(&p);
        let uv_n = cam.project_mat(&p);
        assert_approx(uv_n[0], uv_cv.x, 1e-6);
        assert_approx(uv_n[1], uv_cv.y, 1e-6);
    }

    #[test]
    fn project_at_principal_point() {
        let cam = make_cam();
        let p = Point3f::new(0.0, 0.0, 1.0);
        let uv = cam.project(&p);
        assert_approx(uv.x, CX, 1e-6);
        assert_approx(uv.y, CY, 1e-6);
    }

    #[test]
    fn unproject_inverts_project() {
        let cam = make_cam();
        let pw = Point3f::new(0.7, -0.3, 1.0);
        let uv = cam.project(&pw);
        let ray = cam.unproject(&uv);
        assert_approx(ray.x, pw.x, 1e-5);
        assert_approx(ray.y, pw.y, 1e-5);
        assert_approx(ray.z, 1.0, 1e-6);
    }

    #[test]
    fn unproject_n_computes_correct_ray() {
        let cam = make_cam();
        let uv = Point2f::new(200.0, 300.0);
        let ray = cam.unproject_n(&uv);
        assert_approx(ray[0], (200.0 - CX) / FX, 1e-5);
        assert_approx(ray[1], (300.0 - CY) / FY, 1e-5);
        assert_approx(ray[2], 1.0, 1e-6);
    }

    #[test]
    fn unproject_principal_point_yields_zero_bearing() {
        let cam = make_cam();
        let ray = cam.unproject(&Point2f::new(CX, CY));
        assert_approx(ray.x, 0.0, 1e-6);
        assert_approx(ray.y, 0.0, 1e-6);
        assert_approx(ray.z, 1.0, 1e-6);
    }

    #[test]
    fn project_unproject_round_trip() {
        let cam = make_cam();
        for i in 0..50 {
            let x = -2.0 + i as f32 * 0.1;
            let y = -1.0 + i as f32 * 0.05;
            let pw = Point3f::new(x, y, 1.0);
            let uv = cam.project(&pw);
            let ray = cam.unproject(&uv);
            assert_approx(ray.x, pw.x, 1e-4);
            assert_approx(ray.y, pw.y, 1e-4);
        }
    }

    #[test]
    fn project_jac_matches_finite_difference() {
        let cam = make_cam();
        let p = Point3::new(1.5_f64, -0.8, 4.0);
        let j = cam.project_jac(&p);
        let eps = 1e-6_f64;
        for col in 0..3 {
            let mut pp = p;
            let mut pm = p;
            pp[col] += eps;
            pm[col] -= eps;
            let dp = (cam.project_n_d(&pp) - cam.project_n_d(&pm)) / (2.0 * eps);
            assert_approx64(j[(0, col)], dp[0], 1e-4);
            assert_approx64(j[(1, col)], dp[1], 1e-4);
        }
    }

    #[test]
    fn epipolar_constrain_accepts_perfect_match() {
        let cam1 = make_cam();
        let cam2 = make_cam();

        let r12 = *Rotation3::from_axis_angle(&Vector3::y_axis(), 5.0_f32.to_radians()).matrix();
        let t12 = Point3::new(0.3_f32, 0.0, 0.0);
        let k = cam1.to_k_n();

        let pw = Vector3::new(0.5_f32, -0.2, 5.0);
        let pc2 = r12 * pw + t12.coords;

        let mut uv1h = k * pw;
        uv1h /= uv1h[2];
        let mut uv2h = k * pc2;
        uv2h /= uv2h[2];

        let kp1 = make_kp(uv1h[0], uv1h[1]);
        let kp2 = make_kp(uv2h[0], uv2h[1]);

        assert!(cam1.epipolar_constrain(&cam2, &kp1, &kp2, &r12, &t12, 1.0, 1.0));
    }

    #[test]
    fn epipolar_constrain_rejects_far_off_point() {
        let cam1 = make_cam();
        let cam2 = make_cam();

        let r12 = *Rotation3::from_axis_angle(&Vector3::y_axis(), 5.0_f32.to_radians()).matrix();
        let t12 = Point3::new(0.3_f32, 0.0, 0.0);

        let kp1 = make_kp(100.0, 100.0);
        let kp2 = make_kp(500.0, 400.0);

        assert!(!cam1.epipolar_constrain(&cam2, &kp1, &kp2, &r12, &t12, 1.0, 1.0));
    }
}
