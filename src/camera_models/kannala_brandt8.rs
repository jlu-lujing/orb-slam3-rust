use nalgebra::{Isometry3, Matrix2x3, Matrix3, Matrix3x4, Matrix4, Point2, Point3, Vector3};
use opencv::{
    calib3d,
    core::{
        CV_32F, KeyPoint, KeyPointTrait, KeyPointTraitConst, Mat, MatExprTraitConst, Point2f,
        Point3f, TermCriteria, TermCriteria_Type, Vector,
    },
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{
    camera_models::{GeometricCamera, Type, next_geometric_camera_id},
    two_view_reconstruction::{ReconstructResult, TwoViewReconstruction},
};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KannalaBrandt8 {
    parameters: Vec<f32>,
    id: u64,
    camera_type: Type,
    #[serde(skip)]
    tvr: Option<TwoViewReconstruction>,
    precision: f32,
    #[serde(default = "default_lapping_area")]
    pub lapping_area: Vec<usize>,
}

fn default_lapping_area() -> Vec<usize> {
    vec![0, 0]
}

struct TriangulateMatchesResult {
    z1: f32,
    p3d: Vector3<f32>,
}
#[derive(Debug)]
enum TriangulateMatchesError {
    Parallax,    // -1
    Z1,          // -2
    Z2,          // -3
    Projection1, // -4,
    Projection2, // -5,
    Triangulate, // -6 (not in original codebase)
}

fn triangulate(
    p1: Point2f,
    p2: Point2f,
    tcw1: Matrix3x4<f32>,
    tcw2: Matrix3x4<f32>,
) -> Option<Vector3<f32>> {
    let mut a = Matrix4::<f32>::zeros();
    a.row_mut(0).copy_from(&(p1.x * tcw1.row(2) - tcw1.row(0)));
    a.row_mut(1).copy_from(&(p1.y * tcw1.row(2) - tcw1.row(1)));
    a.row_mut(2).copy_from(&(p2.x * tcw2.row(2) - tcw2.row(0)));
    a.row_mut(3).copy_from(&(p2.y * tcw2.row(2) - tcw2.row(1)));

    let svd = a.svd(false, true);
    let v_t = svd.v_t?;
    let x3d_h = v_t.row(3).transpose();

    let w = x3d_h[3];
    if w.abs() < f32::EPSILON {
        return None;
    }

    Some(Vector3::new(x3d_h[0] / w, x3d_h[1] / w, x3d_h[2] / w))
}

fn isometry_to_matrix3x4(tcw: &Isometry3<f32>) -> Matrix3x4<f32> {
    let r = tcw.rotation.to_rotation_matrix().into_inner();
    let t = tcw.translation.vector;

    Matrix3x4::new(
        r[(0, 0)],
        r[(0, 1)],
        r[(0, 2)],
        t[0],
        r[(1, 0)],
        r[(1, 1)],
        r[(1, 2)],
        t[1],
        r[(2, 0)],
        r[(2, 1)],
        r[(2, 2)],
        t[2],
    )
}

impl KannalaBrandt8 {
    fn project_xyz(&self, x: f32, y: f32, z: f32) -> (f32, f32) {
        let x2_plus_y2 = x * x + y * y;
        let theta = x2_plus_y2.sqrt().atan2(z);
        let psi = y.atan2(x);

        let theta2 = theta * theta;
        let theta3 = theta * theta2;
        let theta5 = theta3 * theta2;
        let theta7 = theta5 * theta2;
        let theta9 = theta7 * theta2;

        let r = theta
            + self.parameters[4] * theta3
            + self.parameters[5] * theta5
            + self.parameters[6] * theta7
            + self.parameters[7] * theta9;

        (
            self.parameters[0] * r * psi.cos() + self.parameters[2],
            self.parameters[1] * r * psi.sin() + self.parameters[3],
        )
    }

    fn with_params_and_precision(params: Vec<f32>, precision: f32) -> Self {
        if params.len() != 8 {
            panic!("Invalid GeometricCamera params size");
        }
        KannalaBrandt8 {
            parameters: params,
            id: next_geometric_camera_id(),
            camera_type: Type::Fisheye,
            tvr: None,
            precision: precision,
            lapping_area: vec![2, 0],
        }
    }

    fn triangulate_matches(
        &self,
        camera: &dyn GeometricCamera,
        kp1: &KeyPoint,
        kp2: &KeyPoint,
        r12: &Matrix3<f32>,
        t12: &Vector3<f32>,
        sigma_level: f32,
        unc: f32,
    ) -> Result<TriangulateMatchesResult, TriangulateMatchesError> {
        let r1 = self.unproject_n(&kp1.pt());
        let r2 = camera.unproject_n(&kp2.pt());

        // Check parallax
        let r21 = r12 * r2;
        let cos_parallax_rays = r1.dot(&r21) / (r1.norm() * r21.norm());
        if cos_parallax_rays > 0.9998 {
            return Err(TriangulateMatchesError::Parallax); // -1
        }

        // Parallax is good, so we try to triangulate
        let p11 = Point2f::new(r1[0], r1[1]);
        let p22 = Point2f::new(r2[0], r2[1]);

        let tcw1 =
            Matrix3x4::<f32>::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        let r21 = r12.transpose();
        let t2: Vector3<f32> = -r21 * t12;
        let tcw2: Matrix3x4<f32> = Matrix3x4::new(
            r21[(0, 0)],
            r21[(0, 1)],
            r21[(0, 2)],
            t2[0],
            r21[(1, 0)],
            r21[(1, 1)],
            r21[(1, 2)],
            t2[1],
            r21[(2, 0)],
            r21[(2, 1)],
            r21[(2, 2)],
            t2[2],
        );

        let x3d: Vector3<f32> =
            triangulate(p11, p22, tcw1, tcw2).ok_or(TriangulateMatchesError::Triangulate)?;

        let z1 = x3d[2];
        if z1 <= 0. {
            return Err(TriangulateMatchesError::Z1); // -2
        }
        let x3d_cam2 = r21 * x3d + t2;
        let z2 = x3d_cam2[2];
        if z2 <= 0. {
            return Err(TriangulateMatchesError::Z2); // -3
        }

        // Check reprojection error
        let uv1 = self.project_n(&x3d.into());
        let errx1 = uv1.x - kp1.pt().x;
        let erry1 = uv1.y - kp1.pt().y;
        if (errx1 * errx1 + erry1 * erry1) > 5.991 * sigma_level {
            return Err(TriangulateMatchesError::Projection1); // -4
        }

        let t2 = Vector3::new(tcw2[(0, 3)], tcw2[(1, 3)], tcw2[(2, 3)]);
        let x3d2 = r21 * x3d + t2;
        let uv2 = camera.project_n(&x3d2.into());
        let errx2 = uv2.x - kp2.pt().x;
        let erry2 = uv2.y - kp2.pt().y;
        if (errx2 * errx2 + erry2 * erry2) > 5.991 * unc {
            return Err(TriangulateMatchesError::Projection2); // -5
        }

        Ok(TriangulateMatchesResult { z1, p3d: x3d })
    }
}

impl GeometricCamera for KannalaBrandt8 {
    fn new() -> Self
    where
        Self: Sized,
    {
        KannalaBrandt8 {
            parameters: Vec::with_capacity(8),
            id: next_geometric_camera_id(),
            camera_type: Type::Fisheye,
            tvr: None,
            precision: 1e-6,
            lapping_area: vec![2, 0],
        }
    }
    fn with_params(params: Vec<f32>) -> Self
    where
        Self: Sized,
    {
        if params.len() != 8 {
            panic!("Invalid GeometricCamera params size");
        }
        KannalaBrandt8 {
            parameters: params,
            id: next_geometric_camera_id(),
            camera_type: Type::Fisheye,
            tvr: None,
            precision: 1e-6,
            lapping_area: vec![2, 0],
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn project(&self, p3d: &Point3f) -> Point2f {
        let (u, v) = self.project_xyz(p3d.x, p3d.y, p3d.z);
        Point2f::new(u, v)
    }
    fn project_n(&self, p3d: &Point3<f32>) -> Point2<f32> {
        let (u, v) = self.project_xyz(p3d.x, p3d.y, p3d.z);
        Point2::new(u, v)
    }
    fn project_n_d(&self, p3d: &Point3<f64>) -> Point2<f64> {
        let x2_plus_y2 = p3d.x * p3d.x + p3d.y * p3d.y;
        let theta = x2_plus_y2.sqrt().atan2(p3d.z);
        let psi = p3d.y.atan2(p3d.x);

        let theta2 = theta * theta;
        let theta3 = theta * theta2;
        let theta5 = theta3 * theta2;
        let theta7 = theta5 * theta2;
        let theta9 = theta7 * theta2;

        let r = theta
            + self.parameters[4] as f64 * theta3
            + self.parameters[5] as f64 * theta5
            + self.parameters[6] as f64 * theta7
            + self.parameters[7] as f64 * theta9;

        Point2::new(
            self.parameters[0] as f64 * r * psi.cos() + self.parameters[2] as f64,
            self.parameters[1] as f64 * r * psi.sin() + self.parameters[3] as f64,
        )
    }
    fn project_mat(&self, p3d: &Point3f) -> Point2<f32> {
        let (u, v) = self.project_xyz(p3d.x, p3d.y, p3d.z);
        Point2::new(u, v)
    }

    fn uncertainty(&self, _p2d: &nalgebra::Matrix2x1<f64>) -> f32 {
        return 1.0;
    }

    fn unproject(&self, p2d: &Point2f) -> Point3f {
        // Use Newthon method to solve for theta with good precision (err ~ e-6)
        let pw = Point2f::new(
            (p2d.x - self.parameters[2]) / self.parameters[0],
            (p2d.y - self.parameters[3]) / self.parameters[1],
        );
        let mut scale = 1.0f32;
        let theta_d = (pw.x * pw.x + pw.y * pw.y)
            .sqrt()
            .clamp(-std::f32::consts::PI / 2.0, std::f32::consts::PI / 2.0);

        if theta_d > 1e-8 {
            // Compensate distortion iteratively
            let mut theta = theta_d;

            for _ in 0..10 {
                let theta2 = theta * theta;
                let theta4 = theta2 * theta2;
                let theta6 = theta4 * theta2;
                let theta8 = theta4 * theta4;

                let t2 = self.parameters[4] * theta2;
                let t4 = self.parameters[5] * theta4;
                let t6 = self.parameters[6] * theta6;
                let t8 = self.parameters[7] * theta8;

                let theta_fix = (theta * (1.0 + t2 + t4 + t6 + t8) - theta_d)
                    / (1.0 + 3.0 * t2 + 5.0 * t4 + 7.0 * t6 + 9.0 * t8);

                theta -= theta_fix;

                if theta_fix.abs() < self.precision {
                    break;
                }
            }
            scale = theta.tan() / theta_d;
        }

        Point3f::new(pw.x * scale, pw.y * scale, 1.)
    }
    fn unproject_n(&self, p2d: &Point2f) -> Vector3<f32> {
        let ray = self.unproject(p2d);
        Vector3::new(ray.x, ray.y, ray.z)
    }

    fn project_jac(&self, p3d: &Point3<f64>) -> nalgebra::Matrix2x3<f64> {
        let x = p3d[0];
        let y = p3d[1];
        let z = p3d[2];
        let x2 = x * x;
        let y2 = y * y;
        let z2 = z * z;
        let r2 = x2 + y2;
        let r = r2.sqrt();
        let r3 = r2 * r;
        let theta = r.atan2(z);

        let theta2 = theta * theta;
        let theta3 = theta * theta2;
        let theta4 = theta2 * theta2;
        let theta5 = theta3 * theta2;
        let theta6 = theta2 * theta4;
        let theta7 = theta6 * theta;
        let theta8 = theta7 * theta;
        let theta9 = theta7 * theta2;

        let fx = self.parameters[0] as f64;
        let fy = self.parameters[1] as f64;
        let k0 = self.parameters[4] as f64;
        let k1 = self.parameters[5] as f64;
        let k2 = self.parameters[6] as f64;
        let k3 = self.parameters[7] as f64;

        let f = theta + theta3 * k0 + theta5 * k1 + theta7 * k2 + theta9 * k3;
        let fd =
            1.0 + 3.0 * k0 * theta2 + 5.0 * k1 * theta4 + 7.0 * k2 * theta6 + 9.0 * k3 * theta8;

        let mut jac = Matrix2x3::zeros();

        jac[(0, 0)] = fx * (fd * z * x2 / (r2 * (r2 + z2)) + f * y2 / r3);
        jac[(1, 0)] = fy * (fd * z * y * x / (r2 * (r2 + z2)) - f * y * x / r3);

        jac[(0, 1)] = fx * (fd * z * y * x / (r2 * (r2 + z2)) - f * y * x / r3);
        jac[(1, 1)] = fy * (fd * z * y2 / (r2 * (r2 + z2)) + f * x2 / r3);

        jac[(0, 2)] = -fx * fd * x / (r2 + z2);
        jac[(1, 2)] = -fy * fd * y / (r2 + z2);

        jac
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

        // Correct FishEye distortion
        let mut pts1 = Vector::<Point2f>::new();
        let mut pts2 = Vector::<Point2f>::new();
        for kp in keys1 {
            pts1.push(kp.pt());
        }
        for kp in keys2 {
            pts2.push(kp.pt());
        }

        let d = Mat::from_slice_2d(&[
            &[self.parameters[4]],
            &[self.parameters[5]],
            &[self.parameters[6]],
            &[self.parameters[7]],
        ])
        .ok()?;
        let r = Mat::eye(3, 3, CV_32F).ok()?.to_mat().ok()?;
        let k = self.to_k();

        let mut pts1_ud = Vector::<Point2f>::new();
        let mut pts2_ud = Vector::<Point2f>::new();

        // reasonable default criteria
        let criteria = TermCriteria::new(
            (TermCriteria_Type::COUNT as i32) | (TermCriteria_Type::EPS as i32),
            10,
            1e-8,
        )
        .ok()?;

        calib3d::fisheye_undistort_points(&pts1, &mut pts1_ud, &k, &d, &r, &k, criteria).ok()?;
        calib3d::fisheye_undistort_points(&pts2, &mut pts2_ud, &k, &d, &r, &k, criteria).ok()?;

        let mut keys_un1 = keys1.clone();
        let mut keys_un2 = keys2.clone();
        for (i, kp) in keys_un1.iter_mut().enumerate() {
            kp.set_pt(pts1_ud.get(i).ok()?);
        }
        for (i, kp) in keys_un2.iter_mut().enumerate() {
            kp.set_pt(pts2_ud.get(i).ok()?);
        }

        if let Some(tvr) = &mut self.tvr {
            tvr.reconstruct(&keys_un1, &keys_un2, matches)
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
        sigma_level: f32,
        unc: f32,
    ) -> bool {
        let result = self.triangulate_matches(
            other_camera,
            kp1,
            kp2,
            r12,
            &nalgebra::Vector3::new(t12.x, t12.y, t12.z),
            sigma_level,
            unc,
        );

        if let Ok(result) = result {
            result.z1 > 0.0001
        } else {
            false
        }
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
        kp1: &opencv::core::KeyPoint,
        kp2: &opencv::core::KeyPoint,
        other: &dyn GeometricCamera,
        tcw1: &nalgebra::Isometry3<f32>,
        tcw2: &nalgebra::Isometry3<f32>,
        sigma_level1: f32,
        sigma_level2: f32,
    ) -> Option<Vector3<f32>> {
        let mat_tcw1 = isometry_to_matrix3x4(tcw1);
        let rcw1: Matrix3<f32> = tcw1.rotation.to_rotation_matrix().into_inner();
        let rwc1 = rcw1.transpose();

        let mat_tcw2 = isometry_to_matrix3x4(tcw2);
        let rcw2: Matrix3<f32> = tcw2.rotation.to_rotation_matrix().into_inner();
        let rwc2 = rcw2.transpose();

        let ray1c = self.unproject(&kp1.pt());
        let ray2c = other.unproject(&kp2.pt());

        let r1 = Vector3::new(ray1c.x, ray1c.y, ray1c.z);
        let r2 = Vector3::new(ray2c.x, ray2c.y, ray2c.z);

        // Check parallax between rays
        let ray1 = rwc1 * r1;
        let ray2 = rwc2 * r2;
        let cos_parallax_rays = ray1.dot(&ray2) / (ray1.norm() * ray2.norm());

        // If parallax is lower than 0.9998, reject the match
        if cos_parallax_rays > 0.9998 {
            return None;
        }

        // Parallax is good, so we try to triangulate
        let p11 = Point2f::new(ray1c.x, ray1c.y);
        let p22 = Point2f::new(ray2c.x, ray2c.y);

        let x3d = triangulate(p11, p22, mat_tcw1, mat_tcw2)?;

        // Check triangulation in front of camera
        let z1 = rcw1.row(2).transpose().dot(&x3d) + tcw1.translation.z;
        if z1 <= 0. {
            return None;
        }
        let z2 = rcw2.row(2).transpose().dot(&x3d) + tcw2.translation.z;
        if z2 <= 0. {
            return None;
        }

        // Check projection error in first keyframe
        // Transform point into camera reference system
        let x3d1 = rcw1 * x3d + tcw1.translation.vector;
        let uv1 = self.project_n(&Point3::from(x3d1));
        let errx1 = uv1.x - kp1.pt().x;
        let erry1 = uv1.y - kp1.pt().y;
        if (errx1 * errx1 + erry1 * erry1) > 5.991 * sigma_level1 {
            //Reprojection error is high
            return None;
        }
        // Check reprojection error in second keyframe
        // Transform point into camera reference system
        let x3d2 = rcw2 * x3d + tcw2.translation.vector;
        let uv2 = other.project_n(&Point3::from(x3d2));
        let errx2 = uv2.x - kp2.pt().x;
        let erry2 = uv2.y - kp2.pt().y;
        if (errx2 * errx2 + erry2 * erry2) > 5.991 * sigma_level2 {
            //Reprojection error is high
            return None;
        }

        Some(x3d)
    }

    fn get_id(&self) -> u64 {
        self.id
    }

    fn get_type(&self) -> Type {
        self.camera_type
    }

    fn is_equal(&self, other: &Arc<dyn GeometricCamera>) -> bool {
        let Some(other) = other.as_any().downcast_ref::<KannalaBrandt8>() else {
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
        if (self.precision - other.precision).abs() > 1e-6 {
            is_same = false;
        }
        is_same
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;
    use opencv::prelude::*;

    const FX: f32 = 190.0;
    const FY: f32 = 190.0;
    const CX: f32 = 254.0;
    const CY: f32 = 256.0;
    const K0: f32 = 0.003;
    const K1: f32 = 0.001;
    const K2: f32 = -0.002;
    const K3: f32 = 0.0002;

    const PARAMS: [f32; 8] = [FX, FY, CX, CY, K0, K1, K2, K3];

    fn make_cam() -> KannalaBrandt8 {
        KannalaBrandt8::with_params(PARAMS.to_vec())
    }

    fn make_kp(x: f32, y: f32) -> KeyPoint {
        KeyPoint::new_point_def(Point2f::new(x, y), 1.0).unwrap()
    }

    fn distort_theta(theta: f32) -> f32 {
        let t2 = theta * theta;
        theta
            + K0 * t2 * theta
            + K1 * t2 * t2 * theta
            + K2 * t2 * t2 * t2 * theta
            + K3 * t2 * t2 * t2 * t2 * theta
    }

    fn project_f64(v: &Point3<f64>) -> Point2<f64> {
        let x2y2 = v[0] * v[0] + v[1] * v[1];
        let theta = x2y2.sqrt().atan2(v[2]);
        let psi = v[1].atan2(v[0]);
        let t2 = theta * theta;
        let r = theta
            + K0 as f64 * t2 * theta
            + K1 as f64 * t2 * t2 * theta
            + K2 as f64 * t2 * t2 * t2 * theta
            + K3 as f64 * t2 * t2 * t2 * t2 * theta;
        Point2::new(
            FX as f64 * r * psi.cos() + CX as f64,
            FY as f64 * r * psi.sin() + CY as f64,
        )
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
        let theta = (1.0_f32 + 4.0).sqrt().atan2(5.0);
        let psi = 2.0_f32.atan2(1.0);
        let r = distort_theta(theta);
        let uv = cam.project(&p);
        assert_approx(uv.x, FX * r * psi.cos() + CX, 1e-5);
        assert_approx(uv.y, FY * r * psi.sin() + CY, 1e-5);
    }

    #[test]
    fn project_n_d_point3_f64() {
        let cam = make_cam();
        let p = Point3::new(1.0_f64, 2.0, 5.0);
        let theta = (1.0_f32 + 4.0).sqrt().atan2(5.0);
        let psi = 2.0_f32.atan2(1.0);
        let r = distort_theta(theta);
        let uv = cam.project_n_d(&p);
        assert_approx64(uv[0], (FX * r * psi.cos() + CX) as f64, 1e-4);
        assert_approx64(uv[1], (FY * r * psi.sin() + CY) as f64, 1e-4);
    }

    #[test]
    fn project_n_point3_f32() {
        let cam = make_cam();
        let p = Point3::new(1.0_f32, 2.0, 5.0);
        let theta = (1.0_f32 + 4.0).sqrt().atan2(5.0);
        let psi = 2.0_f32.atan2(1.0);
        let r = distort_theta(theta);
        let uv = cam.project_n(&p);
        assert_approx(uv[0], FX * r * psi.cos() + CX, 1e-5);
        assert_approx(uv[1], FY * r * psi.sin() + CY, 1e-5);
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
        assert_approx(uv.x, CX, 1e-5);
        assert_approx(uv.y, CY, 1e-5);
    }

    #[test]
    fn all_project_overloads_agree() {
        let cam = make_cam();
        let pcv = Point3f::new(2.0, -1.5, 3.0);
        let pd = Point3::new(2.0_f64, -1.5, 3.0);
        let pf = Point3::new(2.0_f32, -1.5, 3.0);

        let uv_cv = cam.project(&pcv);
        let uv_d = cam.project_n_d(&pd);
        let uv_f = cam.project_n(&pf);
        let uv_m = cam.project_mat(&pcv);

        assert_approx64(uv_d[0], uv_cv.x as f64, 1e-4);
        assert_approx64(uv_d[1], uv_cv.y as f64, 1e-4);
        assert_approx(uv_f[0], uv_cv.x, 1e-5);
        assert_approx(uv_f[1], uv_cv.y, 1e-5);
        assert_approx(uv_m[0], uv_cv.x, 1e-5);
        assert_approx(uv_m[1], uv_cv.y, 1e-5);
    }

    #[test]
    fn unproject_inverts_project_small_angle() {
        let cam = make_cam();
        let pw = Point3f::new(0.3, -0.2, 1.0);
        let uv = cam.project(&pw);
        let ray = cam.unproject(&uv);
        let expected_norm = (pw.x * pw.x + pw.y * pw.y).sqrt();
        let actual_norm = (ray.x * ray.x + ray.y * ray.y).sqrt();
        if expected_norm > 1e-6 {
            assert_approx(ray.x / actual_norm, pw.x / expected_norm, 1e-3);
            assert_approx(ray.y / actual_norm, pw.y / expected_norm, 1e-3);
        }
        assert_approx(ray.z, 1.0, 1e-6);
    }

    #[test]
    fn unproject_n_agrees_with_unproject() {
        let cam = make_cam();
        let uv = Point2f::new(200.0, 300.0);
        let ray_cv = cam.unproject(&uv);
        let ray_n = cam.unproject_n(&uv);
        assert_approx(ray_n[0], ray_cv.x, 1e-6);
        assert_approx(ray_n[1], ray_cv.y, 1e-6);
        assert_approx(ray_n[2], ray_cv.z, 1e-6);
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
    fn project_unproject_preserves_bearing_direction() {
        let cam = make_cam();
        for i in 0..50 {
            let x = -0.5 + i as f32 * 0.02;
            let y = -0.3 + i as f32 * 0.015;
            let pw = Point3f::new(x, y, 1.0);
            let uv = cam.project(&pw);
            let ray = cam.unproject(&uv);

            let norm_pw = (pw.x * pw.x + pw.y * pw.y).sqrt();
            let norm_ray = (ray.x * ray.x + ray.y * ray.y).sqrt();
            if norm_pw > 1e-6 {
                assert_approx(ray.x / norm_ray, pw.x / norm_pw, 1e-3);
                assert_approx(ray.y / norm_ray, pw.y / norm_pw, 1e-3);
            }
        }
    }

    #[test]
    fn project_jac_matches_finite_difference() {
        let cam = make_cam();
        let p = Point3::new(1.5_f64, -0.8, 4.0);
        let j = cam.project_jac(&p);
        let eps = 1e-7_f64;
        for col in 0..3 {
            let mut pp = p;
            let mut pm = p;
            pp[col] += eps;
            pm[col] -= eps;
            let dp = (project_f64(&pp) - project_f64(&pm)) / (2.0 * eps);
            assert_approx64(j[(0, col)], dp[0], 1e-4);
            assert_approx64(j[(1, col)], dp[1], 1e-4);
        }
    }

    #[test]
    fn project_jac_at_multiple_points() {
        let cam = make_cam();
        let pts = [
            Point3::new(0.5_f64, 0.5, 3.0),
            Point3::new(-1.0, 2.0, 5.0),
            Point3::new(0.1, -0.1, 1.0),
            Point3::new(2.0, 0.0, 4.0),
        ];
        let eps = 1e-7_f64;
        for p in &pts {
            let j = cam.project_jac(p);
            for col in 0..3 {
                let mut pp = *p;
                let mut pm = *p;
                pp[col] += eps;
                pm[col] -= eps;
                let dp = (project_f64(&pp) - project_f64(&pm)) / (2.0 * eps);
                assert_approx64(j[(0, col)], dp[0], 1e-4);
                assert_approx64(j[(1, col)], dp[1], 1e-4);
            }
        }
    }

    #[test]
    fn triangulate_matches_succeeds_for_correct_stereo_pair() {
        let cam1 = make_cam();
        let cam2 = make_cam();

        let r12 = *Rotation3::from_axis_angle(&Vector3::y_axis(), 10.0_f32.to_radians()).matrix();
        let t12 = Vector3::new(0.5_f32, 0.0, 0.0);
        let r21 = r12.transpose();

        let pw = Vector3::new(0.3_f32, -0.1, 3.0);
        let pc2 = r21 * (pw - t12);

        let uv1 = cam1.project(&Point3f::new(pw[0], pw[1], pw[2]));
        let uv2 = cam2.project(&Point3f::new(pc2[0], pc2[1], pc2[2]));

        let kp1 = make_kp(uv1.x, uv1.y);
        let kp2 = make_kp(uv2.x, uv2.y);

        let result = cam1.triangulate_matches(&cam2, &kp1, &kp2, &r12, &t12, 5.0, 5.0);
        assert!(result.is_ok());
        assert!(result.unwrap().z1 > 0.0);
    }

    #[test]
    fn epipolar_constrain_accepts_correct_match() {
        let cam1 = make_cam();
        let cam2 = make_cam();

        let r12 = *Rotation3::from_axis_angle(&Vector3::y_axis(), 10.0_f32.to_radians()).matrix();
        let t12 = Point3::new(0.5_f32, 0.0, 0.0);
        let t12_v = Vector3::new(0.5_f32, 0.0, 0.0);
        let r21 = r12.transpose();

        let pw = Vector3::new(0.3_f32, -0.1, 3.0);
        let pc2 = r21 * (pw - t12_v);

        let uv1 = cam1.project(&Point3f::new(pw[0], pw[1], pw[2]));
        let uv2 = cam2.project(&Point3f::new(pc2[0], pc2[1], pc2[2]));

        let kp1 = make_kp(uv1.x, uv1.y);
        let kp2 = make_kp(uv2.x, uv2.y);

        assert!(cam1.epipolar_constrain(&cam2, &kp1, &kp2, &r12, &t12, 5.0, 5.0));
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
