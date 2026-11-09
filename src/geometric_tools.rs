use nalgebra::{Matrix3, Matrix3x4, Matrix4, Vector3};

use crate::key_frame::KeyFrame;

/// Computes the fundamental matrix relating camera 1 to camera 2.
pub fn compute_f12(kf1: &KeyFrame, kf2: &KeyFrame) -> Matrix3<f32> {
    let r1c2 = kf1.get_rotation() * kf2.get_rotation().transpose();
    let t1c2 = -r1c2 * kf2.get_camera_center() + kf1.get_camera_center();
    let t1c2_skew = t1c2.cross_matrix();

    kf1.k_matrix
        .try_inverse()
        .expect("valid camera 1 intrinsics")
        .transpose()
        * t1c2_skew
        * r1c2
        * kf2
            .k_matrix
            .try_inverse()
            .expect("valid camera 2 intrinsics")
}

/// Triangulates a 3D point from two 2D points in normalized/pixel coordinates.
/// `x_c1`, `x_c2`: 2D points in camera 1 and 2 (can be pixel or normalized, depending on tc1w/tc2w).
/// `tc1w`, `tc2w`: 3x4 projection matrices for camera 1 and 2.
pub fn triangulate(
    x_c1: &Vector3<f32>,
    x_c2: &Vector3<f32>,
    tc1w: &Matrix3x4<f32>,
    tc2w: &Matrix3x4<f32>,
) -> Option<Vector3<f32>> {
    let mut a = Matrix4::<f32>::zeros();

    a.row_mut(0)
        .copy_from(&(x_c1[0] * tc1w.row(2) - tc1w.row(0)));
    a.row_mut(1)
        .copy_from(&(x_c1[1] * tc1w.row(2) - tc1w.row(1)));
    a.row_mut(2)
        .copy_from(&(x_c2[0] * tc2w.row(2) - tc2w.row(0)));
    a.row_mut(3)
        .copy_from(&(x_c2[1] * tc2w.row(2) - tc2w.row(1)));

    let svd = a.svd(false, true);
    let v_t = svd.v_t.expect("SVD V^T should be computed");

    let x3dh = v_t.row(3);

    if x3dh[3].abs() == 0.0 {
        return None;
    }

    Some(Vector3::new(
        x3dh[0] / x3dh[3],
        x3dh[1] / x3dh[3],
        x3dh[2] / x3dh[3],
    ))
}
