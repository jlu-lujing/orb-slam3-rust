use nalgebra::{Matrix3x4, Matrix4, Vector3};

// TODO
/*pub fn compute_f12() -> Matrix3<f32> {
    let r1c2 = r1w * r2w.transpose();
    let t1c2 = -r1c2 * t2w + t1w;

    // Skew-symmetric matrix of t1c2
    // `Rotation3::hat` is only available for `f64` in sophus (`IsScalar`); promote, then cast back.
    let omega = VecF64::<3>::from_array([t1c2[0] as f64, t1c2[1] as f64, t1c2[2] as f64]);
    let t1c2x = Rotation3F64::hat(omega).map(|v| v as f32);

    // K is an upper triangular matrix with fx, fy > 0, so it is always invertible for a valid camera.
    let k1_inv = k1.try_inverse().expect("K1 must be invertible");
    let k2_inv = k2.try_inverse().expect("K2 must be invertible");

    k1_inv.transpose() * t1c2x * r1c2 * k2_inv
}*/

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
