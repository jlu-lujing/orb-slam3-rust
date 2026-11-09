//! Conversions between OpenCV `cv::Mat` and `nalgebra` types, mirroring
//! ORB-SLAM3's `Converter` utility class.
//!
//! In the original C++ code, several conversions targeted `g2o::SE3Quat`,
//! `g2o::Sim3`, `Sophus::SE3f`, and `Sophus::Sim3f`. Here we use the closest
//! `nalgebra` equivalents:
//!
//! * `g2o::SE3Quat` and `Sophus::SE3<T>` → [`nalgebra::Isometry3`]
//! * `g2o::Sim3` and `Sophus::Sim3f`     → [`nalgebra::Similarity3`]
//!
//! All `cv::Mat` inputs are expected to be `CV_32F`.

use nalgebra::{Isometry3, Matrix3, Matrix4, Quaternion, Similarity3, UnitQuaternion, Vector3};
use opencv::core::{CV_32F, Mat, Point3f};
use opencv::prelude::*;

// -----------------------------------------------------------------------------
// cv::Mat → nalgebra
// -----------------------------------------------------------------------------

/// Split each row of a descriptor `Mat` into its own `Mat` (row view, cloned).
pub fn to_descriptor_vector(descriptors: &Mat) -> opencv::Result<Vec<Mat>> {
    let rows = descriptors.rows();
    let mut v = Vec::with_capacity(rows as usize);
    for j in 0..rows {
        // `row` returns a header pointing to the same data; clone for ownership.
        v.push(descriptors.row(j)?.try_clone()?);
    }
    Ok(v)
}

/// Convert a 4×4 `CV_32F` `Mat` to an [`Isometry3<f64>`] (SE(3)).
pub fn mat_to_isometry3_f64(cv_t: &Mat) -> opencv::Result<Isometry3<f64>> {
    let r = mat_to_matrix3d(cv_t)?;
    let t = Vector3::new(
        *cv_t.at_2d::<f32>(0, 3)? as f64,
        *cv_t.at_2d::<f32>(1, 3)? as f64,
        *cv_t.at_2d::<f32>(2, 3)? as f64,
    );
    let q = UnitQuaternion::from_matrix(&r);
    Ok(Isometry3::from_parts(t.into(), q))
}

/// Convert a `CV_32F` 3-vector (`Mat`) to a `nalgebra` `Vector3<f64>`.
pub fn mat_to_vector3d(cv_vec: &Mat) -> opencv::Result<Vector3<f64>> {
    Ok(Vector3::new(
        *cv_vec.at::<f32>(0)? as f64,
        *cv_vec.at::<f32>(1)? as f64,
        *cv_vec.at::<f32>(2)? as f64,
    ))
}

/// Convert a `CV_32F` 3-vector (`Mat`) to a `nalgebra` `Vector3<f32>`.
pub fn mat_to_vector3f(cv_vec: &Mat) -> opencv::Result<Vector3<f32>> {
    Ok(Vector3::new(
        *cv_vec.at::<f32>(0)?,
        *cv_vec.at::<f32>(1)?,
        *cv_vec.at::<f32>(2)?,
    ))
}

/// Convert a `cv::Point3f` to a `Vector3<f64>`.
pub fn point3f_to_vector3d(p: &Point3f) -> Vector3<f64> {
    Vector3::new(p.x as f64, p.y as f64, p.z as f64)
}

/// Convert a 3×3 `CV_32F` `Mat` to a `Matrix3<f64>`.
pub fn mat_to_matrix3d(mat: &Mat) -> opencv::Result<Matrix3<f64>> {
    Ok(Matrix3::new(
        *mat.at_2d::<f32>(0, 0)? as f64,
        *mat.at_2d::<f32>(0, 1)? as f64,
        *mat.at_2d::<f32>(0, 2)? as f64,
        *mat.at_2d::<f32>(1, 0)? as f64,
        *mat.at_2d::<f32>(1, 1)? as f64,
        *mat.at_2d::<f32>(1, 2)? as f64,
        *mat.at_2d::<f32>(2, 0)? as f64,
        *mat.at_2d::<f32>(2, 1)? as f64,
        *mat.at_2d::<f32>(2, 2)? as f64,
    ))
}

/// Convert a 3×3 `CV_32F` `Mat` to a `Matrix3<f32>`.
pub fn mat_to_matrix3f(mat: &Mat) -> opencv::Result<Matrix3<f32>> {
    Ok(Matrix3::new(
        *mat.at_2d::<f32>(0, 0)?,
        *mat.at_2d::<f32>(0, 1)?,
        *mat.at_2d::<f32>(0, 2)?,
        *mat.at_2d::<f32>(1, 0)?,
        *mat.at_2d::<f32>(1, 1)?,
        *mat.at_2d::<f32>(1, 2)?,
        *mat.at_2d::<f32>(2, 0)?,
        *mat.at_2d::<f32>(2, 1)?,
        *mat.at_2d::<f32>(2, 2)?,
    ))
}

/// Convert a 4×4 `CV_32F` `Mat` to a `Matrix4<f64>`.
pub fn mat_to_matrix4d(mat: &Mat) -> opencv::Result<Matrix4<f64>> {
    let mut m = Matrix4::<f64>::zeros();
    for i in 0..4 {
        for j in 0..4 {
            m[(i, j)] = *mat.at_2d::<f32>(i as i32, j as i32)? as f64;
        }
    }
    Ok(m)
}

/// Convert a 4×4 `CV_32F` `Mat` to a `Matrix4<f32>`.
pub fn mat_to_matrix4f(mat: &Mat) -> opencv::Result<Matrix4<f32>> {
    let mut m = Matrix4::<f32>::zeros();
    for i in 0..4 {
        for j in 0..4 {
            m[(i, j)] = *mat.at_2d::<f32>(i as i32, j as i32)?;
        }
    }
    Ok(m)
}

/// Convert a 4×4 `CV_32F` `Mat` to an [`Isometry3<f32>`].
pub fn mat_to_isometry3_f32(mat: &Mat) -> opencv::Result<Isometry3<f32>> {
    let r = mat_to_matrix3f(mat)?;
    let t = Vector3::new(
        *mat.at_2d::<f32>(0, 3)?,
        *mat.at_2d::<f32>(1, 3)?,
        *mat.at_2d::<f32>(2, 3)?,
    );
    let q = UnitQuaternion::from_matrix(&r);
    Ok(Isometry3::from_parts(t.into(), q))
}

/// Convert a 3×3 rotation `Mat` (`CV_32F`) to a quaternion `(x, y, z, w)`.
pub fn mat_to_quaternion(m: &Mat) -> opencv::Result<[f32; 4]> {
    let r = mat_to_matrix3d(m)?;
    let q = UnitQuaternion::from_matrix(&r);
    let qq: &Quaternion<f64> = q.as_ref();
    Ok([qq.i as f32, qq.j as f32, qq.k as f32, qq.w as f32])
}

/// Build the 3×3 skew-symmetric matrix from a 3-vector stored as a `CV_32F` `Mat`.
pub fn mat_skew(v: &Mat) -> opencv::Result<Mat> {
    let v0 = *v.at::<f32>(0)?;
    let v1 = *v.at::<f32>(1)?;
    let v2 = *v.at::<f32>(2)?;
    mat_from_2d(&[[0.0, -v2, v1], [v2, 0.0, -v0], [-v1, v0, 0.0]])
}

/// Check whether a 3×3 `CV_32F` `Mat` is a valid rotation matrix.
pub fn is_rotation_matrix(r: &Mat) -> opencv::Result<bool> {
    let m = mat_to_matrix3d(r)?;
    let should_be_id = m.transpose() * m;
    let diff = should_be_id - Matrix3::<f64>::identity();
    Ok(diff.norm() < 1e-6)
}

/// Convert a 3×3 rotation `Mat` to ZYX Euler angles `[x, y, z]` (radians).
pub fn mat_to_euler(r: &Mat) -> opencv::Result<[f32; 3]> {
    debug_assert!(is_rotation_matrix(r)?);
    let r00 = *r.at_2d::<f32>(0, 0)?;
    let r10 = *r.at_2d::<f32>(1, 0)?;
    let r11 = *r.at_2d::<f32>(1, 1)?;
    let r12 = *r.at_2d::<f32>(1, 2)?;
    let r20 = *r.at_2d::<f32>(2, 0)?;
    let r21 = *r.at_2d::<f32>(2, 1)?;
    let r22 = *r.at_2d::<f32>(2, 2)?;

    let sy = (r00 * r00 + r10 * r10).sqrt();
    let singular = sy < 1e-6;

    let (x, y, z) = if !singular {
        (r21.atan2(r22), (-r20).atan2(sy), r10.atan2(r00))
    } else {
        ((-r12).atan2(r11), (-r20).atan2(sy), 0.0)
    };
    Ok([x, y, z])
}

// -----------------------------------------------------------------------------
// nalgebra → cv::Mat
// -----------------------------------------------------------------------------

/// Build a `CV_32F` `Mat` of shape (rows × cols) and fill it with `f`.
fn make_cv_mat_32f(
    rows: i32,
    cols: i32,
    mut f: impl FnMut(i32, i32) -> f32,
) -> opencv::Result<Mat> {
    // SAFETY: filled immediately below.
    let mut m = unsafe { Mat::new_rows_cols(rows, cols, CV_32F) }?;
    for i in 0..rows {
        for j in 0..cols {
            *m.at_2d_mut::<f32>(i, j)? = f(i, j);
        }
    }
    Ok(m)
}

/// Build a small `CV_32F` `Mat` from a 2D slice literal.
fn mat_from_2d<const R: usize, const C: usize>(data: &[[f32; C]; R]) -> opencv::Result<Mat> {
    make_cv_mat_32f(R as i32, C as i32, |i, j| data[i as usize][j as usize])
}

/// Convert an [`Isometry3<f64>`] to a 4×4 `CV_32F` `Mat`.
pub fn isometry3_to_mat_f64(t: &Isometry3<f64>) -> opencv::Result<Mat> {
    matrix4d_to_mat(&t.to_homogeneous())
}

/// Convert a [`Similarity3<f64>`] to a 4×4 `CV_32F` `Mat` (top-left = s·R, last col = t).
pub fn similarity3_to_mat_f64(s: &Similarity3<f64>) -> opencv::Result<Mat> {
    let r = s.isometry.rotation.to_rotation_matrix();
    let scaled = r.matrix() * s.scaling();
    let t = s.isometry.translation.vector;
    matrix3d_and_t_to_se3_mat(&scaled, &t)
}

/// Convert a `Matrix4<f64>` to a 4×4 `CV_32F` `Mat`.
pub fn matrix4d_to_mat(m: &Matrix4<f64>) -> opencv::Result<Mat> {
    make_cv_mat_32f(4, 4, |i, j| m[(i as usize, j as usize)] as f32)
}

/// Convert a `Matrix4<f32>` to a 4×4 `CV_32F` `Mat`.
pub fn matrix4f_to_mat(m: &Matrix4<f32>) -> opencv::Result<Mat> {
    make_cv_mat_32f(4, 4, |i, j| m[(i as usize, j as usize)])
}

/// Convert a 3×4 `SMatrix<f32, 3, 4>` to a 3×4 `CV_32F` `Mat`.
pub fn matrix3x4f_to_mat(m: &nalgebra::SMatrix<f32, 3, 4>) -> opencv::Result<Mat> {
    make_cv_mat_32f(3, 4, |i, j| m[(i as usize, j as usize)])
}

/// Convert a `Matrix3<f64>` to a 3×3 `CV_32F` `Mat`.
pub fn matrix3d_to_mat(m: &Matrix3<f64>) -> opencv::Result<Mat> {
    make_cv_mat_32f(3, 3, |i, j| m[(i as usize, j as usize)] as f32)
}

/// Convert a `Matrix3<f32>` to a 3×3 `CV_32F` `Mat`.
pub fn matrix3f_to_mat(m: &Matrix3<f32>) -> opencv::Result<Mat> {
    make_cv_mat_32f(3, 3, |i, j| m[(i as usize, j as usize)])
}

/// Convert a dynamically-sized `nalgebra` matrix (f32) to a `CV_32F` `Mat`.
pub fn matrixx_f32_to_mat(m: &nalgebra::DMatrix<f32>) -> opencv::Result<Mat> {
    make_cv_mat_32f(m.nrows() as i32, m.ncols() as i32, |i, j| {
        m[(i as usize, j as usize)]
    })
}

/// Convert a dynamically-sized `nalgebra` matrix (f64) to a `CV_32F` `Mat`.
pub fn matrixx_f64_to_mat(m: &nalgebra::DMatrix<f64>) -> opencv::Result<Mat> {
    make_cv_mat_32f(m.nrows() as i32, m.ncols() as i32, |i, j| {
        m[(i as usize, j as usize)] as f32
    })
}

/// Convert a `Vector3<f64>` to a 3×1 `CV_32F` `Mat`.
pub fn vector3d_to_mat(v: &Vector3<f64>) -> opencv::Result<Mat> {
    make_cv_mat_32f(3, 1, |i, _| v[i as usize] as f32)
}

/// Convert a `Vector3<f32>` to a 3×1 `CV_32F` `Mat`.
pub fn vector3f_to_mat(v: &Vector3<f32>) -> opencv::Result<Mat> {
    make_cv_mat_32f(3, 1, |i, _| v[i as usize])
}

/// Build a 4×4 `CV_32F` SE(3) `Mat` from rotation and translation.
pub fn matrix3d_and_t_to_se3_mat(r: &Matrix3<f64>, t: &Vector3<f64>) -> opencv::Result<Mat> {
    make_cv_mat_32f(4, 4, |i, j| {
        if i < 3 && j < 3 {
            r[(i as usize, j as usize)] as f32
        } else if i < 3 && j == 3 {
            t[i as usize] as f32
        } else if i == 3 && j == 3 {
            1.0
        } else {
            0.0
        }
    })
}

// -----------------------------------------------------------------------------
// SE3 / Sim3 between nalgebra representations
// -----------------------------------------------------------------------------

/// Convert an [`Isometry3<f32>`] (analogous to `Sophus::SE3f`) to an
/// [`Isometry3<f64>`] (analogous to `g2o::SE3Quat`).
pub fn isometry3_f32_to_f64(t: &Isometry3<f32>) -> Isometry3<f64> {
    let q = t.rotation;
    let nq = UnitQuaternion::new_normalize(Quaternion::new(
        q.w as f64, q.i as f64, q.j as f64, q.k as f64,
    ));
    let nt = Vector3::new(
        t.translation.x as f64,
        t.translation.y as f64,
        t.translation.z as f64,
    );
    Isometry3::from_parts(nt.into(), nq)
}

/// Convert a [`Similarity3<f64>`] (analogous to `g2o::Sim3`) to a
/// [`Similarity3<f32>`] (analogous to `Sophus::Sim3f`).
pub fn similarity3_f64_to_f32(s: &Similarity3<f64>) -> Similarity3<f32> {
    let q = s.isometry.rotation;
    let nq = UnitQuaternion::new_normalize(Quaternion::new(
        q.w as f32, q.i as f32, q.j as f32, q.k as f32,
    ));
    let nt = Vector3::new(
        s.isometry.translation.x as f32,
        s.isometry.translation.y as f32,
        s.isometry.translation.z as f32,
    );
    Similarity3::from_parts(nt.into(), nq, s.scaling() as f32)
}

// =============================================================================
// Tests — ports of ORB_SLAM3/tests/tests_converter.cpp
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, SMatrix};
    use opencv::core::{CV_8U, Scalar};

    const EPS_F: f32 = 1e-5;

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /// Rotation from ZYX Euler (rx, ry, rz): Rz * Ry * Rx, matching the C++ helper.
    fn make_rot_d(rx: f64, ry: f64, rz: f64) -> Matrix3<f64> {
        let qx = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), rx);
        let qy = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), ry);
        let qz = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), rz);
        (qz * qy * qx).to_rotation_matrix().into_inner()
    }

    fn make_cv_se3(r: &Matrix3<f64>, t: &Vector3<f64>) -> Mat {
        matrix3d_and_t_to_se3_mat(r, t).unwrap()
    }

    fn cv_from_mat3d(r: &Matrix3<f64>) -> Mat {
        matrix3d_to_mat(r).unwrap()
    }

    fn approx_eq_f32(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    // -------------------------------------------------------------------------
    // to_descriptor_vector
    // -------------------------------------------------------------------------

    #[test]
    fn to_descriptor_vector_splits_rows() {
        let mut desc = Mat::new_rows_cols_with_default(5, 32, CV_8U, Scalar::all(0.0)).unwrap();
        for r in 0..5 {
            for c in 0..32 {
                *desc.at_2d_mut::<u8>(r, c).unwrap() = ((r * 32 + c) & 0xFF) as u8;
            }
        }

        let v = to_descriptor_vector(&desc).unwrap();
        assert_eq!(v.len(), 5);
        for r in 0..5 {
            assert_eq!(v[r as usize].rows(), 1);
            assert_eq!(v[r as usize].cols(), 32);
            for c in 0..32 {
                assert_eq!(
                    *v[r as usize].at_2d::<u8>(0, c).unwrap(),
                    *desc.at_2d::<u8>(r, c).unwrap()
                );
            }
        }
    }

    #[test]
    fn to_descriptor_vector_empty_input() {
        let desc = Mat::new_rows_cols_with_default(0, 32, CV_8U, Scalar::all(0.0)).unwrap();
        let v = to_descriptor_vector(&desc).unwrap();
        assert!(v.is_empty());
    }

    // -------------------------------------------------------------------------
    // SE(3) ↔ cv::Mat round-trips
    // -------------------------------------------------------------------------

    #[test]
    fn mat_to_isometry3_f64_round_trip() {
        let r = make_rot_d(0.1, -0.2, 0.3);
        let t = Vector3::new(1.0, 2.0, 3.0);
        let cv_t = make_cv_se3(&r, &t);

        let iso = mat_to_isometry3_f64(&cv_t).unwrap();
        let r_back = iso.rotation.to_rotation_matrix().into_inner();
        let t_back = iso.translation.vector;

        assert!((r_back - r).norm() < 1e-5);
        assert!((t_back - t).norm() < 1e-5);
    }

    #[test]
    fn isometry3_f32_to_f64_matches_input() {
        let r = make_rot_d(0.5, 0.1, -0.4);
        let t = Vector3::new(0.5_f32, -1.5, 2.25);
        let q32 = UnitQuaternion::from_matrix(&r).cast::<f32>();
        let iso_f32 = Isometry3::<f32>::from_parts(t.into(), q32);

        let iso_f64 = isometry3_f32_to_f64(&iso_f32);
        let r_back = iso_f64
            .rotation
            .to_rotation_matrix()
            .into_inner()
            .cast::<f32>();
        let t_back = iso_f64.translation.vector.cast::<f32>();

        assert!((r_back - r.cast::<f32>()).norm() < 1e-5);
        assert!((t_back - t).norm() < 1e-5);
    }

    #[test]
    fn isometry3_to_mat_f64_round_trip() {
        let r = make_rot_d(-0.3, 0.7, 0.2);
        let t = Vector3::new(4.0, -5.0, 6.0);
        let q = UnitQuaternion::from_matrix(&r);
        let iso = Isometry3::from_parts(t.into(), q);

        let cv_t = isometry3_to_mat_f64(&iso).unwrap();
        assert_eq!(cv_t.rows(), 4);
        assert_eq!(cv_t.cols(), 4);
        assert_eq!(cv_t.typ(), CV_32F);

        let h = iso.to_homogeneous();
        for i in 0..4 {
            for j in 0..4 {
                assert!(approx_eq_f32(
                    *cv_t.at_2d::<f32>(i, j).unwrap(),
                    h[(i as usize, j as usize)] as f32,
                    EPS_F,
                ));
            }
        }

        let iso_b = mat_to_isometry3_f64(&cv_t).unwrap();
        assert!((iso_b.rotation.to_rotation_matrix().into_inner() - r).norm() < 1e-5);
        assert!((iso_b.translation.vector - t).norm() < 1e-5);
    }

    #[test]
    fn similarity3_to_mat_embeds_scale_into_rotation() {
        let r = make_rot_d(0.2, 0.3, 0.4);
        let t = Vector3::new(0.5, -0.5, 1.0);
        let s = 2.5;
        let q = UnitQuaternion::from_matrix(&r);
        let sim = Similarity3::from_parts(t.into(), q, s);

        let m = similarity3_to_mat_f64(&sim).unwrap();
        assert_eq!(m.rows(), 4);
        assert_eq!(m.cols(), 4);

        let s_r = r * s;
        for i in 0..3 {
            for j in 0..3 {
                assert!(approx_eq_f32(
                    *m.at_2d::<f32>(i, j).unwrap(),
                    s_r[(i as usize, j as usize)] as f32,
                    EPS_F,
                ));
            }
            assert!(approx_eq_f32(
                *m.at_2d::<f32>(i, 3).unwrap(),
                t[i as usize] as f32,
                EPS_F,
            ));
        }
        assert_eq!(*m.at_2d::<f32>(3, 0).unwrap(), 0.0);
        assert_eq!(*m.at_2d::<f32>(3, 1).unwrap(), 0.0);
        assert_eq!(*m.at_2d::<f32>(3, 2).unwrap(), 0.0);
        assert_eq!(*m.at_2d::<f32>(3, 3).unwrap(), 1.0);
    }

    // -------------------------------------------------------------------------
    // toCvMat overloads
    // -------------------------------------------------------------------------

    #[test]
    fn matrix4d_to_mat_elementwise() {
        let m = Matrix4::<f64>::new(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let cv = matrix4d_to_mat(&m).unwrap();
        assert_eq!(cv.rows(), 4);
        assert_eq!(cv.cols(), 4);
        assert_eq!(cv.typ(), CV_32F);
        for i in 0..4 {
            for j in 0..4 {
                assert!(approx_eq_f32(
                    *cv.at_2d::<f32>(i, j).unwrap(),
                    m[(i as usize, j as usize)] as f32,
                    EPS_F,
                ));
            }
        }
    }

    #[test]
    fn matrix4f_to_mat_elementwise() {
        let m = Matrix4::<f32>::new(
            1.5, 2.0, 3.0, 4.0, 5.0, 6.25, 7.0, 8.0, 9.0, 10.0, 11.125, 12.0, 13.0, 14.0, 15.0,
            16.5,
        );
        let cv = matrix4f_to_mat(&m).unwrap();
        assert_eq!(cv.rows(), 4);
        assert_eq!(cv.cols(), 4);
        for i in 0..4 {
            for j in 0..4 {
                assert!(approx_eq_f32(
                    *cv.at_2d::<f32>(i, j).unwrap(),
                    m[(i as usize, j as usize)],
                    EPS_F,
                ));
            }
        }
    }

    #[test]
    fn matrix3x4f_to_mat_elementwise() {
        let m = SMatrix::<f32, 3, 4>::new(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        );
        let cv = matrix3x4f_to_mat(&m).unwrap();
        assert_eq!(cv.rows(), 3);
        assert_eq!(cv.cols(), 4);
        assert_eq!(cv.typ(), CV_32F);
        for i in 0..3 {
            for j in 0..4 {
                assert!(approx_eq_f32(
                    *cv.at_2d::<f32>(i, j).unwrap(),
                    m[(i as usize, j as usize)],
                    EPS_F,
                ));
            }
        }
    }

    #[test]
    fn matrix3_d_and_f_to_mat() {
        let md = Matrix3::<f64>::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        let cd = matrix3d_to_mat(&md).unwrap();
        assert_eq!(cd.rows(), 3);
        assert_eq!(cd.cols(), 3);
        assert_eq!(cd.typ(), CV_32F);
        for i in 0..3 {
            for j in 0..3 {
                assert!(approx_eq_f32(
                    *cd.at_2d::<f32>(i, j).unwrap(),
                    md[(i as usize, j as usize)] as f32,
                    EPS_F,
                ));
            }
        }

        let mf: Matrix3<f32> = md.cast();
        let cf = matrix3f_to_mat(&mf).unwrap();
        assert_eq!(cf.rows(), 3);
        assert_eq!(cf.cols(), 3);
        for i in 0..3 {
            for j in 0..3 {
                assert!(approx_eq_f32(
                    *cf.at_2d::<f32>(i, j).unwrap(),
                    mf[(i as usize, j as usize)],
                    EPS_F,
                ));
            }
        }
    }

    #[test]
    fn vector3_d_and_f_to_mat() {
        let vd = Vector3::new(1.0, -2.0, 3.5);
        let cd = vector3d_to_mat(&vd).unwrap();
        assert_eq!(cd.rows(), 3);
        assert_eq!(cd.cols(), 1);
        assert_eq!(cd.typ(), CV_32F);
        for i in 0..3 {
            assert!(approx_eq_f32(
                *cd.at::<f32>(i).unwrap(),
                vd[i as usize] as f32,
                EPS_F,
            ));
        }

        let vf = Vector3::new(0.25_f32, 7.0, -8.5);
        let cf = vector3f_to_mat(&vf).unwrap();
        assert_eq!(cf.rows(), 3);
        assert_eq!(cf.cols(), 1);
        for i in 0..3 {
            assert!(approx_eq_f32(
                *cf.at::<f32>(i).unwrap(),
                vf[i as usize],
                EPS_F,
            ));
        }
    }

    #[test]
    fn matrixx_f32_and_f64_to_mat() {
        let mut mf = DMatrix::<f32>::zeros(2, 5);
        for i in 0..2 {
            for j in 0..5 {
                mf[(i, j)] = (i * 10 + j) as f32;
            }
        }
        let cf = matrixx_f32_to_mat(&mf).unwrap();
        assert_eq!(cf.rows(), 2);
        assert_eq!(cf.cols(), 5);
        assert_eq!(cf.typ(), CV_32F);
        for i in 0..2 {
            for j in 0..5 {
                assert!(approx_eq_f32(
                    *cf.at_2d::<f32>(i, j).unwrap(),
                    mf[(i as usize, j as usize)],
                    EPS_F,
                ));
            }
        }

        let md = DMatrix::<f64>::from_row_slice(3, 2, &[0.5, 1.5, 2.5, 3.5, 4.5, 5.5]);
        let cd = matrixx_f64_to_mat(&md).unwrap();
        assert_eq!(cd.rows(), 3);
        assert_eq!(cd.cols(), 2);
        for i in 0..3 {
            for j in 0..2 {
                assert!(approx_eq_f32(
                    *cd.at_2d::<f32>(i, j).unwrap(),
                    md[(i as usize, j as usize)] as f32,
                    EPS_F,
                ));
            }
        }
    }

    // -------------------------------------------------------------------------
    // toCvSE3
    // -------------------------------------------------------------------------

    #[test]
    fn matrix3d_and_t_to_se3_mat_assembles_bottom_row() {
        let r = make_rot_d(0.1, 0.2, 0.3);
        let t = Vector3::new(7.0, 8.0, 9.0);
        let cv_t = matrix3d_and_t_to_se3_mat(&r, &t).unwrap();
        assert_eq!(cv_t.rows(), 4);
        assert_eq!(cv_t.cols(), 4);
        assert_eq!(cv_t.typ(), CV_32F);

        for i in 0..3 {
            for j in 0..3 {
                assert!(approx_eq_f32(
                    *cv_t.at_2d::<f32>(i, j).unwrap(),
                    r[(i as usize, j as usize)] as f32,
                    EPS_F,
                ));
            }
            assert!(approx_eq_f32(
                *cv_t.at_2d::<f32>(i, 3).unwrap(),
                t[i as usize] as f32,
                EPS_F,
            ));
        }
        assert_eq!(*cv_t.at_2d::<f32>(3, 0).unwrap(), 0.0);
        assert_eq!(*cv_t.at_2d::<f32>(3, 1).unwrap(), 0.0);
        assert_eq!(*cv_t.at_2d::<f32>(3, 2).unwrap(), 0.0);
        assert_eq!(*cv_t.at_2d::<f32>(3, 3).unwrap(), 1.0);
    }

    // -------------------------------------------------------------------------
    // mat → vector / matrix readers
    // -------------------------------------------------------------------------

    #[test]
    fn mat_to_vector3d_reads_three_floats() {
        let v = vector3f_to_mat(&Vector3::new(1.0, -2.0, 3.5)).unwrap();
        let ev = mat_to_vector3d(&v).unwrap();
        assert!((ev - Vector3::new(1.0, -2.0, 3.5)).norm() < 1e-6);
    }

    #[test]
    fn mat_to_vector3f_reads_three_floats() {
        let v = vector3f_to_mat(&Vector3::new(4.0, 5.0, 6.0)).unwrap();
        let ev = mat_to_vector3f(&v).unwrap();
        assert!((ev - Vector3::new(4.0_f32, 5.0, 6.0)).norm() < 1e-6);
    }

    #[test]
    fn point3f_to_vector3d_basic() {
        let p = Point3f::new(0.5, -1.5, 9.0);
        let ev = point3f_to_vector3d(&p);
        assert!((ev - Vector3::new(0.5, -1.5, 9.0)).norm() < 1e-12);
    }

    #[test]
    fn mat_to_matrix3d_and_3f() {
        let r = make_rot_d(0.1, 0.2, 0.3);
        let m = cv_from_mat3d(&r);
        let rd = mat_to_matrix3d(&m).unwrap();
        let rf = mat_to_matrix3f(&m).unwrap();
        assert!((rd - r).norm() < 1e-5);
        assert!((rf - r.cast::<f32>()).norm() < 1e-5);
    }

    #[test]
    fn mat_to_matrix4d_and_4f() {
        let h = Matrix4::<f64>::new(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let m = matrix4d_to_mat(&h).unwrap();
        let hd = mat_to_matrix4d(&m).unwrap();
        let hf = mat_to_matrix4f(&m).unwrap();
        assert!((hd - h).norm() < 1e-5);
        assert!((hf - h.cast::<f32>()).norm() < 1e-5);
    }

    // -------------------------------------------------------------------------
    // mat_to_quaternion
    // -------------------------------------------------------------------------

    #[test]
    fn mat_to_quaternion_round_trips_through_rotation() {
        let r = make_rot_d(0.2, -0.3, 0.5);
        let m = cv_from_mat3d(&r);
        let q = mat_to_quaternion(&m).unwrap();

        let n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
        assert!(approx_eq_f32(n2, 1.0, 1e-5));

        // Quaternion sign may differ between libraries; compare matrices instead.
        let uq = UnitQuaternion::from_quaternion(Quaternion::new(
            q[3] as f64,
            q[0] as f64,
            q[1] as f64,
            q[2] as f64,
        ));
        let r_back = uq.to_rotation_matrix().into_inner();
        assert!((r_back - r).norm() < 1e-5);
    }

    #[test]
    fn mat_to_quaternion_of_identity_is_unit_w() {
        let i = cv_from_mat3d(&Matrix3::<f64>::identity());
        let q = mat_to_quaternion(&i).unwrap();
        assert!(approx_eq_f32(q[0], 0.0, 1e-5));
        assert!(approx_eq_f32(q[1], 0.0, 1e-5));
        assert!(approx_eq_f32(q[2], 0.0, 1e-5));
        assert!(approx_eq_f32(q[3].abs(), 1.0, 1e-5));
    }

    // -------------------------------------------------------------------------
    // mat_skew
    // -------------------------------------------------------------------------

    #[test]
    fn mat_skew_is_skew_and_implements_cross_product() {
        let v = vector3f_to_mat(&Vector3::new(1.0, 2.0, 3.0)).unwrap();
        let s = mat_skew(&v).unwrap();
        assert_eq!(s.rows(), 3);
        assert_eq!(s.cols(), 3);

        let s_mat = mat_to_matrix3f(&s).unwrap();
        assert!((s_mat + s_mat.transpose()).norm() < 1e-6);

        let v_nv = mat_to_vector3f(&v).unwrap();
        assert!((s_mat * v_nv).norm() < 1e-6);

        let a = Vector3::new(4.0_f32, -1.0, 2.0);
        let sa = s_mat * a;
        let cross = v_nv.cross(&a);
        assert!((sa - cross).norm() < 1e-6);
    }

    // -------------------------------------------------------------------------
    // is_rotation_matrix
    // -------------------------------------------------------------------------

    #[test]
    fn is_rotation_matrix_accepts_valid_rotations() {
        let i = cv_from_mat3d(&Matrix3::<f64>::identity());
        assert!(is_rotation_matrix(&i).unwrap());

        let r = make_rot_d(0.3, -0.4, 1.1);
        let r_cv = cv_from_mat3d(&r);
        assert!(is_rotation_matrix(&r_cv).unwrap());
    }

    #[test]
    fn is_rotation_matrix_rejects_non_rotations() {
        let m = cv_from_mat3d(&Matrix3::new(2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0));
        assert!(!is_rotation_matrix(&m).unwrap());

        let z = cv_from_mat3d(&Matrix3::<f64>::zeros());
        assert!(!is_rotation_matrix(&z).unwrap());
    }

    // -------------------------------------------------------------------------
    // mat_to_euler
    // -------------------------------------------------------------------------

    #[test]
    fn mat_to_euler_recovers_zyx_angles() {
        let (rx, ry, rz) = (0.2_f64, -0.4, 0.6);
        let r = make_rot_d(rx, ry, rz);
        let r_cv = cv_from_mat3d(&r);
        let e = mat_to_euler(&r_cv).unwrap();
        assert!(approx_eq_f32(e[0], rx as f32, 1e-4));
        assert!(approx_eq_f32(e[1], ry as f32, 1e-4));
        assert!(approx_eq_f32(e[2], rz as f32, 1e-4));
    }

    #[test]
    fn mat_to_euler_identity_is_zero() {
        let i = cv_from_mat3d(&Matrix3::<f64>::identity());
        let e = mat_to_euler(&i).unwrap();
        assert!(approx_eq_f32(e[0], 0.0, 1e-6));
        assert!(approx_eq_f32(e[1], 0.0, 1e-6));
        assert!(approx_eq_f32(e[2], 0.0, 1e-6));
    }

    // -------------------------------------------------------------------------
    // Sophus equivalents
    // -------------------------------------------------------------------------

    #[test]
    fn mat_to_isometry3_f32_from_4x4() {
        let r = make_rot_d(0.1, 0.2, -0.3);
        let t = Vector3::new(1.0, -1.0, 2.0);
        let cv_t = make_cv_se3(&r, &t);

        let iso = mat_to_isometry3_f32(&cv_t).unwrap();
        let r_back = iso.rotation.to_rotation_matrix().into_inner();
        assert!((r_back - r.cast::<f32>()).norm() < 1e-4);
        assert!((iso.translation.vector - t.cast::<f32>()).norm() < 1e-5);
    }

    #[test]
    fn similarity3_f64_to_f32_preserves_components() {
        let r = make_rot_d(0.4, -0.1, 0.25);
        let t = Vector3::new(0.1, 0.2, 0.3);
        let s = 1.75_f64;
        let q = UnitQuaternion::from_matrix(&r);
        let sim_d = Similarity3::from_parts(t.into(), q, s);

        let sim_f = similarity3_f64_to_f32(&sim_d);
        assert!(approx_eq_f32(sim_f.scaling(), s as f32, 1e-5));
        let r_back = sim_f.isometry.rotation.to_rotation_matrix().into_inner();
        assert!((r_back - r.cast::<f32>()).norm() < 1e-4);
        assert!((sim_f.isometry.translation.vector - t.cast::<f32>()).norm() < 1e-5);
    }
}
