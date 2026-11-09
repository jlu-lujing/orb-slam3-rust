//! Concrete g2o vertices and edges used by `Optimizer.cc`.
//!
//! Ports of `ORB_SLAM3/src/OptimizableTypes.cpp` and the handful of
//! `g2o/types/types_six_dof_expmap` builtins ORB-SLAM3 instantiates directly.
//! Analytic Jacobians match upstream exactly.

use std::any::Any;
use std::sync::Arc;

use nalgebra::{DMatrix, DVector, Matrix2, Matrix3, Matrix3x6, Point3, Vector2, Vector3, Vector6};

use crate::camera_models::GeometricCamera;
use crate::g2o_core::{Edge, EdgeLinearization, SE3Quat, Vertex};

/// Convert a fixed-size nalgebra matrix to a dynamic one.
fn to_dmat<const R: usize, const C: usize>(m: &nalgebra::SMatrix<f64, R, C>) -> DMatrix<f64> {
    DMatrix::from_iterator(R, C, m.iter().copied())
}
fn to_dvec<const R: usize>(v: &nalgebra::SVector<f64, R>) -> DVector<f64> {
    DVector::from_iterator(R, v.iter().copied())
}

/// The `[0 z -y; -z 0 x; y -x 0 | I₃]` derivative block shared by all
/// SE(3) projection Jacobians (`SE3deriv` in OptimizableTypes.cpp).
#[inline]
fn se3_deriv(x: f64, y: f64, z: f64) -> Matrix3x6<f64> {
    Matrix3x6::new(
        0.0, z, -y, 1.0, 0.0, 0.0, //
        -z, 0.0, x, 0.0, 1.0, 0.0, //
        y, -x, 0.0, 0.0, 0.0, 1.0,
    )
}

// ===========================================================================
// VertexSE3Expmap  (g2o::VertexSE3Expmap)
// ===========================================================================

/// Camera pose vertex `Tcw` as an [`SE3Quat`]. `oplus`: `est = exp(δ) · est`.
pub struct VertexSE3Expmap {
    estimate: SE3Quat,
    backup: Vec<SE3Quat>,
    fixed: bool,
    hessian_index: i32,
}

impl VertexSE3Expmap {
    pub fn new(estimate: SE3Quat) -> Self {
        VertexSE3Expmap {
            estimate,
            backup: Vec::new(),
            fixed: false,
            hessian_index: -1,
        }
    }
    pub fn estimate(&self) -> SE3Quat {
        self.estimate
    }
    pub fn set_estimate(&mut self, e: SE3Quat) {
        self.estimate = e;
    }
}

impl Vertex for VertexSE3Expmap {
    fn dim(&self) -> usize {
        6
    }
    fn oplus(&mut self, delta: &[f64]) {
        let update = Vector6::new(delta[0], delta[1], delta[2], delta[3], delta[4], delta[5]);
        self.estimate = SE3Quat::exp(&update).mul(&self.estimate);
    }
    fn push(&mut self) {
        self.backup.push(self.estimate);
    }
    fn pop(&mut self) {
        if let Some(e) = self.backup.pop() {
            self.estimate = e;
        }
    }
    fn discard_top(&mut self) {
        self.backup.pop();
    }
    fn fixed(&self) -> bool {
        self.fixed
    }
    fn set_fixed(&mut self, fixed: bool) {
        self.fixed = fixed;
    }
    fn hessian_index(&self) -> i32 {
        self.hessian_index
    }
    fn set_hessian_index(&mut self, idx: i32) {
        self.hessian_index = idx;
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ===========================================================================
// VertexSBAPointXYZ  (g2o::VertexSBAPointXYZ)
// ===========================================================================

/// 3D point vertex. `oplus`: `est += δ`. Marginalized (landmark) by default.
pub struct VertexSBAPointXYZ {
    estimate: Vector3<f64>,
    backup: Vec<Vector3<f64>>,
    fixed: bool,
    marginalized: bool,
    hessian_index: i32,
}

impl VertexSBAPointXYZ {
    pub fn new(estimate: Vector3<f64>) -> Self {
        VertexSBAPointXYZ {
            estimate,
            backup: Vec::new(),
            fixed: false,
            marginalized: true,
            hessian_index: -1,
        }
    }
    pub fn estimate(&self) -> Vector3<f64> {
        self.estimate
    }
    pub fn set_marginalized(&mut self, m: bool) {
        self.marginalized = m;
    }
}

impl Vertex for VertexSBAPointXYZ {
    fn dim(&self) -> usize {
        3
    }
    fn oplus(&mut self, delta: &[f64]) {
        self.estimate += Vector3::new(delta[0], delta[1], delta[2]);
    }
    fn push(&mut self) {
        self.backup.push(self.estimate);
    }
    fn pop(&mut self) {
        if let Some(e) = self.backup.pop() {
            self.estimate = e;
        }
    }
    fn discard_top(&mut self) {
        self.backup.pop();
    }
    fn fixed(&self) -> bool {
        self.fixed
    }
    fn set_fixed(&mut self, fixed: bool) {
        self.fixed = fixed;
    }
    fn marginalized(&self) -> bool {
        self.marginalized
    }
    fn hessian_index(&self) -> i32 {
        self.hessian_index
    }
    fn set_hessian_index(&mut self, idx: i32) {
        self.hessian_index = idx;
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

fn se3_of(v: &dyn Vertex) -> SE3Quat {
    v.as_any()
        .downcast_ref::<VertexSE3Expmap>()
        .expect("expected VertexSE3Expmap")
        .estimate()
}

fn point_of(v: &dyn Vertex) -> Vector3<f64> {
    v.as_any()
        .downcast_ref::<VertexSBAPointXYZ>()
        .expect("expected VertexSBAPointXYZ")
        .estimate()
}

// ===========================================================================
// EdgeSE3ProjectXYZOnlyPose  (ORB_SLAM3, mono, generic camera)
// ===========================================================================

/// Mono reprojection error against a fixed world point, optimizing pose only.
/// Uses the generic camera model (`pCamera->project` / `projectJac`).
pub struct EdgeSE3ProjectXYZOnlyPose {
    vertices: [usize; 1],
    pub measurement: Vector2<f64>,
    pub information: Matrix2<f64>,
    robust_delta: Option<f64>,
    level: i32,
    error: Vector2<f64>,
    pub xw: Vector3<f64>,
    pub camera: Arc<dyn GeometricCamera>,
}

impl EdgeSE3ProjectXYZOnlyPose {
    pub fn new(vertex: usize, xw: Vector3<f64>, camera: Arc<dyn GeometricCamera>) -> Self {
        EdgeSE3ProjectXYZOnlyPose {
            vertices: [vertex],
            measurement: Vector2::zeros(),
            information: Matrix2::identity(),
            robust_delta: None,
            level: 0,
            error: Vector2::zeros(),
            xw,
            camera,
        }
    }
    pub fn set_measurement(&mut self, m: Vector2<f64>) {
        self.measurement = m;
    }
    pub fn set_information(&mut self, info: Matrix2<f64>) {
        self.information = info;
    }
}

impl Edge for EdgeSE3ProjectXYZOnlyPose {
    fn vertices(&self) -> &[usize] {
        &self.vertices
    }
    fn dim(&self) -> usize {
        2
    }
    fn level(&self) -> i32 {
        self.level
    }
    fn set_level(&mut self, level: i32) {
        self.level = level;
    }
    fn robust_kernel(&self) -> Option<f64> {
        self.robust_delta
    }
    fn set_robust_kernel(&mut self, delta: Option<f64>) {
        self.robust_delta = delta;
    }
    fn compute_error(&mut self, vertices: &[&dyn Vertex]) {
        let t = se3_of(vertices[0]);
        let xyz = t.map(&self.xw);
        let proj = self.camera.project_n_d(&Point3::from(xyz));
        self.error = self.measurement - Vector2::new(proj.x, proj.y);
    }
    fn chi2(&self) -> f64 {
        (self.error.transpose() * self.information * self.error)[(0, 0)]
    }
    fn linearize(&mut self, vertices: &[&dyn Vertex]) -> EdgeLinearization {
        self.compute_error(vertices);
        let t = se3_of(vertices[0]);
        let xyz = t.map(&self.xw);
        let proj_jac = self.camera.project_jac(&Point3::from(xyz)); // 2×3
        let jac = -proj_jac * se3_deriv(xyz[0], xyz[1], xyz[2]); // 2×6
        EdgeLinearization {
            error: to_dvec(&self.error),
            information: to_dmat(&self.information),
            jacobians: vec![to_dmat(&jac)],
        }
    }
}

// ===========================================================================
// EdgeSE3ProjectXYZ  (ORB_SLAM3, mono binary: point + pose)
// ===========================================================================

/// Mono reprojection error optimizing both the 3D point (`vertex 0`) and the
/// camera pose (`vertex 1`). Generic camera model.
pub struct EdgeSE3ProjectXYZ {
    vertices: [usize; 2],
    pub measurement: Vector2<f64>,
    pub information: Matrix2<f64>,
    robust_delta: Option<f64>,
    level: i32,
    error: Vector2<f64>,
    pub camera: Arc<dyn GeometricCamera>,
}

impl EdgeSE3ProjectXYZ {
    /// `point` and `pose` are vertex slot indices.
    pub fn new(point: usize, pose: usize, camera: Arc<dyn GeometricCamera>) -> Self {
        EdgeSE3ProjectXYZ {
            vertices: [point, pose],
            measurement: Vector2::zeros(),
            information: Matrix2::identity(),
            robust_delta: None,
            level: 0,
            error: Vector2::zeros(),
            camera,
        }
    }
    pub fn set_measurement(&mut self, m: Vector2<f64>) {
        self.measurement = m;
    }
    pub fn set_information(&mut self, info: Matrix2<f64>) {
        self.information = info;
    }
}

impl Edge for EdgeSE3ProjectXYZ {
    fn vertices(&self) -> &[usize] {
        &self.vertices
    }
    fn dim(&self) -> usize {
        2
    }
    fn level(&self) -> i32 {
        self.level
    }
    fn set_level(&mut self, level: i32) {
        self.level = level;
    }
    fn robust_kernel(&self) -> Option<f64> {
        self.robust_delta
    }
    fn set_robust_kernel(&mut self, delta: Option<f64>) {
        self.robust_delta = delta;
    }
    fn compute_error(&mut self, vertices: &[&dyn Vertex]) {
        let p = point_of(vertices[0]);
        let t = se3_of(vertices[1]);
        let proj = self.camera.project_n_d(&Point3::from(t.map(&p)));
        self.error = self.measurement - Vector2::new(proj.x, proj.y);
    }
    fn chi2(&self) -> f64 {
        (self.error.transpose() * self.information * self.error)[(0, 0)]
    }
    fn linearize(&mut self, vertices: &[&dyn Vertex]) -> EdgeLinearization {
        self.compute_error(vertices);
        let p = point_of(vertices[0]);
        let t = se3_of(vertices[1]);
        let xyz = t.map(&p);
        let proj_jac = -self.camera.project_jac(&Point3::from(xyz)); // -(2×3)
        let ji = proj_jac * t.rotation_matrix(); // 2×3 (point)
        let jj = proj_jac * se3_deriv(xyz[0], xyz[1], xyz[2]); // 2×6 (pose)
        EdgeLinearization {
            error: to_dvec(&self.error),
            information: to_dmat(&self.information),
            jacobians: vec![to_dmat(&ji), to_dmat(&jj)],
        }
    }
    fn depth_positive(&self, vertices: &[&dyn Vertex]) -> bool {
        let t = se3_of(vertices[1]);
        let p = point_of(vertices[0]);
        t.map(&p)[2] > 0.0
    }
}

// ===========================================================================
// EdgeSE3ProjectXYZToBody  (ORB_SLAM3, mono binary, rig right camera)
// ===========================================================================

/// Like [`EdgeSE3ProjectXYZ`] but through the body→right-camera transform `mTrl`.
pub struct EdgeSE3ProjectXYZToBody {
    vertices: [usize; 2],
    pub measurement: Vector2<f64>,
    pub information: Matrix2<f64>,
    robust_delta: Option<f64>,
    level: i32,
    error: Vector2<f64>,
    pub camera: Arc<dyn GeometricCamera>,
    pub m_trl: SE3Quat,
}

impl EdgeSE3ProjectXYZToBody {
    pub fn new(
        point: usize,
        pose: usize,
        camera: Arc<dyn GeometricCamera>,
        m_trl: SE3Quat,
    ) -> Self {
        EdgeSE3ProjectXYZToBody {
            vertices: [point, pose],
            measurement: Vector2::zeros(),
            information: Matrix2::identity(),
            robust_delta: None,
            level: 0,
            error: Vector2::zeros(),
            camera,
            m_trl,
        }
    }
    pub fn set_measurement(&mut self, m: Vector2<f64>) {
        self.measurement = m;
    }
    pub fn set_information(&mut self, info: Matrix2<f64>) {
        self.information = info;
    }
}

impl Edge for EdgeSE3ProjectXYZToBody {
    fn vertices(&self) -> &[usize] {
        &self.vertices
    }
    fn dim(&self) -> usize {
        2
    }
    fn level(&self) -> i32 {
        self.level
    }
    fn set_level(&mut self, level: i32) {
        self.level = level;
    }
    fn robust_kernel(&self) -> Option<f64> {
        self.robust_delta
    }
    fn set_robust_kernel(&mut self, delta: Option<f64>) {
        self.robust_delta = delta;
    }
    fn compute_error(&mut self, vertices: &[&dyn Vertex]) {
        let p = point_of(vertices[0]);
        let t_lw = se3_of(vertices[1]);
        let x_r = self.m_trl.map(&t_lw.map(&p));
        let proj = self.camera.project_n_d(&Point3::from(x_r));
        self.error = self.measurement - Vector2::new(proj.x, proj.y);
    }
    fn chi2(&self) -> f64 {
        (self.error.transpose() * self.information * self.error)[(0, 0)]
    }
    fn linearize(&mut self, vertices: &[&dyn Vertex]) -> EdgeLinearization {
        self.compute_error(vertices);
        let p = point_of(vertices[0]);
        let t_lw = se3_of(vertices[1]);
        let t_rw = self.m_trl.mul(&t_lw);
        let x_l = t_lw.map(&p);
        let x_r = self.m_trl.map(&x_l);
        let proj_jac = -self.camera.project_jac(&Point3::from(x_r)); // -(2×3)
        let ji = proj_jac * t_rw.rotation_matrix(); // 2×3 (point)
        let jj = proj_jac * self.m_trl.rotation_matrix() * se3_deriv(x_l[0], x_l[1], x_l[2]); // 2×6
        EdgeLinearization {
            error: to_dvec(&self.error),
            information: to_dmat(&self.information),
            jacobians: vec![to_dmat(&ji), to_dmat(&jj)],
        }
    }
    fn depth_positive(&self, vertices: &[&dyn Vertex]) -> bool {
        let p = point_of(vertices[0]);
        let t_lw = se3_of(vertices[1]);
        self.m_trl.map(&t_lw.map(&p))[2] > 0.0
    }
}

// ===========================================================================
// EdgeStereoSE3ProjectXYZ  (g2o builtin, stereo binary: point + pose)
// ===========================================================================

/// Stereo reprojection error optimizing both 3D point (`vertex 0`) and camera
/// pose (`vertex 1`). Explicit intrinsics.
pub struct EdgeStereoSE3ProjectXYZ {
    vertices: [usize; 2],
    pub measurement: Vector3<f64>,
    pub information: Matrix3<f64>,
    robust_delta: Option<f64>,
    level: i32,
    error: Vector3<f64>,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub bf: f64,
}

impl EdgeStereoSE3ProjectXYZ {
    #[allow(clippy::too_many_arguments)]
    pub fn new(point: usize, pose: usize, fx: f64, fy: f64, cx: f64, cy: f64, bf: f64) -> Self {
        EdgeStereoSE3ProjectXYZ {
            vertices: [point, pose],
            measurement: Vector3::zeros(),
            information: Matrix3::identity(),
            robust_delta: None,
            level: 0,
            error: Vector3::zeros(),
            fx,
            fy,
            cx,
            cy,
            bf,
        }
    }
    pub fn set_measurement(&mut self, m: Vector3<f64>) {
        self.measurement = m;
    }
    pub fn set_information(&mut self, info: Matrix3<f64>) {
        self.information = info;
    }
    fn cam_project(&self, xyz: &Vector3<f64>) -> Vector3<f64> {
        let invz = 1.0 / xyz[2];
        let u = xyz[0] * invz * self.fx + self.cx;
        let v = xyz[1] * invz * self.fy + self.cy;
        Vector3::new(u, v, u - self.bf * invz)
    }
}

impl Edge for EdgeStereoSE3ProjectXYZ {
    fn vertices(&self) -> &[usize] {
        &self.vertices
    }
    fn dim(&self) -> usize {
        3
    }
    fn level(&self) -> i32 {
        self.level
    }
    fn set_level(&mut self, level: i32) {
        self.level = level;
    }
    fn robust_kernel(&self) -> Option<f64> {
        self.robust_delta
    }
    fn set_robust_kernel(&mut self, delta: Option<f64>) {
        self.robust_delta = delta;
    }
    fn compute_error(&mut self, vertices: &[&dyn Vertex]) {
        let p = point_of(vertices[0]);
        let t = se3_of(vertices[1]);
        self.error = self.measurement - self.cam_project(&t.map(&p));
    }
    fn chi2(&self) -> f64 {
        (self.error.transpose() * self.information * self.error)[(0, 0)]
    }
    fn linearize(&mut self, vertices: &[&dyn Vertex]) -> EdgeLinearization {
        self.compute_error(vertices);
        let p = point_of(vertices[0]);
        let t = se3_of(vertices[1]);
        let r = t.rotation_matrix();
        let xyz = t.map(&p);
        let (x, y, z) = (xyz[0], xyz[1], xyz[2]);
        let z_2 = z * z;
        let (fx, fy, bf) = (self.fx, self.fy, self.bf);

        // Jacobian w.r.t. point (3×3).
        let mut ji = Matrix3::<f64>::zeros();
        ji[(0, 0)] = -fx * r[(0, 0)] / z + fx * x * r[(2, 0)] / z_2;
        ji[(0, 1)] = -fx * r[(0, 1)] / z + fx * x * r[(2, 1)] / z_2;
        ji[(0, 2)] = -fx * r[(0, 2)] / z + fx * x * r[(2, 2)] / z_2;
        ji[(1, 0)] = -fy * r[(1, 0)] / z + fy * y * r[(2, 0)] / z_2;
        ji[(1, 1)] = -fy * r[(1, 1)] / z + fy * y * r[(2, 1)] / z_2;
        ji[(1, 2)] = -fy * r[(1, 2)] / z + fy * y * r[(2, 2)] / z_2;
        ji[(2, 0)] = ji[(0, 0)] - bf * r[(2, 0)] / z_2;
        ji[(2, 1)] = ji[(0, 1)] - bf * r[(2, 1)] / z_2;
        ji[(2, 2)] = ji[(0, 2)] - bf * r[(2, 2)] / z_2;

        // Jacobian w.r.t. pose (3×6).
        let mut jj = nalgebra::SMatrix::<f64, 3, 6>::zeros();
        jj[(0, 0)] = x * y / z_2 * fx;
        jj[(0, 1)] = -(1.0 + (x * x / z_2)) * fx;
        jj[(0, 2)] = y / z * fx;
        jj[(0, 3)] = -1.0 / z * fx;
        jj[(0, 4)] = 0.0;
        jj[(0, 5)] = x / z_2 * fx;
        jj[(1, 0)] = (1.0 + y * y / z_2) * fy;
        jj[(1, 1)] = -x * y / z_2 * fy;
        jj[(1, 2)] = -x / z * fy;
        jj[(1, 3)] = 0.0;
        jj[(1, 4)] = -1.0 / z * fy;
        jj[(1, 5)] = y / z_2 * fy;
        jj[(2, 0)] = jj[(0, 0)] - bf * y / z_2;
        jj[(2, 1)] = jj[(0, 1)] + bf * x / z_2;
        jj[(2, 2)] = jj[(0, 2)];
        jj[(2, 3)] = jj[(0, 3)];
        jj[(2, 4)] = 0.0;
        jj[(2, 5)] = jj[(0, 5)] - bf / z_2;

        EdgeLinearization {
            error: to_dvec(&self.error),
            information: to_dmat(&self.information),
            jacobians: vec![to_dmat(&ji), to_dmat(&jj)],
        }
    }
    fn depth_positive(&self, vertices: &[&dyn Vertex]) -> bool {
        let t = se3_of(vertices[1]);
        let p = point_of(vertices[0]);
        t.map(&p)[2] > 0.0
    }
}

// ===========================================================================
// EdgeSE3ProjectXYZOnlyPoseToBody  (ORB_SLAM3, mono, rig right camera)
// ===========================================================================

/// Like [`EdgeSE3ProjectXYZOnlyPose`] but projects through a body→right-camera
/// transform `mTrl` (fisheye stereo rig).
pub struct EdgeSE3ProjectXYZOnlyPoseToBody {
    vertices: [usize; 1],
    pub measurement: Vector2<f64>,
    pub information: Matrix2<f64>,
    robust_delta: Option<f64>,
    level: i32,
    error: Vector2<f64>,
    pub xw: Vector3<f64>,
    pub camera: Arc<dyn GeometricCamera>,
    pub m_trl: SE3Quat,
}

impl EdgeSE3ProjectXYZOnlyPoseToBody {
    pub fn new(
        vertex: usize,
        xw: Vector3<f64>,
        camera: Arc<dyn GeometricCamera>,
        m_trl: SE3Quat,
    ) -> Self {
        EdgeSE3ProjectXYZOnlyPoseToBody {
            vertices: [vertex],
            measurement: Vector2::zeros(),
            information: Matrix2::identity(),
            robust_delta: None,
            level: 0,
            error: Vector2::zeros(),
            xw,
            camera,
            m_trl,
        }
    }
    pub fn set_measurement(&mut self, m: Vector2<f64>) {
        self.measurement = m;
    }
    pub fn set_information(&mut self, info: Matrix2<f64>) {
        self.information = info;
    }
}

impl Edge for EdgeSE3ProjectXYZOnlyPoseToBody {
    fn vertices(&self) -> &[usize] {
        &self.vertices
    }
    fn dim(&self) -> usize {
        2
    }
    fn level(&self) -> i32 {
        self.level
    }
    fn set_level(&mut self, level: i32) {
        self.level = level;
    }
    fn robust_kernel(&self) -> Option<f64> {
        self.robust_delta
    }
    fn set_robust_kernel(&mut self, delta: Option<f64>) {
        self.robust_delta = delta;
    }
    fn compute_error(&mut self, vertices: &[&dyn Vertex]) {
        let t_lw = se3_of(vertices[0]);
        let x_r = self.m_trl.map(&t_lw.map(&self.xw));
        let proj = self.camera.project_n_d(&Point3::from(x_r));
        self.error = self.measurement - Vector2::new(proj.x, proj.y);
    }
    fn chi2(&self) -> f64 {
        (self.error.transpose() * self.information * self.error)[(0, 0)]
    }
    fn linearize(&mut self, vertices: &[&dyn Vertex]) -> EdgeLinearization {
        self.compute_error(vertices);
        let t_lw = se3_of(vertices[0]);
        let x_l = t_lw.map(&self.xw);
        let x_r = self.m_trl.map(&x_l);
        let proj_jac = self.camera.project_jac(&Point3::from(x_r)); // 2×3
        let jac = -proj_jac * self.m_trl.rotation_matrix() * se3_deriv(x_l[0], x_l[1], x_l[2]); // 2×6
        EdgeLinearization {
            error: to_dvec(&self.error),
            information: to_dmat(&self.information),
            jacobians: vec![to_dmat(&jac)],
        }
    }
}

// ===========================================================================
// EdgeStereoSE3ProjectXYZOnlyPose  (g2o builtin, stereo, explicit intrinsics)
// ===========================================================================

/// Stereo reprojection error (u, v, u_right) against a fixed world point,
/// optimizing pose only. Uses explicit `fx, fy, cx, cy, bf`.
pub struct EdgeStereoSE3ProjectXYZOnlyPose {
    vertices: [usize; 1],
    pub measurement: Vector3<f64>,
    pub information: Matrix3<f64>,
    robust_delta: Option<f64>,
    level: i32,
    error: Vector3<f64>,
    pub xw: Vector3<f64>,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub bf: f64,
}

impl EdgeStereoSE3ProjectXYZOnlyPose {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vertex: usize,
        xw: Vector3<f64>,
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        bf: f64,
    ) -> Self {
        EdgeStereoSE3ProjectXYZOnlyPose {
            vertices: [vertex],
            measurement: Vector3::zeros(),
            information: Matrix3::identity(),
            robust_delta: None,
            level: 0,
            error: Vector3::zeros(),
            xw,
            fx,
            fy,
            cx,
            cy,
            bf,
        }
    }
    pub fn set_measurement(&mut self, m: Vector3<f64>) {
        self.measurement = m;
    }
    pub fn set_information(&mut self, info: Matrix3<f64>) {
        self.information = info;
    }
    fn cam_project(&self, xyz: &Vector3<f64>) -> Vector3<f64> {
        let invz = 1.0 / xyz[2];
        let u = xyz[0] * invz * self.fx + self.cx;
        let v = xyz[1] * invz * self.fy + self.cy;
        Vector3::new(u, v, u - self.bf * invz)
    }
}

impl Edge for EdgeStereoSE3ProjectXYZOnlyPose {
    fn vertices(&self) -> &[usize] {
        &self.vertices
    }
    fn dim(&self) -> usize {
        3
    }
    fn level(&self) -> i32 {
        self.level
    }
    fn set_level(&mut self, level: i32) {
        self.level = level;
    }
    fn robust_kernel(&self) -> Option<f64> {
        self.robust_delta
    }
    fn set_robust_kernel(&mut self, delta: Option<f64>) {
        self.robust_delta = delta;
    }
    fn compute_error(&mut self, vertices: &[&dyn Vertex]) {
        let t = se3_of(vertices[0]);
        let xyz = t.map(&self.xw);
        self.error = self.measurement - self.cam_project(&xyz);
    }
    fn chi2(&self) -> f64 {
        (self.error.transpose() * self.information * self.error)[(0, 0)]
    }
    fn linearize(&mut self, vertices: &[&dyn Vertex]) -> EdgeLinearization {
        self.compute_error(vertices);
        let t = se3_of(vertices[0]);
        let xyz = t.map(&self.xw);
        let (x, y) = (xyz[0], xyz[1]);
        let invz = 1.0 / xyz[2];
        let invz_2 = invz * invz;
        let (fx, fy, bf) = (self.fx, self.fy, self.bf);

        let mut j = nalgebra::SMatrix::<f64, 3, 6>::zeros();
        j[(0, 0)] = x * y * invz_2 * fx;
        j[(0, 1)] = -(1.0 + (x * x * invz_2)) * fx;
        j[(0, 2)] = y * invz * fx;
        j[(0, 3)] = -invz * fx;
        j[(0, 4)] = 0.0;
        j[(0, 5)] = x * invz_2 * fx;

        j[(1, 0)] = (1.0 + y * y * invz_2) * fy;
        j[(1, 1)] = -x * y * invz_2 * fy;
        j[(1, 2)] = -x * invz * fy;
        j[(1, 3)] = 0.0;
        j[(1, 4)] = -invz * fy;
        j[(1, 5)] = y * invz_2 * fy;

        j[(2, 0)] = j[(0, 0)] - bf * y * invz_2;
        j[(2, 1)] = j[(0, 1)] + bf * x * invz_2;
        j[(2, 2)] = j[(0, 2)];
        j[(2, 3)] = j[(0, 3)];
        j[(2, 4)] = 0.0;
        j[(2, 5)] = j[(0, 5)] - bf * invz_2;

        EdgeLinearization {
            error: to_dvec(&self.error),
            information: to_dmat(&self.information),
            jacobians: vec![to_dmat(&j)],
        }
    }
}

// ===========================================================================
// VertexSim3Expmap  (ORB_SLAM3 / g2o Sim3 pose vertex)
// ===========================================================================

use crate::g2o_core::{Sim3, Vector7};

/// Similarity-transform vertex `S12`. `oplus`: `est = Sim3(update)·est`, with the
/// scale component frozen when `fix_scale` is set (`OptimizableTypes.cpp`).
pub struct VertexSim3Expmap {
    estimate: Sim3,
    backup: Vec<Sim3>,
    fixed: bool,
    hessian_index: i32,
    pub fix_scale: bool,
}

impl VertexSim3Expmap {
    pub fn new(estimate: Sim3, fix_scale: bool) -> Self {
        VertexSim3Expmap {
            estimate,
            backup: Vec::new(),
            fixed: false,
            hessian_index: -1,
            fix_scale,
        }
    }
    pub fn estimate(&self) -> Sim3 {
        self.estimate
    }
    pub fn set_estimate(&mut self, e: Sim3) {
        self.estimate = e;
    }
}

impl Vertex for VertexSim3Expmap {
    fn dim(&self) -> usize {
        7
    }
    fn oplus(&mut self, delta: &[f64]) {
        let mut update = Vector7::from_column_slice(delta);
        if self.fix_scale {
            update[6] = 0.0;
        }
        self.estimate = Sim3::from_update(&update).mul(&self.estimate);
    }
    fn push(&mut self) {
        self.backup.push(self.estimate);
    }
    fn pop(&mut self) {
        if let Some(e) = self.backup.pop() {
            self.estimate = e;
        }
    }
    fn discard_top(&mut self) {
        self.backup.pop();
    }
    fn fixed(&self) -> bool {
        self.fixed
    }
    fn set_fixed(&mut self, fixed: bool) {
        self.fixed = fixed;
    }
    fn hessian_index(&self) -> i32 {
        self.hessian_index
    }
    fn set_hessian_index(&mut self, idx: i32) {
        self.hessian_index = idx;
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

fn sim3_of(v: &dyn Vertex) -> (Sim3, bool) {
    let v = v
        .as_any()
        .downcast_ref::<VertexSim3Expmap>()
        .expect("expected VertexSim3Expmap");
    (v.estimate(), v.fix_scale)
}

/// Central-difference numerical Jacobian (2×7) of an error function of the Sim3
/// vertex, matching g2o's default `linearizeOplus` (delta = 1e-9). The scale
/// column is frozen when `fix_scale` is set, via the same `oplus` rule.
fn numeric_sim3_jacobian(
    s: &Sim3,
    fix_scale: bool,
    err: impl Fn(&Sim3) -> Vector2<f64>,
) -> nalgebra::SMatrix<f64, 2, 7> {
    let delta = 1e-9;
    let scalar = 1.0 / (2.0 * delta);
    let mut jac = nalgebra::SMatrix::<f64, 2, 7>::zeros();
    for d in 0..7 {
        let mut up = Vector7::zeros();
        up[d] = delta;
        if fix_scale {
            up[6] = 0.0;
        }
        let s_plus = Sim3::from_update(&up).mul(s);
        let mut um = Vector7::zeros();
        um[d] = -delta;
        if fix_scale {
            um[6] = 0.0;
        }
        let s_minus = Sim3::from_update(&um).mul(s);
        let col = (err(&s_plus) - err(&s_minus)) * scalar;
        jac[(0, d)] = col[0];
        jac[(1, d)] = col[1];
    }
    jac
}

// ===========================================================================
// EdgeSim3ProjectXYZ      x1 = S12 * X2   (forward)
// EdgeInverseSim3ProjectXYZ  x2 = S12⁻¹ * X1  (inverse)
// ===========================================================================

/// `EdgeSim3ProjectXYZ`: reprojection of point `X2` (vertex 0, fixed) into
/// camera 1 through `S12` (vertex 1). Jacobian is numerical (as in g2o).
pub struct EdgeSim3ProjectXYZ {
    vertices: [usize; 2],
    pub measurement: Vector2<f64>,
    pub information: Matrix2<f64>,
    robust_delta: Option<f64>,
    level: i32,
    error: Vector2<f64>,
    pub camera1: Arc<dyn GeometricCamera>,
}

impl EdgeSim3ProjectXYZ {
    pub fn new(point: usize, sim3: usize, camera1: Arc<dyn GeometricCamera>) -> Self {
        EdgeSim3ProjectXYZ {
            vertices: [point, sim3],
            measurement: Vector2::zeros(),
            information: Matrix2::identity(),
            robust_delta: None,
            level: 0,
            error: Vector2::zeros(),
            camera1,
        }
    }
    pub fn set_measurement(&mut self, m: Vector2<f64>) {
        self.measurement = m;
    }
    pub fn set_information(&mut self, info: Matrix2<f64>) {
        self.information = info;
    }
    fn err_at(&self, s12: &Sim3, x2: &Vector3<f64>) -> Vector2<f64> {
        let proj = self.camera1.project_n_d(&Point3::from(s12.map(x2)));
        self.measurement - Vector2::new(proj.x, proj.y)
    }
}

impl Edge for EdgeSim3ProjectXYZ {
    fn vertices(&self) -> &[usize] {
        &self.vertices
    }
    fn dim(&self) -> usize {
        2
    }
    fn level(&self) -> i32 {
        self.level
    }
    fn set_level(&mut self, level: i32) {
        self.level = level;
    }
    fn robust_kernel(&self) -> Option<f64> {
        self.robust_delta
    }
    fn set_robust_kernel(&mut self, delta: Option<f64>) {
        self.robust_delta = delta;
    }
    fn compute_error(&mut self, vertices: &[&dyn Vertex]) {
        let (s12, _) = sim3_of(vertices[1]);
        let x2 = point_of(vertices[0]);
        self.error = self.err_at(&s12, &x2);
    }
    fn chi2(&self) -> f64 {
        (self.error.transpose() * self.information * self.error)[(0, 0)]
    }
    fn linearize(&mut self, vertices: &[&dyn Vertex]) -> EdgeLinearization {
        self.compute_error(vertices);
        let (s12, fix_scale) = sim3_of(vertices[1]);
        let x2 = point_of(vertices[0]);
        let jac_sim3 = numeric_sim3_jacobian(&s12, fix_scale, |s| self.err_at(s, &x2));
        // Point vertex is fixed in OptimizeSim3 (jacobian unused).
        let jac_point = nalgebra::Matrix2x3::<f64>::zeros();
        EdgeLinearization {
            error: to_dvec(&self.error),
            information: to_dmat(&self.information),
            jacobians: vec![to_dmat(&jac_point), to_dmat(&jac_sim3)],
        }
    }
}

/// `EdgeInverseSim3ProjectXYZ`: reprojection of point `X1` (vertex 0, fixed)
/// into camera 2 through `S12⁻¹`. Jacobian is numerical.
pub struct EdgeInverseSim3ProjectXYZ {
    vertices: [usize; 2],
    pub measurement: Vector2<f64>,
    pub information: Matrix2<f64>,
    robust_delta: Option<f64>,
    level: i32,
    error: Vector2<f64>,
    pub camera2: Arc<dyn GeometricCamera>,
}

impl EdgeInverseSim3ProjectXYZ {
    pub fn new(point: usize, sim3: usize, camera2: Arc<dyn GeometricCamera>) -> Self {
        EdgeInverseSim3ProjectXYZ {
            vertices: [point, sim3],
            measurement: Vector2::zeros(),
            information: Matrix2::identity(),
            robust_delta: None,
            level: 0,
            error: Vector2::zeros(),
            camera2,
        }
    }
    pub fn set_measurement(&mut self, m: Vector2<f64>) {
        self.measurement = m;
    }
    pub fn set_information(&mut self, info: Matrix2<f64>) {
        self.information = info;
    }
    fn err_at(&self, s12: &Sim3, x1: &Vector3<f64>) -> Vector2<f64> {
        let proj = self
            .camera2
            .project_n_d(&Point3::from(s12.inverse().map(x1)));
        self.measurement - Vector2::new(proj.x, proj.y)
    }
}

impl Edge for EdgeInverseSim3ProjectXYZ {
    fn vertices(&self) -> &[usize] {
        &self.vertices
    }
    fn dim(&self) -> usize {
        2
    }
    fn level(&self) -> i32 {
        self.level
    }
    fn set_level(&mut self, level: i32) {
        self.level = level;
    }
    fn robust_kernel(&self) -> Option<f64> {
        self.robust_delta
    }
    fn set_robust_kernel(&mut self, delta: Option<f64>) {
        self.robust_delta = delta;
    }
    fn compute_error(&mut self, vertices: &[&dyn Vertex]) {
        let (s12, _) = sim3_of(vertices[1]);
        let x1 = point_of(vertices[0]);
        self.error = self.err_at(&s12, &x1);
    }
    fn chi2(&self) -> f64 {
        (self.error.transpose() * self.information * self.error)[(0, 0)]
    }
    fn linearize(&mut self, vertices: &[&dyn Vertex]) -> EdgeLinearization {
        self.compute_error(vertices);
        let (s12, fix_scale) = sim3_of(vertices[1]);
        let x1 = point_of(vertices[0]);
        let jac_sim3 = numeric_sim3_jacobian(&s12, fix_scale, |s| self.err_at(s, &x1));
        let jac_point = nalgebra::Matrix2x3::<f64>::zeros();
        EdgeLinearization {
            error: to_dvec(&self.error),
            information: to_dmat(&self.information),
            jacobians: vec![to_dmat(&jac_point), to_dmat(&jac_sim3)],
        }
    }
}

// ===========================================================================
// EdgeSim3  (g2o builtin 7-DoF Sim3 pose-graph edge)
// ===========================================================================

use crate::g2o_core::Vector7 as V7;

/// Relative-Sim3 constraint between two [`VertexSim3Expmap`] vertices:
/// `error = log(C · S0 · S1⁻¹)` where `C` is the measured `Sji`. Numerical
/// 7×7 Jacobians (g2o has no analytic `linearizeOplus` for this edge).
pub struct EdgeSim3 {
    vertices: [usize; 2],
    pub measurement: Sim3,
    pub information: nalgebra::SMatrix<f64, 7, 7>,
    level: i32,
    error: V7,
}

impl EdgeSim3 {
    pub fn new(v0: usize, v1: usize, measurement: Sim3) -> Self {
        EdgeSim3 {
            vertices: [v0, v1],
            measurement,
            information: nalgebra::SMatrix::<f64, 7, 7>::identity(),
            level: 0,
            error: V7::zeros(),
        }
    }
    fn err_at(&self, s0: &Sim3, s1: &Sim3) -> V7 {
        self.measurement.mul(s0).mul(&s1.inverse()).log()
    }
}

impl Edge for EdgeSim3 {
    fn vertices(&self) -> &[usize] {
        &self.vertices
    }
    fn dim(&self) -> usize {
        7
    }
    fn level(&self) -> i32 {
        self.level
    }
    fn set_level(&mut self, level: i32) {
        self.level = level;
    }
    fn robust_kernel(&self) -> Option<f64> {
        None
    }
    fn set_robust_kernel(&mut self, _delta: Option<f64>) {}
    fn compute_error(&mut self, vertices: &[&dyn Vertex]) {
        let (s0, _) = sim3_of(vertices[0]);
        let (s1, _) = sim3_of(vertices[1]);
        self.error = self.err_at(&s0, &s1);
    }
    fn chi2(&self) -> f64 {
        (self.error.transpose() * self.information * self.error)[(0, 0)]
    }
    fn linearize(&mut self, vertices: &[&dyn Vertex]) -> EdgeLinearization {
        self.compute_error(vertices);
        let (s0, fs0) = sim3_of(vertices[0]);
        let (s1, fs1) = sim3_of(vertices[1]);

        let delta = 1e-9;
        let scalar = 1.0 / (2.0 * delta);
        let mut j0 = nalgebra::SMatrix::<f64, 7, 7>::zeros();
        let mut j1 = nalgebra::SMatrix::<f64, 7, 7>::zeros();
        for d in 0..7 {
            // w.r.t. vertex 0
            let (mut up, mut um) = (V7::zeros(), V7::zeros());
            up[d] = delta;
            um[d] = -delta;
            if fs0 {
                up[6] = 0.0;
                um[6] = 0.0;
            }
            let s0p = Sim3::from_update(&up).mul(&s0);
            let s0m = Sim3::from_update(&um).mul(&s0);
            let col0 = (self.err_at(&s0p, &s1) - self.err_at(&s0m, &s1)) * scalar;
            j0.set_column(d, &col0);

            // w.r.t. vertex 1
            let (mut up1, mut um1) = (V7::zeros(), V7::zeros());
            up1[d] = delta;
            um1[d] = -delta;
            if fs1 {
                up1[6] = 0.0;
                um1[6] = 0.0;
            }
            let s1p = Sim3::from_update(&up1).mul(&s1);
            let s1m = Sim3::from_update(&um1).mul(&s1);
            let col1 = (self.err_at(&s0, &s1p) - self.err_at(&s0, &s1m)) * scalar;
            j1.set_column(d, &col1);
        }

        EdgeLinearization {
            error: to_dvec(&self.error),
            information: to_dmat(&self.information),
            jacobians: vec![to_dmat(&j0), to_dmat(&j1)],
        }
    }
}
