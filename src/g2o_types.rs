//! g2o vertex/edge types used by `Optimizer.cc`, including the full IMU
//! (inertial) set ported from `ORB_SLAM3/src/G2oTypes.cc`.
//!
//! All analytic Jacobians match upstream exactly (parity with g2o). Multi-vertex
//! edges (`EdgeInertial` over 6 vertices, `EdgeInertialGS` over 8,
//! `EdgePriorPoseImu` over 4) are supported by the generic [`crate::g2o_core`]
//! engine.

use std::any::Any;
use std::sync::Arc;

use nalgebra::{
    DMatrix, DVector, Matrix2, Matrix3, Point3, SMatrix, SVector, SymmetricEigen, Vector2, Vector3,
};

use crate::camera_models::GeometricCamera;
use crate::g2o_core::{
    Edge, EdgeLinearization, Vertex, exp_so3, inverse_right_jacobian_so3, log_so3,
    normalize_rotation, right_jacobian_so3, skew,
};
use crate::imu_types::{Bias, Preintegrated};
use crate::optimizable_types::VertexSBAPointXYZ;

const GRAVITY_VALUE: f64 = 9.81;

fn dmat<const R: usize, const C: usize>(m: &SMatrix<f64, R, C>) -> DMatrix<f64> {
    DMatrix::from_iterator(R, C, m.iter().copied())
}
fn dvec<const R: usize>(v: &SVector<f64, R>) -> DVector<f64> {
    DVector::from_iterator(R, v.iter().copied())
}

type Matrix15d = SMatrix<f64, 15, 15>;

/// IMU pose+bias prior constraint (`ORB_SLAM3::ConstraintPoseImu`). Stores a
/// reference state and its 15×15 information matrix (symmetrized + eigen-clamped).
#[derive(Clone)]
pub struct ConstraintPoseIMU {
    rwb: Matrix3<f64>,
    twb: Vector3<f64>,
    vwb: Vector3<f64>,
    bg: Vector3<f64>,
    ba: Vector3<f64>,
    h: Matrix15d,
}
impl ConstraintPoseIMU {
    pub fn new(
        rwb: Matrix3<f64>,
        twb: Vector3<f64>,
        vwb: Vector3<f64>,
        bg: Vector3<f64>,
        ba: Vector3<f64>,
        h: SMatrix<f64, 15, 15>,
    ) -> Self {
        let h = (h + h.transpose()) / 2.;
        let es = SymmetricEigen::new(h);
        let mut eigs = es.eigenvalues;
        for i in 0..15 {
            if eigs[i] < 1e-12 {
                eigs[i] = 0.;
            }
        }
        let h = es.eigenvectors * Matrix15d::from_diagonal(&eigs) * es.eigenvectors.transpose();

        ConstraintPoseIMU {
            rwb,
            twb,
            vwb,
            bg,
            ba,
            h,
        }
    }

    pub fn rwb(&self) -> Matrix3<f64> {
        self.rwb
    }
    pub fn twb(&self) -> Vector3<f64> {
        self.twb
    }
    pub fn vwb(&self) -> Vector3<f64> {
        self.vwb
    }
    pub fn bg(&self) -> Vector3<f64> {
        self.bg
    }
    pub fn ba(&self) -> Vector3<f64> {
        self.ba
    }
    pub fn h(&self) -> Matrix15d {
        self.h
    }
}
// ===========================================================================
// ImuCamPose
// ===========================================================================

/// IMU body pose plus the derived camera poses for a set of cameras
/// (`ORB_SLAM3::ImuCamPose`). Optimization variable of [`VertexPose`].
#[derive(Clone)]
pub struct ImuCamPose {
    pub rwb: Matrix3<f64>,
    pub twb: Vector3<f64>,
    pub rcw: Vec<Matrix3<f64>>,
    pub tcw: Vec<Vector3<f64>>,
    pub rcb: Vec<Matrix3<f64>>,
    pub rbc: Vec<Matrix3<f64>>,
    pub tcb: Vec<Vector3<f64>>,
    pub tbc: Vec<Vector3<f64>>,
    pub bf: f64,
    pub cameras: Vec<Arc<dyn GeometricCamera>>,
    its: i32,
    /// For the 4-DoF pose-graph update (`UpdateW`): reference rotation + the
    /// accumulated yaw-only delta.
    pub rwb0: Matrix3<f64>,
    pub dr: Matrix3<f64>,
}

impl ImuCamPose {
    /// Build from camera world poses `Rcw/tcw` and body↔camera extrinsics
    /// `Rbc/tbc` (mirrors `ImuCamPose::SetParam`). `Rwb/twb` are derived from
    /// camera 0.
    pub fn new(
        rcw: Vec<Matrix3<f64>>,
        tcw: Vec<Vector3<f64>>,
        rbc: Vec<Matrix3<f64>>,
        tbc: Vec<Vector3<f64>>,
        bf: f64,
        cameras: Vec<Arc<dyn GeometricCamera>>,
    ) -> Self {
        let num = rbc.len();
        let mut rcb = Vec::with_capacity(num);
        let mut tcb = Vec::with_capacity(num);
        for i in 0..num {
            rcb.push(rbc[i].transpose());
            tcb.push(-rcb[i] * tbc[i]);
        }
        let rwb = rcw[0].transpose() * rcb[0];
        let twb = rcw[0].transpose() * (tcb[0] - tcw[0]);
        ImuCamPose {
            rwb,
            twb,
            rcw,
            tcw,
            rcb,
            rbc,
            tcb,
            tbc,
            bf,
            cameras,
            its: 0,
            rwb0: rwb,
            dr: Matrix3::identity(),
        }
    }

    /// `UpdateW` — 4-DoF-style update in the world reference (yaw + translation),
    /// used by [`VertexPose4DoF`]. `pu = [ωr(3), υt(3)]`.
    pub fn update_w(&mut self, pu: &[f64]) {
        let ur = Vector3::new(pu[0], pu[1], pu[2]);
        let ut = Vector3::new(pu[3], pu[4], pu[5]);
        self.dr = exp_so3(&ur) * self.dr;
        self.rwb = self.dr * self.rwb0;
        self.twb += ut;
        self.its += 1;
        if self.its >= 5 {
            self.dr[(0, 2)] = 0.0;
            self.dr[(1, 2)] = 0.0;
            self.dr[(2, 0)] = 0.0;
            self.dr[(2, 1)] = 0.0;
            self.dr = normalize_rotation(&self.dr);
            self.its = 0;
        }
        let rbw = self.rwb.transpose();
        let tbw = -rbw * self.twb;
        for i in 0..self.cameras.len() {
            self.rcw[i] = self.rcb[i] * rbw;
            self.tcw[i] = self.rcb[i] * tbw + self.tcb[i];
        }
    }

    /// `Update` — 6-DoF increment in the IMU reference: `pu = [ωr, ωt]`.
    pub fn update(&mut self, pu: &[f64]) {
        let ur = Vector3::new(pu[0], pu[1], pu[2]);
        let ut = Vector3::new(pu[3], pu[4], pu[5]);

        self.twb += self.rwb * ut;
        self.rwb *= exp_so3(&ur);

        self.its += 1;
        if self.its >= 3 {
            self.rwb = normalize_rotation(&self.rwb);
            self.its = 0;
        }

        let rbw = self.rwb.transpose();
        let tbw = -rbw * self.twb;
        for i in 0..self.cameras.len() {
            self.rcw[i] = self.rcb[i] * rbw;
            self.tcw[i] = self.rcb[i] * tbw + self.tcb[i];
        }
    }

    /// Mono projection of world point `xw` into camera `cam_idx`.
    pub fn project(&self, xw: &Vector3<f64>, cam_idx: usize) -> Vector2<f64> {
        let xc = self.rcw[cam_idx] * xw + self.tcw[cam_idx];
        let p = self.cameras[cam_idx].project_n_d(&nalgebra::Point3::from(xc));
        Vector2::new(p.x, p.y)
    }

    /// Stereo projection (u, v, u_right).
    pub fn project_stereo(&self, xw: &Vector3<f64>, cam_idx: usize) -> Vector3<f64> {
        let pc = self.rcw[cam_idx] * xw + self.tcw[cam_idx];
        let inv_z = 1.0 / pc[2];
        let uv = self.cameras[cam_idx].project_n_d(&nalgebra::Point3::from(pc));
        Vector3::new(uv.x, uv.y, uv.x - self.bf * inv_z)
    }

    pub fn is_depth_positive(&self, xw: &Vector3<f64>, cam_idx: usize) -> bool {
        (self.rcw[cam_idx].row(2) * xw)[0] + self.tcw[cam_idx][2] > 0.0
    }
}

// ===========================================================================
// Vertices
// ===========================================================================

macro_rules! impl_vertex_common {
    ($name:ty) => {
        fn push(&mut self) {
            self.backup.push(self.estimate.clone());
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
    };
}

/// 6-DoF IMU pose vertex (`VertexPose`).
pub struct VertexPose {
    estimate: ImuCamPose,
    backup: Vec<ImuCamPose>,
    fixed: bool,
    hessian_index: i32,
}
impl VertexPose {
    pub fn new(estimate: ImuCamPose) -> Self {
        VertexPose {
            estimate,
            backup: Vec::new(),
            fixed: false,
            hessian_index: -1,
        }
    }
    pub fn estimate(&self) -> &ImuCamPose {
        &self.estimate
    }
}
impl Vertex for VertexPose {
    fn dim(&self) -> usize {
        6
    }
    fn oplus(&mut self, delta: &[f64]) {
        self.estimate.update(delta);
    }
    impl_vertex_common!(VertexPose);
}

/// Generic 3-vector vertex (`oplus`: `est += δ`) — velocity / gyro / acc bias.
macro_rules! vector3_vertex {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        pub struct $name {
            estimate: Vector3<f64>,
            backup: Vec<Vector3<f64>>,
            fixed: bool,
            hessian_index: i32,
        }
        impl $name {
            pub fn new(estimate: Vector3<f64>) -> Self {
                $name {
                    estimate,
                    backup: Vec::new(),
                    fixed: false,
                    hessian_index: -1,
                }
            }
            pub fn estimate(&self) -> Vector3<f64> {
                self.estimate
            }
        }
        impl Vertex for $name {
            fn dim(&self) -> usize {
                3
            }
            fn oplus(&mut self, delta: &[f64]) {
                self.estimate += Vector3::new(delta[0], delta[1], delta[2]);
            }
            impl_vertex_common!($name);
        }
    };
}
vector3_vertex!(VertexVelocity, "IMU velocity vertex (`VertexVelocity`).");
vector3_vertex!(VertexGyroBias, "Gyroscope bias vertex (`VertexGyroBias`).");
vector3_vertex!(
    VertexAccBias,
    "Accelerometer bias vertex (`VertexAccBias`)."
);

/// Scale vertex (`VertexScale`); `oplus`: `est *= exp(δ)`.
pub struct VertexScale {
    estimate: f64,
    backup: Vec<f64>,
    fixed: bool,
    hessian_index: i32,
}
impl VertexScale {
    pub fn new(estimate: f64) -> Self {
        VertexScale {
            estimate,
            backup: Vec::new(),
            fixed: false,
            hessian_index: -1,
        }
    }
    pub fn estimate(&self) -> f64 {
        self.estimate
    }
}
impl Vertex for VertexScale {
    fn dim(&self) -> usize {
        1
    }
    fn oplus(&mut self, delta: &[f64]) {
        self.estimate *= delta[0].exp();
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

/// Gravity-direction vertex (`VertexGDir`); stores `Rwg`, `oplus`:
/// `Rwg = Rwg · ExpSO3(δx, δy, 0)`.
pub struct VertexGDir {
    estimate: Matrix3<f64>,
    backup: Vec<Matrix3<f64>>,
    fixed: bool,
    hessian_index: i32,
}
impl VertexGDir {
    pub fn new(rwg: Matrix3<f64>) -> Self {
        VertexGDir {
            estimate: rwg,
            backup: Vec::new(),
            fixed: false,
            hessian_index: -1,
        }
    }
    pub fn estimate(&self) -> Matrix3<f64> {
        self.estimate
    }
}
impl Vertex for VertexGDir {
    fn dim(&self) -> usize {
        2
    }
    fn oplus(&mut self, delta: &[f64]) {
        self.estimate *= exp_so3(&Vector3::new(delta[0], delta[1], 0.0));
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

// --- downcast helpers ---
fn pose_of(v: &dyn Vertex) -> &ImuCamPose {
    v.as_any()
        .downcast_ref::<VertexPose>()
        .expect("VertexPose")
        .estimate()
}
fn vel_of(v: &dyn Vertex) -> Vector3<f64> {
    v.as_any()
        .downcast_ref::<VertexVelocity>()
        .expect("VertexVelocity")
        .estimate()
}
fn gyro_of(v: &dyn Vertex) -> Vector3<f64> {
    v.as_any()
        .downcast_ref::<VertexGyroBias>()
        .expect("VertexGyroBias")
        .estimate()
}
fn acc_of(v: &dyn Vertex) -> Vector3<f64> {
    v.as_any()
        .downcast_ref::<VertexAccBias>()
        .expect("VertexAccBias")
        .estimate()
}
fn point_of(v: &dyn Vertex) -> Vector3<f64> {
    use crate::optimizable_types::VertexSBAPointXYZ;
    v.as_any()
        .downcast_ref::<VertexSBAPointXYZ>()
        .expect("VertexSBAPointXYZ")
        .estimate()
}

/// The `[0 z -y; -z 0 x; y -x 0 | I₃]` block (`SE3deriv`).
fn se3_deriv(x: f64, y: f64, z: f64) -> nalgebra::SMatrix<f64, 3, 6> {
    nalgebra::SMatrix::<f64, 3, 6>::from_row_slice(&[
        0.0, z, -y, 1.0, 0.0, 0.0, //
        -z, 0.0, x, 0.0, 1.0, 0.0, //
        y, -x, 0.0, 0.0, 0.0, 1.0,
    ])
}

// ===========================================================================
// Reprojection edges through VertexPose
// ===========================================================================

/// Mono reprojection optimizing point (`vertex 0`) + IMU pose (`vertex 1`).
pub struct EdgeMono {
    vertices: [usize; 2],
    pub measurement: Vector2<f64>,
    pub information: nalgebra::Matrix2<f64>,
    robust_delta: Option<f64>,
    level: i32,
    error: Vector2<f64>,
    cam_idx: usize,
}
impl EdgeMono {
    pub fn new(point: usize, pose: usize, cam_idx: usize) -> Self {
        EdgeMono {
            vertices: [point, pose],
            measurement: Vector2::zeros(),
            information: nalgebra::Matrix2::identity(),
            robust_delta: None,
            level: 0,
            error: Vector2::zeros(),
            cam_idx,
        }
    }
    pub fn set_measurement(&mut self, m: Vector2<f64>) {
        self.measurement = m;
    }
    pub fn set_information(&mut self, i: nalgebra::Matrix2<f64>) {
        self.information = i;
    }
}
impl Edge for EdgeMono {
    fn vertices(&self) -> &[usize] {
        &self.vertices
    }
    fn dim(&self) -> usize {
        2
    }
    fn level(&self) -> i32 {
        self.level
    }
    fn set_level(&mut self, l: i32) {
        self.level = l;
    }
    fn robust_kernel(&self) -> Option<f64> {
        self.robust_delta
    }
    fn set_robust_kernel(&mut self, d: Option<f64>) {
        self.robust_delta = d;
    }
    fn compute_error(&mut self, vertices: &[&dyn Vertex]) {
        let p = point_of(vertices[0]);
        let pose = pose_of(vertices[1]);
        self.error = self.measurement - pose.project(&p, self.cam_idx);
    }
    fn chi2(&self) -> f64 {
        (self.error.transpose() * self.information * self.error)[(0, 0)]
    }
    fn depth_positive(&self, vertices: &[&dyn Vertex]) -> bool {
        pose_of(vertices[1]).is_depth_positive(&point_of(vertices[0]), self.cam_idx)
    }
    fn linearize(&mut self, vertices: &[&dyn Vertex]) -> EdgeLinearization {
        self.compute_error(vertices);
        let p = point_of(vertices[0]);
        let pose = pose_of(vertices[1]);
        let ci = self.cam_idx;
        let xc = pose.rcw[ci] * p + pose.tcw[ci];
        let xb = pose.rbc[ci] * xc + pose.tbc[ci];
        let proj_jac = pose.cameras[ci].project_jac(&nalgebra::Point3::from(xc)); // 2×3
        let ji = -proj_jac * pose.rcw[ci]; // 2×3 (point)
        let jj = proj_jac * pose.rcb[ci] * se3_deriv(xb[0], xb[1], xb[2]); // 2×6 (pose)
        EdgeLinearization {
            error: dvec(&self.error),
            information: dmat(&self.information),
            jacobians: vec![dmat(&ji), dmat(&jj)],
        }
    }
}

/// Mono reprojection against a fixed world point, optimizing IMU pose only.
pub struct EdgeMonoOnlyPose {
    vertices: [usize; 1],
    pub measurement: Vector2<f64>,
    pub information: nalgebra::Matrix2<f64>,
    robust_delta: Option<f64>,
    level: i32,
    error: Vector2<f64>,
    xw: Vector3<f64>,
    cam_idx: usize,
}
impl EdgeMonoOnlyPose {
    pub fn new(pose: usize, xw: Vector3<f64>, cam_idx: usize) -> Self {
        EdgeMonoOnlyPose {
            vertices: [pose],
            measurement: Vector2::zeros(),
            information: nalgebra::Matrix2::identity(),
            robust_delta: None,
            level: 0,
            error: Vector2::zeros(),
            xw,
            cam_idx,
        }
    }
    pub fn set_measurement(&mut self, m: Vector2<f64>) {
        self.measurement = m;
    }
    pub fn set_information(&mut self, i: nalgebra::Matrix2<f64>) {
        self.information = i;
    }
}
impl Edge for EdgeMonoOnlyPose {
    fn vertices(&self) -> &[usize] {
        &self.vertices
    }
    fn dim(&self) -> usize {
        2
    }
    fn level(&self) -> i32 {
        self.level
    }
    fn set_level(&mut self, l: i32) {
        self.level = l;
    }
    fn robust_kernel(&self) -> Option<f64> {
        self.robust_delta
    }
    fn set_robust_kernel(&mut self, d: Option<f64>) {
        self.robust_delta = d;
    }
    fn compute_error(&mut self, vertices: &[&dyn Vertex]) {
        let pose = pose_of(vertices[0]);
        self.error = self.measurement - pose.project(&self.xw, self.cam_idx);
    }
    fn chi2(&self) -> f64 {
        (self.error.transpose() * self.information * self.error)[(0, 0)]
    }
    fn depth_positive(&self, vertices: &[&dyn Vertex]) -> bool {
        pose_of(vertices[0]).is_depth_positive(&self.xw, self.cam_idx)
    }
    fn linearize(&mut self, vertices: &[&dyn Vertex]) -> EdgeLinearization {
        self.compute_error(vertices);
        let pose = pose_of(vertices[0]);
        let ci = self.cam_idx;
        let xc = pose.rcw[ci] * self.xw + pose.tcw[ci];
        let xb = pose.rbc[ci] * xc + pose.tbc[ci];
        let proj_jac = pose.cameras[ci].project_jac(&nalgebra::Point3::from(xc));
        let ji = proj_jac * pose.rcb[ci] * se3_deriv(xb[0], xb[1], xb[2]); // 2×6
        EdgeLinearization {
            error: dvec(&self.error),
            information: dmat(&self.information),
            jacobians: vec![dmat(&ji)],
        }
    }
}

/// Stereo reprojection optimizing point (`vertex 0`) + IMU pose (`vertex 1`).
pub struct EdgeStereo {
    vertices: [usize; 2],
    pub measurement: Vector3<f64>,
    pub information: Matrix3<f64>,
    robust_delta: Option<f64>,
    level: i32,
    error: Vector3<f64>,
    cam_idx: usize,
}
impl EdgeStereo {
    pub fn new(point: usize, pose: usize, cam_idx: usize) -> Self {
        EdgeStereo {
            vertices: [point, pose],
            measurement: Vector3::zeros(),
            information: Matrix3::identity(),
            robust_delta: None,
            level: 0,
            error: Vector3::zeros(),
            cam_idx,
        }
    }
    pub fn set_measurement(&mut self, m: Vector3<f64>) {
        self.measurement = m;
    }
    pub fn set_information(&mut self, i: Matrix3<f64>) {
        self.information = i;
    }
    fn stereo_proj_jac(&self, pose: &ImuCamPose, xc: &Vector3<f64>) -> Matrix3<f64> {
        let pj = pose.cameras[self.cam_idx].project_jac(&nalgebra::Point3::from(*xc)); // 2×3
        let inv_z2 = 1.0 / (xc[2] * xc[2]);
        let mut j = Matrix3::zeros();
        j.fixed_view_mut::<2, 3>(0, 0).copy_from(&pj);
        let row0 = pj.row(0).into_owned();
        j.fixed_view_mut::<1, 3>(2, 0).copy_from(&row0);
        j[(2, 2)] += pose.bf * inv_z2;
        j
    }
}
impl Edge for EdgeStereo {
    fn vertices(&self) -> &[usize] {
        &self.vertices
    }
    fn dim(&self) -> usize {
        3
    }
    fn level(&self) -> i32 {
        self.level
    }
    fn set_level(&mut self, l: i32) {
        self.level = l;
    }
    fn robust_kernel(&self) -> Option<f64> {
        self.robust_delta
    }
    fn set_robust_kernel(&mut self, d: Option<f64>) {
        self.robust_delta = d;
    }
    fn compute_error(&mut self, vertices: &[&dyn Vertex]) {
        let p = point_of(vertices[0]);
        let pose = pose_of(vertices[1]);
        self.error = self.measurement - pose.project_stereo(&p, self.cam_idx);
    }
    fn chi2(&self) -> f64 {
        (self.error.transpose() * self.information * self.error)[(0, 0)]
    }
    fn depth_positive(&self, vertices: &[&dyn Vertex]) -> bool {
        pose_of(vertices[1]).is_depth_positive(&point_of(vertices[0]), self.cam_idx)
    }
    fn linearize(&mut self, vertices: &[&dyn Vertex]) -> EdgeLinearization {
        self.compute_error(vertices);
        let p = point_of(vertices[0]);
        let pose = pose_of(vertices[1]);
        let ci = self.cam_idx;
        let xc = pose.rcw[ci] * p + pose.tcw[ci];
        let xb = pose.rbc[ci] * xc + pose.tbc[ci];
        let proj_jac = self.stereo_proj_jac(pose, &xc); // 3×3
        let ji = -proj_jac * pose.rcw[ci]; // 3×3 (point)
        let jj = proj_jac * pose.rcb[ci] * se3_deriv(xb[0], xb[1], xb[2]); // 3×6 (pose)
        EdgeLinearization {
            error: dvec(&self.error),
            information: dmat(&self.information),
            jacobians: vec![dmat(&ji), dmat(&jj)],
        }
    }
}

/// Stereo reprojection against a fixed world point, optimizing IMU pose only.
pub struct EdgeStereoOnlyPose {
    vertices: [usize; 1],
    pub measurement: Vector3<f64>,
    pub information: Matrix3<f64>,
    robust_delta: Option<f64>,
    level: i32,
    error: Vector3<f64>,
    xw: Vector3<f64>,
    cam_idx: usize,
}
impl EdgeStereoOnlyPose {
    pub fn new(pose: usize, xw: Vector3<f64>, cam_idx: usize) -> Self {
        EdgeStereoOnlyPose {
            vertices: [pose],
            measurement: Vector3::zeros(),
            information: Matrix3::identity(),
            robust_delta: None,
            level: 0,
            error: Vector3::zeros(),
            xw,
            cam_idx,
        }
    }
    pub fn set_measurement(&mut self, m: Vector3<f64>) {
        self.measurement = m;
    }
    pub fn set_information(&mut self, i: Matrix3<f64>) {
        self.information = i;
    }
}
impl Edge for EdgeStereoOnlyPose {
    fn vertices(&self) -> &[usize] {
        &self.vertices
    }
    fn dim(&self) -> usize {
        3
    }
    fn level(&self) -> i32 {
        self.level
    }
    fn set_level(&mut self, l: i32) {
        self.level = l;
    }
    fn robust_kernel(&self) -> Option<f64> {
        self.robust_delta
    }
    fn set_robust_kernel(&mut self, d: Option<f64>) {
        self.robust_delta = d;
    }
    fn compute_error(&mut self, vertices: &[&dyn Vertex]) {
        let pose = pose_of(vertices[0]);
        self.error = self.measurement - pose.project_stereo(&self.xw, self.cam_idx);
    }
    fn chi2(&self) -> f64 {
        (self.error.transpose() * self.information * self.error)[(0, 0)]
    }
    fn depth_positive(&self, vertices: &[&dyn Vertex]) -> bool {
        pose_of(vertices[0]).is_depth_positive(&self.xw, self.cam_idx)
    }
    fn linearize(&mut self, vertices: &[&dyn Vertex]) -> EdgeLinearization {
        self.compute_error(vertices);
        let pose = pose_of(vertices[0]);
        let ci = self.cam_idx;
        let xc = pose.rcw[ci] * self.xw + pose.tcw[ci];
        let xb = pose.rbc[ci] * xc + pose.tbc[ci];
        let pj = pose.cameras[ci].project_jac(&nalgebra::Point3::from(xc));
        let inv_z2 = 1.0 / (xc[2] * xc[2]);
        let mut proj_jac = Matrix3::zeros();
        proj_jac.fixed_view_mut::<2, 3>(0, 0).copy_from(&pj);
        let row0 = pj.row(0).into_owned();
        proj_jac.fixed_view_mut::<1, 3>(2, 0).copy_from(&row0);
        proj_jac[(2, 2)] += pose.bf * inv_z2;
        let ji = proj_jac * pose.rcb[ci] * se3_deriv(xb[0], xb[1], xb[2]); // 3×6
        EdgeLinearization {
            error: dvec(&self.error),
            information: dmat(&self.information),
            jacobians: vec![dmat(&ji)],
        }
    }
}

// ===========================================================================
// Bias random-walk edges + bias priors
// ===========================================================================

/// Random-walk edge between two bias vertices (`error = b2 − b1`). Used for both
/// gyro (`EdgeGyroRW`) and accelerometer (`EdgeAccRW`).
pub struct EdgeBiasRW {
    vertices: [usize; 2],
    pub information: Matrix3<f64>,
    level: i32,
    error: Vector3<f64>,
    is_acc: bool,
}
impl EdgeBiasRW {
    pub fn new_gyro(v1: usize, v2: usize) -> Self {
        EdgeBiasRW {
            vertices: [v1, v2],
            information: Matrix3::identity(),
            level: 0,
            error: Vector3::zeros(),
            is_acc: false,
        }
    }
    pub fn new_acc(v1: usize, v2: usize) -> Self {
        EdgeBiasRW {
            is_acc: true,
            ..Self::new_gyro(v1, v2)
        }
    }
    pub fn set_information(&mut self, i: Matrix3<f64>) {
        self.information = i;
    }
    fn b(&self, v: &dyn Vertex) -> Vector3<f64> {
        if self.is_acc { acc_of(v) } else { gyro_of(v) }
    }
}
impl Edge for EdgeBiasRW {
    fn vertices(&self) -> &[usize] {
        &self.vertices
    }
    fn dim(&self) -> usize {
        3
    }
    fn level(&self) -> i32 {
        self.level
    }
    fn set_level(&mut self, l: i32) {
        self.level = l;
    }
    fn robust_kernel(&self) -> Option<f64> {
        None
    }
    fn set_robust_kernel(&mut self, _d: Option<f64>) {}
    fn compute_error(&mut self, vertices: &[&dyn Vertex]) {
        self.error = self.b(vertices[1]) - self.b(vertices[0]);
    }
    fn chi2(&self) -> f64 {
        (self.error.transpose() * self.information * self.error)[(0, 0)]
    }
    fn linearize(&mut self, vertices: &[&dyn Vertex]) -> EdgeLinearization {
        self.compute_error(vertices);
        let ji = -Matrix3::<f64>::identity();
        let jj = Matrix3::<f64>::identity();
        EdgeLinearization {
            error: dvec(&self.error),
            information: dmat(&self.information),
            jacobians: vec![dmat(&ji), dmat(&jj)],
        }
    }
}

/// Bias prior `error = bprior − b` (unary). Jacobian = `+I` (matches g2o).
pub struct EdgeBiasPrior {
    vertices: [usize; 1],
    pub information: Matrix3<f64>,
    level: i32,
    error: Vector3<f64>,
    bprior: Vector3<f64>,
    is_acc: bool,
}
impl EdgeBiasPrior {
    pub fn new_gyro(v: usize, bprior: Vector3<f64>) -> Self {
        EdgeBiasPrior {
            vertices: [v],
            information: Matrix3::identity(),
            level: 0,
            error: Vector3::zeros(),
            bprior,
            is_acc: false,
        }
    }
    pub fn new_acc(v: usize, bprior: Vector3<f64>) -> Self {
        EdgeBiasPrior {
            is_acc: true,
            ..Self::new_gyro(v, bprior)
        }
    }
    pub fn set_information(&mut self, i: Matrix3<f64>) {
        self.information = i;
    }
    fn b(&self, v: &dyn Vertex) -> Vector3<f64> {
        if self.is_acc { acc_of(v) } else { gyro_of(v) }
    }
}
impl Edge for EdgeBiasPrior {
    fn vertices(&self) -> &[usize] {
        &self.vertices
    }
    fn dim(&self) -> usize {
        3
    }
    fn level(&self) -> i32 {
        self.level
    }
    fn set_level(&mut self, l: i32) {
        self.level = l;
    }
    fn robust_kernel(&self) -> Option<f64> {
        None
    }
    fn set_robust_kernel(&mut self, _d: Option<f64>) {}
    fn compute_error(&mut self, vertices: &[&dyn Vertex]) {
        self.error = self.bprior - self.b(vertices[0]);
    }
    fn chi2(&self) -> f64 {
        (self.error.transpose() * self.information * self.error)[(0, 0)]
    }
    fn linearize(&mut self, vertices: &[&dyn Vertex]) -> EdgeLinearization {
        self.compute_error(vertices);
        // g2o sets the Jacobian to +Identity here.
        let j = Matrix3::<f64>::identity();
        EdgeLinearization {
            error: dvec(&self.error),
            information: dmat(&self.information),
            jacobians: vec![dmat(&j)],
        }
    }
}

/// Build an IMU [`Bias`] from gyro + accelerometer bias vectors.
fn bias_from(gyro: Vector3<f64>, acc: Vector3<f64>) -> Bias {
    Bias::from_params(
        acc[0] as f32,
        acc[1] as f32,
        acc[2] as f32,
        gyro[0] as f32,
        gyro[1] as f32,
        gyro[2] as f32,
    )
}

// ===========================================================================
// Inertial edges (preintegration)
// ===========================================================================

type Matrix9 = nalgebra::SMatrix<f64, 9, 9>;
type Vector9 = nalgebra::SVector<f64, 9>;

fn m3(m: &Matrix3<f32>) -> Matrix3<f64> {
    m.cast::<f64>()
}
fn v3(v: &Vector3<f32>) -> Vector3<f64> {
    v.cast::<f64>()
}

/// Symmetrize + clamp negative eigenvalues, then return the information matrix
/// `inv(C₉ₓ₉)` exactly as `EdgeInertial`'s constructor in g2o.
fn info_from_cov9(c: &nalgebra::SMatrix<f32, 15, 15>) -> Matrix9 {
    let c9: Matrix9 = c.fixed_view::<9, 9>(0, 0).into_owned().cast::<f64>();
    let mut info = c9.try_inverse().expect("preintegration covariance inverse");
    info = (info + info.transpose()) / 2.0;
    let es = nalgebra::SymmetricEigen::new(info);
    let mut eigs = es.eigenvalues;
    for i in 0..9 {
        if eigs[i] < 1e-12 {
            eigs[i] = 0.0;
        }
    }
    es.eigenvectors * Matrix9::from_diagonal(&eigs) * es.eigenvectors.transpose()
}

/// `EdgeInertial` — 9-DoF IMU preintegration constraint linking
/// `[VP1, VV1, VG1, VA1, VP2, VV2]`. Full analytic Jacobians (parity with g2o).
pub struct EdgeInertial {
    vertices: [usize; 6],
    information: Matrix9,
    robust_delta: Option<f64>,
    level: i32,
    error: Vector9,
    preint: Arc<Preintegrated>,
    jrg: Matrix3<f64>,
    jvg: Matrix3<f64>,
    jpg: Matrix3<f64>,
    jva: Matrix3<f64>,
    jpa: Matrix3<f64>,
    dt: f64,
    g: Vector3<f64>,
}

impl EdgeInertial {
    /// `[vp1, vv1, vg1, va1, vp2, vv2]` vertex slots.
    pub fn new(vertices: [usize; 6], preint: Arc<Preintegrated>) -> Self {
        let information = info_from_cov9(&preint.c);
        EdgeInertial {
            vertices,
            information,
            robust_delta: None,
            level: 0,
            error: Vector9::zeros(),
            jrg: m3(&preint.jrg),
            jvg: m3(&preint.jvg),
            jpg: m3(&preint.jpg),
            jva: m3(&preint.jva),
            jpa: m3(&preint.jpa),
            dt: preint.dt as f64,
            g: Vector3::new(0.0, 0.0, -GRAVITY_VALUE),
            preint,
        }
    }

    /// Scale the (precomputed) information matrix — used by LocalInertialBA to
    /// down-weight the boundary inertial link.
    pub fn scale_information(&mut self, s: f64) {
        self.information *= s;
    }
}

impl Edge for EdgeInertial {
    fn vertices(&self) -> &[usize] {
        &self.vertices
    }
    fn dim(&self) -> usize {
        9
    }
    fn level(&self) -> i32 {
        self.level
    }
    fn set_level(&mut self, l: i32) {
        self.level = l;
    }
    fn robust_kernel(&self) -> Option<f64> {
        self.robust_delta
    }
    fn set_robust_kernel(&mut self, d: Option<f64>) {
        self.robust_delta = d;
    }
    fn compute_error(&mut self, vertices: &[&dyn Vertex]) {
        let vp1 = pose_of(vertices[0]);
        let vv1 = vel_of(vertices[1]);
        let vp2 = pose_of(vertices[4]);
        let vv2 = vel_of(vertices[5]);
        let b1 = bias_from(gyro_of(vertices[2]), acc_of(vertices[3]));

        let dr = m3(&self.preint.get_delta_rotation(&b1));
        let dv = v3(&self.preint.get_delta_velocity(&b1));
        let dp = v3(&self.preint.get_delta_position(&b1));

        let er = log_so3(&(dr.transpose() * vp1.rwb.transpose() * vp2.rwb));
        let ev = vp1.rwb.transpose() * (vv2 - vv1 - self.g * self.dt) - dv;
        let ep = vp1.rwb.transpose()
            * (vp2.twb - vp1.twb - vv1 * self.dt - self.g * self.dt * self.dt / 2.0)
            - dp;
        self.error = Vector9::from_iterator(er.iter().chain(ev.iter()).chain(ep.iter()).copied());
    }
    fn chi2(&self) -> f64 {
        (self.error.transpose() * self.information * self.error)[(0, 0)]
    }
    fn linearize(&mut self, vertices: &[&dyn Vertex]) -> EdgeLinearization {
        self.compute_error(vertices);
        let vp1 = pose_of(vertices[0]);
        let vv1 = vel_of(vertices[1]);
        let vp2 = pose_of(vertices[4]);
        let vv2 = vel_of(vertices[5]);
        let b1 = bias_from(gyro_of(vertices[2]), acc_of(vertices[3]));

        // Gyro bias delta.
        let dbg = Vector3::new(
            (b1.bwx - self.preint.b.bwx) as f64,
            (b1.bwy - self.preint.b.bwy) as f64,
            (b1.bwz - self.preint.b.bwz) as f64,
        );

        let rwb1 = vp1.rwb;
        let rbw1 = rwb1.transpose();
        let rwb2 = vp2.rwb;
        let dr = m3(&self.preint.get_delta_rotation(&b1));
        let er_mat = dr.transpose() * rbw1 * rwb2;
        let er = log_so3(&er_mat);
        let inv_jr = inverse_right_jacobian_so3(&er);
        let dt = self.dt;

        let mut j0 = nalgebra::SMatrix::<f64, 9, 6>::zeros();
        j0.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(-inv_jr * rwb2.transpose() * rwb1));
        j0.fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&skew(&(rbw1 * (vv2 - vv1 - self.g * dt))));
        j0.fixed_view_mut::<3, 3>(6, 0).copy_from(&skew(
            &(rbw1 * (vp2.twb - vp1.twb - vv1 * dt - 0.5 * self.g * dt * dt)),
        ));
        j0.fixed_view_mut::<3, 3>(6, 3)
            .copy_from(&(-Matrix3::<f64>::identity()));

        let mut j1 = nalgebra::SMatrix::<f64, 9, 3>::zeros();
        j1.fixed_view_mut::<3, 3>(3, 0).copy_from(&(-rbw1));
        j1.fixed_view_mut::<3, 3>(6, 0).copy_from(&(-rbw1 * dt));

        let mut j2 = nalgebra::SMatrix::<f64, 9, 3>::zeros();
        j2.fixed_view_mut::<3, 3>(0, 0).copy_from(
            &(-inv_jr * er_mat.transpose() * right_jacobian_so3(&(self.jrg * dbg)) * self.jrg),
        );
        j2.fixed_view_mut::<3, 3>(3, 0).copy_from(&(-self.jvg));
        j2.fixed_view_mut::<3, 3>(6, 0).copy_from(&(-self.jpg));

        let mut j3 = nalgebra::SMatrix::<f64, 9, 3>::zeros();
        j3.fixed_view_mut::<3, 3>(3, 0).copy_from(&(-self.jva));
        j3.fixed_view_mut::<3, 3>(6, 0).copy_from(&(-self.jpa));

        let mut j4 = nalgebra::SMatrix::<f64, 9, 6>::zeros();
        j4.fixed_view_mut::<3, 3>(0, 0).copy_from(&inv_jr);
        j4.fixed_view_mut::<3, 3>(6, 3).copy_from(&(rbw1 * rwb2));

        let mut j5 = nalgebra::SMatrix::<f64, 9, 3>::zeros();
        j5.fixed_view_mut::<3, 3>(3, 0).copy_from(&rbw1);

        EdgeLinearization {
            error: dvec(&self.error),
            information: dmat(&self.information),
            jacobians: vec![
                dmat(&j0),
                dmat(&j1),
                dmat(&j2),
                dmat(&j3),
                dmat(&j4),
                dmat(&j5),
            ],
        }
    }
}

/// `EdgeInertialGS` — like [`EdgeInertial`] but with gravity direction and scale
/// as optimizable variables. Links `[VP1, VV1, VG, VA, VP2, VV2, VGDir, VS]`.
pub struct EdgeInertialGS {
    vertices: [usize; 8],
    information: Matrix9,
    robust_delta: Option<f64>,
    level: i32,
    error: Vector9,
    preint: Arc<Preintegrated>,
    jrg: Matrix3<f64>,
    jvg: Matrix3<f64>,
    jpg: Matrix3<f64>,
    jva: Matrix3<f64>,
    jpa: Matrix3<f64>,
    dt: f64,
    gi: Vector3<f64>,
}

impl EdgeInertialGS {
    /// `[vp1, vv1, vg, va, vp2, vv2, vgdir, vs]` vertex slots.
    pub fn new(vertices: [usize; 8], preint: Arc<Preintegrated>) -> Self {
        let information = info_from_cov9(&preint.c);
        EdgeInertialGS {
            vertices,
            information,
            robust_delta: None,
            level: 0,
            error: Vector9::zeros(),
            jrg: m3(&preint.jrg),
            jvg: m3(&preint.jvg),
            jpg: m3(&preint.jpg),
            jva: m3(&preint.jva),
            jpa: m3(&preint.jpa),
            dt: preint.dt as f64,
            gi: Vector3::new(0.0, 0.0, -GRAVITY_VALUE),
            preint,
        }
    }
}

impl Edge for EdgeInertialGS {
    fn vertices(&self) -> &[usize] {
        &self.vertices
    }
    fn dim(&self) -> usize {
        9
    }
    fn level(&self) -> i32 {
        self.level
    }
    fn set_level(&mut self, l: i32) {
        self.level = l;
    }
    fn robust_kernel(&self) -> Option<f64> {
        self.robust_delta
    }
    fn set_robust_kernel(&mut self, d: Option<f64>) {
        self.robust_delta = d;
    }
    fn compute_error(&mut self, vertices: &[&dyn Vertex]) {
        let vp1 = pose_of(vertices[0]);
        let vv1 = vel_of(vertices[1]);
        let vp2 = pose_of(vertices[4]);
        let vv2 = vel_of(vertices[5]);
        let b = bias_from(gyro_of(vertices[2]), acc_of(vertices[3]));
        let rwg = vertices[6]
            .as_any()
            .downcast_ref::<VertexGDir>()
            .unwrap()
            .estimate();
        let s = vertices[7]
            .as_any()
            .downcast_ref::<VertexScale>()
            .unwrap()
            .estimate();
        let g = rwg * self.gi;

        let dr = m3(&self.preint.get_delta_rotation(&b));
        let dv = v3(&self.preint.get_delta_velocity(&b));
        let dp = v3(&self.preint.get_delta_position(&b));

        let er = log_so3(&(dr.transpose() * vp1.rwb.transpose() * vp2.rwb));
        let ev = vp1.rwb.transpose() * (s * (vv2 - vv1) - g * self.dt) - dv;
        let ep = vp1.rwb.transpose()
            * (s * (vp2.twb - vp1.twb - vv1 * self.dt) - g * self.dt * self.dt / 2.0)
            - dp;
        self.error = Vector9::from_iterator(er.iter().chain(ev.iter()).chain(ep.iter()).copied());
    }
    fn chi2(&self) -> f64 {
        (self.error.transpose() * self.information * self.error)[(0, 0)]
    }
    fn linearize(&mut self, vertices: &[&dyn Vertex]) -> EdgeLinearization {
        self.compute_error(vertices);
        let vp1 = pose_of(vertices[0]);
        let vv1 = vel_of(vertices[1]);
        let vp2 = pose_of(vertices[4]);
        let vv2 = vel_of(vertices[5]);
        let b = bias_from(gyro_of(vertices[2]), acc_of(vertices[3]));
        let rwg = vertices[6]
            .as_any()
            .downcast_ref::<VertexGDir>()
            .unwrap()
            .estimate();
        let s = vertices[7]
            .as_any()
            .downcast_ref::<VertexScale>()
            .unwrap()
            .estimate();
        let g = rwg * self.gi;

        let dbg = Vector3::new(
            (b.bwx - self.preint.b.bwx) as f64,
            (b.bwy - self.preint.b.bwy) as f64,
            (b.bwz - self.preint.b.bwz) as f64,
        );

        let rwb1 = vp1.rwb;
        let rbw1 = rwb1.transpose();
        let rwb2 = vp2.rwb;
        let mut gm = nalgebra::SMatrix::<f64, 3, 2>::zeros();
        gm[(0, 1)] = -GRAVITY_VALUE;
        gm[(1, 0)] = GRAVITY_VALUE;
        let dg_dtheta = rwg * gm; // 3×2
        let dt = self.dt;
        let dr = m3(&self.preint.get_delta_rotation(&b));
        let er_mat = dr.transpose() * rbw1 * rwb2;
        let er = log_so3(&er_mat);
        let inv_jr = inverse_right_jacobian_so3(&er);

        let mut j0 = nalgebra::SMatrix::<f64, 9, 6>::zeros();
        j0.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(-inv_jr * rwb2.transpose() * rwb1));
        j0.fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&skew(&(rbw1 * (s * (vv2 - vv1) - g * dt))));
        j0.fixed_view_mut::<3, 3>(6, 0).copy_from(&skew(
            &(rbw1 * (s * (vp2.twb - vp1.twb - vv1 * dt) - 0.5 * g * dt * dt)),
        ));
        j0.fixed_view_mut::<3, 3>(6, 3)
            .copy_from(&Matrix3::from_diagonal(&Vector3::new(-s, -s, -s)));

        let mut j1 = nalgebra::SMatrix::<f64, 9, 3>::zeros();
        j1.fixed_view_mut::<3, 3>(3, 0).copy_from(&(-s * rbw1));
        j1.fixed_view_mut::<3, 3>(6, 0).copy_from(&(-s * rbw1 * dt));

        let mut j2 = nalgebra::SMatrix::<f64, 9, 3>::zeros();
        j2.fixed_view_mut::<3, 3>(0, 0).copy_from(
            &(-inv_jr * er_mat.transpose() * right_jacobian_so3(&(self.jrg * dbg)) * self.jrg),
        );
        j2.fixed_view_mut::<3, 3>(3, 0).copy_from(&(-self.jvg));
        j2.fixed_view_mut::<3, 3>(6, 0).copy_from(&(-self.jpg));

        let mut j3 = nalgebra::SMatrix::<f64, 9, 3>::zeros();
        j3.fixed_view_mut::<3, 3>(3, 0).copy_from(&(-self.jva));
        j3.fixed_view_mut::<3, 3>(6, 0).copy_from(&(-self.jpa));

        let mut j4 = nalgebra::SMatrix::<f64, 9, 6>::zeros();
        j4.fixed_view_mut::<3, 3>(0, 0).copy_from(&inv_jr);
        j4.fixed_view_mut::<3, 3>(6, 3)
            .copy_from(&(s * rbw1 * rwb2));

        let mut j5 = nalgebra::SMatrix::<f64, 9, 3>::zeros();
        j5.fixed_view_mut::<3, 3>(3, 0).copy_from(&(s * rbw1));

        let mut j6 = nalgebra::SMatrix::<f64, 9, 2>::zeros();
        j6.fixed_view_mut::<3, 2>(3, 0)
            .copy_from(&(-rbw1 * dg_dtheta * dt));
        j6.fixed_view_mut::<3, 2>(6, 0)
            .copy_from(&(-0.5 * rbw1 * dg_dtheta * dt * dt));

        let mut j7 = nalgebra::SMatrix::<f64, 9, 1>::zeros();
        j7.fixed_view_mut::<3, 1>(3, 0)
            .copy_from(&(rbw1 * (vv2 - vv1)));
        j7.fixed_view_mut::<3, 1>(6, 0)
            .copy_from(&(rbw1 * (vp2.twb - vp1.twb - vv1 * dt)));

        EdgeLinearization {
            error: dvec(&self.error),
            information: dmat(&self.information),
            jacobians: vec![
                dmat(&j0),
                dmat(&j1),
                dmat(&j2),
                dmat(&j3),
                dmat(&j4),
                dmat(&j5),
                dmat(&j6),
                dmat(&j7),
            ],
        }
    }
}

// ===========================================================================
// EdgePriorPoseImu  (15-DoF prior over [VP, VV, VG, VA])
// ===========================================================================

type Matrix15 = nalgebra::SMatrix<f64, 15, 15>;
type Vector15 = nalgebra::SVector<f64, 15>;

/// 15-DoF prior linking `[VP, VV, VG, VA]` to a stored pose/velocity/bias with a
/// full information matrix (`EdgePriorPoseImu`, from a [`ConstraintPoseImu`]).
pub struct EdgePriorPoseImu {
    vertices: [usize; 4],
    information: Matrix15,
    robust_delta: Option<f64>,
    level: i32,
    error: Vector15,
    rwb: Matrix3<f64>,
    twb: Vector3<f64>,
    vwb: Vector3<f64>,
    bg: Vector3<f64>,
    ba: Vector3<f64>,
}

impl EdgePriorPoseImu {
    /// `[vp, vv, vg, va]` vertex slots, constrained to `c` (`ConstraintPoseImu`).
    pub fn new(vertices: [usize; 4], c: &ConstraintPoseIMU) -> Self {
        EdgePriorPoseImu {
            vertices,
            information: c.h(),
            robust_delta: None,
            level: 0,
            error: Vector15::zeros(),
            rwb: c.rwb(),
            twb: c.twb(),
            vwb: c.vwb(),
            bg: c.bg(),
            ba: c.ba(),
        }
    }
}

impl Edge for EdgePriorPoseImu {
    fn vertices(&self) -> &[usize] {
        &self.vertices
    }
    fn dim(&self) -> usize {
        15
    }
    fn level(&self) -> i32 {
        self.level
    }
    fn set_level(&mut self, l: i32) {
        self.level = l;
    }
    fn robust_kernel(&self) -> Option<f64> {
        self.robust_delta
    }
    fn set_robust_kernel(&mut self, d: Option<f64>) {
        self.robust_delta = d;
    }
    fn compute_error(&mut self, vertices: &[&dyn Vertex]) {
        let vp = pose_of(vertices[0]);
        let vv = vel_of(vertices[1]);
        let vg = gyro_of(vertices[2]);
        let va = acc_of(vertices[3]);
        let er = log_so3(&(self.rwb.transpose() * vp.rwb));
        let et = self.rwb.transpose() * (vp.twb - self.twb);
        let ev = vv - self.vwb;
        let ebg = vg - self.bg;
        let eba = va - self.ba;
        self.error = Vector15::from_iterator(
            er.iter()
                .chain(et.iter())
                .chain(ev.iter())
                .chain(ebg.iter())
                .chain(eba.iter())
                .copied(),
        );
    }
    fn chi2(&self) -> f64 {
        (self.error.transpose() * self.information * self.error)[(0, 0)]
    }
    fn linearize(&mut self, vertices: &[&dyn Vertex]) -> EdgeLinearization {
        self.compute_error(vertices);
        let vp = pose_of(vertices[0]);
        let er = log_so3(&(self.rwb.transpose() * vp.rwb));

        let mut j0 = nalgebra::SMatrix::<f64, 15, 6>::zeros();
        j0.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&inverse_right_jacobian_so3(&er));
        j0.fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(self.rwb.transpose() * vp.rwb));
        let mut j1 = nalgebra::SMatrix::<f64, 15, 3>::zeros();
        j1.fixed_view_mut::<3, 3>(6, 0)
            .copy_from(&Matrix3::<f64>::identity());
        let mut j2 = nalgebra::SMatrix::<f64, 15, 3>::zeros();
        j2.fixed_view_mut::<3, 3>(9, 0)
            .copy_from(&Matrix3::<f64>::identity());
        let mut j3 = nalgebra::SMatrix::<f64, 15, 3>::zeros();
        j3.fixed_view_mut::<3, 3>(12, 0)
            .copy_from(&Matrix3::<f64>::identity());

        EdgeLinearization {
            error: dvec(&self.error),
            information: dmat(&self.information),
            jacobians: vec![dmat(&j0), dmat(&j1), dmat(&j2), dmat(&j3)],
        }
    }
}

// ===========================================================================
// VertexPose4DoF + Edge4DoF  (4-DoF inertial pose graph)
// ===========================================================================

type Vector6 = nalgebra::SVector<f64, 6>;

/// 4-DoF pose vertex (yaw + translation), `oplus` via [`ImuCamPose::update_w`]
/// with the 4-vector mapped to `[0, 0, yaw, tx, ty, tz]` (`VertexPose4DoF`).
pub struct VertexPose4DoF {
    estimate: ImuCamPose,
    backup: Vec<ImuCamPose>,
    fixed: bool,
    hessian_index: i32,
}
impl VertexPose4DoF {
    pub fn new(estimate: ImuCamPose) -> Self {
        VertexPose4DoF {
            estimate,
            backup: Vec::new(),
            fixed: false,
            hessian_index: -1,
        }
    }
    pub fn estimate(&self) -> &ImuCamPose {
        &self.estimate
    }
}
impl Vertex for VertexPose4DoF {
    fn dim(&self) -> usize {
        4
    }
    fn oplus(&mut self, delta: &[f64]) {
        let u6 = [0.0, 0.0, delta[0], delta[1], delta[2], delta[3]];
        self.estimate.update_w(&u6);
    }
    fn push(&mut self) {
        self.backup.push(self.estimate.clone());
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

fn pose4dof_of(v: &dyn Vertex) -> &ImuCamPose {
    v.as_any()
        .downcast_ref::<VertexPose4DoF>()
        .expect("VertexPose4DoF")
        .estimate()
}

/// Relative-pose 4-DoF constraint between two [`VertexPose4DoF`] (`Edge4DoF`).
/// Numerical 6×4 Jacobians (g2o has no analytic `linearizeOplus`).
pub struct Edge4DoF {
    vertices: [usize; 2],
    pub information: nalgebra::SMatrix<f64, 6, 6>,
    level: i32,
    error: Vector6,
    drij: Matrix3<f64>,
    dtij: Vector3<f64>,
}
impl Edge4DoF {
    /// `delta_t` is the measured relative transform `Tij` (4×4).
    pub fn new(v0: usize, v1: usize, drij: Matrix3<f64>, dtij: Vector3<f64>) -> Self {
        Edge4DoF {
            vertices: [v0, v1],
            information: nalgebra::SMatrix::<f64, 6, 6>::identity(),
            level: 0,
            error: Vector6::zeros(),
            drij,
            dtij,
        }
    }
    fn err_at(&self, pi: &ImuCamPose, pj: &ImuCamPose) -> Vector6 {
        let er = log_so3(&(pi.rcw[0] * pj.rcw[0].transpose() * self.drij.transpose()));
        let et = pi.rcw[0] * (-pj.rcw[0].transpose() * pj.tcw[0]) + pi.tcw[0] - self.dtij;
        Vector6::new(er[0], er[1], er[2], et[0], et[1], et[2])
    }
}
impl Edge for Edge4DoF {
    fn vertices(&self) -> &[usize] {
        &self.vertices
    }
    fn dim(&self) -> usize {
        6
    }
    fn level(&self) -> i32 {
        self.level
    }
    fn set_level(&mut self, l: i32) {
        self.level = l;
    }
    fn robust_kernel(&self) -> Option<f64> {
        None
    }
    fn set_robust_kernel(&mut self, _d: Option<f64>) {}
    fn compute_error(&mut self, vertices: &[&dyn Vertex]) {
        self.error = self.err_at(pose4dof_of(vertices[0]), pose4dof_of(vertices[1]));
    }
    fn chi2(&self) -> f64 {
        (self.error.transpose() * self.information * self.error)[(0, 0)]
    }
    fn linearize(&mut self, vertices: &[&dyn Vertex]) -> EdgeLinearization {
        self.compute_error(vertices);
        let pi = pose4dof_of(vertices[0]).clone();
        let pj = pose4dof_of(vertices[1]).clone();
        let delta = 1e-9;
        let scalar = 1.0 / (2.0 * delta);
        let mut j0 = nalgebra::SMatrix::<f64, 6, 4>::zeros();
        let mut j1 = nalgebra::SMatrix::<f64, 6, 4>::zeros();
        for d in 0..4 {
            let mut up = [0.0f64; 4];
            let mut um = [0.0f64; 4];
            up[d] = delta;
            um[d] = -delta;
            let map_u = |u: &[f64; 4]| [0.0, 0.0, u[0], u[1], u[2], u[3]];

            let mut pip = pi.clone();
            pip.update_w(&map_u(&up));
            let mut pim = pi.clone();
            pim.update_w(&map_u(&um));
            let c0 = (self.err_at(&pip, &pj) - self.err_at(&pim, &pj)) * scalar;
            j0.set_column(d, &c0);

            let mut pjp = pj.clone();
            pjp.update_w(&map_u(&up));
            let mut pjm = pj.clone();
            pjm.update_w(&map_u(&um));
            let c1 = (self.err_at(&pi, &pjp) - self.err_at(&pi, &pjm)) * scalar;
            j1.set_column(d, &c1);
        }
        EdgeLinearization {
            error: dvec(&self.error),
            information: dmat(&self.information),
            jacobians: vec![dmat(&j0), dmat(&j1)],
        }
    }
}
