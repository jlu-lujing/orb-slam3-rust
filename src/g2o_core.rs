//! A focused Rust port of the slice of [g2o](https://github.com/RainerKuemmerle/g2o)
//!
//! This is **not** a general graph-optimization framework. It implements exactly
//! the pieces ORB-SLAM3 uses — `SE3Quat`, a Levenberg–Marquardt sparse optimizer
//! with the same trust-region update rule as upstream, the Huber robust kernel,
//! and a small set of vertex/edge traits — so that the ported optimizer routines
//! reproduce upstream numerics as closely as possible.
//!
//! References (all paths under `ORB_SLAM3/Thirdparty/g2o/g2o`):
//! * `types/se3quat.h`                            — [`SE3Quat`]
//! * `core/optimization_algorithm_levenberg.cpp`  — [`SparseOptimizer::optimize`]
//! * `core/base_unary_edge.hpp` / `base_binary_edge.hpp` — quadratic-form assembly
//! * `core/robust_kernel_impl.cpp`                — [`RobustKernelHuber`]

use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use nalgebra::SVector;
use nalgebra::{
    DMatrix, DVector, Isometry3, Matrix3, Quaternion, Translation3, UnitQuaternion, Vector3,
    Vector6,
};

/// 7-vector type alias.
pub type Vector7 = SVector<f64, 7>;

/// `SE3Quat` — a rigid-body transform stored as a unit quaternion + translation,
/// mirroring `g2o::SE3Quat` (`Thirdparty/g2o/g2o/types/se3quat.h`).
///
/// The Lie-algebra ordering matches g2o: the 6-vector is `[ω(0..3), υ(3..6)]`
/// (rotation first, then translation).
#[derive(Clone, Copy, Debug)]
pub struct SE3Quat {
    r: UnitQuaternion<f64>,
    t: Vector3<f64>,
}

/// Skew-symmetric matrix `[v]ₓ` (`g2o::skew`, `se3_ops.hpp`).
#[inline]
pub fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -v[2], v[1], //
        v[2], 0.0, -v[0], //
        -v[1], v[0], 0.0,
    )
}

/// `deltaR` (`g2o::deltaR`, `se3_ops.hpp`): the off-diagonal vector of `R`.
#[inline]
fn delta_r(r: &Matrix3<f64>) -> Vector3<f64> {
    Vector3::new(
        r[(2, 1)] - r[(1, 2)],
        r[(0, 2)] - r[(2, 0)],
        r[(1, 0)] - r[(0, 1)],
    )
}

impl Default for SE3Quat {
    fn default() -> Self {
        Self::identity()
    }
}

impl SE3Quat {
    /// Identity transform.
    pub fn identity() -> Self {
        SE3Quat {
            r: UnitQuaternion::identity(),
            t: Vector3::zeros(),
        }
    }

    /// Construct from a (already unit) quaternion and translation.
    pub fn new(r: UnitQuaternion<f64>, t: Vector3<f64>) -> Self {
        let mut s = SE3Quat { r, t };
        s.normalize_rotation();
        s
    }

    /// Construct from a rotation matrix and translation.
    pub fn from_rt(r: &Matrix3<f64>, t: Vector3<f64>) -> Self {
        let q = UnitQuaternion::from_matrix(r);
        SE3Quat::new(q, t)
    }

    /// Quaternion part.
    pub fn rotation(&self) -> UnitQuaternion<f64> {
        self.r
    }

    /// Rotation matrix.
    pub fn rotation_matrix(&self) -> Matrix3<f64> {
        self.r.to_rotation_matrix().into_inner()
    }

    /// Translation part.
    pub fn translation(&self) -> Vector3<f64> {
        self.t
    }

    /// g2o's `normalizeRotation`: keep `w >= 0` and renormalize.
    fn normalize_rotation(&mut self) {
        if self.r.w < 0.0 {
            // Flip sign — same rotation, canonical hemisphere.
            let q = self.r.quaternion();
            self.r = UnitQuaternion::new_unchecked(Quaternion::new(-q.w, -q.i, -q.j, -q.k));
        }
    }

    /// `map(xyz) = R·xyz + t`.
    #[inline]
    pub fn map(&self, xyz: &Vector3<f64>) -> Vector3<f64> {
        self.r * xyz + self.t
    }

    /// `inverse()`.
    pub fn inverse(&self) -> SE3Quat {
        let ri = self.r.conjugate();
        SE3Quat {
            r: ri,
            t: ri * (-self.t),
        }
    }

    /// Composition `self * other`.
    pub fn mul(&self, other: &SE3Quat) -> SE3Quat {
        let mut result = SE3Quat {
            r: self.r * other.r,
            t: self.t + self.r * other.t,
        };
        result.normalize_rotation();
        result
    }

    /// `SE3Quat::exp(update)` with `update = [ω, υ]` (`se3quat.h`).
    pub fn exp(update: &Vector6<f64>) -> SE3Quat {
        let omega = Vector3::new(update[0], update[1], update[2]);
        let upsilon = Vector3::new(update[3], update[4], update[5]);

        let theta = omega.norm();
        let omega_hat = skew(&omega);

        let (rot, v): (Matrix3<f64>, Matrix3<f64>);
        if theta < 0.00001 {
            let r = Matrix3::identity() + omega_hat + omega_hat * omega_hat;
            rot = r;
            v = r;
        } else {
            let omega2 = omega_hat * omega_hat;
            rot = Matrix3::identity()
                + (theta.sin() / theta) * omega_hat
                + ((1.0 - theta.cos()) / (theta * theta)) * omega2;
            v = Matrix3::identity()
                + ((1.0 - theta.cos()) / (theta * theta)) * omega_hat
                + ((theta - theta.sin()) / theta.powi(3)) * omega2;
        }
        SE3Quat::from_rt(&rot, v * upsilon)
    }

    /// Build an [`SE3Quat`] from a single-precision [`Isometry3`] (e.g. a frame
    /// pose `Tcw`), casting to `f64`.
    pub fn from_isometry_f32(t: &Isometry3<f32>) -> SE3Quat {
        SE3Quat::new(t.rotation.cast::<f64>(), t.translation.vector.cast::<f64>())
    }

    /// Convert back to a single-precision [`Isometry3`].
    pub fn to_isometry_f32(&self) -> Isometry3<f32> {
        Isometry3::from_parts(
            Translation3::from(self.t.cast::<f32>()),
            self.r.cast::<f32>(),
        )
    }

    /// `log()` -> `[ω, υ]` (`se3quat.h`).
    pub fn log(&self) -> Vector6<f64> {
        let r = self.rotation_matrix();
        let d = 0.5 * (r[(0, 0)] + r[(1, 1)] + r[(2, 2)] - 1.0);
        let dr = delta_r(&r);

        let (omega, v_inv): (Vector3<f64>, Matrix3<f64>);
        if d > 0.99999 {
            omega = 0.5 * dr;
            let omega_hat = skew(&omega);
            v_inv = Matrix3::identity() - 0.5 * omega_hat + (1.0 / 12.0) * (omega_hat * omega_hat);
        } else {
            let theta = d.acos();
            omega = theta / (2.0 * (1.0 - d * d).sqrt()) * dr;
            let omega_hat = skew(&omega);
            v_inv = Matrix3::identity() - 0.5 * omega_hat
                + (1.0 - theta / (2.0 * (theta / 2.0).tan())) / (theta * theta)
                    * (omega_hat * omega_hat);
        }
        let upsilon = v_inv * self.t;
        Vector6::new(
            omega[0], omega[1], omega[2], upsilon[0], upsilon[1], upsilon[2],
        )
    }
}

/// `Sim3` — a 3D similarity transform (rotation, translation, scale), mirroring
/// `g2o::Sim3` (`Thirdparty/g2o/g2o/types/sim3.h`).
///
/// `map(xyz) = s·(R·xyz) + t`. The Lie-algebra 7-vector is `[ω(0..3), υ(3..6), σ]`.
#[derive(Clone, Copy, Debug)]
pub struct Sim3 {
    r: UnitQuaternion<f64>,
    t: Vector3<f64>,
    s: f64,
}

impl Default for Sim3 {
    fn default() -> Self {
        Sim3 {
            r: UnitQuaternion::identity(),
            t: Vector3::zeros(),
            s: 1.0,
        }
    }
}

impl Sim3 {
    pub fn identity() -> Self {
        Sim3::default()
    }

    pub fn new(r: UnitQuaternion<f64>, t: Vector3<f64>, s: f64) -> Self {
        Sim3 { r, t, s }
    }

    pub fn from_rt(r: &Matrix3<f64>, t: Vector3<f64>, s: f64) -> Self {
        Sim3 {
            r: UnitQuaternion::from_matrix(r),
            t,
            s,
        }
    }

    pub fn rotation(&self) -> UnitQuaternion<f64> {
        self.r
    }
    pub fn rotation_matrix(&self) -> Matrix3<f64> {
        self.r.to_rotation_matrix().into_inner()
    }
    pub fn translation(&self) -> Vector3<f64> {
        self.t
    }
    pub fn scale(&self) -> f64 {
        self.s
    }

    /// `map(xyz) = s·R·xyz + t`.
    #[inline]
    pub fn map(&self, xyz: &Vector3<f64>) -> Vector3<f64> {
        self.s * (self.r * xyz) + self.t
    }

    /// `inverse()`.
    pub fn inverse(&self) -> Sim3 {
        let ri = self.r.conjugate();
        Sim3 {
            r: ri,
            t: ri * ((-1.0 / self.s) * self.t),
            s: 1.0 / self.s,
        }
    }

    /// Composition `self * other`.
    pub fn mul(&self, other: &Sim3) -> Sim3 {
        Sim3 {
            r: self.r * other.r,
            t: self.s * (self.r * other.t) + self.t,
            s: self.s * other.s,
        }
    }

    /// Exponential map from a 7-vector update (`g2o::Sim3(const Vector7d&)`).
    pub fn from_update(update: &Vector7) -> Sim3 {
        let omega = Vector3::new(update[0], update[1], update[2]);
        let upsilon = Vector3::new(update[3], update[4], update[5]);
        let sigma = update[6];
        let theta = omega.norm();
        let omega_hat = skew(&omega);
        let s = sigma.exp();
        let omega2 = omega_hat * omega_hat;
        let i = Matrix3::identity();
        let eps = 0.00001;
        let (a, b, c, rot): (f64, f64, f64, Matrix3<f64>);
        if sigma.abs() < eps {
            c = 1.0;
            if theta < eps {
                a = 0.5;
                b = 1.0 / 6.0;
                rot = i + omega_hat + omega2;
            } else {
                let theta2 = theta * theta;
                a = (1.0 - theta.cos()) / theta2;
                b = (theta - theta.sin()) / (theta2 * theta);
                rot = i
                    + (theta.sin() / theta) * omega_hat
                    + ((1.0 - theta.cos()) / (theta * theta)) * omega2;
            }
        } else {
            c = (s - 1.0) / sigma;
            if theta < eps {
                let sigma2 = sigma * sigma;
                a = ((sigma - 1.0) * s + 1.0) / sigma2;
                b = ((0.5 * sigma2 - sigma + 1.0) * s) / (sigma2 * sigma);
                rot = i + omega_hat + omega2;
            } else {
                rot = i
                    + (theta.sin() / theta) * omega_hat
                    + ((1.0 - theta.cos()) / (theta * theta)) * omega2;
                let aa = s * theta.sin();
                let bb = s * theta.cos();
                let theta2 = theta * theta;
                let sigma2 = sigma * sigma;
                let cc = theta2 + sigma2;
                a = (aa * sigma + (1.0 - bb) * theta) / (theta * cc);
                b = (c - ((bb - 1.0) * sigma + aa * theta) / cc) / theta2;
            }
        }
        let w = a * omega_hat + b * omega2 + c * i;
        Sim3::from_rt(&rot, w * upsilon, s)
    }

    /// Logarithm map -> 7-vector `[ω, υ, σ]` (`g2o::Sim3::log`).
    pub fn log(&self) -> Vector7 {
        let sigma = self.s.ln();
        let r = self.rotation_matrix();
        let d = 0.5 * (r[(0, 0)] + r[(1, 1)] + r[(2, 2)] - 1.0);
        let eps = 0.00001;
        let i = Matrix3::identity();
        let (a, b, c, omega): (f64, f64, f64, Vector3<f64>);
        let omega_hat;
        if sigma.abs() < eps {
            c = 1.0;
            if d > 1.0 - eps {
                omega = 0.5 * delta_r(&r);
                omega_hat = skew(&omega);
                a = 0.5;
                b = 1.0 / 6.0;
            } else {
                let theta = d.acos();
                let theta2 = theta * theta;
                omega = theta / (2.0 * (1.0 - d * d).sqrt()) * delta_r(&r);
                omega_hat = skew(&omega);
                a = (1.0 - theta.cos()) / theta2;
                b = (theta - theta.sin()) / (theta2 * theta);
            }
        } else {
            c = (self.s - 1.0) / sigma;
            if d > 1.0 - eps {
                let sigma2 = sigma * sigma;
                omega = 0.5 * delta_r(&r);
                omega_hat = skew(&omega);
                a = ((sigma - 1.0) * self.s + 1.0) / sigma2;
                b = ((0.5 * sigma2 - sigma + 1.0) * self.s) / (sigma2 * sigma);
            } else {
                let theta = d.acos();
                omega = theta / (2.0 * (1.0 - d * d).sqrt()) * delta_r(&r);
                omega_hat = skew(&omega);
                let theta2 = theta * theta;
                let aa = self.s * theta.sin();
                let bb = self.s * theta.cos();
                let cc = theta2 + sigma * sigma;
                a = (aa * sigma + (1.0 - bb) * theta) / (theta * cc);
                b = (c - ((bb - 1.0) * sigma + aa * theta) / cc) / theta2;
            }
        }
        let w = a * omega_hat + b * (omega_hat * omega_hat) + c * i;
        let upsilon = w.lu().solve(&self.t).expect("Sim3::log W solve");
        Vector7::from_column_slice(&[
            omega[0], omega[1], omega[2], upsilon[0], upsilon[1], upsilon[2], sigma,
        ])
    }
}

// ---------------------------------------------------------------------------
// SO(3) helpers (f64) — ports of ORB_SLAM3 G2oTypes.cc
// ---------------------------------------------------------------------------

/// `ExpSO3` — exponential map of a rotation vector (`G2oTypes.cc`).
pub fn exp_so3(v: &Vector3<f64>) -> Matrix3<f64> {
    let d2 = v.dot(v);
    let d = d2.sqrt();
    let w = skew(v);
    let r = if d < 1e-5 {
        Matrix3::identity() + w + 0.5 * w * w
    } else {
        Matrix3::identity() + w * (d.sin() / d) + w * w * ((1.0 - d.cos()) / d2)
    };
    normalize_rotation(&r)
}

/// `LogSO3` — logarithm map of a rotation matrix (`G2oTypes.cc`).
pub fn log_so3(r: &Matrix3<f64>) -> Vector3<f64> {
    let tr = r[(0, 0)] + r[(1, 1)] + r[(2, 2)];
    let w = Vector3::new(
        (r[(2, 1)] - r[(1, 2)]) / 2.0,
        (r[(0, 2)] - r[(2, 0)]) / 2.0,
        (r[(1, 0)] - r[(0, 1)]) / 2.0,
    );
    let costheta = (tr - 1.0) * 0.5;
    if costheta > 1.0 || costheta < -1.0 {
        return w;
    }
    let theta = costheta.acos();
    let s = theta.sin();
    if s.abs() < 1e-5 { w } else { theta * w / s }
}

/// `RightJacobianSO3` (`G2oTypes.cc`).
pub fn right_jacobian_so3(v: &Vector3<f64>) -> Matrix3<f64> {
    let d2 = v.dot(v);
    let d = d2.sqrt();
    let w = skew(v);
    if d < 1e-5 {
        Matrix3::identity()
    } else {
        Matrix3::identity() - w * ((1.0 - d.cos()) / d2) + w * w * ((d - d.sin()) / (d2 * d))
    }
}

/// `InverseRightJacobianSO3` (`G2oTypes.cc`).
pub fn inverse_right_jacobian_so3(v: &Vector3<f64>) -> Matrix3<f64> {
    let d2 = v.dot(v);
    let d = d2.sqrt();
    let w = skew(v);
    if d < 1e-5 {
        Matrix3::identity()
    } else {
        Matrix3::identity() + w / 2.0 + w * w * (1.0 / d2 - (1.0 + d.cos()) / (2.0 * d * d.sin()))
    }
}

/// `NormalizeRotation` — SVD orthonormalization `U·Vᵀ` (`G2oTypes.h`).
pub fn normalize_rotation(r: &Matrix3<f64>) -> Matrix3<f64> {
    let svd = r.svd(true, true);
    let u = svd.u.unwrap();
    let vt = svd.v_t.unwrap();
    u * vt
}

// ---------------------------------------------------------------------------
// Robust kernel
// ---------------------------------------------------------------------------

/// Huber robust kernel (`g2o::RobustKernelHuber`).
///
/// `robustify(e²)` returns `[ρ, ρ', ρ'']` for squared error `e²`.
#[derive(Clone, Copy, Debug)]
pub struct RobustKernelHuber {
    delta: f64,
    dsqr: f64,
}

impl RobustKernelHuber {
    pub fn new(delta: f64) -> Self {
        RobustKernelHuber {
            delta,
            dsqr: delta * delta,
        }
    }

    /// `robustify` (`robust_kernel_impl.cpp`).
    #[inline]
    pub fn robustify(&self, e: f64) -> [f64; 3] {
        if e <= self.dsqr {
            [e, 1.0, 0.0]
        } else {
            let sqrte = e.sqrt();
            let rho1 = self.delta / sqrte;
            [2.0 * sqrte * self.delta - self.dsqr, rho1, -0.5 * rho1 / e]
        }
    }
}

// ---------------------------------------------------------------------------
// Vertex / Edge traits
// ---------------------------------------------------------------------------

/// A graph vertex (optimization variable). Mirrors `g2o::OptimizableGraph::Vertex`
/// plus `BaseVertex` state (estimate + backup stack).
pub trait Vertex: Any {
    /// Minimal (tangent-space) dimension.
    fn dim(&self) -> usize;
    /// Apply an increment: `estimate = oplus(estimate, delta)`.
    fn oplus(&mut self, delta: &[f64]);
    /// Backup the current estimate (LM `push`).
    fn push(&mut self);
    /// Restore the last backed-up estimate (LM `pop`).
    fn pop(&mut self);
    /// Discard the last backup without restoring (LM `discardTop`).
    fn discard_top(&mut self);
    /// Whether this vertex is held fixed.
    fn fixed(&self) -> bool;
    fn set_fixed(&mut self, fixed: bool);
    /// Index into the assembled Hessian (`-1` if not active).
    fn hessian_index(&self) -> i32;
    fn set_hessian_index(&mut self, idx: i32);
    /// Whether the vertex is marginalized (landmark) — affects index ordering.
    fn marginalized(&self) -> bool {
        false
    }
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Per-edge linearization result: error, information matrix, and the Jacobian
/// block w.r.t. each of the edge's vertices (in `vertices()` order).
pub struct EdgeLinearization {
    /// Error vector `e` (length = edge dim).
    pub error: DVector<f64>,
    /// Information matrix `Ω` (edge dim × edge dim).
    pub information: DMatrix<f64>,
    /// Jacobian blocks `J_k` (edge dim × vertex_k dim), one per vertex.
    pub jacobians: Vec<DMatrix<f64>>,
}

/// A graph edge (cost term). Mirrors `g2o::OptimizableGraph::Edge`.
pub trait Edge {
    /// Hessian indices / vertex slots this edge connects, in a fixed order.
    fn vertices(&self) -> &[usize];
    /// Error dimension.
    fn dim(&self) -> usize;
    /// Activation level (g2o `level()`); only edges at the active level are used.
    fn level(&self) -> i32;
    fn set_level(&mut self, level: i32);
    /// Huber delta if a robust kernel is set.
    fn robust_kernel(&self) -> Option<f64>;
    fn set_robust_kernel(&mut self, delta: Option<f64>);
    /// Recompute and cache the error from current vertex estimates.
    fn compute_error(&mut self, vertices: &[&dyn Vertex]);
    /// Cached `χ² = eᵀ Ω e`.
    fn chi2(&self) -> f64;
    /// Compute error + Jacobians + information for system assembly.
    fn linearize(&mut self, vertices: &[&dyn Vertex]) -> EdgeLinearization;
    /// `isDepthPositive()` — whether the observed point is in front of the
    /// camera. Defaults to `true` for edges without a depth notion.
    fn depth_positive(&self, _vertices: &[&dyn Vertex]) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// SparseOptimizer + Levenberg–Marquardt
// ---------------------------------------------------------------------------

/// A sparse nonlinear least-squares optimizer with a Levenberg–Marquardt
/// trust-region solver, mirroring `g2o::SparseOptimizer` driven by
/// `g2o::OptimizationAlgorithmLevenberg`.
///
/// Vertices and edges are owned here; edges reference vertices by the slot index
/// returned from [`SparseOptimizer::add_vertex`].
pub struct SparseOptimizer {
    vertices: Vec<Box<dyn Vertex>>,
    edges: Vec<Box<dyn Edge>>,
    /// Indices into `edges` that are active at the current level.
    active_edges: Vec<usize>,
    /// Indices into `vertices`, in Hessian (block) order.
    iv_map: Vec<usize>,
    /// Hessian offset (scalar) for each entry of `iv_map`.
    offsets: Vec<usize>,
    /// Total active dimension.
    n: usize,
    /// Active dimension of the non-marginalized (pose) block.
    n_p: usize,
    /// LM tuning (matches g2o defaults).
    tau: f64,
    good_step_upper: f64,
    good_step_lower: f64,
    max_trials_after_failure: i32,
    /// External abort flag (g2o `setForceStopFlag`).
    force_stop: Option<Arc<AtomicBool>>,
    /// User-specified initial λ (g2o `setUserLambdaInit`); `<= 0` means auto.
    user_lambda_init: f64,
    /// Optimization algorithm: Levenberg–Marquardt (default) or Gauss–Newton.
    gauss_newton: bool,
}

impl Default for SparseOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseOptimizer {
    pub fn new() -> Self {
        SparseOptimizer {
            vertices: Vec::new(),
            edges: Vec::new(),
            active_edges: Vec::new(),
            iv_map: Vec::new(),
            offsets: Vec::new(),
            n: 0,
            n_p: 0,
            tau: 1e-5,
            good_step_upper: 2.0 / 3.0,
            good_step_lower: 1.0 / 3.0,
            max_trials_after_failure: 10,
            force_stop: None,
            user_lambda_init: 0.0,
            gauss_newton: false,
        }
    }

    /// Use the Gauss–Newton algorithm (`OptimizationAlgorithmGaussNewton`)
    /// instead of Levenberg–Marquardt. ORB-SLAM3 uses this for the
    /// gravity/scale `InertialOptimization` routines.
    pub fn set_gauss_newton(&mut self, on: bool) {
        self.gauss_newton = on;
    }

    /// Override the initial LM damping λ (`setUserLambdaInit`). ORB-SLAM3 uses
    /// `100.0` for inertial local BA.
    pub fn set_user_lambda_init(&mut self, lambda: f64) {
        self.user_lambda_init = lambda;
    }

    /// Install an external abort flag (`setForceStopFlag`). When set to `true`
    /// the optimizer stops at the next opportunity.
    pub fn set_force_stop_flag(&mut self, flag: Arc<AtomicBool>) {
        self.force_stop = Some(flag);
    }

    /// `terminate()`: whether an external stop has been requested.
    fn terminate(&self) -> bool {
        self.force_stop
            .as_ref()
            .is_some_and(|f| f.load(Ordering::SeqCst))
    }

    /// Add a vertex, returning its slot index (used by edges to reference it).
    pub fn add_vertex(&mut self, v: Box<dyn Vertex>) -> usize {
        self.vertices.push(v);
        self.vertices.len() - 1
    }

    /// Add an edge, returning its index.
    pub fn add_edge(&mut self, e: Box<dyn Edge>) -> usize {
        self.edges.push(e);
        self.edges.len() - 1
    }

    /// Immutable access to a vertex (downcast with `as_any`).
    pub fn vertex(&self, idx: usize) -> &dyn Vertex {
        &*self.vertices[idx]
    }

    /// Mutable access to a vertex.
    pub fn vertex_mut(&mut self, idx: usize) -> &mut dyn Vertex {
        &mut *self.vertices[idx]
    }

    /// Fix/unfix a vertex by slot index.
    pub fn set_vertex_fixed(&mut self, idx: usize, fixed: bool) {
        self.vertices[idx].set_fixed(fixed);
    }

    /// Immutable access to an edge.
    pub fn edge(&self, idx: usize) -> &dyn Edge {
        &*self.edges[idx]
    }

    /// Mutable access to an edge.
    pub fn edge_mut(&mut self, idx: usize) -> &mut dyn Edge {
        &mut *self.edges[idx]
    }

    pub fn num_active_edges(&self) -> usize {
        self.active_edges.len()
    }

    /// Recompute the cached error of every active edge (`computeActiveErrors`).
    pub fn compute_active_errors(&mut self) {
        let edges = &mut self.edges;
        let vertices = &self.vertices;
        for &ei in &self.active_edges {
            let vidx = edges[ei].vertices().to_vec();
            let refs: Vec<&dyn Vertex> = vidx
                .iter()
                .map(|&vi| &*vertices[vi] as &dyn Vertex)
                .collect();
            edges[ei].compute_error(&refs);
        }
    }

    /// `isDepthPositive()` for an edge, evaluated at the current estimates.
    pub fn edge_depth_positive(&self, ei: usize) -> bool {
        let vidx = self.edges[ei].vertices();
        let refs: Vec<&dyn Vertex> = vidx
            .iter()
            .map(|&vi| &*self.vertices[vi] as &dyn Vertex)
            .collect();
        self.edges[ei].depth_positive(&refs)
    }

    /// Linearize a single edge at the current estimates (error + Jacobians +
    /// information). Used to assemble marginalization Hessians.
    pub fn edge_linearization(&mut self, ei: usize) -> EdgeLinearization {
        let edges = &mut self.edges;
        let vertices = &self.vertices;
        let vidx = edges[ei].vertices().to_vec();
        let refs: Vec<&dyn Vertex> = vidx
            .iter()
            .map(|&vi| &*vertices[vi] as &dyn Vertex)
            .collect();
        edges[ei].linearize(&refs)
    }

    /// Recompute the cached error of a single edge (`e->computeError()`).
    pub fn compute_edge_error(&mut self, ei: usize) {
        let edges = &mut self.edges;
        let vertices = &self.vertices;
        let vidx = edges[ei].vertices().to_vec();
        let refs: Vec<&dyn Vertex> = vidx
            .iter()
            .map(|&vi| &*vertices[vi] as &dyn Vertex)
            .collect();
        edges[ei].compute_error(&refs);
    }

    /// `activeRobustChi2`: Σ over active edges of ρ(χ²) or χ².
    pub fn active_robust_chi2(&self) -> f64 {
        let mut chi = 0.0;
        for &ei in &self.active_edges {
            let e = &self.edges[ei];
            let c = e.chi2();
            chi += match e.robust_kernel() {
                Some(delta) => RobustKernelHuber::new(delta).robustify(c)[0],
                None => c,
            };
        }
        chi
    }

    /// Build the active-edge set and Hessian index mapping for the given level
    /// (`initializeOptimization`). Vertices are ordered non-marginalized first,
    /// then marginalized, mirroring g2o's `buildIndexMapping`.
    pub fn initialize_optimization(&mut self, level: i32) {
        // Active edges: matching level and not all-fixed.
        self.active_edges.clear();
        for (ei, e) in self.edges.iter().enumerate() {
            if e.level() != level {
                continue;
            }
            let all_fixed = e.vertices().iter().all(|&vi| self.vertices[vi].fixed());
            if all_fixed {
                continue;
            }
            self.active_edges.push(ei);
        }

        // Active vertices: those touched by an active edge and not fixed.
        let mut active_vertex_flags = vec![false; self.vertices.len()];
        for &ei in &self.active_edges {
            for &vi in self.edges[ei].vertices() {
                if !self.vertices[vi].fixed() {
                    active_vertex_flags[vi] = true;
                }
            }
        }

        // Reset all hessian indices.
        for v in self.vertices.iter_mut() {
            v.set_hessian_index(-1);
        }

        // Order: non-marginalized (k=0) then marginalized (k=1).
        self.iv_map.clear();
        self.offsets.clear();
        let mut offset = 0usize;
        for k in 0..2 {
            for vi in 0..self.vertices.len() {
                if !active_vertex_flags[vi] {
                    continue;
                }
                if (self.vertices[vi].marginalized() as usize) != k {
                    continue;
                }
                let pos = self.iv_map.len() as i32;
                self.vertices[vi].set_hessian_index(pos);
                self.iv_map.push(vi);
                self.offsets.push(offset);
                offset += self.vertices[vi].dim();
            }
            if k == 0 {
                // End of the non-marginalized (pose) block.
                self.n_p = offset;
            }
        }
        self.n = offset;
    }

    /// Assemble the block-structured normal equations over active edges,
    /// applying the Huber robust weighting `ρ'·Ω` exactly as g2o. Marginalized
    /// (landmark) vertices are kept as separate diagonal blocks so they can be
    /// eliminated by Schur complement in [`BlockSystem::solve`].
    fn build_block_system(&mut self) -> BlockSystem {
        let n_p = self.n_p;
        let mut hpp = DMatrix::<f64>::zeros(n_p, n_p);
        let mut bp = DVector::<f64>::zeros(n_p);

        // One landmark block per marginalized active vertex, addressed by its
        // Hessian position. `lm_index[pos]` maps the iv_map position to an index
        // into `landmarks` (or usize::MAX for pose vertices).
        let mut landmarks: Vec<LandmarkBlock> = Vec::new();
        let mut lm_index = vec![usize::MAX; self.iv_map.len()];
        for (pos, &vi) in self.iv_map.iter().enumerate() {
            if self.vertices[vi].marginalized() {
                lm_index[pos] = landmarks.len();
                let dim = self.vertices[vi].dim();
                landmarks.push(LandmarkBlock {
                    off: self.offsets[pos],
                    dim,
                    hll: DMatrix::zeros(dim, dim),
                    bl: DVector::zeros(dim),
                    conns: Vec::new(),
                });
            }
        }

        let edges = &mut self.edges;
        let vertices = &self.vertices;
        for &ei in &self.active_edges {
            let vidx = edges[ei].vertices().to_vec();
            let refs: Vec<&dyn Vertex> = vidx
                .iter()
                .map(|&vi| &*vertices[vi] as &dyn Vertex)
                .collect();
            let lin = edges[ei].linearize(&refs);
            let chi2 = edges[ei].chi2();

            // Robust (weighted) information.
            let weighted = match edges[ei].robust_kernel() {
                Some(delta) => {
                    let rho = RobustKernelHuber::new(delta).robustify(chi2);
                    rho[1] * &lin.information
                }
                None => lin.information.clone(),
            };

            // Per-vertex: (pos, off, marginalized, JᵀW). Skip fixed/inactive.
            struct Blk {
                pos: usize,
                off: usize,
                marg: bool,
                jt_w: DMatrix<f64>,
            }
            let mut blocks: Vec<Option<Blk>> = Vec::with_capacity(vidx.len());
            for (k, &vi) in vidx.iter().enumerate() {
                let v = &vertices[vi];
                let hidx = v.hessian_index();
                if v.fixed() || hidx < 0 {
                    blocks.push(None);
                    continue;
                }
                let pos = hidx as usize;
                blocks.push(Some(Blk {
                    pos,
                    off: self.offsets[pos],
                    marg: v.marginalized(),
                    jt_w: lin.jacobians[k].transpose() * &weighted,
                }));
            }

            for blk in blocks.iter().flatten() {
                // b contribution: b_k -= JᵀW · e
                let bk = &blk.jt_w * &lin.error;
                if blk.marg {
                    let lm = &mut landmarks[lm_index[blk.pos]];
                    for r in 0..bk.nrows() {
                        lm.bl[r] -= bk[r];
                    }
                } else {
                    for r in 0..bk.nrows() {
                        bp[blk.off + r] -= bk[r];
                    }
                }
            }

            // H contribution: H_kl += JᵀW · J_l
            for k in 0..blocks.len() {
                let Some(bk) = &blocks[k] else { continue };
                for l in 0..blocks.len() {
                    let Some(bl) = &blocks[l] else { continue };
                    let hkl = &bk.jt_w * &lin.jacobians[l]; // (dk × dl)
                    match (bk.marg, bl.marg) {
                        (false, false) => {
                            add_block(&mut hpp, bk.off, bl.off, &hkl);
                        }
                        (true, true) => {
                            // Landmark-landmark: only the diagonal block (k==l)
                            // is supported by the Schur path.
                            assert!(bk.pos == bl.pos, "unsupported landmark-landmark coupling");
                            let lm = &mut landmarks[lm_index[bk.pos]];
                            add_block(&mut lm.hll, 0, 0, &hkl);
                        }
                        (false, true) => {
                            // Pose (k) × landmark (l): store Hpl on the landmark.
                            let pose_off = bk.off;
                            let lm = &mut landmarks[lm_index[bl.pos]];
                            lm.conns.push((pose_off, hkl));
                        }
                        (true, false) => {
                            // Landmark (k) × pose (l): redundant transpose of the
                            // (false,true) case; skip to avoid double counting.
                        }
                    }
                }
            }
        }

        BlockSystem {
            n: self.n,
            n_p,
            hpp,
            bp,
            landmarks,
        }
    }

    /// `computeScale`: Σ_j x_j (λ x_j + b_j).
    fn compute_scale(x: &DVector<f64>, b: &DVector<f64>, lambda: f64) -> f64 {
        let mut scale = 0.0;
        for j in 0..x.len() {
            scale += x[j] * (lambda * x[j] + b[j]);
        }
        scale
    }

    fn push_active(&mut self) {
        for &vi in &self.iv_map {
            self.vertices[vi].push();
        }
    }
    fn pop_active(&mut self) {
        for &vi in &self.iv_map {
            self.vertices[vi].pop();
        }
    }
    fn discard_top_active(&mut self) {
        for &vi in &self.iv_map {
            self.vertices[vi].discard_top();
        }
    }

    fn update(&mut self, x: &DVector<f64>) {
        for (pos, &vi) in self.iv_map.iter().enumerate() {
            let off = self.offsets[pos];
            let dim = self.vertices[vi].dim();
            self.vertices[vi].oplus(&x.as_slice()[off..off + dim]);
        }
    }

    /// Gauss–Newton iterations (`OptimizationAlgorithmGaussNewton`): each step
    /// rebuilds the system and applies the undamped solution, with no trust
    /// region / rollback.
    fn optimize_gauss_newton(&mut self, iterations: i32) -> i32 {
        let mut cj = 0;
        for _ in 0..iterations {
            if self.terminate() {
                break;
            }
            self.compute_active_errors();
            let system = self.build_block_system();
            if let Some(x) = system.solve(0.0) {
                self.update(&x);
            } else {
                break;
            }
            cj += 1;
        }
        cj
    }

    /// Run up to `iterations` LM iterations. Returns the number of iterations
    /// actually executed. Mirrors `SparseOptimizer::optimize` +
    /// `OptimizationAlgorithmLevenberg::solve`.
    pub fn optimize(&mut self, iterations: i32) -> i32 {
        if self.iv_map.is_empty() {
            return -1;
        }

        if self.gauss_newton {
            return self.optimize_gauss_newton(iterations);
        }

        let mut lambda = -1.0f64;
        let mut ni = 2.0f64;
        let mut n_bad = 0i32;
        let mut cj_iterations = 0;

        for iter in 0..iterations {
            if self.terminate() {
                break;
            }
            // --- solve(iter) ---
            self.compute_active_errors();
            let mut current_chi = self.active_robust_chi2();
            let ini_chi = current_chi;

            let system = self.build_block_system();
            let full_b = system.full_b();

            if iter == 0 {
                lambda = if self.user_lambda_init > 0.0 {
                    self.user_lambda_init
                } else {
                    self.tau * system.max_diagonal()
                };
                ni = 2.0;
                n_bad = 0;
            }

            let mut rho = 0.0f64;
            let mut qmax = 0i32;
            loop {
                self.push_active();
                let solved = system.solve(lambda);
                let solve_ok = solved.is_some();
                let x = solved.unwrap_or_else(|| DVector::zeros(self.n));
                self.update(&x);

                self.compute_active_errors();
                let temp_chi = if solve_ok {
                    self.active_robust_chi2()
                } else {
                    f64::MAX
                };

                rho = current_chi - temp_chi;
                let scale = Self::compute_scale(&x, &full_b, lambda) + 1e-3;
                rho /= scale;

                if rho > 0.0 && temp_chi.is_finite() {
                    let alpha = 1.0 - (2.0 * rho - 1.0).powi(3);
                    let alpha = alpha.min(self.good_step_upper);
                    let scale_factor = alpha.max(self.good_step_lower);
                    lambda *= scale_factor;
                    ni = 2.0;
                    current_chi = temp_chi;
                    self.discard_top_active();
                } else {
                    lambda *= ni;
                    ni *= 2.0;
                    self.pop_active();
                }
                qmax += 1;
                if !(rho < 0.0 && qmax < self.max_trials_after_failure && !self.terminate()) {
                    break;
                }
            }
            cj_iterations += 1;

            if qmax == self.max_trials_after_failure || rho == 0.0 {
                break;
            }

            // Raul's stop criterion.
            if (ini_chi - current_chi) * 1e3 < ini_chi {
                n_bad += 1;
            } else {
                n_bad = 0;
            }
            if n_bad >= 3 {
                break;
            }
        }
        cj_iterations
    }
}

// ---------------------------------------------------------------------------
// Block-structured normal equations with Schur-complement solve
// ---------------------------------------------------------------------------

/// Accumulate a small block into `dst` at `(row0, col0)`.
fn add_block(dst: &mut DMatrix<f64>, row0: usize, col0: usize, blk: &DMatrix<f64>) {
    for r in 0..blk.nrows() {
        for c in 0..blk.ncols() {
            dst[(row0 + r, col0 + c)] += blk[(r, c)];
        }
    }
}

/// One marginalized (landmark) diagonal block plus its couplings to pose blocks.
struct LandmarkBlock {
    /// Offset of this landmark in the full solution vector.
    off: usize,
    dim: usize,
    /// `H_ll` (without λ damping).
    hll: DMatrix<f64>,
    /// `b_l`.
    bl: DVector<f64>,
    /// Couplings `(pose_offset, H_pl)` with `H_pl` of shape `pose_dim × dim`.
    conns: Vec<(usize, DMatrix<f64>)>,
}

/// Block form of `H x = b`: a dense pose block `H_pp` plus per-landmark diagonal
/// blocks. Mirrors g2o's `BlockSolver` (e.g. `BlockSolver_6_3`): landmarks are
/// eliminated by Schur complement and recovered by back-substitution. The
/// numerical result is identical to solving the full dense system.
struct BlockSystem {
    n: usize,
    n_p: usize,
    hpp: DMatrix<f64>,
    bp: DVector<f64>,
    landmarks: Vec<LandmarkBlock>,
}

impl BlockSystem {
    /// Full right-hand side `b` (poses first, then landmarks), for `computeScale`.
    fn full_b(&self) -> DVector<f64> {
        let mut b = DVector::zeros(self.n);
        b.rows_mut(0, self.n_p).copy_from(&self.bp);
        for lm in &self.landmarks {
            b.rows_mut(lm.off, lm.dim).copy_from(&lm.bl);
        }
        b
    }

    /// `τ`-free max absolute diagonal across pose and landmark blocks
    /// (used by `computeLambdaInit`).
    fn max_diagonal(&self) -> f64 {
        let mut m = 0.0f64;
        for i in 0..self.n_p {
            m = m.max(self.hpp[(i, i)].abs());
        }
        for lm in &self.landmarks {
            for d in 0..lm.dim {
                m = m.max(lm.hll[(d, d)].abs());
            }
        }
        m
    }

    /// Solve `(H + λ I) x = b` via Schur complement on the landmark blocks.
    fn solve(&self, lambda: f64) -> Option<DVector<f64>> {
        // Reduced (camera/pose) system, starting from H_pp + λI.
        let mut hpp_r = self.hpp.clone();
        for i in 0..self.n_p {
            hpp_r[(i, i)] += lambda;
        }
        let mut bp_r = self.bp.clone();

        // Precompute each landmark's damped inverse and subtract its Schur term.
        let mut hll_invs: Vec<DMatrix<f64>> = Vec::with_capacity(self.landmarks.len());
        for lm in &self.landmarks {
            let mut hll = lm.hll.clone();
            for d in 0..lm.dim {
                hll[(d, d)] += lambda;
            }
            let hll_inv = hll.try_inverse()?;

            for (off_a, m_a) in &lm.conns {
                let t_a = m_a * &hll_inv; // pose_dim × dim
                // bp_r[a] -= H_pl H_ll⁻¹ b_l
                let cb = &t_a * &lm.bl;
                for r in 0..cb.nrows() {
                    bp_r[off_a + r] -= cb[r];
                }
                // H_pp'[a,b] -= H_pl H_ll⁻¹ H_plᵀ
                for (off_b, m_b) in &lm.conns {
                    let contrib = &t_a * m_b.transpose();
                    for r in 0..contrib.nrows() {
                        for c in 0..contrib.ncols() {
                            hpp_r[(off_a + r, off_b + c)] -= contrib[(r, c)];
                        }
                    }
                }
            }
            hll_invs.push(hll_inv);
        }

        // Solve the reduced pose system.
        let dp = {
            if let Some(chol) = hpp_r.clone().cholesky() {
                chol.solve(&bp_r)
            } else {
                hpp_r.lu().solve(&bp_r)?
            }
        };

        // Assemble full solution: poses then back-substituted landmarks.
        let mut x = DVector::zeros(self.n);
        x.rows_mut(0, self.n_p).copy_from(&dp);
        for (li, lm) in self.landmarks.iter().enumerate() {
            let mut rhs = lm.bl.clone();
            for (off_a, m_a) in &lm.conns {
                let dp_a = dp.rows(*off_a, m_a.nrows());
                rhs -= m_a.transpose() * dp_a;
            }
            let dl = &hll_invs[li] * rhs;
            x.rows_mut(lm.off, lm.dim).copy_from(&dl);
        }
        Some(x)
    }
}
