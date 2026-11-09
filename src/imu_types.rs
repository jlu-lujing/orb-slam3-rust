use nalgebra::{Isometry3, Matrix3, Matrix6, Matrix6x1, Rotation3, SMatrix, Vector3, Vector6};
use serde::{Deserialize, Serialize};

const EPS: f32 = 1e-4;

fn normalize_rotation(r: Matrix3<f32>) -> Matrix3<f32> {
    let svd = r.svd(true, true);
    let u = svd.u.expect("SVD U should be computed");
    let v_t = svd.v_t.expect("SVD V^T should be computed");
    u * v_t
}

pub fn right_jacobian_so3(v: Vector3<f32>) -> Matrix3<f32> {
    let d2 = v.norm_squared();
    let d = d2.sqrt();
    let w = v.cross_matrix();
    if d < EPS {
        Matrix3::identity()
    } else {
        Matrix3::identity() - w * ((1.0 - d.cos()) / d2) + w * w * ((d - d.sin()) / (d2 * d))
    }
}

pub fn right_jacobian_so3_xyz(x: f32, y: f32, z: f32) -> Matrix3<f32> {
    right_jacobian_so3(Vector3::new(x, y, z))
}

pub fn inverse_right_jacobian_so3(v: Vector3<f32>) -> Matrix3<f32> {
    let d2 = v.norm_squared();
    let d = d2.sqrt();
    let w = v.cross_matrix();
    if d < EPS {
        Matrix3::identity()
    } else {
        Matrix3::identity()
            + w * 0.5_f32
            + w * w * (1.0 / d2 - (1.0 + d.cos()) / (2.0 * d * d.sin()))
    }
}

pub fn inverse_right_jacobian_so3_xyz(x: f32, y: f32, z: f32) -> Matrix3<f32> {
    inverse_right_jacobian_so3(Vector3::new(x, y, z))
}

// IMU measurement (gyro, accelerometer and timestamp)
pub struct Point {
    pub a: Vector3<f32>,
    pub w: Vector3<f32>,
    pub t: f64,
}
impl Point {
    pub fn new(a: Vector3<f32>, w: Vector3<f32>, t: f64) -> Self {
        Point { a, w, t }
    }
    pub fn from_params(
        acc_x: f32,
        acc_y: f32,
        acc_z: f32,
        ang_vel_x: f32,
        ang_vel_y: f32,
        ang_vel_z: f32,
        t: f64,
    ) -> Self {
        Point {
            a: Vector3::new(acc_x, acc_y, acc_z),
            w: Vector3::new(ang_vel_x, ang_vel_y, ang_vel_z),
            t,
        }
    }
}

// IMU biases (gyro and accelerometer)
#[derive(Debug, Serialize, Deserialize, Default, Clone, Copy)]
pub struct Bias {
    pub bax: f32,
    pub bay: f32,
    pub baz: f32,
    pub bwx: f32,
    pub bwy: f32,
    pub bwz: f32,
}
impl Bias {
    pub fn empty() -> Self {
        Bias {
            bax: 0.0,
            bay: 0.0,
            baz: 0.0,
            bwx: 0.0,
            bwy: 0.0,
            bwz: 0.0,
        }
    }
    pub fn from_params(bax: f32, bay: f32, baz: f32, bwx: f32, bwy: f32, bwz: f32) -> Self {
        Bias {
            bax,
            bay,
            baz,
            bwx,
            bwy,
            bwz,
        }
    }
}

// IMU calibration (Tbc, Tcb, noise)
#[derive(Serialize, Deserialize, Clone)]
pub struct Calib {
    pub tcb: Isometry3<f32>,
    pub tbc: Isometry3<f32>,
    cov: Vector6<f32>,
    cov_walk: Vector6<f32>,
    pub is_set: bool,
}
impl Calib {
    pub fn new() -> Self {
        Calib {
            tcb: Isometry3::identity(),
            tbc: Isometry3::identity(),
            cov: Vector6::zeros(),
            cov_walk: Vector6::zeros(),
            is_set: false,
        }
    }
    pub fn from_params(tbc: Isometry3<f32>, g: f32, a: f32, gw: f32, aw: f32) -> Self {
        let mut calib = Calib::new();
        calib.set(tbc, g, a, gw, aw);
        calib
    }
    pub fn set(&mut self, tbc: Isometry3<f32>, g: f32, a: f32, gw: f32, aw: f32) {
        self.is_set = true;
        self.tbc = tbc;
        self.tcb = self.tbc.inverse();
        let g2 = g * g;
        let a2 = a * a;
        let gw2 = gw * gw;
        let aw2 = aw * aw;
        self.cov = Vector6::new(g2, g2, g2, a2, a2, a2);
        self.cov_walk = Vector6::new(gw2, gw2, gw2, aw2, aw2, aw2);
    }
    pub fn set_cov(&mut self, cov: Vector6<f32>) {
        self.cov = cov;
    }
    pub fn set_cov_walk(&mut self, cov_walk: Vector6<f32>) {
        self.cov_walk = cov_walk;
    }

    pub fn get_cov(&self) -> Matrix6<f32> {
        Matrix6::from_diagonal(&self.cov)
    }
    pub fn get_cov_walk(&self) -> Matrix6<f32> {
        Matrix6::from_diagonal(&self.cov_walk)
    }
}

pub struct IntegratedRotation {
    pub delta_t: f32, // integration time
    pub delta_r: Matrix3<f32>,
    pub right_j: Matrix3<f32>, // right jacobian
}
impl IntegratedRotation {
    pub fn new() -> Self {
        IntegratedRotation {
            delta_t: 0.0,
            delta_r: Matrix3::identity(),
            right_j: Matrix3::identity(),
        }
    }
    pub fn from_params(ang_vel: &Vector3<f32>, imu_bias: &Bias, time: f32) -> Self {
        let x = (ang_vel[0] - imu_bias.bwx) * time;
        let y = (ang_vel[1] - imu_bias.bwy) * time;
        let z = (ang_vel[2] - imu_bias.bwz) * time;

        let d2 = x * x + y * y + z * z;
        let d = d2.sqrt();

        let v = Vector3::new(x, y, z);
        let w = v.cross_matrix();

        let delta_t = 0.0;
        let delta_r;
        let right_j;
        if d < EPS {
            delta_r = Matrix3::identity() + w;
            right_j = Matrix3::identity();
        } else {
            delta_r = Matrix3::identity() + w * (d.sin() / d) + w * w * ((1.0_f32 - d.cos()) / d2);
            right_j = Matrix3::identity() - w * ((1.0_f32 - d.cos()) / d2)
                + w * w * ((d - d.sin()) / (d2 * d));
        }

        IntegratedRotation {
            delta_t,
            delta_r,
            right_j,
        }
    }
}

// Preintegration of IMU measurements
#[derive(Debug, Serialize, Deserialize, Default, Clone)]
struct Integrable {
    a: Vector3<f32>,
    w: Vector3<f32>,
    t: f32,
}
#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct Preintegrated {
    pub dt: f32,
    pub c: SMatrix<f32, 15, 15>,
    pub info: SMatrix<f32, 15, 15>,
    pub nga: Vector6<f32>,
    pub nga_walk: Vector6<f32>,
    pub b: Bias,
    pub dr: Matrix3<f32>,
    pub dv: Vector3<f32>,
    pub dp: Vector3<f32>,
    pub jrg: Matrix3<f32>,
    pub jvg: Matrix3<f32>,
    pub jva: Matrix3<f32>,
    pub jpg: Matrix3<f32>,
    pub jpa: Matrix3<f32>,
    pub avg_a: Vector3<f32>,
    pub avg_w: Vector3<f32>,
    // Updated bias
    bu: Bias,
    // Dif between original and updated bias, this is used to compute the updated values of preintegration
    db: Matrix6x1<f32>,
    measurements: Vec<Integrable>,
}
impl Preintegrated {
    pub fn from_bias_and_calib(b: &Bias, calib: &Calib) -> Self {
        let mut item = Self::default();
        item.nga = calib.cov;
        item.nga_walk = calib.cov_walk;
        item.initialize(b);
        item
    }
    pub fn initialize(&mut self, b: &Bias) {
        self.dr = Matrix3::identity();
        self.dv = Vector3::zeros();
        self.dp = Vector3::zeros();
        self.jrg = Matrix3::zeros();
        self.jvg = Matrix3::zeros();
        self.jva = Matrix3::zeros();
        self.jpg = Matrix3::zeros();
        self.jpa = Matrix3::zeros();
        self.c = SMatrix::zeros();
        self.info = SMatrix::zeros();
        self.db = Matrix6x1::zeros();
        self.b = *b;
        self.bu = *b;
        self.avg_a = Vector3::zeros();
        self.avg_w = Vector3::zeros();
        self.dt = 0.0;
        self.measurements.clear();
    }

    pub fn reintegrate(&mut self) {
        let aux = self.measurements.clone();
        let bu = self.bu; // Bias is Copy
        self.initialize(&bu);
        for m in &aux {
            self.integrate_new_measurement(m.a, m.w, m.t);
        }
    }

    pub fn integrate_new_measurement(&mut self, acc: Vector3<f32>, ang: Vector3<f32>, dt: f32) {
        self.measurements.push(Integrable {
            a: acc,
            w: ang,
            t: dt,
        });

        // Position is updated firstly, as it depends on previously computed velocity and rotations
        // Velocity is updated secondly, as it depends on previously computer rotation
        // Rotation is the last to be updated

        // Matrices to compute covariance
        let mut a: SMatrix<f32, 9, 9> = SMatrix::identity();
        let mut b: SMatrix<f32, 9, 6> = SMatrix::zeros();

        let acc: Vector3<f32> = Vector3::new(
            acc[0] - self.b.bax,
            acc[1] - self.b.bay,
            acc[2] - self.b.baz,
        );
        let acc_w: Vector3<f32> = Vector3::new(
            ang[0] - self.b.bwx,
            ang[1] - self.b.bwy,
            ang[2] - self.b.bwz,
        );

        self.avg_a = (self.dt * self.avg_a + self.dr * acc * dt) / (self.dt + dt);
        self.avg_w = (self.dt * self.avg_w + acc_w * dt) / (self.dt + dt);

        // Update delta position dp and velocity dv (rely on no-updated delta rotation)
        self.dp = self.dp + self.dv * dt + 0.5 * self.dr * acc * dt * dt;
        self.dv = self.dv + self.dr * acc * dt;

        // Compute velocity and position parts of matrices A and B (rely on non-updated delta rotation)
        let w_acc = acc.cross_matrix();

        a.fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&(-self.dr * dt * w_acc));
        a.fixed_view_mut::<3, 3>(6, 0)
            .copy_from(&(-0.5 * self.dr * dt * dt * w_acc));
        a.fixed_view_mut::<3, 3>(6, 3)
            .copy_from(&(Matrix3::identity() * dt));
        b.fixed_view_mut::<3, 3>(3, 3).copy_from(&(self.dr * dt));
        b.fixed_view_mut::<3, 3>(6, 3)
            .copy_from(&(0.5_f32 * self.dr * dt * dt));

        // Update positions and velocity jacobians wrt bias correction
        self.jpa = self.jpa + self.jva * dt - 0.5 * self.dr * dt * dt;
        self.jpg = self.jpg + self.jvg * dt - 0.5 * self.dr * dt * dt * w_acc * self.jrg;
        self.jva = self.jva - self.dr * dt;
        self.jvg = self.jvg - self.dr * dt * w_acc * self.jrg;

        // Update delta rotation
        let dri = IntegratedRotation::from_params(&ang, &self.b, dt);
        self.dr = normalize_rotation(self.dr * dri.delta_r);

        // Compute rotation parts of matrices A and B
        a.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&dri.delta_r.transpose());
        b.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(dri.right_j * dt));

        // Update covariance
        // nga and nga_walk are stored as diagonal vectors; expand to 6×6 diagonal matrices
        // for the matrix products (mirrors Eigen::DiagonalMatrix<float,6> in the C++ original).
        let nga_mat = SMatrix::<f32, 6, 6>::from_diagonal(&self.nga);
        let nga_walk_mat = Matrix6::from_diagonal(&self.nga_walk);
        let c_sub = self.c.fixed_view::<9, 9>(0, 0).into_owned();
        self.c
            .fixed_view_mut::<9, 9>(0, 0)
            .copy_from(&(a * c_sub * a.transpose() + b * nga_mat * b.transpose()));
        let c_sub2 = self.c.fixed_view::<6, 6>(9, 9).into_owned();
        self.c
            .fixed_view_mut::<6, 6>(9, 9)
            .copy_from(&(c_sub2 + nga_walk_mat));

        // Update rotation jacobian wrt bias correction
        self.jrg = dri.delta_r.transpose() * self.jrg - dri.right_j * dt;

        // Total integrated time
        self.dt += dt;
    }

    pub fn merge_previous(&mut self, prev: &Preintegrated) {
        let bu = self.bu; // Bias is Copy
        let aux1 = prev.measurements.clone();
        let aux2 = self.measurements.clone();
        self.initialize(&bu);
        for m in &aux1 {
            self.integrate_new_measurement(m.a, m.w, m.t);
        }
        for m in &aux2 {
            self.integrate_new_measurement(m.a, m.w, m.t);
        }
    }

    pub fn set_new_bias(&mut self, b: &Bias) {
        self.bu = *b;
        self.db[0] = b.bwx - self.b.bwx;
        self.db[1] = b.bwy - self.b.bwy;
        self.db[2] = b.bwz - self.b.bwz;
        self.db[3] = b.bax - self.b.bax;
        self.db[4] = b.bay - self.b.bay;
        self.db[5] = b.baz - self.b.baz;
    }

    pub fn compute_bias_diff(&self, b: &Bias) -> Bias {
        Bias::from_params(
            b.bax - self.b.bax,
            b.bay - self.b.bay,
            b.baz - self.b.baz,
            b.bwx - self.b.bwx,
            b.bwy - self.b.bwy,
            b.bwz - self.b.bwz,
        )
    }

    pub fn get_delta_rotation(&self, b: &Bias) -> Matrix3<f32> {
        let dbg = Vector3::new(b.bwx - self.b.bwx, b.bwy - self.b.bwy, b.bwz - self.b.bwz);
        normalize_rotation(self.dr * Rotation3::new(self.jrg * dbg).into_inner())
    }

    pub fn get_delta_velocity(&self, b: &Bias) -> Vector3<f32> {
        let dbg = Vector3::new(b.bwx - self.b.bwx, b.bwy - self.b.bwy, b.bwz - self.b.bwz);
        let dba = Vector3::new(b.bax - self.b.bax, b.bay - self.b.bay, b.baz - self.b.baz);
        self.dv + self.jvg * dbg + self.jva * dba
    }

    pub fn get_delta_position(&self, b: &Bias) -> Vector3<f32> {
        let dbg = Vector3::new(b.bwx - self.b.bwx, b.bwy - self.b.bwy, b.bwz - self.b.bwz);
        let dba = Vector3::new(b.bax - self.b.bax, b.bay - self.b.bay, b.baz - self.b.baz);
        self.dp + self.jpg * dbg + self.jpa * dba
    }

    pub fn get_updated_delta_rotation(&self) -> Matrix3<f32> {
        let dbg = self.db.fixed_rows::<3>(0);
        normalize_rotation(self.dr * Rotation3::new(self.jrg * dbg).into_inner())
    }

    pub fn get_updated_delta_velocity(&self) -> Vector3<f32> {
        let dbg = self.db.fixed_rows::<3>(0);
        let dba = self.db.fixed_rows::<3>(3);
        self.dv + self.jvg * dbg + self.jva * dba
    }

    pub fn get_updated_delta_position(&self) -> Vector3<f32> {
        let dbg = self.db.fixed_rows::<3>(0);
        let dba = self.db.fixed_rows::<3>(3);
        self.dp + self.jpg * dbg + self.jpa * dba
    }

    pub fn get_original_delta_rotation(&self) -> Matrix3<f32> {
        self.dr
    }

    pub fn get_original_delta_velocity(&self) -> Vector3<f32> {
        self.dv
    }

    pub fn get_original_delta_position(&self) -> Vector3<f32> {
        self.dp
    }

    pub fn get_original_bias(&self) -> Bias {
        self.b
    }

    pub fn get_updated_bias(&self) -> Bias {
        self.bu
    }

    pub fn get_delta_bias(&self) -> Matrix6x1<f32> {
        self.db
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Rotation3, Unit};

    // Helpers
    fn zero_bias() -> Bias {
        Bias::default()
    }
    fn make_calib() -> Calib {
        Calib::from_params(
            Isometry3::identity(),
            1e-3_f32,
            1e-2_f32,
            1e-5_f32,
            1e-4_f32,
        )
    }
    fn is_rotation(r: &Matrix3<f32>, tol: f32) -> bool {
        let i = r.transpose() * r;
        (i - Matrix3::identity()).norm() < tol && (r.determinant() - 1.0_f32).abs() < tol
    }

    // calib
    #[test]
    fn calib_set_stores_tbc_and_computes_tcb_as_inverse() {
        // Non-identity Tbc: rotation 0.1 rad about x, translation (0.05, 0, 0).
        let tbc = Isometry3::new(
            Vector3::new(0.05_f32, 0.0, 0.0),
            Vector3::new(0.1_f32, 0.0, 0.0),
        );
        let c = Calib::from_params(tbc, 1e-3, 1e-2, 1e-5, 1e-4);

        assert!(c.is_set);
        let diff_tbc = (c.tbc.to_homogeneous() - tbc.to_homogeneous()).norm();
        assert!(diff_tbc < 1e-5, "tbc mismatch: {diff_tbc}");
        let diff_tcb = (c.tcb.to_homogeneous() - tbc.inverse().to_homogeneous()).norm();
        assert!(diff_tcb < 1e-5, "tcb mismatch: {diff_tcb}");
    }

    #[test]
    fn calib_set_populates_noise_covariance_diagonals_correctly() {
        let (ng, na, ngw, naw) = (2e-3_f32, 3e-2_f32, 4e-5_f32, 5e-4_f32);
        let c = Calib::from_params(Isometry3::identity(), ng, na, ngw, naw);

        let cov = c.get_cov();
        for i in 0..3 {
            assert!((cov[(i, i)] - ng * ng).abs() < 1e-12, "Cov[{i},{i}] wrong");
        }
        for i in 3..6 {
            assert!((cov[(i, i)] - na * na).abs() < 1e-12, "Cov[{i},{i}] wrong");
        }
        let cov_walk = c.get_cov_walk();
        for i in 0..3 {
            assert!(
                (cov_walk[(i, i)] - ngw * ngw).abs() < 1e-15,
                "CovWalk[{i},{i}] wrong"
            );
        }
        for i in 3..6 {
            assert!(
                (cov_walk[(i, i)] - naw * naw).abs() < 1e-14,
                "CovWalk[{i},{i}] wrong"
            );
        }
    }

    #[test]
    fn calib_clone_duplicates_all_fields() {
        let c1 = make_calib();
        let c2 = c1.clone();

        assert_eq!(c2.is_set, c1.is_set);
        assert!((c2.tbc.to_homogeneous() - c1.tbc.to_homogeneous()).norm() < 1e-6);
        assert!((c2.tcb.to_homogeneous() - c1.tcb.to_homogeneous()).norm() < 1e-6);
        assert!((c2.get_cov().diagonal() - c1.get_cov().diagonal()).norm() < 1e-9);
        assert!((c2.get_cov_walk().diagonal() - c1.get_cov_walk().diagonal()).norm() < 1e-9);
    }
    // normalize rotation
    #[test]
    fn normalize_rotation_identity_in_identity_out() {
        let rn = normalize_rotation(Matrix3::identity());
        assert!((rn - Matrix3::<f32>::identity()).norm() < 1e-5);
    }

    #[test]
    fn normalize_rotation_output_is_always_valid_rotation() {
        let axis = Unit::new_normalize(Vector3::new(1.0_f32, 1.0, 0.0));
        let mut r = Rotation3::from_axis_angle(&axis, 0.7_f32).into_inner();
        r[(0, 0)] += 0.1; // perturb to make non-orthogonal
        let rn = normalize_rotation(r);
        assert!(is_rotation(&rn, 1e-5));
    }

    #[test]
    fn normalize_rotation_valid_rotation_is_preserved() {
        let axis = Unit::new_normalize(Vector3::new(1.0_f32, 0.5, 0.3));
        let r = Rotation3::from_axis_angle(&axis, 1.2_f32).into_inner();
        let rn = normalize_rotation(r);
        assert!((rn - r).norm() < 1e-5);
    }

    // jacobians
    #[test]
    fn right_jacobian_so3_near_zero_returns_identity() {
        let j = right_jacobian_so3(Vector3::zeros());
        assert!((j - Matrix3::<f32>::identity()).norm() < 1e-5);
    }

    #[test]
    fn inverse_right_jacobian_so3_near_zero_returns_identity() {
        let ji = inverse_right_jacobian_so3(Vector3::zeros());
        assert!((ji - Matrix3::<f32>::identity()).norm() < 1e-5);
    }

    #[test]
    fn right_jacobian_so3_xyz_and_vector_overloads_agree() {
        let (x, y, z) = (0.2_f32, -0.3, 0.15);
        let j1 = right_jacobian_so3_xyz(x, y, z);
        let j2 = right_jacobian_so3(Vector3::new(x, y, z));
        assert!((j1 - j2).norm() < 1e-7);
    }

    #[test]
    fn inverse_right_jacobian_so3_xyz_and_vector_overloads_agree() {
        let (x, y, z) = (-0.1_f32, 0.4, -0.2);
        let ji1 = inverse_right_jacobian_so3_xyz(x, y, z);
        let ji2 = inverse_right_jacobian_so3(Vector3::new(x, y, z));
        assert!((ji1 - ji2).norm() < 1e-7);
    }

    #[test]
    fn right_jacobian_times_inverse_right_jacobian_is_identity() {
        let v = Vector3::new(0.1_f32, 0.2, 0.3);
        let j = right_jacobian_so3(v);
        let ji = inverse_right_jacobian_so3(v);
        assert!((j * ji - Matrix3::<f32>::identity()).norm() < 1e-5);
    }

    #[test]
    fn inverse_right_jacobian_times_right_jacobian_is_identity() {
        let v = Vector3::new(0.4_f32, -0.1, 0.5);
        let j = right_jacobian_so3(v);
        let ji = inverse_right_jacobian_so3(v);
        assert!((ji * j - Matrix3::<f32>::identity()).norm() < 1e-5);
    }

    // integrated rotation
    #[test]
    fn integrated_rotation_zero_angular_velocity_gives_identity() {
        let ir = IntegratedRotation::from_params(&Vector3::zeros(), &zero_bias(), 0.01);
        assert!(is_rotation(&ir.delta_r, 1e-5));
        assert!((ir.delta_r - Matrix3::<f32>::identity()).norm() < 1e-5);
    }

    #[test]
    fn integrated_rotation_result_is_always_valid_rotation() {
        let bias = Bias::from_params(0.0, 0.0, 0.0, 0.01, 0.02, -0.01);
        let ang_vel = Vector3::new(0.1_f32, 0.2, 0.3);
        let ir = IntegratedRotation::from_params(&ang_vel, &bias, 0.05);
        assert!(is_rotation(&ir.delta_r, 1e-5));
    }

    #[test]
    fn integrated_rotation_bias_subtracted_from_angular_velocity() {
        // When angVel == bias the net corrected angular velocity is zero → identity.
        let bias = Bias::from_params(0.0, 0.0, 0.0, 0.1, 0.2, 0.3);
        let ang_vel = Vector3::new(0.1_f32, 0.2, 0.3);
        let ir = IntegratedRotation::from_params(&ang_vel, &bias, 0.01);
        assert!(
            (ir.delta_r - Matrix3::<f32>::identity()).norm() < 1e-5,
            "delta_r should be identity when angVel == bias"
        );
    }

    #[test]
    fn integrated_rotation_matches_rodrigues_for_rotation_about_z() {
        let omega = 0.5_f32;
        let ir = IntegratedRotation::from_params(&Vector3::new(0.0, 0.0, omega), &zero_bias(), 1.0);
        let expected = Rotation3::from_axis_angle(&Vector3::z_axis(), omega).into_inner();
        assert!((ir.delta_r - expected).norm() < 1e-5);
    }

    #[test]
    fn integrated_rotation_right_j_matches_right_jacobian_so3() {
        let ang_vel = Vector3::new(0.1_f32, -0.2, 0.3);
        let dt = 0.1_f32;
        let ir = IntegratedRotation::from_params(&ang_vel, &zero_bias(), dt);
        let expected_j = right_jacobian_so3(ang_vel * dt);
        assert!(
            (ir.right_j - expected_j).norm() < 1e-5,
            "rightJ mismatch: {}",
            (ir.right_j - expected_j).norm()
        );
    }

    // preintegrated
    #[test]
    fn preintegrated_constant_acceleration_zero_angular_velocity_follows_kinematics() {
        // acc = (1,0,0), angVel = 0, N*dt = T = 1 s
        // Expected: dV = (T,0,0), dP = (0.5*T^2, 0, 0), dR = I
        let mut pre = Preintegrated::from_bias_and_calib(&zero_bias(), &make_calib());
        let acc = Vector3::new(1.0_f32, 0.0, 0.0);
        let n = 100;
        let dt = 0.01_f32;
        let t = n as f32 * dt;

        for _ in 0..n {
            pre.integrate_new_measurement(acc, Vector3::zeros(), dt);
        }

        assert!((pre.dt - t).abs() < 1e-5);
        assert!((pre.dv[0] - t).abs() < 1e-4);
        assert!(pre.dv[1].abs() < 1e-6);
        assert!(pre.dv[2].abs() < 1e-6);
        assert!((pre.dp[0] - 0.5 * t * t).abs() < 1e-3);
        assert!(pre.dp[1].abs() < 1e-6);
        assert!(pre.dp[2].abs() < 1e-6);
        assert!(is_rotation(&pre.dr, 1e-4));
        assert!((pre.dr - Matrix3::<f32>::identity()).norm() < 1e-4);
    }

    #[test]
    fn preintegrated_dt_accumulates_total_integration_time() {
        let mut pre = Preintegrated::from_bias_and_calib(&zero_bias(), &make_calib());
        let dt = 0.005_f32;
        let n = 200;
        for _ in 0..n {
            pre.integrate_new_measurement(Vector3::zeros(), Vector3::zeros(), dt);
        }
        assert!((pre.dt - n as f32 * dt).abs() < 1e-4);
    }

    #[test]
    fn preintegrated_avg_a_converges_to_measured_acceleration() {
        let mut pre = Preintegrated::from_bias_and_calib(&zero_bias(), &make_calib());
        let acc = Vector3::new(0.3_f32, -0.5, 9.8);
        for _ in 0..50 {
            pre.integrate_new_measurement(acc, Vector3::zeros(), 0.01);
        }
        assert!((pre.avg_a[0] - acc[0]).abs() < 1e-4);
        assert!((pre.avg_a[1] - acc[1]).abs() < 1e-4);
        assert!((pre.avg_a[2] - acc[2]).abs() < 1e-4);
    }

    #[test]
    fn preintegrated_avg_w_converges_to_measured_angular_velocity() {
        let mut pre = Preintegrated::from_bias_and_calib(&zero_bias(), &make_calib());
        let ang_vel = Vector3::new(0.05_f32, -0.03, 0.1);
        for _ in 0..50 {
            pre.integrate_new_measurement(Vector3::zeros(), ang_vel, 0.01);
        }
        assert!((pre.avg_w[0] - ang_vel[0]).abs() < 1e-4);
        assert!((pre.avg_w[1] - ang_vel[1]).abs() < 1e-4);
        assert!((pre.avg_w[2] - ang_vel[2]).abs() < 1e-4);
    }

    #[test]
    fn preintegrated_constant_angular_velocity_integrates_to_correct_rotation() {
        // angVel = (0,0,omega_z), acc = 0, N*dt = T = 1 s
        // Expected: dR = Rz(omega_z * T)
        // KNOWN BUG: from_params computes v = angVel - bias*dt instead of
        // (angVel - bias)*dt, so with dt = 0.02 the per-step rotation angle
        // equals omega_z (not omega_z * dt), causing a 50x error in dR.
        let mut pre = Preintegrated::from_bias_and_calib(&zero_bias(), &make_calib());
        let omega_z = 0.1_f32;
        let ang_vel = Vector3::new(0.0_f32, 0.0, omega_z);
        let n = 50;
        let dt = 0.02_f32;
        let t = n as f32 * dt; // = 1.0 s

        for _ in 0..n {
            pre.integrate_new_measurement(Vector3::zeros(), ang_vel, dt);
        }

        let expected = Rotation3::from_axis_angle(&Vector3::z_axis(), omega_z * t).into_inner();
        assert!(
            (pre.dr - expected).norm() < 1e-3,
            "dR mismatch: {}",
            (pre.dr - expected).norm()
        );
        assert!(pre.dv.norm() < 1e-6);
        assert!(pre.dp.norm() < 1e-6);
    }

    #[test]
    fn preintegrated_get_original_bias_returns_construction_time_bias() {
        let b = Bias::from_params(0.1, 0.2, 0.3, 0.4, 0.5, 0.6);
        let pre = Preintegrated::from_bias_and_calib(&b, &make_calib());
        let orig = pre.get_original_bias();
        assert!((orig.bax - 0.1_f32).abs() < 1e-7);
        assert!((orig.bay - 0.2_f32).abs() < 1e-7);
        assert!((orig.baz - 0.3_f32).abs() < 1e-7);
        assert!((orig.bwx - 0.4_f32).abs() < 1e-7);
        assert!((orig.bwy - 0.5_f32).abs() < 1e-7);
        assert!((orig.bwz - 0.6_f32).abs() < 1e-7);
    }

    #[test]
    fn preintegrated_set_new_bias_and_get_updated_bias() {
        let mut pre = Preintegrated::from_bias_and_calib(&zero_bias(), &make_calib());
        let new_bias = Bias::from_params(0.01, 0.02, 0.03, 0.04, 0.05, 0.06);
        pre.set_new_bias(&new_bias);

        let updated = pre.get_updated_bias();
        assert!((updated.bax - 0.01_f32).abs() < 1e-7);
        assert!((updated.bay - 0.02_f32).abs() < 1e-7);
        assert!((updated.baz - 0.03_f32).abs() < 1e-7);
        assert!((updated.bwx - 0.04_f32).abs() < 1e-7);
        assert!((updated.bwy - 0.05_f32).abs() < 1e-7);
        assert!((updated.bwz - 0.06_f32).abs() < 1e-7);
        // Original must remain unchanged
        assert_eq!(pre.get_original_bias().bax, 0.0_f32);
    }

    #[test]
    fn preintegrated_get_delta_bias_vector_reflects_updated_minus_original() {
        // db layout: [bwx, bwy, bwz, bax, bay, baz] deltas.
        let mut pre = Preintegrated::from_bias_and_calib(&zero_bias(), &make_calib());
        // Only gyro bias changes.
        let new_bias = Bias::from_params(0.0, 0.0, 0.0, 0.1, 0.2, 0.3);
        pre.set_new_bias(&new_bias);

        let db = pre.get_delta_bias();
        assert!((db[0] - 0.1_f32).abs() < 1e-7); // bwx delta
        assert!((db[1] - 0.2_f32).abs() < 1e-7); // bwy delta
        assert!((db[2] - 0.3_f32).abs() < 1e-7); // bwz delta
        assert!(db[3].abs() < 1e-7); // bax delta
        assert!(db[4].abs() < 1e-7); // bay delta
        assert!(db[5].abs() < 1e-7); // baz delta
    }

    #[test]
    fn preintegrated_compute_bias_diff_computes_per_component_difference() {
        let b0 = Bias::from_params(0.1, 0.2, 0.3, 0.4, 0.5, 0.6);
        let b1 = Bias::from_params(0.2, 0.3, 0.4, 0.5, 0.6, 0.7);
        let pre = Preintegrated::from_bias_and_calib(&b0, &make_calib());
        let delta = pre.compute_bias_diff(&b1);
        assert!((delta.bax - 0.1_f32).abs() < 1e-5);
        assert!((delta.bay - 0.1_f32).abs() < 1e-5);
        assert!((delta.baz - 0.1_f32).abs() < 1e-5);
        assert!((delta.bwx - 0.1_f32).abs() < 1e-5);
        assert!((delta.bwy - 0.1_f32).abs() < 1e-5);
        assert!((delta.bwz - 0.1_f32).abs() < 1e-5);
    }

    #[test]
    fn preintegrated_get_delta_rotation_with_original_bias_equals_dr() {
        let bias = zero_bias();
        let mut pre = Preintegrated::from_bias_and_calib(&bias, &make_calib());
        for _ in 0..20 {
            pre.integrate_new_measurement(
                Vector3::new(0.0, 0.0, 1.0),
                Vector3::new(0.0, 0.1, 0.0),
                0.01,
            );
        }
        let diff = (pre.get_delta_rotation(&bias) - pre.get_original_delta_rotation()).norm();
        assert!(diff < 1e-5, "get_delta_rotation mismatch: {diff}");
    }

    #[test]
    fn preintegrated_get_delta_velocity_with_original_bias_equals_dv() {
        let bias = zero_bias();
        let mut pre = Preintegrated::from_bias_and_calib(&bias, &make_calib());
        for _ in 0..20 {
            pre.integrate_new_measurement(Vector3::new(0.5, 0.0, 0.0), Vector3::zeros(), 0.01);
        }
        let diff = (pre.get_delta_velocity(&bias) - pre.get_original_delta_velocity()).norm();
        assert!(diff < 1e-5, "get_delta_velocity mismatch: {diff}");
    }

    #[test]
    fn preintegrated_get_delta_position_with_original_bias_equals_dp() {
        let bias = zero_bias();
        let mut pre = Preintegrated::from_bias_and_calib(&bias, &make_calib());
        for _ in 0..20 {
            pre.integrate_new_measurement(Vector3::new(0.0, 1.0, 0.0), Vector3::zeros(), 0.01);
        }
        let diff = (pre.get_delta_position(&bias) - pre.get_original_delta_position()).norm();
        assert!(diff < 1e-5, "get_delta_position mismatch: {diff}");
    }

    #[test]
    fn preintegrated_get_updated_delta_equals_original_when_bias_unchanged() {
        let bias = zero_bias();
        let mut pre = Preintegrated::from_bias_and_calib(&bias, &make_calib());
        for _ in 0..30 {
            pre.integrate_new_measurement(
                Vector3::new(0.1, 0.2, 9.8),
                Vector3::new(0.01, -0.02, 0.03),
                0.01,
            );
        }
        // Setting the same bias makes db = 0.
        pre.set_new_bias(&bias);

        assert!(
            (pre.get_updated_delta_rotation() - pre.get_original_delta_rotation()).norm() < 1e-5
        );
        assert!(
            (pre.get_updated_delta_velocity() - pre.get_original_delta_velocity()).norm() < 1e-5
        );
        assert!(
            (pre.get_updated_delta_position() - pre.get_original_delta_position()).norm() < 1e-5
        );
    }

    #[test]
    fn preintegrated_reintegrate_with_unchanged_bias_reproduces_same_result() {
        let mut pre = Preintegrated::from_bias_and_calib(&zero_bias(), &make_calib());
        let acc = Vector3::new(0.3_f32, -0.1, 9.8);
        let ang_vel = Vector3::new(0.05_f32, -0.02, 0.1);
        for _ in 0..50 {
            pre.integrate_new_measurement(acc, ang_vel, 0.01);
        }

        let dr_before = pre.get_original_delta_rotation();
        let dv_before = pre.get_original_delta_velocity();
        let dp_before = pre.get_original_delta_position();
        let dt_before = pre.dt;

        pre.reintegrate();

        assert!((pre.dt - dt_before).abs() < 1e-5);
        assert!((pre.get_original_delta_rotation() - dr_before).norm() < 1e-4);
        assert!((pre.get_original_delta_velocity() - dv_before).norm() < 1e-4);
        assert!((pre.get_original_delta_position() - dp_before).norm() < 1e-4);
    }

    #[test]
    fn preintegrated_clone_duplicates_all_preintegrated_state() {
        let mut pre = Preintegrated::from_bias_and_calib(&zero_bias(), &make_calib());
        for _ in 0..10 {
            pre.integrate_new_measurement(Vector3::new(0.1, 0.0, 0.0), Vector3::zeros(), 0.01);
        }
        let copy = pre.clone();

        assert!((copy.dt - pre.dt).abs() < 1e-7);
        assert!(
            (copy.get_original_delta_velocity() - pre.get_original_delta_velocity()).norm() < 1e-6
        );
        assert!(
            (copy.get_original_delta_position() - pre.get_original_delta_position()).norm() < 1e-6
        );
        assert!(
            (copy.get_original_delta_rotation() - pre.get_original_delta_rotation()).norm() < 1e-6
        );
    }

    #[test]
    fn preintegrated_copy_from_via_clone_duplicates_all_state() {
        let mut pre = Preintegrated::from_bias_and_calib(&zero_bias(), &make_calib());
        for _ in 0..15 {
            pre.integrate_new_measurement(
                Vector3::new(0.0, 0.5, 0.0),
                Vector3::new(0.0, 0.0, 0.1),
                0.01,
            );
        }
        // CopyFrom in C++ is equivalent to Clone in Rust.
        let copy = pre.clone();

        assert!((copy.dt - pre.dt).abs() < 1e-7);
        assert!(
            (copy.get_original_delta_rotation() - pre.get_original_delta_rotation()).norm() < 1e-6
        );
        assert!(
            (copy.get_original_delta_velocity() - pre.get_original_delta_velocity()).norm() < 1e-6
        );
        assert!(
            (copy.get_original_delta_position() - pre.get_original_delta_position()).norm() < 1e-6
        );
    }

    #[test]
    fn preintegrated_merge_previous_equivalent_to_integrating_all_in_order() {
        let calib = make_calib();
        let bias = zero_bias();
        let acc = Vector3::new(0.2_f32, 0.0, 0.0);
        let ang_vel = Vector3::new(0.0_f32, 0.0, 0.05);
        let dt = 0.01_f32;

        // Reference: 20 consecutive measurements in one object.
        let mut reference = Preintegrated::from_bias_and_calib(&bias, &calib);
        for _ in 0..20 {
            reference.integrate_new_measurement(acc, ang_vel, dt);
        }

        // Split: first 10 → pre1, next 10 → pre2, then merge.
        let mut pre1 = Preintegrated::from_bias_and_calib(&bias, &calib);
        for _ in 0..10 {
            pre1.integrate_new_measurement(acc, ang_vel, dt);
        }
        let mut pre2 = Preintegrated::from_bias_and_calib(&bias, &calib);
        for _ in 0..10 {
            pre2.integrate_new_measurement(acc, ang_vel, dt);
        }
        pre2.merge_previous(&pre1);

        assert!((pre2.dt - reference.dt).abs() < 1e-5);
        assert!(
            (pre2.get_original_delta_rotation() - reference.get_original_delta_rotation()).norm()
                < 1e-4
        );
        assert!(
            (pre2.get_original_delta_velocity() - reference.get_original_delta_velocity()).norm()
                < 1e-4
        );
        assert!(
            (pre2.get_original_delta_position() - reference.get_original_delta_position()).norm()
                < 1e-4
        );
    }
}
