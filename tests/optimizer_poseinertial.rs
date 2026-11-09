//! Parity test for `Optimizer::PoseInertialOptimizationLastKeyFrame`.
//!
//! Replays the fixture from `optimizer_poseinertial_fixture_gen` (real g2o:
//! EdgeMonoOnlyPose/EdgeStereoOnlyPose + EdgeInertial + bias RW, Gauss-Newton).

use std::sync::Arc;

use nalgebra::{Isometry3, Matrix3, SMatrix, Vector2, Vector3};

use orb_slam3_rs::camera_models::GeometricCamera;
use orb_slam3_rs::camera_models::pinhole::Pinhole;
use orb_slam3_rs::g2o_core::log_so3;
use orb_slam3_rs::imu_types::{Bias, Calib, Preintegrated};
use orb_slam3_rs::optimizer::{ImuState, InertialPoseObs, pose_inertial_optimization_last_kf_core};

struct Tok {
    v: Vec<f64>,
    i: usize,
}
impl Tok {
    fn new(t: &str) -> Self {
        let v = t
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .flat_map(|l| l.split_whitespace())
            .map(|s| s.parse().unwrap())
            .collect();
        Tok { v, i: 0 }
    }
    fn n(&mut self) -> f64 {
        let x = self.v[self.i];
        self.i += 1;
        x
    }
    fn m3(&mut self) -> Matrix3<f64> {
        let mut m = Matrix3::zeros();
        for r in 0..3 {
            for c in 0..3 {
                m[(r, c)] = self.n();
            }
        }
        m
    }
    fn v3(&mut self) -> Vector3<f64> {
        Vector3::new(self.n(), self.n(), self.n())
    }
    fn m3f(&mut self) -> Matrix3<f32> {
        self.m3().cast::<f32>()
    }
    fn v3f(&mut self) -> Vector3<f32> {
        self.v3().cast::<f32>()
    }
    fn state(&mut self) -> ImuState {
        ImuState {
            rwb: self.m3(),
            twb: self.v3(),
            vel: self.v3(),
            bg: self.v3(),
            ba: self.v3(),
        }
    }
}

#[test]
fn pose_inertial_last_kf_matches_g2o() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/pose_inertial.txt"
    ))
    .expect("fixture present");
    let mut t = Tok::new(&text);

    let (fx, fy, cx, cy, bf) = (t.n(), t.n(), t.n(), t.n(), t.n());
    let last = t.state();
    let cur = t.state();

    // Preintegration.
    let int_bias = Bias::from_params(
        last.ba[0] as f32,
        last.ba[1] as f32,
        last.ba[2] as f32,
        last.bg[0] as f32,
        last.bg[1] as f32,
        last.bg[2] as f32,
    );
    let calib = Calib::from_params(Isometry3::identity(), 1.7e-4, 2.0e-3, 1.9e-5, 3.0e-3);
    let mut p = Preintegrated::from_bias_and_calib(&int_bias, &calib);
    p.dr = t.m3f();
    p.dv = t.v3f();
    p.dp = t.v3f();
    p.jrg = t.m3f();
    p.jvg = t.m3f();
    p.jva = t.m3f();
    p.jpg = t.m3f();
    p.jpa = t.m3f();
    let mut c = SMatrix::<f32, 15, 15>::zeros();
    for r in 0..15 {
        for col in 0..15 {
            c[(r, col)] = t.n() as f32;
        }
    }
    p.c = c;
    p.dt = t.n() as f32;
    p.b = int_bias;
    let preint = Arc::new(p);

    let n_obs = t.n() as usize;
    let mut obs = Vec::with_capacity(n_obs);
    for i in 0..n_obs {
        let kind = t.n() as i32;
        // Real MapPoint positions are f32; the C++ edges store Vector3f. Match
        // that precision so the (large) prior Hessian agrees.
        let xw = t.v3().map(|c| c as f32 as f64);
        let o = t.v3();
        let inv_sigma2 = t.n();
        let track_depth = t.n() as f32;
        if kind == 0 {
            obs.push(InertialPoseObs::Mono {
                xw,
                obs: Vector2::new(o[0], o[1]),
                inv_sigma2,
                track_depth,
                cam_idx: 0,
                idx: i,
            });
        } else {
            obs.push(InertialPoseObs::Stereo {
                xw,
                obs: o,
                inv_sigma2,
                cam_idx: 0,
                idx: i,
            });
        }
    }

    let ref_state = t.state();
    let ref_inliers = t.n() as i32;
    let ref_flags: Vec<bool> = (0..n_obs).map(|_| t.n() == 1.0).collect();
    let mut ref_h = SMatrix::<f64, 15, 15>::zeros();
    for r in 0..15 {
        for col in 0..15 {
            ref_h[(r, col)] = t.n();
        }
    }

    let camera: Arc<dyn GeometricCamera> = Arc::new(Pinhole::with_params(vec![
        fx as f32, fy as f32, cx as f32, cy as f32,
    ]));
    let rbc = [Matrix3::identity()];
    let tbc = [Vector3::zeros()];

    let res = pose_inertial_optimization_last_kf_core(
        &cur,
        &last,
        preint,
        &obs,
        &[camera],
        &rbc,
        &tbc,
        bf,
        false,
    );

    // Outlier classification + inlier count.
    let got_flags: Vec<bool> = res.outliers.iter().map(|(_, b)| *b).collect();
    assert_eq!(got_flags, ref_flags, "outlier classification differs");
    assert_eq!(res.n_inliers, ref_inliers, "inlier count differs");

    // Optimized state.
    let dr = (res.state.rwb.transpose() * ref_state.rwb).into();
    assert!(log_so3(&dr).norm() < 1e-6, "rotation differs");
    assert!(
        (res.state.twb - ref_state.twb).norm() < 1e-6,
        "translation differs"
    );
    assert!(
        (res.state.vel - ref_state.vel).norm() < 1e-6,
        "velocity differs"
    );
    assert!(
        (res.state.bg - ref_state.bg).norm() < 1e-6,
        "gyro bias differs"
    );
    assert!(
        (res.state.ba - ref_state.ba).norm() < 1e-6,
        "acc bias differs"
    );

    // Marginalization prior Hessian.
    // The IMU information block is ~1e11-conditioned; tiny entries in its
    // near-null-space carry benign eigensolver differences. Compare via the
    // relative Frobenius norm, which reflects the meaningful (large) entries.
    let fro_diff = (res.prior_h_raw - ref_h).norm() / ref_h.norm();
    assert!(
        fro_diff < 1e-6,
        "prior Hessian relative Frobenius diff {fro_diff}"
    );
}
