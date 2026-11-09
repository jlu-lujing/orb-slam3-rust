//! Parity test for `Optimizer::PoseInertialOptimizationLastFrame`.
//!
//! Replays the fixture from `optimizer_poseinertiallf_fixture_gen` (real g2o:
//! reprojection + EdgeInertial + bias RW + EdgePriorPoseImu, Gauss-Newton, then
//! 30->15 marginalization of the previous frame).

use std::sync::Arc;

use nalgebra::{Isometry3, Matrix3, SMatrix, Vector2, Vector3};

use orb_slam3_rs::camera_models::GeometricCamera;
use orb_slam3_rs::camera_models::pinhole::Pinhole;
use orb_slam3_rs::g2o_core::log_so3;
use orb_slam3_rs::g2o_types::ConstraintPoseIMU;
use orb_slam3_rs::imu_types::{Bias, Calib, Preintegrated};
use orb_slam3_rs::optimizer::{
    ImuState, InertialPoseObs, pose_inertial_optimization_last_frame_core,
};

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
    fn m15(&mut self) -> SMatrix<f64, 15, 15> {
        let mut m = SMatrix::<f64, 15, 15>::zeros();
        for r in 0..15 {
            for c in 0..15 {
                m[(r, c)] = self.n();
            }
        }
        m
    }
}

#[test]
fn pose_inertial_last_frame_matches_g2o() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/pose_inertial_lf.txt"
    ))
    .expect("fixture present");
    let mut t = Tok::new(&text);

    let (fx, fy, cx, cy, bf) = (t.n(), t.n(), t.n(), t.n(), t.n());
    let prev = t.state();
    let cur = t.state();

    let int_bias = Bias::from_params(
        prev.ba[0] as f32,
        prev.ba[1] as f32,
        prev.ba[2] as f32,
        prev.bg[0] as f32,
        prev.bg[1] as f32,
        prev.bg[2] as f32,
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
    let info_g: Matrix3<f64> = preint
        .c
        .fixed_view::<3, 3>(9, 9)
        .into_owned()
        .cast::<f64>()
        .try_inverse()
        .unwrap();
    let info_a: Matrix3<f64> = preint
        .c
        .fixed_view::<3, 3>(12, 12)
        .into_owned()
        .cast::<f64>()
        .try_inverse()
        .unwrap();

    let n_obs = t.n() as usize;
    let mut obs = Vec::with_capacity(n_obs);
    for i in 0..n_obs {
        let kind = t.n() as i32;
        let xw = t.v3().map(|x| x as f32 as f64);
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
    let h_prior = t.m15(); // input to prev cpi
    let ref_hc = t.m15(); // output marginalized current prior

    let prev_cpi = ConstraintPoseIMU::new(prev.rwb, prev.twb, prev.vel, prev.bg, prev.ba, h_prior);

    let camera: Arc<dyn GeometricCamera> = Arc::new(Pinhole::with_params(vec![
        fx as f32, fy as f32, cx as f32, cy as f32,
    ]));
    let rbc = [Matrix3::identity()];
    let tbc = [Vector3::zeros()];

    let res = pose_inertial_optimization_last_frame_core(
        &cur,
        &prev,
        preint,
        info_g,
        info_a,
        &prev_cpi,
        &obs,
        &[camera],
        &rbc,
        &tbc,
        bf,
        false,
    );

    let got_flags: Vec<bool> = res.outliers.iter().map(|(_, b)| *b).collect();
    assert_eq!(got_flags, ref_flags, "outlier classification differs");
    assert_eq!(res.n_inliers, ref_inliers, "inlier count differs");

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

    let fro = (res.prior_h_raw - ref_hc).norm() / ref_hc.norm();
    assert!(
        fro < 1e-6,
        "marginalized prior relative Frobenius diff {fro}"
    );
}
