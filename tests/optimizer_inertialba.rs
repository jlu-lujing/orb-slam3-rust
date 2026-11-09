//! Parity test for the inertial bundle-adjustment core (shared by
//! FullInertialBA / LocalInertialBA / MergeInertialBA).
//!
//! Replays the fixture from `optimizer_inertialba_fixture_gen` (real g2o:
//! EdgeInertial + bias RW + EdgeMono/EdgeStereo, Levenberg-Marquardt).

use std::sync::Arc;

use nalgebra::{Isometry3, Matrix3, SMatrix, Vector2, Vector3};

use orb_slam3_rs::camera_models::GeometricCamera;
use orb_slam3_rs::camera_models::pinhole::Pinhole;
use orb_slam3_rs::g2o_core::log_so3;
use orb_slam3_rs::imu_types::{Bias, Calib, Preintegrated};
use orb_slam3_rs::optimizer::{
    ImuState, InertialBaKf, InertialBaObs, InertialLink, inertial_ba_core,
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
}

#[test]
fn inertial_ba_matches_g2o() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/inertial_ba.txt"
    ))
    .expect("fixture present");
    let mut t = Tok::new(&text);

    let (fx, fy, cx, cy, bf) = (t.n(), t.n(), t.n(), t.n(), t.n());
    let camera: Arc<dyn GeometricCamera> = Arc::new(Pinhole::with_params(vec![
        fx as f32, fy as f32, cx as f32, cy as f32,
    ]));
    let int_bias = Bias::from_params(0.01, -0.02, 0.015, 0.001, -0.0015, 0.0008);
    let calib = Calib::from_params(Isometry3::identity(), 1.7e-4, 2.0e-3, 1.9e-5, 3.0e-3);

    let n_kf = t.n() as usize;
    let mut kfs = Vec::with_capacity(n_kf);
    for _ in 0..n_kf {
        let fixed = t.n() == 1.0;
        let state = t.state();
        kfs.push(InertialBaKf {
            state,
            fixed,
            camera: camera.clone(),
            rbc: Matrix3::identity(),
            tbc: Vector3::zeros(),
            bf,
        });
    }

    let n_links = t.n() as usize;
    let mut links = Vec::with_capacity(n_links);
    for _ in 0..n_links {
        let prev = t.n() as usize;
        let cur = t.n() as usize;
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
        links.push(InertialLink {
            prev,
            cur,
            preint: Arc::new(p),
            robust_delta: Some(16.92_f64.sqrt()),
            info_scale: 1.0,
        });
    }

    let n_pts = t.n() as usize;
    let points: Vec<Vector3<f64>> = (0..n_pts)
        .map(|_| t.v3().map(|x| x as f32 as f64))
        .collect();

    let n_obs = t.n() as usize;
    let mut obs = Vec::with_capacity(n_obs);
    for _ in 0..n_obs {
        let kind = t.n() as i32;
        let kf = t.n() as usize;
        let mp = t.n() as usize;
        let o = t.v3();
        let inv_sigma2 = t.n();
        if kind == 0 {
            obs.push(InertialBaObs::Mono {
                kf,
                mp,
                obs: Vector2::new(o[0], o[1]),
                inv_sigma2,
                cam_idx: 0,
            });
        } else {
            obs.push(InertialBaObs::Stereo {
                kf,
                mp,
                obs: o,
                inv_sigma2,
                cam_idx: 0,
            });
        }
    }

    let ref_states: Vec<ImuState> = (0..n_kf).map(|_| t.state()).collect();
    let ref_points: Vec<Vector3<f64>> = (0..n_pts).map(|_| t.v3()).collect();

    let res = inertial_ba_core(&kfs, &links, &points, &obs, 10, 1e-5, None);
    let (states, pts) = (res.states, res.points);

    for (i, (g, r)) in states.iter().zip(ref_states.iter()).enumerate() {
        let dr = (g.rwb.transpose() * r.rwb).into();
        assert!(log_so3(&dr).norm() < 1e-5, "KF {i} rotation differs");
        assert!((g.twb - r.twb).norm() < 1e-5, "KF {i} translation differs");
        assert!((g.vel - r.vel).norm() < 1e-5, "KF {i} velocity differs");
        assert!((g.bg - r.bg).norm() < 1e-5, "KF {i} gyro bias differs");
        assert!((g.ba - r.ba).norm() < 1e-5, "KF {i} acc bias differs");
    }
    for (j, (g, r)) in pts.iter().zip(ref_points.iter()).enumerate() {
        assert!(
            (g - r).norm() < 1e-4,
            "point {j} differs by {}",
            (g - r).norm()
        );
    }
}
