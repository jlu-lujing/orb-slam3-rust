//! Parity test for `Optimizer::InertialOptimization` full overload
//! (velocity + bias + gravity + scale, with bias priors, LM).

use std::sync::Arc;

use nalgebra::{Isometry3, Matrix3, SMatrix, Vector3};

use orb_slam3_rs::imu_types::{Bias, Calib, Preintegrated};
use orb_slam3_rs::optimizer::{InertialKf, InertialOptConfig, inertial_optimization_core};

struct Tok {
    vals: Vec<f64>,
    i: usize,
}
impl Tok {
    fn new(t: &str) -> Self {
        let vals = t
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .flat_map(|l| l.split_whitespace())
            .map(|s| s.parse().unwrap())
            .collect();
        Tok { vals, i: 0 }
    }
    fn n(&mut self) -> f64 {
        let v = self.vals[self.i];
        self.i += 1;
        v
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
}

#[test]
fn inertial_optimization_full_matches_g2o() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/inertial_optimization_full.txt"
    ))
    .expect("fixture present");
    let mut t = Tok::new(&text);

    let n_kf = t.n() as usize;
    let _dt = t.n();
    let _steps = t.n();
    let front_gyro = t.v3();
    let front_acc = t.v3();
    let int_bias = Bias::from_params(
        t.n() as f32,
        t.n() as f32,
        t.n() as f32,
        t.n() as f32,
        t.n() as f32,
        t.n() as f32,
    );
    let rwg_init = t.m3();
    let scale_init = t.n();

    let mut rwb = Vec::new();
    let mut twb = Vec::new();
    let mut vel = Vec::new();
    for _ in 0..n_kf {
        rwb.push(t.m3());
        twb.push(t.v3());
        vel.push(t.v3());
    }

    let calib = Calib::from_params(Isometry3::identity(), 1.7e-4, 2.0e-3, 1.9e-5, 3.0e-3);
    let mut preints: Vec<Option<Arc<Preintegrated>>> = vec![None];
    for _ in 1..n_kf {
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
        preints.push(Some(Arc::new(p)));
    }

    let ref_rwg = t.m3();
    let ref_scale = t.n();
    let ref_bg = t.v3();
    let ref_ba = t.v3();
    let ref_vels: Vec<Vector3<f64>> = (0..n_kf).map(|_| t.v3()).collect();

    let kfs: Vec<InertialKf> = (0..n_kf)
        .map(|i| InertialKf {
            rwb: rwb[i],
            twb: twb[i],
            vel: vel[i],
            preint: preints[i].clone(),
        })
        .collect();

    let cfg = InertialOptConfig {
        fix_vel: false,
        fix_bias: false,
        fix_gdir: false,
        fix_scale: false,
        prior_g: 1e2,
        prior_a: 1e6,
        robust_delta: None,
        user_lambda: Some(1e3),
        gauss_newton: false,
        iterations: 200,
    };
    let r = inertial_optimization_core(&kfs, front_gyro, front_acc, rwg_init, scale_init, &cfg);

    let dr = (r.rwg.transpose() * ref_rwg).into();
    assert!(
        orb_slam3_rs::g2o_core::log_so3(&dr).norm() < 1e-5,
        "Rwg differs"
    );
    assert!(
        (r.scale - ref_scale).abs() < 1e-5,
        "scale {} vs {}",
        r.scale,
        ref_scale
    );
    assert!((r.bg - ref_bg).norm() < 1e-6, "bg differs");
    assert!((r.ba - ref_ba).norm() < 1e-6, "ba differs");
    for (i, (g, e)) in r.vels.iter().zip(ref_vels.iter()).enumerate() {
        assert!(
            (g - e).norm() < 1e-5,
            "vel {i} differs by {}",
            (g - e).norm()
        );
    }
}
