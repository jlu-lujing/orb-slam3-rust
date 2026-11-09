//! Parity test for `Optimizer::InertialOptimization(Map, Rwg, scale)`.
//!
//! Replays the fixture from `optimizer_inertialgs_fixture_gen` (real g2o
//! EdgeInertialGS + Gauss-Newton). The IMU preintegration fields are dumped by
//! the C++ side and reconstructed here so both run on identical inputs.

use std::sync::Arc;

use nalgebra::{Isometry3, Matrix3, SMatrix, Vector3};

use orb_slam3_rs::imu_types::{Bias, Calib, Preintegrated};
use orb_slam3_rs::optimizer::{InertialKf, inertial_optimization_gs_core};

struct Tok {
    vals: Vec<f64>,
    i: usize,
}
impl Tok {
    fn new(text: &str) -> Self {
        let vals = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .flat_map(|l| l.split_whitespace())
            .map(|s| s.parse::<f64>().unwrap())
            .collect();
        Tok { vals, i: 0 }
    }
    fn next(&mut self) -> f64 {
        let v = self.vals[self.i];
        self.i += 1;
        v
    }
    fn mat3(&mut self) -> Matrix3<f64> {
        let mut m = Matrix3::zeros();
        for r in 0..3 {
            for c in 0..3 {
                m[(r, c)] = self.next();
            }
        }
        m
    }
    fn vec3(&mut self) -> Vector3<f64> {
        Vector3::new(self.next(), self.next(), self.next())
    }
    fn mat3f(&mut self) -> Matrix3<f32> {
        self.mat3().cast::<f32>()
    }
    fn vec3f(&mut self) -> Vector3<f32> {
        self.vec3().cast::<f32>()
    }
}

#[test]
fn inertial_optimization_gs_matches_g2o() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/inertial_optimization_gs.txt"
    ))
    .expect("fixture present");
    let mut t = Tok::new(&text);

    let n_kf = t.next() as usize;
    let _dt = t.next();
    let _steps = t.next();
    let front_gyro = t.vec3();
    let front_acc = t.vec3();
    let int_bias = Bias::from_params(
        t.next() as f32, // bax
        t.next() as f32, // bay
        t.next() as f32, // baz
        t.next() as f32, // bwx
        t.next() as f32, // bwy
        t.next() as f32, // bwz
    );
    let rwg_init = t.mat3();
    let scale_init = t.next();

    let mut rwb = Vec::new();
    let mut twb = Vec::new();
    let mut vel = Vec::new();
    for _ in 0..n_kf {
        rwb.push(t.mat3());
        twb.push(t.vec3());
        vel.push(t.vec3());
    }

    // Reconstruct preintegration objects from the dumped fields.
    let calib = Calib::from_params(Isometry3::identity(), 1.7e-4, 2.0e-3, 1.9e-5, 3.0e-3);
    let mut preints: Vec<Option<Arc<Preintegrated>>> = vec![None];
    for _ in 1..n_kf {
        let mut p = Preintegrated::from_bias_and_calib(&int_bias, &calib);
        p.dr = t.mat3f();
        p.dv = t.vec3f();
        p.dp = t.vec3f();
        p.jrg = t.mat3f();
        p.jvg = t.mat3f();
        p.jva = t.mat3f();
        p.jpg = t.mat3f();
        p.jpa = t.mat3f();
        let mut c = SMatrix::<f32, 15, 15>::zeros();
        for r in 0..15 {
            for col in 0..15 {
                c[(r, col)] = t.next() as f32;
            }
        }
        p.c = c;
        p.dt = t.next() as f32;
        p.b = int_bias;
        preints.push(Some(Arc::new(p)));
    }

    let ref_rwg = t.mat3();
    let ref_scale = t.next();

    let kfs: Vec<InertialKf> = (0..n_kf)
        .map(|i| InertialKf {
            rwb: rwb[i],
            twb: twb[i],
            vel: vel[i],
            preint: preints[i].clone(),
        })
        .collect();

    let (rwg, scale) =
        inertial_optimization_gs_core(&kfs, front_gyro, front_acc, rwg_init, scale_init);

    let dr = (rwg.transpose() * ref_rwg).into();
    let ang = orb_slam3_rs::g2o_core::log_so3(&dr).norm();
    assert!(ang < 1e-5, "Rwg differs by {ang} rad");
    assert!(
        (scale - ref_scale).abs() < 1e-5,
        "scale: got {scale}, ref {ref_scale}"
    );
}
