//! Parity test for `Optimizer::OptimizeSim3` (Sim3 + numerical Jacobians).
//!
//! Replays the fixtures from `ORB_SLAM3/tests/optimizer_sim3_fixture_gen`
//! (real g2o) through the Rust mini-g2o port, for both the Sim3 (free scale)
//! and SE3 (fixed scale) cases.

use std::sync::Arc;

use nalgebra::{Quaternion, UnitQuaternion, Vector2, Vector3};

use orb_slam3_rs::camera_models::GeometricCamera;
use orb_slam3_rs::camera_models::pinhole::Pinhole;
use orb_slam3_rs::g2o_core::Sim3;
use orb_slam3_rs::optimizer::{Sim3Correspondence, optimize_sim3_core};

fn data_lines(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect()
}
fn nums(line: &str) -> Vec<f64> {
    line.split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect()
}
fn parse_sim3(v: &[f64]) -> Sim3 {
    let q = UnitQuaternion::new_normalize(Quaternion::new(v[0], v[1], v[2], v[3]));
    Sim3::new(q, Vector3::new(v[4], v[5], v[6]), v[7])
}

fn run(path: &str) {
    let text = std::fs::read_to_string(path).expect("fixture present");
    let lines = data_lines(&text);
    let mut it = lines.iter();

    let intr = nums(it.next().unwrap());
    let (fx, fy, cx, cy) = (intr[0], intr[1], intr[2], intr[3]);
    let fs = nums(it.next().unwrap());
    let (fix_scale, th2) = (fs[0] == 1.0, fs[1]);
    let s12_init = parse_sim3(&nums(it.next().unwrap()));
    let n: usize = it.next().unwrap().parse().unwrap();

    let mut corrs = Vec::with_capacity(n);
    for _ in 0..n {
        let v = nums(it.next().unwrap());
        corrs.push(Sim3Correspondence {
            x1c: Vector3::new(v[0], v[1], v[2]),
            x2c: Vector3::new(v[3], v[4], v[5]),
            obs1: Vector2::new(v[6], v[7]),
            obs2: Vector2::new(v[8], v[9]),
            inv_sigma1: v[10],
            inv_sigma2: v[11],
        });
    }
    let ref_n_in: i32 = it.next().unwrap().parse().unwrap();
    let ref_s12 = parse_sim3(&nums(it.next().unwrap()));
    let ref_kept: Vec<bool> = it
        .next()
        .unwrap()
        .split_whitespace()
        .map(|s| s == "1")
        .collect();

    let cam: Arc<dyn GeometricCamera> = Arc::new(Pinhole::with_params(vec![
        fx as f32, fy as f32, cx as f32, cy as f32,
    ]));

    let (n_in, s12, kept) = optimize_sim3_core(s12_init, &cam, &cam, &corrs, th2, fix_scale);

    assert_eq!(kept, ref_kept, "inlier classification differs");
    assert_eq!(n_in, ref_n_in, "inlier count differs");

    let dr = s12.rotation().angle_to(&ref_s12.rotation());
    let dt = (s12.translation() - ref_s12.translation()).norm();
    let ds = (s12.scale() - ref_s12.scale()).abs();
    assert!(dr < 1e-6, "rotation differs by {dr}");
    assert!(dt < 1e-6, "translation differs by {dt}");
    assert!(ds < 1e-6, "scale differs by {ds}");
}

#[test]
fn optimize_sim3_free_scale_matches_g2o() {
    run(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/optimize_sim3.txt"
    ));
}

#[test]
fn optimize_sim3_fixed_scale_matches_g2o() {
    run(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/optimize_sim3_fixscale.txt"
    ));
}
