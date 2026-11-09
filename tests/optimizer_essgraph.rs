//! Parity test for `Optimizer::OptimizeEssentialGraph` (Sim3 pose graph).
//!
//! Replays the fixtures from `optimizer_essgraph_fixture_gen` (real g2o
//! EdgeSim3 + BlockSolver_7_3) through the Rust mini-g2o port.

use nalgebra::{Quaternion, UnitQuaternion, Vector3};

use orb_slam3_rs::g2o_core::Sim3;
use orb_slam3_rs::optimizer::{EssentialGraphEdge, optimize_essential_graph_core};

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

    let fix_scale = it.next().unwrap().trim() == "1";
    let n_kf: usize = it.next().unwrap().parse().unwrap();
    let poses: Vec<Sim3> = (0..n_kf)
        .map(|_| parse_sim3(&nums(it.next().unwrap())))
        .collect();
    let mut fixed = vec![false; n_kf];
    fixed[0] = true;

    let n_edges: usize = it.next().unwrap().parse().unwrap();
    let mut edges = Vec::with_capacity(n_edges);
    for _ in 0..n_edges {
        let v = nums(it.next().unwrap());
        edges.push(EssentialGraphEdge {
            i: v[0] as usize,
            j: v[1] as usize,
            sji: parse_sim3(&v[2..10]),
        });
    }
    let ref_out: Vec<Sim3> = (0..n_kf)
        .map(|_| parse_sim3(&nums(it.next().unwrap())))
        .collect();

    let out = optimize_essential_graph_core(&poses, &fixed, fix_scale, &edges, 20);

    for (i, (got, exp)) in out.iter().zip(ref_out.iter()).enumerate() {
        let dr = got.rotation().angle_to(&exp.rotation());
        let dt = (got.translation() - exp.translation()).norm();
        let ds = (got.scale() - exp.scale()).abs();
        assert!(dr < 1e-6, "KF {i} rotation differs by {dr}");
        assert!(dt < 1e-6, "KF {i} translation differs by {dt}");
        assert!(ds < 1e-6, "KF {i} scale differs by {ds}");
    }
}

#[test]
fn essential_graph_free_scale_matches_g2o() {
    run(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/optimize_essential_graph.txt"
    ));
}

#[test]
fn essential_graph_fixed_scale_matches_g2o() {
    run(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/optimize_essential_graph_fixscale.txt"
    ));
}
