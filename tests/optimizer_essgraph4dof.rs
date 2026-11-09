//! Parity test for `Optimizer::OptimizeEssentialGraph4DoF` (yaw+translation
//! pose graph: g2o VertexPose4DoF + Edge4DoF, numerical Jacobians, LM).

use nalgebra::{Matrix3, Vector3};

use orb_slam3_rs::g2o_core::log_so3;
use orb_slam3_rs::optimizer::{Edge4DoFConstraint, Pose4DoF, optimize_essential_graph_4dof_core};

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
}

#[test]
fn essential_graph_4dof_matches_g2o() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/essential_graph_4dof.txt"
    ))
    .expect("fixture present");
    let mut t = Tok::new(&text);

    let n_kf = t.n() as usize;
    let mut verts = Vec::with_capacity(n_kf);
    for _ in 0..n_kf {
        let fixed = t.n() == 1.0;
        let rcw = t.m3();
        let tcw = t.v3();
        verts.push(Pose4DoF { rcw, tcw, fixed });
    }
    let n_e = t.n() as usize;
    let mut edges = Vec::with_capacity(n_e);
    for _ in 0..n_e {
        let i = t.n() as usize;
        let j = t.n() as usize;
        let drij = t.m3();
        let dtij = t.v3();
        edges.push(Edge4DoFConstraint { i, j, drij, dtij });
    }
    let ref_out: Vec<(Matrix3<f64>, Vector3<f64>)> = (0..n_kf).map(|_| (t.m3(), t.v3())).collect();

    let out = optimize_essential_graph_4dof_core(&verts, &edges, 20);

    for (i, ((rg, tg), (re, te))) in out.iter().zip(ref_out.iter()).enumerate() {
        let dr = (rg.transpose() * re).into();
        assert!(
            log_so3(&dr).norm() < 1e-5,
            "KF {i} rotation differs by {}",
            log_so3(&dr).norm()
        );
        assert!(
            (tg - te).norm() < 1e-5,
            "KF {i} translation differs by {}",
            (tg - te).norm()
        );
    }
}
