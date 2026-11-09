//! Self-consistency check for the analytic Jacobians of the inertial g2o edges.
//!
//! Each edge's analytic Jacobian (from `linearize`) is compared against a
//! central finite-difference of its `compute_error`, perturbing every vertex
//! through its own `oplus`. This catches any sign/transpose/derivative error in
//! the ported `G2oTypes.cc` Jacobians without needing a C++ fixture.

use std::sync::Arc;

use nalgebra::{Isometry3, Matrix3, Vector2, Vector3};

use orb_slam3_rs::camera_models::GeometricCamera;
use orb_slam3_rs::camera_models::pinhole::Pinhole;
use orb_slam3_rs::g2o_core::{Edge, Vertex, exp_so3};
use orb_slam3_rs::g2o_types::{
    ConstraintPoseIMU, EdgeInertial, EdgeInertialGS, EdgeMono, EdgePriorPoseImu, EdgeStereo,
    ImuCamPose, VertexAccBias, VertexGDir, VertexGyroBias, VertexPose, VertexScale, VertexVelocity,
};
use orb_slam3_rs::imu_types::{Bias, Calib, Preintegrated};
use orb_slam3_rs::optimizable_types::VertexSBAPointXYZ;

fn refs(v: &[Box<dyn Vertex>]) -> Vec<&dyn Vertex> {
    v.iter().map(|b| &**b as &dyn Vertex).collect()
}

/// Compare analytic vs numerical Jacobian for every non-fixed vertex of `edge`.
///
/// `delta`/`tol` are configurable because the inertial edges route part of their
/// error through the (single-precision) preintegration, which limits how small a
/// finite-difference step is meaningful.
fn check(mut verts: Vec<Box<dyn Vertex>>, mut edge: Box<dyn Edge>, tag: &str) {
    check_with(&mut verts, &mut *edge, tag, 1e-6, 1e-4, &[]);
}

fn check_with(
    verts: &mut Vec<Box<dyn Vertex>>,
    edge: &mut dyn Edge,
    tag: &str,
    delta: f64,
    tol: f64,
    skip: &[usize],
) {
    let edim = edge.dim();
    let analytic = {
        let r = refs(verts);
        edge.linearize(&r).jacobians
    };
    for k in 0..verts.len() {
        if skip.contains(&k) {
            continue;
        }
        let vdim = verts[k].dim();
        for d in 0..vdim {
            let mut plus = vec![0.0; vdim];
            let mut minus = vec![0.0; vdim];
            plus[d] = delta;
            minus[d] = -delta;

            verts[k].push();
            verts[k].oplus(&plus);
            let ep = {
                let r = refs(verts);
                edge.linearize(&r).error
            };
            verts[k].pop();

            verts[k].push();
            verts[k].oplus(&minus);
            let em = {
                let r = refs(verts);
                edge.linearize(&r).error
            };
            verts[k].pop();

            for row in 0..edim {
                let num = (ep[row] - em[row]) / (2.0 * delta);
                let ana = analytic[k][(row, d)];
                assert!(
                    (num - ana).abs() < tol,
                    "{tag}: vertex {k} d{d} row{row}: analytic {ana}, numerical {num}"
                );
            }
        }
    }
}

fn camera() -> Arc<dyn GeometricCamera> {
    Arc::new(Pinhole::with_params(vec![
        458.654, 457.296, 367.215, 248.375,
    ]))
}

fn make_pose(rwb_axis: Vector3<f64>, ang: f64, twb: Vector3<f64>) -> ImuCamPose {
    // Body->camera: a small fixed rotation + offset so Rcb is non-trivial.
    let rbc = exp_so3(&Vector3::new(0.05, -0.1, 0.02));
    let tbc = Vector3::new(0.01, -0.02, 0.03);
    let rwb = exp_so3(&(rwb_axis.normalize() * ang));
    // Rcw = Rcb * Rbw ; tcw = Rcb*(-Rbw*twb) + tcb
    let rcb = rbc.transpose();
    let tcb = -rcb * tbc;
    let rbw = rwb.transpose();
    let tbw = -rbw * twb;
    let rcw = rcb * rbw;
    let tcw = rcb * tbw + tcb;
    ImuCamPose::new(
        vec![rcw],
        vec![tcw],
        vec![rbc],
        vec![tbc],
        47.0 * 0.01 * 458.654,
        vec![camera()],
    )
}

fn synthetic_preint() -> Arc<Preintegrated> {
    let calib = Calib::from_params(Isometry3::identity(), 1.7e-4, 2.0e-3, 1.9e-5, 3.0e-3);
    let bias = Bias::from_params(0.01, -0.02, 0.015, 0.001, -0.0015, 0.0008);
    let mut p = Preintegrated::from_bias_and_calib(&bias, &calib);
    let dt = 0.005f32;
    for i in 0..40 {
        let t = i as f32 * dt;
        let acc = Vector3::new(0.3 * t.cos(), 0.2 * t.sin(), 9.81 + 0.1 * t);
        let ang = Vector3::new(0.05 * t.sin(), 0.04 * t.cos(), 0.03);
        p.integrate_new_measurement(acc, ang, dt);
    }
    Arc::new(p)
}

#[test]
fn edge_mono_jacobian() {
    let verts: Vec<Box<dyn Vertex>> = vec![
        Box::new(VertexSBAPointXYZ::new(Vector3::new(0.4, -0.3, 5.0))),
        Box::new(VertexPose::new(make_pose(
            Vector3::new(0.2, 1.0, 0.1),
            0.15,
            Vector3::new(0.1, -0.05, 0.02),
        ))),
    ];
    let mut e = EdgeMono::new(0, 1, 0);
    e.set_measurement(Vector2::new(370.0, 250.0));
    check(verts, Box::new(e), "EdgeMono");
}

#[test]
fn edge_stereo_jacobian() {
    let verts: Vec<Box<dyn Vertex>> = vec![
        Box::new(VertexSBAPointXYZ::new(Vector3::new(-0.6, 0.2, 6.0))),
        Box::new(VertexPose::new(make_pose(
            Vector3::new(-0.3, 0.8, 0.2),
            0.2,
            Vector3::new(-0.1, 0.07, 0.05),
        ))),
    ];
    let mut e = EdgeStereo::new(0, 1, 0);
    e.set_measurement(Vector3::new(360.0, 255.0, 320.0));
    check(verts, Box::new(e), "EdgeStereo");
}

#[test]
fn edge_inertial_jacobian() {
    let preint = synthetic_preint();
    let verts: Vec<Box<dyn Vertex>> = vec![
        Box::new(VertexPose::new(make_pose(
            Vector3::new(0.1, 1.0, 0.2),
            0.1,
            Vector3::new(0.0, 0.0, 0.0),
        ))),
        Box::new(VertexVelocity::new(Vector3::new(0.5, -0.2, 0.1))),
        Box::new(VertexGyroBias::new(Vector3::new(0.001, -0.0015, 0.0008))),
        Box::new(VertexAccBias::new(Vector3::new(0.01, -0.02, 0.015))),
        Box::new(VertexPose::new(make_pose(
            Vector3::new(0.15, 0.9, 0.25),
            0.18,
            Vector3::new(0.3, -0.1, 0.05),
        ))),
        Box::new(VertexVelocity::new(Vector3::new(0.4, -0.25, 0.15))),
    ];
    let mut verts = verts;
    let mut e = EdgeInertial::new([0, 1, 2, 3, 4, 5], preint);
    // Preintegration is single-precision -> larger FD step, looser tolerance.
    check_with(&mut verts, &mut e, "EdgeInertial", 1e-4, 5e-3, &[]);
}

#[test]
fn edge_inertial_gs_jacobian() {
    let preint = synthetic_preint();
    let verts: Vec<Box<dyn Vertex>> = vec![
        Box::new(VertexPose::new(make_pose(
            Vector3::new(0.1, 1.0, 0.2),
            0.1,
            Vector3::new(0.0, 0.0, 0.0),
        ))),
        Box::new(VertexVelocity::new(Vector3::new(0.5, -0.2, 0.1))),
        Box::new(VertexGyroBias::new(Vector3::new(0.001, -0.0015, 0.0008))),
        Box::new(VertexAccBias::new(Vector3::new(0.01, -0.02, 0.015))),
        Box::new(VertexPose::new(make_pose(
            Vector3::new(0.15, 0.9, 0.25),
            0.18,
            Vector3::new(0.3, -0.1, 0.05),
        ))),
        Box::new(VertexVelocity::new(Vector3::new(0.4, -0.25, 0.15))),
        Box::new(VertexGDir::new(exp_so3(&Vector3::new(0.05, -0.03, 0.0)))),
        Box::new(VertexScale::new(1.1)),
    ];
    let mut verts = verts;
    let mut e = EdgeInertialGS::new([0, 1, 2, 3, 4, 5, 6, 7], preint);
    // Skip vertex 7 (scale): g2o's analytic Jacobian for the scale column omits
    // the `·s` chain factor of the `s·exp(u)` oplus (an upstream approximation),
    // so the true finite difference differs by exactly `s`. We match g2o.
    check_with(&mut verts, &mut e, "EdgeInertialGS", 1e-4, 5e-3, &[7]);
}

#[test]
fn edge_prior_pose_imu_jacobian() {
    let c = ConstraintPoseIMU::new(
        Matrix3::identity(),
        Vector3::new(0.1, 0.2, 0.3),
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::zeros(),
        Vector3::zeros(),
        nalgebra::SMatrix::<f64, 15, 15>::identity(),
    );
    let verts: Vec<Box<dyn Vertex>> = vec![
        Box::new(VertexPose::new(make_pose(
            Vector3::new(0.2, 1.0, 0.1),
            0.12,
            Vector3::new(0.1, 0.2, 0.3),
        ))),
        Box::new(VertexVelocity::new(Vector3::new(0.5, -0.2, 0.1))),
        Box::new(VertexGyroBias::new(Vector3::new(0.001, -0.0015, 0.0008))),
        Box::new(VertexAccBias::new(Vector3::new(0.01, -0.02, 0.015))),
    ];
    let e = EdgePriorPoseImu::new([0, 1, 2, 3], &c);
    check(verts, Box::new(e), "EdgePriorPoseImu");
}
