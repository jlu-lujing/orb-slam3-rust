//! Built on the focused mini-g2o in [`crate::g2o_core`] and the vertex/edge
//! types in [`crate::optimizable_types`]. Each routine mirrors the upstream
//! structure (graph assembly + Levenberg–Marquardt) closely enough to reproduce
//! its numerics.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use nalgebra::{Isometry3, Matrix2, Matrix3, Vector2, Vector3};
use opencv::prelude::*;

use crate::camera_models::GeometricCamera;
use crate::frame::Frame;
use crate::g2o_core::EdgeLinearization;
use crate::g2o_core::{Edge, SE3Quat, Sim3, SparseOptimizer, Vertex};
use crate::g2o_types::{
    ConstraintPoseIMU, Edge4DoF, EdgeBiasPrior, EdgeBiasRW, EdgeInertial, EdgeInertialGS, EdgeMono,
    EdgeMonoOnlyPose, EdgePriorPoseImu, EdgeStereo, EdgeStereoOnlyPose, ImuCamPose, VertexAccBias,
    VertexGDir, VertexGyroBias, VertexPose, VertexPose4DoF, VertexScale, VertexVelocity,
};
use crate::imu_types::{Bias, Preintegrated};
use crate::key_frame::KeyFrame;
use crate::map::Map;
use crate::map_point::MapPoint;
use crate::optimizable_types::{
    EdgeInverseSim3ProjectXYZ, EdgeSE3ProjectXYZ, EdgeSE3ProjectXYZOnlyPose,
    EdgeSE3ProjectXYZOnlyPoseToBody, EdgeSE3ProjectXYZToBody, EdgeSim3, EdgeSim3ProjectXYZ,
    EdgeStereoSE3ProjectXYZ, EdgeStereoSE3ProjectXYZOnlyPose, VertexSBAPointXYZ, VertexSE3Expmap,
    VertexSim3Expmap,
};

/// χ² thresholds (95% for 2 and 3 DoF) used for outlier classification.
const CHI2_MONO: f64 = 5.991;
const CHI2_STEREO: f64 = 7.815;

/// One map-point observation fed to [`pose_optimization_core`]. Mirrors the
/// per-keypoint branch in `Optimizer::PoseOptimization`.
pub enum PoseObs {
    /// Monocular observation through the (generic) left camera.
    Mono {
        xw: Vector3<f64>,
        obs: Vector2<f64>,
        inv_sigma2: f64,
        /// Index into the frame's outlier flag array.
        idx: usize,
    },
    /// Stereo observation (u, v, u_right) with explicit intrinsics.
    Stereo {
        xw: Vector3<f64>,
        obs: Vector3<f64>,
        inv_sigma2: f64,
        idx: usize,
    },
    /// Right-camera observation through the body→right transform `m_trl`.
    MonoBody {
        xw: Vector3<f64>,
        obs: Vector2<f64>,
        inv_sigma2: f64,
        m_trl: SE3Quat,
        idx: usize,
    },
}

impl PoseObs {
    fn idx(&self) -> usize {
        match self {
            PoseObs::Mono { idx, .. }
            | PoseObs::Stereo { idx, .. }
            | PoseObs::MonoBody { idx, .. } => *idx,
        }
    }
}

/// Records an added edge so it can be reclassified between optimization passes.
struct EdgeRec {
    /// Index into the optimizer's edge list.
    ei: usize,
    /// Index into the frame outlier array.
    idx: usize,
    /// χ² threshold (mono vs stereo).
    thr: f64,
}

/// `Optimizer::PoseOptimization` (Optimizer.cc:814).
///
/// Motion-only bundle adjustment: optimizes the frame pose against its matched
/// map points (held fixed). Updates `frame.outlier` and the frame pose in place,
/// returning the number of inlier correspondences.
pub fn pose_optimization(frame: &mut Frame) -> i32 {
    // Gather observations from the frame (the C++ does this inline).
    let intr = &frame.constants.intrinsics;
    let (fx, fy, cx, cy) = (
        intr.fx as f64,
        intr.fy as f64,
        intr.cx as f64,
        intr.cy as f64,
    );
    let bf = frame.b_fx as f64;
    let has_rig = frame.camera2.is_some();
    let n_left = frame.n_left.unwrap_or(0);
    let m_trl = if has_rig {
        Some(SE3Quat::from_isometry_f32(&frame.get_relative_pose_trl()))
    } else {
        None
    };

    let mut observations: Vec<PoseObs> = Vec::with_capacity(frame.n);
    for i in 0..frame.n {
        let Some(mp) = frame.map_points[i].clone() else {
            continue;
        };
        let xw = mp.get_world_pos().cast::<f64>();
        frame.outlier[i] = false;

        if !has_rig {
            let kp = &frame.keys_un.as_ref().unwrap()[i];
            let inv_sigma2 = frame.inv_level_sigma2[kp.octave() as usize] as f64;
            if frame.u_right[i] < 0.0 {
                observations.push(PoseObs::Mono {
                    xw,
                    obs: Vector2::new(kp.pt().x as f64, kp.pt().y as f64),
                    inv_sigma2,
                    idx: i,
                });
            } else {
                observations.push(PoseObs::Stereo {
                    xw,
                    obs: Vector3::new(kp.pt().x as f64, kp.pt().y as f64, frame.u_right[i] as f64),
                    inv_sigma2,
                    idx: i,
                });
            }
        } else if i < n_left {
            let kp = &frame.keys[i];
            let inv_sigma2 = frame.inv_level_sigma2[kp.octave() as usize] as f64;
            observations.push(PoseObs::Mono {
                xw,
                obs: Vector2::new(kp.pt().x as f64, kp.pt().y as f64),
                inv_sigma2,
                idx: i,
            });
        } else {
            let kp = &frame.keys_right.as_ref().unwrap()[i - n_left];
            let inv_sigma2 = frame.inv_level_sigma2[kp.octave() as usize] as f64;
            observations.push(PoseObs::MonoBody {
                xw,
                obs: Vector2::new(kp.pt().x as f64, kp.pt().y as f64),
                inv_sigma2,
                m_trl: m_trl.unwrap(),
                idx: i,
            });
        }
    }

    let left_camera = frame.camera.clone();
    let right_camera = frame.camera2.clone();

    let (pose, n_inliers, outliers) = pose_optimization_core(
        frame.get_pose(),
        &left_camera,
        right_camera.as_ref(),
        fx,
        fy,
        cx,
        cy,
        bf,
        &observations,
    );

    for (idx, is_out) in outliers {
        frame.outlier[idx] = is_out;
    }
    frame.set_pose(pose);
    n_inliers
}

// ===========================================================================
// BundleAdjustment / GlobalBundleAdjustment
// ===========================================================================

/// A keyframe input to [`bundle_adjustment_core`].
pub struct BaKeyframe {
    /// Camera pose `Tcw`.
    pub pose: Isometry3<f32>,
    /// Whether the pose is held fixed (e.g. the origin/init keyframe).
    pub fixed: bool,
    /// Left (generic) camera model.
    pub camera: Arc<dyn GeometricCamera>,
    /// Explicit intrinsics (for stereo edges).
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub bf: f64,
}

/// One observation of a point in a keyframe.
pub enum BaObs {
    /// Mono observation through the keyframe's left camera.
    Mono {
        kf: usize,
        mp: usize,
        obs: Vector2<f64>,
        inv_sigma2: f64,
    },
    /// Stereo observation (u, v, u_right).
    Stereo {
        kf: usize,
        mp: usize,
        obs: Vector3<f64>,
        inv_sigma2: f64,
    },
}

/// Core of [`bundle_adjustment`], decoupled from `Map`/`KeyFrame`/`MapPoint`
/// for testing. Optimizes keyframe poses and point positions jointly
/// (full bundle adjustment, landmarks Schur-marginalized).
///
/// Returns `(optimized poses, optimized points)`.
pub fn bundle_adjustment_core(
    keyframes: &[BaKeyframe],
    points: &[Vector3<f64>],
    observations: &[BaObs],
    n_iterations: i32,
    robust: bool,
) -> (Vec<Isometry3<f32>>, Vec<Vector3<f64>>) {
    let mut optimizer = SparseOptimizer::new();

    let th_huber_2d = 5.99_f64.sqrt();
    let th_huber_3d = 7.815_f64.sqrt();

    // KeyFrame pose vertices (added first => ordered before points).
    let mut kf_vertex = Vec::with_capacity(keyframes.len());
    for kf in keyframes {
        let mut v = VertexSE3Expmap::new(SE3Quat::from_isometry_f32(&kf.pose));
        v.set_fixed(kf.fixed);
        kf_vertex.push(optimizer.add_vertex(Box::new(v)));
    }

    // MapPoint vertices (marginalized).
    let mut mp_vertex = Vec::with_capacity(points.len());
    for p in points {
        let v = VertexSBAPointXYZ::new(*p);
        mp_vertex.push(optimizer.add_vertex(Box::new(v)));
    }

    for ob in observations {
        match ob {
            BaObs::Mono {
                kf,
                mp,
                obs,
                inv_sigma2,
            } => {
                let mut e = EdgeSE3ProjectXYZ::new(
                    mp_vertex[*mp],
                    kf_vertex[*kf],
                    keyframes[*kf].camera.clone(),
                );
                e.set_measurement(*obs);
                e.set_information(Matrix2::identity() * *inv_sigma2);
                if robust {
                    e.set_robust_kernel(Some(th_huber_2d));
                }
                optimizer.add_edge(Box::new(e));
            }
            BaObs::Stereo {
                kf,
                mp,
                obs,
                inv_sigma2,
            } => {
                let k = &keyframes[*kf];
                let mut e = EdgeStereoSE3ProjectXYZ::new(
                    mp_vertex[*mp],
                    kf_vertex[*kf],
                    k.fx,
                    k.fy,
                    k.cx,
                    k.cy,
                    k.bf,
                );
                e.set_measurement(*obs);
                e.set_information(Matrix3::identity() * *inv_sigma2);
                if robust {
                    e.set_robust_kernel(Some(th_huber_3d));
                }
                optimizer.add_edge(Box::new(e));
            }
        }
    }

    optimizer.initialize_optimization(0);
    optimizer.optimize(n_iterations);

    let out_poses = kf_vertex
        .iter()
        .map(|&vi| {
            optimizer
                .vertex(vi)
                .as_any()
                .downcast_ref::<VertexSE3Expmap>()
                .unwrap()
                .estimate()
                .to_isometry_f32()
        })
        .collect();
    let out_points = mp_vertex
        .iter()
        .map(|&vi| {
            optimizer
                .vertex(vi)
                .as_any()
                .downcast_ref::<VertexSBAPointXYZ>()
                .unwrap()
                .estimate()
        })
        .collect();
    (out_poses, out_points)
}

/// `Optimizer::GlobalBundleAdjustemnt` (Optimizer.cc:52).
///
/// Runs full [`bundle_adjustment`] over every keyframe and map point in `map`.
pub fn global_bundle_adjustment(
    map: &Map,
    n_iterations: i32,
    stop_flag: Option<Arc<AtomicBool>>,
    loop_kf: u64,
    robust: bool,
) {
    let kfs = map.get_all_key_frames();
    let mps = map.get_all_map_points();
    bundle_adjustment(map, &kfs, &mps, n_iterations, stop_flag, loop_kf, robust);
}

/// `Optimizer::BundleAdjustment` (Optimizer.cc:60).
///
/// Joint optimization of the given keyframe poses and map-point positions.
/// When `loop_kf` equals the map origin keyframe id the optimized values are
/// written straight back; otherwise they are staged in the keyframes'/points'
/// global-BA fields (`mTcwGBA` / `mPosGBA`) for a later loop-closing merge.
pub fn bundle_adjustment(
    map: &Map,
    kfs: &[Arc<KeyFrame>],
    mps: &[Arc<MapPoint>],
    n_iterations: i32,
    stop_flag: Option<Arc<AtomicBool>>,
    loop_kf: u64,
    robust: bool,
) {
    if kfs.is_empty() {
        return;
    }
    let mut optimizer = SparseOptimizer::new();
    if let Some(flag) = stop_flag {
        optimizer.set_force_stop_flag(flag);
    }

    let init_kf_id = map.get_init_kf_id();
    let origin_kf_id = map.get_origin_kf().map(|kf| kf.id);

    let th_huber_2d = 5.99_f64.sqrt();
    let th_huber_3d = 7.815_f64.sqrt();

    // --- KeyFrame vertices ---
    let mut max_kf_id: u64 = 0;
    let mut kf_vertex: HashMap<u64, usize> = HashMap::new();
    for kf in kfs {
        if kf.is_bad() {
            continue;
        }
        let mut v = VertexSE3Expmap::new(SE3Quat::from_isometry_f32(&kf.get_pose()));
        v.set_fixed(kf.id == init_kf_id);
        let vi = optimizer.add_vertex(Box::new(v));
        kf_vertex.insert(kf.id, vi);
        max_kf_id = max_kf_id.max(kf.id);
    }

    // --- MapPoint vertices + reprojection edges ---
    let mut mp_vertex: HashMap<usize, usize> = HashMap::new();
    let mut not_included = vec![false; mps.len()];

    for (i, mp) in mps.iter().enumerate() {
        if mp.is_bad() {
            continue;
        }
        let mut v = VertexSBAPointXYZ::new(mp.get_world_pos().cast::<f64>());
        v.set_marginalized(true);
        let mp_vi = optimizer.add_vertex(Box::new(v));
        mp_vertex.insert(mp.id, mp_vi);

        let mut n_edges = 0;
        for (kf, (left, right)) in mp.get_observations() {
            if kf.is_bad() || kf.id > max_kf_id {
                continue;
            }
            let Some(&kf_vi) = kf_vertex.get(&kf.id) else {
                continue;
            };
            n_edges += 1;

            let left = left as i64;
            if left != -1 && kf.u_right[left as usize] < 0.0 {
                // Mono observation.
                let kp = &kf.keys_un[left as usize];
                let inv_sigma2 = kf.inv_level_sigma2[kp.octave() as usize] as f64;
                let mut e = EdgeSE3ProjectXYZ::new(mp_vi, kf_vi, kf.camera.clone());
                e.set_measurement(Vector2::new(kp.pt().x as f64, kp.pt().y as f64));
                e.set_information(Matrix2::identity() * inv_sigma2);
                if robust {
                    e.set_robust_kernel(Some(th_huber_2d));
                }
                optimizer.add_edge(Box::new(e));
            } else if left != -1 && kf.u_right[left as usize] >= 0.0 {
                // Stereo observation.
                let kp = &kf.keys_un[left as usize];
                let kp_ur = kf.u_right[left as usize] as f64;
                let inv_sigma2 = kf.inv_level_sigma2[kp.octave() as usize] as f64;
                let mut e = EdgeStereoSE3ProjectXYZ::new(
                    mp_vi,
                    kf_vi,
                    kf.fx as f64,
                    kf.fy as f64,
                    kf.cx as f64,
                    kf.cy as f64,
                    kf.bf as f64,
                );
                e.set_measurement(Vector3::new(kp.pt().x as f64, kp.pt().y as f64, kp_ur));
                e.set_information(Matrix3::identity() * inv_sigma2);
                if robust {
                    e.set_robust_kernel(Some(th_huber_3d));
                }
                optimizer.add_edge(Box::new(e));
            }

            // Right-camera (fisheye rig) observation.
            if let Some(cam2) = &kf.camera2 {
                let right = right as i64;
                if let Some(keys_right) = &kf.keys_right {
                    if right != -1 && (right as usize) < keys_right.len() {
                        let n_left = kf.n_left.unwrap_or(0) as i64;
                        let kp = &keys_right[(right - n_left) as usize];
                        let inv_sigma2 = kf.inv_level_sigma2[kp.octave() as usize] as f64;
                        let m_trl = SE3Quat::from_isometry_f32(&kf.get_relative_pose_trl());
                        let mut e = EdgeSE3ProjectXYZToBody::new(mp_vi, kf_vi, cam2.clone(), m_trl);
                        e.set_measurement(Vector2::new(kp.pt().x as f64, kp.pt().y as f64));
                        e.set_information(Matrix2::identity() * inv_sigma2);
                        if robust {
                            e.set_robust_kernel(Some(th_huber_2d));
                        }
                        optimizer.add_edge(Box::new(e));
                    }
                }
            }
        }

        if n_edges == 0 {
            // No constraints: drop the (now isolated) vertex from optimization.
            optimizer.set_vertex_fixed(mp_vi, true);
            not_included[i] = true;
        }
    }

    // Optimize!
    optimizer.initialize_optimization(0);
    optimizer.optimize(n_iterations);

    // --- Recover keyframe poses ---
    let is_origin = Some(loop_kf) == origin_kf_id;
    for kf in kfs {
        if kf.is_bad() {
            continue;
        }
        let Some(&vi) = kf_vertex.get(&kf.id) else {
            continue;
        };
        let pose = optimizer
            .vertex(vi)
            .as_any()
            .downcast_ref::<VertexSE3Expmap>()
            .unwrap()
            .estimate()
            .to_isometry_f32();
        if is_origin {
            kf.set_pose(pose);
        } else {
            kf.set_tcw_gba(pose, loop_kf as i64);
        }
    }

    // --- Recover point positions ---
    for (i, mp) in mps.iter().enumerate() {
        if not_included[i] || mp.is_bad() {
            continue;
        }
        let Some(&vi) = mp_vertex.get(&mp.id) else {
            continue;
        };
        let pos = optimizer
            .vertex(vi)
            .as_any()
            .downcast_ref::<VertexSBAPointXYZ>()
            .unwrap()
            .estimate()
            .cast::<f32>();
        if is_origin {
            mp.set_world_pos(pos);
            mp.update_normal_and_depth();
        } else {
            mp.set_pos_gba(pos, loop_kf as i64);
        }
    }
}

/// `Optimizer::LocalBundleAdjustment` (non-inertial, Optimizer.cc:1116).
///
/// Optimizes the current keyframe and its covisible neighbours ("local"), with
/// keyframes that observe the local map points but are not local held fixed.
/// Map points seen by the local keyframes are optimized. Outlier observations
/// are erased afterwards. Returns `(num_fixed_kf, num_opt_kf, num_mps,
/// num_edges)`.
pub fn local_bundle_adjustment(
    kf: &Arc<KeyFrame>,
    stop_flag: Option<Arc<AtomicBool>>,
    map: &Map,
) -> (usize, usize, usize, usize) {
    let current_map_id = map.get_id();
    let same_map = |k: &Arc<KeyFrame>| k.get_map().map(|m| m.get_id()) == Some(current_map_id);

    // --- Local keyframes: current + covisible ---
    let mut local_kfs: Vec<Arc<KeyFrame>> = vec![kf.clone()];
    let mut local_kf_ids: HashSet<u64> = HashSet::from([kf.id]);
    for nb in kf.get_vector_covisible_keyframes() {
        if local_kf_ids.insert(nb.id) && !nb.is_bad() && same_map(&nb) {
            local_kfs.push(nb);
        }
    }

    // --- Local map points (seen by local keyframes) ---
    let init_kf_id = map.get_init_kf_id();
    let mut num_fixed_kf = 0usize;
    let mut local_mps: Vec<Arc<MapPoint>> = Vec::new();
    let mut local_mp_ids: HashSet<usize> = HashSet::new();
    for lkf in &local_kfs {
        if lkf.id == init_kf_id {
            num_fixed_kf = 1;
        }
        for mp in lkf.get_map_point_matches().into_iter().flatten() {
            if !mp.is_bad()
                && mp.get_map().map(|m| m.get_id()) == Some(current_map_id)
                && local_mp_ids.insert(mp.id)
            {
                local_mps.push(mp);
            }
        }
    }

    // --- Fixed cameras: see local map points but are not local ---
    let mut fixed_kfs: Vec<Arc<KeyFrame>> = Vec::new();
    let mut fixed_kf_ids: HashSet<u64> = HashSet::new();
    for mp in &local_mps {
        for (obs_kf, _) in mp.get_observations() {
            if !local_kf_ids.contains(&obs_kf.id)
                && !fixed_kf_ids.contains(&obs_kf.id)
                && !obs_kf.is_bad()
                && same_map(&obs_kf)
            {
                fixed_kf_ids.insert(obs_kf.id);
                fixed_kfs.push(obs_kf);
            }
        }
    }
    num_fixed_kf += fixed_kfs.len();
    if num_fixed_kf == 0 {
        return (0, local_kfs.len(), local_mps.len(), 0);
    }

    // --- Optimizer ---
    let mut optimizer = SparseOptimizer::new();
    if map.is_inertial() {
        optimizer.set_user_lambda_init(100.0);
    }
    if let Some(flag) = stop_flag.clone() {
        optimizer.set_force_stop_flag(flag);
    }

    let mut kf_vertex: HashMap<u64, usize> = HashMap::new();
    for lkf in &local_kfs {
        let mut v = VertexSE3Expmap::new(SE3Quat::from_isometry_f32(&lkf.get_pose()));
        v.set_fixed(lkf.id == init_kf_id);
        kf_vertex.insert(lkf.id, optimizer.add_vertex(Box::new(v)));
    }
    for fkf in &fixed_kfs {
        let mut v = VertexSE3Expmap::new(SE3Quat::from_isometry_f32(&fkf.get_pose()));
        v.set_fixed(true);
        kf_vertex.insert(fkf.id, optimizer.add_vertex(Box::new(v)));
    }
    let num_opt_kf = local_kfs.len();

    let th_huber_mono = 5.991_f64.sqrt();
    let th_huber_stereo = 7.815_f64.sqrt();

    // edge record: (edge index, threshold, kf, mp)
    struct Rec {
        ei: usize,
        thr: f64,
        kf: Arc<KeyFrame>,
        mp: Arc<MapPoint>,
    }
    let mut recs: Vec<Rec> = Vec::new();
    let mut mp_vertex: HashMap<usize, usize> = HashMap::new();

    for mp in &local_mps {
        let mut v = VertexSBAPointXYZ::new(mp.get_world_pos().cast::<f64>());
        v.set_marginalized(true);
        let mp_vi = optimizer.add_vertex(Box::new(v));
        mp_vertex.insert(mp.id, mp_vi);

        for (obs_kf, (left, right)) in mp.get_observations() {
            if obs_kf.is_bad() || !same_map(&obs_kf) {
                continue;
            }
            let Some(&kf_vi) = kf_vertex.get(&obs_kf.id) else {
                continue;
            };
            let left = left as i64;
            if left != -1 && obs_kf.u_right[left as usize] < 0.0 {
                let kp = &obs_kf.keys_un[left as usize];
                let inv_sigma2 = obs_kf.inv_level_sigma2[kp.octave() as usize] as f64;
                let mut e = EdgeSE3ProjectXYZ::new(mp_vi, kf_vi, obs_kf.camera.clone());
                e.set_measurement(Vector2::new(kp.pt().x as f64, kp.pt().y as f64));
                e.set_information(Matrix2::identity() * inv_sigma2);
                e.set_robust_kernel(Some(th_huber_mono));
                let ei = optimizer.add_edge(Box::new(e));
                recs.push(Rec {
                    ei,
                    thr: CHI2_MONO,
                    kf: obs_kf.clone(),
                    mp: mp.clone(),
                });
            } else if left != -1 && obs_kf.u_right[left as usize] >= 0.0 {
                let kp = &obs_kf.keys_un[left as usize];
                let kp_ur = obs_kf.u_right[left as usize] as f64;
                let inv_sigma2 = obs_kf.inv_level_sigma2[kp.octave() as usize] as f64;
                let mut e = EdgeStereoSE3ProjectXYZ::new(
                    mp_vi,
                    kf_vi,
                    obs_kf.fx as f64,
                    obs_kf.fy as f64,
                    obs_kf.cx as f64,
                    obs_kf.cy as f64,
                    obs_kf.bf as f64,
                );
                e.set_measurement(Vector3::new(kp.pt().x as f64, kp.pt().y as f64, kp_ur));
                e.set_information(Matrix3::identity() * inv_sigma2);
                e.set_robust_kernel(Some(th_huber_stereo));
                let ei = optimizer.add_edge(Box::new(e));
                recs.push(Rec {
                    ei,
                    thr: CHI2_STEREO,
                    kf: obs_kf.clone(),
                    mp: mp.clone(),
                });
            }

            if let Some(cam2) = &obs_kf.camera2 {
                let right = right as i64;
                if right != -1 {
                    if let Some(keys_right) = &obs_kf.keys_right {
                        let n_left = obs_kf.n_left.unwrap_or(0) as i64;
                        let kp = &keys_right[(right - n_left) as usize];
                        let inv_sigma2 = obs_kf.inv_level_sigma2[kp.octave() as usize] as f64;
                        let m_trl = SE3Quat::from_isometry_f32(&obs_kf.get_relative_pose_trl());
                        let mut e = EdgeSE3ProjectXYZToBody::new(mp_vi, kf_vi, cam2.clone(), m_trl);
                        e.set_measurement(Vector2::new(kp.pt().x as f64, kp.pt().y as f64));
                        e.set_information(Matrix2::identity() * inv_sigma2);
                        e.set_robust_kernel(Some(th_huber_mono));
                        let ei = optimizer.add_edge(Box::new(e));
                        recs.push(Rec {
                            ei,
                            thr: CHI2_MONO,
                            kf: obs_kf.clone(),
                            mp: mp.clone(),
                        });
                    }
                }
            }
        }
    }
    let num_edges = recs.len();

    if stop_flag.as_ref().is_some_and(|f| f.load(Ordering::SeqCst)) {
        return (num_fixed_kf, num_opt_kf, local_mps.len(), num_edges);
    }

    optimizer.initialize_optimization(0);
    optimizer.optimize(10);

    // --- Cull outlier observations ---
    let mut to_erase: Vec<(Arc<KeyFrame>, Arc<MapPoint>)> = Vec::new();
    for rec in &recs {
        if rec.mp.is_bad() {
            continue;
        }
        if optimizer.edge(rec.ei).chi2() > rec.thr || !optimizer.edge_depth_positive(rec.ei) {
            to_erase.push((rec.kf.clone(), rec.mp.clone()));
        }
    }
    for (ekf, emp) in &to_erase {
        ekf.erase_map_point_match(emp);
        emp.erase_observation(ekf);
    }

    // --- Recover optimized keyframe poses ---
    for lkf in &local_kfs {
        let vi = kf_vertex[&lkf.id];
        let pose = optimizer
            .vertex(vi)
            .as_any()
            .downcast_ref::<VertexSE3Expmap>()
            .unwrap()
            .estimate()
            .to_isometry_f32();
        lkf.set_pose(pose);
    }
    // --- Recover optimized point positions ---
    for mp in &local_mps {
        let vi = mp_vertex[&mp.id];
        let pos = optimizer
            .vertex(vi)
            .as_any()
            .downcast_ref::<VertexSBAPointXYZ>()
            .unwrap()
            .estimate()
            .cast::<f32>();
        mp.set_world_pos(pos);
        mp.update_normal_and_depth();
    }

    map.increase_change_index();
    (num_fixed_kf, num_opt_kf, local_mps.len(), num_edges)
}

// ===========================================================================
// OptimizeSim3
// ===========================================================================

/// One Sim3 correspondence: a map point known in both keyframes, with its
/// position expressed in each camera frame and its observation in each image.
pub struct Sim3Correspondence {
    /// Point of KF1 in camera-1 frame (`P3D1c`).
    pub x1c: Vector3<f64>,
    /// Point of KF2 in camera-2 frame (`P3D2c`).
    pub x2c: Vector3<f64>,
    /// Observation in KF1 image.
    pub obs1: Vector2<f64>,
    /// Observation in KF2 image.
    pub obs2: Vector2<f64>,
    pub inv_sigma1: f64,
    pub inv_sigma2: f64,
}

/// Core of [`optimize_sim3`], decoupled from `KeyFrame` for testing.
///
/// Optimizes the relative similarity `S12` against bidirectional reprojection
/// edges, with two rounds and outlier rejection. Returns
/// `(n_inliers, optimized S12, per-correspondence inlier flags)`.
pub fn optimize_sim3_core(
    s12_init: Sim3,
    camera1: &Arc<dyn GeometricCamera>,
    camera2: &Arc<dyn GeometricCamera>,
    corrs: &[Sim3Correspondence],
    th2: f64,
    fix_scale: bool,
) -> (i32, Sim3, Vec<bool>) {
    let mut optimizer = SparseOptimizer::new();

    let v_sim3 = optimizer.add_vertex(Box::new(VertexSim3Expmap::new(s12_init, fix_scale)));
    let delta_huber = th2.sqrt();

    struct Pair {
        e12: usize,
        e21: usize,
        removed: bool,
    }
    let mut pairs: Vec<Pair> = Vec::with_capacity(corrs.len());

    for c in corrs {
        // Both points are fixed; only S12 is optimized.
        let mut vp1 = VertexSBAPointXYZ::new(c.x1c);
        vp1.set_marginalized(false);
        let v_p1 = optimizer.add_vertex(Box::new(vp1));
        optimizer.set_vertex_fixed(v_p1, true);

        let mut vp2 = VertexSBAPointXYZ::new(c.x2c);
        vp2.set_marginalized(false);
        let v_p2 = optimizer.add_vertex(Box::new(vp2));
        optimizer.set_vertex_fixed(v_p2, true);

        // e12: x1 = S12 * X2  (project X2 into camera 1)
        let mut e12 = EdgeSim3ProjectXYZ::new(v_p2, v_sim3, camera1.clone());
        e12.set_measurement(c.obs1);
        e12.set_information(Matrix2::identity() * c.inv_sigma1);
        e12.set_robust_kernel(Some(delta_huber));
        let e12i = optimizer.add_edge(Box::new(e12));

        // e21: x2 = S12⁻¹ * X1  (project X1 into camera 2)
        let mut e21 = EdgeInverseSim3ProjectXYZ::new(v_p1, v_sim3, camera2.clone());
        e21.set_measurement(c.obs2);
        e21.set_information(Matrix2::identity() * c.inv_sigma2);
        e21.set_robust_kernel(Some(delta_huber));
        let e21i = optimizer.add_edge(Box::new(e21));

        pairs.push(Pair {
            e12: e12i,
            e21: e21i,
            removed: false,
        });
    }

    let n_correspondences = pairs.len() as i32;

    optimizer.initialize_optimization(0);
    optimizer.optimize(5);

    // Check inliers after the first round.
    let mut n_bad = 0;
    for p in &mut pairs {
        let chi12 = optimizer.edge(p.e12).chi2();
        let chi21 = optimizer.edge(p.e21).chi2();
        if chi12 > th2 || chi21 > th2 {
            optimizer.edge_mut(p.e12).set_level(1);
            optimizer.edge_mut(p.e21).set_level(1);
            p.removed = true;
            n_bad += 1;
        } else {
            optimizer.edge_mut(p.e12).set_robust_kernel(None);
            optimizer.edge_mut(p.e21).set_robust_kernel(None);
        }
    }

    let n_more = if n_bad > 0 { 10 } else { 5 };
    if n_correspondences - n_bad < 10 {
        let recov = sim3_estimate(&optimizer, v_sim3);
        return (0, recov, pairs.iter().map(|p| !p.removed).collect());
    }

    optimizer.initialize_optimization(0);
    optimizer.optimize(n_more);

    let mut n_in = 0;
    for p in &mut pairs {
        if p.removed {
            continue;
        }
        optimizer.compute_edge_error(p.e12);
        optimizer.compute_edge_error(p.e21);
        if optimizer.edge(p.e12).chi2() > th2 || optimizer.edge(p.e21).chi2() > th2 {
            p.removed = true;
        } else {
            n_in += 1;
        }
    }

    let recov = sim3_estimate(&optimizer, v_sim3);
    (n_in, recov, pairs.iter().map(|p| !p.removed).collect())
}

/// `Optimizer::OptimizeSim3` (Optimizer.cc:2115).
///
/// Optimizes the relative similarity `s12` between `kf1` and `kf2` against their
/// matched map points (`matches1`, indexed by KF1 keypoint). Outlier matches
/// are set to `None`; `s12` is updated in place. Returns the inlier count.
#[allow(clippy::too_many_arguments)]
pub fn optimize_sim3(
    kf1: &Arc<KeyFrame>,
    kf2: &Arc<KeyFrame>,
    matches1: &mut [Option<Arc<MapPoint>>],
    s12: &mut Sim3,
    th2: f64,
    fix_scale: bool,
    all_points: bool,
) -> i32 {
    let r1w = kf1.get_rotation().cast::<f64>();
    let t1w = kf1.get_translation().cast::<f64>();
    let r2w = kf2.get_rotation().cast::<f64>();
    let t2w = kf2.get_translation().cast::<f64>();

    let mps1 = kf1.get_map_point_matches();

    let mut corrs: Vec<Sim3Correspondence> = Vec::new();
    let mut corr_index: Vec<usize> = Vec::new(); // KF1 keypoint index per corr

    for (i, m) in matches1.iter().enumerate() {
        let Some(mp2) = m else { continue };
        let Some(mp1) = mps1.get(i).and_then(|o| o.clone()) else {
            continue;
        };
        if mp1.is_bad() || mp2.is_bad() {
            continue;
        }
        let i2 = mp2.get_index_in_keyframe(kf2).0;
        if i2 < 0 && !all_points {
            continue;
        }

        let p3d1w = mp1.get_world_pos().cast::<f64>();
        let x1c = r1w * p3d1w + t1w;
        let p3d2w = mp2.get_world_pos().cast::<f64>();
        let x2c = r2w * p3d2w + t2w;
        if x2c[2] < 0.0 {
            continue;
        }

        let kp1 = &kf1.keys_un[i];
        let obs1 = Vector2::new(kp1.pt().x as f64, kp1.pt().y as f64);
        let inv_sigma1 = kf1.inv_level_sigma2[kp1.octave() as usize] as f64;

        let (obs2, inv_sigma2) = if i2 >= 0 {
            let kp2 = &kf2.keys_un[i2 as usize];
            (
                Vector2::new(kp2.pt().x as f64, kp2.pt().y as f64),
                kf2.inv_level_sigma2[kp2.octave() as usize] as f64,
            )
        } else {
            // Point not in KF2: synthesize the observation from the projection.
            let invz = 1.0 / x2c[2];
            let obs = Vector2::new(x2c[0] * invz, x2c[1] * invz);
            let lvl = mp2.track_scale_level.max(0) as usize;
            (obs, kf2.inv_level_sigma2[lvl] as f64)
        };

        corrs.push(Sim3Correspondence {
            x1c,
            x2c,
            obs1,
            obs2,
            inv_sigma1,
            inv_sigma2,
        });
        corr_index.push(i);
    }

    let (n_in, s12_opt, kept) =
        optimize_sim3_core(*s12, &kf1.camera, &kf2.camera, &corrs, th2, fix_scale);
    *s12 = s12_opt;
    for (ci, keep) in kept.iter().enumerate() {
        if !keep {
            matches1[corr_index[ci]] = None;
        }
    }
    n_in
}

// ===========================================================================
// Marginalize  (Schur complement of a Hessian block)
// ===========================================================================

/// `Optimizer::Marginalize` — marginalize the block `[start..=end]` of `H` by
/// Schur complement, returning a matrix with that block zeroed. Faithful port
/// of the upstream reorder + pseudo-inverse implementation.
pub fn marginalize(h: &nalgebra::DMatrix<f64>, start: usize, end: usize) -> nalgebra::DMatrix<f64> {
    use nalgebra::DMatrix;
    let total = h.ncols();
    let a = start;
    let b = end - start + 1;
    let c = total - (end + 1);

    // Reorder so the marginalized block is last.
    let mut hn = DMatrix::<f64>::zeros(total, total);
    if a > 0 {
        hn.view_mut((0, 0), (a, a))
            .copy_from(&h.view((0, 0), (a, a)));
        hn.view_mut((0, a + c), (a, b))
            .copy_from(&h.view((0, a), (a, b)));
        hn.view_mut((a + c, 0), (b, a))
            .copy_from(&h.view((a, 0), (b, a)));
    }
    if a > 0 && c > 0 {
        hn.view_mut((0, a), (a, c))
            .copy_from(&h.view((0, a + b), (a, c)));
        hn.view_mut((a, 0), (c, a))
            .copy_from(&h.view((a + b, 0), (c, a)));
    }
    if c > 0 {
        hn.view_mut((a, a), (c, c))
            .copy_from(&h.view((a + b, a + b), (c, c)));
        hn.view_mut((a, a + c), (c, b))
            .copy_from(&h.view((a + b, a), (c, b)));
        hn.view_mut((a + c, a), (b, c))
            .copy_from(&h.view((a, a + b), (b, c)));
    }
    hn.view_mut((a + c, a + c), (b, b))
        .copy_from(&h.view((a, a), (b, b)));

    // Schur complement using a pseudo-inverse of the marginalized block.
    let hb = hn.view((a + c, a + c), (b, b)).into_owned();
    let svd = hb.svd(true, true);
    let mut sv_inv = svd.singular_values;
    for i in 0..b {
        sv_inv[i] = if sv_inv[i] > 1e-6 {
            1.0 / sv_inv[i]
        } else {
            0.0
        };
    }
    let u = svd.u.unwrap();
    let vt = svd.v_t.unwrap();
    let inv_hb = vt.transpose() * DMatrix::from_diagonal(&sv_inv) * u.transpose();

    let ac = a + c;
    if ac > 0 {
        let top_right = hn.view((0, a + c), (ac, b)).into_owned();
        let bottom_left = hn.view((a + c, 0), (b, ac)).into_owned();
        let reduced = hn.view((0, 0), (ac, ac)) - top_right * &inv_hb * bottom_left;
        hn.view_mut((0, 0), (ac, ac)).copy_from(&reduced);
    }
    hn.view_mut((a + c, a + c), (b, b)).fill(0.0);
    hn.view_mut((0, a + c), (ac, b)).fill(0.0);
    hn.view_mut((a + c, 0), (b, ac)).fill(0.0);

    // Inverse reorder.
    let mut res = DMatrix::<f64>::zeros(total, total);
    if a > 0 {
        res.view_mut((0, 0), (a, a))
            .copy_from(&hn.view((0, 0), (a, a)));
        res.view_mut((0, a), (a, b))
            .copy_from(&hn.view((0, a + c), (a, b)));
        res.view_mut((a, 0), (b, a))
            .copy_from(&hn.view((a + c, 0), (b, a)));
    }
    if a > 0 && c > 0 {
        res.view_mut((0, a + b), (a, c))
            .copy_from(&hn.view((0, a), (a, c)));
        res.view_mut((a + b, 0), (c, a))
            .copy_from(&hn.view((a, 0), (c, a)));
    }
    if c > 0 {
        res.view_mut((a + b, a + b), (c, c))
            .copy_from(&hn.view((a, a), (c, c)));
        res.view_mut((a + b, a), (c, b))
            .copy_from(&hn.view((a, a + c), (c, b)));
        res.view_mut((a, a + b), (b, c))
            .copy_from(&hn.view((a + c, a), (b, c)));
    }
    res.view_mut((a, a), (b, b))
        .copy_from(&hn.view((a + c, a + c), (b, b)));
    res
}

// ===========================================================================
// InertialOptimization (gravity direction + scale)
// ===========================================================================

/// Per-keyframe IMU state for [`inertial_optimization_gs_core`]. Keyframes are
/// listed in temporal order; `preint` links each KF to its predecessor.
pub struct InertialKf {
    pub rwb: nalgebra::Matrix3<f64>,
    pub twb: Vector3<f64>,
    pub vel: Vector3<f64>,
    /// Preintegration from the previous keyframe to this one (`None` for the first).
    pub preint: Option<Arc<Preintegrated>>,
}

/// Build an [`ImuCamPose`] with the given body pose (camera params irrelevant
/// for inertial-only edges). Uses the `SetParam` relationship
/// `Rcw = Rwbᵀ`, `tcw = -Rwbᵀ·twb` with identity body↔camera extrinsics.
fn imu_cam_pose(
    rwb: nalgebra::Matrix3<f64>,
    twb: Vector3<f64>,
    cam: &Arc<dyn GeometricCamera>,
) -> ImuCamPose {
    let rcw = rwb.transpose();
    let tcw = -rcw * twb;
    ImuCamPose::new(
        vec![rcw],
        vec![tcw],
        vec![nalgebra::Matrix3::identity()],
        vec![Vector3::zeros()],
        0.0,
        vec![cam.clone()],
    )
}

/// Configuration for [`inertial_optimization_core`], covering all three
/// `Optimizer::InertialOptimization` overloads.
pub struct InertialOptConfig {
    pub fix_vel: bool,
    pub fix_bias: bool,
    pub fix_gdir: bool,
    pub fix_scale: bool,
    /// Bias prior information weights (`0` = no prior edge).
    pub prior_g: f64,
    pub prior_a: f64,
    /// Huber delta for `EdgeInertialGS` (`Some(1.0)` for the gravity+scale
    /// overload; `None` for the others).
    pub robust_delta: Option<f64>,
    /// Initial LM damping (`None` = auto).
    pub user_lambda: Option<f64>,
    /// Gauss-Newton instead of Levenberg-Marquardt.
    pub gauss_newton: bool,
    pub iterations: i32,
}

/// Result of [`inertial_optimization_core`].
pub struct InertialOptResult {
    pub rwg: nalgebra::Matrix3<f64>,
    pub scale: f64,
    pub bg: Vector3<f64>,
    pub ba: Vector3<f64>,
    pub vels: Vec<Vector3<f64>>,
}

/// Core of the IMU-initialization `Optimizer::InertialOptimization` routines
/// (Optimizer.cc:3042 / 3227 / 3389): keyframe poses fixed, an `EdgeInertialGS`
/// per consecutive pair links a single shared gyro/acc bias, gravity direction
/// and scale; velocities/biases/gravity/scale are optimized per the config,
/// with optional bias priors.
pub fn inertial_optimization_core(
    kfs: &[InertialKf],
    init_gyro: Vector3<f64>,
    init_acc: Vector3<f64>,
    rwg_init: nalgebra::Matrix3<f64>,
    scale_init: f64,
    cfg: &InertialOptConfig,
) -> InertialOptResult {
    let cam: Arc<dyn GeometricCamera> =
        Arc::new(crate::camera_models::pinhole::Pinhole::with_params(vec![
            1.0, 1.0, 0.0, 0.0,
        ]));

    let mut opt = SparseOptimizer::new();
    opt.set_gauss_newton(cfg.gauss_newton);
    if let Some(l) = cfg.user_lambda {
        opt.set_user_lambda_init(l);
    }

    let mut vpose = Vec::with_capacity(kfs.len());
    let mut vvel = Vec::with_capacity(kfs.len());
    for kf in kfs {
        let mut vp = VertexPose::new(imu_cam_pose(kf.rwb, kf.twb, &cam));
        vp.set_fixed(true);
        vpose.push(opt.add_vertex(Box::new(vp)));
        let mut vv = VertexVelocity::new(kf.vel);
        vv.set_fixed(cfg.fix_vel);
        vvel.push(opt.add_vertex(Box::new(vv)));
    }
    let mut vg = VertexGyroBias::new(init_gyro);
    vg.set_fixed(cfg.fix_bias);
    let vg = opt.add_vertex(Box::new(vg));
    let mut va = VertexAccBias::new(init_acc);
    va.set_fixed(cfg.fix_bias);
    let va = opt.add_vertex(Box::new(va));

    if cfg.prior_a != 0.0 {
        let mut e = EdgeBiasPrior::new_acc(va, Vector3::zeros());
        e.set_information(nalgebra::Matrix3::identity() * cfg.prior_a);
        opt.add_edge(Box::new(e));
    }
    if cfg.prior_g != 0.0 {
        let mut e = EdgeBiasPrior::new_gyro(vg, Vector3::zeros());
        e.set_information(nalgebra::Matrix3::identity() * cfg.prior_g);
        opt.add_edge(Box::new(e));
    }

    let mut vgdir = VertexGDir::new(rwg_init);
    vgdir.set_fixed(cfg.fix_gdir);
    let vgdir = opt.add_vertex(Box::new(vgdir));
    let mut vs = VertexScale::new(scale_init);
    vs.set_fixed(cfg.fix_scale);
    let vs = opt.add_vertex(Box::new(vs));

    for i in 1..kfs.len() {
        if let Some(preint) = &kfs[i].preint {
            let mut e = EdgeInertialGS::new(
                [
                    vpose[i - 1],
                    vvel[i - 1],
                    vg,
                    va,
                    vpose[i],
                    vvel[i],
                    vgdir,
                    vs,
                ],
                preint.clone(),
            );
            e.set_robust_kernel(cfg.robust_delta);
            opt.add_edge(Box::new(e));
        }
    }

    opt.initialize_optimization(0);
    opt.optimize(cfg.iterations);

    InertialOptResult {
        rwg: opt
            .vertex(vgdir)
            .as_any()
            .downcast_ref::<VertexGDir>()
            .unwrap()
            .estimate(),
        scale: opt
            .vertex(vs)
            .as_any()
            .downcast_ref::<VertexScale>()
            .unwrap()
            .estimate(),
        bg: opt
            .vertex(vg)
            .as_any()
            .downcast_ref::<VertexGyroBias>()
            .unwrap()
            .estimate(),
        ba: opt
            .vertex(va)
            .as_any()
            .downcast_ref::<VertexAccBias>()
            .unwrap()
            .estimate(),
        vels: vvel
            .iter()
            .map(|&vi| {
                opt.vertex(vi)
                    .as_any()
                    .downcast_ref::<VertexVelocity>()
                    .unwrap()
                    .estimate()
            })
            .collect(),
    }
}

/// Gather map keyframes (id order) as [`InertialKf`]s with preintegration links,
/// plus the front keyframe's gyro/acc bias. Returns the keyframe arcs too.
fn gather_inertial_kfs(
    map: &Map,
) -> (
    Vec<InertialKf>,
    Vec<Arc<KeyFrame>>,
    Vector3<f64>,
    Vector3<f64>,
) {
    let mut arcs = map.get_all_key_frames();
    arcs.sort_by_key(|k| k.id);
    let max_kf_id = map.get_max_kf_id();
    arcs.retain(|k| k.id <= max_kf_id && !k.is_bad());

    let mut kfs = Vec::with_capacity(arcs.len());
    for (i, kf) in arcs.iter().enumerate() {
        // Link to the previous chain keyframe only if it is the array predecessor.
        let preint = if i > 0 {
            match kf.get_prev_kf() {
                Some(p) if p.id == arcs[i - 1].id => kf.imu_preintegrated.clone(),
                _ => None,
            }
        } else {
            None
        };
        kfs.push(InertialKf {
            rwb: kf.get_imu_rotation().cast::<f64>(),
            twb: kf.get_imu_position().cast::<f64>(),
            vel: kf.get_velocity().cast::<f64>(),
            preint,
        });
    }
    let fb = arcs
        .first()
        .map(|k| k.get_imu_bias())
        .unwrap_or_else(Bias::empty);
    let front_gyro = Vector3::new(fb.bwx as f64, fb.bwy as f64, fb.bwz as f64);
    let front_acc = Vector3::new(fb.bax as f64, fb.bay as f64, fb.baz as f64);
    (kfs, arcs, front_gyro, front_acc)
}

/// Write back optimized velocities + a shared bias to the keyframes (with
/// reintegration when the gyro bias changed), as the inertial-init routines do.
fn write_back_inertial(arcs: &[Arc<KeyFrame>], result: &InertialOptResult) {
    let b = Bias::from_params(
        result.ba[0] as f32,
        result.ba[1] as f32,
        result.ba[2] as f32,
        result.bg[0] as f32,
        result.bg[1] as f32,
        result.bg[2] as f32,
    );
    for (i, kf) in arcs.iter().enumerate() {
        kf.set_velocity(result.vels[i].cast::<f32>());
        kf.set_new_bias(b);
    }
}

/// `Optimizer::InertialOptimization(Map, Rwg, scale, bg, ba, bMono, ...)`
/// (Optimizer.cc:3042) full overload: optimize velocities, shared bias, gravity
/// direction and (mono only) scale, with bias priors. Updates `(rwg, scale, bg,
/// ba)` and writes velocities + biases to the map.
#[allow(clippy::too_many_arguments)]
pub fn inertial_optimization(
    map: &Map,
    rwg: &mut nalgebra::Matrix3<f64>,
    scale: &mut f64,
    bg: &mut Vector3<f64>,
    ba: &mut Vector3<f64>,
    b_mono: bool,
    b_fixed_vel: bool,
    prior_g: f64,
    prior_a: f64,
) {
    let (kfs, arcs, fg, fa) = gather_inertial_kfs(map);
    if kfs.len() < 2 {
        return;
    }
    let cfg = InertialOptConfig {
        fix_vel: b_fixed_vel,
        fix_bias: b_fixed_vel,
        fix_gdir: false,
        fix_scale: !b_mono,
        prior_g,
        prior_a,
        robust_delta: None,
        user_lambda: if prior_g != 0.0 { Some(1e3) } else { None },
        gauss_newton: false,
        iterations: 200,
    };
    let r = inertial_optimization_core(&kfs, fg, fa, *rwg, *scale, &cfg);
    *rwg = r.rwg;
    *scale = r.scale;
    *bg = r.bg;
    *ba = r.ba;
    write_back_inertial(&arcs, &r);
}

/// `Optimizer::InertialOptimization(Map, bg, ba, ...)` (Optimizer.cc:3227)
/// bias-only overload: gravity fixed (identity), scale fixed; optimize
/// velocities + shared bias with priors.
pub fn inertial_optimization_bias(
    map: &Map,
    bg: &mut Vector3<f64>,
    ba: &mut Vector3<f64>,
    prior_g: f64,
    prior_a: f64,
) {
    let (kfs, arcs, fg, fa) = gather_inertial_kfs(map);
    if kfs.len() < 2 {
        return;
    }
    let cfg = InertialOptConfig {
        fix_vel: false,
        fix_bias: false,
        fix_gdir: true,
        fix_scale: true,
        prior_g,
        prior_a,
        robust_delta: None,
        user_lambda: Some(1e3),
        gauss_newton: false,
        iterations: 200,
    };
    let r = inertial_optimization_core(&kfs, fg, fa, nalgebra::Matrix3::identity(), 1.0, &cfg);
    *bg = r.bg;
    *ba = r.ba;
    write_back_inertial(&arcs, &r);
}

/// `Optimizer::InertialOptimization(Map, Rwg, scale)` gravity+scale overload
/// (Optimizer.cc:3389): all keyframe states fixed; estimate gravity & scale.
pub fn inertial_optimization_gravity_scale(
    map: &Map,
    rwg: &mut nalgebra::Matrix3<f64>,
    scale: &mut f64,
) {
    let (kfs, _arcs, fg, fa) = gather_inertial_kfs(map);
    if kfs.len() < 2 {
        return;
    }
    let (r, s) = inertial_optimization_gs_core(&kfs, fg, fa, *rwg, *scale);
    *rwg = r;
    *scale = s;
}

/// `Optimizer::InertialOptimization(Map, Rwg, scale)` (Optimizer.cc:3389):
/// gravity direction + scale, all keyframe states fixed.
pub fn inertial_optimization_gs_core(
    kfs: &[InertialKf],
    front_gyro: Vector3<f64>,
    front_acc: Vector3<f64>,
    rwg_init: nalgebra::Matrix3<f64>,
    scale_init: f64,
) -> (nalgebra::Matrix3<f64>, f64) {
    let r = inertial_optimization_core(
        kfs,
        front_gyro,
        front_acc,
        rwg_init,
        scale_init,
        &InertialOptConfig {
            fix_vel: true,
            fix_bias: true,
            fix_gdir: false,
            fix_scale: false,
            prior_g: 0.0,
            prior_a: 0.0,
            robust_delta: Some(1.0),
            user_lambda: None,
            gauss_newton: true,
            iterations: 10,
        },
    );
    (r.rwg, r.scale)
}

/// `Optimizer::LocalBundleAdjustment` 2nd overload (Optimizer.cc:3498): the
/// merge-welding visual BA. `adjust_kfs` are optimized, `fixed_kfs` held fixed,
/// over the map points observed by `main_kf`; outliers are culled.
pub fn local_bundle_adjustment_merge(
    main_kf: &Arc<KeyFrame>,
    adjust_kfs: &[Arc<KeyFrame>],
    fixed_kfs: &[Arc<KeyFrame>],
    stop_flag: Option<Arc<AtomicBool>>,
) {
    let Some(map) = main_kf.get_map() else { return };
    let mut optimizer = SparseOptimizer::new();
    if let Some(f) = stop_flag {
        optimizer.set_force_stop_flag(f);
    }

    let th_huber_2d = 5.99_f64.sqrt();
    let th_huber_3d = 7.815_f64.sqrt();

    let mut kf_vertex: HashMap<u64, usize> = HashMap::new();
    let mut adjust_arcs: Vec<Arc<KeyFrame>> = Vec::new();
    for kf in adjust_kfs {
        if kf.is_bad() {
            continue;
        }
        let v = VertexSE3Expmap::new(SE3Quat::from_isometry_f32(&kf.get_pose()));
        kf_vertex.insert(kf.id, optimizer.add_vertex(Box::new(v)));
        adjust_arcs.push(kf.clone());
    }
    for kf in fixed_kfs {
        if kf.is_bad() || kf_vertex.contains_key(&kf.id) {
            continue;
        }
        let mut v = VertexSE3Expmap::new(SE3Quat::from_isometry_f32(&kf.get_pose()));
        v.set_fixed(true);
        kf_vertex.insert(kf.id, optimizer.add_vertex(Box::new(v)));
    }

    // Map points seen by the welding-area keyframes (main + adjust).
    let mut mps: Vec<Arc<MapPoint>> = Vec::new();
    let mut mp_seen: HashSet<usize> = HashSet::new();
    for kf in std::iter::once(main_kf).chain(adjust_kfs.iter()) {
        for mp in kf.get_map_point_matches().into_iter().flatten() {
            if !mp.is_bad() && mp_seen.insert(mp.id) {
                mps.push(mp);
            }
        }
    }

    struct Rec {
        ei: usize,
        thr: f64,
        kf: Arc<KeyFrame>,
        mp: Arc<MapPoint>,
    }
    let mut recs: Vec<Rec> = Vec::new();
    let mut mp_vertex: HashMap<usize, usize> = HashMap::new();
    for mp in &mps {
        let mut v = VertexSBAPointXYZ::new(mp.get_world_pos().cast::<f64>());
        v.set_marginalized(true);
        let mp_vi = optimizer.add_vertex(Box::new(v));
        mp_vertex.insert(mp.id, mp_vi);
        for (obs_kf, (left, _r)) in mp.get_observations() {
            let Some(&kf_vi) = kf_vertex.get(&obs_kf.id) else {
                continue;
            };
            let left = left as i64;
            if left < 0 {
                continue;
            }
            let li = left as usize;
            let kp = &obs_kf.keys_un[li];
            let inv_sigma2 = obs_kf.inv_level_sigma2[kp.octave() as usize] as f64;
            if obs_kf.u_right[li] < 0.0 {
                let mut e = EdgeSE3ProjectXYZ::new(mp_vi, kf_vi, obs_kf.camera.clone());
                e.set_measurement(Vector2::new(kp.pt().x as f64, kp.pt().y as f64));
                e.set_information(Matrix2::identity() * inv_sigma2);
                e.set_robust_kernel(Some(th_huber_2d));
                let ei = optimizer.add_edge(Box::new(e));
                recs.push(Rec {
                    ei,
                    thr: CHI2_MONO,
                    kf: obs_kf.clone(),
                    mp: mp.clone(),
                });
            } else {
                let mut e = EdgeStereoSE3ProjectXYZ::new(
                    mp_vi,
                    kf_vi,
                    obs_kf.fx as f64,
                    obs_kf.fy as f64,
                    obs_kf.cx as f64,
                    obs_kf.cy as f64,
                    obs_kf.bf as f64,
                );
                e.set_measurement(Vector3::new(
                    kp.pt().x as f64,
                    kp.pt().y as f64,
                    obs_kf.u_right[li] as f64,
                ));
                e.set_information(Matrix3::identity() * inv_sigma2);
                e.set_robust_kernel(Some(th_huber_3d));
                let ei = optimizer.add_edge(Box::new(e));
                recs.push(Rec {
                    ei,
                    thr: CHI2_STEREO,
                    kf: obs_kf.clone(),
                    mp: mp.clone(),
                });
            }
        }
    }

    optimizer.initialize_optimization(0);
    optimizer.optimize(5);

    // Cull outliers, then re-optimize without them.
    for rec in &recs {
        if optimizer.edge(rec.ei).chi2() > rec.thr || !optimizer.edge_depth_positive(rec.ei) {
            optimizer.edge_mut(rec.ei).set_level(1);
        }
    }
    optimizer.initialize_optimization(0);
    optimizer.optimize(10);

    let mut to_erase: Vec<(Arc<KeyFrame>, Arc<MapPoint>)> = Vec::new();
    for rec in &recs {
        if optimizer.edge(rec.ei).chi2() > rec.thr || !optimizer.edge_depth_positive(rec.ei) {
            to_erase.push((rec.kf.clone(), rec.mp.clone()));
        }
    }
    for (ekf, emp) in &to_erase {
        ekf.erase_map_point_match(emp);
        emp.erase_observation(ekf);
    }

    for kf in &adjust_arcs {
        let vi = kf_vertex[&kf.id];
        let pose = optimizer
            .vertex(vi)
            .as_any()
            .downcast_ref::<VertexSE3Expmap>()
            .unwrap()
            .estimate()
            .to_isometry_f32();
        kf.set_pose(pose);
    }
    for mp in &mps {
        let vi = mp_vertex[&mp.id];
        let pos = optimizer
            .vertex(vi)
            .as_any()
            .downcast_ref::<VertexSBAPointXYZ>()
            .unwrap()
            .estimate()
            .cast::<f32>();
        mp.set_world_pos(pos);
        mp.update_normal_and_depth();
    }
    map.increase_change_index();
}

// ===========================================================================
// PoseInertialOptimizationLastKeyFrame
// ===========================================================================

/// IMU navigation state: body rotation/translation, velocity, gyro & acc biases.
#[derive(Clone)]
pub struct ImuState {
    pub rwb: nalgebra::Matrix3<f64>,
    pub twb: Vector3<f64>,
    pub vel: Vector3<f64>,
    pub bg: Vector3<f64>,
    pub ba: Vector3<f64>,
}

/// One reprojection observation for the inertial pose optimizer (point fixed).
pub enum InertialPoseObs {
    Mono {
        xw: Vector3<f64>,
        obs: Vector2<f64>,
        inv_sigma2: f64,
        track_depth: f32,
        cam_idx: usize,
        idx: usize,
    },
    Stereo {
        xw: Vector3<f64>,
        obs: Vector3<f64>,
        inv_sigma2: f64,
        cam_idx: usize,
        idx: usize,
    },
}

/// Result of [`pose_inertial_optimization_last_kf_core`].
pub struct PoseInertialResult {
    pub state: ImuState,
    pub n_inliers: i32,
    /// `(frame index, is_outlier)` per observation.
    pub outliers: Vec<(usize, bool)>,
    /// Marginalized 15-DoF prior for the next frame (`mpcpi`).
    pub prior: ConstraintPoseIMU,
    /// Raw assembled 15×15 Hessian before the prior's symmetrize/eigen-clamp.
    pub prior_h_raw: nalgebra::SMatrix<f64, 15, 15>,
}

/// Build an [`ImuCamPose`] from a body pose + per-camera extrinsics.
fn imu_cam_pose_full(
    rwb: nalgebra::Matrix3<f64>,
    twb: Vector3<f64>,
    cameras: &[Arc<dyn GeometricCamera>],
    rbc: &[nalgebra::Matrix3<f64>],
    tbc: &[Vector3<f64>],
    bf: f64,
) -> ImuCamPose {
    let rbw = rwb.transpose();
    let tbw = -rbw * twb;
    let mut rcw = Vec::with_capacity(cameras.len());
    let mut tcw = Vec::with_capacity(cameras.len());
    for i in 0..cameras.len() {
        let rcb = rbc[i].transpose();
        let tcb = -rcb * tbc[i];
        rcw.push(rcb * rbw);
        tcw.push(rcb * tbw + tcb);
    }
    ImuCamPose::new(rcw, tcw, rbc.to_vec(), tbc.to_vec(), bf, cameras.to_vec())
}

/// Gather inertial pose-only observations from a frame (left mono / stereo;
/// fisheye-rig right-camera observations are not yet gathered). Mirrors the
/// per-keypoint branching in `PoseInertialOptimization*`.
fn gather_inertial_pose_obs(frame: &Frame) -> Vec<InertialPoseObs> {
    let mut obs = Vec::with_capacity(frame.n);
    let has_rig = frame.camera2.is_some();
    let n_left = frame.n_left.unwrap_or(0);
    for i in 0..frame.n {
        let Some(mp) = frame.map_points[i].clone() else {
            continue;
        };
        let xw = mp.get_world_pos().cast::<f64>();
        let td = mp.track_depth;
        if has_rig && i >= n_left {
            continue; // rig right-camera path not yet gathered
        }
        let kp = &frame.keys_un.as_ref().unwrap_or(&frame.keys)[i];
        let inv_sigma2 = frame.inv_level_sigma2[kp.octave() as usize] as f64;
        let o = Vector2::new(kp.pt().x as f64, kp.pt().y as f64);
        let unc2 = frame.camera.uncertainty(&o) as f64;
        let inv_sigma2 = inv_sigma2 / unc2;
        if frame.u_right[i] < 0.0 {
            obs.push(InertialPoseObs::Mono {
                xw,
                obs: o,
                inv_sigma2,
                track_depth: td,
                cam_idx: 0,
                idx: i,
            });
        } else {
            obs.push(InertialPoseObs::Stereo {
                xw,
                obs: Vector3::new(o[0], o[1], frame.u_right[i] as f64),
                inv_sigma2,
                cam_idx: 0,
                idx: i,
            });
        }
    }
    obs
}

fn imu_state_from_frame(frame: &Frame) -> ImuState {
    let b = frame.imu_bias;
    ImuState {
        rwb: frame.get_imu_rotation().cast::<f64>(),
        twb: frame.get_imu_position().cast::<f64>(),
        vel: frame.get_velocity().cast::<f64>(),
        bg: Vector3::new(b.bwx as f64, b.bwy as f64, b.bwz as f64),
        ba: Vector3::new(b.bax as f64, b.bay as f64, b.baz as f64),
    }
}

fn frame_extrinsics(frame: &Frame) -> (nalgebra::Matrix3<f64>, Vector3<f64>) {
    let rbc = frame
        .imu_calib
        .tbc
        .rotation
        .to_rotation_matrix()
        .into_inner()
        .cast::<f64>();
    let tbc = frame.imu_calib.tbc.translation.vector.cast::<f64>();
    (rbc, tbc)
}

fn apply_pose_inertial_result(frame: &mut Frame, res: &PoseInertialResult) {
    frame.set_imu_pose_velocity(
        res.state.rwb.cast::<f32>(),
        res.state.twb.cast::<f32>(),
        res.state.vel.cast::<f32>(),
    );
    frame.imu_bias = Bias::from_params(
        res.state.ba[0] as f32,
        res.state.ba[1] as f32,
        res.state.ba[2] as f32,
        res.state.bg[0] as f32,
        res.state.bg[1] as f32,
        res.state.bg[2] as f32,
    );
    for &(idx, is_out) in &res.outliers {
        frame.outlier[idx] = is_out;
    }
    frame.cpi = Some(res.prior.clone());
}

/// `Optimizer::PoseInertialOptimizationLastKeyFrame` (Optimizer.cc:4491):
/// inertial pose tracking against the last keyframe. Updates the frame's pose,
/// velocity, bias, outlier flags and prior; returns the inlier count.
pub fn pose_inertial_optimization_last_keyframe(frame: &mut Frame, b_rec_init: bool) -> i32 {
    let Some(last_kf) = frame.last_keyframe.clone() else {
        return 0;
    };
    let Some(preint) = frame.imu_preintegrated.clone() else {
        return 0;
    };
    let cur = imu_state_from_frame(frame);
    let lb = last_kf.get_imu_bias();
    let last = ImuState {
        rwb: last_kf.get_imu_rotation().cast::<f64>(),
        twb: last_kf.get_imu_position().cast::<f64>(),
        vel: last_kf.get_velocity().cast::<f64>(),
        bg: Vector3::new(lb.bwx as f64, lb.bwy as f64, lb.bwz as f64),
        ba: Vector3::new(lb.bax as f64, lb.bay as f64, lb.baz as f64),
    };
    let (rbc, tbc) = frame_extrinsics(frame);
    let bf = frame.b_fx as f64;
    let camera = frame.camera.clone();
    let obs = gather_inertial_pose_obs(frame);

    let res = pose_inertial_optimization_last_kf_core(
        &cur,
        &last,
        preint,
        &obs,
        &[camera],
        &[rbc],
        &[tbc],
        bf,
        b_rec_init,
    );
    let n = res.n_inliers;
    apply_pose_inertial_result(frame, &res);
    n
}

/// `Optimizer::PoseInertialOptimizationLastFrame` (Optimizer.cc:4875): inertial
/// pose tracking against the previous frame (constrained by its prior). Updates
/// the current frame and returns the inlier count.
pub fn pose_inertial_optimization_last_frame(frame: &mut Frame, b_rec_init: bool) -> i32 {
    let Some(prev_frame) = frame.prev_frame.clone() else {
        return 0;
    };
    let Some(preint) = frame.imu_preintegrated_frame.clone() else {
        return 0;
    };
    let Some(prev_cpi) = prev_frame.cpi.clone() else {
        return 0;
    };
    // Bias random-walk info comes from the keyframe preintegration.
    let kf_preint = frame
        .imu_preintegrated
        .clone()
        .unwrap_or_else(|| preint.clone());
    let info_g: nalgebra::Matrix3<f64> = kf_preint
        .c
        .fixed_view::<3, 3>(9, 9)
        .into_owned()
        .cast::<f64>()
        .try_inverse()
        .unwrap();
    let info_a: nalgebra::Matrix3<f64> = kf_preint
        .c
        .fixed_view::<3, 3>(12, 12)
        .into_owned()
        .cast::<f64>()
        .try_inverse()
        .unwrap();

    let cur = imu_state_from_frame(frame);
    let prev = imu_state_from_frame(&prev_frame);
    let (rbc, tbc) = frame_extrinsics(frame);
    let bf = frame.b_fx as f64;
    let camera = frame.camera.clone();
    let obs = gather_inertial_pose_obs(frame);

    let res = pose_inertial_optimization_last_frame_core(
        &cur,
        &prev,
        preint,
        info_g,
        info_a,
        &prev_cpi,
        &obs,
        &[camera],
        &[rbc],
        &[tbc],
        bf,
        b_rec_init,
    );
    let n = res.n_inliers;
    apply_pose_inertial_result(frame, &res);
    n
}

/// Core of `Optimizer::PoseInertialOptimizationLastKeyFrame` (Optimizer.cc:4491).
///
/// Optimizes the current frame's pose/velocity/biases against the (fixed) last
/// keyframe via an IMU preintegration edge + bias random-walk edges + visual
/// reprojection edges, using Gauss-Newton with 4 outlier-rejection rounds.
/// Produces the optimized state, inlier count, per-observation outlier flags,
/// and a marginalized 15-DoF prior for the next frame.
#[allow(clippy::too_many_arguments)]
pub fn pose_inertial_optimization_last_kf_core(
    cur: &ImuState,
    last: &ImuState,
    preint: Arc<Preintegrated>,
    obs: &[InertialPoseObs],
    cameras: &[Arc<dyn GeometricCamera>],
    rbc: &[nalgebra::Matrix3<f64>],
    tbc: &[Vector3<f64>],
    bf: f64,
    b_rec_init: bool,
) -> PoseInertialResult {
    use nalgebra::DMatrix;

    let mut opt = SparseOptimizer::new();
    opt.set_gauss_newton(true);

    // Current frame vertices (free).
    let vp = opt.add_vertex(Box::new(VertexPose::new(imu_cam_pose_full(
        cur.rwb, cur.twb, cameras, rbc, tbc, bf,
    ))));
    let vv = opt.add_vertex(Box::new(VertexVelocity::new(cur.vel)));
    let vg = opt.add_vertex(Box::new(VertexGyroBias::new(cur.bg)));
    let va = opt.add_vertex(Box::new(VertexAccBias::new(cur.ba)));

    let th_huber_mono = 5.991_f64.sqrt();
    let th_huber_stereo = 7.815_f64.sqrt();

    enum Rec {
        Mono {
            ei: usize,
            idx: usize,
            track_depth: f32,
        },
        Stereo {
            ei: usize,
            idx: usize,
        },
    }
    let mut recs: Vec<Rec> = Vec::new();
    for o in obs {
        match o {
            InertialPoseObs::Mono {
                xw,
                obs,
                inv_sigma2,
                track_depth,
                cam_idx,
                idx,
            } => {
                let mut e = EdgeMonoOnlyPose::new(vp, *xw, *cam_idx);
                e.set_measurement(*obs);
                e.set_information(nalgebra::Matrix2::identity() * *inv_sigma2);
                e.set_robust_kernel(Some(th_huber_mono));
                let ei = opt.add_edge(Box::new(e));
                recs.push(Rec::Mono {
                    ei,
                    idx: *idx,
                    track_depth: *track_depth,
                });
            }
            InertialPoseObs::Stereo {
                xw,
                obs,
                inv_sigma2,
                cam_idx,
                idx,
            } => {
                let mut e = EdgeStereoOnlyPose::new(vp, *xw, *cam_idx);
                e.set_measurement(*obs);
                e.set_information(nalgebra::Matrix3::identity() * *inv_sigma2);
                e.set_robust_kernel(Some(th_huber_stereo));
                let ei = opt.add_edge(Box::new(e));
                recs.push(Rec::Stereo { ei, idx: *idx });
            }
        }
    }

    // Last keyframe vertices (fixed).
    let vpk = opt.add_vertex(Box::new({
        let mut v = VertexPose::new(imu_cam_pose_full(last.rwb, last.twb, cameras, rbc, tbc, bf));
        v.set_fixed(true);
        v
    }));
    let vvk = opt.add_vertex(Box::new({
        let mut v = VertexVelocity::new(last.vel);
        v.set_fixed(true);
        v
    }));
    let vgk = opt.add_vertex(Box::new({
        let mut v = VertexGyroBias::new(last.bg);
        v.set_fixed(true);
        v
    }));
    let vak = opt.add_vertex(Box::new({
        let mut v = VertexAccBias::new(last.ba);
        v.set_fixed(true);
        v
    }));

    let ei = opt.add_edge(Box::new(EdgeInertial::new(
        [vpk, vvk, vgk, vak, vp, vv],
        preint.clone(),
    )));

    let info_g: nalgebra::Matrix3<f64> = preint
        .c
        .fixed_view::<3, 3>(9, 9)
        .into_owned()
        .cast::<f64>()
        .try_inverse()
        .unwrap();
    let info_a: nalgebra::Matrix3<f64> = preint
        .c
        .fixed_view::<3, 3>(12, 12)
        .into_owned()
        .cast::<f64>()
        .try_inverse()
        .unwrap();
    let egr = {
        let mut e = EdgeBiasRW::new_gyro(vgk, vg);
        e.set_information(info_g);
        opt.add_edge(Box::new(e))
    };
    let ear = {
        let mut e = EdgeBiasRW::new_acc(vak, va);
        e.set_information(info_a);
        opt.add_edge(Box::new(e))
    };

    let n_initial = recs.len() as i32;
    let mut outlier = vec![false; recs.len()];
    let chi2_mono = [12.0, 7.5, 5.991, 5.991];
    let chi2_stereo = [15.6, 9.8, 7.815, 7.815];
    let its = [10i32; 4];
    let mut n_bad = 0;
    let mut n_inliers;

    for it in 0..4 {
        opt.initialize_optimization(0);
        opt.optimize(its[it]);
        n_bad = 0;
        n_inliers = 0;
        let chi2close = 1.5 * chi2_mono[it];
        for (ri, rec) in recs.iter().enumerate() {
            match *rec {
                Rec::Mono {
                    ei, track_depth, ..
                } => {
                    if outlier[ri] {
                        opt.compute_edge_error(ei);
                    }
                    let chi2 = opt.edge(ei).chi2();
                    let b_close = track_depth < 10.0;
                    let depth_ok = opt.edge_depth_positive(ei);
                    if (chi2 > chi2_mono[it] && !b_close)
                        || (b_close && chi2 > chi2close)
                        || !depth_ok
                    {
                        outlier[ri] = true;
                        opt.edge_mut(ei).set_level(1);
                        n_bad += 1;
                    } else {
                        outlier[ri] = false;
                        opt.edge_mut(ei).set_level(0);
                        n_inliers += 1;
                    }
                    if it == 2 {
                        opt.edge_mut(ei).set_robust_kernel(None);
                    }
                }
                Rec::Stereo { ei, .. } => {
                    if outlier[ri] {
                        opt.compute_edge_error(ei);
                    }
                    let chi2 = opt.edge(ei).chi2();
                    if chi2 > chi2_stereo[it] {
                        outlier[ri] = true;
                        opt.edge_mut(ei).set_level(1);
                        n_bad += 1;
                    } else {
                        outlier[ri] = false;
                        opt.edge_mut(ei).set_level(0);
                        n_inliers += 1;
                    }
                    if it == 2 {
                        opt.edge_mut(ei).set_robust_kernel(None);
                    }
                }
            }
        }
        let _ = n_inliers;
        if opt.num_active_edges() < 10 {
            break;
        }
    }

    // Recompute inlier count after the last round.
    let mut n_inliers_final = 0;
    for (ri, _) in recs.iter().enumerate() {
        if !outlier[ri] {
            n_inliers_final += 1;
        }
    }

    // If few tracks, recover not-too-bad points.
    if n_inliers_final < 30 && !b_rec_init {
        n_bad = 0;
        for (ri, rec) in recs.iter().enumerate() {
            match *rec {
                Rec::Mono { ei, idx, .. } => {
                    opt.compute_edge_error(ei);
                    if opt.edge(ei).chi2() < 18.0 {
                        outlier[ri] = false;
                    } else {
                        n_bad += 1;
                    }
                    let _ = idx;
                }
                Rec::Stereo { ei, .. } => {
                    opt.compute_edge_error(ei);
                    if opt.edge(ei).chi2() < 24.0 {
                        outlier[ri] = false;
                    } else {
                        n_bad += 1;
                    }
                }
            }
        }
    }

    // Recover optimized state.
    let pose = opt
        .vertex(vp)
        .as_any()
        .downcast_ref::<VertexPose>()
        .unwrap()
        .estimate()
        .clone();
    let vel = opt
        .vertex(vv)
        .as_any()
        .downcast_ref::<VertexVelocity>()
        .unwrap()
        .estimate();
    let bg = opt
        .vertex(vg)
        .as_any()
        .downcast_ref::<VertexGyroBias>()
        .unwrap()
        .estimate();
    let ba = opt
        .vertex(va)
        .as_any()
        .downcast_ref::<VertexAccBias>()
        .unwrap()
        .estimate();
    let state = ImuState {
        rwb: pose.rwb,
        twb: pose.twb,
        vel,
        bg,
        ba,
    };

    // Marginalize KF states -> 15-DoF prior for the next frame.
    let mut h = DMatrix::<f64>::zeros(15, 15);
    // EdgeInertial GetHessian2: J = [jac[4](9x6) | jac[5](9x3)] -> 9x9.
    let lin = opt.edge_linearization(ei);
    let mut jei = DMatrix::<f64>::zeros(9, 9);
    jei.view_mut((0, 0), (9, 6)).copy_from(&lin.jacobians[4]);
    jei.view_mut((0, 6), (9, 3)).copy_from(&lin.jacobians[5]);
    let h2 = jei.transpose() * &lin.information * &jei;
    {
        let mut blk = h.view_mut((0, 0), (9, 9));
        blk += &h2;
    }
    // Bias RW GetHessian2 = info (Xj jacobian = I).
    {
        let mut blk = h.view_mut((9, 9), (3, 3));
        blk += &opt.edge_linearization(egr).information;
    }
    {
        let mut blk = h.view_mut((12, 12), (3, 3));
        blk += &opt.edge_linearization(ear).information;
    }
    // Reprojection GetHessian (6x6) for inliers.
    for (ri, rec) in recs.iter().enumerate() {
        if outlier[ri] {
            continue;
        }
        let e_ei = match *rec {
            Rec::Mono { ei, .. } => ei,
            Rec::Stereo { ei, .. } => ei,
        };
        let lin = opt.edge_linearization(e_ei);
        let j = &lin.jacobians[0]; // (dim x 6)
        let hh = j.transpose() * &lin.information * j;
        let mut blk = h.view_mut((0, 0), (6, 6));
        blk += &hh;
    }
    let h15 = nalgebra::SMatrix::<f64, 15, 15>::from_iterator(h.iter().copied());
    let prior = ConstraintPoseIMU::new(state.rwb, state.twb, state.vel, state.bg, state.ba, h15);
    let prior_h_raw = h15;

    let outliers = obs
        .iter()
        .enumerate()
        .map(|(ri, o)| {
            let idx = match o {
                InertialPoseObs::Mono { idx, .. } | InertialPoseObs::Stereo { idx, .. } => *idx,
            };
            (idx, outlier[ri])
        })
        .collect();

    PoseInertialResult {
        state,
        n_inliers: n_initial - n_bad,
        outliers,
        prior,
        prior_h_raw,
    }
}

// ===========================================================================
// PoseInertialOptimizationLastFrame
// ===========================================================================

/// Scatter an edge's `JᵀΩJ` into a global Hessian at per-vertex offsets `offs`.
fn scatter_hessian(h: &mut nalgebra::DMatrix<f64>, lin: &EdgeLinearization, offs: &[usize]) {
    let info = &lin.information;
    for k in 0..offs.len() {
        let jt_info = lin.jacobians[k].transpose() * info;
        for l in 0..offs.len() {
            let blk = &jt_info * &lin.jacobians[l];
            let mut v = h.view_mut((offs[k], offs[l]), (blk.nrows(), blk.ncols()));
            v += &blk;
        }
    }
}

/// Core of `Optimizer::PoseInertialOptimizationLastFrame` (Optimizer.cc:4875).
///
/// Like [`pose_inertial_optimization_last_kf_core`] but the "previous" entity is
/// the prior frame (its states are *free*), constrained by an
/// [`EdgePriorPoseImu`] from its marginalization prior. After optimization the
/// previous-frame states are marginalized out (30→15 DoF) to form the current
/// frame's new prior.
#[allow(clippy::too_many_arguments)]
pub fn pose_inertial_optimization_last_frame_core(
    cur: &ImuState,
    prev: &ImuState,
    preint_frame: Arc<Preintegrated>,
    info_g: nalgebra::Matrix3<f64>,
    info_a: nalgebra::Matrix3<f64>,
    prev_cpi: &ConstraintPoseIMU,
    obs: &[InertialPoseObs],
    cameras: &[Arc<dyn GeometricCamera>],
    rbc: &[nalgebra::Matrix3<f64>],
    tbc: &[Vector3<f64>],
    bf: f64,
    b_rec_init: bool,
) -> PoseInertialResult {
    use nalgebra::DMatrix;

    let mut opt = SparseOptimizer::new();
    opt.set_gauss_newton(true);

    // Current frame vertices (free).
    let vp = opt.add_vertex(Box::new(VertexPose::new(imu_cam_pose_full(
        cur.rwb, cur.twb, cameras, rbc, tbc, bf,
    ))));
    let vv = opt.add_vertex(Box::new(VertexVelocity::new(cur.vel)));
    let vg = opt.add_vertex(Box::new(VertexGyroBias::new(cur.bg)));
    let va = opt.add_vertex(Box::new(VertexAccBias::new(cur.ba)));

    let th_huber_mono = 5.991_f64.sqrt();
    let th_huber_stereo = 7.815_f64.sqrt();

    enum Rec {
        Mono {
            ei: usize,
            idx: usize,
            track_depth: f32,
        },
        Stereo {
            ei: usize,
            idx: usize,
        },
    }
    let mut recs: Vec<Rec> = Vec::new();
    for o in obs {
        match o {
            InertialPoseObs::Mono {
                xw,
                obs,
                inv_sigma2,
                track_depth,
                cam_idx,
                idx,
            } => {
                let mut e = EdgeMonoOnlyPose::new(vp, *xw, *cam_idx);
                e.set_measurement(*obs);
                e.set_information(nalgebra::Matrix2::identity() * *inv_sigma2);
                e.set_robust_kernel(Some(th_huber_mono));
                let ei = opt.add_edge(Box::new(e));
                recs.push(Rec::Mono {
                    ei,
                    idx: *idx,
                    track_depth: *track_depth,
                });
            }
            InertialPoseObs::Stereo {
                xw,
                obs,
                inv_sigma2,
                cam_idx,
                idx,
            } => {
                let mut e = EdgeStereoOnlyPose::new(vp, *xw, *cam_idx);
                e.set_measurement(*obs);
                e.set_information(nalgebra::Matrix3::identity() * *inv_sigma2);
                e.set_robust_kernel(Some(th_huber_stereo));
                let ei = opt.add_edge(Box::new(e));
                recs.push(Rec::Stereo { ei, idx: *idx });
            }
        }
    }

    // Previous frame vertices (free).
    let vpk = opt.add_vertex(Box::new(VertexPose::new(imu_cam_pose_full(
        prev.rwb, prev.twb, cameras, rbc, tbc, bf,
    ))));
    let vvk = opt.add_vertex(Box::new(VertexVelocity::new(prev.vel)));
    let vgk = opt.add_vertex(Box::new(VertexGyroBias::new(prev.bg)));
    let vak = opt.add_vertex(Box::new(VertexAccBias::new(prev.ba)));

    let ei = opt.add_edge(Box::new(EdgeInertial::new(
        [vpk, vvk, vgk, vak, vp, vv],
        preint_frame,
    )));
    let egr = {
        let mut e = EdgeBiasRW::new_gyro(vgk, vg);
        e.set_information(info_g);
        opt.add_edge(Box::new(e))
    };
    let ear = {
        let mut e = EdgeBiasRW::new_acc(vak, va);
        e.set_information(info_a);
        opt.add_edge(Box::new(e))
    };
    let ep = {
        let mut e = EdgePriorPoseImu::new([vpk, vvk, vgk, vak], prev_cpi);
        e.set_robust_kernel(Some(5.0));
        opt.add_edge(Box::new(e))
    };

    let n_initial = recs.len() as i32;
    let mut outlier = vec![false; recs.len()];
    let chi2_mono = [5.991, 5.991, 5.991, 5.991];
    let chi2_stereo = [15.6, 9.8, 7.815, 7.815];
    let mut n_bad = 0;

    for it in 0..4 {
        opt.initialize_optimization(0);
        opt.optimize(10);
        n_bad = 0;
        let chi2close = 1.5 * chi2_mono[it];
        for (ri, rec) in recs.iter().enumerate() {
            match *rec {
                Rec::Mono {
                    ei, track_depth, ..
                } => {
                    if outlier[ri] {
                        opt.compute_edge_error(ei);
                    }
                    let chi2 = opt.edge(ei).chi2();
                    let b_close = track_depth < 10.0;
                    if (chi2 > chi2_mono[it] && !b_close)
                        || (b_close && chi2 > chi2close)
                        || !opt.edge_depth_positive(ei)
                    {
                        outlier[ri] = true;
                        opt.edge_mut(ei).set_level(1);
                        n_bad += 1;
                    } else {
                        outlier[ri] = false;
                        opt.edge_mut(ei).set_level(0);
                    }
                    if it == 2 {
                        opt.edge_mut(ei).set_robust_kernel(None);
                    }
                }
                Rec::Stereo { ei, .. } => {
                    if outlier[ri] {
                        opt.compute_edge_error(ei);
                    }
                    let chi2 = opt.edge(ei).chi2();
                    if chi2 > chi2_stereo[it] {
                        outlier[ri] = true;
                        opt.edge_mut(ei).set_level(1);
                        n_bad += 1;
                    } else {
                        outlier[ri] = false;
                        opt.edge_mut(ei).set_level(0);
                    }
                    if it == 2 {
                        opt.edge_mut(ei).set_robust_kernel(None);
                    }
                }
            }
        }
        if opt.num_active_edges() < 10 {
            break;
        }
    }

    let mut n_inliers = 0;
    for &o in outlier.iter() {
        if !o {
            n_inliers += 1;
        }
    }
    if n_inliers < 30 && !b_rec_init {
        n_bad = 0;
        for (ri, rec) in recs.iter().enumerate() {
            let (ce, thr) = match *rec {
                Rec::Mono { ei, .. } => (ei, 18.0),
                Rec::Stereo { ei, .. } => (ei, 24.0),
            };
            opt.compute_edge_error(ce);
            if opt.edge(ce).chi2() < thr {
                outlier[ri] = false;
            } else {
                n_bad += 1;
            }
        }
    }

    // Recover optimized current state.
    let pose = opt
        .vertex(vp)
        .as_any()
        .downcast_ref::<VertexPose>()
        .unwrap()
        .estimate()
        .clone();
    let vel = opt
        .vertex(vv)
        .as_any()
        .downcast_ref::<VertexVelocity>()
        .unwrap()
        .estimate();
    let bg = opt
        .vertex(vg)
        .as_any()
        .downcast_ref::<VertexGyroBias>()
        .unwrap()
        .estimate();
    let ba = opt
        .vertex(va)
        .as_any()
        .downcast_ref::<VertexAccBias>()
        .unwrap()
        .estimate();
    let state = ImuState {
        rwb: pose.rwb,
        twb: pose.twb,
        vel,
        bg,
        ba,
    };

    // 30-DoF Hessian: prev frame [0..15), current frame [15..30).
    // Layout: pose(0-5/15-20), vel(6-8/21-23), gyro(9-11/24-26), acc(12-14/27-29).
    let mut h = DMatrix::<f64>::zeros(30, 30);
    scatter_hessian(&mut h, &opt.edge_linearization(ei), &[0, 6, 9, 12, 15, 21]);
    scatter_hessian(&mut h, &opt.edge_linearization(egr), &[9, 24]);
    scatter_hessian(&mut h, &opt.edge_linearization(ear), &[12, 27]);
    scatter_hessian(&mut h, &opt.edge_linearization(ep), &[0, 6, 9, 12]);
    for (ri, rec) in recs.iter().enumerate() {
        if outlier[ri] {
            continue;
        }
        let ce = match *rec {
            Rec::Mono { ei, .. } | Rec::Stereo { ei, .. } => ei,
        };
        scatter_hessian(&mut h, &opt.edge_linearization(ce), &[15]);
    }

    let h = marginalize(&h, 0, 14);
    let h15 =
        nalgebra::SMatrix::<f64, 15, 15>::from_iterator(h.view((15, 15), (15, 15)).iter().copied());
    let prior = ConstraintPoseIMU::new(state.rwb, state.twb, state.vel, state.bg, state.ba, h15);

    let outliers = obs
        .iter()
        .enumerate()
        .map(|(ri, o)| {
            let idx = match o {
                InertialPoseObs::Mono { idx, .. } | InertialPoseObs::Stereo { idx, .. } => *idx,
            };
            (idx, outlier[ri])
        })
        .collect();

    PoseInertialResult {
        state,
        n_inliers: n_initial - n_bad,
        outliers,
        prior,
        prior_h_raw: h15,
    }
}

// ===========================================================================
// Inertial bundle adjustment (shared core for Full / Local / Merge InertialBA)
// ===========================================================================

/// A keyframe in an inertial bundle adjustment.
pub struct InertialBaKf {
    pub state: ImuState,
    pub fixed: bool,
    pub camera: Arc<dyn GeometricCamera>,
    pub rbc: nalgebra::Matrix3<f64>,
    pub tbc: Vector3<f64>,
    pub bf: f64,
}

/// An IMU preintegration link between two keyframes (`prev` -> `cur` indices).
pub struct InertialLink {
    pub prev: usize,
    pub cur: usize,
    pub preint: Arc<Preintegrated>,
    /// Huber delta for the inertial edge (`None` = no robust kernel).
    pub robust_delta: Option<f64>,
    /// Multiplier on the inertial information (LocalInertialBA boundary = 1e-2).
    pub info_scale: f64,
}

/// A reprojection observation in an inertial BA.
pub enum InertialBaObs {
    Mono {
        kf: usize,
        mp: usize,
        obs: Vector2<f64>,
        inv_sigma2: f64,
        cam_idx: usize,
    },
    Stereo {
        kf: usize,
        mp: usize,
        obs: Vector3<f64>,
        inv_sigma2: f64,
        cam_idx: usize,
    },
}

/// Result of [`inertial_ba_core`]: optimized states + points plus per-edge
/// diagnostics for outlier culling.
pub struct InertialBaResult {
    pub states: Vec<ImuState>,
    pub points: Vec<Vector3<f64>>,
    /// Per-observation `χ²` (input order).
    pub obs_chi2: Vec<f64>,
    /// Per-observation depth positivity.
    pub obs_depth_positive: Vec<bool>,
    /// Active robust χ² before / after optimization (for the divergence guard).
    pub err_start: f64,
    pub err_end: f64,
}

/// Shared core for the inertial bundle-adjustment routines (`FullInertialBA`,
/// `LocalInertialBA`, `MergeInertialBA`): keyframe pose/velocity/bias vertices,
/// `EdgeInertial` + bias random-walk links, reprojection edges to marginalized
/// points, Levenberg-Marquardt. Returns optimized states/points + per-edge
/// diagnostics.
pub fn inertial_ba_core(
    kfs: &[InertialBaKf],
    links: &[InertialLink],
    points: &[Vector3<f64>],
    obs: &[InertialBaObs],
    n_iterations: i32,
    user_lambda_init: f64,
    stop_flag: Option<Arc<AtomicBool>>,
) -> InertialBaResult {
    let mut opt = SparseOptimizer::new();
    if user_lambda_init > 0.0 {
        opt.set_user_lambda_init(user_lambda_init);
    }
    if let Some(f) = stop_flag {
        opt.set_force_stop_flag(f);
    }

    struct KfV {
        vp: usize,
        vv: usize,
        vg: usize,
        va: usize,
    }
    let mut kfv = Vec::with_capacity(kfs.len());
    for kf in kfs {
        let mut vp = VertexPose::new(imu_cam_pose_full(
            kf.state.rwb,
            kf.state.twb,
            &[kf.camera.clone()],
            &[kf.rbc],
            &[kf.tbc],
            kf.bf,
        ));
        vp.set_fixed(kf.fixed);
        let vp = opt.add_vertex(Box::new(vp));
        let mut vv = VertexVelocity::new(kf.state.vel);
        vv.set_fixed(kf.fixed);
        let vv = opt.add_vertex(Box::new(vv));
        let mut vg = VertexGyroBias::new(kf.state.bg);
        vg.set_fixed(kf.fixed);
        let vg = opt.add_vertex(Box::new(vg));
        let mut va = VertexAccBias::new(kf.state.ba);
        va.set_fixed(kf.fixed);
        let va = opt.add_vertex(Box::new(va));
        kfv.push(KfV { vp, vv, vg, va });
    }

    for link in links {
        let (p, c) = (&kfv[link.prev], &kfv[link.cur]);
        let mut ei = EdgeInertial::new([p.vp, p.vv, p.vg, p.va, c.vp, c.vv], link.preint.clone());
        if link.info_scale != 1.0 {
            ei.scale_information(link.info_scale);
        }
        ei.set_robust_kernel(link.robust_delta);
        opt.add_edge(Box::new(ei));

        let info_g: nalgebra::Matrix3<f64> = link
            .preint
            .c
            .fixed_view::<3, 3>(9, 9)
            .into_owned()
            .cast::<f64>()
            .try_inverse()
            .unwrap();
        let mut egr = EdgeBiasRW::new_gyro(p.vg, c.vg);
        egr.set_information(info_g);
        opt.add_edge(Box::new(egr));

        let info_a: nalgebra::Matrix3<f64> = link
            .preint
            .c
            .fixed_view::<3, 3>(12, 12)
            .into_owned()
            .cast::<f64>()
            .try_inverse()
            .unwrap();
        let mut ear = EdgeBiasRW::new_acc(p.va, c.va);
        ear.set_information(info_a);
        opt.add_edge(Box::new(ear));
    }

    let th_mono = 5.991_f64.sqrt();
    let th_stereo = 7.815_f64.sqrt();
    let mut mpv = Vec::with_capacity(points.len());
    for p in points {
        let mut v = VertexSBAPointXYZ::new(*p);
        v.set_marginalized(true);
        mpv.push(opt.add_vertex(Box::new(v)));
    }
    let mut obs_edges: Vec<usize> = Vec::with_capacity(obs.len());
    for o in obs {
        let ei = match o {
            InertialBaObs::Mono {
                kf,
                mp,
                obs,
                inv_sigma2,
                cam_idx,
            } => {
                let mut e = EdgeMono::new(mpv[*mp], kfv[*kf].vp, *cam_idx);
                e.set_measurement(*obs);
                e.set_information(nalgebra::Matrix2::identity() * *inv_sigma2);
                e.set_robust_kernel(Some(th_mono));
                opt.add_edge(Box::new(e))
            }
            InertialBaObs::Stereo {
                kf,
                mp,
                obs,
                inv_sigma2,
                cam_idx,
            } => {
                let mut e = EdgeStereo::new(mpv[*mp], kfv[*kf].vp, *cam_idx);
                e.set_measurement(*obs);
                e.set_information(nalgebra::Matrix3::identity() * *inv_sigma2);
                e.set_robust_kernel(Some(th_stereo));
                opt.add_edge(Box::new(e))
            }
        };
        obs_edges.push(ei);
    }

    opt.initialize_optimization(0);
    opt.compute_active_errors();
    let err_start = opt.active_robust_chi2();
    opt.optimize(n_iterations);
    opt.compute_active_errors();
    let err_end = opt.active_robust_chi2();

    let obs_chi2: Vec<f64> = obs_edges.iter().map(|&ei| opt.edge(ei).chi2()).collect();
    let obs_depth_positive: Vec<bool> = obs_edges
        .iter()
        .map(|&ei| opt.edge_depth_positive(ei))
        .collect();

    let states = kfv
        .iter()
        .map(|k| {
            let pose = opt
                .vertex(k.vp)
                .as_any()
                .downcast_ref::<VertexPose>()
                .unwrap()
                .estimate();
            ImuState {
                rwb: pose.rwb,
                twb: pose.twb,
                vel: opt
                    .vertex(k.vv)
                    .as_any()
                    .downcast_ref::<VertexVelocity>()
                    .unwrap()
                    .estimate(),
                bg: opt
                    .vertex(k.vg)
                    .as_any()
                    .downcast_ref::<VertexGyroBias>()
                    .unwrap()
                    .estimate(),
                ba: opt
                    .vertex(k.va)
                    .as_any()
                    .downcast_ref::<VertexAccBias>()
                    .unwrap()
                    .estimate(),
            }
        })
        .collect();
    let out_points = mpv
        .iter()
        .map(|&vi| {
            opt.vertex(vi)
                .as_any()
                .downcast_ref::<VertexSBAPointXYZ>()
                .unwrap()
                .estimate()
        })
        .collect();
    InertialBaResult {
        states,
        points: out_points,
        obs_chi2,
        obs_depth_positive,
        err_start,
        err_end,
    }
}

/// Build an [`InertialBaKf`] / [`ImuState`] from a keyframe.
fn inertial_ba_kf_from(kf: &Arc<KeyFrame>, fixed: bool) -> InertialBaKf {
    let b = kf.get_imu_bias();
    let state = ImuState {
        rwb: kf.get_imu_rotation().cast::<f64>(),
        twb: kf.get_imu_position().cast::<f64>(),
        vel: kf.get_velocity().cast::<f64>(),
        bg: Vector3::new(b.bwx as f64, b.bwy as f64, b.bwz as f64),
        ba: Vector3::new(b.bax as f64, b.bay as f64, b.baz as f64),
    };
    InertialBaKf {
        state,
        fixed,
        camera: kf.camera.clone(),
        rbc: kf
            .imu_calib
            .tbc
            .rotation
            .to_rotation_matrix()
            .into_inner()
            .cast::<f64>(),
        tbc: kf.imu_calib.tbc.translation.vector.cast::<f64>(),
        bf: kf.bf as f64,
    }
}

/// Convert an optimized [`ImuState`] back to a camera pose `Tcw`.
fn imu_state_to_tcw(
    state: &ImuState,
    rbc: &nalgebra::Matrix3<f64>,
    tbc: &Vector3<f64>,
) -> Isometry3<f32> {
    let rcb = rbc.transpose();
    let tcb = -rcb * tbc;
    let rcw = rcb * state.rwb.transpose();
    let tcw = rcb * (-state.rwb.transpose() * state.twb) + tcb;
    let q = nalgebra::UnitQuaternion::from_matrix(&rcw.cast::<f32>());
    Isometry3::from_parts(nalgebra::Translation3::from(tcw.cast::<f32>()), q)
}

/// `Optimizer::FullInertialBA` (Optimizer.cc:392), non-init path (per-keyframe
/// bias). Inertial bundle adjustment over every keyframe + map point in `map`.
/// When `loop_kf == 0` the result is written back directly; otherwise keyframe
/// poses / point positions are staged in their global-BA fields.
///
/// The IMU-initialization variant (`b_init`, with a single shared bias vertex +
/// bias priors) is not yet wired here; the underlying [`inertial_ba_core`]
/// already validates the shared per-keyframe-bias numerics. (`vSingVal`/`bHess`
/// in the C++ signature are dead outputs and are omitted.)
#[allow(clippy::too_many_arguments)]
pub fn full_inertial_ba(
    map: &Map,
    iterations: i32,
    b_fix_local: bool,
    loop_kf: u64,
    stop_flag: Option<Arc<AtomicBool>>,
    b_init: bool,
    _prior_g: f32,
    _prior_a: f32,
) {
    let _ = b_init; // shared-bias init path not yet wired (see doc).
    let kfs_all = map.get_all_key_frames();
    let mps_all = map.get_all_map_points();
    let max_kf_id = map.get_max_kf_id();

    let mut kfs: Vec<InertialBaKf> = Vec::new();
    let mut kf_index: HashMap<u64, usize> = HashMap::new();
    let mut kf_arcs: Vec<Arc<KeyFrame>> = Vec::new();
    for kf in &kfs_all {
        if kf.id > max_kf_id {
            continue;
        }
        // bFixLocal keyframe fixing needs mnBALocalForKF/mnBAFixedForKF
        // bookkeeping (not yet plumbed); default to the bFixLocal==false case,
        // where no keyframe is fixed and the gauge is fixed by LM damping.
        let _ = b_fix_local;
        let fixed = false;
        kf_index.insert(kf.id, kfs.len());
        kfs.push(inertial_ba_kf_from(kf, fixed));
        kf_arcs.push(kf.clone());
    }

    let mut links: Vec<InertialLink> = Vec::new();
    for kf in &kf_arcs {
        let Some(prev) = kf.get_prev_kf() else {
            continue;
        };
        let (Some(&ci), Some(&pi)) = (kf_index.get(&kf.id), kf_index.get(&prev.id)) else {
            continue;
        };
        let Some(preint) = kf.imu_preintegrated.clone() else {
            continue;
        };
        links.push(InertialLink {
            prev: pi,
            cur: ci,
            preint,
            robust_delta: Some(16.92_f64.sqrt()),
            info_scale: 1.0,
        });
    }

    let mut points: Vec<Vector3<f64>> = Vec::new();
    let mut mp_index: HashMap<usize, usize> = HashMap::new();
    let mut mp_arcs: Vec<Arc<MapPoint>> = Vec::new();
    for mp in &mps_all {
        if mp.is_bad() {
            continue;
        }
        mp_index.insert(mp.id, points.len());
        points.push(mp.get_world_pos().cast::<f64>());
        mp_arcs.push(mp.clone());
    }

    let mut obs: Vec<InertialBaObs> = Vec::new();
    for mp in &mp_arcs {
        let mp_i = mp_index[&mp.id];
        for (obs_kf, (left, _right)) in mp.get_observations() {
            let Some(&kf_i) = kf_index.get(&obs_kf.id) else {
                continue;
            };
            let left = left as i64;
            if left < 0 {
                continue;
            }
            let li = left as usize;
            let kp = &obs_kf.keys_un[li];
            let inv_sigma2 = obs_kf.inv_level_sigma2[kp.octave() as usize] as f64;
            if obs_kf.u_right[li] < 0.0 {
                obs.push(InertialBaObs::Mono {
                    kf: kf_i,
                    mp: mp_i,
                    obs: Vector2::new(kp.pt().x as f64, kp.pt().y as f64),
                    inv_sigma2,
                    cam_idx: 0,
                });
            } else {
                obs.push(InertialBaObs::Stereo {
                    kf: kf_i,
                    mp: mp_i,
                    obs: Vector3::new(
                        kp.pt().x as f64,
                        kp.pt().y as f64,
                        obs_kf.u_right[li] as f64,
                    ),
                    inv_sigma2,
                    cam_idx: 0,
                });
            }
        }
    }

    let r = inertial_ba_core(&kfs, &links, &points, &obs, iterations, 1e-5, stop_flag);
    let (states, out_points) = (r.states, r.points);

    let direct = loop_kf == 0;
    for (i, kf) in kf_arcs.iter().enumerate() {
        let s = &states[i];
        let tcw = imu_state_to_tcw(s, &kfs[i].rbc, &kfs[i].tbc);
        if direct {
            kf.set_pose(tcw);
            kf.set_velocity(s.vel.cast::<f32>());
            kf.set_new_bias(crate::imu_types::Bias::from_params(
                s.ba[0] as f32,
                s.ba[1] as f32,
                s.ba[2] as f32,
                s.bg[0] as f32,
                s.bg[1] as f32,
                s.bg[2] as f32,
            ));
        } else {
            kf.set_tcw_gba(tcw, loop_kf as i64);
        }
    }
    for (mp, p) in mp_arcs.iter().zip(out_points.iter()) {
        if direct {
            mp.set_world_pos(p.cast::<f32>());
            mp.update_normal_and_depth();
        } else {
            mp.set_pos_gba(p.cast::<f32>(), loop_kf as i64);
        }
    }
    map.increase_change_index();
}

/// `Optimizer::LocalInertialBA` (Optimizer.cc:2383).
///
/// Inertial bundle adjustment over a temporal window of recent keyframes
/// (with the keyframe just before the window held fixed) plus the map points
/// they observe. Builds on the validated [`inertial_ba_core`].
#[allow(clippy::type_complexity)]
pub fn local_inertial_ba(
    kf: &Arc<KeyFrame>,
    stop_flag: Option<Arc<AtomicBool>>,
    map: &Map,
    large: bool,
    rec_init: bool,
) -> (usize, usize, usize, usize) {
    let opt_it = if large { 4 } else { 10 };
    let user_lambda = if large { 1e-2 } else { 1e0 };
    let max_opt = if large { 25 } else { 10 };
    let nd = ((map.key_frames_in_map() as i64 - 2).min(max_opt)).max(0) as usize;

    // Temporal window: current KF + its `Nd` predecessors via prev links.
    let mut window: Vec<Arc<KeyFrame>> = vec![kf.clone()];
    for _ in 1..nd {
        match window.last().unwrap().get_prev_kf() {
            Some(p) => window.push(p),
            None => break,
        }
    }
    // The keyframe before the window is the fixed boundary.
    let fixed_boundary = window.last().unwrap().get_prev_kf();

    // Local map points seen by the window.
    let mut local_mps: Vec<Arc<MapPoint>> = Vec::new();
    let mut mp_seen: HashSet<usize> = HashSet::new();
    for wkf in &window {
        for mp in wkf.get_map_point_matches().into_iter().flatten() {
            if !mp.is_bad() && mp_seen.insert(mp.id) {
                local_mps.push(mp);
            }
        }
    }

    // Assemble keyframes for the core: window (free) + fixed boundary.
    let mut kfs: Vec<InertialBaKf> = Vec::new();
    let mut kf_index: HashMap<u64, usize> = HashMap::new();
    let mut kf_arcs: Vec<Arc<KeyFrame>> = Vec::new();
    for wkf in &window {
        kf_index.insert(wkf.id, kfs.len());
        kfs.push(inertial_ba_kf_from(wkf, false));
        kf_arcs.push(wkf.clone());
    }
    if let Some(fb) = &fixed_boundary {
        kf_index.insert(fb.id, kfs.len());
        kfs.push(inertial_ba_kf_from(fb, true));
        kf_arcs.push(fb.clone());
    }

    // Inertial links between consecutive keyframes (each KF -> its prev).
    let n_window = window.len();
    let mut links: Vec<InertialLink> = Vec::new();
    for (i, wkf) in kf_arcs.iter().enumerate() {
        let Some(prev) = wkf.get_prev_kf() else {
            continue;
        };
        let (Some(&ci), Some(&pi)) = (kf_index.get(&wkf.id), kf_index.get(&prev.id)) else {
            continue;
        };
        let Some(preint) = wkf.imu_preintegrated.clone() else {
            continue;
        };
        // Boundary link (window edge) or bRecInit: robust kernel; boundary also
        // down-weights the information (mirrors upstream).
        let is_boundary = i == n_window - 1;
        links.push(InertialLink {
            prev: pi,
            cur: ci,
            preint,
            robust_delta: if is_boundary || rec_init {
                Some(16.92_f64.sqrt())
            } else {
                None
            },
            info_scale: if is_boundary { 1e-2 } else { 1.0 },
        });
    }

    // Points + observations.
    let mut points: Vec<Vector3<f64>> = Vec::with_capacity(local_mps.len());
    let mut mp_index: HashMap<usize, usize> = HashMap::new();
    for mp in &local_mps {
        mp_index.insert(mp.id, points.len());
        points.push(mp.get_world_pos().cast::<f64>());
    }
    let mut obs: Vec<InertialBaObs> = Vec::new();
    // Parallel metadata per observation for outlier culling.
    let mut obs_meta: Vec<(Arc<KeyFrame>, Arc<MapPoint>, bool, f32)> = Vec::new();
    for mp in &local_mps {
        let mp_i = mp_index[&mp.id];
        for (obs_kf, (left, _right)) in mp.get_observations() {
            let Some(&kf_i) = kf_index.get(&obs_kf.id) else {
                continue;
            };
            let left = left as i64;
            if left < 0 {
                continue;
            }
            let li = left as usize;
            let kp = &obs_kf.keys_un[li];
            let inv_sigma2 = obs_kf.inv_level_sigma2[kp.octave() as usize] as f64;
            let is_mono = obs_kf.u_right[li] < 0.0;
            if is_mono {
                obs.push(InertialBaObs::Mono {
                    kf: kf_i,
                    mp: mp_i,
                    obs: Vector2::new(kp.pt().x as f64, kp.pt().y as f64),
                    inv_sigma2,
                    cam_idx: 0,
                });
            } else {
                obs.push(InertialBaObs::Stereo {
                    kf: kf_i,
                    mp: mp_i,
                    obs: Vector3::new(
                        kp.pt().x as f64,
                        kp.pt().y as f64,
                        obs_kf.u_right[li] as f64,
                    ),
                    inv_sigma2,
                    cam_idx: 0,
                });
            }
            obs_meta.push((obs_kf.clone(), mp.clone(), is_mono, mp.track_depth));
        }
    }

    let num_fixed = fixed_boundary.is_some() as usize;
    let num_opt = window.len();
    let num_mps = local_mps.len();
    let num_edges = obs.len();

    let r = inertial_ba_core(&kfs, &links, &points, &obs, opt_it, user_lambda, stop_flag);

    // Divergence guard (mirrors upstream): bail without applying the result.
    if (2.0 * r.err_start < r.err_end || r.err_start.is_nan() || r.err_end.is_nan()) && !large {
        return (num_fixed, num_opt, num_mps, num_edges);
    }

    // Cull outlier observations (chi2 / depth), erasing the matches.
    let chi2_mono = 5.991;
    let chi2_stereo = 7.815;
    for (oi, (obs_kf, mp, is_mono, track_depth)) in obs_meta.iter().enumerate() {
        if mp.is_bad() {
            continue;
        }
        let chi2 = r.obs_chi2[oi];
        let bad = if *is_mono {
            let b_close = *track_depth < 10.0;
            (chi2 > chi2_mono && !b_close)
                || (chi2 > 1.5 * chi2_mono && b_close)
                || !r.obs_depth_positive[oi]
        } else {
            chi2 > chi2_stereo
        };
        if bad {
            obs_kf.erase_map_point_match(mp);
            mp.erase_observation(obs_kf);
        }
    }

    // Write back the window keyframes (skip the fixed boundary).
    for (i, wkf) in window.iter().enumerate() {
        let s = &r.states[i];
        wkf.set_pose(imu_state_to_tcw(s, &kfs[i].rbc, &kfs[i].tbc));
        wkf.set_velocity(s.vel.cast::<f32>());
        wkf.set_new_bias(crate::imu_types::Bias::from_params(
            s.ba[0] as f32,
            s.ba[1] as f32,
            s.ba[2] as f32,
            s.bg[0] as f32,
            s.bg[1] as f32,
            s.bg[2] as f32,
        ));
    }
    for (mp, p) in local_mps.iter().zip(r.points.iter()) {
        mp.set_world_pos(p.cast::<f32>());
        mp.update_normal_and_depth();
    }
    map.increase_change_index();
    (num_fixed, num_opt, num_mps, num_edges)
}

// ===========================================================================
// OptimizeEssentialGraph4DoF
// ===========================================================================

/// A keyframe camera pose `(Rcw, tcw)` for the 4-DoF pose graph.
pub struct Pose4DoF {
    pub rcw: nalgebra::Matrix3<f64>,
    pub tcw: Vector3<f64>,
    pub fixed: bool,
}

/// A relative 4-DoF constraint `Tij = (drij, dtij)` between vertices `i`, `j`.
pub struct Edge4DoFConstraint {
    pub i: usize,
    pub j: usize,
    pub drij: nalgebra::Matrix3<f64>,
    pub dtij: Vector3<f64>,
}

/// Core of `Optimizer::OptimizeEssentialGraph4DoF` (Optimizer.cc:5292):
/// a yaw+translation (4-DoF) pose graph for inertial loop closing. Returns the
/// optimized `(Rcw, tcw)` per vertex.
pub fn optimize_essential_graph_4dof_core(
    verts: &[Pose4DoF],
    edges: &[Edge4DoFConstraint],
    iterations: i32,
) -> Vec<(nalgebra::Matrix3<f64>, Vector3<f64>)> {
    let cam: Arc<dyn GeometricCamera> =
        Arc::new(crate::camera_models::pinhole::Pinhole::with_params(vec![
            1.0, 1.0, 0.0, 0.0,
        ]));
    let mut opt = SparseOptimizer::new();

    let mut vtx = Vec::with_capacity(verts.len());
    for v in verts {
        let icp = ImuCamPose::new(
            vec![v.rcw],
            vec![v.tcw],
            vec![nalgebra::Matrix3::identity()],
            vec![Vector3::zeros()],
            0.0,
            vec![cam.clone()],
        );
        let mut vp = VertexPose4DoF::new(icp);
        vp.set_fixed(v.fixed);
        vtx.push(opt.add_vertex(Box::new(vp)));
    }

    let mut mat_lambda = nalgebra::SMatrix::<f64, 6, 6>::identity();
    mat_lambda[(0, 0)] = 1e3;
    mat_lambda[(1, 1)] = 1e3;
    for e in edges {
        let mut ed = Edge4DoF::new(vtx[e.i], vtx[e.j], e.drij, e.dtij);
        ed.information = mat_lambda;
        opt.add_edge(Box::new(ed));
    }

    opt.initialize_optimization(0);
    opt.optimize(iterations);

    vtx.iter()
        .map(|&vi| {
            let p = opt
                .vertex(vi)
                .as_any()
                .downcast_ref::<VertexPose4DoF>()
                .unwrap()
                .estimate();
            (p.rcw[0], p.tcw[0])
        })
        .collect()
}

/// `Optimizer::OptimizeEssentialGraph4DoF` (Optimizer.cc:5292): the inertial
/// loop-closing yaw+translation pose graph over `&Map`, built from the
/// (non-)corrected Sim3 maps + keyframe poses, optimized via
/// [`optimize_essential_graph_4dof_core`], then recovered to SE3 keyframe poses
/// and applied to the map points.
pub fn optimize_essential_graph_4dof(
    map: &Map,
    loop_kf: &Arc<KeyFrame>,
    cur_kf: &Arc<KeyFrame>,
    non_corrected: &KeyFrameAndPose,
    corrected: &KeyFrameAndPose,
    loop_connections: &LoopConnections,
) {
    const MIN_FEAT: i32 = 100;
    let init_kf_id = map.get_init_kf_id();
    let mut kfs = map.get_all_key_frames();
    kfs.sort_by_key(|k| k.id);
    let max_kf_id = map.get_max_kf_id();
    kfs.retain(|k| k.id <= max_kf_id && !k.is_bad());
    let idx: HashMap<u64, usize> = kfs.iter().enumerate().map(|(i, k)| (k.id, i)).collect();

    // Per-keyframe Scw (corrected if available, else current pose).
    let vscw: Vec<Sim3> = kfs
        .iter()
        .map(|k| {
            corrected
                .get(&k.id)
                .copied()
                .unwrap_or_else(|| sim3_from_pose(&k.get_pose()))
        })
        .collect();
    let verts: Vec<Pose4DoF> = kfs
        .iter()
        .enumerate()
        .map(|(i, k)| Pose4DoF {
            rcw: vscw[i].rotation_matrix(),
            tcw: vscw[i].translation(),
            fixed: k.id == init_kf_id,
        })
        .collect();
    let siw_for = |kf: &Arc<KeyFrame>| -> Sim3 {
        non_corrected
            .get(&kf.id)
            .copied()
            .unwrap_or(vscw[idx[&kf.id]])
    };

    let mut edges: Vec<Edge4DoFConstraint> = Vec::new();
    let mut inserted: HashSet<(u64, u64)> = HashSet::new();
    let key = |a: u64, b: u64| (a.min(b), a.max(b));
    let rel = |si: &Sim3, sj: &Sim3| -> (nalgebra::Matrix3<f64>, Vector3<f64>) {
        // Tij such that Edge4DoF error vanishes: drij = Rcw_i Rcw_jᵀ, dtij = tcw_i - drij tcw_j.
        let (ri, ti) = (si.rotation_matrix(), si.translation());
        let (rj, tj) = (sj.rotation_matrix(), sj.translation());
        let drij = ri * rj.transpose();
        (drij, ti - drij * tj)
    };

    for (&id_i, conns) in loop_connections.iter() {
        let Some(&i) = idx.get(&id_i) else { continue };
        let si = vscw[i];
        for &id_j in conns {
            let Some(&j) = idx.get(&id_j) else { continue };
            if (id_i != cur_kf.id || id_j != loop_kf.id) && kfs[i].get_weight(&kfs[j]) < MIN_FEAT {
                continue;
            }
            let (drij, dtij) = rel(&vscw[j], &si);
            edges.push(Edge4DoFConstraint { i, j, drij, dtij });
            inserted.insert(key(id_i, id_j));
        }
    }
    for (i, kf) in kfs.iter().enumerate() {
        let si = siw_for(kf);
        if let Some(parent) = kf.get_parent() {
            if let Some(&j) = idx.get(&parent.id) {
                let (drij, dtij) = rel(&siw_for(&parent), &si);
                edges.push(Edge4DoFConstraint { i, j, drij, dtij });
            }
        }
        for lkf in kf.get_loop_edges() {
            if lkf.id < kf.id {
                if let Some(&j) = idx.get(&lkf.id) {
                    let (drij, dtij) = rel(&siw_for(&lkf), &si);
                    edges.push(Edge4DoFConstraint { i, j, drij, dtij });
                }
            }
        }
        for nkf in kf.get_covisibles_by_weight(MIN_FEAT) {
            let is_parent = kf.get_parent().map(|p| p.id == nkf.id).unwrap_or(false);
            if is_parent || kf.has_child(&nkf) || nkf.is_bad() || nkf.id >= kf.id {
                continue;
            }
            if inserted.contains(&key(kf.id, nkf.id)) {
                continue;
            }
            if let Some(&j) = idx.get(&nkf.id) {
                let (drij, dtij) = rel(&siw_for(&nkf), &si);
                edges.push(Edge4DoFConstraint { i, j, drij, dtij });
            }
        }
    }

    let out = optimize_essential_graph_4dof_core(&verts, &edges, 20);

    let mut vcorrected_swc: Vec<Sim3> = Vec::with_capacity(kfs.len());
    for (i, kf) in kfs.iter().enumerate() {
        let (rcw, tcw) = out[i];
        let csiw = Sim3::new(nalgebra::UnitQuaternion::from_matrix(&rcw), tcw, 1.0);
        vcorrected_swc.push(csiw.inverse());
        kf.set_pose(Isometry3::from_parts(
            nalgebra::Translation3::from(tcw.cast::<f32>()),
            nalgebra::UnitQuaternion::from_matrix(&rcw.cast::<f32>()),
        ));
    }
    for mp in map.get_all_map_points() {
        if mp.is_bad() {
            continue;
        }
        let Some(ref_kf) = mp.get_reference_keyframe() else {
            continue;
        };
        let Some(&ridx) = idx.get(&ref_kf.id) else {
            continue;
        };
        let corrected_p3dw =
            vcorrected_swc[ridx].map(&vscw[ridx].map(&mp.get_world_pos().cast::<f64>()));
        mp.set_world_pos(corrected_p3dw.cast::<f32>());
        mp.update_normal_and_depth();
    }
    map.increase_change_index();
}

// ===========================================================================
// MergeInertialBA
// ===========================================================================

/// `Optimizer::MergeInertialBA` (Optimizer.cc:3948).
///
/// Inertial bundle adjustment over a welding window spanning both maps being
/// merged: a sliding window around `curr_kf` and one around `merge_kf`, with the
/// keyframes just before each window held fixed. Builds on the validated
/// [`inertial_ba_core`]; the covisible visual-only keyframes upstream adds as
/// pose-only vertices are included here as fixed inertial keyframes.
pub fn merge_inertial_ba(
    curr_kf: &Arc<KeyFrame>,
    merge_kf: &Arc<KeyFrame>,
    stop_flag: Option<Arc<AtomicBool>>,
    map: &Map,
) -> KeyFrameAndPose {
    let nd = 6usize;
    // Window back from current KF.
    let mut window: Vec<Arc<KeyFrame>> = vec![curr_kf.clone()];
    for _ in 1..nd {
        match window.last().unwrap().get_prev_kf() {
            Some(p) => window.push(p),
            None => break,
        }
    }
    // Window around the merge KF (predecessors).
    let mut merge_window: Vec<Arc<KeyFrame>> = vec![merge_kf.clone()];
    for _ in 1..(nd / 2) {
        match merge_window.last().unwrap().get_prev_kf() {
            Some(p) => merge_window.push(p),
            None => break,
        }
    }

    let mut seen: HashSet<u64> = HashSet::new();
    let mut optimizable: Vec<Arc<KeyFrame>> = Vec::new();
    for k in window.iter().chain(merge_window.iter()) {
        if seen.insert(k.id) {
            optimizable.push(k.clone());
        }
    }
    // Fixed boundary: the predecessor of each window's last keyframe.
    let mut fixed: Vec<Arc<KeyFrame>> = Vec::new();
    for w in [&window, &merge_window] {
        if let Some(b) = w.last().and_then(|k| k.get_prev_kf()) {
            if !seen.contains(&b.id) && seen.insert(b.id) {
                fixed.push(b);
            }
        }
    }

    // Assemble core inputs.
    let mut kfs: Vec<InertialBaKf> = Vec::new();
    let mut kf_index: HashMap<u64, usize> = HashMap::new();
    let mut kf_arcs: Vec<Arc<KeyFrame>> = Vec::new();
    for k in &optimizable {
        kf_index.insert(k.id, kfs.len());
        kfs.push(inertial_ba_kf_from(k, false));
        kf_arcs.push(k.clone());
    }
    let n_opt = optimizable.len();
    for k in &fixed {
        kf_index.insert(k.id, kfs.len());
        kfs.push(inertial_ba_kf_from(k, true));
        kf_arcs.push(k.clone());
    }

    let mut links: Vec<InertialLink> = Vec::new();
    for k in &kf_arcs {
        let Some(prev) = k.get_prev_kf() else {
            continue;
        };
        let (Some(&ci), Some(&pi)) = (kf_index.get(&k.id), kf_index.get(&prev.id)) else {
            continue;
        };
        let Some(preint) = k.imu_preintegrated.clone() else {
            continue;
        };
        links.push(InertialLink {
            prev: pi,
            cur: ci,
            preint,
            robust_delta: None,
            info_scale: 1.0,
        });
    }

    let mut local_mps: Vec<Arc<MapPoint>> = Vec::new();
    let mut mp_seen: HashSet<usize> = HashSet::new();
    for k in &optimizable {
        for mp in k.get_map_point_matches().into_iter().flatten() {
            if !mp.is_bad() && mp_seen.insert(mp.id) {
                local_mps.push(mp);
            }
        }
    }
    let mut points: Vec<Vector3<f64>> = Vec::new();
    let mut mp_index: HashMap<usize, usize> = HashMap::new();
    for mp in &local_mps {
        mp_index.insert(mp.id, points.len());
        points.push(mp.get_world_pos().cast::<f64>());
    }
    let mut obs: Vec<InertialBaObs> = Vec::new();
    for mp in &local_mps {
        let mp_i = mp_index[&mp.id];
        for (obs_kf, (left, _r)) in mp.get_observations() {
            let Some(&kf_i) = kf_index.get(&obs_kf.id) else {
                continue;
            };
            let left = left as i64;
            if left < 0 {
                continue;
            }
            let li = left as usize;
            let kp = &obs_kf.keys_un[li];
            let inv_sigma2 = obs_kf.inv_level_sigma2[kp.octave() as usize] as f64;
            if obs_kf.u_right[li] < 0.0 {
                obs.push(InertialBaObs::Mono {
                    kf: kf_i,
                    mp: mp_i,
                    obs: Vector2::new(kp.pt().x as f64, kp.pt().y as f64),
                    inv_sigma2,
                    cam_idx: 0,
                });
            } else {
                obs.push(InertialBaObs::Stereo {
                    kf: kf_i,
                    mp: mp_i,
                    obs: Vector3::new(
                        kp.pt().x as f64,
                        kp.pt().y as f64,
                        obs_kf.u_right[li] as f64,
                    ),
                    inv_sigma2,
                    cam_idx: 0,
                });
            }
        }
    }

    let r = inertial_ba_core(&kfs, &links, &points, &obs, 8, 1e0, stop_flag);
    let (states, out_points) = (r.states, r.points);

    for (i, k) in kf_arcs.iter().take(n_opt).enumerate() {
        let s = &states[i];
        k.set_pose(imu_state_to_tcw(s, &kfs[i].rbc, &kfs[i].tbc));
        k.set_velocity(s.vel.cast::<f32>());
        k.set_new_bias(Bias::from_params(
            s.ba[0] as f32,
            s.ba[1] as f32,
            s.ba[2] as f32,
            s.bg[0] as f32,
            s.bg[1] as f32,
            s.bg[2] as f32,
        ));
    }
    for (mp, p) in local_mps.iter().zip(out_points.iter()) {
        mp.set_world_pos(p.cast::<f32>());
        mp.update_normal_and_depth();
    }
    map.increase_change_index();

    // Corrected poses of the optimized keyframes (`corrPoses` output).
    let mut corr_poses = KeyFrameAndPose::new();
    for (i, k) in kf_arcs.iter().take(n_opt).enumerate() {
        corr_poses.insert(
            k.id,
            sim3_from_pose(&imu_state_to_tcw(&states[i], &kfs[i].rbc, &kfs[i].tbc)),
        );
    }
    corr_poses
}

// ===========================================================================
// OptimizeEssentialGraph (real &Map wrappers)
// ===========================================================================

/// Stub of `LoopClosing::KeyFrameAndPose`: keyframe id -> corrected `Sim3` (`Scw`).
pub type KeyFrameAndPose = HashMap<u64, Sim3>;
/// Stub of the loop-connection map: keyframe id -> set of connected keyframe ids.
pub type LoopConnections = HashMap<u64, HashSet<u64>>;

fn sim3_from_pose(tcw: &Isometry3<f32>) -> Sim3 {
    Sim3::new(
        tcw.rotation.cast::<f64>(),
        tcw.translation.vector.cast::<f64>(),
        1.0,
    )
}

/// `Optimizer::OptimizeEssentialGraph` (Optimizer.cc:1501): the loop-closing
/// Sim3 pose graph. Builds vertices from the (non-)corrected Sim3 maps + KF
/// poses, adds loop / spanning-tree / loop-edge / covisibility / inertial
/// `EdgeSim3` constraints, optimizes via [`optimize_essential_graph_core`], then
/// recovers SE3 keyframe poses and transforms the map points.
#[allow(clippy::too_many_arguments)]
pub fn optimize_essential_graph(
    map: &Map,
    loop_kf: &Arc<KeyFrame>,
    cur_kf: &Arc<KeyFrame>,
    non_corrected: &KeyFrameAndPose,
    corrected: &KeyFrameAndPose,
    loop_connections: &LoopConnections,
    fix_scale: bool,
) {
    const MIN_FEAT: i32 = 100;
    let init_kf_id = map.get_init_kf_id();
    let mut kfs = map.get_all_key_frames();
    kfs.sort_by_key(|k| k.id);
    let max_kf_id = map.get_max_kf_id();
    kfs.retain(|k| k.id <= max_kf_id && !k.is_bad());

    let idx: HashMap<u64, usize> = kfs.iter().enumerate().map(|(i, k)| (k.id, i)).collect();
    let n = kfs.len();

    // Per-keyframe initial Scw (corrected if available, else the current pose).
    let mut vscw: Vec<Sim3> = Vec::with_capacity(n);
    let mut poses: Vec<Sim3> = Vec::with_capacity(n);
    let mut fixed = vec![false; n];
    for (i, kf) in kfs.iter().enumerate() {
        let siw = corrected
            .get(&kf.id)
            .copied()
            .unwrap_or_else(|| sim3_from_pose(&kf.get_pose()));
        vscw.push(siw);
        poses.push(siw);
        if kf.id == init_kf_id {
            fixed[i] = true;
        }
    }
    // `Siw` as used for edge measurements (non-corrected if present).
    let siw_for = |kf: &Arc<KeyFrame>| -> Sim3 {
        non_corrected
            .get(&kf.id)
            .copied()
            .unwrap_or(vscw[idx[&kf.id]])
    };

    let mut edges: Vec<EssentialGraphEdge> = Vec::new();
    let mut inserted: HashSet<(u64, u64)> = HashSet::new();
    let key = |a: u64, b: u64| (a.min(b), a.max(b));

    // Loop edges (LoopConnections).
    for (&id_i, conns) in loop_connections.iter() {
        let Some(&i) = idx.get(&id_i) else { continue };
        let siw = vscw[i];
        let swi = siw.inverse();
        let kf_i = &kfs[i];
        for &id_j in conns {
            let Some(&j) = idx.get(&id_j) else { continue };
            // Keep the (cur,loop) edge unconditionally, else require min weight.
            if (id_i != cur_kf.id || id_j != loop_kf.id) && kf_i.get_weight(&kfs[j]) < MIN_FEAT {
                continue;
            }
            let sji = vscw[j].mul(&swi);
            edges.push(EssentialGraphEdge { i, j, sji });
            inserted.insert(key(id_i, id_j));
        }
    }

    // Normal edges: spanning tree, loop edges, covisibility, inertial.
    for (i, kf) in kfs.iter().enumerate() {
        let swi = siw_for(kf).inverse();

        if let Some(parent) = kf.get_parent() {
            if let Some(&j) = idx.get(&parent.id) {
                let sjw = siw_for(&parent);
                edges.push(EssentialGraphEdge {
                    i,
                    j,
                    sji: sjw.mul(&swi),
                });
            }
        }
        for lkf in kf.get_loop_edges() {
            if lkf.id < kf.id {
                if let Some(&j) = idx.get(&lkf.id) {
                    let slw = siw_for(&lkf);
                    edges.push(EssentialGraphEdge {
                        i,
                        j,
                        sji: slw.mul(&swi),
                    });
                }
            }
        }
        for nkf in kf.get_covisibles_by_weight(MIN_FEAT) {
            let is_parent = kf.get_parent().map(|p| p.id == nkf.id).unwrap_or(false);
            if is_parent || kf.has_child(&nkf) || nkf.is_bad() || nkf.id >= kf.id {
                continue;
            }
            if inserted.contains(&key(kf.id, nkf.id)) {
                continue;
            }
            if let Some(&j) = idx.get(&nkf.id) {
                let snw = siw_for(&nkf);
                edges.push(EssentialGraphEdge {
                    i,
                    j,
                    sji: snw.mul(&swi),
                });
            }
        }
        if kf.imu {
            if let Some(prev) = kf.get_prev_kf() {
                if let Some(&j) = idx.get(&prev.id) {
                    let spw = siw_for(&prev);
                    edges.push(EssentialGraphEdge {
                        i,
                        j,
                        sji: spw.mul(&swi),
                    });
                }
            }
        }
    }

    let corrected_siw = optimize_essential_graph_core(&poses, &fixed, fix_scale, &edges, 20);

    // Recover SE3 keyframe poses (Sim3 [sR t] -> SE3 [R t/s]).
    let mut vcorrected_swc: Vec<Sim3> = Vec::with_capacity(n);
    for (i, kf) in kfs.iter().enumerate() {
        let csiw = corrected_siw[i];
        vcorrected_swc.push(csiw.inverse());
        let s = csiw.scale();
        let tiw = Isometry3::from_parts(
            nalgebra::Translation3::from((csiw.translation() / s).cast::<f32>()),
            csiw.rotation().cast::<f32>(),
        );
        kf.set_pose(tiw);
    }

    // Transform map points by their reference keyframe's correction.
    for mp in map.get_all_map_points() {
        if mp.is_bad() {
            continue;
        }
        let Some(ref_kf) = mp.get_reference_keyframe() else {
            continue;
        };
        let Some(&ridx) = idx.get(&ref_kf.id) else {
            continue;
        };
        let srw = vscw[ridx];
        let corrected_swr = vcorrected_swc[ridx];
        let p3dw = mp.get_world_pos().cast::<f64>();
        let corrected_p3dw = corrected_swr.map(&srw.map(&p3dw));
        mp.set_world_pos(corrected_p3dw.cast::<f32>());
        mp.update_normal_and_depth();
    }
    map.increase_change_index();
}

/// `Optimizer::OptimizeEssentialGraph` 2nd overload (Optimizer.cc:1785): the
/// map-merge welding pose graph. `fixed_kfs` (and `fixed_corrected_kfs`) keep
/// their pose; `non_fixed_kfs` are optimized; `non_corrected_mps` are
/// transformed by their reference keyframe's correction afterwards.
pub fn optimize_essential_graph_merge(
    cur_kf: &Arc<KeyFrame>,
    fixed_kfs: &[Arc<KeyFrame>],
    fixed_corrected_kfs: &[Arc<KeyFrame>],
    non_fixed_kfs: &[Arc<KeyFrame>],
    non_corrected_mps: &[Arc<MapPoint>],
) {
    const MIN_FEAT: i32 = 100;
    let Some(map) = cur_kf.get_map() else { return };

    // Collect all participating keyframes (fixed + fixed-corrected + non-fixed).
    let mut kfs: Vec<Arc<KeyFrame>> = Vec::new();
    let mut fixed_flags: Vec<bool> = Vec::new();
    let mut seen: HashSet<u64> = HashSet::new();
    for (group, is_fixed) in [
        (fixed_kfs, true),
        (fixed_corrected_kfs, true),
        (non_fixed_kfs, false),
    ] {
        for kf in group {
            if !kf.is_bad() && seen.insert(kf.id) {
                kfs.push(kf.clone());
                fixed_flags.push(is_fixed);
            }
        }
    }
    if kfs.is_empty() {
        return;
    }
    let idx: HashMap<u64, usize> = kfs.iter().enumerate().map(|(i, k)| (k.id, i)).collect();

    // Initial Scw from current keyframe poses.
    let vscw: Vec<Sim3> = kfs.iter().map(|k| sim3_from_pose(&k.get_pose())).collect();
    let poses = vscw.clone();
    let siw_for = |kf: &Arc<KeyFrame>| -> Sim3 { vscw[idx[&kf.id]] };

    let mut edges: Vec<EssentialGraphEdge> = Vec::new();
    let mut inserted: HashSet<(u64, u64)> = HashSet::new();
    let key = |a: u64, b: u64| (a.min(b), a.max(b));
    for (i, kf) in kfs.iter().enumerate() {
        let swi = siw_for(kf).inverse();
        if let Some(parent) = kf.get_parent() {
            if let Some(&j) = idx.get(&parent.id) {
                edges.push(EssentialGraphEdge {
                    i,
                    j,
                    sji: siw_for(&parent).mul(&swi),
                });
                inserted.insert(key(kf.id, parent.id));
            }
        }
        for lkf in kf.get_loop_edges() {
            if let Some(&j) = idx.get(&lkf.id) {
                if kf.id < lkf.id {
                    edges.push(EssentialGraphEdge {
                        i,
                        j,
                        sji: siw_for(&lkf).mul(&swi),
                    });
                    inserted.insert(key(kf.id, lkf.id));
                }
            }
        }
        for nkf in kf.get_covisibles_by_weight(MIN_FEAT) {
            if nkf.is_bad() || nkf.id >= kf.id || inserted.contains(&key(kf.id, nkf.id)) {
                continue;
            }
            if let Some(&j) = idx.get(&nkf.id) {
                edges.push(EssentialGraphEdge {
                    i,
                    j,
                    sji: siw_for(&nkf).mul(&swi),
                });
                inserted.insert(key(kf.id, nkf.id));
            }
        }
    }

    let corrected_siw = optimize_essential_graph_core(&poses, &fixed_flags, true, &edges, 20);

    let mut vcorrected_swc: Vec<Sim3> = Vec::with_capacity(kfs.len());
    for (i, kf) in kfs.iter().enumerate() {
        let csiw = corrected_siw[i];
        vcorrected_swc.push(csiw.inverse());
        let s = csiw.scale();
        let tiw = Isometry3::from_parts(
            nalgebra::Translation3::from((csiw.translation() / s).cast::<f32>()),
            csiw.rotation().cast::<f32>(),
        );
        kf.set_pose(tiw);
    }
    for mp in non_corrected_mps {
        if mp.is_bad() {
            continue;
        }
        let Some(ref_kf) = mp.get_reference_keyframe() else {
            continue;
        };
        let Some(&ridx) = idx.get(&ref_kf.id) else {
            continue;
        };
        let corrected_p3dw =
            vcorrected_swc[ridx].map(&vscw[ridx].map(&mp.get_world_pos().cast::<f64>()));
        mp.set_world_pos(corrected_p3dw.cast::<f32>());
        mp.update_normal_and_depth();
    }
    map.increase_change_index();
}

// ===========================================================================
// OptimizeEssentialGraph (core)
// ===========================================================================

/// A relative-Sim3 constraint `Sji` between vertices `i` (slot 0) and `j`
/// (slot 1): `error = log(Sji · Si · Sj⁻¹)`.
pub struct EssentialGraphEdge {
    pub i: usize,
    pub j: usize,
    pub sji: Sim3,
}

/// Core of [`optimize_essential_graph`], decoupled from `Map`/`KeyFrame`.
///
/// Sim3 pose-graph optimization: `poses[k]` is the initial `Skw`, `fixed[k]`
/// holds it constant. Returns the optimized `Skw` for every vertex.
pub fn optimize_essential_graph_core(
    poses: &[Sim3],
    fixed: &[bool],
    fix_scale: bool,
    edges: &[EssentialGraphEdge],
    iterations: i32,
) -> Vec<Sim3> {
    let mut optimizer = SparseOptimizer::new();
    optimizer.set_user_lambda_init(1e-16);

    let mut vtx = Vec::with_capacity(poses.len());
    for (k, s) in poses.iter().enumerate() {
        let mut v = VertexSim3Expmap::new(*s, fix_scale);
        v.set_fixed(fixed[k]);
        vtx.push(optimizer.add_vertex(Box::new(v)));
    }
    for e in edges {
        optimizer.add_edge(Box::new(EdgeSim3::new(vtx[e.i], vtx[e.j], e.sji)));
    }

    optimizer.initialize_optimization(0);
    optimizer.optimize(iterations);

    vtx.iter()
        .map(|&vi| {
            optimizer
                .vertex(vi)
                .as_any()
                .downcast_ref::<VertexSim3Expmap>()
                .unwrap()
                .estimate()
        })
        .collect()
}

fn sim3_estimate(optimizer: &SparseOptimizer, v: usize) -> Sim3 {
    optimizer
        .vertex(v)
        .as_any()
        .downcast_ref::<VertexSim3Expmap>()
        .unwrap()
        .estimate()
}

/// Core of [`pose_optimization`], decoupled from [`Frame`] for testing.
///
/// Returns `(optimized Tcw, n_inliers, outlier_flags)` where `outlier_flags`
/// pairs each observation's frame index with its final inlier/outlier state.
#[allow(clippy::too_many_arguments)]
pub fn pose_optimization_core(
    tcw_init: Isometry3<f32>,
    left_camera: &Arc<dyn GeometricCamera>,
    right_camera: Option<&Arc<dyn GeometricCamera>>,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    bf: f64,
    observations: &[PoseObs],
) -> (Isometry3<f32>, i32, Vec<(usize, bool)>) {
    let mut optimizer = SparseOptimizer::new();

    // Frame pose vertex (id 0).
    let mut v_se3 = VertexSE3Expmap::new(SE3Quat::from_isometry_f32(&tcw_init));
    v_se3.set_fixed(false);
    let v_pose = optimizer.add_vertex(Box::new(v_se3));

    let n_initial_correspondences = observations.len() as i32;
    let delta_mono = 5.991_f64.sqrt();
    let delta_stereo = 7.815_f64.sqrt();

    // outlier state per observation, in observation order.
    let mut outlier = vec![false; observations.len()];

    let mut recs: Vec<EdgeRec> = Vec::with_capacity(observations.len());
    for (oi, ob) in observations.iter().enumerate() {
        match ob {
            PoseObs::Mono {
                xw,
                obs,
                inv_sigma2,
                ..
            } => {
                let mut e = EdgeSE3ProjectXYZOnlyPose::new(v_pose, *xw, left_camera.clone());
                e.set_measurement(*obs);
                e.set_information(Matrix2::identity() * *inv_sigma2);
                e.set_robust_kernel(Some(delta_mono));
                let ei = optimizer.add_edge(Box::new(e));
                recs.push(EdgeRec {
                    ei,
                    idx: oi,
                    thr: CHI2_MONO,
                });
            }
            PoseObs::Stereo {
                xw,
                obs,
                inv_sigma2,
                ..
            } => {
                let mut e = EdgeStereoSE3ProjectXYZOnlyPose::new(v_pose, *xw, fx, fy, cx, cy, bf);
                e.set_measurement(*obs);
                e.set_information(Matrix3::identity() * *inv_sigma2);
                e.set_robust_kernel(Some(delta_stereo));
                let ei = optimizer.add_edge(Box::new(e));
                recs.push(EdgeRec {
                    ei,
                    idx: oi,
                    thr: CHI2_STEREO,
                });
            }
            PoseObs::MonoBody {
                xw,
                obs,
                inv_sigma2,
                m_trl,
                ..
            } => {
                let cam = right_camera.expect("MonoBody observation requires a right camera");
                let mut e = EdgeSE3ProjectXYZOnlyPoseToBody::new(v_pose, *xw, cam.clone(), *m_trl);
                e.set_measurement(*obs);
                e.set_information(Matrix2::identity() * *inv_sigma2);
                e.set_robust_kernel(Some(delta_mono));
                let ei = optimizer.add_edge(Box::new(e));
                recs.push(EdgeRec {
                    ei,
                    idx: oi,
                    thr: CHI2_MONO,
                });
            }
        }
    }

    if n_initial_correspondences < 3 {
        let flags = observations.iter().map(|o| (o.idx(), false)).collect();
        return (tcw_init, 0, flags);
    }

    let its = [10i32; 4];
    let mut n_bad = 0;
    for it in 0..4 {
        // Reset the estimate to the initial pose each pass (matches upstream).
        optimizer
            .vertex_mut(v_pose)
            .as_any_mut()
            .downcast_mut::<VertexSE3Expmap>()
            .unwrap()
            .set_estimate(SE3Quat::from_isometry_f32(&tcw_init));

        optimizer.initialize_optimization(0);
        optimizer.optimize(its[it]);

        n_bad = 0;
        for rec in &recs {
            if outlier[rec.idx] {
                optimizer.compute_edge_error(rec.ei);
            }
            let chi2 = optimizer.edge(rec.ei).chi2();
            if chi2 > rec.thr {
                outlier[rec.idx] = true;
                optimizer.edge_mut(rec.ei).set_level(1);
                n_bad += 1;
            } else {
                outlier[rec.idx] = false;
                optimizer.edge_mut(rec.ei).set_level(0);
            }
            if it == 2 {
                optimizer.edge_mut(rec.ei).set_robust_kernel(None);
            }
        }

        if optimizer.num_active_edges() < 10 {
            break;
        }
    }

    let recov = optimizer
        .vertex(v_pose)
        .as_any()
        .downcast_ref::<VertexSE3Expmap>()
        .unwrap()
        .estimate();

    let flags = observations
        .iter()
        .enumerate()
        .map(|(oi, o)| (o.idx(), outlier[oi]))
        .collect();
    (
        recov.to_isometry_f32(),
        n_initial_correspondences - n_bad,
        flags,
    )
}
