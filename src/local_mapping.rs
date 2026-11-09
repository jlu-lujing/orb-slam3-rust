//! Local mapping thread.
//!
//! `LocalMapping` is shared as `Arc<LocalMapping>`: it owns its `run()` loop on a
//! dedicated `std::thread`, while `Tracking` (and `LoopClosing`) call into it
//! concurrently to enqueue keyframes and drive the stop/finish/reset handshakes.
//! Every method therefore takes `&self` and all mutable state lives behind
//! interior-mutability fields.
//!
//! # Locking strategy
//!
//! This mirrors the granularity of upstream ORB-SLAM3's `LocalMapping`, which
//! deliberately splits state across several `std::mutex`es rather than one big
//! lock. The split is load-bearing: the local-mapping thread holds the *map*
//! lock for a long time during bundle adjustment, while `Tracking` must keep
//! polling the lightweight control-plane flags (`accept_key_frames`,
//! `stop_requested`, …) without blocking on that heavy work.
//!
//! Two flags that upstream mutates *without* a lock (intentionally racy signals
//! polled by other threads) are modelled as atomics: `abort_ba` (the `bool*`
//! handed to the optimizer so it can bail out of BA) and `bad_imu` (read by
//! `Tracking` to trigger a map reset). `accept_key_frames` and `initializing`
//! are lone booleans behind their own upstream mutex — a plain `AtomicBool` is
//! a faithful, lock-free equivalent.
//!
//! ## Lock ordering (deadlock avoidance)
//!
//! Upstream contains a latent lock-order inversion: `Release()` takes
//! `mMutexStop` then `mMutexFinish`, while `SetFinish()` takes `mMutexFinish`
//! then `mMutexStop`. It survives only because the window is tiny. We do **not**
//! replicate it. The single global acquisition order is:
//!
//! ```text
//! finish  <  stop  <  new_key_frames
//! imu_init  <  Map::map_update          (BA / IMU init take the map lock last)
//! ```
//!
//! Never hold a lock that is "earlier" in this list while waiting on an
//! "earlier-still" one. `reset` is only ever taken alone.

use nalgebra::{
    Isometry3, Matrix3, Matrix3x4, Point3, Rotation3, SMatrix, Translation3, UnitQuaternion,
    Vector3,
};
use opencv::core::{KeyPoint, KeyPointTraitConst};
use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Duration;
use tracing::info;

use crate::{
    atlas::Atlas, geometric_tools, imu_types::Bias, key_frame::KeyFrame,
    key_frame_database::KeyFrameId, loop_closing::LoopClosing, map::Map, map_point::MapPoint,
    optimizer, orb_matcher::OrbMatcher, system::System, tracking::Tracking,
};

/// Reset-request state
#[derive(Default)]
struct ResetState {
    reset_requested: bool,
    reset_requested_active_map: bool,
    map_to_reset: Option<Arc<Map>>,
}

/// Finish handshake state
struct FinishState {
    finish_requested: bool,
    finished: bool,
}

/// Stop handshake state
#[derive(Default)]
struct StopState {
    stopped: bool,
    stop_requested: bool,
    not_stop: bool,
}

/// IMU-initialization parameters and statistics
/// Only touched by the local-mapping thread inside `initialize_imu` /
/// `scale_refinement` / `run`, but guarded so cross-thread readers stay sound.
struct ImuInitState {
    /// World→gravity rotation
    rwg: Matrix3<f64>,
    /// Gyroscope bias
    bg: Vector3<f64>,
    /// Accelerometer bias
    ba: Vector3<f64>,
    /// Scale factor
    scale: f64,
    init_time: f64,
    cost_time: f64,
    init_sect: usize,
    idx_init: usize,
    n_kfs: usize,
    first_ts: f64,
    matches_inliers: usize,

    // For debugging (erased in normal mode upstream).
    init_fr: usize,
    idx_iteration: usize,
    last_scale_refine: f64,

    not_ba1: bool,
    not_ba2: bool,

    /// Inertial information matrix
    info_inertial: SMatrix<f64, 9, 9>,
    num_lm: usize,
    num_kf_culling: usize,
    /// Elapsed time since first keyframe used for IMU init
    t_init: f64,
}

impl ImuInitState {
    fn new() -> Self {
        ImuInitState {
            rwg: Matrix3::identity(),
            bg: Vector3::zeros(),
            ba: Vector3::zeros(),
            scale: 1.,
            init_time: 0.,
            cost_time: 0.,
            init_sect: 0,
            idx_init: 0,
            n_kfs: 0,
            first_ts: 0.,
            matches_inliers: 0,
            init_fr: 0,
            idx_iteration: 0,
            last_scale_refine: -1.0,
            not_ba1: true,
            not_ba2: true,
            info_inertial: SMatrix::zeros(),
            num_lm: 0,
            num_kf_culling: 0,
            t_init: 0.,
        }
    }
}

/// Per-stage timing/diagnostic counters
#[cfg(feature = "register-times")]
#[derive(Default)]
struct RegisterTimes {
    kf_insert_ms: Vec<f64>,
    mp_culling_ms: Vec<f64>,
    mp_creation_ms: Vec<f64>,
    lba_ms: Vec<f64>,
    kf_culling_ms: Vec<f64>,
    lm_total_ms: Vec<f64>,
    lba_sync_ms: Vec<f64>,
    kf_culling_sync_ms: Vec<f64>,
    lba_edges: Vec<usize>,
    lba_kf_opt: Vec<usize>,
    lba_kf_fixed: Vec<usize>,
    lba_mps: Vec<usize>,
    lba_exec: usize,
    lba_abort: usize,
}

/// Times `$body` and—when the `register-times` feature is active—appends the
/// elapsed milliseconds to `$rt.lock().unwrap().$field`.
///
/// When the feature is disabled the macro reduces to a plain expression, so
/// call sites need no `#[cfg]` attributes at all.
macro_rules! timed {
    ($rt:expr, $field:ident, $body:expr) => {{
        #[cfg(feature = "register-times")]
        let _t0 = ::std::time::Instant::now();
        let _v = { $body };
        #[cfg(feature = "register-times")]
        {
            $rt.lock()
                .unwrap()
                .$field
                .push(_t0.elapsed().as_secs_f64() * 1000.);
        }
        _v
    }};
}

pub struct LocalMapping {
    // --- immutable wiring (set once at/after construction) -----------------
    system: Weak<System>,
    atlas: Arc<Atlas>,
    monocular: bool,
    inertial: bool,
    sequence: String,

    /// `false` to ignore far stereo points (forced off in upstream).
    far_points: bool,
    /// Depth threshold for far-point rejection (forced off in upstream).
    th_far_points: f32,

    loop_closer: OnceLock<Arc<LoopClosing>>,
    tracker: OnceLock<Weak<Mutex<Tracking>>>,

    // --- lock-free signals (intentionally racy in upstream) ---------------
    /// BA abort flag — the optimizer polls this to bail out early
    abort_ba: Arc<AtomicBool>,
    /// Whether new keyframes are currently accepted
    accept_key_frames: Arc<AtomicBool>,
    /// IMU initialization in progress
    initializing: Arc<AtomicBool>,
    /// Bad-IMU flag polled by `Tracking` to request a map reset
    bad_imu: Arc<AtomicBool>,

    // --- locked state (see module-level lock ordering) --------------------
    /// New-keyframe queue (upstream `mMutexNewKFs` / `mlNewKeyFrames`).
    new_key_frames: Mutex<VecDeque<Arc<KeyFrame>>>,
    /// Keyframe currently being processed (`mpCurrentKeyFrame`).
    current_key_frame: Mutex<Option<Arc<KeyFrame>>>,
    /// Map points recently created, pending culling (`mlpRecentAddedMapPoints`).
    recent_added_map_points: Mutex<Vec<Arc<MapPoint>>>,

    reset: Mutex<ResetState>,
    finish: Mutex<FinishState>,
    stop: Mutex<StopState>,
    imu_init: Mutex<ImuInitState>,

    #[cfg(feature = "register-times")]
    register_times: Mutex<RegisterTimes>,
}

impl LocalMapping {
    pub fn new(
        system: Arc<System>,
        atlas: Arc<Atlas>,
        monocular: bool,
        inertial: bool,
        sequence: String,
    ) -> Self {
        LocalMapping {
            system: Arc::downgrade(&system),
            atlas,
            monocular,
            inertial,
            sequence,
            far_points: false, // forced off in the C++ code
            th_far_points: 0., // forced off in the C++ code
            loop_closer: OnceLock::new(),
            tracker: OnceLock::new(),
            abort_ba: Arc::new(AtomicBool::new(false)),
            accept_key_frames: Arc::new(AtomicBool::new(true)),
            initializing: Arc::new(AtomicBool::new(false)),
            bad_imu: Arc::new(AtomicBool::new(false)),
            new_key_frames: Mutex::new(VecDeque::new()),
            current_key_frame: Mutex::new(None),
            recent_added_map_points: Mutex::new(Vec::new()),
            reset: Mutex::new(ResetState::default()),
            finish: Mutex::new(FinishState {
                finish_requested: false,
                finished: true,
            }),
            stop: Mutex::new(StopState::default()),
            imu_init: Mutex::new(ImuInitState::new()),
            #[cfg(feature = "register-times")]
            register_times: Mutex::new(RegisterTimes::default()),
        }
    }

    pub fn set_loop_closer(&self, loop_closer: Arc<LoopClosing>) {
        let _ = self.loop_closer.set(loop_closer);
    }
    pub fn set_tracker(&self, tracker: Arc<Mutex<Tracking>>) {
        let _ = self.tracker.set(Arc::downgrade(&tracker));
    }

    // Main function
    pub fn run(&self) {
        self.finish.lock().unwrap().finished = false;

        loop {
            // tracking will see that local mapping is busy
            self.set_accept_key_frames(false);

            // check if there are keyframes in the queue
            if self.check_new_key_frames() && !self.bad_imu.load(Ordering::SeqCst) {
                // BoW conversion and insertion in Map
                timed!(
                    self.register_times,
                    kf_insert_ms,
                    self.process_new_key_frame()
                );

                // Check recent MapPoints
                timed!(self.register_times, mp_culling_ms, self.map_point_culling());

                timed!(self.register_times, mp_creation_ms, {
                    // Triangulate new MapPoints
                    self.create_new_map_points();

                    self.abort_ba.store(false, Ordering::SeqCst);

                    if !self.check_new_key_frames() {
                        // Find more matches in neighbor keyframes and fuse point duplications
                        self.search_in_neighbors();
                    }
                });

                if !self.check_new_key_frames()
                    && !self.stop_requested()
                    && let Some(current_keyframe) = self.get_curr_kf()
                {
                    if let Some(key_frames) = self.atlas.key_frames_in_map()
                        && key_frames > 2
                        && let Some(map) = current_keyframe.get_map()
                    {
                        #[cfg(feature = "register-times")]
                        let start = std::time::Instant::now();
                        let result = if self.inertial && map.is_imu_initialized() {
                            if let Some(previous_kf) = current_keyframe.get_prev_kf()
                                && let Some(previous_previous_kf) = previous_kf.get_prev_kf()
                            {
                                let dist = (previous_kf.get_camera_center()
                                    - current_keyframe.get_camera_center())
                                .norm()
                                    + (previous_previous_kf.get_camera_center()
                                        - previous_kf.get_camera_center())
                                    .norm();

                                if dist > 0.05 {
                                    self.imu_init.lock().unwrap().t_init +=
                                        current_keyframe.timestamp - previous_kf.timestamp;
                                }

                                if !map.get_inertial_ba2()
                                    && self.imu_init.lock().unwrap().t_init < 10.
                                    && dist < 0.02
                                {
                                    info!("Not enough motion for initializing. Resetting...");
                                    {
                                        let mut reset = self.reset.lock().unwrap();
                                        reset.reset_requested_active_map = true;
                                        reset.map_to_reset = Some(map.clone());
                                        self.bad_imu.store(true, Ordering::SeqCst);
                                    }
                                }

                                let inliers = self
                                    .tracker
                                    .get()
                                    .and_then(Weak::upgrade)
                                    .map(|tracker| tracker.lock().unwrap().get_matches_inliers())
                                    .unwrap_or(0);
                                let large = ((inliers > 75) && self.monocular)
                                    || ((inliers > 100) && !self.monocular);
                                Some(optimizer::local_inertial_ba(
                                    &current_keyframe,
                                    Some(self.abort_ba.clone()),
                                    &map,
                                    large,
                                    false,
                                ))
                            } else {
                                None
                            }
                        } else {
                            current_keyframe.get_map().map(|map| {
                                optimizer::local_bundle_adjustment(
                                    &current_keyframe,
                                    Some(self.abort_ba.clone()),
                                    &map,
                                )
                            })
                        };
                        #[cfg(feature = "register-times")]
                        if let Some(result) = result {
                            let duration = start.elapsed().as_secs_f64() * 1000.;
                            let mut register = self.register_times.lock().unwrap();
                            register.lba_ms.push(duration);
                            register.lba_exec += 1;
                            if self.abort_ba.load(Ordering::SeqCst) {
                                register.lba_abort += 1;
                            }
                            register.lba_kf_fixed.push(result.0);
                            register.lba_kf_opt.push(result.1);
                            register.lba_mps.push(result.2);
                            register.lba_edges.push(result.3);
                        }
                    }

                    if let Some(map) = current_keyframe.get_map()
                        && !map.is_imu_initialized()
                        && self.inertial
                    {
                        let prior_a = if self.monocular { 1e10 } else { 1e5 };
                        self.initialize_imu(1e2, prior_a, true);
                    }

                    if self.inertial
                        && self.monocular
                        && current_keyframe
                            .get_map()
                            .is_some_and(|map| map.is_imu_initialized())
                    {
                        let elapsed = self.imu_init.lock().unwrap().t_init;
                        let in_refinement_window = [
                            (25.0, 25.5),
                            (35.0, 35.5),
                            (45.0, 45.5),
                            (55.0, 55.5),
                            (65.0, 65.5),
                            (75.0, 75.5),
                        ]
                        .iter()
                        .any(|(start, end)| elapsed > *start && elapsed < *end);
                        if in_refinement_window {
                            let should_refine = {
                                let mut imu = self.imu_init.lock().unwrap();
                                let ready = elapsed - imu.last_scale_refine >= 5.0;
                                if ready {
                                    imu.last_scale_refine = elapsed;
                                }
                                ready
                            };
                            if should_refine {
                                self.scale_refinement();
                            }
                        }
                    }

                    // Redundant-keyframe culling is a separate mapping stage.
                    self.key_frame_culling();
                }

                if let Some(current) = self.get_curr_kf()
                    && !current.is_bad()
                    && let Some(loop_closer) = self.loop_closer.get()
                {
                    loop_closer.insert_key_frame(current);
                }
            } else if self.stop() && !self.bad_imu.load(Ordering::SeqCst) {
                while self.is_stopped() && !self.check_finish() {
                    std::thread::sleep(Duration::from_micros(3000));
                }
                if self.check_finish() {
                    break;
                }
            }

            self.reset_if_requested();
            self.set_accept_key_frames(true);

            if self.check_finish() {
                break;
            }

            std::thread::sleep(Duration::from_micros(3000));
        }

        self.set_finished();
    }

    /// Enqueue a keyframe and request BA abort (upstream `InsertKeyFrame`).
    pub fn insert_key_frame(&self, kf: Arc<KeyFrame>) {
        // SAFETY of unwrap: poisoned only if another thread panicked while
        // holding the lock, which is unrecoverable here.
        self.new_key_frames.lock().unwrap().push_back(kf);
        self.abort_ba.store(true, Ordering::SeqCst);
    }

    pub fn empty_queue(&self) {
        while self.check_new_key_frames() {
            self.process_new_key_frame();
        }
    }

    // --- Thread sync -------------------------------------------------------

    /// Request the loop to stop and abort any in-flight BA.
    pub fn request_stop(&self) {
        self.stop.lock().unwrap().stop_requested = true;
        self.abort_ba.store(true, Ordering::SeqCst);
    }

    /// Request an atlas reset and block until the loop has processed it.
    pub fn request_reset(&self) {
        {
            self.reset.lock().unwrap().reset_requested = true;
        }
        loop {
            if !self.reset.lock().unwrap().reset_requested {
                break;
            }
            std::thread::sleep(Duration::from_micros(3000));
        }
    }

    /// Request an active-map reset and block until processed.
    pub fn request_reset_active_map(&self, map: Arc<Map>) {
        {
            let mut reset = self.reset.lock().unwrap();
            reset.reset_requested_active_map = true;
            reset.map_to_reset = Some(map);
        }
        loop {
            if !self.reset.lock().unwrap().reset_requested_active_map {
                break;
            }
            std::thread::sleep(Duration::from_micros(3000));
        }
    }

    /// Try to enter the stopped state. Returns `true` if it actually stopped.
    pub fn stop(&self) -> bool {
        let mut stop = self.stop.lock().unwrap();
        if stop.stop_requested && !stop.not_stop {
            stop.stopped = true;
            return true;
        }
        false
    }

    /// Resume the loop and clear the keyframe queue
    /// Acquires `finish` before `stop` to honour the global lock order (this is
    /// the deadlock fix relative to upstream's `Release`).
    pub fn release(&self) {
        let finish = self.finish.lock().unwrap();
        if finish.finished {
            return;
        }
        {
            let mut stop = self.stop.lock().unwrap();
            stop.stopped = false;
            stop.stop_requested = false;
        }
        self.new_key_frames.lock().unwrap().clear();
        drop(finish);
    }

    pub fn is_stopped(&self) -> bool {
        self.stop.lock().unwrap().stopped
    }
    pub fn stop_requested(&self) -> bool {
        self.stop.lock().unwrap().stop_requested
    }
    pub fn accept_key_frames(&self) -> bool {
        self.accept_key_frames.load(Ordering::SeqCst)
    }
    pub fn set_accept_key_frames(&self, flag: bool) {
        self.accept_key_frames.store(flag, Ordering::SeqCst);
    }

    /// Set the not-stop flag. Returns `false` if it could not be set because
    /// the loop is already stopped
    pub fn set_not_stop(&self, flag: bool) -> bool {
        let mut stop = self.stop.lock().unwrap();
        if flag && stop.stopped {
            return false;
        }
        stop.not_stop = flag;
        true
    }

    /// Signal the optimizer to abort the current bundle adjustment.
    pub fn interrupt_ba(&self) {
        self.abort_ba.store(true, Ordering::SeqCst);
    }

    pub fn request_finish(&self) {
        self.finish.lock().unwrap().finish_requested = true;
    }
    pub fn is_finished(&self) -> bool {
        self.finish.lock().unwrap().finished
    }

    pub fn keyframes_in_queue(&self) -> usize {
        self.new_key_frames.lock().unwrap().len()
    }
    pub fn is_initializing(&self) -> bool {
        self.initializing.load(Ordering::SeqCst)
    }
    pub fn get_curr_kf_time(&self) -> f64 {
        match &*self.current_key_frame.lock().unwrap() {
            Some(kf) => kf.timestamp,
            None => 0.,
        }
    }
    /// Elapsed IMU-initialization interval (`mTinit` upstream).
    pub fn get_time_from_imu_init(&self) -> f64 {
        let elapsed = self.imu_init.lock().unwrap().t_init;
        if elapsed > 0.0 && self.atlas.is_imu_initialized().unwrap_or(false) {
            elapsed
        } else {
            0.0
        }
    }
    pub fn get_curr_kf(&self) -> Option<Arc<KeyFrame>> {
        self.current_key_frame.lock().unwrap().clone()
    }

    /// Whether the bad-IMU flag is set (polled by `Tracking`).
    pub fn is_bad_imu(&self) -> bool {
        self.bad_imu.load(Ordering::SeqCst)
    }

    // --- internal loop steps -----------------------

    fn check_new_key_frames(&self) -> bool {
        !self.new_key_frames.lock().unwrap().is_empty()
    }
    fn process_new_key_frame(&self) {
        let next = self.new_key_frames.lock().unwrap().pop_front();
        let Some(kf) = next else {
            return;
        };
        *self.current_key_frame.lock().unwrap() = Some(kf.clone());

        // BoW is already computed when a Frame is promoted to KeyFrame. The
        // descriptors and vocabulary are immutable from that point onward.
        let mut recent = Vec::new();
        for (idx, mp) in kf.get_map_point_matches().into_iter().enumerate() {
            let Some(mp) = mp else {
                continue;
            };
            if mp.is_bad() {
                continue;
            }

            if mp.is_in_keyframe(&kf) {
                recent.push(mp);
            } else {
                mp.add_observation(&kf, idx as i32);
                mp.update_normal_and_depth();
                mp.compute_distinctive_descriptors();
            }
        }

        if !recent.is_empty() {
            self.recent_added_map_points.lock().unwrap().extend(recent);
        }

        kf.update_connections();
        self.atlas.add_key_frame(kf);
    }
    fn create_new_map_points(&self) {
        let Some(current) = self.get_curr_kf() else {
            return;
        };
        let Some(map) = current.get_map() else {
            return;
        };

        let nn = if self.monocular { 30 } else { 10 };
        let mut neighbors = current.get_best_covisibility_keyframes(nn);
        if self.inertial {
            let mut previous = Some(current.clone());
            for _ in 0..nn {
                let Some(prev) = previous.and_then(|kf| kf.get_prev_kf()) else {
                    break;
                };
                if !neighbors.iter().any(|kf| kf.id == prev.id) {
                    neighbors.push(prev.clone());
                }
                previous = Some(prev);
            }
        }

        let matcher = OrbMatcher::default();
        let mut created = Vec::new();

        for (neighbor_index, neighbor) in neighbors.iter().enumerate() {
            if neighbor_index > 0 && self.check_new_key_frames() {
                break;
            }

            let ow1_left = current.get_camera_center();
            let ow2_left = neighbor.get_camera_center();
            let baseline = (ow2_left - ow1_left).norm();
            if !self.monocular {
                if baseline < neighbor.b {
                    continue;
                }
            } else {
                let median_depth = neighbor.compute_scene_median_depth(2);
                if median_depth <= 0.0 || baseline / median_depth < 0.01 {
                    continue;
                }
            }

            let mut matched_pairs = Vec::new();
            matcher.search_for_triangulation(&current, neighbor, &mut matched_pairs, false, false);

            for (idx1, idx2) in matched_pairs {
                let kp1 = keyframe_keypoint(&current, idx1);
                let kp2 = keyframe_keypoint(neighbor, idx2);
                let right1 = keyframe_keypoint_is_right(&current, idx1);
                let right2 = keyframe_keypoint_is_right(neighbor, idx2);

                let (pose1, camera1) = if current.camera2.is_some() && right1 {
                    (
                        current.get_right_pose(),
                        current.camera2.as_ref().expect("right camera").clone(),
                    )
                } else {
                    (current.get_pose(), current.camera.clone())
                };
                let (pose2, camera2) = if neighbor.camera2.is_some() && right2 {
                    (
                        neighbor.get_right_pose(),
                        neighbor.camera2.as_ref().expect("right camera").clone(),
                    )
                } else {
                    (neighbor.get_pose(), neighbor.camera.clone())
                };

                let rcw1 = pose1.rotation.to_rotation_matrix().into_inner();
                let rcw2 = pose2.rotation.to_rotation_matrix().into_inner();
                let rwc1 = rcw1.transpose();
                let rwc2 = rcw2.transpose();
                let ow1 = pose1.inverse().translation.vector;
                let ow2 = pose2.inverse().translation.vector;

                let xn1 = camera1.unproject_n(&kp1.pt());
                let xn2 = camera2.unproject_n(&kp2.pt());
                let ray1 = rwc1 * xn1;
                let ray2 = rwc2 * xn2;
                let denom = ray1.norm() * ray2.norm();
                if denom <= f32::EPSILON {
                    continue;
                }
                let cos_parallax_rays = ray1.dot(&ray2) / denom;

                let u_right1 = current.u_right.get(idx1).copied().unwrap_or(-1.0);
                let u_right2 = neighbor.u_right.get(idx2).copied().unwrap_or(-1.0);
                let stereo1 = current.camera2.is_none() && u_right1 >= 0.0;
                let stereo2 = neighbor.camera2.is_none() && u_right2 >= 0.0;

                let cos_parallax_stereo1 = if stereo1 {
                    let depth = current.depth.get(idx1).copied().unwrap_or(-1.0);
                    if depth > 0.0 {
                        (2.0 * (current.b * 0.5).atan2(depth)).cos()
                    } else {
                        2.0
                    }
                } else {
                    2.0
                };
                let cos_parallax_stereo2 = if stereo2 {
                    let depth = neighbor.depth.get(idx2).copied().unwrap_or(-1.0);
                    if depth > 0.0 {
                        (2.0 * (neighbor.b * 0.5).atan2(depth)).cos()
                    } else {
                        2.0
                    }
                } else {
                    2.0
                };
                let cos_parallax_stereo = cos_parallax_stereo1.min(cos_parallax_stereo2);

                let x3d = if cos_parallax_rays < cos_parallax_stereo
                    && cos_parallax_rays > 0.0
                    && (stereo1
                        || stereo2
                        || (cos_parallax_rays < 0.9996 && self.inertial)
                        || (cos_parallax_rays < 0.9998 && !self.inertial))
                {
                    let x1 = Vector3::new(xn1.x, xn1.y, 1.0);
                    let x2 = Vector3::new(xn2.x, xn2.y, 1.0);
                    let tcw1 = isometry_to_matrix3x4(&pose1);
                    let tcw2 = isometry_to_matrix3x4(&pose2);
                    geometric_tools::triangulate(&x1, &x2, &tcw1, &tcw2)
                } else if stereo1 && cos_parallax_stereo1 < cos_parallax_stereo2 {
                    current.unproject_stereo(idx1)
                } else if stereo2 && cos_parallax_stereo2 < cos_parallax_stereo1 {
                    neighbor.unproject_stereo(idx2)
                } else {
                    None
                };
                let Some(x3d) = x3d else {
                    continue;
                };

                let z1 = rcw1.row(2).transpose().dot(&x3d) + pose1.translation.z;
                let z2 = rcw2.row(2).transpose().dot(&x3d) + pose2.translation.z;
                if z1 <= 0.0 || z2 <= 0.0 {
                    continue;
                }

                let level1 = kp1.octave() as usize;
                let level2 = kp2.octave() as usize;
                if level1 >= current.level_sigma2.len() || level2 >= neighbor.level_sigma2.len() {
                    continue;
                }

                let p1 = rcw1 * x3d + pose1.translation.vector;
                let p2 = rcw2 * x3d + pose2.translation.vector;
                if !reprojection_is_valid(
                    &current,
                    idx1,
                    &p1,
                    &camera1,
                    u_right1,
                    stereo1,
                    current.level_sigma2[level1],
                ) || !reprojection_is_valid(
                    neighbor,
                    idx2,
                    &p2,
                    &camera2,
                    u_right2,
                    stereo2,
                    neighbor.level_sigma2[level2],
                ) {
                    continue;
                }

                let dist1 = (x3d - ow1).norm();
                let dist2 = (x3d - ow2).norm();
                if dist1 <= f32::EPSILON || dist2 <= f32::EPSILON {
                    continue;
                }
                if self.far_points && (dist1 >= self.th_far_points || dist2 >= self.th_far_points) {
                    continue;
                }

                let ratio_dist = dist2 / dist1;
                let ratio_octave = current.scale_factors[level1] / neighbor.scale_factors[level2];
                let ratio_factor = 1.5 * current.scale_factor;
                if ratio_dist * ratio_factor < ratio_octave
                    || ratio_dist > ratio_octave * ratio_factor
                {
                    continue;
                }

                let mp = MapPoint::from_pos_ref(&x3d, current.clone(), map.clone());
                mp.add_observation(&current, idx1 as i32);
                mp.add_observation(neighbor, idx2 as i32);
                current.add_map_point(mp.clone(), idx1);
                neighbor.add_map_point(mp.clone(), idx2);
                mp.compute_distinctive_descriptors();
                mp.update_normal_and_depth();
                self.atlas.add_map_point(mp.clone());
                created.push(mp);
            }
        }

        if !created.is_empty() {
            self.recent_added_map_points.lock().unwrap().extend(created);
        }
    }
    fn map_point_culling(&self) {
        let current_kf_id = match self.get_curr_kf() {
            Some(kf) => kf.id,
            None => return,
        };
        let min_observations = if self.monocular { 2 } else { 3 };
        let pending = std::mem::take(&mut *self.recent_added_map_points.lock().unwrap());
        let mut retained = Vec::with_capacity(pending.len());

        for mp in pending {
            if mp.is_bad() || mp.get_found_ratio() < 0.25 {
                mp.set_bad_flag();
                continue;
            }

            // Tracking-created points carry upstream's sentinel `mnFirstKFid=-1`
            // (`None` here) until they have been observed for several frames.
            let first_kf_id = mp.first_kf_id.map(i128::from).unwrap_or(-1);
            let age = current_kf_id as i128 - first_kf_id;
            if age >= 2 && mp.observations() <= min_observations {
                mp.set_bad_flag();
            } else if age < 3 {
                retained.push(mp);
            }
        }

        *self.recent_added_map_points.lock().unwrap() = retained;
    }
    fn search_in_neighbors(&self) {
        let Some(current) = self.get_curr_kf() else {
            return;
        };
        let nn = if self.monocular { 30 } else { 10 };
        let mut target = Vec::new();
        let mut target_ids = HashSet::new();

        for neighbor in current.get_best_covisibility_keyframes(nn) {
            if !neighbor.is_bad() && target_ids.insert(neighbor.id) {
                target.push(neighbor);
            }
        }

        let first_layer = target.len();
        for i in 0..first_layer {
            for second in target[i].get_best_covisibility_keyframes(20) {
                if second.id != current.id && !second.is_bad() && target_ids.insert(second.id) {
                    target.push(second);
                }
            }
            if self.abort_ba.load(Ordering::SeqCst) {
                break;
            }
        }

        if self.inertial {
            let mut previous = current.get_prev_kf();
            while target.len() < 20 {
                let Some(kf) = previous else {
                    break;
                };
                previous = kf.get_prev_kf();
                if !kf.is_bad() && target_ids.insert(kf.id) {
                    target.push(kf);
                }
            }
        }

        let matcher = OrbMatcher::default();
        let current_matches = current.get_map_point_matches();
        for kf in &target {
            matcher.fuse(kf, &current_matches, 3.0, false);
            if kf.camera2.is_some() {
                matcher.fuse(kf, &current_matches, 3.0, true);
            }
        }

        if self.abort_ba.load(Ordering::SeqCst) {
            return;
        }

        let mut candidates = Vec::new();
        let mut candidate_ids = HashSet::new();
        for kf in &target {
            for mp in kf.get_map_point_matches().into_iter().flatten() {
                if !mp.is_bad() && candidate_ids.insert(mp.id) {
                    candidates.push(Some(mp));
                }
            }
        }
        matcher.fuse(&current, &candidates, 3.0, false);
        if current.camera2.is_some() {
            matcher.fuse(&current, &candidates, 3.0, true);
        }

        for mp in current.get_map_point_matches().into_iter().flatten() {
            if !mp.is_bad() {
                mp.compute_distinctive_descriptors();
                mp.update_normal_and_depth();
            }
        }

        current.update_connections();
    }
    fn key_frame_culling(&self) {
        let Some(current) = self.get_curr_kf() else {
            return;
        };

        let init_kf_id = current
            .get_map()
            .map(|map| map.get_init_kf_id())
            .unwrap_or(0);
        let redundant_ratio = 0.9;
        let last_window_id = if self.inertial {
            let mut last = current.clone();
            for _ in 0..21 {
                let Some(previous) = last.get_prev_kf() else {
                    break;
                };
                last = previous;
            }
            last.id
        } else {
            u64::MAX
        };

        for kf in current.get_vector_covisible_keyframes() {
            if kf.id == init_kf_id || kf.is_bad() {
                continue;
            }

            let mut n_map_points = 0usize;
            let mut n_redundant = 0usize;
            for (idx, mp) in kf.get_map_point_matches().into_iter().enumerate() {
                let Some(mp) = mp else {
                    continue;
                };
                if mp.is_bad() {
                    continue;
                }
                if !self.monocular {
                    let depth = kf.depth.get(idx).copied().unwrap_or(-1.0);
                    if depth < 0.0 || depth > kf.th_depth {
                        continue;
                    }
                }

                n_map_points += 1;
                if mp.observations() <= 3 {
                    continue;
                }

                let level = keyframe_keypoint(&kf, idx).octave();
                let mut observations = 0;
                for (other, (left, right)) in mp.get_observations() {
                    if other.id == kf.id || other.is_bad() {
                        continue;
                    }

                    let mut other_level = i32::MAX;
                    if left >= 0 {
                        other_level = keyframe_keypoint(&other, left as usize).octave();
                    }
                    if right >= 0 {
                        let right_level = keyframe_keypoint(&other, right as usize).octave();
                        other_level = other_level.min(right_level);
                    }

                    if other_level <= level + 1 {
                        observations += 1;
                        if observations > 3 {
                            break;
                        }
                    }
                }

                if observations > 3 {
                    n_redundant += 1;
                }
            }

            if n_map_points > 0 && n_redundant as f32 > redundant_ratio * n_map_points as f32 {
                if !self.inertial {
                    kf.set_bad_flag();
                    if let Some(database) = self.atlas.get_key_frame_database() {
                        database.erase(KeyFrameId(kf.id), &kf.bow_vec);
                    }
                    continue;
                }
                let Some(map) = kf.get_map() else {
                    continue;
                };
                if map.key_frames_in_map() <= 21 || kf.id > current.id.saturating_sub(2) {
                    continue;
                }
                let (Some(previous), Some(next)) = (kf.get_prev_kf(), kf.get_next_kf()) else {
                    continue;
                };
                let temporal_span = next.timestamp - previous.timestamp;
                let merge_for_initialization =
                    map.is_imu_initialized() && kf.id < last_window_id && temporal_span < 3.0;
                let merge_for_short_interval = temporal_span < 0.5;
                let merge_for_low_motion = !map.get_inertial_ba2()
                    && (kf.get_imu_position() - previous.get_imu_position()).norm() < 0.02
                    && temporal_span < 3.0;
                if !(merge_for_initialization || merge_for_short_interval || merge_for_low_motion) {
                    continue;
                }
                let (Some(previous_preintegrated), Some(next_preintegrated)) = (
                    previous.get_imu_preintegrated(),
                    next.get_imu_preintegrated(),
                ) else {
                    continue;
                };
                let mut merged = next_preintegrated.as_ref().clone();
                merged.merge_previous(&previous_preintegrated);
                next.set_imu_preintegrated(Some(Arc::new(merged)));
                next.set_prev_kf(&previous);
                previous.set_next_kf(&next);
                kf.clear_prev_kf();
                kf.clear_next_kf();
                kf.set_bad_flag();
                if let Some(database) = self.atlas.get_key_frame_database() {
                    database.erase(KeyFrameId(kf.id), &kf.bow_vec);
                }
            }
        }
    }

    /// Apply a pending reset if one was requested
    fn reset_if_requested(&self) {
        let mut reset = self.reset.lock().unwrap();
        if !reset.reset_requested && !reset.reset_requested_active_map {
            return;
        }
        self.new_key_frames.lock().unwrap().clear();
        self.recent_added_map_points.lock().unwrap().clear();

        // Inertial parameters
        {
            let mut imu = self.imu_init.lock().unwrap();
            imu.t_init = 0.;
            imu.not_ba1 = true;
            imu.not_ba2 = true;
            if reset.reset_requested {
                imu.idx_init = 0;
            }
        }
        self.bad_imu.store(false, Ordering::SeqCst);

        reset.reset_requested = false;
        reset.reset_requested_active_map = false;
        reset.map_to_reset = None;
    }

    fn check_finish(&self) -> bool {
        self.finish.lock().unwrap().finish_requested
    }

    /// Mark the loop finished and stopped
    /// Acquires `finish` before `stop`, matching the global lock order.
    fn set_finished(&self) {
        let mut finish = self.finish.lock().unwrap();
        finish.finished = true;
        self.stop.lock().unwrap().stopped = true;
    }

    fn initialize_imu(
        &self,
        prior_g: f32, // default 1e2
        prior_a: f32, // default 1e6
        first: bool,  // default false
    ) {
        if self.reset.lock().unwrap().reset_requested {
            return;
        }
        let Some(current) = self.get_curr_kf() else {
            return;
        };
        let Some(map) = current.get_map() else {
            return;
        };

        let min_time = if self.monocular { 2.0 } else { 1.0 };
        let min_keyframes = 10usize;
        if map.key_frames_in_map() < min_keyframes {
            return;
        }

        let mut chain = Vec::new();
        let mut cursor = Some(current.clone());
        let mut seen = HashSet::new();
        while let Some(kf) = cursor {
            if !seen.insert(kf.id) {
                break;
            }
            cursor = kf.get_prev_kf();
            chain.push(kf);
        }
        chain.reverse();
        if chain.len() < min_keyframes || current.timestamp - chain[0].timestamp < min_time {
            return;
        }

        self.initializing.store(true, Ordering::SeqCst);
        while self.check_new_key_frames() {
            self.process_new_key_frame();
        }
        if let Some(updated) = self.get_curr_kf()
            && chain.last().is_none_or(|kf| kf.id != updated.id)
        {
            chain.push(updated);
        }

        let mut rwg = Matrix3::<f64>::identity();
        let mut scale = 1.0;
        let mut bg = current.get_gyro_bias().cast::<f64>();
        let mut ba = current.get_acc_bias().cast::<f64>();
        let mut imu = self.imu_init.lock().unwrap();
        imu.first_ts = chain[0].timestamp;

        if !map.is_imu_initialized() {
            let mut gravity_direction = Vector3::<f32>::zeros();
            for kf in &chain {
                let Some(previous) = kf.get_prev_kf() else {
                    continue;
                };
                let Some(preintegrated) = kf.get_imu_preintegrated() else {
                    continue;
                };
                gravity_direction -=
                    previous.get_imu_rotation() * preintegrated.get_updated_delta_velocity();
                let dt = (kf.timestamp - previous.timestamp) as f32;
                if dt > 0.0 {
                    let velocity = (kf.get_imu_position() - previous.get_imu_position()) / dt;
                    kf.set_velocity(velocity);
                    previous.set_velocity(velocity);
                }
            }

            if gravity_direction.norm() > 1e-6 {
                gravity_direction.normalize_mut();
                let gravity_inertial = Vector3::new(0.0, 0.0, -1.0);
                let cross = gravity_inertial.cross(&gravity_direction);
                let axis_norm = cross.norm();
                if axis_norm > 1e-6 {
                    let angle = gravity_inertial
                        .dot(&gravity_direction)
                        .clamp(-1.0, 1.0)
                        .acos();
                    rwg = Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(cross), angle)
                        .into_inner()
                        .cast::<f64>();
                }
            }
            imu.rwg = rwg;
            imu.t_init = current.timestamp - chain[0].timestamp;
        } else {
            rwg = imu.rwg;
            bg = imu.bg;
            ba = imu.ba;
        }

        imu.init_time = current.timestamp - chain[0].timestamp;
        let start = std::time::Instant::now();
        optimizer::inertial_optimization(
            &map,
            &mut rwg,
            &mut scale,
            &mut bg,
            &mut ba,
            self.monocular,
            false,
            prior_g as f64,
            prior_a as f64,
        );
        imu.cost_time = start.elapsed().as_secs_f64();

        if scale < 0.1 || !scale.is_finite() {
            self.initializing.store(false, Ordering::SeqCst);
            return;
        }

        imu.rwg = rwg;
        imu.scale = scale;
        imu.bg = bg;
        imu.ba = ba;
        imu.n_kfs = chain.len();

        let rotation = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(
            rwg.transpose().cast::<f32>(),
        ));
        let transform = Isometry3::from_parts(Translation3::new(0.0, 0.0, 0.0), rotation);
        if (scale - 1.0).abs() > 1e-5 || !self.monocular {
            map.apply_scaled_rotation(transform, scale as f32, true);
        }
        let bias = Bias::from_params(
            ba[0] as f32,
            ba[1] as f32,
            ba[2] as f32,
            bg[0] as f32,
            bg[1] as f32,
            bg[2] as f32,
        );
        for kf in &chain {
            kf.set_inertial();
            kf.set_new_bias(bias);
        }
        if !map.is_imu_initialized() {
            map.set_imu_initialized();
        }
        if let Some(tracker) = self.tracker.get().and_then(Weak::upgrade)
            && let Ok(mut tracker) = tracker.lock()
        {
            tracker.update_frame_imu(scale as f32, bias, current.clone());
        }

        if first && (prior_a != 0.0 || prior_g != 0.0) {
            optimizer::full_inertial_ba(
                &map,
                100,
                false,
                current.id,
                Some(self.abort_ba.clone()),
                true,
                prior_g,
                prior_a,
            );
        }
        map.increase_change_index();
        self.initializing.store(false, Ordering::SeqCst);
    }
    fn scale_refinement(&self) {
        if self.reset.lock().unwrap().reset_requested {
            return;
        }
        let Some(current) = self.get_curr_kf() else {
            return;
        };
        let Some(map) = current.get_map() else {
            return;
        };
        if !map.is_imu_initialized() {
            return;
        }
        self.initializing.store(true, Ordering::SeqCst);
        while self.check_new_key_frames() {
            self.process_new_key_frame();
        }
        let mut rwg = Matrix3::<f64>::identity();
        let mut scale = 1.0;
        optimizer::inertial_optimization_gravity_scale(&map, &mut rwg, &mut scale);
        if scale >= 0.1 && scale.is_finite() && (scale - 1.0).abs() > 0.002 {
            let rotation = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(
                rwg.transpose().cast::<f32>(),
            ));
            map.apply_scaled_rotation(
                Isometry3::from_parts(Translation3::new(0.0, 0.0, 0.0), rotation),
                scale as f32,
                true,
            );
            if let Some(tracker) = self.tracker.get().and_then(Weak::upgrade)
                && let Ok(mut tracker) = tracker.lock()
            {
                tracker.update_frame_imu(scale as f32, current.get_imu_bias(), current.clone());
            }
        }
        map.increase_change_index();
        self.initializing.store(false, Ordering::SeqCst);
    }
}

fn keyframe_keypoint(kf: &KeyFrame, idx: usize) -> &KeyPoint {
    match kf.n_left {
        None => &kf.keys_un[idx],
        Some(n_left) if idx < n_left => &kf.keys[idx],
        Some(n_left) => &kf.keys_right.as_ref().expect("keys_right")[idx - n_left],
    }
}

fn keyframe_keypoint_is_right(kf: &KeyFrame, idx: usize) -> bool {
    kf.n_left.is_some_and(|n_left| idx >= n_left)
}

fn isometry_to_matrix3x4(tcw: &Isometry3<f32>) -> Matrix3x4<f32> {
    let r = tcw.rotation.to_rotation_matrix().into_inner();
    let t = tcw.translation.vector;
    Matrix3x4::new(
        r[(0, 0)],
        r[(0, 1)],
        r[(0, 2)],
        t[0],
        r[(1, 0)],
        r[(1, 1)],
        r[(1, 2)],
        t[1],
        r[(2, 0)],
        r[(2, 1)],
        r[(2, 2)],
        t[2],
    )
}

fn reprojection_is_valid(
    kf: &KeyFrame,
    idx: usize,
    point_camera: &Vector3<f32>,
    camera: &Arc<dyn crate::camera_models::GeometricCamera>,
    right_coordinate: f32,
    is_stereo: bool,
    sigma_square: f32,
) -> bool {
    let keypoint = keyframe_keypoint(kf, idx);
    if !is_stereo {
        let uv = camera.project_n(&Point3::from(*point_camera));
        let ex = uv.x - keypoint.pt().x;
        let ey = uv.y - keypoint.pt().y;
        return ex * ex + ey * ey <= 5.991 * sigma_square;
    }

    let invz = 1.0 / point_camera.z;
    let u = kf.fx * point_camera.x * invz + kf.cx;
    let v = kf.fy * point_camera.y * invz + kf.cy;
    let ur = u - kf.bf * invz;
    let ex = u - keypoint.pt().x;
    let ey = v - keypoint.pt().y;
    let er = ur - right_coordinate;
    ex * ex + ey * ey + er * er <= 7.8 * sigma_square
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::build_frame;

    fn make_mapping(atlas: Arc<Atlas>) -> LocalMapping {
        LocalMapping::new(
            Arc::new(System::new_for_test(crate::system::Sensor::Stereo)),
            atlas,
            true,
            false,
            "test".to_owned(),
        )
    }

    #[test]
    fn empty_queue_registers_keyframes_observations_and_connections() {
        let atlas = Arc::new(Atlas::from_kf_id(0));
        let map = atlas.get_current_map();
        let frame = build_frame();
        let a = KeyFrame::from_frame(&frame, map.clone());
        let b = KeyFrame::from_frame(&frame, map.clone());

        for i in 0..16 {
            let pos = Vector3::new(i as f32, 0.0, 5.0);
            let mp = MapPoint::from_pos_ref(&pos, a.clone(), map.clone());
            mp.add_observation(&a, i);
            mp.add_observation(&b, i);
            a.add_map_point(mp.clone(), i as usize);
            b.add_map_point(mp.clone(), i as usize);
            map.add_map_point(mp);
        }

        let mapping = make_mapping(atlas.clone());
        mapping.insert_key_frame(a.clone());
        mapping.insert_key_frame(b.clone());
        mapping.empty_queue();

        assert_eq!(mapping.keyframes_in_queue(), 0);
        assert_eq!(mapping.get_curr_kf().expect("current keyframe").id, b.id);
        assert_eq!(atlas.key_frames_in_map(), Some(2));
        assert_eq!(a.get_weight(&b), 16);
        assert_eq!(b.get_weight(&a), 16);
        for i in 0..16 {
            let mp = a.get_map_point(i).expect("map point match");
            assert!(mp.is_in_keyframe(&a));
            assert!(mp.is_in_keyframe(&b));
        }
    }

    #[test]
    fn map_point_culling_removes_poor_early_points() {
        let atlas = Arc::new(Atlas::from_kf_id(0));
        let map = atlas.get_current_map();
        let kf = KeyFrame::from_frame(&build_frame(), map.clone());
        let mapping = make_mapping(atlas);
        mapping.insert_key_frame(kf.clone());
        mapping.empty_queue();

        let mp = MapPoint::from_pos_ref(&Vector3::new(0.0, 0.0, 5.0), kf.clone(), map);
        mp.add_observation(&kf, 0);
        mp.increase_visible(4);
        assert!(mp.get_found_ratio() < 0.25);
        mapping
            .recent_added_map_points
            .lock()
            .unwrap()
            .push(mp.clone());

        mapping.map_point_culling();

        assert!(mp.is_bad());
        assert!(mapping.recent_added_map_points.lock().unwrap().is_empty());
    }
}
