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

use nalgebra::{Matrix3, SMatrix, Vector3};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use tracing::info;

use crate::optimizer;
use crate::{
    atlas::Atlas, key_frame::KeyFrame, loop_closing::LoopClosing, map::Map, map_point::MapPoint,
    system::System, tracking::Tracking,
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
    system: Arc<System>,
    atlas: Arc<Atlas>,
    monocular: bool,
    inertial: bool,
    sequence: String,

    /// `false` to ignore far stereo points (forced off in upstream).
    far_points: bool,
    /// Depth threshold for far-point rejection (forced off in upstream).
    th_far_points: f32,

    loop_closer: OnceLock<Arc<LoopClosing>>,
    tracker: OnceLock<Arc<Tracking>>,

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
            system,
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
    pub fn set_tracker(&self, tracker: Arc<Tracking>) {
        let _ = self.tracker.set(tracker);
    }

    // Main function
    pub fn run(&self) {
        self.finish.lock().unwrap().finished = true;

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

                    self.abort_ba.store(true, Ordering::SeqCst);

                    if !self.check_new_key_frames() {
                        // Find more matches in neighbor keyframes and fuse point duplications
                        self.search_in_neighbors();
                    }
                });

                if !self.check_new_key_frames()
                    && !self.stop_requested()
                    && let Some(current_keyframe) = self.current_key_frame.lock().unwrap().as_ref()
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

                                if !map.get_inertial_ba2() {
                                    if self.imu_init.lock().unwrap().t_init < 10. && dist < 0.02 {
                                        info!("Not enough motion for initializing. Resetting...");
                                        {
                                            let mut reset = self.reset.lock().unwrap();
                                            reset.reset_requested_active_map = true;
                                            reset.map_to_reset = Some(map.clone());
                                            self.bad_imu.store(true, Ordering::SeqCst);
                                        }
                                    }
                                }

                                let inliers = self.tracker.get().unwrap().get_matches_inliers();
                                let large = ((inliers > 75) && self.monocular)
                                    || ((inliers > 100) && !self.monocular);
                                Some(optimizer::local_inertial_ba(
                                    current_keyframe,
                                    Some(self.abort_ba.clone()),
                                    &map,
                                    large,
                                    false,
                                ))
                            } else {
                                None
                            }
                        } else if let Some(map) = current_keyframe.get_map() {
                            Some(optimizer::local_bundle_adjustment(
                                current_keyframe,
                                Some(self.abort_ba.clone()),
                                &map,
                            ))
                        } else {
                            None
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

                    // Initialize the IMU here
                    if let Some(map) = current_keyframe.get_map()
                        && !map.is_imu_initialized()
                        && self.inertial
                    {
                        if self.monocular {
                            // TODO
                        } else {
                            // TODO
                        }
                    }

                    // TODO
                }
                // TODO
            }
        }
    }

    fn initialize_imu(&mut self, prior_g: f32, prior_a: f32, fiba: bool) {
        if self.reset.lock().unwrap().reset_requested {
            return;
        }

        let (min_time, min_kf): (f32, usize) = if self.monocular { (2.0, 10) } else { (1.0, 10) };

        if let Some(keyframes) = self.atlas.key_frames_in_map()
            && keyframes < min_kf
        {
            return;
        }

        // Retrieve all keyframe in temporal order
        let mut kfs: Vec<Arc<KeyFrame>> = Vec::new();
        let mut current = self.current_key_frame.lock().unwrap().clone();
        while let Some(kf) = current {
            kfs.push(kf.clone());
            current = kf.get_prev_kf().clone();
        }
        kfs.reverse();

        if kfs.len() < min_kf {
            return;
        }

        let first_ts = kfs[0].timestamp;

        // TODO
    }

    /// Enqueue a keyframe and request BA abort (upstream `InsertKeyFrame`).
    pub fn insert_key_frame(&self, kf: Arc<KeyFrame>) {
        // SAFETY of unwrap: poisoned only if another thread panicked while
        // holding the lock, which is unrecoverable here.
        self.new_key_frames.lock().unwrap().push_back(kf);
        self.abort_ba.store(true, Ordering::SeqCst);
    }

    pub fn empty_queue(&self) {
        // TODO: drain and process the queue (upstream `EmptyQueue`).
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
        if let Some(kf) = next {
            *self.current_key_frame.lock().unwrap() = Some(kf);
            // TODO: BoW, observations, connections, map insertion.
        }
    }
    fn create_new_map_points(&self) {}
    fn map_point_culling(&self) {}
    fn search_in_neighbors(&self) {}
    fn key_frame_culling(&self) {}

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
        _prior_g: f32, // default 1e2
        _prior_a: f32, // default 1e6
        _first: bool,  // default false
    ) {
    }
    fn scale_refinement(&self) {}
}
