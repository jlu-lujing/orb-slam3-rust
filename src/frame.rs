//! A handful of structural decisions differ from the C++ original to stay idiomatic and lock-free:
//!
//! * The C++ `Frame` keeps a pile of **static** members (`fx`, `fy`, `cx`,
//!   `cy`, the undistorted image bounds and the grid-cell sizes) guarded by the
//!   one-shot `mbInitialComputations` flag. Those are per-calibration constants,
//!   not per-frame state, so they live in a shared [`FrameConstants`] held
//!   behind an `Arc` instead.
//!
//! * Optional data that only exists for some sensor configurations
//!   (`mvKeysRight`, `mvKeysUn`, the right grid, the stereo-fisheye matches, …)
//!   is modelled with `Option`/empty collections rather than the sentinel
//!   values (`Nleft == -1`) used in C++.
//!
//! ## Threading / mutability
//!
//! In ORB-SLAM3 a `Frame` is, in practice, **owned by the Tracking thread**.
//! The only field the C++ code actually guards with a mutex (`mpMutexImu`) is
//! the `mbImuPreintegrated` boolean, which LocalMapping flips once it has
//! finished preintegrating the frame's IMU measurements while Tracking polls
//! it.
//!
//! The cleanest lock-free shape for the Rust port, as implemented here:
//!
//! * Keep `Frame` single-owner inside Tracking and hand out **immutable
//!   snapshots** via `Arc<Frame>` to other threads. Pose/feature data is then
//!   read-only for everyone but the owner, so `get_pose`/`set_pose` need no
//!   synchronisation — `set_pose` takes `&mut self`, which the borrow checker
//!   already proves is exclusive. The same goes for the owner-only `is_set` /
//!   `has_pose` / `has_velocity` flags.
//! * The one genuinely cross-thread field (`is_imu_preintegrated`, the C++
//!   `mpMutexImu`-guarded `mbImuPreintegrated`) is a single [`AtomicBool`]
//!   accessed with `Ordering::Acquire`/`Release`. It is one word, never blocks,
//!   and expresses the exact publish/observe handshake LocalMapping↔Tracking
//!   need: LocalMapping calls `set_integrated(&self)` — note the shared `&self`,
//!   the atomic gives interior mutability — and Tracking polls
//!   `imu_is_preintegrated()`. This is strictly cheaper than a `std::mutex` and
//!   removes the only lock that was on the hot path.
//! * Anything that must be *mutated* through a shared `Arc` (e.g. a
//!   `MapPoint`'s tracking scratch fields written by `is_in_frustum`) is pushed
//!   out of the shared object: `is_in_frustum` is pure and **returns** the
//!   projection instead of writing it back, so no interior mutability / lock is
//!   needed on the shared `MapPoint`.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use nalgebra::{Isometry3, Matrix3, Point3, Vector3};
use opencv::calib3d::undistort_points;
use opencv::core::{
    CV_32F, KeyPoint, KeyPointTrait, KeyPointTraitConst, Mat, MatTrait, MatTraitConst,
    NORM_HAMMING, NORM_L1, Point2f, Rect, Scalar, Vector,
};
use opencv::core::{DMatch, norm2, vconcat2};
use opencv::features2d::{BFMatcher, prelude::DescriptorMatcherTraitConst};
use opencv::prelude::MatTraitConstManual;

use crate::camera_models::GeometricCamera;
use crate::camera_models::kannala_brandt8::KannalaBrandt8;
use crate::converter::mat_to_matrix3f;
use crate::g2o_types::ConstraintPoseIMU;
use crate::imu_types::{Bias, Calib, Preintegrated};
use crate::key_frame::KeyFrame;
use crate::map_point::MapPoint;
use crate::orb_extractor::{ExtractionError, OrbExtractResult, OrbExtractor};
use crate::orb_matcher::{TH_HIGH, TH_LOW, descriptor_distance};
use crate::orb_vocabulary::{BowVector, DESC_LEN, Descriptor, FeatureVector, OrbVocabulary};

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

/// Current value of the global frame-id counter (for serialization).
pub(crate) fn peek_next_frame_id() -> usize {
    NEXT_ID.load(Ordering::SeqCst)
}
/// Restore the global frame-id counter (after deserialization).
pub(crate) fn set_next_frame_id(v: usize) {
    NEXT_ID.store(v, Ordering::SeqCst);
}

/// Lock-free boolean flag with `Clone` semantics suitable for a `#[derive(Clone)]`
/// struct: cloning snapshots the current value into a fresh atomic.
///
/// Used for the cross-thread `is_imu_preintegrated` flag so LocalMapping can
/// publish it through a shared `&Frame` without a mutex (see the module docs).
#[derive(Debug)]
pub struct AtomicFlag(AtomicBool);

impl AtomicFlag {
    #[inline]
    pub fn new(value: bool) -> Self {
        AtomicFlag(AtomicBool::new(value))
    }
    /// Observe the flag (`Acquire`).
    #[inline]
    pub fn get(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
    /// Publish the flag (`Release`).
    #[inline]
    pub fn set(&self, value: bool) {
        self.0.store(value, Ordering::Release);
    }
}

impl Clone for AtomicFlag {
    fn clone(&self) -> Self {
        AtomicFlag::new(self.get())
    }
}

/// Number of rows/cols in the keypoint-to-cell grid.
pub const FRAME_GRID_ROWS: usize = 48;
pub const FRAME_GRID_COLS: usize = 64;

#[derive(Clone)]
pub struct Frame {
    // Current Frame id
    pub id: usize,

    // Vocabulary used for relocalization
    pub orb_vocabulary: Arc<OrbVocabulary>,

    // Feature extractor. The right is used only in the stereo case.
    pub extractor_left: Arc<OrbExtractor>,
    pub extractor_right: Arc<OrbExtractor>,

    // Frame timestamp
    pub timestamp: f64,

    // Per-camera precomputed constants (intrinsics, bounds, grid sizing, K).
    // Replaces the C++ `Frame` statics guarded by `mbInitialComputations`.
    pub constants: Arc<FrameConstants>,

    // Stereo baseline multiplied by fx
    pub b_fx: f32,

    // Stereo baseline in meters
    pub b: f32,

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.
    pub th_depth: f32,

    // Number of KeyPoints
    pub n: usize,

    // Vector of keypoints (originally for visualization) and undistorted (actually used by the system).
    // In the stereo case, keys_un is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    pub keys: Vec<KeyPoint>,
    pub keys_right: Option<Vec<KeyPoint>>,
    pub keys_un: Option<Vec<KeyPoint>>,

    // Corresponding stereo coordinate and depth for each keypoint
    pub map_points: Vec<Option<Arc<MapPoint>>>,
    // "Monocular" keypoints have a negative value
    pub u_right: Vec<f32>,
    pub depth: Vec<f32>,

    // Bag of Word Vector structures
    pub bow_vec: BowVector,
    pub feat_vec: FeatureVector,

    // ORB descriptor, each row associated to a keypoint
    pub descriptors: Mat,
    pub descriptors_right: Option<Mat>,

    // Flag to identify outlier associations.
    pub outlier: Vec<bool>,
    pub close_mps: usize,

    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints
    pub grid: Vec<Vec<usize>>,

    // Prediction bias
    pub pred_bias: Bias,

    // IMU bias
    pub imu_bias: Bias,

    // IMU calibration
    pub imu_calib: Arc<Calib>,

    // IMU preintegration from last keyframe
    pub imu_preintegrated: Option<Arc<Preintegrated>>,
    pub last_keyframe: Option<Arc<KeyFrame>>,

    // Pointer to previous frame
    pub prev_frame: Option<Arc<Frame>>,
    pub imu_preintegrated_frame: Option<Arc<Preintegrated>>,

    // Reference KeyFrame
    pub reference_kf: Option<Arc<KeyFrame>>,

    // Scale pyramid info
    pub scale_levels: usize,
    pub scale_factor: f32,
    pub log_scale_factor: f32,
    pub scale_factors: Vec<f32>,
    pub inv_scale_factors: Vec<f32>,
    pub level_sigma2: Vec<f32>,
    pub inv_level_sigma2: Vec<f32>,

    pub project_points: HashMap<usize, Point2f>,
    pub matched_in_image: HashMap<usize, Point2f>,

    pub name_file: String,
    pub dataset: usize,

    pub camera: Arc<dyn GeometricCamera>,
    pub camera2: Option<Arc<dyn GeometricCamera>>,

    // Number of KeyPoints extracted in the left and right images
    pub n_left: Option<usize>,
    pub n_right: Option<usize>,
    // Number of non lapping KeyPoints
    pub mono_left: Option<usize>,
    pub mono_right: Option<usize>,

    // For stereo matching. `usize::MAX` marks an unmatched keypoint.
    pub left_to_right_match: Option<Vec<usize>>,
    pub right_to_left_match: Option<Vec<usize>>,

    // Triangulated stereo observations using as reference the left camera.
    // These are computed during compute_stereo_fish_eye_matches
    pub stereo_3d_points: Option<Vec<Vector3<f32>>>,

    // Grid for the right image
    pub grid_right: Vec<Vec<usize>>,

    #[cfg(feature = "register-times")]
    pub time_orb_ext: f64,
    #[cfg(feature = "register-times")]
    pub time_stereo_match: f64,

    // Optimization constraint set by the optimizer.
    pub cpi: Option<ConstraintPoseIMU>,

    pub img_left: Option<Mat>,
    pub img_right: Option<Mat>,

    // nalgebra migration
    t_cw: Isometry3<f32>,
    r_wc: Matrix3<f32>,
    o_w: Vector3<f32>,
    r_cw: Matrix3<f32>,
    t_cw_vec: Vector3<f32>,
    has_pose: bool,

    t_lr: Isometry3<f32>,
    t_rl: Isometry3<f32>,
    r_lr: Matrix3<f32>,
    t_lr_vec: Vector3<f32>,

    // IMU linear velocity
    vw: Vector3<f32>,
    has_velocity: bool,

    is_set: bool,
    // Cross-thread flag flipped by LocalMapping once preintegration is done and
    // polled by Tracking. Lock-free `AtomicBool` instead of the C++ `mpMutexImu`
    // (see the module-level "Threading / mutability analysis").
    is_imu_preintegrated: AtomicFlag,
}

/// Per-camera projection of a [`MapPoint`] computed by [`Frame::is_in_frustum`].
///
/// In C++ these values are written straight into the `MapPoint`'s `mTrack*`
/// scratch fields. Returning them instead keeps the shared `MapPoint`
/// immutable (see the module-level threading analysis); the caller copies them
/// where it needs them.
#[derive(Debug, Clone, Copy, Default)]
pub struct TrackInfo {
    pub proj_x: f32,
    pub proj_y: f32,
    /// Right-image x coordinate (`u - bf/z` for rectified stereo).
    pub proj_xr: f32,
    pub depth: f32,
    pub scale_level: i32,
    pub view_cos: f32,
}

/// Result of a frustum check: the left and/or right projection, if visible.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrustumResult {
    pub left: Option<TrackInfo>,
    pub right: Option<TrackInfo>,
}

impl FrustumResult {
    /// `true` if the point projects into either camera.
    pub fn in_view(&self) -> bool {
        self.left.is_some() || self.right.is_some()
    }
}

/// Scale-pyramid info copied out of an [`OrbExtractor`] into every [`Frame`].
struct ScaleInfo {
    scale_levels: usize,
    scale_factor: f32,
    log_scale_factor: f32,
    scale_factors: Vec<f32>,
    inv_scale_factors: Vec<f32>,
    level_sigma2: Vec<f32>,
    inv_level_sigma2: Vec<f32>,
}

impl ScaleInfo {
    fn from_extractor(extractor: &OrbExtractor) -> Self {
        let scale_factor = extractor.get_scale_factor();
        ScaleInfo {
            scale_levels: extractor.get_levels(),
            scale_factor,
            log_scale_factor: scale_factor.ln(),
            scale_factors: extractor.get_scale_factors(),
            inv_scale_factors: extractor.get_inverse_scale_factors(),
            level_sigma2: extractor.get_scale_sigma2(),
            inv_level_sigma2: extractor.get_inverse_scale_sigma2(),
        }
    }
}

/// Shared inputs that every constructor forwards to [`Frame::assemble`].
struct FrameConfig {
    orb_vocabulary: Arc<OrbVocabulary>,
    extractor_left: Arc<OrbExtractor>,
    extractor_right: Arc<OrbExtractor>,
    timestamp: f64,
    constants: Arc<FrameConstants>,
    b_fx: f32,
    b: f32,
    th_depth: f32,
    camera: Arc<dyn GeometricCamera>,
    camera2: Option<Arc<dyn GeometricCamera>>,
    imu_calib: Arc<Calib>,
    prev_frame: Option<Arc<Frame>>,
}

/// Sensor-specific feature data produced by each constructor and merged into a
/// [`Frame`] by [`Frame::assemble`].
struct ExtractedFrame {
    keys: Vec<KeyPoint>,
    keys_right: Option<Vec<KeyPoint>>,
    keys_un: Option<Vec<KeyPoint>>,
    descriptors: Mat,
    descriptors_right: Option<Mat>,
    u_right: Vec<f32>,
    depth: Vec<f32>,
    n: usize,
    n_left: Option<usize>,
    n_right: Option<usize>,
    mono_left: Option<usize>,
    mono_right: Option<usize>,
    left_to_right_match: Option<Vec<usize>>,
    right_to_left_match: Option<Vec<usize>>,
    stereo_3d_points: Option<Vec<Vector3<f32>>>,
    grid: Vec<Vec<usize>>,
    grid_right: Vec<Vec<usize>>,
    close_mps: usize,
    /// Stereo-fisheye left→right rigid transform (identity otherwise).
    t_lr: Isometry3<f32>,
    #[cfg(feature = "register-times")]
    time_orb_ext: f64,
    #[cfg(feature = "register-times")]
    time_stereo_match: f64,
    img_left: Option<Mat>,
    img_right: Option<Mat>,
}

impl Frame {
    /// Constructor for rectified stereo cameras (single pinhole model).
    #[allow(clippy::too_many_arguments)]
    pub fn from_stereo(
        im_left: &Mat,
        im_right: &Mat,
        timestamp: f64,
        extractor_left: Arc<OrbExtractor>,
        extractor_right: Arc<OrbExtractor>,
        orb_vocabulary: Arc<OrbVocabulary>,
        constants: Arc<FrameConstants>,
        b_fx: f32,
        th_depth: f32,
        camera: Arc<dyn GeometricCamera>,
        prev_frame: Option<Arc<Frame>>,
        imu_calib: Arc<Calib>, // default to default constructor
    ) -> Option<Self> {
        #[cfg(feature = "register-times")]
        let time_start_ext_orb = std::time::Instant::now();

        let (left_res, right_res) = extract_orb_stereo(
            extractor_left.as_ref(),
            extractor_right.as_ref(),
            im_left,
            im_right,
            [0, 0],
            [0, 0],
        );

        #[cfg(feature = "register-times")]
        let time_orb_ext = time_start_ext_orb.elapsed().as_secs_f64() * 1000.0;

        let keys = left_res.keypoints;
        let keys_right = right_res.keypoints;
        let descriptors = left_res.descriptors.unwrap_or_default();
        let descriptors_right = right_res.descriptors.unwrap_or_default();
        let n = keys.len();
        if n == 0 {
            return None;
        }

        // Undistort keypoints (no-op for already-rectified images).
        let keys_un =
            undistort_keypoints(&constants.dist_coef, &camera.to_k(), &constants.k, &keys);

        let b = b_fx / constants.intrinsics.fx;

        #[cfg(feature = "register-times")]
        let time_start_stereo = std::time::Instant::now();

        let (u_right, depth) = compute_stereo_matches(
            &keys,
            &keys_right,
            &descriptors,
            &descriptors_right,
            &left_res.image_pyramid,
            &right_res.image_pyramid,
            &extractor_left.get_scale_factors(),
            &extractor_left.get_inverse_scale_factors(),
            b,
            b_fx,
        );

        #[cfg(feature = "register-times")]
        let time_stereo_match = time_start_stereo.elapsed().as_secs_f64() * 1000.0;

        let (grid, grid_right) =
            assign_features_to_grid(n, None, &keys, None, Some(&keys_un), &constants.bounds);

        Some(Self::assemble(
            FrameConfig {
                orb_vocabulary,
                extractor_left,
                extractor_right,
                timestamp,
                constants,
                b_fx,
                b,
                th_depth,
                camera,
                camera2: None,
                imu_calib,
                prev_frame,
            },
            ExtractedFrame {
                keys,
                keys_right: Some(keys_right),
                keys_un: Some(keys_un),
                descriptors,
                descriptors_right: Some(descriptors_right),
                u_right,
                depth,
                n,
                n_left: None,
                n_right: None,
                mono_left: None,
                mono_right: None,
                left_to_right_match: None,
                right_to_left_match: None,
                stereo_3d_points: None,
                grid,
                grid_right,
                close_mps: 0,
                t_lr: Isometry3::identity(),
                #[cfg(feature = "register-times")]
                time_orb_ext,
                #[cfg(feature = "register-times")]
                time_stereo_match,
                img_left: None,
                img_right: None,
            },
        ))
    }

    /// Constructor for RGB-D cameras.
    #[allow(clippy::too_many_arguments)]
    pub fn from_rgbd(
        im_gray: &Mat,
        im_depth: &Mat,
        timestamp: f64,
        extractor: Arc<OrbExtractor>,
        orb_vocabulary: Arc<OrbVocabulary>,
        constants: Arc<FrameConstants>,
        b_fx: f32,
        th_depth: f32,
        camera: Arc<dyn GeometricCamera>,
        prev_frame: Option<Arc<Frame>>,
        imu_calib: Arc<Calib>, // default to default constructor
    ) -> Option<Self> {
        #[cfg(feature = "register-times")]
        let time_start_ext_orb = std::time::Instant::now();

        let res = extract_orb(extractor.as_ref(), im_gray, [0, 0]);

        #[cfg(feature = "register-times")]
        let time_orb_ext = time_start_ext_orb.elapsed().as_secs_f64() * 1000.0;

        let keys = res.keypoints;
        let descriptors = res.descriptors.unwrap_or_default();
        let n = keys.len();
        if n == 0 {
            return None;
        }

        let keys_un =
            undistort_keypoints(&constants.dist_coef, &camera.to_k(), &constants.k, &keys);

        let b = b_fx / constants.intrinsics.fx;
        let (u_right, depth) = compute_stereo_from_rgbd(&keys, &keys_un, im_depth, b_fx);

        let (grid, grid_right) =
            assign_features_to_grid(n, None, &keys, None, Some(&keys_un), &constants.bounds);

        Some(Self::assemble(
            FrameConfig {
                orb_vocabulary,
                extractor_left: extractor.clone(),
                extractor_right: extractor,
                timestamp,
                constants,
                b_fx,
                b,
                th_depth,
                camera,
                camera2: None,
                imu_calib,
                prev_frame,
            },
            ExtractedFrame {
                keys,
                keys_right: None,
                keys_un: Some(keys_un),
                descriptors,
                descriptors_right: None,
                u_right,
                depth,
                n,
                n_left: None,
                n_right: None,
                mono_left: None,
                mono_right: None,
                left_to_right_match: None,
                right_to_left_match: None,
                stereo_3d_points: None,
                grid,
                grid_right,
                close_mps: 0,
                t_lr: Isometry3::identity(),
                #[cfg(feature = "register-times")]
                time_orb_ext,
                #[cfg(feature = "register-times")]
                time_stereo_match: 0.0,
                img_left: None,
                img_right: None,
            },
        ))
    }

    /// Constructor for monocular cameras.
    #[allow(clippy::too_many_arguments)]
    pub fn from_monocular(
        im_gray: &Mat,
        timestamp: f64,
        extractor: Arc<OrbExtractor>,
        orb_vocabulary: Arc<OrbVocabulary>,
        constants: Arc<FrameConstants>,
        b_fx: f32,
        th_depth: f32,
        camera: Arc<dyn GeometricCamera>,
        prev_frame: Option<Arc<Frame>>,
        imu_calib: Arc<Calib>, // default to default constructor
    ) -> Option<Self> {
        #[cfg(feature = "register-times")]
        let time_start_ext_orb = std::time::Instant::now();

        let res = extract_orb(extractor.as_ref(), im_gray, [0, 1000]);

        #[cfg(feature = "register-times")]
        let time_orb_ext = time_start_ext_orb.elapsed().as_secs_f64() * 1000.0;

        let keys = res.keypoints;
        let descriptors = res.descriptors.unwrap_or_default();
        let n = keys.len();
        if n == 0 {
            return None;
        }

        let keys_un =
            undistort_keypoints(&constants.dist_coef, &camera.to_k(), &constants.k, &keys);

        // No stereo information for monocular.
        let u_right = vec![-1.0f32; n];
        let depth = vec![-1.0f32; n];

        let b = b_fx / constants.intrinsics.fx;
        let (grid, grid_right) =
            assign_features_to_grid(n, None, &keys, None, Some(&keys_un), &constants.bounds);

        Some(Self::assemble(
            FrameConfig {
                orb_vocabulary,
                extractor_left: extractor.clone(),
                extractor_right: extractor,
                timestamp,
                constants,
                b_fx,
                b,
                th_depth,
                camera,
                camera2: None,
                imu_calib,
                prev_frame,
            },
            ExtractedFrame {
                keys,
                keys_right: None,
                keys_un: Some(keys_un),
                descriptors,
                descriptors_right: None,
                u_right,
                depth,
                n,
                n_left: None,
                n_right: None,
                mono_left: None,
                mono_right: None,
                left_to_right_match: None,
                right_to_left_match: None,
                stereo_3d_points: None,
                grid,
                grid_right,
                close_mps: 0,
                t_lr: Isometry3::identity(),
                #[cfg(feature = "register-times")]
                time_orb_ext,
                #[cfg(feature = "register-times")]
                time_stereo_match: 0.0,
                img_left: None,
                img_right: None,
            },
        ))
    }

    /// Constructor for two-camera stereo fisheye (Kannala-Brandt) rigs.
    ///
    /// `t_lr` is the rigid transform mapping right-camera coordinates into the
    /// left camera (Sophus `Tlr`).
    #[allow(clippy::too_many_arguments)]
    pub fn from_stereo_fisheye(
        im_left: &Mat,
        im_right: &Mat,
        timestamp: f64,
        extractor_left: Arc<OrbExtractor>,
        extractor_right: Arc<OrbExtractor>,
        orb_vocabulary: Arc<OrbVocabulary>,
        constants: Arc<FrameConstants>,
        b_fx: f32,
        th_depth: f32,
        camera: Arc<dyn GeometricCamera>,
        camera2: Arc<dyn GeometricCamera>,
        t_lr: Isometry3<f32>,
        prev_frame: Option<Arc<Frame>>,
        imu_calib: Arc<Calib>,
    ) -> Option<Self> {
        let img_left = im_left.clone();
        let img_right = im_right.clone();

        let lap_left = lapping_area(camera.as_ref());
        let lap_right = lapping_area(camera2.as_ref());

        #[cfg(feature = "register-times")]
        let time_start_ext_orb = std::time::Instant::now();

        let (left_res, right_res) = extract_orb_stereo(
            extractor_left.as_ref(),
            extractor_right.as_ref(),
            im_left,
            im_right,
            lap_left,
            lap_right,
        );

        #[cfg(feature = "register-times")]
        let time_orb_ext = time_start_ext_orb.elapsed().as_secs_f64() * 1000.0;

        let keys = left_res.keypoints;
        let keys_right = right_res.keypoints;
        let descriptors_left = left_res.descriptors.unwrap_or_default();
        let descriptors_right = right_res.descriptors.unwrap_or_default();

        let n_left = keys.len();
        let n_right = keys_right.len();
        let n = n_left + n_right;
        if n == 0 {
            return None;
        }
        let mono_left = left_res.mono_index.max(0) as usize;
        let mono_right = right_res.mono_index.max(0) as usize;

        let b = b_fx / constants.intrinsics.fx;
        let t_rl = t_lr.inverse();

        #[cfg(feature = "register-times")]
        let time_start_stereo = std::time::Instant::now();

        let scale_info = ScaleInfo::from_extractor(extractor_left.as_ref());
        let fish = compute_stereo_fisheye_matches(
            &keys,
            &keys_right,
            &descriptors_left,
            &descriptors_right,
            mono_left,
            mono_right,
            n_left,
            n_right,
            camera.as_ref(),
            camera2.as_ref(),
            &t_rl,
            &scale_info.level_sigma2,
        );

        #[cfg(feature = "register-times")]
        let time_stereo_match = time_start_stereo.elapsed().as_secs_f64() * 1000.0;

        // Put all descriptors in the same matrix (left rows then right rows).
        let mut descriptors = Mat::default();
        vconcat2(&descriptors_left, &descriptors_right, &mut descriptors)
            .expect("vconcat descriptors");

        let (grid, grid_right) = assign_features_to_grid(
            n,
            Some(n_left),
            &keys,
            Some(&keys_right),
            None,
            &constants.bounds,
        );

        Some(Self::assemble(
            FrameConfig {
                orb_vocabulary,
                extractor_left,
                extractor_right,
                timestamp,
                constants,
                b_fx,
                b,
                th_depth,
                camera,
                camera2: Some(camera2),
                imu_calib,
                prev_frame,
            },
            ExtractedFrame {
                keys,
                keys_right: Some(keys_right),
                keys_un: None,
                descriptors,
                descriptors_right: Some(descriptors_right),
                u_right: fish.u_right,
                depth: fish.depth,
                n,
                n_left: Some(n_left),
                n_right: Some(n_right),
                mono_left: Some(mono_left),
                mono_right: Some(mono_right),
                left_to_right_match: Some(fish.left_to_right_match),
                right_to_left_match: Some(fish.right_to_left_match),
                stereo_3d_points: Some(fish.stereo_3d_points),
                grid,
                grid_right,
                close_mps: 0,
                t_lr,
                #[cfg(feature = "register-times")]
                time_orb_ext,
                #[cfg(feature = "register-times")]
                time_stereo_match,
                img_left: Some(img_left),
                img_right: Some(img_right),
            },
        ))
    }

    /// Merge the shared config and the sensor-specific feature data into a
    /// fully-initialised [`Frame`], filling in pose/velocity defaults.
    fn assemble(cfg: FrameConfig, ext: ExtractedFrame) -> Self {
        let scale = ScaleInfo::from_extractor(cfg.extractor_left.as_ref());

        // Inherit velocity from the previous frame, like the C++ constructors.
        let (vw, has_velocity) = match &cfg.prev_frame {
            Some(pf) if pf.has_velocity => (pf.vw, true),
            _ => (Vector3::zeros(), false),
        };

        // Stereo-fisheye relative pose (identity for every other sensor).
        let t_rl = ext.t_lr.inverse();
        let r_lr = ext.t_lr.rotation.to_rotation_matrix().into_inner();
        let t_lr_vec = ext.t_lr.translation.vector;

        Frame {
            id: NEXT_ID.fetch_add(1, Ordering::SeqCst),
            orb_vocabulary: cfg.orb_vocabulary,
            extractor_left: cfg.extractor_left,
            extractor_right: cfg.extractor_right,
            timestamp: cfg.timestamp,
            constants: cfg.constants,
            b_fx: cfg.b_fx,
            b: cfg.b,
            th_depth: cfg.th_depth,
            n: ext.n,
            keys: ext.keys,
            keys_right: ext.keys_right,
            keys_un: ext.keys_un,
            map_points: vec![None; ext.n],
            u_right: ext.u_right,
            depth: ext.depth,
            bow_vec: BowVector::default(),
            feat_vec: FeatureVector::default(),
            descriptors: ext.descriptors,
            descriptors_right: ext.descriptors_right,
            outlier: vec![false; ext.n],
            close_mps: ext.close_mps,
            grid: ext.grid,
            pred_bias: Bias::empty(),
            imu_bias: Bias::empty(),
            imu_calib: cfg.imu_calib,
            imu_preintegrated: None,
            last_keyframe: None,
            prev_frame: cfg.prev_frame,
            imu_preintegrated_frame: None,
            reference_kf: None,
            scale_levels: scale.scale_levels,
            scale_factor: scale.scale_factor,
            log_scale_factor: scale.log_scale_factor,
            scale_factors: scale.scale_factors,
            inv_scale_factors: scale.inv_scale_factors,
            level_sigma2: scale.level_sigma2,
            inv_level_sigma2: scale.inv_level_sigma2,
            project_points: HashMap::new(),
            matched_in_image: HashMap::new(),
            name_file: String::new(),
            dataset: 0,
            camera: cfg.camera,
            camera2: cfg.camera2,
            n_left: ext.n_left,
            n_right: ext.n_right,
            mono_left: ext.mono_left,
            mono_right: ext.mono_right,
            left_to_right_match: ext.left_to_right_match,
            right_to_left_match: ext.right_to_left_match,
            stereo_3d_points: ext.stereo_3d_points,
            grid_right: ext.grid_right,
            #[cfg(feature = "register-times")]
            time_orb_ext: ext.time_orb_ext,
            #[cfg(feature = "register-times")]
            time_stereo_match: ext.time_stereo_match,
            cpi: None,
            t_cw: Isometry3::identity(),
            r_wc: Matrix3::identity(),
            o_w: Vector3::zeros(),
            r_cw: Matrix3::identity(),
            t_cw_vec: Vector3::zeros(),
            has_pose: false,
            t_lr: ext.t_lr,
            t_rl,
            r_lr,
            t_lr_vec,
            vw,
            has_velocity,
            is_set: false,
            is_imu_preintegrated: AtomicFlag::new(false),
            img_left: ext.img_left,
            img_right: ext.img_right,
        }
    }

    // --- Bag of Words ---------------------------------------------------

    /// Compute the Bag-of-Words representation of the frame's descriptors.
    pub fn compute_bow(&mut self) {
        if self.bow_vec.is_empty() {
            let descs = descriptors_to_array(&self.descriptors);
            let (bow, feat) = self.orb_vocabulary.transform(&descs, 4);
            self.bow_vec = bow;
            self.feat_vec = feat;
        }
    }

    // --- Pose -----------------------------------------------------------

    /// Set the camera pose `Tcw`. The IMU pose is not modified.
    pub fn set_pose(&mut self, t_cw: Isometry3<f32>) {
        self.t_cw = t_cw;
        self.update_pose_matrices();
        self.is_set = true;
        self.has_pose = true;
    }

    /// Set IMU pose (`Rwb`, `twb`) and velocity, implicitly setting the camera pose.
    pub fn set_imu_pose_velocity(
        &mut self,
        r_wb: Matrix3<f32>,
        t_wb: Vector3<f32>,
        v_wb: Vector3<f32>,
    ) {
        self.vw = v_wb;
        self.has_velocity = true;

        let rotation = nalgebra::UnitQuaternion::from_rotation_matrix(
            &nalgebra::Rotation3::from_matrix_unchecked(r_wb),
        );
        let t_wb = Isometry3::from_parts(t_wb.into(), rotation);
        let t_bw = t_wb.inverse();
        self.t_cw = self.imu_calib.tcb * t_bw;
        self.update_pose_matrices();
        self.is_set = true;
        self.has_pose = true;
    }

    /// Recompute rotation, translation and camera-center matrices from `Tcw`.
    fn update_pose_matrices(&mut self) {
        let t_wc = self.t_cw.inverse();
        self.r_wc = t_wc.rotation.to_rotation_matrix().into_inner();
        self.o_w = t_wc.translation.vector;
        self.r_cw = self.t_cw.rotation.to_rotation_matrix().into_inner();
        self.t_cw_vec = self.t_cw.translation.vector;
    }

    pub fn get_pose(&self) -> Isometry3<f32> {
        self.t_cw
    }
    pub fn has_pose(&self) -> bool {
        self.has_pose
    }
    pub fn is_set(&self) -> bool {
        self.is_set
    }
    /// Camera center in world coordinates (`mOw`).
    pub fn get_camera_center(&self) -> Vector3<f32> {
        self.o_w
    }
    /// Inverse of the rotation (`mRwc`).
    pub fn get_rotation_inverse(&self) -> Matrix3<f32> {
        self.r_wc
    }
    pub fn get_rwc(&self) -> Matrix3<f32> {
        self.r_wc
    }
    pub fn get_ow(&self) -> Vector3<f32> {
        self.o_w
    }

    // --- Velocity -------------------------------------------------------

    pub fn set_velocity(&mut self, v_wb: Vector3<f32>) {
        self.vw = v_wb;
        self.has_velocity = true;
    }
    pub fn get_velocity(&self) -> Vector3<f32> {
        self.vw
    }
    pub fn has_velocity(&self) -> bool {
        self.has_velocity
    }

    // --- IMU ------------------------------------------------------------

    /// IMU position in world coordinates.
    pub fn get_imu_position(&self) -> Vector3<f32> {
        self.r_wc * self.imu_calib.tcb.translation.vector + self.o_w
    }
    /// IMU rotation in world coordinates.
    pub fn get_imu_rotation(&self) -> Matrix3<f32> {
        self.r_wc
            * self
                .imu_calib
                .tcb
                .rotation
                .to_rotation_matrix()
                .into_inner()
    }
    /// IMU pose (`Twb`-style `Tcw⁻¹ · Tcb`).
    pub fn get_imu_pose(&self) -> Isometry3<f32> {
        self.t_cw.inverse() * self.imu_calib.tcb
    }

    pub fn set_new_bias(&mut self, b: Bias) {
        self.imu_bias = b;
        if let Some(preint) = self.imu_preintegrated.as_ref() {
            // `Preintegrated` mutation through a shared `Arc` is part of the
            // wider interior-mutability decision; this is a no-op stub until then.
            let _ = preint;
        }
    }

    pub fn imu_is_preintegrated(&self) -> bool {
        self.is_imu_preintegrated.get()
    }
    /// Mark the frame's IMU measurements as preintegrated.
    ///
    /// Takes `&self`: the [`AtomicFlag`] gives interior mutability so
    /// LocalMapping can publish this through a shared `Arc<Frame>` without a
    /// lock or exclusive borrow.
    pub fn set_integrated(&self) {
        self.is_imu_preintegrated.set(true);
    }

    // --- Relative stereo pose ------------------------------------------

    pub fn get_relative_pose_trl(&self) -> Isometry3<f32> {
        self.t_rl
    }
    pub fn get_relative_pose_tlr(&self) -> Isometry3<f32> {
        self.t_lr
    }
    pub fn get_relative_pose_tlr_rotation(&self) -> Matrix3<f32> {
        self.r_lr
    }
    pub fn get_relative_pose_tlr_translation(&self) -> Vector3<f32> {
        self.t_lr_vec
    }

    // --- Projection / geometry -----------------------------------------

    /// Transform a world point into this frame's camera coordinates.
    pub fn in_ref_coordinates(&self, p_cw: Vector3<f32>) -> Vector3<f32> {
        self.r_cw * p_cw + self.t_cw_vec
    }

    /// Check whether a [`MapPoint`] falls in the camera frustum and, if so,
    /// return the projection data for the left and/or right camera.
    /// Unlike the C++ version this does not mutate the `MapPoint`.
    pub fn is_in_frustum(&self, mp: &MapPoint, viewing_cos_limit: f32) -> FrustumResult {
        if self.n_left.is_none() {
            // Pinhole / rectified-stereo path.
            let p = mp.get_world_pos();
            let pc = self.r_cw * p + self.t_cw_vec;
            let pc_dist = pc.norm();
            let pcz = pc.z;
            if pcz < 0.0 {
                return FrustumResult::default();
            }
            let invz = 1.0 / pcz;

            let uv = self.camera.project_n(&Point3::from(pc));
            let bounds = &self.constants.bounds;
            if uv.x < bounds.min_x || uv.x > bounds.max_x {
                return FrustumResult::default();
            }
            if uv.y < bounds.min_y || uv.y > bounds.max_y {
                return FrustumResult::default();
            }

            let max_distance = mp.get_max_distance_invariance();
            let min_distance = mp.get_min_distance_invariance();
            let po = p - self.o_w;
            let dist = po.norm();
            if dist < min_distance || dist > max_distance {
                return FrustumResult::default();
            }

            let pn = mp.get_normal();
            let view_cos = po.dot(&pn) / dist;
            if view_cos < viewing_cos_limit {
                return FrustumResult::default();
            }

            let predicted_level = mp.predict_scale(dist, self) as i32;
            FrustumResult {
                left: Some(TrackInfo {
                    proj_x: uv.x,
                    proj_y: uv.y,
                    proj_xr: uv.x - self.b_fx * invz,
                    depth: pc_dist,
                    scale_level: predicted_level,
                    view_cos,
                }),
                right: None,
            }
        } else {
            // Stereo-fisheye path: check both cameras.
            FrustumResult {
                left: self.is_in_frustum_checks(mp, viewing_cos_limit, false),
                right: self.is_in_frustum_checks(mp, viewing_cos_limit, true),
            }
        }
    }

    /// Frustum check against a single camera of a stereo-fisheye rig.
    fn is_in_frustum_checks(
        &self,
        mp: &MapPoint,
        viewing_cos_limit: f32,
        right: bool,
    ) -> Option<TrackInfo> {
        let p = mp.get_world_pos();

        let (rotation, translation, twc, cam): (
            Matrix3<f32>,
            Vector3<f32>,
            Vector3<f32>,
            &dyn GeometricCamera,
        ) = if right {
            let rrl = self.t_rl.rotation.to_rotation_matrix().into_inner();
            let trl = self.t_rl.translation.vector;
            (
                rrl * self.r_cw,
                rrl * self.t_cw_vec + trl,
                self.r_wc * self.t_lr_vec + self.o_w,
                self.camera2
                    .as_deref()
                    .expect("camera2 for fisheye right check"),
            )
        } else {
            (self.r_cw, self.t_cw_vec, self.o_w, self.camera.as_ref())
        };

        let pc = rotation * p + translation;
        let pc_dist = pc.norm();
        if pc.z < 0.0 {
            return None;
        }

        let uv = cam.project_n(&Point3::from(pc));
        let bounds = &self.constants.bounds;
        if uv.x < bounds.min_x || uv.x > bounds.max_x {
            return None;
        }
        if uv.y < bounds.min_y || uv.y > bounds.max_y {
            return None;
        }

        let max_distance = mp.get_max_distance_invariance();
        let min_distance = mp.get_min_distance_invariance();
        let po = p - twc;
        let dist = po.norm();
        if dist < min_distance || dist > max_distance {
            return None;
        }

        let pn = mp.get_normal();
        let view_cos = po.dot(&pn) / dist;
        if view_cos < viewing_cos_limit {
            return None;
        }

        let predicted_level = mp.predict_scale(dist, self) as i32;
        Some(TrackInfo {
            proj_x: uv.x,
            proj_y: uv.y,
            proj_xr: uv.x,
            depth: pc_dist,
            scale_level: predicted_level,
            view_cos,
        })
    }

    /// Project a [`MapPoint`] applying OpenCV radial-tangential distortion.
    /// Returns the distorted image point, or `None` if behind / outside.
    pub fn project_point_distort(&self, mp: &MapPoint) -> Option<Point2f> {
        let intr = &self.constants.intrinsics;
        let p = mp.get_world_pos();
        let pc = self.r_cw * p + self.t_cw_vec;
        if pc.z < 0.0 {
            return None;
        }

        let invz = 1.0 / pc.z;
        let u = intr.fx * pc.x * invz + intr.cx;
        let v = intr.fy * pc.y * invz + intr.cy;

        let bounds = &self.constants.bounds;
        if u < bounds.min_x || u > bounds.max_x {
            return None;
        }
        if v < bounds.min_y || v > bounds.max_y {
            return None;
        }

        let x = (u - intr.cx) * intr.invfx;
        let y = (v - intr.cy) * intr.invfy;
        let r2 = x * x + y * y;

        let dist = &self.constants.dist_coef;
        let k1 = *dist.at::<f32>(0).expect("k1");
        let k2 = *dist.at::<f32>(1).expect("k2");
        let p1 = *dist.at::<f32>(2).expect("p1");
        let p2 = *dist.at::<f32>(3).expect("p2");
        let k3 = if dist.total() == 5 {
            *dist.at::<f32>(4).expect("k3")
        } else {
            0.0
        };

        let radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
        let x_distort = x * radial + (2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x));
        let y_distort = y * radial + (p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y);

        Some(Point2f::new(
            x_distort * intr.fx + intr.cx,
            y_distort * intr.fy + intr.cy,
        ))
    }

    /// Backproject keypoint `i` into 3D world coordinates, if it has valid depth.
    pub fn unproject_stereo(&self, i: usize) -> Option<Vector3<f32>> {
        let z = self.depth[i];
        if z > 0.0 {
            let intr = &self.constants.intrinsics;
            let kp = &self.keys_un.as_ref().expect("keys_un for unproject")[i];
            let u = kp.pt().x;
            let v = kp.pt().y;
            let x = (u - intr.cx) * z * intr.invfx;
            let y = (v - intr.cy) * z * intr.invfy;
            Some(self.r_wc * Vector3::new(x, y, z) + self.o_w)
        } else {
            None
        }
    }

    /// Backproject a stereo-fisheye match (already triangulated in the left frame).
    pub fn unproject_stereo_fisheye(&self, i: usize) -> Vector3<f32> {
        let pts = self
            .stereo_3d_points
            .as_ref()
            .expect("stereo_3d_points for fisheye unproject");
        self.r_wc * pts[i] + self.o_w
    }

    // --- Feature grid ---------------------------------------------------

    /// Indices of features whose cells overlap a circle of radius `r` around
    /// `(x, y)`, optionally restricted to a scale-level band and to the right
    /// image (stereo fisheye).
    pub fn get_features_in_area(
        &self,
        x: f32,
        y: f32,
        r: f32,
        min_level: i32,
        max_level: i32,
        right: bool,
    ) -> Vec<usize> {
        let mut indices = Vec::with_capacity(self.n);
        let bounds = &self.constants.bounds;

        let min_cell_x = (((x - bounds.min_x - r) * bounds.grid_w_inv).floor() as i32).max(0);
        if min_cell_x >= FRAME_GRID_COLS as i32 {
            return indices;
        }
        let max_cell_x = (((x - bounds.min_x + r) * bounds.grid_w_inv).ceil() as i32)
            .min(FRAME_GRID_COLS as i32 - 1);
        if max_cell_x < 0 {
            return indices;
        }
        let min_cell_y = (((y - bounds.min_y - r) * bounds.grid_h_inv).floor() as i32).max(0);
        if min_cell_y >= FRAME_GRID_ROWS as i32 {
            return indices;
        }
        let max_cell_y = (((y - bounds.min_y + r) * bounds.grid_h_inv).ceil() as i32)
            .min(FRAME_GRID_ROWS as i32 - 1);
        if max_cell_y < 0 {
            return indices;
        }

        let check_levels = (min_level > 0) || (max_level >= 0);
        let grid = if right { &self.grid_right } else { &self.grid };

        for ix in min_cell_x..=max_cell_x {
            for iy in min_cell_y..=max_cell_y {
                let cell = &grid[grid_index(ix as usize, iy as usize)];
                for &j in cell {
                    let kp = self.keypoint_for_area(j, right);
                    if check_levels {
                        let octave = kp.octave();
                        if octave < min_level {
                            continue;
                        }
                        if max_level >= 0 && octave > max_level {
                            continue;
                        }
                    }
                    let distx = kp.pt().x - x;
                    let disty = kp.pt().y - y;
                    if distx.abs() < r && disty.abs() < r {
                        indices.push(j);
                    }
                }
            }
        }
        indices
    }

    /// Keypoint used by [`Self::get_features_in_area`]: undistorted for the
    /// pinhole/rectified case, raw left/right keys for stereo fisheye.
    fn keypoint_for_area(&self, idx: usize, right: bool) -> &KeyPoint {
        match self.n_left {
            None => &self.keys_un.as_ref().expect("keys_un")[idx],
            Some(_) if !right => &self.keys[idx],
            Some(_) => &self.keys_right.as_ref().expect("keys_right")[idx],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CameraIntrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub invfx: f32,
    pub invfy: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct ImageBounds {
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
    pub grid_w_inv: f32,
    pub grid_h_inv: f32,
}

/// Per-camera precomputed constants shared by every Frame produced from
/// the same calibration.
#[derive(Debug)]
pub struct FrameConstants {
    /// 3×3 intrinsics matrix as an OpenCV `Mat` (`CV_32F`).
    pub k: Mat,
    /// 3×3 intrinsics matrix as a nalgebra type.
    pub k_matrix: Matrix3<f32>,
    /// OpenCV distortion coefficients (may be empty / all-zero for fisheye).
    pub dist_coef: Mat,
    /// Scalar intrinsics derived from `k`.
    pub intrinsics: CameraIntrinsics,
    /// Undistorted image bounds and grid-cell inverse sizes.
    pub bounds: ImageBounds,
}

#[derive(Debug)]
pub enum FrameConstantsError {
    /// `k` could not be converted to a `Matrix3<f32>`.
    InvalidK(opencv::Error),
    /// Undistortion of the image corners failed.
    Undistort(opencv::Error),
    /// Image dimensions must be strictly positive.
    InvalidImageSize { cols: i32, rows: i32 },
}
impl std::fmt::Display for FrameConstantsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidK(e) => write!(f, "invalid camera matrix K: {e}"),
            Self::Undistort(e) => write!(f, "failed to undistort image corners: {e}"),
            Self::InvalidImageSize { cols, rows } => {
                write!(f, "invalid image size: {cols}x{rows}")
            }
        }
    }
}
impl std::error::Error for FrameConstantsError {}

impl FrameConstants {
    /// Build the per-camera constants from the calibration and the image size.
    ///
    /// This performs all the work the original C++ code did once inside the
    /// `if(mbInitialComputations)` block of every `Frame` constructor.
    ///
    /// `k` is the 3×3 intrinsics matrix (`CV_32F`). `dist_coef` is the OpenCV
    /// distortion-coefficient vector; pass an empty `Mat` (or one whose first
    /// element is zero) for cameras that don't use OpenCV-style distortion
    /// (e.g. fisheye/Kannala–Brandt), in which case the raw image rectangle is
    /// used as the bounds
    pub fn new(
        k: Mat,
        dist_coef: Mat,
        image_cols: i32,
        image_rows: i32,
    ) -> Result<Self, FrameConstantsError> {
        if image_cols <= 0 || image_rows <= 0 {
            return Err(FrameConstantsError::InvalidImageSize {
                cols: image_cols,
                rows: image_rows,
            });
        }

        let k_matrix = mat_to_matrix3f(&k).map_err(FrameConstantsError::InvalidK)?;
        let fx = k_matrix[(0, 0)];
        let fy = k_matrix[(1, 1)];
        let cx = k_matrix[(0, 2)];
        let cy = k_matrix[(1, 2)];
        let intrinsics = CameraIntrinsics {
            fx,
            fy,
            cx,
            cy,
            invfx: 1.0 / fx,
            invfy: 1.0 / fy,
        };

        let (min_x, max_x, min_y, max_y) =
            compute_image_bounds(&k, &dist_coef, image_cols, image_rows)?;
        let bounds = ImageBounds {
            min_x,
            max_x,
            min_y,
            max_y,
            grid_w_inv: FRAME_GRID_COLS as f32 / (max_x - min_x),
            grid_h_inv: FRAME_GRID_ROWS as f32 / (max_y - min_y),
        };

        Ok(Self {
            k,
            k_matrix,
            dist_coef,
            intrinsics,
            bounds,
        })
    }
}

fn compute_image_bounds(
    k: &Mat,
    dist_coef: &Mat,
    image_cols: i32,
    image_rows: i32,
) -> Result<(f32, f32, f32, f32), FrameConstantsError> {
    if !has_distortion(dist_coef) {
        return Ok((0.0, image_cols as f32, 0.0, image_rows as f32));
    }

    // 4×1 of CV_32FC2 — the format `undistortPoints` expects.
    let cols = image_cols as f32;
    let rows = image_rows as f32;
    let corners = Mat::from_slice_2d(&[
        [Point2f::new(0.0, 0.0)],
        [Point2f::new(cols, 0.0)],
        [Point2f::new(0.0, rows)],
        [Point2f::new(cols, rows)],
    ])
    .map_err(FrameConstantsError::Undistort)?;

    let mut undistorted = Mat::default();
    undistort_points(&corners, &mut undistorted, k, dist_coef, &Mat::default(), k)
        .map_err(FrameConstantsError::Undistort)?;

    let p = |i: i32| -> Result<Point2f, FrameConstantsError> {
        undistorted
            .at::<Point2f>(i)
            .copied()
            .map_err(FrameConstantsError::Undistort)
    };
    let (p0, p1, p2, p3) = (p(0)?, p(1)?, p(2)?, p(3)?);

    Ok((
        p0.x.min(p2.x),
        p1.x.max(p3.x),
        p0.y.min(p1.y),
        p2.y.max(p3.y),
    ))
}

/// Whether the OpenCV distortion vector actually carries distortion.
fn has_distortion(dist_coef: &Mat) -> bool {
    !dist_coef.empty() && dist_coef.at::<f32>(0).map(|v| *v != 0.0).unwrap_or(false)
}

/// The Kannala-Brandt lapping (stereo overlap) area `[x0, x1]` for a camera.
fn lapping_area(camera: &dyn GeometricCamera) -> [i32; 2] {
    let kb = camera
        .as_any()
        .downcast_ref::<KannalaBrandt8>()
        .expect("fisheye frame requires a KannalaBrandt8 camera");
    [kb.lapping_area[0] as i32, kb.lapping_area[1] as i32]
}

/// Run ORB extraction on a single image.
fn extract_orb(extractor: &OrbExtractor, im: &Mat, lapping: [i32; 2]) -> OrbExtractResult {
    extract_orb_result(extractor, im, lapping).expect("ORB extraction failed")
}

fn extract_orb_result(
    extractor: &OrbExtractor,
    im: &Mat,
    lapping: [i32; 2],
) -> Result<OrbExtractResult, ExtractionError> {
    extractor.compute(im, &Mat::default(), lapping)
}

/// Extract ORB on the left and right images in parallel.
fn extract_orb_stereo(
    extractor_left: &OrbExtractor,
    extractor_right: &OrbExtractor,
    im_left: &Mat,
    im_right: &Mat,
    lap_left: [i32; 2],
    lap_right: [i32; 2],
) -> (OrbExtractResult, OrbExtractResult) {
    let (left, right) = std::thread::scope(|s| {
        let left_handle = s.spawn(|| extract_orb_result(extractor_left, im_left, lap_left));
        let right = extract_orb_result(extractor_right, im_right, lap_right);
        let left = left_handle.join().expect("left ORB thread panicked");
        (left, right)
    });
    (
        left.expect("left ORB extraction failed"),
        right.expect("right ORB extraction failed"),
    )
}

/// Grid cell of a keypoint, or `None` if its undistorted position falls
/// outside the grid.
pub fn pos_in_grid(kp: &KeyPoint, bounds: &ImageBounds) -> Option<(usize, usize)> {
    let pt = kp.pt();
    let pos_x = ((pt.x - bounds.min_x) * bounds.grid_w_inv).round() as i32;
    let pos_y = ((pt.y - bounds.min_y) * bounds.grid_h_inv).round() as i32;

    // Keypoints' coordinates are undistorted, which could push them outside
    // the image rectangle.
    if pos_x < 0 || pos_x >= FRAME_GRID_COLS as i32 || pos_y < 0 || pos_y >= FRAME_GRID_ROWS as i32
    {
        return None;
    }
    Some((pos_x as usize, pos_y as usize))
}

#[inline]
pub fn grid_index(col: usize, row: usize) -> usize {
    col * FRAME_GRID_ROWS + row
}

fn assign_features_to_grid(
    n: usize,
    n_left: Option<usize>,
    keys: &[KeyPoint],
    keys_right: Option<&[KeyPoint]>,
    keys_un: Option<&[KeyPoint]>,
    bounds: &ImageBounds,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let n_cells = FRAME_GRID_COLS * FRAME_GRID_ROWS;
    let reserve = (0.5 * n as f32 / n_cells as f32) as usize;

    let mut grid: Vec<Vec<usize>> = (0..n_cells).map(|_| Vec::with_capacity(reserve)).collect();
    let mut grid_right: Vec<Vec<usize>> = if n_left.is_some() {
        (0..n_cells).map(|_| Vec::with_capacity(reserve)).collect()
    } else {
        Vec::new()
    };

    // Index-based: the keypoint source (left / right / undistorted) depends on `i`.
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let kp = match n_left {
            None => &keys_un.expect("keys_un required for non-fisheye frames")[i],
            Some(nl) if i < nl => &keys[i],
            Some(nl) => &keys_right.expect("keys_right required for stereo-fisheye frames")[i - nl],
        };

        let Some((gx, gy)) = pos_in_grid(kp, bounds) else {
            continue;
        };
        let idx = grid_index(gx, gy);
        match n_left {
            None => grid[idx].push(i),
            Some(nl) if i < nl => grid[idx].push(i),
            Some(nl) => grid_right[idx].push(i - nl),
        }
    }

    (grid, grid_right)
}

/// Undistort the raw keypoints using the OpenCV distortion model. Returns a
/// copy of `keys` unchanged when there is no distortion (rectified / fisheye).
fn undistort_keypoints(
    dist_coef: &Mat,
    camera_k: &Mat,
    k: &Mat,
    keys: &[KeyPoint],
) -> Vec<KeyPoint> {
    if !has_distortion(dist_coef) {
        return keys.to_vec();
    }

    let n = keys.len() as i32;

    // Fill matrix with points.
    let mut mat =
        Mat::new_rows_cols_with_default(n, 2, CV_32F, Scalar::default()).expect("create mat");
    for (i, kp) in keys.iter().enumerate() {
        *mat.at_2d_mut::<f32>(i as i32, 0).unwrap() = kp.pt().x;
        *mat.at_2d_mut::<f32>(i as i32, 1).unwrap() = kp.pt().y;
    }

    // Reinterpret as Nx1 CV_32FC2, undistort, then back to Nx2 CV_32F.
    let mat_2c = mat.reshape(2, 0).expect("reshape to 2 channels");
    let mut undistorted = Mat::default();
    undistort_points(
        &mat_2c,
        &mut undistorted,
        camera_k,
        dist_coef,
        &Mat::default(),
        k,
    )
    .expect("undistortPoints");
    let undistorted = undistorted.reshape(1, 0).expect("reshape to 1 channel");

    // Write the undistorted coordinates back into a new keypoint vector.
    let mut keys_un = Vec::with_capacity(keys.len());
    for (i, kp) in keys.iter().enumerate() {
        let mut kp = kp.clone();
        let x = *undistorted.at_2d::<f32>(i as i32, 0).unwrap();
        let y = *undistorted.at_2d::<f32>(i as i32, 1).unwrap();
        kp.set_pt(Point2f::new(x, y));
        keys_un.push(kp);
    }
    keys_un
}

/// Search a match for each left keypoint in the right (rectified) image,
/// returning `(u_right, depth)` per left keypoint (`-1` when unmatched).
#[allow(clippy::too_many_arguments)]
fn compute_stereo_matches(
    keys: &[KeyPoint],
    keys_right: &[KeyPoint],
    descriptors: &Mat,
    descriptors_right: &Mat,
    left_pyramid: &[Mat],
    right_pyramid: &[Mat],
    scale_factors: &[f32],
    inv_scale_factors: &[f32],
    mb: f32,
    b_fx: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = keys.len();
    let mut u_right = vec![-1.0f32; n];
    let mut depth = vec![-1.0f32; n];

    let th_orb_dist = (TH_HIGH + TH_LOW) / 2;
    let n_rows = left_pyramid[0].rows();
    if n_rows <= 0 {
        return (u_right, depth);
    }

    // Assign right keypoints to a row table.
    let mut row_indices: Vec<Vec<usize>> = vec![Vec::with_capacity(200); n_rows as usize];
    for (i_r, kp) in keys_right.iter().enumerate() {
        let kp_y = kp.pt().y;
        let r = 2.0 * scale_factors[kp.octave() as usize];
        let max_r = (kp_y + r).ceil() as i32;
        let min_r = (kp_y - r).floor() as i32;
        for yi in min_r.max(0)..=max_r.min(n_rows - 1) {
            row_indices[yi as usize].push(i_r);
        }
    }

    // Search limits.
    let min_z = mb;
    let min_d = 0.0;
    let max_d = b_fx / min_z;

    // (best_dist, left_index) for the median-based outlier rejection pass.
    let mut dist_idx: Vec<(i32, usize)> = Vec::with_capacity(n);

    for i_l in 0..n {
        let kp_l = &keys[i_l];
        let level_l = kp_l.octave();
        let v_l = kp_l.pt().y;
        let u_l = kp_l.pt().x;

        let candidates = &row_indices[v_l.round().clamp(0.0, (n_rows - 1) as f32) as usize];
        if candidates.is_empty() {
            continue;
        }

        let min_u = u_l - max_d;
        let max_u = u_l - min_d;
        if max_u < 0.0 {
            continue;
        }

        let mut best_dist = TH_HIGH;
        let mut best_idx_r = 0usize;
        let d_l = descriptors.row(i_l as i32).expect("left descriptor row");

        for &i_r in candidates {
            let kp_r = &keys_right[i_r];
            if kp_r.octave() < level_l - 1 || kp_r.octave() > level_l + 1 {
                continue;
            }
            let u_r = kp_r.pt().x;
            if u_r >= min_u && u_r <= max_u {
                let d_r = descriptors_right
                    .row(i_r as i32)
                    .expect("right descriptor row");
                let dist = descriptor_distance(&d_l, &d_r);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx_r = i_r;
                }
            }
        }

        if best_dist >= th_orb_dist {
            continue;
        }

        // Subpixel match by sliding-window correlation at the keypoint's scale.
        let u_r0 = keys_right[best_idx_r].pt().x;
        let scale = inv_scale_factors[kp_l.octave() as usize];
        let scaled_u_l = (kp_l.pt().x * scale).round();
        let scaled_v_l = (kp_l.pt().y * scale).round();
        let scaled_u_r0 = (u_r0 * scale).round();

        let w = 5;
        let level = kp_l.octave() as usize;
        let Some(il) = roi_window(&left_pyramid[level], scaled_u_l, scaled_v_l, w) else {
            continue;
        };

        let big_l = 5;
        let mut dists = vec![0.0f32; (2 * big_l + 1) as usize];
        let ini_u = scaled_u_r0 + big_l as f32 - w as f32;
        let end_u = scaled_u_r0 + big_l as f32 + w as f32 + 1.0;
        if ini_u < 0.0 || end_u >= right_pyramid[level].cols() as f32 {
            continue;
        }

        let mut best_corr = f32::MAX;
        let mut best_inc_r = 0i32;
        for inc_r in -big_l..=big_l {
            let Some(ir) = roi_window(
                &right_pyramid[level],
                scaled_u_r0 + inc_r as f32,
                scaled_v_l,
                w,
            ) else {
                continue;
            };
            let dist = norm2(&il, &ir, NORM_L1, &Mat::default()).expect("norm L1") as f32;
            if dist < best_corr {
                best_corr = dist;
                best_inc_r = inc_r;
            }
            dists[(big_l + inc_r) as usize] = dist;
        }

        if best_inc_r == -big_l || best_inc_r == big_l {
            continue;
        }

        // Sub-pixel match (parabola fitting).
        let dist1 = dists[(big_l + best_inc_r - 1) as usize];
        let dist2 = dists[(big_l + best_inc_r) as usize];
        let dist3 = dists[(big_l + best_inc_r + 1) as usize];
        let denom = 2.0 * (dist1 + dist3 - 2.0 * dist2);
        if denom == 0.0 {
            continue;
        }
        let delta_r = (dist1 - dist3) / denom;
        if !(-1.0..=1.0).contains(&delta_r) {
            continue;
        }

        // Re-scaled coordinate.
        let mut best_u_r =
            scale_factors[kp_l.octave() as usize] * (scaled_u_r0 + best_inc_r as f32 + delta_r);
        let mut disparity = u_l - best_u_r;
        if disparity >= min_d && disparity < max_d {
            if disparity <= 0.0 {
                disparity = 0.01;
                best_u_r = u_l - 0.01;
            }
            depth[i_l] = b_fx / disparity;
            u_right[i_l] = best_u_r;
            dist_idx.push((best_corr as i32, i_l));
        }
    }

    // Reject matches whose correlation is far above the median.
    if dist_idx.is_empty() {
        return (u_right, depth);
    }
    dist_idx.sort_unstable();
    let median = dist_idx[dist_idx.len() / 2].0 as f32;
    let th_dist = 1.5 * 1.4 * median;
    for &(dist, i_l) in dist_idx.iter().rev() {
        if (dist as f32) < th_dist {
            break;
        }
        u_right[i_l] = -1.0;
        depth[i_l] = -1.0;
    }

    (u_right, depth)
}

/// Extract a `(2w+1)×(2w+1)` window centred on `(cx, cy)`, or `None` if it
/// would fall outside the image.
fn roi_window(img: &Mat, cx: f32, cy: f32, w: i32) -> Option<opencv::boxed_ref::BoxedRef<'_, Mat>> {
    let x = cx as i32 - w;
    let y = cy as i32 - w;
    let side = 2 * w + 1;
    if x < 0 || y < 0 || x + side > img.cols() || y + side > img.rows() {
        return None;
    }
    Mat::roi(img, Rect::new(x, y, side, side)).ok()
}

/// Associate a right coordinate and depth to each keypoint from a depth map.
fn compute_stereo_from_rgbd(
    keys: &[KeyPoint],
    keys_un: &[KeyPoint],
    im_depth: &Mat,
    b_fx: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = keys.len();
    let mut u_right = vec![-1.0f32; n];
    let mut depth = vec![-1.0f32; n];

    for (i, (kp, kp_u)) in keys.iter().zip(keys_un).enumerate() {
        let v = kp.pt().y as i32;
        let u = kp.pt().x as i32;
        let d = *im_depth.at_2d::<f32>(v, u).expect("depth lookup");
        if d > 0.0 {
            depth[i] = d;
            u_right[i] = kp_u.pt().x - b_fx / d;
        }
    }
    (u_right, depth)
}

/// Convert an `N×32` `CV_8U` descriptor matrix into per-row [`Descriptor`]s
/// for the vocabulary transform.
fn descriptors_to_array(descriptors: &Mat) -> Vec<Descriptor> {
    let rows = descriptors.rows();
    let mut out = Vec::with_capacity(rows as usize);
    for i in 0..rows {
        let row = descriptors.row(i).expect("descriptor row");
        let bytes = row.data_bytes().expect("descriptor bytes");
        let mut d: Descriptor = [0u8; DESC_LEN];
        d.copy_from_slice(&bytes[..DESC_LEN]);
        out.push(d);
    }
    out
}

/// Output of [`compute_stereo_fisheye_matches`].
struct FisheyeMatches {
    left_to_right_match: Vec<usize>,
    right_to_left_match: Vec<usize>,
    u_right: Vec<f32>,
    depth: Vec<f32>,
    stereo_3d_points: Vec<Vector3<f32>>,
}

/// Brute-force match left↔right keypoints in the lapping area of a fisheye
/// rig and triangulate the good matches in the left-camera frame.
#[allow(clippy::too_many_arguments)]
fn compute_stereo_fisheye_matches(
    keys: &[KeyPoint],
    keys_right: &[KeyPoint],
    descriptors_left: &Mat,
    descriptors_right: &Mat,
    mono_left: usize,
    mono_right: usize,
    n_left: usize,
    n_right: usize,
    camera: &dyn GeometricCamera,
    camera2: &dyn GeometricCamera,
    t_rl: &Isometry3<f32>,
    level_sigma2: &[f32],
) -> FisheyeMatches {
    let mut left_to_right_match = vec![usize::MAX; n_left];
    let mut right_to_left_match = vec![usize::MAX; n_right];
    let u_right = vec![-1.0f32; n_left];
    let mut depth = vec![-1.0f32; n_left];
    let mut stereo_3d_points = vec![Vector3::zeros(); n_left];

    // Only the lapping-area descriptors participate in stereo matching.
    let stereo_desc_left = descriptors_left
        .row_range(&opencv::core::Range::new(mono_left as i32, descriptors_left.rows()).unwrap())
        .expect("left stereo descriptors");
    let stereo_desc_right = descriptors_right
        .row_range(&opencv::core::Range::new(mono_right as i32, descriptors_right.rows()).unwrap())
        .expect("right stereo descriptors");

    let matcher = BFMatcher::new(NORM_HAMMING, false).expect("BFMatcher");
    let mut matches: Vector<Vector<DMatch>> = Vector::new();
    matcher
        .knn_train_match_def(&stereo_desc_left, &stereo_desc_right, &mut matches, 2)
        .expect("knn match");

    let identity = Isometry3::identity();
    for pair in matches.iter() {
        if pair.len() < 2 {
            continue;
        }
        let m0 = pair.get(0).unwrap();
        let m1 = pair.get(1).unwrap();
        // Lowe's ratio test.
        if m0.distance >= m1.distance * 0.7 {
            continue;
        }

        let left_idx = m0.query_idx as usize + mono_left;
        let right_idx = m0.train_idx as usize + mono_right;
        let sigma1 = level_sigma2[keys[left_idx].octave() as usize];
        let sigma2 = level_sigma2[keys_right[right_idx].octave() as usize];

        // Triangulate with the left camera as the world reference.
        if let Some(p3d) = camera.match_and_triangulate(
            &keys[left_idx],
            &keys_right[right_idx],
            camera2,
            &identity,
            t_rl,
            sigma1,
            sigma2,
        ) {
            let z = p3d.z;
            if z > 0.0001 {
                left_to_right_match[left_idx] = right_idx;
                right_to_left_match[right_idx] = left_idx;
                stereo_3d_points[left_idx] = p3d;
                depth[left_idx] = z;
            }
        }
    }

    FisheyeMatches {
        left_to_right_match,
        right_to_left_match,
        u_right,
        depth,
        stereo_3d_points,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::excessive_precision)]
    use std::sync::Arc;

    use nalgebra::Vector3;
    use opencv::core::KeyPointTrait;

    use super::*;
    use crate::map::Map;
    use crate::map_point::MapPoint;
    use crate::test_helpers::*;

    #[test]
    fn image_bounds_and_intrinsics() {
        let c = constants();
        let b = &c.bounds;
        assert!(approx(b.min_x, -135.795639, 1e-2), "min_x={}", b.min_x);
        assert!(approx(b.max_x, 895.507324, 1e-2), "max_x={}", b.max_x);
        assert!(approx(b.min_y, -92.875015, 1e-2), "min_y={}", b.min_y);
        assert!(approx(b.max_y, 565.553101, 1e-2), "max_y={}", b.max_y);
        assert!(approx(b.grid_w_inv, 0.062057417, 1e-6));
        assert!(approx(b.grid_h_inv, 0.072900899, 1e-6));

        let i = &c.intrinsics;
        assert!(approx(i.fx, FX, 1e-2) && approx(i.fy, FY, 1e-2));
        assert!(approx(i.cx, CX, 1e-2) && approx(i.cy, CY, 1e-2));
    }

    #[test]
    fn pose_matrices() {
        let mut f = build_frame();
        f.set_pose(make_pose());

        assert_vec(f.get_ow(), [-1.0, -2.0, -3.0], 1e-4);
        assert_vec(f.get_camera_center(), [-1.0, -2.0, -3.0], 1e-4);
        assert_mat3(
            f.get_rwc(),
            [
                0.935754836,
                0.302932680,
                -0.180540055,
                -0.283164918,
                0.950580657,
                0.127334565,
                0.210191682,
                -0.068031318,
                0.975290298,
            ],
            1e-4,
        );
        assert!(f.has_pose());
        assert!(f.is_set());
    }

    #[test]
    fn imu_getters() {
        let mut f = build_frame();
        f.set_pose(make_pose());

        assert_vec(
            f.get_imu_position(),
            [-1.038241386, -1.966795921, -3.020858765],
            1e-4,
        );
        assert_mat3(
            f.get_imu_rotation(),
            [
                0.929613411,
                0.332612872,
                -0.158706099,
                -0.314113677,
                0.940329552,
                0.130816728,
                0.192747355,
                -0.071757227,
                0.978621125,
            ],
            1e-4,
        );
        assert_vec(
            f.get_imu_pose().translation.vector,
            [-1.038241386, -1.966795921, -3.020858765],
            1e-4,
        );
    }

    #[test]
    fn in_ref_coordinates_applies_rcw_p_plus_tcw() {
        let mut f = build_frame();
        f.set_pose(make_pose());
        let got = f.in_ref_coordinates(Vector3::new(0.5, -0.3, 4.0));
        assert_vec(got, [2.393593788, 1.594166875, 6.772690773], 1e-4);
    }

    #[test]
    fn unproject_stereo_backprojects_keypoint() {
        let mut f = build_frame();
        f.set_pose(make_pose());

        f.keys_un = Some(vec![keypoint(400.0, 300.0)]);
        f.depth = vec![2.5];

        let x3d = f.unproject_stereo(0).expect("has depth");
        assert_vec(x3d, [-1.198632002, -1.463983774, -0.543413162], 1e-3);

        f.depth[0] = -1.0;
        assert!(f.unproject_stereo(0).is_none());
    }

    #[test]
    fn get_features_in_area() {
        let mut f = build_frame();
        let c = f.constants.clone();

        f.n = 3;
        let kps = vec![
            keypoint(100.0, 100.0),
            keypoint(105.0, 103.0),
            keypoint(400.0, 300.0),
        ];
        f.keys_un = Some(kps.clone());

        let mut grid = vec![Vec::<usize>::new(); FRAME_GRID_COLS * FRAME_GRID_ROWS];
        for (i, kp) in kps.iter().enumerate() {
            let (gx, gy) = pos_in_grid(kp, &c.bounds).unwrap();
            grid[grid_index(gx, gy)].push(i);
        }
        f.grid = grid;

        let mut near = f.get_features_in_area(100.0, 100.0, 10.0, -1, -1, false);
        near.sort();
        assert_eq!(near, vec![0, 1]);

        assert!(
            f.get_features_in_area(600.0, 50.0, 5.0, -1, -1, false)
                .is_empty()
        );

        let far = f.get_features_in_area(400.0, 300.0, 10.0, -1, -1, false);
        assert_eq!(far, vec![2]);
    }

    #[test]
    fn project_point_distort() {
        let mut f = build_frame();
        f.set_pose(make_pose());

        let mp = Arc::new(MapPoint::new());
        mp.set_world_pos(Vector3::new(-1.772220612, -1.225883961, 2.907996178));

        let kp = f.project_point_distort(&mp).expect("in front");
        assert!(approx(kp.x, 390.129882812, 1e-2), "u={}", kp.x);
        assert!(approx(kp.y, 255.990905762, 1e-2), "v={}", kp.y);
    }

    #[test]
    fn is_in_frustum() {
        let mut f = build_frame();
        f.set_pose(make_pose());
        assert!(f.n > 0);
        f.keys_un.as_mut().unwrap()[0].set_octave(0);

        let map = Arc::new(Map::new());
        let cam_pt = Vector3::new(0.2, 0.1, 5.0);
        let world = f.get_rwc() * cam_pt + f.get_ow();
        let mp = MapPoint::from_frame(&world, map, &f, 0);

        let res = f.is_in_frustum(&mp, 0.5);
        assert!(res.in_view());
        let t = res.left.expect("left projection");
        assert!(approx(t.proj_x, 385.561157227, 1e-2), "projX={}", t.proj_x);
        assert!(approx(t.proj_y, 257.520935059, 1e-2), "projY={}", t.proj_y);
        assert!(
            approx(t.proj_xr, 342.447692871, 1e-2),
            "projXR={}",
            t.proj_xr
        );
        assert!(approx(t.depth, 5.004997730, 1e-3), "depth={}", t.depth);
        assert_eq!(t.scale_level, 0);
        assert!(approx(t.view_cos, 1.0, 1e-4), "viewCos={}", t.view_cos);

        // Behind the camera -> not in view.
        let world_behind = f.get_rwc() * Vector3::new(0.2, 0.1, -5.0) + f.get_ow();
        let mp2 = Arc::new(MapPoint::new());
        mp2.set_world_pos(world_behind);
        assert!(!f.is_in_frustum(&mp2, 0.5).in_view());
    }
}
