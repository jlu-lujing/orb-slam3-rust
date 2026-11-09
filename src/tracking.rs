use crate::{
    atlas::Atlas,
    camera_models::GeometricCamera,
    frame::{Frame, FrameConstants},
    imu_types::{Bias, Calib, Point, Preintegrated},
    key_frame::KeyFrame,
    key_frame_database::{KeyFrameDatabase, KeyFrameId, KeyFrameView, MapId},
    local_mapping::LocalMapping,
    loop_closing::LoopClosing,
    map_point::MapPoint,
    optimizer,
    orb_extractor::OrbExtractor,
    orb_matcher::OrbMatcher,
    orb_vocabulary::{BowVector, OrbVocabulary},
    pnp::{PnpCorrespondence, solve_pnp_ransac},
    settings::{CameraType, IMUInfo, Settings},
    system::{Sensor, System},
    viewer::{ViewerFeature, ViewerImage, ViewerKeyFrame, ViewerSnapshot},
};
use nalgebra::{Isometry3, Matrix3, Vector3};
use opencv::imgproc::cvt_color;
use opencv::prelude::*;
use opencv::{
    core::{CV_32F, Mat, MatExprTraitConst, Point2f, Point3f, Scalar},
    imgproc::ColorConversionCodes,
};
use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, Ordering},
    },
};
use tracing::{info, warn};

pub struct Tracking {
    pub state: TrackingState,
    pub last_processed_state: Option<TrackingState>,

    // Input sensor
    pub sensor: Sensor,

    // Current frame
    pub current_frame: Option<Frame>,
    pub last_frame: Option<Arc<Frame>>,

    pub im_gray: Option<Mat>,
    pub im_right: Option<Mat>,

    // Initialization variables (Monocular)
    pub ini_matches: Vec<i32>,
    pub prev_matched: Vec<Point2f>,
    pub ini_p3d: Vec<Point3f>,
    pub initial_frame: Option<Frame>,

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    pub relative_frame_poses: Vec<Isometry3<f32>>,
    pub references: Vec<Arc<KeyFrame>>,
    pub frame_times: Vec<f64>,
    pub lost: Vec<bool>,

    // Frames with estimated pose
    pub tracked_fr: usize,
    pub step: bool,

    // True if local mapping is deactivated and we are performing only localization
    pub only_tracking: bool,

    pub init_with_3kfs: bool,
    pub t0: f64,    // timestamp of first read frame
    pub t0vis: f64, // timestamp of first inserted keyframe
    pub t0imu: f64, // timestamp of IMU initialization
    pub fast_init: bool,

    #[cfg(feature = "register-times")]
    pub rect_stereo_ms: Vec<f64>,
    #[cfg(feature = "register-times")]
    pub resize_image_ms: Vec<f64>,
    #[cfg(feature = "register-times")]
    pub orb_extract_ms: Vec<f64>,
    #[cfg(feature = "register-times")]
    pub stereo_match_ms: Vec<f64>,
    #[cfg(feature = "register-times")]
    pub imu_integ_ms: Vec<f64>,
    #[cfg(feature = "register-times")]
    pub pose_pred_ms: Vec<f64>,
    #[cfg(feature = "register-times")]
    pub lm_track_ms: Vec<f64>,
    #[cfg(feature = "register-times")]
    pub new_kf_ms: Vec<f64>,
    #[cfg(feature = "register-times")]
    pub track_total_ms: Vec<f64>,

    map_updated: bool,

    // IMU preintegration from last frame
    imu_preintegrated_from_last_kf: Option<Arc<Preintegrated>>,

    // Queue of IMU measurements between frames
    queue_imu_data: Vec<Point>,

    // Vector of IMU measurements from previous to current frame (to be filled by preintegrate_imu)
    imu_from_last_frame: Vec<Point>,

    // IMU calibration parameters
    imu_calib: Option<Arc<Calib>>,

    // Last Bias estimation (at keyframe creation)
    last_bias: Option<Bias>,

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    visual_odometry: bool,

    // Other thread pointers
    local_mapper: Option<Arc<LocalMapping>>,
    loop_closing: Option<Arc<LoopClosing>>,

    // ORB
    orb_extractor_left: Arc<OrbExtractor>,
    orb_extractor_right: Option<Arc<OrbExtractor>>,
    ini_orb_extractor: Option<Arc<OrbExtractor>>,

    // BoW
    orb_vocabulary: Arc<OrbVocabulary>,
    key_frame_database: Arc<KeyFrameDatabase>,

    // Initialization (only for monocular)
    ready_to_initialize: bool,

    // Local map
    reference_kf: Option<Arc<KeyFrame>>,
    local_key_frames: Vec<Arc<KeyFrame>>,
    local_map_points: Vec<Arc<MapPoint>>,

    // System
    system: Weak<System>,

    // Atlas
    atlas: Arc<Atlas>,

    // Calibration matrix
    k: Mat,
    k_n: Matrix3<f32>,
    dist_coef: Mat,

    // Per-camera constants shared by every Frame (C++ Frame statics:
    // image bounds, grid inv sizes, intrinsics).
    frame_constants: Arc<FrameConstants>,
    bf: Option<f32>,
    image_scale: f32,

    imu_freq: f32,
    imu_per: f64,
    insert_kfs_lost: bool,

    // New KeyFrame rules (according to fps)
    min_frames: i32,
    max_frames: i32,

    first_imu_frame_id: Option<usize>,
    frames_to_reset_imu: u64,

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points require a match in two keyframes
    th_depth: Option<f32>,

    // For RGB-D inputs only. For some datasets (eg TUM) the depthmap values are scaled
    depth_map_factor: Option<f32>,

    // Current matches in frame
    matches_inliers: usize,

    // Last Frame, KeyFrame and Relocalisation info
    last_key_frame: Option<Arc<KeyFrame>>,
    last_key_frame_id: Option<usize>,
    last_reloc_frame_id: usize,
    timestamp_lost: Option<f64>,
    timestamp_recently_lost: f64,

    first_frame_id: usize,
    initial_frame_id: usize,
    last_init_frame_id: Option<usize>,

    created_map: bool,

    // Motion model
    velocity: Option<Isometry3<f32>>,

    // Color order (true RGB, false BGR, ignored if grayscale)
    rgb: bool,

    temporal_points: Vec<Arc<MapPoint>>,

    num_dataset: usize,

    time_pre_int_imu: f64,
    time_pose_pred: f64,
    time_local_map_track: f64,
    time_new_kf_dec: f64,

    camera: Arc<dyn GeometricCamera>,
    camera2: Option<Arc<dyn GeometricCamera>>,

    init_id: usize,
    last_id: usize,

    #[cfg(feature = "tracker-pause-resume")]
    stopped: AtomicBool,
    #[cfg(feature = "tracker-pause-resume")]
    stop_requested: AtomicBool,

    tlr: Option<Isometry3<f32>>,
}

struct TrackingKeyFrameView {
    key_frames: HashMap<KeyFrameId, Arc<KeyFrame>>,
    bad_maps: HashSet<MapId>,
}

impl TrackingKeyFrameView {
    fn from_atlas(atlas: &Atlas) -> Self {
        let key_frames = atlas
            .get_atlas_key_frames()
            .into_iter()
            .map(|(id, keyframe)| (KeyFrameId(id), keyframe))
            .collect();
        let bad_maps = atlas
            .get_all_maps()
            .into_iter()
            .filter(|map| map.is_bad())
            .map(|map| MapId(map.get_id() as u64))
            .collect();
        Self {
            key_frames,
            bad_maps,
        }
    }
}

impl KeyFrameView for TrackingKeyFrameView {
    fn bow_vec(&self, keyframe: KeyFrameId) -> Option<&BowVector> {
        self.key_frames
            .get(&keyframe)
            .map(|keyframe| &keyframe.bow_vec)
    }

    fn map_id(&self, keyframe: KeyFrameId) -> Option<MapId> {
        self.key_frames
            .get(&keyframe)
            .and_then(|keyframe| keyframe.get_map())
            .map(|map| MapId(map.get_id() as u64))
    }

    fn is_bad(&self, keyframe: KeyFrameId) -> bool {
        self.key_frames
            .get(&keyframe)
            .is_none_or(|keyframe| keyframe.is_bad())
    }

    fn is_map_bad(&self, map: MapId) -> bool {
        self.bad_maps.contains(&map)
    }

    fn best_covisibility(&self, keyframe: KeyFrameId, n: usize) -> Vec<KeyFrameId> {
        self.key_frames
            .get(&keyframe)
            .map(|keyframe| {
                keyframe
                    .get_best_covisibility_keyframes(n)
                    .into_iter()
                    .map(|keyframe| KeyFrameId(keyframe.id))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn connected_keyframes(&self, keyframe: KeyFrameId) -> HashSet<KeyFrameId> {
        self.key_frames
            .get(&keyframe)
            .map(|keyframe| {
                keyframe
                    .get_connected_keyframes()
                    .into_iter()
                    .map(|keyframe| KeyFrameId(keyframe.id))
                    .collect()
            })
            .unwrap_or_default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackingState {
    SystemNotReady,
    NoImagesYet,
    NotInitialized,
    OK,
    RecentlyLost,
    Lost,
    OKKLT,
}

impl Tracking {
    pub fn new(
        system: Arc<System>,
        orb_vocabulary: Arc<OrbVocabulary>,
        atlas: Arc<Atlas>,
        key_frame_database: Arc<KeyFrameDatabase>,
        sensor: Sensor,
        settings: Arc<Settings>,
    ) -> Self {
        // Load camera parameters from settings file
        let camera = settings
            .camera1
            .as_ref()
            .expect("settings missing camera1")
            .calibration
            .clone();
        let camera = atlas.add_camera(camera);

        let dist_coef = if settings.need_to_undistort {
            settings.camera1_distortion_coef()
        } else {
            Mat::new_rows_cols_with_default(4, 1, CV_32F, Scalar::all(0.0)).unwrap()
        };

        let fx = camera.get_parameter(0);
        let fy = camera.get_parameter(1);
        let cx = camera.get_parameter(2);
        let cy = camera.get_parameter(3);
        let mut k = Mat::eye(3, 3, CV_32F).unwrap().to_mat().unwrap();
        *k.at_2d_mut::<f32>(0, 0).unwrap() = fx;
        *k.at_2d_mut::<f32>(1, 1).unwrap() = fy;
        *k.at_2d_mut::<f32>(0, 2).unwrap() = cx;
        *k.at_2d_mut::<f32>(1, 2).unwrap() = cy;
        // The working image size is known up-front from the settings: every
        // incoming image is resized to `new_size` before a Frame is built, and
        // the intrinsics in `k` are already scaled to those pixel coordinates.
        // This replaces the C++ lazy static computation.
        let frame_constants = Arc::new(
            FrameConstants::new(
                k.clone(),
                dist_coef.clone(),
                settings.image.new_size.width,
                settings.image.new_size.height,
            )
            // Infallible for a valid calibration + positive image size, both of
            // which are validated during settings parsing.
            .expect("failed to build frame constants from settings"),
        );

        let k_n: Matrix3<f32> = Matrix3::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);

        let (camera2, tlr) =
            if matches!(sensor, Sensor::Stereo | Sensor::IMUStereo | Sensor::IMURGBD)
                && settings.camera_model == CameraType::KannalaBrandt
            {
                let ret = settings.camera2.as_ref().unwrap().calibration.clone();
                let ret = atlas.add_camera(ret);
                // The desktop viewer reads immutable stereo snapshots.
                (
                    Some(ret),
                    Some(settings.stereo.as_ref().expect("missing stereo info").tlr),
                )
            } else {
                (None, None)
            };

        // Sensor dependant settings
        let (bf, th_depth, depth_map_factor) = match sensor {
            Sensor::Stereo | Sensor::IMUStereo => {
                let s = settings.stereo.as_ref().expect("missing stereo info");
                (Some(s.bf), Some(s.b * s.th_depth), None)
            }
            Sensor::RGBD | Sensor::IMURGBD => {
                let r = settings.rgbd.as_ref().expect("missing rgbd info");
                let mut factor = r.depth_map_factor;
                if factor.abs() < 1e-5 {
                    factor = 1.
                } else {
                    factor = 1. / factor;
                }
                (Some(r.bf), Some(r.b * r.th_depth), Some(factor))
            }
            _ => (None, None, None),
        };

        let max_frames = settings.image.fps;
        let rgb = settings.image.rgb;

        // ORB parameters
        let orb = &settings.orb;
        let orb_extractor_left = Arc::new(OrbExtractor::new(
            orb.n_features as usize,
            orb.scale_factor,
            orb.n_levels as usize,
            orb.init_th_fast,
            orb.min_th_fast,
        ));
        let orb_extractor_right = if matches!(sensor, Sensor::Stereo | Sensor::IMUStereo) {
            Some(Arc::new(OrbExtractor::new(
                orb.n_features as usize,
                orb.scale_factor,
                orb.n_levels as usize,
                orb.init_th_fast,
                orb.min_th_fast,
            )))
        } else {
            None
        };
        let ini_orb_extractor = if matches!(sensor, Sensor::Monocular | Sensor::IMUMonocular) {
            Some(Arc::new(OrbExtractor::new(
                orb.n_features as usize,
                orb.scale_factor,
                orb.n_levels as usize,
                orb.init_th_fast,
                orb.min_th_fast,
            )))
        } else {
            None
        };

        // IMU parameters
        let (insert_kfs_lost, imu_freq, imu_per, imu_calib, imu_preintegrated_from_last_kf) =
            if let Some(imu) = settings.imu.as_ref() {
                let calib = Arc::new(imu_calibration_from_settings(imu));
                (
                    imu.insert_kfs_when_lost,
                    imu.imu_frequency,
                    0.001, // Original C++ codebase comment: 1.0 / (double) mImuFreq; Is this OK?
                    Some(calib.clone()),
                    Some(Arc::new(Preintegrated::from_bias_and_calib(
                        &Bias::empty(),
                        &calib,
                    ))),
                )
            } else {
                (false, 0., 0., None, None)
            };

        // Log camera info
        let cameras = atlas.get_all_cameras();
        info!("There are {} cameras in the atlas", cameras.len());
        for cam in cameras {
            info!("Camera {} is {:?}", cam.get_id(), cam.get_type());
        }

        Tracking {
            state: TrackingState::NoImagesYet,
            last_processed_state: None,
            sensor,
            tracked_fr: 0,
            step: false,
            only_tracking: false,
            map_updated: false,
            visual_odometry: false,
            orb_vocabulary,
            key_frame_database,
            ready_to_initialize: false,
            system: Arc::downgrade(&system),
            atlas,
            last_reloc_frame_id: 0,
            timestamp_recently_lost: 5.,
            initial_frame_id: 0,
            created_map: false,
            first_frame_id: 0,
            last_key_frame: None,
            init_id: 0,
            last_id: 0,
            init_with_3kfs: false,
            num_dataset: 0,
            camera,
            camera2,
            tlr,
            dist_coef,
            frame_constants,
            image_scale: 1.,
            k,
            k_n,
            bf,
            th_depth,
            depth_map_factor,
            min_frames: 0,
            max_frames,
            // C++ applies this reset interval to the legacy camera configuration.
            frames_to_reset_imu: max_frames as u64,
            rgb,
            orb_extractor_left,
            orb_extractor_right,
            ini_orb_extractor,
            insert_kfs_lost,
            imu_freq,
            imu_per,
            imu_calib,
            imu_preintegrated_from_last_kf,
            current_frame: None,
            last_frame: None,
            fast_init: false,
            first_imu_frame_id: None,
            frame_times: Vec::new(),
            im_gray: None,
            im_right: None,
            imu_from_last_frame: Vec::new(),
            ini_matches: Vec::new(),
            prev_matched: Vec::new(),
            ini_p3d: Vec::new(),
            initial_frame: None,
            last_bias: None,
            last_init_frame_id: None,
            last_key_frame_id: None,
            local_key_frames: Vec::new(),
            local_map_points: Vec::new(),
            local_mapper: None,
            loop_closing: None,
            lost: Vec::new(),
            matches_inliers: 0,
            #[cfg(feature = "tracker-pause-resume")]
            stopped: AtomicBool::new(false),
            #[cfg(feature = "tracker-pause-resume")]
            stop_requested: AtomicBool::new(false),
            queue_imu_data: Vec::new(),
            reference_kf: None,
            references: Vec::new(),
            relative_frame_poses: Vec::new(),
            t0: 0.,
            t0vis: 0.,
            t0imu: 0.,
            temporal_points: Vec::new(),
            time_local_map_track: 0.,
            time_new_kf_dec: 0.,
            time_pose_pred: 0.,
            time_pre_int_imu: 0.,
            timestamp_lost: None,
            velocity: None,
            #[cfg(feature = "register-times")]
            rect_stereo_ms: Vec::new(),
            #[cfg(feature = "register-times")]
            resize_image_ms: Vec::new(),
            #[cfg(feature = "register-times")]
            orb_extract_ms: Vec::new(),
            #[cfg(feature = "register-times")]
            stereo_match_ms: Vec::new(),
            #[cfg(feature = "register-times")]
            imu_integ_ms: Vec::new(),
            #[cfg(feature = "register-times")]
            pose_pred_ms: Vec::new(),
            #[cfg(feature = "register-times")]
            lm_track_ms: Vec::new(),
            #[cfg(feature = "register-times")]
            new_kf_ms: Vec::new(),
            #[cfg(feature = "register-times")]
            track_total_ms: Vec::new(),
        }
    }

    // Preprocess the input and call track(). Extract features and performs stereo matching.
    pub fn grab_image_stereo(
        &mut self,
        im_rect_left: Mat,
        im_rect_right: Mat,
        timestamp: f64,
        filename: String,
    ) -> Option<Isometry3<f32>> {
        let mut im_gray = im_rect_left;
        let mut im_right = im_rect_right;

        // Convert the colors to gray
        for image in [&mut im_gray, &mut im_right] {
            let conversion = match image.channels() {
                1 => None,
                3 if self.rgb => Some(ColorConversionCodes::COLOR_RGB2GRAY),
                3 => Some(ColorConversionCodes::COLOR_BGR2GRAY),
                4 if self.rgb => Some(ColorConversionCodes::COLOR_RGBA2GRAY),
                4 => Some(ColorConversionCodes::COLOR_BGRA2GRAY),
                channels => {
                    warn!("Unsupported image channel count: {channels}");
                    return None;
                }
            };
            if let Some(conversion) = conversion {
                cvt_color(
                    &image.clone(),
                    image,
                    conversion as i32,
                    0,
                    opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
                )
                .ok()?;
            }
        }

        // Construct the frame. Two orthogonal axes of variation:
        //   - IMU vs non-IMU: whether we chain to the previous frame and which
        //     IMU calibration is used.
        //   - fisheye (two cameras) vs single pinhole: which constructor and
        //     whether `camera2`/`tlr` are supplied.
        let (prev_frame, imu_calib) = match self.sensor {
            Sensor::Stereo => (None, Arc::new(Calib::new())),
            Sensor::IMUStereo => (
                self.last_frame.clone(),
                self.imu_calib.as_ref().expect("missing imu calib").clone(),
            ),
            // grab_image_stereo is only valid for stereo sensors.
            _ => return None,
        };

        let extractor_left = self.orb_extractor_left.clone();
        let extractor_right = self
            .orb_extractor_right
            .as_ref()
            .expect("missing right extractor")
            .clone();
        let bf = self.bf.expect("missing bf");
        let th_depth = self.th_depth.expect("missing th_depth");

        let previous_frame = self.current_frame.take();
        let next_frame = if let Some(camera2) = &self.camera2 {
            Frame::from_stereo_fisheye(
                &im_gray,
                &im_right,
                timestamp,
                extractor_left,
                extractor_right,
                self.orb_vocabulary.clone(),
                self.frame_constants.clone(),
                bf,
                th_depth,
                self.camera.clone(),
                camera2.clone(),
                self.tlr.expect("missing tlr"),
                prev_frame,
                imu_calib,
            )
        } else {
            Frame::from_stereo(
                &im_gray,
                &im_right,
                timestamp,
                extractor_left,
                extractor_right,
                self.orb_vocabulary.clone(),
                self.frame_constants.clone(),
                bf,
                th_depth,
                self.camera.clone(),
                prev_frame,
                imu_calib,
            )
        };
        self.current_frame = next_frame;
        if self.current_frame.is_none() {
            self.current_frame = previous_frame;
            self.im_gray = Some(im_gray);
            self.im_right = Some(im_right);
            return None;
        }
        if let Some(previous) = previous_frame {
            self.last_frame = Some(Arc::new(previous));
        }

        if let Some(current_frame) = self.current_frame.as_mut() {
            current_frame.name_file = filename;
            current_frame.dataset = self.num_dataset;

            #[cfg(feature = "register-times")]
            self.orb_extract_ms.push(current_frame.time_orb_ext);
            #[cfg(feature = "register-times")]
            self.stereo_match_ms.push(current_frame.time_stereo_match);
        }

        self.track();
        self.current_frame.as_ref().map(Frame::get_pose)
    }

    pub fn grab_image_monocular(
        &mut self,
        image: Mat,
        timestamp: f64,
        filename: String,
    ) -> Option<Isometry3<f32>> {
        let mut image = image;
        if image.channels() > 1 {
            let conversion = match (image.channels(), self.rgb) {
                (4, true) => ColorConversionCodes::COLOR_RGBA2GRAY,
                (4, false) => ColorConversionCodes::COLOR_BGRA2GRAY,
                (_, true) => ColorConversionCodes::COLOR_RGB2GRAY,
                (_, false) => ColorConversionCodes::COLOR_BGR2GRAY,
            };
            cvt_color(
                &image.clone(),
                &mut image,
                conversion as i32,
                0,
                opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
            )
            .ok()?;
        }

        let previous_frame = self.current_frame.take();
        let imu_calib = if matches!(self.sensor, Sensor::IMUMonocular) {
            self.imu_calib.as_ref().expect("missing imu calib").clone()
        } else {
            Arc::new(Calib::new())
        };
        let previous = previous_frame.as_ref().map(|frame| Arc::new(frame.clone()));
        let next = Frame::from_monocular(
            &image,
            timestamp,
            self.orb_extractor_left.clone(),
            self.orb_vocabulary.clone(),
            self.frame_constants.clone(),
            0.0,
            0.0,
            self.camera.clone(),
            previous,
            imu_calib,
        );
        self.current_frame = next;
        if self.current_frame.is_none() {
            self.current_frame = previous_frame;
            self.im_gray = Some(image);
            return None;
        }
        if let Some(previous) = previous_frame {
            self.last_frame = Some(Arc::new(previous));
        }
        if let Some(current) = self.current_frame.as_mut() {
            current.name_file = filename;
            current.dataset = self.num_dataset;
        }
        self.track();
        self.current_frame.as_ref().map(Frame::get_pose)
    }

    pub fn grab_image_rgbd(
        &mut self,
        image: Mat,
        mut depth: Mat,
        timestamp: f64,
        filename: String,
    ) -> Option<Isometry3<f32>> {
        let mut image = image;
        if image.channels() > 1 {
            let conversion = if self.rgb {
                ColorConversionCodes::COLOR_RGB2GRAY
            } else {
                ColorConversionCodes::COLOR_BGR2GRAY
            };
            cvt_color(
                &image.clone(),
                &mut image,
                conversion as i32,
                0,
                opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
            )
            .ok()?;
        }
        if depth.channels() != 1 {
            return None;
        }
        if let Some(factor) = self.depth_map_factor {
            let mut scaled = Mat::default();
            depth
                .convert_to(&mut scaled, CV_32F, factor as f64, 0.0)
                .ok()?;
            depth = scaled;
        } else if depth.depth() != CV_32F {
            let mut converted = Mat::default();
            depth.convert_to(&mut converted, CV_32F, 1.0, 0.0).ok()?;
            depth = converted;
        }

        let previous_frame = self.current_frame.take();
        let imu_calib = if matches!(self.sensor, Sensor::IMURGBD) {
            self.imu_calib.as_ref().expect("missing imu calib").clone()
        } else {
            Arc::new(Calib::new())
        };
        let previous = previous_frame.as_ref().map(|frame| Arc::new(frame.clone()));
        self.current_frame = Frame::from_rgbd(
            &image,
            &depth,
            timestamp,
            self.orb_extractor_left.clone(),
            self.orb_vocabulary.clone(),
            self.frame_constants.clone(),
            self.bf.unwrap_or(0.0),
            self.th_depth.unwrap_or(0.0),
            self.camera.clone(),
            previous,
            imu_calib,
        );
        if self.current_frame.is_none() {
            self.current_frame = previous_frame;
            self.im_gray = Some(image);
            return None;
        }
        if let Some(previous) = previous_frame {
            self.last_frame = Some(Arc::new(previous));
        }
        if let Some(current) = self.current_frame.as_mut() {
            current.name_file = filename;
            current.dataset = self.num_dataset;
        }
        self.track();
        self.current_frame.as_ref().map(Frame::get_pose)
    }

    fn insert_stereo_map_points(
        &self,
        frame: &mut Frame,
        kf: &Arc<KeyFrame>,
        map: Arc<crate::map::Map>,
        limit: Option<usize>,
    ) {
        let mut candidates = Vec::new();
        if frame.camera2.is_none() {
            for i in 0..frame.n {
                let depth = frame.depth.get(i).copied().unwrap_or(-1.0);
                if depth > 0.0 {
                    candidates.push((depth, i));
                }
            }
        } else {
            let n_left = frame.n_left.unwrap_or(0);
            if let Some(left_to_right) = &frame.left_to_right_match {
                for i in 0..n_left {
                    let depth = frame.depth.get(i).copied().unwrap_or(-1.0);
                    if depth > 0.0 && left_to_right[i] != usize::MAX {
                        candidates.push((depth, i));
                    }
                }
            }
        }
        candidates.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut created = 0usize;
        for (depth, i) in candidates {
            let existing_is_valid = frame
                .map_points
                .get(i)
                .and_then(|mp| mp.as_ref())
                .is_some_and(|mp| mp.observations() > 0 && !mp.is_bad());
            if existing_is_valid {
                continue;
            }

            let world_pos = if frame.camera2.is_none() {
                frame.unproject_stereo(i)
            } else {
                Some(frame.unproject_stereo_fisheye(i))
            };
            let Some(world_pos) = world_pos else {
                continue;
            };

            let mp = MapPoint::from_pos_ref(&world_pos, kf.clone(), map.clone());
            mp.add_observation(kf, i as i32);
            kf.add_map_point(mp.clone(), i);
            frame.map_points[i] = Some(mp.clone());

            if frame.camera2.is_some()
                && let Some(n_left) = frame.n_left
                && let Some(left_to_right) = &frame.left_to_right_match
                && left_to_right[i] != usize::MAX
            {
                let right_idx = n_left + left_to_right[i];
                mp.add_observation(kf, right_idx as i32);
                kf.add_map_point(mp.clone(), right_idx);
                frame.map_points[right_idx] = Some(mp.clone());
            }

            mp.compute_distinctive_descriptors();
            mp.update_normal_and_depth();
            self.atlas.add_map_point(mp);
            created += 1;

            if limit.is_some_and(|max| created >= max && depth > frame.th_depth) {
                break;
            }
        }
    }

    fn stereo_initialization(&mut self, frame: &mut Frame) {
        if frame.n <= 500 {
            return;
        }

        if matches!(self.sensor, Sensor::IMUStereo | Sensor::IMURGBD) {
            let r_wb = frame
                .imu_calib
                .tcb
                .rotation
                .to_rotation_matrix()
                .into_inner();
            let t_wb = frame.imu_calib.tcb.translation.vector;
            frame.set_imu_pose_velocity(r_wb, t_wb, Vector3::zeros());
        } else {
            frame.set_pose(Isometry3::identity());
        }

        let map = self.atlas.get_current_map();
        let kf = KeyFrame::from_frame(frame, map.clone());
        self.key_frame_database.add(KeyFrameId(kf.id), &kf.bow_vec);
        self.atlas.add_key_frame(kf.clone());
        self.insert_stereo_map_points(frame, &kf, map.clone(), None);

        if let Some(local_mapper) = &self.local_mapper {
            local_mapper.insert_key_frame(kf.clone());
        }

        frame.reference_kf = Some(kf.clone());
        frame.last_keyframe = Some(kf.clone());
        self.reference_kf = Some(kf.clone());
        self.last_key_frame = Some(kf.clone());
        self.last_key_frame_id = Some(frame.id);
        self.local_key_frames = vec![kf.clone()];
        self.local_map_points = self.atlas.get_all_map_points().unwrap_or_default();
        self.atlas
            .set_reference_map_points(self.local_map_points.clone());
        map.add_key_frame_origin(kf);
        self.state = TrackingState::OK;
        self.created_map = false;
    }

    fn monocular_initialization(&mut self, frame: &mut Frame) {
        if !self.ready_to_initialize {
            if frame.n > 100 {
                self.initial_frame = Some(frame.clone());
                self.last_frame = Some(Arc::new(frame.clone()));
                self.prev_matched = frame
                    .keys_un
                    .as_ref()
                    .expect("monocular keys_un")
                    .iter()
                    .map(|keypoint| keypoint.pt())
                    .collect();
                self.ini_matches = vec![-1; frame.n];
                self.ready_to_initialize = true;
            }
            return;
        }

        let Some(initial) = self.initial_frame.clone() else {
            self.ready_to_initialize = false;
            return;
        };
        if frame.n <= 100
            || (matches!(self.sensor, Sensor::IMUMonocular)
                && frame.timestamp - initial.timestamp > 1.0)
        {
            self.ready_to_initialize = false;
            return;
        }

        let matcher = OrbMatcher::new(0.9, true);
        let (raw_matches, mut n_matches) =
            matcher.search_for_initialization(&initial, frame, &mut self.prev_matched, 100);
        if n_matches < 100 {
            self.ready_to_initialize = false;
            return;
        }
        let matches = raw_matches
            .iter()
            .map(|&index| (index >= 0).then_some(index as usize))
            .collect::<Vec<_>>();
        let reconstruction = self.camera.reconstruct_with_two_views(
            initial.keys_un.as_ref().expect("initial keys"),
            frame.keys_un.as_ref().expect("current keys"),
            &matches,
        );
        let Some(reconstruction) = reconstruction else {
            self.ready_to_initialize = false;
            return;
        };

        let triangulated = reconstruction.triangulated();
        let mut filtered_matches = raw_matches;
        for (index, match_index) in filtered_matches.iter_mut().enumerate() {
            if *match_index >= 0 && !triangulated.get(index).copied().unwrap_or(false) {
                *match_index = -1;
                n_matches -= 1;
            }
        }
        if n_matches < 100 {
            self.ready_to_initialize = false;
            return;
        }

        self.initial_frame
            .as_mut()
            .expect("initial frame")
            .set_pose(Isometry3::identity());
        frame.set_pose(reconstruction.t21());
        self.ini_matches = filtered_matches;
        self.ini_p3d = reconstruction.points_3d().to_vec();
        self.initial_frame = Some(initial);
        self.create_initial_map_monocular(frame);
    }

    fn create_initial_map_monocular(&mut self, frame: &mut Frame) {
        let Some(initial) = self.initial_frame.as_ref().cloned() else {
            return;
        };
        let map = self.atlas.get_current_map();
        let initial_kf = KeyFrame::from_frame(&initial, map.clone());
        let current_kf = KeyFrame::from_frame(frame, map.clone());
        self.key_frame_database
            .add(KeyFrameId(initial_kf.id), &initial_kf.bow_vec);
        self.key_frame_database
            .add(KeyFrameId(current_kf.id), &current_kf.bow_vec);
        self.atlas.add_key_frame(initial_kf.clone());
        self.atlas.add_key_frame(current_kf.clone());

        for (initial_index, &current_index) in self.ini_matches.iter().enumerate() {
            if current_index < 0 || initial_index >= self.ini_p3d.len() {
                continue;
            }
            let point = &self.ini_p3d[initial_index];
            let world = Vector3::new(point.x, point.y, point.z);
            let mp = MapPoint::from_pos_ref(&world, current_kf.clone(), map.clone());
            initial_kf.add_map_point(mp.clone(), initial_index);
            current_kf.add_map_point(mp.clone(), current_index as usize);
            mp.add_observation(&initial_kf, initial_index as i32);
            mp.add_observation(&current_kf, current_index);
            mp.compute_distinctive_descriptors();
            mp.update_normal_and_depth();
            self.atlas.add_map_point(mp.clone());
            frame.map_points[current_index as usize] = Some(mp);
        }

        initial_kf.update_connections();
        current_kf.update_connections();
        optimizer::global_bundle_adjustment(&map, 20, None, 0, false);
        let median_depth = initial_kf.compute_scene_median_depth(2);
        if median_depth <= 0.0 || current_kf.tracked_map_points(1) < 50 {
            self.ready_to_initialize = false;
            return;
        }
        let inverse_median_depth = 1.0 / median_depth;
        let mut scaled_pose = current_kf.get_pose();
        scaled_pose.translation.vector *= inverse_median_depth;
        current_kf.set_pose(scaled_pose);
        for mp in frame.map_points.iter().flatten() {
            mp.set_world_pos(mp.get_world_pos() * inverse_median_depth);
            mp.update_normal_and_depth();
        }

        if let Some(mapper) = &self.local_mapper {
            mapper.insert_key_frame(initial_kf.clone());
            mapper.insert_key_frame(current_kf.clone());
        }
        frame.reference_kf = Some(current_kf.clone());
        frame.last_keyframe = Some(current_kf.clone());
        self.reference_kf = Some(current_kf.clone());
        self.last_key_frame = Some(current_kf.clone());
        self.last_key_frame_id = Some(frame.id);
        self.local_key_frames = vec![current_kf.clone(), initial_kf];
        self.local_map_points = self.atlas.get_all_map_points().unwrap_or_default();
        self.atlas
            .set_reference_map_points(self.local_map_points.clone());
        self.last_frame = Some(Arc::new(frame.clone()));
        self.ready_to_initialize = false;
        self.state = TrackingState::OK;
        self.initial_frame_id = current_kf.id as usize;
        self.init_id = current_kf.id as usize;
        map.add_key_frame_origin(current_kf);
    }

    fn check_replaced_in_last_frame(&mut self) {
        if let Some(last) = self.last_frame.as_mut().and_then(|last| Arc::get_mut(last)) {
            for mp in &mut last.map_points {
                if let Some(current) = mp
                    && let Some(replacement) = current.get_replaced()
                {
                    *mp = Some(replacement);
                }
            }
        }
    }

    fn clean_outliers(&mut self, frame: &mut Frame) -> usize {
        let mut inliers = 0usize;
        for i in 0..frame.n {
            let Some(mp) = frame.map_points[i].clone() else {
                continue;
            };
            if frame.outlier[i] {
                mp.set_track_in_view(false);
                mp.set_track_in_view_r(false);
                mp.set_last_frame_seen(frame.id as u64);
                frame.map_points[i] = None;
                frame.outlier[i] = false;
            } else if mp.observations() > 0 && !mp.is_bad() {
                inliers += 1;
            }
        }
        inliers
    }

    fn track_reference_key_frame(&mut self, frame: &mut Frame) -> bool {
        let Some(reference) = self.reference_kf.as_ref() else {
            return false;
        };
        frame.compute_bow();
        let matcher = OrbMatcher::new(0.7, true);
        let mut matches = Vec::new();
        let n_matches = matcher.search_by_bow_frame(reference, frame, &mut matches);
        if n_matches < 15 {
            return false;
        }

        frame.map_points = matches;
        if let Some(last) = &self.last_frame {
            frame.set_pose(last.get_pose());
        } else {
            frame.set_pose(reference.get_pose());
        }
        optimizer::pose_optimization(frame);
        let inliers = self.clean_outliers(frame);
        inliers >= 10
    }

    fn track_with_motion_model(&mut self, frame: &mut Frame) -> bool {
        let Some(last) = self.last_frame.clone() else {
            return false;
        };
        if self.atlas.is_imu_initialized().unwrap_or(false) && self.predict_state_imu(frame) {
            return true;
        }
        let Some(velocity) = self.velocity else {
            return self.track_reference_key_frame(frame);
        };

        frame.set_pose(velocity * last.get_pose());
        frame.map_points.fill(None);
        frame.outlier.fill(false);

        let matcher = OrbMatcher::new(0.9, true);
        let monocular = matches!(self.sensor, Sensor::Monocular | Sensor::IMUMonocular);
        let th = if matches!(self.sensor, Sensor::Stereo) {
            7.0
        } else {
            15.0
        };
        let mut n_matches = matcher.search_by_projection_last_frame(frame, &last, th, monocular);
        if n_matches < 20 {
            frame.map_points.fill(None);
            n_matches = matcher.search_by_projection_last_frame(frame, &last, 2.0 * th, monocular);
        }
        if n_matches < 20 {
            return matches!(
                self.sensor,
                Sensor::IMUMonocular | Sensor::IMUStereo | Sensor::IMURGBD
            );
        }

        optimizer::pose_optimization(frame);
        let inliers = self.clean_outliers(frame);
        if self.only_tracking {
            self.visual_odometry = inliers < 10;
            return n_matches > 20;
        }
        inliers >= 10
    }

    fn predict_state_imu(&mut self, frame: &mut Frame) -> bool {
        let (position, rotation, velocity, preintegrated, bias) = if self.map_updated {
            let (Some(keyframe), Some(preintegrated)) = (
                self.last_key_frame.as_ref(),
                self.imu_preintegrated_from_last_kf.as_ref(),
            ) else {
                return false;
            };
            (
                keyframe.get_imu_position(),
                keyframe.get_imu_rotation(),
                keyframe.get_velocity(),
                preintegrated.clone(),
                keyframe.get_imu_bias(),
            )
        } else {
            let (Some(last), Some(preintegrated)) = (
                self.last_frame.as_ref(),
                self.last_frame
                    .as_ref()
                    .and_then(|frame| frame.imu_preintegrated_frame.as_ref()),
            ) else {
                return false;
            };
            (
                last.get_imu_position(),
                last.get_imu_rotation(),
                last.get_velocity(),
                preintegrated.clone(),
                last.imu_bias,
            )
        };

        let dt = preintegrated.total_dt();
        if dt <= 0.0 {
            return false;
        }
        let gravity = Vector3::new(0.0, 0.0, -9.81);
        let predicted_rotation = rotation * preintegrated.get_delta_rotation(&bias);
        let predicted_position = position
            + velocity * dt
            + 0.5 * dt * dt * gravity
            + rotation * preintegrated.get_delta_position(&bias);
        let predicted_velocity =
            velocity + gravity * dt + rotation * preintegrated.get_delta_velocity(&bias);
        frame.set_imu_pose_velocity(predicted_rotation, predicted_position, predicted_velocity);
        frame.imu_bias = bias;
        frame.pred_bias = bias;
        true
    }

    fn update_local_points(&mut self, frame: &Frame) {
        self.local_map_points.clear();
        let mut seen = HashSet::new();
        for kf in self.local_key_frames.iter().rev() {
            for mp in kf.get_map_point_matches().into_iter().flatten() {
                if !mp.is_bad() && seen.insert(mp.id) {
                    self.local_map_points.push(mp);
                }
            }
        }
        self.atlas
            .set_reference_map_points(self.local_map_points.clone());
        let _ = frame;
    }

    fn update_local_key_frames(&mut self, frame: &Frame) {
        let mut counts: HashMap<u64, (Arc<KeyFrame>, i32)> = HashMap::new();
        for mp in frame.map_points.iter().flatten() {
            if mp.is_bad() {
                continue;
            }
            for (kf, _) in mp.get_observations() {
                if !kf.is_bad() {
                    counts
                        .entry(kf.id)
                        .and_modify(|(_, count)| *count += 1)
                        .or_insert((kf, 1));
                }
            }
        }

        let best = counts
            .values()
            .max_by_key(|(_, count)| *count)
            .map(|(kf, _)| kf.clone());
        let mut local = Vec::with_capacity(counts.len() * 3);
        let mut ids = HashSet::new();
        for (kf, _) in counts.into_values() {
            if ids.insert(kf.id) {
                local.push(kf);
            }
        }

        let first_layer = local.len();
        for i in 0..first_layer {
            if local.len() > 80 {
                break;
            }
            for neighbor in local[i].get_best_covisibility_keyframes(10) {
                if !neighbor.is_bad() && ids.insert(neighbor.id) {
                    local.push(neighbor);
                    break;
                }
            }
            for child in local[i].get_children() {
                if !child.is_bad() && ids.insert(child.id) {
                    local.push(child);
                    break;
                }
            }
            if let Some(parent) = local[i].get_parent()
                && !parent.is_bad()
                && ids.insert(parent.id)
            {
                local.push(parent);
            }
        }

        if matches!(
            self.sensor,
            Sensor::IMUMonocular | Sensor::IMUStereo | Sensor::IMURGBD
        ) && local.len() < 80
        {
            let mut temporal = self.last_key_frame.clone();
            for _ in 0..20 {
                let Some(kf) = temporal.take() else {
                    break;
                };
                if !kf.is_bad() && ids.insert(kf.id) {
                    local.push(kf.clone());
                }
                temporal = kf.get_prev_kf();
            }
        }

        self.local_key_frames = local;
        if let Some(reference) = best {
            self.reference_kf = Some(reference.clone());
            if let Some(current) = self.current_frame.as_mut() {
                current.reference_kf = Some(reference);
            }
        }
    }

    fn update_local_map(&mut self, frame: &Frame) {
        self.update_local_key_frames(frame);
        self.update_local_points(frame);
    }

    fn search_local_points(&self, frame: &mut Frame) {
        for i in 0..frame.n {
            let Some(mp) = frame.map_points[i].clone() else {
                continue;
            };
            if mp.is_bad() {
                frame.map_points[i] = None;
            } else {
                mp.increase_visible(1);
                mp.set_last_frame_seen(frame.id as u64);
                mp.set_track_in_view(false);
                mp.set_track_in_view_r(false);
            }
        }

        for mp in &self.local_map_points {
            if mp.last_frame_seen() == frame.id as u64 || mp.is_bad() {
                continue;
            }
            let frustum = frame.is_in_frustum(mp, 0.5);
            if let Some(left) = frustum.left {
                mp.set_track_in_view(true);
                mp.set_track_proj_x(left.proj_x);
                mp.set_track_proj_y(left.proj_y);
                mp.set_track_proj_xr(left.proj_xr);
                mp.set_track_depth(left.depth);
                mp.set_track_scale_level(left.scale_level);
                mp.set_track_view_cos(left.view_cos);
                mp.increase_visible(1);
            }
            if let Some(right) = frustum.right {
                mp.set_track_in_view_r(true);
                mp.set_track_proj_xr(right.proj_x);
                mp.set_track_proj_yr(right.proj_y);
                mp.set_track_depth_r(right.depth);
                mp.set_track_scale_level_r(right.scale_level);
                mp.set_track_view_cos_r(right.view_cos);
                mp.increase_visible(1);
            }
        }

        let th = if matches!(self.sensor, Sensor::IMUMonocular | Sensor::IMUStereo) {
            10.0
        } else {
            1.0
        };
        let matcher = OrbMatcher::new(0.8, true);
        matcher.search_by_projection_keypoints_mappoints(
            frame,
            &self.local_map_points,
            th,
            false,
            0.0,
        );
    }

    fn track_local_map(&mut self, frame: &mut Frame) -> bool {
        self.tracked_fr += 1;
        self.update_local_map(frame);
        self.search_local_points(frame);

        let map = self.atlas.get_current_map();
        if map.is_imu_initialized() && !self.map_updated {
            optimizer::pose_inertial_optimization_last_frame(frame, false);
        } else if map.is_imu_initialized() {
            optimizer::pose_inertial_optimization_last_keyframe(frame, false);
        } else {
            optimizer::pose_optimization(frame);
        }

        self.matches_inliers = 0;
        for i in 0..frame.n {
            let Some(mp) = frame.map_points[i].as_ref() else {
                continue;
            };
            if !frame.outlier[i] {
                mp.increase_found(1);
                if !self.only_tracking {
                    if mp.observations() > 0 {
                        self.matches_inliers += 1;
                    }
                } else {
                    self.matches_inliers += 1;
                }
            } else if matches!(self.sensor, Sensor::Stereo) {
                frame.map_points[i] = None;
            }
        }

        let min_inliers = if matches!(
            self.sensor,
            Sensor::IMUMonocular | Sensor::IMUStereo | Sensor::IMURGBD
        ) {
            15
        } else {
            30
        };
        self.matches_inliers >= min_inliers
    }

    fn relocalization(&mut self, frame: &mut Frame) -> bool {
        frame.compute_bow();
        let map = self.atlas.get_current_map();
        let view = TrackingKeyFrameView::from_atlas(&self.atlas);
        let candidates = self.key_frame_database.detect_relocalization_candidates(
            &view,
            &frame.bow_vec,
            MapId(map.get_id() as u64),
        );
        let matcher = OrbMatcher::new(0.75, true);

        for candidate in candidates {
            let Some(candidate_kf) = view.key_frames.get(&candidate).cloned() else {
                continue;
            };
            frame.map_points.fill(None);
            frame.outlier.fill(false);
            frame.set_pose(candidate_kf.get_pose());
            let matches = matcher.search_by_projection_keyframe(
                frame,
                &candidate_kf,
                HashSet::new(),
                10.0,
                100,
            );
            if matches < 20 {
                continue;
            }
            let Some((pnp_pose, inliers)) = pnp_relocalization_pose(frame) else {
                continue;
            };
            let map_points = std::mem::take(&mut frame.map_points);
            for (idx, map_point) in map_points.into_iter().enumerate() {
                if inliers.contains(&idx) {
                    frame.map_points[idx] = map_point;
                }
            }
            frame.set_pose(pnp_pose);
            optimizer::pose_optimization(frame);
            let inliers = self.clean_outliers(frame);
            tracing::info!(
                "Relocalization candidate {}: {} inliers",
                candidate_kf.id,
                inliers
            );
            if inliers >= 50 {
                self.reference_kf = Some(candidate_kf.clone());
                frame.reference_kf = Some(candidate_kf);
                self.velocity = None;
                self.matches_inliers = inliers;
                self.state = TrackingState::OK;
                return true;
            }
        }
        false
    }

    fn need_new_key_frame(&self, frame: &Frame) -> bool {
        let Some(local_mapper) = &self.local_mapper else {
            return false;
        };
        if matches!(
            self.sensor,
            Sensor::IMUMonocular | Sensor::IMUStereo | Sensor::IMURGBD
        ) && !self.atlas.is_imu_initialized().unwrap_or(false)
        {
            return self
                .last_key_frame
                .as_ref()
                .is_some_and(|kf| frame.timestamp - kf.timestamp >= 0.25);
        }
        if self.only_tracking || local_mapper.is_stopped() || local_mapper.stop_requested() {
            return false;
        }

        let n_kfs = self.atlas.key_frames_in_map().unwrap_or(0);
        if frame.id < self.last_reloc_frame_id + self.max_frames as usize
            && n_kfs > self.max_frames as usize
        {
            return false;
        }
        let Some(reference) = &self.reference_kf else {
            return false;
        };
        let min_obs = if n_kfs <= 2 { 2 } else { 3 };
        let reference_matches = reference.tracked_map_points(min_obs) as usize;

        let mut tracked_close = 0usize;
        let mut untracked_close = 0usize;
        let n_left = frame.n_left.unwrap_or(frame.n);
        for i in 0..n_left {
            let depth = frame.depth.get(i).copied().unwrap_or(-1.0);
            if depth > 0.0 && depth < frame.th_depth {
                if frame.map_points[i].is_some() && !frame.outlier[i] {
                    tracked_close += 1;
                } else {
                    untracked_close += 1;
                }
            }
        }

        let need_close = tracked_close < 100 && untracked_close > 70;
        let ratio = if n_kfs < 2 { 0.4 } else { 0.75 };
        let c1a = frame.id.saturating_sub(self.last_key_frame_id.unwrap_or(0))
            >= self.max_frames as usize;
        let c1b = frame.id.saturating_sub(self.last_key_frame_id.unwrap_or(0))
            >= self.min_frames as usize
            && local_mapper.accept_key_frames();
        let c2 = (self.matches_inliers < (reference_matches as f32 * ratio) as usize || need_close)
            && self.matches_inliers > 15;
        let c3 = self.last_key_frame.as_ref().is_some_and(|kf| {
            matches!(
                self.sensor,
                Sensor::IMUMonocular | Sensor::IMUStereo | Sensor::IMURGBD
            ) && frame.timestamp - kf.timestamp >= 0.5
        });

        if (c1a || c1b) && c2 || c3 {
            if local_mapper.accept_key_frames() || local_mapper.is_initializing() {
                true
            } else {
                local_mapper.interrupt_ba();
                local_mapper.keyframes_in_queue() < 3
            }
        } else {
            false
        }
    }

    fn create_new_key_frame(&mut self, frame: &mut Frame) {
        let Some(local_mapper) = self.local_mapper.clone() else {
            return;
        };
        if local_mapper.is_initializing() && !self.atlas.is_imu_initialized().unwrap_or(false) {
            return;
        }
        if !local_mapper.set_not_stop(true) {
            return;
        }

        let map = self.atlas.get_current_map();
        let kf = KeyFrame::from_frame(frame, map.clone());
        self.key_frame_database.add(KeyFrameId(kf.id), &kf.bow_vec);
        kf.set_new_bias(frame.imu_bias);
        self.reference_kf = Some(kf.clone());
        frame.reference_kf = Some(kf.clone());

        if let Some(last) = &self.last_key_frame {
            kf.set_prev_kf(last);
            last.set_next_kf(&kf);
        }
        if let Some(calib) = &self.imu_calib {
            self.imu_preintegrated_from_last_kf = Some(Arc::new(
                Preintegrated::from_bias_and_calib(&frame.imu_bias, calib),
            ));
        }

        self.insert_stereo_map_points(frame, &kf, map, Some(100));
        local_mapper.insert_key_frame(kf.clone());
        local_mapper.set_not_stop(false);
        self.last_key_frame_id = Some(frame.id);
        self.last_key_frame = Some(kf);
        self.created_map = false;
    }

    fn preintegrate_imu(&mut self, frame: &mut Frame) {
        let Some(calib) = &self.imu_calib else {
            return;
        };
        let Some(previous_frame) = frame.prev_frame.clone().or_else(|| self.last_frame.clone())
        else {
            frame.set_integrated();
            return;
        };

        let previous_timestamp = previous_frame.timestamp;
        let current_timestamp = frame.timestamp;
        let boundary_epsilon = self.imu_per;
        let queue = std::mem::take(&mut self.queue_imu_data);
        let mut selected = Vec::new();
        let mut retained = Vec::new();
        for (index, measurement) in queue.iter().enumerate() {
            if measurement.t < previous_timestamp - boundary_epsilon {
                continue;
            }
            if measurement.t < current_timestamp - boundary_epsilon {
                selected.push(measurement.clone());
            } else {
                selected.push(measurement.clone());
                retained.extend(queue[index..].iter().cloned());
                break;
            }
        }
        self.queue_imu_data = retained;
        self.imu_from_last_frame = selected.clone();
        if selected.len() < 2 {
            frame.set_integrated();
            return;
        }

        let mut from_last_kf = self
            .imu_preintegrated_from_last_kf
            .as_deref()
            .cloned()
            .unwrap_or_else(|| Preintegrated::from_bias_and_calib(&frame.imu_bias, calib));
        let mut frame_preintegrated = Preintegrated::from_bias_and_calib(&frame.imu_bias, calib);

        let count = selected.len();
        if count == 2 {
            let dt = (current_timestamp - previous_timestamp) as f32;
            if dt > 0.0 {
                from_last_kf.integrate_new_measurement(selected[0].a, selected[0].w, dt);
                frame_preintegrated.integrate_new_measurement(selected[0].a, selected[0].w, dt);
            }
        } else {
            for index in 0..count - 1 {
                let current = &selected[index];
                let next = &selected[index + 1];
                let interval = (next.t - current.t) as f32;
                let (acceleration, angular_velocity, dt) = if index == 0 {
                    let start = (current.t - previous_timestamp) as f32;
                    let ratio = if interval > 0.0 {
                        start / interval
                    } else {
                        0.0
                    };
                    (
                        (current.a + next.a - (next.a - current.a) * ratio) * 0.5,
                        (current.w + next.w - (next.w - current.w) * ratio) * 0.5,
                        (next.t - previous_timestamp) as f32,
                    )
                } else if index < count - 2 {
                    (
                        (current.a + next.a) * 0.5,
                        (current.w + next.w) * 0.5,
                        interval,
                    )
                } else {
                    let end = (next.t - current_timestamp) as f32;
                    let ratio = if interval > 0.0 { end / interval } else { 0.0 };
                    (
                        (current.a + next.a - (next.a - current.a) * ratio) * 0.5,
                        (current.w + next.w - (next.w - current.w) * ratio) * 0.5,
                        (current_timestamp - current.t) as f32,
                    )
                };
                if dt > 0.0 {
                    from_last_kf.integrate_new_measurement(acceleration, angular_velocity, dt);
                    frame_preintegrated.integrate_new_measurement(
                        acceleration,
                        angular_velocity,
                        dt,
                    );
                }
            }
        }

        frame.imu_preintegrated = Some(Arc::new(from_last_kf.clone()));
        frame.imu_preintegrated_frame = Some(Arc::new(frame_preintegrated));
        self.imu_preintegrated_from_last_kf = Some(Arc::new(from_last_kf));
        frame.set_integrated();
    }

    // Main tracking function. It is independent of the input sensor
    fn track(&mut self) {
        if self
            .local_mapper
            .as_ref()
            .is_some_and(|mapper| mapper.is_bad_imu())
        {
            warn!("Reset requested because LocalMapping reported a bad IMU");
            self.state = TrackingState::Lost;
            return;
        }

        let Some(mut frame) = self.current_frame.take() else {
            return;
        };
        if self.state == TrackingState::NoImagesYet {
            self.state = TrackingState::NotInitialized;
        }
        self.last_processed_state = Some(self.state);

        let active_map = self.atlas.get_current_map();
        let change_index = active_map.get_map_change_index();
        self.map_updated = change_index > active_map.get_last_map_change();
        if self.map_updated {
            active_map.set_last_map_change(change_index);
        }

        let is_inertial = matches!(
            self.sensor,
            Sensor::IMUMonocular | Sensor::IMUStereo | Sensor::IMURGBD
        );
        if is_inertial && self.imu_calib.is_some() {
            self.preintegrate_imu(&mut frame);
        }

        if self.state == TrackingState::NotInitialized {
            match self.sensor {
                Sensor::Stereo | Sensor::IMUStereo | Sensor::RGBD | Sensor::IMURGBD => {
                    self.stereo_initialization(&mut frame)
                }
                Sensor::Monocular | Sensor::IMUMonocular => {
                    self.monocular_initialization(&mut frame)
                }
            }
            if self.state != TrackingState::OK {
                self.current_frame = Some(frame);
                return;
            }
        } else if self.state != TrackingState::Lost {
            self.check_replaced_in_last_frame();
            let mut ok = if self.state == TrackingState::RecentlyLost {
                let lost_time = self.timestamp_lost.unwrap_or(frame.timestamp);
                if frame.timestamp - lost_time > 3.0 {
                    self.state = TrackingState::Lost;
                    false
                } else {
                    self.relocalization(&mut frame)
                }
            } else if self.velocity.is_none() && !self.atlas.is_imu_initialized().unwrap_or(false) {
                self.track_reference_key_frame(&mut frame)
            } else {
                let tracked = self.track_with_motion_model(&mut frame);
                tracked || self.track_reference_key_frame(&mut frame)
            };

            if ok {
                ok = self.track_local_map(&mut frame);
            }
            self.state = if ok {
                TrackingState::OK
            } else {
                TrackingState::RecentlyLost
            };
            if !ok {
                self.timestamp_lost = Some(frame.timestamp);
            }
        } else {
            self.state = TrackingState::Lost;
        }

        if self.state == TrackingState::OK || self.state == TrackingState::RecentlyLost {
            if let Some(last) = &self.last_frame
                && frame.is_set()
            {
                self.velocity = Some(frame.get_pose() * last.get_pose().inverse());
            }
            for i in 0..frame.n {
                if frame.outlier[i] && frame.map_points[i].is_some() {
                    frame.map_points[i] = None;
                }
            }

            let need_keyframe = self.need_new_key_frame(&frame);
            if need_keyframe && self.state == TrackingState::OK {
                self.create_new_key_frame(&mut frame);
            }

            if let Some(reference) = &self.reference_kf
                && frame.is_set()
            {
                frame.reference_kf = Some(reference.clone());
                self.relative_frame_poses
                    .push(frame.get_pose() * reference.get_pose_inverse());
                self.references.push(reference.clone());
                self.frame_times.push(frame.timestamp);
                self.lost.push(self.state == TrackingState::Lost);
            }
        }

        #[cfg(feature = "tracker-pause-resume")]
        if self.stop() {
            while self.is_stopped() {
                std::thread::sleep(std::time::Duration::from_micros(3000));
            }
        }

        self.current_frame = Some(frame);
    }

    pub fn get_matches_inliers(&self) -> usize {
        self.matches_inliers
    }

    pub fn get_image_scale(&self) -> f32 {
        self.image_scale
    }

    pub fn new_dataset(&mut self) {
        self.num_dataset = self.num_dataset.saturating_add(1);
    }

    pub fn get_number_dataset(&self) -> usize {
        self.num_dataset
    }

    pub fn viewer_snapshot(&self) -> ViewerSnapshot {
        let map = self.atlas.get_current_map();
        let map_points = map
            .get_all_map_points()
            .into_iter()
            .filter(|point| !point.is_bad())
            .map(|point| point.get_world_pos())
            .collect::<Vec<_>>();
        let keyframes = map
            .get_all_key_frames()
            .into_iter()
            .filter(|keyframe| !keyframe.is_bad())
            .map(|keyframe| ViewerKeyFrame {
                id: keyframe.id,
                pose: keyframe.get_pose(),
                parent: keyframe.get_parent().map(|parent| parent.id),
                connections: keyframe
                    .get_vector_covisible_keyframes()
                    .into_iter()
                    .map(|connected| connected.id)
                    .collect(),
            })
            .collect::<Vec<_>>();
        let current = self.current_frame.as_ref();
        let features = current
            .map(|frame| {
                let count = frame.n_left.unwrap_or(frame.n);
                (0..count)
                    .filter_map(|index| {
                        let x = if frame.n_left.is_none() {
                            frame.keys_un.as_ref()?.get(index)?.pt().x
                        } else {
                            frame.keys.get(index)?.pt().x
                        };
                        let y = if frame.n_left.is_none() {
                            frame.keys_un.as_ref()?.get(index)?.pt().y
                        } else {
                            frame.keys.get(index)?.pt().y
                        };
                        Some(ViewerFeature {
                            x,
                            y,
                            has_map_point: frame.map_points.get(index).is_some_and(Option::is_some),
                            outlier: frame.outlier.get(index).copied().unwrap_or(false),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        ViewerSnapshot {
            state: self.state,
            image: current
                .and_then(|frame| frame.img_left.as_ref())
                .and_then(ViewerImage::from_mat),
            current_pose: current.map(Frame::get_pose),
            feature_count: current.map_or(0, |frame| frame.n),
            inlier_count: self.matches_inliers,
            map_point_count: map_points.len(),
            map_points,
            keyframes,
            features,
            map_changed: self.map_updated,
            imu_initialized: map.is_imu_initialized(),
        }
    }

    pub fn frame_trajectory(&self) -> Vec<(f64, Isometry3<f32>)> {
        self.relative_frame_poses
            .iter()
            .zip(&self.references)
            .zip(&self.frame_times)
            .map(|((relative, reference), &timestamp)| (timestamp, relative * reference.get_pose()))
            .collect()
    }

    pub fn frame_trajectory_for_export(
        &self,
        map: &Arc<crate::map::Map>,
        imu_body: bool,
        timestamp_scale: f64,
    ) -> Vec<(f64, Isometry3<f32>)> {
        let mut keyframes = map.get_all_key_frames();
        keyframes.sort_by_key(|keyframe| keyframe.id);
        let Some(first) = keyframes.first() else {
            return Vec::new();
        };
        let origin = if imu_body {
            first.get_imu_pose()
        } else {
            first.get_pose_inverse()
        };

        self.relative_frame_poses
            .iter()
            .zip(&self.references)
            .zip(&self.frame_times)
            .zip(&self.lost)
            .filter_map(|(((relative, reference), &timestamp), &lost)| {
                if lost {
                    return None;
                }
                let (resolved_pose, tbc) = self.resolve_reference_pose(reference, map)?;
                let trw = resolved_pose * origin;
                let pose = if imu_body {
                    (tbc * *relative * trw).inverse()
                } else {
                    (*relative * trw).inverse()
                };
                Some((timestamp * timestamp_scale, pose))
            })
            .collect()
    }

    pub fn keyframe_trajectory(&self) -> Vec<(f64, Isometry3<f32>)> {
        let mut trajectory = self
            .atlas
            .get_current_map()
            .get_all_key_frames()
            .into_iter()
            .filter(|keyframe| !keyframe.is_bad())
            .map(|keyframe| (keyframe.timestamp, keyframe.get_pose()))
            .collect::<Vec<_>>();
        trajectory.sort_by(|left, right| left.0.total_cmp(&right.0));
        trajectory
    }

    pub fn keyframe_trajectory_for_export(
        &self,
        map: &Arc<crate::map::Map>,
        imu_body: bool,
        timestamp_scale: f64,
        normalize_origin: bool,
    ) -> Vec<(f64, Isometry3<f32>)> {
        export_keyframe_trajectory(map, imu_body, timestamp_scale, normalize_origin)
    }

    fn resolve_reference_pose(
        &self,
        reference: &Arc<KeyFrame>,
        map: &Arc<crate::map::Map>,
    ) -> Option<(Isometry3<f32>, Isometry3<f32>)> {
        let mut keyframe = reference.clone();
        let mut correction = Isometry3::identity();
        let mut guard = 0;
        while keyframe.is_bad() {
            correction *= keyframe
                .get_pose_to_parent()
                .unwrap_or(Isometry3::identity());
            keyframe = keyframe.get_parent()?;
            guard += 1;
            if guard > 10_000 {
                return None;
            }
        }
        if keyframe.get_map().map(|owner| owner.get_id()) != Some(map.get_id()) {
            return None;
        }
        Some((correction * keyframe.get_pose(), keyframe.imu_calib.tbc))
    }

    pub fn set_local_mapper(&mut self, local_mapper: Arc<LocalMapping>) {
        self.local_mapper = Some(local_mapper);
    }

    pub fn set_loop_closing(&mut self, loop_closing: Arc<LoopClosing>) {
        self.loop_closing = Some(loop_closing);
    }

    pub fn grab_imu_data(&mut self, measurement: Point) {
        self.queue_imu_data.push(measurement);
    }

    pub fn get_tracking_state(&self) -> TrackingState {
        self.state
    }

    pub fn is_lost(&self) -> bool {
        self.state == TrackingState::Lost
    }

    pub fn reset_active_map(&mut self, map: &Arc<crate::map::Map>) {
        mark_lost_for_map(&mut self.lost, &self.references, map.get_id());

        let next_frame_id = self
            .current_frame
            .as_ref()
            .map_or(self.init_id + 1, |frame| frame.id.saturating_add(1));
        self.reset_runtime_state(next_frame_id);
    }

    /// Full reset (upstream `Tracking::Reset`): discard all trajectory history
    /// and restart identifiers/state.
    pub fn reset(&mut self) {
        let next_frame_id = self
            .current_frame
            .as_ref()
            .map_or(0, |frame| frame.id.saturating_add(1));
        self.relative_frame_poses.clear();
        self.references.clear();
        self.frame_times.clear();
        self.lost.clear();
        self.reset_runtime_state(next_frame_id);
        self.init_id = 0;
        self.first_frame_id = 0;
        self.initial_frame_id = 0;
        self.last_reloc_frame_id = 0;
    }

    fn reset_runtime_state(&mut self, next_frame_id: usize) {
        self.state = TrackingState::NoImagesYet;
        self.last_processed_state = None;
        self.ready_to_initialize = false;
        self.created_map = false;
        self.current_frame = None;
        self.last_frame = None;
        self.initial_frame = None;
        self.reference_kf = None;
        self.last_key_frame = None;
        self.last_key_frame_id = None;
        self.last_init_frame_id = None;
        self.first_imu_frame_id = None;
        self.initial_frame_id = next_frame_id;
        self.last_reloc_frame_id = next_frame_id;
        self.init_id = next_frame_id;
        self.velocity = None;
        self.last_bias = None;
        self.matches_inliers = 0;
        self.ini_matches.clear();
        self.prev_matched.clear();
        self.ini_p3d.clear();
        self.queue_imu_data.clear();
        self.imu_from_last_frame.clear();
        self.local_key_frames.clear();
        self.local_map_points.clear();
        self.t0 = 0.0;
        self.t0vis = 0.0;
        self.t0imu = 0.0;
        self.timestamp_lost = None;
    }

    pub fn update_frame_imu(&mut self, scale: f32, bias: Bias, current_keyframe: Arc<KeyFrame>) {
        for relative in &mut self.relative_frame_poses {
            relative.translation.vector *= scale;
        }
        self.last_bias = Some(bias);
        self.last_key_frame = Some(current_keyframe.clone());
        self.reference_kf = Some(current_keyframe.clone());
        if let Some(frame) = self.current_frame.as_mut() {
            frame.set_new_bias(bias);
            frame.last_keyframe = Some(current_keyframe);
            frame.reference_kf = Some(self.reference_kf.clone().expect("reference set above"));
        }
        self.state = TrackingState::OK;
    }

    #[cfg(feature = "register-times")]
    pub fn local_map_stats_to_file(&self) {
        if std::fs::write(
            "LocalMapTimes.txt",
            "Local mapping timing is recorded by the LocalMapping worker.\n",
        )
        .is_err()
        {
            warn!("Failed to write LocalMapTimes.txt");
        }
    }
    #[cfg(feature = "register-times")]
    pub fn track_stats_to_file(&self) {
        let mut output = String::from("category,mean_ms,min_ms,max_ms\n");
        for (name, values) in [
            ("orb_extract", &self.orb_extract_ms),
            ("stereo_match", &self.stereo_match_ms),
            ("imu_integrate", &self.imu_integ_ms),
            ("pose_prediction", &self.pose_pred_ms),
            ("local_map", &self.lm_track_ms),
            ("new_keyframe", &self.new_kf_ms),
            ("total", &self.track_total_ms),
        ] {
            if values.is_empty() {
                continue;
            }
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            output.push_str(&format!("{name},{mean:.6},{min:.6},{max:.6}\n"));
        }
        if std::fs::write("TrackTimes.txt", output).is_err() {
            warn!("Failed to write TrackTimes.txt");
        }
    }
    #[cfg(feature = "register-times")]
    pub fn print_time_stats(&self) {
        let count = self.track_total_ms.len();
        if count > 0 {
            let total = self.track_total_ms.iter().sum::<f64>();
            info!(
                "Tracking: {} frames, mean {:.3} ms, total {:.3} s",
                count,
                total / count as f64,
                total / 1000.0
            );
        }
    }

    #[cfg(feature = "tracker-pause-resume")]
    pub fn request_stop(&self) {
        self.stop_requested.store(true, Ordering::SeqCst);
    }
    #[cfg(feature = "tracker-pause-resume")]
    pub fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::SeqCst)
    }
    #[cfg(feature = "tracker-pause-resume")]
    pub fn release(&self) {
        self.stop_requested.store(false, Ordering::SeqCst);
        self.stopped.store(false, Ordering::SeqCst);
    }
    #[cfg(feature = "tracker-pause-resume")]
    pub fn step(&self) {
        // Release only this wait. `stop_requested` remains set, so the next
        // completed frame immediately re-enters the paused state.
        self.stopped.store(false, Ordering::SeqCst);
    }
    #[cfg(feature = "tracker-pause-resume")]
    pub fn stop_requested(&self) -> bool {
        self.stop_requested.load(Ordering::SeqCst)
    }
    #[cfg(feature = "tracker-pause-resume")]
    fn stop(&self) -> bool {
        if self.stop_requested.load(Ordering::SeqCst) {
            self.stopped.store(true, Ordering::SeqCst);
            true
        } else {
            false
        }
    }
}

fn pnp_relocalization_pose(frame: &Frame) -> Option<(Isometry3<f32>, Vec<usize>)> {
    let rig = frame.camera2.as_ref().zip(frame.n_left);
    let mut groups: [Vec<(usize, PnpCorrespondence)>; 2] = [Vec::new(), Vec::new()];
    for (idx, map_point) in frame.map_points.iter().enumerate() {
        let Some(map_point) = map_point else {
            continue;
        };
        let (camera, keypoint, cam_idx) = if let Some((camera2, n_left)) = rig {
            if idx < n_left {
                let Some(keypoint) = frame.keys.get(idx) else {
                    continue;
                };
                (frame.camera.as_ref(), keypoint, 0)
            } else {
                let Some(keypoint) = frame
                    .keys_right
                    .as_ref()
                    .and_then(|keys| keys.get(idx - n_left))
                else {
                    continue;
                };
                (camera2.as_ref(), keypoint, 1)
            }
        } else {
            let Some(keypoint) = frame.keys_un.as_ref().unwrap_or(&frame.keys).get(idx) else {
                continue;
            };
            (frame.camera.as_ref(), keypoint, 0)
        };
        let ray = camera.unproject_n(&keypoint.pt());
        if ray.z.abs() < 1e-6 {
            continue;
        }
        let world = map_point.get_world_pos();
        groups[cam_idx].push((
            idx,
            PnpCorrespondence {
                world: opencv::core::Point3f::new(world.x, world.y, world.z),
                bearing: opencv::core::Point2f::new(ray.x / ray.z, ray.y / ray.z),
            },
        ));
    }

    let mut groups: Vec<_> = groups.into_iter().enumerate().collect();
    groups.sort_by_key(|(_, group)| std::cmp::Reverse(group.len()));
    for (cam_idx, group) in groups {
        let correspondences: Vec<_> = group
            .iter()
            .map(|(_, correspondence)| *correspondence)
            .collect();
        let Some(solution) = solve_pnp_ransac(&correspondences) else {
            continue;
        };
        let tcw = if cam_idx == 1 {
            frame.get_relative_pose_trl().inverse() * solution.tcw
        } else {
            solution.tcw
        };
        let indices = solution
            .inliers
            .into_iter()
            .filter_map(|pnp_idx| group.get(pnp_idx).map(|(idx, _)| *idx))
            .collect();
        return Some((tcw, indices));
    }
    None
}

fn mark_lost_for_map(lost: &mut [bool], references: &[Arc<KeyFrame>], map_id: u32) {
    for (lost, reference) in lost.iter_mut().zip(references) {
        if reference.origin_map_id == map_id {
            *lost = true;
        }
    }
}

fn export_keyframe_trajectory(
    map: &crate::map::Map,
    imu_body: bool,
    timestamp_scale: f64,
    normalize_origin: bool,
) -> Vec<(f64, Isometry3<f32>)> {
    let mut keyframes = map.get_all_key_frames();
    keyframes.sort_by_key(|keyframe| keyframe.id);
    let origin = keyframes.first().map(|first| {
        if normalize_origin {
            if imu_body {
                first.get_imu_pose()
            } else {
                first.get_pose_inverse()
            }
        } else {
            Isometry3::identity()
        }
    });
    let Some(origin) = origin else {
        return Vec::new();
    };
    keyframes
        .into_iter()
        .filter(|keyframe| !keyframe.is_bad())
        .map(|keyframe| {
            let pose = if imu_body {
                (keyframe.imu_calib.tbc * keyframe.get_pose() * origin).inverse()
            } else {
                (keyframe.get_pose() * origin).inverse()
            };
            (keyframe.timestamp * timestamp_scale, pose)
        })
        .collect()
}

fn imu_calibration_from_settings(imu: &IMUInfo) -> Calib {
    let scale = imu.imu_frequency.sqrt();
    Calib::from_params(
        imu.tbc,
        imu.noise_gyro * scale,
        imu.noise_acc * scale,
        imu.gyro_walk / scale,
        imu.acc_walk / scale,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;

    use crate::key_frame::KeyFrame;
    use crate::test_helpers::{build_frame, keypoint, make_pose};

    #[test]
    fn imu_calibration_matches_orbslam_noise_scaling() {
        let imu = IMUInfo {
            noise_gyro: 0.01,
            noise_acc: 0.02,
            gyro_walk: 0.03,
            acc_walk: 0.04,
            imu_frequency: 400.0,
            tbc: Isometry3::identity(),
            insert_kfs_when_lost: true,
        };
        let calibration = imu_calibration_from_settings(&imu);
        let noise = calibration.get_cov();
        let walk = calibration.get_cov_walk();
        assert!((noise[(0, 0)] - (0.01_f32 * 20.0).powi(2)).abs() < 1e-8);
        assert!((noise[(3, 3)] - (0.02_f32 * 20.0).powi(2)).abs() < 1e-8);
        assert!((walk[(0, 0)] - (0.03_f32 / 20.0).powi(2)).abs() < 1e-10);
        assert!((walk[(3, 3)] - (0.04_f32 / 20.0).powi(2)).abs() < 1e-10);
    }

    #[test]
    fn keyframe_export_keeps_world_pose_and_only_normalizes_when_requested() {
        let map = Arc::new(crate::map::Map::new());
        let first = KeyFrame::from_frame(&build_frame(), map.clone());
        first.set_pose(make_pose());
        map.add_key_frame(first.clone());

        let direct = export_keyframe_trajectory(&map, false, 1.0, false);
        assert_eq!(direct.len(), 1);
        assert!((direct[0].1.to_matrix() - first.get_pose_inverse().to_matrix()).norm() < 1e-6);

        let normalized = export_keyframe_trajectory(&map, false, 1.0, true);
        assert_eq!(normalized.len(), 1);
        assert!((normalized[0].1.to_matrix() - Isometry3::identity().to_matrix()).norm() < 1e-6);
    }

    #[test]
    fn relocalization_pnp_recovers_pose_from_frame_bearings() {
        let map = Arc::new(crate::map::Map::new());
        let mut frame = build_frame();
        let expected = make_pose();
        let world_points = [
            Point3::new(-1.0, -0.5, 4.0),
            Point3::new(1.0, -0.5, 4.0),
            Point3::new(1.0, 0.5, 4.0),
            Point3::new(-1.0, 0.5, 4.0),
            Point3::new(-0.8, -0.4, 5.0),
            Point3::new(0.8, -0.4, 5.0),
            Point3::new(0.8, 0.4, 5.0),
            Point3::new(-0.8, 0.4, 5.0),
        ];
        let mut keys = Vec::new();
        let mut map_points = Vec::new();
        for world in world_points {
            let camera_point = expected * Point3::new(world.x, world.y, world.z);
            let uv = frame.camera.project_n_d(&camera_point.cast::<f64>());
            keys.push(keypoint(uv.x as f32, uv.y as f32));
            let point = MapPoint::from_frame(
                &Vector3::new(world.x, world.y, world.z),
                map.clone(),
                &frame,
                0,
            );
            point.set_world_pos(Vector3::new(world.x, world.y, world.z));
            map_points.push(Some(point));
        }
        frame.n = keys.len();
        frame.keys = keys.clone();
        frame.keys_un = Some(keys);
        frame.map_points = map_points;

        let (solved, inliers) = pnp_relocalization_pose(&frame).expect("PnP pose");
        let delta = solved.inverse() * expected;
        assert!(delta.rotation.angle().abs() < 1e-3);
        assert!(delta.translation.vector.norm() < 1e-3);
        assert_eq!(inliers.len(), frame.n);
    }

    #[test]
    fn reset_marks_only_active_map_trajectory_entries_lost() {
        let map_a = Arc::new(crate::map::Map::new());
        let map_b = Arc::new(crate::map::Map::new());
        let reference_a = KeyFrame::from_frame(&build_frame(), map_a.clone());
        let reference_b = KeyFrame::from_frame(&build_frame(), map_b.clone());
        let references = [reference_a, reference_b];
        let mut lost = [false, false];

        map_a.clear();
        mark_lost_for_map(&mut lost, &references, map_a.get_id());

        assert_eq!(lost, [true, false]);
    }

    #[test]
    fn relocalization_pnp_uses_rig_right_camera_observations() {
        let map = Arc::new(crate::map::Map::new());
        let mut frame = build_frame();
        let expected = make_pose();
        let camera2: Arc<dyn crate::camera_models::GeometricCamera> =
            Arc::new(crate::camera_models::pinhole::Pinhole::with_params(vec![
                458.654, 457.296, 367.215, 248.375,
            ]));
        frame.camera2 = Some(camera2.clone());
        frame.n_left = Some(0);
        frame.n_right = Some(8);
        frame.keys = Vec::new();
        frame.keys_un = None;
        let world_points = [
            Point3::new(-1.0, -0.5, 4.0),
            Point3::new(1.0, -0.5, 4.0),
            Point3::new(1.0, 0.5, 4.0),
            Point3::new(-1.0, 0.5, 4.0),
            Point3::new(-0.8, -0.4, 5.0),
            Point3::new(0.8, -0.4, 5.0),
            Point3::new(0.8, 0.4, 5.0),
            Point3::new(-0.8, 0.4, 5.0),
        ];
        let mut keys_right = Vec::new();
        for &world in &world_points {
            let camera_point = expected * Point3::new(world.x, world.y, world.z);
            let uv = camera2.project_n_d(&camera_point.cast::<f64>());
            keys_right.push(keypoint(uv.x as f32, uv.y as f32));
        }
        frame.keys_right = Some(keys_right);
        let mut map_points = Vec::new();
        for world in world_points {
            let point = MapPoint::from_frame(
                &Vector3::new(world.x, world.y, world.z),
                map.clone(),
                &frame,
                0,
            );
            point.set_world_pos(Vector3::new(world.x, world.y, world.z));
            map_points.push(Some(point));
        }
        frame.n = 8;
        frame.u_right = vec![-1.0; 8];
        frame.map_points = map_points;

        let (solved, inliers) = pnp_relocalization_pose(&frame).expect("right-camera PnP pose");
        let delta = solved.inverse() * expected;
        assert!(delta.rotation.angle().abs() < 1e-3);
        assert!(delta.translation.vector.norm() < 1e-3);
        assert_eq!(inliers.len(), frame.n);
    }
}
