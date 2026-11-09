use md5::{Digest, Md5};
use opencv::core::Mat;
use opencv::imgproc::{INTER_LINEAR, remap, resize};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicBool, Ordering},
};
use std::thread::JoinHandle;
use std::{
    fs::File,
    io::{self, Read, Write},
    path::Path,
};
use tracing::info;

use crate::{
    atlas::Atlas,
    imu_types::Point,
    key_frame_database::{KeyFrameDatabase, MapId},
    local_mapping::LocalMapping,
    loop_closing::{AtlasKeyFrameView, LoopClosing},
    orb_vocabulary::{OrbVocabulary, VocabularyError},
    settings::{Settings, SettingsError},
    tracking::{Tracking, TrackingState},
    viewer::{Viewer, ViewerSnapshot},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sensor {
    Monocular = 0,
    Stereo = 1,
    RGBD = 2,
    IMUMonocular = 3,
    IMUStereo = 4,
    IMURGBD = 5,
}

pub struct System {
    sensor: Sensor,
    settings: Option<Arc<Settings>>,
    atlas: Arc<Atlas>,
    vocabulary: Option<Arc<OrbVocabulary>>,
    keyframe_database: Option<Arc<KeyFrameDatabase>>,
    tracking: OnceLock<Arc<Mutex<Tracking>>>,
    local_mapping: OnceLock<Arc<LocalMapping>>,
    loop_closing: OnceLock<Arc<LoopClosing>>,
    #[cfg(feature = "viewer")]
    viewer: Option<Arc<Viewer>>,
    reset: AtomicBool,
    reset_active_map: AtomicBool,
    activate_localization_mode: AtomicBool,
    deactivate_localization_mode: AtomicBool,
    shutdown: AtomicBool,
    threads: Mutex<Vec<JoinHandle<()>>>,
    vocabulary_file_path: PathBuf,
    load_atlas_file_path: Option<String>,
    save_atlas_file_path: Option<String>,
}

#[derive(Debug)]
pub enum SystemError {
    InvalidSettings(SettingsError),
    InvalidORBVocabulary(VocabularyError),
    AtlasLoad(AtlasLoadError),
}

impl std::fmt::Display for SystemError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SystemError::InvalidSettings(error) => write!(formatter, "invalid settings: {error:?}"),
            SystemError::InvalidORBVocabulary(error) => {
                write!(formatter, "invalid ORB vocabulary: {error:?}")
            }
            SystemError::AtlasLoad(error) => write!(formatter, "failed to load Atlas: {error:?}"),
        }
    }
}

impl std::error::Error for SystemError {}

impl System {
    /// Builds the SLAM runtime and starts the mapping/loop-closing workers.
    pub fn new(
        vocabulary_path: &PathBuf,
        settings_path: &PathBuf,
        sensor: Sensor,
        use_viewer: bool,
        sequence: Option<String>,
    ) -> Result<Arc<Self>, SystemError> {
        info!("orb-slam3-rs");
        info!(
            "ORB-SLAM3 Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza."
        );
        info!(
            "ORB-SLAM2 Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza."
        );
        info!(
            "This program comes with ABSOLUTELY NO WARRANTY; This is free software, and you are welcome to redistribute it under certain conditions. See LICENSE."
        );
        info!("Input sensor was set to: {:?}", sensor);

        // Load settings
        let settings =
            Arc::new(Settings::new(settings_path, sensor).map_err(SystemError::InvalidSettings)?);

        // Load ORB VOB vocabulary
        info!("Loading ORB vocabulary. This could take a while...");
        let vocabulary = OrbVocabulary::load_from_text_file(vocabulary_path)
            .map_err(SystemError::InvalidORBVocabulary)?;
        info!("ORB vocabulary loaded!");
        let vocabulary = Arc::new(vocabulary);
        // Create keyframe database
        let keyframe_database = Arc::new(KeyFrameDatabase::new(vocabulary.clone()));

        let atlas = if let Some(load_path) = &settings.load_and_save.load_from {
            info!("Initialization of Atlas from file: {}", load_path);
            let ret = load_atlas(
                load_path,
                vocabulary_path,
                keyframe_database.clone(),
                vocabulary.clone(),
            )
            .map_err(SystemError::AtlasLoad)?;
            ret.create_new_map();
            ret
        } else {
            info!("Initialization of Atlas from scratch");
            Atlas::from_kf_id(0)
        };

        if matches!(
            sensor,
            Sensor::IMUStereo | Sensor::IMUMonocular | Sensor::IMURGBD
        ) {
            atlas.set_inertial_sensor();
        }

        let atlas = Arc::new(atlas);
        let system = Arc::new(System {
            sensor,
            settings: Some(settings.clone()),
            atlas: atlas.clone(),
            vocabulary: Some(vocabulary.clone()),
            keyframe_database: Some(keyframe_database.clone()),
            tracking: OnceLock::new(),
            local_mapping: OnceLock::new(),
            loop_closing: OnceLock::new(),
            viewer: None,
            reset: AtomicBool::new(false),
            reset_active_map: AtomicBool::new(false),
            activate_localization_mode: AtomicBool::new(false),
            deactivate_localization_mode: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            threads: Mutex::new(Vec::new()),
            vocabulary_file_path: vocabulary_path.clone(),
            load_atlas_file_path: settings.load_and_save.load_from.clone(),
            save_atlas_file_path: settings.load_and_save.save_to.clone(),
        });

        let tracking = Arc::new(Mutex::new(Tracking::new(
            system.clone(),
            vocabulary,
            atlas.clone(),
            keyframe_database,
            sensor,
            settings.clone(),
        )));
        let monocular = matches!(sensor, Sensor::Monocular | Sensor::IMUMonocular);
        let inertial = matches!(
            sensor,
            Sensor::IMUMonocular | Sensor::IMUStereo | Sensor::IMURGBD
        );
        let sequence = sequence.unwrap_or_else(|| {
            settings
                .camera1
                .as_ref()
                .map(|_| String::new())
                .unwrap_or_default()
        });
        let local_mapping = Arc::new(LocalMapping::new(
            system.clone(),
            atlas.clone(),
            monocular,
            inertial,
            sequence.clone(),
        ));
        let loop_closing = Arc::new(LoopClosing::new(
            atlas.clone(),
            system
                .keyframe_database
                .as_ref()
                .expect("database initialized")
                .clone(),
            system
                .vocabulary
                .as_ref()
                .expect("vocabulary initialized")
                .clone(),
            !monocular,
            settings.loop_closing,
        ));

        {
            let mut tracker = tracking.lock().unwrap();
            tracker.set_local_mapper(local_mapping.clone());
            tracker.set_loop_closing(loop_closing.clone());
        }
        local_mapping.set_tracker(tracking.clone());
        local_mapping.set_loop_closer(loop_closing.clone());
        loop_closing.set_local_mapper(local_mapping.clone());
        let _ = system.tracking.set(tracking);
        let _ = system.local_mapping.set(local_mapping.clone());
        let _ = system.loop_closing.set(loop_closing.clone());

        if use_viewer {
            info!("GUI viewer requested; snapshots will be attached by the frontend");
        }
        if !sequence.is_empty() {
            info!("Seq. Name: {}", sequence);
        }

        let mapping_thread = local_mapping.clone();
        let mapping_handle = std::thread::Builder::new()
            .name("orb-local-mapping".to_owned())
            .spawn(move || mapping_thread.run())
            .map_err(|_| SystemError::InvalidSettings(SettingsError::InvalidFile))?;
        let loop_thread = loop_closing;
        let loop_handle = std::thread::Builder::new()
            .name("orb-loop-closing".to_owned())
            .spawn(move || loop_thread.run())
            .map_err(|_| SystemError::InvalidSettings(SettingsError::InvalidFile))?;
        system
            .threads
            .lock()
            .unwrap()
            .extend([mapping_handle, loop_handle]);

        Ok(system)
    }

    pub fn track_stereo(
        &self,
        image_left: Mat,
        image_right: Mat,
        timestamp: f64,
        imu_measurements: &[Point],
        filename: impl Into<String>,
    ) -> Option<nalgebra::Isometry3<f32>> {
        if self.shutdown.load(Ordering::SeqCst) {
            return None;
        }
        let (image_left, image_right) = self.preprocess_stereo(image_left, image_right)?;
        let tracking = self.tracking.get()?.clone();
        let mut tracking = tracking.lock().unwrap();
        for measurement in imu_measurements {
            tracking.grab_imu_data(Point::new(measurement.a, measurement.w, measurement.t));
        }
        tracking.grab_image_stereo(image_left, image_right, timestamp, filename.into())
    }

    pub fn track_monocular(
        &self,
        image: Mat,
        timestamp: f64,
        imu_measurements: &[Point],
        filename: impl Into<String>,
    ) -> Option<nalgebra::Isometry3<f32>> {
        if self.shutdown.load(Ordering::SeqCst) {
            return None;
        }
        let image = self.preprocess_image(image)?;
        let tracking = self.tracking.get()?.clone();
        let mut tracking = tracking.lock().unwrap();
        for measurement in imu_measurements {
            tracking.grab_imu_data(Point::new(measurement.a, measurement.w, measurement.t));
        }
        tracking.grab_image_monocular(image, timestamp, filename.into())
    }

    pub fn track_rgbd(
        &self,
        image: Mat,
        depth: Mat,
        timestamp: f64,
        imu_measurements: &[Point],
        filename: impl Into<String>,
    ) -> Option<nalgebra::Isometry3<f32>> {
        if self.shutdown.load(Ordering::SeqCst) {
            return None;
        }
        let (image, depth) = self.preprocess_rgbd(image, depth)?;
        let tracking = self.tracking.get()?.clone();
        let mut tracking = tracking.lock().unwrap();
        for measurement in imu_measurements {
            tracking.grab_imu_data(Point::new(measurement.a, measurement.w, measurement.t));
        }
        tracking.grab_image_rgbd(image, depth, timestamp, filename.into())
    }

    fn preprocess_image(&self, image: Mat) -> Option<Mat> {
        let Some(settings) = self.settings.as_ref() else {
            return Some(image);
        };
        if !settings.image.need_to_resize {
            return Some(image);
        }
        let mut resized = Mat::default();
        resize(
            &image,
            &mut resized,
            settings.image.new_size,
            0.0,
            0.0,
            INTER_LINEAR,
        )
        .ok()?;
        Some(resized)
    }

    fn preprocess_rgbd(&self, image: Mat, depth: Mat) -> Option<(Mat, Mat)> {
        let Some(settings) = self.settings.as_ref() else {
            return Some((image, depth));
        };
        if !settings.image.need_to_resize {
            return Some((image, depth));
        }
        let mut resized_image = Mat::default();
        let mut resized_depth = Mat::default();
        resize(
            &image,
            &mut resized_image,
            settings.image.new_size,
            0.0,
            0.0,
            INTER_LINEAR,
        )
        .ok()?;
        resize(
            &depth,
            &mut resized_depth,
            settings.image.new_size,
            0.0,
            0.0,
            INTER_LINEAR,
        )
        .ok()?;
        Some((resized_image, resized_depth))
    }

    fn preprocess_stereo(&self, image_left: Mat, image_right: Mat) -> Option<(Mat, Mat)> {
        let Some(settings) = self.settings.as_ref() else {
            return Some((image_left, image_right));
        };
        let Some(stereo) = settings.stereo.as_ref() else {
            return Some((image_left, image_right));
        };
        let mut left = Mat::default();
        let mut right = Mat::default();
        if settings.need_to_rectify {
            remap(
                &image_left,
                &mut left,
                &stereo.m1l,
                &stereo.m2l,
                INTER_LINEAR,
                opencv::core::BORDER_CONSTANT,
                opencv::core::Scalar::default(),
            )
            .ok()?;
            remap(
                &image_right,
                &mut right,
                &stereo.m1r,
                &stereo.m2r,
                INTER_LINEAR,
                opencv::core::BORDER_CONSTANT,
                opencv::core::Scalar::default(),
            )
            .ok()?;
            Some((left, right))
        } else if settings.image.need_to_resize {
            resize(
                &image_left,
                &mut left,
                settings.image.new_size,
                0.0,
                0.0,
                INTER_LINEAR,
            )
            .ok()?;
            resize(
                &image_right,
                &mut right,
                settings.image.new_size,
                0.0,
                0.0,
                INTER_LINEAR,
            )
            .ok()?;
            Some((left, right))
        } else {
            Some((image_left, image_right))
        }
    }

    pub fn grab_imu_data(&self, measurement: Point) {
        if let Some(tracking) = self.tracking.get()
            && let Ok(mut tracking) = tracking.lock()
        {
            tracking.grab_imu_data(measurement);
        }
    }

    pub fn tracking_state(&self) -> Option<TrackingState> {
        self.tracking.get().and_then(|tracking| {
            tracking
                .lock()
                .ok()
                .map(|tracking| tracking.get_tracking_state())
        })
    }

    pub fn is_lost(&self) -> bool {
        self.tracking
            .get()
            .and_then(|tracking| tracking.lock().ok().map(|tracking| tracking.is_lost()))
            .unwrap_or(true)
    }

    pub fn get_atlas(&self) -> Arc<Atlas> {
        self.atlas.clone()
    }

    pub fn get_tracking(&self) -> Option<Arc<Mutex<Tracking>>> {
        self.tracking.get().cloned()
    }

    pub fn get_local_mapping(&self) -> Option<Arc<LocalMapping>> {
        self.local_mapping.get().cloned()
    }

    pub fn viewer_snapshot(&self) -> Option<ViewerSnapshot> {
        self.tracking.get().and_then(|tracking| {
            tracking
                .lock()
                .ok()
                .map(|tracking| tracking.viewer_snapshot())
        })
    }

    pub fn activate_localization_mode(&self) {
        if let Some(mapping) = self.local_mapping.get() {
            mapping.request_stop();
        }
        if let Some(tracking) = self.tracking.get()
            && let Ok(mut tracking) = tracking.lock()
        {
            tracking.only_tracking = true;
        }
        self.activate_localization_mode
            .store(false, Ordering::SeqCst);
    }

    #[cfg(feature = "tracker-pause-resume")]
    pub fn pause_tracking(&self) {
        if let Some(tracking) = self.tracking.get() {
            tracking.lock().unwrap().request_stop();
        }
    }

    #[cfg(feature = "tracker-pause-resume")]
    pub fn resume_tracking(&self) {
        if let Some(tracking) = self.tracking.get() {
            tracking.lock().unwrap().release();
        }
    }

    pub fn deactivate_localization_mode(&self) {
        if let Some(mapping) = self.local_mapping.get() {
            mapping.release();
        }
        if let Some(tracking) = self.tracking.get()
            && let Ok(mut tracking) = tracking.lock()
        {
            tracking.only_tracking = false;
        }
        self.deactivate_localization_mode
            .store(false, Ordering::SeqCst);
    }

    pub fn reset_active_map(&self) {
        let current_map = self.atlas.get_current_map();
        if let Some(mapping) = self.local_mapping.get() {
            mapping.request_reset_active_map(current_map.clone());
        }
        if let Some(loop_closing) = self.loop_closing.get() {
            loop_closing.request_reset_active_map(current_map.clone());
        }
        if let Some(database) = self.keyframe_database.as_ref() {
            database.clear_map(
                &AtlasKeyFrameView::from_atlas(&self.atlas),
                MapId(current_map.get_id() as u64),
            );
        }
        self.atlas.clear_map();
        if let Some(tracking) = self.tracking.get()
            && let Ok(mut tracking) = tracking.lock()
        {
            tracking.reset_active_map(&current_map);
        }
        self.reset_active_map.store(false, Ordering::SeqCst);
    }

    pub fn shutdown(&self) {
        if self.shutdown.swap(true, Ordering::SeqCst) {
            return;
        }
        if let Some(mapping) = self.local_mapping.get() {
            mapping.request_finish();
            mapping.interrupt_ba();
        }
        if let Some(loop_closing) = self.loop_closing.get() {
            loop_closing.request_finish();
        }
        let handles = std::mem::take(&mut *self.threads.lock().unwrap());
        for handle in handles {
            if let Err(error) = handle.join() {
                tracing::error!("SLAM worker thread panicked: {:?}", error);
            }
        }
    }

    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::SeqCst)
    }

    pub fn save_trajectory_tum(&self, path: impl AsRef<Path>) -> io::Result<()> {
        if self.sensor == Sensor::Monocular {
            return Err(unsupported_trajectory_format("TUM", self.sensor));
        }
        let Some(map) = self.largest_map() else {
            return write_tum_trajectory(path, &[]);
        };
        let trajectory = self
            .tracking
            .get()
            .map(|tracking| {
                tracking
                    .lock()
                    .unwrap()
                    .frame_trajectory_for_export(&map, false, 1.0)
            })
            .unwrap_or_default();
        write_tum_trajectory(path, &trajectory)
    }

    pub fn save_keyframe_trajectory_tum(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let Some(map) = self.largest_map() else {
            return write_tum_trajectory(path, &[]);
        };
        let trajectory = self
            .tracking
            .get()
            .map(|tracking| {
                tracking
                    .lock()
                    .unwrap()
                    .keyframe_trajectory_for_export(&map, false, 1.0, false)
            })
            .unwrap_or_default();
        write_tum_trajectory(path, &trajectory)
    }

    pub fn save_trajectory_euroc(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let Some(map) = self.largest_map() else {
            return write_tum_trajectory(path, &[]);
        };
        let trajectory = self
            .tracking
            .get()
            .map(|tracking| {
                tracking.lock().unwrap().frame_trajectory_for_export(
                    &map,
                    self.uses_imu_body_trajectory(),
                    1e9,
                )
            })
            .unwrap_or_default();
        write_tum_trajectory(path, &trajectory)
    }

    pub fn save_keyframe_trajectory_euroc(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let Some(map) = self.largest_map() else {
            return write_tum_trajectory(path, &[]);
        };
        let trajectory = self
            .tracking
            .get()
            .map(|tracking| {
                tracking.lock().unwrap().keyframe_trajectory_for_export(
                    &map,
                    self.uses_imu_body_trajectory(),
                    1e9,
                    false,
                )
            })
            .unwrap_or_default();
        write_tum_trajectory(path, &trajectory)
    }

    pub fn save_trajectory_kitti(&self, path: impl AsRef<Path>) -> io::Result<()> {
        if self.sensor == Sensor::Monocular {
            return Err(unsupported_trajectory_format("KITTI", self.sensor));
        }
        let Some(map) = self.largest_map() else {
            return Ok(());
        };
        let trajectory = self
            .tracking
            .get()
            .map(|tracking| {
                tracking
                    .lock()
                    .unwrap()
                    .frame_trajectory_for_export(&map, false, 1.0)
            })
            .unwrap_or_default();
        let mut file = File::create(path)?;
        for (_, pose) in trajectory {
            let r = pose.rotation.to_rotation_matrix().into_inner();
            let t = pose.translation.vector;
            writeln!(
                file,
                "{:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e}",
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
            )?;
        }
        Ok(())
    }

    fn largest_map(&self) -> Option<Arc<crate::map::Map>> {
        self.atlas
            .get_all_maps()
            .into_iter()
            .max_by_key(|map| map.key_frames_in_map())
    }

    fn uses_imu_body_trajectory(&self) -> bool {
        matches!(
            self.sensor,
            Sensor::IMUMonocular | Sensor::IMUStereo | Sensor::IMURGBD
        )
    }

    pub fn save_atlas(&self, path: impl AsRef<Path>) -> Result<(), SystemSaveError> {
        let path = path.as_ref();
        let checksum = calculate_checksum(&self.vocabulary_file_path)?;
        let snapshot = SessionSnapshotRef {
            file_voc: self.vocabulary_file_path.to_string_lossy().into_owned(),
            file_voc_checksum: checksum,
            atlas: &self.atlas,
        };
        let bytes = postcard::to_allocvec(&snapshot)?;
        fs::write(format!("{}.postcard", path.display()), bytes)?;
        Ok(())
    }

    /// Minimal system handle for unit tests that exercise another component's
    /// wiring without loading the ORB vocabulary or settings files.
    #[cfg(test)]
    pub(crate) fn new_for_test(sensor: Sensor) -> Self {
        System {
            sensor,
            settings: None,
            atlas: Arc::new(Atlas::from_kf_id(0)),
            vocabulary: None,
            keyframe_database: None,
            tracking: OnceLock::new(),
            local_mapping: OnceLock::new(),
            loop_closing: OnceLock::new(),
            #[cfg(feature = "viewer")]
            viewer: None,
            reset: AtomicBool::new(false),
            reset_active_map: AtomicBool::new(false),
            activate_localization_mode: AtomicBool::new(false),
            deactivate_localization_mode: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            threads: Mutex::new(Vec::new()),
            vocabulary_file_path: PathBuf::new(),
            load_atlas_file_path: None,
            save_atlas_file_path: None,
        }
    }
}

#[derive(Debug)]
pub enum SystemSaveError {
    Io(io::Error),
    Postcard(postcard::Error),
}

impl From<io::Error> for SystemSaveError {
    fn from(error: io::Error) -> Self {
        SystemSaveError::Io(error)
    }
}

impl From<postcard::Error> for SystemSaveError {
    fn from(error: postcard::Error) -> Self {
        SystemSaveError::Postcard(error)
    }
}

impl std::fmt::Display for SystemSaveError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SystemSaveError::Io(error) => write!(formatter, "I/O error: {error}"),
            SystemSaveError::Postcard(error) => write!(formatter, "serialization error: {error}"),
        }
    }
}

impl std::error::Error for SystemSaveError {}

impl Drop for System {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(mapping) = self.local_mapping.get() {
            mapping.request_finish();
            mapping.interrupt_ba();
        }
        if let Some(loop_closing) = self.loop_closing.get() {
            loop_closing.request_finish();
        }
        let handles = std::mem::take(&mut *self.threads.lock().unwrap());
        for handle in handles {
            let _ = handle.join();
        }
    }
}

#[derive(Debug)]
pub enum AtlasLoadError {
    Io(std::io::Error),
    Postcard(postcard::Error),
    IncompatibleVocabulary(String),
}
impl From<std::io::Error> for AtlasLoadError {
    fn from(err: std::io::Error) -> Self {
        AtlasLoadError::Io(err)
    }
}
impl From<postcard::Error> for AtlasLoadError {
    fn from(err: postcard::Error) -> Self {
        AtlasLoadError::Postcard(err)
    }
}
#[derive(Serialize, Deserialize)]
struct SessionSnapshot {
    file_voc: String,
    file_voc_checksum: String,
    atlas: Atlas,
}

#[derive(Serialize)]
struct SessionSnapshotRef<'a> {
    file_voc: String,
    file_voc_checksum: String,
    atlas: &'a Atlas,
}

fn write_tum_trajectory(
    path: impl AsRef<Path>,
    trajectory: &[(f64, nalgebra::Isometry3<f32>)],
) -> io::Result<()> {
    let mut file = File::create(path)?;
    for (timestamp, pose) in trajectory {
        let t = pose.translation.vector;
        let q = pose.rotation.quaternion();
        writeln!(
            file,
            "{timestamp:.6} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9}",
            t.x, t.y, t.z, q.i, q.j, q.k, q.w
        )?;
    }
    Ok(())
}

fn unsupported_trajectory_format(format: &str, sensor: Sensor) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("{format} trajectory export cannot be used for {sensor:?}"),
    )
}

fn load_atlas(
    path: &str,
    vocabulary_path: &PathBuf,
    key_frame_database: Arc<KeyFrameDatabase>,
    orb_vocabulary: Arc<OrbVocabulary>,
) -> Result<Atlas, AtlasLoadError> {
    let path_load_file_name = format!("./{}.postcard", path);
    let bytes = fs::read(path_load_file_name)?;
    let snapshot: SessionSnapshot = postcard::from_bytes(&bytes)?;

    // Check if the vocabulary is compatible
    let checksum = calculate_checksum(vocabulary_path)?;
    if checksum != snapshot.file_voc_checksum {
        return Err(AtlasLoadError::IncompatibleVocabulary(snapshot.file_voc));
    }

    let atlas = snapshot.atlas;
    atlas.set_key_frame_database(key_frame_database);
    atlas.set_orb_vocabulary(orb_vocabulary);
    atlas.post_load();

    Ok(atlas)
}

pub fn calculate_checksum(filename: impl AsRef<Path>) -> io::Result<String> {
    let mut f = File::open(filename)?;
    let mut hasher = Md5::new();
    let mut buffer = [0u8; 1024];
    loop {
        let count = f.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let digest = hasher.finalize();
    let checksum = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(checksum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{CV_8UC1, MatTraitConst, Scalar};

    use crate::key_frame::KeyFrame;
    use crate::key_frame_database::KeyFrameId;
    use crate::orb_vocabulary::{BowVector, OrbVocabulary};
    use crate::test_helpers::build_frame;

    #[test]
    fn stereo_preprocessing_applies_rectification_maps() {
        let settings_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config/euroc.yaml");
        let settings = Settings::new(&settings_path, Sensor::IMUStereo).expect("EuRoC settings");
        assert!(settings.need_to_rectify);
        let expected_size = settings.image.new_size;

        let mut system = System::new_for_test(Sensor::IMUStereo);
        system.settings = Some(Arc::new(settings));
        let left = Mat::new_rows_cols_with_default(480, 752, CV_8UC1, Scalar::all(80.0))
            .expect("left image");
        let right = Mat::new_rows_cols_with_default(480, 752, CV_8UC1, Scalar::all(120.0))
            .expect("right image");
        let (left, right) = system
            .preprocess_stereo(left, right)
            .expect("rectification");
        assert_eq!(left.size().expect("left size"), expected_size);
        assert_eq!(right.size().expect("right size"), expected_size);
        assert!(!left.empty());
        assert!(!right.empty());
    }

    #[test]
    fn reset_active_map_clears_database_before_dropping_keyframes() {
        let mut system = System::new_for_test(Sensor::Stereo);
        let map = system.atlas.get_current_map();
        let keyframe = KeyFrame::from_frame(&build_frame(), map.clone());
        map.add_key_frame(keyframe.clone());

        let mut vocabulary_text = String::from("1 1 0 0\n0 1 ");
        for _ in 0..crate::orb_vocabulary::DESC_LEN {
            vocabulary_text.push_str("0 ");
        }
        vocabulary_text.push_str("1.0\n");
        let vocabulary = Arc::new(
            OrbVocabulary::load_from_reader(vocabulary_text.as_bytes())
                .expect("one-word vocabulary"),
        );
        let database = Arc::new(KeyFrameDatabase::new(vocabulary));
        let bow = BowVector([(0, 1.0)].into());
        database.add(KeyFrameId(keyframe.id), &bow);
        assert!(database._entry_count() > 0);
        system.keyframe_database = Some(database.clone());

        system.reset_active_map();

        assert_eq!(map.key_frames_in_map(), 0);
        assert_eq!(database._entry_count(), 0);
        assert!(!system.reset_active_map.load(Ordering::SeqCst));
    }

    #[test]
    fn trajectory_sensor_semantics_match_upstream_exports() {
        let inertial = System::new_for_test(Sensor::IMUStereo);
        assert!(inertial.uses_imu_body_trajectory());
        let stereo = System::new_for_test(Sensor::Stereo);
        assert!(!stereo.uses_imu_body_trajectory());

        let mono = System::new_for_test(Sensor::Monocular);
        assert!(mono.save_trajectory_tum("unused.txt").is_err());
        assert!(mono.save_trajectory_kitti("unused.txt").is_err());
    }
}
