use md5::{Digest, Md5};
use opencv::core::Mat;
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
    key_frame_database::KeyFrameDatabase,
    local_mapping::LocalMapping,
    loop_closing::LoopClosing,
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
    // TODO: handle initFr and strSequence
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
        let tracking = self.tracking.get()?.clone();
        let mut tracking = tracking.lock().unwrap();
        for measurement in imu_measurements {
            tracking.grab_imu_data(Point::new(measurement.a, measurement.w, measurement.t));
        }
        tracking.grab_image_rgbd(image, depth, timestamp, filename.into())
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
            mapping.request_reset_active_map(current_map);
        }
        self.atlas.clear_map();
        if let Some(tracking) = self.tracking.get()
            && let Ok(mut tracking) = tracking.lock()
        {
            tracking.reset_active_map();
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
        let trajectory = self
            .tracking
            .get()
            .map(|tracking| tracking.lock().unwrap().frame_trajectory())
            .unwrap_or_default();
        write_tum_trajectory(path, &trajectory)
    }

    pub fn save_keyframe_trajectory_tum(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let trajectory = self
            .tracking
            .get()
            .map(|tracking| tracking.lock().unwrap().keyframe_trajectory())
            .unwrap_or_default();
        write_tum_trajectory(path, &trajectory)
    }

    pub fn save_trajectory_euroc(&self, path: impl AsRef<Path>) -> io::Result<()> {
        self.save_trajectory_tum(path)
    }

    pub fn save_keyframe_trajectory_euroc(&self, path: impl AsRef<Path>) -> io::Result<()> {
        self.save_keyframe_trajectory_tum(path)
    }

    pub fn save_trajectory_kitti(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let trajectory = self
            .tracking
            .get()
            .map(|tracking| tracking.lock().unwrap().frame_trajectory())
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
            "{timestamp:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9}",
            t.x, t.y, t.z, q.i, q.j, q.k, q.w
        )?;
    }
    Ok(())
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
