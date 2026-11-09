use md5::{Digest, Md5};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::{
    fs::File,
    io::{self, Read},
    path::Path,
};
use tracing::info;

use crate::{
    atlas::Atlas,
    key_frame_database::KeyFrameDatabase,
    orb_vocabulary::{OrbVocabulary, VocabularyError},
    settings::{Settings, SettingsError},
    viewer::Viewer,
};
use crate::{key_frame_database, orb_vocabulary};

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
    // TODO
    #[cfg(feature = "viewer")]
    viewer: Option<Arc<Viewer>>,
    reset: bool,
    reset_active_map: bool,
    activate_localization_mode: bool,
    deactivate_localization_mode: bool,
    shutdown: bool,
    vocabulary_file_path: PathBuf,
    load_atlas_file_path: Option<String>,
    save_atlas_file_path: Option<String>,
}

#[derive(Debug)]
pub enum SystemError {
    InvalidSettings(SettingsError),
    InvalidORBVocabulary(VocabularyError),
}

impl System {
    // TODO: handle initFr and strSequence
    pub fn new(
        vocabulary_path: &PathBuf,
        settings_path: &PathBuf,
        sensor: Sensor,
        use_viewer: bool,
        sequence: Option<String>,
    ) -> Result<Self, SystemError> {
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
            Settings::new(settings_path, sensor).map_err(|e| SystemError::InvalidSettings(e))?;
        let loop_closing = settings.loop_closing;

        // Load ORB VOB vocabulary
        info!("Loading ORB vocabulary. This could take a while...");
        let vocabulary = OrbVocabulary::load_from_text_file(vocabulary_path)
            .map_err(|e| SystemError::InvalidORBVocabulary(e))?;
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
            .expect("Error loading Atlas file, please try with other session file or vocabulary");
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

        // Create drawers for the viewer
        #[cfg(feature = "viewer")]
        if use_viewer {
            // TODO: implement this later, viewer stuff is optional
        }

        if let Some(sequence) = sequence {
            info!("Seq. Name: {}", sequence);
        }

        // Initialize the tracking thread
        // It will live in the main thread of execution, the one that called this constructor
        //

        Ok(System {
            sensor,
            viewer: None,
            reset: false,
            reset_active_map: false,
            activate_localization_mode: false,
            deactivate_localization_mode: false,
            shutdown: false,
            vocabulary_file_path: vocabulary_path.clone(),
            load_atlas_file_path: settings.load_and_save.load_from.clone(),
            save_atlas_file_path: settings.load_and_save.save_to.clone(),
        })
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
