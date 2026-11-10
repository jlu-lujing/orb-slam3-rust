//! TUM RGB-D association loading and execution helpers.

use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use opencv::imgcodecs::{IMREAD_COLOR, IMREAD_UNCHANGED, imread};
use opencv::prelude::MatTraitConst;

use crate::system::System;

#[derive(Debug)]
pub enum TumRgbdError {
    Io(io::Error),
    InvalidAssociation { line: usize, value: String },
    InvalidTimestamp { line: usize, value: String },
    ImageDecode(PathBuf),
}

impl std::fmt::Display for TumRgbdError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TumRgbdError::Io(error) => write!(formatter, "{error}"),
            TumRgbdError::InvalidAssociation { line, value } => {
                write!(formatter, "invalid association at line {line}: {value}")
            }
            TumRgbdError::InvalidTimestamp { line, value } => {
                write!(formatter, "invalid timestamp at line {line}: {value}")
            }
            TumRgbdError::ImageDecode(path) => {
                write!(formatter, "failed to decode image {}", path.display())
            }
        }
    }
}

impl std::error::Error for TumRgbdError {}

impl From<io::Error> for TumRgbdError {
    fn from(error: io::Error) -> Self {
        TumRgbdError::Io(error)
    }
}

#[derive(Debug, Clone)]
pub struct TumRgbdFrame {
    pub timestamp: f64,
    pub rgb: PathBuf,
    pub depth: PathBuf,
}

#[derive(Debug, Clone)]
pub struct TumRgbdSequence {
    pub root: PathBuf,
    pub name: String,
    pub frames: Vec<TumRgbdFrame>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TumRgbdRunStats {
    pub images: usize,
    pub mean_tracking_ms: f32,
}

impl TumRgbdSequence {
    pub fn load(
        root: impl AsRef<Path>,
        associations: impl AsRef<Path>,
    ) -> Result<Self, TumRgbdError> {
        let root = root.as_ref().canonicalize()?;
        let associations = if associations.as_ref().is_absolute() {
            associations.as_ref().to_path_buf()
        } else if associations.as_ref().is_file() {
            // The upstream TUM example resolves the association file from the
            // process working directory, while the convenience API also allows
            // a path relative to the dataset root.
            associations.as_ref().to_path_buf()
        } else {
            root.join(associations)
        };
        let mut frames = Vec::new();
        for (line_index, line) in BufReader::new(File::open(&associations)?)
            .lines()
            .enumerate()
        {
            let line = line?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let fields: Vec<_> = line.split_whitespace().collect();
            if fields.len() != 4 {
                return Err(TumRgbdError::InvalidAssociation {
                    line: line_index + 1,
                    value: line.to_owned(),
                });
            }
            let timestamp =
                fields[0]
                    .parse::<f64>()
                    .map_err(|_| TumRgbdError::InvalidTimestamp {
                        line: line_index + 1,
                        value: fields[0].to_owned(),
                    })?;
            let depth_timestamp =
                fields[2]
                    .parse::<f64>()
                    .map_err(|_| TumRgbdError::InvalidTimestamp {
                        line: line_index + 1,
                        value: fields[2].to_owned(),
                    })?;
            if (timestamp - depth_timestamp).abs() > 0.02 {
                return Err(TumRgbdError::InvalidAssociation {
                    line: line_index + 1,
                    value: line.to_owned(),
                });
            }
            frames.push(TumRgbdFrame {
                timestamp,
                rgb: root.join(fields[1]),
                depth: root.join(fields[3]),
            });
        }
        frames.sort_by(|left, right| left.timestamp.total_cmp(&right.timestamp));
        let name = root
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| root.display().to_string());
        Ok(Self { root, name, frames })
    }
}

pub fn run_sequence(
    system: Arc<System>,
    sequence: &TumRgbdSequence,
) -> Result<TumRgbdRunStats, TumRgbdError> {
    run_sequence_with_sync(system, sequence, false)
}

/// Run a sequence, optionally waiting for Local Mapping to finish processing
/// each keyframe before submitting the next image.
/// Monocular variant of [`run_sequence_with_sync`]: the RGB image is fed
/// through `System::track_monocular` and the depth image is ignored. Used to
/// smoke-test the monocular pipeline against the same sequences (monocular
/// scale is arbitrary, so only execution/consistency is checked).
pub fn run_monocular_sequence(
    system: Arc<System>,
    sequence: &TumRgbdSequence,
    sync_local_mapping: bool,
) -> Result<TumRgbdRunStats, TumRgbdError> {
    let mut total_ms = 0.0f32;
    let mut images = 0usize;
    for frame in &sequence.frames {
        if system.is_shutdown() {
            break;
        }
        let rgb = imread(&frame.rgb.to_string_lossy(), IMREAD_COLOR)
            .map_err(|_| TumRgbdError::ImageDecode(frame.rgb.clone()))?;
        if rgb.empty() {
            return Err(TumRgbdError::ImageDecode(frame.rgb.clone()));
        }
        let started = std::time::Instant::now();
        let filename = frame
            .rgb
            .strip_prefix(&sequence.root)
            .unwrap_or(&frame.rgb)
            .to_string_lossy()
            .into_owned();
        let _ = system.track_monocular(rgb, frame.timestamp, &[], filename);
        if sync_local_mapping {
            wait_for_local_mapping(&system);
        }
        let atlas = system.get_atlas();
        let map = atlas.get_current_map();
        tracing::debug!(
            timestamp = frame.timestamp,
            state = ?system.tracking_state(),
            keyframes = map.key_frames_in_map(),
            map_points = map.map_points_in_map(),
            "monocular frame completed"
        );
        total_ms += started.elapsed().as_secs_f32() * 1000.0;
        images += 1;
    }
    Ok(TumRgbdRunStats {
        images,
        mean_tracking_ms: if images == 0 {
            0.0
        } else {
            total_ms / images as f32
        },
    })
}

pub fn run_sequence_with_sync(
    system: Arc<System>,
    sequence: &TumRgbdSequence,
    sync_local_mapping: bool,
) -> Result<TumRgbdRunStats, TumRgbdError> {
    let mut total_ms = 0.0f32;
    let mut images = 0usize;
    for frame in &sequence.frames {
        if system.is_shutdown() {
            break;
        }
        let rgb = imread(&frame.rgb.to_string_lossy(), IMREAD_COLOR)
            .map_err(|_| TumRgbdError::ImageDecode(frame.rgb.clone()))?;
        let depth = imread(&frame.depth.to_string_lossy(), IMREAD_UNCHANGED)
            .map_err(|_| TumRgbdError::ImageDecode(frame.depth.clone()))?;
        if rgb.empty() {
            return Err(TumRgbdError::ImageDecode(frame.rgb.clone()));
        }
        if depth.empty() {
            return Err(TumRgbdError::ImageDecode(frame.depth.clone()));
        }
        let started = std::time::Instant::now();
        let filename = frame
            .rgb
            .strip_prefix(&sequence.root)
            .unwrap_or(&frame.rgb)
            .to_string_lossy()
            .into_owned();
        let _ = system.track_rgbd(rgb, depth, frame.timestamp, &[], filename);
        if sync_local_mapping {
            wait_for_local_mapping(&system);
        }
        let atlas = system.get_atlas();
        let map = atlas.get_current_map();
        tracing::debug!(
            timestamp = frame.timestamp,
            state = ?system.tracking_state(),
            keyframes = map.key_frames_in_map(),
            map_points = map.map_points_in_map(),
            "RGB-D frame completed"
        );
        total_ms += started.elapsed().as_secs_f32() * 1000.0;
        images += 1;
    }
    Ok(TumRgbdRunStats {
        images,
        mean_tracking_ms: if images == 0 {
            0.0
        } else {
            total_ms / images as f32
        },
    })
}

fn wait_for_local_mapping(system: &System) {
    let Some(mapper) = system.get_local_mapping() else {
        return;
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let idle = mapper.accept_key_frames()
            && mapper.keyframes_in_queue() == 0
            && !mapper.is_initializing();
        if idle {
            break;
        }
        if system.is_shutdown() || std::time::Instant::now() >= deadline {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}
