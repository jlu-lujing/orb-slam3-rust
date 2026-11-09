//! EuRoC dataset loading and stereo-inertial execution helpers.

use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use nalgebra::Vector3;
use opencv::imgcodecs::{IMREAD_GRAYSCALE, imread};
use opencv::prelude::MatTraitConst;

use crate::imu_types::Point;
use crate::system::System;

#[derive(Debug)]
pub enum EurocError {
    Io(io::Error),
    InvalidData(String),
    ImageDecode(PathBuf),
}

impl From<io::Error> for EurocError {
    fn from(error: io::Error) -> Self {
        EurocError::Io(error)
    }
}

impl std::fmt::Display for EurocError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EurocError::Io(error) => write!(formatter, "I/O error: {error}"),
            EurocError::InvalidData(message) => write!(formatter, "invalid EuRoC data: {message}"),
            EurocError::ImageDecode(path) => {
                write!(formatter, "failed to decode image {}", path.display())
            }
        }
    }
}

impl std::error::Error for EurocError {}

#[derive(Debug, Clone)]
pub struct EurocSequence {
    pub name: String,
    pub images_left: Vec<PathBuf>,
    pub images_right: Vec<PathBuf>,
    pub image_timestamps: Vec<f64>,
    pub acc: Vec<Vector3<f32>>,
    pub gyro: Vec<Vector3<f32>>,
    pub imu_timestamps: Vec<f64>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EurocRunStats {
    pub images: usize,
    pub mean_tracking_ms: f32,
}

impl EurocSequence {
    pub fn load(root: impl AsRef<Path>, timestamps: impl AsRef<Path>) -> Result<Self, EurocError> {
        let root = root.as_ref();
        let name = root
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| root.display().to_string());
        let images = load_images(
            &root.join("mav0").join("cam0").join("data"),
            &root.join("mav0").join("cam1").join("data"),
            timestamps.as_ref(),
        )?;
        let imu = load_imu(&root.join("mav0").join("imu0").join("data.csv"))?;
        Ok(Self {
            name,
            images_left: images.left,
            images_right: images.right,
            image_timestamps: images.timestamps,
            acc: imu.acc,
            gyro: imu.gyro,
            imu_timestamps: imu.timestamps,
        })
    }
}

pub fn run_sequences(
    system: Arc<System>,
    sequences: &[EurocSequence],
    realtime: bool,
) -> Result<EurocRunStats, EurocError> {
    let mut total_images = 0usize;
    let mut total_ms = 0.0f32;
    for (sequence_index, sequence) in sequences.iter().enumerate() {
        let mut imu_cursor = sequence
            .imu_timestamps
            .iter()
            .position(|&t| t > *sequence.image_timestamps.first().unwrap_or(&0.0))
            .unwrap_or(0)
            .saturating_sub(1);
        let mut previous_image_timestamp: Option<f64> = None;

        for frame_index in 0..sequence.image_timestamps.len() {
            if system.is_shutdown() {
                return Ok(EurocRunStats {
                    images: total_images,
                    mean_tracking_ms: if total_images == 0 {
                        0.0
                    } else {
                        total_ms / total_images as f32
                    },
                });
            }
            let timestamp = sequence.image_timestamps[frame_index];
            if realtime && let Some(previous) = previous_image_timestamp {
                let delay = (timestamp - previous).max(0.0);
                std::thread::sleep(std::time::Duration::from_secs_f64(delay));
            }
            previous_image_timestamp = Some(timestamp);

            let mut measurements = Vec::new();
            while imu_cursor < sequence.imu_timestamps.len()
                && sequence.imu_timestamps[imu_cursor] <= timestamp
            {
                measurements.push(Point::new(
                    sequence.acc[imu_cursor],
                    sequence.gyro[imu_cursor],
                    sequence.imu_timestamps[imu_cursor],
                ));
                imu_cursor += 1;
            }

            let left_path = &sequence.images_left[frame_index];
            let right_path = &sequence.images_right[frame_index];
            let image_left = imread(&left_path.to_string_lossy(), IMREAD_GRAYSCALE)
                .map_err(|_| EurocError::ImageDecode(left_path.clone()))?;
            let image_right = imread(&right_path.to_string_lossy(), IMREAD_GRAYSCALE)
                .map_err(|_| EurocError::ImageDecode(right_path.clone()))?;
            if image_left.empty() {
                return Err(EurocError::ImageDecode(left_path.clone()));
            }
            if image_right.empty() {
                return Err(EurocError::ImageDecode(right_path.clone()));
            }

            let start = std::time::Instant::now();
            let _ = system.track_stereo(
                image_left,
                image_right,
                timestamp,
                &measurements,
                format!(
                    "{}/{}",
                    sequence.name,
                    left_path
                        .file_name()
                        .map(|name| name.to_string_lossy())
                        .unwrap_or_default()
                ),
            );
            total_ms += start.elapsed().as_secs_f32() * 1000.0;
            total_images += 1;
        }

        if sequence_index + 1 < sequences.len() && !system.is_shutdown() {
            system.reset_active_map();
        }
    }
    Ok(EurocRunStats {
        images: total_images,
        mean_tracking_ms: if total_images == 0 {
            0.0
        } else {
            total_ms / total_images as f32
        },
    })
}

struct LoadedImages {
    left: Vec<PathBuf>,
    right: Vec<PathBuf>,
    timestamps: Vec<f64>,
}

fn load_images(
    path_left: &Path,
    path_right: &Path,
    path_times: &Path,
) -> Result<LoadedImages, EurocError> {
    let reader = BufReader::new(File::open(path_times)?);
    let mut left = Vec::new();
    let mut right = Vec::new();
    let mut timestamps = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let timestamp_text = line.split(',').next().unwrap_or(line);
        let timestamp_ns = timestamp_text
            .parse::<f64>()
            .map_err(|_| EurocError::InvalidData(format!("invalid timestamp line: {line}")))?;
        left.push(path_left.join(format!("{timestamp_text}.png")));
        right.push(path_right.join(format!("{timestamp_text}.png")));
        timestamps.push(timestamp_ns / 1e9);
    }
    Ok(LoadedImages {
        left,
        right,
        timestamps,
    })
}

struct LoadedImu {
    acc: Vec<Vector3<f32>>,
    gyro: Vec<Vector3<f32>>,
    timestamps: Vec<f64>,
}

fn load_imu(path: &Path) -> Result<LoadedImu, EurocError> {
    let reader = BufReader::new(File::open(path)?);
    let mut acc = Vec::new();
    let mut gyro = Vec::new();
    let mut timestamps = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let values = line
            .split(',')
            .map(str::trim)
            .map(|value| {
                value
                    .parse::<f64>()
                    .map_err(|_| EurocError::InvalidData(format!("invalid IMU line: {line}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if values.len() != 7 {
            return Err(EurocError::InvalidData(format!(
                "expected 7 IMU values, got {} in line: {line}",
                values.len()
            )));
        }
        timestamps.push(values[0] / 1e9);
        gyro.push(Vector3::new(
            values[1] as f32,
            values[2] as f32,
            values[3] as f32,
        ));
        acc.push(Vector3::new(
            values[4] as f32,
            values[5] as f32,
            values[6] as f32,
        ));
    }
    Ok(LoadedImu {
        acc,
        gyro,
        timestamps,
    })
}
