use clap::{Arg, Command};
use opencv::core::Point3f;
use orb_slam3_rs::system::System;
use std::fs::File;
use std::{
    io::{self, BufRead, BufReader},
    path::{Path, PathBuf},
};
use tracing::{error, info};

fn build_cli() -> Command {
    Command::new("stereo_intertial_euroc")
        .about("Run stereo interial EuRoC sequences")
        .arg(
            Arg::new("vocabulary")
                .help("Path to vocabulary")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::new("settings")
                .help("Path to settings")
                .required(true)
                .index(2),
        )
        .arg(
            Arg::new("sequence_folders")
                .help("Paths to sequences and times")
                .required(true)
                .num_args(2..)
                .index(3)
                .trailing_var_arg(true),
        )
}

#[derive(Debug)]
enum RuntimeError {
    InvalidSequenceArgs,
    ImageLoadFailed(io::Error),
    IMULoadFailed(io::Error),
    SettingsLoadFailed,
}

struct LoadedImages {
    images_left: Vec<PathBuf>,
    images_right: Vec<PathBuf>,
    timestamps: Vec<f64>,
}
fn load_images(
    path_left: &PathBuf,
    path_right: &PathBuf,
    path_times: &PathBuf,
) -> io::Result<LoadedImages> {
    let file = File::open(path_times)?;
    let reader = BufReader::new(file);

    let mut images_left = Vec::with_capacity(50);
    let mut images_right = Vec::with_capacity(50);
    let mut timestamps = Vec::with_capacity(50);

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let t: f64 = line
            .split(',')
            .next()
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid timestamp line: {line}"),
                )
            })?
            .parse()
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid timestamp line: {line}"),
                )
            })?;
        images_left.push(path_left.join(format!("{line}.png")));
        images_right.push(path_right.join(format!("{line}.png")));
        timestamps.push(t / 1e9);
    }

    Ok(LoadedImages {
        images_left,
        images_right,
        timestamps,
    })
}

struct LoadedIMU {
    acc: Vec<Point3f>,
    gyro: Vec<Point3f>,
    timestamps: Vec<f64>,
}
fn load_imu(path_imu: &PathBuf) -> io::Result<LoadedIMU> {
    let file = File::open(path_imu)?;
    let reader = BufReader::new(file);

    let mut acc = Vec::with_capacity(50);
    let mut gyro = Vec::with_capacity(50);
    let mut timestamps = Vec::with_capacity(50);

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let values: Vec<f64> = line
            .split(',')
            .map(str::trim)
            .map(|s| {
                s.parse::<f64>().map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("invalid IMU line: {line}"),
                    )
                })
            })
            .collect::<Result<_, _>>()?;

        if values.len() != 7 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "expected 7 comma-separated values, got {} in line: {line}",
                    values.len()
                ),
            ));
        }
        timestamps.push(values[0] / 1e9);
        gyro.push(Point3f::new(
            values[1] as f32,
            values[2] as f32,
            values[3] as f32,
        ));
        acc.push(Point3f::new(
            values[4] as f32,
            values[5] as f32,
            values[6] as f32,
        ));
    }

    Ok(LoadedIMU {
        acc,
        gyro,
        timestamps,
    })
}

fn main() -> Result<(), RuntimeError> {
    // install global subscriber
    tracing_subscriber::fmt::init();

    // argument handling
    let matches = build_cli().get_matches();
    let vocabulary = matches.get_one::<String>("vocabulary").expect("required");
    let settings = matches.get_one::<String>("settings").expect("required");
    let sequence_folders: Vec<_> = matches
        .get_many::<String>("sequence_folders")
        .expect("required")
        .map(|s| s.as_str())
        .collect();
    if sequence_folders.len() % 2 != 0 {
        error!("Mismatch between number of sequences and times");
        return Err(RuntimeError::InvalidSequenceArgs);
    }
    let vocabulary_path = PathBuf::from(vocabulary);
    let settings_path = PathBuf::from(settings);

    // number of sequences
    let n_seq: usize = sequence_folders.len() / 2;
    info!("num_seq = {}", n_seq);

    // load all the sequences
    let mut images_left: Vec<Vec<PathBuf>> = Vec::with_capacity(n_seq);
    let mut images_right: Vec<Vec<PathBuf>> = Vec::with_capacity(n_seq);
    let mut timestamps_cam: Vec<Vec<f64>> = Vec::with_capacity(n_seq);
    let mut acc: Vec<Vec<Point3f>> = Vec::with_capacity(n_seq);
    let mut gyro: Vec<Vec<Point3f>> = Vec::with_capacity(n_seq);
    let mut timestamps_imu: Vec<Vec<f64>> = Vec::with_capacity(n_seq);
    let mut n_images: Vec<usize> = Vec::with_capacity(n_seq);
    let mut n_imu: Vec<usize> = Vec::with_capacity(n_seq);
    let mut first_imu: Vec<usize> = vec![0; n_seq];

    let mut total_images: usize = 0;
    for seq in 0..n_seq {
        let path_sequence = Path::new(
            sequence_folders
                .get(2 * seq)
                .expect("missing sequence path"),
        );
        let path_timestamps = PathBuf::from(
            sequence_folders
                .get(2 * seq + 1)
                .expect("missing timestamps path"),
        );
        let path_cam0 = path_sequence.join("mav0").join("cam0").join("data");
        let path_cam1 = path_sequence.join("mav0").join("cam1").join("data");
        let path_imu = path_sequence.join("mav0").join("imu0").join("data.csv");

        info!("Loading images for sequence {}...", seq);
        let images = load_images(&path_cam0, &path_cam1, &path_timestamps)
            .map_err(|e| RuntimeError::ImageLoadFailed(e))?;
        let n = images.timestamps.len();
        n_images.push(n);
        total_images += n;
        images_left.push(images.images_left);
        images_right.push(images.images_right);
        timestamps_cam.push(images.timestamps);
        info!("Images loaded");

        info!("Loading IMU for sequence {}...", seq);
        let imu = load_imu(&path_imu).map_err(|e| RuntimeError::IMULoadFailed(e))?;
        n_imu.push(imu.timestamps.len());
        acc.push(imu.acc);
        gyro.push(imu.gyro);
        timestamps_imu.push(imu.timestamps);
        info!("IMU data loaded");

        // find first imu to be considered, supposing imu measurements start first
        let cam_t0 = timestamps_cam.last().unwrap().first().unwrap();
        first_imu.push(
            timestamps_imu
                .last()
                .unwrap()
                .iter()
                .position(|&t| t > *cam_t0)
                .map(|i| i.saturating_sub(1))
                .unwrap_or_else(|| timestamps_imu.first().unwrap().len().saturating_sub(1)),
        );
    }

    // Track time statistics
    let mut times_track: Vec<f32> = Vec::with_capacity(total_images);

    // Create SLAM system. It initializes all system threads and gets ready to process frames
    let slam = System::new(
        &vocabulary_path,
        &settings_path,
        orb_slam3_rs::system::Sensor::IMUStereo,
        true,
        None,
    );

    // TODO: here

    Ok(())
}
