use std::path::PathBuf;

use clap::{Arg, ArgAction, Command};
use orb_slam3_rs::system::{Sensor, System};
use orb_slam3_rs::tum_rgbd::{TumRgbdSequence, run_monocular_sequence, run_sequence_with_sync};
use tracing::info;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let matches = Command::new("tum_rgbd")
        .arg(Arg::new("vocabulary").required(true).index(1))
        .arg(Arg::new("settings").required(true).index(2))
        .arg(Arg::new("dataset").required(true).index(3))
        .arg(
            Arg::new("associations")
                .long("associations")
                .default_value("associations.txt"),
        )
        .arg(
            Arg::new("sync-local-mapping")
                .long("sync-local-mapping")
                .help("Wait for Local Mapping to become idle after each frame")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("mono")
                .long("mono")
                .help("Run the monocular pipeline (depth ignored); trajectory scale is arbitrary")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("localization")
                .long("localization")
                .help("Track against the current map without Local Mapping updates")
                .action(ArgAction::SetTrue),
        )
        .get_matches();
    let vocabulary = PathBuf::from(matches.get_one::<String>("vocabulary").expect("required"));
    let settings = PathBuf::from(matches.get_one::<String>("settings").expect("required"));
    let dataset = PathBuf::from(matches.get_one::<String>("dataset").expect("required"));
    let associations = matches.get_one::<String>("associations").expect("default");
    let sequence = TumRgbdSequence::load(&dataset, associations)?;
    let mono = matches.get_flag("mono");
    let sensor = if mono {
        Sensor::Monocular
    } else {
        Sensor::RGBD
    };
    let system = System::new(&vocabulary, &settings, sensor, true, None)?;
    if matches.get_flag("localization") {
        system.activate_localization_mode();
    }
    let stats = if mono {
        run_monocular_sequence(
            system.clone(),
            &sequence,
            matches.get_flag("sync-local-mapping"),
        )?
    } else {
        run_sequence_with_sync(
            system.clone(),
            &sequence,
            matches.get_flag("sync-local-mapping"),
        )?
    };
    if mono {
        system.save_trajectory_euroc("CameraTrajectory.txt")?;
        system.save_keyframe_trajectory_euroc("KeyFrameTrajectory.txt")?;
    } else {
        system.save_trajectory_tum("CameraTrajectory.txt")?;
        system.save_keyframe_trajectory_tum("KeyFrameTrajectory.txt")?;
    }
    system.shutdown();
    info!(
        "Processed {} images from {}, mean tracking {:.3} ms",
        stats.images, sequence.name, stats.mean_tracking_ms
    );
    Ok(())
}
