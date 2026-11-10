use std::path::PathBuf;

use clap::{Arg, ArgAction, Command};
use orb_slam3_rs::evaluation::{
    evaluate_ate_with_offset, evaluate_rpe_with_offset, load_euroc_csv, load_tum,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("trajectory_eval")
        .arg(Arg::new("estimated").required(true).index(1))
        .arg(Arg::new("ground_truth").required(true).index(2))
        .arg(
            Arg::new("euroc")
                .long("euroc")
                .help("Ground truth uses EuRoC CSV (timestamp,p,q) ordering")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("no-scale")
                .long("no-scale")
                .help("Disable scale estimation in Umeyama alignment")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("max-diff")
                .long("max-diff")
                .value_name("SECONDS")
                .default_value("0.02"),
        )
        .arg(
            Arg::new("offset")
                .long("offset")
                .value_name("SECONDS")
                .default_value("0.0")
                .help("Offset added to estimated timestamps before association"),
        )
        .arg(
            Arg::new("rpe-delta")
                .long("rpe-delta")
                .value_name("SECONDS")
                // Match `scripts/eval/evaluate_trajectory.py --rpe-delta` in the
                // simulation workspace so the RPE numbers are comparable.
                .default_value("0.5"),
        )
        .get_matches();
    let estimated_path = PathBuf::from(matches.get_one::<String>("estimated").expect("required"));
    let truth_path = PathBuf::from(matches.get_one::<String>("ground_truth").expect("required"));
    let estimated = load_tum(estimated_path)?;
    let truth = if matches.get_flag("euroc") {
        load_euroc_csv(truth_path)?
    } else {
        load_tum(truth_path)?
    };
    let max_diff = matches
        .get_one::<String>("max-diff")
        .expect("default")
        .parse::<f64>()?;
    let offset = matches
        .get_one::<String>("offset")
        .expect("default")
        .parse::<f64>()?;
    let rpe_delta = matches
        .get_one::<String>("rpe-delta")
        .expect("default")
        .parse::<f64>()?;
    let ate = evaluate_ate_with_offset(
        &estimated,
        &truth,
        !matches.get_flag("no-scale"),
        offset,
        max_diff,
    );
    let rpe = evaluate_rpe_with_offset(&estimated, &truth, rpe_delta, offset, max_diff);

    println!("associated: {}", ate.associated);
    println!("scale: {:.9}", ate.scale);
    println!("ate_rmse: {:.9}", ate.rmse);
    println!("ate_mean: {:.9}", ate.mean);
    println!("ate_median: {:.9}", ate.median);
    println!("ate_max: {:.9}", ate.max);
    println!("rpe_pairs: {}", rpe.pairs);
    println!("rpe_translation_rmse: {:.9}", rpe.translational_rmse);
    println!("rpe_rotation_rmse_rad: {:.9}", rpe.rotational_rmse);
    Ok(())
}
