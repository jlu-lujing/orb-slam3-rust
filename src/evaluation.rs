//! Trajectory association and quantitative SLAM evaluation.

use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use nalgebra::{Isometry3, Matrix3, UnitQuaternion, Vector3};

#[derive(Debug, Clone, Copy)]
pub struct PoseSample {
    pub timestamp: f64,
    pub pose: Isometry3<f64>,
}

#[derive(Debug, Clone)]
pub struct AteResult {
    pub associated: usize,
    pub rmse: f64,
    pub mean: f64,
    pub median: f64,
    pub max: f64,
    pub scale: f64,
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
}

#[derive(Debug, Clone)]
pub struct RpeResult {
    pub pairs: usize,
    pub translational_rmse: f64,
    pub rotational_rmse: f64,
}

/// Load TUM/ORB-SLAM trajectory format:
/// `timestamp tx ty tz qx qy qz qw`.
pub fn load_tum(path: impl AsRef<Path>) -> io::Result<Vec<PoseSample>> {
    load_space_or_comma(path.as_ref(), false)
}

/// Load EuRoC ground-truth CSV:
/// `timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z`.
pub fn load_euroc_csv(path: impl AsRef<Path>) -> io::Result<Vec<PoseSample>> {
    load_space_or_comma(path.as_ref(), true)
}

fn load_space_or_comma(path: &Path, euroc_csv: bool) -> io::Result<Vec<PoseSample>> {
    let mut samples = Vec::new();
    for (line_index, line) in BufReader::new(File::open(path)?).lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let values: Vec<_> = if euroc_csv {
            line.split(',').collect()
        } else {
            line.split_whitespace().collect()
        };
        if values.len() < 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}:{}: expected 8 fields", path.display(), line_index + 1),
            ));
        }
        let parsed = values[..8]
            .iter()
            .map(|value| value.trim().parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{}:{}: {error}", path.display(), line_index + 1),
                )
            })?;
        let (timestamp, translation, quaternion) = if euroc_csv {
            (
                parsed[0],
                Vector3::new(parsed[1], parsed[2], parsed[3]),
                UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                    parsed[4], parsed[5], parsed[6], parsed[7],
                )),
            )
        } else {
            (
                parsed[0],
                Vector3::new(parsed[1], parsed[2], parsed[3]),
                UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                    parsed[7], parsed[4], parsed[5], parsed[6],
                )),
            )
        };
        samples.push(PoseSample {
            timestamp,
            pose: Isometry3::from_parts(translation.into(), quaternion),
        });
    }
    samples.sort_by(|left, right| left.timestamp.total_cmp(&right.timestamp));
    Ok(samples)
}

/// Associate timestamps one-to-one, preferring the globally smallest
/// differences, matching ORB-SLAM3's `associate.py`.
pub fn associate(
    estimated: &[PoseSample],
    ground_truth: &[PoseSample],
    max_difference: f64,
) -> Vec<(PoseSample, PoseSample)> {
    associate_with_offset(estimated, ground_truth, 0.0, max_difference)
}

/// Associate timestamps after adding `offset` to each estimated timestamp.
pub fn associate_with_offset(
    estimated: &[PoseSample],
    ground_truth: &[PoseSample],
    offset: f64,
    max_difference: f64,
) -> Vec<(PoseSample, PoseSample)> {
    let mut candidates = Vec::new();
    for (estimate_index, estimate) in estimated.iter().enumerate() {
        for (truth_index, truth) in ground_truth.iter().enumerate() {
            let difference = (truth.timestamp - (estimate.timestamp + offset)).abs();
            if difference < max_difference {
                candidates.push((difference, estimate_index, truth_index));
            }
        }
    }
    candidates.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then(left.1.cmp(&right.1))
            .then(left.2.cmp(&right.2))
    });

    let mut used_estimates = vec![false; estimated.len()];
    let mut used_truth = vec![false; ground_truth.len()];
    let mut matches = Vec::new();
    for (_, estimate_index, truth_index) in candidates {
        if !used_estimates[estimate_index] && !used_truth[truth_index] {
            used_estimates[estimate_index] = true;
            used_truth[truth_index] = true;
            matches.push((estimated[estimate_index], ground_truth[truth_index]));
        }
    }
    matches.sort_by(|left, right| left.0.timestamp.total_cmp(&right.0.timestamp));
    matches
}

/// Align estimated positions to ground truth and compute ATE.
pub fn evaluate_ate(
    estimated: &[PoseSample],
    ground_truth: &[PoseSample],
    align_scale: bool,
    max_difference: f64,
) -> AteResult {
    evaluate_ate_with_offset(estimated, ground_truth, align_scale, 0.0, max_difference)
}

/// Offset-aware variant of [`evaluate_ate`].
pub fn evaluate_ate_with_offset(
    estimated: &[PoseSample],
    ground_truth: &[PoseSample],
    align_scale: bool,
    offset: f64,
    max_difference: f64,
) -> AteResult {
    let pairs = associate_with_offset(estimated, ground_truth, offset, max_difference);
    if pairs.is_empty() {
        return AteResult {
            associated: 0,
            rmse: f64::NAN,
            mean: f64::NAN,
            median: f64::NAN,
            max: f64::NAN,
            scale: 1.0,
            rotation: Matrix3::identity(),
            translation: Vector3::zeros(),
        };
    }
    let estimated_positions: Vec<_> = pairs
        .iter()
        .map(|(estimate, _)| estimate.pose.translation.vector)
        .collect();
    let truth_positions: Vec<_> = pairs
        .iter()
        .map(|(_, truth)| truth.pose.translation.vector)
        .collect();
    let (scale, rotation, translation) =
        solve_umeyama(&estimated_positions, &truth_positions, align_scale);
    let mut errors: Vec<_> = estimated_positions
        .iter()
        .zip(&truth_positions)
        .map(|(estimate, truth)| (scale * (rotation * estimate) + translation - truth).norm())
        .collect();
    errors.sort_by(f64::total_cmp);
    let rmse = (errors.iter().map(|error| error * error).sum::<f64>() / errors.len() as f64).sqrt();
    let mean = errors.iter().sum::<f64>() / errors.len() as f64;
    let median = if errors.len() % 2 == 0 {
        0.5 * (errors[errors.len() / 2 - 1] + errors[errors.len() / 2])
    } else {
        errors[errors.len() / 2]
    };
    let max = errors.last().copied().unwrap_or(0.0);
    AteResult {
        associated: pairs.len(),
        rmse,
        mean,
        median,
        max,
        scale,
        rotation,
        translation,
    }
}

/// Relative pose error on associated pose pairs separated by at least
/// `delta_seconds`.
pub fn evaluate_rpe(
    estimated: &[PoseSample],
    ground_truth: &[PoseSample],
    delta_seconds: f64,
    max_difference: f64,
) -> RpeResult {
    evaluate_rpe_with_offset(estimated, ground_truth, delta_seconds, 0.0, max_difference)
}

/// Offset-aware variant of [`evaluate_rpe`].
pub fn evaluate_rpe_with_offset(
    estimated: &[PoseSample],
    ground_truth: &[PoseSample],
    delta_seconds: f64,
    offset: f64,
    max_difference: f64,
) -> RpeResult {
    let pairs = associate_with_offset(estimated, ground_truth, offset, max_difference);
    let mut translational_errors = Vec::new();
    let mut rotational_errors = Vec::new();
    let mut end = 0usize;
    for start in 0..pairs.len() {
        end = end.max(start + 1);
        while end < pairs.len() && pairs[end].1.timestamp - pairs[start].1.timestamp < delta_seconds
        {
            end += 1;
        }
        if end >= pairs.len() {
            break;
        }
        let (estimate_start, truth_start) = pairs[start];
        let (estimate_end, truth_end) = pairs[end];
        let estimate_relative = estimate_start.pose.inverse() * estimate_end.pose;
        let truth_relative = truth_start.pose.inverse() * truth_end.pose;
        let delta = truth_relative.inverse() * estimate_relative;
        translational_errors.push(delta.translation.vector.norm());
        rotational_errors.push(delta.rotation.angle());
    }
    let rms = |values: &[f64]| {
        if values.is_empty() {
            f64::NAN
        } else {
            (values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64).sqrt()
        }
    };
    RpeResult {
        pairs: translational_errors.len(),
        translational_rmse: rms(&translational_errors),
        rotational_rmse: rms(&rotational_errors),
    }
}

fn solve_umeyama(
    estimated: &[Vector3<f64>],
    ground_truth: &[Vector3<f64>],
    align_scale: bool,
) -> (f64, Matrix3<f64>, Vector3<f64>) {
    let n = estimated.len() as f64;
    let estimated_mean = estimated.iter().sum::<Vector3<f64>>() / n;
    let truth_mean = ground_truth.iter().sum::<Vector3<f64>>() / n;
    let mut covariance = Matrix3::zeros();
    let mut estimated_variance = 0.0;
    for (estimate, truth) in estimated.iter().zip(ground_truth) {
        let estimate = estimate - estimated_mean;
        let truth = truth - truth_mean;
        covariance += truth * estimate.transpose();
        estimated_variance += estimate.norm_squared();
    }
    covariance /= n;
    estimated_variance /= n;
    if estimated_variance <= 1e-15 {
        return (1.0, Matrix3::identity(), truth_mean - estimated_mean);
    }
    let svd = covariance.svd(true, true);
    let Some(u) = svd.u else {
        return (1.0, Matrix3::identity(), truth_mean - estimated_mean);
    };
    let Some(v_t) = svd.v_t else {
        return (1.0, Matrix3::identity(), truth_mean - estimated_mean);
    };
    let mut sign = Matrix3::identity();
    if (u * v_t).determinant() < 0.0 {
        sign[(2, 2)] = -1.0;
    }
    let rotation = u * sign * v_t;
    let scale = if align_scale {
        (svd.singular_values.transpose() * sign.diagonal()).sum() / estimated_variance
    } else {
        1.0
    };
    let translation = truth_mean - scale * (rotation * estimated_mean);
    (scale, rotation, translation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Translation3;

    fn trajectory(length: usize, offset: Vector3<f64>, scale: f64) -> Vec<PoseSample> {
        (0..length)
            .map(|index| {
                let timestamp = index as f64;
                let position = Vector3::new(index as f64 * scale, 0.0, 0.0) + offset;
                PoseSample {
                    timestamp,
                    pose: Isometry3::from_parts(position.into(), UnitQuaternion::identity()),
                }
            })
            .collect()
    }

    #[test]
    fn umeyama_alignment_recovers_similarity() {
        let truth = trajectory(20, Vector3::new(2.0, -1.0, 0.5), 1.0);
        let estimate = trajectory(20, Vector3::new(4.0, 3.0, -2.0), 2.0);
        let ate = evaluate_ate(&estimate, &truth, true, 1e-9);
        assert_eq!(ate.associated, 20);
        assert!((ate.scale - 0.5).abs() < 1e-9);
        assert!(ate.rmse < 1e-9);
    }

    #[test]
    fn ate_reports_perturbation_and_rpe_pairs() {
        let truth = trajectory(20, Vector3::zeros(), 1.0);
        let mut estimate = truth.clone();
        for (index, sample) in estimate.iter_mut().enumerate() {
            sample.pose = Isometry3::from_parts(
                Translation3::new(index as f64, 0.0, 0.0),
                UnitQuaternion::identity(),
            );
        }
        estimate[10].pose.translation.y += 0.5;
        let ate = evaluate_ate(&estimate, &truth, false, 1e-9);
        assert!(ate.rmse > 0.0);
        let rpe = evaluate_rpe(&estimate, &truth, 2.0, 1e-9);
        assert!(rpe.pairs > 0);
        assert!(rpe.translational_rmse >= 0.0);
    }

    #[test]
    fn static_trajectory_has_finite_identity_alignment() {
        let truth = trajectory(5, Vector3::new(1.0, 2.0, 3.0), 0.0);
        let estimate = trajectory(5, Vector3::new(4.0, 5.0, 6.0), 0.0);
        let ate = evaluate_ate(&estimate, &truth, true, 1e-9);
        assert_eq!(ate.associated, 5);
        assert!((ate.scale - 1.0).abs() < 1e-12);
        assert!(ate.rmse.is_finite());
    }

    #[test]
    fn association_is_one_to_one_like_the_reference_script() {
        let estimate = vec![PoseSample {
            timestamp: 1.0,
            pose: Isometry3::identity(),
        }];
        let truth = vec![
            PoseSample {
                timestamp: 0.99,
                pose: Isometry3::identity(),
            },
            PoseSample {
                timestamp: 1.01,
                pose: Isometry3::identity(),
            },
        ];
        let matches = associate(&estimate, &truth, 0.02);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].1.timestamp, 0.99);

        let estimates = vec![
            PoseSample {
                timestamp: 1.001,
                pose: Isometry3::identity(),
            },
            PoseSample {
                timestamp: 1.002,
                pose: Isometry3::identity(),
            },
        ];
        let one_truth = vec![PoseSample {
            timestamp: 1.0,
            pose: Isometry3::identity(),
        }];
        assert_eq!(associate(&estimates, &one_truth, 0.02).len(), 1);
    }
}
