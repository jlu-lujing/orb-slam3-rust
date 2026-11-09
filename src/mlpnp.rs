//! Maximum-likelihood PnP on bearing vectors.
//!
//! This follows the structure of ORB-SLAM3's `MLPnPsolver`: a homogeneous
//! nullspace linear estimate is used to seed a robust RANSAC, followed by a
//! bearing-residual Gauss-Newton refinement. The original uses a large
//! hand-derived Jacobian; this port evaluates the same residual numerically,
//! which keeps the optimization stable without duplicating that fragile
//! expression.

use nalgebra::{DMatrix, DVector, Isometry3, Matrix3, Point3, UnitQuaternion, Vector3};
use opencv::core::{Point2f, Point3f};

use crate::camera_models::GeometricCamera;
use crate::pnp::{PnpCorrespondence, solve_pnp_ransac};

/// One image-to-world correspondence for [`solve_mlpnp_ransac`].
#[derive(Debug, Clone, Copy)]
pub struct MlpnpCorrespondence {
    pub world: Vector3<f64>,
    /// Unit bearing in camera coordinates.
    pub bearing: Vector3<f64>,
    pub observed_px: Point2f,
    /// Per-observation squared pixel error bound.
    pub max_error2: f64,
}

/// RANSAC output.
#[derive(Debug, Clone)]
pub struct MlpnpSolution {
    pub tcw: Isometry3<f64>,
    pub inliers: Vec<usize>,
}

/// ORB-SLAM3 uses a 99% confidence, six-point minimal set and five-pixel
/// squared threshold (`5.991 * level_sigma²`).
pub fn solve_mlpnp_ransac(
    camera: &dyn GeometricCamera,
    correspondences: &[MlpnpCorrespondence],
) -> Option<MlpnpSolution> {
    solve_mlpnp_ransac_with(camera, correspondences, 0.99, 8, 300, 6, 0.4)
}

#[allow(clippy::too_many_arguments)]
pub fn solve_mlpnp_ransac_with(
    camera: &dyn GeometricCamera,
    correspondences: &[MlpnpCorrespondence],
    probability: f64,
    min_inliers: usize,
    max_iterations: usize,
    min_set: usize,
    epsilon: f64,
) -> Option<MlpnpSolution> {
    let n = correspondences.len();
    if n < min_set.max(6) {
        return None;
    }

    let mut required = ((n as f64) * epsilon).ceil() as usize;
    required = required.max(min_inliers).max(min_set);
    let effective_epsilon = (required as f64 / n as f64).max(epsilon);
    let iterations = if required == n {
        1
    } else {
        let denominator = 1.0 - effective_epsilon.powi(3);
        if denominator <= 0.0 || denominator >= 1.0 {
            max_iterations
        } else {
            ((1.0 - probability).ln() / denominator.ln()).ceil() as usize
        }
    }
    .clamp(1, max_iterations);

    let mut rng = XorShift64::new(0xA24B_AED4_963E_E407 ^ n as u64);
    let mut best_tcw = None;
    let mut best_inliers = Vec::new();

    for _ in 0..iterations {
        let mut available: Vec<usize> = (0..n).collect();
        let mut sample = Vec::with_capacity(min_set);
        for _ in 0..min_set {
            let index = rng.range(available.len());
            sample.push(available.swap_remove(index));
        }
        let sample_bearings: Vec<_> = sample
            .iter()
            .map(|&index| correspondences[index].bearing)
            .collect();
        let sample_points: Vec<_> = sample
            .iter()
            .map(|&index| correspondences[index].world)
            .collect();
        let Some(tcw) = compute_pose(&sample_bearings, &sample_points) else {
            continue;
        };
        let inliers = check_inliers(camera, correspondences, &tcw);
        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
            best_tcw = Some(tcw);
        }
        if best_inliers.len() >= required {
            let refined = refine(camera, correspondences, &best_inliers, best_tcw?);
            if refined.inliers.len() >= required {
                return Some(refined);
            }
        }
    }

    let tcw = best_tcw?;
    if best_inliers.len() >= required {
        let refined = refine(camera, correspondences, &best_inliers, tcw);
        (refined.inliers.len() >= required).then_some(refined)
    } else {
        None
    }
}

fn refine(
    camera: &dyn GeometricCamera,
    correspondences: &[MlpnpCorrespondence],
    inlier_indices: &[usize],
    initial: Isometry3<f64>,
) -> MlpnpSolution {
    let bearings: Vec<_> = inlier_indices
        .iter()
        .map(|&index| correspondences[index].bearing)
        .collect();
    let points: Vec<_> = inlier_indices
        .iter()
        .map(|&index| correspondences[index].world)
        .collect();
    let tcw = refine_pose(&bearings, &points, initial);
    MlpnpSolution {
        inliers: check_inliers(camera, correspondences, &tcw),
        tcw,
    }
}

fn check_inliers(
    camera: &dyn GeometricCamera,
    correspondences: &[MlpnpCorrespondence],
    tcw: &Isometry3<f64>,
) -> Vec<usize> {
    correspondences
        .iter()
        .enumerate()
        .filter_map(|(index, correspondence)| {
            let camera_point = tcw * Point3::from(correspondence.world);
            let projected = camera.project(&Point3f::new(
                camera_point.x as f32,
                camera_point.y as f32,
                camera_point.z as f32,
            ));
            let dx = projected.x - correspondence.observed_px.x;
            let dy = projected.y - correspondence.observed_px.y;
            ((dx * dx + dy * dy) as f64 <= correspondence.max_error2).then_some(index)
        })
        .collect()
}

fn compute_pose(
    bearings: &[Vector3<f64>],
    world_points: &[Vector3<f64>],
) -> Option<Isometry3<f64>> {
    if bearings.len() < 6 || bearings.len() != world_points.len() {
        return None;
    }

    let nullspaces: Vec<_> = bearings.iter().map(|bearing| nullspace(*bearing)).collect();
    let mut points = world_points.to_vec();
    let mut planar = false;
    let mut eigen_rotation = Matrix3::identity();

    let mut covariance = Matrix3::zeros();
    for point in &points {
        covariance += point * point.transpose();
    }
    let svd = covariance.svd(true, true);
    let max_singular = svd.singular_values.iter().copied().fold(0.0, f64::max);
    let rank = svd
        .singular_values
        .iter()
        .filter(|&&value| value > max_singular * 1e-10)
        .count();
    if rank == 2 {
        planar = true;
        let eigen = covariance.symmetric_eigen();
        eigen_rotation = eigen.eigenvectors.transpose();
        for point in &mut points {
            *point = eigen_rotation * *point;
        }
    }

    let columns = if planar { 9 } else { 12 };
    let mut normal = DMatrix::<f64>::zeros(columns, columns);
    for (index, point) in points.iter().enumerate() {
        let (nr, ns) = &nullspaces[index];
        for basis in [nr, ns] {
            let mut row = DVector::<f64>::zeros(columns);
            if planar {
                row[0] = basis.x * point.y;
                row[1] = basis.x * point.z;
                row[2] = basis.y * point.y;
                row[3] = basis.y * point.z;
                row[4] = basis.z * point.y;
                row[5] = basis.z * point.z;
                row[6] = basis.x;
                row[7] = basis.y;
                row[8] = basis.z;
            } else {
                row[0] = basis.x * point.x;
                row[1] = basis.x * point.y;
                row[2] = basis.x * point.z;
                row[3] = basis.y * point.x;
                row[4] = basis.y * point.y;
                row[5] = basis.y * point.z;
                row[6] = basis.z * point.x;
                row[7] = basis.z * point.y;
                row[8] = basis.z * point.z;
                row[9] = basis.x;
                row[10] = basis.y;
                row[11] = basis.z;
            }
            normal += &row * row.transpose();
        }
    }

    let eigen = normal.symmetric_eigen();
    let smallest = eigen
        .eigenvalues
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| left.total_cmp(right))?
        .0;
    let solution = eigen.eigenvectors.column(smallest).into_owned();

    let candidates = if planar {
        planar_candidates(&solution, &eigen_rotation, bearings, world_points)
    } else {
        nonplanar_candidates(&solution, bearings, world_points)
    };
    let mut tcw = candidates.into_iter().min_by(|left, right| {
        angular_cost(left, bearings, world_points).total_cmp(&angular_cost(
            right,
            bearings,
            world_points,
        ))
    })?;
    if planar
        && let Some(fallback) = planar_pnp_fallback(bearings, world_points)
        && angular_cost(&fallback, bearings, world_points)
            < angular_cost(&tcw, bearings, world_points)
    {
        tcw = fallback;
    }
    Some(refine_pose(bearings, world_points, tcw))
}

fn planar_pnp_fallback(
    bearings: &[Vector3<f64>],
    world_points: &[Vector3<f64>],
) -> Option<Isometry3<f64>> {
    let correspondences: Vec<_> = world_points
        .iter()
        .zip(bearings)
        .filter_map(|(point, bearing)| {
            (bearing.z.abs() > 1e-8).then_some(PnpCorrespondence {
                world: Point3f::new(point.x as f32, point.y as f32, point.z as f32),
                bearing: Point2f::new(
                    (bearing.x / bearing.z) as f32,
                    (bearing.y / bearing.z) as f32,
                ),
            })
        })
        .collect();
    let solution = solve_pnp_ransac(&correspondences)?;
    Some(crate::converter::isometry3_f32_to_f64(&solution.tcw))
}

fn nonplanar_candidates(
    solution: &DVector<f64>,
    bearings: &[Vector3<f64>],
    world_points: &[Vector3<f64>],
) -> Vec<Isometry3<f64>> {
    let linear_rotation = Matrix3::new(
        solution[0],
        solution[3],
        solution[6],
        solution[1],
        solution[4],
        solution[7],
        solution[2],
        solution[5],
        solution[8],
    );
    let scale = 1.0
        / (linear_rotation.column(0).norm()
            * linear_rotation.column(1).norm()
            * linear_rotation.column(2).norm())
        .abs()
        .cbrt();
    let rotation = nearest_rotation(linear_rotation);
    let translation = rotation * (scale * Vector3::new(solution[9], solution[10], solution[11]));
    vec![pose(rotation, translation), pose(rotation, -translation)]
        .into_iter()
        .filter(|candidate| angular_cost(candidate, bearings, world_points).is_finite())
        .collect()
}

fn planar_candidates(
    solution: &DVector<f64>,
    eigen_rotation: &Matrix3<f64>,
    bearings: &[Vector3<f64>],
    world_points: &[Vector3<f64>],
) -> Vec<Isometry3<f64>> {
    let mut raw = Matrix3::new(
        0.0,
        solution[0],
        solution[1],
        0.0,
        solution[2],
        solution[3],
        0.0,
        solution[4],
        solution[5],
    );
    raw.set_column(0, &raw.column(1).cross(&raw.column(2)));
    raw.transpose_mut();
    let scale = 1.0 / (raw.column(1).norm() * raw.column(2).norm()).abs().sqrt();
    let base = nearest_rotation(raw);
    let in_eigen = eigen_rotation.transpose() * base;
    let rotations = [base, base.transpose(), in_eigen, in_eigen.transpose()];
    let translation = scale * Vector3::new(solution[6], solution[7], solution[8]);
    let mut candidates = Vec::new();
    for rotation in rotations {
        for translation in [
            translation,
            -translation,
            rotation * translation,
            -(rotation * translation),
        ] {
            candidates.push(pose(rotation, translation));
        }
    }
    candidates
        .into_iter()
        .filter(|candidate| angular_cost(candidate, bearings, world_points).is_finite())
        .collect()
}

fn refine_pose(
    bearings: &[Vector3<f64>],
    world_points: &[Vector3<f64>],
    initial: Isometry3<f64>,
) -> Isometry3<f64> {
    let mut x = DVector::<f64>::from_row_slice(&[
        initial.rotation.scaled_axis().x,
        initial.rotation.scaled_axis().y,
        initial.rotation.scaled_axis().z,
        initial.translation.x,
        initial.translation.y,
        initial.translation.z,
    ]);
    let nullspaces: Vec<_> = bearings.iter().map(|bearing| nullspace(*bearing)).collect();
    for _ in 0..8 {
        let residual = bearing_residual(&x, &nullspaces, world_points);
        let mut jacobian = DMatrix::<f64>::zeros(residual.len(), 6);
        for column in 0..6 {
            let step = 1e-7;
            let mut plus = x.clone();
            let mut minus = x.clone();
            plus[column] += step;
            minus[column] -= step;
            let derivative = (bearing_residual(&plus, &nullspaces, world_points)
                - bearing_residual(&minus, &nullspaces, world_points))
                / (2.0 * step);
            jacobian.set_column(column, &derivative);
        }
        let lhs = jacobian.transpose() * &jacobian;
        let rhs = jacobian.transpose() * residual;
        let Some(dx) = lhs.lu().solve(&rhs) else {
            break;
        };
        if dx.iter().any(|value| !value.is_finite()) || dx.norm() > 10.0 {
            break;
        }
        x -= &dx;
        if dx.norm() < 1e-8 {
            break;
        }
    }
    pose(
        UnitQuaternion::from_scaled_axis(Vector3::new(x[0], x[1], x[2]))
            .to_rotation_matrix()
            .into_inner(),
        Vector3::new(x[3], x[4], x[5]),
    )
}

fn bearing_residual(
    x: &DVector<f64>,
    nullspaces: &[(Vector3<f64>, Vector3<f64>)],
    world_points: &[Vector3<f64>],
) -> DVector<f64> {
    let rotation =
        UnitQuaternion::from_scaled_axis(Vector3::new(x[0], x[1], x[2])).to_rotation_matrix();
    let translation = Vector3::new(x[3], x[4], x[5]);
    let mut residual = DVector::<f64>::zeros(2 * world_points.len());
    for (index, point) in world_points.iter().enumerate() {
        let camera_point = rotation * point + translation;
        let unit = if camera_point.norm() > 1e-12 {
            camera_point.normalize()
        } else {
            Vector3::zeros()
        };
        residual[2 * index] = nullspaces[index].0.dot(&unit);
        residual[2 * index + 1] = nullspaces[index].1.dot(&unit);
    }
    residual
}

fn angular_cost(
    tcw: &Isometry3<f64>,
    bearings: &[Vector3<f64>],
    world_points: &[Vector3<f64>],
) -> f64 {
    world_points
        .iter()
        .zip(bearings)
        .map(|(point, bearing)| {
            let camera_point = tcw * Point3::from(*point);
            if camera_point.coords.norm() <= 1e-12 {
                f64::INFINITY
            } else {
                1.0 - camera_point.coords.normalize().dot(bearing)
            }
        })
        .sum()
}

fn pose(rotation: Matrix3<f64>, translation: Vector3<f64>) -> Isometry3<f64> {
    Isometry3::from_parts(
        translation.into(),
        UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(
            nearest_rotation(rotation),
        )),
    )
}

fn nearest_rotation(matrix: Matrix3<f64>) -> Matrix3<f64> {
    let svd = matrix.svd(true, true);
    let Some(u) = svd.u else {
        return Matrix3::identity();
    };
    let Some(v_t) = svd.v_t else {
        return Matrix3::identity();
    };
    let mut rotation = u * v_t;
    if rotation.determinant() < 0.0 {
        let mut u = u;
        u.set_column(2, &(-u.column(2)));
        rotation = u * v_t;
    }
    rotation
}

fn nullspace(bearing: Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let bearing = if bearing.norm() > 1e-12 {
        bearing.normalize()
    } else {
        Vector3::z()
    };
    let helper = if bearing.x.abs() < 0.9 {
        Vector3::x()
    } else {
        Vector3::y()
    };
    let first = bearing.cross(&helper).normalize();
    let second = bearing.cross(&first).normalize();
    (first, second)
}

struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn range(&mut self, upper: usize) -> usize {
        (self.next() % upper as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera_models::pinhole::Pinhole;
    use nalgebra::Translation3;

    fn correspondences(
        expected: Isometry3<f64>,
        points: &[Vector3<f64>],
        planar: bool,
        outlier: Option<usize>,
    ) -> Vec<MlpnpCorrespondence> {
        let camera = Pinhole::with_params(vec![400.0, 400.0, 320.0, 240.0]);
        points
            .iter()
            .enumerate()
            .map(|(index, point)| {
                let mut point = *point;
                if planar {
                    point.z = 0.0;
                }
                let camera_point = expected * Point3::from(point);
                let bearing = camera_point.coords.normalize();
                let projected = camera.project(&Point3f::new(
                    camera_point.x as f32,
                    camera_point.y as f32,
                    camera_point.z as f32,
                ));
                MlpnpCorrespondence {
                    world: point,
                    bearing,
                    observed_px: Point2f::new(
                        projected.x + if outlier == Some(index) { 20.0 } else { 0.0 },
                        projected.y,
                    ),
                    max_error2: 5.991,
                }
            })
            .collect()
    }

    #[test]
    fn mlpnp_recovers_nonplanar_pose_and_rejects_outlier() {
        let expected = Isometry3::from_parts(
            Translation3::new(0.2, -0.1, 0.3),
            UnitQuaternion::from_euler_angles(0.03, -0.02, 0.05),
        );
        let points = [
            Vector3::new(-1.0, -0.5, 4.0),
            Vector3::new(1.0, -0.5, 4.2),
            Vector3::new(1.0, 0.5, 4.4),
            Vector3::new(-1.0, 0.5, 4.6),
            Vector3::new(-0.8, -0.4, 5.0),
            Vector3::new(0.8, -0.4, 5.2),
            Vector3::new(0.8, 0.4, 5.4),
            Vector3::new(-0.8, 0.4, 5.6),
            Vector3::new(-0.5, -0.2, 4.8),
            Vector3::new(0.5, 0.2, 5.1),
        ];
        let correspondences = correspondences(expected, &points, false, Some(3));
        let camera = Pinhole::with_params(vec![400.0, 400.0, 320.0, 240.0]);
        let solution = solve_mlpnp_ransac(&camera, &correspondences).expect("MLPnP should solve");
        let delta = solution.tcw.inverse() * expected;
        assert!(delta.rotation.angle().abs() < 1e-2);
        assert!(delta.translation.vector.norm() < 1e-2);
        assert_eq!(solution.inliers.len(), points.len() - 1);
        assert!(!solution.inliers.contains(&3));
    }

    #[test]
    fn mlpnp_recovers_planar_pose() {
        let expected = Isometry3::from_parts(
            Translation3::new(0.1, 0.2, 0.4),
            UnitQuaternion::from_euler_angles(0.02, 0.01, -0.03),
        );
        let points = [
            Vector3::new(-1.0, -0.5, 0.0),
            Vector3::new(1.0, -0.5, 0.0),
            Vector3::new(1.0, 0.5, 0.0),
            Vector3::new(-1.0, 0.5, 0.0),
            Vector3::new(-0.6, -0.2, 0.0),
            Vector3::new(0.6, -0.2, 0.0),
            Vector3::new(0.6, 0.2, 0.0),
            Vector3::new(-0.6, 0.2, 0.0),
        ];
        let correspondences = correspondences(expected, &points, true, None);
        let camera = Pinhole::with_params(vec![400.0, 400.0, 320.0, 240.0]);
        let solution =
            solve_mlpnp_ransac(&camera, &correspondences).expect("planar MLPnP should solve");
        let delta = solution.tcw.inverse() * expected;
        assert!(delta.rotation.angle().abs() < 2e-2);
        assert!(delta.translation.vector.norm() < 2e-2);
    }
}
