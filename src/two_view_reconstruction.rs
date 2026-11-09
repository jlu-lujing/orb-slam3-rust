use nalgebra::{
    DMatrix, Isometry3, Matrix3, Rotation3, SMatrix, SVD, Translation3, UnitQuaternion, Vector3,
};
use opencv::core::{KeyPoint, Point2f, Point3f};
use opencv::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::index;

use crate::geometric_tools::triangulate;

#[derive(Debug, Clone)]
pub struct TwoViewReconstruction {
    // Keypoints reference frame (1)
    keys1: Vec<KeyPoint>,
    // Keypoints current frame (2)
    keys2: Vec<KeyPoint>,
    // Matches from reference to current
    matches12: Vec<(usize, usize)>,
    matched1: Vec<bool>,
    // Calibration
    k: Matrix3<f32>,
    // Standard deviation and variance
    sigma: f32,
    sigma2: f32,
    // Ransac max iterations
    max_iterations: u32,
    // Ransac sets
    sets: Vec<Vec<usize>>,
}

/// Wrapper to send &KeyPoint across thread boundaries.
/// KeyPoint contains a `*mut c_void` from the OpenCV C++ FFI, making it !Sync.
/// The fields we access (pt, size, angle, etc.) are plain data that is safe to read
/// concurrently. We only share immutable references, so no data races should occur in this context.
#[derive(Clone, Copy)]
struct SendKeyPoints<'a>(&'a [KeyPoint]);
unsafe impl Send for SendKeyPoints<'_> {}
unsafe impl Sync for SendKeyPoints<'_> {}

pub struct ReconstructResult {
    t21: Isometry3<f32>,
    triangulated: Vec<bool>,
}

impl TwoViewReconstruction {
    pub fn new(k: Matrix3<f32>) -> Self {
        let sigma = 1.0;
        let iterations = 200;
        TwoViewReconstruction {
            keys1: Vec::new(),
            keys2: Vec::new(),
            matches12: Vec::new(),
            matched1: Vec::new(),
            k: k,
            sigma: sigma,
            sigma2: sigma * sigma,
            max_iterations: iterations,
            sets: Vec::new(),
        }
    }

    // Computes in parallel a fundamental matrix and a homography
    // Selects a model and tries to recover the motion and the structure from motion
    pub fn reconstruct(
        &mut self,
        keys1: &[KeyPoint],
        keys2: &[KeyPoint],
        matches12: &[Option<usize>],
    ) -> Option<ReconstructResult> {
        self.keys1 = keys1.to_vec();
        self.keys2 = keys2.to_vec();

        // Fill structure with current keypoints and matches with reference frame
        self.matches12.clear();
        self.matches12.reserve(self.keys2.len());
        self.matched1.resize(self.keys1.len(), false);
        for (i, matched) in matches12.iter().enumerate() {
            if let Some(&j) = matched.as_ref() {
                self.matches12.push((i, j));
                self.matched1[i] = true;
            } else {
                self.matched1[i] = false;
            }
        }

        let n = self.matches12.len();

        // Generate sets of 8 points for each RANSAC iteration
        self.sets = (0..self.max_iterations).map(|_| vec![0usize; 8]).collect();
        let mut rng = StdRng::seed_from_u64(0);
        for set in &mut self.sets {
            for (j, idx) in index::sample(&mut rng, n, 8).into_iter().enumerate() {
                set[j] = idx;
            }
        }

        let matches12 = &self.matches12;
        let keys1 = SendKeyPoints(&self.keys1);
        let keys2 = SendKeyPoints(&self.keys2);
        let sets = &self.sets;
        let max_iterations = self.max_iterations;
        let sigma = self.sigma;
        let (homography, fundamental) = rayon::join(
            move || find_homography(matches12, keys1, keys2, sets, max_iterations, sigma),
            move || find_fundamental(matches12, keys1, keys2, sets, max_iterations, sigma),
        );

        // Compute ratio of scores
        if homography.score + fundamental.score == 0. {
            return None;
        }
        let rh = homography.score / (homography.score + fundamental.score);

        // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
        const MIN_PARALLAX: f32 = 1.0f32;
        const MIN_TRIANGULATED: usize = 50;
        if rh > 0.5 {
            self.reconstruct_h(homography, MIN_PARALLAX, MIN_TRIANGULATED)
        } else {
            self.reconstruct_f(fundamental, MIN_PARALLAX, MIN_TRIANGULATED)
        }
    }

    fn reconstruct_h(
        &self,
        result: HomographyResult,
        min_parallax: f32,
        min_triangulated: usize,
    ) -> Option<ReconstructResult> {
        let n = result.matches_inliers.iter().filter(|i| **i).count();

        // We recover 8 motion hypotheses using the method of Faugeras et al.
        // Motion and structure from motion in a piecewice planar environment
        // International Journal of Pattern Recognition nad Artificial Intelligence, 1988
        let k_inv = &self.k.try_inverse().unwrap();
        let a = k_inv * result.h21 * &self.k;

        let svd = a.svd(true, true);
        let u = svd.u.unwrap();
        let v_t = svd.v_t.unwrap();
        let v = v_t.transpose();
        let w = svd.singular_values;

        let s = u.determinant() * v_t.determinant();

        let d1 = w[0];
        let d2 = w[1];
        let d3 = w[2];

        if d1 / d2 < 1.00001 || d2 / d3 < 1.00001 {
            return None;
        }

        let mut vr = Vec::<Matrix3<f32>>::with_capacity(8);
        let (mut vt, mut vn) = (
            Vec::<Vector3<f32>>::with_capacity(8),
            Vec::<Vector3<f32>>::with_capacity(8),
        );

        //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
        let aux1 = ((d1 * d1 - d2 * d2) / (d1 * d1 - d3 * d3)).sqrt();
        let aux3 = ((d2 * d2 - d3 * d3) / (d1 * d1 - d3 * d3)).sqrt();
        let x1 = [aux1, aux1, -aux1, -aux1];
        let x3 = [aux3, -aux3, aux3, -aux3];

        //case d'=d2
        let aux_stheta = ((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 + d3) * d2).sqrt();

        let ctheta = (d2 * d2 + d1 * d3) / ((d1 + d3) * d2);
        let stheta = [aux_stheta, -aux_stheta, -aux_stheta, aux_stheta];

        for i in 0..4 {
            let mut rp = Matrix3::<f32>::zeros();
            rp[(0, 0)] = ctheta;
            rp[(0, 2)] = -stheta[i];
            rp[(1, 1)] = 1.;
            rp[(2, 0)] = stheta[i];
            rp[(2, 2)] = ctheta;

            let r = s * u * rp * v_t;
            vr.push(r);

            let mut tp = Vector3::<f32>::new(x1[i], 0., -x3[i]);
            tp *= d1 - d3;

            let t = u * tp;
            vt.push(t / t.norm());

            let np = Vector3::<f32>::new(x1[i], 0., x3[i]);

            let mut n = v * np;
            if n[2] < 0. {
                n = -n;
            }
            vn.push(n);
        }

        //case d'=-d2
        let aux_sphi = ((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 - d3) * d2).sqrt();
        let cphi = (d1 * d3 - d2 * d2) / ((d1 - d3) * d2);
        let sphi = [aux_sphi, -aux_sphi, -aux_sphi, aux_sphi];

        for i in 0..4 {
            let mut rp = Matrix3::<f32>::zeros();
            rp[(0, 0)] = cphi;
            rp[(0, 2)] = sphi[i];
            rp[(1, 1)] = -1.;
            rp[(2, 0)] = sphi[i];
            rp[(2, 2)] = -cphi;

            let r = s * u * rp * v_t;
            vr.push(r);

            let mut tp = Vector3::<f32>::new(x1[i], 0., x3[i]);
            tp *= d1 + d3;

            let t = u * tp;
            vt.push(t / t.norm());

            let np = Vector3::<f32>::new(x1[i], 0., x3[i]);

            let mut n = v * np;
            if n[2] < 0. {
                n = -n;
            }
            vn.push(n);
        }

        let mut best_good = 0;
        let mut second_best_good = 0;
        let mut best_solution_idx = 0;
        let mut best_parallax = -1.0f32;
        let mut best_triangulated = Vec::<bool>::new();

        // Instead of applying the visibility constraints proposed in the Faugeras paper (which could fail for points seen with low parallax)
        // We reconstruct all hypotheses and check in terms of triangulated points and parallax
        for i in 0..8 {
            let result = check_rt(
                &vr[i],
                &vt[i],
                &self.keys1,
                &self.keys2,
                &self.matches12,
                &result.matches_inliers,
                &self.k,
                4.0 * self.sigma2,
            );
            if result.num_good > best_good {
                second_best_good = best_good;
                best_good = result.num_good;
                best_solution_idx = i;
                best_parallax = result.parallax;
                best_triangulated = result.good;
            } else if result.num_good > second_best_good {
                second_best_good = result.num_good;
            }
        }

        if second_best_good * 4 < best_good * 3
            && best_parallax >= min_parallax
            && best_good > min_triangulated
            && best_good * 10 > 9 * n
        {
            Some(ReconstructResult {
                t21: pose_from_rt(&vr[best_solution_idx], &vt[best_solution_idx]),
                triangulated: best_triangulated,
            })
        } else {
            None
        }
    }

    fn reconstruct_f(
        &self,
        result: FundamentalResult,
        min_parallax: f32,
        min_triangulated: usize,
    ) -> Option<ReconstructResult> {
        let n = result.matches_inliers.iter().filter(|i| **i).count();

        // Compute essential matrix from fundamental matrix
        let e21 = self.k.transpose() * result.f21 * self.k;

        // Recover the 4 motion hypotheses
        let (r1, r2, t) = decompose_e(&e21);
        let t1 = t;
        let t2 = -t;

        // Reconstruct with the 4 hypotheses and check
        let sigma = self.sigma2 * 4.0;
        let hypotheses = [(&r1, &t1), (&r2, &t1), (&r1, &t2), (&r2, &t2)].map(|(r, t)| {
            (
                check_rt(
                    r,
                    t,
                    &self.keys1,
                    &self.keys2,
                    &self.matches12,
                    &result.matches_inliers,
                    &self.k,
                    sigma,
                ),
                r,
                t,
            )
        });

        let max_good = hypotheses
            .iter()
            .map(|(res, _, _)| res.num_good)
            .max()
            .unwrap();

        let min_good = ((9 * n) / 10).max(min_triangulated);

        let similar = hypotheses
            .iter()
            .filter(|(res, _, _)| res.num_good * 10 > max_good * 7)
            .count();

        // If there is not a clear winner or not enough triangulated points, reject initialization
        if max_good < min_good || similar > 1 {
            return None;
        }

        // Pick the first hypothesis with max_good
        if let Some((chosen, r, t)) = hypotheses
            .iter()
            .find(|(res, _, _)| res.num_good == max_good)
        {
            if chosen.parallax > min_parallax {
                return Some(ReconstructResult {
                    t21: pose_from_rt(r, t),
                    triangulated: chosen.good.clone(),
                });
            }
        }

        None
    }
}

fn normalize(keys: &[KeyPoint]) -> (Vec<Point2f>, Matrix3<f32>) {
    let n = keys.len();
    let mut normalized_points = Vec::<Point2f>::with_capacity(n);

    let (mut mean_x, mut mean_y) = (0.0f32, 0.0f32);
    for it in keys {
        mean_x += it.pt().x;
        mean_y += it.pt().y;
    }
    mean_x = mean_x / n as f32;
    mean_y = mean_y / n as f32;

    let (mut mean_dev_x, mut mean_dev_y) = (0.0f32, 0.0f32);
    for i in 0..n {
        let p = Point2f::new(keys[i].pt().x - mean_x, keys[i].pt().y - mean_y);
        mean_dev_x += p.x.abs();
        mean_dev_y += p.y.abs();
        normalized_points.push(p);
    }
    mean_dev_x = mean_dev_x / n as f32;
    mean_dev_y = mean_dev_y / n as f32;

    // Guard against degenerate (collinear) point sets where all coordinates
    // along one axis are identical, producing zero mean deviation. Without
    // this check `1.0 / 0.0 = inf` poisons every downstream matrix with NaN,
    // causing the iterative SVD to loop forever.
    if mean_dev_x < f32::EPSILON {
        mean_dev_x = 1.0;
    }
    if mean_dev_y < f32::EPSILON {
        mean_dev_y = 1.0;
    }

    let s_x = 1.0 / mean_dev_x;
    let s_y = 1.0 / mean_dev_y;
    for it in normalized_points.iter_mut() {
        it.x = it.x * s_x;
        it.y = it.y * s_y;
    }

    let mut t = Matrix3::<f32>::zeros();
    t[(0, 0)] = s_x;
    t[(1, 1)] = s_y;
    t[(0, 2)] = -mean_x * s_x;
    t[(1, 2)] = -mean_y * s_y;
    t[(2, 2)] = 1.0;

    (normalized_points, t)
}

fn compute_h21(p1: &[Point2f], p2: &[Point2f]) -> Matrix3<f32> {
    let n = p1.len();
    let mut a = DMatrix::<f32>::zeros(2 * n, 9);

    for i in 0..n {
        let u1 = p1[i].x;
        let v1 = p1[i].y;
        let u2 = p2[i].x;
        let v2 = p2[i].y;

        a[(2 * i, 0)] = 0.0;
        a[(2 * i, 1)] = 0.0;
        a[(2 * i, 2)] = 0.0;
        a[(2 * i, 3)] = -u1;
        a[(2 * i, 4)] = -v1;
        a[(2 * i, 5)] = -1.0;
        a[(2 * i, 6)] = v2 * u1;
        a[(2 * i, 7)] = v2 * v1;
        a[(2 * i, 8)] = v2;

        a[(2 * i + 1, 0)] = u1;
        a[(2 * i + 1, 1)] = v1;
        a[(2 * i + 1, 2)] = 1.0;
        a[(2 * i + 1, 3)] = 0.0;
        a[(2 * i + 1, 4)] = 0.0;
        a[(2 * i + 1, 5)] = 0.0;
        a[(2 * i + 1, 6)] = -u2 * u1;
        a[(2 * i + 1, 7)] = -u2 * v1;
        a[(2 * i + 1, 8)] = -u2;
    }

    // A = U Σ Vᵀ; homography is the right singular vector for the smallest singular value.
    // nalgebra stores Vᵀ (`v_t`); column c of V is row c of Vᵀ. With descending σ, that is
    // the last row of `v_t` (Eigen full V, col(8); thin 8×9 case uses col(7) — same index k).
    let svd = a.svd(false, true);
    let v_t = svd.v_t.expect("svd: V^T not computed");
    let k = svd.singular_values.len() - 1;
    Matrix3::from_row_slice(&[
        v_t[(k, 0)],
        v_t[(k, 1)],
        v_t[(k, 2)],
        v_t[(k, 3)],
        v_t[(k, 4)],
        v_t[(k, 5)],
        v_t[(k, 6)],
        v_t[(k, 7)],
        v_t[(k, 8)],
    ])
}

fn compute_f21(p1: &[Point2f], p2: &[Point2f]) -> Matrix3<f32> {
    let n = p1.len();
    // Pad rows to at least 9 so the thin SVD produces a full 9×9 Vᵀ.
    // nalgebra only computes thin SVD: Vᵀ is min(rows,9) × 9. With fewer
    // than 9 rows the null-space vector (column 8 of V) is never returned.
    let nrows = n.max(9);
    let mut a = DMatrix::<f32>::zeros(nrows, 9);

    for i in 0..n {
        let u1 = p1[i].x;
        let v1 = p1[i].y;
        let u2 = p2[i].x;
        let v2 = p2[i].y;

        a[(i, 0)] = u2 * u1;
        a[(i, 1)] = u2 * v1;
        a[(i, 2)] = u2;
        a[(i, 3)] = v2 * u1;
        a[(i, 4)] = v2 * v1;
        a[(i, 5)] = v2;
        a[(i, 6)] = u1;
        a[(i, 7)] = v1;
        a[(i, 8)] = 1.;
    }

    let svd = a.svd(false, true);
    let v_t = svd.v_t.expect("svd: V^T not computed");
    let k = v_t.nrows() - 1;
    let f_pre = Matrix3::from_row_slice(&[
        v_t[(k, 0)],
        v_t[(k, 1)],
        v_t[(k, 2)],
        v_t[(k, 3)],
        v_t[(k, 4)],
        v_t[(k, 5)],
        v_t[(k, 6)],
        v_t[(k, 7)],
        v_t[(k, 8)],
    ]);
    let svd2 = f_pre.svd(true, true);
    let u = svd2.u.expect("svd2: U not computed");
    let v_t2 = svd2.v_t.expect("svd2: V^T not computed");
    let mut w = svd2.singular_values;
    w[2] = 0.0;

    u * Matrix3::from_diagonal(&w) * v_t2
}

struct HomographyResult {
    score: f32,
    matches_inliers: Vec<bool>,
    h21: Matrix3<f32>,
}
fn find_homography(
    matches12: &[(usize, usize)],
    keys1: SendKeyPoints<'_>,
    keys2: SendKeyPoints<'_>,
    sets: &[Vec<usize>],
    max_iterations: u32,
    sigma: f32,
) -> HomographyResult {
    let (keys1, keys2) = (keys1.0, keys2.0);
    let n = matches12.len();

    let (pn1, t1) = normalize(keys1);
    let (pn2, t2) = normalize(keys2);
    let t2inv = t2.try_inverse().unwrap();

    // best results
    let mut score = 0.0f32;
    let mut matches_inliers = vec![false; n];

    // iteration variables
    let (mut pn1_i, mut pn2_i) = (vec![Point2f::default(); 8], vec![Point2f::default(); 8]);
    let (mut h21_i, mut h12_i);
    let mut current_inliers = vec![false; n];
    let mut current_score;
    let mut h21 = Matrix3::<f32>::default();

    // perform all RANSAC iterations and save solution with highest score
    for i in 0..max_iterations as usize {
        // select a minimum set
        for j in 0..8 {
            let idx = sets[i][j];
            pn1_i[j] = pn1[matches12[idx].0];
            pn2_i[j] = pn2[matches12[idx].1];
        }

        let hn = compute_h21(&pn1_i, &pn2_i);
        h21_i = t2inv * hn * t1;
        let Some(inv) = h21_i.try_inverse() else {
            continue;
        };
        h12_i = inv;

        (current_score, current_inliers) =
            check_homography(matches12, keys1, keys2, &h21_i, &h12_i, sigma);

        if current_score > score {
            h21 = h21_i;
            matches_inliers = current_inliers;
            score = current_score;
        }
    }

    HomographyResult {
        score,
        matches_inliers,
        h21,
    }
}

fn check_homography(
    matches12: &[(usize, usize)],
    keys1: &[KeyPoint],
    keys2: &[KeyPoint],
    h21: &Matrix3<f32>,
    h12: &Matrix3<f32>,
    sigma: f32,
) -> (f32, Vec<bool>) {
    const TH: f32 = 5.991;
    let inv_sigma_sq = 1.0 / (sigma * sigma);
    let n = matches12.len();
    let mut score = 0.0f32;
    let mut inliers = vec![false; n];

    for (i, &(i1, i2)) in matches12.iter().enumerate() {
        let u1 = keys1[i1].pt().x;
        let v1 = keys1[i1].pt().y;
        let u2 = keys2[i2].pt().x;
        let v2 = keys2[i2].pt().y;

        // x2in1 = H12 * x2 (reprojection in image 1)
        let w2in1_inv = 1.0 / (h12[(2, 0)] * u2 + h12[(2, 1)] * v2 + h12[(2, 2)]);
        let u2in1 = (h12[(0, 0)] * u2 + h12[(0, 1)] * v2 + h12[(0, 2)]) * w2in1_inv;
        let v2in1 = (h12[(1, 0)] * u2 + h12[(1, 1)] * v2 + h12[(1, 2)]) * w2in1_inv;
        let chi_sq1 = ((u1 - u2in1).powi(2) + (v1 - v2in1).powi(2)) * inv_sigma_sq;

        let mut is_inlier = true;
        if chi_sq1 > TH {
            is_inlier = false;
        } else {
            score += TH - chi_sq1;
        }

        // x1in2 = H21 * x1 (reprojection in image 2)
        let w1in2_inv = 1.0 / (h21[(2, 0)] * u1 + h21[(2, 1)] * v1 + h21[(2, 2)]);
        let u1in2 = (h21[(0, 0)] * u1 + h21[(0, 1)] * v1 + h21[(0, 2)]) * w1in2_inv;
        let v1in2 = (h21[(1, 0)] * u1 + h21[(1, 1)] * v1 + h21[(1, 2)]) * w1in2_inv;
        let chi_sq2 = ((u2 - u1in2).powi(2) + (v2 - v1in2).powi(2)) * inv_sigma_sq;

        if chi_sq2 > TH {
            is_inlier = false;
        } else {
            score += TH - chi_sq2;
        }

        inliers[i] = is_inlier;
    }

    (score, inliers)
}

struct FundamentalResult {
    score: f32,
    matches_inliers: Vec<bool>,
    f21: Matrix3<f32>,
}
fn find_fundamental(
    matches12: &[(usize, usize)],
    keys1: SendKeyPoints<'_>,
    keys2: SendKeyPoints<'_>,
    sets: &[Vec<usize>],
    max_iterations: u32,
    sigma: f32,
) -> FundamentalResult {
    let (keys1, keys2) = (keys1.0, keys2.0);
    let n = matches12.len();

    let (pn1, t1) = normalize(keys1);
    let (pn2, t2) = normalize(keys2);
    let t2t = t2.transpose();

    // best result
    let mut score = 0.0;
    let mut matches_inliers = vec![false; n];

    // iteration variables
    let (mut pn1_i, mut pn2_i) = (vec![Point2f::default(); 8], vec![Point2f::default(); 8]);
    let mut f21i;
    let mut current_inliers = vec![false; n];
    let mut current_score;
    let mut f21 = Matrix3::<f32>::default();

    // perform all RANSAC iterations and save solution with highest score
    for i in 0..max_iterations as usize {
        // select a minimum set
        for j in 0..8 {
            let idx = sets[i][j];
            pn1_i[j] = pn1[matches12[idx].0];
            pn2_i[j] = pn2[matches12[idx].1];
        }

        let f = compute_f21(&pn1_i, &pn2_i);
        f21i = t2t * f * t1;

        (current_score, current_inliers) = check_fundamental(matches12, keys1, keys2, &f21i, sigma);

        if current_score > score {
            f21 = f21i;
            matches_inliers = current_inliers;
            score = current_score;
        }
    }

    FundamentalResult {
        score,
        matches_inliers,
        f21,
    }
}

fn check_fundamental(
    matches12: &[(usize, usize)],
    keys1: &[KeyPoint],
    keys2: &[KeyPoint],
    f21: &Matrix3<f32>,
    sigma: f32,
) -> (f32, Vec<bool>) {
    const TH: f32 = 3.841;
    const TH_SCORE: f32 = 5.991;
    let inv_sigma_sq = 1.0 / (sigma * sigma);
    let n = matches12.len();
    let mut score = 0.0f32;
    let mut inliers = vec![false; n];

    for (i, &(i1, i2)) in matches12.iter().enumerate() {
        let u1 = keys1[i1].pt().x;
        let v1 = keys1[i1].pt().y;
        let u2 = keys2[i2].pt().x;
        let v2 = keys2[i2].pt().y;

        let mut is_inlier = true;

        // l2 = F21 * x1 = (a2, b2, c2) — epipolar line in image 2 / reprojection error
        let a2 = f21[(0, 0)] * u1 + f21[(0, 1)] * v1 + f21[(0, 2)];
        let b2 = f21[(1, 0)] * u1 + f21[(1, 1)] * v1 + f21[(1, 2)];
        let c2 = f21[(2, 0)] * u1 + f21[(2, 1)] * v1 + f21[(2, 2)];
        let num2 = a2 * u2 + b2 * v2 + c2;
        let chi_sq1 = (num2 * num2 / (a2 * a2 + b2 * b2)) * inv_sigma_sq;

        if chi_sq1 > TH {
            is_inlier = false;
        } else {
            score += TH_SCORE - chi_sq1;
        }

        // l1 = x2^T * F21 = (a1, b1, c1) — epipolar line in image 1 / reprojection error
        let a1 = f21[(0, 0)] * u2 + f21[(1, 0)] * v2 + f21[(2, 0)];
        let b1 = f21[(0, 1)] * u2 + f21[(1, 1)] * v2 + f21[(2, 1)];
        let c1 = f21[(0, 2)] * u2 + f21[(1, 2)] * v2 + f21[(2, 2)];
        let num1 = a1 * u1 + b1 * v1 + c1;
        let chi_sq2 = (num1 * num1 / (a1 * a1 + b1 * b1)) * inv_sigma_sq;

        if chi_sq2 > TH {
            is_inlier = false;
        } else {
            score += TH_SCORE - chi_sq2;
        }

        inliers[i] = is_inlier;
    }

    (score, inliers)
}

struct RTResult {
    good: Vec<bool>,
    num_good: usize,
    p3d: Vec<Point3f>,
    parallax: f32,
}
fn check_rt(
    r: &Matrix3<f32>,
    t: &Vector3<f32>,
    keys1: &[KeyPoint],
    keys2: &[KeyPoint],
    matches12: &[(usize, usize)],
    matches_inliers: &[bool],
    k: &Matrix3<f32>,
    th2: f32,
) -> RTResult {
    // Calibration parameters
    let fx = k[(0, 0)];
    let fy = k[(1, 1)];
    let cx = k[(0, 2)];
    let cy = k[(1, 2)];

    let mut good = vec![false; keys1.len()];
    let mut p3d = vec![Point3f::new(0.0, 0.0, 0.0); keys1.len()];

    let mut v_cos_parallax = Vec::<f32>::with_capacity(keys1.len());

    // camera 1 projection matrix k[I|0]
    let mut p1 = SMatrix::<f32, 3, 4>::zeros();
    p1.fixed_view_mut::<3, 3>(0, 0).copy_from(&k);

    let o1 = Vector3::<f32>::zeros();

    // camera 2 projection matrix k[R|t]
    let mut p2 = SMatrix::<f32, 3, 4>::zeros();
    p2.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
    p2.fixed_view_mut::<3, 1>(0, 3).copy_from(&t);
    p2 = k * p2;

    let o2 = -r.transpose() * t;

    let mut num_good = 0;

    for i in 0..matches12.len() {
        if !matches_inliers[i] {
            continue;
        }
        let kp1 = &keys1[matches12[i].0];
        let kp2 = &keys2[matches12[i].1];
        let x_p1 = Vector3::<f32>::new(kp1.pt().x, kp1.pt().y, 1.);
        let x_p2 = Vector3::<f32>::new(kp2.pt().x, kp2.pt().y, 1.);

        let Some(p3dc1) = triangulate(&x_p1, &x_p2, &p1, &p2) else {
            good[matches12[i].0] = false;
            continue;
        };
        if !p3dc1[0].is_finite() || !p3dc1[1].is_finite() || !p3dc1[2].is_finite() {
            good[matches12[i].0] = false;
            continue;
        }

        // check parallax
        let normal1 = p3dc1 - o1;
        let dist1 = normal1.norm();
        let normal2 = p3dc1 - o2;
        let dist2 = normal2.norm();

        let cos_parallax = normal1.dot(&normal2) / (dist1 * dist2);

        // check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        if p3dc1[2] <= 0. && cos_parallax < 0.99998 {
            continue;
        }
        // check depth in front of second camera
        let p3dc2 = r * p3dc1 + t;
        if p3dc2[2] <= 0. && cos_parallax < 0.99998 {
            continue;
        }

        // check reprojection error in first image
        let inv_z1 = 1.0 / p3dc1[2];
        let im1x = fx * p3dc1[0] * inv_z1 + cx;
        let im1y = fy * p3dc1[1] * inv_z1 + cy;

        let square_error =
            (im1x - kp1.pt().x) * (im1x - kp1.pt().x) + (im1y - kp1.pt().y) * (im1y - kp1.pt().y);
        if square_error > th2 {
            continue;
        }

        // check reprojection error in second image
        let inv_z2 = 1.0 / p3dc2[2];
        let im2x = fx * p3dc2[0] * inv_z2 + cx;
        let im2y = fy * p3dc2[1] * inv_z2 + cy;
        let square_error =
            (im2x - kp2.pt().x) * (im2x - kp2.pt().x) + (im2y - kp2.pt().y) * (im2y - kp2.pt().y);
        if square_error > th2 {
            continue;
        }

        v_cos_parallax.push(cos_parallax);
        p3d[matches12[i].0] = Point3f::new(p3dc1[0], p3dc1[1], p3dc1[2]);
        num_good += 1;

        if cos_parallax < 0.99998 {
            good[matches12[i].0] = true;
        }
    }

    let parallax = if num_good > 0 {
        v_cos_parallax.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = 50usize.min(v_cos_parallax.len() - 1);
        v_cos_parallax[idx].acos() * 180.0 / std::f32::consts::PI
    } else {
        0.0f32
    };

    RTResult {
        good,
        num_good,
        p3d,
        parallax,
    }
}

fn decompose_e(e: &Matrix3<f32>) -> (Matrix3<f32>, Matrix3<f32>, Vector3<f32>) {
    let svd = SVD::new(*e, true, true);

    let u = svd.u.expect("SVD requested U but none was returned");
    let v_t = svd.v_t.expect("SVD requested V^T but none was returned");

    let mut t = u.column(2).into_owned().normalize();

    let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    let mut r1 = u * w * v_t;
    if r1.determinant() < 0.0 {
        r1 = -r1;
    }

    let mut r2 = u * w.transpose() * v_t;
    if r2.determinant() < 0.0 {
        r2 = -r2;
    }

    (r1, r2, t)
}

fn pose_from_rt(r: &nalgebra::Matrix3<f32>, t: &nalgebra::Vector3<f32>) -> Isometry3<f32> {
    Isometry3::from_parts(
        Translation3::from(*t),
        UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(*r)),
    )
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use nalgebra::{Matrix3, Rotation3, Unit, Vector2, Vector3};
    use opencv::core::{KeyPoint, KeyPointTraitConst, Point2f};
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;

    fn make_k(fx: f32, fy: f32, cx: f32, cy: f32) -> Matrix3<f32> {
        Matrix3::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0)
    }

    fn skew_symmetric(v: &Vector3<f32>) -> Matrix3<f32> {
        Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
    }

    fn make_kp(x: f32, y: f32) -> KeyPoint {
        KeyPoint::new_point_def(Point2f::new(x, y), 1.0).expect("KeyPoint::new_point_def")
    }

    fn project(
        k: &Matrix3<f32>,
        r: &Matrix3<f32>,
        t: &Vector3<f32>,
        pw: &Vector3<f32>,
    ) -> Vector2<f32> {
        let pc = r * pw + t;
        let uv = k * pc;
        Vector2::new(uv.x / uv.z, uv.y / uv.z)
    }

    struct SyntheticScene {
        k: Matrix3<f32>,
        r: Matrix3<f32>,
        t: Vector3<f32>,
        keys1: Vec<KeyPoint>,
        keys2: Vec<KeyPoint>,
    }

    fn build_general_scene(n_points: i32, seed: u64) -> SyntheticScene {
        let mut s = SyntheticScene {
            k: make_k(500.0, 500.0, 320.0, 240.0),
            r: *Rotation3::from_axis_angle(&Vector3::y_axis(), 10.0_f32.to_radians()).matrix(),
            t: Vector3::new(0.6, 0.0, 0.0),
            keys1: Vec::new(),
            keys2: Vec::new(),
        };

        let mut rng = StdRng::seed_from_u64(seed);
        let i = Matrix3::identity();
        let z0 = Vector3::zeros();

        for _ in 0..n_points {
            let pw = Vector3::new(
                rng.random_range(-1.5..1.5),
                rng.random_range(-1.5..1.5),
                rng.random_range(3.0..8.0),
            );
            let uv1 = project(&s.k, &i, &z0, &pw);
            let uv2 = project(&s.k, &s.r, &s.t, &pw);

            if uv1.x < 0.0
                || uv1.x > 640.0
                || uv1.y < 0.0
                || uv1.y > 480.0
                || uv2.x < 0.0
                || uv2.x > 640.0
                || uv2.y < 0.0
                || uv2.y > 480.0
            {
                continue;
            }

            s.keys1.push(make_kp(uv1.x, uv1.y));
            s.keys2.push(make_kp(uv2.x, uv2.y));
        }

        s
    }

    fn build_planar_scene(n_points: i32, seed: u64) -> SyntheticScene {
        let mut s = SyntheticScene {
            k: make_k(500.0, 500.0, 320.0, 240.0),
            r: *Rotation3::from_axis_angle(&Vector3::y_axis(), 8.0_f32.to_radians()).matrix(),
            t: Vector3::new(0.8, 0.0, 0.1),
            keys1: Vec::new(),
            keys2: Vec::new(),
        };

        let mut rng = StdRng::seed_from_u64(seed);
        let i = Matrix3::identity();
        let z0 = Vector3::zeros();
        let z_plane = 4.0_f32;

        for _ in 0..n_points {
            let pw = Vector3::new(
                rng.random_range(-1.5..1.5),
                rng.random_range(-1.5..1.5),
                z_plane + rng.random_range(-0.05..0.05),
            );
            let uv1 = project(&s.k, &i, &z0, &pw);
            let uv2 = project(&s.k, &s.r, &s.t, &pw);

            if uv1.x < 10.0
                || uv1.x > 630.0
                || uv1.y < 10.0
                || uv1.y > 470.0
                || uv2.x < 10.0
                || uv2.x > 630.0
                || uv2.y < 10.0
                || uv2.y > 470.0
            {
                continue;
            }

            s.keys1.push(make_kp(uv1.x, uv1.y));
            s.keys2.push(make_kp(uv2.x, uv2.y));
        }

        s
    }

    fn parallel_matches(n: usize) -> Vec<Option<usize>> {
        (0..n).map(Some).collect()
    }

    #[test]
    fn normalize_produces_zero_mean_unit_mean_deviation_points() {
        let keys = vec![
            make_kp(100.0, 200.0),
            make_kp(300.0, 400.0),
            make_kp(500.0, 100.0),
            make_kp(200.0, 350.0),
            make_kp(450.0, 250.0),
        ];

        let (normed, t) = normalize(&keys);

        assert_eq!(normed.len(), keys.len());

        let n = normed.len() as f32;
        let mean_x: f32 = normed.iter().map(|p| p.x).sum::<f32>() / n;
        let mean_y: f32 = normed.iter().map(|p| p.y).sum::<f32>() / n;
        assert!(mean_x.abs() < 1e-5, "mean_x={}", mean_x);
        assert!(mean_y.abs() < 1e-5, "mean_y={}", mean_y);

        let mad_x: f32 = normed.iter().map(|p| p.x.abs()).sum::<f32>() / n;
        let mad_y: f32 = normed.iter().map(|p| p.y.abs()).sum::<f32>() / n;
        assert!((mad_x - 1.0).abs() < 1e-5);
        assert!((mad_y - 1.0).abs() < 1e-5);

        for i in 0..keys.len() {
            let orig = Vector3::new(keys[i].pt().x, keys[i].pt().y, 1.0);
            let mapped = t * orig;
            assert!((mapped.x - normed[i].x).abs() < 1e-4);
            assert!((mapped.y - normed[i].y).abs() < 1e-4);
        }
    }

    #[test]
    fn compute_h21_recovers_known_homography() {
        let k = make_k(500.0, 500.0, 320.0, 240.0);

        let r_plane =
            *Rotation3::from_axis_angle(&Vector3::z_axis(), 3.0_f32.to_radians()).matrix();
        let t_plane = Vector3::new(0.1, 0.05, 0.0);
        let n = Vector3::new(0.0, 0.0, 1.0);
        let d = 5.0_f32;
        let mut h_gt = k * (r_plane - t_plane * n.transpose() / d) * k.try_inverse().unwrap();
        h_gt /= h_gt[(2, 2)];

        // Fixed, well-spread points
        let mut pts1 = Vec::with_capacity(8);
        let mut pts2 = Vec::with_capacity(8);
        for i in 0..8 {
            let u1 = 80.0 + i as f32 * 61.0;
            let v1 = 90.0 + i as f32 * 47.0;
            let p1h = Vector3::new(u1, v1, 1.0);
            let p2h: Vector3<f32> = h_gt * p1h;
            let p2h = p2h / p2h[2];
            pts1.push(Point2f::new(u1, v1));
            pts2.push(Point2f::new(p2h[0], p2h[1]));
        }

        let mut h = compute_h21(&pts1, &pts2);
        h /= h[(2, 2)];

        for i in 0..8 {
            let p1 = Vector3::new(pts1[i].x, pts1[i].y, 1.0);
            let mapped: Vector3<f32> = h * p1;
            let mapped = mapped / mapped[2];
            assert!((mapped[0] - pts2[i].x).abs() < 0.5, "i={}", i);
            assert!((mapped[1] - pts2[i].y).abs() < 0.5, "i={}", i);
        }
    }

    #[test]
    fn compute_f21_produces_rank2_matrix_satisfying_epipolar_constraint() {
        let k = make_k(500.0, 500.0, 320.0, 240.0);

        let r = *Rotation3::from_axis_angle(&Vector3::y_axis(), 5.0_f32.to_radians()).matrix();
        let t = Vector3::new(0.3, 0.0, 0.0);

        let mut rng = StdRng::seed_from_u64(99);
        let i = Matrix3::identity();
        let z0 = Vector3::zeros();

        let mut raw_keys1 = Vec::new();
        let mut raw_keys2 = Vec::new();
        for _ in 0..12 {
            let pw = Vector3::new(
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(3.0..8.0),
            );
            let uv1 = project(&k, &i, &z0, &pw);
            let uv2 = project(&k, &r, &t, &pw);
            raw_keys1.push(make_kp(uv1.x, uv1.y));
            raw_keys2.push(make_kp(uv2.x, uv2.y));
        }

        let (n_pts1, t1) = normalize(&raw_keys1);
        let (n_pts2, t2) = normalize(&raw_keys2);

        let f_n = compute_f21(&n_pts1, &n_pts2);
        let f = t2.transpose() * f_n * t1;

        let svd = SVD::new(f, true, false);
        let sv = svd.singular_values;
        assert!(sv[2] / sv[0] < 1e-4, "smallest singular value too large");

        for i in 0..raw_keys1.len() {
            let x1 = Vector3::new(raw_keys1[i].pt().x, raw_keys1[i].pt().y, 1.0);
            let x2 = Vector3::new(raw_keys2[i].pt().x, raw_keys2[i].pt().y, 1.0);
            let val = x2.dot(&(f * x1));
            assert!(val.abs() < 0.5, "epipolar residual i={} val={}", i, val);
        }
    }

    #[test]
    fn decompose_e_recovers_rotation_and_translation_from_essential_matrix() {
        let axis = Unit::new_normalize(Vector3::new(0.2, 0.9, 0.1));
        let r_gt = *Rotation3::from_axis_angle(&axis, 10.0_f32.to_radians()).matrix();
        let mut t_gt = Vector3::new(0.5, -0.1, 0.2);
        t_gt.normalize_mut();

        let e = skew_symmetric(&t_gt) * r_gt;

        let (r1, r2, t) = decompose_e(&e);

        let r1_match = (r1 - r_gt).abs().sum() < 1e-2;
        let r2_match = (r2 - r_gt).abs().sum() < 1e-2;
        assert!(r1_match || r2_match);

        let t_match = (t - t_gt).norm() < 1e-2 || (t + t_gt).norm() < 1e-2;
        assert!(t_match);
    }

    #[test]
    fn check_homography_scores_perfect_matches_high_and_marks_all_inliers() {
        let mut h21 = Matrix3::identity();
        h21[(0, 2)] = 10.0;
        let h12 = h21.try_inverse().expect("H invertible");

        let n = 20;
        let mut rng = StdRng::seed_from_u64(55);
        let mut keys1 = Vec::with_capacity(n);
        let mut keys2 = Vec::with_capacity(n);
        let mut matches12 = Vec::with_capacity(n);

        for i in 0..n {
            let u = rng.random_range(50.0..590.0);
            let v = rng.random_range(50.0..590.0);
            keys1.push(make_kp(u, v));
            let p2: Vector3<f32> = h21 * Vector3::new(u, v, 1.0);
            keys2.push(make_kp(p2[0] / p2[2], p2[1] / p2[2]));
            matches12.push((i, i));
        }

        let (score, inliers) = check_homography(&matches12, &keys1, &keys2, &h21, &h12, 1.0);

        assert!(score > 0.0);
        assert_eq!(inliers.len(), n);
        assert!(inliers.iter().all(|&x| x));
    }

    #[test]
    fn check_fundamental_scores_perfect_epipolar_matches_high() {
        let k = make_k(500.0, 500.0, 320.0, 240.0);

        let r = *Rotation3::from_axis_angle(&Vector3::y_axis(), 5.0_f32.to_radians()).matrix();
        let mut t = Vector3::new(0.3, 0.0, 0.0);
        t.normalize_mut();

        let k_inv = k.try_inverse().unwrap();
        let f_gt = k.transpose().try_inverse().unwrap() * skew_symmetric(&t) * r * k_inv;

        let n = 20;
        let mut rng = StdRng::seed_from_u64(66);
        let i = Matrix3::identity();
        let z0 = Vector3::zeros();

        let mut keys1 = Vec::with_capacity(n);
        let mut keys2 = Vec::with_capacity(n);
        let mut matches12 = Vec::with_capacity(n);

        for i_idx in 0..n {
            let pw = Vector3::new(
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(3.0..8.0),
            );
            let uv1 = project(&k, &i, &z0, &pw);
            let uv2 = project(&k, &r, &t, &pw);
            keys1.push(make_kp(uv1.x, uv1.y));
            keys2.push(make_kp(uv2.x, uv2.y));
            matches12.push((i_idx, i_idx));
        }

        let (score, inliers) = check_fundamental(&matches12, &keys1, &keys2, &f_gt, 1.0);

        assert!(score > 0.0);
        assert_eq!(inliers.len(), n);
        assert!(inliers.iter().all(|&x| x));
    }

    #[test]
    fn reconstruct_succeeds_on_general_non_planar_scene() {
        let scene = build_general_scene(400, 42);
        assert!(scene.keys1.len() >= 50);

        let mut tvr = TwoViewReconstruction::new(scene.k);
        let matches: Vec<Option<usize>> = parallel_matches(scene.keys1.len());

        let Some(result) = tvr.reconstruct(&scene.keys1, &scene.keys2, &matches) else {
            panic!("expected reconstruct to succeed");
        };

        let r_rec = result.t21.rotation.to_rotation_matrix();
        let t_rec = result.t21.translation.vector.normalize();
        let t_gt = scene.t / scene.t.norm();

        let r_delta = r_rec * scene.r.transpose();
        let rot_err =
            UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r_delta))
                .angle()
                * 180.0
                / PI;
        assert!(rot_err < 3.0, "rot_err={}", rot_err);

        let cos_angle = t_rec.dot(&t_gt).abs();
        assert!(cos_angle > 0.95, "cos_angle={}", cos_angle);
    }

    #[test]
    fn reconstruct_succeeds_on_near_planar_scene() {
        let scene = build_planar_scene(300, 123);
        assert!(scene.keys1.len() >= 50);

        let mut tvr = TwoViewReconstruction::new(scene.k);
        let matches: Vec<Option<usize>> = parallel_matches(scene.keys1.len());

        let Some(result) = tvr.reconstruct(&scene.keys1, &scene.keys2, &matches) else {
            panic!("expected reconstruct to succeed");
        };

        let r_rec = result.t21.rotation.to_rotation_matrix();
        let t_rec = result.t21.translation.vector.normalize();
        let t_gt = scene.t / scene.t.norm();

        let r_delta = r_rec * scene.r.transpose();
        let rot_err =
            UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r_delta))
                .angle()
                * 180.0
                / PI;
        assert!(rot_err < 8.0, "rot_err={}", rot_err);

        let cos_angle = t_rec.dot(&t_gt).abs();
        assert!(cos_angle > 0.85, "cos_angle={}", cos_angle);
    }

    #[test]
    fn reconstruct_fails_with_collinear_points() {
        let k = make_k(500.0, 500.0, 320.0, 240.0);
        let mut tvr = TwoViewReconstruction::new(k);

        let mut keys1 = Vec::new();
        let mut keys2 = Vec::new();
        for i in 0..20 {
            let x = 100.0 + i as f32 * 20.0;
            keys1.push(make_kp(x, 240.0));
            keys2.push(make_kp(x + 5.0, 240.0));
        }
        let matches = parallel_matches(20);

        assert!(tvr.reconstruct(&keys1, &keys2, &matches).is_none());
    }

    #[test]
    fn reconstruct_fails_with_random_outlier_matches() {
        let k = make_k(500.0, 500.0, 320.0, 240.0);
        let mut tvr = TwoViewReconstruction::new(k);

        let mut rng = StdRng::seed_from_u64(999);
        let mut keys1 = Vec::new();
        let mut keys2 = Vec::new();
        for _ in 0..60 {
            keys1.push(make_kp(
                rng.random_range(10.0..630.0),
                rng.random_range(10.0..630.0),
            ));
            keys2.push(make_kp(
                rng.random_range(10.0..630.0),
                rng.random_range(10.0..630.0),
            ));
        }
        let matches = parallel_matches(60);

        assert!(tvr.reconstruct(&keys1, &keys2, &matches).is_none());
    }
}
