//! Validates `Optimizer::Marginalize` against the direct Schur complement.

use nalgebra::DMatrix;
use orb_slam3_rs::optimizer::marginalize;

#[test]
fn marginalize_matches_schur_complement() {
    // Build a deterministic symmetric positive-definite 9x9 matrix.
    let n = 9;
    let mut a = DMatrix::<f64>::zeros(n, n);
    let mut seed = 1u64;
    let mut rnd = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    };
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = rnd();
        }
    }
    let h = &a * a.transpose() + DMatrix::identity(n, n) * (n as f64); // SPD

    // Marginalize variables [start..=end].
    let (start, end) = (0usize, 2usize);
    let res = marginalize(&h, start, end);

    // Direct Schur complement of the kept block [3..9] after removing [0..3].
    let b = end - start + 1; // 3
    let keep = n - b; // 6
    let hkk = h.view((b, b), (keep, keep)).into_owned();
    let hkm = h.view((b, 0), (keep, b)).into_owned();
    let hmk = h.view((0, b), (b, keep)).into_owned();
    let hmm = h.view((0, 0), (b, b)).into_owned();
    let schur = &hkk - &hkm * hmm.try_inverse().unwrap() * &hmk;

    // The kept block of the result must equal the Schur complement.
    let got = res.view((b, b), (keep, keep)).into_owned();
    let diff = (&got - &schur).norm() / schur.norm();
    assert!(diff < 1e-10, "Schur block differs by {diff}");

    // The marginalized block must be zeroed.
    assert!(
        res.view((start, start), (b, b)).norm() < 1e-12,
        "marginalized block not zeroed"
    );
}
