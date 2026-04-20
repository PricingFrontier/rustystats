// =============================================================================
// Partial Dependence Aggregation
// =============================================================================
//
// Helpers for the categorical partial-dependence loop in
// `DiagnosticsComputer.compute_partial_dependence`. The Python caller already
// holds u32 codes per categorical factor (via `cat_unique_cache`), so the only
// work left is to bucket-sum mu by level for each factor. With 6+ factors and
// n on the order of 1M, doing this in NumPy via two `np.bincount` calls per
// factor sequentially leaves a lot of parallelism on the floor; this module
// runs all factors in parallel via rayon. (OPT-20.)
// =============================================================================
use ndarray::Array1;
use rayon::prelude::*;

/// Per-factor categorical partial-dependence aggregates.
///
/// For each factor (entry of `codes`), produces:
///   - `counts[lvl]`  = number of rows whose code equals `lvl`
///   - `mu_sums[lvl]` = sum of `mu[i]` for rows whose code equals `lvl`
///
/// The Python caller divides `mu_sums / counts` to obtain the per-level mean
/// prediction (with a `base_pred` fallback for empty levels).
///
/// # Arguments
/// * `codes` - slice of length `k`; entry `j` is a length-`n` slice of `u32`
///   codes for factor `j`. Codes that are out-of-range
///   (`>= n_levels_per_factor[j]`) are silently dropped. Taking a slice-of-
///   slices (rather than a stacked `(n, k)` matrix) lets the PyO3 wrapper
///   forward the caller's already-contiguous 1D numpy buffers without the
///   transient `np.stack` allocation (400 MB @ n=1M × k=100).
/// * `mu` - length-`n` predictions.
/// * `n_levels_per_factor` - length-`k` slice giving the level count for each
///   factor; sizes the output buckets.
///
/// # Returns
/// A `Vec` of length `k`. Entry `j` is `(counts, mu_sums)`, both of length
/// `n_levels_per_factor[j]`. `f64` is used for `counts` so the Python side
/// can divide directly without casting; at `n = 1M` this is exact.
pub fn partial_dependence_categorical_batch(
    codes: &[&[u32]],
    mu: &Array1<f64>,
    n_levels_per_factor: &[usize],
) -> Vec<(Vec<f64>, Vec<f64>)> {
    let n = mu.len();
    let k = codes.len();
    debug_assert_eq!(n_levels_per_factor.len(), k);
    debug_assert!(codes.iter().all(|c| c.len() == n));

    let mu_slice = mu.as_slice().expect("mu must be contiguous");

    (0..k)
        .into_par_iter()
        .map(|fac| {
            let m = n_levels_per_factor[fac];
            let mut counts = vec![0.0_f64; m];
            let mut mu_sums = vec![0.0_f64; m];
            let col = codes[fac];
            for i in 0..n {
                let lvl = col[i] as usize;
                if lvl < m {
                    counts[lvl] += 1.0;
                    mu_sums[lvl] += mu_slice[i];
                }
            }
            (counts, mu_sums)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    /// Helper: build a `Vec<&[u32]>` of column slices from an `Array2<u32>` for
    /// tests that were originally written against the matrix signature. The
    /// slices borrow from a side vector of column copies whose lifetime must
    /// outlive the call — hence the explicit `columns` parameter.
    fn columns_as_slices(columns: &[Vec<u32>]) -> Vec<&[u32]> {
        columns.iter().map(|c| c.as_slice()).collect()
    }

    /// Transpose an `Array2<u32>` of shape (n, k) into `k` column vectors so
    /// each can be viewed as a contiguous `&[u32]`. `Array2::column(j)` is
    /// *not* contiguous in row-major memory, so a copy is required; we collect
    /// into owned `Vec<u32>` here.
    fn columns_of(mat: &Array2<u32>) -> Vec<Vec<u32>> {
        (0..mat.ncols())
            .map(|j| mat.column(j).iter().copied().collect())
            .collect()
    }

    #[test]
    fn partial_dependence_categorical_batch_basic() {
        // 5 rows, 2 factors:
        //   factor 0: codes [0, 1, 0, 2, 1], 3 levels
        //   factor 1: codes [1, 0, 1, 1, 0], 2 levels
        let codes_mat = array![[0u32, 1], [1, 0], [0, 1], [2, 1], [1, 0]];
        let cols = columns_of(&codes_mat);
        let codes = columns_as_slices(&cols);
        let mu = array![0.5, 1.0, 1.5, 2.0, 2.5];
        let n_levels = vec![3usize, 2];

        let result = partial_dependence_categorical_batch(&codes, &mu, &n_levels);

        assert_eq!(result.len(), 2);

        // Factor 0: level 0 -> rows 0, 2 -> count=2, mu_sum=2.0
        //           level 1 -> rows 1, 4 -> count=2, mu_sum=3.5
        //           level 2 -> row 3 ->     count=1, mu_sum=2.0
        let (counts0, mu_sums0) = &result[0];
        assert_eq!(counts0, &vec![2.0, 2.0, 1.0]);
        assert_eq!(mu_sums0, &vec![2.0, 3.5, 2.0]);

        // Factor 1: level 0 -> rows 1, 4 -> count=2, mu_sum=3.5
        //           level 1 -> rows 0, 2, 3 -> count=3, mu_sum=4.0
        let (counts1, mu_sums1) = &result[1];
        assert_eq!(counts1, &vec![2.0, 3.0]);
        assert_eq!(mu_sums1, &vec![3.5, 4.0]);
    }

    #[test]
    fn partial_dependence_skips_out_of_range_codes() {
        // Code 5 is out of range when n_levels = 2; it should be dropped
        // rather than panic.
        let col0: Vec<u32> = vec![0, 1, 5, 0];
        let codes: Vec<&[u32]> = vec![col0.as_slice()];
        let mu = array![1.0, 2.0, 3.0, 4.0];
        let n_levels = vec![2usize];

        let result = partial_dependence_categorical_batch(&codes, &mu, &n_levels);
        let (counts, mu_sums) = &result[0];
        assert_eq!(counts, &vec![2.0, 1.0]); // rows 0,3 -> level 0; row 1 -> level 1
        assert_eq!(mu_sums, &vec![5.0, 2.0]);
    }

    #[test]
    fn partial_dependence_handles_empty_levels() {
        // 3 levels but only level 0 is observed.
        let col0: Vec<u32> = vec![0, 0, 0];
        let codes: Vec<&[u32]> = vec![col0.as_slice()];
        let mu = array![1.0, 2.0, 3.0];
        let n_levels = vec![3usize];

        let result = partial_dependence_categorical_batch(&codes, &mu, &n_levels);
        let (counts, mu_sums) = &result[0];
        assert_eq!(counts, &vec![3.0, 0.0, 0.0]);
        assert_eq!(mu_sums, &vec![6.0, 0.0, 0.0]);
    }

    // =========================================================================
    // Regression guards for the Change 7 memory-hardening refactor of
    // `partial_dependence_categorical_batch_py`. The PyO3 wrapper now accepts
    // a `Vec<PyReadonlyArray1<u32>>` instead of a stacked `(n, k)` matrix;
    // the core function here also moved to `&[&[u32]]`. These tests pin the
    // numerical output on realistic data so the wrapper re-plumbing can be
    // validated without risk of silent drift.
    // =========================================================================

    /// Build a deterministic length-n u32 code column with `n_levels` levels
    /// using a simple modular LCG — no rand-crate dependency, no tape flip.
    fn gen_codes(n: usize, n_levels: u32, seed: u64) -> Vec<u32> {
        // Classic Numerical Recipes LCG constants — deterministic, good enough
        // for spreading codes across levels in tests.
        let a = 1664525_u64;
        let c = 1013904223_u64;
        let mut state = seed.wrapping_mul(a).wrapping_add(c);
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(a).wrapping_add(c);
                ((state >> 16) as u32) % n_levels
            })
            .collect()
    }

    fn gen_mu(n: usize, seed: u64) -> Vec<f64> {
        let a = 6364136223846793005_u64;
        let c = 1442695040888963407_u64;
        let mut state = seed.wrapping_mul(a).wrapping_add(c);
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(a).wrapping_add(c);
                // Map to [0.1, 2.1) — mu-like positive predictions.
                0.1 + ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0
            })
            .collect()
    }

    /// Independent reference implementation (serial, per-factor bincount) used
    /// as the "golden" oracle for the parallel rayon path.
    fn reference_partial_dependence(
        codes: &[&[u32]],
        mu: &Array1<f64>,
        n_levels_per_factor: &[usize],
    ) -> Vec<(Vec<f64>, Vec<f64>)> {
        let n = mu.len();
        let k = codes.len();
        let mut out = Vec::with_capacity(k);
        for fac in 0..k {
            let m = n_levels_per_factor[fac];
            let mut counts = vec![0.0_f64; m];
            let mut mu_sums = vec![0.0_f64; m];
            let col = codes[fac];
            for i in 0..n {
                let lvl = col[i] as usize;
                if lvl < m {
                    counts[lvl] += 1.0;
                    mu_sums[lvl] += mu[i];
                }
            }
            out.push((counts, mu_sums));
        }
        out
    }

    #[test]
    fn partial_dependence_realistic_5_factors_varied_levels() {
        // Change 7 regression guard: n=10k rows × k=5 factors with level
        // counts [3, 5, 10, 20, 50] matches the per-factor code arrays the
        // Python caller now forwards directly. Deterministic codes + mu via
        // LCG; compared against a serial reference implementation.
        let n = 10_000;
        let n_levels_per_factor = vec![3usize, 5, 10, 20, 50];
        let k = n_levels_per_factor.len();

        let cols: Vec<Vec<u32>> = (0..k)
            .map(|fac| {
                let mut col = Vec::with_capacity(n);
                for i in 0..n {
                    let seed = (i as u64)
                        .wrapping_mul(0x9E3779B97F4A7C15)
                        .wrapping_add(fac as u64);
                    let v = gen_codes(1, n_levels_per_factor[fac] as u32, seed);
                    col.push(v[0]);
                }
                col
            })
            .collect();
        let codes: Vec<&[u32]> = cols.iter().map(|c| c.as_slice()).collect();
        let mu = Array1::from(gen_mu(n, 0xDEADBEEF));

        let got = partial_dependence_categorical_batch(&codes, &mu, &n_levels_per_factor);
        let want = reference_partial_dependence(&codes, &mu, &n_levels_per_factor);

        assert_eq!(got.len(), k);
        for fac in 0..k {
            let (got_counts, got_mu) = &got[fac];
            let (want_counts, want_mu) = &want[fac];
            assert_eq!(got_counts.len(), n_levels_per_factor[fac]);
            assert_eq!(got_mu.len(), n_levels_per_factor[fac]);
            // Counts are exact integers stored as f64; compare bit-exact.
            assert_eq!(got_counts, want_counts, "counts mismatch at factor {}", fac);
            // mu_sums involve floating-point addition: order-independent for
            // this serial vs serial comparison (parallelism is over factors,
            // not rows), so bit-exact equality is fine here.
            assert_eq!(got_mu, want_mu, "mu_sums mismatch at factor {}", fac);
            // Invariant: total count equals n (no out-of-range codes generated).
            let total: f64 = got_counts.iter().sum();
            assert_eq!(total, n as f64);
            // Invariant: total mu_sum equals sum of mu (all rows bucketed).
            let total_mu: f64 = got_mu.iter().sum();
            let expected_mu_total: f64 = mu.iter().sum();
            assert!(
                (total_mu - expected_mu_total).abs() < 1e-9 * expected_mu_total.abs().max(1.0),
                "total mu_sum {} != sum(mu) {} at factor {}",
                total_mu,
                expected_mu_total,
                fac
            );
        }
    }

    #[test]
    fn partial_dependence_golden_hand_computed() {
        // Small hand-computable instance pinning exact expected aggregates.
        // 8 rows × 3 factors, level counts [2, 3, 4].
        let codes_mat = array![
            [0u32, 0, 0],
            [1, 0, 1],
            [0, 1, 2],
            [1, 1, 3],
            [0, 2, 0],
            [1, 2, 1],
            [0, 0, 2],
            [1, 1, 3],
        ];
        let cols = columns_of(&codes_mat);
        let codes = columns_as_slices(&cols);
        let mu = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let n_levels = vec![2usize, 3, 4];

        let result = partial_dependence_categorical_batch(&codes, &mu, &n_levels);

        // Factor 0 (2 levels):
        //   level 0 -> rows 0, 2, 4, 6 -> count=4, mu_sum=1+3+5+7=16
        //   level 1 -> rows 1, 3, 5, 7 -> count=4, mu_sum=2+4+6+8=20
        assert_eq!(result[0].0, vec![4.0, 4.0]);
        assert_eq!(result[0].1, vec![16.0, 20.0]);

        // Factor 1 (3 levels):
        //   level 0 -> rows 0, 1, 6 -> count=3, mu_sum=1+2+7=10
        //   level 1 -> rows 2, 3, 7 -> count=3, mu_sum=3+4+8=15
        //   level 2 -> rows 4, 5    -> count=2, mu_sum=5+6=11
        assert_eq!(result[1].0, vec![3.0, 3.0, 2.0]);
        assert_eq!(result[1].1, vec![10.0, 15.0, 11.0]);

        // Factor 2 (4 levels):
        //   level 0 -> rows 0, 4 -> count=2, mu_sum=1+5=6
        //   level 1 -> rows 1, 5 -> count=2, mu_sum=2+6=8
        //   level 2 -> rows 2, 6 -> count=2, mu_sum=3+7=10
        //   level 3 -> rows 3, 7 -> count=2, mu_sum=4+8=12
        assert_eq!(result[2].0, vec![2.0, 2.0, 2.0, 2.0]);
        assert_eq!(result[2].1, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn partial_dependence_parity_against_reference_with_out_of_range() {
        // Exercise the out-of-range code path at scale: factor 2's level count
        // is intentionally under-sized so ~half its codes get dropped. The
        // reference implementation mirrors this so aggregates must still agree.
        let n = 2_000;
        let n_levels_per_factor = vec![4usize, 8, 3]; // factor 2 under-sized
        let gen_levels = vec![4u32, 8, 6]; // generate 0..6 but only 0..3 counted

        let k = n_levels_per_factor.len();
        let cols: Vec<Vec<u32>> = (0..k)
            .map(|fac| {
                let mut col = Vec::with_capacity(n);
                for i in 0..n {
                    let v = gen_codes(1, gen_levels[fac], (i as u64) * 31 + fac as u64);
                    col.push(v[0]);
                }
                col
            })
            .collect();
        let codes: Vec<&[u32]> = cols.iter().map(|c| c.as_slice()).collect();
        let mu = Array1::from(gen_mu(n, 0xBADC0FFEE));

        let got = partial_dependence_categorical_batch(&codes, &mu, &n_levels_per_factor);
        let want = reference_partial_dependence(&codes, &mu, &n_levels_per_factor);

        for fac in 0..k {
            assert_eq!(got[fac].0, want[fac].0, "counts diverged at factor {}", fac);
            assert_eq!(
                got[fac].1, want[fac].1,
                "mu_sums diverged at factor {}",
                fac
            );
        }
        // Factor 2 should have dropped rows: total count < n.
        let total_fac2: f64 = got[2].0.iter().sum();
        assert!(total_fac2 < n as f64, "expected some codes to be dropped");
    }
}
