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
use ndarray::{Array1, Array2};
use rayon::prelude::*;

/// Per-factor categorical partial-dependence aggregates.
///
/// For each factor (column of `codes_matrix`), produces:
///   - `counts[lvl]`  = number of rows whose code equals `lvl`
///   - `mu_sums[lvl]` = sum of `mu[i]` for rows whose code equals `lvl`
///
/// The Python caller divides `mu_sums / counts` to obtain the per-level mean
/// prediction (with a `base_pred` fallback for empty levels).
///
/// # Arguments
/// * `codes_matrix` - shape `(n, k)` of `u32` codes; column `j` is the code
///   per row for factor `j`. Codes that are out-of-range
///   (`>= n_levels_per_factor[j]`) are silently dropped.
/// * `mu` - length-`n` predictions.
/// * `n_levels_per_factor` - length-`k` vector giving the level count for each
///   factor; sizes the output buckets.
///
/// # Returns
/// A `Vec` of length `k`. Entry `j` is `(counts, mu_sums)`, both of length
/// `n_levels_per_factor[j]`. `f64` is used for `counts` so the Python side
/// can divide directly without casting; at `n = 1M` this is exact.
pub fn partial_dependence_categorical_batch(
    codes_matrix: &Array2<u32>,
    mu: &Array1<f64>,
    n_levels_per_factor: &[usize],
) -> Vec<(Vec<f64>, Vec<f64>)> {
    let n = mu.len();
    let k = codes_matrix.ncols();
    debug_assert_eq!(n_levels_per_factor.len(), k);
    debug_assert_eq!(codes_matrix.nrows(), n);

    (0..k)
        .into_par_iter()
        .map(|fac| {
            let m = n_levels_per_factor[fac];
            let mut counts = vec![0.0_f64; m];
            let mut mu_sums = vec![0.0_f64; m];
            let col = codes_matrix.column(fac);
            for i in 0..n {
                let lvl = col[i] as usize;
                if lvl < m {
                    counts[lvl] += 1.0;
                    mu_sums[lvl] += mu[i];
                }
            }
            (counts, mu_sums)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn partial_dependence_categorical_batch_basic() {
        // 5 rows, 2 factors:
        //   factor 0: codes [0, 1, 0, 2, 1], 3 levels
        //   factor 1: codes [1, 0, 1, 1, 0], 2 levels
        let codes = array![[0u32, 1], [1, 0], [0, 1], [2, 1], [1, 0]];
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
        let codes = array![[0u32], [1], [5], [0]];
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
        let codes = array![[0u32], [0], [0]];
        let mu = array![1.0, 2.0, 3.0];
        let n_levels = vec![3usize];

        let result = partial_dependence_categorical_batch(&codes, &mu, &n_levels);
        let (counts, mu_sums) = &result[0];
        assert_eq!(counts, &vec![3.0, 0.0, 0.0]);
        assert_eq!(mu_sums, &vec![6.0, 0.0, 0.0]);
    }
}
