// =============================================================================
// Decile A/E Metrics
// =============================================================================
//
// Computes per-decile actual / predicted / exposure aggregates for a model's
// predictions, sorted by the predicted value. Used by the dataset-diagnostics
// orchestrator to populate the standard 10-decile A/E table.
//
// A pre-computed sort index can be threaded in to avoid re-sorting `mu` when
// the caller already holds an `argsort(mu)` array.
//
// =============================================================================

use ndarray::Array1;

/// Raw aggregates for a single decile (sorted by predicted value).
#[derive(Clone, Debug)]
pub struct DecileMetricsRaw {
    /// 1-based decile index (1..=n_deciles).
    pub decile: usize,
    /// Number of observations in this decile.
    pub n: usize,
    /// Sum of `y` in this decile.
    pub actual_sum: f64,
    /// Sum of `mu` in this decile.
    pub predicted_sum: f64,
    /// Sum of `exposure` in this decile (or `n` when no exposure passed).
    pub exposure_sum: f64,
    /// Min `mu` value in this decile.
    pub mu_min: f64,
    /// Max `mu` value in this decile.
    pub mu_max: f64,
}

/// Compute A/E and exposure metrics per decile, sorted by `mu`.
///
/// Accepts an optional pre-sorted index to avoid redundant sorts when the
/// caller already holds `argsort(mu)`.
///
/// The decile loop is sequential: each decile aggregates a different range of
/// `n / n_deciles` rows so the per-decile work is fast (and rayon overhead at
/// k=10 deciles outweighs any gains for typical n).
pub fn compute_ae_by_decile(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    exposure: Option<&Array1<f64>>,
    n_deciles: usize,
    sort_idx: Option<&Array1<usize>>,
) -> Vec<DecileMetricsRaw> {
    let n = y.len();

    // If the caller didn't pass a sort index, build one locally. Keep it owned
    // separately so the borrow lives for the whole function.
    let owned_sort: Option<Array1<usize>> = match sort_idx {
        Some(_) => None,
        None => {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&a, &b| {
                mu[a]
                    .partial_cmp(&mu[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            Some(Array1::from_vec(idx))
        }
    };
    let sort_idx_view = sort_idx.unwrap_or_else(|| {
        owned_sort
            .as_ref()
            .expect("owned_sort must be Some when sort_idx is None")
    });

    // Gather y/mu (and exposure when provided) into contiguous buffers in
    // sort-index order. This mirrors NumPy's `arr[sort_idx]` strategy: the
    // indirect access happens once during the gather, then per-decile sums
    // run on contiguous slices the optimiser can vectorise.
    let y_slice = y.as_slice().expect("y must be contiguous");
    let mu_slice = mu.as_slice().expect("mu must be contiguous");
    let idx_slice = sort_idx_view
        .as_slice()
        .expect("sort_idx must be contiguous");

    // Sequential gather: the per-decile sums are tiny (10 chunks of n/10
    // contiguous f64s) so rayon's thread fan-out overhead dominates. The
    // gather loop below is autovectorisable and runs at ~memory bandwidth.
    let y_sorted: Vec<f64> = idx_slice.iter().map(|&row| y_slice[row]).collect();
    let mu_sorted: Vec<f64> = idx_slice.iter().map(|&row| mu_slice[row]).collect();
    let exp_sorted: Option<Vec<f64>> = exposure.map(|e| {
        let e_slice = e.as_slice().expect("exposure must be contiguous");
        idx_slice.iter().map(|&row| e_slice[row]).collect()
    });

    let decile_size = n / n_deciles;
    let mut out = Vec::with_capacity(n_deciles);
    for d in 0..n_deciles {
        let start = d * decile_size;
        let end = if d == n_deciles - 1 {
            n
        } else {
            (d + 1) * decile_size
        };

        // Sequential sums on contiguous slices — autovectorisable.
        let actual: f64 = y_sorted[start..end].iter().sum();
        let predicted: f64 = mu_sorted[start..end].iter().sum();
        let expo: f64 = match &exp_sorted {
            Some(v) => v[start..end].iter().sum(),
            None => (end - start) as f64,
        };
        // mu is already sorted ascending in this slice. Guard against empty
        // deciles (n < n_deciles) where the slice is zero-length.
        let (mu_min, mu_max) = if start < end {
            (mu_sorted[start], mu_sorted[end - 1])
        } else {
            (f64::NAN, f64::NAN)
        };

        out.push(DecileMetricsRaw {
            decile: d + 1,
            n: end - start,
            actual_sum: actual,
            predicted_sum: predicted,
            exposure_sum: expo,
            mu_min,
            mu_max,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(v: &[f64]) -> Array1<f64> {
        Array1::from_vec(v.to_vec())
    }

    #[test]
    fn basic_two_deciles_sorted() {
        let y = arr(&[1.0, 2.0, 3.0, 4.0]);
        let mu = arr(&[0.1, 0.2, 0.3, 0.4]);
        let result = compute_ae_by_decile(&y, &mu, None, 2, None);
        assert_eq!(result.len(), 2);
        // First decile: rows 0,1 (mu 0.1, 0.2) → actual = 3, predicted = 0.3
        assert_eq!(result[0].decile, 1);
        assert_eq!(result[0].n, 2);
        assert!((result[0].actual_sum - 3.0).abs() < 1e-12);
        assert!((result[0].predicted_sum - 0.3).abs() < 1e-12);
        assert!((result[0].mu_min - 0.1).abs() < 1e-12);
        assert!((result[0].mu_max - 0.2).abs() < 1e-12);
        // Second decile: rows 2,3 (mu 0.3, 0.4) → actual = 7, predicted = 0.7
        assert_eq!(result[1].decile, 2);
        assert!((result[1].actual_sum - 7.0).abs() < 1e-12);
        assert!((result[1].predicted_sum - 0.7).abs() < 1e-12);
    }

    #[test]
    fn unsorted_mu_uses_argsort() {
        let y = arr(&[1.0, 2.0, 3.0, 4.0]);
        let mu = arr(&[0.4, 0.1, 0.3, 0.2]);
        let result = compute_ae_by_decile(&y, &mu, None, 2, None);
        // Sorted mu order: [0.1, 0.2, 0.3, 0.4] → rows [1, 3, 2, 0]
        // Decile 1: y[1]+y[3] = 2+4=6, mu sum = 0.1+0.2=0.3
        assert!((result[0].actual_sum - 6.0).abs() < 1e-12);
        assert!((result[0].predicted_sum - 0.3).abs() < 1e-12);
        // Decile 2: y[2]+y[0] = 3+1=4, mu sum = 0.3+0.4=0.7
        assert!((result[1].actual_sum - 4.0).abs() < 1e-12);
        assert!((result[1].predicted_sum - 0.7).abs() < 1e-12);
    }

    #[test]
    fn exposure_sum_when_provided() {
        let y = arr(&[1.0, 2.0, 3.0, 4.0]);
        let mu = arr(&[0.1, 0.2, 0.3, 0.4]);
        let exp = arr(&[10.0, 20.0, 30.0, 40.0]);
        let result = compute_ae_by_decile(&y, &mu, Some(&exp), 2, None);
        assert!((result[0].exposure_sum - 30.0).abs() < 1e-12);
        assert!((result[1].exposure_sum - 70.0).abs() < 1e-12);
    }

    #[test]
    fn no_exposure_returns_count() {
        let y = arr(&[1.0, 2.0, 3.0, 4.0]);
        let mu = arr(&[0.1, 0.2, 0.3, 0.4]);
        let result = compute_ae_by_decile(&y, &mu, None, 2, None);
        assert!((result[0].exposure_sum - 2.0).abs() < 1e-12);
        assert!((result[1].exposure_sum - 2.0).abs() < 1e-12);
    }

    #[test]
    fn presorted_index_path() {
        let y = arr(&[1.0, 2.0, 3.0, 4.0]);
        let mu = arr(&[0.4, 0.1, 0.3, 0.2]);
        // argsort(mu) = [1, 3, 2, 0]
        let sort_idx = Array1::from_vec(vec![1usize, 3, 2, 0]);
        let result_with = compute_ae_by_decile(&y, &mu, None, 2, Some(&sort_idx));
        let result_without = compute_ae_by_decile(&y, &mu, None, 2, None);
        for (a, b) in result_with.iter().zip(result_without.iter()) {
            assert!((a.actual_sum - b.actual_sum).abs() < 1e-12);
            assert!((a.predicted_sum - b.predicted_sum).abs() < 1e-12);
            assert!((a.exposure_sum - b.exposure_sum).abs() < 1e-12);
            assert_eq!(a.n, b.n);
        }
    }

    #[test]
    fn small_n_with_empty_deciles() {
        // n=8 < n_deciles=10 → decile_size=0 → first 9 deciles are empty,
        // last decile gets all 8 rows. Must not panic on empty slices.
        let y = arr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mu = arr(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        let result = compute_ae_by_decile(&y, &mu, None, 10, None);
        assert_eq!(result.len(), 10);
        for d in result.iter().take(9) {
            assert_eq!(d.n, 0);
            assert_eq!(d.actual_sum, 0.0);
            assert_eq!(d.predicted_sum, 0.0);
            assert!(d.mu_min.is_nan());
            assert!(d.mu_max.is_nan());
        }
        let last = &result[9];
        assert_eq!(last.n, 8);
        assert!((last.actual_sum - 36.0).abs() < 1e-12);
    }

    #[test]
    fn last_decile_absorbs_remainder() {
        // 11 rows, 2 deciles → decile 1 has 5, decile 2 has 6.
        let y: Vec<f64> = (0..11).map(|i| i as f64 + 1.0).collect();
        let mu: Vec<f64> = (0..11).map(|i| (i + 1) as f64 * 0.1).collect();
        let result =
            compute_ae_by_decile(&Array1::from_vec(y), &Array1::from_vec(mu), None, 2, None);
        assert_eq!(result[0].n, 5);
        assert_eq!(result[1].n, 6);
    }
}
