//! Weighted standardisation moments (`specs/out-of-core-fit.md` §6.7).
//!
//! Two strictly sequential passes over the design rows, both reduced with
//! [`crate::rows`] so the result is independent of thread count and of how
//! rows arrive:
//!
//! * M1: `S_j = Σ w·x_j` (or `Σ x_j` unweighted) and `W = Σ w` (or `n`);
//!   `m_j = S_j / W`.
//! * M2: `V_j = Σ w·(x_j − m_j)²` (or `Σ (x_j − m_j)²`), formed row-wise from the
//!   subtracted value; `var = max(V_j / W, 0)`, `sd = √var`.
//!
//! The per-row folds are public so the out-of-core moments pass can feed them
//! encoded blocks; [`standardization_moments`] is the in-memory driver.

use std::ops::Range;

use ndarray::{ArrayView1, ArrayView2};

use crate::error::{Result, RustyStatsError};
use crate::rows::reduce_rows;

/// A column whose weighted standard deviation is at or below this keeps centre 0
/// and scale 1.
pub const MIN_WEIGHTED_STD: f64 = 1e-12;

/// Standardisation of the active (penalised) columns of a design.
#[derive(Debug, Clone, PartialEq)]
pub struct StandardizationMoments {
    pub center: Vec<f64>,
    pub scale: Vec<f64>,
    pub ridge_xtx_diag: Vec<f64>,
}

/// M1 fold: add `w·x_j` (or `x_j`) for the active columns of `rows` into `sums`,
/// and `w` (or 1) into `weight_sum` (the last element of `sums`).
pub fn fold_first_moments(
    sums: &mut [f64],
    x: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    active: &[usize],
    rows: Range<usize>,
) {
    fold_first_moments_where(sums, x, weights, active, rows, |_| true);
}

/// [`fold_first_moments`] over the rows `include` accepts (a fold's training rows,
/// §7.9); the others are skipped.
pub fn fold_first_moments_where(
    sums: &mut [f64],
    x: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    active: &[usize],
    rows: Range<usize>,
    include: impl Fn(usize) -> bool,
) {
    let weight_slot = active.len();
    for row in rows {
        if !include(row) {
            continue;
        }
        let x_row = x.row(row);
        match weights {
            Some(w) => {
                let wk = w[row];
                for (slot, &col) in active.iter().enumerate() {
                    sums[slot] += wk * x_row[col];
                }
                sums[weight_slot] += wk;
            }
            None => {
                for (slot, &col) in active.iter().enumerate() {
                    sums[slot] += x_row[col];
                }
                sums[weight_slot] += 1.0;
            }
        }
    }
}

/// M2 fold: add `w·(x_j − m_j)²` (or `(x_j − m_j)²`) for the active columns.
pub fn fold_second_moments(
    squares: &mut [f64],
    x: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    active: &[usize],
    means: &[f64],
    rows: Range<usize>,
) {
    fold_second_moments_where(squares, x, weights, active, means, rows, |_| true);
}

/// [`fold_second_moments`] over the rows `include` accepts.
pub fn fold_second_moments_where(
    squares: &mut [f64],
    x: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    active: &[usize],
    means: &[f64],
    rows: Range<usize>,
    include: impl Fn(usize) -> bool,
) {
    for row in rows {
        if !include(row) {
            continue;
        }
        let x_row = x.row(row);
        match weights {
            Some(w) => {
                let wk = w[row];
                for (slot, &col) in active.iter().enumerate() {
                    let delta = x_row[col] - means[slot];
                    squares[slot] += wk * delta * delta;
                }
            }
            None => {
                for (slot, &col) in active.iter().enumerate() {
                    let delta = x_row[col] - means[slot];
                    squares[slot] += delta * delta;
                }
            }
        }
    }
}

/// Turn the reduced M1/M2 sums into centres, scales and the ridge diagonal.
pub fn finalize_moments(
    p: usize,
    active: &[usize],
    sums: &[f64],
    squares: &[f64],
    fit_intercept: bool,
) -> Result<StandardizationMoments> {
    let mut center = vec![0.0; p];
    let mut scale = vec![1.0; p];
    let mut ridge_xtx_diag = vec![0.0; p];
    let weight_sum = sums[active.len()];
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return Err(RustyStatsError::InvalidValue(format!(
            "standardization requires positive finite sum(weights), got {weight_sum}."
        )));
    }
    for (slot, &col) in active.iter().enumerate() {
        let mean = sums[slot] / weight_sum;
        let var = (squares[slot] / weight_sum).max(0.0);
        let sd = var.sqrt();
        let raw_diag = weight_sum * (var + mean * mean);
        let valid = mean.is_finite() && sd.is_finite() && sd > MIN_WEIGHTED_STD;

        ridge_xtx_diag[col] = if valid {
            if fit_intercept {
                weight_sum
            } else {
                raw_diag / (sd * sd)
            }
        } else {
            raw_diag
        };

        if valid {
            if fit_intercept {
                center[col] = mean;
            }
            scale[col] = sd;
        }
    }
    Ok(StandardizationMoments {
        center,
        scale,
        ridge_xtx_diag,
    })
}

/// Means of the active columns from M1 sums (the last element is `W`).
pub fn means_from_sums(sums: &[f64], active_len: usize) -> Vec<f64> {
    let weight_sum = sums[active_len];
    sums[..active_len]
        .iter()
        .map(|sum| sum / weight_sum)
        .collect()
}

/// In-memory standardisation moments over every row of `x`.
pub fn standardization_moments(
    x: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    active: &[usize],
    fit_intercept: bool,
) -> Result<StandardizationMoments> {
    standardization_moments_where(x, weights, active, fit_intercept, |_| true)
}

/// In-memory standardisation moments over the rows of `x` that `include` accepts
/// (a fold's training rows, §7.9), reduced in blocks of global rows.
pub fn standardization_moments_where(
    x: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    active: &[usize],
    fit_intercept: bool,
    include: impl Fn(usize) -> bool + Sync,
) -> Result<StandardizationMoments> {
    let (n, p) = x.dim();
    if let Some(w) = weights {
        if w.len() != n {
            return Err(RustyStatsError::dim_mismatch(
                n,
                w.len(),
                "standardization weights vs X rows",
            ));
        }
    }
    if p == 0 || active.is_empty() {
        return Ok(StandardizationMoments {
            center: vec![0.0; p],
            scale: vec![1.0; p],
            ridge_xtx_diag: vec![0.0; p],
        });
    }
    let active_len = active.len();
    let sums = reduce_rows(vec![0.0; active_len + 1], n, |sums, rows| {
        fold_first_moments_where(sums, x, weights, active, rows, &include)
    });
    let weight_sum = sums[active_len];
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return Err(RustyStatsError::InvalidValue(format!(
            "standardization requires positive finite sum(weights), got {weight_sum}."
        )));
    }
    let means = means_from_sums(&sums, active_len);
    let squares = reduce_rows(vec![0.0; active_len], n, |squares, rows| {
        fold_second_moments_where(squares, x, weights, active, &means, rows, &include)
    });
    finalize_moments(p, active, &sums, &squares, fit_intercept)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};

    #[test]
    fn moments_match_two_pass_definition() {
        let x = array![[1.0, 2.0, 10.0], [1.0, 4.0, 10.0], [1.0, 9.0, 10.0]];
        let w = array![1.0, 2.0, 1.0];
        let moments =
            standardization_moments(x.view(), Some(w.view()), &[1, 2], true).expect("moments");
        let mean = (2.0 + 8.0 + 9.0) / 4.0;
        let var =
            ((2.0f64 - mean).powi(2) + 2.0 * (4.0 - mean).powi(2) + (9.0 - mean).powi(2)) / 4.0;
        assert_eq!(moments.center[1], mean);
        assert!((moments.scale[1] - var.sqrt()).abs() < 1e-15);
        // A constant column is invalid: centre 0, scale 1, raw ridge diagonal.
        assert_eq!(moments.center[2], 0.0);
        assert_eq!(moments.scale[2], 1.0);
        assert_eq!(moments.ridge_xtx_diag[2], 4.0 * 100.0);
        assert_eq!(moments.ridge_xtx_diag[1], 4.0);
        assert_eq!(moments.center[0], 0.0);
        assert_eq!(moments.scale[0], 1.0);
    }

    #[test]
    fn moments_do_not_depend_on_thread_count() {
        let n = 3 * crate::rows::GRAM_BLOCK_ROWS + 11;
        let x = Array2::from_shape_fn((n, 3), |(row, col)| {
            ((row * 37 + col * 11) % 101) as f64 * 10f64.powi((row % 9) as i32 - 4)
        });
        let w: Array1<f64> = (0..n).map(|row| 0.5 + (row % 7) as f64).collect();
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool")
                .install(|| {
                    standardization_moments(x.view(), Some(w.view()), &[0, 1, 2], false)
                        .expect("moments")
                })
        };
        let one = run(1);
        for threads in [2, 5, 8] {
            assert_eq!(run(threads), one);
        }
    }

    #[test]
    fn non_positive_weight_sum_is_refused() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let w = array![0.0, 0.0];
        let error = standardization_moments(x.view(), Some(w.view()), &[1], true)
            .expect_err("zero weights are refused");
        assert!(error.to_string().contains("positive finite sum(weights)"));
    }
}
