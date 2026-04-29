//! Rust helpers for pre-fit data exploration.
//!
//! These routines back the Python `explore_data()` API. They operate on
//! already-factorized integer codes so Python can preserve the public output
//! shape while Rust handles the row-heavy aggregation work.

use ndarray::Array2;
use rayon::prelude::*;
use std::collections::HashMap;

use super::distributions::f_cdf;
use super::interactions::InteractionCandidate;

const DENSE_CELL_LIMIT: usize = 2_000_000;

#[derive(Clone, Copy, Debug, Default)]
struct CellAgg {
    count: usize,
    y_sum: f64,
    exposure_sum: f64,
    weighted_rate_sq_sum: f64,
}

impl CellAgg {
    fn add(&mut self, y: f64, exposure: f64) {
        self.count += 1;
        self.y_sum += y;
        self.exposure_sum += exposure;
        if exposure > 0.0 {
            self.weighted_rate_sq_sum += y * y / exposure;
        }
    }
}

/// Compute Cramer's V for a pair of pre-factorized categorical columns.
pub fn cramers_v_from_codes(
    x: &[u32],
    y: &[u32],
    x_levels: usize,
    y_levels: usize,
) -> Result<f64, String> {
    let n = x.len();
    if n != y.len() {
        return Err(format!(
            "Cramer's V inputs have different lengths: {} vs {}",
            n,
            y.len()
        ));
    }
    if x_levels < 2 || y_levels < 2 || n == 0 {
        return Ok(0.0);
    }

    match x_levels.checked_mul(y_levels) {
        Some(product) if product <= DENSE_CELL_LIMIT => {
            cramers_v_dense(x, y, x_levels, y_levels, product)
        }
        _ => cramers_v_sparse(x, y, x_levels, y_levels),
    }
}

/// Compute a full symmetric Cramer's V matrix from pre-factorized columns.
pub fn cramers_v_matrix_from_codes(
    codes: &[&[u32]],
    n_levels: &[usize],
) -> Result<Array2<f64>, String> {
    let k = codes.len();
    if n_levels.len() != k {
        return Err(format!(
            "codes has {} columns but n_levels has {} entries",
            k,
            n_levels.len()
        ));
    }
    if k == 0 {
        return Ok(Array2::zeros((0, 0)));
    }

    let n = codes[0].len();
    for (j, col) in codes.iter().enumerate() {
        if col.len() != n {
            return Err(format!(
                "codes[{}] has {} rows but codes[0] has {} rows",
                j,
                col.len(),
                n
            ));
        }
    }

    let pairs: Vec<(usize, usize)> = (0..k)
        .flat_map(|i| ((i + 1)..k).map(move |j| (i, j)))
        .collect();

    let pair_values: Result<Vec<(usize, usize, f64)>, String> = pairs
        .into_par_iter()
        .map(|(i, j)| {
            cramers_v_from_codes(codes[i], codes[j], n_levels[i], n_levels[j]).map(|v| (i, j, v))
        })
        .collect();

    let mut matrix = Array2::<f64>::eye(k);
    for (i, j, v) in pair_values? {
        matrix[[i, j]] = v;
        matrix[[j, i]] = v;
    }
    Ok(matrix)
}

fn cramers_v_dense(
    x: &[u32],
    y: &[u32],
    x_levels: usize,
    y_levels: usize,
    product: usize,
) -> Result<f64, String> {
    let mut row_sums = vec![0.0_f64; x_levels];
    let mut col_sums = vec![0.0_f64; y_levels];
    let mut counts = vec![0_u32; product];

    for (&xi_u32, &yi_u32) in x.iter().zip(y.iter()) {
        let xi = xi_u32 as usize;
        let yi = yi_u32 as usize;
        if xi >= x_levels || yi >= y_levels {
            return Err(format!(
                "Cramer's V code out of range: ({}, {}) for levels ({}, {})",
                xi, yi, x_levels, y_levels
            ));
        }
        row_sums[xi] += 1.0;
        col_sums[yi] += 1.0;
        counts[xi * y_levels + yi] += 1;
    }

    finish_cramers_v(&row_sums, &col_sums, |visit| {
        for i in 0..x_levels {
            let row_start = i * y_levels;
            for j in 0..y_levels {
                let observed = counts[row_start + j];
                if observed > 0 {
                    visit(i, j, observed as f64);
                }
            }
        }
    })
}

fn cramers_v_sparse(x: &[u32], y: &[u32], x_levels: usize, y_levels: usize) -> Result<f64, String> {
    let mut row_sums = vec![0.0_f64; x_levels];
    let mut col_sums = vec![0.0_f64; y_levels];
    let mut counts: HashMap<u64, u32> = HashMap::with_capacity(x.len().min(65_536));
    let y_levels_u64 = y_levels as u64;

    for (&xi_u32, &yi_u32) in x.iter().zip(y.iter()) {
        let xi = xi_u32 as usize;
        let yi = yi_u32 as usize;
        if xi >= x_levels || yi >= y_levels {
            return Err(format!(
                "Cramer's V code out of range: ({}, {}) for levels ({}, {})",
                xi, yi, x_levels, y_levels
            ));
        }
        row_sums[xi] += 1.0;
        col_sums[yi] += 1.0;
        let key = xi as u64 * y_levels_u64 + yi as u64;
        *counts.entry(key).or_insert(0) += 1;
    }

    finish_cramers_v(&row_sums, &col_sums, |visit| {
        for (key, observed) in counts {
            let i = (key / y_levels_u64) as usize;
            let j = (key % y_levels_u64) as usize;
            visit(i, j, observed as f64);
        }
    })
}

fn finish_cramers_v<F>(row_sums: &[f64], col_sums: &[f64], visit_nonzero: F) -> Result<f64, String>
where
    F: FnOnce(&mut dyn FnMut(usize, usize, f64)),
{
    if row_sums.contains(&0.0) || col_sums.contains(&0.0) {
        return Err("Cramer's V calculation has zero expected frequencies. \
             This indicates empty cells in the contingency table between factors."
            .to_string());
    }

    let n = row_sums.iter().sum::<f64>();
    if n == 0.0 {
        return Ok(0.0);
    }

    let mut chi2_plus_n = 0.0_f64;
    visit_nonzero(&mut |i, j, observed| {
        let expected = row_sums[i] * col_sums[j] / n;
        chi2_plus_n += observed * observed / expected;
    });

    let chi2 = (chi2_plus_n - n).max(0.0);
    let min_dim = (row_sums.len() - 1).min(col_sums.len() - 1);
    if min_dim == 0 {
        return Ok(0.0);
    }

    Ok((chi2 / (n * min_dim as f64)).sqrt())
}

/// Detect pre-fit interaction candidates using response/exposure aggregates.
///
/// `factor_codes` should contain one integer-coded column per factor. Continuous
/// factors should already be discretized into the desired interaction bins.
pub fn detect_exploratory_interactions_from_codes(
    factor_names: &[String],
    factor_codes: &[&[u32]],
    n_levels: &[usize],
    y: &[f64],
    exposure: &[f64],
    max_factors: usize,
    min_effect_size: f64,
    max_candidates: usize,
    min_cell_count: usize,
) -> Result<Vec<InteractionCandidate>, String> {
    let n_factors = factor_codes.len();
    if factor_names.len() != n_factors || n_levels.len() != n_factors {
        return Err(format!(
            "factor_names, factor_codes, and n_levels must have the same length (got {}, {}, {})",
            factor_names.len(),
            n_factors,
            n_levels.len()
        ));
    }
    let n = y.len();
    if exposure.len() != n {
        return Err(format!(
            "y has {} rows but exposure has {} rows",
            n,
            exposure.len()
        ));
    }
    for (j, codes) in factor_codes.iter().enumerate() {
        if codes.len() != n {
            return Err(format!(
                "factor_codes[{}] has {} rows but y has {} rows",
                j,
                codes.len(),
                n
            ));
        }
    }
    if n == 0 || n_factors < 2 {
        return Ok(Vec::new());
    }

    let total_exposure: f64 = exposure.iter().sum();
    if total_exposure <= 0.0 {
        return Ok(Vec::new());
    }
    let total_y: f64 = y.iter().sum();
    let overall_mean = total_y / total_exposure;

    let ss_total = y
        .iter()
        .zip(exposure.iter())
        .filter(|(_, &exp)| exp > 0.0)
        .map(|(&yi, &exp)| {
            let rate = yi / exp;
            exp * (rate - overall_mean).powi(2)
        })
        .sum::<f64>();

    if ss_total == 0.0 {
        return Ok(Vec::new());
    }

    let mut factor_scores: Vec<(usize, f64)> = (0..n_factors)
        .into_par_iter()
        .map(|idx| {
            let score = eta_squared_from_codes(
                factor_codes[idx],
                n_levels[idx],
                y,
                exposure,
                overall_mean,
                ss_total,
            );
            (idx, score)
        })
        .collect();

    factor_scores.sort_by(|a, b| b.1.total_cmp(&a.1));
    let top_factors: Vec<usize> = factor_scores
        .into_iter()
        .filter(|(_, score)| *score >= min_effect_size)
        .take(max_factors)
        .map(|(idx, _)| idx)
        .collect();

    if top_factors.len() < 2 {
        return Ok(Vec::new());
    }

    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..top_factors.len() {
        for j in (i + 1)..top_factors.len() {
            pairs.push((top_factors[i], top_factors[j]));
        }
    }

    let mut candidates: Vec<InteractionCandidate> = pairs
        .into_par_iter()
        .filter_map(|(i, j)| {
            interaction_strength_from_codes(
                &factor_names[i],
                factor_codes[i],
                n_levels[i],
                &factor_names[j],
                factor_codes[j],
                n_levels[j],
                y,
                exposure,
                min_cell_count,
            )
        })
        .collect();

    candidates.sort_by(|a, b| b.interaction_strength.total_cmp(&a.interaction_strength));
    candidates.truncate(max_candidates);
    Ok(candidates)
}

fn eta_squared_from_codes(
    codes: &[u32],
    n_levels: usize,
    y: &[f64],
    exposure: &[f64],
    overall_mean: f64,
    ss_total: f64,
) -> f64 {
    if n_levels == 0 || ss_total == 0.0 {
        return 0.0;
    }

    let mut level_y = vec![0.0_f64; n_levels];
    let mut level_exp = vec![0.0_f64; n_levels];

    for ((&code, &yi), &exp) in codes.iter().zip(y.iter()).zip(exposure.iter()) {
        let idx = code as usize;
        if idx < n_levels {
            level_y[idx] += yi;
            level_exp[idx] += exp;
        }
    }

    let ss_between = level_y
        .iter()
        .zip(level_exp.iter())
        .filter(|(_, &exp)| exp > 0.0)
        .map(|(&sum_y, &sum_exp)| {
            let level_mean = sum_y / sum_exp;
            sum_exp * (level_mean - overall_mean).powi(2)
        })
        .sum::<f64>();

    ss_between / ss_total
}

#[allow(clippy::too_many_arguments)]
fn interaction_strength_from_codes(
    name1: &str,
    codes1: &[u32],
    n_levels1: usize,
    name2: &str,
    codes2: &[u32],
    n_levels2: usize,
    y: &[f64],
    exposure: &[f64],
    min_cell_count: usize,
) -> Option<InteractionCandidate> {
    if y.len() < min_cell_count.saturating_mul(4) {
        return None;
    }

    let cells = match n_levels1.checked_mul(n_levels2) {
        Some(product) if product <= DENSE_CELL_LIMIT => {
            aggregate_interaction_cells_dense(codes1, n_levels1, codes2, n_levels2, y, exposure)
        }
        _ => aggregate_interaction_cells_sparse(codes1, n_levels2, codes2, y, exposure),
    };

    finish_interaction_candidate(name1, name2, cells, min_cell_count)
}

fn aggregate_interaction_cells_dense(
    codes1: &[u32],
    n_levels1: usize,
    codes2: &[u32],
    n_levels2: usize,
    y: &[f64],
    exposure: &[f64],
) -> Vec<CellAgg> {
    let mut cells = vec![CellAgg::default(); n_levels1 * n_levels2];
    for (((&c1, &c2), &yi), &exp) in codes1
        .iter()
        .zip(codes2.iter())
        .zip(y.iter())
        .zip(exposure.iter())
    {
        let i = c1 as usize;
        let j = c2 as usize;
        if i < n_levels1 && j < n_levels2 {
            cells[i * n_levels2 + j].add(yi, exp);
        }
    }
    cells
}

fn aggregate_interaction_cells_sparse(
    codes1: &[u32],
    n_levels2: usize,
    codes2: &[u32],
    y: &[f64],
    exposure: &[f64],
) -> Vec<CellAgg> {
    let mut cells: HashMap<u64, CellAgg> = HashMap::with_capacity(codes1.len().min(65_536));
    let n_levels2_u64 = n_levels2 as u64;
    for (((&c1, &c2), &yi), &exp) in codes1
        .iter()
        .zip(codes2.iter())
        .zip(y.iter())
        .zip(exposure.iter())
    {
        let key = c1 as u64 * n_levels2_u64 + c2 as u64;
        cells.entry(key).or_default().add(yi, exp);
    }
    cells.into_values().collect()
}

fn finish_interaction_candidate(
    name1: &str,
    name2: &str,
    cells: Vec<CellAgg>,
    min_cell_count: usize,
) -> Option<InteractionCandidate> {
    let valid_cells: Vec<CellAgg> = cells
        .into_iter()
        .filter(|cell| cell.count >= min_cell_count && cell.exposure_sum > 0.0)
        .collect();

    let n_valid_cells = valid_cells.len();
    if n_valid_cells < 4 {
        return None;
    }

    let (n_valid, total_y, total_exposure, weighted_rate_sq_sum) = valid_cells.iter().fold(
        (0_usize, 0.0_f64, 0.0_f64, 0.0_f64),
        |(n_acc, y_acc, exp_acc, wr2_acc), cell| {
            (
                n_acc + cell.count,
                y_acc + cell.y_sum,
                exp_acc + cell.exposure_sum,
                wr2_acc + cell.weighted_rate_sq_sum,
            )
        },
    );

    if n_valid < min_cell_count.saturating_mul(4) || total_exposure <= 0.0 {
        return None;
    }

    let overall_mean = total_y / total_exposure;
    let ss_total = weighted_rate_sq_sum - total_exposure * overall_mean * overall_mean;
    if ss_total <= 0.0 {
        return None;
    }

    let ss_model = valid_cells
        .iter()
        .map(|cell| {
            let cell_mean = cell.y_sum / cell.exposure_sum;
            cell.exposure_sum * (cell_mean - overall_mean).powi(2)
        })
        .sum::<f64>();

    let r_squared = (ss_model / ss_total).clamp(0.0, 1.0);
    let df_model = n_valid_cells - 1;
    let df_resid = n_valid - n_valid_cells;

    let pvalue = if df_model > 0 && df_resid > 0 && ss_total > ss_model {
        let f_stat = (ss_model / df_model as f64) / ((ss_total - ss_model) / df_resid as f64);
        1.0 - f_cdf(f_stat, df_model as f64, df_resid as f64)
    } else {
        f64::NAN
    };

    Some(InteractionCandidate {
        factor1: name1.to_string(),
        factor2: name2.to_string(),
        interaction_strength: r_squared,
        pvalue,
        n_cells: n_valid_cells,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn cramers_v_matches_hand_computed_2x2() {
        let x = [0_u32, 0, 0, 1, 1, 1, 1, 1];
        let y = [0_u32, 0, 1, 0, 0, 1, 1, 1];
        let v = cramers_v_from_codes(&x, &y, 2, 2).expect("2x2 Cramer's V should compute");

        let n = 8.0_f64;
        let expected = (0.5333333333333333_f64 / n).sqrt();
        assert_abs_diff_eq!(v, expected, epsilon = 1e-12);
    }

    #[test]
    fn cramers_v_matrix_is_symmetric() {
        let a = [0_u32, 0, 1, 1, 2, 2];
        let b = [0_u32, 1, 0, 1, 0, 1];
        let c = [1_u32, 1, 1, 0, 0, 0];
        let matrix = cramers_v_matrix_from_codes(&[&a, &b, &c], &[3, 2, 2])
            .expect("Cramer's V matrix should compute");

        assert_eq!(matrix.nrows(), 3);
        assert_eq!(matrix.ncols(), 3);
        for i in 0..3 {
            assert_abs_diff_eq!(matrix[[i, i]], 1.0, epsilon = 0.0);
            for j in 0..3 {
                assert_abs_diff_eq!(matrix[[i, j]], matrix[[j, i]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn exploratory_interactions_identifies_signal_pair() {
        let n = 1_000;
        let mut y = Vec::with_capacity(n);
        let exposure = vec![1.0_f64; n];
        let mut a = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        let mut noise = Vec::with_capacity(n);

        for i in 0..n {
            let ai = (i % 4) as u32;
            let bi = ((i / 4) % 4) as u32;
            a.push(ai);
            b.push(bi);
            noise.push(((i / 16) % 5) as u32);
            let interaction = if ai == bi { 3.0 } else { 0.2 };
            y.push(interaction + (i % 7) as f64 * 0.01);
        }

        let names = vec!["a".to_string(), "b".to_string(), "noise".to_string()];
        let candidates = detect_exploratory_interactions_from_codes(
            &names,
            &[&a, &b, &noise],
            &[4, 4, 5],
            &y,
            &exposure,
            3,
            0.0,
            3,
            20,
        )
        .expect("exploratory interaction detection should compute");

        assert!(!candidates.is_empty());
        let top = &candidates[0];
        assert_eq!(top.factor1, "a");
        assert_eq!(top.factor2, "b");
        assert!(top.interaction_strength > 0.2);
    }
}
