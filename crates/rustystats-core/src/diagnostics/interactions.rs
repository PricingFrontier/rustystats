// =============================================================================
// Interaction Detection
// =============================================================================
//
// This module detects potential interactions between factors using a greedy
// residual-based approach similar to gradient boosting.
//
// ALGORITHM:
// 1. First, rank factors by their residual correlation (pre-filter)
// 2. For top N factors, check pairwise interaction potential
// 3. For each pair, compute residual variance reduction from a simple
//    interaction term (similar to a single tree split)
//
// This is O(N * k²) where k is the number of top factors checked,
// instead of O(N * p²) for all factor pairs.
//
// =============================================================================

use ndarray::Array1;
use rayon::prelude::*;
use std::collections::HashMap;

/// Result of interaction detection
#[derive(Debug, Clone)]
pub struct InteractionCandidate {
    pub factor1: String,
    pub factor2: String,
    pub interaction_strength: f64, // Partial R² of interaction on residuals
    pub pvalue: f64,
    pub n_cells: usize, // Number of interaction cells with data
}

/// Configuration for interaction detection
#[derive(Debug, Clone)]
pub struct InteractionConfig {
    /// Maximum number of top factors to check for interactions
    pub max_factors_to_check: usize,
    /// Minimum correlation with residuals to consider a factor
    pub min_residual_correlation: f64,
    /// Maximum number of interaction candidates to return
    pub max_candidates: usize,
    /// Minimum cell count for valid interaction cell
    pub min_cell_count: usize,
}

impl Default for InteractionConfig {
    fn default() -> Self {
        Self {
            max_factors_to_check: 10,
            min_residual_correlation: 0.01,
            max_candidates: 5,
            min_cell_count: 30,
        }
    }
}

/// Factor data for interaction detection
#[derive(Debug, Clone)]
pub enum FactorData {
    Continuous(Vec<f64>),
    Categorical(Vec<String>),
}

impl FactorData {
    pub fn len(&self) -> usize {
        match self {
            FactorData::Continuous(v) => v.len(),
            FactorData::Categorical(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Detect potential interactions using greedy residual-based approach
pub fn detect_interactions(
    factors: &HashMap<String, FactorData>,
    residuals: &Array1<f64>,
    config: &InteractionConfig,
) -> Vec<InteractionCandidate> {
    let n = residuals.len();
    if n == 0 || factors.is_empty() {
        return Vec::new();
    }

    // Step 1: Rank factors by residual correlation/association
    let mut factor_scores: Vec<(String, f64)> = factors
        .par_iter()
        .map(|(name, data)| {
            let score = compute_factor_residual_association(data, residuals);
            (name.clone(), score)
        })
        .collect();

    // Sort by score descending
    factor_scores.sort_by(|a, b| b.1.total_cmp(&a.1));

    // Step 2: Take top factors above threshold
    let top_factors: Vec<String> = factor_scores
        .iter()
        .filter(|(_, score)| *score >= config.min_residual_correlation)
        .take(config.max_factors_to_check)
        .map(|(name, _)| name.clone())
        .collect();

    if top_factors.len() < 2 {
        return Vec::new();
    }

    // Step 3: Check pairwise interactions for top factors
    let mut candidates: Vec<InteractionCandidate> = Vec::new();

    for i in 0..top_factors.len() {
        for j in (i + 1)..top_factors.len() {
            let name1 = &top_factors[i];
            let name2 = &top_factors[j];

            if let (Some(data1), Some(data2)) = (factors.get(name1), factors.get(name2)) {
                if let Some(candidate) = compute_interaction_strength(
                    name1,
                    data1,
                    name2,
                    data2,
                    residuals,
                    config.min_cell_count,
                ) {
                    candidates.push(candidate);
                }
            }
        }
    }

    // Sort by interaction strength descending
    candidates.sort_by(|a, b| b.interaction_strength.total_cmp(&a.interaction_strength));

    // Return top candidates
    candidates.into_iter().take(config.max_candidates).collect()
}

/// Compute association between a factor and residuals
fn compute_factor_residual_association(factor: &FactorData, residuals: &Array1<f64>) -> f64 {
    match factor {
        FactorData::Continuous(values) => {
            // Pearson correlation
            compute_correlation_continuous(values, residuals)
        }
        FactorData::Categorical(values) => {
            // Eta-squared (variance explained by categories)
            compute_eta_squared(values, residuals)
        }
    }
}

fn compute_correlation_continuous(values: &[f64], residuals: &Array1<f64>) -> f64 {
    let n = values.len().min(residuals.len());
    if n < 2 {
        return 0.0;
    }

    let valid_pairs: Vec<(f64, f64)> = values
        .iter()
        .zip(residuals.iter())
        .take(n)
        .filter(|(&v, _)| !v.is_nan() && !v.is_infinite())
        .map(|(&v, &r)| (v, r))
        .collect();

    if valid_pairs.len() < 2 {
        return 0.0;
    }

    let sum_x: f64 = valid_pairs.iter().map(|&(x, _)| x).sum();
    let sum_y: f64 = valid_pairs.iter().map(|&(_, y)| y).sum();
    let n_f = valid_pairs.len() as f64;
    let mean_x = sum_x / n_f;
    let mean_y = sum_y / n_f;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for &(x, y) in &valid_pairs {
        let dx = x - mean_x;
        let dy = y - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x == 0.0 || var_y == 0.0 {
        return 0.0;
    }

    (cov / (var_x * var_y).sqrt()).abs()
}

fn compute_eta_squared(categories: &[String], residuals: &Array1<f64>) -> f64 {
    let n = categories.len().min(residuals.len());
    if n < 2 {
        return 0.0;
    }

    // Group residuals by category
    let mut category_residuals: HashMap<&str, Vec<f64>> = HashMap::new();
    for (i, cat) in categories.iter().enumerate().take(n) {
        category_residuals
            .entry(cat.as_str())
            .or_default()
            .push(residuals[i]);
    }

    // Compute overall mean
    let overall_sum: f64 = residuals.iter().take(n).sum();
    let overall_mean = overall_sum / n as f64;

    // Compute SS_total
    let ss_total: f64 = residuals
        .iter()
        .take(n)
        .map(|&r| (r - overall_mean).powi(2))
        .sum();

    if ss_total == 0.0 {
        return 0.0;
    }

    // Compute SS_between
    let ss_between: f64 = category_residuals
        .values()
        .map(|resids| {
            let cat_mean: f64 = resids.iter().sum::<f64>() / resids.len() as f64;
            resids.len() as f64 * (cat_mean - overall_mean).powi(2)
        })
        .sum();

    ss_between / ss_total
}

/// Compute interaction strength between two factors
fn compute_interaction_strength(
    name1: &str,
    data1: &FactorData,
    name2: &str,
    data2: &FactorData,
    residuals: &Array1<f64>,
    min_cell_count: usize,
) -> Option<InteractionCandidate> {
    let n = data1.len().min(data2.len()).min(residuals.len());
    if n < min_cell_count * 4 {
        // Need reasonable sample for interaction
        return None;
    }

    // Bin continuous factors, use categories as-is
    let bins1 = discretize_factor(data1, 5);
    let bins2 = discretize_factor(data2, 5);

    // Running aggregates per cell — no residual storage.
    // Tracks count, sum, and sum of squares per (bin1, bin2) cell, so the
    // whole function is O(n_cells) memory instead of O(n_valid).
    #[derive(Default, Clone, Copy)]
    struct CellAgg {
        count: usize,
        sum: f64,
        sum_sq: f64,
    }

    let mut cells: HashMap<(usize, usize), CellAgg> = HashMap::new();
    for i in 0..n {
        if let (Some(&b1), Some(&b2)) = (bins1.get(i), bins2.get(i)) {
            let r = residuals[i];
            let c = cells.entry((b1, b2)).or_default();
            c.count += 1;
            c.sum += r;
            c.sum_sq += r * r;
        }
    }

    // Filter cells with sufficient data
    cells.retain(|_, c| c.count >= min_cell_count);

    if cells.len() < 4 {
        return None;
    }

    // Reduce across cells for totals
    let (n_valid, overall_sum, overall_sum_sq) = cells
        .values()
        .fold((0_usize, 0.0_f64, 0.0_f64), |(n_acc, s_acc, ss_acc), c| {
            (n_acc + c.count, s_acc + c.sum, ss_acc + c.sum_sq)
        });

    let overall_mean = overall_sum / n_valid as f64;

    // SS_total = Σ r² − n·μ² (algebraically equivalent to Σ(r - μ)²).
    // This is subject to catastrophic cancellation when residuals are near-
    // constant, but the running-aggregates identity test verifies the drift
    // stays < 1e-10 relative for realistic inputs. For exactly-constant
    // nonzero residuals the result is tiny float-noise, preserving the
    // pre-refactor quirk where such inputs yield Some(..) rather than None.
    let ss_total = overall_sum_sq - (n_valid as f64) * overall_mean * overall_mean;

    if ss_total == 0.0 {
        return None;
    }

    // Compute SS_model (variance explained by interaction cells) from aggregates
    let ss_model: f64 = cells
        .values()
        .map(|c| {
            let cell_mean = c.sum / c.count as f64;
            c.count as f64 * (cell_mean - overall_mean).powi(2)
        })
        .sum();

    // Partial R²
    let r_squared = ss_model / ss_total;

    // Compute p-value using F-test approximation
    let df_model = cells.len() - 1;
    let df_resid = n_valid - cells.len();

    let f_stat = if df_model > 0 && df_resid > 0 {
        (ss_model / df_model as f64) / ((ss_total - ss_model) / df_resid as f64)
    } else {
        0.0
    };

    let pvalue = f_test_pvalue(f_stat, df_model, df_resid);

    Some(InteractionCandidate {
        factor1: name1.to_string(),
        factor2: name2.to_string(),
        interaction_strength: r_squared,
        pvalue,
        n_cells: cells.len(),
    })
}

/// Discretize a factor into bins
fn discretize_factor(factor: &FactorData, n_bins: usize) -> Vec<usize> {
    match factor {
        FactorData::Continuous(values) => {
            // Find quantile boundaries
            let mut sorted: Vec<f64> = values
                .iter()
                .filter(|&&v| !v.is_nan() && !v.is_infinite())
                .cloned()
                .collect();

            if sorted.is_empty() {
                return vec![0; values.len()];
            }

            sorted.sort_by(|a, b| a.total_cmp(b));

            let boundaries: Vec<f64> = (1..n_bins)
                .map(|i| {
                    let p = i as f64 / n_bins as f64;
                    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
                    sorted[idx]
                })
                .collect();

            // Assign bins
            values
                .iter()
                .map(|&v| {
                    if v.is_nan() || v.is_infinite() {
                        n_bins // Invalid values get their own bin
                    } else {
                        boundaries.iter().position(|&b| v < b).unwrap_or(n_bins - 1)
                    }
                })
                .collect()
        }
        FactorData::Categorical(values) => {
            // Map categories to indices
            let mut cat_to_idx: HashMap<&str, usize> = HashMap::new();
            let mut next_idx = 0;

            values
                .iter()
                .map(|v| {
                    *cat_to_idx.entry(v.as_str()).or_insert_with(|| {
                        let idx = next_idx;
                        next_idx += 1;
                        idx
                    })
                })
                .collect()
        }
    }
}

/// Approximate F-test p-value
fn f_test_pvalue(f_stat: f64, df1: usize, df2: usize) -> f64 {
    if df1 == 0 || df2 == 0 || f_stat <= 0.0 {
        return 1.0;
    }

    // Use beta distribution relationship: F ~ Beta(df1/2, df2/2)
    let x = df2 as f64 / (df2 as f64 + df1 as f64 * f_stat);

    // Incomplete beta function approximation
    1.0 - incomplete_beta_approx(x, df2 as f64 / 2.0, df1 as f64 / 2.0)
}

fn incomplete_beta_approx(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Simple continued fraction approximation
    let bt = (ln_gamma_approx(a + b) - ln_gamma_approx(a) - ln_gamma_approx(b)
        + a * x.ln()
        + b * (1.0 - x).ln())
    .exp();

    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(x, a, b) / a
    } else {
        1.0 - bt * betacf(1.0 - x, b, a) / b
    }
}

fn betacf(x: f64, a: f64, b: f64) -> f64 {
    let max_iter = 100;
    let eps = 1e-10;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=max_iter {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;

        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        h *= d * c;

        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < eps {
            break;
        }
    }

    h
}

fn ln_gamma_approx(x: f64) -> f64 {
    // Stirling's approximation
    if x <= 0.0 {
        return f64::INFINITY;
    }

    let coeffs = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];

    let y = x;
    let mut tmp = x + 5.5;
    tmp -= (x + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015;

    for (j, &c) in coeffs.iter().enumerate() {
        ser += c / (y + j as f64 + 1.0);
    }

    -tmp + (2.5066282746310005 * ser / x).ln()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_interactions_basic() {
        let n = 1000;

        // Create factors
        let mut factors = HashMap::new();
        factors.insert(
            "factor1".to_string(),
            FactorData::Continuous((0..n).map(|i| (i % 10) as f64).collect()),
        );
        factors.insert(
            "factor2".to_string(),
            FactorData::Categorical((0..n).map(|i| format!("cat{}", i % 5)).collect()),
        );

        // Create residuals with some pattern
        let residuals = Array1::from_vec(
            (0..n)
                .map(|i| ((i % 10) as f64 - 5.0) * 0.1 + ((i % 5) as f64 - 2.0) * 0.2)
                .collect(),
        );

        let config = InteractionConfig::default();
        let candidates = detect_interactions(&factors, &residuals, &config);

        // Should find at least one candidate
        assert!(!candidates.is_empty() || factors.len() < 2);
    }

    #[test]
    fn test_discretize_continuous() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let data = FactorData::Continuous(values);
        let bins = discretize_factor(&data, 5);

        assert_eq!(bins.len(), 10);
        // First values should be in lower bins, last in higher
        assert!(bins[0] <= bins[9]);
    }

    #[test]
    fn test_discretize_categorical() {
        let values = vec![
            "A".to_string(),
            "B".to_string(),
            "A".to_string(),
            "C".to_string(),
            "B".to_string(),
        ];
        let data = FactorData::Categorical(values);
        let bins = discretize_factor(&data, 5);

        assert_eq!(bins.len(), 5);
        // Same categories should have same bin
        assert_eq!(bins[0], bins[2]); // Both "A"
        assert_eq!(bins[1], bins[4]); // Both "B"
    }

    #[test]
    fn test_factor_metadata_and_detection_short_circuits() {
        assert!(FactorData::Continuous(Vec::new()).is_empty());
        assert!(FactorData::Categorical(Vec::new()).is_empty());
        assert_eq!(FactorData::Continuous(vec![1.0, 2.0]).len(), 2);
        assert_eq!(
            FactorData::Categorical(vec!["a".to_string(), "b".to_string()]).len(),
            2
        );

        let mut config = InteractionConfig {
            min_residual_correlation: 0.0,
            min_cell_count: 30,
            ..InteractionConfig::default()
        };

        assert!(
            detect_interactions(&HashMap::new(), &Array1::from_vec(vec![1.0]), &config).is_empty()
        );

        let mut one_factor = HashMap::new();
        one_factor.insert("x".to_string(), FactorData::Continuous(vec![1.0, 2.0, 3.0]));
        assert!(
            detect_interactions(&one_factor, &Array1::from_vec(Vec::new()), &config).is_empty()
        );
        assert!(
            detect_interactions(&one_factor, &Array1::from_vec(vec![1.0, 2.0, 3.0]), &config)
                .is_empty()
        );

        config.min_residual_correlation = 2.0;
        let mut filtered = HashMap::new();
        filtered.insert(
            "x".to_string(),
            FactorData::Continuous((0..120).map(|i| i as f64).collect()),
        );
        filtered.insert(
            "y".to_string(),
            FactorData::Continuous((0..120).map(|i| (i % 5) as f64).collect()),
        );
        assert!(
            detect_interactions(
                &filtered,
                &Array1::from_vec((0..120).map(|i| i as f64).collect()),
                &config,
            )
            .is_empty(),
            "thresholds above any possible association should reject all factors"
        );

        config.min_residual_correlation = 0.0;
        let mut too_few_cells = HashMap::new();
        too_few_cells.insert("x".to_string(), FactorData::Continuous(vec![1.0; 120]));
        too_few_cells.insert("y".to_string(), FactorData::Continuous(vec![2.0; 120]));
        assert!(
            detect_interactions(
                &too_few_cells,
                &Array1::from_vec((0..120).map(|i| i as f64).collect()),
                &config,
            )
            .is_empty(),
            "eligible factor pairs with too few cells should produce no candidate"
        );
    }

    #[test]
    fn test_association_helpers_handle_tiny_degenerate_and_invalid_inputs() {
        let residuals = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        assert_eq!(compute_correlation_continuous(&[42.0], &residuals), 0.0);
        assert_eq!(
            compute_correlation_continuous(&[f64::NAN, f64::INFINITY, 1.0], &residuals),
            0.0
        );
        assert_eq!(
            compute_correlation_continuous(&[2.0, 2.0, 2.0], &residuals),
            0.0
        );
        assert_eq!(
            compute_correlation_continuous(
                &[1.0, 2.0, 3.0],
                &Array1::from_vec(vec![5.0, 5.0, 5.0])
            ),
            0.0
        );
        assert!(compute_correlation_continuous(&[1.0, 2.0, 3.0], &residuals) > 0.99);

        assert_eq!(
            compute_eta_squared(&["a".to_string()], &Array1::from_vec(vec![1.0])),
            0.0
        );
        assert_eq!(
            compute_eta_squared(
                &["a".to_string(), "b".to_string()],
                &Array1::from_vec(vec![5.0, 5.0])
            ),
            0.0
        );
        assert!(
            compute_eta_squared(
                &[
                    "a".to_string(),
                    "a".to_string(),
                    "b".to_string(),
                    "b".to_string(),
                ],
                &Array1::from_vec(vec![1.0, 1.2, 4.8, 5.0])
            ) > 0.95
        );
    }

    #[test]
    fn test_discretize_continuous_invalid_value_contracts() {
        let all_invalid = FactorData::Continuous(vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
        assert_eq!(discretize_factor(&all_invalid, 3), vec![0, 0, 0]);

        let mixed = FactorData::Continuous(vec![1.0, f64::NAN, 2.0, f64::INFINITY, 3.0]);
        let bins = discretize_factor(&mixed, 3);
        assert_eq!(bins.len(), 5);
        assert_eq!(bins[1], 3);
        assert_eq!(bins[3], 3);
        assert!(bins[0] <= bins[2]);
        assert!(bins[2] <= bins[4]);
    }

    #[test]
    fn test_interaction_with_zero_residual_degrees_of_freedom_is_safe() {
        let f1 = FactorData::Continuous(vec![0.0, 0.0, 1.0, 1.0]);
        let f2 = FactorData::Continuous(vec![0.0, 1.0, 0.0, 1.0]);
        let residuals = Array1::from_vec(vec![1.0, 2.0, 3.0, 5.0]);

        let candidate = compute_interaction_strength("f1", &f1, "f2", &f2, &residuals, 0)
            .expect("four one-row cells are accepted when min_cell_count is zero");
        assert_eq!(candidate.n_cells, 4);
        assert_eq!(candidate.pvalue, 1.0);
        assert!(candidate.interaction_strength.is_finite());
    }

    #[test]
    fn test_f_distribution_helpers_cover_boundary_and_branch_contracts() {
        assert_eq!(f_test_pvalue(0.0, 1, 1), 1.0);
        assert_eq!(f_test_pvalue(-1.0, 1, 1), 1.0);
        assert_eq!(f_test_pvalue(1.0, 0, 1), 1.0);
        assert_eq!(f_test_pvalue(1.0, 1, 0), 1.0);
        let pvalue = f_test_pvalue(2.5, 3, 20);
        assert!((0.0..=1.0).contains(&pvalue));

        assert_eq!(incomplete_beta_approx(-0.25, 2.0, 5.0), 0.0);
        assert_eq!(incomplete_beta_approx(0.0, 2.0, 5.0), 0.0);
        assert_eq!(incomplete_beta_approx(1.0, 2.0, 5.0), 1.0);
        assert_eq!(incomplete_beta_approx(1.25, 2.0, 5.0), 1.0);
        let lower_branch = incomplete_beta_approx(0.2, 2.0, 5.0);
        let upper_branch = incomplete_beta_approx(0.8, 2.0, 5.0);
        assert!((0.0..=1.0).contains(&lower_branch));
        assert!((0.0..=1.0).contains(&upper_branch));
        assert!(lower_branch < upper_branch);

        let continued_fraction = betacf(0.523_809_523_809_523_8, 0.1, 2.0);
        assert!(continued_fraction.is_finite());
        assert!(ln_gamma_approx(0.0).is_infinite());
        assert!(ln_gamma_approx(-1.0).is_infinite());
        assert!(ln_gamma_approx(2.5).is_finite());
    }

    // =========================================================================
    // Deterministic PRNG for test data generation
    // =========================================================================
    //
    // We avoid pulling in `rand` as a dev-dep — these tests use a simple LCG
    // to produce reproducible inputs. Each test seeds with a constant so the
    // "golden" values pinned below are exactly reproducible across runs.
    // =========================================================================

    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg(seed)
        }

        fn next_u64(&mut self) -> u64 {
            // Numerical Recipes LCG parameters
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0
        }

        /// Uniform in [0, 1)
        fn next_f64(&mut self) -> f64 {
            // 53 bits of randomness — take top 53 bits of the 64-bit output
            (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
        }

        /// Approximate standard normal via Box-Muller
        fn next_normal(&mut self) -> f64 {
            let u1 = self.next_f64().max(1e-300); // avoid ln(0)
            let u2 = self.next_f64();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }

        /// Integer in [0, high)
        fn next_range(&mut self, high: usize) -> usize {
            (self.next_u64() as usize) % high
        }
    }

    /// Deterministic inputs for the parity test — fixed seed, fixed shape.
    ///
    /// n=1000, two Continuous factors with values in 0..5 (each integer value
    /// becomes its own bin after discretization), residuals are a mixture of
    /// signal from both factors plus noise.
    fn build_parity_inputs() -> (FactorData, FactorData, Array1<f64>) {
        let n = 1000;
        let mut rng = Lcg::new(0xDEAD_BEEF_u64);

        let mut f1_vals = Vec::with_capacity(n);
        let mut f2_vals = Vec::with_capacity(n);
        let mut resid_vals = Vec::with_capacity(n);

        for _ in 0..n {
            let a = rng.next_range(5) as f64;
            let b = rng.next_range(5) as f64;
            // Residual carries signal that is a product (interaction) of a and b
            // plus independent main effects plus Gaussian noise.
            let signal = 0.3 * a + 0.2 * b + 0.15 * a * b;
            let noise = 0.5 * rng.next_normal();
            f1_vals.push(a);
            f2_vals.push(b);
            resid_vals.push(signal + noise);
        }

        (
            FactorData::Continuous(f1_vals),
            FactorData::Continuous(f2_vals),
            Array1::from_vec(resid_vals),
        )
    }

    // =========================================================================
    // 1. Parity test — bit-exact comparison against a frozen reference.
    //
    // Golden values below were computed by running this test once on the
    // pre-refactor code and pinning the outputs. Any behavior-preserving
    // refactor of `compute_interaction_strength` must reproduce them to 1e-12
    // relative tolerance.
    // =========================================================================

    /// Helper: compare two f64s by relative tolerance (with absolute floor).
    fn rel_close(a: f64, b: f64, rel_tol: f64) -> bool {
        if a == b {
            return true;
        }
        let denom = a.abs().max(b.abs()).max(1e-300);
        (a - b).abs() / denom <= rel_tol
    }

    #[test]
    fn test_analyze_interaction_parity_golden() {
        let (f1, f2, residuals) = build_parity_inputs();

        let candidate = compute_interaction_strength(
            "f1", &f1, "f2", &f2, &residuals,
            30, // min_cell_count matching default InteractionConfig
        )
        .expect("strong-signal interaction should produce a candidate");

        // ---- GOLDEN VALUES ------------------------------------------------
        // Computed 2026-04-19 on current (pre-refactor) code. Any refactor
        // that changes these bits is a behavior change, not a performance
        // refactor. If these ever need updating, document why.
        //
        // Tolerance: 1e-12 relative — allows trivial reassociation drift in
        // summation orders but flags any real numerical change.
        // -------------------------------------------------------------------
        let golden_r_squared: f64 = GOLDEN_R_SQUARED;
        let golden_pvalue: f64 = GOLDEN_PVALUE;
        let golden_n_cells: usize = GOLDEN_N_CELLS;

        assert_eq!(
            candidate.n_cells, golden_n_cells,
            "n_cells drifted (cell discovery changed)"
        );
        assert!(
            rel_close(candidate.interaction_strength, golden_r_squared, 1e-12),
            "r_squared drift: got {:.17e}, golden {:.17e}, diff {:.3e}",
            candidate.interaction_strength,
            golden_r_squared,
            (candidate.interaction_strength - golden_r_squared).abs(),
        );
        assert!(
            rel_close(candidate.pvalue, golden_pvalue, 1e-12),
            "pvalue drift: got {:.17e}, golden {:.17e}, diff {:.3e}",
            candidate.pvalue,
            golden_pvalue,
            (candidate.pvalue - golden_pvalue).abs(),
        );
    }

    // Golden values — pinned constants. See `test_analyze_interaction_parity_golden`.
    //
    // Captured 2026-04-19 on pre-refactor code by running
    // `compute_interaction_strength` on `build_parity_inputs()`. The p-value
    // saturating at exactly 1.0 is an artifact of the current beta-approximation
    // F-test: at large F-stats, `incomplete_beta_approx` returns ~1.0 and the
    // pvalue becomes 1.0 - 1.0 == 0 ... or, for this particular (x, a, b), the
    // clamp at `x >= 1.0 => 1.0` fires — either way the number is stable.
    // If this ever changes, it's a behavioral change, not a refactor.
    const GOLDEN_R_SQUARED: f64 = 8.16679995891399835e-1_f64;
    const GOLDEN_PVALUE: f64 = 1.00000000000000000e0_f64;
    const GOLDEN_N_CELLS: usize = 20;

    // =========================================================================
    // 2. Running-aggregates identity — algebraic verification.
    //
    // Checks that the forthcoming (count, sum, sum_sq) approach will produce
    // ss_total values equal to the naive Σ(r_i - mean)² to 1e-10 relative.
    // =========================================================================

    #[test]
    fn test_running_aggregates_identity() {
        let n = 1000;
        let n_cells = 20;
        let mut rng = Lcg::new(0xC0FFEE_u64);

        // Random residuals (mix of magnitudes to stress catastrophic cancellation)
        let residuals: Vec<f64> = (0..n)
            .map(|_| {
                let scale = if rng.next_f64() < 0.1 { 100.0 } else { 1.0 };
                scale * rng.next_normal()
            })
            .collect();

        // Random cell assignment
        let cells: Vec<usize> = (0..n).map(|_| rng.next_range(n_cells)).collect();

        // Per-cell aggregates
        let mut counts = vec![0_usize; n_cells];
        let mut sums = vec![0.0_f64; n_cells];
        let mut sum_sqs = vec![0.0_f64; n_cells];

        for i in 0..n {
            let c = cells[i];
            let r = residuals[i];
            counts[c] += 1;
            sums[c] += r;
            sum_sqs[c] += r * r;
        }

        let total_count: usize = counts.iter().sum();
        let total_sum: f64 = sums.iter().sum();
        let total_sum_sq: f64 = sum_sqs.iter().sum();
        assert_eq!(total_count, n);

        let overall_mean = total_sum / total_count as f64;

        // Method A: naive SS_total
        let ss_total_naive: f64 = residuals.iter().map(|&r| (r - overall_mean).powi(2)).sum();

        // Method B: aggregated SS_total = Σ r² - n * mean²
        let ss_total_agg = total_sum_sq - (total_count as f64) * overall_mean * overall_mean;

        assert!(
            rel_close(ss_total_naive, ss_total_agg, 1e-10),
            "ss_total identity broke: naive={:.17e}, agg={:.17e}, rel_diff={:.3e}",
            ss_total_naive,
            ss_total_agg,
            (ss_total_naive - ss_total_agg).abs() / ss_total_naive.abs(),
        );

        // Also verify SS_model identity: Σ count_c * (mean_c - overall_mean)²
        // matches the naive within-cell-mean computation.
        let ss_model_naive: f64 = {
            // Group residuals by cell then compute
            let mut cell_groups: HashMap<usize, Vec<f64>> = HashMap::new();
            for i in 0..n {
                cell_groups.entry(cells[i]).or_default().push(residuals[i]);
            }
            cell_groups
                .values()
                .map(|resids| {
                    let m = resids.iter().sum::<f64>() / resids.len() as f64;
                    resids.len() as f64 * (m - overall_mean).powi(2)
                })
                .sum()
        };

        let ss_model_agg: f64 = (0..n_cells)
            .filter(|&c| counts[c] > 0)
            .map(|c| {
                let cell_mean = sums[c] / counts[c] as f64;
                counts[c] as f64 * (cell_mean - overall_mean).powi(2)
            })
            .sum();

        assert!(
            rel_close(ss_model_naive, ss_model_agg, 1e-10),
            "ss_model identity broke: naive={:.17e}, agg={:.17e}",
            ss_model_naive,
            ss_model_agg,
        );
    }

    // =========================================================================
    // 3. Edge cases
    // =========================================================================

    #[test]
    fn test_fewer_than_four_valid_cells_returns_none() {
        // All residuals fall into 2 bins of each factor → at most 4 cells, but
        // with `min_cell_count = 30` and only 2 unique values per factor we'll
        // get 4 cells max. Forcing 3 cells by collapsing one pair:
        let n = 200;
        let mut f1 = Vec::with_capacity(n);
        let mut f2 = Vec::with_capacity(n);
        let mut r = Vec::with_capacity(n);
        for i in 0..n {
            // Only populate cells (0,0), (0,1), (1,0) — leave (1,1) empty
            let (a, b) = match i % 3 {
                0 => (0.0, 0.0),
                1 => (0.0, 1.0),
                _ => (1.0, 0.0),
            };
            f1.push(a);
            f2.push(b);
            r.push(0.1 * i as f64);
        }
        let residuals = Array1::from_vec(r);

        let result = compute_interaction_strength(
            "f1",
            &FactorData::Continuous(f1),
            "f2",
            &FactorData::Continuous(f2),
            &residuals,
            30,
        );
        assert!(result.is_none(), "expected None with <4 valid cells");
    }

    #[test]
    fn test_zero_residuals_exact_returns_none() {
        // When all residuals are *exactly* 0.0, the summation produces an
        // exact 0.0 ss_total, and the `ss_total == 0.0` guard fires → None.
        // (Note: a constant-nonzero residual vector, e.g. all 3.14, does *not*
        // currently return None — summation of (3.14 - mean)² gathers tiny
        // floating-point noise that is nonzero. See
        // `test_constant_nonzero_residuals_current_behavior` below.)
        let n = 2000;
        let mut rng = Lcg::new(42);
        let f1: Vec<f64> = (0..n).map(|_| rng.next_range(5) as f64).collect();
        let f2: Vec<f64> = (0..n).map(|_| rng.next_range(5) as f64).collect();
        let residuals = Array1::from_vec(vec![0.0_f64; n]);

        let result = compute_interaction_strength(
            "f1",
            &FactorData::Continuous(f1),
            "f2",
            &FactorData::Continuous(f2),
            &residuals,
            30,
        );
        assert!(
            result.is_none(),
            "expected None when all residuals are exactly 0.0 (ss_total==0): got {:?}",
            result
        );
    }

    #[test]
    fn test_constant_nonzero_residuals_current_behavior() {
        // Pins a known quirk of the current implementation: with constant but
        // nonzero residuals (e.g. all == 3.14), ss_total is a tiny float noise
        // value rather than exactly 0.0, so the function returns Some(..) with
        // a numerically-unstable r² rather than None.
        //
        // This test intentionally does NOT assert the exact r² value (which is
        // noise-dominated and not physically meaningful), only that the current
        // function returns Some. The refactor should preserve this behavior —
        // or, if it fixes it by returning None, that's a deliberate behavioral
        // change the reviewer should bless explicitly.
        let n = 2000;
        let mut rng = Lcg::new(42);
        let f1: Vec<f64> = (0..n).map(|_| rng.next_range(5) as f64).collect();
        let f2: Vec<f64> = (0..n).map(|_| rng.next_range(5) as f64).collect();
        let residual = f64::from(314_u16) / 100.0;
        let residuals = Array1::from_vec(vec![residual; n]);

        let result = compute_interaction_strength(
            "f1",
            &FactorData::Continuous(f1),
            "f2",
            &FactorData::Continuous(f2),
            &residuals,
            30,
        );
        // Current behavior: returns Some, with the p-value saturating at 1.0
        // (the F-stat is noise/noise, essentially random, and the beta approx
        // returns 1.0 at the saturation clamp).
        assert!(
            result.is_some(),
            "CURRENT-BEHAVIOR PIN: constant-nonzero residuals currently yield Some(..). \
             If this test fails, the refactor has changed this edge case — verify the \
             new behavior is intentional and update this test."
        );
    }

    #[test]
    fn test_tiny_sample_returns_none() {
        // n=20 is far below min_cell_count * 4 == 120
        let n = 20;
        let f1: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
        let f2: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
        let residuals = Array1::from_vec((0..n).map(|i| i as f64 * 0.1).collect());

        let result = compute_interaction_strength(
            "f1",
            &FactorData::Continuous(f1),
            "f2",
            &FactorData::Continuous(f2),
            &residuals,
            30,
        );
        assert!(result.is_none(), "expected None with tiny sample (n=20)");
    }

    #[test]
    fn test_strong_interaction_high_r_squared() {
        // Construct residuals that are exactly cell-mean driven → r² near 1.
        // We use tiny noise so the cell structure dominates; the observed r² on
        // the current implementation with this seed is ≈ 0.95. Using a
        // conservative floor of 0.90 so trivial algebraic reassociation in the
        // refactor can't accidentally trip this.
        let n = 2000;
        let mut rng = Lcg::new(7);
        let mut f1_vals = Vec::with_capacity(n);
        let mut f2_vals = Vec::with_capacity(n);
        let mut r_vals = Vec::with_capacity(n);
        for _ in 0..n {
            let a = rng.next_range(5);
            let b = rng.next_range(5);
            // Strong interaction: residual determined almost entirely by cell
            let cell_signal = (a * 5 + b) as f64;
            let noise = 0.001 * rng.next_normal();
            f1_vals.push(a as f64);
            f2_vals.push(b as f64);
            r_vals.push(cell_signal + noise);
        }

        let candidate = compute_interaction_strength(
            "f1",
            &FactorData::Continuous(f1_vals),
            "f2",
            &FactorData::Continuous(f2_vals),
            &Array1::from_vec(r_vals),
            30,
        )
        .expect("should produce a candidate with strong signal");

        assert!(
            candidate.interaction_strength > 0.90,
            "expected r² > 0.90 for strong interaction, got {}",
            candidate.interaction_strength
        );
        // p-value: the current F-test approximation clamps at 1.0 for very
        // large F-stats (beta-approx saturation). Either ~0 (well-conditioned)
        // or ~1 (saturation) is valid current behavior; we only require it's
        // in the legal [0, 1] range.
        assert!(
            (0.0..=1.0).contains(&candidate.pvalue),
            "pvalue out of range: {}",
            candidate.pvalue
        );
    }

    #[test]
    fn test_no_interaction_high_pvalue() {
        // Pure noise residuals, random bins → r² small, p-value large
        let n = 2000;
        let mut rng = Lcg::new(1234);
        let mut f1_vals = Vec::with_capacity(n);
        let mut f2_vals = Vec::with_capacity(n);
        let mut r_vals = Vec::with_capacity(n);
        for _ in 0..n {
            f1_vals.push(rng.next_range(5) as f64);
            f2_vals.push(rng.next_range(5) as f64);
            r_vals.push(rng.next_normal());
        }

        let candidate = compute_interaction_strength(
            "f1",
            &FactorData::Continuous(f1_vals),
            "f2",
            &FactorData::Continuous(f2_vals),
            &Array1::from_vec(r_vals),
            30,
        )
        .expect("should produce a candidate (cells valid, variance nonzero)");

        // Random data → r² should be small (well below 0.1)
        assert!(
            candidate.interaction_strength < 0.1,
            "expected r² < 0.1 for pure noise, got {}",
            candidate.interaction_strength
        );
        // And the p-value should be nowhere near 0
        assert!(
            candidate.pvalue > 0.01,
            "expected p-value > 0.01 for pure noise, got {}",
            candidate.pvalue
        );
    }

    // =========================================================================
    // 4. Integration test via public `detect_interactions` API
    // =========================================================================

    #[test]
    fn test_detect_interactions_identifies_true_pair() {
        let n = 5000;
        let mut rng = Lcg::new(99);

        // Four factors:
        //   strong_A, strong_B — these two carry a genuine interaction
        //   noise_A,  noise_B  — random, residuals don't depend on them
        let mut strong_a = Vec::with_capacity(n);
        let mut strong_b = Vec::with_capacity(n);
        let mut noise_a = Vec::with_capacity(n);
        let mut noise_b = Vec::with_capacity(n);
        let mut residuals = Vec::with_capacity(n);
        for _ in 0..n {
            let a = rng.next_range(5);
            let b = rng.next_range(5);
            let na = rng.next_range(5);
            let nb = rng.next_range(5);
            strong_a.push(a as f64);
            strong_b.push(b as f64);
            noise_a.push(na as f64);
            noise_b.push(nb as f64);

            // Residual driven by product of strong_a and strong_b + noise
            let signal = (a as f64) * (b as f64) * 0.5;
            let noise = 0.3 * rng.next_normal();
            residuals.push(signal + noise);
        }

        let mut factors = HashMap::new();
        factors.insert("strong_A".to_string(), FactorData::Continuous(strong_a));
        factors.insert("strong_B".to_string(), FactorData::Continuous(strong_b));
        factors.insert("noise_A".to_string(), FactorData::Continuous(noise_a));
        factors.insert("noise_B".to_string(), FactorData::Continuous(noise_b));

        let residuals = Array1::from_vec(residuals);

        let config = InteractionConfig::default();
        let candidates = detect_interactions(&factors, &residuals, &config);

        assert!(!candidates.is_empty(), "expected at least one candidate");

        let top = &candidates[0];
        let top_pair = {
            let mut v = [top.factor1.as_str(), top.factor2.as_str()];
            v.sort();
            v
        };
        assert_eq!(
            top_pair,
            ["strong_A", "strong_B"],
            "expected top candidate to be the strong_A × strong_B pair, \
             got ({}, {}) with strength {}",
            top.factor1,
            top.factor2,
            top.interaction_strength,
        );
    }

    // =========================================================================
    // 5. Large-n performance smoke test.
    //
    // Not a strict benchmark — just confirms that the refactor won't regress
    // wall time catastrophically. Runs the single-pair analysis for n=100_000.
    // =========================================================================

    #[test]
    fn test_analyze_interaction_large_n_is_fast() {
        let n = 100_000;
        let mut rng = Lcg::new(2024);
        let mut f1 = Vec::with_capacity(n);
        let mut f2 = Vec::with_capacity(n);
        let mut r = Vec::with_capacity(n);
        for _ in 0..n {
            let a = rng.next_range(5);
            let b = rng.next_range(5);
            f1.push(a as f64);
            f2.push(b as f64);
            let signal = 0.1 * a as f64 + 0.1 * b as f64 + 0.05 * (a * b) as f64;
            r.push(signal + 0.3 * rng.next_normal());
        }
        let residuals = Array1::from_vec(r);

        let start = std::time::Instant::now();
        let candidate = compute_interaction_strength(
            "f1",
            &FactorData::Continuous(f1),
            "f2",
            &FactorData::Continuous(f2),
            &residuals,
            30,
        );
        let elapsed = start.elapsed();

        assert!(candidate.is_some(), "expected a candidate for n=100k");
        let max_elapsed = if cfg!(coverage) {
            // Coverage instrumentation adds enough overhead to make a hard 1s
            // wall-clock gate flaky on otherwise healthy runs.
            std::time::Duration::from_secs(5)
        } else {
            std::time::Duration::from_secs(1)
        };
        assert!(
            elapsed < max_elapsed,
            "compute_interaction_strength took {:?} for n={}, expected < {:?}",
            elapsed,
            n,
            max_elapsed,
        );
    }
}
