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
use std::cmp::Ordering;

/// Result of interaction detection
#[derive(Debug, Clone)]
pub struct InteractionCandidate {
    pub factor1: String,
    pub factor2: String,
    pub interaction_strength: f64,  // Partial R² of interaction on residuals
    pub pvalue: f64,
    pub n_cells: usize,  // Number of interaction cells with data
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
    let mut factor_scores: Vec<(String, f64)> = factors.par_iter()
        .map(|(name, data)| {
            let score = compute_factor_residual_association(data, residuals);
            (name.clone(), score)
        })
        .collect();
    
    // Sort by score descending
    factor_scores.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
    });
    
    // Step 2: Take top factors above threshold
    let top_factors: Vec<String> = factor_scores.iter()
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
                    name1, data1,
                    name2, data2,
                    residuals,
                    config.min_cell_count,
                ) {
                    candidates.push(candidate);
                }
            }
        }
    }
    
    // Sort by interaction strength descending
    candidates.sort_by(|a, b| {
        b.interaction_strength.partial_cmp(&a.interaction_strength).unwrap_or(Ordering::Equal)
    });
    
    // Return top candidates
    candidates.into_iter()
        .take(config.max_candidates)
        .collect()
}

/// Compute association between a factor and residuals
fn compute_factor_residual_association(
    factor: &FactorData,
    residuals: &Array1<f64>,
) -> f64 {
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
    
    let valid_pairs: Vec<(f64, f64)> = values.iter()
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
        category_residuals.entry(cat.as_str())
            .or_insert_with(Vec::new)
            .push(residuals[i]);
    }
    
    // Compute overall mean
    let overall_sum: f64 = residuals.iter().take(n).sum();
    let overall_mean = overall_sum / n as f64;
    
    // Compute SS_total
    let ss_total: f64 = residuals.iter()
        .take(n)
        .map(|&r| (r - overall_mean).powi(2))
        .sum();
    
    if ss_total == 0.0 {
        return 0.0;
    }
    
    // Compute SS_between
    let ss_between: f64 = category_residuals.iter()
        .map(|(_, resids)| {
            let cat_mean: f64 = resids.iter().sum::<f64>() / resids.len() as f64;
            resids.len() as f64 * (cat_mean - overall_mean).powi(2)
        })
        .sum();
    
    ss_between / ss_total
}

/// Compute interaction strength between two factors
fn compute_interaction_strength(
    name1: &str, data1: &FactorData,
    name2: &str, data2: &FactorData,
    residuals: &Array1<f64>,
    min_cell_count: usize,
) -> Option<InteractionCandidate> {
    let n = data1.len().min(data2.len()).min(residuals.len());
    if n < min_cell_count * 4 {  // Need reasonable sample for interaction
        return None;
    }
    
    // Bin continuous factors, use categories as-is
    let bins1 = discretize_factor(data1, 5);
    let bins2 = discretize_factor(data2, 5);
    
    // Create interaction cells
    let mut cell_residuals: HashMap<(usize, usize), Vec<f64>> = HashMap::new();
    for i in 0..n {
        if let (Some(&b1), Some(&b2)) = (bins1.get(i), bins2.get(i)) {
            cell_residuals.entry((b1, b2))
                .or_insert_with(Vec::new)
                .push(residuals[i]);
        }
    }
    
    // Filter cells with sufficient data
    let valid_cells: HashMap<(usize, usize), Vec<f64>> = cell_residuals.into_iter()
        .filter(|(_, resids)| resids.len() >= min_cell_count)
        .collect();
    
    if valid_cells.len() < 4 {
        return None;
    }
    
    // Compute SS_total (using only observations in valid cells)
    let all_residuals: Vec<f64> = valid_cells.values()
        .flat_map(|v| v.iter().cloned())
        .collect();
    let n_valid = all_residuals.len();
    
    if n_valid < min_cell_count * 4 {
        return None;
    }
    
    let overall_mean: f64 = all_residuals.iter().sum::<f64>() / n_valid as f64;
    let ss_total: f64 = all_residuals.iter()
        .map(|&r| (r - overall_mean).powi(2))
        .sum();
    
    if ss_total == 0.0 {
        return None;
    }
    
    // Compute SS_model (variance explained by interaction cells)
    let ss_model: f64 = valid_cells.iter()
        .map(|(_, resids)| {
            let cell_mean: f64 = resids.iter().sum::<f64>() / resids.len() as f64;
            resids.len() as f64 * (cell_mean - overall_mean).powi(2)
        })
        .sum();
    
    // Partial R² 
    let r_squared = ss_model / ss_total;
    
    // Compute p-value using F-test approximation
    let df_model = valid_cells.len() - 1;
    let df_resid = n_valid - valid_cells.len();
    
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
        n_cells: valid_cells.len(),
    })
}

/// Discretize a factor into bins
fn discretize_factor(factor: &FactorData, n_bins: usize) -> Vec<usize> {
    match factor {
        FactorData::Continuous(values) => {
            // Find quantile boundaries
            let mut sorted: Vec<f64> = values.iter()
                .filter(|&&v| !v.is_nan() && !v.is_infinite())
                .cloned()
                .collect();
            
            if sorted.is_empty() {
                return vec![0; values.len()];
            }
            
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            
            let boundaries: Vec<f64> = (1..n_bins)
                .map(|i| {
                    let p = i as f64 / n_bins as f64;
                    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
                    sorted[idx]
                })
                .collect();
            
            // Assign bins
            values.iter()
                .map(|&v| {
                    if v.is_nan() || v.is_infinite() {
                        n_bins  // Invalid values get their own bin
                    } else {
                        boundaries.iter()
                            .position(|&b| v < b)
                            .unwrap_or(n_bins - 1)
                    }
                })
                .collect()
        }
        FactorData::Categorical(values) => {
            // Map categories to indices
            let mut cat_to_idx: HashMap<&str, usize> = HashMap::new();
            let mut next_idx = 0;
            
            values.iter()
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
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }
    
    // Simple continued fraction approximation
    let bt = if x == 0.0 || x == 1.0 {
        0.0
    } else {
        (ln_gamma_approx(a + b) - ln_gamma_approx(a) - ln_gamma_approx(b)
            + a * x.ln() + b * (1.0 - x).ln()).exp()
    };
    
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
    if d.abs() < 1e-30 { d = 1e-30; }
    d = 1.0 / d;
    let mut h = d;
    
    for m in 1..=max_iter {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;
        
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 { d = 1e-30; }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 { c = 1e-30; }
        d = 1.0 / d;
        h *= d * c;
        
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 { d = 1e-30; }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 { c = 1e-30; }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        
        if (del - 1.0).abs() < eps { break; }
    }
    
    h
}

fn ln_gamma_approx(x: f64) -> f64 {
    // Stirling's approximation
    if x <= 0.0 { return f64::INFINITY; }
    
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
    use ndarray::array;

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
            (0..n).map(|i| ((i % 10) as f64 - 5.0) * 0.1 + ((i % 5) as f64 - 2.0) * 0.2).collect()
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
            "A".to_string(), "B".to_string(), "A".to_string(), 
            "C".to_string(), "B".to_string(),
        ];
        let data = FactorData::Categorical(values);
        let bins = discretize_factor(&data, 5);
        
        assert_eq!(bins.len(), 5);
        // Same categories should have same bin
        assert_eq!(bins[0], bins[2]); // Both "A"
        assert_eq!(bins[1], bins[4]); // Both "B"
    }
}
