// =============================================================================
// Regularization Module
// =============================================================================
//
// This module provides regularization (penalty) support for GLMs.
// Regularization helps with:
//   - High-dimensional data (more predictors than observations)
//   - Multicollinearity (correlated predictors)
//   - Variable selection (Lasso/Elastic Net set coefficients to zero)
//   - Overfitting prevention
//
// PENALTY TYPES
// -------------
// Ridge (L2):  penalty = λ × Σβ²
//   - Shrinks coefficients toward zero but never exactly zero
//   - Good for multicollinearity
//   - Closed-form solution in IRLS
//
// Lasso (L1):  penalty = λ × Σ|β|
//   - Can shrink coefficients exactly to zero → variable selection
//   - Requires coordinate descent (not differentiable at 0)
//
// Elastic Net: penalty = λ × [α×Σ|β| + (1-α)×Σβ²/2]
//   - Combines L1 and L2 penalties
//   - α=1 is pure Lasso, α=0 is pure Ridge
//   - Groups correlated variables (advantage over Lasso)
//
// INTERCEPT HANDLING
// ------------------
// The intercept (first column if present) is NEVER penalized.
// This is standard practice - we don't want to shrink the baseline.
//
// STANDARDIZATION
// ---------------
// For regularization to work fairly across predictors:
//   - Predictors should be standardized (mean=0, std=1)
//   - Or use penalty weights inversely proportional to scale
//
// =============================================================================

use ndarray::Array2;
use std::ops::Range;

/// Penalty type for regularized GLMs.
///
/// Controls which type of regularization is applied to the coefficients.
#[derive(Debug, Clone)]
pub enum Penalty {
    /// No regularization (standard GLM)
    None,

    /// Ridge (L2) penalty: λ × Σβ²
    ///
    /// - Shrinks all coefficients toward zero
    /// - Never produces exact zeros
    /// - Good for multicollinearity
    /// - Can be solved with modified IRLS (add λI to X'WX)
    Ridge(f64),

    /// Lasso (L1) penalty: λ × Σ|β|
    ///
    /// - Can shrink coefficients exactly to zero
    /// - Automatic variable selection
    /// - Requires coordinate descent solver
    Lasso(f64),

    /// Elastic Net: λ × [α×Σ|β| + (1-α)×Σβ²/2]
    ///
    /// - First f64 is λ (overall regularization strength)
    /// - Second f64 is α (L1 ratio: 1.0 = pure Lasso, 0.0 = pure Ridge)
    /// - Combines benefits of Ridge and Lasso
    /// - Groups correlated predictors together
    ElasticNet {
        /// Overall regularization strength (λ)
        lambda: f64,
        /// L1 ratio (α): 1.0 = pure Lasso, 0.0 = pure Ridge
        l1_ratio: f64,
    },

    /// Smooth penalty for penalized splines (P-splines, GAMs).
    ///
    /// Uses structured penalty matrices S = D'D where D is a difference matrix.
    /// This penalizes wiggliness of the smooth function rather than coefficient magnitude.
    ///
    /// The penalty is: Σ_j λ_j × β_j' S_j β_j
    ///
    /// where each smooth term j has its own penalty matrix S_j and smoothing
    /// parameter λ_j.
    ///
    /// Can be combined with scalar Ridge/Lasso for parametric terms.
    Smooth(SmoothPenalty),
}

/// Penalty configuration for smooth terms in GAMs.
///
/// Each smooth term (e.g., `s(age)`) has:
/// - A penalty matrix S encoding the smoothness constraint
/// - A smoothing parameter λ (selected via GCV or fixed)
/// - Column indices indicating which coefficients it applies to
#[derive(Debug, Clone)]
pub struct SmoothPenalty {
    /// Penalty matrices, one per smooth term (each is k_j × k_j)
    pub penalty_matrices: Vec<Array2<f64>>,
    /// Smoothing parameters, one per smooth term
    pub lambdas: Vec<f64>,
    /// Column indices for each smooth term in the design matrix
    pub term_indices: Vec<Range<usize>>,
    /// Optional scalar L2 penalty for parametric (non-smooth) terms
    pub parametric_l2: Option<f64>,
}

impl SmoothPenalty {
    /// Create a new smooth penalty with no terms.
    pub fn new() -> Self {
        Self {
            penalty_matrices: Vec::new(),
            lambdas: Vec::new(),
            term_indices: Vec::new(),
            parametric_l2: None,
        }
    }

    /// Add a smooth term with its penalty matrix and column indices.
    pub fn add_term(&mut self, penalty: Array2<f64>, lambda: f64, indices: Range<usize>) {
        self.penalty_matrices.push(penalty);
        self.lambdas.push(lambda);
        self.term_indices.push(indices);
    }

    /// Set the scalar L2 penalty for parametric terms.
    pub fn with_parametric_l2(mut self, lambda: f64) -> Self {
        self.parametric_l2 = Some(lambda);
        self
    }

    /// Build the combined penalty matrix for the full design matrix.
    ///
    /// Returns a p × p matrix where p is the total number of coefficients.
    /// The matrix is block-diagonal for smooth terms, with optional
    /// diagonal entries for parametric L2 regularization.
    pub fn build_penalty_matrix(&self, total_cols: usize, parametric_cols: usize) -> Array2<f64> {
        let mut combined: Array2<f64> = Array2::zeros((total_cols, total_cols));

        // Add smooth term penalties (block-diagonal structure)
        for (idx, penalty) in self.penalty_matrices.iter().enumerate() {
            let range = &self.term_indices[idx];
            let lambda = self.lambdas[idx];
            for i in 0..penalty.nrows() {
                for j in 0..penalty.ncols() {
                    combined[[range.start + i, range.start + j]] = lambda * penalty[[i, j]];
                }
            }
        }

        // Add parametric L2 penalty (diagonal, skip intercept at col 0)
        if let Some(l2) = self.parametric_l2 {
            for i in 1..parametric_cols {  // Skip intercept
                combined[[i, i]] += l2;
            }
        }

        combined
    }

    /// Get total number of smooth terms.
    pub fn n_terms(&self) -> usize {
        self.penalty_matrices.len()
    }

    /// Check if there are any smooth terms.
    pub fn is_empty(&self) -> bool {
        self.penalty_matrices.is_empty()
    }
}

impl Default for SmoothPenalty {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for SmoothPenalty {
    fn eq(&self, other: &Self) -> bool {
        // Simple comparison - check lengths and lambdas
        self.lambdas == other.lambdas && 
        self.term_indices == other.term_indices &&
        self.parametric_l2 == other.parametric_l2
    }
}

impl PartialEq for Penalty {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Penalty::None, Penalty::None) => true,
            (Penalty::Ridge(a), Penalty::Ridge(b)) => a == b,
            (Penalty::Lasso(a), Penalty::Lasso(b)) => a == b,
            (Penalty::ElasticNet { lambda: l1, l1_ratio: r1 }, 
             Penalty::ElasticNet { lambda: l2, l1_ratio: r2 }) => l1 == l2 && r1 == r2,
            (Penalty::Smooth(a), Penalty::Smooth(b)) => a == b,
            _ => false,
        }
    }
}

impl Penalty {
    /// Create a Ridge penalty with the given regularization strength.
    pub fn ridge(lambda: f64) -> Self {
        Penalty::Ridge(lambda)
    }

    /// Create a Lasso penalty with the given regularization strength.
    pub fn lasso(lambda: f64) -> Self {
        Penalty::Lasso(lambda)
    }

    /// Create an Elastic Net penalty.
    ///
    /// # Arguments
    /// * `lambda` - Overall regularization strength
    /// * `l1_ratio` - Mix between L1 and L2 (1.0 = Lasso, 0.0 = Ridge)
    pub fn elastic_net(lambda: f64, l1_ratio: f64) -> Self {
        Penalty::ElasticNet { lambda, l1_ratio }
    }

    /// Returns true if this is a no-penalty (unregularized) model.
    pub fn is_none(&self) -> bool {
        matches!(self, Penalty::None)
    }

    /// Returns true if this penalty requires coordinate descent (has L1 component).
    pub fn requires_coordinate_descent(&self) -> bool {
        match self {
            Penalty::Lasso(_) => true,
            Penalty::ElasticNet { l1_ratio, .. } => *l1_ratio > 0.0,
            _ => false,
        }
    }

    /// Returns true if this penalty can use modified IRLS (pure L2).
    pub fn can_use_irls(&self) -> bool {
        match self {
            Penalty::None | Penalty::Ridge(_) => true,
            Penalty::Smooth(_) => true,  // Smooth penalty uses IRLS with penalty matrix
            Penalty::ElasticNet { l1_ratio, .. } => *l1_ratio == 0.0,
            Penalty::Lasso(_) => false,
        }
    }

    /// Returns true if this is a smooth (GAM) penalty.
    pub fn is_smooth(&self) -> bool {
        matches!(self, Penalty::Smooth(_))
    }

    /// Get the smooth penalty configuration, if this is a smooth penalty.
    pub fn as_smooth(&self) -> Option<&SmoothPenalty> {
        match self {
            Penalty::Smooth(sp) => Some(sp),
            _ => None,
        }
    }

    /// Get mutable reference to smooth penalty configuration.
    pub fn as_smooth_mut(&mut self) -> Option<&mut SmoothPenalty> {
        match self {
            Penalty::Smooth(sp) => Some(sp),
            _ => None,
        }
    }

    /// Create a smooth penalty from a SmoothPenalty configuration.
    pub fn smooth(smooth_penalty: SmoothPenalty) -> Self {
        Penalty::Smooth(smooth_penalty)
    }

    /// Get the L2 (Ridge) component of the penalty.
    ///
    /// Returns 0.0 for no penalty or pure Lasso.
    /// For Smooth penalties, returns the parametric L2 component if set.
    pub fn l2_penalty(&self) -> f64 {
        match self {
            Penalty::None => 0.0,
            Penalty::Ridge(lambda) => *lambda,
            Penalty::Lasso(_) => 0.0,
            Penalty::ElasticNet { lambda, l1_ratio } => *lambda * (1.0 - l1_ratio),
            Penalty::Smooth(sp) => sp.parametric_l2.unwrap_or(0.0),
        }
    }

    /// Get the L1 (Lasso) component of the penalty.
    ///
    /// Returns 0.0 for no penalty or pure Ridge.
    pub fn l1_penalty(&self) -> f64 {
        match self {
            Penalty::None => 0.0,
            Penalty::Ridge(_) => 0.0,
            Penalty::Lasso(lambda) => *lambda,
            Penalty::ElasticNet { lambda, l1_ratio } => *lambda * l1_ratio,
            Penalty::Smooth(_) => 0.0,  // Smooth penalties don't have L1 component
        }
    }

    /// Get the overall regularization strength (lambda).
    /// For smooth penalties, returns 0.0 (use as_smooth() to get per-term lambdas).
    pub fn lambda(&self) -> f64 {
        match self {
            Penalty::None => 0.0,
            Penalty::Ridge(lambda) | Penalty::Lasso(lambda) => *lambda,
            Penalty::ElasticNet { lambda, .. } => *lambda,
            Penalty::Smooth(_) => 0.0,  // Per-term lambdas accessed via as_smooth()
        }
    }
}

impl Default for Penalty {
    fn default() -> Self {
        Penalty::None
    }
}

/// Configuration for regularized GLM fitting.
#[derive(Debug, Clone)]
pub struct RegularizationConfig {
    /// The penalty type and strength
    pub penalty: Penalty,

    /// Whether to standardize predictors before fitting.
    ///
    /// If true:
    /// - Predictors are centered (mean=0) and scaled (std=1)
    /// - Coefficients are transformed back to original scale
    /// - Intercept is computed on original scale
    ///
    /// Default: true (recommended for regularization)
    pub standardize: bool,

    /// Whether the first column of X is an intercept.
    ///
    /// If true, the first coefficient is never penalized.
    /// Default: true
    pub fit_intercept: bool,

    /// Per-predictor penalty weights.
    ///
    /// If provided, the penalty for coefficient j is multiplied by weight[j].
    /// Use this to:
    /// - Exclude certain predictors from penalization (weight=0)
    /// - Apply adaptive Lasso (weight = 1/|β_OLS|)
    ///
    /// Length must equal number of predictors (excluding intercept if fit_intercept=true).
    /// Default: None (all predictors weighted equally with weight=1)
    pub penalty_weights: Option<Vec<f64>>,

    /// Maximum number of coordinate descent iterations (for L1 penalties).
    ///
    /// Default: 1000
    pub max_cd_iterations: usize,

    /// Convergence tolerance for coordinate descent.
    ///
    /// Default: 1e-7
    pub cd_tolerance: f64,
}

impl Default for RegularizationConfig {
    fn default() -> Self {
        Self {
            penalty: Penalty::None,
            standardize: true,
            fit_intercept: true,
            penalty_weights: None,
            max_cd_iterations: 1000,
            cd_tolerance: 1e-7,
        }
    }
}

impl RegularizationConfig {
    /// Create a new config with no regularization.
    pub fn none() -> Self {
        Self::default()
    }

    /// Create a Ridge (L2) regularization config.
    pub fn ridge(lambda: f64) -> Self {
        Self {
            penalty: Penalty::ridge(lambda),
            ..Default::default()
        }
    }

    /// Create a Lasso (L1) regularization config.
    pub fn lasso(lambda: f64) -> Self {
        Self {
            penalty: Penalty::lasso(lambda),
            ..Default::default()
        }
    }

    /// Create an Elastic Net regularization config.
    pub fn elastic_net(lambda: f64, l1_ratio: f64) -> Self {
        Self {
            penalty: Penalty::elastic_net(lambda, l1_ratio),
            ..Default::default()
        }
    }

    /// Set whether to standardize predictors.
    pub fn with_standardize(mut self, standardize: bool) -> Self {
        self.standardize = standardize;
        self
    }

    /// Set whether the first column is an intercept.
    pub fn with_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set custom penalty weights for each predictor.
    pub fn with_penalty_weights(mut self, weights: Vec<f64>) -> Self {
        self.penalty_weights = Some(weights);
        self
    }
}

// =============================================================================
// Soft Thresholding (for Lasso/Elastic Net)
// =============================================================================

/// Soft thresholding operator for Lasso.
///
/// S(z, γ) = sign(z) × max(0, |z| - γ)
///
/// This is the proximal operator for the L1 penalty.
/// It shrinks the coefficient toward zero, setting it exactly to zero
/// if |z| ≤ γ.
#[inline]
pub fn soft_threshold(z: f64, gamma: f64) -> f64 {
    if z > gamma {
        z - gamma
    } else if z < -gamma {
        z + gamma
    } else {
        0.0
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_penalty_creation() {
        let ridge = Penalty::ridge(0.1);
        assert_eq!(ridge.l2_penalty(), 0.1);
        assert_eq!(ridge.l1_penalty(), 0.0);
        assert!(ridge.can_use_irls());
        assert!(!ridge.requires_coordinate_descent());

        let lasso = Penalty::lasso(0.5);
        assert_eq!(lasso.l1_penalty(), 0.5);
        assert_eq!(lasso.l2_penalty(), 0.0);
        assert!(!lasso.can_use_irls());
        assert!(lasso.requires_coordinate_descent());

        let elastic = Penalty::elastic_net(1.0, 0.5);
        assert_eq!(elastic.l1_penalty(), 0.5);
        assert_eq!(elastic.l2_penalty(), 0.5);
        assert!(!elastic.can_use_irls());
        assert!(elastic.requires_coordinate_descent());
    }

    #[test]
    fn test_elastic_net_extremes() {
        // l1_ratio = 1.0 should be pure Lasso
        let pure_lasso = Penalty::elastic_net(1.0, 1.0);
        assert_eq!(pure_lasso.l1_penalty(), 1.0);
        assert_eq!(pure_lasso.l2_penalty(), 0.0);

        // l1_ratio = 0.0 should be pure Ridge
        let pure_ridge = Penalty::elastic_net(1.0, 0.0);
        assert_eq!(pure_ridge.l1_penalty(), 0.0);
        assert_eq!(pure_ridge.l2_penalty(), 1.0);
        assert!(pure_ridge.can_use_irls()); // Pure L2 can use IRLS
    }

    #[test]
    fn test_soft_threshold() {
        // Above threshold
        assert!((soft_threshold(5.0, 2.0) - 3.0).abs() < 1e-10);
        // Below negative threshold
        assert!((soft_threshold(-5.0, 2.0) - (-3.0)).abs() < 1e-10);
        // Within threshold -> zero
        assert!((soft_threshold(1.5, 2.0) - 0.0).abs() < 1e-10);
        assert!((soft_threshold(-1.5, 2.0) - 0.0).abs() < 1e-10);
        // At boundary
        assert!((soft_threshold(2.0, 2.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_config_builders() {
        let config = RegularizationConfig::ridge(0.1)
            .with_standardize(false)
            .with_intercept(true);
        
        assert_eq!(config.penalty.l2_penalty(), 0.1);
        assert!(!config.standardize);
        assert!(config.fit_intercept);
    }

    #[test]
    fn test_penalty_default() {
        let penalty = Penalty::default();
        assert!(penalty.is_none());
        assert_eq!(penalty.lambda(), 0.0);
    }
}
