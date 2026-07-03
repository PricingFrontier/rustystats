use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3};

use crate::error::{Result, RustyStatsError};
use crate::regularization::{soft_threshold, Standardization};
use crate::solvers::{
    build_sparse_row_cache_if_beneficial, compute_xtwx_with_sparse_cache, SparseRowCache,
};

const DEFAULT_HESSIAN_MEMORY_LIMIT_BYTES: usize = 256 * 1024 * 1024;
const DEFAULT_MAX_DENSE_PARAMETERS: usize = 5000;
const DEFAULT_MAX_ITERATIONS: usize = 100;
const DEFAULT_TOLERANCE: f64 = 1e-8;
const MAX_HALF_STEPS: usize = 30;
const MIN_LOG_PROBABILITY: f64 = 1e-300;
const MAX_COEFFICIENT_NORM: f64 = 1e6;
const PROXIMAL_NEWTON_MAX_CD_ITERATIONS: usize = 1000;
const PROXIMAL_NEWTON_CD_TOLERANCE: f64 = 1e-8;
const PROXIMAL_NEWTON_ZERO_TOLERANCE: f64 = 1e-10;
const PROXIMAL_NEWTON_MIN_DIAGONAL: f64 = 1e-12;

#[derive(Debug, Clone)]
pub struct MultinomialConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub alpha: f64,
    pub l1_ratio: f64,
    pub fit_intercept: bool,
    pub standardize: bool,
    pub skip_covariance: bool,
    pub hessian_memory_limit_bytes: usize,
    pub max_dense_parameters: usize,
    pub verbose: bool,
    pub initial_theta: Option<Array1<f64>>,
    pub smooth_penalties: Vec<MultinomialSmoothPenalty>,
    pub nonneg_indices: Vec<usize>,
    pub nonpos_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct MultinomialSmoothPenalty {
    pub col_start: usize,
    pub col_end: usize,
    pub penalty: Array2<f64>,
    pub lambda: f64,
}

impl Default for MultinomialConfig {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_MAX_ITERATIONS,
            tolerance: DEFAULT_TOLERANCE,
            alpha: 0.0,
            l1_ratio: 0.0,
            fit_intercept: true,
            standardize: true,
            skip_covariance: false,
            hessian_memory_limit_bytes: DEFAULT_HESSIAN_MEMORY_LIMIT_BYTES,
            max_dense_parameters: DEFAULT_MAX_DENSE_PARAMETERS,
            verbose: false,
            initial_theta: None,
            smooth_penalties: Vec::new(),
            nonneg_indices: Vec::new(),
            nonpos_indices: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MultinomialResult {
    pub coefficients: Array2<f64>,
    pub alternative_generic_coefficients: Array1<f64>,
    pub alternative_specific_coefficients: Array2<f64>,
    pub fitted_probabilities: Array2<f64>,
    pub linear_predictor: Array2<f64>,
    pub log_likelihood: f64,
    pub deviance: f64,
    pub null_deviance: f64,
    pub iterations: usize,
    pub converged: bool,
    pub covariance_unscaled: Option<Array2<f64>>,
    pub prior_weights: Array1<f64>,
    pub y_codes: Array1<usize>,
    pub reference_index: usize,
    pub warnings: Vec<String>,
    pub solver_status: String,
    pub smooth_edfs: Vec<f64>,
    pub total_edf: Option<f64>,
}

#[derive(Debug)]
struct PreparedInputs {
    y_codes: Array1<usize>,
    availability: Array2<bool>,
    offset: Array2<f64>,
    weights: Array1<f64>,
    alternative_generic: Array3<f64>,
    alternative_specific: Array3<f64>,
    class_to_block: Vec<Option<usize>>,
    non_reference_classes: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct AlternativeSpecificStandardization {
    pub center: Array2<f64>,
    pub scale: Array2<f64>,
}

impl AlternativeSpecificStandardization {
    pub fn new(center: Array2<f64>, scale: Array2<f64>) -> Result<Self> {
        let std = Self { center, scale };
        std.validate(std.center.dim())?;
        Ok(std)
    }

    pub fn validate(&self, expected: (usize, usize)) -> Result<()> {
        if self.center.dim() != expected {
            return Err(RustyStatsError::InvalidValue(format!(
                "class-specific alternative standardization center has shape {:?}, expected {:?}",
                self.center.dim(),
                expected
            )));
        }
        if self.scale.dim() != expected {
            return Err(RustyStatsError::InvalidValue(format!(
                "class-specific alternative standardization scale has shape {:?}, expected {:?}",
                self.scale.dim(),
                expected
            )));
        }
        if self.center.iter().any(|v| !v.is_finite()) {
            return Err(RustyStatsError::InvalidValue(
                "class-specific alternative standardization center values must be finite"
                    .to_string(),
            ));
        }
        if self.scale.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(RustyStatsError::InvalidValue(
                "class-specific alternative standardization scale values must be finite and > 0"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

struct MultinomialTransform {
    rows: Vec<Vec<(usize, f64)>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MultinomialParameterLayout {
    n_shared: usize,
    n_non_reference: usize,
    n_alt_generic: usize,
    n_alt_specific: usize,
    fit_intercept: bool,
}

impl MultinomialParameterLayout {
    fn new(
        n_shared: usize,
        n_classes: usize,
        n_alt_generic: usize,
        n_alt_specific: usize,
        fit_intercept: bool,
    ) -> Result<Self> {
        if n_classes < 2 {
            return Err(RustyStatsError::InvalidValue(
                "multinomial layout requires at least two classes".to_string(),
            ));
        }
        Ok(Self {
            n_shared,
            n_non_reference: n_classes - 1,
            n_alt_generic,
            n_alt_specific,
            fit_intercept,
        })
    }

    fn len(&self) -> Result<usize> {
        let shared = self.shared_len()?;
        let specific = self
            .n_alt_specific
            .checked_mul(self.n_non_reference)
            .ok_or_else(|| {
                RustyStatsError::InvalidValue(
                    "multinomial class-specific alternative parameter count overflowed usize"
                        .into(),
                )
            })?;
        shared
            .checked_add(self.n_alt_generic)
            .and_then(|value| value.checked_add(specific))
            .ok_or_else(|| {
                RustyStatsError::InvalidValue(
                    "multinomial total parameter count overflowed usize".into(),
                )
            })
    }

    fn shared_len(&self) -> Result<usize> {
        self.n_shared
            .checked_mul(self.n_non_reference)
            .ok_or_else(|| {
                RustyStatsError::InvalidValue(
                    "multinomial shared parameter count overflowed usize".into(),
                )
            })
    }

    fn shared(&self, block_idx: usize, feature_idx: usize) -> Result<usize> {
        if block_idx >= self.n_non_reference {
            return Err(RustyStatsError::InvalidValue(format!(
                "non-reference block {} is out of range for {} blocks",
                block_idx, self.n_non_reference
            )));
        }
        if feature_idx >= self.n_shared {
            return Err(RustyStatsError::InvalidValue(format!(
                "shared feature {} is out of range for {} shared features",
                feature_idx, self.n_shared
            )));
        }
        block_idx
            .checked_mul(self.n_shared)
            .and_then(|value| value.checked_add(feature_idx))
            .ok_or_else(|| {
                RustyStatsError::InvalidValue(
                    "multinomial shared parameter index overflowed usize".into(),
                )
            })
    }

    fn alternative_generic_start(&self) -> Result<usize> {
        self.shared_len()
    }

    fn alternative_generic(&self, term_idx: usize) -> Result<usize> {
        if term_idx >= self.n_alt_generic {
            return Err(RustyStatsError::InvalidValue(format!(
                "generic alternative term {} is out of range for {} terms",
                term_idx, self.n_alt_generic
            )));
        }
        self.alternative_generic_start()?
            .checked_add(term_idx)
            .ok_or_else(|| {
                RustyStatsError::InvalidValue(
                    "multinomial generic alternative parameter index overflowed usize".into(),
                )
            })
    }

    fn alternative_specific_start(&self) -> Result<usize> {
        self.alternative_generic_start()?
            .checked_add(self.n_alt_generic)
            .ok_or_else(|| {
                RustyStatsError::InvalidValue(
                    "multinomial class-specific alternative start overflowed usize".into(),
                )
            })
    }

    fn alternative_specific(&self, block_idx: usize, term_idx: usize) -> Result<usize> {
        if block_idx >= self.n_non_reference {
            return Err(RustyStatsError::InvalidValue(format!(
                "non-reference block {} is out of range for {} blocks",
                block_idx, self.n_non_reference
            )));
        }
        if term_idx >= self.n_alt_specific {
            return Err(RustyStatsError::InvalidValue(format!(
                "class-specific alternative term {} is out of range for {} terms",
                term_idx, self.n_alt_specific
            )));
        }
        self.alternative_specific_start()?
            .checked_add(block_idx.checked_mul(self.n_alt_specific).ok_or_else(|| {
                RustyStatsError::InvalidValue(
                    "multinomial class-specific alternative block offset overflowed usize".into(),
                )
            })?)
            .and_then(|value| value.checked_add(term_idx))
            .ok_or_else(|| {
                RustyStatsError::InvalidValue(
                    "multinomial class-specific alternative parameter index overflowed usize"
                        .into(),
                )
            })
    }

    fn is_shared_intercept(&self, idx: usize) -> bool {
        self.fit_intercept
            && idx < self.shared_len().unwrap_or(0)
            && idx.is_multiple_of(self.n_shared)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DenseMultinomialOperation {
    Fit,
    #[allow(dead_code)]
    // Constructed by the dense-guard test to exercise the covariance-workspace peak.
    Covariance,
}

impl DenseMultinomialOperation {
    fn label(self) -> &'static str {
        match self {
            Self::Fit => "fit",
            Self::Covariance => "covariance",
        }
    }

    fn peak_factor(self) -> usize {
        match self {
            Self::Fit => 1,
            Self::Covariance => 3,
        }
    }
}

#[derive(Debug, Clone)]
struct QuadraticPenaltyBlock {
    indices: Vec<usize>,
    matrix: Array2<f64>,
    weight: f64,
}

impl QuadraticPenaltyBlock {
    fn validate(&self, q: usize) -> Result<()> {
        if self.matrix.nrows() != self.indices.len() || self.matrix.ncols() != self.indices.len() {
            return Err(RustyStatsError::InvalidValue(format!(
                "quadratic penalty matrix has shape {:?}, expected ({}, {})",
                self.matrix.dim(),
                self.indices.len(),
                self.indices.len()
            )));
        }
        if !self.weight.is_finite() || self.weight < 0.0 {
            return Err(RustyStatsError::InvalidValue(
                "quadratic penalty weight must be finite and non-negative".to_string(),
            ));
        }
        for &idx in &self.indices {
            if idx >= q {
                return Err(RustyStatsError::InvalidValue(format!(
                    "quadratic penalty index {} is out of range for {} parameters",
                    idx, q
                )));
            }
        }
        if self.matrix.iter().any(|value| !value.is_finite()) {
            return Err(RustyStatsError::InvalidValue(
                "quadratic penalty matrix values must be finite".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct L1Penalty {
    alpha: f64,
    mask: Vec<bool>,
}

impl L1Penalty {
    fn inert(q: usize) -> Self {
        Self {
            alpha: 0.0,
            mask: vec![false; q],
        }
    }

    fn validate(&self, q: usize) -> Result<()> {
        if self.mask.len() != q {
            return Err(RustyStatsError::dim_mismatch(
                q,
                self.mask.len(),
                "L1 penalty mask vs parameter count",
            ));
        }
        if !self.alpha.is_finite() || self.alpha < 0.0 {
            return Err(RustyStatsError::InvalidValue(
                "L1 penalty alpha must be finite and non-negative".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
struct BoundConstraints {
    nonneg_indices: Vec<usize>,
    nonpos_indices: Vec<usize>,
}

impl BoundConstraints {
    fn is_empty(&self) -> bool {
        self.nonneg_indices.is_empty() && self.nonpos_indices.is_empty()
    }

    fn signs(&self, q: usize) -> Result<Vec<i8>> {
        let mut signs = vec![0_i8; q];
        for &idx in &self.nonneg_indices {
            if idx >= q {
                return Err(RustyStatsError::InvalidValue(format!(
                    "bound constraint index {} is out of range for {} parameters",
                    idx, q
                )));
            }
            if signs[idx] != 0 {
                return Err(RustyStatsError::InvalidValue(format!(
                    "bound constraint index {} appears more than once or has conflicting signs",
                    idx
                )));
            }
            signs[idx] = 1;
        }
        for &idx in &self.nonpos_indices {
            if idx >= q {
                return Err(RustyStatsError::InvalidValue(format!(
                    "bound constraint index {} is out of range for {} parameters",
                    idx, q
                )));
            }
            if signs[idx] != 0 {
                return Err(RustyStatsError::InvalidValue(format!(
                    "bound constraint index {} appears more than once or has conflicting signs",
                    idx
                )));
            }
            signs[idx] = -1;
        }
        Ok(signs)
    }

    fn validate(&self, q: usize) -> Result<()> {
        self.signs(q).map(|_| ())
    }
}

#[derive(Debug, Clone)]
struct OptimizationExtensions {
    quadratic_penalties: Vec<QuadraticPenaltyBlock>,
    l1_penalty: L1Penalty,
    bound_constraints: BoundConstraints,
}

impl OptimizationExtensions {
    fn inert(q: usize) -> Self {
        Self {
            quadratic_penalties: Vec::new(),
            l1_penalty: L1Penalty::inert(q),
            bound_constraints: BoundConstraints::default(),
        }
    }

    fn validate(&self, q: usize) -> Result<()> {
        for penalty in &self.quadratic_penalties {
            penalty.validate(q)?;
        }
        self.l1_penalty.validate(q)?;
        self.bound_constraints.validate(q)
    }
}

fn optimization_extensions_from_config(
    layout: &MultinomialParameterLayout,
    l1_alpha: f64,
    smooth_penalties: &[MultinomialSmoothPenalty],
    nonneg_indices: &[usize],
    nonpos_indices: &[usize],
) -> Result<OptimizationExtensions> {
    let q = layout.len()?;
    let mut extensions = OptimizationExtensions::inert(q);
    for smooth in smooth_penalties {
        extensions
            .quadratic_penalties
            .extend(expand_smooth_penalty(layout, smooth)?);
    }
    if l1_alpha > 0.0 {
        extensions.l1_penalty = L1Penalty {
            alpha: l1_alpha,
            mask: default_l1_penalty_mask(layout)?,
        };
    }
    extensions.bound_constraints = BoundConstraints {
        nonneg_indices: nonneg_indices.to_vec(),
        nonpos_indices: nonpos_indices.to_vec(),
    };
    Ok(extensions)
}

fn expand_smooth_penalty(
    layout: &MultinomialParameterLayout,
    smooth: &MultinomialSmoothPenalty,
) -> Result<Vec<QuadraticPenaltyBlock>> {
    if smooth.col_start >= smooth.col_end || smooth.col_end > layout.n_shared {
        return Err(RustyStatsError::InvalidValue(format!(
            "smooth penalty column range {}..{} is invalid for {} shared columns",
            smooth.col_start, smooth.col_end, layout.n_shared
        )));
    }
    let width = smooth.col_end - smooth.col_start;
    if smooth.penalty.nrows() != width || smooth.penalty.ncols() != width {
        return Err(RustyStatsError::InvalidValue(format!(
            "smooth penalty matrix has shape {:?}, expected ({}, {}) for columns {}..{}",
            smooth.penalty.dim(),
            width,
            width,
            smooth.col_start,
            smooth.col_end
        )));
    }
    if !smooth.lambda.is_finite() || smooth.lambda < 0.0 {
        return Err(RustyStatsError::InvalidValue(
            "smooth penalty lambda must be finite and non-negative".to_string(),
        ));
    }
    if smooth.penalty.iter().any(|value| !value.is_finite()) {
        return Err(RustyStatsError::InvalidValue(
            "smooth penalty matrix values must be finite".to_string(),
        ));
    }
    for row in 0..width {
        for col in 0..row {
            if (smooth.penalty[[row, col]] - smooth.penalty[[col, row]]).abs() > 1e-10 {
                return Err(RustyStatsError::InvalidValue(
                    "smooth penalty matrix must be symmetric".to_string(),
                ));
            }
        }
    }

    let mut blocks = Vec::with_capacity(layout.n_non_reference);
    for block_idx in 0..layout.n_non_reference {
        let mut indices = Vec::with_capacity(width);
        for feature_idx in smooth.col_start..smooth.col_end {
            indices.push(layout.shared(block_idx, feature_idx)?);
        }
        blocks.push(QuadraticPenaltyBlock {
            indices,
            matrix: smooth.penalty.clone(),
            weight: smooth.lambda,
        });
    }
    Ok(blocks)
}

fn default_l1_penalty_mask(layout: &MultinomialParameterLayout) -> Result<Vec<bool>> {
    let q = layout.len()?;
    let mut mask = vec![true; q];
    for idx in 0..layout.shared_len()? {
        if layout.is_shared_intercept(idx) {
            mask[idx] = false;
        }
    }
    Ok(mask)
}

#[derive(Debug)]
struct Evaluation {
    objective: f64,
    gradient: Array1<f64>,
    hessian: Array2<f64>,
}

#[derive(Debug)]
struct NewtonProposal {
    target: Array1<f64>,
}

/// Fit a baseline-category multinomial logit model.
pub fn fit_multinomial(
    y_codes: &Array1<usize>,
    x: ArrayView2<'_, f64>,
    n_classes: usize,
    reference_index: usize,
    config: &MultinomialConfig,
    availability: Option<&Array2<bool>>,
    offset: Option<&Array2<f64>>,
    weights: Option<&Array1<f64>>,
    standardization: Option<&Standardization>,
) -> Result<MultinomialResult> {
    fit_multinomial_with_alternatives(
        y_codes,
        x,
        n_classes,
        reference_index,
        config,
        availability,
        offset,
        weights,
        standardization,
        None,
        None,
        None,
        None,
    )
}

/// Fit a baseline-category multinomial logit model with wide-format
/// alternative-specific covariates.
#[allow(clippy::too_many_arguments)]
pub fn fit_multinomial_with_alternatives(
    y_codes: &Array1<usize>,
    x: ArrayView2<'_, f64>,
    n_classes: usize,
    reference_index: usize,
    config: &MultinomialConfig,
    availability: Option<&Array2<bool>>,
    offset: Option<&Array2<f64>>,
    weights: Option<&Array1<f64>>,
    standardization: Option<&Standardization>,
    alternative_generic: Option<ArrayView3<'_, f64>>,
    alternative_specific: Option<ArrayView3<'_, f64>>,
    alternative_generic_standardization: Option<&Standardization>,
    alternative_specific_standardization: Option<&AlternativeSpecificStandardization>,
) -> Result<MultinomialResult> {
    fit_multinomial_internal(
        y_codes,
        x,
        n_classes,
        reference_index,
        config,
        availability,
        offset,
        weights,
        standardization,
        alternative_generic,
        alternative_specific,
        alternative_generic_standardization,
        alternative_specific_standardization,
        true,
    )
}

fn should_standardize_shared_inputs(
    config: &MultinomialConfig,
    standardization: Option<&Standardization>,
) -> bool {
    config.alpha > 0.0 && config.standardize && standardization.is_some()
}

fn parameter_l2_distance(candidate: &Array1<f64>, current: &Array1<f64>) -> f64 {
    candidate
        .iter()
        .zip(current.iter())
        .map(|(candidate_value, current_value)| {
            let delta = candidate_value - current_value;
            delta * delta
        })
        .sum::<f64>()
        .sqrt()
}

fn interpolate_parameters(
    current: &Array1<f64>,
    target: &Array1<f64>,
    step_fraction: f64,
) -> Array1<f64> {
    current
        .iter()
        .zip(target.iter())
        .map(|(current_value, target_value)| {
            current_value + step_fraction * (target_value - current_value)
        })
        .collect::<Array1<f64>>()
}

fn line_search_accepts(candidate_objective: f64, current_objective: f64, tolerance: f64) -> bool {
    candidate_objective.is_finite() && candidate_objective <= current_objective + tolerance * 1e-3
}

fn next_half_step_fraction(step_fraction: f64) -> f64 {
    step_fraction * 0.5
}

fn next_iteration_count(iter: usize) -> usize {
    iter + 1
}

fn coefficient_l2_norm(theta: &Array1<f64>) -> f64 {
    theta.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn coefficient_norm_is_extreme(coefficient_norm: f64) -> bool {
    coefficient_norm > MAX_COEFFICIENT_NORM
}

fn relative_objective_improvement(previous_objective: f64, new_objective: f64) -> f64 {
    (previous_objective - new_objective).abs() / (1.0 + previous_objective.abs())
}

fn scaled_newton_step_norm(step_fraction: f64, step_norm: f64, coefficient_norm: f64) -> f64 {
    step_fraction * step_norm / (1.0 + coefficient_norm)
}

fn convergence_recheck_needed(
    relative_improvement: f64,
    scaled_step_norm: f64,
    tolerance: f64,
) -> bool {
    relative_improvement <= tolerance || scaled_step_norm <= tolerance
}

fn kkt_converged(optimality: f64, tolerance: f64) -> bool {
    optimality <= tolerance
}

fn should_emit_max_iteration_warning(converged: bool, solver_status: &str) -> bool {
    !converged && solver_status == "max_iterations"
}

fn multinomial_deviance(log_likelihood: f64) -> f64 {
    -2.0 * log_likelihood
}

#[allow(clippy::too_many_arguments)]
fn fit_multinomial_internal(
    y_codes: &Array1<usize>,
    x: ArrayView2<'_, f64>,
    n_classes: usize,
    reference_index: usize,
    config: &MultinomialConfig,
    availability: Option<&Array2<bool>>,
    offset: Option<&Array2<f64>>,
    weights: Option<&Array1<f64>>,
    standardization: Option<&Standardization>,
    alternative_generic: Option<ArrayView3<'_, f64>>,
    alternative_specific: Option<ArrayView3<'_, f64>>,
    alternative_generic_standardization: Option<&Standardization>,
    alternative_specific_standardization: Option<&AlternativeSpecificStandardization>,
    compute_null_deviance: bool,
) -> Result<MultinomialResult> {
    let mut prepared = validate_and_prepare(
        y_codes,
        x,
        n_classes,
        reference_index,
        config,
        availability,
        offset,
        weights,
        alternative_generic,
        alternative_specific,
    )?;
    let layout = MultinomialParameterLayout::new(
        x.ncols(),
        n_classes,
        prepared.alternative_generic.dim().2,
        prepared.alternative_specific.dim().2,
        config.fit_intercept,
    )?;
    let l1_alpha = config.alpha * config.l1_ratio;
    let l2_alpha = config.alpha * (1.0 - config.l1_ratio);
    let optimization_extensions = optimization_extensions_from_config(
        &layout,
        l1_alpha,
        &config.smooth_penalties,
        &config.nonneg_indices,
        &config.nonpos_indices,
    )?;
    optimization_extensions.validate(layout.len()?)?;

    let use_standardization = should_standardize_shared_inputs(config, standardization);
    let x_work = if use_standardization {
        standardization
            .expect("checked is_some")
            .standardize_matrix(x)?
    } else {
        x.to_owned()
    };
    let x_fit = x_work.view();
    let sparse_cache = build_sparse_row_cache_if_beneficial(x_fit);
    let alternative_generic_raw = prepared.alternative_generic.clone();
    let alternative_specific_raw = prepared.alternative_specific.clone();
    standardize_alternative_inputs(
        &mut prepared,
        config,
        alternative_generic_standardization,
        alternative_specific_standardization,
    )?;

    let mut theta = initialize_coefficients(&layout, n_classes, &prepared, config)?;
    theta = apply_bound_projection(theta, &optimization_extensions);
    let mut warnings = Vec::new();
    let mut solver_status = "max_iterations".to_string();
    let mut converged = false;
    let mut iterations = 0usize;

    for iter in 0..config.max_iterations {
        let evaluation = evaluate(
            &theta,
            x_fit,
            n_classes,
            &prepared,
            l2_alpha,
            config.fit_intercept,
            sparse_cache.as_ref(),
            &optimization_extensions.quadratic_penalties,
        )?;
        let current_objective = penalized_objective_from_smooth(
            evaluation.objective,
            &theta,
            &optimization_extensions.l1_penalty,
        );

        let optimality = check_kkt(
            &theta,
            &evaluation.gradient,
            &optimization_extensions.l1_penalty,
            &optimization_extensions.bound_constraints,
        )?;
        if optimality <= config.tolerance {
            converged = true;
            solver_status = "converged".to_string();
            iterations = iter;
            break;
        }

        let proposal =
            solve_newton_proposal(&theta, &evaluation, &optimization_extensions, &mut warnings)?;
        let step_norm = parameter_l2_distance(&proposal.target, &theta);
        if !step_norm.is_finite() {
            return Err(RustyStatsError::NumericalError(
                "multinomial Newton step produced a non-finite norm".to_string(),
            ));
        }

        let mut accepted = false;
        let mut step_fraction = 1.0;
        let mut accepted_theta = theta.clone();
        let mut accepted_objective = current_objective;

        for _half_step in 0..=MAX_HALF_STEPS {
            let candidate = interpolate_parameters(&theta, &proposal.target, step_fraction);
            let candidate = apply_bound_projection(candidate, &optimization_extensions);
            let candidate_objective = penalized_objective(
                &candidate,
                x_fit,
                n_classes,
                &prepared,
                l2_alpha,
                config.fit_intercept,
                &optimization_extensions.l1_penalty,
                &optimization_extensions.quadratic_penalties,
            )?;

            if line_search_accepts(candidate_objective, current_objective, config.tolerance) {
                accepted = true;
                accepted_theta = candidate;
                accepted_objective = candidate_objective;
                break;
            }
            step_fraction = next_half_step_fraction(step_fraction);
        }

        if !accepted {
            solver_status = "step_halving_no_improvement".to_string();
            warnings.push(
                "multinomial Newton step could not improve the objective; \
                 the fit may be separated or ill-conditioned. Try ridge regularization."
                    .to_string(),
            );
            break;
        }

        let previous_objective = current_objective;
        let new_objective = accepted_objective;
        theta = accepted_theta;
        iterations = next_iteration_count(iter);

        let coefficient_norm = coefficient_l2_norm(&theta);
        if coefficient_norm_is_extreme(coefficient_norm) {
            warnings.push(
                "multinomial coefficient norm is extremely large; possible separation. \
                 Try ridge regularization."
                    .to_string(),
            );
        }

        let relative_improvement =
            relative_objective_improvement(previous_objective, new_objective);
        let scaled_step_norm = scaled_newton_step_norm(step_fraction, step_norm, coefficient_norm);
        if convergence_recheck_needed(relative_improvement, scaled_step_norm, config.tolerance) {
            // Gradient-only recheck: check_kkt consumes only the gradient, so avoid
            // rebuilding the full Hessian that evaluate(...) would discard here.
            let recheck_gradient = evaluate_gradient(
                &theta,
                x_fit,
                n_classes,
                &prepared,
                l2_alpha,
                config.fit_intercept,
                &optimization_extensions.quadratic_penalties,
            )?;
            let next_optimality = check_kkt(
                &theta,
                &recheck_gradient,
                &optimization_extensions.l1_penalty,
                &optimization_extensions.bound_constraints,
            )?;
            if kkt_converged(next_optimality, config.tolerance) {
                converged = true;
                solver_status = "converged".to_string();
                break;
            }
        }

        if config.verbose {
            eprintln!(
                "multinomial iter={} objective={:.8} rel_improvement={:.3e}",
                iterations, new_objective, relative_improvement
            );
        }
    }

    if should_emit_max_iteration_warning(converged, &solver_status) {
        warnings.push(format!(
            "multinomial Newton solver did not converge after {} iterations; \
             try increasing max_iter, simplifying the model, or using ridge regularization.",
            config.max_iterations
        ));
    }

    let final_evaluation = evaluate(
        &theta,
        x_fit,
        n_classes,
        &prepared,
        l2_alpha,
        config.fit_intercept,
        sparse_cache.as_ref(),
        &optimization_extensions.quadratic_penalties,
    )?;
    let covariance_work = if optimization_extensions.l1_penalty.alpha > 0.0 {
        if !config.skip_covariance {
            warnings.push(
                "standard errors are unavailable for multinomial lasso/elastic-net fits; \
                 the fit is post-selection and covariance is not computed."
                    .to_string(),
            );
        }
        None
    } else if config.skip_covariance {
        None
    } else {
        // A failed inversion (singular/ill-conditioned Hessian, typically from
        // separation or collinearity) must not discard an otherwise usable fit.
        // Degrade to no covariance and warn instead of propagating the error.
        match invert_hessian(&final_evaluation.hessian) {
            Ok(cov) => Some(cov),
            Err(_) => {
                warnings.push(
                    "could not invert the multinomial Hessian for the covariance \
                     matrix; standard errors are unavailable. The fit may be \
                     separated or ill-conditioned. Try ridge regularization, drop \
                     collinear terms, or set compute_covariance=false."
                        .to_string(),
                );
                None
            }
        }
    };

    let n_alt_generic = prepared.alternative_generic.dim().2;
    let n_alt_specific = prepared.alternative_specific.dim().2;
    let original_transform = original_parameter_transform(
        &layout,
        if use_standardization {
            standardization
        } else {
            None
        },
        active_alternative_generic_standardization(config, alternative_generic_standardization),
        active_alternative_specific_standardization(config, alternative_specific_standardization),
    )?;
    let theta_original = transform_parameter_vector(&theta, &original_transform)?;
    let coefficients = unflatten_coefficients(&theta_original, n_classes, x_fit.ncols());
    let alternative_generic_coefficients = unflatten_alternative_generic_coefficients(
        &theta_original,
        x_fit.ncols(),
        n_classes,
        n_alt_generic,
    );
    let alternative_specific_coefficients = unflatten_alternative_specific_coefficients(
        &theta_original,
        x_fit.ncols(),
        n_classes,
        n_alt_generic,
        n_alt_specific,
    );
    let covariance_unscaled = match covariance_work {
        Some(covariance) => Some(transform_covariance(&covariance, &original_transform)?),
        None => None,
    };
    let (smooth_edfs, total_edf) = if config.smooth_penalties.is_empty() {
        (Vec::new(), None)
    } else {
        let hessian_unpenalized = hessian(
            &theta,
            x_fit,
            &prepared,
            &final_evaluation_probabilities(&theta, x_fit, n_classes, &prepared)?,
            0.0,
            config.fit_intercept,
            sparse_cache.as_ref(),
        )?;
        let (edfs, total) = smooth_edf_diagnostics(
            &hessian_unpenalized,
            &final_evaluation.hessian,
            &layout,
            &config.smooth_penalties,
        )?;
        (edfs, Some(total))
    };

    let (linear_predictor, fitted_probabilities) = linear_predictor_and_probabilities(
        &coefficients,
        &alternative_generic_coefficients,
        &alternative_specific_coefficients,
        x,
        &alternative_generic_raw,
        &alternative_specific_raw,
        n_classes,
        &prepared.class_to_block,
        &prepared.availability,
        &prepared.offset,
    )?;
    let log_likelihood = log_likelihood_from_probabilities(
        &fitted_probabilities,
        &prepared.y_codes,
        &prepared.weights,
    )?;
    let deviance = multinomial_deviance(log_likelihood);
    let null_deviance = if compute_null_deviance {
        compute_null_deviance_value(
            &prepared,
            n_classes,
            reference_index,
            config,
            standardization,
        )?
    } else {
        f64::NAN
    };

    Ok(MultinomialResult {
        coefficients,
        alternative_generic_coefficients,
        alternative_specific_coefficients,
        fitted_probabilities,
        linear_predictor,
        log_likelihood,
        deviance,
        null_deviance,
        iterations,
        converged,
        covariance_unscaled,
        prior_weights: prepared.weights,
        y_codes: prepared.y_codes,
        reference_index,
        warnings,
        solver_status,
        smooth_edfs,
        total_edf,
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_and_prepare(
    y_codes: &Array1<usize>,
    x: ArrayView2<'_, f64>,
    n_classes: usize,
    reference_index: usize,
    config: &MultinomialConfig,
    availability: Option<&Array2<bool>>,
    offset: Option<&Array2<f64>>,
    weights: Option<&Array1<f64>>,
    alternative_generic: Option<ArrayView3<'_, f64>>,
    alternative_specific: Option<ArrayView3<'_, f64>>,
) -> Result<PreparedInputs> {
    let n = x.nrows();
    let p = x.ncols();
    if n == 0 {
        return Err(RustyStatsError::EmptyInput("X has no rows".to_string()));
    }
    if p == 0 {
        return Err(RustyStatsError::EmptyInput("X has no columns".to_string()));
    }
    if n_classes < 2 {
        return Err(RustyStatsError::InvalidValue(
            "multinomial requires at least two classes".to_string(),
        ));
    }
    if reference_index >= n_classes {
        return Err(RustyStatsError::InvalidValue(format!(
            "reference_index {} is out of range for {} classes",
            reference_index, n_classes
        )));
    }
    if y_codes.len() != n {
        return Err(RustyStatsError::dim_mismatch(
            n,
            y_codes.len(),
            "X rows vs y_codes length",
        ));
    }
    if config.max_iterations == 0 {
        return Err(RustyStatsError::InvalidValue(
            "max_iterations must be positive".to_string(),
        ));
    }
    if !config.tolerance.is_finite() || config.tolerance <= 0.0 {
        return Err(RustyStatsError::InvalidValue(
            "tolerance must be finite and positive".to_string(),
        ));
    }
    if !config.alpha.is_finite() || config.alpha < 0.0 {
        return Err(RustyStatsError::InvalidValue(
            "alpha must be finite and non-negative".to_string(),
        ));
    }
    if !config.l1_ratio.is_finite() || config.l1_ratio < 0.0 || config.l1_ratio > 1.0 {
        return Err(RustyStatsError::InvalidValue(
            "l1_ratio must be finite and in [0, 1]".to_string(),
        ));
    }
    let alternative_generic_matrix =
        validate_alternative_tensor(alternative_generic, n, n_classes, "alternative_generic")?;
    let alternative_specific_matrix =
        validate_alternative_tensor(alternative_specific, n, n_classes, "alternative_specific")?;

    let layout = MultinomialParameterLayout::new(
        p,
        n_classes,
        alternative_generic_matrix.dim().2,
        alternative_specific_matrix.dim().2,
        config.fit_intercept,
    )?;
    validate_dense_multinomial_size(
        &layout,
        config,
        DenseMultinomialOperation::Fit,
        !config.skip_covariance,
        !config.smooth_penalties.is_empty(),
    )?;

    let weights_vec = match weights {
        Some(w) => {
            if w.len() != n {
                return Err(RustyStatsError::dim_mismatch(
                    n,
                    w.len(),
                    "X rows vs weights length",
                ));
            }
            if w.iter().any(|value| !value.is_finite() || *value < 0.0) {
                return Err(RustyStatsError::InvalidValue(
                    "weights must be finite and non-negative".to_string(),
                ));
            }
            w.clone()
        }
        None => Array1::ones(n),
    };
    let total_weight = weights_vec.sum();
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return Err(RustyStatsError::InvalidValue(
            "total effective weight must be positive".to_string(),
        ));
    }

    let mut class_counts = vec![0usize; n_classes];
    let mut class_weight = vec![0.0f64; n_classes];
    for (row, &code) in y_codes.iter().enumerate() {
        if code >= n_classes {
            return Err(RustyStatsError::InvalidValue(format!(
                "y_codes[{}] = {} is out of range for {} classes",
                row, code, n_classes
            )));
        }
        class_counts[code] += 1;
        class_weight[code] += weights_vec[row];
    }
    for class_idx in 0..n_classes {
        if class_counts[class_idx] == 0 {
            return Err(RustyStatsError::InvalidValue(format!(
                "class {} has no observed rows; empty classes are not supported",
                class_idx
            )));
        }
        if class_weight[class_idx] <= 0.0 {
            return Err(RustyStatsError::InvalidValue(format!(
                "class {} has zero positive effective weight",
                class_idx
            )));
        }
    }

    let availability_matrix = match availability {
        Some(a) => {
            if a.nrows() != n || a.ncols() != n_classes {
                return Err(RustyStatsError::InvalidValue(format!(
                    "availability must have shape ({}, {}), got ({}, {})",
                    n,
                    n_classes,
                    a.nrows(),
                    a.ncols()
                )));
            }
            a.clone()
        }
        None => Array2::from_elem((n, n_classes), true),
    };

    for row in 0..n {
        let mut any_available = false;
        for class_idx in 0..n_classes {
            any_available |= availability_matrix[[row, class_idx]];
        }
        if !any_available {
            return Err(RustyStatsError::InvalidValue(format!(
                "availability row {} has no available classes",
                row
            )));
        }
        let observed = y_codes[row];
        if !availability_matrix[[row, observed]] {
            return Err(RustyStatsError::InvalidValue(format!(
                "observed class {} is unavailable on row {}",
                observed, row
            )));
        }
    }

    let offset_matrix = match offset {
        Some(o) => {
            if o.nrows() != n || o.ncols() != n_classes {
                return Err(RustyStatsError::InvalidValue(format!(
                    "offset must have shape ({}, {}), got ({}, {})",
                    n,
                    n_classes,
                    o.nrows(),
                    o.ncols()
                )));
            }
            if o.iter().any(|value| !value.is_finite()) {
                return Err(RustyStatsError::InvalidValue(
                    "offset values must be finite".to_string(),
                ));
            }
            o.clone()
        }
        None => Array2::zeros((n, n_classes)),
    };

    let mut class_to_block = vec![None; n_classes];
    let mut non_reference_classes = Vec::with_capacity(n_classes - 1);
    for class_idx in 0..n_classes {
        if class_idx != reference_index {
            class_to_block[class_idx] = Some(non_reference_classes.len());
            non_reference_classes.push(class_idx);
        }
    }

    Ok(PreparedInputs {
        y_codes: y_codes.clone(),
        availability: availability_matrix,
        offset: offset_matrix,
        weights: weights_vec,
        alternative_generic: alternative_generic_matrix,
        alternative_specific: alternative_specific_matrix,
        class_to_block,
        non_reference_classes,
    })
}

fn validate_alternative_tensor(
    values: Option<ArrayView3<'_, f64>>,
    n: usize,
    n_classes: usize,
    name: &str,
) -> Result<Array3<f64>> {
    match values {
        Some(tensor) => {
            let (rows, classes, _) = tensor.dim();
            if rows != n || classes != n_classes {
                return Err(RustyStatsError::InvalidValue(format!(
                    "{} must have shape ({}, {}, n_terms), got ({}, {}, {})",
                    name,
                    n,
                    n_classes,
                    rows,
                    classes,
                    tensor.dim().2
                )));
            }
            if tensor.iter().any(|value| !value.is_finite()) {
                return Err(RustyStatsError::InvalidValue(format!(
                    "{} values must be finite",
                    name
                )));
            }
            Ok(tensor.to_owned())
        }
        None => Ok(Array3::zeros((n, n_classes, 0))),
    }
}

fn validate_dense_multinomial_size(
    layout: &MultinomialParameterLayout,
    config: &MultinomialConfig,
    operation: DenseMultinomialOperation,
    compute_covariance: bool,
    compute_edf: bool,
) -> Result<()> {
    let q = layout.len()?;
    if q > config.max_dense_parameters {
        return Err(RustyStatsError::InvalidValue(format!(
            "multinomial dense {} would estimate q={} parameters \
             (shared p={} columns x K-1={} non-reference classes, \
             {} generic alternative terms, {} class-specific alternative terms), exceeding \
             max_dense_parameters={}. Reduce design width/classes or wait for \
             the large-p solver.",
            operation.label(),
            q,
            layout.n_shared,
            layout.n_non_reference,
            layout.n_alt_generic,
            layout.n_alt_specific,
            config.max_dense_parameters
        )));
    }
    let hessian_bytes = q
        .checked_mul(q)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f64>()))
        .ok_or_else(|| {
            RustyStatsError::InvalidValue(
                "multinomial Hessian memory estimate overflowed usize".to_string(),
            )
        })?;
    let peak_factor = operation
        .peak_factor()
        .max(if compute_covariance || compute_edf {
            3
        } else {
            1
        });
    let peak_bytes = hessian_bytes.checked_mul(peak_factor).ok_or_else(|| {
        RustyStatsError::InvalidValue(
            "multinomial dense workspace memory estimate overflowed usize".to_string(),
        )
    })?;
    if hessian_bytes > config.hessian_memory_limit_bytes {
        let estimated_mb = hessian_bytes as f64 / (1024.0 * 1024.0);
        let limit_mb = config.hessian_memory_limit_bytes as f64 / (1024.0 * 1024.0);
        return Err(RustyStatsError::InvalidValue(format!(
            "multinomial dense Hessian would require {:.1} MB for q={} \
             parameters (shared p={} columns x K-1={} non-reference classes, \
             {} generic alternative terms, {} class-specific alternative terms), exceeding \
             the configured {:.1} MB limit. Reduce high-cardinality terms, \
             remove classes, or set compute_covariance=false only if the \
             coefficient solve itself remains within the dense guard.",
            estimated_mb,
            q,
            layout.n_shared,
            layout.n_non_reference,
            layout.n_alt_generic,
            layout.n_alt_specific,
            limit_mb
        )));
    }
    if peak_bytes > config.hessian_memory_limit_bytes {
        let estimated_mb = peak_bytes as f64 / (1024.0 * 1024.0);
        let limit_mb = config.hessian_memory_limit_bytes as f64 / (1024.0 * 1024.0);
        return Err(RustyStatsError::InvalidValue(format!(
            "multinomial dense {} workspace may require {:.1} MB for q={} \
             parameters, exceeding the configured {:.1} MB limit. Disable \
             covariance/edf computation where possible or reduce design width/classes.",
            operation.label(),
            estimated_mb,
            q,
            limit_mb
        )));
    }
    Ok(())
}

fn total_parameter_count(
    p: usize,
    n_classes: usize,
    n_alt_generic: usize,
    n_alt_specific: usize,
) -> Result<usize> {
    MultinomialParameterLayout::new(p, n_classes, n_alt_generic, n_alt_specific, true)?.len()
}

fn active_alternative_generic_standardization<'a>(
    config: &MultinomialConfig,
    standardization: Option<&'a Standardization>,
) -> Option<&'a Standardization> {
    (config.alpha > 0.0 && config.standardize)
        .then_some(standardization)
        .flatten()
}

fn active_alternative_specific_standardization<'a>(
    config: &MultinomialConfig,
    standardization: Option<&'a AlternativeSpecificStandardization>,
) -> Option<&'a AlternativeSpecificStandardization> {
    (config.alpha > 0.0 && config.standardize)
        .then_some(standardization)
        .flatten()
}

fn standardize_alternative_inputs(
    prepared: &mut PreparedInputs,
    config: &MultinomialConfig,
    generic_standardization: Option<&Standardization>,
    specific_standardization: Option<&AlternativeSpecificStandardization>,
) -> Result<()> {
    if let Some(std) = active_alternative_generic_standardization(config, generic_standardization) {
        std.validate(prepared.alternative_generic.dim().2)?;
        for term_idx in 0..prepared.alternative_generic.dim().2 {
            let center = std.center[term_idx];
            let scale = std.scale[term_idx];
            if center == 0.0 && scale == 1.0 {
                continue;
            }
            for row in 0..prepared.alternative_generic.dim().0 {
                for class_idx in 0..prepared.alternative_generic.dim().1 {
                    prepared.alternative_generic[[row, class_idx, term_idx]] =
                        (prepared.alternative_generic[[row, class_idx, term_idx]] - center) / scale;
                }
            }
        }
    }

    if let Some(std) = active_alternative_specific_standardization(config, specific_standardization)
    {
        let n_alt_specific = prepared.alternative_specific.dim().2;
        std.validate((prepared.non_reference_classes.len(), n_alt_specific))?;
        for (block_idx, &class_idx) in prepared.non_reference_classes.iter().enumerate() {
            for term_idx in 0..n_alt_specific {
                let center = std.center[[block_idx, term_idx]];
                let scale = std.scale[[block_idx, term_idx]];
                if center == 0.0 && scale == 1.0 {
                    continue;
                }
                for row in 0..prepared.alternative_specific.dim().0 {
                    prepared.alternative_specific[[row, class_idx, term_idx]] =
                        (prepared.alternative_specific[[row, class_idx, term_idx]] - center)
                            / scale;
                }
            }
        }
    }

    Ok(())
}

fn initialize_coefficients(
    layout: &MultinomialParameterLayout,
    n_classes: usize,
    prepared: &PreparedInputs,
    config: &MultinomialConfig,
) -> Result<Array1<f64>> {
    if let Some(initial_theta) = &config.initial_theta {
        let expected = layout.len()?;
        if initial_theta.len() != expected {
            return Err(RustyStatsError::dim_mismatch(
                expected,
                initial_theta.len(),
                "multinomial initial_theta length",
            ));
        }
        if initial_theta.iter().any(|value| !value.is_finite()) {
            return Err(RustyStatsError::InvalidValue(
                "multinomial initial_theta values must be finite".to_string(),
            ));
        }
        return Ok(initial_theta.clone());
    }

    let p = layout.n_shared;
    let q = layout.len().expect("validated parameter count");
    let mut theta = Array1::zeros(q);
    if !config.fit_intercept || p == 0 || !all_classes_available(&prepared.availability) {
        return Ok(theta);
    }
    if prepared.offset.iter().any(|value| *value != 0.0) {
        return Ok(theta);
    }

    let mut totals = vec![0.0f64; n_classes];
    for row in 0..prepared.y_codes.len() {
        totals[prepared.y_codes[row]] += prepared.weights[row];
    }
    let reference_total = totals
        .iter()
        .enumerate()
        .find_map(|(class_idx, total)| {
            prepared.class_to_block[class_idx]
                .is_none()
                .then_some(*total)
        })
        .unwrap_or(0.0);
    if reference_total <= 0.0 {
        return Ok(theta);
    }

    for (block_idx, &class_idx) in prepared.non_reference_classes.iter().enumerate() {
        let class_total = totals[class_idx];
        if class_total > 0.0 {
            let intercept_idx = layout
                .shared(block_idx, 0)
                .expect("validated shared intercept index");
            theta[intercept_idx] = (class_total / reference_total).ln();
        }
    }
    Ok(theta)
}

fn all_classes_available(availability: &Array2<bool>) -> bool {
    availability.iter().all(|available| *available)
}

fn evaluate(
    theta: &Array1<f64>,
    x: ArrayView2<'_, f64>,
    n_classes: usize,
    prepared: &PreparedInputs,
    alpha: f64,
    fit_intercept: bool,
    sparse_cache: Option<&SparseRowCache>,
    quadratic_penalties: &[QuadraticPenaltyBlock],
) -> Result<Evaluation> {
    let (_, probabilities) = linear_predictor_and_probabilities_from_theta(
        theta,
        x,
        n_classes,
        &prepared.class_to_block,
        &prepared.availability,
        &prepared.offset,
        &prepared.alternative_generic,
        &prepared.alternative_specific,
    )?;
    let log_likelihood =
        log_likelihood_from_probabilities(&probabilities, &prepared.y_codes, &prepared.weights)?;
    let ridge = ridge_penalty(theta, x.ncols(), n_classes, alpha, fit_intercept);
    let objective = -log_likelihood + ridge + quadratic_penalty(theta, quadratic_penalties);
    let mut gradient = gradient(theta, x, prepared, &probabilities, alpha, fit_intercept)?;
    add_quadratic_gradient(&mut gradient, theta, quadratic_penalties);
    let mut hessian = hessian(
        theta,
        x,
        prepared,
        &probabilities,
        alpha,
        fit_intercept,
        sparse_cache,
    )?;
    add_quadratic_hessian(&mut hessian, quadratic_penalties);

    Ok(Evaluation {
        objective,
        gradient,
        hessian,
    })
}

/// Gradient-only evaluation for the end-of-iteration KKT recheck.
///
/// `evaluate` unconditionally assembles the full q x q Hessian, but the
/// convergence recheck only feeds the (penalized) gradient into `check_kkt`.
/// Reuse the existing probability and gradient kernels and skip the Hessian
/// assembly so a slow-progress fit near the optimum does not re-pay the
/// O(K^2 * n * p^2) Hessian build every iteration.
fn evaluate_gradient(
    theta: &Array1<f64>,
    x: ArrayView2<'_, f64>,
    n_classes: usize,
    prepared: &PreparedInputs,
    alpha: f64,
    fit_intercept: bool,
    quadratic_penalties: &[QuadraticPenaltyBlock],
) -> Result<Array1<f64>> {
    let probabilities = final_evaluation_probabilities(theta, x, n_classes, prepared)?;
    let mut gradient = gradient(theta, x, prepared, &probabilities, alpha, fit_intercept)?;
    add_quadratic_gradient(&mut gradient, theta, quadratic_penalties);
    Ok(gradient)
}

fn final_evaluation_probabilities(
    theta: &Array1<f64>,
    x: ArrayView2<'_, f64>,
    n_classes: usize,
    prepared: &PreparedInputs,
) -> Result<Array2<f64>> {
    let (_, probabilities) = linear_predictor_and_probabilities_from_theta(
        theta,
        x,
        n_classes,
        &prepared.class_to_block,
        &prepared.availability,
        &prepared.offset,
        &prepared.alternative_generic,
        &prepared.alternative_specific,
    )?;
    Ok(probabilities)
}

fn objective_only(
    theta: &Array1<f64>,
    x: ArrayView2<'_, f64>,
    n_classes: usize,
    prepared: &PreparedInputs,
    alpha: f64,
    fit_intercept: bool,
) -> Result<f64> {
    let (_, probabilities) = linear_predictor_and_probabilities_from_theta(
        theta,
        x,
        n_classes,
        &prepared.class_to_block,
        &prepared.availability,
        &prepared.offset,
        &prepared.alternative_generic,
        &prepared.alternative_specific,
    )?;
    let log_likelihood =
        log_likelihood_from_probabilities(&probabilities, &prepared.y_codes, &prepared.weights)?;
    Ok(-log_likelihood + ridge_penalty(theta, x.ncols(), n_classes, alpha, fit_intercept))
}

fn penalized_objective(
    theta: &Array1<f64>,
    x: ArrayView2<'_, f64>,
    n_classes: usize,
    prepared: &PreparedInputs,
    l2_alpha: f64,
    fit_intercept: bool,
    l1_penalty: &L1Penalty,
    quadratic_penalties: &[QuadraticPenaltyBlock],
) -> Result<f64> {
    let smooth = objective_only(theta, x, n_classes, prepared, l2_alpha, fit_intercept)?;
    Ok(penalized_objective_from_smooth(
        smooth + quadratic_penalty(theta, quadratic_penalties),
        theta,
        l1_penalty,
    ))
}

fn penalized_objective_from_smooth(
    smooth_objective: f64,
    theta: &Array1<f64>,
    l1_penalty: &L1Penalty,
) -> f64 {
    smooth_objective + l1_norm(theta, l1_penalty)
}

fn l1_norm(theta: &Array1<f64>, l1_penalty: &L1Penalty) -> f64 {
    if l1_penalty.alpha <= 0.0 {
        return 0.0;
    }
    theta
        .iter()
        .zip(l1_penalty.mask.iter())
        .filter_map(|(value, penalized)| penalized.then_some(value.abs()))
        .sum::<f64>()
        * l1_penalty.alpha
}

fn quadratic_penalty(theta: &Array1<f64>, penalties: &[QuadraticPenaltyBlock]) -> f64 {
    penalties
        .iter()
        .map(|penalty| {
            let mut value = 0.0;
            for (row_pos, &row_idx) in penalty.indices.iter().enumerate() {
                for (col_pos, &col_idx) in penalty.indices.iter().enumerate() {
                    value += theta[row_idx] * penalty.matrix[[row_pos, col_pos]] * theta[col_idx];
                }
            }
            0.5 * penalty.weight * value
        })
        .sum()
}

fn add_quadratic_gradient(
    gradient: &mut Array1<f64>,
    theta: &Array1<f64>,
    penalties: &[QuadraticPenaltyBlock],
) {
    for penalty in penalties {
        for (row_pos, &row_idx) in penalty.indices.iter().enumerate() {
            let mut value = 0.0;
            for (col_pos, &col_idx) in penalty.indices.iter().enumerate() {
                value += penalty.matrix[[row_pos, col_pos]] * theta[col_idx];
            }
            gradient[row_idx] += penalty.weight * value;
        }
    }
}

fn add_quadratic_hessian(hessian: &mut Array2<f64>, penalties: &[QuadraticPenaltyBlock]) {
    for penalty in penalties {
        for (row_pos, &row_idx) in penalty.indices.iter().enumerate() {
            for (col_pos, &col_idx) in penalty.indices.iter().enumerate() {
                hessian[[row_idx, col_idx]] += penalty.weight * penalty.matrix[[row_pos, col_pos]];
            }
        }
    }
}

fn ridge_penalty(
    theta: &Array1<f64>,
    p: usize,
    n_classes: usize,
    alpha: f64,
    fit_intercept: bool,
) -> f64 {
    if alpha <= 0.0 {
        return 0.0;
    }
    let layout = MultinomialParameterLayout::new(p, n_classes, 0, 0, fit_intercept)
        .expect("validated multinomial layout");
    theta
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| (!layout.is_shared_intercept(idx)).then_some(value * value))
        .sum::<f64>()
        * 0.5
        * alpha
}

fn linear_predictor_and_probabilities_from_theta(
    theta: &Array1<f64>,
    x: ArrayView2<'_, f64>,
    n_classes: usize,
    class_to_block: &[Option<usize>],
    availability: &Array2<bool>,
    offset: &Array2<f64>,
    alternative_generic: &Array3<f64>,
    alternative_specific: &Array3<f64>,
) -> Result<(Array2<f64>, Array2<f64>)> {
    let p = x.ncols();
    let n_alt_generic = alternative_generic.dim().2;
    let n_alt_specific = alternative_specific.dim().2;
    let expected = total_parameter_count(p, n_classes, n_alt_generic, n_alt_specific)?;
    if theta.len() != expected {
        return Err(RustyStatsError::dim_mismatch(
            expected,
            theta.len(),
            "theta length vs multinomial parameter count",
        ));
    }
    let coefficients = unflatten_coefficients(theta, n_classes, p);
    let alternative_generic_coefficients =
        unflatten_alternative_generic_coefficients(theta, p, n_classes, n_alt_generic);
    let alternative_specific_coefficients = unflatten_alternative_specific_coefficients(
        theta,
        p,
        n_classes,
        n_alt_generic,
        n_alt_specific,
    );
    linear_predictor_and_probabilities(
        &coefficients,
        &alternative_generic_coefficients,
        &alternative_specific_coefficients,
        x,
        alternative_generic,
        alternative_specific,
        n_classes,
        class_to_block,
        availability,
        offset,
    )
}

fn linear_predictor_and_probabilities(
    coefficients: &Array2<f64>,
    alternative_generic_coefficients: &Array1<f64>,
    alternative_specific_coefficients: &Array2<f64>,
    x: ArrayView2<'_, f64>,
    alternative_generic: &Array3<f64>,
    alternative_specific: &Array3<f64>,
    n_classes: usize,
    class_to_block: &[Option<usize>],
    availability: &Array2<bool>,
    offset: &Array2<f64>,
) -> Result<(Array2<f64>, Array2<f64>)> {
    let n = x.nrows();
    let p = x.ncols();
    if coefficients.nrows() != n_classes - 1 || coefficients.ncols() != p {
        return Err(RustyStatsError::InvalidValue(format!(
            "coefficient matrix must have shape ({}, {}), got ({}, {})",
            n_classes - 1,
            p,
            coefficients.nrows(),
            coefficients.ncols()
        )));
    }
    if alternative_generic.dim().0 != n
        || alternative_generic.dim().1 != n_classes
        || alternative_generic.dim().2 != alternative_generic_coefficients.len()
    {
        return Err(RustyStatsError::InvalidValue(format!(
            "generic alternative tensor shape {:?} is incompatible with {} rows, {} classes, \
             and {} coefficients",
            alternative_generic.dim(),
            n,
            n_classes,
            alternative_generic_coefficients.len()
        )));
    }
    if alternative_specific.dim().0 != n
        || alternative_specific.dim().1 != n_classes
        || alternative_specific.dim().2 != alternative_specific_coefficients.ncols()
        || alternative_specific_coefficients.nrows() != n_classes - 1
    {
        return Err(RustyStatsError::InvalidValue(format!(
            "class-specific alternative tensor shape {:?} is incompatible with {} rows, \
             {} classes, and coefficient shape {:?}",
            alternative_specific.dim(),
            n,
            n_classes,
            alternative_specific_coefficients.dim()
        )));
    }

    let mut logits = Array2::zeros((n, n_classes));
    for row in 0..n {
        for class_idx in 0..n_classes {
            let mut eta = offset[[row, class_idx]];
            for term_idx in 0..alternative_generic_coefficients.len() {
                eta += alternative_generic[[row, class_idx, term_idx]]
                    * alternative_generic_coefficients[term_idx];
            }
            if let Some(block_idx) = class_to_block[class_idx] {
                let mut dot = 0.0;
                for feature_idx in 0..p {
                    dot += x[[row, feature_idx]] * coefficients[[block_idx, feature_idx]];
                }
                eta += dot;
                for term_idx in 0..alternative_specific_coefficients.ncols() {
                    eta += alternative_specific[[row, class_idx, term_idx]]
                        * alternative_specific_coefficients[[block_idx, term_idx]];
                }
            }
            logits[[row, class_idx]] = eta;
        }
    }

    let probabilities = masked_softmax(&logits, availability)?;
    Ok((logits, probabilities))
}

fn masked_softmax(logits: &Array2<f64>, availability: &Array2<bool>) -> Result<Array2<f64>> {
    if logits.dim() != availability.dim() {
        return Err(RustyStatsError::InvalidValue(format!(
            "logits shape {:?} does not match availability shape {:?}",
            logits.dim(),
            availability.dim()
        )));
    }

    let (n, k) = logits.dim();
    let mut probabilities = Array2::zeros((n, k));
    for row in 0..n {
        let mut max_eta = f64::NEG_INFINITY;
        for class_idx in 0..k {
            if availability[[row, class_idx]] {
                let eta = logits[[row, class_idx]];
                if !eta.is_finite() {
                    return Err(RustyStatsError::NumericalError(format!(
                        "non-finite multinomial logit on row {}, class {}",
                        row, class_idx
                    )));
                }
                max_eta = max_eta.max(eta);
            }
        }
        if !max_eta.is_finite() {
            return Err(RustyStatsError::InvalidValue(format!(
                "availability row {} has no available classes",
                row
            )));
        }

        let mut denom = 0.0;
        for class_idx in 0..k {
            if availability[[row, class_idx]] {
                let value = (logits[[row, class_idx]] - max_eta).exp();
                probabilities[[row, class_idx]] = value;
                denom += value;
            }
        }
        for class_idx in 0..k {
            if availability[[row, class_idx]] {
                probabilities[[row, class_idx]] /= denom;
            }
        }
    }
    Ok(probabilities)
}

fn log_likelihood_from_probabilities(
    probabilities: &Array2<f64>,
    y_codes: &Array1<usize>,
    weights: &Array1<f64>,
) -> Result<f64> {
    let n = probabilities.nrows();
    if y_codes.len() != n || weights.len() != n {
        return Err(RustyStatsError::dim_mismatch(
            n,
            y_codes.len().min(weights.len()),
            "probability rows vs y/weights length",
        ));
    }

    let mut log_likelihood = 0.0;
    for row in 0..n {
        let class_idx = y_codes[row];
        let probability = probabilities[[row, class_idx]].max(MIN_LOG_PROBABILITY);
        log_likelihood += weights[row] * probability.ln();
    }
    Ok(log_likelihood)
}

fn gradient(
    theta: &Array1<f64>,
    x: ArrayView2<'_, f64>,
    prepared: &PreparedInputs,
    probabilities: &Array2<f64>,
    alpha: f64,
    fit_intercept: bool,
) -> Result<Array1<f64>> {
    let n = x.nrows();
    let p = x.ncols();
    let mut gradient = Array1::zeros(theta.len());
    let n_classes = probabilities.ncols();
    let n_alt_generic = prepared.alternative_generic.dim().2;
    let n_alt_specific = prepared.alternative_specific.dim().2;
    let layout = MultinomialParameterLayout::new(
        p,
        n_classes,
        n_alt_generic,
        n_alt_specific,
        fit_intercept,
    )?;
    let generic_start = layout.alternative_generic_start()?;

    for (block_idx, &class_idx) in prepared.non_reference_classes.iter().enumerate() {
        for row in 0..n {
            let observed = if prepared.y_codes[row] == class_idx {
                1.0
            } else {
                0.0
            };
            let residual = prepared.weights[row] * (probabilities[[row, class_idx]] - observed);
            for feature_idx in 0..p {
                let param_idx = layout.shared(block_idx, feature_idx)?;
                gradient[param_idx] += x[[row, feature_idx]] * residual;
            }
        }
    }

    for term_idx in 0..n_alt_generic {
        for row in 0..n {
            let mut expected = 0.0;
            for class_idx in 0..n_classes {
                expected += probabilities[[row, class_idx]]
                    * prepared.alternative_generic[[row, class_idx, term_idx]];
            }
            let observed = prepared.alternative_generic[[row, prepared.y_codes[row], term_idx]];
            gradient[generic_start + term_idx] += prepared.weights[row] * (expected - observed);
        }
    }

    for (block_idx, &class_idx) in prepared.non_reference_classes.iter().enumerate() {
        for term_idx in 0..n_alt_specific {
            let param_idx = layout.alternative_specific(block_idx, term_idx)?;
            for row in 0..n {
                let observed = if prepared.y_codes[row] == class_idx {
                    1.0
                } else {
                    0.0
                };
                let residual = prepared.weights[row] * (probabilities[[row, class_idx]] - observed);
                gradient[param_idx] +=
                    prepared.alternative_specific[[row, class_idx, term_idx]] * residual;
            }
        }
    }

    if alpha > 0.0 {
        for idx in 0..theta.len() {
            if !layout.is_shared_intercept(idx) {
                gradient[idx] += alpha * theta[idx];
            }
        }
    }

    Ok(gradient)
}

#[allow(clippy::too_many_arguments)]
fn hessian(
    theta: &Array1<f64>,
    x: ArrayView2<'_, f64>,
    prepared: &PreparedInputs,
    probabilities: &Array2<f64>,
    alpha: f64,
    fit_intercept: bool,
    sparse_cache: Option<&SparseRowCache>,
) -> Result<Array2<f64>> {
    let n = x.nrows();
    let p = x.ncols();
    let q = theta.len();
    let n_classes = probabilities.ncols();
    let n_alt_generic = prepared.alternative_generic.dim().2;
    let n_alt_specific = prepared.alternative_specific.dim().2;
    let layout = MultinomialParameterLayout::new(
        p,
        n_classes,
        n_alt_generic,
        n_alt_specific,
        fit_intercept,
    )?;
    let generic_start = layout.alternative_generic_start()?;
    let mut hessian = Array2::zeros((q, q));
    let mut pair_weights = Array1::zeros(n);

    for (block_k, &class_k) in prepared.non_reference_classes.iter().enumerate() {
        for (block_l, &class_l) in prepared
            .non_reference_classes
            .iter()
            .enumerate()
            .skip(block_k)
        {
            for row in 0..n {
                let delta = if class_k == class_l { 1.0 } else { 0.0 };
                pair_weights[row] = prepared.weights[row]
                    * probabilities[[row, class_k]]
                    * (delta - probabilities[[row, class_l]]);
            }
            let gram = compute_xtwx_with_sparse_cache(x, &pair_weights, sparse_cache)?;
            copy_hessian_block(&mut hessian, &gram, block_k, block_l, p);
        }
    }

    let mut generic_expected = Array2::zeros((n, n_alt_generic));
    for row in 0..n {
        for term_idx in 0..n_alt_generic {
            let mut expected = 0.0;
            for class_idx in 0..n_classes {
                expected += probabilities[[row, class_idx]]
                    * prepared.alternative_generic[[row, class_idx, term_idx]];
            }
            generic_expected[[row, term_idx]] = expected;
        }
    }

    for (block_idx, &class_idx) in prepared.non_reference_classes.iter().enumerate() {
        for feature_idx in 0..p {
            let shared_idx = layout.shared(block_idx, feature_idx)?;
            for term_idx in 0..n_alt_generic {
                let mut value = 0.0;
                for row in 0..n {
                    value += prepared.weights[row]
                        * x[[row, feature_idx]]
                        * probabilities[[row, class_idx]]
                        * (prepared.alternative_generic[[row, class_idx, term_idx]]
                            - generic_expected[[row, term_idx]]);
                }
                set_symmetric(&mut hessian, shared_idx, generic_start + term_idx, value);
            }
        }
    }

    for (block_l, &class_l) in prepared.non_reference_classes.iter().enumerate() {
        for feature_idx in 0..p {
            let shared_idx = layout.shared(block_l, feature_idx)?;
            for (block_m, &class_m) in prepared.non_reference_classes.iter().enumerate() {
                for term_idx in 0..n_alt_specific {
                    let specific_idx = layout.alternative_specific(block_m, term_idx)?;
                    let mut value = 0.0;
                    for row in 0..n {
                        let delta = if class_l == class_m { 1.0 } else { 0.0 };
                        value += prepared.weights[row]
                            * x[[row, feature_idx]]
                            * probabilities[[row, class_l]]
                            * (delta - probabilities[[row, class_m]])
                            * prepared.alternative_specific[[row, class_m, term_idx]];
                    }
                    set_symmetric(&mut hessian, shared_idx, specific_idx, value);
                }
            }
        }
    }

    for term_j in 0..n_alt_generic {
        let param_j = generic_start + term_j;
        for term_m in term_j..n_alt_generic {
            let param_m = generic_start + term_m;
            let mut value = 0.0;
            for row in 0..n {
                let mut expected_product = 0.0;
                for class_idx in 0..n_classes {
                    expected_product += probabilities[[row, class_idx]]
                        * prepared.alternative_generic[[row, class_idx, term_j]]
                        * prepared.alternative_generic[[row, class_idx, term_m]];
                }
                value += prepared.weights[row]
                    * (expected_product
                        - generic_expected[[row, term_j]] * generic_expected[[row, term_m]]);
            }
            set_symmetric(&mut hessian, param_j, param_m, value);
        }
    }

    for term_j in 0..n_alt_generic {
        let generic_idx = generic_start + term_j;
        for (block_m, &class_m) in prepared.non_reference_classes.iter().enumerate() {
            for term_m in 0..n_alt_specific {
                let specific_idx = layout.alternative_specific(block_m, term_m)?;
                let mut value = 0.0;
                for row in 0..n {
                    value += prepared.weights[row]
                        * probabilities[[row, class_m]]
                        * prepared.alternative_specific[[row, class_m, term_m]]
                        * (prepared.alternative_generic[[row, class_m, term_j]]
                            - generic_expected[[row, term_j]]);
                }
                set_symmetric(&mut hessian, generic_idx, specific_idx, value);
            }
        }
    }

    for (block_l, &class_l) in prepared.non_reference_classes.iter().enumerate() {
        for term_l in 0..n_alt_specific {
            let idx_l = layout.alternative_specific(block_l, term_l)?;
            for (block_m, &class_m) in prepared
                .non_reference_classes
                .iter()
                .enumerate()
                .skip(block_l)
            {
                let start_term = if block_l == block_m { term_l } else { 0 };
                for term_m in start_term..n_alt_specific {
                    let idx_m = layout.alternative_specific(block_m, term_m)?;
                    let mut value = 0.0;
                    for row in 0..n {
                        let delta = if class_l == class_m { 1.0 } else { 0.0 };
                        value += prepared.weights[row]
                            * probabilities[[row, class_l]]
                            * prepared.alternative_specific[[row, class_l, term_l]]
                            * (delta * prepared.alternative_specific[[row, class_l, term_m]]
                                - probabilities[[row, class_m]]
                                    * prepared.alternative_specific[[row, class_m, term_m]]);
                    }
                    set_symmetric(&mut hessian, idx_l, idx_m, value);
                }
            }
        }
    }

    if alpha > 0.0 {
        for idx in 0..q {
            if !layout.is_shared_intercept(idx) {
                hessian[[idx, idx]] += alpha;
            }
        }
    }

    Ok(hessian)
}

fn set_symmetric(hessian: &mut Array2<f64>, row: usize, col: usize, value: f64) {
    hessian[[row, col]] = value;
    hessian[[col, row]] = value;
}

fn copy_hessian_block(
    hessian: &mut Array2<f64>,
    block: &Array2<f64>,
    block_k: usize,
    block_l: usize,
    p: usize,
) {
    let row_start = block_k * p;
    let col_start = block_l * p;
    let mirror_row_start = block_l * p;
    let mirror_col_start = block_k * p;

    for feature_i in 0..p {
        for feature_j in 0..p {
            let value = block[[feature_i, feature_j]];
            hessian[[row_start + feature_i, col_start + feature_j]] = value;
            hessian[[mirror_row_start + feature_j, mirror_col_start + feature_i]] = value;
        }
    }
}

fn positive_l1_penalty(l1_penalty: &L1Penalty) -> bool {
    l1_penalty.alpha > 0.0
}

fn active_bound_constraints(bounds: &BoundConstraints) -> bool {
    !bounds.is_empty()
}

fn solve_newton_proposal(
    theta: &Array1<f64>,
    evaluation: &Evaluation,
    extensions: &OptimizationExtensions,
    warnings: &mut Vec<String>,
) -> Result<NewtonProposal> {
    if positive_l1_penalty(&extensions.l1_penalty) {
        solve_proximal_newton_subproblem(
            theta,
            &evaluation.gradient,
            &evaluation.hessian,
            &extensions.l1_penalty,
            &extensions.bound_constraints,
            warnings,
        )
    } else if active_bound_constraints(&extensions.bound_constraints) {
        solve_bound_projected_newton_step(
            theta,
            &evaluation.gradient,
            &evaluation.hessian,
            &extensions.bound_constraints,
        )
    } else {
        let step = solve_newton_step(&evaluation.hessian, &evaluation.gradient)?;
        let mut target = theta
            .iter()
            .zip(step.iter())
            .map(|(coef, delta)| coef - delta)
            .collect::<Array1<f64>>();
        target = apply_bound_projection(target, extensions);
        Ok(NewtonProposal { target })
    }
}

fn lower_bound_blocks_descent(sign: i8, theta_j: f64, gradient_j: f64) -> bool {
    sign > 0 && theta_j <= PROXIMAL_NEWTON_ZERO_TOLERANCE && gradient_j >= 0.0
}

fn upper_bound_blocks_descent(sign: i8, theta_j: f64, gradient_j: f64) -> bool {
    sign < 0 && theta_j >= -PROXIMAL_NEWTON_ZERO_TOLERANCE && gradient_j <= 0.0
}

fn coordinate_free_under_bounds(sign: i8, theta_j: f64, gradient_j: f64) -> bool {
    !lower_bound_blocks_descent(sign, theta_j, gradient_j)
        && !upper_bound_blocks_descent(sign, theta_j, gradient_j)
}

fn solve_bound_projected_newton_step(
    theta: &Array1<f64>,
    gradient: &Array1<f64>,
    hessian: &Array2<f64>,
    bounds: &BoundConstraints,
) -> Result<NewtonProposal> {
    let q = theta.len();
    if gradient.len() != q || hessian.nrows() != q || hessian.ncols() != q {
        return Err(RustyStatsError::InvalidValue(
            "bound-projected Newton inputs have inconsistent dimensions".to_string(),
        ));
    }
    let signs = bounds.signs(q)?;
    let mut free_indices = Vec::with_capacity(q);
    for idx in 0..q {
        if coordinate_free_under_bounds(signs[idx], theta[idx], gradient[idx]) {
            free_indices.push(idx);
        }
    }

    if free_indices.is_empty() {
        return Ok(NewtonProposal {
            target: project_with_signs(theta.clone(), &signs),
        });
    }
    let free_q = free_indices.len();
    let mut reduced_hessian = Array2::zeros((free_q, free_q));
    let mut reduced_gradient = Array1::zeros(free_q);
    for (row_pos, &row_idx) in free_indices.iter().enumerate() {
        reduced_gradient[row_pos] = gradient[row_idx];
        for (col_pos, &col_idx) in free_indices.iter().enumerate() {
            reduced_hessian[[row_pos, col_pos]] = hessian[[row_idx, col_idx]];
        }
    }
    let reduced_step = solve_newton_step(&reduced_hessian, &reduced_gradient)?;
    let mut target = theta.clone();
    for (pos, &idx) in free_indices.iter().enumerate() {
        target[idx] -= reduced_step[pos];
    }
    Ok(NewtonProposal {
        target: project_with_signs(target, &signs),
    })
}

fn use_proximal_active_subset(cd_iter: usize, active_len: usize, q: usize) -> bool {
    !cd_iter.is_multiple_of(5) && active_len < q
}

fn refresh_proximal_active_set(cd_iter: usize) -> bool {
    cd_iter == 0 || cd_iter.is_multiple_of(5)
}

fn coefficient_delta_magnitude(old: f64, updated: f64) -> f64 {
    (updated - old).abs()
}

fn zero_small_penalized_coordinate(updated: f64, penalized: bool) -> bool {
    penalized && updated.abs() < PROXIMAL_NEWTON_ZERO_TOLERANCE
}

fn proximal_coordinate_descent_tolerance(beta_scale: f64) -> f64 {
    PROXIMAL_NEWTON_CD_TOLERANCE * (1.0 + beta_scale)
}

fn proximal_coordinate_descent_converged(max_change: f64, beta_scale: f64) -> bool {
    max_change <= proximal_coordinate_descent_tolerance(beta_scale)
}

fn solve_proximal_newton_subproblem(
    theta: &Array1<f64>,
    gradient: &Array1<f64>,
    hessian: &Array2<f64>,
    l1_penalty: &L1Penalty,
    bounds: &BoundConstraints,
    warnings: &mut Vec<String>,
) -> Result<NewtonProposal> {
    let q = theta.len();
    if gradient.len() != q || hessian.nrows() != q || hessian.ncols() != q {
        return Err(RustyStatsError::InvalidValue(
            "proximal Newton inputs have inconsistent dimensions".to_string(),
        ));
    }
    l1_penalty.validate(q)?;
    bounds.validate(q)?;

    let h_theta = hessian.dot(theta);
    let linear = gradient - &h_theta;
    let mut beta = theta.clone();
    let all_indices: Vec<usize> = (0..q).collect();
    let mut active_indices = active_l1_indices(theta, gradient, l1_penalty);
    let mut cd_converged = false;

    for cd_iter in 0..PROXIMAL_NEWTON_MAX_CD_ITERATIONS {
        let indices: &[usize] = if use_proximal_active_subset(cd_iter, active_indices.len(), q) {
            &active_indices
        } else {
            &all_indices
        };
        let mut max_change = 0.0_f64;

        for &idx in indices {
            let diag = hessian[[idx, idx]];
            if diag <= PROXIMAL_NEWTON_MIN_DIAGONAL {
                if l1_penalty.mask[idx] {
                    let old = beta[idx];
                    beta[idx] = 0.0;
                    max_change = max_change.max(coefficient_delta_magnitude(old, beta[idx]));
                    continue;
                }
                return Err(RustyStatsError::LinearAlgebraError(
                    "multinomial proximal Newton Hessian has a near-zero unpenalized diagonal; \
                     try ridge regularization or remove collinear terms"
                        .to_string(),
                ));
            }

            let old = beta[idx];
            let mut partial = linear[idx];
            for col in 0..q {
                if col != idx {
                    partial += hessian[[idx, col]] * beta[col];
                }
            }
            let mut updated = if l1_penalty.mask[idx] {
                soft_threshold(-partial, l1_penalty.alpha) / diag
            } else {
                -partial / diag
            };
            updated = project_single_bound(idx, updated, bounds);
            if zero_small_penalized_coordinate(updated, l1_penalty.mask[idx]) {
                updated = 0.0;
            }
            beta[idx] = updated;
            max_change = max_change.max(coefficient_delta_magnitude(old, updated));
        }

        if refresh_proximal_active_set(cd_iter) {
            active_indices = active_l1_indices(&beta, gradient, l1_penalty);
        }
        let beta_scale = beta.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
        if proximal_coordinate_descent_converged(max_change, beta_scale) {
            cd_converged = true;
            break;
        }
    }

    if !cd_converged {
        // Degrade-and-warn rather than abort: `beta` started at `theta` and is only
        // ever updated by finite soft-threshold/bound-projected steps, so it is a
        // usable best-effort descent target. The outer line search (objective-gated)
        // and the KKT check vet acceptance. Mirrors coordinate_descent.rs and this
        // solver's own degrade-and-warn contract (covariance, step-halving).
        let warning = "multinomial proximal-Newton inner coordinate descent did not \
                       fully converge; continuing with best-effort coefficients. \
                       Increase max_iter or add ridge if this warning persists."
            .to_string();
        if !warnings.contains(&warning) {
            warnings.push(warning);
        }
    }

    Ok(NewtonProposal { target: beta })
}

fn l1_coordinate_is_active(
    theta_j: f64,
    smooth_gradient_j: f64,
    l1_penalty: &L1Penalty,
    idx: usize,
) -> bool {
    !l1_penalty.mask[idx]
        || theta_j.abs() > PROXIMAL_NEWTON_ZERO_TOLERANCE
        || l1_coordinate_violation(theta_j, smooth_gradient_j, l1_penalty.alpha)
            > PROXIMAL_NEWTON_ZERO_TOLERANCE
}

fn active_l1_indices(
    theta: &Array1<f64>,
    smooth_gradient: &Array1<f64>,
    l1_penalty: &L1Penalty,
) -> Vec<usize> {
    if l1_penalty.alpha <= 0.0 {
        return (0..theta.len()).collect();
    }
    let mut indices = Vec::new();
    for idx in 0..theta.len() {
        if l1_coordinate_is_active(theta[idx], smooth_gradient[idx], l1_penalty, idx) {
            indices.push(idx);
        }
    }
    if indices.is_empty() {
        indices.extend((0..theta.len()).filter(|idx| !l1_penalty.mask[*idx]));
    }
    indices
}

#[cfg(test)]
fn check_l1_kkt(
    theta: &Array1<f64>,
    smooth_gradient: &Array1<f64>,
    l1_penalty: &L1Penalty,
) -> Result<f64> {
    check_kkt(
        theta,
        smooth_gradient,
        l1_penalty,
        &BoundConstraints::default(),
    )
}

fn check_kkt(
    theta: &Array1<f64>,
    smooth_gradient: &Array1<f64>,
    l1_penalty: &L1Penalty,
    bounds: &BoundConstraints,
) -> Result<f64> {
    if smooth_gradient.len() != theta.len() {
        return Err(RustyStatsError::dim_mismatch(
            theta.len(),
            smooth_gradient.len(),
            "theta length vs multinomial gradient length",
        ));
    }
    l1_penalty.validate(theta.len())?;
    let signs = bounds.signs(theta.len())?;
    let mut max_violation = 0.0_f64;
    for idx in 0..theta.len() {
        let alpha = if l1_penalty.alpha > 0.0 && l1_penalty.mask[idx] {
            l1_penalty.alpha
        } else {
            0.0
        };
        let violation = match signs[idx] {
            1 => {
                let feasibility = (-theta[idx]).max(0.0);
                let stationarity = if theta[idx] <= PROXIMAL_NEWTON_ZERO_TOLERANCE {
                    (-(smooth_gradient[idx] + alpha)).max(0.0)
                } else {
                    l1_coordinate_violation(theta[idx], smooth_gradient[idx], alpha)
                };
                feasibility.max(stationarity)
            }
            -1 => {
                let feasibility = theta[idx].max(0.0);
                let stationarity = if theta[idx] >= -PROXIMAL_NEWTON_ZERO_TOLERANCE {
                    (smooth_gradient[idx] - alpha).max(0.0)
                } else {
                    l1_coordinate_violation(theta[idx], smooth_gradient[idx], alpha)
                };
                feasibility.max(stationarity)
            }
            _ => l1_coordinate_violation(theta[idx], smooth_gradient[idx], alpha),
        };
        max_violation = max_violation.max(violation);
    }
    Ok(max_violation)
}

fn l1_coordinate_violation(theta_j: f64, gradient_j: f64, alpha: f64) -> f64 {
    if alpha <= 0.0 {
        return gradient_j.abs();
    }
    if theta_j > PROXIMAL_NEWTON_ZERO_TOLERANCE {
        (gradient_j + alpha).abs()
    } else if theta_j < -PROXIMAL_NEWTON_ZERO_TOLERANCE {
        (gradient_j - alpha).abs()
    } else {
        (gradient_j.abs() - alpha).max(0.0)
    }
}

fn apply_bound_projection(theta: Array1<f64>, extensions: &OptimizationExtensions) -> Array1<f64> {
    let mut projected = theta;
    for &idx in &extensions.bound_constraints.nonneg_indices {
        if idx < projected.len() && projected[idx] < 0.0 {
            projected[idx] = 0.0;
        }
    }
    for &idx in &extensions.bound_constraints.nonpos_indices {
        if idx < projected.len() && projected[idx] > 0.0 {
            projected[idx] = 0.0;
        }
    }
    projected
}

fn project_with_signs(theta: Array1<f64>, signs: &[i8]) -> Array1<f64> {
    let mut projected = theta;
    for (idx, sign) in signs.iter().enumerate().take(projected.len()) {
        if (*sign > 0 && projected[idx] < 0.0) || (*sign < 0 && projected[idx] > 0.0) {
            projected[idx] = 0.0;
        }
    }
    projected
}

fn project_single_bound(idx: usize, value: f64, bounds: &BoundConstraints) -> f64 {
    if bounds.nonneg_indices.contains(&idx) {
        value.max(0.0)
    } else if bounds.nonpos_indices.contains(&idx) {
        value.min(0.0)
    } else {
        value
    }
}

fn solve_newton_step(hessian: &Array2<f64>, gradient: &Array1<f64>) -> Result<Array1<f64>> {
    let q = gradient.len();
    let h = array2_to_dmatrix(hessian);
    let g = DVector::from_iterator(q, gradient.iter().copied());
    if let Some(chol) = h.clone().cholesky() {
        return Ok(Array1::from_iter(chol.solve(&g).iter().copied()));
    }
    match h.lu().solve(&g) {
        Some(solution) => Ok(Array1::from_iter(solution.iter().copied())),
        None => Err(RustyStatsError::LinearAlgebraError(
            "multinomial Hessian is singular; try ridge regularization or remove collinear terms"
                .to_string(),
        )),
    }
}

fn invert_hessian(hessian: &Array2<f64>) -> Result<Array2<f64>> {
    let q = hessian.nrows();
    let h = array2_to_dmatrix(hessian);
    let inverse = if let Some(chol) = h.clone().cholesky() {
        let identity = DMatrix::identity(q, q);
        chol.solve(&identity)
    } else {
        h.try_inverse().ok_or_else(|| {
            RustyStatsError::LinearAlgebraError(
                "failed to invert multinomial Hessian for covariance; \
                 try compute_covariance=false or ridge regularization"
                    .to_string(),
            )
        })?
    };

    let mut out = Array2::zeros((q, q));
    for row in 0..q {
        for col in 0..q {
            out[[row, col]] = inverse[(row, col)];
        }
    }
    Ok(out)
}

fn smooth_edf_diagnostics(
    hessian_unpenalized: &Array2<f64>,
    hessian_penalized: &Array2<f64>,
    layout: &MultinomialParameterLayout,
    smooth_penalties: &[MultinomialSmoothPenalty],
) -> Result<(Vec<f64>, f64)> {
    let q = hessian_penalized.nrows();
    if hessian_penalized.ncols() != q
        || hessian_unpenalized.nrows() != q
        || hessian_unpenalized.ncols() != q
    {
        return Err(RustyStatsError::InvalidValue(format!(
            "EDF Hessian dimensions are inconsistent: unpenalized {:?}, penalized {:?}",
            hessian_unpenalized.dim(),
            hessian_penalized.dim()
        )));
    }

    // Reuse the crate's hat-matrix solver (Cholesky -> LU -> SVD pseudo-inverse).
    // It never fails: a singular/ill-conditioned penalized Hessian must not discard
    // an otherwise-usable fit -- the covariance path degrades identically, and EDF is
    // a diagnostics-only quantity, so a pseudo-inverse value is acceptable here.
    // influence = (H_unpen + P)^-1 H_unpen, whose trace is the effective dof.
    let influence = crate::convert::solve_symmetric_matrix(hessian_penalized, hessian_unpenalized);

    let total_edf = (0..q).map(|idx| influence[[idx, idx]]).sum::<f64>();
    let mut smooth_edfs = Vec::with_capacity(smooth_penalties.len());
    for smooth in smooth_penalties {
        let mut edf = 0.0;
        for block_idx in 0..layout.n_non_reference {
            for feature_idx in smooth.col_start..smooth.col_end {
                let idx = layout.shared(block_idx, feature_idx)?;
                edf += influence[[idx, idx]];
            }
        }
        smooth_edfs.push(edf);
    }
    Ok((smooth_edfs, total_edf))
}

fn array2_to_dmatrix(values: &Array2<f64>) -> DMatrix<f64> {
    let (rows, cols) = values.dim();
    DMatrix::from_fn(rows, cols, |row, col| values[[row, col]])
}

fn unflatten_coefficients(theta: &Array1<f64>, n_classes: usize, p: usize) -> Array2<f64> {
    let layout =
        MultinomialParameterLayout::new(p, n_classes, 0, 0, true).expect("validated layout");
    let mut coefficients = Array2::zeros((n_classes - 1, p));
    for block_idx in 0..(n_classes - 1) {
        for feature_idx in 0..p {
            let source_idx = layout
                .shared(block_idx, feature_idx)
                .expect("validated shared coefficient index");
            coefficients[[block_idx, feature_idx]] = theta[source_idx];
        }
    }
    coefficients
}

fn unflatten_alternative_generic_coefficients(
    theta: &Array1<f64>,
    p: usize,
    n_classes: usize,
    n_alt_generic: usize,
) -> Array1<f64> {
    let layout = MultinomialParameterLayout::new(p, n_classes, n_alt_generic, 0, true)
        .expect("validated multinomial layout");
    let start = layout
        .alternative_generic_start()
        .expect("validated generic alternative start");
    Array1::from_iter((0..n_alt_generic).map(|term_idx| theta[start + term_idx]))
}

fn unflatten_alternative_specific_coefficients(
    theta: &Array1<f64>,
    p: usize,
    n_classes: usize,
    n_alt_generic: usize,
    n_alt_specific: usize,
) -> Array2<f64> {
    let layout = MultinomialParameterLayout::new(p, n_classes, n_alt_generic, n_alt_specific, true)
        .expect("validated multinomial layout");
    let mut coefficients = Array2::zeros((n_classes - 1, n_alt_specific));
    for block_idx in 0..(n_classes - 1) {
        for term_idx in 0..n_alt_specific {
            let source_idx = layout
                .alternative_specific(block_idx, term_idx)
                .expect("validated class-specific alternative index");
            coefficients[[block_idx, term_idx]] = theta[source_idx];
        }
    }
    coefficients
}

fn original_parameter_transform(
    layout: &MultinomialParameterLayout,
    shared_standardization: Option<&Standardization>,
    generic_standardization: Option<&Standardization>,
    specific_standardization: Option<&AlternativeSpecificStandardization>,
) -> Result<MultinomialTransform> {
    let p = layout.n_shared;
    let n_blocks = layout.n_non_reference;
    let n_alt_generic = layout.n_alt_generic;
    let n_alt_specific = layout.n_alt_specific;
    let fit_intercept = layout.fit_intercept;
    let q = layout.len()?;
    if let Some(std) = shared_standardization {
        std.validate(p)?;
    }
    if let Some(std) = generic_standardization {
        std.validate(n_alt_generic)?;
    }
    if let Some(std) = specific_standardization {
        std.validate((n_blocks, n_alt_specific))?;
    }

    let mut rows = Vec::with_capacity(q);
    for block_idx in 0..n_blocks {
        for feature_idx in 0..p {
            let source_idx = layout.shared(block_idx, feature_idx)?;
            let mut row = vec![(
                source_idx,
                shared_scale_factor(shared_standardization, feature_idx, fit_intercept),
            )];
            if fit_intercept && feature_idx == 0 {
                if let Some(std) = shared_standardization {
                    for centered_feature in 1..p {
                        let factor = -std.center[centered_feature] / std.scale[centered_feature];
                        if factor != 0.0 {
                            row.push((layout.shared(block_idx, centered_feature)?, factor));
                        }
                    }
                }
                if let Some(std) = specific_standardization {
                    for term_idx in 0..n_alt_specific {
                        let factor =
                            -std.center[[block_idx, term_idx]] / std.scale[[block_idx, term_idx]];
                        if factor != 0.0 {
                            row.push((layout.alternative_specific(block_idx, term_idx)?, factor));
                        }
                    }
                }
            }
            rows.push(row);
        }
    }

    for term_idx in 0..n_alt_generic {
        rows.push(vec![(
            layout.alternative_generic(term_idx)?,
            generic_standardization
                .map(|std| 1.0 / std.scale[term_idx])
                .unwrap_or(1.0),
        )]);
    }

    for block_idx in 0..n_blocks {
        for term_idx in 0..n_alt_specific {
            rows.push(vec![(
                layout.alternative_specific(block_idx, term_idx)?,
                specific_standardization
                    .map(|std| 1.0 / std.scale[[block_idx, term_idx]])
                    .unwrap_or(1.0),
            )]);
        }
    }

    Ok(MultinomialTransform { rows })
}

fn shared_scale_factor(
    standardization: Option<&Standardization>,
    feature_idx: usize,
    fit_intercept: bool,
) -> f64 {
    if fit_intercept && feature_idx == 0 {
        1.0
    } else {
        standardization
            .map(|std| 1.0 / std.scale[feature_idx])
            .unwrap_or(1.0)
    }
}

fn transform_parameter_vector(
    theta: &Array1<f64>,
    transform: &MultinomialTransform,
) -> Result<Array1<f64>> {
    if theta.len() != transform.rows.len() {
        return Err(RustyStatsError::dim_mismatch(
            transform.rows.len(),
            theta.len(),
            "parameter transform vs theta length",
        ));
    }
    let mut out = Array1::zeros(theta.len());
    for (row_idx, row) in transform.rows.iter().enumerate() {
        out[row_idx] = row
            .iter()
            .map(|(source_idx, factor)| theta[*source_idx] * factor)
            .sum::<f64>();
    }
    Ok(out)
}

fn transform_covariance(
    covariance: &Array2<f64>,
    transform: &MultinomialTransform,
) -> Result<Array2<f64>> {
    let q = transform.rows.len();
    if covariance.nrows() != q || covariance.ncols() != q {
        return Err(RustyStatsError::InvalidValue(format!(
            "multinomial covariance has shape ({}, {}), expected ({}, {})",
            covariance.nrows(),
            covariance.ncols(),
            q,
            q
        )));
    }

    let mut out = Array2::zeros((q, q));
    for row_idx in 0..q {
        for col_idx in row_idx..q {
            let mut value = 0.0;
            for (source_row, row_factor) in &transform.rows[row_idx] {
                for (source_col, col_factor) in &transform.rows[col_idx] {
                    value += row_factor * col_factor * covariance[[*source_row, *source_col]];
                }
            }
            out[[row_idx, col_idx]] = value;
            out[[col_idx, row_idx]] = value;
        }
    }
    Ok(out)
}

fn compute_null_deviance_value(
    prepared: &PreparedInputs,
    n_classes: usize,
    reference_index: usize,
    config: &MultinomialConfig,
    _standardization: Option<&Standardization>,
) -> Result<f64> {
    if all_classes_available(&prepared.availability) && prepared.offset.iter().all(|v| *v == 0.0) {
        let mut class_weight = vec![0.0f64; n_classes];
        for row in 0..prepared.y_codes.len() {
            class_weight[prepared.y_codes[row]] += prepared.weights[row];
        }
        let total = class_weight.iter().sum::<f64>();
        let mut ll = 0.0;
        for row in 0..prepared.y_codes.len() {
            let share = (class_weight[prepared.y_codes[row]] / total).max(MIN_LOG_PROBABILITY);
            ll += prepared.weights[row] * share.ln();
        }
        return Ok(-2.0 * ll);
    }

    let x_null = Array2::ones((prepared.y_codes.len(), 1));
    let mut null_config = config.clone();
    null_config.alpha = 0.0;
    null_config.l1_ratio = 0.0;
    null_config.skip_covariance = true;
    null_config.fit_intercept = true;
    null_config.standardize = false;
    null_config.initial_theta = None;
    null_config.smooth_penalties.clear();
    null_config.nonneg_indices.clear();
    null_config.nonpos_indices.clear();
    null_config.max_dense_parameters = null_config.max_dense_parameters.max(n_classes - 1);

    let result = fit_multinomial_internal(
        &prepared.y_codes,
        x_null.view(),
        n_classes,
        reference_index,
        &null_config,
        Some(&prepared.availability),
        Some(&prepared.offset),
        Some(&prepared.weights),
        None,
        None,
        None,
        None,
        None,
        false,
    )?;
    Ok(result.deviance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn default_config() -> MultinomialConfig {
        MultinomialConfig {
            skip_covariance: true,
            ..Default::default()
        }
    }

    fn assert_error_contains(error: RustyStatsError, expected: &str) {
        let message = error.to_string();
        assert!(
            message.contains(expected),
            "expected error to contain '{expected}', got '{message}'"
        );
    }

    #[test]
    fn default_config_constants_are_exact() {
        let config = MultinomialConfig::default();
        assert_eq!(DEFAULT_HESSIAN_MEMORY_LIMIT_BYTES, 256 * 1024 * 1024);
        assert_eq!(config.hessian_memory_limit_bytes, 268_435_456);
        assert_eq!(config.max_dense_parameters, DEFAULT_MAX_DENSE_PARAMETERS);
        assert_eq!(config.max_iterations, DEFAULT_MAX_ITERATIONS);
        assert_abs_diff_eq!(config.tolerance, DEFAULT_TOLERANCE, epsilon = 0.0);
    }

    #[test]
    fn parameter_layout_maps_shared_and_alternative_blocks() {
        let layout = MultinomialParameterLayout::new(3, 4, 2, 5, true).expect("valid layout");

        assert_eq!(layout.len().expect("len"), 26);
        assert_eq!(layout.shared(0, 0).expect("shared"), 0);
        assert_eq!(layout.shared(1, 2).expect("shared"), 5);
        assert_eq!(
            layout.alternative_generic_start().expect("generic start"),
            9
        );
        assert_eq!(layout.alternative_generic(1).expect("generic"), 10);
        assert_eq!(
            layout.alternative_specific_start().expect("specific start"),
            11
        );
        assert_eq!(layout.alternative_specific(2, 4).expect("specific"), 25);
    }

    #[test]
    fn parameter_layout_and_alt_specific_standardization_reject_bad_metadata() {
        assert!(MultinomialParameterLayout::new(2, 1, 0, 0, true).is_err());
        let layout = MultinomialParameterLayout::new(3, 4, 2, 2, true).expect("valid layout");
        assert!(layout.shared(3, 0).is_err());
        assert!(layout.shared(0, 3).is_err());
        assert!(layout.alternative_generic(2).is_err());
        assert!(layout.alternative_specific(3, 0).is_err());
        assert!(layout.alternative_specific(0, 2).is_err());
        assert!(layout.is_shared_intercept(0));
        assert!(layout.is_shared_intercept(3));
        assert!(!layout.is_shared_intercept(1));
        assert!(!layout.is_shared_intercept(9));
        let no_shared = MultinomialParameterLayout::new(0, 4, 0, 0, true).expect("valid");
        assert!(!no_shared.is_shared_intercept(0));
        let no_intercept = MultinomialParameterLayout::new(3, 4, 0, 0, false).expect("valid");
        assert!(!no_intercept.is_shared_intercept(0));

        let center = array![[0.0, 1.0], [2.0, 3.0]];
        let scale = array![[1.0, 2.0], [3.0, 4.0]];
        let std = AlternativeSpecificStandardization::new(center.clone(), scale.clone())
            .expect("valid class-specific standardization");
        std.validate((2, 2)).expect("expected shape");
        assert!(AlternativeSpecificStandardization {
            center: Array2::zeros((1, 2)),
            scale: scale.clone(),
        }
        .validate((2, 2))
        .is_err());
        assert!(AlternativeSpecificStandardization {
            center: center.clone(),
            scale: Array2::ones((2, 1)),
        }
        .validate((2, 2))
        .is_err());
        assert!(AlternativeSpecificStandardization {
            center: array![[0.0, f64::NAN], [2.0, 3.0]],
            scale: scale.clone(),
        }
        .validate((2, 2))
        .is_err());
        assert!(AlternativeSpecificStandardization {
            center,
            scale: array![[1.0, 0.0], [3.0, 4.0]],
        }
        .validate((2, 2))
        .is_err());
    }

    #[test]
    fn parameter_layout_overflow_guards_are_reported() {
        let shared_overflow = MultinomialParameterLayout {
            n_shared: usize::MAX,
            n_non_reference: 2,
            n_alt_generic: 0,
            n_alt_specific: 0,
            fit_intercept: true,
        };
        assert_error_contains(
            shared_overflow
                .shared_len()
                .expect_err("shared length should overflow"),
            "shared parameter count overflowed",
        );
        assert_error_contains(
            shared_overflow
                .shared(1, 1)
                .expect_err("shared parameter index should overflow"),
            "shared parameter index overflowed",
        );

        let specific_count_overflow = MultinomialParameterLayout {
            n_shared: 0,
            n_non_reference: 2,
            n_alt_generic: 0,
            n_alt_specific: usize::MAX,
            fit_intercept: true,
        };
        assert_error_contains(
            specific_count_overflow
                .len()
                .expect_err("specific count should overflow"),
            "class-specific alternative parameter count overflowed",
        );

        let total_overflow = MultinomialParameterLayout {
            n_shared: 0,
            n_non_reference: 1,
            n_alt_generic: usize::MAX,
            n_alt_specific: 1,
            fit_intercept: true,
        };
        assert_error_contains(
            total_overflow
                .len()
                .expect_err("total count should overflow"),
            "total parameter count overflowed",
        );

        let generic_index_overflow = MultinomialParameterLayout {
            n_shared: usize::MAX - 1,
            n_non_reference: 1,
            n_alt_generic: usize::MAX,
            n_alt_specific: 0,
            fit_intercept: true,
        };
        assert_error_contains(
            generic_index_overflow
                .alternative_generic(2)
                .expect_err("generic parameter index should overflow"),
            "generic alternative parameter index overflowed",
        );
        assert_error_contains(
            generic_index_overflow
                .alternative_specific_start()
                .expect_err("specific start should overflow"),
            "class-specific alternative start overflowed",
        );

        let specific_index_overflow = MultinomialParameterLayout {
            n_shared: 0,
            n_non_reference: usize::MAX,
            n_alt_generic: 0,
            n_alt_specific: 2,
            fit_intercept: true,
        };
        assert_error_contains(
            specific_index_overflow
                .alternative_specific(usize::MAX - 1, 0)
                .expect_err("specific block offset should overflow"),
            "class-specific alternative block offset overflowed",
        );

        let specific_parameter_index_overflow = MultinomialParameterLayout {
            n_shared: usize::MAX - 1,
            n_non_reference: 1,
            n_alt_generic: 1,
            n_alt_specific: 2,
            fit_intercept: true,
        };
        assert_error_contains(
            specific_parameter_index_overflow
                .alternative_specific(0, 1)
                .expect_err("specific parameter index should overflow"),
            "class-specific alternative parameter index overflowed",
        );
    }

    #[test]
    fn dense_guard_reports_operation_peak_workspace() {
        let layout = MultinomialParameterLayout::new(1, 4, 0, 0, true).expect("valid layout");
        let config = MultinomialConfig {
            hessian_memory_limit_bytes: 100,
            max_dense_parameters: 100,
            ..default_config()
        };

        validate_dense_multinomial_size(
            &layout,
            &config,
            DenseMultinomialOperation::Fit,
            false,
            false,
        )
        .expect("single Hessian fits under the limit");
        let err = validate_dense_multinomial_size(
            &layout,
            &config,
            DenseMultinomialOperation::Covariance,
            false,
            false,
        )
        .expect_err("covariance workspace should exceed the limit");
        assert!(err.to_string().contains("covariance workspace"));

        let hessian_err = validate_dense_multinomial_size(
            &layout,
            &MultinomialConfig {
                hessian_memory_limit_bytes: 50,
                max_dense_parameters: 100,
                ..default_config()
            },
            DenseMultinomialOperation::Fit,
            false,
            false,
        )
        .expect_err("fit Hessian itself should exceed the smaller limit");
        assert!(hessian_err.to_string().contains("dense Hessian"));

        let exact_hessian_limit = MultinomialConfig {
            hessian_memory_limit_bytes: 72,
            max_dense_parameters: 100,
            ..default_config()
        };
        validate_dense_multinomial_size(
            &layout,
            &exact_hessian_limit,
            DenseMultinomialOperation::Fit,
            false,
            false,
        )
        .expect("hessian bytes equal to the limit are allowed");

        let exact_peak_limit = MultinomialConfig {
            hessian_memory_limit_bytes: 216,
            max_dense_parameters: 100,
            ..default_config()
        };
        validate_dense_multinomial_size(
            &layout,
            &exact_peak_limit,
            DenseMultinomialOperation::Fit,
            true,
            false,
        )
        .expect("peak workspace bytes equal to the limit are allowed");

        let large_hessian_layout =
            MultinomialParameterLayout::new(1024, 2, 0, 0, true).expect("large layout");
        let large_hessian_err = validate_dense_multinomial_size(
            &large_hessian_layout,
            &MultinomialConfig {
                hessian_memory_limit_bytes: 4 * 1024 * 1024,
                max_dense_parameters: 2000,
                ..default_config()
            },
            DenseMultinomialOperation::Fit,
            false,
            false,
        )
        .expect_err("8 MB Hessian should exceed a 4 MB limit");
        let message = large_hessian_err.to_string();
        assert!(
            message.contains("would require 8.0 MB for q=1024"),
            "{message}"
        );
        assert!(message.contains("configured 4.0 MB limit"), "{message}");

        let large_peak_layout =
            MultinomialParameterLayout::new(512, 2, 0, 0, true).expect("large peak layout");
        let large_peak_err = validate_dense_multinomial_size(
            &large_peak_layout,
            &MultinomialConfig {
                hessian_memory_limit_bytes: 4 * 1024 * 1024,
                max_dense_parameters: 2000,
                ..default_config()
            },
            DenseMultinomialOperation::Fit,
            true,
            false,
        )
        .expect_err("6 MB peak workspace should exceed a 4 MB limit");
        let message = large_peak_err.to_string();
        assert!(
            message.contains("workspace may require 6.0 MB for q=512"),
            "{message}"
        );
        assert!(message.contains("configured 4.0 MB limit"), "{message}");

        let max_parameter_boundary =
            MultinomialParameterLayout::new(5, 2, 0, 0, true).expect("boundary layout");
        validate_dense_multinomial_size(
            &max_parameter_boundary,
            &MultinomialConfig {
                max_dense_parameters: 5,
                hessian_memory_limit_bytes: 10_000,
                ..default_config()
            },
            DenseMultinomialOperation::Fit,
            false,
            false,
        )
        .expect("q exactly equal to max_dense_parameters is allowed");
    }

    #[test]
    fn optimization_extensions_validate_inert_and_reject_bad_metadata() {
        let inert = OptimizationExtensions::inert(4);
        inert.validate(4).expect("inert extensions are valid");

        let bad_l1 = OptimizationExtensions {
            l1_penalty: L1Penalty {
                alpha: 1.0,
                mask: vec![true],
            },
            ..OptimizationExtensions::inert(4)
        };
        assert!(bad_l1.validate(4).is_err());

        let bad_quadratic = OptimizationExtensions {
            quadratic_penalties: vec![QuadraticPenaltyBlock {
                indices: vec![0, 1],
                matrix: Array2::ones((1, 1)),
                weight: 1.0,
            }],
            ..OptimizationExtensions::inert(4)
        };
        assert!(bad_quadratic.validate(4).is_err());

        let bad_quadratic_rows = QuadraticPenaltyBlock {
            indices: vec![0, 1],
            matrix: Array2::ones((1, 2)),
            weight: 1.0,
        };
        assert!(bad_quadratic_rows.validate(4).is_err());
        let bad_quadratic_cols = QuadraticPenaltyBlock {
            indices: vec![0, 1],
            matrix: Array2::ones((2, 1)),
            weight: 1.0,
        };
        assert!(bad_quadratic_cols.validate(4).is_err());
        let zero_weight_quadratic = QuadraticPenaltyBlock {
            indices: vec![0],
            matrix: Array2::eye(1),
            weight: 0.0,
        };
        zero_weight_quadratic
            .validate(4)
            .expect("zero penalty weight is valid");

        let bad_bound = OptimizationExtensions {
            bound_constraints: BoundConstraints {
                nonneg_indices: vec![4],
                nonpos_indices: Vec::new(),
            },
            ..OptimizationExtensions::inert(4)
        };
        assert!(bad_bound.validate(4).is_err());

        let duplicate_bound = OptimizationExtensions {
            bound_constraints: BoundConstraints {
                nonneg_indices: vec![1],
                nonpos_indices: vec![1],
            },
            ..OptimizationExtensions::inert(4)
        };
        assert!(duplicate_bound.validate(4).is_err());

        let bad_l1_alpha = OptimizationExtensions {
            l1_penalty: L1Penalty {
                alpha: f64::NAN,
                mask: vec![true; 4],
            },
            ..OptimizationExtensions::inert(4)
        };
        assert!(bad_l1_alpha.validate(4).is_err());

        let bad_quadratic_weight = OptimizationExtensions {
            quadratic_penalties: vec![QuadraticPenaltyBlock {
                indices: vec![0],
                matrix: Array2::ones((1, 1)),
                weight: -1.0,
            }],
            ..OptimizationExtensions::inert(4)
        };
        assert!(bad_quadratic_weight.validate(4).is_err());

        let bad_quadratic_index = OptimizationExtensions {
            quadratic_penalties: vec![QuadraticPenaltyBlock {
                indices: vec![4],
                matrix: Array2::ones((1, 1)),
                weight: 1.0,
            }],
            ..OptimizationExtensions::inert(4)
        };
        assert!(bad_quadratic_index.validate(4).is_err());

        let bad_quadratic_value = OptimizationExtensions {
            quadratic_penalties: vec![QuadraticPenaltyBlock {
                indices: vec![0],
                matrix: array![[f64::INFINITY]],
                weight: 1.0,
            }],
            ..OptimizationExtensions::inert(4)
        };
        assert!(bad_quadratic_value.validate(4).is_err());

        let bad_nonpos_bound = OptimizationExtensions {
            bound_constraints: BoundConstraints {
                nonneg_indices: Vec::new(),
                nonpos_indices: vec![4],
            },
            ..OptimizationExtensions::inert(4)
        };
        assert!(bad_nonpos_bound.validate(4).is_err());

        let duplicate_nonneg_bound = OptimizationExtensions {
            bound_constraints: BoundConstraints {
                nonneg_indices: vec![1, 1],
                nonpos_indices: Vec::new(),
            },
            ..OptimizationExtensions::inert(4)
        };
        assert!(duplicate_nonneg_bound.validate(4).is_err());

        assert!(BoundConstraints::default().is_empty());
        assert!(!BoundConstraints {
            nonneg_indices: vec![0],
            nonpos_indices: Vec::new(),
        }
        .is_empty());
        assert!(!BoundConstraints {
            nonneg_indices: Vec::new(),
            nonpos_indices: vec![1],
        }
        .is_empty());

        let layout = MultinomialParameterLayout::new(2, 3, 0, 0, true).expect("layout");
        let no_l1 = optimization_extensions_from_config(&layout, 0.0, &[], &[], &[])
            .expect("zero alpha is inert");
        assert!(no_l1.l1_penalty.mask.iter().all(|active| !*active));
        let with_l1 = optimization_extensions_from_config(&layout, 0.25, &[], &[], &[])
            .expect("positive alpha activates non-intercepts");
        assert_eq!(with_l1.l1_penalty.alpha, 0.25);
        assert!(!with_l1.l1_penalty.mask[0]);
        assert!(with_l1.l1_penalty.mask[1]);
    }

    #[test]
    fn smooth_penalty_metadata_error_paths_are_explicit() {
        let layout = MultinomialParameterLayout::new(3, 3, 0, 0, true).expect("valid layout");
        for smooth in [
            MultinomialSmoothPenalty {
                col_start: 2,
                col_end: 2,
                penalty: Array2::eye(1),
                lambda: 1.0,
            },
            MultinomialSmoothPenalty {
                col_start: 1,
                col_end: 3,
                penalty: Array2::eye(1),
                lambda: 1.0,
            },
            MultinomialSmoothPenalty {
                col_start: 1,
                col_end: 2,
                penalty: Array2::eye(1),
                lambda: f64::NAN,
            },
            MultinomialSmoothPenalty {
                col_start: 1,
                col_end: 2,
                penalty: array![[f64::INFINITY]],
                lambda: 1.0,
            },
        ] {
            assert!(expand_smooth_penalty(&layout, &smooth).is_err());
        }

        let empty_width = MultinomialSmoothPenalty {
            col_start: 2,
            col_end: 2,
            penalty: Array2::zeros((0, 0)),
            lambda: 1.0,
        };
        assert_error_contains(
            expand_smooth_penalty(&layout, &empty_width)
                .expect_err("empty column range should be rejected before expansion"),
            "column range",
        );
        let out_of_range = MultinomialSmoothPenalty {
            col_start: 2,
            col_end: 4,
            penalty: Array2::eye(2),
            lambda: 1.0,
        };
        assert_error_contains(
            expand_smooth_penalty(&layout, &out_of_range)
                .expect_err("range beyond shared columns should be rejected"),
            "column range",
        );
        let bad_rows = MultinomialSmoothPenalty {
            col_start: 0,
            col_end: 2,
            penalty: Array2::ones((1, 2)),
            lambda: 1.0,
        };
        assert_error_contains(
            expand_smooth_penalty(&layout, &bad_rows)
                .expect_err("penalty row mismatch should be rejected"),
            "expected (2, 2)",
        );
        let bad_cols = MultinomialSmoothPenalty {
            col_start: 0,
            col_end: 2,
            penalty: Array2::ones((2, 1)),
            lambda: 1.0,
        };
        assert_error_contains(
            expand_smooth_penalty(&layout, &bad_cols)
                .expect_err("penalty column mismatch should be rejected"),
            "expected (2, 2)",
        );
        let zero_lambda = MultinomialSmoothPenalty {
            col_start: 1,
            col_end: 3,
            penalty: Array2::eye(2),
            lambda: 0.0,
        };
        let zero_lambda_blocks =
            expand_smooth_penalty(&layout, &zero_lambda).expect("zero lambda is valid");
        assert_eq!(zero_lambda_blocks.len(), 2);
        assert!(zero_lambda_blocks
            .iter()
            .all(|block| block.weight == 0.0 && block.indices.len() == 2));
        assert_eq!(zero_lambda_blocks[0].indices, vec![1, 2]);
        assert_eq!(zero_lambda_blocks[1].indices, vec![4, 5]);
        let tolerance_boundary = MultinomialSmoothPenalty {
            col_start: 0,
            col_end: 2,
            penalty: array![[1.0, 1e-10], [0.0, 1.0]],
            lambda: 1.0,
        };
        expand_smooth_penalty(&layout, &tolerance_boundary)
            .expect("symmetry tolerance is strict above 1e-10, not at 1e-10");
    }

    #[test]
    fn multinomial_validation_rejects_config_and_auxiliary_shape_errors() {
        let x = Array2::ones((3, 2));
        let y = array![0usize, 1, 2];
        let cfg = default_config();

        for config in [
            MultinomialConfig {
                max_iterations: 0,
                ..cfg.clone()
            },
            MultinomialConfig {
                tolerance: 0.0,
                ..cfg.clone()
            },
            MultinomialConfig {
                alpha: -1.0,
                ..cfg.clone()
            },
            MultinomialConfig {
                l1_ratio: 1.5,
                ..cfg.clone()
            },
            MultinomialConfig {
                l1_ratio: -0.1,
                ..cfg.clone()
            },
            MultinomialConfig {
                l1_ratio: f64::NAN,
                ..cfg.clone()
            },
        ] {
            assert!(validate_and_prepare(
                &y,
                x.view(),
                3,
                0,
                &config,
                None,
                None,
                None,
                None,
                None
            )
            .is_err());
        }
        for ratio in [0.0, 1.0] {
            validate_and_prepare(
                &y,
                x.view(),
                3,
                0,
                &MultinomialConfig {
                    l1_ratio: ratio,
                    ..cfg.clone()
                },
                None,
                None,
                None,
                None,
                None,
            )
            .expect("l1_ratio endpoints are valid");
        }

        assert!(
            validate_and_prepare(&y, x.view(), 1, 0, &cfg, None, None, None, None, None).is_err()
        );
        assert!(
            validate_and_prepare(&y, x.view(), 3, 3, &cfg, None, None, None, None, None).is_err()
        );
        assert!(validate_and_prepare(
            &array![0usize, 1],
            x.view(),
            3,
            0,
            &cfg,
            None,
            None,
            None,
            None,
            None
        )
        .is_err());

        let bad_weights_len = array![1.0, 2.0];
        let bad_weights_value = array![1.0, -0.5, 1.0];
        let bad_weights_nan = array![1.0, f64::NAN, 1.0];
        let zero_weights = Array1::zeros(3);
        let one_zero_weight = array![1.0, 0.0, 1.0, 1.0];
        let y_with_repeated_class = array![0usize, 1, 2, 1];
        let x_with_repeated_class = Array2::ones((4, 2));
        let overflowing_total_weights = array![f64::MAX, f64::MAX, f64::MAX];
        assert!(validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &cfg,
            None,
            None,
            Some(&bad_weights_len),
            None,
            None
        )
        .is_err());
        assert_error_contains(
            validate_and_prepare(
                &y,
                x.view(),
                3,
                0,
                &cfg,
                None,
                None,
                Some(&bad_weights_nan),
                None,
                None,
            )
            .expect_err("NaN row weight should be rejected early"),
            "weights must be finite and non-negative",
        );
        assert_error_contains(
            validate_and_prepare(
                &y,
                x.view(),
                3,
                0,
                &cfg,
                None,
                None,
                Some(&bad_weights_value),
                None,
                None,
            )
            .expect_err("negative row weight should be rejected early"),
            "weights must be finite and non-negative",
        );
        validate_and_prepare(
            &y_with_repeated_class,
            x_with_repeated_class.view(),
            3,
            0,
            &cfg,
            None,
            None,
            Some(&one_zero_weight),
            None,
            None,
        )
        .expect("individual zero weights are allowed when each class has positive weight");
        assert!(validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &cfg,
            None,
            None,
            Some(&overflowing_total_weights),
            None,
            None
        )
        .is_err());
        assert!(validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &cfg,
            None,
            None,
            Some(&zero_weights),
            None,
            None
        )
        .is_err());

        let bad_availability_shape = Array2::from_elem((2, 3), true);
        let no_available = array![
            [true, true, true],
            [false, false, false],
            [true, true, true]
        ];
        assert!(validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &cfg,
            Some(&bad_availability_shape),
            None,
            None,
            None,
            None
        )
        .is_err());
        assert!(validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &cfg,
            Some(&no_available),
            None,
            None,
            None,
            None
        )
        .is_err());

        let bad_offset_shape = Array2::<f64>::zeros((3, 2));
        let bad_offset_value = array![[0.0, 0.0, 0.0], [0.0, f64::NAN, 0.0], [0.0, 0.0, 0.0]];
        assert!(validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &cfg,
            None,
            Some(&bad_offset_shape),
            None,
            None,
            None
        )
        .is_err());
        assert!(validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &cfg,
            None,
            Some(&bad_offset_value),
            None,
            None,
            None
        )
        .is_err());

        let bad_alt_shape = Array3::<f64>::zeros((2, 3, 1));
        let mut bad_alt_value = Array3::<f64>::zeros((3, 3, 1));
        bad_alt_value[[1, 2, 0]] = f64::INFINITY;
        assert!(validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &cfg,
            None,
            None,
            None,
            Some(bad_alt_shape.view()),
            None
        )
        .is_err());
        assert!(validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &cfg,
            None,
            None,
            None,
            None,
            Some(bad_alt_value.view())
        )
        .is_err());

        let x_empty_rows = Array2::<f64>::zeros((0, 2));
        assert!(validate_and_prepare(
            &Array1::<usize>::zeros(0),
            x_empty_rows.view(),
            3,
            0,
            &cfg,
            None,
            None,
            None,
            None,
            None
        )
        .is_err());

        let x_empty_cols = Array2::<f64>::zeros((3, 0));
        assert!(validate_and_prepare(
            &y,
            x_empty_cols.view(),
            3,
            0,
            &cfg,
            None,
            None,
            None,
            None,
            None
        )
        .is_err());

        let zero_class_weight = array![1.0, 0.0, 1.0];
        assert!(validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &cfg,
            None,
            None,
            Some(&zero_class_weight),
            None,
            None
        )
        .is_err());

        let dense_flag_layout_config = MultinomialConfig {
            hessian_memory_limit_bytes: 160,
            max_dense_parameters: 100,
            skip_covariance: true,
            smooth_penalties: Vec::new(),
            ..cfg.clone()
        };
        validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &dense_flag_layout_config,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("single fit Hessian is within the dense guard");
        assert!(validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &MultinomialConfig {
                skip_covariance: false,
                ..dense_flag_layout_config.clone()
            },
            None,
            None,
            None,
            None,
            None
        )
        .is_err());
        assert!(validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &MultinomialConfig {
                smooth_penalties: vec![MultinomialSmoothPenalty {
                    col_start: 0,
                    col_end: 1,
                    penalty: Array2::eye(1),
                    lambda: 1.0,
                }],
                ..dense_flag_layout_config.clone()
            },
            None,
            None,
            None,
            None,
            None
        )
        .is_err());
    }

    #[test]
    fn multinomial_probability_and_likelihood_helpers_validate_shapes() {
        let x = Array2::ones((2, 2));
        let availability = Array2::from_elem((2, 3), true);
        let offset = Array2::zeros((2, 3));
        let class_to_block = vec![None, Some(0), Some(1)];
        let generic = Array3::<f64>::zeros((2, 3, 1));
        let specific = Array3::<f64>::zeros((2, 3, 1));
        let theta = array![0.1, 0.2, -0.1, 0.3, 0.4, -0.2, 0.5];

        let (logits, probabilities) = linear_predictor_and_probabilities_from_theta(
            &theta,
            x.view(),
            3,
            &class_to_block,
            &availability,
            &offset,
            &generic,
            &specific,
        )
        .expect("valid linear predictor");
        assert_eq!(logits.dim(), (2, 3));
        for row in 0..2 {
            assert_abs_diff_eq!(probabilities.row(row).sum(), 1.0, epsilon = 1e-12);
        }

        assert!(linear_predictor_and_probabilities_from_theta(
            &array![0.1],
            x.view(),
            3,
            &class_to_block,
            &availability,
            &offset,
            &generic,
            &specific,
        )
        .is_err());

        assert!(linear_predictor_and_probabilities(
            &Array2::zeros((1, 2)),
            &Array1::zeros(1),
            &Array2::zeros((2, 1)),
            x.view(),
            &generic,
            &specific,
            3,
            &class_to_block,
            &availability,
            &offset,
        )
        .is_err());
        assert!(linear_predictor_and_probabilities(
            &Array2::zeros((2, 2)),
            &Array1::zeros(2),
            &Array2::zeros((2, 1)),
            x.view(),
            &generic,
            &specific,
            3,
            &class_to_block,
            &availability,
            &offset,
        )
        .is_err());
        assert!(linear_predictor_and_probabilities(
            &Array2::zeros((2, 2)),
            &Array1::zeros(1),
            &Array2::zeros((1, 1)),
            x.view(),
            &generic,
            &specific,
            3,
            &class_to_block,
            &availability,
            &offset,
        )
        .is_err());
        assert_error_contains(
            linear_predictor_and_probabilities(
                &Array2::zeros((2, 2)),
                &Array1::zeros(1),
                &Array2::zeros((2, 1)),
                x.view(),
                &Array3::<f64>::zeros((1, 3, 1)),
                &specific,
                3,
                &class_to_block,
                &availability,
                &offset,
            )
            .expect_err("generic row mismatch should be rejected"),
            "generic alternative tensor",
        );
        assert_error_contains(
            linear_predictor_and_probabilities(
                &Array2::zeros((2, 2)),
                &Array1::zeros(1),
                &Array2::zeros((2, 1)),
                x.view(),
                &Array3::<f64>::zeros((2, 2, 1)),
                &specific,
                3,
                &class_to_block,
                &availability,
                &offset,
            )
            .expect_err("generic class mismatch should be rejected"),
            "generic alternative tensor",
        );
        assert_error_contains(
            linear_predictor_and_probabilities(
                &Array2::zeros((2, 2)),
                &Array1::zeros(1),
                &Array2::zeros((2, 1)),
                x.view(),
                &Array3::<f64>::zeros((2, 3, 2)),
                &specific,
                3,
                &class_to_block,
                &availability,
                &offset,
            )
            .expect_err("generic term mismatch should be rejected"),
            "generic alternative tensor",
        );
        assert_error_contains(
            linear_predictor_and_probabilities(
                &Array2::zeros((2, 2)),
                &Array1::zeros(1),
                &Array2::zeros((2, 1)),
                x.view(),
                &generic,
                &Array3::<f64>::zeros((1, 3, 1)),
                3,
                &class_to_block,
                &availability,
                &offset,
            )
            .expect_err("specific row mismatch should be rejected"),
            "class-specific alternative tensor",
        );
        assert_error_contains(
            linear_predictor_and_probabilities(
                &Array2::zeros((2, 2)),
                &Array1::zeros(1),
                &Array2::zeros((2, 1)),
                x.view(),
                &generic,
                &Array3::<f64>::zeros((2, 2, 1)),
                3,
                &class_to_block,
                &availability,
                &offset,
            )
            .expect_err("specific class mismatch should be rejected"),
            "class-specific alternative tensor",
        );
        assert_error_contains(
            linear_predictor_and_probabilities(
                &Array2::zeros((2, 2)),
                &Array1::zeros(1),
                &Array2::zeros((2, 2)),
                x.view(),
                &generic,
                &specific,
                3,
                &class_to_block,
                &availability,
                &offset,
            )
            .expect_err("specific coefficient term mismatch should be rejected"),
            "class-specific alternative tensor",
        );
        assert_error_contains(
            linear_predictor_and_probabilities(
                &Array2::zeros((2, 2)),
                &Array1::zeros(1),
                &Array2::zeros((1, 1)),
                x.view(),
                &generic,
                &specific,
                3,
                &class_to_block,
                &availability,
                &offset,
            )
            .expect_err("specific coefficient row mismatch should be rejected"),
            "class-specific alternative tensor",
        );

        assert!(masked_softmax(&Array2::zeros((2, 3)), &Array2::from_elem((2, 2), true)).is_err());
        assert!(
            masked_softmax(&array![[0.0, f64::NAN, 1.0]], &array![[true, true, true]],).is_err()
        );
        assert!(masked_softmax(&array![[0.0, 1.0]], &array![[false, false]]).is_err());
        let shifted = masked_softmax(&array![[10.0, 12.0, 11.0]], &array![[true, true, true]])
            .expect("shifted softmax");
        let shifted_denom = (-2.0_f64).exp() + 1.0 + (-1.0_f64).exp();
        assert_abs_diff_eq!(
            shifted[[0, 0]],
            (-2.0_f64).exp() / shifted_denom,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(shifted[[0, 1]], 1.0 / shifted_denom, epsilon = 1e-12);
        assert_abs_diff_eq!(
            shifted[[0, 2]],
            (-1.0_f64).exp() / shifted_denom,
            epsilon = 1e-12
        );
        let stable_large_logits =
            masked_softmax(&array![[700.0, 699.0, 698.0]], &array![[true, true, true]])
                .expect("large-logit softmax should remain finite");
        assert!(stable_large_logits.iter().all(|value| value.is_finite()));
        assert_abs_diff_eq!(stable_large_logits.row(0).sum(), 1.0, epsilon = 1e-12);
        assert!(stable_large_logits[[0, 0]] > stable_large_logits[[0, 1]]);
        assert!(stable_large_logits[[0, 1]] > stable_large_logits[[0, 2]]);

        assert!(log_likelihood_from_probabilities(
            &probabilities,
            &array![0usize],
            &Array1::ones(2)
        )
        .is_err());
    }

    #[test]
    fn objective_and_penalty_helpers_have_exact_scalar_contracts() {
        let x = array![[1.0, 0.5], [1.0, -1.0], [1.0, 2.0]];
        let y = array![0usize, 1, 2];
        let weights = array![1.0, 2.0, 3.0];
        let cfg = default_config();
        let prepared = validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &cfg,
            None,
            None,
            Some(&weights),
            None,
            None,
        )
        .expect("valid prepared inputs");
        let theta = array![0.1, 0.2, -0.3, 0.4];
        let l2_alpha = 0.5;
        let (_, probabilities) = linear_predictor_and_probabilities_from_theta(
            &theta,
            x.view(),
            3,
            &prepared.class_to_block,
            &prepared.availability,
            &prepared.offset,
            &prepared.alternative_generic,
            &prepared.alternative_specific,
        )
        .expect("probabilities");
        let log_likelihood =
            log_likelihood_from_probabilities(&probabilities, &prepared.y_codes, &prepared.weights)
                .expect("log likelihood");
        let ridge = ridge_penalty(&theta, x.ncols(), 3, l2_alpha, true);
        assert_abs_diff_eq!(ridge, 0.05, epsilon = 1e-12);
        assert_abs_diff_eq!(
            ridge_penalty(&theta, x.ncols(), 3, l2_alpha, false),
            0.075,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            ridge_penalty(&theta, x.ncols(), 3, 0.0, true),
            0.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            ridge_penalty(&theta, x.ncols(), 3, -0.25, true),
            0.0,
            epsilon = 1e-12
        );

        let smooth_objective =
            objective_only(&theta, x.view(), 3, &prepared, l2_alpha, true).expect("objective");
        assert_abs_diff_eq!(smooth_objective, -log_likelihood + ridge, epsilon = 1e-12);

        let l1 = L1Penalty {
            alpha: 0.7,
            mask: vec![true, false, true, false],
        };
        assert_abs_diff_eq!(l1_norm(&theta, &l1), 0.28, epsilon = 1e-12);
        assert_abs_diff_eq!(
            l1_norm(
                &theta,
                &L1Penalty {
                    alpha: 0.0,
                    mask: vec![true; 4],
                },
            ),
            0.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            l1_norm(
                &theta,
                &L1Penalty {
                    alpha: -1.0,
                    mask: vec![true; 4],
                },
            ),
            0.0,
            epsilon = 1e-12
        );

        let quadratic = QuadraticPenaltyBlock {
            indices: vec![1, 3],
            matrix: array![[2.0, 0.5], [0.5, 3.0]],
            weight: 0.2,
        };
        let quadratic_value = quadratic_penalty(&theta, std::slice::from_ref(&quadratic));
        assert_abs_diff_eq!(quadratic_value, 0.064, epsilon = 1e-12);
        assert_abs_diff_eq!(
            penalized_objective_from_smooth(smooth_objective + quadratic_value, &theta, &l1),
            smooth_objective + quadratic_value + l1_norm(&theta, &l1),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            penalized_objective(
                &theta,
                x.view(),
                3,
                &prepared,
                l2_alpha,
                true,
                &l1,
                &[quadratic]
            )
            .expect("penalized objective"),
            smooth_objective + quadratic_value + l1_norm(&theta, &l1),
            epsilon = 1e-12
        );
    }

    #[test]
    fn alternative_standardization_and_parameter_transform_contracts() {
        let x = Array2::ones((3, 2));
        let y = array![0usize, 1, 2];
        let cfg = MultinomialConfig {
            alpha: 0.1,
            standardize: true,
            ..default_config()
        };
        let generic_std_for_activation = Standardization::new(vec![1.0], vec![2.0]).expect("std");
        let specific_std_for_activation =
            AlternativeSpecificStandardization::new(array![[1.0], [2.0]], array![[2.0], [3.0]])
                .expect("specific std");
        assert!(active_alternative_generic_standardization(
            &cfg,
            Some(&generic_std_for_activation)
        )
        .is_some());
        assert!(active_alternative_specific_standardization(
            &cfg,
            Some(&specific_std_for_activation)
        )
        .is_some());
        assert!(active_alternative_generic_standardization(
            &MultinomialConfig {
                alpha: 0.0,
                ..cfg.clone()
            },
            Some(&generic_std_for_activation)
        )
        .is_none());
        assert!(active_alternative_specific_standardization(
            &MultinomialConfig {
                alpha: 0.0,
                ..cfg.clone()
            },
            Some(&specific_std_for_activation)
        )
        .is_none());
        assert!(active_alternative_specific_standardization(
            &MultinomialConfig {
                standardize: false,
                ..cfg.clone()
            },
            Some(&specific_std_for_activation)
        )
        .is_none());
        assert!(active_alternative_generic_standardization(&cfg, None).is_none());

        let generic = Array3::from_shape_vec(
            (3, 3, 1),
            vec![10.0, 20.0, 30.0, 12.0, 22.0, 32.0, 14.0, 24.0, 34.0],
        )
        .expect("valid generic tensor");
        let specific = Array3::from_shape_vec(
            (3, 3, 1),
            vec![1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0],
        )
        .expect("valid specific tensor");
        let mut prepared = validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &cfg,
            None,
            None,
            None,
            Some(generic.view()),
            Some(specific.view()),
        )
        .expect("valid prepared inputs");
        let generic_std = Standardization::new(vec![10.0], vec![2.0]).expect("valid std");
        let specific_std =
            AlternativeSpecificStandardization::new(array![[5.0], [9.0]], array![[5.0], [2.0]])
                .expect("valid specific std");

        standardize_alternative_inputs(
            &mut prepared,
            &cfg,
            Some(&generic_std),
            Some(&specific_std),
        )
        .expect("standardization succeeds");
        assert_abs_diff_eq!(
            prepared.alternative_generic[[1, 2, 0]],
            11.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            prepared.alternative_specific[[1, 1, 0]],
            0.2,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            prepared.alternative_specific[[2, 2, 0]],
            1.0,
            epsilon = 1e-12
        );

        let mut identity_prepared = validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &cfg,
            None,
            None,
            None,
            Some(generic.view()),
            Some(specific.view()),
        )
        .expect("valid prepared inputs");
        let identity_generic = Standardization::new(vec![0.0], vec![1.0]).expect("identity std");
        let identity_specific =
            AlternativeSpecificStandardization::new(Array2::zeros((2, 1)), Array2::ones((2, 1)))
                .expect("identity specific std");
        standardize_alternative_inputs(
            &mut identity_prepared,
            &cfg,
            Some(&identity_generic),
            Some(&identity_specific),
        )
        .expect("identity standardization should be a no-op");
        assert_abs_diff_eq!(identity_prepared.alternative_generic[[2, 2, 0]], 34.0);
        assert_abs_diff_eq!(identity_prepared.alternative_specific[[2, 2, 0]], 11.0);

        let mut scale_only_prepared = validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &cfg,
            None,
            None,
            None,
            Some(generic.view()),
            Some(specific.view()),
        )
        .expect("valid prepared inputs");
        let scale_only_generic = Standardization::new(vec![0.0], vec![2.0]).expect("scale std");
        let shift_only_specific =
            AlternativeSpecificStandardization::new(array![[5.0], [9.0]], Array2::ones((2, 1)))
                .expect("shift std");
        standardize_alternative_inputs(
            &mut scale_only_prepared,
            &cfg,
            Some(&scale_only_generic),
            Some(&shift_only_specific),
        )
        .expect("partial identity standardization still applies");
        assert_abs_diff_eq!(
            scale_only_prepared.alternative_generic[[0, 1, 0]],
            10.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            scale_only_prepared.alternative_specific[[0, 1, 0]],
            0.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            scale_only_prepared.alternative_specific[[0, 2, 0]],
            0.0,
            epsilon = 1e-12
        );

        let mut shift_and_scale_edges_prepared = validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &cfg,
            None,
            None,
            None,
            Some(generic.view()),
            Some(specific.view()),
        )
        .expect("valid prepared inputs");
        let shift_only_generic = Standardization::new(vec![10.0], vec![1.0]).expect("shift std");
        let scale_only_specific =
            AlternativeSpecificStandardization::new(Array2::zeros((2, 1)), array![[5.0], [2.0]])
                .expect("scale specific std");
        standardize_alternative_inputs(
            &mut shift_and_scale_edges_prepared,
            &cfg,
            Some(&shift_only_generic),
            Some(&scale_only_specific),
        )
        .expect("shift-only and scale-only standardization still apply");
        assert_abs_diff_eq!(
            shift_and_scale_edges_prepared.alternative_generic[[0, 1, 0]],
            10.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            shift_and_scale_edges_prepared.alternative_specific[[0, 1, 0]],
            1.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            shift_and_scale_edges_prepared.alternative_specific[[0, 2, 0]],
            4.5,
            epsilon = 1e-12
        );

        let layout = MultinomialParameterLayout::new(3, 3, 1, 1, true).expect("layout");
        let shared_std =
            Standardization::new(vec![0.0, 2.0, 4.0], vec![1.0, 2.0, 4.0]).expect("shared std");
        let generic_std = Standardization::new(vec![0.0], vec![5.0]).expect("generic std");
        let specific_std =
            AlternativeSpecificStandardization::new(array![[3.0], [6.0]], array![[3.0], [2.0]])
                .expect("specific std");
        let transform = original_parameter_transform(
            &layout,
            Some(&shared_std),
            Some(&generic_std),
            Some(&specific_std),
        )
        .expect("transform");
        let theta = Array1::from_iter((0..layout.len().expect("len")).map(|i| i as f64));
        let public_theta = transform_parameter_vector(&theta, &transform).expect("transform theta");
        assert_abs_diff_eq!(public_theta[0], -10.0, epsilon = 1e-12);
        assert_abs_diff_eq!(public_theta[1], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(public_theta[2], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(public_theta[6], 1.2, epsilon = 1e-12);
        assert_abs_diff_eq!(public_theta[7], 7.0 / 3.0, epsilon = 1e-12);
        assert_eq!(
            unflatten_alternative_generic_coefficients(&theta, 3, 3, 1),
            array![6.0]
        );
        let two_generic_layout = MultinomialParameterLayout::new(3, 3, 2, 0, true).expect("layout");
        let two_generic_theta =
            Array1::from_iter((0..two_generic_layout.len().expect("len")).map(|idx| idx as f64));
        assert_eq!(
            unflatten_alternative_generic_coefficients(&two_generic_theta, 3, 3, 2),
            array![6.0, 7.0]
        );

        assert!(transform_parameter_vector(&Array1::zeros(theta.len() - 1), &transform).is_err());
        assert!(transform_covariance(&Array2::eye(theta.len() - 1), &transform).is_err());
        assert!(
            transform_covariance(&Array2::zeros((theta.len() - 1, theta.len())), &transform)
                .is_err()
        );
        assert!(
            transform_covariance(&Array2::zeros((theta.len(), theta.len() - 1)), &transform)
                .is_err()
        );
        let covariance = Array2::eye(theta.len());
        let public_cov = transform_covariance(&covariance, &transform).expect("transform cov");
        assert_eq!(public_cov.dim(), (theta.len(), theta.len()));
        assert!(public_cov[[0, 0]] > 1.0);

        let exact_transform = MultinomialTransform {
            rows: vec![vec![(0, 2.0), (1, -1.0)], vec![(1, 3.0)]],
        };
        let exact_cov = array![[4.0, 5.0], [5.0, 9.0]];
        let exact_public_cov =
            transform_covariance(&exact_cov, &exact_transform).expect("exact transform");
        assert_abs_diff_eq!(exact_public_cov[[0, 0]], 5.0, epsilon = 1e-12);
        assert_abs_diff_eq!(exact_public_cov[[0, 1]], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(exact_public_cov[[1, 0]], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(exact_public_cov[[1, 1]], 81.0, epsilon = 1e-12);
    }

    #[test]
    fn initialize_coefficients_validates_warm_starts_and_zero_initializers() {
        let x = Array2::ones((6, 2));
        let y = array![0usize, 1, 2, 0, 1, 2];
        let cfg = default_config();
        let prepared = validate_and_prepare(&y, x.view(), 3, 0, &cfg, None, None, None, None, None)
            .expect("valid prepared inputs");
        let layout = MultinomialParameterLayout::new(2, 3, 0, 0, true).expect("layout");
        let q = layout.len().expect("parameter count");

        let bad_len_config = MultinomialConfig {
            initial_theta: Some(Array1::zeros(q - 1)),
            ..cfg.clone()
        };
        assert!(initialize_coefficients(&layout, 3, &prepared, &bad_len_config).is_err());

        let mut nonfinite_theta = Array1::zeros(q);
        nonfinite_theta[1] = f64::NAN;
        let bad_value_config = MultinomialConfig {
            initial_theta: Some(nonfinite_theta),
            ..cfg.clone()
        };
        assert!(initialize_coefficients(&layout, 3, &prepared, &bad_value_config).is_err());

        let warm_theta = Array1::from_iter((0..q).map(|idx| idx as f64 * 0.1));
        let warm_config = MultinomialConfig {
            initial_theta: Some(warm_theta.clone()),
            ..cfg.clone()
        };
        assert_eq!(
            initialize_coefficients(&layout, 3, &prepared, &warm_config).expect("valid warm start"),
            warm_theta
        );

        let nonuniform_weights = array![2.0, 3.0, 1.0, 0.0, 1.0, 11.0];
        let weighted_prepared = validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &cfg,
            None,
            None,
            Some(&nonuniform_weights),
            None,
            None,
        )
        .expect("valid weighted prepared inputs");
        let weighted_initial =
            initialize_coefficients(&layout, 3, &weighted_prepared, &cfg).expect("weighted init");
        assert_abs_diff_eq!(weighted_initial[0], 2.0_f64.ln(), epsilon = 1e-12);
        assert_abs_diff_eq!(weighted_initial[2], 6.0_f64.ln(), epsilon = 1e-12);
        assert_abs_diff_eq!(weighted_initial[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(weighted_initial[3], 0.0, epsilon = 1e-12);

        let no_intercept_config = MultinomialConfig {
            fit_intercept: false,
            ..cfg.clone()
        };
        let no_intercept_layout =
            MultinomialParameterLayout::new(2, 3, 0, 0, false).expect("layout");
        let no_intercept_prepared = validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &no_intercept_config,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("valid prepared inputs");
        assert!(initialize_coefficients(
            &no_intercept_layout,
            3,
            &no_intercept_prepared,
            &no_intercept_config,
        )
        .expect("no-intercept initialization")
        .iter()
        .all(|value| *value == 0.0));
        let no_intercept_weighted_prepared = validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &no_intercept_config,
            None,
            None,
            Some(&nonuniform_weights),
            None,
            None,
        )
        .expect("valid weighted no-intercept prepared inputs");
        assert!(initialize_coefficients(
            &no_intercept_layout,
            3,
            &no_intercept_weighted_prepared,
            &no_intercept_config,
        )
        .expect("no-intercept weighted initialization")
        .iter()
        .all(|value| *value == 0.0));

        let offset = array![
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        let offset_prepared = validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &cfg,
            None,
            Some(&offset),
            None,
            None,
            None,
        )
        .expect("valid prepared inputs with offset");
        assert!(initialize_coefficients(&layout, 3, &offset_prepared, &cfg)
            .expect("offset initialization")
            .iter()
            .all(|value| *value == 0.0));
        let offset_weighted_prepared = validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &cfg,
            None,
            Some(&offset),
            Some(&nonuniform_weights),
            None,
            None,
        )
        .expect("valid weighted prepared inputs with offset");
        assert!(
            initialize_coefficients(&layout, 3, &offset_weighted_prepared, &cfg)
                .expect("offset weighted initialization")
                .iter()
                .all(|value| *value == 0.0)
        );

        let availability = array![
            [true, true, true],
            [true, true, false],
            [true, true, true],
            [true, true, true],
            [true, true, true],
            [true, true, true],
        ];
        let availability_prepared = validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &cfg,
            Some(&availability),
            None,
            None,
            None,
            None,
        )
        .expect("valid prepared inputs with partial availability");
        assert!(
            initialize_coefficients(&layout, 3, &availability_prepared, &cfg)
                .expect("partial-availability initialization")
                .iter()
                .all(|value| *value == 0.0)
        );
        let availability_weighted_prepared = validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &cfg,
            Some(&availability),
            None,
            Some(&nonuniform_weights),
            None,
            None,
        )
        .expect("valid weighted prepared inputs with partial availability");
        assert!(
            initialize_coefficients(&layout, 3, &availability_weighted_prepared, &cfg)
                .expect("partial-availability weighted initialization")
                .iter()
                .all(|value| *value == 0.0)
        );

        assert!(all_classes_available(&Array2::from_elem((2, 3), true)));
        assert!(!all_classes_available(&array![
            [true, true, true],
            [true, false, true]
        ]));

        let manual_zero_reference = PreparedInputs {
            y_codes: array![0usize, 1, 2],
            availability: Array2::from_elem((3, 3), true),
            offset: Array2::zeros((3, 3)),
            weights: array![0.0, 2.0, 3.0],
            alternative_generic: Array3::zeros((3, 3, 0)),
            alternative_specific: Array3::zeros((3, 3, 0)),
            class_to_block: vec![None, Some(0), Some(1)],
            non_reference_classes: vec![1, 2],
        };
        assert!(
            initialize_coefficients(&layout, 3, &manual_zero_reference, &cfg)
                .expect("zero reference total is guarded")
                .iter()
                .all(|value| *value == 0.0)
        );

        let manual_zero_nonreference = PreparedInputs {
            y_codes: array![0usize, 1, 2],
            availability: Array2::from_elem((3, 3), true),
            offset: Array2::zeros((3, 3)),
            weights: array![5.0, 0.0, 3.0],
            alternative_generic: Array3::zeros((3, 3, 0)),
            alternative_specific: Array3::zeros((3, 3, 0)),
            class_to_block: vec![None, Some(0), Some(1)],
            non_reference_classes: vec![1, 2],
        };
        let manual_initial = initialize_coefficients(&layout, 3, &manual_zero_nonreference, &cfg)
            .expect("manual init");
        assert_abs_diff_eq!(manual_initial[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(manual_initial[2], (3.0_f64 / 5.0).ln(), epsilon = 1e-12);
        assert!(manual_initial.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn fit_loop_numeric_helpers_have_exact_contracts() {
        let alpha_zero_config = MultinomialConfig {
            alpha: 0.0,
            standardize: true,
            ..default_config()
        };
        let alpha_positive_config = MultinomialConfig {
            alpha: 0.2,
            standardize: true,
            ..default_config()
        };
        let standardization = Standardization::new(vec![0.0, 0.0], vec![1.0, 2.0]).expect("std");
        assert!(!should_standardize_shared_inputs(
            &alpha_zero_config,
            Some(&standardization)
        ));
        assert!(should_standardize_shared_inputs(
            &alpha_positive_config,
            Some(&standardization)
        ));
        assert!(!should_standardize_shared_inputs(
            &MultinomialConfig {
                standardize: false,
                ..alpha_positive_config.clone()
            },
            Some(&standardization)
        ));
        assert!(!should_standardize_shared_inputs(
            &alpha_positive_config,
            None
        ));

        let current = array![1.0, -2.0];
        let target = array![4.0, 3.0];
        assert_abs_diff_eq!(
            parameter_l2_distance(&target, &current),
            34.0_f64.sqrt(),
            epsilon = 1e-12
        );
        assert_eq!(
            interpolate_parameters(&current, &target, 0.25),
            array![1.75, -0.75]
        );
        assert!(line_search_accepts(10.0001, 10.0, 0.2));
        assert!(!line_search_accepts(10.001, 10.0, 0.2));
        assert!(!line_search_accepts(f64::NAN, 10.0, 0.2));
        assert_abs_diff_eq!(next_half_step_fraction(1.0), 0.5, epsilon = 1e-12);
        assert_eq!(next_iteration_count(4), 5);

        assert_abs_diff_eq!(coefficient_l2_norm(&array![3.0, 4.0]), 5.0, epsilon = 1e-12);
        assert!(!coefficient_norm_is_extreme(MAX_COEFFICIENT_NORM));
        assert!(coefficient_norm_is_extreme(
            MAX_COEFFICIENT_NORM * 1.000_001
        ));
        assert_abs_diff_eq!(
            relative_objective_improvement(9.0, 6.0),
            0.3,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(scaled_newton_step_norm(0.5, 4.0, 3.0), 0.5, epsilon = 1e-12);
        assert!(convergence_recheck_needed(0.01, 0.5, 0.02));
        assert!(convergence_recheck_needed(0.5, 0.01, 0.02));
        assert!(!convergence_recheck_needed(0.5, 0.5, 0.02));
        assert!(kkt_converged(0.02, 0.02));
        assert!(!kkt_converged(0.020_001, 0.02));
        assert!(should_emit_max_iteration_warning(false, "max_iterations"));
        assert!(!should_emit_max_iteration_warning(true, "max_iterations"));
        assert!(!should_emit_max_iteration_warning(
            false,
            "step_halving_no_improvement"
        ));
        assert_abs_diff_eq!(multinomial_deviance(-3.0), 6.0, epsilon = 1e-12);
    }

    #[test]
    fn shared_standardization_is_ignored_only_when_regularization_is_inactive() {
        let x = array![
            [1.0, -1.0],
            [1.0, -0.5],
            [1.0, 0.2],
            [1.0, 0.7],
            [1.0, 1.1],
            [1.0, 1.6],
        ];
        let y = array![0usize, 1, 2, 0, 1, 2];
        let mismatched_standardization = Standardization::new(vec![0.0], vec![1.0]).expect("std");
        let unregularized = MultinomialConfig {
            alpha: 0.0,
            standardize: true,
            ..default_config()
        };
        let result = fit_multinomial(
            &y,
            x.view(),
            3,
            0,
            &unregularized,
            None,
            None,
            None,
            Some(&mismatched_standardization),
        )
        .expect("alpha=0 fit should ignore supplied standardization metadata");
        assert!(result.converged);

        let regularized = MultinomialConfig {
            alpha: 0.1,
            standardize: true,
            ..default_config()
        };
        assert!(fit_multinomial(
            &y,
            x.view(),
            3,
            0,
            &regularized,
            None,
            None,
            None,
            Some(&mismatched_standardization),
        )
        .is_err());
    }

    #[test]
    fn newton_bound_l1_and_linear_algebra_helpers_cover_edge_contracts() {
        let hessian = Array2::eye(2);
        let gradient = array![1.0, -2.0];
        let theta = array![0.0, 0.0];

        assert!(solve_bound_projected_newton_step(
            &theta,
            &array![1.0],
            &hessian,
            &BoundConstraints::default(),
        )
        .is_err());
        assert!(solve_bound_projected_newton_step(
            &theta,
            &gradient,
            &Array2::zeros((1, 2)),
            &BoundConstraints::default(),
        )
        .is_err());
        assert!(solve_bound_projected_newton_step(
            &theta,
            &gradient,
            &Array2::zeros((2, 1)),
            &BoundConstraints::default(),
        )
        .is_err());

        let active_bounds = BoundConstraints {
            nonneg_indices: vec![0],
            nonpos_indices: vec![1],
        };
        assert!(!positive_l1_penalty(&L1Penalty::inert(2)));
        assert!(positive_l1_penalty(&L1Penalty {
            alpha: 0.1,
            mask: vec![true, false],
        }));
        assert!(!active_bound_constraints(&BoundConstraints::default()));
        assert!(active_bound_constraints(&active_bounds));

        let tol = PROXIMAL_NEWTON_ZERO_TOLERANCE;
        assert!(lower_bound_blocks_descent(1, tol, 0.0));
        assert!(!lower_bound_blocks_descent(1, tol * 2.0, 0.0));
        assert!(!lower_bound_blocks_descent(1, 0.0, -tol));
        assert!(upper_bound_blocks_descent(-1, -tol, 0.0));
        assert!(!upper_bound_blocks_descent(-1, -2.0 * tol, 0.0));
        assert!(!upper_bound_blocks_descent(-1, 0.0, tol));
        assert!(!coordinate_free_under_bounds(1, 0.0, 0.0));
        assert!(coordinate_free_under_bounds(0, -1.0, 1.0));

        assert!(!use_proximal_active_subset(0, 1, 3));
        assert!(use_proximal_active_subset(1, 1, 3));
        assert!(!use_proximal_active_subset(5, 1, 3));
        assert!(!use_proximal_active_subset(1, 3, 3));
        assert!(refresh_proximal_active_set(0));
        assert!(!refresh_proximal_active_set(1));
        assert!(refresh_proximal_active_set(5));
        assert_abs_diff_eq!(coefficient_delta_magnitude(2.0, -5.0), 7.0, epsilon = 1e-12);
        assert!(zero_small_penalized_coordinate(0.5 * tol, true));
        assert!(!zero_small_penalized_coordinate(tol, true));
        assert!(!zero_small_penalized_coordinate(0.5 * tol, false));
        let convergence_boundary = proximal_coordinate_descent_tolerance(2.0);
        assert_abs_diff_eq!(
            convergence_boundary,
            PROXIMAL_NEWTON_CD_TOLERANCE * 3.0,
            epsilon = 1e-18
        );
        assert!(proximal_coordinate_descent_converged(
            convergence_boundary,
            2.0
        ));
        assert!(!proximal_coordinate_descent_converged(
            convergence_boundary * (1.0 + 1e-8),
            2.0
        ));

        let routed_evaluation = Evaluation {
            objective: 0.0,
            gradient: array![1.0, -1.0],
            hessian: array![[10.0, -3.0], [-3.0, 1.0]],
        };
        let routed_extensions = OptimizationExtensions {
            bound_constraints: active_bounds.clone(),
            ..OptimizationExtensions::inert(2)
        };
        let routed_proposal = solve_newton_proposal(
            &theta,
            &routed_evaluation,
            &routed_extensions,
            &mut Vec::new(),
        )
        .expect("bounded proposal should route through active-set Newton");
        assert_eq!(routed_proposal.target, theta);

        let proposal =
            solve_bound_projected_newton_step(&theta, &array![1.0, -1.0], &hessian, &active_bounds)
                .expect("all constrained coordinates active at the boundary");
        assert_eq!(proposal.target, theta);

        let extensions = OptimizationExtensions {
            bound_constraints: active_bounds.clone(),
            ..OptimizationExtensions::inert(3)
        };
        assert_eq!(
            apply_bound_projection(array![-1.0, 2.0, 3.0], &extensions),
            array![0.0, 0.0, 3.0]
        );
        let projected_signed_zero = apply_bound_projection(array![-0.0, -0.0], &extensions);
        assert_eq!(projected_signed_zero[0].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(projected_signed_zero[1].to_bits(), (-0.0_f64).to_bits());
        let out_of_range_extensions = OptimizationExtensions {
            bound_constraints: BoundConstraints {
                nonneg_indices: vec![3],
                nonpos_indices: vec![4],
            },
            ..OptimizationExtensions::inert(3)
        };
        assert_eq!(
            apply_bound_projection(array![-1.0, 2.0, -3.0], &out_of_range_extensions),
            array![-1.0, 2.0, -3.0]
        );
        let exact_len_nonpos_extensions = OptimizationExtensions {
            bound_constraints: BoundConstraints {
                nonneg_indices: Vec::new(),
                nonpos_indices: vec![3],
            },
            ..OptimizationExtensions::inert(3)
        };
        assert_eq!(
            apply_bound_projection(array![-1.0, 2.0, -3.0], &exact_len_nonpos_extensions),
            array![-1.0, 2.0, -3.0]
        );
        assert_eq!(
            project_with_signs(array![-2.0, 2.0, 3.0, 0.0, -0.0], &[1, -1, 0, 1, -1]),
            array![0.0, 0.0, 3.0, 0.0, -0.0]
        );
        let projected_with_signed_zero = project_with_signs(array![-0.0, -0.0, -2.0], &[1, -1, 0]);
        assert_eq!(
            projected_with_signed_zero[0].to_bits(),
            (-0.0_f64).to_bits()
        );
        assert_eq!(
            projected_with_signed_zero[1].to_bits(),
            (-0.0_f64).to_bits()
        );
        assert_abs_diff_eq!(projected_with_signed_zero[2], -2.0, epsilon = 1e-12);
        assert_eq!(project_single_bound(0, -2.0, &active_bounds), 0.0);
        assert_eq!(project_single_bound(1, 2.0, &active_bounds), 0.0);
        assert_eq!(project_single_bound(2, 2.0, &active_bounds), 2.0);

        let free_all = solve_bound_projected_newton_step(
            &array![1.0, -1.0],
            &gradient,
            &hessian,
            &active_bounds,
        )
        .expect("inactive bounds should use full Newton step");
        assert!(free_all.target[0] >= 0.0);
        assert!(free_all.target[1] <= 0.0);

        let partial = solve_bound_projected_newton_step(
            &array![0.0, 1.0],
            &array![1.0, -2.0],
            &hessian,
            &BoundConstraints {
                nonneg_indices: vec![0],
                nonpos_indices: Vec::new(),
            },
        )
        .expect("partially active bounds should solve the reduced system");
        assert_eq!(partial.target[0], 0.0);
        assert_abs_diff_eq!(partial.target[1], 3.0, epsilon = 1e-12);

        let l1 = L1Penalty {
            alpha: 0.5,
            mask: vec![true, false],
        };
        assert!(solve_proximal_newton_subproblem(
            &theta,
            &array![1.0],
            &hessian,
            &l1,
            &BoundConstraints::default(),
            &mut Vec::new(),
        )
        .is_err());
        assert!(solve_proximal_newton_subproblem(
            &theta,
            &gradient,
            &Array2::zeros((1, 2)),
            &l1,
            &BoundConstraints::default(),
            &mut Vec::new(),
        )
        .is_err());
        assert!(solve_proximal_newton_subproblem(
            &theta,
            &gradient,
            &Array2::zeros((2, 1)),
            &l1,
            &BoundConstraints::default(),
            &mut Vec::new(),
        )
        .is_err());
        let proximal = solve_proximal_newton_subproblem(
            &array![0.0],
            &array![-3.0],
            &array![[2.0]],
            &L1Penalty {
                alpha: 0.5,
                mask: vec![true],
            },
            &BoundConstraints::default(),
            &mut Vec::new(),
        )
        .expect("one-dimensional proximal Newton step");
        assert_abs_diff_eq!(proximal.target[0], 1.25, epsilon = 1e-12);
        let proximal_projected = solve_proximal_newton_subproblem(
            &array![0.0],
            &array![-3.0],
            &array![[2.0]],
            &L1Penalty {
                alpha: 0.5,
                mask: vec![true],
            },
            &BoundConstraints {
                nonneg_indices: Vec::new(),
                nonpos_indices: vec![0],
            },
            &mut Vec::new(),
        )
        .expect("proximal Newton step honors upper sign bound");
        assert_abs_diff_eq!(proximal_projected.target[0], 0.0, epsilon = 1e-12);
        let zero_diag_penalized = solve_proximal_newton_subproblem(
            &array![1.0],
            &array![0.0],
            &array![[0.0]],
            &L1Penalty {
                alpha: 0.5,
                mask: vec![true],
            },
            &BoundConstraints::default(),
            &mut Vec::new(),
        )
        .expect("penalized zero diagonal should degrade to a zero coefficient");
        assert_eq!(zero_diag_penalized.target, array![0.0]);
        assert!(solve_proximal_newton_subproblem(
            &array![1.0],
            &array![0.0],
            &array![[0.0]],
            &L1Penalty {
                alpha: 0.5,
                mask: vec![false],
            },
            &BoundConstraints::default(),
            &mut Vec::new(),
        )
        .is_err());

        assert_eq!(
            active_l1_indices(&array![0.0, 0.0], &array![0.0, 0.0], &L1Penalty::inert(2)),
            vec![0, 1]
        );
        assert_eq!(
            active_l1_indices(
                &array![0.0, 0.0],
                &array![0.0, 0.0],
                &L1Penalty {
                    alpha: 1.0,
                    mask: vec![true, false],
                },
            ),
            vec![1]
        );
        assert_eq!(
            active_l1_indices(
                &array![2.0e-10, 0.0, 0.0],
                &array![0.0, 1.2, 0.2],
                &L1Penalty {
                    alpha: 1.0,
                    mask: vec![true, true, true],
                },
            ),
            vec![0, 1]
        );
        let strict_l1 = L1Penalty {
            alpha: 1.0,
            mask: vec![true, true, false],
        };
        assert!(!l1_coordinate_is_active(tol, 0.0, &strict_l1, 0));
        assert!(l1_coordinate_is_active(2.0 * tol, 0.0, &strict_l1, 0));
        assert!(l1_coordinate_is_active(0.0, 1.0 + 2.0 * tol, &strict_l1, 1));
        assert!(l1_coordinate_is_active(0.0, 0.0, &strict_l1, 2));
        assert!(!l1_coordinate_is_active(
            0.0,
            tol,
            &L1Penalty {
                alpha: 0.0,
                mask: vec![true],
            },
            0,
        ));
        assert!(active_l1_indices(
            &array![tol],
            &array![0.0],
            &L1Penalty {
                alpha: 1.0,
                mask: vec![true],
            },
        )
        .is_empty());
        assert_abs_diff_eq!(
            l1_coordinate_violation(0.2, -1.0, 1.0),
            0.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            l1_coordinate_violation(-0.2, 1.0, 1.0),
            0.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            l1_coordinate_violation(0.0, 1.25, 1.0),
            0.25,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(l1_coordinate_violation(0.0, 0.5, 1.0), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(
            l1_coordinate_violation(tol, 0.25, 1.0),
            0.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            l1_coordinate_violation(2.0 * tol, 0.25, 1.0),
            1.25,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            l1_coordinate_violation(-tol, -0.25, 1.0),
            0.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            l1_coordinate_violation(-2.0 * tol, -0.25, 1.0),
            1.25,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            l1_coordinate_violation(0.2, -3.0, 0.0),
            3.0,
            epsilon = 1e-12
        );

        assert!(check_kkt(
            &array![0.0, 0.0],
            &array![0.0],
            &L1Penalty::inert(2),
            &BoundConstraints::default(),
        )
        .is_err());
        assert!(
            check_kkt(
                &array![-0.1, 0.1],
                &array![-1.0, 1.0],
                &L1Penalty {
                    alpha: 0.25,
                    mask: vec![true, true],
                },
                &active_bounds,
            )
            .expect("bounded KKT should compute")
                > 0.0
        );
        assert_abs_diff_eq!(
            check_kkt(
                &array![0.0],
                &array![-0.75],
                &L1Penalty {
                    alpha: 0.25,
                    mask: vec![true],
                },
                &BoundConstraints {
                    nonneg_indices: vec![0],
                    nonpos_indices: Vec::new(),
                },
            )
            .expect("lower-bound stationarity"),
            0.5,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            check_kkt(
                &array![0.0],
                &array![0.75],
                &L1Penalty {
                    alpha: 0.25,
                    mask: vec![true],
                },
                &BoundConstraints {
                    nonneg_indices: vec![0],
                    nonpos_indices: Vec::new(),
                },
            )
            .expect("lower-bound outward gradient is KKT-feasible"),
            0.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            check_kkt(
                &array![0.0],
                &array![0.75],
                &L1Penalty {
                    alpha: 0.25,
                    mask: vec![true],
                },
                &BoundConstraints {
                    nonneg_indices: Vec::new(),
                    nonpos_indices: vec![0],
                },
            )
            .expect("upper-bound stationarity"),
            0.5,
            epsilon = 1e-12
        );

        let indefinite = array![[0.0, 1.0], [1.0, 0.0]];
        let step = solve_newton_step(&indefinite, &gradient).expect("LU fallback should solve");
        assert_abs_diff_eq!(step[0], -2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(step[1], 1.0, epsilon = 1e-12);
        assert!(solve_newton_step(&Array2::zeros((2, 2)), &gradient).is_err());

        let inverse = invert_hessian(&indefinite).expect("LU inverse fallback should solve");
        assert_abs_diff_eq!(inverse[[0, 1]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(inverse[[1, 0]], 1.0, epsilon = 1e-12);
        assert!(invert_hessian(&Array2::zeros((2, 2))).is_err());
    }

    #[test]
    fn null_deviance_fallback_handles_availability_and_offset_models() {
        let x = Array2::ones((6, 2));
        let y = array![0usize, 1, 2, 0, 1, 2];
        let fast_weights = array![2.0, 3.0, 1.0, 0.0, 1.0, 11.0];
        let cfg = default_config();
        let fast_prepared = validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &cfg,
            None,
            None,
            Some(&fast_weights),
            None,
            None,
        )
        .expect("valid fast-path prepared inputs");
        let fast_expected = {
            let class_weight = [2.0, 4.0, 12.0];
            let total = class_weight.iter().sum::<f64>();
            let ll = y
                .iter()
                .zip(fast_weights.iter())
                .map(|(&code, &weight)| weight * (class_weight[code] / total).ln())
                .sum::<f64>();
            -2.0 * ll
        };
        assert_abs_diff_eq!(
            compute_null_deviance_value(&fast_prepared, 3, 0, &cfg, None).expect("fast null"),
            fast_expected,
            epsilon = 1e-12
        );

        let availability = array![
            [true, true, false],
            [true, true, true],
            [true, false, true],
            [true, true, true],
            [true, true, false],
            [true, true, true],
        ];
        let offset = array![
            [0.0, 0.1, 0.0],
            [0.0, 0.0, -0.2],
            [0.0, 0.0, 0.2],
            [0.0, -0.1, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, -0.2],
        ];
        let prepared = validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &cfg,
            Some(&availability),
            Some(&offset),
            None,
            None,
            None,
        )
        .expect("valid prepared inputs");

        let null_deviance =
            compute_null_deviance_value(&prepared, 3, 0, &cfg, None).expect("null fallback fit");
        assert!(null_deviance.is_finite());
        assert!(null_deviance >= 0.0);

        let all_available_with_offset = validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &cfg,
            None,
            Some(&offset),
            None,
            None,
            None,
        )
        .expect("all classes available with offset");
        let offset_fallback =
            compute_null_deviance_value(&all_available_with_offset, 3, 0, &cfg, None)
                .expect("offset fallback");
        let offset_fast_formula = {
            let class_weight = [2.0, 2.0, 2.0];
            let total = class_weight.iter().sum::<f64>();
            let ll = y
                .iter()
                .map(|&code| (class_weight[code] / total).ln())
                .sum::<f64>();
            -2.0 * ll
        };
        assert!((offset_fallback - offset_fast_formula).abs() > 1e-6);

        let partial_available_zero_offset = validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &cfg,
            Some(&availability),
            None,
            None,
            None,
            None,
        )
        .expect("partial availability without offset");
        let availability_fallback =
            compute_null_deviance_value(&partial_available_zero_offset, 3, 0, &cfg, None)
                .expect("availability fallback");
        assert!((availability_fallback - offset_fast_formula).abs() > 1e-6);
    }

    #[test]
    fn smooth_penalty_expands_to_each_non_reference_shared_block() {
        let layout = MultinomialParameterLayout::new(4, 3, 0, 0, true).expect("valid layout");
        let smooth = MultinomialSmoothPenalty {
            col_start: 1,
            col_end: 3,
            penalty: array![[1.0, 0.25], [0.25, 2.0]],
            lambda: 4.0,
        };

        let extensions = optimization_extensions_from_config(&layout, 0.0, &[smooth], &[], &[])
            .expect("extensions");

        assert_eq!(extensions.quadratic_penalties.len(), 2);
        assert_eq!(extensions.quadratic_penalties[0].indices, vec![1, 2]);
        assert_eq!(extensions.quadratic_penalties[1].indices, vec![5, 6]);
        for penalty in &extensions.quadratic_penalties {
            assert_abs_diff_eq!(penalty.weight, 4.0, epsilon = 1e-12);
            assert_abs_diff_eq!(penalty.matrix[[0, 0]], 1.0, epsilon = 1e-12);
            assert_abs_diff_eq!(penalty.matrix[[0, 1]], 0.25, epsilon = 1e-12);
            assert_abs_diff_eq!(penalty.matrix[[1, 1]], 2.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn smooth_quadratic_gradient_and_hessian_match_finite_differences() {
        let penalty = QuadraticPenaltyBlock {
            indices: vec![1, 3],
            matrix: array![[2.0, -0.5], [-0.5, 1.5]],
            weight: 0.7,
        };
        let penalties = vec![penalty];
        let theta = array![0.3, -0.8, 0.1, 1.2];
        let mut gradient = Array1::zeros(theta.len());
        let mut hessian = Array2::zeros((theta.len(), theta.len()));

        add_quadratic_gradient(&mut gradient, &theta, &penalties);
        add_quadratic_hessian(&mut hessian, &penalties);

        let eps = 1e-6;
        for idx in 0..theta.len() {
            let mut plus = theta.clone();
            let mut minus = theta.clone();
            plus[idx] += eps;
            minus[idx] -= eps;
            let fd_grad = (quadratic_penalty(&plus, &penalties)
                - quadratic_penalty(&minus, &penalties))
                / (2.0 * eps);
            assert_abs_diff_eq!(gradient[idx], fd_grad, epsilon = 1e-6);
        }

        assert_abs_diff_eq!(hessian[[1, 1]], 1.4, epsilon = 1e-12);
        assert_abs_diff_eq!(hessian[[1, 3]], -0.35, epsilon = 1e-12);
        assert_abs_diff_eq!(hessian[[3, 1]], -0.35, epsilon = 1e-12);
        assert_abs_diff_eq!(hessian[[3, 3]], 1.05, epsilon = 1e-12);
        assert_abs_diff_eq!(hessian[[0, 0]], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn gradient_and_hessian_match_hand_computed_rich_alternative_fixture() {
        let x = array![[2.0], [-1.0]];
        let y = array![1usize, 2];
        let weights = array![1.5, 0.5];
        let generic = Array3::from_shape_vec((2, 3, 1), vec![0.0, 1.0, 2.0, 1.0, -1.0, 0.5])
            .expect("generic shape");
        let specific = Array3::from_shape_vec((2, 3, 1), vec![0.0, 3.0, 5.0, 0.0, -2.0, 4.0])
            .expect("specific shape");
        let prepared = PreparedInputs {
            y_codes: y,
            availability: Array2::from_elem((2, 3), true),
            offset: Array2::zeros((2, 3)),
            weights,
            alternative_generic: generic,
            alternative_specific: specific,
            class_to_block: vec![None, Some(0), Some(1)],
            non_reference_classes: vec![1, 2],
        };
        let theta = array![0.2, -0.3, 0.4, -0.5, 0.6];
        let probabilities = Array2::from_elem((2, 3), 1.0 / 3.0);
        let grad =
            gradient(&theta, x.view(), &prepared, &probabilities, 0.4, true).expect("gradient");

        assert_abs_diff_eq!(grad[0], -13.0 / 6.0, epsilon = 1e-12);
        assert_abs_diff_eq!(grad[1], 4.0 / 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(grad[2], -1.0 / 150.0, epsilon = 1e-12);
        assert_abs_diff_eq!(grad[3], -53.0 / 15.0, epsilon = 1e-12);
        assert_abs_diff_eq!(grad[4], 211.0 / 150.0, epsilon = 1e-12);

        let hess =
            hessian(&theta, x.view(), &prepared, &probabilities, 0.4, true, None).expect("hessian");
        assert_abs_diff_eq!(hess[[0, 0]], 13.0 / 9.0, epsilon = 1e-12);
        assert_abs_diff_eq!(hess[[0, 1]], -13.0 / 18.0, epsilon = 1e-12);
        assert_abs_diff_eq!(hess[[1, 1]], 13.0 / 9.0, epsilon = 1e-12);
        assert_abs_diff_eq!(hess[[0, 2]], 7.0 / 36.0, epsilon = 1e-12);
        assert_abs_diff_eq!(hess[[1, 2]], 17.0 / 18.0, epsilon = 1e-12);
        assert_abs_diff_eq!(hess[[2, 2]], 49.0 / 36.0 + 0.4, epsilon = 1e-12);
        assert_abs_diff_eq!(hess[[3, 4]], -37.0 / 18.0, epsilon = 1e-12);
        assert_abs_diff_eq!(hess[[4, 4]], 91.0 / 9.0 + 0.4, epsilon = 1e-12);
        assert_abs_diff_eq!(hess[[2, 0]], hess[[0, 2]], epsilon = 1e-12);
        assert_abs_diff_eq!(hess[[4, 3]], hess[[3, 4]], epsilon = 1e-12);
    }

    #[test]
    fn gradient_and_hessian_use_forward_offsets_for_multiple_alternative_terms() {
        let x = array![[1.0]];
        let generic = Array3::from_shape_vec((1, 3, 2), vec![0.0, 10.0, 2.0, 20.0, 4.0, 40.0])
            .expect("generic shape");
        let specific = Array3::from_shape_vec((1, 3, 2), vec![0.0, 0.0, 3.0, 5.0, 7.0, 11.0])
            .expect("specific shape");
        let prepared = PreparedInputs {
            y_codes: array![1usize],
            availability: Array2::from_elem((1, 3), true),
            offset: Array2::zeros((1, 3)),
            weights: array![2.0],
            alternative_generic: generic,
            alternative_specific: specific,
            class_to_block: vec![None, Some(0), Some(1)],
            non_reference_classes: vec![1, 2],
        };
        let theta = array![0.4, -0.3, 0.2, -0.1, 0.7, -0.5, 0.6, -0.4];
        let probabilities = array![[0.2, 0.3, 0.5]];

        let grad =
            gradient(&theta, x.view(), &prepared, &probabilities, 0.0, true).expect("gradient");
        assert_abs_diff_eq!(grad[0], -1.4, epsilon = 1e-12);
        assert_abs_diff_eq!(grad[1], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(grad[2], 1.2, epsilon = 1e-12);
        assert_abs_diff_eq!(grad[3], 16.0, epsilon = 1e-12);
        assert_abs_diff_eq!(grad[4], -4.2, epsilon = 1e-12);
        assert_abs_diff_eq!(grad[5], -7.0, epsilon = 1e-12);
        assert_abs_diff_eq!(grad[6], 7.0, epsilon = 1e-12);
        assert_abs_diff_eq!(grad[7], 11.0, epsilon = 1e-12);

        let hess =
            hessian(&theta, x.view(), &prepared, &probabilities, 0.0, true, None).expect("hessian");
        assert_abs_diff_eq!(hess[[0, 2]], -0.36, epsilon = 1e-12);
        assert_abs_diff_eq!(hess[[0, 3]], -4.8, epsilon = 1e-12);
        assert_abs_diff_eq!(hess[[1, 3]], 12.0, epsilon = 1e-12);
        assert_abs_diff_eq!(hess[[2, 2]], 4.88, epsilon = 1e-12);
        assert_abs_diff_eq!(hess[[2, 3]], 38.4, epsilon = 1e-12);
        assert_abs_diff_eq!(hess[[3, 3]], 312.0, epsilon = 1e-12);
        assert_abs_diff_eq!(hess[[3, 5]], -24.0, epsilon = 1e-12);
        assert_abs_diff_eq!(hess[[3, 7]], 132.0, epsilon = 1e-12);
        assert_abs_diff_eq!(hess[[4, 5]], 6.3, epsilon = 1e-12);
        assert_abs_diff_eq!(hess[[5, 6]], -10.5, epsilon = 1e-12);
        assert_abs_diff_eq!(hess[[5, 7]], -16.5, epsilon = 1e-12);
        assert_abs_diff_eq!(hess[[3, 0]], hess[[0, 3]], epsilon = 1e-12);
        assert_abs_diff_eq!(hess[[7, 3]], hess[[3, 7]], epsilon = 1e-12);
    }

    #[test]
    fn smooth_edf_decreases_as_lambda_increases() {
        let layout = MultinomialParameterLayout::new(3, 2, 0, 0, true).expect("valid layout");
        let penalty_matrix = array![[1.0, 0.0], [0.0, 1.0]];
        let low_lambda = MultinomialSmoothPenalty {
            col_start: 1,
            col_end: 3,
            penalty: penalty_matrix.clone(),
            lambda: 0.1,
        };
        let high_lambda = MultinomialSmoothPenalty {
            col_start: 1,
            col_end: 3,
            penalty: penalty_matrix,
            lambda: 100.0,
        };
        let mut hessian_unpenalized = Array2::zeros((3, 3));
        for idx in 0..3 {
            hessian_unpenalized[[idx, idx]] = 10.0;
        }
        let mut hessian_low = hessian_unpenalized.clone();
        add_quadratic_hessian(
            &mut hessian_low,
            &expand_smooth_penalty(&layout, &low_lambda).expect("low penalty"),
        );
        let mut hessian_high = hessian_unpenalized.clone();
        add_quadratic_hessian(
            &mut hessian_high,
            &expand_smooth_penalty(&layout, &high_lambda).expect("high penalty"),
        );

        let (edf_low, total_low) =
            smooth_edf_diagnostics(&hessian_unpenalized, &hessian_low, &layout, &[low_lambda])
                .expect("low edf");
        let (edf_high, total_high) =
            smooth_edf_diagnostics(&hessian_unpenalized, &hessian_high, &layout, &[high_lambda])
                .expect("high edf");

        assert!(edf_high[0] < edf_low[0]);
        assert!(total_high < total_low);
    }

    #[test]
    fn smooth_edf_matches_hand_computed_trace_and_limits() {
        // K=3 (n_non_reference=2) so the per-block EDF summation loop is exercised.
        // Shared design p=3: col 0 is an unpenalized intercept-like column, cols 1..3
        // are the smooth basis penalized by S = I_2. With H_unpenalized = I the
        // influence matrix (H+P)^-1 H is diagonal and the EDF is closed-form:
        //   total_edf  = 2 (unpenalized cols) + 4/(1+lambda)
        //   smooth_edf = 4/(1+lambda)   (two basis cols across both class blocks)
        let layout = MultinomialParameterLayout::new(3, 3, 0, 0, true).expect("valid layout");
        let s = array![[1.0, 0.0], [0.0, 1.0]];
        let q = layout.len().expect("len");
        let mut hessian_unpenalized = Array2::zeros((q, q));
        for idx in 0..q {
            hessian_unpenalized[[idx, idx]] = 1.0;
        }

        let edf_for = |lambda: f64| -> (f64, f64) {
            let smooth = MultinomialSmoothPenalty {
                col_start: 1,
                col_end: 3,
                penalty: s.clone(),
                lambda,
            };
            let mut hessian_penalized = hessian_unpenalized.clone();
            add_quadratic_hessian(
                &mut hessian_penalized,
                &expand_smooth_penalty(&layout, &smooth).expect("penalty"),
            );
            let (smooth_edfs, total_edf) = smooth_edf_diagnostics(
                &hessian_unpenalized,
                &hessian_penalized,
                &layout,
                std::slice::from_ref(&smooth),
            )
            .expect("edf");
            (smooth_edfs[0], total_edf)
        };

        // lambda = 1 => closed-form total = 4.0, smooth = 2.0.
        let (smooth_one, total_one) = edf_for(1.0);
        assert_abs_diff_eq!(total_one, 4.0, epsilon = 1e-9);
        assert_abs_diff_eq!(smooth_one, 2.0, epsilon = 1e-9);
        assert!(
            smooth_edf_diagnostics(&Array2::eye(q - 1), &hessian_unpenalized, &layout, &[])
                .is_err()
        );
        assert!(smooth_edf_diagnostics(
            &Array2::zeros((q, q - 1)),
            &hessian_unpenalized,
            &layout,
            &[]
        )
        .is_err());
        assert!(smooth_edf_diagnostics(
            &hessian_unpenalized,
            &Array2::zeros((q, q - 1)),
            &layout,
            &[]
        )
        .is_err());
        assert!(smooth_edf_diagnostics(
            &hessian_unpenalized,
            &Array2::zeros((q - 1, q)),
            &layout,
            &[]
        )
        .is_err());

        // lambda -> 0: smooth EDF -> (2 basis cols) * (K-1 = 2) = 4, total -> q = 6.
        let (smooth_small, total_small) = edf_for(1e-8);
        assert_abs_diff_eq!(smooth_small, 4.0, epsilon = 1e-6);
        assert_abs_diff_eq!(total_small, 6.0, epsilon = 1e-6);

        // lambda -> large: smooth EDF -> 0, total -> 2 unpenalized columns.
        let (smooth_large, total_large) = edf_for(1e8);
        assert!(smooth_large < 1e-6);
        assert_abs_diff_eq!(total_large, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn smooth_penalty_rejects_non_symmetric_matrix() {
        let layout = MultinomialParameterLayout::new(3, 2, 0, 0, true).expect("valid layout");
        let penalty = MultinomialSmoothPenalty {
            col_start: 1,
            col_end: 3,
            penalty: array![[1.0, 2.0], [0.0, 1.0]],
            lambda: 1.0,
        };

        let err = expand_smooth_penalty(&layout, &penalty)
            .expect_err("non-symmetric smooth penalty should be rejected");
        assert!(err.to_string().contains("symmetric"));
    }

    #[test]
    fn smooth_edf_degrades_on_singular_penalized_hessian() {
        // A rank-deficient penalized Hessian must not abort the fit: EDF is a
        // diagnostics-only quantity, so smooth_edf_diagnostics returns finite
        // pseudo-inverse values (mirroring the covariance degrade-and-warn path)
        // instead of propagating a LinearAlgebraError.
        let layout = MultinomialParameterLayout::new(2, 3, 0, 0, true).expect("valid layout");
        let q = layout.len().expect("len");
        let mut hessian_unpenalized = Array2::zeros((q, q));
        for idx in 0..q {
            hessian_unpenalized[[idx, idx]] = 1.0;
        }
        // Zero a diagonal entry so both Hessians are singular and the (empty)
        // penalty does not cover the null space -- Cholesky and LU both fail.
        hessian_unpenalized[[1, 1]] = 0.0;
        let hessian_penalized = hessian_unpenalized.clone();
        let smooth = MultinomialSmoothPenalty {
            col_start: 1,
            col_end: 2,
            penalty: array![[0.0]],
            lambda: 0.0,
        };

        let (smooth_edfs, total_edf) = smooth_edf_diagnostics(
            &hessian_unpenalized,
            &hessian_penalized,
            &layout,
            std::slice::from_ref(&smooth),
        )
        .expect("singular penalized Hessian must degrade to finite EDF, not error");
        assert!(total_edf.is_finite());
        assert!(smooth_edfs.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn proximal_newton_nonconvergence_degrades_with_warning() {
        // A positive-definite but severely ill-conditioned Hessian makes the inner
        // Gauss-Seidel coordinate descent converge far slower than the iteration cap,
        // so it exits unconverged while beta stays finite. The solver must
        // degrade-and-warn -- return a best-effort proposal plus a warning -- rather
        // than abort the whole fit, matching coordinate_descent.rs and this solver's
        // other degrade paths (covariance, step-halving).
        let theta = array![0.0, 0.0];
        let gradient = array![1.0, -1.0];
        let hessian = array![[1.0, 0.999], [0.999, 1.0]];
        let l1_penalty = L1Penalty {
            alpha: 0.01,
            mask: vec![true, true],
        };
        let bounds = BoundConstraints::default();
        let mut warnings = Vec::new();

        let proposal = solve_proximal_newton_subproblem(
            &theta,
            &gradient,
            &hessian,
            &l1_penalty,
            &bounds,
            &mut warnings,
        )
        .expect("inner non-convergence must degrade, not error");

        assert!(proposal.target.iter().all(|value| value.is_finite()));
        assert!(warnings.iter().any(|warning| warning.contains("did not")));
    }

    #[test]
    fn smooth_fit_with_availability_reports_finite_edf() {
        let x = array![
            [1.0, -1.3, 0.5],
            [1.0, -0.9, -0.2],
            [1.0, -0.4, 0.8],
            [1.0, 0.0, -0.5],
            [1.0, 0.3, 0.2],
            [1.0, 0.7, -0.7],
            [1.0, 1.0, 0.4],
            [1.0, 1.4, -0.1],
            [1.0, 1.8, 0.9],
        ];
        let y = array![0usize, 0, 1, 1, 2, 2, 0, 1, 2];
        let availability = array![
            [true, true, false],
            [true, true, true],
            [true, true, true],
            [true, true, false],
            [true, false, true],
            [true, true, true],
            [true, false, true],
            [true, true, true],
            [true, true, true],
        ];
        let config = MultinomialConfig {
            smooth_penalties: vec![MultinomialSmoothPenalty {
                col_start: 1,
                col_end: 3,
                penalty: array![[1.0, -0.25], [-0.25, 1.0]],
                lambda: 0.8,
            }],
            ..default_config()
        };

        let result = fit_multinomial(
            &y,
            x.view(),
            3,
            0,
            &config,
            Some(&availability),
            None,
            None,
            None,
        )
        .expect("smooth fit with availability succeeds");

        assert!(result.converged);
        assert_eq!(result.smooth_edfs.len(), 1);
        assert!(result.smooth_edfs[0].is_finite());
        assert!(result.smooth_edfs[0] > 0.0);
        assert!(result.total_edf.is_some_and(f64::is_finite));
        assert_abs_diff_eq!(result.fitted_probabilities[[0, 2]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(result.fitted_probabilities[[3, 2]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(result.fitted_probabilities[[4, 1]], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn validation_rejects_invalid_class_code() {
        let x = Array2::ones((3, 1));
        let y = array![0usize, 1, 3];
        let err = fit_multinomial(
            &y,
            x.view(),
            3,
            0,
            &default_config(),
            None,
            None,
            None,
            None,
        )
        .expect_err("invalid class should fail");
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn validation_rejects_empty_class() {
        let x = Array2::ones((3, 1));
        let y = array![0usize, 1, 1];
        let err = fit_multinomial(
            &y,
            x.view(),
            3,
            0,
            &default_config(),
            None,
            None,
            None,
            None,
        )
        .expect_err("empty class should fail");
        assert!(err.to_string().contains("no observed rows"));
    }

    #[test]
    fn validation_rejects_unavailable_observed_class() {
        let x = Array2::ones((3, 1));
        let y = array![0usize, 1, 2];
        let availability = array![[true, true, true], [true, false, true], [true, true, true],];
        let err = fit_multinomial(
            &y,
            x.view(),
            3,
            0,
            &default_config(),
            Some(&availability),
            None,
            None,
            None,
        )
        .expect_err("unavailable observed class should fail");
        assert!(err.to_string().contains("unavailable"));
    }

    #[test]
    fn validation_rejects_memory_guard_before_allocation() {
        let x = Array2::ones((4, 4));
        let y = array![0usize, 1, 2, 1];
        let config = MultinomialConfig {
            max_dense_parameters: 2,
            ..default_config()
        };
        let err = fit_multinomial(&y, x.view(), 3, 0, &config, None, None, None, None)
            .expect_err("dense guard should fail");
        assert!(err.to_string().contains("max_dense_parameters"));
    }

    #[test]
    fn initial_theta_warm_start_reuses_solution() {
        let x = array![
            [1.0, -1.2],
            [1.0, -0.4],
            [1.0, 0.1],
            [1.0, 0.3],
            [1.0, 0.9],
            [1.0, 1.4],
            [1.0, -0.8],
            [1.0, 0.6],
            [1.0, 1.1],
        ];
        let y = array![0usize, 0, 1, 1, 2, 2, 0, 1, 2];
        let cold_config = MultinomialConfig {
            alpha: 0.25,
            ..default_config()
        };
        let cold = fit_multinomial(&y, x.view(), 3, 0, &cold_config, None, None, None, None)
            .expect("cold fit succeeds");
        assert!(cold.converged);

        let warm_config = MultinomialConfig {
            alpha: 0.25,
            initial_theta: Some(Array1::from_iter(cold.coefficients.iter().copied())),
            ..default_config()
        };
        let warm = fit_multinomial(&y, x.view(), 3, 0, &warm_config, None, None, None, None)
            .expect("warm fit succeeds");

        assert!(warm.converged);
        assert_eq!(warm.iterations, 0);
        assert_abs_diff_eq!(warm.coefficients, cold.coefficients, epsilon = 1e-8);
    }

    #[test]
    fn masked_softmax_respects_availability() {
        let logits = array![[1.0, 2.0, 100.0], [0.0, -1.0, 1.0]];
        let availability = array![[true, true, false], [false, true, true]];
        let probabilities = masked_softmax(&logits, &availability).expect("softmax");

        assert_eq!(probabilities[[0, 2]], 0.0);
        assert_eq!(probabilities[[1, 0]], 0.0);
        for row in 0..2 {
            assert_abs_diff_eq!(probabilities.row(row).sum(), 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn intercept_only_recovers_weighted_class_shares() {
        let x = Array2::ones((6, 1));
        let y = array![0usize, 0, 1, 1, 1, 2];
        let result = fit_multinomial(
            &y,
            x.view(),
            3,
            0,
            &default_config(),
            None,
            None,
            None,
            None,
        )
        .expect("fit");
        let mean_prob = result
            .fitted_probabilities
            .mean_axis(ndarray::Axis(0))
            .expect("means");
        assert_abs_diff_eq!(mean_prob[0], 2.0 / 6.0, epsilon = 1e-8);
        assert_abs_diff_eq!(mean_prob[1], 3.0 / 6.0, epsilon = 1e-8);
        assert_abs_diff_eq!(mean_prob[2], 1.0 / 6.0, epsilon = 1e-8);
        assert!(result.converged);
    }

    #[test]
    fn availability_changes_fitted_probabilities() {
        let x = Array2::ones((6, 1));
        let y = array![0usize, 0, 1, 1, 1, 2];
        let all_available = fit_multinomial(
            &y,
            x.view(),
            3,
            0,
            &default_config(),
            None,
            None,
            None,
            None,
        )
        .expect("all available");
        let availability = array![
            [true, true, true],
            [true, true, true],
            [true, true, true],
            [true, true, true],
            [true, true, false],
            [true, true, true],
        ];
        let masked = fit_multinomial(
            &y,
            x.view(),
            3,
            0,
            &default_config(),
            Some(&availability),
            None,
            None,
            None,
        )
        .expect("masked");
        let diff = (&all_available.fitted_probabilities - &masked.fitted_probabilities)
            .mapv(f64::abs)
            .sum();
        assert!(diff > 1e-4);
        assert_eq!(masked.fitted_probabilities[[4, 2]], 0.0);
    }

    #[test]
    fn hessian_matches_objective_finite_difference() {
        let x = array![[1.0, -0.5], [1.0, 0.2], [1.0, 1.0], [1.0, 1.5]];
        let y = array![0usize, 1, 2, 1];
        let config = default_config();
        let prepared =
            validate_and_prepare(&y, x.view(), 3, 0, &config, None, None, None, None, None)
                .expect("prepare");
        let theta = array![0.2, -0.3, -0.1, 0.4];
        let eval = evaluate(&theta, x.view(), 3, &prepared, 0.0, true, None, &[]).expect("eval");
        let eps = 1e-5;

        for idx in 0..theta.len() {
            let mut plus = theta.clone();
            let mut minus = theta.clone();
            plus[idx] += eps;
            minus[idx] -= eps;
            let obj_plus =
                objective_only(&plus, x.view(), 3, &prepared, 0.0, true).expect("obj plus");
            let obj_minus =
                objective_only(&minus, x.view(), 3, &prepared, 0.0, true).expect("obj minus");
            let fd_grad = (obj_plus - obj_minus) / (2.0 * eps);
            assert_abs_diff_eq!(eval.gradient[idx], fd_grad, epsilon = 1e-5);
        }

        for row in 0..theta.len() {
            for col in 0..theta.len() {
                let mut plus = theta.clone();
                let mut minus = theta.clone();
                plus[col] += eps;
                minus[col] -= eps;
                let grad_plus =
                    evaluate(&plus, x.view(), 3, &prepared, 0.0, true, None, &[]).expect("plus");
                let grad_minus =
                    evaluate(&minus, x.view(), 3, &prepared, 0.0, true, None, &[]).expect("minus");
                let fd_hessian = (grad_plus.gradient[row] - grad_minus.gradient[row]) / (2.0 * eps);
                assert_abs_diff_eq!(eval.hessian[[row, col]], fd_hessian, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn alternative_hessian_matches_objective_finite_difference() {
        let x = array![[1.0, -0.5], [1.0, 0.2], [1.0, 1.0], [1.0, 1.5]];
        let y = array![0usize, 1, 2, 1];
        let alternative_generic = array![
            [[0.0], [1.2], [1.8]],
            [[0.0], [0.9], [1.3]],
            [[0.0], [1.6], [1.0]],
            [[0.0], [1.1], [2.1]],
        ];
        let alternative_specific = array![
            [[0.0], [0.4], [0.8]],
            [[0.0], [0.3], [0.7]],
            [[0.0], [0.6], [0.5]],
            [[0.0], [0.2], [0.9]],
        ];
        let config = default_config();
        let prepared = validate_and_prepare(
            &y,
            x.view(),
            3,
            0,
            &config,
            None,
            None,
            None,
            Some(alternative_generic.view()),
            Some(alternative_specific.view()),
        )
        .expect("prepare");
        let theta = array![0.2, -0.3, -0.1, 0.4, -0.5, 0.25, -0.15];
        let eval = evaluate(&theta, x.view(), 3, &prepared, 0.0, true, None, &[]).expect("eval");
        let eps = 1e-5;

        for idx in 0..theta.len() {
            let mut plus = theta.clone();
            let mut minus = theta.clone();
            plus[idx] += eps;
            minus[idx] -= eps;
            let obj_plus =
                objective_only(&plus, x.view(), 3, &prepared, 0.0, true).expect("obj plus");
            let obj_minus =
                objective_only(&minus, x.view(), 3, &prepared, 0.0, true).expect("obj minus");
            let fd_grad = (obj_plus - obj_minus) / (2.0 * eps);
            assert_abs_diff_eq!(eval.gradient[idx], fd_grad, epsilon = 1e-5);
        }

        for row in 0..theta.len() {
            for col in 0..theta.len() {
                let mut plus = theta.clone();
                let mut minus = theta.clone();
                plus[col] += eps;
                minus[col] -= eps;
                let grad_plus =
                    evaluate(&plus, x.view(), 3, &prepared, 0.0, true, None, &[]).expect("plus");
                let grad_minus =
                    evaluate(&minus, x.view(), 3, &prepared, 0.0, true, None, &[]).expect("minus");
                let fd_hessian = (grad_plus.gradient[row] - grad_minus.gradient[row]) / (2.0 * eps);
                assert_abs_diff_eq!(eval.hessian[[row, col]], fd_hessian, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn covariance_failure_degrades_to_none_with_warning() {
        // Two identical columns make the design rank-deficient. With equal class
        // weights the intercept warm-start already sits at the optimum, so the
        // solver converges at iteration 0, but the singular Hessian cannot be
        // inverted for the covariance. The fit must still succeed, returning no
        // covariance plus a warning, rather than propagating an error.
        let x = array![
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ];
        let y = array![0usize, 0, 1, 1, 2, 2];
        let config = MultinomialConfig {
            skip_covariance: false,
            ..Default::default()
        };
        let result = fit_multinomial(&y, x.view(), 3, 0, &config, None, None, None, None)
            .expect("fit should degrade gracefully rather than error");
        assert!(result.converged);
        assert!(result.covariance_unscaled.is_none());
        assert!(result
            .warnings
            .iter()
            .any(|message| message.contains("could not invert")));
    }

    #[test]
    fn ridge_shrinks_non_intercept_coefficients() {
        let x = array![
            [1.0, -2.0],
            [1.0, -1.0],
            [1.0, -0.5],
            [1.0, 0.5],
            [1.0, 1.0],
            [1.0, 2.0],
        ];
        let y = array![0usize, 0, 1, 1, 2, 2];
        let unpenalized = fit_multinomial(
            &y,
            x.view(),
            3,
            0,
            &default_config(),
            None,
            None,
            None,
            None,
        )
        .expect("unpenalized");
        let ridge_config = MultinomialConfig {
            alpha: 10.0,
            ..default_config()
        };
        let ridge = fit_multinomial(&y, x.view(), 3, 0, &ridge_config, None, None, None, None)
            .expect("ridge");
        let unpenalized_slope_norm = unpenalized
            .coefficients
            .column(1)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let ridge_slope_norm = ridge
            .coefficients
            .column(1)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!(ridge_slope_norm < unpenalized_slope_norm);
    }

    #[test]
    fn lasso_large_alpha_zeroes_penalized_coefficients_not_intercepts() {
        let x = array![
            [1.0, -2.0],
            [1.0, -1.4],
            [1.0, -0.8],
            [1.0, -0.2],
            [1.0, 0.3],
            [1.0, 0.7],
            [1.0, 1.1],
            [1.0, 1.6],
            [1.0, 2.2],
        ];
        let y = array![0usize, 0, 0, 0, 1, 1, 1, 2, 2];
        let config = MultinomialConfig {
            alpha: 1_000.0,
            l1_ratio: 1.0,
            ..default_config()
        };

        let result =
            fit_multinomial(&y, x.view(), 3, 0, &config, None, None, None, None).expect("lasso");

        assert!(result.converged);
        assert_abs_diff_eq!(result.coefficients[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result.coefficients[[1, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(
            result.coefficients[[0, 0]],
            (3.0_f64 / 4.0).ln(),
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            result.coefficients[[1, 0]],
            (2.0_f64 / 4.0).ln(),
            epsilon = 1e-6
        );
    }

    #[test]
    fn lasso_fit_with_requested_covariance_warns_standard_errors_unavailable() {
        let x = array![
            [1.0, -2.0],
            [1.0, -1.4],
            [1.0, -0.8],
            [1.0, -0.2],
            [1.0, 0.3],
            [1.0, 0.7],
            [1.0, 1.1],
            [1.0, 1.6],
            [1.0, 2.2],
        ];
        let y = array![0usize, 0, 0, 0, 1, 1, 1, 2, 2];
        let config = MultinomialConfig {
            alpha: 1_000.0,
            l1_ratio: 1.0,
            skip_covariance: false,
            ..default_config()
        };

        let result =
            fit_multinomial(&y, x.view(), 3, 0, &config, None, None, None, None).expect("lasso");

        assert!(result.converged);
        assert!(result.covariance_unscaled.is_none());
        assert!(result.warnings.iter().any(|warning| {
            warning.contains("standard errors are unavailable for multinomial lasso")
        }));
    }

    #[test]
    fn lasso_solution_satisfies_kkt_conditions() {
        let x = array![
            [1.0, -1.8, 0.2],
            [1.0, -1.2, -0.3],
            [1.0, -0.6, 0.5],
            [1.0, -0.1, -0.1],
            [1.0, 0.4, 0.4],
            [1.0, 0.8, -0.5],
            [1.0, 1.3, 0.1],
            [1.0, 1.9, -0.2],
            [1.0, 2.4, 0.3],
        ];
        let y = array![0usize, 0, 1, 0, 1, 1, 2, 2, 2];
        let config = MultinomialConfig {
            alpha: 0.25,
            l1_ratio: 1.0,
            standardize: false,
            max_iterations: 200,
            ..default_config()
        };
        let result =
            fit_multinomial(&y, x.view(), 3, 0, &config, None, None, None, None).expect("lasso");
        assert!(result.converged);

        let prepared =
            validate_and_prepare(&y, x.view(), 3, 0, &config, None, None, None, None, None)
                .expect("prepare");
        let theta = Array1::from_iter(result.coefficients.iter().copied());
        let evaluation =
            evaluate(&theta, x.view(), 3, &prepared, 0.0, true, None, &[]).expect("eval");
        let layout = MultinomialParameterLayout::new(3, 3, 0, 0, true).expect("layout");
        let l1_penalty = L1Penalty {
            alpha: config.alpha,
            mask: default_l1_penalty_mask(&layout).expect("mask"),
        };
        let violation =
            check_l1_kkt(&theta, &evaluation.gradient, &l1_penalty).expect("kkt violation");

        assert!(violation <= 1e-5, "KKT violation was {violation}");
    }

    #[test]
    fn bound_constrained_fit_hits_boundary_when_data_fights_constraint() {
        let x = array![
            [1.0, -2.0],
            [1.0, -1.5],
            [1.0, -1.0],
            [1.0, 1.0],
            [1.0, 1.5],
            [1.0, 2.0],
        ];
        let y = array![0usize, 0, 0, 1, 1, 1];
        let config = MultinomialConfig {
            nonpos_indices: vec![1],
            max_iterations: 50,
            ..default_config()
        };

        let result =
            fit_multinomial(&y, x.view(), 2, 0, &config, None, None, None, None).expect("fit");

        assert!(result.converged);
        assert_abs_diff_eq!(result.coefficients[[0, 1]], 0.0, epsilon = 1e-10);

        let prepared =
            validate_and_prepare(&y, x.view(), 2, 0, &config, None, None, None, None, None)
                .expect("prepare");
        let theta = Array1::from_iter(result.coefficients.iter().copied());
        let evaluation =
            evaluate(&theta, x.view(), 2, &prepared, 0.0, true, None, &[]).expect("eval");
        let violation = check_kkt(
            &theta,
            &evaluation.gradient,
            &L1Penalty::inert(theta.len()),
            &BoundConstraints {
                nonneg_indices: Vec::new(),
                nonpos_indices: vec![1],
            },
        )
        .expect("kkt");
        assert!(violation <= 1e-8, "KKT violation was {violation}");
    }

    #[test]
    fn bound_constraints_allow_interior_solution_and_combine_with_penalties() {
        let x = array![
            [1.0, -2.0],
            [1.0, -1.5],
            [1.0, -1.0],
            [1.0, -0.5],
            [1.0, 0.5],
            [1.0, 1.0],
            [1.0, 1.5],
            [1.0, 2.0],
        ];
        let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];
        let config = MultinomialConfig {
            nonneg_indices: vec![1],
            max_iterations: 50,
            ..default_config()
        };

        let result =
            fit_multinomial(&y, x.view(), 2, 0, &config, None, None, None, None).expect("fit");
        assert!(result.converged);
        assert!(result.coefficients[[0, 1]] > 0.0);

        for (alpha, l1_ratio) in [(1.0, 0.0), (0.5, 1.0)] {
            let penalized_config = MultinomialConfig {
                alpha,
                l1_ratio,
                nonneg_indices: vec![1],
                max_iterations: 100,
                ..default_config()
            };
            let penalized = fit_multinomial(
                &y,
                x.view(),
                2,
                0,
                &penalized_config,
                None,
                None,
                None,
                None,
            )
            .expect("penalized");
            assert!(penalized.converged);
            assert!(penalized.coefficients[[0, 1]] >= -1e-10);
        }
    }

    #[test]
    fn elastic_net_shrinks_and_keeps_valid_probabilities() {
        let x = array![
            [1.0, -2.0],
            [1.0, -1.0],
            [1.0, -0.5],
            [1.0, 0.5],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 2.5],
            [1.0, -2.5],
        ];
        let y = array![0usize, 0, 1, 1, 2, 2, 2, 0];
        let en_config = MultinomialConfig {
            alpha: 1.0,
            l1_ratio: 0.5,
            ..default_config()
        };
        let unpenalized = fit_multinomial(
            &y,
            x.view(),
            3,
            0,
            &default_config(),
            None,
            None,
            None,
            None,
        )
        .expect("unpenalized");
        let elastic_net = fit_multinomial(&y, x.view(), 3, 0, &en_config, None, None, None, None)
            .expect("elastic net");
        let unpenalized_slope_norm = unpenalized
            .coefficients
            .column(1)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let en_slope_norm = elastic_net
            .coefficients
            .column(1)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();

        assert!(elastic_net.converged);
        assert!(en_slope_norm < unpenalized_slope_norm);
        for row in 0..elastic_net.fitted_probabilities.nrows() {
            assert_abs_diff_eq!(
                elastic_net.fitted_probabilities.row(row).sum(),
                1.0,
                epsilon = 1e-12
            );
        }
    }
}
