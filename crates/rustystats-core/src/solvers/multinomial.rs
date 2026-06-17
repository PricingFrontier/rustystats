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
            && self.n_shared > 0
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
        for col in (row + 1)..width {
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

    let use_standardization = config.alpha > 0.0 && config.standardize && standardization.is_some();
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
        let step_norm = proposal
            .target
            .iter()
            .zip(theta.iter())
            .map(|(candidate, current)| {
                let delta = candidate - current;
                delta * delta
            })
            .sum::<f64>()
            .sqrt();
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
            let candidate = theta
                .iter()
                .zip(proposal.target.iter())
                .map(|(current, target)| current + step_fraction * (target - current))
                .collect::<Array1<f64>>();
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

            if candidate_objective.is_finite()
                && candidate_objective <= current_objective + config.tolerance * 1e-3
            {
                accepted = true;
                accepted_theta = candidate;
                accepted_objective = candidate_objective;
                break;
            }
            step_fraction *= 0.5;
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
        iterations = iter + 1;

        let coefficient_norm = theta.iter().map(|v| v * v).sum::<f64>().sqrt();
        if coefficient_norm > MAX_COEFFICIENT_NORM {
            warnings.push(
                "multinomial coefficient norm is extremely large; possible separation. \
                 Try ridge regularization."
                    .to_string(),
            );
        }

        let relative_improvement =
            (previous_objective - new_objective).abs() / (1.0 + previous_objective.abs());
        let scaled_step_norm = step_fraction * step_norm / (1.0 + coefficient_norm);
        if relative_improvement <= config.tolerance || scaled_step_norm <= config.tolerance {
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
            if next_optimality <= config.tolerance {
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

    if !converged && solver_status == "max_iterations" {
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
    let deviance = -2.0 * log_likelihood;
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
    if peak_bytes > config.hessian_memory_limit_bytes && peak_factor > 1 {
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
        if !denom.is_finite() || denom <= 0.0 {
            return Err(RustyStatsError::NumericalError(format!(
                "invalid multinomial softmax denominator on row {}",
                row
            )));
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

fn solve_newton_proposal(
    theta: &Array1<f64>,
    evaluation: &Evaluation,
    extensions: &OptimizationExtensions,
    warnings: &mut Vec<String>,
) -> Result<NewtonProposal> {
    if extensions.l1_penalty.alpha > 0.0 {
        solve_proximal_newton_subproblem(
            theta,
            &evaluation.gradient,
            &evaluation.hessian,
            &extensions.l1_penalty,
            &extensions.bound_constraints,
            warnings,
        )
    } else if !extensions.bound_constraints.is_empty() {
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
        let active_lower =
            signs[idx] > 0 && theta[idx] <= PROXIMAL_NEWTON_ZERO_TOLERANCE && gradient[idx] >= 0.0;
        let active_upper =
            signs[idx] < 0 && theta[idx] >= -PROXIMAL_NEWTON_ZERO_TOLERANCE && gradient[idx] <= 0.0;
        if !active_lower && !active_upper {
            free_indices.push(idx);
        }
    }

    if free_indices.is_empty() {
        return Ok(NewtonProposal {
            target: project_with_signs(theta.clone(), &signs),
        });
    }
    if free_indices.len() == q {
        let step = solve_newton_step(hessian, gradient)?;
        let target = theta
            .iter()
            .zip(step.iter())
            .map(|(coef, delta)| coef - delta)
            .collect::<Array1<f64>>();
        return Ok(NewtonProposal {
            target: project_with_signs(target, &signs),
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
        let indices: &[usize] = if cd_iter > 0 && cd_iter % 5 != 0 && active_indices.len() < q {
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
                    max_change = max_change.max((old - beta[idx]).abs());
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
            if updated.abs() < PROXIMAL_NEWTON_ZERO_TOLERANCE && l1_penalty.mask[idx] {
                updated = 0.0;
            }
            beta[idx] = updated;
            max_change = max_change.max((updated - old).abs());
        }

        if cd_iter == 0 || cd_iter % 5 == 0 {
            active_indices = active_l1_indices(&beta, gradient, l1_penalty);
        }
        let beta_scale = beta.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
        if max_change <= PROXIMAL_NEWTON_CD_TOLERANCE * (1.0 + beta_scale) {
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
        if !l1_penalty.mask[idx]
            || theta[idx].abs() > PROXIMAL_NEWTON_ZERO_TOLERANCE
            || l1_coordinate_violation(theta[idx], smooth_gradient[idx], l1_penalty.alpha)
                > PROXIMAL_NEWTON_ZERO_TOLERANCE
        {
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
