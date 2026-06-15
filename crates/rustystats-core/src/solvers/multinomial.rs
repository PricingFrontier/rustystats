use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3};

use crate::error::{Result, RustyStatsError};
use crate::regularization::Standardization;
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

#[derive(Debug)]
struct Evaluation {
    objective: f64,
    gradient: Array1<f64>,
    hessian: Array2<f64>,
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

    let mut theta = initialize_coefficients(x_fit.ncols(), n_classes, &prepared, config);
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
            config.alpha,
            config.fit_intercept,
            sparse_cache.as_ref(),
        )?;
        let current_objective = evaluation.objective;

        let max_gradient = evaluation
            .gradient
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        if max_gradient <= config.tolerance {
            converged = true;
            solver_status = "converged".to_string();
            iterations = iter;
            break;
        }

        let step = solve_newton_step(&evaluation.hessian, &evaluation.gradient)?;
        let step_norm = step.iter().map(|v| v * v).sum::<f64>().sqrt();
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
                .zip(step.iter())
                .map(|(coef, delta)| coef - step_fraction * delta)
                .collect::<Array1<f64>>();
            let candidate_objective = objective_only(
                &candidate,
                x_fit,
                n_classes,
                &prepared,
                config.alpha,
                config.fit_intercept,
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
            converged = true;
            solver_status = "converged".to_string();
            break;
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
        config.alpha,
        config.fit_intercept,
        sparse_cache.as_ref(),
    )?;
    let covariance_work = if config.skip_covariance {
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
        x_fit.ncols(),
        n_classes,
        n_alt_generic,
        n_alt_specific,
        config.fit_intercept,
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
    if config.alpha > 0.0 && config.l1_ratio != 0.0 {
        return Err(RustyStatsError::InvalidValue(
            "multinomial Phase 1 supports ridge only; use l1_ratio=0.0".to_string(),
        ));
    }
    let alternative_generic_matrix =
        validate_alternative_tensor(alternative_generic, n, n_classes, "alternative_generic")?;
    let alternative_specific_matrix =
        validate_alternative_tensor(alternative_specific, n, n_classes, "alternative_specific")?;

    check_dense_parameter_guard(
        p,
        n_classes,
        alternative_generic_matrix.dim().2,
        alternative_specific_matrix.dim().2,
        config,
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

fn check_dense_parameter_guard(
    p: usize,
    n_classes: usize,
    n_alt_generic: usize,
    n_alt_specific: usize,
    config: &MultinomialConfig,
) -> Result<()> {
    let q = total_parameter_count(p, n_classes, n_alt_generic, n_alt_specific)?;
    if q > config.max_dense_parameters {
        return Err(RustyStatsError::InvalidValue(format!(
            "multinomial dense Newton would estimate q={} parameters \
             (shared p={} columns x K-1={} non-reference classes, \
             {} generic alternative terms, {} class-specific alternative terms), exceeding \
             max_dense_parameters={}. Reduce design width/classes or wait for \
             the large-p solver.",
            q,
            p,
            n_classes - 1,
            n_alt_generic,
            n_alt_specific,
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
            p,
            n_classes - 1,
            n_alt_generic,
            n_alt_specific,
            limit_mb
        )));
    }
    Ok(())
}

fn shared_parameter_count(p: usize, n_classes: usize) -> Result<usize> {
    p.checked_mul(n_classes - 1).ok_or_else(|| {
        RustyStatsError::InvalidValue("multinomial shared parameter count overflowed usize".into())
    })
}

fn total_parameter_count(
    p: usize,
    n_classes: usize,
    n_alt_generic: usize,
    n_alt_specific: usize,
) -> Result<usize> {
    let k_nonref = n_classes - 1;
    let shared = shared_parameter_count(p, n_classes)?;
    let specific = n_alt_specific.checked_mul(k_nonref).ok_or_else(|| {
        RustyStatsError::InvalidValue(
            "multinomial class-specific alternative parameter count overflowed usize".into(),
        )
    })?;
    shared
        .checked_add(n_alt_generic)
        .and_then(|value| value.checked_add(specific))
        .ok_or_else(|| {
            RustyStatsError::InvalidValue(
                "multinomial total parameter count overflowed usize".into(),
            )
        })
}

fn alternative_generic_start(p: usize, n_classes: usize) -> usize {
    p * (n_classes - 1)
}

fn alternative_specific_start(p: usize, n_classes: usize, n_alt_generic: usize) -> usize {
    alternative_generic_start(p, n_classes) + n_alt_generic
}

fn specific_parameter_index(
    p: usize,
    n_classes: usize,
    n_alt_generic: usize,
    block_idx: usize,
    term_idx: usize,
    n_alt_specific: usize,
) -> usize {
    alternative_specific_start(p, n_classes, n_alt_generic) + block_idx * n_alt_specific + term_idx
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
    p: usize,
    n_classes: usize,
    prepared: &PreparedInputs,
    config: &MultinomialConfig,
) -> Array1<f64> {
    let q = total_parameter_count(
        p,
        n_classes,
        prepared.alternative_generic.dim().2,
        prepared.alternative_specific.dim().2,
    )
    .expect("validated parameter count");
    let mut theta = Array1::zeros(q);
    if !config.fit_intercept || p == 0 || !all_classes_available(&prepared.availability) {
        return theta;
    }
    if prepared.offset.iter().any(|value| *value != 0.0) {
        return theta;
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
        return theta;
    }

    for (block_idx, &class_idx) in prepared.non_reference_classes.iter().enumerate() {
        let class_total = totals[class_idx];
        if class_total > 0.0 {
            theta[block_idx * p] = (class_total / reference_total).ln();
        }
    }
    theta
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
    let objective = -log_likelihood + ridge;
    let gradient = gradient(theta, x, prepared, &probabilities, alpha, fit_intercept);
    let hessian = hessian(
        theta,
        x,
        prepared,
        &probabilities,
        alpha,
        fit_intercept,
        sparse_cache,
    )?;

    Ok(Evaluation {
        objective,
        gradient,
        hessian,
    })
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
    let shared_count = p * (n_classes - 1);
    theta
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| {
            let is_shared_intercept = idx < shared_count && p > 0 && fit_intercept && idx % p == 0;
            (!is_shared_intercept).then_some(value * value)
        })
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
) -> Array1<f64> {
    let n = x.nrows();
    let p = x.ncols();
    let mut gradient = Array1::zeros(theta.len());
    let n_classes = probabilities.ncols();
    let n_alt_generic = prepared.alternative_generic.dim().2;
    let n_alt_specific = prepared.alternative_specific.dim().2;
    let generic_start = alternative_generic_start(p, n_classes);

    for (block_idx, &class_idx) in prepared.non_reference_classes.iter().enumerate() {
        for row in 0..n {
            let observed = if prepared.y_codes[row] == class_idx {
                1.0
            } else {
                0.0
            };
            let residual = prepared.weights[row] * (probabilities[[row, class_idx]] - observed);
            for feature_idx in 0..p {
                gradient[block_idx * p + feature_idx] += x[[row, feature_idx]] * residual;
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
            let param_idx = specific_parameter_index(
                p,
                n_classes,
                n_alt_generic,
                block_idx,
                term_idx,
                n_alt_specific,
            );
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
            let is_shared_intercept = idx < generic_start && p > 0 && fit_intercept && idx % p == 0;
            if !is_shared_intercept {
                gradient[idx] += alpha * theta[idx];
            }
        }
    }

    gradient
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
    let generic_start = alternative_generic_start(p, n_classes);
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
            let shared_idx = block_idx * p + feature_idx;
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
            let shared_idx = block_l * p + feature_idx;
            for (block_m, &class_m) in prepared.non_reference_classes.iter().enumerate() {
                for term_idx in 0..n_alt_specific {
                    let specific_idx = specific_parameter_index(
                        p,
                        n_classes,
                        n_alt_generic,
                        block_m,
                        term_idx,
                        n_alt_specific,
                    );
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
                let specific_idx = specific_parameter_index(
                    p,
                    n_classes,
                    n_alt_generic,
                    block_m,
                    term_m,
                    n_alt_specific,
                );
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
            let idx_l = specific_parameter_index(
                p,
                n_classes,
                n_alt_generic,
                block_l,
                term_l,
                n_alt_specific,
            );
            for (block_m, &class_m) in prepared
                .non_reference_classes
                .iter()
                .enumerate()
                .skip(block_l)
            {
                let start_term = if block_l == block_m { term_l } else { 0 };
                for term_m in start_term..n_alt_specific {
                    let idx_m = specific_parameter_index(
                        p,
                        n_classes,
                        n_alt_generic,
                        block_m,
                        term_m,
                        n_alt_specific,
                    );
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
            let is_shared_intercept = idx < generic_start && p > 0 && fit_intercept && idx % p == 0;
            if !is_shared_intercept {
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

fn array2_to_dmatrix(values: &Array2<f64>) -> DMatrix<f64> {
    let (rows, cols) = values.dim();
    DMatrix::from_fn(rows, cols, |row, col| values[[row, col]])
}

fn unflatten_coefficients(theta: &Array1<f64>, n_classes: usize, p: usize) -> Array2<f64> {
    let mut coefficients = Array2::zeros((n_classes - 1, p));
    for block_idx in 0..(n_classes - 1) {
        for feature_idx in 0..p {
            coefficients[[block_idx, feature_idx]] = theta[block_idx * p + feature_idx];
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
    let start = alternative_generic_start(p, n_classes);
    Array1::from_iter((0..n_alt_generic).map(|term_idx| theta[start + term_idx]))
}

fn unflatten_alternative_specific_coefficients(
    theta: &Array1<f64>,
    p: usize,
    n_classes: usize,
    n_alt_generic: usize,
    n_alt_specific: usize,
) -> Array2<f64> {
    let mut coefficients = Array2::zeros((n_classes - 1, n_alt_specific));
    for block_idx in 0..(n_classes - 1) {
        for term_idx in 0..n_alt_specific {
            coefficients[[block_idx, term_idx]] = theta[specific_parameter_index(
                p,
                n_classes,
                n_alt_generic,
                block_idx,
                term_idx,
                n_alt_specific,
            )];
        }
    }
    coefficients
}

#[allow(clippy::too_many_arguments)]
fn original_parameter_transform(
    p: usize,
    n_classes: usize,
    n_alt_generic: usize,
    n_alt_specific: usize,
    fit_intercept: bool,
    shared_standardization: Option<&Standardization>,
    generic_standardization: Option<&Standardization>,
    specific_standardization: Option<&AlternativeSpecificStandardization>,
) -> Result<MultinomialTransform> {
    let n_blocks = n_classes - 1;
    let q = total_parameter_count(p, n_classes, n_alt_generic, n_alt_specific)?;
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
            let source_idx = block_idx * p + feature_idx;
            let mut row = vec![(
                source_idx,
                shared_scale_factor(shared_standardization, feature_idx, fit_intercept),
            )];
            if fit_intercept && feature_idx == 0 {
                if let Some(std) = shared_standardization {
                    for centered_feature in 1..p {
                        let factor = -std.center[centered_feature] / std.scale[centered_feature];
                        if factor != 0.0 {
                            row.push((block_idx * p + centered_feature, factor));
                        }
                    }
                }
                if let Some(std) = specific_standardization {
                    for term_idx in 0..n_alt_specific {
                        let factor =
                            -std.center[[block_idx, term_idx]] / std.scale[[block_idx, term_idx]];
                        if factor != 0.0 {
                            row.push((
                                specific_parameter_index(
                                    p,
                                    n_classes,
                                    n_alt_generic,
                                    block_idx,
                                    term_idx,
                                    n_alt_specific,
                                ),
                                factor,
                            ));
                        }
                    }
                }
            }
            rows.push(row);
        }
    }

    for term_idx in 0..n_alt_generic {
        rows.push(vec![(
            alternative_generic_start(p, n_classes) + term_idx,
            generic_standardization
                .map(|std| 1.0 / std.scale[term_idx])
                .unwrap_or(1.0),
        )]);
    }

    for block_idx in 0..n_blocks {
        for term_idx in 0..n_alt_specific {
            rows.push(vec![(
                specific_parameter_index(
                    p,
                    n_classes,
                    n_alt_generic,
                    block_idx,
                    term_idx,
                    n_alt_specific,
                ),
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
        let eval = evaluate(&theta, x.view(), 3, &prepared, 0.0, true, None).expect("eval");
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
                    evaluate(&plus, x.view(), 3, &prepared, 0.0, true, None).expect("plus");
                let grad_minus =
                    evaluate(&minus, x.view(), 3, &prepared, 0.0, true, None).expect("minus");
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
        let eval = evaluate(&theta, x.view(), 3, &prepared, 0.0, true, None).expect("eval");
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
                    evaluate(&plus, x.view(), 3, &prepared, 0.0, true, None).expect("plus");
                let grad_minus =
                    evaluate(&minus, x.view(), 3, &prepared, 0.0, true, None).expect("minus");
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
}
