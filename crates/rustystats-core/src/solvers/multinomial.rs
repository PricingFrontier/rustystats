use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2, ArrayView2};

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
    class_to_block: Vec<Option<usize>>,
    non_reference_classes: Vec<usize>,
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
    compute_null_deviance: bool,
) -> Result<MultinomialResult> {
    let prepared = validate_and_prepare(
        y_codes,
        x,
        n_classes,
        reference_index,
        config,
        availability,
        offset,
        weights,
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

    let coefficients_work = unflatten_coefficients(&theta, n_classes, x_fit.ncols());
    let (coefficients, covariance_unscaled) = if use_standardization {
        let std = standardization.expect("checked is_some");
        let coefficients = to_original_coefficient_matrix(&coefficients_work, std, config)?;
        let covariance = match covariance_work {
            Some(cov) => Some(to_original_multinomial_covariance(
                &cov,
                x_fit.ncols(),
                n_classes - 1,
                std,
                config.fit_intercept,
            )?),
            None => None,
        };
        (coefficients, covariance)
    } else {
        (coefficients_work, covariance_work)
    };

    let (linear_predictor, fitted_probabilities) = linear_predictor_and_probabilities(
        &coefficients,
        x,
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
    check_dense_parameter_guard(p, n_classes, config)?;

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
        class_to_block,
        non_reference_classes,
    })
}

fn check_dense_parameter_guard(
    p: usize,
    n_classes: usize,
    config: &MultinomialConfig,
) -> Result<()> {
    let k_nonref = n_classes - 1;
    let q = p.checked_mul(k_nonref).ok_or_else(|| {
        RustyStatsError::InvalidValue("multinomial parameter count overflowed usize".to_string())
    })?;
    if q > config.max_dense_parameters {
        return Err(RustyStatsError::InvalidValue(format!(
            "multinomial dense Newton would estimate q={} parameters \
             (p={} columns x K-1={} non-reference classes), exceeding \
             max_dense_parameters={}. Reduce design width/classes or wait for \
             the large-p solver.",
            q, p, k_nonref, config.max_dense_parameters
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
             parameters (p={} columns x K-1={} non-reference classes), exceeding \
             the configured {:.1} MB limit. Reduce high-cardinality terms, \
             remove classes, or set compute_covariance=false only if the \
             coefficient solve itself remains within the dense guard.",
            estimated_mb, q, p, k_nonref, limit_mb
        )));
    }
    Ok(())
}

fn initialize_coefficients(
    p: usize,
    n_classes: usize,
    prepared: &PreparedInputs,
    config: &MultinomialConfig,
) -> Array1<f64> {
    let q = p * (n_classes - 1);
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
    )?;
    let log_likelihood =
        log_likelihood_from_probabilities(&probabilities, &prepared.y_codes, &prepared.weights)?;
    let ridge = ridge_penalty(theta, x.ncols(), alpha, fit_intercept);
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
    )?;
    let log_likelihood =
        log_likelihood_from_probabilities(&probabilities, &prepared.y_codes, &prepared.weights)?;
    Ok(-log_likelihood + ridge_penalty(theta, x.ncols(), alpha, fit_intercept))
}

fn ridge_penalty(theta: &Array1<f64>, p: usize, alpha: f64, fit_intercept: bool) -> f64 {
    if alpha <= 0.0 {
        return 0.0;
    }
    theta
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| {
            let local_feature = idx % p;
            (!(fit_intercept && local_feature == 0)).then_some(value * value)
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
) -> Result<(Array2<f64>, Array2<f64>)> {
    let p = x.ncols();
    if theta.len() != p * (n_classes - 1) {
        return Err(RustyStatsError::dim_mismatch(
            p * (n_classes - 1),
            theta.len(),
            "theta length vs multinomial parameter count",
        ));
    }
    let coefficients = unflatten_coefficients(theta, n_classes, p);
    linear_predictor_and_probabilities(
        &coefficients,
        x,
        n_classes,
        class_to_block,
        availability,
        offset,
    )
}

fn linear_predictor_and_probabilities(
    coefficients: &Array2<f64>,
    x: ArrayView2<'_, f64>,
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

    let mut logits = Array2::zeros((n, n_classes));
    for row in 0..n {
        for class_idx in 0..n_classes {
            let mut eta = offset[[row, class_idx]];
            if let Some(block_idx) = class_to_block[class_idx] {
                let mut dot = 0.0;
                for feature_idx in 0..p {
                    dot += x[[row, feature_idx]] * coefficients[[block_idx, feature_idx]];
                }
                eta += dot;
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

    if alpha > 0.0 {
        for idx in 0..theta.len() {
            let local_feature = idx % p;
            if !(fit_intercept && local_feature == 0) {
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

    if alpha > 0.0 {
        for idx in 0..q {
            let local_feature = idx % p;
            if !(fit_intercept && local_feature == 0) {
                hessian[[idx, idx]] += alpha;
            }
        }
    }

    Ok(hessian)
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

fn to_original_coefficient_matrix(
    coefficients: &Array2<f64>,
    standardization: &Standardization,
    config: &MultinomialConfig,
) -> Result<Array2<f64>> {
    let (blocks, p) = coefficients.dim();
    let mut out = Array2::zeros((blocks, p));
    for block_idx in 0..blocks {
        let beta = coefficients.row(block_idx).to_owned();
        let original = standardization.to_original_coefficients(&beta, config.fit_intercept)?;
        for feature_idx in 0..p {
            out[[block_idx, feature_idx]] = original[feature_idx];
        }
    }
    Ok(out)
}

fn to_original_multinomial_covariance(
    covariance: &Array2<f64>,
    p: usize,
    n_blocks: usize,
    standardization: &Standardization,
    fit_intercept: bool,
) -> Result<Array2<f64>> {
    if covariance.nrows() != p * n_blocks || covariance.ncols() != p * n_blocks {
        return Err(RustyStatsError::InvalidValue(format!(
            "multinomial covariance has shape ({}, {}), expected ({}, {})",
            covariance.nrows(),
            covariance.ncols(),
            p * n_blocks,
            p * n_blocks
        )));
    }
    standardization.validate(p)?;

    let mut out = Array2::zeros(covariance.dim());
    for block_row in 0..n_blocks {
        for block_col in 0..n_blocks {
            transform_covariance_block(
                covariance,
                &mut out,
                block_row * p,
                block_col * p,
                p,
                standardization,
                fit_intercept,
            );
        }
    }
    Ok(out)
}

fn transform_covariance_block(
    source: &Array2<f64>,
    target: &mut Array2<f64>,
    row_start: usize,
    col_start: usize,
    p: usize,
    standardization: &Standardization,
    fit_intercept: bool,
) {
    if !fit_intercept || p == 0 {
        for row in 0..p {
            let row_scale = standardization.scale[row];
            for col in 0..p {
                target[[row_start + row, col_start + col]] = source
                    [[row_start + row, col_start + col]]
                    / (row_scale * standardization.scale[col]);
            }
        }
        return;
    }

    for row in 1..p {
        let row_scale = standardization.scale[row];
        for col in 1..p {
            target[[row_start + row, col_start + col]] = source[[row_start + row, col_start + col]]
                / (row_scale * standardization.scale[col]);
        }
    }

    for col in 1..p {
        let mut value = source[[row_start, col_start + col]];
        for j in 1..p {
            value -= (standardization.center[j] / standardization.scale[j])
                * source[[row_start + j, col_start + col]];
        }
        target[[row_start, col_start + col]] = value / standardization.scale[col];
    }

    for row in 1..p {
        let mut value = source[[row_start + row, col_start]];
        for j in 1..p {
            value -= (standardization.center[j] / standardization.scale[j])
                * source[[row_start + row, col_start + j]];
        }
        target[[row_start + row, col_start]] = value / standardization.scale[row];
    }

    let mut intercept = source[[row_start, col_start]];
    for j in 1..p {
        let factor_j = standardization.center[j] / standardization.scale[j];
        intercept -= factor_j * source[[row_start + j, col_start]];
        intercept -= factor_j * source[[row_start, col_start + j]];
    }
    for j in 1..p {
        let factor_j = standardization.center[j] / standardization.scale[j];
        for l in 1..p {
            let factor_l = standardization.center[l] / standardization.scale[l];
            intercept += factor_j * factor_l * source[[row_start + j, col_start + l]];
        }
    }
    target[[row_start, col_start]] = intercept;
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
            validate_and_prepare(&y, x.view(), 3, 0, &config, None, None, None).expect("prepare");
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
