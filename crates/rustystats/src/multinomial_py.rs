use ndarray::{Array1, Array2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use rustystats_core::regularization::Standardization;
use rustystats_core::solvers::{
    fit_multinomial_with_alternatives, AlternativeSpecificStandardization, MultinomialConfig,
    MultinomialResult,
};

use crate::fitting_py::build_standardization;

#[pyclass(name = "MultinomialResults", skip_from_py_object)]
#[derive(Clone)]
pub struct PyMultinomialResults {
    pub(crate) coefficients: Array2<f64>,
    pub(crate) alternative_generic_coefficients: Array1<f64>,
    pub(crate) alternative_specific_coefficients: Array2<f64>,
    pub(crate) fitted_probabilities: Array2<f64>,
    pub(crate) linear_predictor: Array2<f64>,
    pub(crate) log_likelihood: f64,
    pub(crate) deviance: f64,
    pub(crate) null_deviance: f64,
    pub(crate) iterations: usize,
    pub(crate) converged: bool,
    pub(crate) covariance_unscaled: Option<Array2<f64>>,
    pub(crate) prior_weights: Array1<f64>,
    pub(crate) y_codes: Array1<i64>,
    pub(crate) reference_index: usize,
    pub(crate) warnings: Vec<String>,
    pub(crate) solver_status: String,
    pub(crate) alpha: f64,
    pub(crate) l1_ratio: f64,
    pub(crate) fit_intercept: bool,
    pub(crate) design_matrix: Option<Array2<f64>>,
}

impl From<(MultinomialResult, f64, f64, bool, Option<Array2<f64>>)> for PyMultinomialResults {
    fn from(
        (result, alpha, l1_ratio, fit_intercept, design_matrix): (
            MultinomialResult,
            f64,
            f64,
            bool,
            Option<Array2<f64>>,
        ),
    ) -> Self {
        Self {
            coefficients: result.coefficients,
            alternative_generic_coefficients: result.alternative_generic_coefficients,
            alternative_specific_coefficients: result.alternative_specific_coefficients,
            fitted_probabilities: result.fitted_probabilities,
            linear_predictor: result.linear_predictor,
            log_likelihood: result.log_likelihood,
            deviance: result.deviance,
            null_deviance: result.null_deviance,
            iterations: result.iterations,
            converged: result.converged,
            covariance_unscaled: result.covariance_unscaled,
            prior_weights: result.prior_weights,
            y_codes: Array1::from_iter(result.y_codes.iter().map(|code| *code as i64)),
            reference_index: result.reference_index,
            warnings: result.warnings,
            solver_status: result.solver_status,
            alpha,
            l1_ratio,
            fit_intercept,
            design_matrix,
        }
    }
}

#[pymethods]
impl PyMultinomialResults {
    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.coefficients.clone().into_pyarray(py)
    }

    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.coefficients.clone().into_pyarray(py)
    }

    #[getter]
    fn coef_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.coefficients.clone().into_pyarray(py)
    }

    #[getter]
    fn alternative_generic_coefficients<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.alternative_generic_coefficients
            .clone()
            .into_pyarray(py)
    }

    #[getter]
    fn alternative_specific_coefficients<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.alternative_specific_coefficients
            .clone()
            .into_pyarray(py)
    }

    #[getter]
    fn fitted_probabilities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.fitted_probabilities.clone().into_pyarray(py)
    }

    #[getter]
    fn fittedvalues<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.fitted_probabilities.clone().into_pyarray(py)
    }

    #[getter]
    fn linear_predictor<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.linear_predictor.clone().into_pyarray(py)
    }

    #[getter]
    fn cov_params_unscaled<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.covariance_unscaled
            .as_ref()
            .map(|cov| cov.clone().into_pyarray(py))
    }

    #[getter]
    fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    fn llf(&self) -> f64 {
        self.log_likelihood
    }

    #[getter]
    fn deviance(&self) -> f64 {
        self.deviance
    }

    #[getter]
    fn null_deviance(&self) -> f64 {
        self.null_deviance
    }

    #[getter]
    fn iterations(&self) -> usize {
        self.iterations
    }

    #[getter]
    fn converged(&self) -> bool {
        self.converged
    }

    #[getter]
    fn solver_status(&self) -> String {
        self.solver_status.clone()
    }

    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }

    #[getter]
    fn prior_weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.prior_weights.clone().into_pyarray(py)
    }

    #[getter]
    fn y_codes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.y_codes.clone().into_pyarray(py)
    }

    #[getter]
    fn reference_index(&self) -> usize {
        self.reference_index
    }

    #[getter]
    fn nobs(&self) -> usize {
        self.fitted_probabilities.nrows()
    }

    #[getter]
    fn n_params(&self) -> usize {
        self.coefficients.len()
            + self.alternative_generic_coefficients.len()
            + self.alternative_specific_coefficients.len()
    }

    #[getter]
    fn df_model(&self) -> usize {
        self.coefficients.len()
    }

    #[getter]
    fn df_resid(&self) -> isize {
        self.nobs() as isize - self.n_params() as isize
    }

    #[getter]
    fn alpha(&self) -> f64 {
        self.alpha
    }

    #[getter]
    fn l1_ratio(&self) -> f64 {
        self.l1_ratio
    }

    #[getter]
    fn is_regularized(&self) -> bool {
        self.alpha > 0.0
    }

    #[getter]
    fn penalty_type(&self) -> &str {
        if self.alpha <= 0.0 {
            "none"
        } else if self.l1_ratio >= 1.0 {
            "lasso"
        } else if self.l1_ratio > 0.0 {
            "elastic_net"
        } else {
            "ridge"
        }
    }

    #[getter]
    fn fit_intercept(&self) -> bool {
        self.fit_intercept
    }

    #[getter]
    fn get_design_matrix<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.design_matrix
            .as_ref()
            .map(|x| x.clone().into_pyarray(py))
    }
}

#[pyfunction]
#[pyo3(signature = (
    y_codes,
    x,
    n_classes,
    reference_index,
    availability=None,
    offset=None,
    weights=None,
    alpha=0.0,
    l1_ratio=0.0,
    max_iter=100,
    tol=1e-8,
    fit_intercept=true,
    center=None,
    scale=None,
    skip_covariance=false,
    hessian_memory_limit_bytes=268435456,
    max_dense_parameters=5000,
    store_design_matrix=false,
    verbose=false,
    alternative_generic=None,
    alternative_specific=None,
    alternative_generic_center=None,
    alternative_generic_scale=None,
    alternative_specific_center=None,
    alternative_specific_scale=None,
    initial_theta=None
))]
#[allow(clippy::too_many_arguments)]
pub fn fit_multinomial_py(
    y_codes: PyReadonlyArray1<i64>,
    x: PyReadonlyArray2<f64>,
    n_classes: usize,
    reference_index: usize,
    availability: Option<PyReadonlyArray2<bool>>,
    offset: Option<PyReadonlyArray2<f64>>,
    weights: Option<PyReadonlyArray1<f64>>,
    alpha: f64,
    l1_ratio: f64,
    max_iter: usize,
    tol: f64,
    fit_intercept: bool,
    center: Option<PyReadonlyArray1<f64>>,
    scale: Option<PyReadonlyArray1<f64>>,
    skip_covariance: bool,
    hessian_memory_limit_bytes: usize,
    max_dense_parameters: usize,
    store_design_matrix: bool,
    verbose: bool,
    alternative_generic: Option<PyReadonlyArray3<f64>>,
    alternative_specific: Option<PyReadonlyArray3<f64>>,
    alternative_generic_center: Option<PyReadonlyArray1<f64>>,
    alternative_generic_scale: Option<PyReadonlyArray1<f64>>,
    alternative_specific_center: Option<PyReadonlyArray2<f64>>,
    alternative_specific_scale: Option<PyReadonlyArray2<f64>>,
    initial_theta: Option<PyReadonlyArray1<f64>>,
) -> PyResult<PyMultinomialResults> {
    let y_codes_array = y_codes
        .as_array()
        .iter()
        .enumerate()
        .map(|(idx, code)| {
            if *code < 0 {
                Err(PyValueError::new_err(format!(
                    "y_codes[{}] must be non-negative, got {}",
                    idx, code
                )))
            } else {
                Ok(*code as usize)
            }
        })
        .collect::<PyResult<Array1<usize>>>()?;
    let x_view = x.as_array();
    let n_params = x_view.ncols();
    let availability_array = availability.map(|a| a.as_array().to_owned());
    let offset_array = offset.map(|o| o.as_array().to_owned());
    let weights_array = weights.map(|w| w.as_array().to_owned());
    let alternative_generic_array = alternative_generic.map(|a| a.as_array().to_owned());
    let alternative_specific_array = alternative_specific.map(|a| a.as_array().to_owned());
    let initial_theta_array = initial_theta.map(|theta| theta.as_array().to_owned());
    let standardization = build_standardization(center, scale, n_params)?;
    let alternative_generic_standardization = build_vector_standardization(
        alternative_generic_center,
        alternative_generic_scale,
        alternative_generic_array
            .as_ref()
            .map_or(0, |array| array.dim().2),
        "alternative_generic",
    )?;
    let alternative_specific_standardization = build_alternative_specific_standardization(
        alternative_specific_center,
        alternative_specific_scale,
        alternative_specific_array
            .as_ref()
            .map_or((n_classes.saturating_sub(1), 0), |array| {
                (n_classes.saturating_sub(1), array.dim().2)
            }),
    )?;

    let use_standardization = standardization.is_some()
        || alternative_generic_standardization.is_some()
        || alternative_specific_standardization.is_some();

    let config = MultinomialConfig {
        max_iterations: max_iter,
        tolerance: tol,
        alpha,
        l1_ratio,
        fit_intercept,
        standardize: use_standardization,
        skip_covariance,
        hessian_memory_limit_bytes,
        max_dense_parameters,
        verbose,
        initial_theta: initial_theta_array,
    };

    let result = fit_multinomial_with_alternatives(
        &y_codes_array,
        x_view,
        n_classes,
        reference_index,
        &config,
        availability_array.as_ref(),
        offset_array.as_ref(),
        weights_array.as_ref(),
        standardization.as_ref(),
        alternative_generic_array.as_ref().map(|a| a.view()),
        alternative_specific_array.as_ref().map(|a| a.view()),
        alternative_generic_standardization.as_ref(),
        alternative_specific_standardization.as_ref(),
    )
    .map_err(|err| PyValueError::new_err(format!("multinomial fitting failed: {}", err)))?;

    Ok(PyMultinomialResults::from((
        result,
        alpha,
        l1_ratio,
        fit_intercept,
        store_design_matrix.then(|| x_view.to_owned()),
    )))
}

fn build_vector_standardization(
    center: Option<PyReadonlyArray1<f64>>,
    scale: Option<PyReadonlyArray1<f64>>,
    expected_len: usize,
    name: &str,
) -> PyResult<Option<Standardization>> {
    match (center, scale) {
        (None, None) => Ok(None),
        (Some(c), Some(s)) => {
            let center_vec = c.as_array().to_vec();
            let scale_vec = s.as_array().to_vec();
            if center_vec.len() != expected_len {
                return Err(PyValueError::new_err(format!(
                    "{}_center length {} does not match expected {}",
                    name,
                    center_vec.len(),
                    expected_len
                )));
            }
            if scale_vec.len() != expected_len {
                return Err(PyValueError::new_err(format!(
                    "{}_scale length {} does not match expected {}",
                    name,
                    scale_vec.len(),
                    expected_len
                )));
            }
            Standardization::new(center_vec, scale_vec)
                .map(Some)
                .map_err(|e| {
                    PyValueError::new_err(format!("Invalid {} standardization: {}", name, e))
                })
        }
        _ => Err(PyValueError::new_err(format!(
            "{}_center and {}_scale must be provided together for standardization",
            name, name
        ))),
    }
}

fn build_alternative_specific_standardization(
    center: Option<PyReadonlyArray2<f64>>,
    scale: Option<PyReadonlyArray2<f64>>,
    expected_shape: (usize, usize),
) -> PyResult<Option<AlternativeSpecificStandardization>> {
    match (center, scale) {
        (None, None) => Ok(None),
        (Some(c), Some(s)) => {
            let center_array = c.as_array().to_owned();
            let scale_array = s.as_array().to_owned();
            if center_array.dim() != expected_shape {
                return Err(PyValueError::new_err(format!(
                    "alternative_specific_center shape {:?} does not match expected {:?}",
                    center_array.dim(),
                    expected_shape
                )));
            }
            if scale_array.dim() != expected_shape {
                return Err(PyValueError::new_err(format!(
                    "alternative_specific_scale shape {:?} does not match expected {:?}",
                    scale_array.dim(),
                    expected_shape
                )));
            }
            AlternativeSpecificStandardization::new(center_array, scale_array)
                .map(Some)
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "Invalid alternative_specific standardization: {}",
                        e
                    ))
                })
        }
        _ => Err(PyValueError::new_err(
            "alternative_specific_center and alternative_specific_scale must be provided together for standardization",
        )),
    }
}
