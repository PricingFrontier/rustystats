"""Type stubs for the rustystats native Rust extension module."""

import numpy as np
import numpy.typing as npt

# =============================================================================
# Link Functions
# =============================================================================

class IdentityLink:
    def __init__(self) -> None: ...
    def name(self) -> str: ...
    def link(self, mu: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def inverse(self, eta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def derivative(self, mu: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

class LogLink:
    def __init__(self) -> None: ...
    def name(self) -> str: ...
    def link(self, mu: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def inverse(self, eta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def derivative(self, mu: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

class LogitLink:
    def __init__(self) -> None: ...
    def name(self) -> str: ...
    def link(self, mu: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def inverse(self, eta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def derivative(self, mu: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

# =============================================================================
# Family Classes
# =============================================================================

class GaussianFamily:
    def __init__(self) -> None: ...
    def name(self) -> str: ...
    def variance(self, mu: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def unit_deviance(
        self, y: npt.NDArray[np.float64], mu: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def deviance(self, y: npt.NDArray[np.float64], mu: npt.NDArray[np.float64]) -> float: ...
    def default_link(self) -> IdentityLink: ...

class PoissonFamily:
    def __init__(self) -> None: ...
    def name(self) -> str: ...
    def variance(self, mu: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def unit_deviance(
        self, y: npt.NDArray[np.float64], mu: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def deviance(self, y: npt.NDArray[np.float64], mu: npt.NDArray[np.float64]) -> float: ...
    def default_link(self) -> LogLink: ...

class BinomialFamily:
    def __init__(self) -> None: ...
    def name(self) -> str: ...
    def variance(self, mu: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def unit_deviance(
        self, y: npt.NDArray[np.float64], mu: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def deviance(self, y: npt.NDArray[np.float64], mu: npt.NDArray[np.float64]) -> float: ...
    def default_link(self) -> LogitLink: ...

class GammaFamily:
    def __init__(self) -> None: ...
    def name(self) -> str: ...
    def variance(self, mu: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def unit_deviance(
        self, y: npt.NDArray[np.float64], mu: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def deviance(self, y: npt.NDArray[np.float64], mu: npt.NDArray[np.float64]) -> float: ...
    def default_link(self) -> LogLink: ...

class QuasiPoissonFamily:
    def __init__(self) -> None: ...
    def name(self) -> str: ...
    def variance(self, mu: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def unit_deviance(
        self, y: npt.NDArray[np.float64], mu: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def deviance(self, y: npt.NDArray[np.float64], mu: npt.NDArray[np.float64]) -> float: ...
    def default_link(self) -> LogLink: ...

class QuasiBinomialFamily:
    def __init__(self) -> None: ...
    def name(self) -> str: ...
    def variance(self, mu: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def unit_deviance(
        self, y: npt.NDArray[np.float64], mu: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def deviance(self, y: npt.NDArray[np.float64], mu: npt.NDArray[np.float64]) -> float: ...
    def default_link(self) -> LogitLink: ...

class TweedieFamily:
    def __init__(self, var_power: float = 1.5, allow_extended_tweedie: bool = False) -> None: ...
    def name(self) -> str: ...
    @property
    def var_power(self) -> float: ...
    def variance(self, mu: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def unit_deviance(
        self, y: npt.NDArray[np.float64], mu: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def deviance(self, y: npt.NDArray[np.float64], mu: npt.NDArray[np.float64]) -> float: ...
    def default_link(self) -> LogLink: ...

class NegativeBinomialFamily:
    def __init__(self, theta: float = 1.0) -> None: ...
    def name(self) -> str: ...
    @property
    def theta(self) -> float: ...
    @property
    def alpha(self) -> float: ...
    def variance(self, mu: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def unit_deviance(
        self, y: npt.NDArray[np.float64], mu: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def deviance(self, y: npt.NDArray[np.float64], mu: npt.NDArray[np.float64]) -> float: ...
    def default_link(self) -> LogLink: ...

# =============================================================================
# GLM Results
# =============================================================================

class GLMResults:
    @property
    def params(self) -> npt.NDArray[np.float64]: ...
    @property
    def coefficients(self) -> npt.NDArray[np.float64]: ...
    @property
    def fittedvalues(self) -> npt.NDArray[np.float64]: ...
    @property
    def linear_predictor(self) -> npt.NDArray[np.float64]: ...
    @property
    def deviance(self) -> float: ...
    @property
    def iterations(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    @property
    def solver_status(self) -> str: ...
    @property
    def step_halving_used(self) -> bool: ...
    @property
    def warnings(self) -> list[str]: ...
    @property
    def nobs(self) -> int: ...
    @property
    def df_resid(self) -> int: ...
    @property
    def df_model(self) -> int: ...
    @property
    def cov_params_unscaled(self) -> npt.NDArray[np.float64]: ...
    @property
    def get_design_matrix(self) -> npt.NDArray[np.float64] | None: ...
    @property
    def get_irls_weights(self) -> npt.NDArray[np.float64]: ...
    @property
    def family(self) -> str: ...
    @property
    def alpha(self) -> float: ...
    @property
    def l1_ratio(self) -> float | None: ...
    @property
    def penalty_type(self) -> str: ...
    @property
    def is_regularized(self) -> bool: ...
    def scale(self) -> float: ...
    def bse(self) -> npt.NDArray[np.float64]: ...
    def tvalues(self) -> npt.NDArray[np.float64]: ...
    def pvalues(self) -> npt.NDArray[np.float64]: ...
    def conf_int(self, alpha: float = 0.05) -> npt.NDArray[np.float64]: ...
    def significance_codes(self) -> list[str]: ...
    def cov_robust(self, hc_type: str = "HC1") -> npt.NDArray[np.float64]: ...
    def bse_robust(self, hc_type: str = "HC1") -> npt.NDArray[np.float64]: ...
    def tvalues_robust(self, hc_type: str = "HC1") -> npt.NDArray[np.float64]: ...
    def pvalues_robust(self, hc_type: str = "HC1") -> npt.NDArray[np.float64]: ...
    def conf_int_robust(
        self, alpha: float = 0.05, hc_type: str = "HC1"
    ) -> npt.NDArray[np.float64]: ...
    def resid_response(self) -> npt.NDArray[np.float64]: ...
    def resid_pearson(self) -> npt.NDArray[np.float64]: ...
    def resid_deviance(self) -> npt.NDArray[np.float64]: ...
    def resid_working(self) -> npt.NDArray[np.float64]: ...
    def pearson_chi2(self) -> float: ...
    def scale_pearson(self) -> float: ...
    def llf(self) -> float: ...
    def aic(self) -> float | None: ...
    def bic(self) -> float | None: ...
    def null_deviance(self) -> float: ...
    def n_nonzero(self) -> int: ...
    def selected_features(self) -> list[int]: ...

class MultinomialResults:
    @property
    def params(self) -> npt.NDArray[np.float64]: ...
    @property
    def coefficients(self) -> npt.NDArray[np.float64]: ...
    @property
    def coef_matrix(self) -> npt.NDArray[np.float64]: ...
    @property
    def alternative_generic_coefficients(self) -> npt.NDArray[np.float64]: ...
    @property
    def alternative_specific_coefficients(self) -> npt.NDArray[np.float64]: ...
    @property
    def fitted_probabilities(self) -> npt.NDArray[np.float64]: ...
    @property
    def fittedvalues(self) -> npt.NDArray[np.float64]: ...
    @property
    def linear_predictor(self) -> npt.NDArray[np.float64]: ...
    @property
    def cov_params_unscaled(self) -> npt.NDArray[np.float64] | None: ...
    @property
    def log_likelihood(self) -> float: ...
    def llf(self) -> float: ...
    @property
    def deviance(self) -> float: ...
    @property
    def null_deviance(self) -> float: ...
    @property
    def iterations(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    @property
    def solver_status(self) -> str: ...
    @property
    def warnings(self) -> list[str]: ...
    @property
    def prior_weights(self) -> npt.NDArray[np.float64]: ...
    @property
    def y_codes(self) -> npt.NDArray[np.int64]: ...
    @property
    def reference_index(self) -> int: ...
    @property
    def nobs(self) -> int: ...
    @property
    def n_params(self) -> int: ...
    @property
    def df_model(self) -> int: ...
    @property
    def df_resid(self) -> int: ...
    @property
    def alpha(self) -> float: ...
    @property
    def l1_ratio(self) -> float: ...
    @property
    def is_regularized(self) -> bool: ...
    @property
    def penalty_type(self) -> str: ...
    @property
    def smooth_edfs(self) -> npt.NDArray[np.float64]: ...
    @property
    def total_edf(self) -> float | None: ...
    @property
    def fit_intercept(self) -> bool: ...
    @property
    def get_design_matrix(self) -> npt.NDArray[np.float64] | None: ...

# =============================================================================
# GLM Fitting Functions
# =============================================================================

def compute_standardization_py(
    x: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64] | None = None,
    pen_mask: npt.NDArray[np.bool_] | None = None,
    fit_intercept: bool = True,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]: ...
def fit_glm_py(
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    family: str,
    link: str | None = None,
    var_power: float = 1.5,
    theta: float = 1.0,
    offset: npt.NDArray[np.float64] | None = None,
    weights: npt.NDArray[np.float64] | None = None,
    alpha: float = 0.0,
    l1_ratio: float = 0.0,
    max_iter: int = 25,
    tol: float = 1e-8,
    nonneg_indices: list[int] | None = None,
    nonpos_indices: list[int] | None = None,
    store_design_matrix: bool = False,
    allow_extended_tweedie: bool = False,
    fit_intercept: bool = True,
    center: npt.NDArray[np.float64] | None = None,
    scale: npt.NDArray[np.float64] | None = None,
    skip_covariance: bool = False,
) -> GLMResults: ...
def fit_multinomial_py(
    y_codes: npt.NDArray[np.int64],
    x: npt.NDArray[np.float64],
    n_classes: int,
    reference_index: int,
    availability: npt.NDArray[np.bool_] | None = None,
    offset: npt.NDArray[np.float64] | None = None,
    weights: npt.NDArray[np.float64] | None = None,
    alpha: float = 0.0,
    l1_ratio: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-8,
    fit_intercept: bool = True,
    center: npt.NDArray[np.float64] | None = None,
    scale: npt.NDArray[np.float64] | None = None,
    skip_covariance: bool = False,
    hessian_memory_limit_bytes: int = 268435456,
    max_dense_parameters: int = 5000,
    store_design_matrix: bool = False,
    verbose: bool = False,
    alternative_generic: npt.NDArray[np.float64] | None = None,
    alternative_specific: npt.NDArray[np.float64] | None = None,
    alternative_generic_center: npt.NDArray[np.float64] | None = None,
    alternative_generic_scale: npt.NDArray[np.float64] | None = None,
    alternative_specific_center: npt.NDArray[np.float64] | None = None,
    alternative_specific_scale: npt.NDArray[np.float64] | None = None,
    initial_theta: npt.NDArray[np.float64] | None = None,
    smooth_col_ranges: list[tuple[int, int]] | None = None,
    smooth_penalties: list[npt.NDArray[np.float64]] | None = None,
    smooth_lambdas: list[float] | None = None,
    nonneg_indices: list[int] | None = None,
    nonpos_indices: list[int] | None = None,
) -> MultinomialResults: ...
def fit_negbinomial_py(
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    link: str | None = None,
    init_theta: float | None = None,
    theta_tol: float = 1e-5,
    max_theta_iter: int = 10,
    offset: npt.NDArray[np.float64] | None = None,
    weights: npt.NDArray[np.float64] | None = None,
    max_iter: int = 25,
    tol: float = 1e-8,
    alpha: float = 0.0,
    l1_ratio: float = 0.0,
    nonneg_indices: list[int] | None = None,
    nonpos_indices: list[int] | None = None,
    store_design_matrix: bool = False,
    center: npt.NDArray[np.float64] | None = None,
    scale: npt.NDArray[np.float64] | None = None,
) -> tuple[GLMResults, dict]: ...
def fit_smooth_glm_unified_py(
    y: npt.NDArray[np.float64],
    x_full: npt.NDArray[np.float64],
    smooth_col_ranges: list[tuple[int, int]],
    smooth_penalties: list[npt.NDArray[np.float64]],
    family: str,
    link: str | None = None,
    offset: npt.NDArray[np.float64] | None = None,
    weights: npt.NDArray[np.float64] | None = None,
    max_iter: int = 25,
    tol: float = 1e-8,
    lambda_min: float = 0.001,
    lambda_max: float = 1000.0,
    smooth_monotonicity: list[str | None] | None = None,
    store_design_matrix: bool = False,
    nonneg_indices: list[int] | None = None,
    nonpos_indices: list[int] | None = None,
    var_power: float = 1.5,
    theta: float = 1.0,
    allow_extended_tweedie: bool = False,
) -> tuple[GLMResults, dict]: ...
def fit_cv_path_py(
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    family: str,
    link: str | None = None,
    var_power: float = 1.5,
    theta: float = 1.0,
    offset: npt.NDArray[np.float64] | None = None,
    weights: npt.NDArray[np.float64] | None = None,
    alphas: list[float] | None = None,
    l1_ratio: float = 0.0,
    n_folds: int = 5,
    max_iter: int = 25,
    tol: float = 1e-8,
    seed: int | None = None,
    nonneg_indices: list[int] | None = None,
    nonpos_indices: list[int] | None = None,
    allow_extended_tweedie: bool = False,
    fit_intercept: bool = True,
    center: npt.NDArray[np.float64] | None = None,
    scale: npt.NDArray[np.float64] | None = None,
) -> dict: ...
def fit_fold_path_py(
    y_train: npt.NDArray[np.float64],
    x_train: npt.NDArray[np.float64],
    y_val: npt.NDArray[np.float64],
    x_val: npt.NDArray[np.float64],
    family: str,
    link: str | None = None,
    var_power: float = 1.5,
    theta: float = 1.0,
    offset_train: npt.NDArray[np.float64] | None = None,
    weights_train: npt.NDArray[np.float64] | None = None,
    offset_val: npt.NDArray[np.float64] | None = None,
    weights_val: npt.NDArray[np.float64] | None = None,
    alphas: list[float] | None = None,
    l1_ratio: float = 0.0,
    max_iter: int = 25,
    tol: float = 1e-8,
    nonneg_indices: list[int] | None = None,
    nonpos_indices: list[int] | None = None,
    allow_extended_tweedie: bool = False,
    fit_intercept: bool = True,
    center: npt.NDArray[np.float64] | None = None,
    scale: npt.NDArray[np.float64] | None = None,
) -> dict: ...

# =============================================================================
# IRLS Residual Helpers
# =============================================================================

def working_response_weights_py(
    y: npt.NDArray[np.float64],
    eta: npt.NDArray[np.float64],
    family: str,
    link: str | None = None,
    var_power: float = 1.5,
    theta: float = 1.0,
    offset: npt.NDArray[np.float64] | None = None,
    weights: npt.NDArray[np.float64] | None = None,
    allow_extended_tweedie: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...

# =============================================================================
# Inference Functions
# =============================================================================

def score_test_continuous_py(
    z: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    bread: npt.NDArray[np.float64],
    family: str,
) -> dict: ...
def score_test_continuous_batch_py(
    zs: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    bread: npt.NDArray[np.float64],
    family: str,
) -> list[dict]:
    """Batched Rao's score test for adding k continuous variables to a fitted model."""
    ...

def score_test_categorical_py(
    z_matrix: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    bread: npt.NDArray[np.float64],
    family: str,
) -> dict: ...
def chi2_cdf_py(x: float, df: float) -> float: ...
def t_cdf_py(x: float, df: float) -> float: ...
def f_cdf_py(x: float, df1: float, df2: float) -> float: ...

# =============================================================================
# Spline Functions
# =============================================================================

def bs_py(
    x: npt.NDArray[np.float64],
    df: int | None = None,
    knots: npt.NDArray[np.float64] | None = None,
    degree: int = 3,
    include_intercept: bool = False,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> npt.NDArray[np.float64]: ...
def ns_py(
    x: npt.NDArray[np.float64],
    df: int | None = None,
    knots: npt.NDArray[np.float64] | None = None,
    include_intercept: bool = False,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> npt.NDArray[np.float64]: ...
def ns_with_knots_py(
    x: npt.NDArray[np.float64],
    df: int | None = None,
    knots: npt.NDArray[np.float64] | None = None,
    include_intercept: bool = False,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
def bs_knots_py(
    x: npt.NDArray[np.float64],
    df: int | None = None,
    knots: npt.NDArray[np.float64] | None = None,
    degree: int = 3,
    include_intercept: bool = False,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
def bs_names_py(prefix: str, n_cols: int) -> list[str]: ...
def ns_names_py(prefix: str, n_cols: int) -> list[str]: ...
def ms_py(
    x: npt.NDArray[np.float64],
    df: int | None = None,
    knots: npt.NDArray[np.float64] | None = None,
    degree: int = 3,
    include_intercept: bool = False,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    increasing: bool = True,
) -> npt.NDArray[np.float64]: ...
def ms_with_knots_py(
    x: npt.NDArray[np.float64],
    df: int | None = None,
    knots: npt.NDArray[np.float64] | None = None,
    degree: int = 3,
    include_intercept: bool = False,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    increasing: bool = True,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
def ms_names_py(prefix: str, n_cols: int) -> list[str]: ...
def compute_knots_py(x: npt.NDArray[np.float64], n_knots: int) -> npt.NDArray[np.float64]: ...
def compute_knots_natural_py(
    x: npt.NDArray[np.float64], n_knots: int
) -> npt.NDArray[np.float64]: ...

# =============================================================================
# Design Matrix Functions
# =============================================================================

def encode_categorical_py(
    values: npt.NDArray[np.float64],
    drop_first: bool = True,
) -> tuple[npt.NDArray[np.float64], list[str]]: ...
def factorize_strings_py(
    values: list[str],
) -> tuple[npt.NDArray[np.int64], list[str]]: ...
def encode_categorical_indices_py(
    indices: npt.NDArray[np.int64],
    n_levels: int,
    drop_first: bool = True,
) -> npt.NDArray[np.float64]: ...
def build_cat_cat_interaction_py(
    col_a: npt.NDArray[np.int64],
    col_b: npt.NDArray[np.int64],
    n_levels_a: int,
    n_levels_b: int,
    drop_first: bool = True,
) -> tuple[npt.NDArray[np.float64], list[str]]: ...
def build_cat_cont_interaction_py(
    cat_col: npt.NDArray[np.int64],
    cont_col: npt.NDArray[np.float64],
    n_levels: int,
    drop_first: bool = True,
) -> tuple[npt.NDArray[np.float64], list[str]]: ...
def build_cat_basis_interaction_py(
    cat_indices: npt.NDArray[np.int64],
    n_levels: int,
    basis: npt.NDArray[np.float64],
    cat_names: list[str],
    basis_names: list[str],
) -> tuple[npt.NDArray[np.float64], list[str]]: ...
def predict_cat_basis_interaction_py(
    cat_indices: npt.NDArray[np.int64],
    n_levels: int,
    basis: npt.NDArray[np.float64],
    params: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def build_two_cat_cont_interaction_py(
    idx1: npt.NDArray[np.int64],
    n_levels1: int,
    idx2: npt.NDArray[np.int64],
    n_levels2: int,
    continuous: npt.NDArray[np.float64],
    names1: list[str],
    names2: list[str],
    cont_name: str,
) -> tuple[npt.NDArray[np.float64], list[str]]: ...
def build_cont_cont_interaction_py(
    col_a: npt.NDArray[np.float64],
    col_b: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def multiply_matrix_by_continuous_py(
    matrix: npt.NDArray[np.float64],
    continuous: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def stack_columns_horizontal_py(
    blocks: list[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Stack a list of (n x c_i) f64 column blocks horizontally into a single (n x sum(c_i)) matrix."""
    ...

# =============================================================================
# Target Encoding Functions
# =============================================================================

def target_encode_py(
    categories: list[str],
    target: npt.NDArray[np.float64],
    var_name: str,
    prior_weight: float = 1.0,
    n_permutations: int = 4,
    seed: int | None = None,
) -> tuple[npt.NDArray[np.float64], str, float, dict[str, tuple[float, int]]]: ...
def apply_target_encoding_py(
    categories: list[str],
    level_stats: dict[str, tuple[float, int]],
    prior: float,
    prior_weight: float = 1.0,
) -> npt.NDArray[np.float64]: ...
def target_encode_with_exposure_py(
    categories: list[str],
    claims: npt.NDArray[np.float64],
    exposure: npt.NDArray[np.float64],
    var_name: str,
    prior_weight: float = 1.0,
    n_permutations: int = 4,
    seed: int | None = None,
) -> tuple[npt.NDArray[np.float64], str, float, dict[str, tuple[float, float]]]: ...
def apply_exposure_weighted_target_encoding_py(
    categories: list[str],
    level_stats: dict[str, tuple[float, float]],
    prior: float,
    prior_weight: float = 1.0,
) -> npt.NDArray[np.float64]: ...
def frequency_encode_py(
    categories: list[str],
    var_name: str,
) -> tuple[npt.NDArray[np.float64], str, dict[str, int], int, int]: ...
def apply_frequency_encoding_py(
    categories: list[str],
    level_counts: dict[str, int],
    max_count: int,
) -> npt.NDArray[np.float64]: ...
def target_encode_interaction_py(
    cat1: list[str],
    cat2: list[str],
    target: npt.NDArray[np.float64],
    var_name1: str,
    var_name2: str,
    prior_weight: float = 1.0,
    n_permutations: int = 4,
    seed: int | None = None,
) -> tuple[npt.NDArray[np.float64], str, float, dict[str, tuple[float, int]]]: ...
def target_encode_interaction_with_exposure_py(
    cat1: list[str],
    cat2: list[str],
    claims: npt.NDArray[np.float64],
    exposure: npt.NDArray[np.float64],
    var_name1: str,
    var_name2: str,
    prior_weight: float = 1.0,
    n_permutations: int = 4,
    seed: int | None = None,
) -> tuple[npt.NDArray[np.float64], str, float, dict[str, tuple[float, float]]]: ...

# =============================================================================
# Diagnostics Functions
# =============================================================================

def compute_calibration_curve_py(
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    exposure: npt.NDArray[np.float64] | None = None,
    n_bins: int = 10,
) -> list[dict]: ...
def compute_discrimination_stats_py(
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    exposure: npt.NDArray[np.float64] | None = None,
) -> dict: ...
def compute_ae_continuous_py(
    values: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    exposure: npt.NDArray[np.float64] | None = None,
    n_bins: int = 10,
    family: str = "poisson",
    prior_weights: npt.NDArray[np.float64] | None = None,
    base: npt.NDArray[np.float64] | None = None,
) -> list[dict]: ...
def compute_ae_continuous_batch_py(
    values_list: list[npt.NDArray[np.float64]],
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    exposure: npt.NDArray[np.float64] | None = None,
    n_bins: int = 10,
    family: str = "poisson",
    prior_weights: npt.NDArray[np.float64] | None = None,
    base: npt.NDArray[np.float64] | None = None,
) -> list[list[dict]]:
    """Compute A/E bins for many continuous factors at once, parallelized over factors.

    `values_list` is a Python list where each entry is a contiguous 1D float64
    array of per-row values for one factor. Length-mismatched entries raise
    ValueError.
    """
    ...

def compute_ae_categorical_py(
    levels: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    exposure: npt.NDArray[np.float64] | None = None,
    rare_threshold_pct: float = 1.0,
    max_levels: int = 20,
    family: str = "poisson",
    prior_weights: npt.NDArray[np.float64] | None = None,
    base: npt.NDArray[np.float64] | None = None,
) -> list[dict]: ...
def compute_ae_categorical_batch_py(
    codes_list: list[npt.NDArray[np.uint32]],
    levels_list: list[list[str]],
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    exposure: npt.NDArray[np.float64] | None = None,
    rare_threshold_pct: float = 1.0,
    max_levels: int = 20,
    family: str = "poisson",
    prior_weights: npt.NDArray[np.float64] | None = None,
    base: npt.NDArray[np.float64] | None = None,
) -> list[list[dict]]:
    """Compute A/E bins for many categorical factors at once, parallelized over factors.

    `codes_list` is a Python list where each entry is a contiguous 1D
    np.uint32 array of per-row codes for one factor; codes index into the
    matching `levels_list[j]`. Length-mismatched entries raise ValueError.
    """
    ...

def compute_ae_by_decile_py(
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    exposure: npt.NDArray[np.float64] | None = None,
    n_deciles: int = 10,
    sort_idx: npt.NDArray[np.uintp] | None = None,
) -> list[dict]:
    """Compute A/E aggregates per decile (sorted by predicted value) from Rust."""
    ...

def partial_dependence_categorical_batch_py(
    codes_list: list[npt.NDArray[np.uint32]],
    mu: npt.NDArray[np.float64],
    n_levels_per_factor: list[int],
) -> list[tuple[list[float], list[float]]]:
    """Per-factor categorical partial-dependence aggregates (counts, mu_sums), parallel over factors.

    `codes_list` is a length-k list of 1D u32 numpy arrays (one per factor),
    avoiding the (n, k) `np.stack` transient required by the prior matrix
    signature.
    """
    ...

def compute_correlation_and_vif_py(
    x: npt.NDArray[np.float64],
    epsilon: float,
    skip_cols: int = 0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute the column correlation matrix and `diag((R + ε·I)^{-1})` of (n, k) X.

    Skips the first ``skip_cols`` columns (typically the intercept) without
    requiring a Python-side slice. Returns (R, vif_diagonal) of shape
    ((k - skip_cols), (k - skip_cols)) and (k - skip_cols,).
    """
    ...

def compute_correlation_moments_py(
    x: npt.NDArray[np.float64],
    skip_cols: int = 0,
) -> tuple[int, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute streamable column sums and upper-triangle Gram moments for X."""
    ...

def aggregate_pair_cells_py(
    codes1: npt.NDArray[np.uint32],
    n_levels1: int,
    codes2: npt.NDArray[np.uint32],
    n_levels2: int,
    y: npt.NDArray[np.float64],
    exposure: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64] | None = None,
) -> list[tuple[int, int, int, float, float, float]]:
    """Aggregate non-empty pair cells as (r, c, count, exposure_sum, y_sum, mu_sum)."""
    ...

def interaction_strength_from_codes_py(
    codes1: npt.NDArray[np.uint32],
    n_levels1: int,
    codes2: npt.NDArray[np.uint32],
    n_levels2: int,
    y: npt.NDArray[np.float64],
    exposure: npt.NDArray[np.float64],
    min_cell_count: int = 0,
) -> float:
    """Compute the exploratory interaction-strength scalar for one coded pair."""
    ...

def compute_factor_deviance_py(
    factor_name: str,
    factor_values: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    family: str = "poisson",
    var_power: float = 1.5,
    theta: float = 1.0,
) -> dict: ...
def compute_factor_deviance_batch_py(
    factor_names: list[str],
    factor_values_list: list[list[str]],
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    family: str = "poisson",
    var_power: float = 1.5,
    theta: float = 1.0,
) -> list[dict]:
    """Compute factor deviance breakdown for many categorical factors at once, parallel over factors."""
    ...

def compute_factor_deviance_batch_from_codes_py(
    factor_names: list[str],
    codes_matrix: npt.NDArray[np.uint32],
    levels_list: list[list[str]],
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    family: str = "poisson",
    var_power: float = 1.5,
    theta: float = 1.0,
) -> list[dict]:
    """Code-based variant of compute_factor_deviance_batch_py (avoids per-row string marshalling)."""
    ...

def compute_factor_significance_batch_py(
    param_indices_per_factor: list[list[int]],
    params: npt.NDArray[np.float64],
    bse: npt.NDArray[np.float64],
    bread: npt.NDArray[np.float64],
) -> list[dict | None]:
    """Batch-compute joint Wald factor significance (chi2, pvalue, df) for k factors in parallel."""
    ...

def compute_loss_metrics_py(
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    family: str,
    weights: npt.NDArray[np.float64] | None = None,
    var_power: float | None = None,
    theta: float | None = None,
) -> dict: ...
def detect_interactions_py(
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    columns: dict,
    weights: npt.NDArray[np.float64] | None = None,
    top_k: int = 20,
    n_bins: int = 10,
) -> list[dict]: ...
def compute_lorenz_curve_py(
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    exposure: npt.NDArray[np.float64] | None = None,
    n_points: int = 20,
) -> list[dict]: ...
def hosmer_lemeshow_test_py(
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    n_groups: int = 10,
) -> dict: ...
def compute_fit_statistics_py(
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    family: str,
    n_params: int,
    weights: npt.NDArray[np.float64] | None = None,
    var_power: float | None = None,
    theta: float | None = None,
) -> dict: ...
def compute_dataset_metrics_py(
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    family: str,
    n_params: int,
    var_power: float = 1.5,
    theta: float = 1.0,
    scale: float | None = None,
) -> dict: ...
def compute_residual_summary_py(
    residuals: npt.NDArray[np.float64],
) -> dict: ...
def compute_residual_pattern_py(
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    family: str,
    var_power: float | None = None,
    theta: float | None = None,
) -> dict: ...
def compute_residual_pattern_batch_py(
    values_list: list[npt.NDArray[np.float64]],
    residuals: npt.NDArray[np.float64],
    n_bins: int = 10,
) -> list[dict]:
    """Compute residual-pattern dicts for many continuous factors at once, parallelized over factors.

    `values_list` is a Python list where each entry is a contiguous 1D float64
    array of per-row values for one factor; the shared `residuals` array is
    reused across all factors. Length-mismatched entries raise ValueError.
    """
    ...

def compute_pearson_residuals_py(
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    family: str,
    var_power: float = 1.5,
    theta: float = 1.0,
) -> npt.NDArray[np.float64]: ...
def compute_deviance_residuals_py(
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    family: str,
    var_power: float = 1.5,
    theta: float = 1.0,
) -> npt.NDArray[np.float64]: ...
def compute_null_deviance_py(
    y: npt.NDArray[np.float64],
    family: str,
    weights: npt.NDArray[np.float64] | None = None,
    offset: npt.NDArray[np.float64] | None = None,
) -> float: ...
def compute_unit_deviance_py(
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    family: str,
    var_power: float = 1.5,
    theta: float = 1.0,
) -> npt.NDArray[np.float64]: ...

# =============================================================================
# ONNX Export Functions
# =============================================================================

def build_onnx_glm_scoring_py(
    coefficients: npt.NDArray[np.float64],
    intercept: float,
    n_features: int,
    link: str,
    family: str,
    metadata_keys: list[str],
    metadata_values: list[str],
) -> bytes: ...
def serialize_onnx_graph_py(
    node_ops: list[str],
    node_inputs: list[list[str]],
    node_outputs: list[list[str]],
    node_attr_names: list[list[str]],
    node_attr_types: list[list[str]],
    node_attr_ints: list[list[int]],
    node_attr_floats: list[list[float]],
    init_names_f64: list[str],
    init_data_f64: list[list[float]],
    init_shapes_f64: list[list[int]],
    init_names_i64: list[str],
    init_data_i64: list[list[int]],
    init_shapes_i64: list[list[int]],
    input_names: list[str],
    input_types: list[int],
    input_shapes: list[list[int]],
    output_names: list[str],
    output_types: list[int],
    output_shapes: list[list[int]],
    ir_version: int,
    opset_version: int,
    producer: str,
    doc_string: str,
    meta_keys: list[str],
    meta_values: list[str],
) -> bytes: ...
