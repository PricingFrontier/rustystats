"""
Top-level diagnostic API functions.

compute_diagnostics() is the main entry point for post-fit model diagnostics.
_compute_smooth_term_diagnostics() handles GAM smooth term analysis.
"""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
import polars as pl

from rustystats._rustystats import (
    chi2_cdf_py as _chi2_cdf,
)
from rustystats.constants import (
    CALIBRATION_AE_LOWER,
    CALIBRATION_AE_UPPER,
    DEFAULT_MAX_CATEGORICAL_LEVELS,
    DEFAULT_MAX_INTERACTION_FACTORS,
    DEFAULT_N_CALIBRATION_BINS,
    DEFAULT_N_FACTOR_BINS,
    DEFAULT_RARE_THRESHOLD_PCT,
    FACTOR_AE_DIFF_THRESHOLD,
    OVERDISPERSION_MILD,
    OVERDISPERSION_MODERATE,
    OVERDISPERSION_SEVERE,
    OVERFITTING_GINI_GAP_THRESHOLD,
    SIGNIFICANCE_THRESHOLD,
)
from rustystats.diagnostics.computer import DiagnosticsComputer, rank_sort_idx
from rustystats.diagnostics.pair_diagnostics import (
    _build_design_correlation_matrix,
    _build_design_correlation_matrix_from_moments,
    _compute_block_gvif,
    _correlation_moments_for_design_chunk,
)
from rustystats.diagnostics.types import (
    BasePredictionsByRole,
    EncodingDiagnostics,
    FactorBinPair,
    FactorCoefficient,
    FactorLevelMetrics,
    FactorSignificance,
    InteractionBlockDiagnostics,
    InteractionDiagnostics,
    ModelDiagnostics,
    SmoothTermDiagnostics,
    TrainTestComparison,
)
from rustystats.exceptions import ValidationError

# Overdispersion severity levels — checked in order, first match wins.
# Used by _classify_overdispersion / _compute_overdispersion to map a
# Pearson dispersion ratio to a severity label and recommendation string.
_OVERDISPERSION_LEVELS = (
    # (threshold, severity, recommendation)
    (OVERDISPERSION_SEVERE, "severe", "Use Negative Binomial or QuasiPoisson"),
    (OVERDISPERSION_MODERATE, "moderate", "Consider Negative Binomial or QuasiPoisson"),
    (OVERDISPERSION_MILD, "mild", "Monitor; Poisson may underestimate standard errors"),
)


def _classify_overdispersion(pearson_dispersion: float) -> tuple[str, str]:
    """Map a Pearson dispersion ratio to (severity, recommendation)."""
    for threshold, severity, recommendation in _OVERDISPERSION_LEVELS:
        if pearson_dispersion > threshold:
            return severity, recommendation
    return "none", "Poisson assumption appears reasonable"


def _compute_smooth_term_diagnostics(
    result: Any,
    warnings: list[dict[str, str]],
) -> list[SmoothTermDiagnostics]:
    """
    Compute diagnostics for smooth terms including EDF and significance tests.

    Uses a Wald-type chi-squared test to assess whether the smooth term as a
    whole is significant. The test statistic is β' × Cov⁻¹ × β where β are
    the coefficients for the smooth term and Cov is the corresponding
    submatrix of the covariance matrix.

    Parameters
    ----------
    result : GLMModel
        Fitted model with smooth terms
    warnings : list
        List to append warnings to

    Returns
    -------
    list of SmoothTermDiagnostics
        Diagnostics for each smooth term
    """
    if not hasattr(result, "smooth_terms") or result.smooth_terms is None:
        return []

    smooth_diagnostics = []
    params = result.params

    # Get covariance matrix (unscaled)
    cov_matrix = None
    if hasattr(result, "get_bread_matrix"):
        cov_matrix = result.get_bread_matrix()
    elif hasattr(result, "_result") and hasattr(result._result, "cov_params_unscaled"):
        cov_matrix = result._result.cov_params_unscaled
    elif hasattr(result, "_result") and hasattr(result._result, "covariance_unscaled"):
        cov_matrix = result._result.covariance_unscaled
    elif hasattr(result, "cov_params"):
        cov_matrix = result.cov_params()

    for st in result.smooth_terms:
        # Extract coefficient indices for this smooth term
        col_start = st.col_start
        col_end = st.col_end
        n_coef = col_end - col_start

        # Get coefficients for this term
        beta = params[col_start:col_end]

        # Compute Wald chi-squared statistic
        chi2 = 0.0
        ref_df = st.edf  # Use EDF as reference df
        p_value = 1.0

        if cov_matrix is not None and n_coef > 0:
            try:
                # Extract covariance submatrix for this term
                cov_sub = cov_matrix[col_start:col_end, col_start:col_end]

                # Compute Wald statistic: β' × Cov⁻¹ × β
                # Use pseudo-inverse for numerical stability
                cov_inv = np.linalg.pinv(cov_sub)
                chi2 = float(beta @ cov_inv @ beta)

                # P-value from chi-squared distribution with EDF degrees of freedom
                # Use EDF as the reference df (as in mgcv)
                if chi2 > 0 and ref_df > 0:
                    p_value = 1.0 - _chi2_cdf(chi2, ref_df)
            except (np.linalg.LinAlgError, ValueError) as e:
                # Singular matrix - warn and fall back to simpler test
                warnings.append(
                    {
                        "type": "smooth_significance_fallback",
                        "message": f"Covariance matrix singular for s({st.variable}), using simplified test: {e}",
                    }
                )
                chi2 = float(np.sum(beta**2))
                ref_df = float(n_coef)
                if chi2 > 0 and ref_df > 0:
                    p_value = 1.0 - _chi2_cdf(chi2, ref_df)

        smooth_diag = SmoothTermDiagnostics(
            variable=st.variable,
            k=st.k,
            edf=st.edf,
            lambda_=st.lambda_,
            gcv=st.gcv,
            ref_df=ref_df,
            chi2=chi2,
            p_value=p_value,
        )
        smooth_diagnostics.append(smooth_diag)

        # Add warning for non-significant smooth terms
        if p_value > SIGNIFICANCE_THRESHOLD:
            warnings.append(
                {
                    "type": "insignificant_smooth",
                    "message": f"Smooth term s({st.variable}) is not significant "
                    f"(p={p_value:.3f}, EDF={st.edf:.1f}). "
                    f"Consider using linear term or removing.",
                }
            )
        # Add warning for EDF close to k (under-smoothed)
        elif st.edf > st.k - 1.5:
            warnings.append(
                {
                    "type": "undersmoothed",
                    "message": f"Smooth term s({st.variable}) has EDF≈k ({st.edf:.1f}/{st.k}). "
                    f"Consider increasing k for more flexibility.",
                }
            )

    return smooth_diagnostics


# ---------------------------------------------------------------------------
# Helpers for compute_diagnostics
# ---------------------------------------------------------------------------


def _normalize_factor_lists(
    categorical_factors: list[str | None] | None,
    continuous_factors: list[str | None] | None,
) -> tuple[list[str], list[str]]:
    """Deduplicate factor lists and remove categorical/continuous overlap."""
    cat = list(dict.fromkeys(categorical_factors or []))
    cont = list(dict.fromkeys(continuous_factors or []))
    cont = [f for f in cont if f not in cat]
    return cat, cont


def _extract_response_and_predictions(
    result: Any,
    train_data: pl.DataFrame,
    exposure: str | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y, mu, lp) for the training data.

    Re-predicts on train_data only when the model contains TE() features (which
    use LOO encoding during fit but full encoding for predict, so fittedvalues
    would be LOO-handicapped). For non-TE models, fittedvalues equals
    predict(train_data) and is reused for free.
    """
    if exposure is None and getattr(result, "_array_exposure_requires_prediction_override", False):
        raise ValidationError(
            "This model was fit with an array exposure that is not reusable diagnostics "
            "metadata. Pass exposure= explicitly to diagnostics()."
        )
    formula_parts = result.formula.split("~") if hasattr(result, "formula") else []
    response_col_temp = formula_parts[0].strip() if formula_parts else None

    if response_col_temp and response_col_temp in train_data.columns:
        y = train_data[response_col_temp].to_numpy().astype(np.float64)
        has_te = any(fn.startswith("TE(") for fn in getattr(result, "feature_names", []))
        if (
            has_te or getattr(result, "_is_deserialized", False) or exposure is not None
        ) and hasattr(result, "predict"):
            # Diagnostics score the model's true mean; the response-ceiling
            # extrapolation guardrail must not distort them, and the
            # extreme-eta guardrail must not abort them (clip keeps the report
            # alive; extreme rows surface as extreme deviance/lift instead).
            mu = np.asarray(
                result.predict(
                    train_data,
                    exposure=exposure,
                    response_ceiling=None,
                    on_extreme_eta="clip",
                ),
                dtype=np.float64,
            )
        else:
            mu = np.asarray(result.fittedvalues, dtype=np.float64)
        lp = np.log(mu) if np.all(mu > 0) else mu
    else:
        # Fallback to fitted values if we can't determine response column
        mu = np.asarray(result.fittedvalues, dtype=np.float64)
        response_resid = np.asarray(result.resid_response(), dtype=np.float64)
        y = mu + response_resid
        lp = np.asarray(result.linear_predictor, dtype=np.float64)
    return y, mu, lp


def _spec_is_transform_output(result: Any, spec: Any) -> bool:
    return isinstance(spec, str) and any(
        transform.get("output") == spec for transform in getattr(result, "_input_transforms", [])
    )


def _data_for_model_spec(result: Any, data: pl.DataFrame, spec: Any) -> pl.DataFrame:
    if _spec_is_transform_output(result, spec) and hasattr(result, "prepare_input"):
        return result.prepare_input(data)
    return data


def _validate_data_length(train_data: pl.DataFrame, mu: np.ndarray) -> None:
    """Raise ValidationError when train_data length disagrees with model fit length."""
    if len(train_data) != len(mu):
        raise ValidationError(
            f"train_data has {len(train_data)} rows but the model was fit on a dataset of "
            f"{len(mu)} rows (inferred from fittedvalues / predict). "
            f"Pass the SAME data the model was fit on as train_data, "
            f"or pass holdout data via the test_data= kwarg."
        )


def _extract_model_metadata(result: Any) -> tuple[str, str, int, float, list[str]]:
    """Return (family, link, n_params, deviance, feature_names) and validate presence."""
    if not hasattr(result, "family"):
        raise ValidationError("Result object missing 'family' attribute")
    if not hasattr(result, "link"):
        raise ValidationError("Result object missing 'link' attribute")
    if not hasattr(result, "feature_names"):
        raise ValidationError("Result object missing 'feature_names' attribute")
    return (
        result.family,
        result.link,
        len(result.params),
        result.deviance,
        result.feature_names,
    )


def _resolve_offset_and_response(
    result: Any,
    train_data: pl.DataFrame,
    exposure_override: str | np.ndarray | None = None,
) -> tuple[str | None, str | None, np.ndarray | None]:
    """Auto-infer response and exposure column names from the model formula."""
    response_col = None
    exposure_col = None
    if hasattr(result, "formula") and result.formula:
        formula_parts = result.formula.split("~")
        if len(formula_parts) >= 1:
            response_col = formula_parts[0].strip()

    exposure = None
    exposure_spec = exposure_override
    if exposure_spec is None:
        exposure_spec = getattr(result, "_exposure_spec", None)
    if exposure_spec is None and getattr(
        result, "_array_exposure_requires_prediction_override", False
    ):
        raise ValidationError(
            "This model was fit with an array exposure that is not reusable diagnostics "
            "metadata. Pass exposure= explicitly to diagnostics()."
        )
    if isinstance(exposure_spec, str):
        exposure_col = exposure_spec
        data_for_exposure = _data_for_model_spec(result, train_data, exposure_col)
        if exposure_col not in data_for_exposure.columns:
            raise ValidationError(
                f"Model requires exposure column '{exposure_col}', but it is not "
                "present in train_data. Pass exposure= explicitly to diagnostics()."
            )
        exposure = data_for_exposure[exposure_col].to_numpy().astype(np.float64)
    elif exposure_spec is not None:
        exposure = np.asarray(exposure_spec, dtype=np.float64)
        if exposure.ndim != 1:
            raise ValidationError(f"exposure must be one-dimensional; got shape {exposure.shape}.")
        if exposure.shape[0] != train_data.height:
            raise ValidationError(
                f"Stored exposure length {exposure.shape[0]} does not match "
                f"train_data length {train_data.height}."
            )

    return response_col, exposure_col, exposure


def _parse_family_params(family: str) -> tuple[float, float]:
    """Extract Tweedie var_power and NegBinomial theta from family string."""
    var_power = 1.5
    theta = 1.0
    if "tweedie" in family.lower():
        match = re.search(r"p=(\d+\.?\d*)", family)
        if match:
            var_power = float(match.group(1))
    if "negbinomial" in family.lower() or "negativebinomial" in family.lower():
        match = re.search(r"theta=(\d+\.?\d*)", family)
        if match:
            theta = float(match.group(1))
    return var_power, theta


def _resolve_null_deviance(result: Any) -> float | None:
    """Return null deviance from the model result, calling it if needed."""
    if not hasattr(result, "null_deviance"):
        return None
    nd = result.null_deviance
    return nd() if callable(nd) else nd


def _precompute_data_caches(
    data: pl.DataFrame,
    categorical_factors: list[str],
    continuous_factors: list[str],
) -> tuple[dict, dict, dict]:
    """Pre-extract categorical and continuous columns using per-column factorization.

    Uses ``pl.Enum`` per column: Enum is a per-column categorical type
    (unlike ``pl.Categorical`` which uses a session-level string cache that
    leaks levels across columns). This gives the correctness of isolated
    factorization with the speed of polars' native string handling
    (~8× faster than ``np.unique`` on an object-dtype array at n=1M).
    """
    cat_cache: dict = {}
    cat_unique_cache: dict = {}
    for name in categorical_factors:
        if name not in data.columns:
            continue

        values = data[name].cast(pl.Utf8)
        sorted_level_list = values.unique().sort().to_list()
        sorted_levels = np.array(sorted_level_list)
        enum_series = values.cast(pl.Enum(sorted_level_list))
        codes = enum_series.to_physical().to_numpy().astype(np.uint32)

        # Materialized string array for the legacy cat_cache (unchanged contract).
        str_vals = sorted_levels[codes]

        cat_cache[name] = str_vals
        cat_unique_cache[name] = (sorted_levels, codes)

    cont_cache: dict = {}
    for name in continuous_factors:
        if name in data.columns:
            cont_cache[name] = data[name].to_numpy().astype(np.float64)

    return cat_cache, cat_unique_cache, cont_cache


def _fit_design_source_data(result: Any, train_data: pl.DataFrame) -> pl.DataFrame | None:
    """Return the fit-time prepared frame when it aligns with ``train_data``."""
    builder = getattr(result, "_builder", None)
    fit_data = getattr(builder, "data", None)
    if isinstance(fit_data, pl.DataFrame) and fit_data.height == train_data.height:
        return fit_data
    return None


def _iter_design_matrix_chunks(result: Any, train_data: pl.DataFrame):
    """Yield design-matrix row chunks for ``train_data`` using the fitted builder."""
    if not (hasattr(result, "_builder") and result._builder is not None):
        return

    from rustystats.formula import _compute_predict_chunk_size

    n_rows = len(train_data)
    n_features = len(result.params)
    chunk_size = _compute_predict_chunk_size(n_features)
    prepared_train = _fit_design_source_data(result, train_data)

    if n_rows <= chunk_size:
        prepared = (
            prepared_train if prepared_train is not None else result.prepare_input(train_data)
        )
        yield result._builder.transform_new_data(prepared)
        return

    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        chunk = (
            prepared_train.slice(start, stop - start)
            if prepared_train is not None
            else result.prepare_input(train_data.slice(start, stop - start))
        )
        yield result._builder.transform_new_data(chunk)


def _extract_score_test_matrices(
    result: Any,
    train_data: pl.DataFrame,
    *,
    build_design_matrix: bool = True,
) -> tuple[Any, Any, Any]:
    """Return (design_matrix, bread_matrix, irls_weights) for Rao score tests.

    Falls back to rebuilding the design matrix from train_data when not stored
    (lean mode).
    """
    design_matrix = None
    bread_matrix = None
    irls_weights = None
    if build_design_matrix and hasattr(result, "get_design_matrix"):
        design_matrix = result.get_design_matrix()
    if (
        build_design_matrix
        and design_matrix is None
        and hasattr(result, "_builder")
        and result._builder is not None
    ):
        # Chunked rebuild: for large n we write row-blocks into a preallocated
        # (n, p) output, so transient peak is ~2*(chunk_size*p*8) rather than
        # doubling during Rust's horizontal stack of the full build. Mirrors
        # the chunked predict() path in formula.py.
        n_rows = len(train_data)
        n_features = len(result.params)
        chunks = _iter_design_matrix_chunks(result, train_data)
        if chunks is not None:
            design_matrix = np.empty((n_rows, n_features), dtype=np.float64)
            cursor = 0
            for X_chunk in chunks:
                stop = cursor + X_chunk.shape[0]
                design_matrix[cursor:stop, :] = X_chunk
                cursor = stop
                # Mark the reference dead so the chunk can be freed before the
                # next iteration allocates.
                del X_chunk
    if hasattr(result, "get_bread_matrix"):
        bread_matrix = result.get_bread_matrix()
    if hasattr(result, "get_irls_weights"):
        irls_weights = result.get_irls_weights()
    return design_matrix, bread_matrix, irls_weights


def _score_tests_need_design_matrix(
    computer: DiagnosticsComputer,
    result: Any,
    categorical_factors: list[str],
    continuous_factors: list[str],
    compute_score_tests: bool,
) -> bool:
    """Whether any requested factor needs the expanded design for a score test."""
    if not compute_score_tests:
        return False

    factor_computer = getattr(computer, "_factors", None)
    if factor_computer is None:
        return bool(categorical_factors or continuous_factors)

    refresh_aliases = getattr(factor_computer, "_refresh_transform_source_aliases", None)
    if callable(refresh_aliases):
        refresh_aliases(result)

    get_feature = getattr(factor_computer, "_get_feature_for", None)
    if not callable(get_feature):
        return bool(categorical_factors or continuous_factors)

    return any(not get_feature(name).indices for name in categorical_factors + continuous_factors)


def _build_design_correlation_context_from_model(
    result: Any,
    train_data: pl.DataFrame,
    *,
    include_inverse: bool,
) -> Any:
    """Stream design chunks into a VIF/GVIF correlation context."""
    chunks = _iter_design_matrix_chunks(result, train_data)
    if chunks is None:
        return None

    total_rows = 0
    sums_total: np.ndarray | None = None
    gram_total: np.ndarray | None = None
    for X_chunk in chunks:
        n_rows, sums, gram_upper = _correlation_moments_for_design_chunk(X_chunk)
        if sums_total is None:
            sums_total = np.zeros_like(sums)
            gram_total = np.zeros_like(gram_upper)
        if (
            sums_total.shape != sums.shape
            or gram_total is None
            or gram_total.shape != gram_upper.shape
        ):
            return None
        total_rows += n_rows
        sums_total += sums
        gram_total += gram_upper
        del X_chunk

    if sums_total is None or gram_total is None:
        return None
    return _build_design_correlation_matrix_from_moments(
        total_rows,
        sums_total,
        gram_total,
        include_inverse=include_inverse,
    )


def _maybe_compute_interactions(
    detect_interactions: bool,
    computer: DiagnosticsComputer,
    train_data: pl.DataFrame,
    categorical_factors: list[str],
    continuous_factors: list[str],
    max_interaction_factors: int,
    cat_cache: dict,
    cont_cache: dict,
) -> list:
    """Run residual-based interaction detection when requested and feasible."""
    if not detect_interactions:
        return []
    if len(categorical_factors) + len(continuous_factors) < 2:
        return []
    all_factors = categorical_factors + continuous_factors
    return computer.detect_interactions(
        data=train_data,
        factor_names=all_factors,
        max_factors=max_interaction_factors,
        cat_column_cache=cat_cache,
        cont_column_cache=cont_cache,
    )


def _compute_pair_diagnostics(
    computer: DiagnosticsComputer,
    interactions: list[Any],
    train_data: pl.DataFrame,
    result: Any,
    response_col: str | None,
    exposure_col: str | None,
    score_test_design_matrix: np.ndarray | None,
    score_test_bread_matrix: np.ndarray | None,
    test_data: pl.DataFrame | None,
    link: str | None,
    correlation_matrix: np.ndarray | None = None,
    test_y: np.ndarray | None = None,
    test_mu: np.ndarray | None = None,
    test_exposure: np.ndarray | None = None,
) -> list[InteractionDiagnostics]:
    """Compute per-pair (interaction) diagnostics for user-supplied pairs.

    Reuses the test arrays, model ``params`` / ``bse``, score-test design /
    bread matrices, and regularized correlation matrix that have already been
    pulled upstream. That keeps pair diagnostics to one prediction pass over
    test data and one O(n*p^2) design-correlation build per diagnostics call.
    """
    # Extract params / bse from the fitted model. Match the lookup pattern
    # used by _FactorDiagnosticsComputer (factors.py:398-403): bse may be a
    # property or a callable.
    params_arr: np.ndarray | None = None
    bse_arr: np.ndarray | None = None
    try:
        params_arr = np.asarray(result.params, dtype=np.float64)
        bse_attr = result.bse
        bse_arr = np.asarray(
            bse_attr() if callable(bse_attr) else bse_attr,
            dtype=np.float64,
        )
    except (AttributeError, TypeError):
        # Deserialized / minimal models may not carry params/bse — proceed
        # without coefficient-block significance and GVIF; the surface grid
        # itself is still computable.
        params_arr = None
        bse_arr = None

    # Fallback for direct internal calls. The orchestrator passes these arrays
    # after extracting them once for both train/test comparison and pair grids.
    test_y_arr = test_y
    test_mu_arr = test_mu
    test_exposure_arr = test_exposure
    if test_data is not None and test_y_arr is None:
        if response_col and response_col in test_data.columns:
            test_y_arr = test_data[response_col].to_numpy().astype(np.float64)
        if hasattr(result, "predict"):
            test_mu_arr = np.asarray(
                result.predict(test_data, response_ceiling=None, on_extreme_eta="clip"),
                dtype=np.float64,
            )
        if exposure_col and exposure_col in test_data.columns:
            test_exposure_arr = test_data[exposure_col].to_numpy().astype(np.float64)
        elif test_y_arr is not None:
            test_exposure_arr = np.ones(len(test_y_arr), dtype=np.float64)

    return computer.compute_pair_diagnostics(
        pairs=list(interactions),
        data=train_data,
        model=result,
        design_matrix=score_test_design_matrix,
        bread_matrix=score_test_bread_matrix,
        params=params_arr,
        bse=bse_arr,
        correlation_matrix=correlation_matrix,
        test_data=test_data,
        test_y=test_y_arr,
        test_mu=test_mu_arr,
        test_exposure=test_exposure_arr,
        link=link,
    )


def _extract_test_arrays(
    test_data: pl.DataFrame | None,
    result: Any,
    response_col: str | None,
    exposure_col: str | None,
    ranking: str = "auto",
    exposure_override: str | np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Extract test response, prediction, exposure, and prediction order once."""
    if test_data is None or response_col is None:
        return None, None, None, None
    if response_col not in test_data.columns:
        raise ValidationError(f"Response column '{response_col}' not found in test_data")
    if not hasattr(result, "predict"):
        raise ValidationError("Model does not support prediction on new data")

    y_test = test_data[response_col].to_numpy().astype(np.float64)
    override_arr = None
    if exposure_override is not None and not isinstance(exposure_override, str):
        override_arr = np.asarray(exposure_override, dtype=np.float64)
        if override_arr.ndim != 1:
            raise ValidationError(
                f"exposure must be one-dimensional; got shape {override_arr.shape}."
            )

    predict_exposure: str | np.ndarray | None
    if isinstance(exposure_override, str):
        predict_exposure = exposure_override
    elif override_arr is not None and override_arr.shape[0] == len(test_data):
        predict_exposure = override_arr
    elif override_arr is not None:
        predict_exposure = None
    elif getattr(result, "_exposure_spec", None) is not None and not isinstance(
        getattr(result, "_exposure_spec", None), str
    ):
        raise ValidationError(
            "test_data diagnostics for a model fit with array exposure require "
            "an explicit exposure= array for the test data."
        )
    else:
        predict_exposure = None
    # Test-data diagnostics score the model's true mean, uncapped; clip (not
    # raise) on extreme eta so the report survives extrapolating test rows.
    mu_test = np.asarray(
        result.predict(
            test_data,
            exposure=predict_exposure,
            response_ceiling=None,
            on_extreme_eta="clip",
        ),
        dtype=np.float64,
    )
    exposure_test = np.ones(len(y_test), dtype=np.float64)
    has_exposure = False
    if isinstance(exposure_override, str):
        exposure_col = exposure_override
    if exposure_col:
        data_for_exposure = _data_for_model_spec(result, test_data, exposure_col)
    else:
        data_for_exposure = test_data
    if exposure_col and exposure_col in data_for_exposure.columns:
        exposure_test = data_for_exposure[exposure_col].to_numpy().astype(np.float64)
        has_exposure = True
    elif override_arr is not None and override_arr.shape[0] == len(y_test):
        exposure_test = override_arr.astype(np.float64)
        has_exposure = True
    return (
        y_test,
        mu_test,
        exposure_test,
        rank_sort_idx(mu_test, exposure_test, has_exposure=has_exposure, ranking=ranking),
    )


def _maybe_compute_vif(
    compute_vif: bool,
    computer: DiagnosticsComputer,
    design_matrix: Any,
    feature_names: list[str],
    correlation_context: Any = None,
) -> Any:
    """Compute VIF/multicollinearity scores when enabled and design matrix available."""
    if not compute_vif:
        return None
    if correlation_context is not None:
        vif_results = computer.compute_vif_from_correlation_context(
            correlation_context,
            feature_names,
        )
        if vif_results is not None:
            return vif_results
    if design_matrix is None:
        return None
    return computer.compute_vif(design_matrix, feature_names)


def _maybe_compute_coefficients(
    compute_coefficients: bool,
    compute_robust_se: bool,
    computer: DiagnosticsComputer,
    result: Any,
    link: str,
    warnings: list | None = None,
) -> tuple[Any, bool]:
    """Build coefficient summary, optionally enriching with HC1 robust SEs.

    When ``compute_robust_se=True`` but robust SEs can't be computed (lean
    model, regularized fit, etc.), append an informational warning to
    ``warnings`` so the caller knows why ``robust_*`` fields are null —
    rather than silently leaving them ``None``.
    """
    if not compute_coefficients:
        return None, False
    inference_status = getattr(result, "inference_status", None)
    if inference_status is not None and inference_status not in {"valid_standard", "valid_robust"}:
        if warnings is not None:
            warnings.append(
                {
                    "type": "coefficient_inference_unavailable",
                    "message": (
                        "Coefficient p-values and standard errors are suppressed because "
                        f"inference_status={inference_status}."
                    ),
                }
            )
        return None, False
    coef_summary = computer.compute_coefficient_summary(result, link=link)
    robust_se_enriched = False
    if compute_robust_se:
        robust_se_enriched, reason = computer.enrich_coefficient_summary_with_robust(
            coef_summary, result, cov_type="HC1"
        )
        if not robust_se_enriched and reason is not None and warnings is not None:
            warnings.append({"type": "robust_se_unavailable", "message": reason})
    return coef_summary, robust_se_enriched


def _maybe_compute_factor_deviance(
    compute_deviance_by_level: bool,
    computer: DiagnosticsComputer,
    train_data: pl.DataFrame,
    categorical_factors: list[str],
    cat_cache: dict,
    cat_unique_cache: dict,
    warnings: list,
) -> Any:
    """Compute deviance contribution per categorical level and warn on problem levels."""
    if not (compute_deviance_by_level and categorical_factors):
        return None
    factor_dev = computer.compute_factor_deviance(
        train_data,
        categorical_factors,
        cat_column_cache=cat_cache,
        cat_unique_cache=cat_unique_cache,
    )
    for fd in factor_dev:
        if fd.problem_levels:
            warnings.append(
                {
                    "type": "problem_factor_levels",
                    "message": f"Factor '{fd.factor}' has problem levels with poor fit: "
                    f"{', '.join(fd.problem_levels[:5])}{'...' if len(fd.problem_levels) > 5 else ''}",
                }
            )
    return factor_dev


def _maybe_compute_lift(
    compute_lift: bool,
    computer: DiagnosticsComputer,
    mu_sort_idx: np.ndarray,
    warnings: list,
    ranking: str,
) -> Any:
    """Compute the full lift chart and warn on weak deciles."""
    if not compute_lift:
        return None
    lift_chart = computer.compute_lift_chart(n_deciles=10, sort_idx=mu_sort_idx, ranking=ranking)
    if lift_chart.weak_deciles:
        warnings.append(
            {
                "type": "weak_discrimination",
                "message": f"Model has weak discrimination in deciles: {lift_chart.weak_deciles}. "
                f"Consider adding features or interactions to improve separation.",
            }
        )
    return lift_chart


def _maybe_compute_partial_dependence(
    compute_partial_dep: bool,
    computer: DiagnosticsComputer,
    train_data: pl.DataFrame,
    result: Any,
    continuous_factors: list[str],
    categorical_factors: list[str],
    link: str,
    cat_cache: dict,
    cat_unique_cache: dict,
    cont_cache: dict,
    warnings: list,
) -> Any:
    """Compute partial dependence and warn on non-linear effects with spline recommendations."""
    if not (compute_partial_dep and (continuous_factors or categorical_factors)):
        return None
    partial_dep = computer.compute_partial_dependence(
        data=train_data,
        result=result,
        continuous_factors=continuous_factors,
        categorical_factors=categorical_factors,
        link=link,
        cat_column_cache=cat_cache,
        cat_unique_cache=cat_unique_cache,
        cont_column_cache=cont_cache,
    )
    for pd in partial_dep:
        if (
            pd.shape in ("u_shaped", "inverted_u", "complex")
            and "spline" in pd.recommendation.lower()
        ):
            warnings.append(
                {
                    "type": "nonlinear_effect",
                    "message": f"Variable '{pd.variable}' shows {pd.shape} pattern. {pd.recommendation}",
                }
            )
    return partial_dep


def _build_train_test_comparison(
    train_diag: Any,
    test_data: pl.DataFrame | None,
    result: Any,
    computer: DiagnosticsComputer,
    response_col: str | None,
    exposure_col: str | None,
    categorical_factors: list[str],
    continuous_factors: list[str],
    warnings: list,
    test_y: np.ndarray | None = None,
    test_mu: np.ndarray | None = None,
    test_exposure: np.ndarray | None = None,
    test_mu_sort_idx: np.ndarray | None = None,
    test_weights: np.ndarray | None = None,
    test_base_mu: np.ndarray | None = None,
    ranking: str = "auto",
) -> TrainTestComparison:
    """Assemble the train/test comparison; populate test_diag and overfitting flags when test_data is supplied."""
    train_test = TrainTestComparison(train=train_diag)
    if test_data is None or response_col is None:
        return train_test

    if test_y is None or test_mu is None or test_exposure is None:
        test_y, test_mu, test_exposure, test_mu_sort_idx = _extract_test_arrays(
            test_data,
            result,
            response_col,
            exposure_col,
            ranking,
        )
    if test_y is None or test_mu is None or test_exposure is None:
        return train_test

    y_test = test_y
    mu_test = test_mu
    exposure_test = test_exposure
    # Pre-compute the rank index of the test predictions once. RS-ACT-004: rank
    # by predicted rate (mu/exposure); exposure_test is ones when absent, so this
    # reduces to argsort(mu) for non-exposure models.
    if test_mu_sort_idx is None:
        has_exposure = exposure_col is not None and exposure_col in test_data.columns
        test_mu_sort_idx = rank_sort_idx(
            mu_test,
            exposure_test,
            has_exposure=has_exposure,
            ranking=ranking,
        )

    # Pre-cache test data columns (same pattern as the train-data cache loop).
    cat_cache_test, cat_unique_cache_test, cont_cache_test = _precompute_data_caches(
        test_data, categorical_factors, continuous_factors
    )

    # Compute test diagnostics
    test_diag = computer.compute_dataset_diagnostics(
        y_test,
        mu_test,
        exposure_test,
        test_data,
        categorical_factors,
        continuous_factors,
        "test",
        result,
        cat_column_cache=cat_cache_test,
        cont_column_cache=cont_cache_test,
        cat_unique_cache=cat_unique_cache_test,
        sort_idx=test_mu_sort_idx,
        weights=test_weights,
        base_mu=test_base_mu,
    )

    # Compute comparison metrics
    gini_gap = train_diag.gini - test_diag.gini
    ae_ratio_diff = abs(train_diag.ae_ratio - test_diag.ae_ratio)

    # Decile comparison
    decile_comparison = []
    for i in range(min(len(train_diag.ae_by_decile), len(test_diag.ae_by_decile))):
        train_d = train_diag.ae_by_decile[i]
        test_d = test_diag.ae_by_decile[i]
        decile_comparison.append(
            {
                "decile": i + 1,
                "train_ae": train_d.ae_ratio,
                "test_ae": test_d.ae_ratio,
                "ae_diff": round(abs((train_d.ae_ratio or 0) - (test_d.ae_ratio or 0)), 4),
            }
        )

    # Factor divergence
    factor_divergence: dict = {}
    unstable_factors_list: list[str] = []
    for factor in categorical_factors:
        if factor in train_diag.factor_diagnostics and factor in test_diag.factor_diagnostics:
            train_levels = {m.level: m for m in train_diag.factor_diagnostics[factor]}
            test_levels = {m.level: m for m in test_diag.factor_diagnostics[factor]}
            divergent = []
            for level in set(train_levels.keys()) | set(test_levels.keys()):
                tr_ae = train_levels.get(
                    level, FactorLevelMetrics(level, 0, 0, 0, 0, None, 0)
                ).ae_ratio
                te_ae = test_levels.get(
                    level, FactorLevelMetrics(level, 0, 0, 0, 0, None, 0)
                ).ae_ratio
                if tr_ae is not None and te_ae is not None:
                    diff = abs(tr_ae - te_ae)
                    if diff > FACTOR_AE_DIFF_THRESHOLD:
                        divergent.append(
                            {
                                "level": level,
                                "train_ae": tr_ae,
                                "test_ae": te_ae,
                                "ae_diff": round(diff, 4),
                            }
                        )
                        unstable_factors_list.append(f"{factor}[{level}]")
            if divergent:
                factor_divergence[factor] = divergent

    # Flags
    overfitting_risk = gini_gap > OVERFITTING_GINI_GAP_THRESHOLD
    calibration_drift = (
        test_diag.ae_ratio < CALIBRATION_AE_LOWER or test_diag.ae_ratio > CALIBRATION_AE_UPPER
    )

    train_test = TrainTestComparison(
        train=train_diag,
        test=test_diag,
        gini_gap=round(gini_gap, 4),
        ae_ratio_diff=round(ae_ratio_diff, 4),
        decile_comparison=decile_comparison,
        factor_divergence=factor_divergence,
        overfitting_risk=overfitting_risk,
        calibration_drift=calibration_drift,
        unstable_factors=unstable_factors_list,
    )

    # Add warnings based on flags
    if overfitting_risk:
        warnings.append(
            {
                "type": "overfitting",
                "message": f"Overfitting detected: Train Gini={train_diag.gini:.3f}, "
                f"Test Gini={test_diag.gini:.3f} (gap={gini_gap:.3f}). "
                f"Consider reducing model complexity or using regularization.",
            }
        )
    if calibration_drift:
        warnings.append(
            {
                "type": "calibration_drift",
                "message": f"Calibration drift: Test A/E={test_diag.ae_ratio:.3f} "
                f"(outside [0.95, 1.05]). Model may not generalize well.",
            }
        )
    if unstable_factors_list:
        warnings.append(
            {
                "type": "unstable_factors",
                "message": f"Unstable factor levels (train/test A/E differ by >0.1): "
                f"{', '.join(unstable_factors_list[:10])}"
                f"{'...' if len(unstable_factors_list) > 10 else ''}",
            }
        )

    return train_test


def _compute_overdispersion(
    family: str,
    result: Any,
    computer: DiagnosticsComputer,
    y: np.ndarray,
    warnings: list,
) -> dict | None:
    """Compute Pearson-based overdispersion stats for Poisson/Binomial families and warn if present."""
    family_lower = family.lower()
    if not any(f in family_lower for f in ["poisson", "binomial", "negativebinomial"]):
        return None

    pearson_chi2 = result.pearson_chi2() if hasattr(result, "pearson_chi2") else None
    df_resid = computer.df_resid
    if pearson_chi2 is None or df_resid <= 0:
        return None

    pearson_dispersion = pearson_chi2 / df_resid

    # Also compute raw dispersion from data (Var/Mean for counts)
    mean_count = float(np.mean(y))
    var_count = float(np.var(y, ddof=1)) if len(y) > 1 else 0.0
    raw_dispersion = var_count / mean_count if mean_count > 0 else 1.0

    # Severity based on Pearson dispersion (more reliable)
    severity, recommendation = _classify_overdispersion(pearson_dispersion)

    overdispersion_result = {
        "pearson_dispersion": round(pearson_dispersion, 4),
        "pearson_chi2": round(pearson_chi2, 2),
        "df_resid": df_resid,
        "raw_dispersion": round(raw_dispersion, 4),
        "mean_count": round(mean_count, 4),
        "var_count": round(var_count, 4),
        "severity": severity,
        "recommendation": recommendation,
    }

    # Add warning if overdispersed
    if pearson_dispersion > OVERDISPERSION_MILD:
        warnings.append(
            {
                "type": "overdispersion",
                "message": f"Overdispersion detected (φ={pearson_dispersion:.2f}). {recommendation}",
            }
        )
    return overdispersion_result


def _extract_spline_info(result: Any) -> Any:
    """Return spline knot info from the model builder when present."""
    if not (hasattr(result, "_builder") and hasattr(result._builder, "get_spline_info")):
        return None
    spline_info = result._builder.get_spline_info()
    if not spline_info:
        return None
    return spline_info


def _compute_smooth_diagnostics(result: Any, warnings: list) -> Any:
    """Compute smooth term diagnostics if the model has smooth terms."""
    if hasattr(result, "has_smooth_terms") and result.has_smooth_terms():
        return _compute_smooth_term_diagnostics(result, warnings)
    return None


def _level_set(data: pl.DataFrame | None, factors: list[str]) -> set[str]:
    """Return observed level keys for one categorical factor or encoded interaction."""
    if data is None or not factors or any(f not in data.columns for f in factors):
        return set()
    if len(factors) == 1:
        return set(str(v) for v in data[factors[0]].cast(pl.Utf8).unique().to_list())
    return set(
        data.select(
            pl.concat_str([pl.col(f).cast(pl.Utf8) for f in factors], separator=":").alias("_level")
        )
        .get_column("_level")
        .unique()
        .to_list()
    )


def _encoding_kind_from_slot(slot: Any) -> str:
    if slot.term_type in {"categorical", "categorical_indicator"}:
        return "categorical"
    if slot.term_type == "target_encoding":
        return "target_encoding"
    if slot.term_type == "frequency_encoding":
        return "frequency_encoding"
    return "unknown"


def _maybe_grouped_kind(name: str, kind: str, notes: list[str]) -> str:
    grouped_suffixes = ("_grp", "_group", "_grouped", "_band", "_bucket")
    if kind in {"categorical", "unknown"} and name.endswith(grouped_suffixes):
        notes.append("kind='grouped_categorical' is inferred from the column name suffix only.")
        return "grouped_categorical"
    return kind


def _compute_encoding_diagnostics(
    result: Any,
    train_data: pl.DataFrame,
    test_data: pl.DataFrame | None,
    categorical_factors: list[str],
) -> list[EncodingDiagnostics] | None:
    """Expose fitted categorical/encoding representation without report opinions."""
    builder = getattr(result, "_builder", None)
    slots = getattr(builder, "_term_slots", None) if builder is not None else None
    entries: list[EncodingDiagnostics] = []
    seen: set[str] = set()

    if slots is not None:
        for slot in slots:
            if slot.term_type not in {
                "categorical",
                "categorical_indicator",
                "target_encoding",
                "frequency_encoding",
            }:
                continue
            source_factors = list(getattr(slot, "factors", []) or [])
            name = str(getattr(slot, "term_name", "") or ":".join(source_factors))
            notes: list[str] = []
            kind = _maybe_grouped_kind(name, _encoding_kind_from_slot(slot), notes)
            train_levels = _level_set(train_data, source_factors)
            test_levels = _level_set(test_data, source_factors)
            entries.append(
                EncodingDiagnostics(
                    name=name,
                    kind=kind,
                    in_model=True,
                    n_levels_train=len(train_levels) if train_levels else None,
                    n_levels_test=len(test_levels) if test_levels else None,
                    unseen_levels_test=len(test_levels - train_levels)
                    if test_levels and train_levels
                    else None,
                    interaction_order=max(1, len(source_factors)),
                    source_factors=source_factors,
                    feature_names=list(getattr(slot, "design_column_names", []) or []),
                    notes=notes,
                )
            )
            seen.add(name)
            seen.update(source_factors)

    for factor in categorical_factors:
        if factor in seen:
            continue
        notes = []
        kind = _maybe_grouped_kind(factor, "unknown", notes)
        train_levels = _level_set(train_data, [factor])
        test_levels = _level_set(test_data, [factor])
        entries.append(
            EncodingDiagnostics(
                name=factor,
                kind=kind,
                in_model=False,
                n_levels_train=len(train_levels) if train_levels else None,
                n_levels_test=len(test_levels) if test_levels else None,
                unseen_levels_test=len(test_levels - train_levels)
                if test_levels and train_levels
                else None,
                source_factors=[factor],
                notes=notes,
            )
        )

    return entries or None


def _resolve_base_predictions_column(
    base_predictions: str | dict[str, str] | None,
    role: str,
) -> str | None:
    """Resolve a base/benchmark prediction column for a dataset role."""
    if base_predictions is None:
        return None
    if isinstance(base_predictions, str):
        return base_predictions
    if isinstance(base_predictions, dict):
        value = base_predictions.get(role)
        return str(value) if value is not None else None
    raise ValidationError(
        "base_predictions must be a column name or a {'train': ..., 'test': ...} mapping."
    )


def _extract_base_predictions_array(
    base_predictions: str | dict[str, str] | None,
    role: str,
    data: pl.DataFrame | None,
    expected_len: int | None,
    warnings: list[dict[str, str]],
    required: bool,
) -> np.ndarray | None:
    """Extract response-scale benchmark predictions for one dataset role."""
    if base_predictions is None or data is None:
        return None
    column = _resolve_base_predictions_column(base_predictions, role)
    if column is None:
        return None
    if column not in data.columns:
        message = f"base_predictions column '{column}' not found in {role}_data"
        if required:
            raise ValidationError(message)
        warnings.append(
            {
                "type": "base_predictions_unavailable",
                "message": f"{message}; leaving {role}-side base comparison empty.",
            }
        )
        return None
    values = data[column].to_numpy().astype(np.float64)
    if expected_len is not None and values.shape[0] != expected_len:
        raise ValidationError(
            f"base_predictions column '{column}' in {role}_data has {values.shape[0]} "
            f"rows but expected {expected_len}."
        )
    return values


def _compute_base_comparison_for_role(
    mu_base: np.ndarray | None,
    role: str,
    computer: DiagnosticsComputer,
    y: np.ndarray,
    mu: np.ndarray,
    exposure: np.ndarray,
    weights: np.ndarray | None,
    warnings: list,
    ranking: str,
    required: bool,
    emit_performance_warning: bool = False,
) -> Any:
    """Compute base predictions comparison (model vs. another set of predictions)."""
    if mu_base is None:
        return None
    if mu_base.shape[0] != y.shape[0]:
        raise ValidationError(
            f"{role}-side base_predictions has {mu_base.shape[0]} rows but expected {y.shape[0]}."
        )
    base_predictions_comparison = computer.compute_base_predictions_comparison(
        y=y,
        mu_model=mu,
        mu_base=mu_base,
        exposure=exposure,
        weights=weights,
        ranking=ranking,
    )
    if not emit_performance_warning:
        return base_predictions_comparison
    if base_predictions_comparison.loss_improvement_pct > 0:
        warnings.append(
            {
                "type": "model_improvement",
                "message": f"Model improves on base predictions: {base_predictions_comparison.loss_improvement_pct:.1f}% lower loss, "
                f"better A/E in {base_predictions_comparison.model_better_deciles}/10 deciles",
            }
        )
    elif base_predictions_comparison.loss_improvement_pct < 0:
        warnings.append(
            {
                "type": "model_regression",
                "message": f"Model is worse than base predictions: {-base_predictions_comparison.loss_improvement_pct:.1f}% higher loss, "
                f"better A/E in only {base_predictions_comparison.model_better_deciles}/10 deciles",
            }
        )
    return base_predictions_comparison


def _compute_base_predictions_by_role(
    base_predictions: str | dict[str, str] | None,
    train_data: pl.DataFrame,
    test_data: pl.DataFrame | None,
    computer: DiagnosticsComputer,
    train_base_mu: np.ndarray | None,
    test_base_mu: np.ndarray | None,
    train_y: np.ndarray,
    train_mu: np.ndarray,
    train_exposure: np.ndarray,
    train_weights: np.ndarray | None,
    test_y: np.ndarray | None,
    test_mu: np.ndarray | None,
    test_exposure: np.ndarray | None,
    test_weights: np.ndarray | None,
    warnings: list,
    ranking: str,
) -> BasePredictionsByRole | None:
    """Compute base/benchmark prediction comparisons for train and optional test roles."""
    if base_predictions is None:
        return None

    train_col = _resolve_base_predictions_column(base_predictions, "train")
    if train_col is None:
        raise ValidationError("base_predictions mapping must include a 'train' column.")

    train_comparison = _compute_base_comparison_for_role(
        mu_base=train_base_mu,
        role="train",
        computer=computer,
        y=train_y,
        mu=train_mu,
        exposure=train_exposure,
        weights=train_weights,
        warnings=warnings,
        ranking=ranking,
        required=True,
        emit_performance_warning=True,
    )

    test_comparison = None
    if (
        test_data is not None
        and test_y is not None
        and test_mu is not None
        and test_exposure is not None
    ):
        # String input means "same column on both roles"; a dict without a
        # "test" key means the caller intentionally provided train-only base
        # data. Either way a named-but-missing test column has already produced
        # a "base_predictions_unavailable" warning (and a None ``test_base_mu``)
        # in ``_extract_base_predictions_array``, so this is best-effort.
        test_col = _resolve_base_predictions_column(base_predictions, "test")
        if test_col is not None:
            test_comparison = _compute_base_comparison_for_role(
                mu_base=test_base_mu,
                role="test",
                computer=computer,
                y=test_y,
                mu=test_mu,
                exposure=test_exposure,
                weights=test_weights,
                warnings=warnings,
                ranking=ranking,
                required=False,
            )

    return BasePredictionsByRole(
        train=train_comparison,
        test=test_comparison,
        ranking=ranking,
        prediction_basis="response",
    )


# TermSlot term_type values that represent a *main effect* on a single
# variable (not an interaction). A factor's design-column range comes from
# the slot matching ``len(slot.factors) == 1``.
_MAIN_EFFECT_TERM_TYPES: frozenset[str] = frozenset(
    {
        "linear",
        "categorical",
        "bs",
        "ns",
        "ms",
        "expression",
        "target_encoding",
        "frequency_encoding",
    }
)

_INTERACTION_SPEC_RESERVED_KEYS: frozenset[str] = frozenset(
    {
        "factor1",
        "factor2",
        "factor3",
        "include_main",
        "target_encoding",
        "frequency_encoding",
        "prior_weight",
    }
)


def _find_main_effect_slot(model: Any, factor_name: str) -> Any | None:
    """Return the single-variable main-effect ``TermSlot`` for ``factor_name``,
    or ``None`` if the factor is not in the model.
    """
    builder = getattr(model, "_builder", None)
    if builder is None:
        return None
    slots = getattr(builder, "_term_slots", None)
    if slots is None:
        return None
    for slot in slots:
        if slot.term_type not in _MAIN_EFFECT_TERM_TYPES:
            continue
        if len(slot.factors) == 1 and slot.factors[0] == factor_name:
            return slot
    return None


def _extract_interaction_spec_factors(spec: Any) -> list[str]:
    """Extract raw factor names from pair or higher-order interaction specs."""
    if isinstance(spec, dict):
        ordered = [
            str(spec[k])
            for k in ("factor1", "factor2", "factor3")
            if k in spec and spec[k] is not None
        ]
        if ordered:
            return ordered
        return [str(k) for k in spec if k not in _INTERACTION_SPEC_RESERVED_KEYS]
    if isinstance(spec, (tuple, list)):
        return [str(v) for v in spec]
    return []


def _split_interaction_specs(interactions: list[Any] | None) -> tuple[list[Any], list[list[str]]]:
    """Split user-specified interactions into legacy pairs and block requests."""
    pairs: list[Any] = []
    blocks: list[list[str]] = []
    for spec in interactions or []:
        factors = _extract_interaction_spec_factors(spec)
        if len(factors) == 2:
            pairs.append(spec)
        elif len(factors) > 2:
            blocks.append(factors)
        else:
            pairs.append(spec)
    return pairs, blocks


def _resolve_diagnostics_interactions(
    result: Any,
    interactions: list[Any] | None,
    include_fitted_interactions: bool,
) -> list[Any] | None:
    """Merge caller-requested interactions with fitted model interactions."""
    if not include_fitted_interactions:
        return interactions
    fitted = list(getattr(result, "_interactions_spec", None) or [])
    if not fitted:
        return interactions
    if interactions is None:
        return fitted
    resolved = list(interactions)
    for spec in fitted:
        if spec not in resolved:
            resolved.append(spec)
    return resolved


def _interaction_diagnostics_data(
    result: Any,
    train_data: pl.DataFrame,
    test_data: pl.DataFrame | None,
    interaction_specs: list[Any] | None,
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Prepare data for interaction diagnostics when specs use transform outputs."""
    if not interaction_specs or not hasattr(result, "prepare_input"):
        return train_data, test_data
    transform_outputs = {
        spec.get("output")
        for spec in getattr(result, "_input_transforms", [])
        if isinstance(spec, dict) and spec.get("output")
    }
    if not transform_outputs:
        return train_data, test_data
    factors = {
        factor for spec in interaction_specs for factor in _extract_interaction_spec_factors(spec)
    }
    needed_outputs = factors & transform_outputs
    if not needed_outputs:
        return train_data, test_data
    train_has_outputs = needed_outputs.issubset(set(train_data.columns))
    test_has_outputs = test_data is None or needed_outputs.issubset(set(test_data.columns))
    if train_has_outputs and test_has_outputs:
        return train_data, test_data
    prepared_train = train_data if train_has_outputs else result.prepare_input(train_data)
    prepared_test = (
        test_data if test_has_outputs or test_data is None else result.prepare_input(test_data)
    )
    return prepared_train, prepared_test


def _find_interaction_slot(model: Any, factors: list[str]) -> Any | None:
    builder = getattr(model, "_builder", None)
    slots = getattr(builder, "_term_slots", None) if builder is not None else None
    if slots is None:
        return None
    target = frozenset(factors)
    for slot in slots:
        if slot.term_type not in {"interaction", "target_encoding", "frequency_encoding"}:
            continue
        if len(slot.factors) == len(factors) and frozenset(slot.factors) == target:
            return slot
    return None


def _interaction_representation(slot: Any | None) -> str | None:
    if slot is None:
        return None
    if slot.term_type == "interaction":
        return "tensor_product"
    if slot.term_type == "target_encoding":
        return "target_encoding"
    if slot.term_type == "frequency_encoding":
        return "frequency_encoding"
    return "unknown"


def _coefficients_for_slot(model: Any, slot: Any) -> list[FactorCoefficient] | None:
    try:
        params = np.asarray(model.params, dtype=np.float64)
        bse_attr = model.bse
        bse = np.asarray(bse_attr() if callable(bse_attr) else bse_attr, dtype=np.float64)
        pvalues = None
        if hasattr(model, "pvalues"):
            pv_attr = model.pvalues
            pvalues = np.asarray(pv_attr() if callable(pv_attr) else pv_attr, dtype=np.float64)
    except (AttributeError, TypeError, ValueError):
        return None

    coefficients: list[FactorCoefficient] = []
    for idx in range(slot.col_start, slot.col_end):
        coef = float(params[idx])
        se = float(bse[idx]) if idx < len(bse) else 0.0
        z_val = coef / se if se > 0 else 0.0
        p_val = float(pvalues[idx]) if pvalues is not None and idx < len(pvalues) else 0.0
        term = (
            slot.design_column_names[idx - slot.col_start]
            if idx - slot.col_start < len(slot.design_column_names)
            else f"{slot.term_name}[{idx - slot.col_start}]"
        )
        coefficients.append(
            FactorCoefficient(
                term=term,
                estimate=round(coef, 6),
                std_error=round(se, 6),
                z_value=round(z_val, 3),
                p_value=round(p_val, 4),
                relativity=None,
            )
        )
    return coefficients or None


def _significance_for_slot(
    model: Any, slot: Any, bread_matrix: np.ndarray | None
) -> FactorSignificance | None:
    try:
        params = np.asarray(model.params, dtype=np.float64)
        bse_attr = model.bse
        bse = np.asarray(bse_attr() if callable(bse_attr) else bse_attr, dtype=np.float64)
    except (AttributeError, TypeError, ValueError):
        return None

    idx = np.arange(slot.col_start, slot.col_end)
    if idx.size == 0:
        return None
    beta = params[idx]
    try:
        if bread_matrix is not None and idx.size > 1:
            bread_sub = bread_matrix[np.ix_(idx, idx)]
            scale = 1.0
            for i in idx:
                if bread_matrix[i, i] > 0 and bse[i] > 0:
                    scale = (bse[i] ** 2) / bread_matrix[i, i]
                    break
            cov_inv = np.linalg.pinv(scale * bread_sub)
            chi2 = float(beta @ cov_inv @ beta)
        else:
            chi2 = float(
                np.sum(np.divide(beta, bse[idx], out=np.zeros_like(beta), where=bse[idx] > 0) ** 2)
            )
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return None
    pvalue = 1.0 - _chi2_cdf(chi2, float(idx.size))
    return FactorSignificance(chi2=round(chi2, 2), p=round(pvalue, 4), dev_contrib=round(chi2, 2))


def _compute_interaction_block_diagnostics(
    block_specs: list[list[str]],
    model: Any,
    bread_matrix: np.ndarray | None,
    correlation_matrix: np.ndarray | None,
    warnings: list[dict[str, str]],
) -> list[InteractionBlockDiagnostics]:
    blocks: list[InteractionBlockDiagnostics] = []
    for factors in block_specs:
        name = ":".join(factors)
        slot = _find_interaction_slot(model, factors)
        if slot is None:
            warnings.append(
                {
                    "type": "interaction_score_test_unavailable",
                    "message": (
                        f"Score test for higher-order interaction '{name}' is not available "
                        "without safely materializing the expanded design block."
                    ),
                }
            )
            blocks.append(
                InteractionBlockDiagnostics(
                    name=name,
                    factors=factors,
                    order=len(factors),
                    in_model=False,
                    representation=None,
                )
            )
            continue

        gvif = (
            _compute_block_gvif(correlation_matrix, slot.col_start, slot.col_end)
            if correlation_matrix is not None and slot.col_end > slot.col_start
            else None
        )
        blocks.append(
            InteractionBlockDiagnostics(
                name=name,
                factors=list(slot.factors),
                order=len(slot.factors),
                in_model=True,
                representation=_interaction_representation(slot),
                coefficients=_coefficients_for_slot(model, slot),
                significance=_significance_for_slot(model, slot, bread_matrix),
                gvif=gvif,
            )
        )
    return blocks


def _build_factor_bin_pairs(
    factor_name: str,
    factor_type: str,
    train_diag: Any,
    test_diag: Any,
    test_data: pl.DataFrame | None = None,
    test_y: np.ndarray | None = None,
    test_mu: np.ndarray | None = None,
    test_exposure: np.ndarray | None = None,
    test_weights: np.ndarray | None = None,
    test_base_mu: np.ndarray | None = None,
) -> list[FactorBinPair] | None:
    """Construct ``train_test_bins`` for a factor by joining the per-bin
    metrics that already exist on the train and test ``DatasetDiagnostics``.

    Returns ``None`` when the factor has no per-bin metrics in either side
    (typical for factors that were never passed in the ``categorical_factors``
    / ``continuous_factors`` lists).
    """
    if factor_type == "categorical":
        train_bins = train_diag.factor_diagnostics.get(factor_name, [])
        test_bins = test_diag.factor_diagnostics.get(factor_name, []) if test_diag else []
        if not train_bins and not test_bins:
            return None
        test_by_level: dict[str, Any] = {b.level: b for b in test_bins}
        pairs: list[FactorBinPair] = []
        for tb in train_bins:
            te = test_by_level.get(tb.level)
            pairs.append(
                FactorBinPair(
                    bin=tb.level,
                    train_n=int(tb.n),
                    train_exposure=float(tb.exposure),
                    train_actual=float(tb.actual),
                    train_predicted=float(tb.predicted),
                    train_ae_ratio=float(tb.ae_ratio) if tb.ae_ratio is not None else None,
                    test_n=int(te.n) if te is not None else 0,
                    test_exposure=float(te.exposure) if te is not None else 0.0,
                    test_actual=float(te.actual) if te is not None else None,
                    test_predicted=float(te.predicted) if te is not None else None,
                    test_ae_ratio=(
                        float(te.ae_ratio) if te is not None and te.ae_ratio is not None else None
                    ),
                    train_actual_total=getattr(tb, "actual_total", None),
                    train_predicted_total=getattr(tb, "predicted_total", None),
                    train_base_predicted=getattr(tb, "base_predicted", None),
                    train_base_predicted_total=getattr(tb, "base_predicted_total", None),
                    train_base_ae_ratio=getattr(tb, "base_ae_ratio", None),
                    test_actual_total=getattr(te, "actual_total", None) if te is not None else None,
                    test_predicted_total=(
                        getattr(te, "predicted_total", None) if te is not None else None
                    ),
                    test_base_predicted=(
                        getattr(te, "base_predicted", None) if te is not None else None
                    ),
                    test_base_predicted_total=(
                        getattr(te, "base_predicted_total", None) if te is not None else None
                    ),
                    test_base_ae_ratio=(
                        getattr(te, "base_ae_ratio", None) if te is not None else None
                    ),
                )
            )
        return pairs or None

    # continuous
    train_bands = train_diag.continuous_diagnostics.get(factor_name, [])
    test_bands = test_diag.continuous_diagnostics.get(factor_name, []) if test_diag else []
    if not train_bands and not test_bands:
        return None
    if (
        test_data is not None
        and test_y is not None
        and test_mu is not None
        and test_exposure is not None
        and factor_name in test_data.columns
    ):
        return _build_continuous_train_test_bin_pairs(
            factor_name,
            train_bands,
            test_data,
            test_y,
            test_mu,
            test_exposure,
            test_weights=test_weights,
            test_base_mu=test_base_mu,
        )
    test_by_band: dict[int, Any] = {b.band: b for b in test_bands}
    pairs2: list[FactorBinPair] = []
    for tb in train_bands:
        te = test_by_band.get(tb.band)
        label = f"{tb.range_min:.4g}-{tb.range_max:.4g}"
        pairs2.append(
            FactorBinPair(
                bin=label,
                train_n=int(tb.n),
                train_exposure=float(tb.exposure),
                train_actual=float(tb.actual),
                train_predicted=float(tb.predicted),
                train_ae_ratio=float(tb.ae_ratio) if tb.ae_ratio is not None else None,
                test_n=int(te.n) if te is not None else 0,
                test_exposure=float(te.exposure) if te is not None else 0.0,
                test_actual=float(te.actual) if te is not None else None,
                test_predicted=float(te.predicted) if te is not None else None,
                test_ae_ratio=(
                    float(te.ae_ratio) if te is not None and te.ae_ratio is not None else None
                ),
                train_actual_total=getattr(tb, "actual_total", None),
                train_predicted_total=getattr(tb, "predicted_total", None),
                train_base_predicted=getattr(tb, "base_predicted", None),
                train_base_predicted_total=getattr(tb, "base_predicted_total", None),
                train_base_ae_ratio=getattr(tb, "base_ae_ratio", None),
                test_actual_total=getattr(te, "actual_total", None) if te is not None else None,
                test_predicted_total=(
                    getattr(te, "predicted_total", None) if te is not None else None
                ),
                test_base_predicted=(
                    getattr(te, "base_predicted", None) if te is not None else None
                ),
                test_base_predicted_total=(
                    getattr(te, "base_predicted_total", None) if te is not None else None
                ),
                test_base_ae_ratio=(getattr(te, "base_ae_ratio", None) if te is not None else None),
            )
        )
    return pairs2 or None


def _build_continuous_train_test_bin_pairs(
    factor_name: str,
    train_bands: list[Any],
    test_data: pl.DataFrame,
    test_y: np.ndarray,
    test_mu: np.ndarray,
    test_exposure: np.ndarray,
    test_weights: np.ndarray | None = None,
    test_base_mu: np.ndarray | None = None,
) -> list[FactorBinPair] | None:
    """Join continuous train bands to test aggregates using train band edges."""
    if not train_bands:
        return None

    ordered_bands = sorted(train_bands, key=lambda b: b.band)
    edges = np.asarray(
        [float(ordered_bands[0].range_min)] + [float(b.range_max) for b in ordered_bands],
        dtype=np.float64,
    )
    if edges.size < 2 or not np.all(np.isfinite(edges)):
        return None

    values = test_data[factor_name].to_numpy().astype(np.float64)
    n_bins = len(ordered_bands)
    valid = np.isfinite(values)
    bin_idx = np.searchsorted(edges, values, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    safe_idx = np.where(valid, bin_idx, n_bins)

    counts = np.bincount(safe_idx, minlength=n_bins + 1)[:n_bins]
    weights_arr = (
        np.ones_like(test_y, dtype=np.float64)
        if test_weights is None
        else np.asarray(test_weights, dtype=np.float64)
    )
    exp_sums = np.bincount(safe_idx, weights=weights_arr * test_exposure, minlength=n_bins + 1)[
        :n_bins
    ]
    y_sums = np.bincount(safe_idx, weights=weights_arr * test_y, minlength=n_bins + 1)[:n_bins]
    mu_sums = np.bincount(safe_idx, weights=weights_arr * test_mu, minlength=n_bins + 1)[:n_bins]
    base_sums = (
        np.bincount(safe_idx, weights=weights_arr * test_base_mu, minlength=n_bins + 1)[:n_bins]
        if test_base_mu is not None
        else None
    )

    pairs: list[FactorBinPair] = []
    for i, tb in enumerate(ordered_bands):
        test_n = int(counts[i])
        test_exp = float(exp_sums[i])
        if test_n > 0 and test_exp > 0.0:
            test_actual = float(y_sums[i] / test_exp)
            test_predicted = float(mu_sums[i] / test_exp)
            test_ae_ratio = float(y_sums[i] / mu_sums[i]) if mu_sums[i] > 0.0 else None
            test_base_total = float(base_sums[i]) if base_sums is not None else None
            test_base_predicted = (
                test_base_total / test_exp if test_base_total is not None else None
            )
            test_base_ae_ratio = (
                float(y_sums[i] / test_base_total)
                if test_base_total is not None and test_base_total > 0.0
                else None
            )
        else:
            test_actual = None
            test_predicted = None
            test_ae_ratio = None
            test_base_total = None
            test_base_predicted = None
            test_base_ae_ratio = None

        pairs.append(
            FactorBinPair(
                bin=f"{float(tb.range_min):.4g}-{float(tb.range_max):.4g}",
                train_n=int(tb.n),
                train_exposure=float(tb.exposure),
                train_actual=float(tb.actual),
                train_predicted=float(tb.predicted),
                train_ae_ratio=float(tb.ae_ratio) if tb.ae_ratio is not None else None,
                test_n=test_n,
                test_exposure=test_exp,
                test_actual=test_actual,
                test_predicted=test_predicted,
                test_ae_ratio=test_ae_ratio,
                train_actual_total=getattr(tb, "actual_total", None),
                train_predicted_total=getattr(tb, "predicted_total", None),
                train_base_predicted=getattr(tb, "base_predicted", None),
                train_base_predicted_total=getattr(tb, "base_predicted_total", None),
                train_base_ae_ratio=getattr(tb, "base_ae_ratio", None),
                test_actual_total=float(y_sums[i]) if test_n > 0 else None,
                test_predicted_total=float(mu_sums[i]) if test_n > 0 else None,
                test_base_predicted=test_base_predicted,
                test_base_predicted_total=test_base_total,
                test_base_ae_ratio=test_base_ae_ratio,
            )
        )
    return pairs or None


def _annotate_factor_extensions(
    factors: list,
    train_diag: Any,
    test_diag: Any,
    correlation_matrix: np.ndarray | None,
    model: Any,
    test_data: pl.DataFrame | None = None,
    test_y: np.ndarray | None = None,
    test_mu: np.ndarray | None = None,
    test_exposure: np.ndarray | None = None,
    test_weights: np.ndarray | None = None,
    test_base_mu: np.ndarray | None = None,
) -> None:
    """Populate ``gvif`` and ``train_test_bins`` on each ``FactorDiagnostics``
    in place. Both are no-refit and reuse data that's already in flight.

    - ``gvif``: block GVIF (Fox-Monette) on the factor's design columns,
      computed against the pre-built regularized correlation matrix
      (one O(n·p²) standardize for the whole diagnostics call, then
      O(p³) determinants per factor).
    - ``train_test_bins``: per-bin train/test pairs joined from the
      already computed per-dataset diagnostics.
    """
    for factor in factors:
        # Block GVIF — only when the factor is fitted and we have the
        # pre-computed correlation matrix.
        if factor.in_model and correlation_matrix is not None:
            slot = _find_main_effect_slot(model, factor.name)
            if slot is not None and slot.col_end > slot.col_start:
                factor.gvif = _compute_block_gvif(correlation_matrix, slot.col_start, slot.col_end)

        # Train/test bin pairs — only when test side is present.
        if test_diag is not None:
            factor.train_test_bins = _build_factor_bin_pairs(
                factor.name,
                factor.factor_type,
                train_diag,
                test_diag,
                test_data=test_data,
                test_y=test_y,
                test_mu=test_mu,
                test_exposure=test_exposure,
                test_weights=test_weights,
                test_base_mu=test_base_mu,
            )


def _annotate_relative_importance(factors: list, model_deviance: float | None) -> None:
    """Mutate factors in-place: set relative_importance, dev_pct, expected_dev_pct."""
    fitted_with_sig = [
        f for f in factors if f.in_model and f.significance and f.significance.dev_contrib
    ]
    if fitted_with_sig:
        total_dev = sum(f.significance.dev_contrib for f in fitted_with_sig)
        if total_dev > 0:
            for f in fitted_with_sig:
                f.relative_importance = round(f.significance.dev_contrib / total_dev * 100, 2)

    if model_deviance and model_deviance > 0:
        for f in factors:
            if f.significance and f.significance.dev_contrib:
                f.significance.dev_pct = round(f.significance.dev_contrib / model_deviance * 100, 2)
            if f.score_test and f.score_test.statistic:
                f.score_test.expected_dev_pct = round(
                    f.score_test.statistic / model_deviance * 100, 2
                )


def _build_model_summary(
    result: Any,
    family: str,
    link: str,
    computer: DiagnosticsComputer,
    n_params: int,
    null_deviance: float | None,
    robust_se_enriched: bool,
) -> dict:
    """Construct the model_summary dict including regularization info when present."""
    if not hasattr(result, "converged"):
        raise ValidationError("Result object missing 'converged' attribute")
    if not hasattr(result, "iterations"):
        raise ValidationError("Result object missing 'iterations' attribute")
    if not hasattr(result, "formula"):
        raise ValidationError("Result object missing 'formula' attribute")

    model_summary = {
        "formula": result.formula,
        "family": family,
        "link": link,
        "n_obs": computer.n_obs,
        "n_params": n_params,
        "df_resid": computer.df_resid,
        "converged": result.converged,
        "iterations": result.iterations,
        "scale": round(result.scale(), 6)
        if hasattr(result, "scale") and callable(getattr(result, "scale", None))
        else None,
        "scale_pearson": round(result.scale_pearson(), 6)
        if hasattr(result, "scale_pearson") and callable(getattr(result, "scale_pearson", None))
        else None,
        "null_deviance": round(float(null_deviance), 2) if null_deviance is not None else None,
    }

    # Add regularization info if present (concise for LLM parsing)
    if hasattr(result, "alpha") and result.alpha > 0:
        reg_type = getattr(result, "regularization_type", None)
        if reg_type is None:
            l1 = getattr(result, "l1_ratio", 0)
            reg_type = "lasso" if l1 >= 1 else "ridge" if l1 <= 0 else "elastic_net"
        model_summary["regularization"] = {
            "type": reg_type,
            "alpha": round(result.alpha, 6),
            "l1_ratio": round(getattr(result, "l1_ratio", 0), 2),
        }
        # Add CV info if available
        if hasattr(result, "cv_deviance") and result.cv_deviance is not None:
            model_summary["regularization"]["cv_deviance"] = round(result.cv_deviance, 6)
            model_summary["regularization"]["cv_folds"] = getattr(result, "n_cv_folds", None)
            model_summary["regularization"]["selection"] = getattr(
                result, "cv_selection_method", None
            )

    if robust_se_enriched:
        model_summary["robust_se_type"] = "HC1"
    boundary_active = getattr(result, "boundary_active_coefficients", None)
    if boundary_active is not None:
        active = boundary_active() if callable(boundary_active) else boundary_active
        if active:
            model_summary["boundary_active_coefficients"] = active

    return model_summary


def _resolve_weights(
    result: Any,
    train_data: pl.DataFrame,
    weights_override: str | np.ndarray | None,
) -> np.ndarray | None:
    """Resolve prior weights for decile/lift aggregates (RS-ACT-004).

    An explicit ``weights_override`` wins and is validated strictly; otherwise
    the model's fitted ``_weights_spec`` is auto-propagated, but leniently — if
    that column/array can't be matched to ``train_data`` we fall back to
    unweighted rather than raising, since diagnostics data need not carry it.
    """
    explicit = weights_override is not None
    spec = weights_override if explicit else getattr(result, "_weights_spec", None)
    if spec is None:
        return None
    if isinstance(spec, str):
        data_for_weights = _data_for_model_spec(result, train_data, spec)
        if spec not in data_for_weights.columns:
            if explicit:
                raise ValidationError(f"weights column '{spec}' is not present in train_data.")
            return None
        return data_for_weights[spec].to_numpy().astype(np.float64)
    arr = np.asarray(spec, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != train_data.height:
        if explicit:
            raise ValidationError(
                f"weights length {arr.shape} does not match train_data length {train_data.height}."
            )
        return None
    return arr


def _resolve_test_weights(
    result: Any,
    test_data: pl.DataFrame | None,
    weights_override: str | np.ndarray | None,
    warnings: list[dict[str, str]],
) -> np.ndarray | None:
    """Resolve held-out prior weights without reusing training arrays.

    A named weights column can be propagated to ``test_data``. Array weights are
    train-row aligned, so they cannot safely describe held-out rows through the
    existing public API; use a column name present in both datasets for weighted
    train/test diagnostics.
    """
    if test_data is None:
        return None
    explicit = weights_override is not None
    spec = weights_override if explicit else getattr(result, "_weights_spec", None)
    if spec is None:
        return None
    if isinstance(spec, str):
        # Mirror the train side: a weights column produced by an input transform
        # only exists after transforms are applied, so resolve through the same
        # prepared-data helper rather than the raw test frame.
        data_for_weights = _data_for_model_spec(result, test_data, spec)
        if spec not in data_for_weights.columns:
            if explicit:
                raise ValidationError(f"weights column '{spec}' is not present in test_data.")
            warnings.append(
                {
                    "type": "test_weights_unavailable",
                    "message": (
                        f"Fitted weights column '{spec}' is not present in test_data; "
                        "held-out decile and Gini diagnostics are unweighted."
                    ),
                }
            )
            return None
        return data_for_weights[spec].to_numpy().astype(np.float64)

    warnings.append(
        {
            "type": "test_weights_unavailable",
            "message": (
                "Array weights are aligned to train_data and are not reused for "
                "test_data diagnostics; pass weights as a column name present in "
                "both datasets to weight held-out diagnostics."
            ),
        }
    )
    return None


def compute_diagnostics(
    result: Any,  # GLMResults or GLMModel
    train_data: pl.DataFrame,
    categorical_factors: list[str | None] | None = None,
    continuous_factors: list[str | None] | None = None,
    n_calibration_bins: int = DEFAULT_N_CALIBRATION_BINS,
    n_factor_bins: int = DEFAULT_N_FACTOR_BINS,
    rare_threshold_pct: float = DEFAULT_RARE_THRESHOLD_PCT,
    max_categorical_levels: int = DEFAULT_MAX_CATEGORICAL_LEVELS,
    detect_interactions: bool = False,
    max_interaction_factors: int = DEFAULT_MAX_INTERACTION_FACTORS,
    # User-specified interaction pairs for per-pair surface diagnostics.
    # Each entry: {"factor1": ..., "factor2": ...} OR (a, b) OR [a, b]. Pairs
    # do NOT need to appear in the fitted model — diagnostics work on raw
    # columns; ``in_model`` is set from TermSlot membership.
    interactions: list[Any] | None = None,
    include_fitted_interactions: bool = False,
    # Test data for overfitting detection (response/exposure auto-inferred from model)
    test_data: pl.DataFrame | None = None,
    # Control which enhanced diagnostics to compute
    compute_vif: bool = True,
    compute_coefficients: bool = True,
    compute_deviance_by_level: bool = True,
    compute_lift: bool = True,
    compute_partial_dep: bool = True,
    # Robust standard errors
    compute_robust_se: bool = True,
    # Score tests on unfitted factors
    compute_score_tests: bool = True,
    # Base/benchmark predictions comparison. String = same column name on
    # train/test; mapping = role-specific columns.
    base_predictions: str | dict[str, str] | None = None,
    ranking: str = "auto",
    exposure: str | np.ndarray | None = None,
    weights: str | np.ndarray | None = None,
) -> ModelDiagnostics:
    """
    Compute comprehensive model diagnostics.

    Results are automatically saved to 'analysis/diagnostics.json'.

    Parameters
    ----------
    result : GLMResults or GLMModel
        Fitted model results.
    train_data : pl.DataFrame
        Training data used for fitting.
    categorical_factors : list of str, optional
        Names of categorical factors to analyze.
    continuous_factors : list of str, optional
        Names of continuous factors to analyze.
    n_calibration_bins : int, default=10
        Number of bins for calibration curve.
    n_factor_bins : int, default=10
        Number of quantile bins for continuous factors.
    rare_threshold_pct : float, default=1.0
        Threshold (%) below which categorical levels are grouped into "Other".
    max_categorical_levels : int, default=20
        Maximum number of categorical levels to show (rest grouped to "Other").
    detect_interactions : bool, default=False
        Whether to detect residual-based interactions post-fit.
        Pre-fit interaction detection is handled by explore().
    max_interaction_factors : int, default=10
        Maximum number of factors to consider for interaction detection.
    interactions : list, optional
        Explicit list of variable pairs for per-pair surface diagnostics
        (separate from the auto-detector controlled by ``detect_interactions``).
        Each entry: ``{"factor1": ..., "factor2": ...}``, ``(a, b)``, or
        ``[a, b]``. Pairs do not need to appear in the fitted model;
        ``InteractionDiagnostics.in_model`` is set from TermSlot membership.
        When ``test_data`` is also supplied, each pair receives a
        ``test_surface_grid`` cell-aligned with the train surface (same bin
        edges / level lists), so the caller can compute element-wise
        train/test divergence in a single subtraction.
    include_fitted_interactions : bool, default=False
        When true, include the model's fitted interaction specs in the
        post-fit interaction diagnostics output. This is opt-in because
        surface diagnostics can be comparatively expensive for wide models.
    test_data : pl.DataFrame, optional
        Test/holdout data for overfitting detection. Response and exposure
        columns are automatically inferred from the model's formula.
    compute_vif : bool, default=True
        Whether to compute VIF/multicollinearity scores (train-only).
        Uses the stored design matrix or rebuilds it from train_data.
    compute_coefficients : bool, default=True
        Whether to compute coefficient summary with interpretations (train-only).
    compute_deviance_by_level : bool, default=True
        Whether to compute deviance breakdown by factor level.
    compute_lift : bool, default=True
        Whether to compute full lift chart.
    compute_partial_dep : bool, default=True
        Whether to compute partial dependence plots.
    compute_score_tests : bool, default=True
        Whether to compute Rao score tests for unfitted factors. Default True.
    base_predictions : str or dict, optional
        Column name containing response-scale predictions from another model
        (for example a GBM teacher or incumbent production model), or a mapping
        like ``{"train": "gbm_oof_mu", "test": "gbm_test_mu"}``. When provided,
        computes:
        - A/E ratio, loss, Gini for base predictions
        - Model vs base decile analysis sorted by model/base ratio
        - Summary of which model performs better in each decile
    ranking : {"auto", "mean", "rate"}, default="auto"
        Decile/lift ranking mode. ``"auto"`` ranks by predicted rate when
        exposure is present and by raw predicted mean otherwise.

    Returns
    -------
    ModelDiagnostics
        Complete diagnostics object with to_json() method.

        Fields for agentic workflows:
        - vif: VIF scores for detecting multicollinearity (train-only)
        - coefficient_summary: Coefficient interpretations (train-only)
        - factor_deviance: Deviance breakdown by categorical levels
        - lift_chart: Full lift chart showing all deciles
        - partial_dependence: Marginal effect shapes for each variable
        - train_test: Comprehensive train vs test comparison with flags:
            - overfitting_risk: True if gini_gap > 0.03
            - calibration_drift: True if test A/E outside [0.95, 1.05]
            - unstable_factors: Factors where train/test A/E differ by > 0.1
        - base_predictions_by_role: Train/test base comparisons when available
          (the train-side comparison is ``base_predictions_by_role.train``)

    Examples
    --------
    >>> result = rs.glm_dict(response="ClaimNb", terms={"Age": {"type": "linear"}, "Region": {"type": "categorical"}}, data=data, family="poisson", exposure="Exposure").fit()
    >>> diagnostics = result.diagnostics(
    ...     train_data=train_data,
    ...     test_data=test_data,
    ...     categorical_factors=["Region", "VehBrand"],
    ...     continuous_factors=["Age", "VehPower"],
    ...     base_predictions="old_model_pred",  # Compare against another model
    ... )
    >>>
    >>> # Check overfitting flags
    >>> if diagnostics.train_test and diagnostics.train_test.overfitting_risk:
    ...     print("Warning: Overfitting detected!")
    """
    # 1. Parse + validate inputs
    categorical_factors, continuous_factors = _normalize_factor_lists(
        categorical_factors, continuous_factors
    )
    y, mu, lp = _extract_response_and_predictions(result, train_data, exposure=exposure)
    _validate_data_length(train_data, mu)
    family, link, n_params, deviance, feature_names = _extract_model_metadata(result)
    response_col, exposure_col, exposure_arr = _resolve_offset_and_response(
        result, train_data, exposure_override=exposure
    )
    var_power, theta = _parse_family_params(family)
    null_deviance = _resolve_null_deviance(result)
    weights_arr = _resolve_weights(result, train_data, weights)
    warnings: list[dict[str, str]] = []
    base_train_mu = _extract_base_predictions_array(
        base_predictions,
        role="train",
        data=train_data,
        expected_len=train_data.height,
        warnings=warnings,
        required=base_predictions is not None,
    )
    base_test_mu = _extract_base_predictions_array(
        base_predictions,
        role="test",
        data=test_data,
        expected_len=test_data.height if test_data is not None else None,
        warnings=warnings,
        required=False,
    )

    # 2. Build computer
    computer = DiagnosticsComputer(
        y=y,
        mu=mu,
        linear_predictor=lp,
        family=family,
        n_params=n_params,
        deviance=deviance,
        exposure=exposure_arr,
        feature_names=feature_names,
        var_power=var_power,
        theta=theta,
        null_deviance=null_deviance,
        weights=weights_arr,
        base_mu=base_train_mu,
    )

    # Pre-compute the rank index once. compute_lift_chart and
    # _compute_ae_by_decile (called from compute_dataset_diagnostics) both bin
    # predictions by decile; sharing the index saves ~200ms of redundant
    # O(n log n) work at 1M rows. RS-ACT-004: rank by predicted rate
    # (mu/exposure) when exposure is present, else by mu, so the decile table and
    # lift chart agree with the rate-ranked calibration/discrimination stats.
    mu_sort_idx = computer._rank_sort_idx(ranking)

    # 3. Pre-cache columns + extract score-test matrices
    cat_cache_train, cat_unique_cache_train, cont_cache_train = _precompute_data_caches(
        train_data, categorical_factors, continuous_factors
    )
    score_tests_need_design = _score_tests_need_design_matrix(
        computer,
        result,
        categorical_factors,
        continuous_factors,
        compute_score_tests,
    )
    score_test_design_matrix, score_test_bread_matrix, score_test_irls_weights = (
        _extract_score_test_matrices(
            result,
            train_data,
            build_design_matrix=score_tests_need_design,
        )
    )

    # 4. Compute core diagnostics
    calibration = computer.compute_calibration(n_calibration_bins, ranking=ranking)
    residual_summary = computer.compute_residual_summary()

    factors = computer.compute_factor_diagnostics(
        data=train_data,
        categorical_factors=categorical_factors,
        continuous_factors=continuous_factors,
        result=result,
        n_bins=n_factor_bins,
        rare_threshold_pct=rare_threshold_pct,
        max_categorical_levels=max_categorical_levels,
        design_matrix=score_test_design_matrix,
        bread_matrix=score_test_bread_matrix,
        irls_weights=score_test_irls_weights,
        cat_column_cache=cat_cache_train,
        cont_column_cache=cont_cache_train,
        cat_unique_cache=cat_unique_cache_train,
        compute_score_tests=compute_score_tests,
    )

    interaction_candidates = _maybe_compute_interactions(
        detect_interactions,
        computer,
        train_data,
        categorical_factors,
        continuous_factors,
        max_interaction_factors,
        cat_cache_train,
        cont_cache_train,
    )

    model_comparison = computer.compute_model_comparison()

    # Train dataset diagnostics — always computed; single source of truth for fit metrics.
    train_diag = computer.compute_dataset_diagnostics(
        y,
        mu,
        computer.exposure,
        train_data,
        categorical_factors,
        continuous_factors,
        "train",
        result,
        cat_column_cache=cat_cache_train,
        cont_column_cache=cont_cache_train,
        cat_unique_cache=cat_unique_cache_train,
        sort_idx=mu_sort_idx,
        weights=weights_arr,
        base_mu=base_train_mu,
    )

    # Generate warnings (use train_diag for fit stats)
    warnings.extend(
        computer.generate_warnings(
            {
                "deviance": train_diag.deviance,
                "aic": train_diag.aic,
                "log_likelihood": train_diag.log_likelihood,
            },
            calibration,
            factors,
            family=family,
        )
    )

    interaction_specs = _resolve_diagnostics_interactions(
        result,
        interactions,
        include_fitted_interactions,
    )
    pair_interactions, block_interactions = _split_interaction_specs(interaction_specs)

    needs_block_gvif = bool(
        pair_interactions or block_interactions or any(f.in_model for f in factors)
    )
    block_gvif_correlation = None
    if compute_vif or needs_block_gvif:
        if score_test_design_matrix is not None:
            block_gvif_correlation = _build_design_correlation_matrix(
                score_test_design_matrix,
                include_inverse=needs_block_gvif,
            )
        else:
            block_gvif_correlation = _build_design_correlation_context_from_model(
                result,
                train_data,
                include_inverse=needs_block_gvif,
            )
            if block_gvif_correlation is None:
                score_test_design_matrix, score_test_bread_matrix, score_test_irls_weights = (
                    _extract_score_test_matrices(
                        result,
                        train_data,
                        build_design_matrix=True,
                    )
                )
                if score_test_design_matrix is not None:
                    block_gvif_correlation = _build_design_correlation_matrix(
                        score_test_design_matrix,
                        include_inverse=needs_block_gvif,
                    )

    # 5. Optional enhanced diagnostics for agentic workflows
    vif_results = _maybe_compute_vif(
        compute_vif,
        computer,
        score_test_design_matrix,
        feature_names,
        correlation_context=block_gvif_correlation,
    )
    coef_summary, robust_se_enriched = _maybe_compute_coefficients(
        compute_coefficients, compute_robust_se, computer, result, link, warnings
    )
    factor_dev = _maybe_compute_factor_deviance(
        compute_deviance_by_level,
        computer,
        train_data,
        categorical_factors,
        cat_cache_train,
        cat_unique_cache_train,
        warnings,
    )
    lift_chart = _maybe_compute_lift(compute_lift, computer, mu_sort_idx, warnings, ranking)
    partial_dep = _maybe_compute_partial_dependence(
        compute_partial_dep,
        computer,
        train_data,
        result,
        continuous_factors,
        categorical_factors,
        link,
        cat_cache_train,
        cat_unique_cache_train,
        cont_cache_train,
        warnings,
    )

    test_y_arr, test_mu_arr, test_exposure_arr, test_mu_sort_idx = _extract_test_arrays(
        test_data,
        result,
        response_col,
        exposure_col,
        ranking,
        exposure,
    )
    test_weights_arr = _resolve_test_weights(result, test_data, weights, warnings)

    # 6. Train/test comparison (test_diag built only if test_data provided)
    train_test = _build_train_test_comparison(
        train_diag,
        test_data,
        result,
        computer,
        response_col,
        exposure_col,
        categorical_factors,
        continuous_factors,
        warnings,
        test_y=test_y_arr,
        test_mu=test_mu_arr,
        test_exposure=test_exposure_arr,
        test_mu_sort_idx=test_mu_sort_idx,
        test_weights=test_weights_arr,
        test_base_mu=base_test_mu,
        ranking=ranking,
    )

    pair_train_data, pair_test_data = _interaction_diagnostics_data(
        result,
        train_data,
        test_data,
        interaction_specs,
    )

    # 6c. User-specified pair (interaction) diagnostics. Only fires when the
    # caller passed ``interactions=[...]``. Independent of the auto-detector
    # at step 5 (interaction_candidates), which keeps populating itself.
    interaction_diagnostics: list[InteractionDiagnostics] = []
    if pair_interactions:
        interaction_diagnostics = _compute_pair_diagnostics(
            computer=computer,
            interactions=pair_interactions,
            train_data=pair_train_data,
            result=result,
            response_col=response_col,
            exposure_col=exposure_col,
            score_test_design_matrix=score_test_design_matrix,
            score_test_bread_matrix=score_test_bread_matrix,
            test_data=pair_test_data,
            link=link,
            correlation_matrix=block_gvif_correlation,
            test_y=test_y_arr,
            test_mu=test_mu_arr,
            test_exposure=test_exposure_arr,
        )
    interaction_blocks = _compute_interaction_block_diagnostics(
        block_specs=block_interactions,
        model=result,
        bread_matrix=score_test_bread_matrix,
        correlation_matrix=block_gvif_correlation,
        warnings=warnings,
    )

    # 6d. Factor extensions: block GVIF (when fitted) and train/test bin
    # pairs (when test_data is supplied). Both are post-processing — no
    # extra prediction, no refit.
    _annotate_factor_extensions(
        factors=factors,
        train_diag=train_test.train,
        test_diag=train_test.test,
        correlation_matrix=block_gvif_correlation,
        model=result,
        test_data=test_data,
        test_y=test_y_arr,
        test_mu=test_mu_arr,
        test_exposure=test_exposure_arr,
        test_weights=test_weights_arr,
        test_base_mu=base_test_mu,
    )

    # 7. Auxiliary diagnostics
    overdispersion_result = _compute_overdispersion(family, result, computer, y, warnings)
    spline_info = _extract_spline_info(result)
    smooth_term_diagnostics = _compute_smooth_diagnostics(result, warnings)
    encoding_diagnostics = _compute_encoding_diagnostics(
        result,
        train_data,
        test_data,
        categorical_factors,
    )
    base_predictions_by_role = _compute_base_predictions_by_role(
        base_predictions=base_predictions,
        train_data=train_data,
        test_data=test_data,
        computer=computer,
        train_base_mu=base_train_mu,
        test_base_mu=base_test_mu,
        train_y=y,
        train_mu=mu,
        train_exposure=computer.exposure,
        train_weights=weights_arr,
        test_y=test_y_arr,
        test_mu=test_mu_arr,
        test_exposure=test_exposure_arr,
        test_weights=test_weights_arr,
        warnings=warnings,
        ranking=ranking,
    )

    # 8. Final assembly: relative importance + model summary, build diagnostics, save
    _annotate_relative_importance(factors, train_diag.deviance)
    model_summary = _build_model_summary(
        result, family, link, computer, n_params, null_deviance, robust_se_enriched
    )

    diagnostics = ModelDiagnostics(
        model_summary=model_summary,
        train_test=train_test,
        calibration=calibration,
        residual_summary=residual_summary,
        factors=factors,
        interaction_candidates=interaction_candidates,
        model_comparison=model_comparison,
        warnings=warnings,
        vif=vif_results,
        coefficient_summary=coef_summary,
        factor_deviance=factor_dev,
        lift_chart=lift_chart,
        partial_dependence=partial_dep,
        overdispersion=overdispersion_result,
        spline_info=spline_info,
        smooth_terms=smooth_term_diagnostics,
        base_predictions_by_role=base_predictions_by_role,
        encoding_diagnostics=encoding_diagnostics,
        interaction_blocks=interaction_blocks,
        interactions=interaction_diagnostics,
    )

    # Auto-save JSON to analysis folder
    os.makedirs("analysis", exist_ok=True)
    with open("analysis/diagnostics.json", "w") as f:
        f.write(diagnostics.to_json(indent=2))

    return diagnostics
