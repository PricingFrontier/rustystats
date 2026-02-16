"""
Top-level diagnostic API functions.

compute_diagnostics() is the main entry point for post-fit model diagnostics.
_compute_smooth_term_diagnostics() handles GAM smooth term analysis.
"""

from __future__ import annotations

import os
import re
from typing import Any, TYPE_CHECKING

import numpy as np

from rustystats._rustystats import (
    compute_discrimination_stats_py as _rust_discrimination_stats,
    compute_dataset_metrics_py as _rust_dataset_metrics,
    chi2_cdf_py as _chi2_cdf,
    factorize_strings_py as _factorize_strings,
)

from rustystats.diagnostics.types import (
    SmoothTermDiagnostics,
    TrainTestComparison,
    FactorLevelMetrics,
    ModelDiagnostics,
)

from rustystats.diagnostics.computer import DiagnosticsComputer
from rustystats.exceptions import ValidationError
from rustystats.constants import (
    DEFAULT_N_CALIBRATION_BINS,
    DEFAULT_N_FACTOR_BINS,
    DEFAULT_RARE_THRESHOLD_PCT,
    DEFAULT_MAX_CATEGORICAL_LEVELS,
    DEFAULT_MAX_INTERACTION_FACTORS,
)

if TYPE_CHECKING:
    import polars as pl


def _compute_smooth_term_diagnostics(
    result: Any,
    warnings: List[Dict[str, str]],
) -> List[SmoothTermDiagnostics]:
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
    if not hasattr(result, 'smooth_terms') or result.smooth_terms is None:
        return []
    
    smooth_diagnostics = []
    params = result.params
    
    # Get covariance matrix (unscaled)
    cov_matrix = None
    if hasattr(result, 'get_bread_matrix'):
        cov_matrix = result.get_bread_matrix()
    elif hasattr(result, '_result') and hasattr(result._result, 'cov_params_unscaled'):
        cov_matrix = result._result.cov_params_unscaled
    elif hasattr(result, '_result') and hasattr(result._result, 'covariance_unscaled'):
        cov_matrix = result._result.covariance_unscaled
    elif hasattr(result, 'cov_params'):
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
                warnings.append({
                    "type": "smooth_significance_fallback",
                    "message": f"Covariance matrix singular for s({st.variable}), using simplified test: {e}"
                })
                chi2 = float(np.sum(beta ** 2))
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
        if p_value > 0.05:
            warnings.append({
                "type": "insignificant_smooth",
                "message": f"Smooth term s({st.variable}) is not significant "
                          f"(p={p_value:.3f}, EDF={st.edf:.1f}). "
                          f"Consider using linear term or removing."
            })
        # Add warning for EDF close to k (under-smoothed)
        elif st.edf > st.k - 1.5:
            warnings.append({
                "type": "undersmoothed",
                "message": f"Smooth term s({st.variable}) has EDF≈k ({st.edf:.1f}/{st.k}). "
                          f"Consider increasing k for more flexibility."
            })
    
    return smooth_diagnostics


def compute_diagnostics(
    result: Any,  # GLMResults or GLMModel
    train_data: "pl.DataFrame",
    categorical_factors: Optional[List[str]] = None,
    continuous_factors: Optional[List[str]] = None,
    n_calibration_bins: int = DEFAULT_N_CALIBRATION_BINS,
    n_factor_bins: int = DEFAULT_N_FACTOR_BINS,
    rare_threshold_pct: float = DEFAULT_RARE_THRESHOLD_PCT,
    max_categorical_levels: int = DEFAULT_MAX_CATEGORICAL_LEVELS,
    detect_interactions: bool = False,
    max_interaction_factors: int = DEFAULT_MAX_INTERACTION_FACTORS,
    # Test data for overfitting detection (response/exposure auto-inferred from model)
    test_data: Optional["pl.DataFrame"] = None,
    # Control which enhanced diagnostics to compute
    compute_vif: bool = True,
    compute_coefficients: bool = True,
    compute_deviance_by_level: bool = True,
    compute_lift: bool = True,
    compute_partial_dep: bool = True,
    # Base predictions comparison (column name in train_data with predictions from another model)
    base_predictions: Optional[str] = None,
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
    base_predictions : str, optional
        Column name in train_data containing predictions from another model 
        (e.g., a base/benchmark model). When provided, computes:
        - A/E ratio, loss, Gini for base predictions
        - Model vs base decile analysis sorted by model/base ratio
        - Summary of which model performs better in each decile
    
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
        - base_predictions_comparison: Comparison against base predictions (if provided)
    
    Examples
    --------
    >>> result = rs.glm_dict(response="ClaimNb", terms={"Age": {"type": "linear"}, "Region": {"type": "categorical"}}, data=data, family="poisson", offset="Exposure").fit()
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
    # Deduplicate factors while preserving order
    categorical_factors = list(dict.fromkeys(categorical_factors or []))
    continuous_factors = list(dict.fromkeys(continuous_factors or []))
    # Remove any overlap (a factor can't be both categorical and continuous)
    continuous_factors = [f for f in continuous_factors if f not in categorical_factors]
    
    # Extract what we need from result
    # ALWAYS re-predict on train_data using result.predict() to get consistent encoding
    # This is critical for TE() which uses LOO encoding during fit but full encoding for predict
    # Using fittedvalues would give artificially high train loss due to LOO handicap
    formula_parts = result.formula.split('~') if hasattr(result, 'formula') else []
    response_col_temp = formula_parts[0].strip() if formula_parts else None
    
    if response_col_temp and response_col_temp in train_data.columns:
        y = train_data[response_col_temp].to_numpy().astype(np.float64)
        mu = np.asarray(result.predict(train_data), dtype=np.float64)
        lp = np.log(mu) if np.all(mu > 0) else mu
    else:
        # Fallback to fitted values if we can't determine response column
        mu = np.asarray(result.fittedvalues, dtype=np.float64)
        response_resid = np.asarray(result.resid_response(), dtype=np.float64)
        y = mu + response_resid
        lp = np.asarray(result.linear_predictor, dtype=np.float64)
    
    # Require essential attributes - fail loudly if missing
    if not hasattr(result, 'family'):
        raise ValidationError("Result object missing 'family' attribute")
    if not hasattr(result, 'link'):
        raise ValidationError("Result object missing 'link' attribute")
    if not hasattr(result, 'feature_names'):
        raise ValidationError("Result object missing 'feature_names' attribute")
    
    family = result.family
    link = result.link
    n_params = len(result.params)
    deviance = result.deviance
    feature_names = result.feature_names
    
    # Auto-infer response and exposure column names from formula
    response_col = None
    exposure_col = None
    if hasattr(result, 'formula') and result.formula:
        # Parse response from formula (left side of ~)
        formula_parts = result.formula.split('~')
        if len(formula_parts) >= 1:
            response_col = formula_parts[0].strip()
    if hasattr(result, '_offset_spec') and isinstance(result._offset_spec, str):
        exposure_col = result._offset_spec
    
    # Get exposure from training data
    exposure = None
    if exposure_col and exposure_col in train_data.columns:
        exposure = train_data[exposure_col].to_numpy().astype(np.float64)
    
    # Extract family parameters
    var_power = 1.5
    theta = 1.0
    if "tweedie" in family.lower():
        # Try to parse var_power from family string
        match = re.search(r'p=(\d+\.?\d*)', family)
        if match:
            var_power = float(match.group(1))
    if "negbinomial" in family.lower() or "negativebinomial" in family.lower():
        match = re.search(r'theta=(\d+\.?\d*)', family)
        if match:
            theta = float(match.group(1))
    
    # Get null deviance from model result (more accurate than recomputing)
    null_deviance = None
    if hasattr(result, 'null_deviance'):
        null_deviance = result.null_deviance() if callable(result.null_deviance) else result.null_deviance
    
    # Create computer
    computer = DiagnosticsComputer(
        y=y,
        mu=mu,
        linear_predictor=lp,
        family=family,
        n_params=n_params,
        deviance=deviance,
        exposure=exposure,
        feature_names=feature_names,
        var_power=var_power,
        theta=theta,
        null_deviance=null_deviance,
    )
    
    # Compute diagnostics
    calibration = computer.compute_calibration(n_calibration_bins)
    residual_summary = computer.compute_residual_summary()
    
    # Pre-extract categorical columns once to avoid repeated .to_numpy().astype(str)
    # Uses Rust HashMap-based factorize for O(n) encoding instead of O(n log n) np.unique
    _cat_cache_train = {}
    _cat_unique_cache_train = {}
    for name in categorical_factors:
        if name in train_data.columns:
            str_list = train_data[name].cast(str).to_list()
            levels, codes = _factorize_strings(str_list)
            str_vals = np.array(str_list)
            _cat_cache_train[name] = str_vals
            _cat_unique_cache_train[name] = (np.array(levels), codes)
    _cont_cache_train = {}
    for name in continuous_factors:
        if name in train_data.columns:
            _cont_cache_train[name] = train_data[name].to_numpy().astype(np.float64)
    
    # Get matrices for score test (for unfitted factors)
    # These are needed for Rao's score test on unfitted variables
    score_test_design_matrix = None
    score_test_bread_matrix = None
    score_test_irls_weights = None
    if hasattr(result, 'get_design_matrix'):
        score_test_design_matrix = result.get_design_matrix()
    # Fallback: rebuild design matrix from train_data when not stored (lean mode)
    if score_test_design_matrix is None and hasattr(result, '_builder') and result._builder is not None:
        score_test_design_matrix = result._builder.transform_new_data(train_data)
    if hasattr(result, 'get_bread_matrix'):
        score_test_bread_matrix = result.get_bread_matrix()
    if hasattr(result, 'get_irls_weights'):
        score_test_irls_weights = result.get_irls_weights()
    
    factors = computer.compute_factor_diagnostics(
        data=train_data,
        categorical_factors=categorical_factors,
        continuous_factors=continuous_factors,
        result=result,  # Pass result for significance tests
        n_bins=n_factor_bins,
        rare_threshold_pct=rare_threshold_pct,
        max_categorical_levels=max_categorical_levels,
        design_matrix=score_test_design_matrix,
        bread_matrix=score_test_bread_matrix,
        irls_weights=score_test_irls_weights,
        cat_column_cache=_cat_cache_train,
        cont_column_cache=_cont_cache_train,
        cat_unique_cache=_cat_unique_cache_train,
    )
    
    # Interaction detection
    interaction_candidates = []
    if detect_interactions and len(categorical_factors) + len(continuous_factors) >= 2:
        all_factors = categorical_factors + continuous_factors
        interaction_candidates = computer.detect_interactions(
            data=train_data,
            factor_names=all_factors,
            max_factors=max_interaction_factors,
            cat_column_cache=_cat_cache_train,
            cont_column_cache=_cont_cache_train,
        )
    
    model_comparison = computer.compute_model_comparison()
    
    # Always compute train_test - this is the single source of truth for metrics
    exposure_train = computer.exposure
    train_diag = computer.compute_dataset_diagnostics(
        y, mu, exposure_train, train_data,
        categorical_factors, continuous_factors, "train", result,
        cat_column_cache=_cat_cache_train,
        cont_column_cache=_cont_cache_train,
        cat_unique_cache=_cat_unique_cache_train,
    )
    
    # Generate warnings (use train_diag for fit stats)
    fit_stats_for_warnings = {
        "deviance": train_diag.deviance,
        "aic": train_diag.aic,
        "log_likelihood": train_diag.log_likelihood,
    }
    warnings = computer.generate_warnings(fit_stats_for_warnings, calibration, factors, family=family)
    
    # =========================================================================
    # NEW: Enhanced diagnostics for agentic workflows
    # =========================================================================
    
    # VIF / Multicollinearity
    # Token optimization: VIF array already contains all info, no separate warnings needed
    vif_results = None
    if compute_vif and score_test_design_matrix is not None:
        vif_results = computer.compute_vif(score_test_design_matrix, feature_names)
    
    # Coefficient summary
    coef_summary = None
    if compute_coefficients:
        coef_summary = computer.compute_coefficient_summary(result, link=link)
        # Token optimization: skip weak_predictors warning (agent can infer from sig=False + rel~1.0)
    
    # Deviance by factor level
    factor_dev = None
    if compute_deviance_by_level and categorical_factors:
        factor_dev = computer.compute_factor_deviance(train_data, categorical_factors)
        # Add warnings for problem levels
        for fd in factor_dev:
            if fd.problem_levels:
                warnings.append({
                    "type": "problem_factor_levels",
                    "message": f"Factor '{fd.factor}' has problem levels with poor fit: "
                              f"{', '.join(fd.problem_levels[:5])}{'...' if len(fd.problem_levels) > 5 else ''}"
                })
    
    # Lift chart
    lift_chart = None
    if compute_lift:
        lift_chart = computer.compute_lift_chart(n_deciles=10)
        # Add warnings for weak discrimination
        if lift_chart.weak_deciles:
            warnings.append({
                "type": "weak_discrimination",
                "message": f"Model has weak discrimination in deciles: {lift_chart.weak_deciles}. "
                          f"Consider adding features or interactions to improve separation."
            })
    
    # Partial dependence
    partial_dep = None
    if compute_partial_dep and (continuous_factors or categorical_factors):
        partial_dep = computer.compute_partial_dependence(
            data=train_data,
            result=result,
            continuous_factors=continuous_factors,
            categorical_factors=categorical_factors,
            link=link,
            cat_column_cache=_cat_cache_train,
            cat_unique_cache=_cat_unique_cache_train,
            cont_column_cache=_cont_cache_train,
        )
        # Add recommendations for non-linear effects
        for pd in partial_dep:
            if pd.shape in ("u_shaped", "inverted_u", "complex") and "spline" in pd.recommendation.lower():
                warnings.append({
                    "type": "nonlinear_effect",
                    "message": f"Variable '{pd.variable}' shows {pd.shape} pattern. {pd.recommendation}"
                })
    
    # Build train_test (train is always present, test is optional)
    train_test = TrainTestComparison(train=train_diag)
    
    if test_data is not None and response_col is not None:
        # Get test response
        if response_col not in test_data.columns:
            raise ValidationError(f"Response column '{response_col}' not found in test_data")
        y_test = test_data[response_col].to_numpy().astype(np.float64)
        
        # Get test predictions
        if not hasattr(result, 'predict'):
            raise ValidationError("Model does not support prediction on new data")
        mu_test = result.predict(test_data)
        
        # Get test exposure
        exposure_test = np.ones(len(y_test))
        if exposure_col and exposure_col in test_data.columns:
            exposure_test = test_data[exposure_col].to_numpy().astype(np.float64)
        
        # Pre-cache test data columns using Rust factorize
        _cat_cache_test = {}
        _cat_unique_cache_test = {}
        for name in categorical_factors:
            if name in test_data.columns:
                str_list = test_data[name].cast(str).to_list()
                levels, codes = _factorize_strings(str_list)
                str_vals = np.array(str_list)
                _cat_cache_test[name] = str_vals
                _cat_unique_cache_test[name] = (np.array(levels), codes)
        _cont_cache_test = {}
        for name in continuous_factors:
            if name in test_data.columns:
                _cont_cache_test[name] = test_data[name].to_numpy().astype(np.float64)
        
        # Compute test diagnostics
        test_diag = computer.compute_dataset_diagnostics(
            y_test, mu_test, exposure_test, test_data,
            categorical_factors, continuous_factors, "test", result,
            cat_column_cache=_cat_cache_test,
            cont_column_cache=_cont_cache_test,
            cat_unique_cache=_cat_unique_cache_test,
        )
        
        # Compute comparison metrics
        gini_gap = train_diag.gini - test_diag.gini
        ae_ratio_diff = abs(train_diag.ae_ratio - test_diag.ae_ratio)
        
        # Decile comparison
        decile_comparison = []
        for i in range(min(len(train_diag.ae_by_decile), len(test_diag.ae_by_decile))):
            train_d = train_diag.ae_by_decile[i]
            test_d = test_diag.ae_by_decile[i]
            decile_comparison.append({
                "decile": i + 1,
                "train_ae": train_d.ae_ratio,
                "test_ae": test_d.ae_ratio,
                "ae_diff": round(abs((train_d.ae_ratio or 0) - (test_d.ae_ratio or 0)), 4),
            })
        
        # Factor divergence
        factor_divergence = {}
        unstable_factors_list = []
        for factor in categorical_factors:
            if factor in train_diag.factor_diagnostics and factor in test_diag.factor_diagnostics:
                train_levels = {m.level: m for m in train_diag.factor_diagnostics[factor]}
                test_levels = {m.level: m for m in test_diag.factor_diagnostics[factor]}
                divergent = []
                for level in set(train_levels.keys()) | set(test_levels.keys()):
                    tr_ae = train_levels.get(level, FactorLevelMetrics(level, 0, 0, 0, 0, None, 0)).ae_ratio
                    te_ae = test_levels.get(level, FactorLevelMetrics(level, 0, 0, 0, 0, None, 0)).ae_ratio
                    if tr_ae is not None and te_ae is not None:
                        diff = abs(tr_ae - te_ae)
                        if diff > 0.1:
                            divergent.append({"level": level, "train_ae": tr_ae, "test_ae": te_ae, "ae_diff": round(diff, 4)})
                            unstable_factors_list.append(f"{factor}[{level}]")
                if divergent:
                    factor_divergence[factor] = divergent
        
        # Flags
        overfitting_risk = gini_gap > 0.03
        calibration_drift = test_diag.ae_ratio < 0.95 or test_diag.ae_ratio > 1.05
        
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
            warnings.append({
                "type": "overfitting",
                "message": f"Overfitting detected: Train Gini={train_diag.gini:.3f}, "
                          f"Test Gini={test_diag.gini:.3f} (gap={gini_gap:.3f}). "
                          f"Consider reducing model complexity or using regularization."
            })
        if calibration_drift:
            warnings.append({
                "type": "calibration_drift",
                "message": f"Calibration drift: Test A/E={test_diag.ae_ratio:.3f} "
                          f"(outside [0.95, 1.05]). Model may not generalize well."
            })
        if unstable_factors_list:
            warnings.append({
                "type": "unstable_factors",
                "message": f"Unstable factor levels (train/test A/E differ by >0.1): "
                          f"{', '.join(unstable_factors_list[:10])}"
                          f"{'...' if len(unstable_factors_list) > 10 else ''}"
            })
    
    # Extract convergence info - require these attributes
    if not hasattr(result, 'converged'):
        raise ValidationError("Result object missing 'converged' attribute")
    if not hasattr(result, 'iterations'):
        raise ValidationError("Result object missing 'iterations' attribute")
    if not hasattr(result, 'formula'):
        raise ValidationError("Result object missing 'formula' attribute")
    
    converged = result.converged
    iterations = result.iterations
    
    # Model summary
    model_summary = {
        "formula": result.formula,
        "family": family,
        "link": link,
        "n_obs": computer.n_obs,
        "n_params": n_params,
        "df_resid": computer.df_resid,
        "converged": converged,
        "iterations": iterations,
    }
    
    # Add regularization info if present (concise for LLM parsing)
    if hasattr(result, 'alpha') and result.alpha > 0:
        reg_type = getattr(result, 'regularization_type', None)
        if reg_type is None:
            l1 = getattr(result, 'l1_ratio', 0)
            reg_type = "lasso" if l1 >= 1 else "ridge" if l1 <= 0 else "elastic_net"
        model_summary["regularization"] = {
            "type": reg_type,
            "alpha": round(result.alpha, 6),
            "l1_ratio": round(getattr(result, 'l1_ratio', 0), 2),
        }
        # Add CV info if available
        if hasattr(result, 'cv_deviance') and result.cv_deviance is not None:
            model_summary["regularization"]["cv_deviance"] = round(result.cv_deviance, 6)
            model_summary["regularization"]["cv_folds"] = getattr(result, 'n_cv_folds', None)
            model_summary["regularization"]["selection"] = getattr(result, 'cv_selection_method', None)
    
    # Compute overdispersion (for Poisson/Binomial families)
    overdispersion_result = None
    family_lower = family.lower()
    if any(f in family_lower for f in ["poisson", "binomial", "negativebinomial"]):
        # Model-based dispersion: Pearson chi-squared / df_resid
        pearson_chi2 = result.pearson_chi2() if hasattr(result, 'pearson_chi2') else None
        df_resid = computer.df_resid
        
        if pearson_chi2 is not None and df_resid > 0:
            pearson_dispersion = pearson_chi2 / df_resid
            
            # Also compute raw dispersion from data (Var/Mean for counts)
            mean_count = float(np.mean(y))
            var_count = float(np.var(y, ddof=1)) if len(y) > 1 else 0.0
            raw_dispersion = var_count / mean_count if mean_count > 0 else 1.0
            
            # Severity based on Pearson dispersion (more reliable)
            if pearson_dispersion > 5:
                severity = "severe"
                recommendation = "Use Negative Binomial or QuasiPoisson"
            elif pearson_dispersion > 2:
                severity = "moderate"
                recommendation = "Consider Negative Binomial or QuasiPoisson"
            elif pearson_dispersion > 1.5:
                severity = "mild"
                recommendation = "Monitor; Poisson may underestimate standard errors"
            else:
                severity = "none"
                recommendation = "Poisson assumption appears reasonable"
            
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
            if pearson_dispersion > 1.5:
                warnings.append({
                    "type": "overdispersion",
                    "message": f"Overdispersion detected (φ={pearson_dispersion:.2f}). {recommendation}"
                })
    
    # Get spline knot information if available
    spline_info = None
    if hasattr(result, '_builder') and hasattr(result._builder, 'get_spline_info'):
        spline_info = result._builder.get_spline_info()
        if not spline_info:  # Empty dict -> None
            spline_info = None
    
    # Smooth term diagnostics with EDF and significance tests
    smooth_term_diagnostics = None
    if hasattr(result, 'has_smooth_terms') and result.has_smooth_terms():
        smooth_term_diagnostics = _compute_smooth_term_diagnostics(result, warnings)
    
    # Base predictions comparison (if provided)
    base_predictions_comparison = None
    if base_predictions is not None:
        if base_predictions not in train_data.columns:
            raise ValidationError(f"base_predictions column '{base_predictions}' not found in train_data")
        mu_base = train_data[base_predictions].to_numpy().astype(np.float64)
        base_predictions_comparison = computer.compute_base_predictions_comparison(
            y=y,
            mu_model=mu,
            mu_base=mu_base,
            exposure=computer.exposure,
        )
        # Add summary to warnings
        if base_predictions_comparison.loss_improvement_pct > 0:
            warnings.append({
                "type": "model_improvement",
                "message": f"Model improves on base predictions: {base_predictions_comparison.loss_improvement_pct:.1f}% lower loss, "
                          f"better A/E in {base_predictions_comparison.model_better_deciles}/10 deciles"
            })
        elif base_predictions_comparison.loss_improvement_pct < 0:
            warnings.append({
                "type": "model_regression",
                "message": f"Model is worse than base predictions: {-base_predictions_comparison.loss_improvement_pct:.1f}% higher loss, "
                          f"better A/E in only {base_predictions_comparison.model_better_deciles}/10 deciles"
            })
    
    # Compute relative importance: normalise Type III deviance contributions to sum to 100%
    fitted_with_sig = [f for f in factors if f.in_model and f.significance and f.significance.dev_contrib]
    if fitted_with_sig:
        total_dev = sum(f.significance.dev_contrib for f in fitted_with_sig)
        if total_dev > 0:
            for f in fitted_with_sig:
                f.relative_importance = round(f.significance.dev_contrib / total_dev * 100, 2)
    
    # Compute dev_pct / expected_dev_pct: normalise by train deviance
    model_deviance = train_diag.deviance
    if model_deviance and model_deviance > 0:
        for f in factors:
            if f.significance and f.significance.dev_contrib:
                f.significance.dev_pct = round(f.significance.dev_contrib / model_deviance * 100, 2)
            if f.score_test and f.score_test.statistic:
                f.score_test.expected_dev_pct = round(f.score_test.statistic / model_deviance * 100, 2)
    
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
        base_predictions_comparison=base_predictions_comparison,
    )
    
    # Auto-save JSON to analysis folder
    os.makedirs("analysis", exist_ok=True)
    with open("analysis/diagnostics.json", "w") as f:
        f.write(diagnostics.to_json(indent=2))
    
    return diagnostics
