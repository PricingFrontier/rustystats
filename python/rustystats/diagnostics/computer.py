"""
Main diagnostics computation orchestrator.

DiagnosticsComputer coordinates focused component classes to produce
unified diagnostics output for fitted GLM models.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from rustystats._rustystats import (
    chi2_cdf_py as _chi2_cdf,
)
from rustystats._rustystats import (
    compute_ae_by_decile_py as _rust_ae_by_decile,
)
from rustystats._rustystats import (
    compute_ae_categorical_batch_py as _rust_ae_categorical_batch,
)
from rustystats._rustystats import (
    compute_ae_continuous_batch_py as _rust_ae_continuous_batch,
)
from rustystats._rustystats import (
    compute_correlation_and_vif_py as _rust_correlation_and_vif,
)
from rustystats._rustystats import (
    compute_dataset_metrics_py as _rust_dataset_metrics,
)
from rustystats._rustystats import (
    compute_deviance_residuals_py as _rust_deviance_residuals,
)
from rustystats._rustystats import (
    compute_discrimination_stats_py as _rust_discrimination_stats,
)
from rustystats._rustystats import (
    compute_fit_statistics_py as _rust_fit_statistics,
)
from rustystats._rustystats import (
    compute_loss_metrics_py as _rust_loss_metrics,
)
from rustystats._rustystats import (
    compute_residual_summary_py as _rust_residual_summary,
)
from rustystats._rustystats import (
    partial_dependence_categorical_batch_py as _rust_partial_dependence_categorical_batch,
)
from rustystats.constants import (
    DEFAULT_MAX_CATEGORICAL_LEVELS,
    DEFAULT_MAX_INTERACTION_FACTORS,
    DEFAULT_N_CALIBRATION_BINS,
    DEFAULT_N_FACTOR_BINS,
    DEFAULT_RARE_THRESHOLD_PCT,
    EPSILON,
    PD_CURVATURE_RELATIVE_THRESHOLD,
    PD_FLAT_RELATIVE_RANGE,
    PD_MONOTONIC_THRESHOLD,
    PD_STEP_FUNCTION_RATIO,
)
from rustystats.diagnostics.components import (
    _CalibrationComputer,
    _DiscriminationComputer,
    _ResidualComputer,
)
from rustystats.diagnostics.factors import _FactorDiagnosticsComputer
from rustystats.diagnostics.interactions import _InteractionDetector
from rustystats.diagnostics.pair_diagnostics import _PairDiagnosticsComputer
from rustystats.diagnostics.types import (
    BasePredictionsComparison,
    BasePredictionsMetrics,
    CoefficientSummary,
    ContinuousBandMetrics,
    DatasetDiagnostics,
    DecileMetrics,
    DevianceByLevel,
    FactorDeviance,
    FactorDiagnostics,
    FactorLevelMetrics,
    InteractionCandidate,
    InteractionDiagnostics,
    LiftChart,
    LiftDecile,
    ModelVsBaseDecile,
    PartialDependence,
    ResidualSummary,
    TrainTestComparison,
    VIFResult,
    _extract_base_variable,
)
from rustystats.exceptions import DesignMatrixError, ValidationError
from rustystats.links import link_forward, link_inverse

if TYPE_CHECKING:
    import polars as pl


def rank_sort_idx(
    mu: np.ndarray,
    exposure: np.ndarray | None = None,
    *,
    has_exposure: bool | None = None,
    ranking: str = "auto",
) -> np.ndarray:
    """Ascending argsort for actuarial diagnostics ranking.

    ``ranking``:
    - ``"auto"`` — rank by ``mu/exposure`` when exposure was supplied, else ``mu``;
    - ``"rate"`` — rank by ``mu/exposure`` (requires exposure);
    - ``"mean"`` — rank by raw ``mu``.

    Stable sorting preserves original row order for equal risk scores, matching
    Rust's `(score, index)` tie-breaker in the default diagnostics sort.
    """
    mu_arr = np.asarray(mu, dtype=np.float64)
    exposure_arr = None if exposure is None else np.asarray(exposure, dtype=np.float64)
    exposure_supplied = exposure_arr is not None if has_exposure is None else has_exposure

    if ranking == "mean":
        key = mu_arr
    elif ranking == "rate":
        if not exposure_supplied or exposure_arr is None:
            raise ValidationError("ranking='rate' requires exposure to be supplied.")
        key = mu_arr / exposure_arr
    elif ranking == "auto":
        key = mu_arr / exposure_arr if exposure_supplied and exposure_arr is not None else mu_arr
    else:
        raise ValidationError(f"ranking must be 'auto', 'mean', or 'rate', got {ranking!r}.")
    return np.argsort(key, kind="stable")


class DiagnosticsComputer:
    """
    Computes comprehensive model diagnostics.

    Coordinates focused component classes to produce unified diagnostics output.
    All results are cached for efficiency.
    """

    def __init__(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        linear_predictor: np.ndarray,
        family: str,
        n_params: int,
        deviance: float,
        exposure: np.ndarray | None = None,
        feature_names: list[str | None] | None = None,
        var_power: float = 1.5,
        theta: float = 1.0,
        null_deviance: float | None = None,
    ):
        self.y = np.asarray(y, dtype=np.float64)
        self.mu = np.asarray(mu, dtype=np.float64)
        self.linear_predictor = np.asarray(linear_predictor, dtype=np.float64)
        self.family = family.lower()
        self.n_params = n_params
        self.deviance = deviance
        self._null_deviance_override = null_deviance  # From model result
        self._has_exposure = exposure is not None
        self.exposure = (
            np.asarray(exposure, dtype=np.float64) if exposure is not None else np.ones_like(y)
        )
        self.feature_names = feature_names or []
        self.var_power = var_power
        self.theta = theta

        self.n_obs = len(y)
        self.df_resid = self.n_obs - n_params

        # Initialize focused components
        self._residuals = _ResidualComputer(self.y, self.mu, self.family, self.exposure)
        self._calibration = _CalibrationComputer(self.y, self.mu, self.exposure)
        self._discrimination = _DiscriminationComputer(self.y, self.mu, self.exposure)
        self._factors = _FactorDiagnosticsComputer(
            self.y,
            self.mu,
            self.exposure,
            self.pearson_residuals,
            self.feature_names,
            self.family,
        )
        self._interactions = _InteractionDetector(
            self.pearson_residuals,
            self.feature_names,
        )
        self._pairs = _PairDiagnosticsComputer(
            y=self.y,
            mu=self.mu,
            exposure=self.exposure,
            family=self.family,
            feature_names=self.feature_names,
            link=None,
        )

    @property
    def pearson_residuals(self) -> np.ndarray:
        return self._residuals.pearson

    @property
    def deviance_residuals(self) -> np.ndarray:
        return self._residuals.deviance

    @property
    def null_deviance(self) -> float:
        # Use override from model if provided, otherwise compute
        if self._null_deviance_override is not None:
            return self._null_deviance_override
        return self._residuals.null_deviance

    def _compute_unit_deviance(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return self._residuals.unit_deviance(y, mu)

    def _compute_loss(
        self, y: np.ndarray, mu: np.ndarray, weights: np.ndarray | None = None
    ) -> float:
        unit_dev = self._compute_unit_deviance(y, mu)
        if weights is not None:
            return np.average(unit_dev, weights=weights)
        return np.mean(unit_dev)

    def compute_fit_statistics(self) -> dict[str, float]:
        """Compute overall fit statistics using Rust backend."""
        return _rust_fit_statistics(
            self.y, self.mu, self.deviance, self.null_deviance, self.n_params, self.family
        )

    def compute_loss_metrics(self) -> dict[str, float]:
        """Compute various loss metrics using Rust backend."""
        rust_loss = _rust_loss_metrics(self.y, self.mu, self.family)
        return {
            "loss": rust_loss["family_loss"],  # Primary metric for model comparison
            "mse": rust_loss["mse"],
            "mae": rust_loss["mae"],
            "rmse": rust_loss["rmse"],
        }

    def compute_calibration(self, n_bins: int = DEFAULT_N_CALIBRATION_BINS) -> dict[str, Any]:
        """Compute calibration metrics using focused component."""
        return self._calibration.compute(n_bins)

    def compute_discrimination(self) -> dict[str, Any | None]:
        """Compute discrimination metrics using focused component."""
        return self._discrimination.compute()

    def compute_residual_summary(self) -> dict[str, ResidualSummary]:
        """Compute residual summary statistics using Rust backend (compressed)."""

        def summarize(resid: np.ndarray) -> ResidualSummary:
            stats = _rust_residual_summary(resid)
            return ResidualSummary(
                mean=round(stats["mean"], 2),
                std=round(stats["std"], 2),
                skewness=round(stats["skewness"], 1),
            )

        return {
            "pearson": summarize(self.pearson_residuals),
            "deviance": summarize(self.deviance_residuals),
        }

    def compute_factor_diagnostics(
        self,
        data: pl.DataFrame,
        categorical_factors: list[str],
        continuous_factors: list[str],
        result=None,
        n_bins: int = DEFAULT_N_FACTOR_BINS,
        rare_threshold_pct: float = DEFAULT_RARE_THRESHOLD_PCT,
        max_categorical_levels: int = DEFAULT_MAX_CATEGORICAL_LEVELS,
        design_matrix: np.ndarray | None = None,
        bread_matrix: np.ndarray | None = None,
        irls_weights: np.ndarray | None = None,
        cat_column_cache: dict[str, np.ndarray | None] | None = None,
        cont_column_cache: dict[str, np.ndarray | None] | None = None,
        cat_unique_cache: dict[str, tuple | None] | None = None,
        compute_score_tests: bool = True,
    ) -> list[FactorDiagnostics]:
        """Compute diagnostics for each specified factor.

        Delegates to _FactorDiagnosticsComputer for per-factor analysis.
        """
        return self._factors.compute_factor_diagnostics(
            data=data,
            categorical_factors=categorical_factors,
            continuous_factors=continuous_factors,
            result=result,
            n_bins=n_bins,
            rare_threshold_pct=rare_threshold_pct,
            max_categorical_levels=max_categorical_levels,
            design_matrix=design_matrix,
            bread_matrix=bread_matrix,
            irls_weights=irls_weights,
            cat_column_cache=cat_column_cache,
            cont_column_cache=cont_column_cache,
            cat_unique_cache=cat_unique_cache,
            compute_score_tests=compute_score_tests,
        )

    def compute_pair_diagnostics(
        self,
        pairs: list[Any],
        data: pl.DataFrame,
        model: Any = None,
        design_matrix: np.ndarray | None = None,
        bread_matrix: np.ndarray | None = None,
        params: np.ndarray | None = None,
        bse: np.ndarray | None = None,
        correlation_matrix: np.ndarray | None = None,
        test_data: pl.DataFrame | None = None,
        test_y: np.ndarray | None = None,
        test_mu: np.ndarray | None = None,
        test_exposure: np.ndarray | None = None,
        link: str | None = None,
    ) -> list[InteractionDiagnostics]:
        """Compute per-pair diagnostics. Delegates to ``_PairDiagnosticsComputer``."""
        if link is not None:
            self._pairs.link = link
        return self._pairs.compute_pair_diagnostics(
            pairs=pairs,
            data=data,
            model=model,
            design_matrix=design_matrix,
            bread_matrix=bread_matrix,
            params=params,
            bse=bse,
            correlation_matrix=correlation_matrix,
            test_data=test_data,
            test_y=test_y,
            test_mu=test_mu,
            test_exposure=test_exposure,
        )

    def detect_interactions(
        self,
        data: pl.DataFrame,
        factor_names: list[str],
        max_factors: int = DEFAULT_MAX_INTERACTION_FACTORS,
        min_correlation: float = 0.01,
        max_candidates: int = 5,
        min_cell_count: int = 30,
        cat_column_cache: dict[str, np.ndarray | None] | None = None,
        cont_column_cache: dict[str, np.ndarray | None] | None = None,
    ) -> list[InteractionCandidate]:
        """Detect potential interactions. Delegates to _InteractionDetector."""
        return self._interactions.detect_interactions(
            data=data,
            factor_names=factor_names,
            max_factors=max_factors,
            min_correlation=min_correlation,
            max_candidates=max_candidates,
            min_cell_count=min_cell_count,
            cat_column_cache=cat_column_cache,
            cont_column_cache=cont_column_cache,
        )

    def compute_model_comparison(self) -> dict[str, float]:
        """Compute model comparison statistics vs null model."""
        null_dev = self.null_deviance

        # Likelihood ratio test
        lr_chi2 = null_dev - self.deviance
        lr_df = self.n_params - 1

        # P-value from chi-square distribution (using Rust CDF)
        lr_pvalue = 1 - _chi2_cdf(lr_chi2, float(lr_df)) if lr_df > 0 else float("nan")

        deviance_reduction_pct = 100 * (1 - self.deviance / null_dev) if null_dev > 0 else 0

        # AIC improvement
        null_aic = null_dev + 2  # Null model has 1 parameter
        model_aic = self.deviance + 2 * self.n_params
        aic_improvement = null_aic - model_aic

        return {
            "likelihood_ratio_chi2": float(lr_chi2),
            "likelihood_ratio_df": lr_df,
            "likelihood_ratio_pvalue": float(lr_pvalue),
            "deviance_reduction_pct": float(deviance_reduction_pct),
            "aic_improvement": float(aic_improvement),
        }

    def generate_warnings(
        self,
        fit_stats: dict[str, float],
        calibration: dict[str, Any],
        factors: list[FactorDiagnostics],
        family: str = "",
    ) -> list[dict[str, str]]:
        """Generate warnings based on diagnostics."""
        warnings = []

        # NegBin-specific warnings
        family_lower = family.lower() if family else ""
        if family_lower.startswith("negativebinomial"):
            # Regularization warning
            warnings.append(
                {
                    "type": "negbinomial_regularization",
                    "message": "Negative binomial fitting applies minimum ridge regularization (alpha=1e-6) for numerical stability. Coefficient bias is negligible but inference is approximate.",
                }
            )

            # Large theta warning (essentially Poisson)
            if "theta=" in family:
                try:
                    theta_str = family.split("theta=")[1].rstrip(")")
                    theta = float(theta_str)
                    if theta >= 100:
                        warnings.append(
                            {
                                "type": "negbinomial_large_theta",
                                "message": f"Estimated theta={theta:.1f} is very large, suggesting minimal overdispersion. Consider using Poisson instead for simpler interpretation.",
                            }
                        )
                    elif theta <= 0.1:
                        warnings.append(
                            {
                                "type": "negbinomial_small_theta",
                                "message": f"Estimated theta={theta:.4f} is very small, indicating severe overdispersion. Check for missing covariates or consider zero-inflated models.",
                            }
                        )
                except (ValueError, IndexError) as e:
                    # Theta parsing failed - this is a bug in family string formatting
                    raise ValidationError(
                        f"Failed to parse theta from family string '{family}': {e}"
                    ) from e

        # High dispersion warning
        dispersion = fit_stats.get("dispersion", 1.0)
        if dispersion > 1.5:
            warnings.append(
                {
                    "type": "high_dispersion",
                    "message": f"Dispersion {dispersion:.2f} suggests overdispersion. Consider quasipoisson or negbinomial.",
                }
            )

        # Poor overall calibration
        ae_ratio = calibration.get("ae_ratio", 1.0)
        if abs(ae_ratio - 1.0) > 0.05:
            direction = "over" if ae_ratio < 1 else "under"
            warnings.append(
                {
                    "type": "poor_calibration",
                    "message": f"Model {direction}-predicts overall (A/E = {ae_ratio:.3f}).",
                }
            )

        # Token optimization: skip per-decile warnings (problem_deciles in calibration has this info)

        # Factors with high residual correlation (not in model)
        for factor in factors:
            if not factor.in_model:
                r2 = factor.residual_pattern.var_explained
                if r2 > 0.02:
                    warnings.append(
                        {
                            "type": "missing_factor",
                            "message": f"Factor '{factor.name}' not in model but explains {100 * r2:.1f}% of residual variance.",
                        }
                    )

        return warnings

    # =========================================================================
    # NEW: Enhanced diagnostics for agentic workflows
    # =========================================================================

    def compute_vif(
        self,
        X: np.ndarray,
        feature_names: list[str],
        threshold_moderate: float = 5.0,
        threshold_severe: float = 10.0,
    ) -> list[VIFResult]:
        """
        Compute Variance Inflation Factors for design matrix columns.

        Uses correlation matrix inverse for O(k³) complexity instead of
        O(k × n × k²) for k features and n observations.

        VIF detects multicollinearity which can cause:
        - Unstable coefficient estimates
        - Inflated standard errors
        - Failed matrix inversions (like VehPower + bs(VehPower, df=4))

        Parameters
        ----------
        X : np.ndarray
            Design matrix (n_obs, n_features)
        feature_names : list of str
            Names of features in X
        threshold_moderate : float
            VIF above this indicates moderate multicollinearity
        threshold_severe : float
            VIF above this indicates severe multicollinearity

        Returns
        -------
        list of VIFResult
            VIF for each feature, sorted by VIF (highest first)
        """
        _n_obs, n_features = X.shape
        results = []

        # Skip intercept column if present
        has_intercept = feature_names and feature_names[0] == "Intercept"
        start_idx = 1 if has_intercept else 0

        if n_features - start_idx <= 1:
            # Only one feature (besides intercept), VIF = 1
            for i in range(start_idx, n_features):
                results.append(
                    VIFResult(
                        feature=feature_names[i] if i < len(feature_names) else f"X{i}",
                        vif=1.0,
                        severity="none",
                        collinear_with=None,
                    )
                )
            return results

        names_no_int = (
            feature_names[start_idx:]
            if feature_names
            else [f"X{i}" for i in range(start_idx, n_features)]
        )
        k = n_features - start_idx

        # Fast VIF via correlation matrix inverse
        # VIF_j = diag((R^{-1}))_j where R is correlation matrix
        try:
            # Push correlation matrix + Cholesky inverse-diagonal into Rust
            # in a single call. The Python implementation that this replaces
            # ran `np.corrcoef(X[:, start_idx:], rowvar=False)`, which mean-
            # centers a copy of X internally — an O(n*k) transient allocation
            # that dominated this method's RSS peak (~800 MB at 1M rows ×
            # 100+ params).
            #
            # Rust takes a zero-copy view of X via PyReadonlyArray2 and skips
            # the first `start_idx` columns (typically the intercept) inside
            # the same call so we never allocate a Python-side slice. The
            # correlation matrix and `diag((R + ε·I)^{-1})` are computed in
            # O(k²) extra memory, and only the small (k, k) and (k,) arrays
            # come back to Python.
            #
            # Zero-variance columns get an all-zero row/col + 0 diagonal in R
            # from Rust. Downstream "find correlated pairs" logic uses
            # `not np.isnan(corr) and abs(corr) > 0.5`, which (a) ignores the
            # 0 entries and (b) the regularized inverse picks up ~1/EPSILON
            # at that diagonal, so the severity classifier flags them severe.
            R, vif_values = _rust_correlation_and_vif(X, EPSILON, start_idx)
            # Non-zero-variance columns: rust sets corr_ii to 1 directly.
            # Zero-variance columns: rust leaves the row/col + diagonal at 0,
            # which (a) keeps the "diagonal == 0" sentinel for collinear-pair
            # display logic and (b) drives `(R + ε·I)^{-1}_ii ≈ 1/ε`, so the
            # Severity classifier later flags them as severe (= 999.0).

            if np.any(np.isnan(vif_values)):
                # Pathological: not positive-definite even after EPSILON
                # regularization. Rare given the regularization above.
                raise np.linalg.LinAlgError(
                    "VIF computation failed: design matrix correlation matrix is not "
                    "positive-definite. This usually means your design matrix has "
                    "exact collinearity (linearly-dependent columns). To diagnose: "
                    "(1) check for duplicate features, (2) check that categorical levels "
                    "haven't created a near-singular design (consider dropping a baseline "
                    "level), (3) for high-cardinality categoricals consider target encoding "
                    "instead of dummy encoding, or (4) use ridge/elastic-net regularization "
                    "with .fit(regularization='ridge')."
                )

            # Correlation matrix for finding collinear pairs (unregularized)
            corr_matrix = R

        except np.linalg.LinAlgError as e:
            raise DesignMatrixError(
                "VIF computation failed: design matrix is singular. "
                "This indicates severe multicollinearity - some columns are exact linear "
                "combinations of others. Check for duplicate or constant columns."
            ) from e

        # Build results
        for i in range(k):
            feature_name = names_no_int[i] if i < len(names_no_int) else f"X{i}"
            vif = vif_values[i]

            # Find most correlated features first (needed for severity assessment)
            correlations = []
            for j in range(k):
                if j != i:
                    corr = corr_matrix[i, j]
                    if not np.isnan(corr) and abs(corr) > 0.5:
                        correlations.append((names_no_int[j], abs(corr)))
            correlations.sort(key=lambda x: -x[1])
            collinear_with = [c[0] for c in correlations[:3]]  # Top 3

            # Determine initial severity based on VIF value
            if np.isnan(vif) or np.isinf(vif) or vif > 100:
                severity = "severe"
                vif = 999.0 if np.isnan(vif) or np.isinf(vif) else vif
            elif vif > threshold_severe:
                severity = "severe"
            elif vif > threshold_moderate:
                severity = "moderate"
            else:
                severity = "none"

            # Downgrade to "expected" if high VIF is only due to same-variable terms
            # (e.g., BonusMalus correlated with I(BonusMalus ** 2) is expected)
            if severity in ("moderate", "severe") and collinear_with:
                base_var = _extract_base_variable(feature_name)
                collinear_bases = [_extract_base_variable(c) for c in collinear_with]
                # If ALL correlated features share the same base variable, it's expected
                if all(cb == base_var for cb in collinear_bases):
                    severity = "expected"

            results.append(
                VIFResult(
                    feature=feature_name,
                    vif=round(float(vif), 2),
                    severity=severity,
                    collinear_with=collinear_with if collinear_with else None,
                )
            )

        # Sort by VIF (highest first)
        results.sort(key=lambda x: -x.vif if not np.isnan(x.vif) else 0)
        return results

    def compute_coefficient_summary(
        self,
        result,  # GLMResults or GLMModel
        link: str = "log",
    ) -> list[CoefficientSummary]:
        """
        Compute coefficient summary with interpretations for agent use.

        Token-optimized compact format with shortened field names.
        Agent can infer impact from z-value sign and relativity magnitude.

        Returns
        -------
        list of CoefficientSummary
            Summary for each coefficient, sorted by absolute z-value
        """
        params = np.asarray(result.params)
        bse = np.asarray(result.bse())
        tvalues = np.asarray(result.tvalues())
        pvalues = np.asarray(result.pvalues())
        ci = np.asarray(result.conf_int(0.05))

        feature_names = (
            self.feature_names if self.feature_names else [f"X{i}" for i in range(len(params))]
        )

        summaries = []
        for i, name in enumerate(feature_names):
            coef_val = float(params[i])
            se_val = float(bse[i])
            z_val = float(tvalues[i])
            p_val = float(pvalues[i])

            # Confidence interval
            ci_low = round(float(ci[i, 0]), 6)
            ci_high = round(float(ci[i, 1]), 6)

            # Relativity for log-link models
            rel = None
            rel_ci = None
            if link == "log":
                rel = round(float(np.exp(coef_val)), 4)
                rel_ci = [round(float(np.exp(ci[i, 0])), 4), round(float(np.exp(ci[i, 1])), 4)]

            summaries.append(
                CoefficientSummary(
                    feature=name,
                    estimate=round(coef_val, 6),
                    std_error=round(se_val, 6),
                    z_value=round(z_val, 3),
                    p_value=round(p_val, 4),
                    significant=p_val < 0.05,
                    conf_int=[ci_low, ci_high],
                    relativity=rel,
                    relativity_ci=rel_ci,
                )
            )

        # Sort by absolute z-value (most significant first), but keep Intercept at end
        intercept = [s for s in summaries if s.feature == "Intercept"]
        others = [s for s in summaries if s.feature != "Intercept"]
        others.sort(key=lambda x: -abs(x.z_value))
        return others + intercept

    def enrich_coefficient_summary_with_robust(
        self,
        summaries: list[CoefficientSummary],
        result,
        cov_type: str = "HC1",
    ) -> tuple[bool, str | None]:
        """Enrich coefficient summaries with robust standard errors in place.

        Returns ``(enriched, reason)``:

        - ``(True, None)`` when robust SEs were attached to every coefficient.
        - ``(False, reason)`` when they could not be computed. ``reason`` is
          a short human-readable string the caller can surface as a warning
          so the user knows why ``robust_*`` fields are null (instead of
          silently leaving them ``None``, which the previous bool-only
          return forced).

        The two known not-applicable cases:

        - Models fit without a stored design matrix (``store_design_matrix=False``,
          deserialized models) — ``AttributeError`` from ``bse_robust``.
        - Regularized fits (``alpha > 0``) where sandwich SEs aren't
          statistically meaningful — ``ValueError`` from ``bse_robust``.
        """
        try:
            robust_bse = np.asarray(result.bse_robust(cov_type))
            robust_tvalues = np.asarray(result.tvalues_robust(cov_type))
            robust_pvalues = np.asarray(result.pvalues_robust(cov_type))
        except AttributeError:
            return False, (
                "robust SE unavailable — fitted model does not expose "
                "bse_robust() (lean / deserialized model)"
            )
        except ValueError as exc:
            return False, f"robust SE unavailable — {exc}"

        # Build a lookup by feature name since summaries may be reordered
        feature_to_idx = {name: i for i, name in enumerate(self.feature_names)}

        for s in summaries:
            idx = feature_to_idx.get(s.feature)
            if idx is not None and idx < len(robust_bse):
                s.robust_std_error = round(float(robust_bse[idx]), 6)
                s.robust_z_value = round(float(robust_tvalues[idx]), 3)
                s.robust_p_value = round(float(robust_pvalues[idx]), 4)
                s.robust_significant = float(robust_pvalues[idx]) < 0.05

        return True, None

    def compute_factor_deviance(
        self,
        data: pl.DataFrame,
        categorical_factors: list[str],
        cat_column_cache: dict[str, np.ndarray | None] | None = None,
        cat_unique_cache: dict[str, tuple | None] | None = None,
    ) -> list[FactorDeviance]:
        """
        Compute deviance breakdown by factor level.

        Uses Rust backend for fast groupby aggregation on large datasets.
        All factors are processed in a single batched Rust call with rayon
        parallelism across factors (OPT-9).

        Fast path: when `cat_unique_cache` is provided (mapping factor name to
        a tuple of (sorted_levels: ndarray[str], codes: ndarray[uint32])), the
        code-based Rust entry is used and per-row string marshalling is
        completely avoided.

        Slow path: when only `cat_column_cache` is available (string arrays),
        the string-based batch entry is used. Without either cache, falls back
        to `data[name].cast(str).to_list()`.

        Identifies which categorical levels are driving poor fit,
        helping the agent pinpoint problem areas.

        Returns
        -------
        list of FactorDeviance
            Deviance breakdown for each categorical factor
        """
        # Fast path: use the code-based batch when uint32 codes are available
        # for every factor. This skips per-row string marshalling entirely.
        valid_factors: list[str] = []
        codes_columns: list[np.ndarray] = []
        levels_per_factor: list[list[str]] = []
        can_use_codes = cat_unique_cache is not None

        if can_use_codes:
            for factor_name in categorical_factors:
                if factor_name not in data.columns:
                    continue
                entry = cat_unique_cache.get(factor_name)
                if entry is None:
                    can_use_codes = False
                    break
                sorted_levels, codes = entry
                valid_factors.append(factor_name)
                codes_columns.append(codes)
                levels_per_factor.append(list(sorted_levels))

        if can_use_codes and valid_factors:
            from rustystats._rustystats import (
                compute_factor_deviance_batch_from_codes_py as _rust_factor_deviance_batch_from_codes,
            )

            # Stack code columns into an (n, k) uint32 matrix for the FFI
            # boundary; this is one allocation, fast.
            codes_matrix = np.stack(codes_columns, axis=1).astype(np.uint32, copy=False)

            rust_results = _rust_factor_deviance_batch_from_codes(
                valid_factors,
                codes_matrix,
                levels_per_factor,
                self.y,
                self.mu,
                self.family,
                getattr(self, "var_power", 1.5),
                getattr(self, "theta", 1.0),
            )
        else:
            # Slow path: collect string values and use the string-based batch.
            # Prefer the pre-materialized array from cat_column_cache to skip
            # cast(str).to_list(). Use ndarray.tolist() — ~6x faster than
            # list(ndarray) AND yields native Python str (not numpy.str_),
            # which PyO3 marshals faster.
            from rustystats._rustystats import (
                compute_factor_deviance_batch_py as _rust_factor_deviance_batch,
            )

            valid_factors = []
            values_list: list[list[str]] = []
            for factor_name in categorical_factors:
                if factor_name not in data.columns:
                    continue

                cached = cat_column_cache.get(factor_name) if cat_column_cache is not None else None
                if cached is not None:
                    values_list.append(cached.tolist())
                else:
                    values_list.append(data[factor_name].cast(str).to_list())
                valid_factors.append(factor_name)

            if not valid_factors:
                return []

            rust_results = _rust_factor_deviance_batch(
                valid_factors,
                values_list,
                self.y,
                self.mu,
                self.family,
                getattr(self, "var_power", 1.5),
                getattr(self, "theta", 1.0),
            )

        if not valid_factors:
            return []

        # Convert each Rust result to Python dataclasses, preserving the
        # existing rounding, NaN handling, and `problem` field semantics.
        results: list[FactorDeviance] = []
        for rust_result in rust_results:
            levels = [
                DevianceByLevel(
                    level=level["level"],
                    n=level["count"],
                    deviance=round(level["deviance"], 2),
                    deviance_pct=round(level["deviance_pct"], 2),
                    mean_deviance=round(level["mean_deviance"], 4),
                    ae_ratio=round(level["ae_ratio"], 3)
                    if not np.isnan(level["ae_ratio"])
                    else None,
                    problem=level["is_problem"],
                )
                for level in rust_result["levels"]
            ]

            results.append(
                FactorDeviance(
                    factor=rust_result["factor_name"],
                    total_deviance=round(rust_result["total_deviance"], 2),
                    levels=levels,
                    problem_levels=rust_result["problem_levels"],
                )
            )

        return results

    def _rank_sort_idx(self, ranking: str = "auto") -> np.ndarray:
        return rank_sort_idx(
            self.mu,
            self.exposure,
            has_exposure=self._has_exposure,
            ranking=ranking,
        )

    def _gini_for_sort_idx(self, sort_idx: np.ndarray) -> float:
        """Compute exposure-weighted Gini using the same rank order as deciles."""
        total_exposure = float(np.sum(self.exposure))
        total_actual = float(np.sum(self.y))
        if total_actual == 0.0 or total_exposure == 0.0:
            return 0.0

        cum_exposure = 0.0
        cum_actual = 0.0
        gini_area = 0.0
        prev_cum_exposure_pct = 0.0
        prev_cum_actual_pct = 0.0

        for idx in sort_idx[::-1]:
            cum_exposure += float(self.exposure[idx])
            cum_actual += float(self.y[idx])
            cum_exposure_pct = cum_exposure / total_exposure
            cum_actual_pct = cum_actual / total_actual
            gini_area += (
                (cum_exposure_pct - prev_cum_exposure_pct)
                * (cum_actual_pct + prev_cum_actual_pct)
                / 2.0
            )
            prev_cum_exposure_pct = cum_exposure_pct
            prev_cum_actual_pct = cum_actual_pct

        return 2.0 * gini_area - 1.0

    def compute_lift_chart(
        self, n_deciles: int = 10, sort_idx: np.ndarray | None = None, ranking: str = "auto"
    ) -> LiftChart:
        """
        Compute full lift chart with all deciles.

        Shows where the model discriminates well vs poorly,
        helping the agent identify risk bands needing attention.

        Parameters
        ----------
        n_deciles : int, default=10
            Number of deciles for binning.
        sort_idx : np.ndarray, optional
            Pre-computed argsort of self.mu. Pass this when multiple downstream
            consumers need the same sort to avoid redundant O(n log n) work.

        Returns
        -------
        LiftChart
            Complete lift chart with discrimination metrics
        """
        # Rank by predicted rate (mu/exposure) when exposure is present, matching
        # the decile/calibration/discrimination diagnostics (RS-ACT-004). Reuse a
        # pre-computed sort when provided.
        if ranking not in {"auto", "mean", "rate"}:
            raise ValidationError(f"ranking must be 'auto', 'mean', or 'rate', got {ranking!r}.")
        if ranking == "rate" and not self._has_exposure:
            raise ValidationError("ranking='rate' requires exposure to be supplied.")
        if sort_idx is None:
            sort_idx = self._rank_sort_idx(ranking)
        y_sorted = self.y[sort_idx]
        mu_sorted = self.mu[sort_idx]
        exp_sorted = self.exposure[sort_idx]

        # Overall rate
        overall_rate = np.sum(self.y) / np.sum(self.exposure)

        # Compute deciles
        n = len(self.y)
        decile_size = n // n_deciles

        deciles = []
        cumulative_actual = 0
        cumulative_predicted = 0
        total_actual = np.sum(self.y)
        total_predicted = np.sum(self.mu)

        max_ks = 0
        ks_decile = 1
        weak_deciles = []

        for d in range(n_deciles):
            start = d * decile_size
            end = (d + 1) * decile_size if d < n_deciles - 1 else n

            y_d = y_sorted[start:end]
            mu_d = mu_sorted[start:end]
            exp_d = exp_sorted[start:end]

            actual = float(np.sum(y_d))
            predicted = float(np.sum(mu_d))
            exposure = float(np.sum(exp_d))
            n_d = len(y_d)

            ae_ratio = actual / predicted if predicted > 0 else float("nan")

            cumulative_actual += actual
            cumulative_predicted += predicted

            cum_actual_pct = 100 * cumulative_actual / total_actual if total_actual > 0 else 0
            cum_pred_pct = (
                100 * cumulative_predicted / total_predicted if total_predicted > 0 else 0
            )

            # Lift: rate in this decile / overall rate
            decile_rate = actual / exposure if exposure > 0 else 0
            lift = decile_rate / overall_rate if overall_rate > 0 else 1.0

            # Cumulative lift
            cum_rate = (
                cumulative_actual / np.sum(exp_sorted[:end]) if np.sum(exp_sorted[:end]) > 0 else 0
            )
            cum_lift = cum_rate / overall_rate if overall_rate > 0 else 1.0

            # KS statistic
            ks = abs(cum_actual_pct - cum_pred_pct)
            if ks > max_ks:
                max_ks = ks
                ks_decile = d + 1

            # Weak deciles: poor A/E or lift close to 1
            if abs(ae_ratio - 1.0) > 0.2 or (d < 3 and lift > 0.8) or (d > 6 and lift < 1.2):
                weak_deciles.append(d + 1)

            predicted_rate = predicted / exposure if exposure > 0 else 0
            deciles.append(
                LiftDecile(
                    decile=d + 1,
                    n=n_d,
                    exposure=round(exposure, 2),
                    actual=round(decile_rate, 6),
                    predicted=round(predicted_rate, 6),
                    ae_ratio=round(ae_ratio, 3) if not np.isnan(ae_ratio) else None,
                    cumulative_actual_pct=round(cum_actual_pct, 2),
                    cumulative_predicted_pct=round(cum_pred_pct, 2),
                    lift=round(lift, 3),
                    cumulative_lift=round(cum_lift, 3),
                )
            )

        gini = self._gini_for_sort_idx(sort_idx)

        return LiftChart(
            deciles=deciles,
            gini=round(gini, 3),
            ks_statistic=round(max_ks, 2),
            ks_decile=ks_decile,
            weak_deciles=weak_deciles,
        )

    def _compute_eta_contribution(
        self,
        var: str,
        grid: np.ndarray,
        result,
        feature_to_idx: dict[str, int],
    ) -> np.ndarray:
        """
        Compute the η (linear-predictor) contribution from variable `var` at each grid value.

        For a GLM, η is additive: ``η = β₀ + Σ_j f_j(x_j) + offset``. This method
        returns ``f_var(grid)`` for the supplied grid. Callers compute the partial
        dependence as ``g_inv(η_baseline + f_var(g) - f_var(baseline_g))``.

        Supports the following feature types found in ``feature_names``:

        - ``var``                 : linear term (``coef * grid``)
        - ``bs(var, ...)``        : B-spline basis (re-evaluated using stored knots)
        - ``ns(var, ...)``        : natural spline basis (re-evaluated using stored knots)
        - ``ms(var, ...)``        : monotonic spline basis (re-evaluated using stored knots)
        - ``s(var, ...)``         : smooth-term variant (treated like its underlying spline type)
        - ``I(<expr>)``           : identity expression (e.g. ``I(var ** 2)``)
        - ``TE(var)``             : target-encoded categorical (no continuous PD; returns 0)

        Parameters
        ----------
        var : str
            Variable name (the user-facing column name, NOT the transformed feature name).
        grid : np.ndarray
            1-D array of grid values to evaluate at, shape (n_grid,).
        result : GLMModel
            The fitted model. Uses ``result.params`` and ``result._builder``.
        feature_to_idx : dict[str, int]
            Pre-built mapping from feature name to coefficient index.

        Returns
        -------
        np.ndarray
            η-contribution at each grid value, shape (n_grid,). Returns zeros if no
            features for ``var`` are found in ``feature_names``.
        """
        params = np.asarray(result.params, dtype=np.float64)
        builder = getattr(result, "_builder", None)
        fitted_splines = getattr(builder, "_fitted_splines", {}) if builder is not None else {}

        grid = np.asarray(grid, dtype=np.float64).ravel()
        eta = np.zeros_like(grid)
        found_any = False

        # FIX-N: route feature filtering through the strict-matching
        # per-variable cache owned by _FactorDiagnosticsComputer (single
        # source of truth). Walking only the matched feature names below
        # avoids substring false positives (e.g. `Age` previously could
        # match `bs(VehAge, 1/4)`) AND avoids scanning the full feature_names
        # list per branch.
        feat = self._factors._get_feature_for(var)
        matched_feature_names = feat.feature_names

        # 1. Linear term: exact name match.
        linear_idx = feature_to_idx.get(var)
        if linear_idx is not None:
            eta = eta + float(params[linear_idx]) * grid
            found_any = True

        # 2. Spline / smooth term: re-evaluate basis on grid using stored knots.
        if var in fitted_splines:
            spline_term = fitted_splines[var]
            # Calling .transform() on a fitted SplineTerm reuses the stored
            # boundary + internal knots (set during training), so the basis on
            # `grid` is exactly the same family as during fit.
            try:
                basis, names = spline_term.transform(grid)
            except Exception:
                # If basis evaluation fails (e.g. unexpected NaN in grid),
                # fall back to no spline contribution rather than error out.
                basis, names = None, []
            if basis is not None and len(names) > 0:
                # Map each generated basis-column name to its coefficient.
                # If a name isn't found (regularization may zero / drop columns),
                # the coefficient is treated as zero.
                col_coefs = np.zeros(len(names), dtype=np.float64)
                for j, nm in enumerate(names):
                    idx = feature_to_idx.get(nm)
                    if idx is not None:
                        col_coefs[j] = float(params[idx])
                        found_any = True
                eta = eta + basis @ col_coefs
        # If `var` has spline features in feature_names but no fitted_splines
        # entry, we conservatively skip — there is no correct way to evaluate
        # the basis without the knots.

        # 3. Identity expression: I(<expr>) where <expr> involves var.
        # Iterate ONLY the strict-index-matched features so an unrelated
        # expression like I(other_var ** 2) cannot be picked up for `var`.
        for name in matched_feature_names:
            if not (name.startswith("I(") and name.endswith(")")):
                continue
            expr = name[2:-1].strip()
            idx = feature_to_idx.get(name)
            if idx is None:
                continue
            coef = float(params[idx])
            if coef == 0.0:
                # Term was zeroed by regularization — no contribution.
                found_any = True
                continue
            # Evaluate the expression with `var = grid` on a single-column
            # polars DataFrame, reusing the builder's expression converter.
            if builder is None or not hasattr(builder, "_convert_expression_to_polars"):
                continue
            try:
                import polars as _pl

                tmp = _pl.DataFrame({var: grid})
                pl_expr = builder._convert_expression_to_polars(expr)
                col = tmp.select(pl_expr.alias("__r__"))["__r__"].to_numpy()
                eta = eta + coef * col.astype(np.float64)
                found_any = True
            except Exception:
                # Expression couldn't be evaluated on the grid (likely
                # involves other columns); skip silently.
                continue

        # 4. Target encoding TE(var): not meaningful for a continuous PD on
        # `var`'s numeric range — TE maps levels (strings) → encoded means.
        # We return whatever the linear/spline/expression contributions gave,
        # plus mark `found_any` if a TE feature exists so callers don't warn.
        # Use the strict index — it correctly recognizes both `TE(var)` and
        # `TE(...:var:...)` interaction-TE features.
        if any(name.startswith("TE(") for name in matched_feature_names):
            found_any = True

        if not found_any:
            return np.zeros_like(grid)
        return eta

    def compute_partial_dependence(
        self,
        data: pl.DataFrame,
        result,  # GLMResults with predict capability
        continuous_factors: list[str],
        categorical_factors: list[str],
        link: str = "log",
        n_grid: int = 20,
        cat_column_cache: dict[str, np.ndarray | None] | None = None,
        cat_unique_cache: dict[str, tuple | None] | None = None,
        cont_column_cache: dict[str, np.ndarray | None] | None = None,
    ) -> list[PartialDependence]:
        """
        Compute partial dependence for each variable.

        Shows the marginal effect shape, helping the agent decide
        between linear, spline, or banding approaches.

        Returns
        -------
        list of PartialDependence
            Partial dependence for each variable
        """
        # OPT-10 Phase 1: cache values that are constant across all factors/grid points.
        # np.mean(self.mu) was previously computed 24 + 6 = 30 times; now once.
        base_pred = float(np.mean(self.mu))
        # Feature-name -> index map avoids O(features) string scan per grid point.
        feature_to_idx = {name: i for i, name in enumerate(self.feature_names)}
        is_log_link = link == "log"

        # FIX-O B2: invert the link once for the baseline mean, so the loop below
        # can compute pred = link.inverse(eta_baseline + delta_eta) for ANY link.
        # Previously the non-log branch used `base_pred + delta_eta`, which is
        # only correct for the identity link and silently produced wrong PD
        # probabilities for binomial models with the canonical logit link.
        eta_baseline_response = float(link_forward(link, base_pred))

        results = []

        # Continuous variables
        for var in continuous_factors:
            if var not in data.columns:
                continue

            values = (
                cont_column_cache[var]
                if cont_column_cache and var in cont_column_cache
                else data[var].to_numpy().astype(np.float64)
            )
            valid_mask = ~np.isnan(values) & ~np.isinf(values)
            valid_values = values[valid_mask]

            if len(valid_values) < 10:
                continue

            # Create grid
            grid = np.linspace(
                np.percentile(valid_values, 1), np.percentile(valid_values, 99), n_grid
            )

            # OPT-10 Phase 1: grid_mean is constant across the 20 grid points.
            grid_mean = float(np.mean(valid_values))

            # FIX-M: Decompose η = Σ f_j(x_j) and use the additive structure of
            # the GLM linear predictor. For a spline like bs(var, df=4), the
            # contribution to η is Σ_k β_k · basis_k(grid), NOT β · (grid - mean)
            # picked from one fuzzy-matched coefficient. The previous fuzzy lookup
            # could pick e.g. coef=1.45 for a spline column and then compute
            # exp(1.45 * 150) → overflow → NaN downstream.
            eta_contrib = self._compute_eta_contribution(var, grid, result, feature_to_idx)
            eta_baseline = self._compute_eta_contribution(
                var, np.array([grid_mean]), result, feature_to_idx
            )[0]
            delta_eta = eta_contrib - eta_baseline

            # FIX-O B2: use the link's inverse to map the linear-predictor
            # contribution back to the response scale uniformly for every link.
            #   eta_baseline = g(base_pred)               # via link_forward
            #   pred         = g⁻¹(eta_baseline + Δη)     # via link_inverse
            # link_inverse already clips its argument to ±50 internally for
            # numerical safety, but we clip delta_eta here too as defence in
            # depth: if clipping ever fires that's a fitting bug, not a PD bug.
            delta_eta_clipped = np.clip(delta_eta, -50.0, 50.0)
            pred_arr = link_inverse(link, eta_baseline_response + delta_eta_clipped)
            predictions = [float(p) for p in pred_arr]

            # Analyze shape
            shape, recommendation = self._analyze_pd_shape(grid, predictions, link)

            # Convert to relativities for log-link
            relativities = None
            if is_log_link and predictions:
                base = predictions[len(predictions) // 2]
                relativities = [p / base if base > 0 else 1.0 for p in predictions]

            results.append(
                PartialDependence(
                    variable=var,
                    variable_type="continuous",
                    grid_values=[round(float(g), 4) for g in grid],
                    predictions=[round(p, 6) for p in predictions],
                    relativities=[round(r, 4) for r in relativities] if relativities else None,
                    shape=shape,
                    recommendation=recommendation,
                )
            )

        # OPT-20: pre-compute per-level (counts, mu_sums) for every categorical
        # factor that has cached codes in a single rayon-parallel Rust call,
        # replacing 6 sequential pairs of `np.bincount` per factor over n=1M.
        # Factors without cached codes still take the per-loop slow path below.
        pd_cat_lookup: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if cat_unique_cache:
            cached_factors: list[str] = []
            cached_levels: list[np.ndarray] = []
            cached_codes: list[np.ndarray] = []
            for var in categorical_factors:
                if var not in data.columns:
                    continue
                entry = cat_unique_cache.get(var)
                if entry is None:
                    continue
                unique_levels, codes = entry
                cached_factors.append(var)
                cached_levels.append(unique_levels)
                cached_codes.append(codes)

            if cached_factors:
                # Memory-hardening (Change 7): pass a list of 1D u32 arrays
                # instead of a stacked (n, k) matrix; this avoids a 400 MB
                # transient `np.stack` allocation at n=1M × k=100. The Rust
                # binding borrows each numpy buffer directly with no copy.
                # `astype(..., copy=False)` returns the input unchanged when
                # it is already u32; otherwise it allocates a single column-
                # sized cast (much smaller than the stacked matrix).
                codes_list = [c.astype(np.uint32, copy=False) for c in cached_codes]
                n_levels_per_factor = [len(lv) for lv in cached_levels]
                batch_results = _rust_partial_dependence_categorical_batch(
                    codes_list, self.mu, n_levels_per_factor
                )
                for var, unique_levels, (counts, mu_sums) in zip(
                    cached_factors, cached_levels, batch_results
                ):
                    pd_cat_lookup[var] = (
                        unique_levels,
                        np.asarray(counts, dtype=np.float64),
                        np.asarray(mu_sums, dtype=np.float64),
                    )

        # Categorical variables
        for var in categorical_factors:
            if var not in data.columns:
                continue

            if var in pd_cat_lookup:
                # OPT-20 fast path: aggregates pre-computed in Rust above.
                unique_levels, counts, mu_sums = pd_cat_lookup[var]
            else:
                # Slow fallback: factor wasn't in `cat_unique_cache`, so
                # extract + factorize here, then bincount in NumPy as before.
                # OPT-10 Phase 1: prefer the cached (levels, codes) pair — avoids
                # re-extracting + re-factorizing the string column per factor.
                if cat_unique_cache and var in cat_unique_cache:
                    unique_levels, inverse = cat_unique_cache[var]
                else:
                    values = (
                        cat_column_cache[var]
                        if cat_column_cache and var in cat_column_cache
                        else data[var].to_numpy().astype(str)
                    )
                    unique_levels, inverse = np.unique(values, return_inverse=True)
                k = len(unique_levels)
                counts = np.bincount(inverse, minlength=k).astype(np.float64)
                mu_sums = np.bincount(inverse, weights=self.mu, minlength=k)

            grid_values = list(unique_levels)
            k = len(unique_levels)
            predictions = []
            for j in range(k):
                if counts[j] > 0:
                    predictions.append(float(mu_sums[j] / counts[j]))
                else:
                    predictions.append(base_pred)

            # Analyze categorical effect
            if len(predictions) > 1:
                max_pred = max(predictions)
                min_pred = min(predictions)
                range_ratio = max_pred / min_pred if min_pred > 0 else float("inf")

                if range_ratio > 2:
                    shape = "high_variation"
                    recommendation = "Keep as categorical - significant level differences"
                elif range_ratio > 1.2:
                    shape = "moderate_variation"
                    recommendation = "Categorical appropriate, consider grouping similar levels"
                else:
                    shape = "low_variation"
                    recommendation = "Consider removing - little variation across levels"
            else:
                shape = "single_level"
                recommendation = "Cannot assess with single level"

            relativities = None
            if is_log_link and predictions:
                base = predictions[0]  # First level as base
                relativities = [p / base if base > 0 else 1.0 for p in predictions]

            results.append(
                PartialDependence(
                    variable=var,
                    variable_type="categorical",
                    grid_values=grid_values,
                    predictions=[round(p, 6) for p in predictions],
                    relativities=[round(r, 4) for r in relativities] if relativities else None,
                    shape=shape,
                    recommendation=recommendation,
                )
            )

        return results

    def _analyze_pd_shape(
        self,
        grid: np.ndarray,
        predictions: list[float],
        link: str,
    ) -> tuple:
        """Analyze partial dependence shape and provide recommendation."""
        if len(predictions) < 3:
            return "insufficient_data", "Need more data points"

        preds = np.array(predictions)

        # Compute differences
        diffs = np.diff(preds)

        # Check monotonicity
        increasing = np.sum(diffs > 0)
        decreasing = np.sum(diffs < 0)
        n_diffs = len(diffs)

        # Analyze curvature
        second_diffs = np.diff(diffs)
        curvature = np.mean(np.abs(second_diffs))

        # Relative range
        pred_range = np.max(preds) - np.min(preds)
        pred_mean = np.mean(preds)
        relative_range = pred_range / pred_mean if pred_mean > 0 else 0

        if relative_range < PD_FLAT_RELATIVE_RANGE:
            return "flat", "May not need in model - negligible effect"

        if increasing >= n_diffs * PD_MONOTONIC_THRESHOLD:
            if curvature < pred_range * PD_CURVATURE_RELATIVE_THRESHOLD:
                return "linear_increasing", "Linear effect adequate"
            else:
                return "monotonic_increasing", "Consider spline for non-linearity"

        if decreasing >= n_diffs * PD_MONOTONIC_THRESHOLD:
            if curvature < pred_range * PD_CURVATURE_RELATIVE_THRESHOLD:
                return "linear_decreasing", "Linear effect adequate"
            else:
                return "monotonic_decreasing", "Consider spline for non-linearity"

        # Check for U-shape
        mid = len(preds) // 2
        left_trend = np.mean(diffs[:mid]) if mid > 0 else 0
        right_trend = np.mean(diffs[mid:]) if mid < len(diffs) else 0

        if left_trend < 0 and right_trend > 0:
            return "u_shaped", "Use spline (df=4+) or polynomial"
        if left_trend > 0 and right_trend < 0:
            return "inverted_u", "Use spline (df=4+) or polynomial"

        # Check for step function
        max_jump = np.max(np.abs(diffs))
        if max_jump > pred_range * PD_STEP_FUNCTION_RATIO:
            return "step_function", "Consider banding/categorical transformation"

        return "complex", "Use spline (df=5+) to capture non-linearity"

    def compute_dataset_diagnostics(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        exposure: np.ndarray,
        data: pl.DataFrame,
        categorical_factors: list[str],
        continuous_factors: list[str],
        dataset_name: str,
        result=None,
        n_bands: int = 10,
        cat_column_cache: dict[str, np.ndarray | None] | None = None,
        cont_column_cache: dict[str, np.ndarray | None] | None = None,
        cat_unique_cache: dict[str, tuple | None] | None = None,
        sort_idx: np.ndarray | None = None,
    ) -> DatasetDiagnostics:
        """Compute comprehensive diagnostics for a single dataset."""
        n_obs = len(y)
        total_exposure = float(np.sum(exposure))
        total_actual = float(np.sum(y))
        total_predicted = float(np.sum(mu))

        # Family deviance metrics (same as GBM loss) using Rust backend
        dataset_metrics = _rust_dataset_metrics(y, mu, self.family, self.n_params)
        deviance = float(dataset_metrics["deviance"])
        mean_deviance = float(dataset_metrics["mean_deviance"])
        log_likelihood = float(dataset_metrics["log_likelihood"])
        aic_val = float(dataset_metrics["aic"])
        bic_val = (
            -2.0 * log_likelihood + self.n_params * math.log(n_obs) if n_obs > 0 else float("nan")
        )

        # Discrimination metrics
        stats = _rust_discrimination_stats(y, mu, exposure)
        gini = float(stats["gini"])
        auc = float(stats["auc"])

        # Overall A/E
        ae_ratio = total_actual / total_predicted if total_predicted > 0 else float("nan")

        # A/E by decile (sorted by predicted value). Forward `sort_idx` so we
        # don't re-sort the same `mu` array the orchestrator already sorted.
        ae_by_decile = self._compute_ae_by_decile(y, mu, exposure, n_deciles=10, sort_idx=sort_idx)

        # Compute deviance residuals once for all factor/continuous diagnostics
        dev_resids = (
            np.asarray(_rust_deviance_residuals(y, mu, self.family))
            if (categorical_factors or continuous_factors)
            else None
        )

        # Factor-level diagnostics. Stage all factors, then dispatch a single
        # Rust batch call (parallelised internally over k via rayon) instead of
        # k sequential per-factor passes through five np.bincount calls.
        factor_diag = self._compute_factor_level_metrics(
            y,
            mu,
            exposure,
            data,
            categorical_factors,
            dev_resids,
            cat_unique_cache=cat_unique_cache,
        )

        # Continuous variable diagnostics. Same batched-Rust strategy as above:
        # one cross-FFI call replaces k per-factor np.percentile + np.digitize +
        # five np.bincount loops on the 1M-row arrays.
        continuous_diag = self._compute_continuous_band_metrics(
            y,
            mu,
            exposure,
            data,
            continuous_factors,
            dev_resids,
            n_bands,
            cont_column_cache=cont_column_cache,
        )

        return DatasetDiagnostics(
            dataset=dataset_name,
            n_obs=n_obs,
            total_exposure=round(total_exposure, 2),
            total_actual=round(total_actual, 2),
            total_predicted=round(total_predicted, 2),
            loss=round(mean_deviance, 6),
            deviance=round(deviance, 2),
            log_likelihood=round(log_likelihood, 2),
            aic=round(aic_val, 2),
            bic=round(bic_val, 2),
            gini=round(gini, 4),
            auc=round(auc, 4),
            ae_ratio=round(ae_ratio, 4),
            ae_by_decile=ae_by_decile,
            factor_diagnostics=factor_diag,
            continuous_diagnostics=continuous_diag,
        )

    def _compute_ae_by_decile(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        exposure: np.ndarray,
        n_deciles: int = 10,
        sort_idx: np.ndarray | None = None,
    ) -> list[DecileMetrics]:
        """Compute A/E by decile sorted by predicted value.

        Pass `sort_idx` (a pre-computed `np.argsort(mu)`) when the caller already
        has it, to skip a redundant O(n log n) sort on the prediction array.

        Per-decile sums (`actual_sum`, `predicted_sum`, `exposure_sum`) are
        computed in Rust via `compute_ae_by_decile_py`; the trivial divisions
        that produce frequencies and the A/E ratio stay Python-side.
        """
        # Hand the pre-computed sort index to Rust as native uintp so the FFI
        # layer can reuse it directly. `np.argsort` returns intp, so this is a
        # no-op on 64-bit platforms but normalises the dtype on 32-bit.
        sort_idx_arg = (
            np.ascontiguousarray(sort_idx, dtype=np.uintp) if sort_idx is not None else None
        )

        raw = _rust_ae_by_decile(y, mu, exposure, n_deciles, sort_idx_arg)

        deciles: list[DecileMetrics] = []
        for r in raw:
            actual = float(r["actual_sum"])
            predicted = float(r["predicted_sum"])
            exp_sum = float(r["exposure_sum"])
            ae = actual / predicted if predicted > 0 else float("nan")
            actual_freq = actual / exp_sum if exp_sum > 0 else 0.0
            predicted_freq = predicted / exp_sum if exp_sum > 0 else 0.0
            deciles.append(
                DecileMetrics(
                    decile=int(r["decile"]),
                    n=int(r["n"]),
                    exposure=round(exp_sum, 2),
                    actual=round(actual_freq, 6),
                    predicted=round(predicted_freq, 6),
                    ae_ratio=round(ae, 4) if not np.isnan(ae) else None,
                )
            )

        return deciles

    def _compute_factor_level_metrics(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        exposure: np.ndarray,
        data: pl.DataFrame,
        categorical_factors: list[str],
        deviance_residuals: np.ndarray | None,
        cat_unique_cache: dict[str, tuple],
    ) -> dict[str, list[FactorLevelMetrics]]:
        """Compute factor-level metrics for all categorical factors in one shot.

        Stacks per-factor (codes, levels) into a (n × k) u32 codes_matrix and
        dispatches a single Rust batch call (`compute_ae_categorical_batch_py`)
        which runs the per-factor sums-by-level work in parallel via rayon.
        Residual means are still computed Python-side via one np.bincount per
        factor against the cached (unique, inverse) array.

        `cat_unique_cache` must contain a (levels, codes) entry for every
        factor present in `data.columns`. Callers (`api.py`, `explorer.py`)
        populate this in lockstep with the factor list; a missing entry
        indicates a caller bug and raises KeyError.
        """
        factor_diag: dict[str, list[FactorLevelMetrics]] = {}

        # Stage per-factor work using the cached (levels, codes) tuple.
        cat_entries: list[dict] = []
        for factor in categorical_factors:
            if factor not in data.columns:
                continue

            unique_levels, inverse = cat_unique_cache[factor]
            cat_entries.append(
                {
                    "name": factor,
                    "unique": unique_levels,
                    "inverse": np.ascontiguousarray(inverse, dtype=np.uint32),
                }
            )

        # Dispatch the batched Rust call with a list of per-factor code arrays.
        # Passing contiguous 1D arrays (instead of an (n, k) matrix) lets the
        # Rust binding zero-copy each column into a `&[u32]` slice, avoiding
        # both the Python matrix allocation and a previously-required
        # per-column Rust-side `to_vec()`.
        if cat_entries:
            codes_list: list[np.ndarray] = [entry["inverse"] for entry in cat_entries]
            levels_list: list[list[str]] = [
                [str(v) for v in entry["unique"]] for entry in cat_entries
            ]

            # Pass loose thresholds so all populated levels are kept (the
            # singular Python helper returns every level; preserve that). The
            # Rust check is `pct < threshold || bin_idx >= max_levels - 1`, so
            # threshold=-1 and max_levels=max_k+2 trigger neither branch.
            max_k_plus_2 = max(len(lst) for lst in levels_list) + 2

            ae_batch = _rust_ae_categorical_batch(
                codes_list,
                levels_list,
                y,
                mu,
                exposure,
                -1.0,  # rare_threshold_pct: never rare
                max_k_plus_2,  # max_levels: never overflow into "_Other"
                self.family,
            )

            # Compute residual_means per factor with one bincount each, then
            # build FactorLevelMetrics from the (level, count, exposure_sum,
            # actual_sum, predicted_sum, ae_ratio) values Rust returned.
            for entry, rust_bins in zip(cat_entries, ae_batch):
                unique_levels = entry["unique"]
                inverse = entry["inverse"]
                k = len(unique_levels)

                if deviance_residuals is not None:
                    resid_sum_by_code = np.bincount(
                        inverse, weights=deviance_residuals, minlength=k
                    )
                else:
                    resid_sum_by_code = np.zeros(k, dtype=np.float64)

                # Map level label -> code for residual lookup. unique_levels
                # is a numpy array of strings; build a small dict (k items).
                level_to_code = {str(lv): i for i, lv in enumerate(unique_levels)}

                metrics: list[FactorLevelMetrics] = []
                for b in rust_bins:
                    n = int(b["count"])
                    if n == 0:
                        continue

                    code = level_to_code.get(b["bin_label"])
                    resid_mean = float(resid_sum_by_code[code] / n) if code is not None else 0.0

                    exp_sum = float(b["exposure"])
                    actual_sum = float(b["actual_sum"])
                    predicted_sum = float(b["predicted_sum"])
                    # Recompute A/E from un-rounded sums in Python to match the
                    # singular path bit-for-bit; reading b["actual_expected_ratio"]
                    # picks up Rust-side division-rounding drift (~5e-3 worst case).
                    ae = actual_sum / predicted_sum if predicted_sum > 0 else float("nan")

                    actual_freq = actual_sum / exp_sum if exp_sum > 0 else 0.0
                    predicted_freq = predicted_sum / exp_sum if exp_sum > 0 else 0.0
                    metrics.append(
                        FactorLevelMetrics(
                            level=b["bin_label"],
                            n=n,
                            exposure=round(exp_sum, 2),
                            actual=round(actual_freq, 6),
                            predicted=round(predicted_freq, 6),
                            ae_ratio=round(ae, 4) if not np.isnan(ae) else None,
                            residual_mean=round(resid_mean, 6),
                        )
                    )

                # Rust already sorts levels by exposure descending; replicate
                # the singular path's explicit sort so output order is stable
                # regardless of any future Rust-side reordering.
                metrics.sort(key=lambda x: -x.exposure)
                factor_diag[entry["name"]] = metrics

        return factor_diag

    def _compute_continuous_band_metrics(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        exposure: np.ndarray,
        data: pl.DataFrame,
        continuous_factors: list[str],
        deviance_residuals: np.ndarray | None,
        n_bands: int,
        cont_column_cache: dict[str, np.ndarray],
    ) -> dict[str, list[ContinuousBandMetrics]]:
        """Compute continuous band metrics for all factors in one batched call.

        Stacks per-factor `values` arrays into an (n × k) f64 matrix and calls
        `compute_ae_continuous_batch_py`, which parallelises the per-factor
        quantile-binning + sums-by-bin work via rayon. For residual_mean we
        still need per-bin assignments — these are recovered cheaply by
        `np.searchsorted` on the bin edges Rust returned for each factor.

        `cont_column_cache` must contain an entry for every factor present in
        `data.columns`. Callers (`api.py`, `explorer.py`) populate this in
        lockstep with the factor list; a missing entry indicates a caller
        bug and raises KeyError.
        """
        continuous_diag: dict[str, list[ContinuousBandMetrics]] = {}

        cont_entries: list[dict] = []
        for var in continuous_factors:
            if var not in data.columns:
                continue
            cont_entries.append({"name": var, "values": cont_column_cache[var]})

        if not cont_entries:
            return continuous_diag

        # Pass a Python list of per-factor `values` arrays directly (each is
        # already contiguous f64). The Rust binding zero-copies into
        # `&[f64]` slices — avoiding both the (n × k) matrix allocation and
        # a previously-required per-column Rust-side `to_vec()`.
        values_list = [
            np.ascontiguousarray(entry["values"], dtype=np.float64) for entry in cont_entries
        ]

        ae_batch = _rust_ae_continuous_batch(
            values_list,
            y,
            mu,
            exposure,
            n_bands,
            self.family,
        )

        for entry, rust_bins in zip(cont_entries, ae_batch):
            values = entry["values"]
            metrics: list[ContinuousBandMetrics] = []

            # Skip empty results (Rust returns empty list when n_valid == 0
            # or when the factor has fewer valid rows than n_bands).
            if not rust_bins:
                continuous_diag[entry["name"]] = metrics
                continue

            # Reconstruct edges from Rust's per-bin bounds. Rust may collapse
            # adjacent identical-bound bins (when many ties), so derive edges
            # by walking the bins in order: edges = [b0.lower, b0.upper,
            # b1.upper, b2.upper, ...].
            edges = [rust_bins[0]["bin_lower"]]
            for b in rust_bins:
                edges.append(b["bin_upper"])
            edges_arr = np.asarray(edges, dtype=np.float64)
            n_bins = len(rust_bins)

            # Compute residual sums per bin in one pass.
            if deviance_residuals is not None:
                # np.searchsorted returns where each value would be inserted
                # in the sorted edges array. side='right' makes [edge_i,
                # edge_{i+1}) inclusive on the left, exclusive on the right;
                # subtract 1 and clip to [0, n_bins-1] to recover the bin
                # index. This matches Rust's binning rule (>= lower &&
                # (< upper || last bin)).
                bin_idx = np.searchsorted(edges_arr, values, side="right") - 1
                bin_idx = np.clip(bin_idx, 0, n_bins - 1)
                # Mask invalid entries so they don't contribute. Use a
                # sentinel bin index n_bins for invalids; minlength ensures
                # bincount allocates that bucket but we never read it.
                valid = np.isfinite(values)
                bin_idx_safe = np.where(valid, bin_idx, n_bins)
                resid_sums = np.bincount(
                    bin_idx_safe, weights=deviance_residuals, minlength=n_bins + 1
                )
            else:
                resid_sums = np.zeros(n_bins + 1, dtype=np.float64)

            for i, b in enumerate(rust_bins):
                n = int(b["count"])
                if n == 0:
                    continue

                lower = float(b["bin_lower"])
                upper = float(b["bin_upper"])
                exp_sum = float(b["exposure"])
                actual_sum = float(b["actual_sum"])
                predicted_sum = float(b["predicted_sum"])
                # Recompute A/E from un-rounded sums in Python to match the
                # singular path bit-for-bit; reading b["actual_expected_ratio"]
                # picks up Rust-side division-rounding drift (~5e-3 worst case).
                ae = actual_sum / predicted_sum if predicted_sum > 0 else float("nan")
                midpoint = (lower + upper) / 2
                partial_dep = predicted_sum / n
                resid_mean = float(resid_sums[i] / n)

                actual_freq = actual_sum / exp_sum if exp_sum > 0 else 0.0
                predicted_freq = predicted_sum / exp_sum if exp_sum > 0 else 0.0
                metrics.append(
                    ContinuousBandMetrics(
                        band=i + 1,
                        range_min=round(lower, 4),
                        range_max=round(upper, 4),
                        midpoint=round(midpoint, 4),
                        n=n,
                        exposure=round(exp_sum, 2),
                        actual=round(actual_freq, 6),
                        predicted=round(predicted_freq, 6),
                        ae_ratio=round(ae, 4) if not np.isnan(ae) else None,
                        partial_dep=round(partial_dep, 6),
                        residual_mean=round(resid_mean, 6),
                    )
                )

            continuous_diag[entry["name"]] = metrics

        return continuous_diag

    def compute_base_predictions_comparison(
        self,
        y: np.ndarray,
        mu_model: np.ndarray,
        mu_base: np.ndarray,
        exposure: np.ndarray,
        n_deciles: int = 10,
    ) -> BasePredictionsComparison:
        """
        Compute comparison between model predictions and base predictions.

        Parameters
        ----------
        y : np.ndarray
            Actual response values
        mu_model : np.ndarray
            Model predictions
        mu_base : np.ndarray
            Base/benchmark model predictions
        exposure : np.ndarray
            Exposure weights
        n_deciles : int
            Number of deciles for ratio analysis

        Returns
        -------
        BasePredictionsComparison
            Complete comparison with metrics and decile analysis
        """
        # Compute base metrics
        total_predicted_base = float(np.sum(mu_base))
        total_actual = float(np.sum(y))
        ae_ratio_base = (
            total_actual / total_predicted_base if total_predicted_base > 0 else float("nan")
        )

        # Base loss using Rust backend
        base_dataset_metrics = _rust_dataset_metrics(y, mu_base, self.family, self.n_params)
        base_loss = float(base_dataset_metrics["mean_deviance"])

        # Base discrimination
        base_stats = _rust_discrimination_stats(y, mu_base, exposure)
        base_gini = float(base_stats["gini"])
        base_auc = float(base_stats["auc"])

        base_metrics = BasePredictionsMetrics(
            total_predicted=round(total_predicted_base, 2),
            ae_ratio=round(ae_ratio_base, 4),
            loss=round(base_loss, 6),
            gini=round(base_gini, 4),
            auc=round(base_auc, 4),
        )

        # Model metrics for side-by-side comparison
        total_predicted_model = float(np.sum(mu_model))
        ae_ratio_model = (
            total_actual / total_predicted_model if total_predicted_model > 0 else float("nan")
        )
        model_dataset_metrics = _rust_dataset_metrics(y, mu_model, self.family, self.n_params)
        model_loss = float(model_dataset_metrics["mean_deviance"])
        model_stats = _rust_discrimination_stats(y, mu_model, exposure)
        model_gini = float(model_stats["gini"])
        model_auc = float(model_stats["auc"])

        model_metrics = BasePredictionsMetrics(
            total_predicted=round(total_predicted_model, 2),
            ae_ratio=round(ae_ratio_model, 4),
            loss=round(model_loss, 6),
            gini=round(model_gini, 4),
            auc=round(model_auc, 4),
        )

        # Compute model/base ratio and sort into deciles
        # Handle divide by zero - use small epsilon where base is 0
        mu_base_safe = np.where(mu_base > EPSILON, mu_base, EPSILON)
        model_base_ratio = mu_model / mu_base_safe

        # Sort by model/base ratio
        sort_idx = np.argsort(model_base_ratio)
        y_sorted = y[sort_idx]
        mu_model_sorted = mu_model[sort_idx]
        mu_base_sorted = mu_base[sort_idx]
        exp_sorted = exposure[sort_idx]
        ratio_sorted = model_base_ratio[sort_idx]

        n = len(y)
        decile_size = n // n_deciles

        deciles = []
        model_better_count = 0
        base_better_count = 0

        for d in range(n_deciles):
            start = d * decile_size
            end = (d + 1) * decile_size if d < n_deciles - 1 else n

            y_d = y_sorted[start:end]
            mu_model_d = mu_model_sorted[start:end]
            mu_base_d = mu_base_sorted[start:end]
            exp_d = exp_sorted[start:end]
            ratio_d = ratio_sorted[start:end]

            actual_sum = float(np.sum(y_d))
            model_sum = float(np.sum(mu_model_d))
            base_sum = float(np.sum(mu_base_d))
            exp_sum = float(np.sum(exp_d))

            model_ae = actual_sum / model_sum if model_sum > 0 else float("nan")
            base_ae = actual_sum / base_sum if base_sum > 0 else float("nan")

            # Frequencies (per exposure)
            actual_freq = actual_sum / exp_sum if exp_sum > 0 else 0.0
            model_freq = model_sum / exp_sum if exp_sum > 0 else 0.0
            base_freq = base_sum / exp_sum if exp_sum > 0 else 0.0

            # Mean ratio in this decile
            ratio_mean = float(np.mean(ratio_d))

            deciles.append(
                ModelVsBaseDecile(
                    decile=d + 1,
                    n=len(y_d),
                    exposure=round(exp_sum, 2),
                    actual=round(actual_freq, 6),
                    model_predicted=round(model_freq, 6),
                    base_predicted=round(base_freq, 6),
                    model_ae_ratio=round(model_ae, 4) if not np.isnan(model_ae) else None,
                    base_ae_ratio=round(base_ae, 4) if not np.isnan(base_ae) else None,
                    model_base_ratio_mean=round(ratio_mean, 4),
                )
            )

            # Count which model is better (A/E closer to 1)
            if not np.isnan(model_ae) and not np.isnan(base_ae):
                model_dist = abs(model_ae - 1.0)
                base_dist = abs(base_ae - 1.0)
                if model_dist < base_dist:
                    model_better_count += 1
                elif base_dist < model_dist:
                    base_better_count += 1

        # Improvement metrics (positive = model is better)
        loss_improvement = 0.0
        if base_loss > 0:
            loss_improvement = (base_loss - model_loss) / base_loss * 100
        gini_improvement = model_gini - base_gini
        auc_improvement = model_auc - base_auc

        return BasePredictionsComparison(
            model_metrics=model_metrics,
            base_metrics=base_metrics,
            model_vs_base_deciles=deciles,
            model_better_deciles=model_better_count,
            base_better_deciles=base_better_count,
            loss_improvement_pct=round(loss_improvement, 2),
            gini_improvement=round(gini_improvement, 4),
            auc_improvement=round(auc_improvement, 4),
        )

    def compute_train_test_comparison(
        self,
        train_data: pl.DataFrame,
        test_data: pl.DataFrame,
        y_train: np.ndarray,
        mu_train: np.ndarray,
        exposure_train: np.ndarray,
        y_test: np.ndarray,
        mu_test: np.ndarray,
        exposure_test: np.ndarray,
        categorical_factors: list[str],
        continuous_factors: list[str],
        result=None,
    ) -> TrainTestComparison:
        """
        Compute comprehensive train vs test comparison with flags.

        Returns
        -------
        TrainTestComparison
            Complete comparison with per-set diagnostics and flags
        """
        # Compute diagnostics for each dataset
        train_diag = self.compute_dataset_diagnostics(
            y_train,
            mu_train,
            exposure_train,
            train_data,
            categorical_factors,
            continuous_factors,
            "train",
            result,
        )
        test_diag = self.compute_dataset_diagnostics(
            y_test,
            mu_test,
            exposure_test,
            test_data,
            categorical_factors,
            continuous_factors,
            "test",
            result,
        )

        # Comparison metrics
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

        # Factor-level divergence
        factor_divergence = {}
        unstable_factors_list = []

        for factor in categorical_factors:
            if factor in train_diag.factor_diagnostics and factor in test_diag.factor_diagnostics:
                train_levels = {m.level: m for m in train_diag.factor_diagnostics[factor]}
                test_levels = {m.level: m for m in test_diag.factor_diagnostics[factor]}

                divergent = []
                for level in set(train_levels.keys()) | set(test_levels.keys()):
                    train_ae = train_levels.get(
                        level, FactorLevelMetrics(level, 0, 0, 0, 0, None, 0)
                    ).ae_ratio
                    test_ae = test_levels.get(
                        level, FactorLevelMetrics(level, 0, 0, 0, 0, None, 0)
                    ).ae_ratio

                    if train_ae is not None and test_ae is not None:
                        diff = abs(train_ae - test_ae)
                        if diff > 0.1:
                            divergent.append(
                                {
                                    "level": level,
                                    "train_ae": train_ae,
                                    "test_ae": test_ae,
                                    "ae_diff": round(diff, 4),
                                }
                            )
                            unstable_factors_list.append(f"{factor}[{level}]")

                if divergent:
                    factor_divergence[factor] = divergent

        # Flags for agent
        overfitting_risk = gini_gap > 0.03
        calibration_drift = test_diag.ae_ratio < 0.95 or test_diag.ae_ratio > 1.05

        return TrainTestComparison(
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
