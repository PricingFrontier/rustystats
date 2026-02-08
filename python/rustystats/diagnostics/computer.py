"""
Main diagnostics computation orchestrator.

DiagnosticsComputer coordinates focused component classes to produce
unified diagnostics output for fitted GLM models.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any, TYPE_CHECKING

import numpy as np

from rustystats._rustystats import (
    compute_discrimination_stats_py as _rust_discrimination_stats,
    compute_loss_metrics_py as _rust_loss_metrics,
    compute_fit_statistics_py as _rust_fit_statistics,
    compute_dataset_metrics_py as _rust_dataset_metrics,
    compute_residual_summary_py as _rust_residual_summary,
    compute_deviance_residuals_py as _rust_deviance_residuals,
    chi2_cdf_py as _chi2_cdf,
)

from rustystats.diagnostics.types import (
    _extract_base_variable,
    ResidualSummary,
    FactorDiagnostics,
    InteractionCandidate,
    VIFResult,
    CoefficientSummary,
    DevianceByLevel,
    FactorDeviance,
    LiftDecile,
    LiftChart,
    PartialDependence,
    DecileMetrics,
    FactorLevelMetrics,
    ContinuousBandMetrics,
    DatasetDiagnostics,
    ModelVsBaseDecile,
    BasePredictionsMetrics,
    BasePredictionsComparison,
)

from rustystats.diagnostics.components import (
    _ResidualComputer,
    _CalibrationComputer,
    _DiscriminationComputer,
)

from rustystats.diagnostics.factors import _FactorDiagnosticsComputer
from rustystats.diagnostics.interactions import _InteractionDetector
from rustystats.exceptions import DesignMatrixError, ValidationError
from rustystats.constants import (
    EPSILON,
    DEFAULT_N_CALIBRATION_BINS,
    DEFAULT_N_FACTOR_BINS,
    DEFAULT_RARE_THRESHOLD_PCT,
    DEFAULT_MAX_CATEGORICAL_LEVELS,
    DEFAULT_MAX_INTERACTION_FACTORS,
)

if TYPE_CHECKING:
    import polars as pl


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
        exposure: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        var_power: float = 1.5,
        theta: float = 1.0,
        null_deviance: Optional[float] = None,
    ):
        self.y = np.asarray(y, dtype=np.float64)
        self.mu = np.asarray(mu, dtype=np.float64)
        self.linear_predictor = np.asarray(linear_predictor, dtype=np.float64)
        self.family = family.lower()
        self.n_params = n_params
        self.deviance = deviance
        self._null_deviance_override = null_deviance  # From model result
        self.exposure = np.asarray(exposure, dtype=np.float64) if exposure is not None else np.ones_like(y)
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
            self.y, self.mu, self.exposure,
            self.pearson_residuals, self.feature_names, self.family,
        )
        self._interactions = _InteractionDetector(
            self.pearson_residuals, self.feature_names,
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
    
    def _compute_loss(self, y: np.ndarray, mu: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        unit_dev = self._compute_unit_deviance(y, mu)
        if weights is not None:
            return np.average(unit_dev, weights=weights)
        return np.mean(unit_dev)
    
    def compute_fit_statistics(self) -> Dict[str, float]:
        """Compute overall fit statistics using Rust backend."""
        return _rust_fit_statistics(
            self.y, self.mu, self.deviance, self.null_deviance, self.n_params, self.family
        )
    
    def compute_loss_metrics(self) -> Dict[str, float]:
        """Compute various loss metrics using Rust backend."""
        rust_loss = _rust_loss_metrics(self.y, self.mu, self.family)
        return {
            "loss": rust_loss["family_loss"],  # Primary metric for model comparison
            "mse": rust_loss["mse"],
            "mae": rust_loss["mae"],
            "rmse": rust_loss["rmse"],
        }
    
    def compute_calibration(self, n_bins: int = DEFAULT_N_CALIBRATION_BINS) -> Dict[str, Any]:
        """Compute calibration metrics using focused component."""
        return self._calibration.compute(n_bins)
    
    def compute_discrimination(self) -> Optional[Dict[str, Any]]:
        """Compute discrimination metrics using focused component."""
        return self._discrimination.compute()
    
    def compute_residual_summary(self) -> Dict[str, ResidualSummary]:
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
        data: "pl.DataFrame",
        categorical_factors: List[str],
        continuous_factors: List[str],
        result=None,
        n_bins: int = DEFAULT_N_FACTOR_BINS,
        rare_threshold_pct: float = DEFAULT_RARE_THRESHOLD_PCT,
        max_categorical_levels: int = DEFAULT_MAX_CATEGORICAL_LEVELS,
        design_matrix: Optional[np.ndarray] = None,
        bread_matrix: Optional[np.ndarray] = None,
        irls_weights: Optional[np.ndarray] = None,
    ) -> List[FactorDiagnostics]:
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
        )
    
    def detect_interactions(
        self,
        data: "pl.DataFrame",
        factor_names: List[str],
        max_factors: int = DEFAULT_MAX_INTERACTION_FACTORS,
        min_correlation: float = 0.01,
        max_candidates: int = 5,
        min_cell_count: int = 30,
    ) -> List[InteractionCandidate]:
        """Detect potential interactions. Delegates to _InteractionDetector."""
        return self._interactions.detect_interactions(
            data=data,
            factor_names=factor_names,
            max_factors=max_factors,
            min_correlation=min_correlation,
            max_candidates=max_candidates,
            min_cell_count=min_cell_count,
        )
    
    def compute_model_comparison(self) -> Dict[str, float]:
        """Compute model comparison statistics vs null model."""
        null_dev = self.null_deviance
        
        # Likelihood ratio test
        lr_chi2 = null_dev - self.deviance
        lr_df = self.n_params - 1
        
        # P-value from chi-square distribution (using Rust CDF)
        lr_pvalue = 1 - _chi2_cdf(lr_chi2, float(lr_df)) if lr_df > 0 else float('nan')
        
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
        fit_stats: Dict[str, float],
        calibration: Dict[str, Any],
        factors: List[FactorDiagnostics],
        family: str = "",
    ) -> List[Dict[str, str]]:
        """Generate warnings based on diagnostics."""
        warnings = []
        
        # NegBin-specific warnings
        family_lower = family.lower() if family else ""
        if family_lower.startswith("negativebinomial"):
            # Regularization warning
            warnings.append({
                "type": "negbinomial_regularization",
                "message": "Negative binomial fitting applies minimum ridge regularization (alpha=1e-6) for numerical stability. Coefficient bias is negligible but inference is approximate."
            })
            
            # Large theta warning (essentially Poisson)
            if "theta=" in family:
                try:
                    theta_str = family.split("theta=")[1].rstrip(")")
                    theta = float(theta_str)
                    if theta >= 100:
                        warnings.append({
                            "type": "negbinomial_large_theta",
                            "message": f"Estimated theta={theta:.1f} is very large, suggesting minimal overdispersion. Consider using Poisson instead for simpler interpretation."
                        })
                    elif theta <= 0.1:
                        warnings.append({
                            "type": "negbinomial_small_theta",
                            "message": f"Estimated theta={theta:.4f} is very small, indicating severe overdispersion. Check for missing covariates or consider zero-inflated models."
                        })
                except (ValueError, IndexError) as e:
                    # Theta parsing failed - this is a bug in family string formatting
                    raise ValidationError(f"Failed to parse theta from family string '{family}': {e}") from e
        
        # High dispersion warning
        dispersion = fit_stats.get("dispersion", 1.0)
        if dispersion > 1.5:
            warnings.append({
                "type": "high_dispersion",
                "message": f"Dispersion {dispersion:.2f} suggests overdispersion. Consider quasipoisson or negbinomial."
            })
        
        # Poor overall calibration
        ae_ratio = calibration.get("ae_ratio", 1.0)
        if abs(ae_ratio - 1.0) > 0.05:
            direction = "over" if ae_ratio < 1 else "under"
            warnings.append({
                "type": "poor_calibration",
                "message": f"Model {direction}-predicts overall (A/E = {ae_ratio:.3f})."
            })
        
        # Token optimization: skip per-decile warnings (problem_deciles in calibration has this info)
        
        # Factors with high residual correlation (not in model)
        for factor in factors:
            if not factor.in_model:
                r2 = factor.residual_pattern.var_explained
                if r2 > 0.02:
                    warnings.append({
                        "type": "missing_factor",
                        "message": f"Factor '{factor.name}' not in model but explains {100*r2:.1f}% of residual variance."
                    })
        
        return warnings
    
    # =========================================================================
    # NEW: Enhanced diagnostics for agentic workflows
    # =========================================================================
    
    def compute_vif(
        self,
        X: np.ndarray,
        feature_names: List[str],
        threshold_moderate: float = 5.0,
        threshold_severe: float = 10.0,
    ) -> List[VIFResult]:
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
        n_obs, n_features = X.shape
        results = []
        
        # Skip intercept column if present
        has_intercept = feature_names and feature_names[0] == "Intercept"
        start_idx = 1 if has_intercept else 0
        
        if n_features - start_idx <= 1:
            # Only one feature (besides intercept), VIF = 1
            for i in range(start_idx, n_features):
                results.append(VIFResult(
                    feature=feature_names[i] if i < len(feature_names) else f"X{i}",
                    vif=1.0, severity="none", collinear_with=None
                ))
            return results
        
        # Extract non-intercept columns
        X_no_int = X[:, start_idx:]
        names_no_int = feature_names[start_idx:] if feature_names else [f"X{i}" for i in range(start_idx, n_features)]
        k = X_no_int.shape[1]
        
        # Fast VIF via correlation matrix inverse
        # VIF_j = diag((R^{-1}))_j where R is correlation matrix
        try:
            # Center and scale columns (standardize to get correlation matrix)
            means = np.mean(X_no_int, axis=0)
            stds = np.std(X_no_int, axis=0, ddof=0)
            stds[stds == 0] = 1.0  # Avoid division by zero
            X_std = (X_no_int - means) / stds
            
            # Correlation matrix R = X'X / n
            R = (X_std.T @ X_std) / n_obs
            
            # Add small regularization for numerical stability
            R += np.eye(k) * EPSILON
            
            # VIF = diagonal of R^{-1}
            R_inv = np.linalg.inv(R)
            vif_values = np.diag(R_inv)
            
            # Also compute correlation matrix for finding collinear pairs
            corr_matrix = R - np.eye(k) * EPSILON  # Remove regularization for reporting
            
        except np.linalg.LinAlgError as e:
            raise DesignMatrixError(
                f"VIF computation failed: design matrix is singular. "
                f"This indicates severe multicollinearity - some columns are exact linear "
                f"combinations of others. Check for duplicate or constant columns."
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
            
            results.append(VIFResult(
                feature=feature_name,
                vif=round(float(vif), 2),
                severity=severity,
                collinear_with=collinear_with if collinear_with else None,
            ))
        
        # Sort by VIF (highest first)
        results.sort(key=lambda x: -x.vif if not np.isnan(x.vif) else 0)
        return results
    
    def compute_coefficient_summary(
        self,
        result,  # GLMResults or GLMModel
        link: str = "log",
    ) -> List[CoefficientSummary]:
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
        
        feature_names = self.feature_names if self.feature_names else [f"X{i}" for i in range(len(params))]
        
        summaries = []
        for i, name in enumerate(feature_names):
            coef_val = float(params[i])
            se_val = float(bse[i])
            z_val = float(tvalues[i])
            p_val = float(pvalues[i])
            
            # Relativity for log-link models
            rel = None
            rel_ci = None
            if link == "log":
                rel = round(float(np.exp(coef_val)), 4)
                rel_ci = [round(float(np.exp(ci[i, 0])), 4), round(float(np.exp(ci[i, 1])), 4)]
            
            summaries.append(CoefficientSummary(
                feature=name,
                estimate=round(coef_val, 6),
                std_error=round(se_val, 6),
                z_value=round(z_val, 3),
                p_value=round(p_val, 4),
                significant=p_val < 0.05,
                relativity=rel,
                relativity_ci=rel_ci,
            ))
        
        # Sort by absolute z-value (most significant first), but keep Intercept at end
        intercept = [s for s in summaries if s.feature == "Intercept"]
        others = [s for s in summaries if s.feature != "Intercept"]
        others.sort(key=lambda x: -abs(x.z_value))
        return others + intercept
    
    def compute_factor_deviance(
        self,
        data: "pl.DataFrame",
        categorical_factors: List[str],
    ) -> List[FactorDeviance]:
        """
        Compute deviance breakdown by factor level.
        
        Uses Rust backend for fast groupby aggregation on large datasets.
        
        Identifies which categorical levels are driving poor fit,
        helping the agent pinpoint problem areas.
        
        Returns
        -------
        list of FactorDeviance
            Deviance breakdown for each categorical factor
        """
        from rustystats._rustystats import compute_factor_deviance_py as _rust_factor_deviance
        
        results = []
        for factor_name in categorical_factors:
            if factor_name not in data.columns:
                continue
            
            values = [str(v) for v in data[factor_name].to_list()]
            
            # Call Rust for fast computation
            rust_result = _rust_factor_deviance(
                factor_name,
                values,
                self.y,
                self.mu,
                self.family,
                getattr(self, 'var_power', 1.5),
                getattr(self, 'theta', 1.0),
            )
            
            # Convert Rust result to Python dataclasses
            levels = [
                DevianceByLevel(
                    level=level["level"],
                    n=level["count"],
                    deviance=round(level["deviance"], 2),
                    deviance_pct=round(level["deviance_pct"], 2),
                    mean_deviance=round(level["mean_deviance"], 4),
                    ae_ratio=round(level["ae_ratio"], 3) if not np.isnan(level["ae_ratio"]) else None,
                    problem=level["is_problem"],
                )
                for level in rust_result["levels"]
            ]
            
            results.append(FactorDeviance(
                factor=factor_name,
                total_deviance=round(rust_result["total_deviance"], 2),
                levels=levels,
                problem_levels=rust_result["problem_levels"],
            ))
        
        return results
    
    def compute_lift_chart(self, n_deciles: int = 10) -> LiftChart:
        """
        Compute full lift chart with all deciles.
        
        Shows where the model discriminates well vs poorly,
        helping the agent identify risk bands needing attention.
        
        Returns
        -------
        LiftChart
            Complete lift chart with discrimination metrics
        """
        # Sort by predicted values
        sort_idx = np.argsort(self.mu)
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
            
            ae_ratio = actual / predicted if predicted > 0 else float('nan')
            
            cumulative_actual += actual
            cumulative_predicted += predicted
            
            cum_actual_pct = 100 * cumulative_actual / total_actual if total_actual > 0 else 0
            cum_pred_pct = 100 * cumulative_predicted / total_predicted if total_predicted > 0 else 0
            
            # Lift: rate in this decile / overall rate
            decile_rate = actual / exposure if exposure > 0 else 0
            lift = decile_rate / overall_rate if overall_rate > 0 else 1.0
            
            # Cumulative lift
            cum_rate = cumulative_actual / np.sum(exp_sorted[:end]) if np.sum(exp_sorted[:end]) > 0 else 0
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
            deciles.append(LiftDecile(
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
            ))
        
        # Compute Gini
        gini = 2 * max_ks / 100  # Approximate from KS
        stats = _rust_discrimination_stats(self.y, self.mu, self.exposure)
        gini = float(stats["gini"])
        
        return LiftChart(
            deciles=deciles,
            gini=round(gini, 3),
            ks_statistic=round(max_ks, 2),
            ks_decile=ks_decile,
            weak_deciles=weak_deciles,
        )
    
    def compute_partial_dependence(
        self,
        data: "pl.DataFrame",
        result,  # GLMResults with predict capability
        continuous_factors: List[str],
        categorical_factors: List[str],
        link: str = "log",
        n_grid: int = 20,
    ) -> List[PartialDependence]:
        """
        Compute partial dependence for each variable.
        
        Shows the marginal effect shape, helping the agent decide
        between linear, spline, or banding approaches.
        
        Returns
        -------
        list of PartialDependence
            Partial dependence for each variable
        """
        results = []
        
        # Continuous variables
        for var in continuous_factors:
            if var not in data.columns:
                continue
            
            values = data[var].to_numpy().astype(np.float64)
            valid_mask = ~np.isnan(values) & ~np.isinf(values)
            valid_values = values[valid_mask]
            
            if len(valid_values) < 10:
                continue
            
            # Create grid
            grid = np.linspace(np.percentile(valid_values, 1), 
                              np.percentile(valid_values, 99), n_grid)
            
            predictions = []
            for g in grid:
                # Mean prediction if we set this variable to g
                # Use the coefficient to approximate partial effect
                var_idx = None
                for i, name in enumerate(self.feature_names):
                    if var == name or var in name:
                        var_idx = i
                        break
                
                if var_idx is not None:
                    # Linear approximation using coefficient
                    coef = result.params[var_idx]
                    base_pred = np.mean(self.mu)
                    if link == "log":
                        pred = base_pred * np.exp(coef * (g - np.mean(valid_values)))
                    else:
                        pred = base_pred + coef * (g - np.mean(valid_values))
                    predictions.append(float(pred))
                else:
                    predictions.append(float(np.mean(self.mu)))
            
            # Analyze shape
            shape, recommendation = self._analyze_pd_shape(grid, predictions, link)
            
            # Convert to relativities for log-link
            relativities = None
            if link == "log" and predictions:
                base = predictions[len(predictions)//2]
                relativities = [p/base if base > 0 else 1.0 for p in predictions]
            
            results.append(PartialDependence(
                variable=var,
                variable_type="continuous",
                grid_values=[round(float(g), 4) for g in grid],
                predictions=[round(p, 6) for p in predictions],
                relativities=[round(r, 4) for r in relativities] if relativities else None,
                std_errors=None,  # Would need bootstrap for this
                shape=shape,
                recommendation=recommendation,
            ))
        
        # Categorical variables
        for var in categorical_factors:
            if var not in data.columns:
                continue
            
            values = data[var].to_numpy().astype(str)
            unique_levels = np.unique(values)
            
            grid_values = list(unique_levels)
            predictions = []
            
            for level in unique_levels:
                mask = values == level
                if np.any(mask):
                    predictions.append(float(np.mean(self.mu[mask])))
                else:
                    predictions.append(float(np.mean(self.mu)))
            
            # Analyze categorical effect
            if len(predictions) > 1:
                max_pred = max(predictions)
                min_pred = min(predictions)
                range_ratio = max_pred / min_pred if min_pred > 0 else float('inf')
                
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
            if link == "log" and predictions:
                base = predictions[0]  # First level as base
                relativities = [p/base if base > 0 else 1.0 for p in predictions]
            
            results.append(PartialDependence(
                variable=var,
                variable_type="categorical",
                grid_values=grid_values,
                predictions=[round(p, 6) for p in predictions],
                relativities=[round(r, 4) for r in relativities] if relativities else None,
                std_errors=None,
                shape=shape,
                recommendation=recommendation,
            ))
        
        return results
    
    def _analyze_pd_shape(
        self, 
        grid: np.ndarray, 
        predictions: List[float],
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
        
        if relative_range < 0.05:
            return "flat", "May not need in model - negligible effect"
        
        if increasing >= n_diffs * 0.8:
            if curvature < pred_range * 0.1:
                return "linear_increasing", "Linear effect adequate"
            else:
                return "monotonic_increasing", "Consider spline for non-linearity"
        
        if decreasing >= n_diffs * 0.8:
            if curvature < pred_range * 0.1:
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
        if max_jump > pred_range * 0.4:
            return "step_function", "Consider banding/categorical transformation"
        
        return "complex", "Use spline (df=5+) to capture non-linearity"
    
    def compute_dataset_diagnostics(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        exposure: np.ndarray,
        data: "pl.DataFrame",
        categorical_factors: List[str],
        continuous_factors: List[str],
        dataset_name: str,
        result=None,
        n_bands: int = 10,
    ) -> DatasetDiagnostics:
        """
        Compute comprehensive diagnostics for a single dataset.
        
        Parameters
        ----------
        y : np.ndarray
            Actual response values
        mu : np.ndarray
            Predicted values
        exposure : np.ndarray
            Exposure weights
        data : pl.DataFrame
            DataFrame with factor columns
        categorical_factors : list of str
            Names of categorical factors
        continuous_factors : list of str
            Names of continuous factors
        dataset_name : str
            "train" or "test"
        result : GLMResults, optional
            Model results for partial dependence
        n_bands : int
            Number of bands for continuous variables
            
        Returns
        -------
        DatasetDiagnostics
        """
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
        
        # Discrimination metrics
        stats = _rust_discrimination_stats(y, mu, exposure)
        gini = float(stats["gini"])
        auc = float(stats["auc"])
        
        # Overall A/E
        ae_ratio = total_actual / total_predicted if total_predicted > 0 else float('nan')
        
        # A/E by decile (sorted by predicted value)
        ae_by_decile = self._compute_ae_by_decile(y, mu, exposure, n_deciles=10)
        
        # Factor-level diagnostics
        factor_diag = {}
        for factor in categorical_factors:
            if factor in data.columns:
                factor_diag[factor] = self._compute_factor_level_metrics(
                    y, mu, exposure, data[factor].to_numpy().astype(str)
                )
        
        # Continuous variable diagnostics
        continuous_diag = {}
        for var in continuous_factors:
            if var in data.columns:
                values = data[var].to_numpy().astype(np.float64)
                continuous_diag[var] = self._compute_continuous_band_metrics(
                    y, mu, exposure, values, result, var, n_bands
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
    ) -> List[DecileMetrics]:
        """Compute A/E by decile sorted by predicted value."""
        # Sort by predicted values
        sort_idx = np.argsort(mu)
        y_sorted = y[sort_idx]
        mu_sorted = mu[sort_idx]
        exp_sorted = exposure[sort_idx]
        
        n = len(y)
        decile_size = n // n_deciles
        
        deciles = []
        for d in range(n_deciles):
            start = d * decile_size
            end = (d + 1) * decile_size if d < n_deciles - 1 else n
            
            y_d = y_sorted[start:end]
            mu_d = mu_sorted[start:end]
            exp_d = exp_sorted[start:end]
            
            actual = float(np.sum(y_d))
            predicted = float(np.sum(mu_d))
            exp_sum = float(np.sum(exp_d))
            ae = actual / predicted if predicted > 0 else float('nan')
            
            actual_freq = actual / exp_sum if exp_sum > 0 else 0.0
            predicted_freq = predicted / exp_sum if exp_sum > 0 else 0.0
            deciles.append(DecileMetrics(
                decile=d + 1,
                n=len(y_d),
                exposure=round(exp_sum, 2),
                actual=round(actual_freq, 6),
                predicted=round(predicted_freq, 6),
                ae_ratio=round(ae, 4) if not np.isnan(ae) else None,
            ))
        
        return deciles
    
    def _compute_factor_level_metrics(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        exposure: np.ndarray,
        factor_values: np.ndarray,
    ) -> List[FactorLevelMetrics]:
        """Compute metrics for each level of a categorical factor."""
        unique_levels = np.unique(factor_values)
        # Use deviance residuals for consistency with continuous band metrics
        residuals = np.asarray(_rust_deviance_residuals(y, mu, self.family))
        
        metrics = []
        for level in unique_levels:
            mask = factor_values == level
            n = int(np.sum(mask))
            
            if n == 0:
                continue
            
            actual = float(np.sum(y[mask]))
            predicted = float(np.sum(mu[mask]))
            exp_sum = float(np.sum(exposure[mask]))
            ae = actual / predicted if predicted > 0 else float('nan')
            resid_mean = float(np.mean(residuals[mask]))
            
            actual_freq = actual / exp_sum if exp_sum > 0 else 0.0
            predicted_freq = predicted / exp_sum if exp_sum > 0 else 0.0
            metrics.append(FactorLevelMetrics(
                level=str(level),
                n=n,
                exposure=round(exp_sum, 2),
                actual=round(actual_freq, 6),
                predicted=round(predicted_freq, 6),
                ae_ratio=round(ae, 4) if not np.isnan(ae) else None,
                residual_mean=round(resid_mean, 6),
            ))
        
        # Sort by exposure (largest first)
        metrics.sort(key=lambda x: -x.exposure)
        return metrics
    
    def _compute_continuous_band_metrics(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        exposure: np.ndarray,
        values: np.ndarray,
        result,
        var_name: str,
        n_bands: int = 10,
    ) -> List[ContinuousBandMetrics]:
        """Compute metrics for bands of a continuous variable."""
        # Remove NaN/Inf
        valid_mask = ~np.isnan(values) & ~np.isinf(values)
        
        if np.sum(valid_mask) < n_bands:
            return []
        
        # Use quantile bands
        percentiles = np.linspace(0, 100, n_bands + 1)
        edges = np.percentile(values[valid_mask], percentiles)
        edges = np.unique(edges)  # Remove duplicates
        
        if len(edges) < 2:
            return []
        
        metrics = []
        # Compute deviance residuals for consistency with categorical diagnostics
        deviance_resids = np.asarray(_rust_deviance_residuals(y, mu, self.family))
        
        for i in range(len(edges) - 1):
            lower, upper = edges[i], edges[i + 1]
            
            if i == len(edges) - 2:
                mask = valid_mask & (values >= lower) & (values <= upper)
            else:
                mask = valid_mask & (values >= lower) & (values < upper)
            
            n = int(np.sum(mask))
            if n == 0:
                continue
            
            actual = float(np.sum(y[mask]))
            predicted = float(np.sum(mu[mask]))
            exp_sum = float(np.sum(exposure[mask]))
            ae = actual / predicted if predicted > 0 else float('nan')
            midpoint = (lower + upper) / 2
            
            # Partial dependence at midpoint
            partial_dep = float(np.mean(mu[mask]))
            
            # Mean deviance residual for this band
            resid_mean = float(np.mean(deviance_resids[mask]))
            
            actual_freq = actual / exp_sum if exp_sum > 0 else 0.0
            predicted_freq = predicted / exp_sum if exp_sum > 0 else 0.0
            metrics.append(ContinuousBandMetrics(
                band=i + 1,
                range_min=round(float(lower), 4),
                range_max=round(float(upper), 4),
                midpoint=round(float(midpoint), 4),
                n=n,
                exposure=round(exp_sum, 2),
                actual=round(actual_freq, 6),
                predicted=round(predicted_freq, 6),
                ae_ratio=round(ae, 4) if not np.isnan(ae) else None,
                partial_dep=round(partial_dep, 6),
                residual_mean=round(resid_mean, 6),
            ))
        
        return metrics
    
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
        ae_ratio_base = total_actual / total_predicted_base if total_predicted_base > 0 else float('nan')
        
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
        ae_ratio_model = total_actual / total_predicted_model if total_predicted_model > 0 else float('nan')
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
            
            model_ae = actual_sum / model_sum if model_sum > 0 else float('nan')
            base_ae = actual_sum / base_sum if base_sum > 0 else float('nan')
            
            # Frequencies (per exposure)
            actual_freq = actual_sum / exp_sum if exp_sum > 0 else 0.0
            model_freq = model_sum / exp_sum if exp_sum > 0 else 0.0
            base_freq = base_sum / exp_sum if exp_sum > 0 else 0.0
            
            # Mean ratio in this decile
            ratio_mean = float(np.mean(ratio_d))
            
            deciles.append(ModelVsBaseDecile(
                decile=d + 1,
                n=len(y_d),
                exposure=round(exp_sum, 2),
                actual=round(actual_freq, 6),
                model_predicted=round(model_freq, 6),
                base_predicted=round(base_freq, 6),
                model_ae_ratio=round(model_ae, 4) if not np.isnan(model_ae) else None,
                base_ae_ratio=round(base_ae, 4) if not np.isnan(base_ae) else None,
                model_base_ratio_mean=round(ratio_mean, 4),
            ))
            
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
        train_data: "pl.DataFrame",
        test_data: "pl.DataFrame",
        y_train: np.ndarray,
        mu_train: np.ndarray,
        exposure_train: np.ndarray,
        y_test: np.ndarray,
        mu_test: np.ndarray,
        exposure_test: np.ndarray,
        categorical_factors: List[str],
        continuous_factors: List[str],
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
            y_train, mu_train, exposure_train, train_data,
            categorical_factors, continuous_factors, "train", result
        )
        test_diag = self.compute_dataset_diagnostics(
            y_test, mu_test, exposure_test, test_data,
            categorical_factors, continuous_factors, "test", result
        )
        
        # Comparison metrics
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
        
        # Factor-level divergence
        factor_divergence = {}
        unstable_factors_list = []
        
        for factor in categorical_factors:
            if factor in train_diag.factor_diagnostics and factor in test_diag.factor_diagnostics:
                train_levels = {m.level: m for m in train_diag.factor_diagnostics[factor]}
                test_levels = {m.level: m for m in test_diag.factor_diagnostics[factor]}
                
                divergent = []
                for level in set(train_levels.keys()) | set(test_levels.keys()):
                    train_ae = train_levels.get(level, FactorLevelMetrics(level, 0, 0, 0, 0, None, 0)).ae_ratio
                    test_ae = test_levels.get(level, FactorLevelMetrics(level, 0, 0, 0, 0, None, 0)).ae_ratio
                    
                    if train_ae is not None and test_ae is not None:
                        diff = abs(train_ae - test_ae)
                        if diff > 0.1:
                            divergent.append({
                                "level": level,
                                "train_ae": train_ae,
                                "test_ae": test_ae,
                                "ae_diff": round(diff, 4),
                            })
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
