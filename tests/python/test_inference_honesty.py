"""RS-ACT-011: honest inference status + solver-status surfacing.

Ordinary standard errors / p-values / AIC / BIC are only valid for an
unpenalized, unselected, unconstrained, non-smooth fit. Every other fit gets a
conservative ``inference_status`` and the summary suppresses the significance
machinery (stars + legend) and warns, rather than presenting it as trustworthy.
The result also surfaces the solver status and optimizer route from RS-ACT-007.
"""

from __future__ import annotations

import pickle

import numpy as np
import polars as pl
import rustystats as rs


def _frame(seed: int = 0, n: int = 600) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    y = rng.poisson(np.exp(0.3 + 0.4 * x + 0.2 * x2)).astype(float)
    return pl.DataFrame({"y": y, "x": x, "x2": x2})


def _fit(data, terms=None, **fit_kwargs):
    terms = terms or {"x": {"type": "linear"}, "x2": {"type": "linear"}}
    return rs.glm_dict(response="y", terms=terms, data=data, family="poisson").fit(**fit_kwargs)


class TestInferenceStatus:
    def test_plain_glm_is_valid_standard(self):
        """011.1: an unpenalized GLM keeps standard inference + the stars legend."""
        result = _fit(_frame())
        assert result.inference_status == "valid_standard"
        summary = result.summary()
        assert "Signif. codes" in summary
        assert "Inference:" in summary
        assert "WARNING" not in summary

    def test_lasso_is_naive_after_selection_and_hides_stars(self):
        """011.2: lasso does not present ordinary p-values as valid."""
        result = _fit(_frame(), alpha=0.1, l1_ratio=1.0)
        assert result.inference_status == "naive_after_selection"
        summary = result.summary()
        assert "WARNING" in summary
        assert "Signif. codes" not in summary
        assert "Std.Err" not in summary
        assert "P>|z|" not in summary
        assert "<0.0001" not in summary
        assert (
            next(line for line in summary.splitlines() if line.startswith("AIC:")).split()[1]
            == "NA"
        )
        assert (
            next(line for line in summary.splitlines() if line.startswith("BIC:")).split()[1]
            == "NA"
        )
        coef_table = result.coef_table()
        assert np.all(np.isnan(coef_table["Std.Error"].to_numpy()))
        assert set(coef_table["Signif"].to_list()) == {""}

    def test_lasso_relativities_and_summary_relativities_suppress_inference(self):
        """011.2: relativities() and summary_relativities() also suppress invalid inference."""
        result = _fit(_frame(), alpha=0.1, l1_ratio=1.0)
        assert result.inference_status == "naive_after_selection"

        # relativities(): keep the point relativity, null the confidence interval.
        rel = result.relativities()
        assert np.all(np.isfinite(rel["Relativity"].to_numpy()))
        assert np.all(np.isnan(rel["CI_Lower"].to_numpy()))
        assert np.all(np.isnan(rel["CI_Upper"].to_numpy()))

        # summary_relativities(): reads inference_status off the result and caveats.
        text = rs.summary_relativities(result)
        assert "Inference suppressed" in text
        assert "naive_after_selection" in text
        assert "<0.0001" not in text

    def test_plain_glm_relativities_keep_inference(self):
        """Guard against over-suppression: a plain GLM keeps CIs/p-values in both paths."""
        result = _fit(_frame())
        rel = result.relativities()
        assert np.all(np.isfinite(rel["CI_Lower"].to_numpy()))
        assert "Inference suppressed" not in rs.summary_relativities(result)

    def test_ridge_is_naive_after_regularization(self):
        result = _fit(_frame(), alpha=0.1, l1_ratio=0.0)
        assert result.inference_status == "naive_after_regularization"

    def test_cv_selection_is_never_valid_standard(self):
        """011.3: a CV-selected fit (even pure ridge) is naive_after_cv_selection."""
        result = _fit(_frame(), cv=3, regularization="ridge", n_alphas=3)
        assert result.inference_status == "naive_after_cv_selection"
        assert "WARNING" in result.summary()

    def test_constrained_fit_is_constrained_boundary(self):
        """011.4: a sign/monotonicity-constrained fit flags constrained inference."""
        result = _fit(_frame(), terms={"x": {"type": "bs", "df": 5, "monotonicity": "increasing"}})
        assert result.inference_status == "constrained_boundary"
        assert isinstance(result.boundary_active_coefficients, list)

    def test_diagnostics_suppresses_nonstandard_coefficient_summary(self):
        df = _frame()
        result = _fit(df, alpha=0.1, l1_ratio=1.0)
        diag = result.diagnostics(
            df,
            continuous_factors=["x"],
            compute_vif=False,
            compute_deviance_by_level=False,
            compute_lift=False,
            compute_partial_dep=False,
            compute_robust_se=True,
            compute_score_tests=False,
        )
        assert diag.coefficient_summary is None
        assert any(w["type"] == "coefficient_inference_unavailable" for w in diag.warnings)

    def test_smooth_uses_effective_df_and_is_unavailable(self):
        """011.5: a penalized smooth fit is non-standard and reports effective df."""
        result = _fit(_frame(), terms={"x": {"type": "bs", "k": 8}})
        assert result.inference_status == "unavailable"
        assert result.optimizer_route == "gcv_penalized"
        expected_aic = -2.0 * result.llf() + 2.0 * result.total_edf
        expected_bic = -2.0 * result.llf() + result.total_edf * np.log(result.nobs)
        assert np.isclose(result.aic(), expected_aic)
        assert np.isclose(result.bic(), expected_bic)
        summary = result.summary()
        assert "Effective df:" in summary
        assert "AIC (edf):" in summary
        assert "BIC (edf):" in summary
        assert "Std.Err" not in summary
        assert "P>|z|" not in summary


class TestSolverStatusSurfacing:
    def test_result_exposes_solver_status_and_route(self):
        """011.6: solver status, route, and step-halving flag are surfaced."""
        result = _fit(_frame())
        assert result.solver_status == "converged"
        assert result.optimizer_route == "irls"
        assert isinstance(result.step_halving_used, bool)


class TestInferenceSerialization:
    def test_inference_and_solver_status_round_trip(self):
        """011.7: serialization preserves inference + solver-status metadata."""
        result = _fit(_frame(), cv=3, regularization="ridge", n_alphas=3)
        state = pickle.loads(result.to_bytes())
        assert state["schema_version"] == 3
        loaded = rs.GLMModel.from_bytes(result.to_bytes())
        assert loaded.inference_status == result.inference_status
        assert loaded.solver_status == result.solver_status
        assert loaded.optimizer_route == result.optimizer_route
        assert loaded.step_halving_used == result.step_halving_used
