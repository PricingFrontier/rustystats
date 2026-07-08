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
import pytest
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

    def test_compute_covariance_false_suppresses_direct_inference(self):
        df = _frame()
        result = _fit(df, compute_covariance=False)
        assert result.inference_status == "covariance_skipped"
        assert result.get_bread_matrix() is None
        with pytest.raises(rs.FittingError, match="Covariance was skipped"):
            result.bse()

        summary = result.summary()
        assert "covariance_skipped" in summary
        assert "Std.Err" not in summary

        diagnostics = result.diagnostics(
            df,
            continuous_factors=["x"],
            compute_vif=False,
            compute_deviance_by_level=False,
            compute_lift=False,
            compute_partial_dep=False,
            compute_score_tests=False,
        )
        assert diagnostics.coefficient_summary is None
        assert all(f.significance is None for f in diagnostics.factors)

        loaded = rs.GLMModel.from_bytes(result.to_bytes())
        assert loaded.inference_status == "covariance_skipped"
        assert loaded.get_bread_matrix() is None
        with pytest.raises(rs.FittingError, match="Covariance was skipped"):
            loaded.bse()

    def test_cv_selection_is_never_valid_standard(self):
        """011.3: a CV-selected fit (even pure ridge) is naive_after_cv_selection."""
        result = _fit(_frame(), cv=3, regularization="ridge", n_alphas=3)
        assert result.inference_status == "naive_after_cv_selection"
        assert "WARNING" in result.summary()

    def test_constrained_fit_is_constrained_boundary(self):
        """011.4: a sign/monotonicity-constrained fit flags constrained inference
        and the summary suppresses the ordinary significance machinery."""
        result = _fit(_frame(), terms={"x": {"type": "bs", "df": 5, "monotonicity": "increasing"}})
        assert result.inference_status == "constrained_boundary"
        assert isinstance(result.boundary_active_coefficients, list)
        summary = result.summary()
        assert "Std.Err" not in summary
        assert "P>|z|" not in summary

    def test_constrained_boundary_marks_clamped_coefficients(self):
        """011.4: coefficients clamped to their constraint boundary are reported.

        A *decreasing* constraint on a strongly increasing effect forces the
        basis coefficients to their boundary, so the marked set is non-empty and
        each entry is genuinely at/above the nonpositive boundary.
        """
        rng = np.random.default_rng(3)
        n = 600
        x = rng.uniform(0.0, 1.0, n)
        y = rng.poisson(np.exp(0.2 + 1.5 * x)).astype(float)
        data = pl.DataFrame({"y": y, "x": x})
        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "bs", "df": 6, "monotonicity": "decreasing"}},
            data=data,
            family="poisson",
        ).fit()
        assert result.inference_status == "constrained_boundary"
        boundary = result.boundary_active_coefficients
        assert len(boundary) >= 1
        for entry in boundary:
            assert set(entry) >= {"feature", "constraint", "coefficient"}
            assert entry["constraint"] == "nonpositive"
            assert entry["coefficient"] >= -1e-10

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
        assert result.optimizer_route == "gcv_smooth"
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
        assert state["schema_version"] == 4
        loaded = rs.GLMModel.from_bytes(result.to_bytes())
        assert loaded.inference_status == result.inference_status
        assert loaded.solver_status == result.solver_status
        assert loaded.optimizer_route == result.optimizer_route
        assert loaded.step_halving_used == result.step_halving_used

    def test_missing_inference_fields_load_with_none_defaults(self):
        """A current-version payload missing the optional inference/solver keys
        (e.g. a model never probed for inference) deserializes cleanly with None
        defaults and still predicts."""
        df = _frame()
        result = _fit(df)
        state = pickle.loads(result.to_bytes())
        for key in ("inference_status", "optimizer_route", "solver_status", "step_halving_used"):
            state["result_state"].pop(key, None)
        loaded = rs.GLMModel.from_bytes(pickle.dumps(state))
        assert loaded.inference_status is None
        assert loaded.solver_status is None
        assert loaded.optimizer_route is None
        assert loaded.step_halving_used is None
        preds = loaded.predict(df)
        assert len(preds) == len(df)

    def test_mismatched_schema_version_fails_loud(self):
        """No backwards compatibility: a payload whose schema_version differs from
        the current writer must be rejected with a clear error rather than loaded
        and silently mis-handled (e.g. the pre-v4 exposure layout)."""
        result = _fit(_frame())
        state = pickle.loads(result.to_bytes())
        state["schema_version"] = 2
        with pytest.raises(rs.ValidationError, match="schema_version"):
            rs.GLMModel.from_bytes(pickle.dumps(state))
        state.pop("schema_version", None)
        with pytest.raises(rs.ValidationError, match="schema_version"):
            rs.GLMModel.from_bytes(pickle.dumps(state))

    def test_weights_spec_round_trips(self):
        """A column-name weights spec must survive serialization so a reloaded
        model still produces weighted (not silently unweighted) diagnostics."""
        df = _frame().with_columns(pl.Series("w", np.linspace(0.5, 2.0, 600)))
        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df,
            family="poisson",
            weights="w",
        ).fit()
        assert result._weights_spec == "w"
        loaded = rs.GLMModel.from_bytes(result.to_bytes())
        assert loaded._weights_spec == "w"

    def test_tweedie_var_power_is_derived_and_round_trips(self):
        """var_power must be recovered from the family string (not defaulted to
        1.5) for both the in-memory and the deserialized model."""
        rng = np.random.default_rng(3)
        n = 400
        x = rng.normal(size=n)
        mu = np.exp(0.2 + 0.3 * x)
        y = rng.gamma(shape=2.0, scale=mu / 2.0)
        df = pl.DataFrame({"y": y, "x": x})
        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}},
            data=df,
            family="tweedie(p=1.7)",
        ).fit()
        assert result.var_power == pytest.approx(1.7, abs=1e-9)
        loaded = rs.GLMModel.from_bytes(result.to_bytes())
        assert loaded.var_power == pytest.approx(1.7, abs=1e-9)


class TestFactorInferenceSuppression:
    """Factor diagnostics must mirror the main coefficient table: Wald factor
    significance and per-coefficient SE/p are only shown for valid inference."""

    def test_factor_significance_and_se_suppressed_under_selection(self):
        df = _frame()
        terms = {"x": {"type": "linear"}, "x2": {"type": "linear"}}
        plain = rs.glm_dict(response="y", terms=terms, data=df, family="poisson").fit()
        lasso = rs.glm_dict(response="y", terms=terms, data=df, family="poisson").fit(
            alpha=0.1, l1_ratio=1.0
        )
        assert plain.inference_status == "valid_standard"
        assert lasso.inference_status == "naive_after_selection"

        d_plain = plain.diagnostics(df, continuous_factors=["x", "x2"])
        d_lasso = lasso.diagnostics(df, continuous_factors=["x", "x2"])

        # Plain GLM: factor significance present.
        assert all(f.significance is not None for f in d_plain.factors)
        # Selected (lasso) GLM: factor significance suppressed, matching the
        # main coefficient table which hides Std.Err / P>|z|.
        assert all(f.significance is None for f in d_lasso.factors)

        # Per-coefficient SE is real for the plain fit and NaN (suppressed) for
        # the selected fit.
        def first_se(diag):
            for f in diag.factors:
                coefs = getattr(f, "coefficients", None)
                if coefs:
                    return coefs[0].std_error
            return None

        assert np.isfinite(first_se(d_plain))
        assert np.isnan(first_se(d_lasso))
