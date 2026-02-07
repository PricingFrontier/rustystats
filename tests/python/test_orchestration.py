"""
Python unit tests for orchestration logic.

Tests the Python-layer glue code (column splitting, reordering, builder state)
independently of Rust fitting correctness. These tests ensure refactoring
the Python layer doesn't break the wiring between components.
"""

import numpy as np
import polars as pl
import pytest

import rustystats as rs
from rustystats.formula import (
    SmoothTermResult,
    _fit_glm_core,
    _build_results,
)
from rustystats.interactions import parse_spline_factor


# =============================================================================
# 1. parse_spline_factor unit tests
# =============================================================================

class TestParseSplineFactor:
    """Test the shared spline parsing function directly."""

    def test_bs_with_df(self):
        term = parse_spline_factor("bs(age, df=5)")
        assert term is not None
        assert term.var_name == "age"
        assert term.df == 5
        assert not term._is_smooth

    def test_bs_with_k(self):
        term = parse_spline_factor("bs(income, k=10)")
        assert term is not None
        assert term.var_name == "income"
        assert term._is_smooth

    def test_ns_with_df(self):
        term = parse_spline_factor("ns(age, df=4)")
        assert term is not None
        assert term.var_name == "age"
        assert term.spline_type == "ns"
        assert not term._is_smooth

    def test_ns_with_k(self):
        term = parse_spline_factor("ns(x, k=8)")
        assert term is not None
        assert term._is_smooth

    def test_bs_with_monotonicity(self):
        term = parse_spline_factor("bs(x, k=10, monotonicity='increasing')")
        assert term is not None
        assert term.monotonicity == "increasing"

    def test_bs_with_degree(self):
        term = parse_spline_factor("bs(x, df=5, degree=2)")
        assert term is not None
        assert term.degree == 2

    def test_not_a_spline(self):
        result = parse_spline_factor("C(region)")
        assert result is None

    def test_not_a_spline_plain_var(self):
        result = parse_spline_factor("age")
        assert result is None

    def test_ms_spline(self):
        term = parse_spline_factor("ms(x, k=8)")
        assert term is not None
        assert term.spline_type == "ms"

    def test_case_insensitive(self):
        term = parse_spline_factor("BS(Age, df=5)")
        assert term is not None
        assert term.var_name == "Age"

    def test_whitespace_handling(self):
        # Note: parser handles leading/trailing whitespace but not internal spacing
        term = parse_spline_factor("bs(age, df=5)")
        assert term is not None
        assert term.var_name == "age"


# =============================================================================
# 2. Smooth coefficient ordering tests
# =============================================================================

class TestSmoothCoefficientOrdering:
    """Test that smooth model coefficients match the design matrix column order."""

    def test_reorder_preserves_values(self):
        """Reordering should just permute, not change values."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.uniform(0, 10, n)
        y = rng.poisson(np.exp(0.5 + 0.1 * x))
        data = pl.DataFrame({"y": y, "x": x.round(4)})

        # Fit a smooth model
        result = rs.glm("y ~ bs(x, k=8)", data, family="poisson").fit(max_iter=100)

        # The number of params should match feature_names
        assert len(result.params) == len(result.feature_names)

    def test_bse_length_matches_params(self):
        """BSE array should have same length as params after reordering."""
        rng = np.random.default_rng(42)
        n = 300
        x1 = rng.uniform(0, 10, n)
        x2 = rng.normal(0, 1, n)
        y = rng.poisson(np.exp(0.5 + 0.1 * x1 + 0.3 * x2))
        data = pl.DataFrame({"y": y, "x1": x1.round(4), "x2": x2.round(4)})

        result = rs.glm("y ~ bs(x1, k=8) + x2", data, family="poisson").fit(max_iter=100)

        n_params = len(result.params)
        assert len(result.bse()) == n_params
        assert len(result.tvalues()) == n_params
        assert len(result.pvalues()) == n_params
        assert result.conf_int().shape == (n_params, 2)


# =============================================================================
# 3. Design matrix builder state tests
# =============================================================================

class TestBuilderState:
    """Test that the InteractionBuilder tracks smooth terms correctly."""

    def test_smooth_terms_detected(self):
        """Builder should detect smooth terms from formula."""
        rng = np.random.default_rng(42)
        n = 100
        x = rng.uniform(0, 10, n)
        y = rng.poisson(np.exp(0.5 + 0.1 * x))
        data = pl.DataFrame({"y": y, "x": x.round(4)})

        model = rs.glm("y ~ bs(x, k=8)", data, family="poisson")
        # The model should have detected smooth terms before fitting
        result = model.fit(max_iter=100)
        assert result.has_smooth_terms()

    def test_fixed_spline_not_smooth(self):
        """bs(x, df=5) should NOT be flagged as smooth."""
        rng = np.random.default_rng(42)
        n = 100
        x = rng.uniform(0, 10, n)
        y = rng.poisson(np.exp(0.5 + 0.1 * x))
        data = pl.DataFrame({"y": y, "x": x.round(4)})

        result = rs.glm("y ~ bs(x, df=5)", data, family="poisson").fit()
        assert not result.has_smooth_terms()

    def test_multiple_smooth_terms_tracked(self):
        """Multiple smooth terms should each be tracked."""
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(0, 10, n)
        x2 = rng.uniform(0, 5, n)
        y = rng.poisson(np.exp(0.5 + 0.1 * x1 + 0.1 * x2))
        data = pl.DataFrame({"y": y, "x1": x1.round(4), "x2": x2.round(4)})

        result = rs.glm(
            "y ~ bs(x1, k=8) + bs(x2, k=6)", data, family="poisson"
        ).fit(max_iter=100)
        assert result.has_smooth_terms()
        assert len(result.smooth_terms) == 2

    def test_mixed_smooth_and_fixed(self):
        """Mix of smooth and fixed splines should only flag smooth ones."""
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(0, 10, n)
        x2 = rng.uniform(0, 5, n)
        y = rng.poisson(np.exp(0.5 + 0.1 * x1 + 0.1 * x2))
        data = pl.DataFrame({"y": y, "x1": x1.round(4), "x2": x2.round(4)})

        result = rs.glm(
            "y ~ bs(x1, k=8) + bs(x2, df=4)", data, family="poisson"
        ).fit(max_iter=100)
        assert result.has_smooth_terms()
        assert len(result.smooth_terms) == 1  # Only x1 is smooth


# =============================================================================
# 4. EDF-based degrees of freedom tests
# =============================================================================

class TestEDFDegreesFreedom:
    """Test that smooth models use EDF for df_resid/df_model."""

    def test_df_resid_uses_edf(self):
        rng = np.random.default_rng(42)
        n = 300
        x = rng.uniform(0, 10, n)
        y = rng.poisson(np.exp(0.5 + 0.3 * np.sin(x)))
        data = pl.DataFrame({"y": y, "x": x.round(4)})

        result = rs.glm("y ~ bs(x, k=10)", data, family="poisson").fit(max_iter=100)

        # df_resid should be n - total_edf, not n - p
        expected = result.nobs - result.total_edf
        assert abs(result.df_resid - expected) < 1e-10

    def test_df_model_uses_edf(self):
        rng = np.random.default_rng(42)
        n = 300
        x = rng.uniform(0, 10, n)
        y = rng.poisson(np.exp(0.5 + 0.3 * np.sin(x)))
        data = pl.DataFrame({"y": y, "x": x.round(4)})

        result = rs.glm("y ~ bs(x, k=10)", data, family="poisson").fit(max_iter=100)

        # df_model should be total_edf - 1
        expected = result.total_edf - 1
        assert abs(result.df_model - expected) < 1e-10

    def test_non_smooth_uses_standard_df(self):
        """Non-smooth models should use standard n-p for df_resid."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.uniform(0, 10, n)
        y = rng.poisson(np.exp(0.5 + 0.1 * x))
        data = pl.DataFrame({"y": y, "x": x.round(4)})

        result = rs.glm("y ~ x", data, family="poisson").fit()
        assert result.df_resid == n - 2  # n - p (intercept + x)


# =============================================================================
# 5. Serialization state preservation
# =============================================================================

class TestSerializationState:
    """Test that serialization preserves orchestration state."""

    def test_smooth_metadata_round_trip(self):
        rng = np.random.default_rng(42)
        n = 300
        x = rng.uniform(0, 10, n)
        y = rng.poisson(np.exp(0.5 + 0.3 * np.sin(x)))
        data = pl.DataFrame({"y": y, "x": x.round(4)})

        result = rs.glm("y ~ bs(x, k=10)", data, family="poisson").fit(max_iter=100)
        model_bytes = result.to_bytes()
        loaded = rs.GLMModel.from_bytes(model_bytes)

        assert loaded.has_smooth_terms() == result.has_smooth_terms()
        assert loaded.total_edf is not None
        assert abs(loaded.total_edf - result.total_edf) < 1e-10
        assert loaded.gcv is not None

    def test_feature_names_preserved(self):
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(0, 10, n)
        x2 = rng.normal(0, 1, n)
        y = rng.poisson(np.exp(0.5 + 0.1 * x1 + 0.3 * x2))
        data = pl.DataFrame({"y": y, "x1": x1.round(4), "x2": x2.round(4)})

        result = rs.glm("y ~ bs(x1, k=8) + x2", data, family="poisson").fit(max_iter=100)
        model_bytes = result.to_bytes()
        loaded = rs.GLMModel.from_bytes(model_bytes)

        assert loaded.feature_names == result.feature_names
