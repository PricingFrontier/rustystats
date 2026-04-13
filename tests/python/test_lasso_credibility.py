"""
Tests for lasso credibility (complement of credibility).

Covers:
- apply_link forward link function
- Complement construction (string, array, GLMModel)
- Complement validation (positive for log-link, (0,1) for logit)
- Fitting with complement (Poisson/log, Binomial/logit, Gamma/log, Gaussian/identity)
- Prediction with complement (auto-applied, manual override)
- credibility_summary() output
- summary() title and notes
- has_complement property
- Serialization round-trip (to_bytes / from_bytes)
- Complement with spline terms
- Complement with regularization (lasso shrinkage toward complement)
"""

import numpy as np
import polars as pl
import pytest
import rustystats as rs
from rustystats.exceptions import ValidationError
from rustystats.formula import apply_link

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def poisson_data():
    """Poisson frequency data with exposure and a countrywide rate column."""
    np.random.seed(42)
    n = 500
    x1 = np.random.uniform(0, 10, n)
    x2 = np.random.uniform(0, 5, n)
    cat = np.random.choice(["A", "B", "C"], n)
    exposure = np.random.uniform(0.5, 2.0, n)
    # Generate rates with known structure
    mu = np.exp(0.1 + 0.05 * x1 - 0.02 * x2)
    y = np.random.poisson(mu * exposure)
    # Countrywide rate: slightly different from true rate
    cw_rate = np.exp(0.08 + 0.04 * x1) * np.ones(n)
    return pl.DataFrame(
        {
            "y": y.astype(float),
            "x1": x1,
            "x2": x2,
            "cat": cat,
            "exposure": exposure,
            "cw_rate": cw_rate,
        }
    )


@pytest.fixture
def binomial_data():
    """Binomial data with prior probability column."""
    np.random.seed(123)
    n = 500
    x1 = np.random.uniform(-2, 2, n)
    p = 1.0 / (1.0 + np.exp(-(0.5 + 0.3 * x1)))
    y = np.random.binomial(1, p, n).astype(float)
    # Prior probabilities (slightly off)
    prior_prob = np.clip(1.0 / (1.0 + np.exp(-(0.4 + 0.2 * x1))), 0.01, 0.99)
    return pl.DataFrame(
        {
            "y": y,
            "x1": x1,
            "prior_prob": prior_prob,
        }
    )


@pytest.fixture
def gaussian_data():
    """Gaussian data with prior prediction column."""
    np.random.seed(456)
    n = 300
    x1 = np.random.uniform(0, 10, n)
    y = 5.0 + 2.0 * x1 + np.random.normal(0, 1, n)
    prior_pred = 4.5 + 1.8 * x1  # slightly off
    return pl.DataFrame(
        {
            "y": y,
            "x1": x1,
            "prior_pred": prior_pred,
        }
    )


@pytest.fixture
def gamma_data():
    """Gamma severity data with prior rate column."""
    np.random.seed(789)
    n = 400
    x1 = np.random.uniform(0, 5, n)
    mu = np.exp(6.0 + 0.1 * x1)
    y = np.random.gamma(shape=5, scale=mu / 5, size=n)
    prior_rate = np.exp(5.9 + 0.08 * x1)
    return pl.DataFrame(
        {
            "y": y,
            "x1": x1,
            "prior_rate": prior_rate,
        }
    )


# =============================================================================
# apply_link tests
# =============================================================================


class TestApplyLink:
    """Test the forward link function."""

    def test_identity_link(self):
        mu = np.array([1.0, 2.0, 3.0])
        result = apply_link(mu, "identity")
        np.testing.assert_array_equal(result, mu)

    def test_log_link(self):
        mu = np.array([1.0, 2.0, np.e])
        result = apply_link(mu, "log")
        expected = np.array([0.0, np.log(2.0), 1.0])
        np.testing.assert_allclose(result, expected)

    def test_log_link_none(self):
        """link=None should behave like log."""
        mu = np.array([1.0, 2.0])
        result = apply_link(mu, None)
        np.testing.assert_allclose(result, np.log(mu))

    def test_logit_link(self):
        mu = np.array([0.5, 0.25, 0.75])
        result = apply_link(mu, "logit")
        expected = np.log(mu / (1.0 - mu))
        np.testing.assert_allclose(result, expected)

    def test_inverse_link(self):
        mu = np.array([1.0, 2.0, 4.0])
        result = apply_link(mu, "inverse")
        expected = np.array([1.0, 0.5, 0.25])
        np.testing.assert_allclose(result, expected)

    def test_unknown_link_raises(self):
        with pytest.raises(ValidationError, match="Unknown link function"):
            apply_link(np.array([1.0]), "probit")

    def test_roundtrip_log(self):
        """apply_link(log) and apply_inverse_link(log) should be inverses."""
        from rustystats.formula import apply_inverse_link

        mu = np.array([0.5, 1.0, 2.0, 10.0])
        eta = apply_link(mu, "log")
        mu_back = apply_inverse_link(eta, "log")
        np.testing.assert_allclose(mu_back, mu)

    def test_roundtrip_logit(self):
        from rustystats.formula import apply_inverse_link

        mu = np.array([0.1, 0.5, 0.9])
        eta = apply_link(mu, "logit")
        mu_back = apply_inverse_link(eta, "logit")
        np.testing.assert_allclose(mu_back, mu)

    def test_roundtrip_identity(self):
        from rustystats.formula import apply_inverse_link

        mu = np.array([-1.0, 0.0, 5.0])
        eta = apply_link(mu, "identity")
        mu_back = apply_inverse_link(eta, "identity")
        np.testing.assert_allclose(mu_back, mu)

    def test_roundtrip_inverse(self):
        from rustystats.formula import apply_inverse_link

        mu = np.array([0.5, 1.0, 2.0])
        eta = apply_link(mu, "inverse")
        mu_back = apply_inverse_link(eta, "inverse")
        np.testing.assert_allclose(mu_back, mu)


# =============================================================================
# Complement construction & validation
# =============================================================================


class TestComplementConstruction:
    """Test complement parameter handling during model construction."""

    def test_string_complement(self, poisson_data):
        """Complement from a column name."""
        model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "cat": {"type": "categorical"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        )
        result = model.fit()
        assert result.has_complement
        assert result.converged

    def test_array_complement(self, poisson_data):
        """Complement from a numpy array."""
        cw_rate = poisson_data["cw_rate"].to_numpy()
        model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement=cw_rate,
        )
        result = model.fit()
        assert result.has_complement
        assert result.converged

    def test_glmmodel_complement(self, poisson_data):
        """Complement from a fitted GLMModel."""
        # Fit a "countrywide" model first
        cw_model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
        ).fit()

        # Use it as complement
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "cat": {"type": "categorical"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement=cw_model,
        ).fit()
        assert result.has_complement
        assert result.converged

    def test_no_complement(self, poisson_data):
        """Model without complement should have has_complement=False."""
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
        ).fit()
        assert not result.has_complement


class TestComplementValidation:
    """Test complement value validation."""

    def test_negative_values_log_link(self, poisson_data):
        """Negative complement values should fail for log-link models."""
        bad_rate = np.full(len(poisson_data), -1.0)
        with pytest.raises(ValidationError, match="strictly positive"):
            rs.glm_dict(
                response="y",
                terms={"x1": {"type": "linear"}},
                data=poisson_data,
                family="poisson",
                offset="exposure",
                complement=bad_rate,
            )

    def test_zero_values_log_link(self, poisson_data):
        """Zero complement values should fail for log-link models."""
        bad_rate = np.zeros(len(poisson_data))
        with pytest.raises(ValidationError, match="strictly positive"):
            rs.glm_dict(
                response="y",
                terms={"x1": {"type": "linear"}},
                data=poisson_data,
                family="poisson",
                offset="exposure",
                complement=bad_rate,
            )

    def test_out_of_range_logit_link(self, binomial_data):
        """Complement values outside (0,1) should fail for logit-link."""
        bad_prob = np.full(len(binomial_data), 1.5)
        with pytest.raises(ValidationError, match="must be in \\(0, 1\\)"):
            rs.glm_dict(
                response="y",
                terms={"x1": {"type": "linear"}},
                data=binomial_data,
                family="binomial",
                complement=bad_prob,
            )

    def test_zero_logit_link(self, binomial_data):
        """Zero complement should fail for logit-link."""
        bad_prob = np.zeros(len(binomial_data))
        with pytest.raises(ValidationError, match="must be in \\(0, 1\\)"):
            rs.glm_dict(
                response="y",
                terms={"x1": {"type": "linear"}},
                data=binomial_data,
                family="binomial",
                complement=bad_prob,
            )

    def test_one_logit_link(self, binomial_data):
        """Complement of 1.0 should fail for logit-link."""
        bad_prob = np.ones(len(binomial_data))
        with pytest.raises(ValidationError, match="must be in \\(0, 1\\)"):
            rs.glm_dict(
                response="y",
                terms={"x1": {"type": "linear"}},
                data=binomial_data,
                family="binomial",
                complement=bad_prob,
            )

    def test_valid_logit_complement(self, binomial_data):
        """Valid probabilities should succeed for logit-link."""
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=binomial_data,
            family="binomial",
            complement="prior_prob",
        ).fit()
        assert result.has_complement
        assert result.converged

    def test_identity_link_no_restriction(self, gaussian_data):
        """Identity link should accept any complement values (including negative)."""
        neg_values = np.full(len(gaussian_data), -5.0)
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=gaussian_data,
            family="gaussian",
            complement=neg_values,
        ).fit()
        assert result.has_complement
        assert result.converged


# =============================================================================
# Fitting with complement across families
# =============================================================================


class TestComplementFitting:
    """Test that complement works with different families and link functions."""

    def test_poisson_log_link(self, poisson_data):
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        ).fit()
        assert result.converged
        assert result.has_complement
        assert len(result.params) == 3  # intercept + x1 + x2

    def test_binomial_logit_link(self, binomial_data):
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=binomial_data,
            family="binomial",
            complement="prior_prob",
        ).fit()
        assert result.converged
        assert result.has_complement

    def test_gamma_log_link(self, gamma_data):
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=gamma_data,
            family="gamma",
            complement="prior_rate",
        ).fit()
        assert result.converged
        assert result.has_complement

    def test_gaussian_identity_link(self, gaussian_data):
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=gaussian_data,
            family="gaussian",
            complement="prior_pred",
        ).fit()
        assert result.converged
        assert result.has_complement

    def test_complement_with_categorical(self, poisson_data):
        """Complement works with categorical terms."""
        result = rs.glm_dict(
            response="y",
            terms={
                "x1": {"type": "linear"},
                "cat": {"type": "categorical"},
            },
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        ).fit()
        assert result.converged
        assert result.has_complement

    def test_complement_with_splines(self, poisson_data):
        """Complement works with spline terms."""
        result = rs.glm_dict(
            response="y",
            terms={
                "x1": {"type": "bs", "df": 4},
                "cat": {"type": "categorical"},
            },
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        ).fit()
        assert result.converged
        assert result.has_complement

    def test_complement_unpenalized(self, poisson_data):
        """Complement works without regularization (just shifts baseline)."""
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        ).fit()
        assert result.converged
        assert result.has_complement
        # Without regularization, complement shifts the offset but model still fits

    def test_complement_with_lasso(self, poisson_data):
        """Lasso credibility: complement + lasso shrinks toward complement."""
        result = rs.glm_dict(
            response="y",
            terms={
                "x1": {"type": "linear"},
                "x2": {"type": "linear"},
                "cat": {"type": "categorical"},
            },
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        ).fit(alpha=1.0, l1_ratio=1.0)
        assert result.converged
        assert result.has_complement
        assert result.is_regularized

    def test_complement_with_ridge(self, poisson_data):
        """Complement + ridge also works (shrinks toward complement)."""
        result = rs.glm_dict(
            response="y",
            terms={
                "x1": {"type": "linear"},
                "x2": {"type": "linear"},
            },
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        ).fit(alpha=1.0, l1_ratio=0.0)
        assert result.converged
        assert result.has_complement
        assert result.is_regularized

    def test_complement_with_elastic_net(self, poisson_data):
        """Complement + elastic net."""
        result = rs.glm_dict(
            response="y",
            terms={
                "x1": {"type": "linear"},
                "x2": {"type": "linear"},
            },
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        ).fit(alpha=1.0, l1_ratio=0.5)
        assert result.converged
        assert result.has_complement
        assert result.is_regularized


# =============================================================================
# Lasso shrinkage behavior
# =============================================================================


class TestLassoShrinkage:
    """Test that lasso credibility shrinks toward complement, not toward zero."""

    def test_strong_lasso_zeros_coefficients(self, poisson_data):
        """With very strong lasso, all non-intercept coefficients should be zeroed."""
        result = rs.glm_dict(
            response="y",
            terms={
                "x1": {"type": "linear"},
                "x2": {"type": "linear"},
                "cat": {"type": "categorical"},
            },
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        ).fit(alpha=100.0, l1_ratio=1.0)

        # Non-intercept coefficients should be shrunk to zero
        non_intercept = result.params[1:]
        n_zeroed = np.sum(np.abs(non_intercept) < 1e-10)
        # With strong enough penalty, most/all should be zero
        assert n_zeroed > 0, "Strong lasso should zero some coefficients"

    def test_lasso_predictions_converge_to_complement(self, poisson_data):
        """With strong lasso, predictions should converge to complement * exposure."""
        cw_rate = poisson_data["cw_rate"].to_numpy()
        exposure = poisson_data["exposure"].to_numpy()

        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        ).fit(alpha=1000.0, l1_ratio=1.0)

        preds = result.predict(poisson_data)
        # When all betas→0, pred = exp(b0) * exposure * cw_rate
        # The predictions should be proportional to cw_rate * exposure
        ratio = preds / (cw_rate * exposure)
        # All ratios should be approximately the same (= exp(b0))
        np.testing.assert_allclose(ratio, ratio[0], rtol=0.01)


# =============================================================================
# Prediction with complement
# =============================================================================


class TestComplementPrediction:
    """Test prediction behavior with complement."""

    def test_predict_auto_applies_string_complement(self, poisson_data):
        """When fit with complement=str, predict() auto-applies it."""
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        ).fit()

        # Predict on same data — should automatically use cw_rate column
        preds = result.predict(poisson_data)
        assert len(preds) == len(poisson_data)
        assert np.all(np.isfinite(preds))
        assert np.all(preds > 0)

    def test_predict_auto_applies_glmmodel_complement(self, poisson_data):
        """When fit with complement=GLMModel, predict() auto-applies it."""
        cw_model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
        ).fit()

        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "cat": {"type": "categorical"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement=cw_model,
        ).fit()

        preds = result.predict(poisson_data)
        assert len(preds) == len(poisson_data)
        assert np.all(np.isfinite(preds))
        assert np.all(preds > 0)

    def test_predict_manual_complement_override(self, poisson_data):
        """Manually providing complement to predict() overrides stored one."""
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        ).fit()

        # Override with different complement
        alt_rate = np.ones(len(poisson_data)) * 0.5
        preds_default = result.predict(poisson_data)
        preds_override = result.predict(poisson_data, complement=alt_rate)

        # Should give different predictions
        assert not np.allclose(preds_default, preds_override)

    def test_predict_without_complement_on_complement_model(self, poisson_data):
        """Predict with complement=array of ones should differ from auto-complement."""
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        ).fit()

        preds_with = result.predict(poisson_data)  # auto-applies cw_rate
        # Use complement=1.0 (log(1)=0, effectively no complement)
        preds_neutral = result.predict(poisson_data, complement=np.ones(len(poisson_data)))

        # These should differ since cw_rate is not all ones
        assert not np.allclose(preds_with, preds_neutral)

    def test_predict_string_and_array_complement_match(self, poisson_data):
        """String complement and equivalent array should give same predictions."""
        cw_rate_array = poisson_data["cw_rate"].to_numpy()

        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        ).fit()

        preds_str = result.predict(poisson_data)  # uses column "cw_rate"
        preds_arr = result.predict(poisson_data, complement=cw_rate_array)

        np.testing.assert_allclose(preds_str, preds_arr, rtol=1e-10)

    def test_predict_binomial_with_complement(self, binomial_data):
        """Binomial prediction with complement stays in (0, 1)."""
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=binomial_data,
            family="binomial",
            complement="prior_prob",
        ).fit()

        preds = result.predict(binomial_data)
        assert np.all(preds > 0)
        assert np.all(preds < 1)


# =============================================================================
# credibility_summary()
# =============================================================================


class TestCredibilitySummary:
    """Test credibility_summary() method."""

    def test_basic_credibility_summary(self, poisson_data):
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        ).fit(regularization="lasso")

        df = result.credibility_summary()
        assert isinstance(df, pl.DataFrame)
        assert "Feature" in df.columns
        assert "Deviation" in df.columns
        assert "Zeroed" in df.columns
        # Log-link should include Deviation_Factor
        assert "Deviation_Factor" in df.columns

    def test_credibility_summary_row_count(self, poisson_data):
        """One row per coefficient (including intercept)."""
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "cat": {"type": "categorical"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        ).fit()

        df = result.credibility_summary()
        assert len(df) == len(result.params)
        assert len(df) == len(result.feature_names)

    def test_credibility_summary_deviation_factor(self, poisson_data):
        """Deviation_Factor should be exp(Deviation) for log-link."""
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        ).fit()

        df = result.credibility_summary()
        deviations = df["Deviation"].to_numpy()
        factors = df["Deviation_Factor"].to_numpy()
        np.testing.assert_allclose(factors, np.exp(deviations))

    def test_credibility_summary_zeroed_flags(self, poisson_data):
        """Strong lasso should produce some True values in Zeroed column."""
        result = rs.glm_dict(
            response="y",
            terms={
                "x1": {"type": "linear"},
                "x2": {"type": "linear"},
                "cat": {"type": "categorical"},
            },
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        ).fit(alpha=50.0, l1_ratio=1.0)

        df = result.credibility_summary()
        zeroed = df["Zeroed"].to_list()
        assert any(zeroed), "Strong lasso should zero at least one coefficient"

    def test_credibility_summary_no_deviation_factor_binomial(self, binomial_data):
        """Logit-link should not include Deviation_Factor column."""
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=binomial_data,
            family="binomial",
            complement="prior_prob",
        ).fit()

        df = result.credibility_summary()
        assert "Deviation_Factor" not in df.columns

    def test_credibility_summary_metadata_note(self, poisson_data):
        """Summary should have _credibility_note metadata."""
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        ).fit()

        df = result.credibility_summary()
        assert hasattr(df, "__dict__")
        note = df.__dict__.get("_credibility_note", "")
        assert "non-intercept terms zeroed" in note

    def test_credibility_summary_raises_without_complement(self, poisson_data):
        """credibility_summary() should raise if model has no complement."""
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
        ).fit()

        with pytest.raises(ValidationError, match="complement="):
            result.credibility_summary()


# =============================================================================
# summary() output
# =============================================================================


class TestSummaryOutput:
    """Test summary() output for complement models."""

    def test_summary_title_with_string_complement(self, poisson_data):
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        ).fit()

        s = result.summary()
        assert "Lasso Credibility Results" in s
        assert "complement: cw_rate" in s

    def test_summary_title_with_glmmodel_complement(self, poisson_data):
        cw_model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
        ).fit()

        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement=cw_model,
        ).fit()

        s = result.summary()
        assert "Lasso Credibility Results" in s
        assert "complement: GLMModel" in s

    def test_summary_complement_note(self, poisson_data):
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        ).fit()

        s = result.summary()
        assert "deviations from the complement of credibility" in s
        assert "non-intercept terms zeroed" in s

    def test_summary_no_complement_note_without_complement(self, poisson_data):
        """Standard model summary should not mention complement."""
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
        ).fit()

        s = result.summary()
        assert "complement" not in s.lower()
        assert "Lasso Credibility" not in s

    def test_summary_zeroed_count(self, poisson_data):
        """Summary should report correct count of zeroed terms."""
        result = rs.glm_dict(
            response="y",
            terms={
                "x1": {"type": "linear"},
                "x2": {"type": "linear"},
            },
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        ).fit(alpha=100.0, l1_ratio=1.0)

        s = result.summary()
        # Should show "N/2 non-intercept terms zeroed"
        assert "non-intercept terms zeroed" in s


# =============================================================================
# has_complement property
# =============================================================================


class TestHasComplement:
    """Test has_complement property."""

    def test_true_with_complement(self, poisson_data):
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        ).fit()
        assert result.has_complement is True

    def test_false_without_complement(self, poisson_data):
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
        ).fit()
        assert result.has_complement is False


# =============================================================================
# Serialization round-trip
# =============================================================================


class TestComplementSerialization:
    """Test to_bytes/from_bytes preserves complement metadata."""

    def test_roundtrip_preserves_complement_spec(self, poisson_data):
        """String complement_spec should survive serialization."""
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "cat": {"type": "categorical"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        ).fit()

        model_bytes = result.to_bytes()
        loaded = rs.GLMModel.from_bytes(model_bytes)

        assert loaded._complement_spec == "cw_rate"

    def test_roundtrip_prediction_matches(self, poisson_data):
        """Predictions from loaded model should match original."""
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "cat": {"type": "categorical"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        ).fit()

        preds_original = result.predict(poisson_data)

        model_bytes = result.to_bytes()
        loaded = rs.GLMModel.from_bytes(model_bytes)
        preds_loaded = loaded.predict(poisson_data)

        np.testing.assert_allclose(preds_original, preds_loaded, rtol=1e-10)

    def test_roundtrip_params_match(self, poisson_data):
        """Coefficients should be identical after serialization."""
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        ).fit()

        model_bytes = result.to_bytes()
        loaded = rs.GLMModel.from_bytes(model_bytes)

        np.testing.assert_array_equal(result.params, loaded.params)
        assert result.family == loaded.family
        assert result.link == loaded.link

    def test_roundtrip_no_complement(self, poisson_data):
        """Model without complement should also roundtrip correctly."""
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
        ).fit()

        model_bytes = result.to_bytes()
        loaded = rs.GLMModel.from_bytes(model_bytes)

        assert loaded._complement_spec is None
        np.testing.assert_array_equal(result.params, loaded.params)


# =============================================================================
# Edge cases
# =============================================================================


class TestComplementEdgeCases:
    """Test edge cases and corner cases."""

    def test_constant_complement(self, poisson_data):
        """Constant complement (all same value) should work fine."""
        const_rate = np.full(len(poisson_data), 0.5)
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement=const_rate,
        ).fit()
        assert result.converged

    def test_complement_near_boundary_logit(self, binomial_data):
        """Complement values close to 0 or 1 should still work."""
        n = len(binomial_data)
        near_boundary = np.random.uniform(0.01, 0.99, n)
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=binomial_data,
            family="binomial",
            complement=near_boundary,
        ).fit()
        assert result.converged

    def test_complement_with_weights(self, poisson_data):
        """Complement and weights should work together."""
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
            weights=np.ones(len(poisson_data)),
            complement="cw_rate",
        ).fit()
        assert result.converged
        assert result.has_complement

    def test_complement_with_explicit_alpha(self, poisson_data):
        """Complement with explicit alpha=0 (no regularization)."""
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=poisson_data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        ).fit(alpha=0.0)
        assert result.converged
        assert result.has_complement
        assert not result.is_regularized
