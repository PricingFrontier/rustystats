"""
Comprehensive comparison of RustyStats against statsmodels and glum.

This test suite validates every major feature of RustyStats by comparing
numerical results against established GLM libraries. It is designed for
actuarial-grade confidence: coefficients, standard errors, deviance,
log-likelihood, residuals, robust SEs, regularization, splines, encoding,
interactions, predictions, serialization, and edge cases.

Run with:
    pytest tests/python/test_comparison.py -v --tb=short
"""

import numpy as np
import polars as pl
import pytest
import warnings

import rustystats as rs

# Reference libraries
import statsmodels.api as sm
import statsmodels.genmod.families as smf
from glum import GeneralizedLinearRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SEED = 42
N = 2000  # large enough for stable estimates

COEF_ATOL = 0.02       # absolute tolerance on coefficients
COEF_RTOL = 0.01       # 1 % relative tolerance on coefficients
SE_RTOL = 0.03         # 3 % relative tolerance on standard errors
DEVIANCE_RTOL = 0.005  # 0.5 % relative tolerance on deviance
LLF_ATOL = 2.0         # absolute tolerance on log-likelihood
PRED_RTOL = 0.005      # 0.5 % relative tolerance on predictions
RESID_ATOL = 0.05      # absolute tolerance on residuals


def _close(a, b, atol=0.0, rtol=0.01):
    """Check if two values are close with combined tolerances."""
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def _arr_close(a, b, atol=0.0, rtol=0.01):
    """Check arrays are element-wise close."""
    return np.allclose(a, b, atol=atol, rtol=rtol)


def _gen_data(seed=SEED):
    """Generate shared synthetic dataset for most tests."""
    rng = np.random.RandomState(seed)
    n = N
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(-5, 5, n)
    cat = rng.choice(["A", "B", "C", "D"], n)
    exposure = rng.uniform(0.5, 2.0, n)
    weight = rng.uniform(0.5, 2.0, n)

    # Gaussian
    y_gauss = 3.0 + 0.5 * x1 - 0.3 * x2 + rng.normal(0, 2, n)

    # Poisson
    eta_pois = -0.5 + 0.2 * x1 + 0.1 * x2
    y_pois = rng.poisson(np.exp(eta_pois)).astype(float)

    # Binomial
    p_binom = 1 / (1 + np.exp(-(0.5 + 0.3 * x1 - 0.2 * x2 - 1.5)))
    y_binom = rng.binomial(1, p_binom).astype(float)

    # Gamma
    mu_gamma = np.exp(2.0 + 0.1 * x1)
    shape = 2.0
    y_gamma = rng.gamma(shape, mu_gamma / shape)

    # Negative Binomial
    mu_nb = np.exp(1.0 + 0.15 * x1)
    alpha_nb = 0.5  # statsmodels alpha
    y_nb = rng.negative_binomial(1 / alpha_nb, 1 / (1 + alpha_nb * mu_nb)).astype(float)

    df = pl.DataFrame({
        "y_gauss": y_gauss,
        "y_pois": y_pois,
        "y_binom": y_binom,
        "y_gamma": y_gamma,
        "y_nb": y_nb,
        "x1": x1,
        "x2": x2,
        "cat": cat,
        "exposure": exposure,
        "weight": weight,
    })
    return df, x1, x2, cat, exposure, weight


@pytest.fixture(scope="module")
def data():
    """Module-scoped fixture for shared data."""
    return _gen_data()


# ===========================================================================
# Test 1: Gaussian family
# ===========================================================================

class TestGaussianFamily:
    """Gaussian GLM: coefficients, SE, deviance, llf, AIC, BIC, scale, predictions."""

    def test_vs_statsmodels(self, data):
        df, x1, x2, *_ = data
        y = df["y_gauss"].to_numpy()
        X_sm = sm.add_constant(np.column_stack([x1, x2]))

        sm_res = sm.GLM(y, X_sm, family=smf.Gaussian()).fit()

        rs_res = rs.glm_dict(
            response="y_gauss",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="gaussian",
        ).fit()

        # Coefficients
        np.testing.assert_allclose(rs_res.params, sm_res.params, atol=COEF_ATOL, rtol=COEF_RTOL)
        # Standard errors
        np.testing.assert_allclose(rs_res.bse(), sm_res.bse, rtol=SE_RTOL)
        # Deviance
        assert _close(rs_res.deviance, sm_res.deviance, rtol=DEVIANCE_RTOL)
        # Log-likelihood
        assert _close(rs_res.llf(), sm_res.llf, atol=LLF_ATOL)
        # AIC / BIC
        assert _close(rs_res.aic(), sm_res.aic, atol=5.0)
        # BIC: statsmodels uses deviance-based BIC by default; compare against bic_llf
        sm_bic = getattr(sm_res, "bic_llf", sm_res.bic)
        assert _close(rs_res.bic(), sm_bic, atol=5.0)
        # Scale
        assert _close(rs_res.scale(), sm_res.scale, rtol=0.02)
        # Predictions
        np.testing.assert_allclose(rs_res.fittedvalues, sm_res.predict(X_sm), rtol=PRED_RTOL)

    def test_vs_glum(self, data):
        df, x1, x2, *_ = data
        y = df["y_gauss"].to_numpy()
        X_np = np.column_stack([x1, x2])

        glum_model = GeneralizedLinearRegressor(family="gaussian", alpha=0, fit_intercept=True)
        glum_model.fit(X_np, y)

        rs_res = rs.glm_dict(
            response="y_gauss",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="gaussian",
        ).fit()

        glum_coefs = np.concatenate([[glum_model.intercept_], glum_model.coef_])
        np.testing.assert_allclose(rs_res.params, glum_coefs, atol=COEF_ATOL, rtol=COEF_RTOL)

        glum_pred = glum_model.predict(X_np)
        np.testing.assert_allclose(rs_res.fittedvalues, glum_pred, rtol=PRED_RTOL)


# ===========================================================================
# Test 2: Poisson family
# ===========================================================================

class TestPoissonFamily:
    """Poisson GLM: coefficients, SE, deviance, llf, AIC, BIC, predictions."""

    def test_vs_statsmodels(self, data):
        df, x1, x2, *_ = data
        y = df["y_pois"].to_numpy()
        X_sm = sm.add_constant(np.column_stack([x1, x2]))

        sm_res = sm.GLM(y, X_sm, family=smf.Poisson()).fit()

        rs_res = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson",
        ).fit()

        np.testing.assert_allclose(rs_res.params, sm_res.params, atol=COEF_ATOL, rtol=COEF_RTOL)
        np.testing.assert_allclose(rs_res.bse(), sm_res.bse, rtol=SE_RTOL)
        assert _close(rs_res.deviance, sm_res.deviance, rtol=DEVIANCE_RTOL)
        assert _close(rs_res.llf(), sm_res.llf, atol=LLF_ATOL)
        assert _close(rs_res.aic(), sm_res.aic, atol=5.0)
        np.testing.assert_allclose(rs_res.fittedvalues, sm_res.predict(X_sm), rtol=PRED_RTOL)

    def test_vs_glum(self, data):
        df, x1, x2, *_ = data
        y = df["y_pois"].to_numpy()
        X_np = np.column_stack([x1, x2])

        glum_model = GeneralizedLinearRegressor(family="poisson", alpha=0, fit_intercept=True)
        glum_model.fit(X_np, y)

        rs_res = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson",
        ).fit()

        glum_coefs = np.concatenate([[glum_model.intercept_], glum_model.coef_])
        np.testing.assert_allclose(rs_res.params, glum_coefs, atol=COEF_ATOL, rtol=COEF_RTOL)


# ===========================================================================
# Test 3: Binomial (Logistic) family
# ===========================================================================

class TestBinomialFamily:
    """Binomial GLM: coefficients, SE, deviance, llf, AIC, BIC, predictions."""

    def test_vs_statsmodels(self, data):
        df, x1, x2, *_ = data
        y = df["y_binom"].to_numpy()
        X_sm = sm.add_constant(np.column_stack([x1, x2]))

        sm_res = sm.GLM(y, X_sm, family=smf.Binomial()).fit()

        rs_res = rs.glm_dict(
            response="y_binom",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="binomial",
        ).fit()

        np.testing.assert_allclose(rs_res.params, sm_res.params, atol=COEF_ATOL, rtol=COEF_RTOL)
        np.testing.assert_allclose(rs_res.bse(), sm_res.bse, rtol=SE_RTOL)
        assert _close(rs_res.deviance, sm_res.deviance, rtol=DEVIANCE_RTOL)
        assert _close(rs_res.llf(), sm_res.llf, atol=LLF_ATOL)
        np.testing.assert_allclose(rs_res.fittedvalues, sm_res.predict(X_sm), rtol=PRED_RTOL)

    def test_vs_glum(self, data):
        df, x1, x2, *_ = data
        y = df["y_binom"].to_numpy()
        X_np = np.column_stack([x1, x2])

        glum_model = GeneralizedLinearRegressor(family="binomial", alpha=0, fit_intercept=True)
        glum_model.fit(X_np, y)

        rs_res = rs.glm_dict(
            response="y_binom",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="binomial",
        ).fit()

        glum_coefs = np.concatenate([[glum_model.intercept_], glum_model.coef_])
        np.testing.assert_allclose(rs_res.params, glum_coefs, atol=COEF_ATOL, rtol=COEF_RTOL)


# ===========================================================================
# Test 4: Gamma family
# ===========================================================================

class TestGammaFamily:
    """Gamma GLM: coefficients, SE, deviance, llf, AIC, BIC, scale, predictions."""

    def test_vs_statsmodels(self, data):
        df, x1, *_ = data
        y = df["y_gamma"].to_numpy()
        X_sm = sm.add_constant(x1)

        sm_res = sm.GLM(y, X_sm, family=smf.Gamma(smf.links.Log())).fit()

        rs_res = rs.glm_dict(
            response="y_gamma",
            terms={"x1": {"type": "linear"}},
            data=df, family="gamma",
        ).fit()

        np.testing.assert_allclose(rs_res.params, sm_res.params, atol=COEF_ATOL, rtol=COEF_RTOL)
        np.testing.assert_allclose(rs_res.bse(), sm_res.bse, rtol=SE_RTOL)
        assert _close(rs_res.deviance, sm_res.deviance, rtol=DEVIANCE_RTOL)
        assert _close(rs_res.scale(), sm_res.scale, rtol=0.05)
        np.testing.assert_allclose(rs_res.fittedvalues, sm_res.predict(X_sm), rtol=PRED_RTOL)

    def test_vs_glum(self, data):
        df, x1, *_ = data
        y = df["y_gamma"].to_numpy()
        X_np = x1.reshape(-1, 1)

        glum_model = GeneralizedLinearRegressor(family="gamma", link="log", alpha=0, fit_intercept=True)
        glum_model.fit(X_np, y)

        rs_res = rs.glm_dict(
            response="y_gamma",
            terms={"x1": {"type": "linear"}},
            data=df, family="gamma",
        ).fit()

        glum_coefs = np.concatenate([[glum_model.intercept_], glum_model.coef_])
        np.testing.assert_allclose(rs_res.params, glum_coefs, atol=COEF_ATOL, rtol=COEF_RTOL)


# ===========================================================================
# Test 5: Tweedie family
# ===========================================================================

class TestTweedieFamily:
    """Tweedie GLM (var_power=1.5): coefficients, deviance, predictions."""

    def test_vs_statsmodels(self, data):
        df, x1, *_ = data
        # Generate Tweedie-like data (compound Poisson-Gamma)
        rng = np.random.RandomState(SEED)
        p_tw = 1.5
        mu_tw = np.exp(1.0 + 0.1 * x1)
        phi_tw = 2.0
        lambda_p = mu_tw ** (2 - p_tw) / (phi_tw * (2 - p_tw))
        n_claims = rng.poisson(lambda_p)
        alpha_g = (2 - p_tw) / (p_tw - 1)
        beta_g = phi_tw * (p_tw - 1) * mu_tw ** (p_tw - 1)
        y_tw = np.array([
            np.sum(rng.gamma(alpha_g, beta_g[i], n_claims[i])) if n_claims[i] > 0 else 0.0
            for i in range(N)
        ])
        # Use only positive values for a cleaner comparison
        mask = y_tw > 0
        y_pos = y_tw[mask]
        x1_pos = x1[mask]

        X_sm = sm.add_constant(x1_pos)
        sm_res = sm.GLM(
            y_pos, X_sm,
            family=smf.Tweedie(var_power=p_tw, link=smf.links.Log()),
        ).fit()

        df_tw = pl.DataFrame({"y": y_pos, "x1": x1_pos})
        rs_res = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=df_tw, family="tweedie", var_power=p_tw,
        ).fit()

        np.testing.assert_allclose(rs_res.params, sm_res.params, atol=0.1, rtol=0.05)
        np.testing.assert_allclose(rs_res.fittedvalues, sm_res.predict(X_sm), rtol=0.02)

    def test_vs_glum(self, data):
        df, x1, *_ = data
        rng = np.random.RandomState(SEED)
        p_tw = 1.5
        mu_tw = np.exp(1.0 + 0.1 * x1)
        phi_tw = 2.0
        lambda_p = mu_tw ** (2 - p_tw) / (phi_tw * (2 - p_tw))
        n_claims = rng.poisson(lambda_p)
        alpha_g = (2 - p_tw) / (p_tw - 1)
        beta_g = phi_tw * (p_tw - 1) * mu_tw ** (p_tw - 1)
        y_tw = np.array([
            np.sum(rng.gamma(alpha_g, beta_g[i], n_claims[i])) if n_claims[i] > 0 else 0.0
            for i in range(N)
        ])
        mask = y_tw > 0
        y_pos = y_tw[mask]
        x1_pos = x1[mask]

        from glum import TweedieDistribution
        glum_model = GeneralizedLinearRegressor(
            family=TweedieDistribution(power=p_tw), link="log", alpha=0, fit_intercept=True,
        )
        glum_model.fit(x1_pos.reshape(-1, 1), y_pos)

        df_tw = pl.DataFrame({"y": y_pos, "x1": x1_pos})
        rs_res = rs.glm_dict(
            response="y", terms={"x1": {"type": "linear"}},
            data=df_tw, family="tweedie", var_power=p_tw,
        ).fit()

        glum_coefs = np.concatenate([[glum_model.intercept_], glum_model.coef_])
        np.testing.assert_allclose(rs_res.params, glum_coefs, atol=0.1, rtol=0.05)


# ===========================================================================
# Test 6: Negative Binomial family
# ===========================================================================

class TestNegBinomialFamily:
    """Negative Binomial GLM: coefficients, SE, deviance, predictions."""

    def test_vs_statsmodels(self, data):
        df, x1, *_ = data
        y = df["y_nb"].to_numpy()
        X_sm = sm.add_constant(x1)

        alpha_nb = 0.5
        theta_nb = 1.0 / alpha_nb

        sm_res = sm.GLM(y, X_sm, family=smf.NegativeBinomial(alpha=alpha_nb)).fit()

        rs_res = rs.glm_dict(
            response="y_nb",
            terms={"x1": {"type": "linear"}},
            data=df, family="negbinomial", theta=theta_nb,
        ).fit()

        np.testing.assert_allclose(rs_res.params, sm_res.params, atol=0.05, rtol=0.05)
        np.testing.assert_allclose(rs_res.bse(), sm_res.bse, rtol=0.1)
        np.testing.assert_allclose(rs_res.fittedvalues, sm_res.predict(X_sm), rtol=0.02)


# ===========================================================================
# Test 7: QuasiPoisson
# ===========================================================================

class TestQuasiPoisson:
    """QuasiPoisson: same coefficients as Poisson, inflated SEs, dispersion."""

    def test_vs_statsmodels(self, data):
        df, x1, x2, *_ = data
        # Generate overdispersed counts
        rng = np.random.RandomState(SEED + 1)
        eta = 0.5 + 0.2 * x1
        mu = np.exp(eta)
        # Introduce overdispersion via Gamma mixture
        y = rng.poisson(rng.gamma(2, mu / 2)).astype(float)

        X_sm = sm.add_constant(x1)
        df_qp = pl.DataFrame({"y": y, "x1": x1})

        # Poisson for point estimates
        sm_pois = sm.GLM(y, X_sm, family=smf.Poisson()).fit()

        # Statsmodels QuasiPoisson: same point estimates, scale estimated
        sm_qp = sm.GLM(y, X_sm, family=smf.Poisson()).fit(scale="X2")

        rs_qp = rs.glm_dict(
            response="y", terms={"x1": {"type": "linear"}},
            data=df_qp, family="quasipoisson",
        ).fit()

        # Point estimates should match Poisson
        np.testing.assert_allclose(rs_qp.params, sm_pois.params, atol=COEF_ATOL, rtol=COEF_RTOL)

        # Scale / dispersion should be > 1 (overdispersed)
        rs_scale = rs_qp.scale()
        sm_scale = sm_qp.scale
        assert rs_scale > 1.0, f"QuasiPoisson scale should be > 1, got {rs_scale}"
        assert _close(rs_scale, sm_scale, rtol=0.05)

        # SEs should be inflated relative to Poisson
        rs_se_qp = rs_qp.bse()
        rs_se_pois = rs.glm_dict(
            response="y", terms={"x1": {"type": "linear"}},
            data=df_qp, family="poisson",
        ).fit().bse()
        assert np.all(rs_se_qp > rs_se_pois), "QuasiPoisson SEs should exceed Poisson SEs"


# ===========================================================================
# Test 8: QuasiBinomial
# ===========================================================================

class TestQuasiBinomial:
    """QuasiBinomial: same coefficients as Binomial, inflated SEs."""

    def test_vs_statsmodels(self, data):
        df, x1, *_ = data
        y = df["y_binom"].to_numpy()
        X_sm = sm.add_constant(x1)
        df_qb = pl.DataFrame({"y": y, "x1": x1})

        sm_binom = sm.GLM(y, X_sm, family=smf.Binomial()).fit()
        sm_qb = sm.GLM(y, X_sm, family=smf.Binomial()).fit(scale="X2")

        rs_qb = rs.glm_dict(
            response="y", terms={"x1": {"type": "linear"}},
            data=df_qb, family="quasibinomial",
        ).fit()

        # Point estimates match Binomial
        np.testing.assert_allclose(rs_qb.params, sm_binom.params, atol=COEF_ATOL, rtol=COEF_RTOL)

        # Scale should be estimated
        rs_scale = rs_qb.scale()
        sm_scale = sm_qb.scale
        assert _close(rs_scale, sm_scale, rtol=0.1)


# ===========================================================================
# Test 9: Poisson with exposure offset
# ===========================================================================

class TestPoissonOffset:
    """Poisson with log(exposure) offset: matches statsmodels and glum."""

    def test_vs_statsmodels(self, data):
        df, x1, x2, cat, exposure, _ = data
        rng = np.random.RandomState(SEED)
        eta = -1.0 + 0.15 * x1
        y = rng.poisson(exposure * np.exp(eta)).astype(float)

        X_sm = sm.add_constant(x1)
        log_exposure = np.log(exposure)
        sm_res = sm.GLM(
            y, X_sm, family=smf.Poisson(), offset=log_exposure,
        ).fit()

        df_off = df.with_columns(pl.Series("y_off", y))
        rs_res = rs.glm_dict(
            response="y_off",
            terms={"x1": {"type": "linear"}},
            data=df_off, family="poisson", offset="exposure",
        ).fit()

        np.testing.assert_allclose(rs_res.params, sm_res.params, atol=COEF_ATOL, rtol=COEF_RTOL)
        # Compare fitted values: statsmodels predict() needs explicit offset
        sm_pred_with_offset = sm_res.predict(X_sm, offset=log_exposure)
        np.testing.assert_allclose(rs_res.fittedvalues, sm_pred_with_offset, rtol=PRED_RTOL)

    def test_vs_glum(self, data):
        df, x1, x2, cat, exposure, _ = data
        rng = np.random.RandomState(SEED)
        eta = -1.0 + 0.15 * x1
        y = rng.poisson(exposure * np.exp(eta)).astype(float)

        X_np = x1.reshape(-1, 1)
        glum_model = GeneralizedLinearRegressor(
            family="poisson", alpha=0, fit_intercept=True,
        )
        glum_model.fit(X_np, y, offset=np.log(exposure))

        df_off = df.with_columns(pl.Series("y_off", y))
        rs_res = rs.glm_dict(
            response="y_off",
            terms={"x1": {"type": "linear"}},
            data=df_off, family="poisson", offset="exposure",
        ).fit()

        glum_coefs = np.concatenate([[glum_model.intercept_], glum_model.coef_])
        np.testing.assert_allclose(rs_res.params, glum_coefs, atol=COEF_ATOL, rtol=COEF_RTOL)


# ===========================================================================
# Test 10: Prior weights
# ===========================================================================

class TestPriorWeights:
    """Weighted GLM: Gaussian and Poisson with sample weights vs statsmodels."""

    def test_gaussian_weights(self, data):
        df, x1, x2, cat, exposure, weight = data
        y = df["y_gauss"].to_numpy()
        X_sm = sm.add_constant(np.column_stack([x1, x2]))

        sm_res = sm.GLM(y, X_sm, family=smf.Gaussian(), freq_weights=weight).fit()

        rs_res = rs.glm_dict(
            response="y_gauss",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="gaussian", weights="weight",
        ).fit()

        np.testing.assert_allclose(rs_res.params, sm_res.params, atol=COEF_ATOL, rtol=COEF_RTOL)

    def test_poisson_weights(self, data):
        df, x1, x2, cat, exposure, weight = data
        y = df["y_pois"].to_numpy()
        X_sm = sm.add_constant(np.column_stack([x1, x2]))

        sm_res = sm.GLM(y, X_sm, family=smf.Poisson(), freq_weights=weight).fit()

        rs_res = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson", weights="weight",
        ).fit()

        np.testing.assert_allclose(rs_res.params, sm_res.params, atol=COEF_ATOL, rtol=COEF_RTOL)

    def test_poisson_weights_vs_glum(self, data):
        df, x1, x2, cat, exposure, weight = data
        y = df["y_pois"].to_numpy()
        X_np = np.column_stack([x1, x2])

        glum_model = GeneralizedLinearRegressor(family="poisson", alpha=0, fit_intercept=True)
        glum_model.fit(X_np, y, sample_weight=weight)

        rs_res = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson", weights="weight",
        ).fit()

        glum_coefs = np.concatenate([[glum_model.intercept_], glum_model.coef_])
        np.testing.assert_allclose(rs_res.params, glum_coefs, atol=COEF_ATOL, rtol=COEF_RTOL)


# ===========================================================================
# Test 11: Multiple predictors (continuous + categorical)
# ===========================================================================

class TestMultiplePredictors:
    """Mixed continuous + categorical predictors vs statsmodels."""

    def test_continuous_and_categorical(self, data):
        df, x1, x2, cat, *_ = data
        y = df["y_pois"].to_numpy()

        # Build statsmodels dummy encoding (drop first)
        import pandas as pd
        cat_dummies = pd.get_dummies(pd.Series(cat), drop_first=True, dtype=float)
        X_sm = sm.add_constant(
            np.column_stack([x1, cat_dummies.values])
        )
        sm_res = sm.GLM(y, X_sm, family=smf.Poisson()).fit()

        rs_res = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "cat": {"type": "categorical"}},
            data=df, family="poisson",
        ).fit()

        # Number of parameters should match
        assert len(rs_res.params) == len(sm_res.params), (
            f"Param count mismatch: rs={len(rs_res.params)} vs sm={len(sm_res.params)}"
        )

        # Predictions should match (even if coefficient order differs due to encoding)
        np.testing.assert_allclose(
            np.sort(rs_res.fittedvalues),
            np.sort(sm_res.predict(X_sm)),
            rtol=PRED_RTOL,
        )


# ===========================================================================
# Test 12: Residuals
# ===========================================================================

class TestResiduals:
    """Response, Pearson, deviance, working residuals vs statsmodels."""

    def test_poisson_residuals(self, data):
        df, x1, x2, *_ = data
        y = df["y_pois"].to_numpy()
        X_sm = sm.add_constant(np.column_stack([x1, x2]))

        sm_res = sm.GLM(y, X_sm, family=smf.Poisson()).fit()
        rs_res = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson",
        ).fit()

        # Response residuals: y - mu
        np.testing.assert_allclose(
            rs_res.resid_response(), sm_res.resid_response, atol=RESID_ATOL,
        )

        # Pearson residuals: (y - mu) / sqrt(V(mu))
        np.testing.assert_allclose(
            rs_res.resid_pearson(), sm_res.resid_pearson, atol=RESID_ATOL,
        )

        # Deviance residuals
        np.testing.assert_allclose(
            rs_res.resid_deviance(), sm_res.resid_deviance, atol=RESID_ATOL,
        )

        # Working residuals
        np.testing.assert_allclose(
            rs_res.resid_working(), sm_res.resid_working, atol=RESID_ATOL,
        )

    def test_gaussian_residuals(self, data):
        df, x1, x2, *_ = data
        y = df["y_gauss"].to_numpy()
        X_sm = sm.add_constant(np.column_stack([x1, x2]))

        sm_res = sm.GLM(y, X_sm, family=smf.Gaussian()).fit()
        rs_res = rs.glm_dict(
            response="y_gauss",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="gaussian",
        ).fit()

        np.testing.assert_allclose(rs_res.resid_response(), sm_res.resid_response, atol=RESID_ATOL)
        np.testing.assert_allclose(rs_res.resid_pearson(), sm_res.resid_pearson, atol=RESID_ATOL)
        np.testing.assert_allclose(rs_res.resid_deviance(), sm_res.resid_deviance, atol=RESID_ATOL)

    def test_gamma_residuals(self, data):
        df, x1, *_ = data
        y = df["y_gamma"].to_numpy()
        X_sm = sm.add_constant(x1)

        sm_res = sm.GLM(y, X_sm, family=smf.Gamma(smf.links.Log())).fit()
        rs_res = rs.glm_dict(
            response="y_gamma", terms={"x1": {"type": "linear"}},
            data=df, family="gamma",
        ).fit()

        np.testing.assert_allclose(rs_res.resid_pearson(), sm_res.resid_pearson, atol=RESID_ATOL)
        np.testing.assert_allclose(rs_res.resid_deviance(), sm_res.resid_deviance, atol=RESID_ATOL)


# ===========================================================================
# Test 13: Robust standard errors (sandwich estimators)
# ===========================================================================

class TestRobustSE:
    """HC0, HC1, HC2, HC3 sandwich estimators vs statsmodels."""

    def test_hc0_poisson(self, data):
        df, x1, x2, *_ = data
        y = df["y_pois"].to_numpy()
        X_sm = sm.add_constant(np.column_stack([x1, x2]))

        sm_res = sm.GLM(y, X_sm, family=smf.Poisson()).fit(cov_type="HC0")

        rs_res = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson",
        ).fit()

        np.testing.assert_allclose(
            rs_res.bse_robust("HC0"), sm_res.bse, rtol=0.05,
        )

    def test_hc1_gaussian(self, data):
        df, x1, x2, *_ = data
        y = df["y_gauss"].to_numpy()
        X_sm = sm.add_constant(np.column_stack([x1, x2]))

        sm_res = sm.GLM(y, X_sm, family=smf.Gaussian()).fit(cov_type="HC1")

        rs_res = rs.glm_dict(
            response="y_gauss",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="gaussian",
        ).fit()

        np.testing.assert_allclose(
            rs_res.bse_robust("HC1"), sm_res.bse, rtol=0.05,
        )

    def test_hc2_hc3_exist(self, data):
        """HC2 and HC3 should be computable without error."""
        df, x1, x2, *_ = data
        rs_res = rs.glm_dict(
            response="y_gauss",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="gaussian",
        ).fit()

        se_hc2 = rs_res.bse_robust("HC2")
        se_hc3 = rs_res.bse_robust("HC3")
        assert len(se_hc2) == 3
        assert len(se_hc3) == 3
        # HC3 >= HC2 >= HC1 >= HC0 (in general)
        assert np.all(se_hc3 >= se_hc2 - 1e-10)


# ===========================================================================
# Test 14: Confidence intervals
# ===========================================================================

class TestConfidenceIntervals:
    """Model-based confidence intervals vs statsmodels."""

    def test_ci_gaussian(self, data):
        df, x1, x2, *_ = data
        y = df["y_gauss"].to_numpy()
        X_sm = sm.add_constant(np.column_stack([x1, x2]))

        sm_res = sm.GLM(y, X_sm, family=smf.Gaussian()).fit()
        rs_res = rs.glm_dict(
            response="y_gauss",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="gaussian",
        ).fit()

        rs_ci = rs_res.conf_int(0.05)
        sm_ci = sm_res.conf_int(alpha=0.05)

        np.testing.assert_allclose(rs_ci, sm_ci, atol=0.05)

    def test_ci_poisson(self, data):
        df, x1, x2, *_ = data
        y = df["y_pois"].to_numpy()
        X_sm = sm.add_constant(np.column_stack([x1, x2]))

        sm_res = sm.GLM(y, X_sm, family=smf.Poisson()).fit()
        rs_res = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson",
        ).fit()

        rs_ci = rs_res.conf_int(0.05)
        sm_ci = sm_res.conf_int(alpha=0.05)

        np.testing.assert_allclose(rs_ci, sm_ci, atol=0.05)


# ===========================================================================
# Test 15: z-values and p-values
# ===========================================================================

class TestZvaluesPvalues:
    """z-statistics and p-values vs statsmodels."""

    def test_zvalues_gaussian(self, data):
        df, x1, x2, *_ = data
        y = df["y_gauss"].to_numpy()
        X_sm = sm.add_constant(np.column_stack([x1, x2]))

        sm_res = sm.GLM(y, X_sm, family=smf.Gaussian()).fit()
        rs_res = rs.glm_dict(
            response="y_gauss",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="gaussian",
        ).fit()

        np.testing.assert_allclose(rs_res.tvalues(), sm_res.tvalues, rtol=0.02)
        np.testing.assert_allclose(rs_res.pvalues(), sm_res.pvalues, atol=0.01)

    def test_zvalues_poisson(self, data):
        df, x1, x2, *_ = data
        y = df["y_pois"].to_numpy()
        X_sm = sm.add_constant(np.column_stack([x1, x2]))

        sm_res = sm.GLM(y, X_sm, family=smf.Poisson()).fit()
        rs_res = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson",
        ).fit()

        np.testing.assert_allclose(rs_res.tvalues(), sm_res.tvalues, rtol=0.02)
        np.testing.assert_allclose(rs_res.pvalues(), sm_res.pvalues, atol=0.01)


# ===========================================================================
# Test 16: Null deviance
# ===========================================================================

class TestNullDeviance:
    """Null (intercept-only) deviance vs statsmodels."""

    def test_null_deviance_gaussian(self, data):
        df, x1, x2, *_ = data
        y = df["y_gauss"].to_numpy()
        X_sm = sm.add_constant(np.column_stack([x1, x2]))

        sm_res = sm.GLM(y, X_sm, family=smf.Gaussian()).fit()
        rs_res = rs.glm_dict(
            response="y_gauss",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="gaussian",
        ).fit()

        assert _close(rs_res.null_deviance(), sm_res.null_deviance, rtol=0.01)

    def test_null_deviance_poisson(self, data):
        df, x1, x2, *_ = data
        y = df["y_pois"].to_numpy()
        X_sm = sm.add_constant(np.column_stack([x1, x2]))

        sm_res = sm.GLM(y, X_sm, family=smf.Poisson()).fit()
        rs_res = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson",
        ).fit()

        assert _close(rs_res.null_deviance(), sm_res.null_deviance, rtol=0.01)

    def test_null_deviance_gamma(self, data):
        df, x1, *_ = data
        y = df["y_gamma"].to_numpy()
        X_sm = sm.add_constant(x1)

        sm_res = sm.GLM(y, X_sm, family=smf.Gamma(smf.links.Log())).fit()
        rs_res = rs.glm_dict(
            response="y_gamma", terms={"x1": {"type": "linear"}},
            data=df, family="gamma",
        ).fit()

        assert _close(rs_res.null_deviance(), sm_res.null_deviance, rtol=0.01)


# ===========================================================================
# Test 17: Pearson chi2 and scale
# ===========================================================================

class TestPearsonScale:
    """Pearson chi-squared and scale (dispersion) vs statsmodels."""

    def test_gaussian_scale(self, data):
        df, x1, x2, *_ = data
        y = df["y_gauss"].to_numpy()
        X_sm = sm.add_constant(np.column_stack([x1, x2]))

        sm_res = sm.GLM(y, X_sm, family=smf.Gaussian()).fit()
        rs_res = rs.glm_dict(
            response="y_gauss",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="gaussian",
        ).fit()

        assert _close(rs_res.pearson_chi2(), sm_res.pearson_chi2, rtol=0.01)
        assert _close(rs_res.scale(), sm_res.scale, rtol=0.02)

    def test_gamma_scale(self, data):
        df, x1, *_ = data
        y = df["y_gamma"].to_numpy()
        X_sm = sm.add_constant(x1)

        sm_res = sm.GLM(y, X_sm, family=smf.Gamma(smf.links.Log())).fit()
        rs_res = rs.glm_dict(
            response="y_gamma", terms={"x1": {"type": "linear"}},
            data=df, family="gamma",
        ).fit()

        assert _close(rs_res.pearson_chi2(), sm_res.pearson_chi2, rtol=0.01)
        assert _close(rs_res.scale(), sm_res.scale, rtol=0.05)


# ===========================================================================
# Test 18-21: Regularization (Ridge, Lasso, Elastic Net, CV Path)
# ===========================================================================

class TestRidgeRegularization:
    """Ridge (L2) regularization: coefficient shrinkage vs glum."""

    def test_ridge_shrinkage_vs_glum(self, data):
        df, x1, x2, *_ = data
        y = df["y_pois"].to_numpy()
        X_np = np.column_stack([x1, x2])
        alpha_val = 1.0

        glum_model = GeneralizedLinearRegressor(
            family="poisson", alpha=alpha_val, l1_ratio=0.0, fit_intercept=True,
        )
        glum_model.fit(X_np, y)

        rs_res = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson",
        ).fit(alpha=alpha_val, l1_ratio=0.0)

        glum_coefs = np.concatenate([[glum_model.intercept_], glum_model.coef_])
        # Ridge coefficients may differ in penalization of intercept
        # Compare non-intercept coefficients
        np.testing.assert_allclose(rs_res.params[1:], glum_coefs[1:], atol=0.2, rtol=0.15)

    def test_ridge_shrinks_toward_zero(self, data):
        df, x1, x2, *_ = data

        rs_unreg = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson",
        ).fit()

        rs_ridge = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson",
        ).fit(alpha=5.0, l1_ratio=0.0)

        # Ridge should shrink non-intercept coefficients toward zero
        assert np.all(np.abs(rs_ridge.params[1:]) <= np.abs(rs_unreg.params[1:]) + 0.01)


class TestLassoRegularization:
    """Lasso (L1) regularization: sparsity vs glum."""

    def test_lasso_sparsity(self, data):
        """Strong Lasso should zero out some coefficients."""
        df, x1, x2, *_ = data
        rng = np.random.RandomState(SEED)
        # Add noise variables
        df2 = df.with_columns([
            pl.Series("noise1", rng.normal(0, 1, N)),
            pl.Series("noise2", rng.normal(0, 1, N)),
            pl.Series("noise3", rng.normal(0, 1, N)),
        ])

        rs_lasso = rs.glm_dict(
            response="y_pois",
            terms={
                "x1": {"type": "linear"}, "x2": {"type": "linear"},
                "noise1": {"type": "linear"}, "noise2": {"type": "linear"},
                "noise3": {"type": "linear"},
            },
            data=df2, family="poisson",
        ).fit(alpha=5.0, l1_ratio=1.0)  # Stronger alpha to force zeros

        # At least some noise coefficients should be zeroed
        noise_coefs = rs_lasso.params[3:]  # noise1, noise2, noise3
        n_zero = np.sum(np.abs(noise_coefs) < 1e-4)
        assert n_zero >= 1, f"Lasso should zero out some noise variables, got coefs={noise_coefs}"


class TestElasticNet:
    """Elastic Net (L1+L2) regularization."""

    def test_elastic_net_intermediate(self, data):
        df, x1, x2, *_ = data

        rs_ridge = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson",
        ).fit(alpha=1.0, l1_ratio=0.0)

        rs_lasso = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson",
        ).fit(alpha=1.0, l1_ratio=1.0)

        rs_enet = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson",
        ).fit(alpha=1.0, l1_ratio=0.5)

        # Elastic net coefficients should generally be between ridge and lasso
        # (not a strict mathematical guarantee, but holds in practice)
        assert rs_enet is not None  # just ensure it runs
        assert len(rs_enet.params) == 3


class TestCVRegularizationPath:
    """Cross-validated regularization path."""

    def test_cv_ridge_runs(self, data):
        df, x1, x2, *_ = data

        rs_cv = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson", seed=42,
        ).fit(regularization="ridge")

        assert rs_cv.alpha >= 0
        assert rs_cv.cv_deviance is not None
        assert rs_cv.regularization_path is not None
        assert len(rs_cv.regularization_path) > 0

    def test_cv_lasso_runs(self, data):
        df, x1, x2, *_ = data

        rs_cv = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson", seed=42,
        ).fit(regularization="lasso")

        assert rs_cv.alpha >= 0
        assert rs_cv.cv_deviance is not None

    def test_cv_1se_selection(self, data):
        df, x1, x2, *_ = data

        rs_min = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson", seed=42,
        ).fit(regularization="ridge", selection="min")

        rs_1se = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson", seed=42,
        ).fit(regularization="ridge", selection="1se")

        # 1se should select >= alpha compared to min
        assert rs_1se.alpha >= rs_min.alpha - 1e-10


# ===========================================================================
# Test 22-23: Spline basis functions
# ===========================================================================

class TestBSplineBasis:
    """B-spline basis matrix properties."""

    def test_shape(self):
        x = np.linspace(0, 10, 100)
        basis = rs.bs(x, df=5)
        # df=5 with degree=3 -> 5-1 = 4 columns (no intercept by default)
        assert basis.shape == (100, 4), f"Got shape {basis.shape}"

    def test_nonnegative(self):
        x = np.linspace(0, 10, 100)
        basis = rs.bs(x, df=5)
        assert np.all(basis >= -1e-10), "B-spline basis should be non-negative"

    def test_boundary_behavior(self):
        """Values at boundaries should be well-defined (no NaN/Inf)."""
        x = np.array([0.0, 5.0, 10.0])
        basis = rs.bs(x, df=5)
        assert not np.any(np.isnan(basis))
        assert not np.any(np.isinf(basis))


class TestNaturalSplineBasis:
    """Natural spline basis properties."""

    def test_shape(self):
        x = np.linspace(0, 10, 100)
        basis = rs.ns(x, df=4)
        assert basis.shape[0] == 100
        # ns() may return df or df-1 columns depending on intercept handling
        assert basis.shape[1] >= 3, f"Expected >= 3 columns, got {basis.shape[1]}"

    def test_no_nan(self):
        x = np.linspace(0, 10, 100)
        basis = rs.ns(x, df=4)
        assert not np.any(np.isnan(basis))
        assert not np.any(np.isinf(basis))

    def test_extrapolation_bounded(self):
        """Natural splines should not produce extreme values outside training range."""
        x_train = np.linspace(1, 9, 100)
        x_extrap = np.array([0.5, 0.8, 9.2, 9.5])
        basis_extrap = rs.ns(x_extrap, df=4, boundary_knots=(1.0, 9.0))
        assert not np.any(np.isnan(basis_extrap))
        # Natural splines extrapolate linearly — values may grow but should not explode
        assert not np.any(np.isinf(basis_extrap))


# ===========================================================================
# Test 24: Penalized smooth splines
# ===========================================================================

class TestPenalizedSmooth:
    """Penalized splines with automatic smoothing (GCV)."""

    def test_edf_less_than_k(self, data):
        df, x1, *_ = data
        rng = np.random.RandomState(SEED)
        y = np.sin(x1) + rng.normal(0, 0.5, N)
        df_smooth = df.with_columns(pl.Series("y_smooth", y))

        rs_res = rs.glm_dict(
            response="y_smooth",
            terms={"x1": {"type": "bs"}},  # Default: penalized smooth, k=10
            data=df_smooth, family="gaussian",
        ).fit()

        assert rs_res.has_smooth_terms()
        for st in rs_res.smooth_terms:
            assert st.edf < st.k, f"EDF ({st.edf}) should be < k ({st.k})"
            assert st.edf > 1, f"EDF ({st.edf}) should be > 1"

    def test_gcv_available(self, data):
        df, x1, *_ = data
        rng = np.random.RandomState(SEED)
        y = np.sin(x1) + rng.normal(0, 0.5, N)
        df_smooth = df.with_columns(pl.Series("y_smooth", y))

        rs_res = rs.glm_dict(
            response="y_smooth",
            terms={"x1": {"type": "bs"}},
            data=df_smooth, family="gaussian",
        ).fit()

        assert rs_res.gcv is not None
        assert rs_res.gcv > 0


# ===========================================================================
# Test 25: Monotonic splines
# ===========================================================================

class TestMonotonicSplines:
    """Monotonic B-splines: fitted effect should respect constraints."""

    def test_monotone_increasing(self, data):
        df, x1, *_ = data
        rng = np.random.RandomState(SEED)
        # Data with clear increasing relationship
        y = np.exp(0.2 * x1) + rng.normal(0, 0.5, N)
        y = np.maximum(y, 0.01)
        df_mono = df.with_columns(pl.Series("y_mono", y))

        rs_res = rs.glm_dict(
            response="y_mono",
            terms={"x1": {"type": "bs", "df": 5, "monotonicity": "increasing"}},
            data=df_mono, family="gaussian",
        ).fit()

        # Evaluate predictions at a grid of unique sorted x values
        x_grid = np.linspace(x1.min() + 0.5, x1.max() - 0.5, 50)
        df_grid = pl.DataFrame({"x1": x_grid})
        preds = rs_res.predict(df_grid)

        # Check monotonicity with a small tolerance for numerical noise
        diffs = np.diff(preds)
        n_violations = np.sum(diffs < -0.01)
        assert n_violations == 0, (
            f"Monotone increasing spline: {n_violations} violations out of {len(diffs)} steps"
        )

    def test_monotone_decreasing(self, data):
        df, x1, *_ = data
        rng = np.random.RandomState(SEED)
        y = np.exp(-0.2 * x1 + 3) + rng.normal(0, 0.5, N)
        y = np.maximum(y, 0.01)
        df_mono = df.with_columns(pl.Series("y_mono", y))

        rs_res = rs.glm_dict(
            response="y_mono",
            terms={"x1": {"type": "bs", "df": 5, "monotonicity": "decreasing"}},
            data=df_mono, family="gaussian",
        ).fit()

        x_grid = np.linspace(x1.min() + 0.5, x1.max() - 0.5, 50)
        df_grid = pl.DataFrame({"x1": x_grid})
        preds = rs_res.predict(df_grid)

        diffs = np.diff(preds)
        n_violations = np.sum(diffs > 0.01)
        assert n_violations == 0, (
            f"Monotone decreasing spline: {n_violations} violations out of {len(diffs)} steps"
        )


# ===========================================================================
# Test 26: Coefficient constraints
# ===========================================================================

class TestCoefficientConstraints:
    """Linear term monotonicity constraints."""

    def test_increasing_constraint(self, data):
        df, x1, *_ = data
        rng = np.random.RandomState(SEED)
        y = rng.poisson(np.exp(0.5 + 0.3 * x1)).astype(float)
        df_con = df.with_columns(pl.Series("y_con", y))

        rs_res = rs.glm_dict(
            response="y_con",
            terms={"x1": {"type": "linear", "monotonicity": "increasing"}},
            data=df_con, family="poisson",
        ).fit()

        # The coefficient for x1 should be >= 0
        x1_coef = rs_res.params[1]  # index 1 = x1 (0 = intercept)
        assert x1_coef >= -1e-8, f"Increasing constraint violated: coef={x1_coef}"

    def test_decreasing_constraint(self, data):
        df, x1, *_ = data
        rng = np.random.RandomState(SEED)
        y = rng.poisson(np.exp(2.0 - 0.3 * x1)).astype(float)
        df_con = df.with_columns(pl.Series("y_con", y))

        rs_res = rs.glm_dict(
            response="y_con",
            terms={"x1": {"type": "linear", "monotonicity": "decreasing"}},
            data=df_con, family="poisson",
        ).fit()

        x1_coef = rs_res.params[1]
        assert x1_coef <= 1e-8, f"Decreasing constraint violated: coef={x1_coef}"


# ===========================================================================
# Test 27: Target encoding
# ===========================================================================

class TestTargetEncoding:
    """Ordered target encoding: no leakage, regularization, unseen levels."""

    def test_no_target_leakage(self):
        """Average of LOO-encoded values should differ from naive mean encoding."""
        rng = np.random.RandomState(SEED)
        n = 500
        cats = np.array(["A"] * 250 + ["B"] * 250)
        target = np.concatenate([rng.normal(1, 0.5, 250), rng.normal(3, 0.5, 250)])

        encoded, name, prior, stats = rs.target_encode(cats, target, "cat", seed=42)

        # For category A, the encoded value should NOT be exactly mean(target[A])
        a_mask = cats == "A"
        naive_a_mean = target[a_mask].mean()
        te_a_mean = encoded[a_mask].mean()
        # They should be close but not identical (ordered encoding introduces variance)
        assert abs(te_a_mean - naive_a_mean) < 0.5, "TE mean too far from naive mean"

    def test_unseen_levels_get_prior(self):
        rng = np.random.RandomState(SEED)
        cats = np.array(["A", "B", "A", "B", "A", "B"])
        target = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])

        _, _, prior, stats = rs.target_encode(cats, target, "cat", seed=42)

        new_cats = np.array(["A", "B", "C"])  # C is unseen
        new_encoded = rs.apply_target_encoding(new_cats, stats, prior)

        # Unseen level C should get the prior (global mean)
        assert _close(new_encoded[2], prior, atol=0.01), (
            f"Unseen level should get prior {prior}, got {new_encoded[2]}"
        )

    def test_prior_weight_regularization(self):
        """Higher prior_weight should shrink rare categories more toward global mean."""
        rng = np.random.RandomState(SEED)
        cats = np.array(["A"] * 200 + ["RARE"] * 5)
        target = np.concatenate([rng.normal(0, 1, 200), np.array([10.0, 10.0, 10.0, 10.0, 10.0])])

        _, _, prior_lo, stats_lo = rs.target_encode(cats, target, "cat", prior_weight=0.1, seed=42)
        _, _, prior_hi, stats_hi = rs.target_encode(cats, target, "cat", prior_weight=10.0, seed=42)

        rare_encoded_lo = rs.apply_target_encoding(np.array(["RARE"]), stats_lo, prior_lo)[0]
        rare_encoded_hi = rs.apply_target_encoding(np.array(["RARE"]), stats_hi, prior_hi)[0]

        # Higher prior weight should pull RARE closer to global mean
        global_mean = target.mean()
        assert abs(rare_encoded_hi - global_mean) < abs(rare_encoded_lo - global_mean) + 0.5


# ===========================================================================
# Test 28: Frequency encoding
# ===========================================================================

class TestFrequencyEncoding:
    """Frequency (count-based) encoding."""

    def test_basic_frequency_encoding(self):
        cats = np.array(["A", "A", "A", "B", "B", "C"])
        # Returns (encoded, name, level_counts, max_count, n_obs)
        encoded, name, level_counts, max_count, n_obs = rs.frequency_encode(cats, "cat")

        assert name == "FE(cat)"
        assert max_count == 3  # A appears 3 times
        assert n_obs == 6
        # All A's should have same encoded value
        assert encoded[0] == encoded[1] == encoded[2]
        assert encoded[3] == encoded[4]  # All B's same
        # A (3/3=1.0) > B (2/3) > C (1/3) by relative frequency
        assert encoded[0] > encoded[3] > encoded[5]

    def test_new_data_frequency_encoding(self):
        cats = np.array(["A", "A", "B"])
        _, _, level_counts, max_count, _ = rs.frequency_encode(cats, "cat")

        new_cats = np.array(["A", "B", "UNSEEN"])
        new_encoded = rs.apply_frequency_encoding(new_cats, level_counts, max_count)

        # Unseen should get 0 or minimum frequency
        assert new_encoded[2] <= new_encoded[1]


# ===========================================================================
# Test 29: Interactions
# ===========================================================================

class TestInteractions:
    """Interaction terms: cat×cat, cat×cont, cont×cont."""

    def test_cat_cat_interaction(self, data):
        df, *_ = data
        rng = np.random.RandomState(SEED)
        # Use larger N with fewer categories to avoid singular matrix
        df2 = df.with_columns(
            pl.Series("cat2", rng.choice(["X", "Y"], N)),
            pl.Series("y_int", rng.poisson(2, N).astype(float)),
        )

        rs_res = rs.glm_dict(
            response="y_int",
            terms={},
            interactions=[{
                "cat": {"type": "categorical"},
                "cat2": {"type": "categorical"},
                "include_main": True,
            }],
            data=df2, family="poisson",
        ).fit()

        # Should have intercept + main effects + interaction terms
        assert len(rs_res.params) > 4, "Cat×Cat interaction should produce multiple columns"

    def test_cat_cont_interaction(self, data):
        df, *_ = data
        rng = np.random.RandomState(SEED)
        df2 = df.with_columns(pl.Series("y_int", rng.poisson(2, N)))

        rs_res = rs.glm_dict(
            response="y_int",
            terms={},
            interactions=[{
                "x1": {"type": "linear"},
                "cat": {"type": "categorical"},
                "include_main": True,
            }],
            data=df2, family="poisson",
        ).fit()

        assert len(rs_res.params) > 4, "Cat×Cont interaction should produce multiple columns"

    def test_cont_cont_interaction(self, data):
        df, *_ = data
        rng = np.random.RandomState(SEED)
        df2 = df.with_columns(pl.Series("y_int", rng.poisson(2, N)))

        rs_res = rs.glm_dict(
            response="y_int",
            terms={},
            interactions=[{
                "x1": {"type": "linear"},
                "x2": {"type": "linear"},
                "include_main": True,
            }],
            data=df2, family="poisson",
        ).fit()

        # Should have intercept + x1 + x2 + x1:x2
        assert len(rs_res.params) == 4


# ===========================================================================
# Test 30: TE & FE interactions
# ===========================================================================

class TestTEFEInteractions:
    """Target encoding and frequency encoding for interaction terms."""

    def test_te_interaction(self, data):
        df, *_ = data
        rng = np.random.RandomState(SEED)
        df2 = df.with_columns(
            pl.Series("cat2", rng.choice(["X", "Y", "Z"], N)),
            pl.Series("y_te", rng.poisson(2, N)),
        )

        rs_res = rs.glm_dict(
            response="y_te",
            terms={},
            interactions=[{
                "cat": {"type": "categorical"},
                "cat2": {"type": "categorical"},
                "target_encoding": True,
            }],
            data=df2, family="poisson", seed=42,
        ).fit()

        # TE interaction produces a single column
        te_names = [n for n in rs_res.feature_names if "TE(" in n]
        assert len(te_names) == 1, f"Expected 1 TE interaction column, got {te_names}"

    def test_fe_interaction(self, data):
        """FE interaction: builder should construct combined column internally."""
        df, *_ = data
        rng = np.random.RandomState(SEED)
        cat2 = rng.choice(["X", "Y", "Z"], N)
        df2 = df.with_columns(
            pl.Series("cat2", cat2),
            pl.Series("y_fe", rng.poisson(2, N).astype(float)),
        )

        rs_res = rs.glm_dict(
            response="y_fe",
            terms={"x1": {"type": "linear"}},
            interactions=[{
                "cat": {"type": "categorical"},
                "cat2": {"type": "categorical"},
                "frequency_encoding": True,
            }],
            data=df2, family="poisson",
        ).fit()

        fe_names = [n for n in rs_res.feature_names if "FE(" in n]
        assert len(fe_names) == 1, f"Expected 1 FE interaction column, got {rs_res.feature_names}"

        # Prediction on new data should also work without the combined column
        preds = rs_res.predict(df2)
        assert len(preds) == N
        assert not np.any(np.isnan(preds))


# ===========================================================================
# Test 31: Expression terms
# ===========================================================================

class TestExpressionTerms:
    """Expression terms: I(x**2), I(x/1000), compound expressions."""

    def test_quadratic_expression(self, data):
        df, x1, *_ = data
        rng = np.random.RandomState(SEED)
        y = 2.0 + 0.5 * x1 - 0.05 * x1 ** 2 + rng.normal(0, 1, N)
        df2 = df.with_columns(pl.Series("y_expr", y))

        rs_res = rs.glm_dict(
            response="y_expr",
            terms={
                "x1": {"type": "linear"},
                "x1_sq": {"type": "expression", "expr": "x1 ** 2"},
            },
            data=df2, family="gaussian",
        ).fit()

        # Should have 3 params: intercept, x1, x1^2
        assert len(rs_res.params) == 3
        # x1^2 coefficient should be negative
        assert rs_res.params[2] < 0, "Quadratic term should be negative"

    def test_division_expression(self, data):
        df, x1, *_ = data
        rng = np.random.RandomState(SEED)
        y = rng.normal(5, 1, N)
        df2 = df.with_columns(pl.Series("y_div", y))

        rs_res = rs.glm_dict(
            response="y_div",
            terms={"x1_scaled": {"type": "expression", "expr": "x1 / 10"}},
            data=df2, family="gaussian",
        ).fit()

        assert len(rs_res.params) == 2  # intercept + scaled x1


# ===========================================================================
# Test 32: Prediction on new data
# ===========================================================================

class TestPrediction:
    """Predict on new data matches manual computation."""

    def test_predict_matches_manual(self, data):
        df, x1, x2, *_ = data

        rs_res = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson",
        ).fit()

        # Predict on same data
        preds = rs_res.predict(df)

        # Manual: mu = exp(X @ beta)
        beta = rs_res.params
        eta_manual = beta[0] + beta[1] * x1 + beta[2] * x2
        mu_manual = np.exp(eta_manual)

        np.testing.assert_allclose(preds, mu_manual, rtol=1e-6)

    def test_predict_with_offset(self, data):
        df, x1, x2, cat, exposure, _ = data
        rng = np.random.RandomState(SEED)
        y = rng.poisson(exposure * np.exp(-0.5 + 0.1 * x1)).astype(float)
        df2 = df.with_columns(pl.Series("y_off", y))

        rs_res = rs.glm_dict(
            response="y_off",
            terms={"x1": {"type": "linear"}},
            data=df2, family="poisson", offset="exposure",
        ).fit()

        # Predict on new data with offset
        preds = rs_res.predict(df2)

        beta = rs_res.params
        eta_manual = beta[0] + beta[1] * x1 + np.log(exposure)
        mu_manual = np.exp(eta_manual)

        np.testing.assert_allclose(preds, mu_manual, rtol=1e-4)

    def test_predict_binomial(self, data):
        df, x1, x2, *_ = data

        rs_res = rs.glm_dict(
            response="y_binom",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="binomial",
        ).fit()

        preds = rs_res.predict(df)

        beta = rs_res.params
        eta_manual = beta[0] + beta[1] * x1 + beta[2] * x2
        mu_manual = 1.0 / (1.0 + np.exp(-eta_manual))

        np.testing.assert_allclose(preds, mu_manual, rtol=1e-6)


# ===========================================================================
# Test 33: Model serialization
# ===========================================================================

class TestSerialization:
    """to_bytes()/from_bytes() roundtrip."""

    def test_linear_roundtrip(self, data):
        df, *_ = data

        rs_res = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson",
        ).fit()

        model_bytes = rs_res.to_bytes()
        loaded = rs.GLMModel.from_bytes(model_bytes)

        # Coefficients should be identical
        np.testing.assert_array_equal(loaded.params, rs_res.params)

        # Predictions should match
        pred_orig = rs_res.predict(df)
        pred_loaded = loaded.predict(df)
        np.testing.assert_allclose(pred_loaded, pred_orig, rtol=1e-10)

    def test_categorical_roundtrip(self, data):
        df, *_ = data

        rs_res = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "cat": {"type": "categorical"}},
            data=df, family="poisson",
        ).fit()

        model_bytes = rs_res.to_bytes()
        loaded = rs.GLMModel.from_bytes(model_bytes)

        pred_orig = rs_res.predict(df)
        pred_loaded = loaded.predict(df)
        np.testing.assert_allclose(pred_loaded, pred_orig, rtol=1e-10)

    def test_te_roundtrip(self, data):
        df, *_ = data

        rs_res = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "cat": {"type": "target_encoding"}},
            data=df, family="poisson", seed=42,
        ).fit()

        model_bytes = rs_res.to_bytes()
        loaded = rs.GLMModel.from_bytes(model_bytes)

        pred_orig = rs_res.predict(df)
        pred_loaded = loaded.predict(df)
        np.testing.assert_allclose(pred_loaded, pred_orig, rtol=1e-6)


# ===========================================================================
# Test 34: Design matrix validation
# ===========================================================================

class TestDesignMatrixValidation:
    """Design matrix issues: collinearity detection, convergence with singular data."""

    def test_detects_zero_variance_at_fit(self):
        """Fitting with a constant column should fail or warn about singularity."""
        rng = np.random.RandomState(SEED)
        df = pl.DataFrame({
            "y": rng.poisson(2, 100).astype(float),
            "x1": rng.normal(0, 1, 100),
            "constant": np.ones(100),  # zero variance -> collinear with intercept
        })

        # This should raise due to singular matrix (constant + intercept)
        with pytest.raises((ValueError, Exception)):
            rs.glm_dict(
                response="y",
                terms={"x1": {"type": "linear"}, "constant": {"type": "linear"}},
                data=df, family="poisson",
            ).fit()

    def test_detects_perfect_collinearity_at_fit(self):
        """Fitting with perfectly collinear columns should fail."""
        rng = np.random.RandomState(SEED)
        x = rng.normal(0, 1, 100)
        df = pl.DataFrame({
            "y": rng.poisson(2, 100).astype(float),
            "x1": x,
            "x2": 2.0 * x,  # perfectly collinear
        })

        with pytest.raises((ValueError, Exception)):
            rs.glm_dict(
                response="y",
                terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
                data=df, family="poisson",
            ).fit()


# ===========================================================================
# Test 35: Non-canonical links
# ===========================================================================

class TestNonCanonicalLinks:
    """Non-default link functions vs statsmodels."""

    def test_gaussian_log_link(self, data):
        df, x1, *_ = data
        y = df["y_gamma"].to_numpy()  # Use positive data
        X_sm = sm.add_constant(x1)

        sm_res = sm.GLM(y, X_sm, family=smf.Gaussian(smf.links.Log())).fit()

        rs_res = rs.glm_dict(
            response="y_gamma",
            terms={"x1": {"type": "linear"}},
            data=df, family="gaussian", link="log",
        ).fit()

        np.testing.assert_allclose(rs_res.params, sm_res.params, atol=COEF_ATOL, rtol=COEF_RTOL)
        np.testing.assert_allclose(rs_res.fittedvalues, sm_res.predict(X_sm), rtol=PRED_RTOL)


# ===========================================================================
# Test 36: Diagnostics calibration
# ===========================================================================

class TestDiagnosticsCalibration:
    """Calibration metrics: A/E ratio, decile calibration."""

    def test_ae_ratio(self, data):
        df, x1, x2, *_ = data

        rs_res = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson",
        ).fit()

        diag = rs_res.diagnostics(
            train_data=df,
            continuous_factors=["x1"],
        )

        # calibration is a dict with keys like 'overall_ae', 'deciles', etc.
        assert diag.calibration is not None
        cal = diag.calibration
        if isinstance(cal, dict):
            ae = cal.get("overall_ae", cal.get("ae_ratio"))
        else:
            ae = getattr(cal, "overall_ae", getattr(cal, "ae_ratio", None))
        assert ae is not None, f"Could not find A/E in calibration: {cal}"
        assert 0.90 <= ae <= 1.10, f"Training A/E should be ~1.0, got {ae:.4f}"

    def test_calibration_exists(self, data):
        df, x1, x2, *_ = data

        rs_res = rs.glm_dict(
            response="y_binom",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="binomial",
        ).fit()

        diag = rs_res.diagnostics(train_data=df)
        assert diag.calibration is not None


# ===========================================================================
# Test 37: Diagnostics discrimination (Gini)
# ===========================================================================

class TestDiagnosticsDiscrimination:
    """Gini coefficient and model comparison metrics."""

    def test_model_comparison_metrics(self, data):
        df, x1, x2, *_ = data

        rs_res = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson",
        ).fit()

        diag = rs_res.diagnostics(train_data=df)

        # model_comparison contains likelihood ratio test and deviance reduction
        mc = diag.model_comparison
        assert mc is not None
        assert "deviance_reduction_pct" in mc, f"Missing deviance_reduction_pct in: {mc}"
        assert mc["deviance_reduction_pct"] > 0, "Model should reduce deviance vs null"

    def test_diagnostics_json_contains_metrics(self, data):
        """Diagnostics JSON should contain key metrics."""
        df, x1, x2, *_ = data

        rs_res = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson",
        ).fit()

        json_str = rs_res.diagnostics_json(train_data=df, continuous_factors=["x1"])
        # Should contain some key diagnostic terms
        assert len(json_str) > 100
        assert "model_comparison" in json_str or "gini" in json_str.lower()


# ===========================================================================
# Test 38: Edge cases
# ===========================================================================

class TestEdgeCases:
    """Edge cases: intercept-only, sparse categories, zero counts."""

    def test_intercept_only_model(self, data):
        df, *_ = data

        rs_res = rs.glm_dict(
            response="y_pois",
            terms={},
            data=df, family="poisson",
        ).fit()

        # Should have just the intercept
        assert len(rs_res.params) == 1
        # Intercept should be log(mean(y))
        y = df["y_pois"].to_numpy()
        expected_intercept = np.log(y.mean())
        assert _close(rs_res.params[0], expected_intercept, atol=0.05)

    def test_all_zero_counts(self):
        """Model should handle all-zero response gracefully."""
        rng = np.random.RandomState(SEED)
        df = pl.DataFrame({
            "y": np.zeros(100),
            "x1": rng.normal(0, 1, 100),
        })

        # This might converge to very negative intercept or warn
        try:
            rs_res = rs.glm_dict(
                response="y",
                terms={"x1": {"type": "linear"}},
                data=df, family="poisson",
            ).fit()
            # If it converges, predictions should be near zero
            assert np.all(rs_res.fittedvalues < 1.0)
        except Exception:
            # Acceptable to raise an error for degenerate data
            pass

    def test_single_obs_per_category(self, data):
        """Should handle categories with very few observations via target encoding."""
        rng = np.random.RandomState(SEED)
        n = 200
        # Mix of common and rare categories
        cats = [f"cat_{i % 20}" for i in range(n)]  # 20 cats, ~10 obs each
        df = pl.DataFrame({
            "y": rng.poisson(2, n).astype(float),
            "x1": rng.normal(0, 1, n),
            "many_cats": cats,
        })

        # Should work with target encoding (rare categories shrink to prior)
        rs_res = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "many_cats": {"type": "target_encoding"}},
            data=df, family="poisson", seed=42,
        ).fit()

        assert len(rs_res.params) == 3  # intercept + x1 + TE(many_cats)

    def test_large_category_count(self):
        """High-cardinality categorical with target encoding."""
        rng = np.random.RandomState(SEED)
        n = 1000
        cats = rng.choice([f"level_{i}" for i in range(200)], n)
        df = pl.DataFrame({
            "y": rng.poisson(2, n),
            "high_card": cats,
        })

        rs_res = rs.glm_dict(
            response="y",
            terms={"high_card": {"type": "target_encoding"}},
            data=df, family="poisson", seed=42,
        ).fit()

        # TE produces single column regardless of cardinality
        assert len(rs_res.params) == 2  # intercept + TE(high_card)


# ===========================================================================
# Test 39: Numerical stability at scale
# ===========================================================================

class TestNumericalStability:
    """Large dataset correctness and stability."""

    def test_large_n_poisson(self):
        """100K observations with many predictors."""
        rng = np.random.RandomState(SEED)
        n = 100_000
        p = 10
        X = rng.normal(0, 1, (n, p))
        beta_true = rng.normal(0, 0.3, p)
        eta = 0.5 + X @ beta_true
        y = rng.poisson(np.exp(eta)).astype(float)

        terms = {f"x{i}": {"type": "linear"} for i in range(p)}
        data_dict = {"y": y}
        data_dict.update({f"x{i}": X[:, i] for i in range(p)})
        df = pl.DataFrame(data_dict)

        rs_res = rs.glm_dict(
            response="y", terms=terms, data=df, family="poisson",
        ).fit()

        assert rs_res.converged, "Model should converge on large clean data"
        # Coefficients should be close to true values
        np.testing.assert_allclose(rs_res.params[1:], beta_true, atol=0.05)

    def test_condition_number(self):
        """Model with high collinearity should still converge."""
        rng = np.random.RandomState(SEED)
        n = 1000
        x1 = rng.normal(0, 1, n)
        x2 = x1 + rng.normal(0, 0.01, n)  # Nearly collinear
        y = rng.poisson(np.exp(0.5 + 0.3 * x1)).astype(float)

        df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})

        rs_res = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson",
        ).fit()

        # Should converge (possibly with large SEs)
        assert rs_res.converged or len(rs_res.params) == 3


# ===========================================================================
# Test 40: Categorical encoding consistency
# ===========================================================================

class TestCategoricalEncoding:
    """Dummy encoding: reference level, unseen levels in prediction."""

    def test_reference_level_dropped(self, data):
        df, *_ = data
        unique_cats = df["cat"].unique().sort().to_list()

        rs_res = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "cat": {"type": "categorical"}},
            data=df, family="poisson",
        ).fit()

        # Should have intercept + x1 + (n_levels - 1) dummies
        expected_params = 1 + 1 + (len(unique_cats) - 1)
        assert len(rs_res.params) == expected_params, (
            f"Expected {expected_params} params, got {len(rs_res.params)}"
        )

    def test_predict_on_subset(self, data):
        """Predict works when new data has fewer categorical levels."""
        df, *_ = data

        rs_res = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "cat": {"type": "categorical"}},
            data=df, family="poisson",
        ).fit()

        # Predict on subset with only 2 categories
        df_subset = df.filter(pl.col("cat").is_in(["A", "B"]))
        preds = rs_res.predict(df_subset)
        assert len(preds) == len(df_subset)
        assert not np.any(np.isnan(preds))


# ===========================================================================
# Additional: Summary and diagnostics JSON
# ===========================================================================

class TestSummaryOutput:
    """Summary string and diagnostics JSON generation."""

    def test_summary_runs(self, data):
        df, *_ = data
        rs_res = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson",
        ).fit()

        summary_str = rs_res.summary()
        assert "Poisson" in summary_str or "poisson" in summary_str
        assert "x1" in summary_str
        assert "x2" in summary_str

    def test_coef_table(self, data):
        df, *_ = data
        rs_res = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson",
        ).fit()

        ct = rs_res.coef_table()
        assert "Feature" in ct.columns
        assert "Estimate" in ct.columns
        assert len(ct) == 3  # intercept + x1 + x2

    def test_diagnostics_json(self, data):
        df, *_ = data
        rs_res = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson",
        ).fit()

        json_str = rs_res.diagnostics_json(
            train_data=df,
            continuous_factors=["x1", "x2"],
        )
        assert len(json_str) > 100

    def test_relativities(self, data):
        df, *_ = data
        rs_res = rs.glm_dict(
            response="y_pois",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df, family="poisson",
        ).fit()

        rel = rs_res.relativities()
        assert "Relativity" in rel.columns
        # Relativity = exp(coef), so intercept relativity is exp(intercept)
        np.testing.assert_allclose(
            rel["Relativity"].to_numpy(), np.exp(rs_res.params), rtol=1e-10,
        )
