"""
Tests for working_response_weights
==================================

`rs.working_response_weights(y, eta, family, ...)` exposes the IRLS working
response z and combined working weight w for an arbitrary linear predictor η
— without requiring a fitted GLM. The canonical use case is link-scale
boosting loops (e.g. destyler), where each iteration needs to re-form the
local quadratic approximation around the running η.

Mathematical contract
---------------------
    μ = g⁻¹(η + offset)
    z = η + (y − μ) · g'(μ)                     # NOTE: η is *excluding* offset
    w = prior_weight × IRLS_weight
        where IRLS_weight = 1 / (V(μ) · g'(μ)²)         [Fisher]
              or          = μ^(2 − p)                    [true Hessian, Tweedie+log, 1<p<2]

Per-family weight formulas (combined with prior_weight=1 below):
    Gaussian + identity:    w = 1
    Poisson  + log:         w = μ
    Gamma    + log:         w = 1                        (Fisher; matches statsmodels)
    Tweedie  + log:         w = μ^(2−p)                  (true Hessian)
    Binomial + logit:       w = μ(1 − μ)
    NegBin   + log:         w = μθ / (θ + μ)
    Quasi*   (matches base) → identical to Poisson/Binomial

Run with: pytest tests/python/test_working_response_weights.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import rustystats as rs
from rustystats.exceptions import ValidationError

# =============================================================================
# Per-family analytic checks at the canonical link
# =============================================================================


class TestWorkingResponseWeightsGaussianIdentity:
    """Gaussian + identity link.

    With g(μ) = μ, g'(μ) = 1, V(μ) = 1, and no offset:
        μ = η  →  z = η + (y − η)·1 = y
        w = 1 / (1 · 1²) = 1
    """

    def test_z_equals_y(self):
        y = np.array([1.0, 2.0, 3.0])
        eta = np.array([1.5, 2.0, 2.5])
        z, _ = rs.working_response_weights(y, eta, family="gaussian", link="identity")
        np.testing.assert_array_almost_equal(z, y)

    def test_w_is_one_everywhere(self):
        y = np.array([1.0, 2.0, 3.0])
        eta = np.array([1.5, 2.0, 2.5])
        _, w = rs.working_response_weights(y, eta, family="gaussian", link="identity")
        np.testing.assert_array_almost_equal(w, np.ones(3))


class TestWorkingResponseWeightsPoissonLog:
    """Poisson + log link.

    μ = exp(η + offset)
    z = η + (y − μ)/μ
    w = μ                                                  (Fisher)
    """

    def test_w_equals_mu(self):
        y = np.array([1.0, 2.0, 3.0])
        eta = np.log(np.array([1.0, 2.0, 3.0]))
        _, w = rs.working_response_weights(y, eta, family="poisson")
        np.testing.assert_array_almost_equal(w, np.exp(eta))

    def test_z_analytic(self):
        y = np.array([1.0, 4.0, 3.0])
        eta = np.log(np.array([2.0, 2.0, 5.0]))
        mu = np.exp(eta)
        z, _ = rs.working_response_weights(y, eta, family="poisson")
        expected = eta + (y - mu) / mu
        np.testing.assert_array_almost_equal(z, expected)

    def test_perfect_fit_gives_zero_correction(self):
        # When y == μ exactly, z collapses to η (no correction term).
        y = np.array([1.0, 2.0, 3.0])
        eta = np.log(y)  # μ = y exactly
        z, _ = rs.working_response_weights(y, eta, family="poisson")
        np.testing.assert_array_almost_equal(z, eta)

    def test_offset_shifts_mu_not_z_directly(self):
        # η stays separate from offset; μ = exp(η + offset).
        # Compare same η with vs without offset: z = η + (y − μ)/μ should differ
        # only through μ. With offset=log(2), μ doubles, so the correction term
        # (y − μ)/μ changes accordingly while the leading η stays put.
        y = np.array([2.0, 4.0])
        eta = np.array([0.0, 0.0])  # log(1) = 0 → μ = 1 without offset
        offset = np.log(np.array([2.0, 2.0]))  # μ becomes 2

        z, _ = rs.working_response_weights(y, eta, family="poisson", offset=offset)

        mu_with_offset = np.array([2.0, 2.0])
        expected_z = eta + (y - mu_with_offset) / mu_with_offset
        np.testing.assert_array_almost_equal(z, expected_z)


class TestWorkingResponseWeightsGammaLog:
    """Gamma + log link, Fisher branch.

    g'(μ) = 1/μ, V(μ) = μ² → w = 1/(μ² · 1/μ²) = 1
    Matches statsmodels (true Hessian is disabled for Gamma+log).
    """

    def test_fisher_w_is_one(self):
        y = np.array([0.5, 2.0, 5.0])
        eta = np.log(np.array([1.0, 2.0, 4.0]))
        _, w = rs.working_response_weights(y, eta, family="gamma")
        np.testing.assert_array_almost_equal(w, np.ones(3))

    def test_z_analytic(self):
        y = np.array([0.5, 2.0, 5.0])
        eta = np.log(np.array([1.0, 2.0, 4.0]))
        mu = np.exp(eta)
        z, _ = rs.working_response_weights(y, eta, family="gamma")
        expected = eta + (y - mu) / mu
        np.testing.assert_array_almost_equal(z, expected)


class TestWorkingResponseWeightsTweedieLog:
    """Tweedie + log link, true Hessian branch (1 < p < 2).

    w = μ^(2 − p)
    """

    def test_true_hessian_w_is_mu_to_2_minus_p(self):
        y = np.array([0.0, 2.0, 4.0])
        eta = np.log(np.array([1.0, 2.0, 5.0]))
        _, w = rs.working_response_weights(y, eta, family="tweedie", var_power=1.5)
        mu = np.exp(eta)
        np.testing.assert_array_almost_equal(w, mu**0.5)

    def test_var_power_kwarg_changes_w(self):
        y = np.array([0.0, 2.0, 4.0])
        eta = np.log(np.array([1.0, 2.0, 5.0]))
        _, w_15 = rs.working_response_weights(y, eta, family="tweedie", var_power=1.5)
        _, w_17 = rs.working_response_weights(y, eta, family="tweedie", var_power=1.7)
        mu = np.exp(eta)
        np.testing.assert_array_almost_equal(w_15, mu**0.5)
        np.testing.assert_array_almost_equal(w_17, mu**0.3)

    def test_embedded_var_power_in_family_string(self):
        # family="tweedie(p=1.7)" and family="tweedie", var_power=1.7 must agree.
        y = np.array([0.0, 2.0, 4.0])
        eta = np.log(np.array([1.0, 2.0, 5.0]))
        z_embedded, w_embedded = rs.working_response_weights(y, eta, family="tweedie(p=1.7)")
        z_kwarg, w_kwarg = rs.working_response_weights(y, eta, family="tweedie", var_power=1.7)
        np.testing.assert_array_almost_equal(z_embedded, z_kwarg)
        np.testing.assert_array_almost_equal(w_embedded, w_kwarg)


class TestWorkingResponseWeightsBinomialLogit:
    """Binomial + logit link.

    g(μ) = log(μ/(1−μ)), g'(μ) = 1/(μ(1−μ)), V(μ) = μ(1−μ)
    w = μ(1 − μ)
    z = logit(μ) + (y − μ)/(μ(1 − μ))
    """

    def test_w_is_mu_one_minus_mu(self):
        y = np.array([0.0, 1.0, 1.0])
        mu = np.array([0.3, 0.7, 0.5])
        eta = np.log(mu / (1.0 - mu))  # logit
        _, w = rs.working_response_weights(y, eta, family="binomial")
        np.testing.assert_array_almost_equal(w, mu * (1.0 - mu))

    def test_z_analytic(self):
        y = np.array([0.0, 1.0, 1.0])
        mu = np.array([0.3, 0.7, 0.5])
        eta = np.log(mu / (1.0 - mu))
        z, _ = rs.working_response_weights(y, eta, family="binomial")
        expected = eta + (y - mu) / (mu * (1.0 - mu))
        np.testing.assert_array_almost_equal(z, expected)


class TestWorkingResponseWeightsNegBinomialLog:
    """Negative Binomial + log link.

    V(μ) = μ + μ²/θ, g'(μ) = 1/μ → w = μ²/(μ + μ²/θ) = μθ/(θ + μ)
    """

    def test_w_matches_theta_formula(self):
        theta = 2.0
        y = np.array([0.0, 1.0, 3.0])
        eta = np.log(np.array([0.5, 2.0, 4.0]))
        _, w = rs.working_response_weights(y, eta, family="negbinomial", theta=theta)
        mu = np.exp(eta)
        np.testing.assert_array_almost_equal(w, mu * theta / (theta + mu))

    def test_theta_kwarg_changes_w(self):
        y = np.array([0.0, 1.0, 3.0])
        eta = np.log(np.array([0.5, 2.0, 4.0]))
        mu = np.exp(eta)
        _, w_low = rs.working_response_weights(y, eta, family="negbinomial", theta=0.5)
        _, w_high = rs.working_response_weights(y, eta, family="negbinomial", theta=10.0)
        np.testing.assert_array_almost_equal(w_low, mu * 0.5 / (0.5 + mu))
        np.testing.assert_array_almost_equal(w_high, mu * 10.0 / (10.0 + mu))

    def test_embedded_theta_in_family_string(self):
        y = np.array([0.0, 1.0, 3.0])
        eta = np.log(np.array([0.5, 2.0, 4.0]))
        z_embedded, w_embedded = rs.working_response_weights(
            y, eta, family="negativebinomial(theta=2.5)"
        )
        z_kwarg, w_kwarg = rs.working_response_weights(y, eta, family="negbinomial", theta=2.5)
        np.testing.assert_array_almost_equal(z_embedded, z_kwarg)
        np.testing.assert_array_almost_equal(w_embedded, w_kwarg)


class TestWorkingResponseWeightsQuasiFamilies:
    """Quasi families have the same V(μ) as their base → identical z, w."""

    def test_quasipoisson_matches_poisson(self):
        y = np.array([0.0, 1.0, 3.0])
        eta = np.log(np.array([0.5, 2.0, 4.0]))
        z_p, w_p = rs.working_response_weights(y, eta, family="poisson")
        z_q, w_q = rs.working_response_weights(y, eta, family="quasipoisson")
        np.testing.assert_array_almost_equal(z_p, z_q)
        np.testing.assert_array_almost_equal(w_p, w_q)

    def test_quasibinomial_matches_binomial(self):
        y = np.array([0.0, 1.0, 1.0])
        mu = np.array([0.2, 0.6, 0.8])
        eta = np.log(mu / (1.0 - mu))
        z_b, w_b = rs.working_response_weights(y, eta, family="binomial")
        z_q, w_q = rs.working_response_weights(y, eta, family="quasibinomial")
        np.testing.assert_array_almost_equal(z_b, z_q)
        np.testing.assert_array_almost_equal(w_b, w_q)


# =============================================================================
# Link resolution
# =============================================================================


class TestWorkingResponseWeightsLinkResolution:
    """Default link resolution and non-canonical link overrides."""

    def test_omitting_link_uses_canonical(self):
        # Poisson's canonical link is log.
        y = np.array([1.0, 2.0, 3.0])
        eta = np.log(np.array([1.0, 2.0, 3.0]))
        z_default, w_default = rs.working_response_weights(y, eta, family="poisson")
        z_log, w_log = rs.working_response_weights(y, eta, family="poisson", link="log")
        np.testing.assert_array_almost_equal(z_default, z_log)
        np.testing.assert_array_almost_equal(w_default, w_log)

    def test_link_default_string_resolves_to_canonical(self):
        y = np.array([1.0, 2.0, 3.0])
        eta = np.log(np.array([1.0, 2.0, 3.0]))
        z_str, w_str = rs.working_response_weights(y, eta, family="poisson", link="default")
        z_log, w_log = rs.working_response_weights(y, eta, family="poisson", link="log")
        np.testing.assert_array_almost_equal(z_str, z_log)
        np.testing.assert_array_almost_equal(w_str, w_log)

    def test_non_canonical_link_works(self):
        # Gaussian with log link — non-canonical but mathematically valid.
        # g'(μ) = 1/μ, V(μ) = 1 → w = 1/(1 · 1/μ²) = μ²
        y = np.array([1.0, 2.0, 3.0])
        eta = np.log(np.array([1.0, 2.0, 3.0]))
        mu = np.exp(eta)
        z, w = rs.working_response_weights(y, eta, family="gaussian", link="log")
        np.testing.assert_array_almost_equal(w, mu**2)
        expected_z = eta + (y - mu) / mu
        np.testing.assert_array_almost_equal(z, expected_z)


# =============================================================================
# Prior weights and offset
# =============================================================================


class TestWorkingResponseWeightsPriorWeights:
    def test_combined_weight_is_prior_times_irls(self):
        # Poisson + log: irls_w = μ → combined = prior × μ
        y = np.array([1.0, 2.0, 3.0])
        eta = np.log(np.array([1.0, 2.0, 3.0]))
        prior = np.array([2.0, 0.5, 3.0])
        _, w = rs.working_response_weights(y, eta, family="poisson", weights=prior)
        mu = np.exp(eta)
        np.testing.assert_array_almost_equal(w, prior * mu)

    def test_no_weights_is_unit_prior(self):
        # Default behaviour: prior weight = 1 → combined = irls.
        y = np.array([1.0, 2.0, 3.0])
        eta = np.log(np.array([1.0, 2.0, 3.0]))
        _, w_default = rs.working_response_weights(y, eta, family="poisson")
        _, w_explicit = rs.working_response_weights(y, eta, family="poisson", weights=np.ones(3))
        np.testing.assert_array_almost_equal(w_default, w_explicit)


class TestWorkingResponseWeightsOffset:
    def test_zero_offset_matches_no_offset(self):
        y = np.array([1.0, 2.0, 3.0])
        eta = np.log(np.array([1.0, 2.0, 3.0]))
        z_none, w_none = rs.working_response_weights(y, eta, family="poisson")
        z_zero, w_zero = rs.working_response_weights(y, eta, family="poisson", offset=np.zeros(3))
        np.testing.assert_array_almost_equal(z_none, z_zero)
        np.testing.assert_array_almost_equal(w_none, w_zero)

    def test_offset_enters_only_through_mu(self):
        # Public contract: z = η + (y − μ)·g'(μ) where the leading η is η-without-offset.
        # Doubling the offset (with log link) doubles μ but leaves the leading η untouched.
        y = np.array([2.0, 4.0])
        eta = np.array([0.0, 0.0])
        offset = np.log(np.array([2.0, 2.0]))
        z, w = rs.working_response_weights(y, eta, family="poisson", offset=offset)

        mu = np.exp(eta + offset)  # = [2, 2]
        expected_z = eta + (y - mu) / mu
        expected_w = mu  # Poisson + log Fisher: w = μ
        np.testing.assert_array_almost_equal(z, expected_z)
        np.testing.assert_array_almost_equal(w, expected_w)


# =============================================================================
# Return types
# =============================================================================


class TestWorkingResponseWeightsReturnTypes:
    def test_returns_two_float64_arrays_of_length_n(self):
        y = np.array([1.0, 2.0, 3.0])
        eta = np.log(np.array([1.0, 2.0, 3.0]))
        out = rs.working_response_weights(y, eta, family="poisson")
        assert isinstance(out, tuple) and len(out) == 2
        z, w = out
        assert z.dtype == np.float64
        assert w.dtype == np.float64
        assert z.shape == (3,)
        assert w.shape == (3,)

    def test_accepts_python_lists(self):
        # Numpy-array-like inputs must be coerced cleanly.
        z, w = rs.working_response_weights([1.0, 2.0, 3.0], [0.0, 0.693, 1.099], family="poisson")
        assert z.shape == (3,)
        assert w.shape == (3,)


# =============================================================================
# Validation / error paths
# =============================================================================


class TestWorkingResponseWeightsValidation:
    def test_constant_response_allowed_for_per_row_helper(self):
        # Unlike GLM fitting, the helper only evaluates per-row IRLS formulas.
        # Constant responses, including all-zero Poisson rows, are valid inputs.
        y = np.zeros(3)
        eta = np.zeros(3)

        z, w = rs.working_response_weights(y, eta, family="poisson")

        np.testing.assert_array_almost_equal(z, np.full(3, -1.0))
        np.testing.assert_array_almost_equal(w, np.ones(3))

    def test_dimension_mismatch_eta(self):
        with pytest.raises(ValidationError, match=r"eta.*length"):
            rs.working_response_weights(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.0, 0.0]),  # wrong length
                family="poisson",
            )

    def test_dimension_mismatch_offset(self):
        with pytest.raises(ValidationError, match=r"offset.*does not match"):
            rs.working_response_weights(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.0, 0.0, 0.0]),
                family="poisson",
                offset=np.array([0.0, 0.0]),
            )

    def test_dimension_mismatch_weights(self):
        with pytest.raises(ValidationError, match=r"weights.*does not match"):
            rs.working_response_weights(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.0, 0.0, 0.0]),
                family="poisson",
                weights=np.array([1.0, 1.0]),
            )

    def test_negative_weights_rejected(self):
        with pytest.raises(ValidationError, match=r"non-negative|negative"):
            rs.working_response_weights(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.0, 0.0, 0.0]),
                family="poisson",
                weights=np.array([1.0, -1.0, 1.0]),
            )

    def test_unknown_family_rejected(self):
        with pytest.raises((ValueError, ValidationError)):
            rs.working_response_weights(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.0, 0.0, 0.0]),
                family="not_a_family",
            )

    @pytest.mark.parametrize(
        "family",
        [
            "tweedie_bad",
            "tweedie nonsense",
            "tweediep=1.7",
            "tweedie(p=1.7)junk",
            "tweedie(theta=1.7)",
            "negativebinomial_bad",
            "negativebinomial(theta=2.0)junk",
            "gaussian(p=1.0)",
        ],
    )
    def test_malformed_family_strings_rejected(self, family):
        with pytest.raises((ValueError, ValidationError)):
            rs.working_response_weights(
                np.array([0.0, 1.0, 2.0]),
                np.array([0.0, 0.0, 0.0]),
                family=family,
            )

    def test_unknown_link_rejected(self):
        with pytest.raises((ValueError, ValidationError)):
            rs.working_response_weights(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.0, 0.0, 0.0]),
                family="poisson",
                link="probit",
            )

    def test_tweedie_bad_var_power_rejected(self):
        # Tweedie var_power outside (0, ∞) or in disallowed regions should fail.
        with pytest.raises((ValueError, ValidationError)):
            rs.working_response_weights(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.0, 0.0, 0.0]),
                family="tweedie",
                var_power=0.5,
            )

    def test_negbinomial_bad_theta_rejected(self):
        with pytest.raises((ValueError, ValidationError)):
            rs.working_response_weights(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.0, 0.0, 0.0]),
                family="negbinomial",
                theta=0.0,
            )

    def test_gamma_zero_y_rejected(self):
        with pytest.raises(ValidationError, match=r"[Gg]amma"):
            rs.working_response_weights(
                np.array([1.0, 0.0, 3.0]),  # zero invalid for Gamma
                np.array([0.0, 0.0, 0.0]),
                family="gamma",
            )

    def test_eta_with_nan_rejected(self):
        with pytest.raises(ValidationError, match=r"NaN|nan"):
            rs.working_response_weights(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.0, np.nan, 0.0]),
                family="poisson",
            )

    def test_y_with_nan_rejected(self):
        with pytest.raises(ValidationError, match=r"NaN|nan"):
            rs.working_response_weights(
                np.array([1.0, np.nan, 3.0]),
                np.array([0.0, 0.0, 0.0]),
                family="poisson",
            )


# =============================================================================
# Solver round-trip
# =============================================================================


def _wls_step(X: np.ndarray, z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Solve β = argmin Σ wᵢ (zᵢ − Xᵢβ)² via numpy lstsq on √w-scaled inputs."""
    sqrt_w = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sqrt_w[:, None], z * sqrt_w, rcond=None)
    return beta


class TestWorkingResponseWeightsAgreesWithSolver:
    """At a fitted model's converged η, the (z, w) returned by
    `working_response_weights` must equal the (z, w) the IRLS solver
    would compute on its next iteration. The cleanest way to verify
    this is to perform one more weighted least-squares step on (z, w)
    and check that it reproduces the converged β to within IRLS tolerance.

    This is the strongest correctness signal — it catches drift between
    the helper and the solver that per-row analytic tests would miss.
    """

    @staticmethod
    def _fit(y, X, family, link, offset=None):
        # Use the raw Rust binding for a self-contained test that doesn't
        # depend on the dict-API / Polars layers. Mirrors how formula.py
        # invokes the solver.
        from rustystats._rustystats import fit_glm_py

        return fit_glm_py(
            y,
            X,
            family,
            link,
            offset=offset,
            tol=1e-12,
            max_iter=200,
        )

    def test_poisson_log(self):
        rng = np.random.default_rng(42)
        n = 500
        X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
        beta_true = np.array([0.5, -0.3, 0.4])
        y = rng.poisson(np.exp(X @ beta_true)).astype(np.float64)

        result = self._fit(y, X, "poisson", "log")
        beta_fit = np.asarray(result.coefficients)
        eta = X @ beta_fit  # η excluding offset (no offset here)

        z, w = rs.working_response_weights(y, eta, family="poisson")
        beta_new = _wls_step(X, z, w)

        # Tolerance is generous because the WLS step's lstsq amplifies the
        # IRLS convergence band (~1e-12) by the condition number of XᵀWX.
        # The point is consistency with the solver, not bit-exactness.
        np.testing.assert_allclose(beta_new, beta_fit, atol=1e-6, rtol=1e-6)

    def test_gamma_log(self):
        rng = np.random.default_rng(7)
        n = 500
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        beta_true = np.array([0.2, 0.3])
        mu_true = np.exp(X @ beta_true)
        y = rng.gamma(shape=2.0, scale=mu_true / 2.0)  # E[y]=mu_true

        result = self._fit(y, X, "gamma", "log")
        beta_fit = np.asarray(result.coefficients)
        eta = X @ beta_fit

        z, w = rs.working_response_weights(y, eta, family="gamma")
        beta_new = _wls_step(X, z, w)

        # Tolerance is generous because the WLS step's lstsq amplifies the
        # IRLS convergence band (~1e-12) by the condition number of XᵀWX.
        # The point is consistency with the solver, not bit-exactness.
        np.testing.assert_allclose(beta_new, beta_fit, atol=1e-6, rtol=1e-6)

    def test_poisson_log_with_offset(self):
        # With offset, the helper expects η-without-offset; the WLS step
        # then recovers β from z = X β + correction.
        rng = np.random.default_rng(123)
        n = 500
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        beta_true = np.array([0.1, -0.2])
        offset = np.log(rng.uniform(0.5, 2.0, size=n))
        mu_true = np.exp(X @ beta_true + offset)
        y = rng.poisson(mu_true).astype(np.float64)

        result = self._fit(y, X, "poisson", "log", offset=offset)
        beta_fit = np.asarray(result.coefficients)
        eta = X @ beta_fit  # excludes offset by construction

        z, w = rs.working_response_weights(y, eta, family="poisson", offset=offset)
        beta_new = _wls_step(X, z, w)

        # Tolerance is generous because the WLS step's lstsq amplifies the
        # IRLS convergence band (~1e-12) by the condition number of XᵀWX.
        # The point is consistency with the solver, not bit-exactness.
        np.testing.assert_allclose(beta_new, beta_fit, atol=1e-6, rtol=1e-6)

    def test_binomial_logit(self):
        rng = np.random.default_rng(99)
        n = 800
        X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
        beta_true = np.array([0.1, 0.8, -0.5])
        p_true = 1.0 / (1.0 + np.exp(-X @ beta_true))
        y = rng.binomial(1, p_true).astype(np.float64)

        result = self._fit(y, X, "binomial", "logit")
        beta_fit = np.asarray(result.coefficients)
        eta = X @ beta_fit

        z, w = rs.working_response_weights(y, eta, family="binomial")
        beta_new = _wls_step(X, z, w)

        # Tolerance is generous because the WLS step's lstsq amplifies the
        # IRLS convergence band (~1e-12) by the condition number of XᵀWX.
        # The point is consistency with the solver, not bit-exactness.
        np.testing.assert_allclose(beta_new, beta_fit, atol=1e-6, rtol=1e-6)


class TestEmbeddedTweediePower:
    """Regression: a Tweedie power embedded in the family string (e.g.
    ``tweedie(p=1.8)``) must be honoured, not silently overridden by the default
    ``var_power=1.5``."""

    def test_embedded_power_changes_weights(self):
        eta = np.log(np.array([2.0, 3.0, 4.0]))  # mu != 1 so mu**p depends on p
        y = np.array([1.0, 2.0, 3.0])
        _, w_low = rs.working_response_weights(y, eta, family="tweedie(p=1.2)")
        _, w_high = rs.working_response_weights(y, eta, family="tweedie(p=1.8)")
        assert not np.allclose(w_low, w_high)
        # Embedded p matches the explicit var_power form.
        _, w_explicit = rs.working_response_weights(y, eta, family="tweedie", var_power=1.8)
        np.testing.assert_allclose(w_high, w_explicit, rtol=1e-12, atol=1e-12)

    def test_embedded_power_conflicting_var_power_raises(self):
        with pytest.raises(ValidationError, match="embeds"):
            rs.working_response_weights(
                np.array([1.0]), np.zeros(1), family="tweedie(p=1.2)", var_power=1.8
            )
