"""
Tests for Link Functions
========================

These tests verify that link function implementations are correct.
We test both mathematical properties and numerical stability.

Run with: pytest tests/python/test_links.py -v
"""

import warnings

import numpy as np
import pytest

# Import our library
import rustystats as rs


class TestIdentityLink:
    """Tests for Identity link function."""

    def test_link_is_identity(self):
        """Identity link should return input unchanged."""
        link = rs.links.Identity()
        mu = np.array([1.0, -5.0, 0.0, 100.0])

        eta = link.link(mu)

        np.testing.assert_array_almost_equal(eta, mu)

    def test_inverse_is_identity(self):
        """Identity inverse should return input unchanged."""
        link = rs.links.Identity()
        eta = np.array([-10.0, 0.0, 10.0])

        mu = link.inverse(eta)

        np.testing.assert_array_almost_equal(mu, eta)

    def test_derivative_is_one(self):
        """Identity derivative should be 1 everywhere."""
        link = rs.links.Identity()
        mu = np.array([1.0, 2.0, 3.0, 100.0])

        deriv = link.derivative(mu)

        expected = np.ones_like(mu)
        np.testing.assert_array_almost_equal(deriv, expected)

    def test_roundtrip(self):
        """link then inverse should return original values."""
        link = rs.links.Identity()
        original = np.array([-5.0, 0.0, 0.5, 10.0])

        eta = link.link(original)
        recovered = link.inverse(eta)

        np.testing.assert_array_almost_equal(recovered, original)


class TestLogLink:
    """Tests for Log link function."""

    def test_link_is_natural_log(self):
        """Log link should compute natural logarithm."""
        link = rs.links.Log()
        mu = np.array([1.0, np.e, 10.0])

        eta = link.link(mu)

        expected = np.log(mu)
        np.testing.assert_array_almost_equal(eta, expected)

    def test_inverse_is_exponential(self):
        """Log inverse should compute exponential."""
        link = rs.links.Log()
        eta = np.array([0.0, 1.0, 2.0])

        mu = link.inverse(eta)

        expected = np.exp(eta)
        np.testing.assert_array_almost_equal(mu, expected)

    def test_inverse_always_positive(self):
        """Inverse of log link should always produce positive values."""
        link = rs.links.Log()
        eta = np.array([-100.0, -10.0, 0.0, 10.0, 100.0])

        mu = link.inverse(eta)

        assert np.all(mu > 0), "All predictions should be positive"

    def test_derivative(self):
        """Log derivative should be 1/μ."""
        link = rs.links.Log()
        mu = np.array([1.0, 2.0, 4.0])

        deriv = link.derivative(mu)

        expected = 1.0 / mu
        np.testing.assert_array_almost_equal(deriv, expected)

    def test_roundtrip(self):
        """link then inverse should return original values."""
        link = rs.links.Log()
        original = np.array([0.1, 1.0, 10.0, 100.0])

        eta = link.link(original)
        recovered = link.inverse(eta)

        np.testing.assert_array_almost_equal(recovered, original)

    def test_multiplicative_interpretation(self):
        """
        Demonstrate the multiplicative interpretation of log link.

        If η increases by Δ, μ is multiplied by exp(Δ).
        This is the foundation of rate relativities in insurance pricing.
        """
        link = rs.links.Log()

        # Two predictions differing by 0.1 on log scale
        eta1 = np.array([1.0])
        eta2 = np.array([1.1])

        mu1 = link.inverse(eta1)
        mu2 = link.inverse(eta2)

        # Ratio should be exp(0.1) ≈ 1.105
        ratio = mu2 / mu1
        expected_ratio = np.exp(0.1)

        np.testing.assert_almost_equal(ratio[0], expected_ratio)


class TestLogitLink:
    """Tests for Logit link function."""

    def test_link_is_log_odds(self):
        """Logit should compute log-odds: log(p/(1-p))."""
        link = rs.links.Logit()
        mu = np.array([0.5, 0.8, 0.2])

        eta = link.link(mu)

        expected = np.log(mu / (1 - mu))
        np.testing.assert_array_almost_equal(eta, expected)

    def test_inverse_is_sigmoid(self):
        """Logit inverse should compute sigmoid function."""
        link = rs.links.Logit()
        eta = np.array([0.0, 2.0, -2.0])

        mu = link.inverse(eta)

        expected = 1 / (1 + np.exp(-eta))
        np.testing.assert_array_almost_equal(mu, expected)

    def test_inverse_at_zero_is_half(self):
        """sigmoid(0) should be 0.5."""
        link = rs.links.Logit()
        eta = np.array([0.0])

        mu = link.inverse(eta)

        np.testing.assert_almost_equal(mu[0], 0.5)

    def test_inverse_always_in_unit_interval(self):
        """Inverse of logit should always be in [0, 1]."""
        link = rs.links.Logit()
        # Use moderate values to avoid floating-point saturation
        eta = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])

        mu = link.inverse(eta)

        assert np.all(mu > 0), "All probabilities should be > 0"
        assert np.all(mu < 1), "All probabilities should be < 1"

        # For extreme values, we expect saturation to 0 or 1
        # This is correct numerical behavior
        extreme_eta = np.array([-100.0, 100.0])
        extreme_mu = link.inverse(extreme_eta)
        assert np.all(extreme_mu >= 0), "Probabilities should be >= 0"
        assert np.all(extreme_mu <= 1), "Probabilities should be <= 1"

    def test_inverse_symmetry(self):
        """sigmoid(-x) should equal 1 - sigmoid(x)."""
        link = rs.links.Logit()
        eta = np.array([2.0])

        mu_pos = link.inverse(eta)
        mu_neg = link.inverse(-eta)

        np.testing.assert_almost_equal(mu_pos[0] + mu_neg[0], 1.0)

    def test_derivative(self):
        """Logit derivative should be 1/(μ(1-μ))."""
        link = rs.links.Logit()
        mu = np.array([0.5, 0.2, 0.8])

        deriv = link.derivative(mu)

        expected = 1 / (mu * (1 - mu))
        np.testing.assert_array_almost_equal(deriv, expected)

    def test_roundtrip(self):
        """link then inverse should return original values."""
        link = rs.links.Logit()
        original = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        eta = link.link(original)
        recovered = link.inverse(eta)

        np.testing.assert_array_almost_equal(recovered, original)

    def test_numerical_stability_large_values(self):
        """Test that extreme values don't cause overflow/NaN."""
        link = rs.links.Logit()
        eta = np.array([-100.0, -50.0, 50.0, 100.0])

        mu = link.inverse(eta)

        # Should not have any NaN or Inf
        assert np.all(np.isfinite(mu)), "All results should be finite"

    def test_odds_ratio_interpretation(self):
        """
        Demonstrate odds ratio interpretation.

        If η increases by β, odds are multiplied by exp(β).
        This is key to interpreting logistic regression coefficients.
        """
        link = rs.links.Logit()

        # Baseline and with effect
        eta_baseline = np.array([0.0])
        eta_with_effect = np.array([0.5])

        prob_baseline = link.inverse(eta_baseline)[0]
        prob_with_effect = link.inverse(eta_with_effect)[0]

        # Compute odds
        odds_baseline = prob_baseline / (1 - prob_baseline)
        odds_with_effect = prob_with_effect / (1 - prob_with_effect)

        # Odds ratio should equal exp(0.5)
        odds_ratio = odds_with_effect / odds_baseline
        expected_or = np.exp(0.5)

        np.testing.assert_almost_equal(odds_ratio, expected_or)


class TestLinkComparisons:
    """
    Cross-link comparison tests.

    These tests highlight how different links transform the same values.
    """

    def test_all_links_roundtrip(self):
        """All links should roundtrip correctly."""
        links = [
            (rs.links.Identity(), np.array([-1.0, 0.0, 1.0, 2.0])),
            (rs.links.Log(), np.array([0.1, 1.0, 10.0])),
            (rs.links.Logit(), np.array([0.1, 0.5, 0.9])),
        ]

        for link, original in links:
            eta = link.link(original)
            recovered = link.inverse(eta)
            np.testing.assert_array_almost_equal(
                recovered, original, err_msg=f"{link.name()} failed roundtrip"
            )

    def test_log_vs_logit_at_small_probabilities(self):
        """
        Compare log and logit behavior for small values.

        For μ close to 0:
        - log(μ) → -∞
        - logit(μ) → -∞ (but slower)

        For μ close to 1:
        - log(μ) → 0
        - logit(μ) → +∞
        """
        small_mu = np.array([0.01])

        log_eta = rs.links.Log().link(small_mu)[0]
        logit_eta = rs.links.Logit().link(small_mu)[0]

        # Both should be negative for small μ
        assert log_eta < 0
        assert logit_eta < 0

        # For small μ, log(μ) ≈ log(μ/(1-μ)) since 1-μ ≈ 1
        # So they should be similar
        assert abs(log_eta - logit_eta) < 1.0  # Within 1 unit


class TestPurePythonLinkHelpers:
    """Direct coverage for diagnostics-oriented pure-Python link helpers."""

    def test_link_inverse_identity_returns_original_object(self):
        eta = np.array([-2.0, 0.0, 3.5])
        assert rs.links.link_inverse("identity", eta) is eta

    def test_link_forward_identity_returns_original_object(self):
        mu = np.array([-2.0, 0.0, 3.5])
        assert rs.links.link_forward("identity", mu) is mu

    def test_link_inverse_log_clips_extreme_eta(self):
        eta = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
        expected = np.exp(np.array([-50.0, -50.0, 0.0, 50.0, 50.0]))
        np.testing.assert_allclose(rs.links.link_inverse("log", eta), expected)

    def test_link_forward_log_clips_nonpositive_mu_to_epsilon(self):
        mu = np.array([-1.0, 0.0, 1e-20, 1.0, np.e])
        eta = rs.links.link_forward("log", mu)
        assert np.all(np.isfinite(eta))
        np.testing.assert_allclose(eta[:3], np.log(np.array([1e-12, 1e-12, 1e-12])))
        np.testing.assert_allclose(eta[3:], np.array([0.0, 1.0]))

    def test_link_inverse_logit_clips_extreme_eta(self):
        eta = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
        expected = 1.0 / (1.0 + np.exp(-np.array([-50.0, -50.0, 0.0, 50.0, 50.0])))
        np.testing.assert_allclose(rs.links.link_inverse("logit", eta), expected)

    def test_link_forward_logit_clips_boundary_probabilities(self):
        mu = np.array([-1.0, 0.0, 0.25, 0.75, 1.0, 2.0])
        eta = rs.links.link_forward("logit", mu)
        assert np.all(np.isfinite(eta))
        np.testing.assert_allclose(eta[2:4], np.log(np.array([0.25 / 0.75, 0.75 / 0.25])))
        assert eta[0] == eta[1]
        assert eta[-1] == eta[-2]

    def test_cloglog_forward_inverse_roundtrip_on_safe_grid(self):
        mu = np.array([0.05, 0.25, 0.75, 0.95])
        eta = rs.links.link_forward("cloglog", mu)
        recovered = rs.links.link_inverse("cloglog", eta)
        np.testing.assert_allclose(recovered, mu, rtol=1e-12, atol=1e-12)

    def test_cloglog_helpers_clip_boundaries_without_nan(self):
        eta = np.array([-1000.0, 0.0, 1000.0])
        mu = rs.links.link_inverse("cloglog", eta)
        assert np.all(np.isfinite(mu))
        assert np.all((mu >= 0.0) & (mu <= 1.0))

        forward = rs.links.link_forward("cloglog", np.array([0.0, 1e-12, 1.0, 2.0]))
        assert np.all(np.isfinite(forward))

    def test_probit_helpers_fail_closed(self):
        with pytest.raises(NotImplementedError, match="Probit"):
            rs.links.link_inverse("probit", np.array([0.0]))
        with pytest.raises(NotImplementedError, match="Probit"):
            rs.links.link_forward("probit", np.array([0.5]))

    def test_unknown_link_helpers_raise_value_error(self):
        with pytest.raises(ValueError, match="Unsupported link"):
            rs.links.link_inverse("not-a-link", np.array([0.0]))
        with pytest.raises(ValueError, match="Unsupported link"):
            rs.links.link_forward("not-a-link", np.array([0.5]))


# =============================================================================
# Numerical-hardening parity tests
# =============================================================================
#
# These tests assert that the Python helpers ``apply_link`` and
# ``apply_inverse_link`` (defined in ``rustystats.formula``) produce the SAME
# finite outputs as the Rust ``Link`` trait implementations for all eta — most
# importantly at the extreme tails where naive ``np.exp(-eta)`` / ``np.exp(eta)``
# overflow to ``inf`` or trigger ``RuntimeWarning: overflow encountered in exp``.
#
# Rust guards (canonical reference):
#   - ``LogitLink::inverse``  — branches on ``x >= 0`` vs ``x < 0`` so
#                               ``exp(-x)`` / ``exp(x)`` is bounded; result
#                               saturates to 0 or 1 cleanly without inf/NaN.
#   - ``LogLink::inverse``    — clamps eta to ``[-700, 700]`` before ``exp``;
#                               result saturates around the IEEE-754 limit
#                               (``exp(700) ≈ 1.014e304``) instead of inf.
#   - ``IdentityLink::inverse`` — pure passthrough, no guards needed.
#
# Comparison strategy: the Rust link impls are exposed directly via PyO3 as
# ``rs.links.Identity()/Log()/Logit()``. We call ``.inverse(eta)`` on those
# wrappers as the canonical reference and assert equality (atol=1e-12) and
# finiteness against ``apply_inverse_link``. No GLMModel-fit detour needed.
# =============================================================================


class TestPythonRustLinkParityExtremeEta:
    """Python helpers in formula.py must match Rust links at extreme eta."""

    # Cases per task spec: normal, large negative, large positive,
    # near-boundary for logit (where 1/(1+exp(-eta)) rounds to 0 or 1).
    NORMAL = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    LARGE_NEG = np.array([-100.0, -500.0, -700.0, -1000.0])
    LARGE_POS = np.array([100.0, 500.0, 700.0, 1000.0])
    NEAR_BOUNDARY_LOGIT = np.array([-50.0, 50.0])

    @pytest.mark.parametrize(
        "eta",
        [NORMAL, LARGE_NEG, LARGE_POS, NEAR_BOUNDARY_LOGIT],
        ids=["normal", "large_neg", "large_pos", "near_boundary_logit"],
    )
    def test_logit_inverse_matches_rust(self, eta):
        """Python apply_inverse_link must equal Rust LogitLink.inverse for all eta."""
        from rustystats.formula import apply_inverse_link

        rust_mu = rs.links.Logit().inverse(eta)
        py_mu = apply_inverse_link(eta, "logit")

        # Both must be finite (Rust always is; Python must be after hardening).
        assert np.all(np.isfinite(py_mu)), (
            f"Python logit inverse produced non-finite values: {py_mu}"
        )
        assert np.all(np.isfinite(rust_mu)), (
            f"Rust logit inverse produced non-finite values: {rust_mu}"
        )

        # Element-wise equal to atol=1e-12.
        np.testing.assert_allclose(
            py_mu,
            rust_mu,
            atol=1e-12,
            err_msg=f"Logit Python/Rust mismatch at eta={eta}",
        )

    @pytest.mark.parametrize(
        "eta",
        [NORMAL, LARGE_NEG, LARGE_POS],
        ids=["normal", "large_neg", "large_pos"],
    )
    def test_log_inverse_matches_rust(self, eta):
        """Python apply_inverse_link must equal Rust LogLink.inverse for all eta."""
        from rustystats.formula import apply_inverse_link

        rust_mu = rs.links.Log().inverse(eta)
        py_mu = apply_inverse_link(eta, "log")

        assert np.all(np.isfinite(py_mu)), f"Python log inverse produced non-finite values: {py_mu}"
        assert np.all(np.isfinite(rust_mu)), (
            f"Rust log inverse produced non-finite values: {rust_mu}"
        )

        np.testing.assert_allclose(
            py_mu,
            rust_mu,
            atol=1e-12,
            err_msg=f"Log Python/Rust mismatch at eta={eta}",
        )

    @pytest.mark.parametrize(
        "eta",
        [NORMAL, LARGE_NEG, LARGE_POS],
        ids=["normal", "large_neg", "large_pos"],
    )
    def test_identity_inverse_matches_rust(self, eta):
        """Identity passthrough — should already match Rust trivially."""
        from rustystats.formula import apply_inverse_link

        rust_mu = rs.links.Identity().inverse(eta)
        py_mu = apply_inverse_link(eta, "identity")

        assert np.all(np.isfinite(py_mu))
        assert np.all(np.isfinite(rust_mu))
        np.testing.assert_allclose(py_mu, rust_mu, atol=1e-12)

    def test_logit_inverse_no_runtime_warning(self):
        """Hardened Python should not emit overflow RuntimeWarnings at extremes."""
        from rustystats.formula import apply_inverse_link

        eta = np.array([-1000.0, -700.0, -500.0, -100.0, 100.0, 500.0, 700.0, 1000.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # promote RuntimeWarnings to errors
            mu = apply_inverse_link(eta, "logit")
        assert np.all(np.isfinite(mu))

    def test_log_inverse_no_runtime_warning(self):
        """Hardened Python should not emit overflow RuntimeWarnings or return inf."""
        from rustystats.formula import apply_inverse_link

        eta = np.array([-1000.0, -700.0, 700.0, 1000.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mu = apply_inverse_link(eta, "log")
        assert np.all(np.isfinite(mu)), f"log inverse must be finite, got {mu}"


class TestRoundtripExtremeEta:
    """``apply_link(apply_inverse_link(eta, link), link) ≈ eta`` on a wide grid.

    Identity roundtrips exactly. Log/logit naturally lose information once eta
    is past the saturation region (e.g. logit-inverse of 1000 is exactly 1.0
    and ``logit(1.0) = +inf``), so we restrict the roundtrip grid to the
    representable interior — but we still go well past the historical Python
    overflow point (~700) so any test failure indicates a genuine guard
    asymmetry between forward and inverse links.
    """

    def test_identity_roundtrip_on_extremes(self):
        from rustystats.formula import apply_inverse_link, apply_link

        eta = np.array([-1e10, -1000.0, -1.0, 0.0, 1.0, 1000.0, 1e10])
        recovered = apply_link(apply_inverse_link(eta, "identity"), "identity")
        np.testing.assert_allclose(recovered, eta, atol=0.0, rtol=0.0)

    def test_log_roundtrip_on_safe_grid(self):
        """Roundtrip on eta in [-700, 700] — within Rust clamp range."""
        from rustystats.formula import apply_inverse_link, apply_link

        eta = np.array([-700.0, -100.0, -1.0, 0.0, 1.0, 100.0, 700.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mu = apply_inverse_link(eta, "log")
            recovered = apply_link(mu, "log")
        np.testing.assert_allclose(recovered, eta, atol=1e-9, rtol=1e-12)

    def test_logit_roundtrip_on_safe_grid(self):
        """Roundtrip on eta in [-20, 20]: well past the historical Python
        overflow point for naive ``1/(1+exp(-eta))`` at eta ≈ -700, but
        comfortably inside the region where sigmoid output hasn't saturated
        to bit-exact 1.0/0.0 (which would lose information through forward
        logit regardless of implementation). This covers the guarded path
        without being confounded by inherent float precision loss at deeper
        extremes (shared between Python and Rust)."""
        from rustystats.formula import apply_inverse_link, apply_link

        eta = np.array([-20.0, -10.0, -1.0, 0.0, 1.0, 10.0, 20.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mu = apply_inverse_link(eta, "logit")
            recovered = apply_link(mu, "logit")
        np.testing.assert_allclose(recovered, eta, atol=1e-6, rtol=1e-6)


class TestPerLinkEdgeCases:
    """Per-link edge cases called out in the TDD spec."""

    def test_identity_passes_huge_eta_unchanged(self):
        """Identity has no transformation — large |eta| is preserved exactly."""
        from rustystats.formula import apply_inverse_link

        eta = np.array([-1e300, -1e10, 1e10, 1e300])
        mu = apply_inverse_link(eta, "identity")
        # Bit-exact passthrough — atol=0, rtol=0.
        np.testing.assert_array_equal(mu, eta)
        assert np.all(np.isfinite(mu))

    def test_log_inverse_at_minus_1000_is_finite(self):
        """``apply_inverse_link(-1000, "log")`` must be finite, not NaN.

        Rust clamps to -700, so the result is ``exp(-700) ≈ 9.86e-305``
        (a tiny denormal-region positive). Underflow to 0.0 would also be
        acceptable per the spec ("essentially 0 is fine, but not NaN"); we
        match the Rust output exactly.
        """
        from rustystats.formula import apply_inverse_link

        mu = apply_inverse_link(np.array([-1000.0]), "log")
        assert np.all(np.isfinite(mu))
        assert not np.any(np.isnan(mu))
        rust_mu = rs.links.Log().inverse(np.array([-1000.0]))
        np.testing.assert_allclose(mu, rust_mu, atol=1e-12)

    def test_log_inverse_at_plus_1000_is_finite(self):
        """``apply_inverse_link(1000, "log")`` currently returns inf — must be
        clamped to a finite value matching Rust (``exp(700) ≈ 1.014e304``)."""
        from rustystats.formula import apply_inverse_link

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mu = apply_inverse_link(np.array([1000.0]), "log")
        assert np.all(np.isfinite(mu)), f"Expected finite, got {mu}"
        rust_mu = rs.links.Log().inverse(np.array([1000.0]))
        np.testing.assert_allclose(mu, rust_mu, atol=1e-12)

    def test_log_forward_at_zero_matches_rust(self):
        """``log(0) = -inf`` is the canonical Rust behavior; NaN is not.
        We assert Python behavior matches Rust, whatever that may be."""
        from rustystats.formula import apply_link

        # numpy's log(0) emits a divide-by-zero RuntimeWarning; suppress
        # because the contract here is "match Rust", not "be silent".
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            py_eta = apply_link(np.array([0.0]), "log")
        rust_eta = rs.links.Log().link(np.array([0.0]))
        # Either both -inf, or both NaN — but they must agree, and NaN is
        # explicitly disallowed by the spec for log.
        assert not np.any(np.isnan(py_eta)), f"log(0) must not produce NaN, got {py_eta}"
        assert not np.any(np.isnan(rust_eta))
        # -inf should compare equal to -inf under assert_array_equal.
        np.testing.assert_array_equal(py_eta, rust_eta)

    def test_logit_inverse_at_minus_1000_is_finite_near_zero(self):
        """``apply_inverse_link(-1000, "logit")`` must be finite ≈ 0+, not NaN.

        Currently the Python branch ``1.0 / (1.0 + np.exp(-eta))`` evaluates
        ``exp(1000)`` → inf, then ``1/(1+inf)`` → 0, but raises a
        RuntimeWarning along the way. After hardening, the result should be
        finite, in ``[0, 0.5)``, and warning-free.
        """
        from rustystats.formula import apply_inverse_link

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mu = apply_inverse_link(np.array([-1000.0]), "logit")
        assert np.all(np.isfinite(mu))
        assert not np.any(np.isnan(mu))
        assert mu[0] >= 0.0 and mu[0] < 0.5
        rust_mu = rs.links.Logit().inverse(np.array([-1000.0]))
        np.testing.assert_allclose(mu, rust_mu, atol=1e-12)

    def test_logit_inverse_at_plus_1000_is_finite_near_one(self):
        """``apply_inverse_link(1000, "logit")`` must be finite, just below or
        equal to 1, not NaN."""
        from rustystats.formula import apply_inverse_link

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mu = apply_inverse_link(np.array([1000.0]), "logit")
        assert np.all(np.isfinite(mu))
        assert not np.any(np.isnan(mu))
        assert mu[0] > 0.5 and mu[0] <= 1.0
        rust_mu = rs.links.Logit().inverse(np.array([1000.0]))
        np.testing.assert_allclose(mu, rust_mu, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
