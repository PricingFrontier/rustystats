"""
Tests for Link Functions
========================

These tests verify that link function implementations are correct.
We test both mathematical properties and numerical stability.

Run with: pytest tests/python/test_links.py -v
"""

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
                recovered, original,
                err_msg=f"{link.name()} failed roundtrip"
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
