"""RS-ACT-003: smooth (penalized) GLM fits must use the requested var_power / theta.

Before this fix ``fit_smooth_glm_unified_py`` hard-coded
``family_from_name(family, 1.5, 1.0)``, so a penalized Tweedie or Negative
Binomial silently ignored the ``var_power`` / ``theta`` the user asked for -- and
the NB family string advertised a theta the fit never used. A penalized smooth
fit is deterministic, so "ignored a parameter" shows up as byte-identical
coefficients across different parameter values; "used it" shows up as different
coefficients. These tests pin that.

A ``{"type": "bs", "k": N}`` term (k, not df) routes through the penalized smooth
solver; ``{"type": "bs", "df": N}`` is the fixed-df path that does not.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import rustystats as rs


def _smooth_count_frame(seed: int = 0, n: int = 800) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 4.0, n)
    mu = np.exp(0.3 + 0.6 * np.sin(x))
    y = rng.poisson(mu).astype(float)
    return pl.DataFrame({"y": y, "x": x})


def _fit_smooth(data: pl.DataFrame, family: str, *, theta=None, var_power=None):
    kwargs = {}
    if theta is not None:
        kwargs["theta"] = theta
    if var_power is not None:
        kwargs["var_power"] = var_power
    return rs.glm_dict(
        response="y",
        terms={"x": {"type": "bs", "k": 8}},  # k => penalized smooth term
        data=data,
        family=family,
        **kwargs,
    ).fit()


class TestSmoothFamilyParams:
    def test_bs_k_spec_routes_through_smooth_solver(self):
        """Guard against vacuity: the k-spec term really is a smooth term."""
        data = _smooth_count_frame()
        model = rs.glm_dict(
            response="y",
            terms={"x": {"type": "bs", "k": 8}},
            data=data,
            family="poisson",
        )
        smooth_terms, _col_indices = model._builder.get_smooth_terms()
        assert len(smooth_terms) == 1

    def test_smooth_negbinomial_respects_theta(self):
        """003.2: smooth NB at theta=0.5 differs from theta=5.0 (theta no longer ignored)."""
        data = _smooth_count_frame()
        low = _fit_smooth(data, "negbinomial", theta=0.5)
        high = _fit_smooth(data, "negbinomial", theta=5.0)
        assert not np.allclose(low.params, high.params)

    def test_smooth_tweedie_respects_var_power(self):
        """003.1 (R): smooth Tweedie at var_power=1.2 differs from var_power=1.8."""
        data = _smooth_count_frame()
        low = _fit_smooth(data, "tweedie", var_power=1.2)
        high = _fit_smooth(data, "tweedie", var_power=1.8)
        assert not np.allclose(low.params, high.params)

    def test_tweedie_family_string_records_var_power(self):
        """003.3: Tweedie result metadata records the p actually used."""
        data = _smooth_count_frame()
        smooth = _fit_smooth(data, "tweedie", var_power=1.2)
        plain = rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}},
            data=data,
            family="tweedie",
            var_power=1.8,
        ).fit()

        assert smooth.family == "Tweedie(p=1.2000)"
        assert plain.family == "Tweedie(p=1.8000)"
        assert np.isfinite(smooth.compute_loss(data))

    def test_smooth_negbinomial_family_string_is_honest(self):
        """003.3: the smooth NB result reports the theta it actually used, non-vacuously."""
        data = _smooth_count_frame()
        result = _fit_smooth(data, "negbinomial", theta=3.0)
        assert "NegativeBinomial" in result.family
        assert "3.0000" in result.family
        # Not vacuous: the fit at theta=3.0 must differ from theta=1.0.
        other = _fit_smooth(data, "negbinomial", theta=1.0)
        assert not np.allclose(result.params, other.params)
