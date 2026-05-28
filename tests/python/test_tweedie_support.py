"""RS-ACT-006 (PR13): Tweedie support contract.

Defaults to the compound Poisson-Gamma interior ``1 < p < 2`` (the actuarial
pure-premium regime). Other powers — ``p <= 0``, ``p == 1``, ``p == 2``, and
``p > 2`` — are valid Tweedie distributions but outside the default; they
require ``allow_extended_tweedie=True`` and bring per-regime support rules
with them. The ``0 < p < 1`` band is rejected *always* because no Tweedie
distribution exists there.

Tests:
* 006.1 — ``0 < p < 1`` errors always, even with ``allow_extended_tweedie=True``.
* 006.2 — ``p <= 0``, ``== 1``, ``== 2``, ``> 2`` error by default; accepted
  with ``allow_extended_tweedie=True``.
* 006.3 — ``1 < p < 2`` accepts zeros and positives.
* 006.4 — ``p == 2`` (Gamma equivalent) rejects exact zeros and never returns
  infinite deviance silently.
* 006.5 — extended ``p >= 2`` rejects exact zeros (``y > 0`` required).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import rustystats as rs
from rustystats.exceptions import ValidationError


def _frame(y: np.ndarray, *, x_seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(x_seed)
    return pl.DataFrame({"y": y, "x1": rng.normal(size=y.size), "x2": rng.normal(size=y.size)})


def _fit(y, var_power, *, allow_extended_tweedie=False):
    df = _frame(y)
    return rs.glm_dict(
        response="y",
        terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
        data=df,
        family="tweedie",
        var_power=var_power,
        allow_extended_tweedie=allow_extended_tweedie,
    ).fit()


# --------------------------------------------------------------------------
# 006.1: 0 < p < 1 is mathematically invalid — error always.
# --------------------------------------------------------------------------


class TestInvalidRegime:
    @pytest.mark.parametrize("p", [0.01, 0.3, 0.5, 0.9, 0.999])
    def test_p_in_open_zero_one_errors_default(self, p):
        y = np.array([1.0, 2.0, 0.0, 3.0, 1.5])
        with pytest.raises(ValidationError, match="0, 1"):
            _fit(y, p)

    @pytest.mark.parametrize("p", [0.01, 0.3, 0.5, 0.9, 0.999])
    def test_p_in_open_zero_one_errors_even_with_extended(self, p):
        """006.1: extended mode does NOT unlock the invalid (0, 1) band."""
        y = np.array([1.0, 2.0, 0.0, 3.0, 1.5])
        with pytest.raises(ValidationError, match="0, 1"):
            _fit(y, p, allow_extended_tweedie=True)

    @pytest.mark.parametrize("p", [0.5])
    def test_direct_tweedie_family_rejects_invalid_p(self, p):
        """Direct rs.TweedieFamily(p=0.5) errors at construction."""
        with pytest.raises(ValueError):
            rs.TweedieFamily(p)


# --------------------------------------------------------------------------
# 006.2: p <= 0, == 1, == 2, > 2 error by default; accepted with extended.
# --------------------------------------------------------------------------


class TestExtendedRegimeGating:
    @pytest.mark.parametrize("p", [-2.0, -1.0, 0.0, 1.0, 2.0, 2.5, 3.0])
    def test_default_rejects_powers_outside_interior(self, p):
        """006.2: every power outside (1, 2) requires opt-in by default."""
        # Use a y that's valid for whichever regime is being tested (positive).
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValidationError, match="allow_extended_tweedie"):
            _fit(y, p)

    @pytest.mark.parametrize("p", [-1.0, 0.0, 1.0, 1.8])
    def test_extended_accepts_supported_combos(self, p):
        """Extended mode plus a per-regime valid y → fit succeeds.

        Powers in (-inf, 1] handle ``y >= 0``; the (1, 2) case is the default
        and is included as a sanity check (extended mode is a no-op there).
        """
        # For p <= 1 (incl. extended), y >= 0 is the support.
        y = np.array([0.5, 1.0, 0.0, 2.0, 1.5])
        if p in (1.8,):
            allow = False
        else:
            allow = True
        result = _fit(y, p, allow_extended_tweedie=allow)
        assert np.isfinite(result.deviance)

    @pytest.mark.parametrize("p", [2.0, 2.5])
    def test_direct_tweedie_unit_deviance_rejects_zero_for_high_p(self, p):
        """Direct family methods enforce the p >= 2 support too."""
        fam = rs.TweedieFamily(p, allow_extended_tweedie=True)
        with pytest.raises(ValueError, match=r"strictly positive|y > 0"):
            fam.unit_deviance(np.array([0.0]), np.array([1.0]))

    @pytest.mark.parametrize("p", [2.0, 2.5])
    def test_low_level_fit_rejects_extended_p_without_opt_in(self, p):
        """The direct Rust binding follows the same default support contract."""
        from rustystats._rustystats import fit_glm_py

        y = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.ones((y.size, 1), dtype=np.float64)
        with pytest.raises(ValueError, match="allow_extended_tweedie"):
            fit_glm_py(y, x, "tweedie", "log", p, 1.0)

    @pytest.mark.parametrize("p", [2.0, 2.5])
    def test_low_level_fit_accepts_extended_p_with_opt_in(self, p):
        from rustystats._rustystats import fit_glm_py

        y = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.ones((y.size, 1), dtype=np.float64)
        result = fit_glm_py(
            y,
            x,
            "tweedie",
            "log",
            p,
            1.0,
            allow_extended_tweedie=True,
        )
        assert np.isfinite(result.deviance)


# --------------------------------------------------------------------------
# 006.3: 1 < p < 2 accepts zeros and positives.
# --------------------------------------------------------------------------


class TestInteriorRegime:
    @pytest.mark.parametrize("p", [1.1, 1.3, 1.5, 1.7, 1.9])
    def test_interior_p_accepts_zeros_and_positives(self, p):
        """006.3: the default compound Poisson-Gamma regime handles the
        actuarial pure-premium shape (zeros + positives) without an opt-in."""
        rng = np.random.default_rng(int(p * 100))
        n = 200
        mu = np.exp(0.5 + 0.3 * rng.normal(size=n))
        # Pure-premium-shaped response: ~30 % zeros, rest positive.
        y = np.where(rng.uniform(size=n) < 0.3, 0.0, rng.gamma(2.0, mu / 2.0))
        result = _fit(y, p)
        assert np.isfinite(result.deviance)
        assert result.converged

    def test_interior_p_negative_y_rejected(self):
        """Even inside (1, 2), Tweedie support excludes negative responses."""
        y = np.array([1.0, 2.0, -0.5, 3.0])
        with pytest.raises(ValidationError, match="non-negative"):
            _fit(y, 1.5)


# --------------------------------------------------------------------------
# 006.4: p == 2 (Gamma) rejects exact zeros and never silently returns inf.
# --------------------------------------------------------------------------


class TestGammaBoundary:
    def test_p_equals_2_default_rejects_with_clear_message(self):
        """006.4: p == 2 needs opt-in and the error mentions the Gamma escape hatch."""
        y = np.array([1.0, 2.0, 3.0, 4.0])  # all positive — would be valid Gamma
        with pytest.raises(ValidationError, match="gamma"):
            _fit(y, 2.0)

    def test_p_equals_2_extended_rejects_exact_zeros(self):
        """006.4: extended Tweedie at p == 2 still requires y > 0.

        Pins the legacy bug: the previous code accepted zeros and the unit
        deviance silently returned ``inf`` (Gamma deviance at y=0 diverges).
        """
        y = np.array([1.0, 2.0, 0.0, 3.0])
        with pytest.raises(ValidationError, match=r"strictly positive|y > 0|requires.*positive"):
            _fit(y, 2.0, allow_extended_tweedie=True)

    def test_p_equals_2_extended_with_positives_does_not_produce_inf_deviance(self):
        """006.4: with all-positive y under extended mode, the fit does not
        return a non-finite deviance (the headline regression we're fixing)."""
        rng = np.random.default_rng(42)
        y = rng.gamma(3.0, 1.0, 200)  # strictly positive
        result = _fit(y, 2.0, allow_extended_tweedie=True)
        assert np.isfinite(result.deviance), (
            f"p=2 with positive y produced non-finite deviance: {result.deviance}"
        )

    def test_p_equals_2_extended_regularized_cv_runs_with_positive_y(self):
        """The extended opt-in is threaded through alpha search and CV scoring."""
        rng = np.random.default_rng(123)
        y = rng.gamma(3.0, 1.0, 120)
        df = _frame(y, x_seed=3)
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=df,
            family="tweedie",
            var_power=2.0,
            allow_extended_tweedie=True,
        ).fit(regularization="lasso", cv=3, n_alphas=4)
        assert np.isfinite(result.deviance)

    def test_p_equals_2_extended_diagnostics_work_after_opt_in(self):
        """A fitted extended Tweedie result keeps diagnostics usable."""
        rng = np.random.default_rng(321)
        y = rng.gamma(3.0, 1.0, 100)
        df = _frame(y, x_seed=5)
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=df,
            family="tweedie",
            var_power=2.0,
            allow_extended_tweedie=True,
        ).fit()

        assert np.isfinite(result.compute_loss(df))
        diagnostics = result.diagnostics(df)
        assert np.isfinite(diagnostics.train_test.train.deviance)

    def test_default_p_equals_2_regularized_cv_fails_before_cv(self, monkeypatch):
        """Unsupported default regimes fail before CV starts fitting folds."""
        import rustystats.regularization_path as rp

        def should_not_start_cv(*args, **kwargs):  # pragma: no cover - should never run
            raise AssertionError("CV path should not start before Tweedie support validation")

        monkeypatch.setattr(rp, "fit_cv_regularization_path", should_not_start_cv)
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        df = _frame(y, x_seed=4)
        with pytest.raises(ValidationError, match="allow_extended_tweedie"):
            rs.glm_dict(
                response="y",
                terms={"x1": {"type": "linear"}},
                data=df,
                family="tweedie",
                var_power=2.0,
            ).fit(regularization="lasso", cv=3, n_alphas=3)


# --------------------------------------------------------------------------
# 006.5: extended p >= 2 rejects exact zeros (y > 0 required).
# --------------------------------------------------------------------------


class TestExtendedHighPowerSupport:
    @pytest.mark.parametrize("p", [2.0, 2.5, 3.0])
    def test_high_p_extended_rejects_exact_zeros(self, p):
        """006.5: for p >= 2 the Tweedie unit deviance at y == 0 is not
        meaningful (returns inf or negative); validation rejects up-front."""
        y = np.array([1.0, 2.0, 0.0, 3.0])
        with pytest.raises(ValidationError, match=r"strictly positive|y > 0|requires.*positive"):
            _fit(y, p, allow_extended_tweedie=True)

    @pytest.mark.parametrize("p", [2.5, 3.0])
    def test_high_p_extended_with_positives_runs(self, p):
        rng = np.random.default_rng(int(p * 100))
        y = rng.gamma(3.0, 1.0, 200)
        result = _fit(y, p, allow_extended_tweedie=True)
        assert np.isfinite(result.deviance)


# --------------------------------------------------------------------------
# 006.6: extended Tweedie threads through the score-test / inference path.
# --------------------------------------------------------------------------


class TestExtendedTweedieInference:
    """Regression for the inference call sites (PR13 review).

    The fix is in ``inference_py.rs`` / ``residuals_py.rs``: those entry
    points now construct families via
    ``family_from_name_with_tweedie_support(..., true)`` so a fitted extended
    Tweedie model's family string (e.g. ``"Tweedie(p=2.5)"``) no longer trips
    the default-regime gate when diagnostics / score tests rebuild the family.
    """

    def test_extended_tweedie_diagnostics_with_score_tests(self):
        """A fitted ``p=2.5`` Tweedie model runs diagnostics + score tests
        without re-raising the default-regime ``ValueError``."""
        rng = np.random.default_rng(2025)
        n = 250
        x_fitted = rng.normal(size=n)
        x_unfitted = rng.normal(size=n)  # candidate continuous factor
        mu = np.exp(1.0 + 0.3 * x_fitted)
        # Gamma-shaped strictly-positive response (p >= 2 requires y > 0).
        y = rng.gamma(2.0, mu / 2.0)
        df = pl.DataFrame({"y": y, "x_fitted": x_fitted, "x_unfitted": x_unfitted})

        result = rs.glm_dict(
            response="y",
            terms={"x_fitted": {"type": "linear"}},
            data=df,
            family="tweedie",
            var_power=2.5,
            allow_extended_tweedie=True,
        ).fit()
        assert np.isfinite(result.deviance)
        # The family string carries the extended ``p`` and is what the
        # inference call sites see when they reconstruct the family.
        assert result.family == "Tweedie(p=2.5000)"

        # diagnostics() with an unfitted continuous factor exercises
        # score_test_continuous_py — this is what would have failed before
        # the inference_py.rs / residuals_py.rs fix.
        diagnostics = result.diagnostics(
            df,
            continuous_factors=["x_unfitted"],
            compute_score_tests=True,
        )
        assert np.isfinite(diagnostics.train_test.train.deviance)

    def test_extended_tweedie_score_test_continuous_py_direct(self):
        """Direct rs binding accepts the extended Tweedie family string."""
        from rustystats._rustystats import score_test_continuous_py

        rng = np.random.default_rng(7)
        n = 120
        mu = np.full(n, 2.0)
        y = rng.gamma(2.0, mu / 2.0)
        x = np.ones((n, 1), dtype=np.float64)
        z = rng.normal(size=n)
        weights = np.ones(n)
        bread = np.array([[1.0]])

        # No ValidationError despite p=2.5 > 2 (extended regime).
        out = score_test_continuous_py(z, x, y, mu, weights, bread, "tweedie(p=2.5)")
        assert "pvalue" in out


# --------------------------------------------------------------------------
# Backward compatibility: existing var_power=1.5 (the default) still works.
# --------------------------------------------------------------------------


class TestDefaultStillWorks:
    def test_default_var_power_unaffected(self):
        """The pure-premium default ``var_power=1.5`` is untouched."""
        rng = np.random.default_rng(0)
        n = 300
        mu = np.exp(0.5 + 0.2 * rng.normal(size=n))
        y = np.where(rng.uniform(size=n) < 0.3, 0.0, rng.gamma(2.0, mu / 2.0))
        df = _frame(y, x_seed=1)
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=df,
            family="tweedie",
            # var_power defaults to 1.5; no allow_extended_tweedie needed.
        ).fit()
        assert result.converged
        assert result.family.startswith("Tweedie")
