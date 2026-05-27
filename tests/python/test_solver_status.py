"""RS-ACT-007: IRLS step acceptance, final-mu clamp, and honest solver status.

The unconstrained IRLS path used to unconditionally accept the last (possibly
worse) half-step and leave the final fitted mu unclamped, and reported no status
on non-convergence. Now: a step that worsens the deviance is never accepted (the
previous iterate is retained), the final fitted mu is clamped into the family's
domain, and the result exposes a terminal `solver_status`.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import rustystats as rs


class TestFinalMuClamp:
    def test_binomial_fitted_values_strictly_inside_unit_interval(self):
        """007.3: separated logistic data still yields fitted mu in (0, 1)."""
        rng = np.random.default_rng(0)
        n = 200
        x = rng.normal(0.0, 1.0, n)
        y = (x > 0).astype(float)  # near-perfect separation drives eta to ±inf
        data = pl.DataFrame({"y": y, "x": x})
        result = rs.glm_dict(
            response="y", terms={"x": {"type": "linear"}}, data=data, family="binomial"
        ).fit(max_iter=50)

        fitted = np.asarray(result.fittedvalues)
        assert np.all(fitted > 0.0)
        assert np.all(fitted < 1.0)


class TestSolverStatus:
    def test_converged_fit_reports_converged(self):
        rng = np.random.default_rng(1)
        n = 400
        x = rng.normal(0.0, 1.0, n)
        y = rng.poisson(np.exp(0.3 + 0.5 * x)).astype(float)
        data = pl.DataFrame({"y": y, "x": x})
        result = rs.glm_dict(
            response="y", terms={"x": {"type": "linear"}}, data=data, family="poisson"
        ).fit(max_iter=50)
        assert result.converged
        assert result.solver_status == "converged"

    def test_non_convergence_reports_max_iterations(self):
        """007.4: a budget-capped fit reports an honest non-converged status."""
        rng = np.random.default_rng(1)
        n = 400
        x = rng.normal(0.0, 1.0, n)
        y = rng.poisson(np.exp(0.3 + 0.5 * x)).astype(float)
        data = pl.DataFrame({"y": y, "x": x})
        result = rs.glm_dict(
            response="y", terms={"x": {"type": "linear"}}, data=data, family="poisson"
        ).fit(max_iter=1)
        assert not result.converged
        assert result.solver_status == "max_iterations"


class TestStepHalvingAcceptance:
    def test_overshooting_gamma_fit_step_halves_and_converges(self):
        """007.1: when the full Newton step overshoots, a half step is accepted and
        the fit still converges to a valid (finite, monotone) solution."""
        rng = np.random.default_rng(0)
        n = 200
        x = rng.normal(0.0, 1.0, n)
        mu = np.exp(0.3 + 3.0 * x)  # steep slope -> full Newton step overshoots
        y = rng.gamma(0.3, mu / 0.3)  # high variance (shape 0.3)
        data = pl.DataFrame({"y": y, "x": x})
        result = rs.glm_dict(
            response="y", terms={"x": {"type": "linear"}}, data=data, family="gamma"
        ).fit(max_iter=100)

        assert result.step_halving_used
        assert result.converged
        assert result.solver_status == "converged"
        assert np.isfinite(result.deviance)
