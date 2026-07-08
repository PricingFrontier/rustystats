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
    def test_first_iteration_never_accepts_catastrophic_full_step(self):
        """007.1b: iteration one is subject to the same deviance acceptance gate."""
        rng = np.random.default_rng(123)
        y = rng.poisson(np.exp(rng.normal(0.0, 2.0, 50))).astype(float)
        x = rng.normal(0.0, 10.0, 50)
        data = pl.DataFrame({"y": y, "x": x})

        result = rs.glm_dict(
            response="y", terms={"x": {"type": "linear"}}, data=data, family="poisson"
        ).fit(max_iter=1)

        assert result.step_halving_used
        assert result.deviance < 1392.5

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


class TestStepHalvingNoImprovement:
    """007.2: when every halved trial step still worsens the deviance, IRLS
    retains the previous iterate and reports `step_halving_no_improvement`."""

    def test_status_step_halving_no_improvement_when_budget_exhausts(self):
        # Mirror of the Rust test:
        # `test_step_halving_no_improvement_retains_previous_iterate`. The
        # fixture pairs huge x with a huge y so the WLS slope sends mu = exp(eta)
        # to +inf at every halved blend, making the deviance non-finite and
        # never acceptable. IRLS must retain the previous iterate's coefficients
        # rather than take a worse step.
        x = np.array([0.0, 1.0, 2.0, 1.0e6, 1.0e6, 1.0e6])
        y = np.array([1.0, 1.0, 1.0, 1.0e300, 1.0e300, 1.0e300])
        data = pl.DataFrame({"y": y, "x": x})

        result = rs.glm_dict(
            response="y", terms={"x": {"type": "linear"}}, data=data, family="poisson"
        ).fit(max_iter=5)

        assert result.solver_status == "step_halving_no_improvement", (
            f"expected halving-exhaustion status, got {result.solver_status!r} "
            f"(deviance={result.deviance}, iters={result.iterations})"
        )
        assert not result.converged
        # Retained coefficients are the previous iterate's: they must be finite
        # (a successful trial would have given a finite deviance too, but the
        # failed full/halved trials would have given inf coefficients).
        coefs = np.asarray(result.coefficients)
        assert np.all(np.isfinite(coefs)), (
            f"retained coefs must be finite (previous iterate), got {coefs}"
        )
        assert np.isfinite(result.deviance)


class TestSmoothSolverStatus:
    """RS-ACT-007 extended into the smooth (PIRLS) solver: the smooth path also
    emits a terminal solver_status and a step_halving_used flag (previously
    untested — the spec's RS-ACT-007 'Relevant code' listed only irls.rs)."""

    def test_smooth_fit_reports_converged_status_and_halving_flag(self):
        rng = np.random.default_rng(2)
        n = 800
        x = rng.uniform(0.0, 4.0, n)
        y = rng.poisson(np.exp(0.3 + 0.6 * np.sin(x))).astype(float)
        data = pl.DataFrame({"y": y, "x": x})
        result = rs.glm_dict(
            response="y", terms={"x": {"type": "bs", "k": 8}}, data=data, family="poisson"
        ).fit()
        assert result.optimizer_route == "gcv_smooth"  # really routed through the smooth solver
        assert result.solver_status == "converged"
        assert isinstance(result.step_halving_used, bool)
        assert np.isfinite(result.deviance)
