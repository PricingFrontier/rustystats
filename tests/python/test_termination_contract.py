"""
Termination-contract acceptance tests for the monotone smooth-GLM solver.

Implements the acceptance criteria from the monotone-solver handoff (bug.md §7):

1. Invariant sweep: a canonical-link GLM with a free intercept, fitted to
   convergence, must satisfy |mean(prediction) - mean(y)| < 1e-3 on its own
   training data, with honest flags (converged=True implies the exit
   stationarity check passed; otherwise a structured non-converged status).
2. ``fit(max_iter=N)`` is a hard TOTAL budget: N=5 visibly truncates, large N
   is never silently stopped at an internal outer-loop cap.
3. Boundary regression: fits with a point mass exactly on the right boundary
   knot (winsorised variables) still satisfy criterion 1. (The basis-level
   assertion lives in Rust: splines::tests::test_boundary_knot_row_activates_last_basis.)
4. No fit-quality regression: deviance of the fixed solver must not be
   materially worse than the v0.8.14 solver's on the same seeds.
5. Serialization round-trip unchanged.

The synthetic generator mirrors the production failure mechanism: three
monotone-decreasing bs terms whose price term's true partial effect is
INCREASING (the constraint binds against the data), heavy-tailed covariates,
~1% point mass exactly at the winsorisation boundary (2.5), binomial response.

For the FULL spec (target encoding), the invariant is asserted on FITTED
values: TE trains on ordered target statistics but ``predict()`` re-encodes
with full training statistics, so mean(predict(train)) differs slightly from
mean(fittedvalues) even at a perfect optimum. The score equation is defined
with respect to the training design, i.e. the fitted values.
"""

import numpy as np
import polars as pl
import pytest
import rustystats as rs

# =============================================================================
# Synthetic generator (bug.md §2.1, verbatim mechanism)
# =============================================================================


def make_data(n: int = 12_000, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    u = rng.normal(0, 1, n)  # latent confound
    # price ratio YoY: heavy tail, winsorised at 2.5 -> ~1% mass AT the boundary knot
    x_price = np.exp(0.04 + 0.30 * u + 0.22 * rng.normal(0, 1, n))
    x_price = np.minimum(x_price, 2.5)
    # skewed premium-like level, correlated with price via u
    x_net = np.exp(5.4 + 0.35 * u + 0.85 * rng.normal(0, 1, n)) - 40.0
    x_net = np.clip(x_net, 25.0, 18_000)
    # heavy-tailed monotone-bs covariates
    x_mkt = np.exp(4.55 + 1.05 * rng.normal(0, 1, n)) + 25 * u
    x_mkt = np.clip(x_mkt, -500, 20_000)
    x_qts = np.clip(np.exp(1.1 + 0.75 * rng.normal(0, 1, n)), 0.2, 80)
    x_q2i = rng.integers(0, 60, n).astype(float)
    cat = rng.integers(0, 10, n)
    cat_eff = rng.normal(0, 0.35, 10)[cat]

    ln_net = np.log(x_net)
    eta = (
        1.9
        + 1.6 * np.log(x_price)  # positive partial effect -> decreasing constraint binds
        - 0.55 * (ln_net - 5.8) ** 2 * 0.35
        - 0.0009 * (np.minimum(x_mkt, 2000) - 300)
        - 0.05 * (x_qts - 4.0)
        + 0.004 * (x_q2i - 30)
        + cat_eff
    )
    p = 1 / (1 + np.exp(-eta))
    y = rng.binomial(1, p)
    return pl.DataFrame(
        {
            "y": y,
            "x_price": x_price,
            "x_net": x_net,
            "x_mkt": x_mkt,
            "x_qts": x_qts,
            "x_q2i": x_q2i,
            "cat": [f"L{c}" for c in cat],
        }
    )


SMALL = {  # minimal failing spec: 3 monotone bs + 2 ns (one skewed)
    "x_mkt": {"type": "bs", "monotonicity": "decreasing"},
    "x_qts": {"type": "bs", "monotonicity": "decreasing"},
    "x_price": {"type": "bs", "monotonicity": "decreasing"},
    "x_q2i": {"type": "ns"},
    "x_net": {"type": "ns"},
}
FULL = {**SMALL, "cat": {"type": "target_encoding"}}

# Per-seed training deviances of rustystats==0.8.14 (PyPI wheel) on the exact
# generator above, plain .fit(). Criterion 4: the fixed solver must not be
# materially worse. The relative slack absorbs legitimate differences in
# REML-selected lambdas (an honestly converged solve may smooth slightly more
# than a stalled trajectory, trading a sliver of raw deviance); it is far too
# small to hide a stalled fit, whose deviance excess is orders larger.
V0814_DEVIANCE_SMALL = {
    0: 10229.535914,
    1: 10347.082368,
    2: 9947.975302,
    3: 10444.724885,
    4: 10423.772721,
    5: 10755.830278,
    6: 11199.363599,
    7: 9427.961716,
}
V0814_DEVIANCE_FULL = {
    0: 10111.943739,
    1: 10274.611985,
    2: 9914.918898,
    3: 10371.477228,
    4: 10296.568470,
    5: 10679.397054,
    6: 11138.014725,
    7: 9293.727541,
}
DEVIANCE_RELATIVE_SLACK = 2e-3

MEAN_GAP_TOL = 1e-3  # |mean(pred) - mean(y)| in proportion terms (0.1pp)

NONCONVERGED_STATUSES = {
    "max_iterations",
    "step_halving_no_improvement",
    "stalled_nonstationary",
}


def _fit(terms, data, **fit_kwargs):
    return rs.glm_dict(
        response="y",
        terms=terms,
        interactions=None,
        data=data,
        family="binomial",
        link="logit",
        seed=42,
    ).fit(**fit_kwargs)


def _assert_honest_flags(model):
    """converged=True must imply the exit stationarity check passed."""
    assert model.stationary is not None, "smooth path must run the stationarity check"
    assert model.max_std_score is not None
    if model.converged:
        assert model.stationary is True
        assert model.solver_status == "converged"
        assert model.max_std_score < 1e-2
    else:
        assert model.solver_status in NONCONVERGED_STATUSES


# =============================================================================
# Criterion 1 + 4: invariant sweep with honest flags, no quality regression
# =============================================================================


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(8))
def test_invariant_sweep_small(seed):
    data = make_data(seed=seed)
    y = data["y"].to_numpy()
    model = _fit(SMALL, data)

    pred_gap = abs(y.mean() - np.asarray(model.predict(data), dtype=float).mean())
    assert pred_gap < MEAN_GAP_TOL, f"seed {seed}: mean gap {100 * pred_gap:+.3f}pp"
    _assert_honest_flags(model)

    baseline = V0814_DEVIANCE_SMALL[seed]
    assert model.deviance <= baseline * (1.0 + DEVIANCE_RELATIVE_SLACK), (
        f"seed {seed}: deviance {model.deviance:.4f} vs 0.8.14 baseline {baseline:.4f}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(8))
def test_invariant_sweep_full(seed):
    data = make_data(seed=seed)
    y = data["y"].to_numpy()
    model = _fit(FULL, data)

    # Fitted values: the design the score equation was solved against.
    fit_gap = abs(y.mean() - np.asarray(model.fittedvalues, dtype=float).mean())
    assert fit_gap < MEAN_GAP_TOL, f"seed {seed}: fitted mean gap {100 * fit_gap:+.3f}pp"
    # Predict re-encodes TE with full training statistics; allow the small
    # ordered-vs-final encoding drift but nothing like the solver-stall bias.
    pred_gap = abs(y.mean() - np.asarray(model.predict(data), dtype=float).mean())
    assert pred_gap < 5e-3, f"seed {seed}: predict mean gap {100 * pred_gap:+.3f}pp"
    _assert_honest_flags(model)

    baseline = V0814_DEVIANCE_FULL[seed]
    assert model.deviance <= baseline * (1.0 + DEVIANCE_RELATIVE_SLACK), (
        f"seed {seed}: deviance {model.deviance:.4f} vs 0.8.14 baseline {baseline:.4f}"
    )


# =============================================================================
# Fast versions (CI default): one seed, reduced n, same contract
# =============================================================================


@pytest.fixture(scope="module")
def small_fit():
    data = make_data(n=3000, seed=5)
    return data, _fit(SMALL, data)


def test_invariant_and_honest_flags_fast(small_fit):
    data, model = small_fit
    y = data["y"].to_numpy()
    pred_gap = abs(y.mean() - np.asarray(model.predict(data), dtype=float).mean())
    assert pred_gap < MEAN_GAP_TOL
    _assert_honest_flags(model)


def test_boundary_point_mass_still_satisfies_invariant(small_fit):
    """Criterion 3: x_price carries ~1% point mass exactly at the winsorisation
    boundary (2.5), which is the right boundary knot of a monotone-decreasing
    bs term. The fit must still satisfy the mean invariant."""
    data, model = small_fit
    boundary_share = (data["x_price"].to_numpy() == 2.5).mean()
    assert boundary_share > 0.003, "generator must place point mass on the boundary knot"
    y = data["y"].to_numpy()
    fit_gap = abs(y.mean() - np.asarray(model.fittedvalues, dtype=float).mean())
    assert fit_gap < MEAN_GAP_TOL


# =============================================================================
# Criterion 2: max_iter is a hard total budget
# =============================================================================


def test_max_iter_truncates_visibly():
    data = make_data(n=3000, seed=5)
    model = _fit(SMALL, data, max_iter=5)
    assert model.iterations <= 5
    assert model.converged is False
    assert model.solver_status == "max_iterations"


def test_max_iter_no_internal_cap():
    data = make_data(n=3000, seed=5)
    model = _fit(SMALL, data, max_iter=5000)
    # Either the solve genuinely converged (honest flags), or it used the full
    # budget. "max_iterations" with iterations far below the budget would mean
    # an internal cap (the old hard-coded 10-outer limit) — that is the bug.
    if model.solver_status == "max_iterations":
        assert model.iterations >= 5000
    else:
        _assert_honest_flags(model)


def test_max_iter_budget_is_monotone():
    data = make_data(n=3000, seed=5)
    iters = [_fit(SMALL, data, max_iter=budget).iterations for budget in (5, 15, 2000)]
    assert iters[0] <= iters[1] <= iters[2]
    assert iters[0] == 5  # truncation is exact, not approximate


def test_stationarity_gate_reports_nonstationary_on_truncated_fit():
    """The gate must be able to say NO — guards against it degenerating into a
    rubber stamp (a mutant that always reports stationary=True survives every
    other test, because the intercept refresh keeps the mean exact anyway).

    A hard-truncated fit with an unpenalized LINEAR coordinate strands that
    coordinate's score far from zero: the refresh restores the intercept but
    must not launder the rest. (In SMALL every non-intercept term is penalized
    or monotone, so a truncated SMALL fit is honestly stationary — the linear
    term is what gives the gate something to reject.)"""
    data = make_data(n=3000, seed=5)
    terms = {**SMALL, "x_q2i": {"type": "linear"}}
    model = _fit(terms, data, max_iter=2)
    assert model.converged is False
    assert model.stationary is False
    assert model.max_std_score > 1e-2
    assert model.solver_status == "max_iterations"
    # The mean backstop still holds even on this truncated fit.
    y = data["y"].to_numpy()
    fit_gap = abs(y.mean() - np.asarray(model.fittedvalues, dtype=float).mean())
    assert fit_gap < MEAN_GAP_TOL


# =============================================================================
# Regularized path: monotonicity is not enforced there — must warn, not hide
# =============================================================================


def test_regularized_path_warns_monotone_not_enforced():
    data = make_data(n=400, seed=0)
    terms = {
        "x_price": {"type": "bs", "monotonicity": "decreasing"},
        "x_q2i": {"type": "ns"},
    }
    with pytest.warns(UserWarning, match="NOT enforced on"):
        _fit(terms, data, alpha=0.05)


# =============================================================================
# Criterion 5: serialization round-trip
# =============================================================================


def test_serialization_roundtrip_unchanged(small_fit):
    data, model = small_fit
    loaded = rs.GLMModel.from_bytes(model.to_bytes())
    np.testing.assert_allclose(
        np.asarray(loaded.predict(data), dtype=float),
        np.asarray(model.predict(data), dtype=float),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(loaded.params, model.params, rtol=0, atol=0)
    assert loaded.solver_status == model.solver_status
    assert loaded.stationary == model.stationary
