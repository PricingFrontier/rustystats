"""The out-of-core pass executor gives the in-memory fit bit for bit.

``specs/out-of-core-fit.md`` §3, §5 and §7.1: an out-of-core fit runs the same
driver and per-row arithmetic as an in-memory fit, with rows reduced in fixed
blocks, so the two agree exactly whatever the source batching and thread count.
These tests serve a design that is already encoded, so they check the executor,
the driver and the abort plumbing independently of the batch encoder.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys

import numpy as np
import pytest
from rustystats._rustystats import (
    compute_standardization_batched_py,
    compute_standardization_py,
    fit_glm_batched_py,
    fit_glm_py,
    response_mean_py,
)

N = 3 * 4096 + 517


def _design(seed: int = 7):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=N)
    x2 = rng.uniform(-1.0, 1.0, size=N)
    level = rng.integers(0, 4, size=N)
    dummies = np.zeros((N, 3))
    for k in range(1, 4):
        dummies[:, k - 1] = (level == k).astype(float)
    x = np.column_stack([np.ones(N), x1, x2, dummies])
    eta = -0.5 + 0.3 * x1 - 0.2 * x2 + dummies @ np.array([0.2, -0.1, 0.4])
    weights = rng.uniform(0.5, 2.0, size=N)
    offset = rng.normal(0.0, 0.1, size=N)
    responses = {
        "gaussian": eta * 3.0 + rng.normal(size=N),
        "poisson": rng.poisson(np.exp(eta + offset)).astype(float),
        "gamma": rng.gamma(2.0, np.exp(eta) / 2.0),
        "binomial": (rng.random(N) < 1.0 / (1.0 + np.exp(-eta))).astype(float),
        "tweedie": np.where(rng.random(N) < 0.4, 0.0, rng.gamma(1.5, np.exp(eta))),
        "negbinomial": rng.negative_binomial(2.0, 2.0 / (2.0 + np.exp(eta))).astype(float),
    }
    return np.ascontiguousarray(x), responses, weights, offset


X, RESPONSES, WEIGHTS, OFFSET = _design()

STATISTICS = (
    "params",
    "cov_params_unscaled",
    "bse",
    "deviance",
    "null_deviance",
    "pearson_chi2",
    "scale",
    "llf",
    "aic",
    "bic",
    "iterations",
    "converged",
    "solver_status",
    "step_halving_used",
    "solver_trace",
)


def _read(result, name):
    value = getattr(result, name)
    value = value() if callable(value) else value
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, np.ndarray):
        return [float(v).hex() for v in np.asarray(value, dtype=np.float64).ravel()]
    return value


def _blocks(y, weights, offset, rows_per_block: int):
    def open_pass(*_kind):
        for start in range(0, N, rows_per_block):
            stop = min(start + rows_per_block, N)
            yield X[start:stop], y[start:stop], weights[start:stop], offset[start:stop]

    return open_pass


def _blocks_of(x, y, weights, offset, rows_per_block: int):
    def open_pass(*_kind):
        for start in range(0, N, rows_per_block):
            stop = min(start + rows_per_block, N)
            yield x[start:stop], y[start:stop], weights[start:stop], offset[start:stop]

    return open_pass


CASES = [
    ("gaussian", {}),
    ("poisson", {"offset": True}),
    ("gamma", {}),
    ("binomial", {}),
    ("tweedie", {"var_power": 1.5}),
    ("negbinomial", {"theta": 2.0}),
]


@pytest.mark.parametrize(("family", "options"), CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("rows_per_block", [1000, 4095, 4096, 4097, N])
def test_batched_fit_equals_in_memory_fit_bitwise(family, options, rows_per_block):
    y = RESPONSES[family]
    offset = OFFSET if options.get("offset") else np.zeros(N)
    common = {
        "var_power": options.get("var_power", 1.5),
        "theta": options.get("theta", 1.0),
    }
    in_memory = fit_glm_py(
        y,
        X,
        family,
        offset=OFFSET if options.get("offset") else None,
        weights=WEIGHTS,
        **common,
    )
    batched = fit_glm_batched_py(
        _blocks(y, WEIGHTS, offset, rows_per_block),
        N,
        X.shape[1],
        response_mean_py(y),
        family,
        has_offset=bool(options.get("offset")),
        **common,
    )
    for name in STATISTICS:
        assert _read(batched, name) == _read(in_memory, name), name


def test_batched_fit_is_thread_independent():
    """In-memory and batched fits agree across thread counts (separate processes)."""
    script = (
        "import numpy as np, sys;"
        "sys.path.insert(0, 'tests/python');"
        "from test_out_of_core_executor import X, RESPONSES, WEIGHTS, OFFSET, N, _blocks, _read;"
        "from rustystats._rustystats import fit_glm_batched_py, response_mean_py;"
        "y = RESPONSES['poisson'];"
        "r = fit_glm_batched_py(_blocks(y, WEIGHTS, OFFSET, 1000), N, X.shape[1],"
        " response_mean_py(y), 'poisson', has_offset=True, robust=['HC3']);"
        "print(_read(r, 'params'), _read(r, 'llf'), _read(r, 'cov_params_unscaled'),"
        " [float(v).hex() for v in r.cov_robust('HC3').ravel()], r.solver_trace)"
    )
    outputs = set()
    for threads in ("1", "4"):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "RAYON_NUM_THREADS": threads},
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        )
        outputs.add(completed.stdout)
    assert len(outputs) == 1


class _Sentinel(ValueError):
    pass


def test_exception_raised_while_serving_a_pass_propagates_unchanged():
    y = RESPONSES["poisson"]
    served = {"passes": 0}

    def open_pass(*_kind):
        served["passes"] += 1
        for start in range(0, N, 4096):
            if served["passes"] == 3 and start >= 4096:
                raise _Sentinel("stop here")
            stop = min(start + 4096, N)
            yield X[start:stop], y[start:stop], WEIGHTS[start:stop], OFFSET[start:stop]

    with pytest.raises(_Sentinel, match="stop here"):
        fit_glm_batched_py(open_pass, N, X.shape[1], response_mean_py(y), "poisson")
    # The abort is latched: no pass is served after it.
    assert served["passes"] == 3


def test_a_source_that_changes_its_row_count_is_refused():
    y = RESPONSES["gaussian"]
    calls = {"n": 0}

    def open_pass(*_kind):
        calls["n"] += 1
        stop = N if calls["n"] == 1 else N - 1
        yield X[:stop], y[:stop], WEIGHTS[:stop], OFFSET[:stop]

    with pytest.raises(ValueError, match="source changed between passes"):
        fit_glm_batched_py(open_pass, N, X.shape[1], response_mean_py(y), "gaussian")


def test_out_of_core_result_refuses_per_row_methods():
    from rustystats.exceptions import OutOfCoreResultError

    y = RESPONSES["gaussian"]
    result = fit_glm_batched_py(
        _blocks(y, WEIGHTS, OFFSET * 0.0, 5000), N, X.shape[1], response_mean_py(y), "gaussian"
    )
    for name in ("fittedvalues", "linear_predictor", "response", "resid_response"):
        with pytest.raises(OutOfCoreResultError, match=name):
            value = getattr(result, name)
            if callable(value):
                value()
    assert math.isfinite(result.llf())
    assert result.nobs == N


HC_TYPES = ("HC0", "HC1", "HC2", "HC3")


def _cov_bits(result, hc_type):
    return [float(v).hex() for v in np.asarray(result.cov_robust(hc_type)).ravel()]


@pytest.mark.parametrize(
    ("family", "options"),
    [CASES[1], CASES[2], CASES[3], CASES[4]],
    ids=["poisson", "gamma", "binomial", "tweedie"],
)
def test_fit_time_robust_covariances_equal_stored_design_bitwise(family, options):
    """robust= at fit time gives the stored-design sandwich bit for bit, in memory
    and from batches (§8)."""
    y = RESPONSES[family]
    offset = OFFSET if options.get("offset") else np.zeros(N)
    common = {"var_power": options.get("var_power", 1.5), "theta": options.get("theta", 1.0)}
    in_memory_kwargs = {
        "offset": OFFSET if options.get("offset") else None,
        "weights": WEIGHTS,
        **common,
    }
    stored = fit_glm_py(y, X, family, store_design_matrix=True, **in_memory_kwargs)
    fit_time = fit_glm_py(y, X, family, robust=list(HC_TYPES), **in_memory_kwargs)
    batched = fit_glm_batched_py(
        _blocks(y, WEIGHTS, offset, 4095),
        N,
        X.shape[1],
        response_mean_py(y),
        family,
        has_offset=bool(options.get("offset")),
        robust=list(HC_TYPES),
        **common,
    )
    for hc_type in HC_TYPES:
        expected = _cov_bits(stored, hc_type)
        assert _cov_bits(fit_time, hc_type) == expected, hc_type
        assert _cov_bits(batched, hc_type) == expected, hc_type
        assert [float(v).hex() for v in batched.bse_robust(hc_type)] == [
            float(v).hex() for v in stored.bse_robust(hc_type)
        ]


@pytest.mark.parametrize(
    "route",
    [
        {"alpha": 0.01, "l1_ratio": 1.0},
        {"alpha": 0.01, "l1_ratio": 0.0, "center": True},
    ],
    ids=["coordinate_descent", "centred_standardisation"],
)
def test_fit_time_robust_on_routes_without_a_row_pass(route):
    """Routes whose final pass does not see the original rows still give the
    stored-design sandwich."""
    y = RESPONSES["poisson"]
    kwargs = {"alpha": route["alpha"], "l1_ratio": route["l1_ratio"], "weights": WEIGHTS}
    if route.get("center"):
        kwargs["center"] = np.r_[0.0, X[:, 1:].mean(axis=0)]
        kwargs["scale"] = np.r_[1.0, X[:, 1:].std(axis=0)]
    stored = fit_glm_py(y, X, "poisson", store_design_matrix=True, **kwargs)
    fit_time = fit_glm_py(y, X, "poisson", robust=["HC1", "HC3"], **kwargs)
    for hc_type in ("HC1", "HC3"):
        assert _cov_bits(fit_time, hc_type) == _cov_bits(stored, hc_type), hc_type


def test_robust_types_not_listed_at_fit_are_refused_on_an_out_of_core_result():
    from rustystats.exceptions import OutOfCoreResultError

    y = RESPONSES["poisson"]
    batched = fit_glm_batched_py(
        _blocks(y, WEIGHTS, OFFSET, 5000),
        N,
        X.shape[1],
        response_mean_py(y),
        "poisson",
        has_offset=True,
        robust=["hc1"],
    )
    assert np.all(np.isfinite(batched.bse_robust("HC1")))
    with pytest.raises(OutOfCoreResultError, match="HC3"):
        batched.bse_robust("HC3")


def test_robust_option_refusals():
    y = RESPONSES["gaussian"]
    with pytest.raises(ValueError, match="Unknown robust covariance type"):
        fit_glm_py(y, X, "gaussian", robust=["HC4"])
    with pytest.raises(ValueError, match="needs the covariance matrix"):
        fit_glm_py(y, X, "gaussian", robust=["HC0"], skip_covariance=True)
    with pytest.raises(ValueError, match="needs the covariance matrix"):
        fit_glm_batched_py(
            _blocks(y, WEIGHTS, OFFSET, 5000),
            N,
            X.shape[1],
            response_mean_py(y),
            "gaussian",
            robust=["HC0"],
            skip_covariance=True,
        )


def _standardization(fit_intercept: bool):
    """Centres and scales as the formula layer passes them (centres zeroed with an
    intercept, as `solver_standardization` does)."""
    center = np.r_[0.0, X[:, 1:].mean(axis=0)]
    scale = np.r_[1.0, X[:, 1:].std(axis=0)]
    if fit_intercept:
        center = np.zeros_like(center)
    return center, scale


PENALISED = [
    ("lasso", {"alpha": 300.0, "l1_ratio": 1.0}),
    ("elastic_net", {"alpha": 300.0, "l1_ratio": 0.5}),
    ("lasso_scaled", {"alpha": 300.0, "l1_ratio": 1.0, "std": True}),
    ("lasso_centred", {"alpha": 300.0, "l1_ratio": 1.0, "std": True, "fit_intercept": False}),
    ("ridge_centred", {"alpha": 300.0, "l1_ratio": 0.0, "std": True, "fit_intercept": False}),
    ("lasso_one_iteration", {"alpha": 300.0, "l1_ratio": 1.0, "max_iter": 1}),
    ("lasso_no_iteration", {"alpha": 300.0, "l1_ratio": 1.0, "max_iter": 0}),
]


@pytest.mark.parametrize("family", ["poisson", "gaussian", "binomial"])
@pytest.mark.parametrize(("name", "route"), PENALISED, ids=[c[0] for c in PENALISED])
def test_penalised_and_standardised_routes_equal_in_memory_bitwise(family, name, route):
    """Coordinate descent and centred standardisation run from batches give the
    in-memory fit bit for bit, robust covariances included (§6.7, §7.5)."""
    y = RESPONSES[family]
    fit_intercept = route.get("fit_intercept", True)
    kwargs = {
        "alpha": route["alpha"],
        "l1_ratio": route["l1_ratio"],
        "fit_intercept": fit_intercept,
        "max_iter": route.get("max_iter", 25),
    }
    if route.get("std"):
        kwargs["center"], kwargs["scale"] = _standardization(fit_intercept)
    robust = ["HC1", "HC3"] if route.get("max_iter", 25) > 0 else []
    in_memory = fit_glm_py(y, X, family, weights=WEIGHTS, store_design_matrix=True, **kwargs)
    batched = fit_glm_batched_py(
        _blocks(y, WEIGHTS, np.zeros(N), 4095),
        N,
        X.shape[1],
        response_mean_py(y),
        family,
        robust=robust,
        **kwargs,
    )
    for field in STATISTICS:
        assert _read(batched, field) == _read(in_memory, field), field
    for hc_type in robust:
        assert _cov_bits(batched, hc_type) == _cov_bits(in_memory, hc_type), hc_type


@pytest.mark.parametrize("rows_per_block", [1000, 4096, N])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_batched_standardization_moments_equal_in_memory_bitwise(
    rows_per_block, weighted, fit_intercept
):
    """Passes M1 and M2 over batches give the in-memory moments bit for bit (§6.7)."""
    pen_mask = np.array([True, True, True, False, True, True])
    expected = compute_standardization_py(X, WEIGHTS if weighted else None, pen_mask, fit_intercept)
    batched = compute_standardization_batched_py(
        _blocks(RESPONSES["gaussian"], WEIGHTS, OFFSET, rows_per_block),
        N,
        X.shape[1],
        weighted=weighted,
        pen_mask=pen_mask.tolist(),
        fit_intercept=fit_intercept,
    )
    for got, want in zip(batched, expected):
        assert [float(v).hex() for v in got] == [float(v).hex() for v in want]


def _smooth_design(k: int = 8):
    """X = [1, x1, x2, dummies, B-spline basis of a smooth covariate] and its penalty."""
    from rustystats._rustystats import bs_py

    rng = np.random.default_rng(11)
    u = rng.uniform(0.0, 10.0, size=N)
    basis = np.asarray(bs_py(u, k, 3, None, False))
    k = basis.shape[1]
    x_full = np.ascontiguousarray(np.column_stack([X, basis]))
    d = np.diff(np.eye(k), 2, axis=0)
    effect = np.sin(u / 2.0) * 0.3
    return x_full, [(X.shape[1], X.shape[1] + k)], [d.T @ d], effect


X_SMOOTH, SMOOTH_RANGES, SMOOTH_PENALTIES, SMOOTH_EFFECT = _smooth_design()

SMOOTH_CASES = [
    ("poisson", {"offset": True}),
    ("gaussian", {}),
    ("gamma", {}),
    ("binomial", {}),
    ("poisson_short", {"offset": True, "max_iter": 2}),
    ("gaussian_fixed_lambda", {"lambda_min": 5.0, "lambda_max": 5.0}),
    ("poisson_monotone", {"offset": True, "monotone": "increasing", "max_iter": 200}),
    ("gaussian_monotone", {"monotone": "increasing", "max_iter": 200}),
    ("gamma_monotone_short", {"monotone": "decreasing", "max_iter": 30}),
    # Refinement, then an intercept refresh from the partitioned state.
    ("poisson_monotone_refresh", {"offset": True, "monotone": "increasing", "max_iter": 12}),
]


@pytest.mark.parametrize(("name", "options"), SMOOTH_CASES, ids=[c[0] for c in SMOOTH_CASES])
def test_batched_smooth_fit_equals_in_memory_bitwise(name, options):
    """The penalised smooth driver gives the in-memory fit bit for bit from batches,
    whatever the batching, with its λ, EDFs, GCV, statistics and robust covariances
    (§7.6)."""
    from rustystats._rustystats import fit_smooth_glm_batched_py, fit_smooth_glm_unified_py

    family = name.split("_")[0]
    rng = np.random.default_rng(5)
    effect = {
        "increasing": np.sort(SMOOTH_EFFECT)[np.argsort(np.argsort(X_SMOOTH[:, 7]))],
        "decreasing": -np.sort(SMOOTH_EFFECT)[np.argsort(np.argsort(X_SMOOTH[:, 7]))],
    }.get(options.get("monotone"), SMOOTH_EFFECT)
    eta = effect + (OFFSET if options.get("offset") else 0.0)
    mean = {
        "poisson": np.exp(eta - 0.5),
        "gaussian": 2.0 + eta,
        "gamma": np.exp(eta + 1.0),
        "binomial": 1.0 / (1.0 + np.exp(-eta)),
    }[family]
    y = {
        "poisson": lambda: rng.poisson(mean).astype(float),
        "gaussian": lambda: mean + rng.normal(size=N),
        "gamma": lambda: rng.gamma(2.0, mean / 2.0),
        "binomial": lambda: (rng.random(N) < mean).astype(float),
    }[family]()
    offset = OFFSET if options.get("offset") else np.zeros(N)
    common = {
        "max_iter": options.get("max_iter", 25),
        "lambda_min": options.get("lambda_min", 0.001),
        "lambda_max": options.get("lambda_max", 1000.0),
        "robust": ["HC1", "HC3"],
    }
    if options.get("monotone"):
        common["smooth_monotonicity"] = [options["monotone"]]
    in_memory, in_memory_meta = fit_smooth_glm_unified_py(
        y,
        X_SMOOTH,
        SMOOTH_RANGES,
        SMOOTH_PENALTIES,
        family,
        offset=OFFSET if options.get("offset") else None,
        weights=WEIGHTS,
        **common,
    )
    for rows_per_block in (4095, N):
        batched, batched_meta = fit_smooth_glm_batched_py(
            _blocks_of(X_SMOOTH, y, WEIGHTS, offset, rows_per_block),
            N,
            X_SMOOTH.shape[1],
            response_mean_py(y),
            SMOOTH_RANGES,
            SMOOTH_PENALTIES,
            family,
            has_offset=bool(options.get("offset")),
            **common,
        )
        for field in STATISTICS:
            assert _read(batched, field) == _read(in_memory, field), field
        for key in ("lambdas", "smooth_edfs", "total_edf", "gcv"):
            got = np.atleast_1d(np.asarray(batched_meta[key], dtype=np.float64))
            want = np.atleast_1d(np.asarray(in_memory_meta[key], dtype=np.float64))
            assert [v.hex() for v in got.tolist()] == [v.hex() for v in want.tolist()], key
        for hc_type in common["robust"]:
            assert _cov_bits(batched, hc_type) == _cov_bits(in_memory, hc_type), hc_type
