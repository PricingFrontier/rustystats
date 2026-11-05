"""Thread pools, row kernels and the θ abort boundary of out-of-core fits.

``specs/out-of-core-fit.md`` §5.2: the dense or sparse row kernel is chosen from the
design's structural nonzeros per row and n, and gives the same bits either way. §11: a
fit's row reductions run in a pool of ``threads`` threads, the count its memory plan
budgets, and the result does not depend on it. §7.1: an exception raised while a pass is
served reaches the caller unchanged, including from Negative Binomial θ estimation's
initial Poisson fit.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import rustystats as rs
from _golden_cases import CASES, RESULT_FIELDS, _read
from rustystats._rustystats import (
    compute_standardization_batched_py,
    fit_cv_path_batched_py,
    fit_cv_path_py,
    fit_glm_batched_py,
    fit_glm_py,
    fit_negbinomial_batched_py,
    fit_negbinomial_py,
    fit_smooth_glm_batched_py,
    fit_smooth_glm_unified_py,
    rayon_threads_py,
    response_mean_py,
    sparse_rows_py,
)
from rustystats.exceptions import OutOfCoreOptionError

ROOT = Path(__file__).resolve().parents[2]
STATISTICS = (
    "params",
    "cov_params_unscaled",
    "bse",
    "deviance",
    "null_deviance",
    "pearson_chi2",
    "llf",
    "iterations",
    "converged",
    "solver_status",
    "step_halving_used",
    "solver_trace",
)


def _bits(value: Any) -> Any:
    if callable(value):
        value = value()
    if isinstance(value, (float, np.floating)):
        return float(value).hex()
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "f":
            return [float(v).hex() for v in value.ravel()]
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_bits(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _bits(v) for k, v in value.items()}
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    return repr(value)


def _statistics(result: Any) -> dict[str, Any]:
    return {name: _bits(_read(result, name)) for name in STATISTICS}


def _path_bits(path: dict[str, Any]) -> dict[str, Any]:
    """A CV path's results, without its timings and the global pool's size."""
    fields = {key: _bits(value) for key, value in path.items() if key != "profile"}
    profile = path["profile"]
    fields["passes"] = profile["passes"]
    fields["folds"] = _bits(profile["folds"])
    return fields


# ── A wide, sparse design: an intercept, two numeric columns and 40 dummies ──────────

N = 2 * 4096 + 1001


def _wide_design(seed: int = 11):
    rng = np.random.default_rng(seed)
    levels = rng.integers(0, 41, N)
    x = np.zeros((N, 43))
    x[:, 0] = 1.0
    x[:, 1] = rng.normal(size=N)
    x[:, 2] = np.where(rng.random(N) < 0.3, 0.0, rng.uniform(-1.0, 1.0, N))
    for level in range(1, 41):
        x[levels == level, 2 + level] = 1.0
    effects = rng.normal(0.0, 0.2, 41)
    eta = -0.3 + 0.2 * x[:, 1] - 0.1 * x[:, 2] + effects[levels]
    y = rng.poisson(np.exp(eta)).astype(float)
    weights = rng.uniform(0.5, 2.0, N)
    offset = rng.normal(0.0, 0.05, N)
    return np.ascontiguousarray(x), y, weights, offset


X, Y, WEIGHTS, OFFSET = _wide_design()
P = X.shape[1]


def _open(x=X, y=Y, weights=WEIGHTS, offset=OFFSET, rows: int = 1000):
    def open_pass(*_kind):
        for start in range(0, len(y), rows):
            stop = min(start + rows, len(y))
            yield x[start:stop], y[start:stop], weights[start:stop], offset[start:stop]

    return open_pass


def _fit_batched(**kwargs: Any) -> Any:
    return fit_glm_batched_py(
        _open(), N, P, response_mean_py(Y), "poisson", has_offset=True, **kwargs
    )


# ── Threads (§11) ────────────────────────────────────────────────────────────────


def _pool_threads() -> int:
    """Live threads of fit pools (named ``rs-fit-N``) in this process."""
    count = 0
    for task in Path("/proc/self/task").iterdir():
        try:
            name = (task / "comm").read_text(encoding="utf-8").strip()
        except OSError:  # the thread exited while we looked
            continue
        count += name.startswith("rs-fit-")
    return count


def _until_no_pool_threads(seconds: float = 10.0) -> int:
    """A dropped pool's threads exit on their own; wait for them."""
    deadline = time.monotonic() + seconds
    while (count := _pool_threads()) and time.monotonic() < deadline:
        time.sleep(0.01)
    return count


@pytest.mark.skipif(not Path("/proc/self/task").is_dir(), reason="needs Linux /proc")
@pytest.mark.parametrize("threads", [None, 1, 3])
def test_a_fit_reduces_its_rows_in_a_pool_of_the_requested_size(threads) -> None:
    assert _until_no_pool_threads() == 0, "an earlier fit's pool has ended"
    seen: list[int] = []

    def open_pass(*kind):
        seen.append(_pool_threads())
        yield from _open()(*kind)

    kwargs = {} if threads is None else {"threads": threads}
    fit_glm_batched_py(open_pass, N, P, response_mean_py(Y), "poisson", **kwargs)
    assert set(seen) == {threads or 0}
    assert _until_no_pool_threads() == 0, "the pool ends with the fit"


def test_results_do_not_depend_on_the_thread_count() -> None:
    reference = _statistics(_fit_batched(robust=["HC3"]))
    for threads in (1, 2, 4, 7):
        result = _fit_batched(robust=["HC3"], threads=threads)
        assert _statistics(result) == reference, threads
    assert rayon_threads_py() >= 1

    paths = [
        fit_cv_path_batched_py(
            _open(),
            N,
            P,
            response_mean_py(Y),
            "poisson",
            has_offset=True,
            alphas=[0.1, 0.01, 0.0],
            l1_ratio=0.5,
            n_folds=3,
            threads=threads,
        )
        for threads in (None, 1, 4)
    ]
    assert len({json.dumps(_path_bits(path)) for path in paths}) == 1

    moments = {
        json.dumps(_bits(compute_standardization_batched_py(_open(), N, P, threads=threads)))
        for threads in (None, 1, 4)
    }
    assert len(moments) == 1


def test_threads_must_be_positive() -> None:
    with pytest.raises(ValueError, match="threads must be positive"):
        _fit_batched(threads=0)
    with pytest.raises(ValueError, match="threads must be positive"):
        compute_standardization_batched_py(_open(), N, P, threads=0)


@pytest.mark.parametrize("threads", [None, 3])
def test_a_cv_path_reports_the_threads_its_reductions_ran_on(threads) -> None:
    kwargs = {} if threads is None else {"threads": threads}
    path = fit_cv_path_batched_py(
        _open(),
        N,
        P,
        response_mean_py(Y),
        "poisson",
        has_offset=True,
        alphas=[0.05, 0.0],
        n_folds=3,
        **kwargs,
    )
    assert path["profile"]["rayon_threads"] == (threads or rayon_threads_py())
    if threads is None:
        in_memory = fit_cv_path_py(Y, X, "poisson", offset=OFFSET, alphas=[0.05, 0.0], n_folds=3)
        assert in_memory["profile"]["rayon_threads"] == rayon_threads_py()


# ── Row kernels (§5.2) ───────────────────────────────────────────────────────────


def test_the_kernel_choice_reads_only_n_p_and_the_structural_bound() -> None:
    assert not sparse_rows_py(100_000, 100)  # no bound known: dense rows
    assert sparse_rows_py(100_000, 100, 35)
    assert not sparse_rows_py(100_000, 100, 36)
    assert not sparse_rows_py(99_999, 100, 1)  # fewer than 10M cells
    assert not sparse_rows_py(10_000_000, 15, 1)  # p below 16


def test_a_row_kernel_other_than_dense_or_sparse_is_refused() -> None:
    with pytest.raises(ValueError, match="row_kernel must be None, 'dense' or 'sparse'"):
        _fit_batched(row_kernel="auto")
    with pytest.raises(ValueError, match="row_kernel must be None, 'dense' or 'sparse'"):
        fit_glm_py(Y, X, "poisson", row_kernel="csr")


ROUTES = {
    "irls": {},
    "ridge_scaled": {"alpha": 0.5, "center": np.zeros(P), "scale": np.full(P, 1.5)},
    "ridge_centred": {
        "alpha": 0.5,
        "center": np.r_[0.0, np.full(P - 1, 0.1)],
        "scale": np.full(P, 1.5),
    },
    "lasso": {"alpha": 0.01, "l1_ratio": 1.0},
    "elastic_net_robust": {"alpha": 0.01, "l1_ratio": 0.5, "robust": ["HC0", "HC3"]},
}


@pytest.mark.parametrize("route", sorted(ROUTES))
@pytest.mark.parametrize("threads", [1, 4])
def test_sparse_rows_give_the_dense_rows_bits(route: str, threads: int) -> None:
    """Both kernels, given explicitly, and both modes agree bit for bit."""
    kwargs = ROUTES[route]
    results = {
        "memory-structural": _statistics(
            fit_glm_py(
                Y, X, "poisson", offset=OFFSET, weights=WEIGHTS, max_row_nonzeros=4, **kwargs
            )
        )
    }
    for kernel in ("dense", "sparse"):
        results[f"batched-{kernel}"] = _statistics(
            _fit_batched(threads=threads, row_kernel=kernel, **kwargs)
        )
        results[f"memory-{kernel}"] = _statistics(
            fit_glm_py(Y, X, "poisson", offset=OFFSET, weights=WEIGHTS, row_kernel=kernel, **kwargs)
        )
    reference = results["memory-dense"]
    for name, statistics in results.items():
        assert statistics == reference, name


@pytest.mark.parametrize("threads", [1, 4])
def test_sparse_rows_give_the_dense_rows_bits_on_every_driver(threads) -> None:
    """Smooth, CV and θ estimation over sparse and dense rows, in both modes."""
    penalty = np.diag(np.r_[0.0, np.ones(40)])

    def runs(kernel: str) -> list[Any]:
        smooth, meta = fit_smooth_glm_batched_py(
            _open(),
            N,
            P,
            response_mean_py(Y),
            [(2, P)],
            [penalty],
            "poisson",
            has_offset=True,
            threads=threads,
            row_kernel=kernel,
        )
        path = fit_cv_path_batched_py(
            _open(),
            N,
            P,
            response_mean_py(Y),
            "poisson",
            has_offset=True,
            weighted=True,
            alphas=[0.05, 0.0],
            n_folds=3,
            threads=threads,
            row_kernel=kernel,
        )
        theta, theta_meta = _negbinomial(threads=threads, row_kernel=kernel)
        return [
            _statistics(smooth),
            _bits(meta),
            _path_bits(path),
            _statistics(theta),
            _bits(theta_meta),
        ]

    def in_memory(kernel: str) -> list[Any]:
        smooth, meta = fit_smooth_glm_unified_py(
            Y, X, [(2, P)], [penalty], "poisson", offset=OFFSET, weights=WEIGHTS, row_kernel=kernel
        )
        path = fit_cv_path_py(
            Y,
            X,
            "poisson",
            offset=OFFSET,
            weights=WEIGHTS,
            alphas=[0.05, 0.0],
            n_folds=3,
            row_kernel=kernel,
        )
        theta, theta_meta = fit_negbinomial_py(
            Y, X, offset=OFFSET, weights=WEIGHTS, row_kernel=kernel
        )
        return [
            _statistics(smooth),
            _bits(meta),
            _path_bits(path),
            _statistics(theta),
            _bits(theta_meta),
        ]

    assert runs("sparse") == runs("dense")
    assert in_memory("sparse") == in_memory("dense")


# ── Negative Binomial θ: aborts cross the initial Poisson fit unchanged (§7.1) ──────


class _Spill:
    """μ kept in RAM for the θ passes of a batched θ estimation."""

    def __init__(self) -> None:
        self.blocks: list[np.ndarray] = []

    def __call__(self, block) -> None:
        if block is None:
            self.blocks = []
        else:
            self.blocks.append(np.array(block, dtype=np.float64))

    def open(self):
        mu = np.concatenate(self.blocks)
        for start in range(0, N, 4096):
            stop = min(start + 4096, N)
            yield Y[start:stop], WEIGHTS[start:stop], mu[start:stop]


def _negbinomial(open_pass=None, **kwargs: Any):
    spill = _Spill()
    return fit_negbinomial_batched_py(
        open_pass or _open(),
        spill,
        spill.open,
        N,
        P,
        response_mean_py(Y),
        has_offset=True,
        weighted=True,
        **kwargs,
    )


class _Sentinel(ValueError):
    pass


def test_an_abort_in_the_initial_poisson_fit_reaches_the_caller_unchanged() -> None:
    kinds: list[str] = []

    def recording(kind):
        kinds.append(kind)
        return _open()(kind)

    _negbinomial(recording)
    # Every pass before the first θ spill: the initial Poisson fit, the moments pass
    # and the first round's fit.
    before_theta = kinds.index("theta")
    assert before_theta >= 3
    for abort_at in range(before_theta):
        raised = _Sentinel(f"stop at pass {abort_at}")
        served = {"passes": 0}

        def open_pass(kind, abort_at=abort_at, raised=raised, served=served):
            index = served["passes"]
            served["passes"] += 1
            for number, block in enumerate(_open()(kind)):
                if index == abort_at and number == 1:
                    raise raised
                yield block

        with pytest.raises(_Sentinel) as caught:
            _negbinomial(open_pass)
        assert caught.value is raised, f"pass {abort_at}"
        assert served["passes"] == abort_at + 1, "no pass is served after the abort"


def test_an_on_batch_abort_during_theta_estimation_reaches_the_caller(tmp_path) -> None:
    frame = pl.DataFrame({"x": X[:, 1], "y": Y})
    glm = {"response": "y", "terms": {"x": {"type": "linear"}}, "family": "negbinomial"}

    def source() -> rs.BatchSource:
        def open_batches(columns):
            selected = frame.select(columns)
            for start in range(0, selected.height, 3000):
                yield selected.slice(start, 3000)

        return rs.BatchSource(open_batches, frame.schema)

    passes: list[tuple[int, str]] = []

    def record(progress) -> None:
        if (progress.pass_index, progress.pass_kind) not in passes:
            passes.append((progress.pass_index, progress.pass_kind))

    rs.glm_dict(data=source(), theta="estimate", **glm).fit(
        out_of_core=rs.OutOfCoreOptions(scratch_dir=tmp_path, on_batch=record)
    )
    kinds = [kind for _index, kind in passes]
    first_solver = kinds.index("start")
    first_theta = kinds.index("theta")
    for index, _kind in passes[first_solver:first_theta]:
        raised = _Sentinel(f"stop at {index}")

        def on_batch(progress, index=index, raised=raised) -> None:
            if progress.pass_index == index:
                raise raised

        with pytest.raises(_Sentinel) as caught:
            rs.glm_dict(data=source(), theta="estimate", **glm).fit(
                out_of_core=rs.OutOfCoreOptions(scratch_dir=tmp_path, on_batch=on_batch)
            )
        assert caught.value is raised
        assert list(tmp_path.iterdir()) == []


# ── Golden cases: both kernels, both modes, 1 and 4 threads ─────────────────────────


def _source(frame: pl.DataFrame, sizes=(997, 4096, 1, 3001)) -> rs.BatchSource:
    def open_batches(columns):
        selected = frame.select(columns)
        start, i = 0, 0
        while start < selected.height:
            size = sizes[i % len(sizes)]
            yield selected.slice(start, size)
            start += size
            i += 1

    return rs.BatchSource(open_batches, frame.schema)


def _case_bits(result: Any, robust: list[str], holdout: Any) -> dict[str, Any]:
    fields = {name: _bits(_read(result, name)) for name in RESULT_FIELDS}
    fields["solver_trace"] = _bits(_read(result, "solver_trace"))
    for hc_type in robust:
        fields[f"bse_robust_{hc_type}"] = _bits(result.bse_robust(hc_type))
    if holdout is not None:
        fields["predict_holdout"] = _bits(np.asarray(result.predict(holdout)))
    return fields


#: The bindings that take ``row_kernel=`` (every fit that reduces rows).
ROW_KERNEL_BINDINGS = (
    "fit_glm_py",
    "fit_negbinomial_py",
    "fit_smooth_glm_unified_py",
    "fit_cv_path_py",
    "fold_path_py",
    "fit_glm_batched_py",
    "fit_smooth_glm_batched_py",
    "fit_negbinomial_batched_py",
    "alpha_max_batched_py",
    "fold_alpha_max_batched_py",
    "fit_cv_path_batched_py",
    "fold_path_batched_py",
)


def force_row_kernel(kernel: str) -> Counter[str]:
    """Make every fit binding run with ``row_kernel=kernel``, wherever RustyStats
    imported it from; returns the calls made through each, for the caller to check.
    For a subprocess that runs nothing else: the bindings stay replaced."""
    import rustystats._rustystats as native
    import rustystats.formula
    import rustystats.out_of_core_fit
    import rustystats.regularization_path  # noqa: F401

    calls: Counter[str] = Counter()
    for name in ROW_KERNEL_BINDINGS:
        original = getattr(native, name)

        def forced(*args: Any, _original=original, _name=name, **kwargs: Any) -> Any:
            calls[_name] += 1
            kwargs["row_kernel"] = kernel
            return _original(*args, **kwargs)

        for module in list(sys.modules.values()):
            module_name = getattr(module, "__name__", "")
            if module_name == "rustystats" or module_name.startswith("rustystats."):
                if getattr(module, name, None) is original:
                    setattr(module, name, forced)
    return calls


def golden_bits(threads: int, scratch: str, kernel: str) -> dict[str, Any]:
    """Every golden case fitted in memory and out of core with ``kernel``'s rows (run
    in a subprocess)."""
    calls = force_row_kernel(kernel)
    out: dict[str, Any] = {}
    for case in CASES:
        glm, fit, holdout = case.build()
        fit = {k: v for k, v in fit.items() if k != "store_design_matrix"}
        robust = ["HC0", "HC1", "HC2", "HC3"] if fit.get("compute_covariance", True) else []
        in_memory = rs.glm_dict(**glm).fit(robust=robust, **fit)
        out[f"{case.name}/memory"] = _case_bits(in_memory, robust, holdout)
        glm, _fit, _holdout = case.build()
        frame = glm.pop("data")
        try:
            out_of_core = rs.glm_dict(data=_source(frame), **glm).fit(
                robust=robust,
                out_of_core=rs.OutOfCoreOptions(
                    encode_rows=4096, threads=threads, scratch_dir=scratch
                ),
                **fit,
            )
        except OutOfCoreOptionError as error:
            out[f"{case.name}/out_of_core"] = {"refused": str(error)}
            continue
        out[f"{case.name}/out_of_core"] = _case_bits(out_of_core, robust, holdout)
    return {"bits": out, "calls": dict(calls)}


def _golden_run(kernel: str, threads: int, scratch: Path) -> dict[str, Any]:
    script = (
        "import json, sys; sys.path.insert(0, 'tests/python');"
        "import test_out_of_core_kernels as t;"
        f"print(json.dumps(t.golden_bits({threads}, {str(scratch)!r}, {kernel!r})))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
        env={**os.environ, "RAYON_NUM_THREADS": str(threads)},
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    run = json.loads(completed.stdout.strip().splitlines()[-1])
    # Both modes' fits went through the forced kernel.
    assert run["calls"].get("fit_glm_py", 0) > 0, run["calls"]
    assert run["calls"].get("fit_glm_batched_py", 0) > 0, run["calls"]
    assert len(run["calls"]) >= 8, run["calls"]
    return run["bits"]


def test_golden_cases_give_the_same_bits_with_either_kernel_at_1_and_4_threads(
    tmp_path,
) -> None:
    runs = {
        (kernel, threads): _golden_run(kernel, threads, tmp_path)
        for kernel in ("dense", "sparse")
        for threads in (1, 4)
    }
    reference = runs[("dense", 1)]
    for key, run in runs.items():
        assert run.keys() == reference.keys(), key
        for name in reference:
            assert run[name] == reference[name], (key, name)
    # Out of core gives the in-memory bits for every field a rowless result serves.
    for case in CASES:
        memory = reference[f"{case.name}/memory"]
        out_of_core = reference[f"{case.name}/out_of_core"]
        if "refused" in out_of_core:
            continue
        for field, value in out_of_core.items():
            if isinstance(value, dict) and set(value) == {"unavailable"}:
                continue
            assert value == memory[field], (case.name, field)
    assert list(tmp_path.iterdir()) == []
