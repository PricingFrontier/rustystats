"""The solver trace (``specs/out-of-core-fit.md`` §4.5, §14 T5, T16).

Each branch of the drivers shows in ``solver_trace``, and the trace is the same for
the in-memory and the out-of-core fit of the same data (and, in a subprocess, at 1
and 4 threads). The golden cases' trace equality is pinned in
``test_out_of_core_fit.py`` (T3) and across target-encoding sort runs in
``test_target_encoding_out_of_core.py`` (T20).
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from typing import Any

import polars as pl
import pytest
import rustystats as rs
from _golden_cases import CASES, design_frame

_CASES = {case.name: case for case in CASES}


def _source(frame: pl.DataFrame) -> rs.BatchSource:
    def open_batches(columns):
        selected = frame.select(columns)
        for start in range(0, selected.height, 1777):
            yield selected.slice(start, 1777)

    return rs.BatchSource(open_batches, frame.schema)


def _both(glm: dict[str, Any], fit: dict[str, Any], tmp_path) -> tuple:
    """The in-memory and out-of-core fits of one model; their traces must agree."""
    frame = glm.pop("data")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        in_memory = rs.glm_dict(data=frame, **glm).fit(**fit)
        out_of_core = rs.glm_dict(data=_source(frame), **glm).fit(
            out_of_core=rs.OutOfCoreOptions(encode_rows=4096, scratch_dir=tmp_path), **fit
        )
    assert out_of_core.solver_trace == in_memory.solver_trace
    assert all(isinstance(step, rs.SolverStep) for step in in_memory.solver_trace)
    return in_memory, out_of_core


def _golden(name: str, tmp_path, **fit_overrides: Any) -> tuple:
    glm, fit, _holdout = _CASES[name].build()
    fit = {k: v for k, v in fit.items() if k != "store_design_matrix"}
    fit.update(fit_overrides)
    return _both(glm, fit, tmp_path)


def _steps(result, kind: str) -> list[rs.SolverStep]:
    return [step for step in result.solver_trace if step.kind == kind]


# ── T5: IRLS and coordinate-descent branches ────────────────────────────────


def test_accepted_halving(tmp_path) -> None:
    result, _ = _golden("halving_first_iteration", tmp_path)
    halved = [step for step in _steps(result, "iteration") if step.status == "halved"]
    # The full step, then the first halving, which was accepted.
    assert halved[0] == rs.SolverStep("irls", "iteration", 2, True, "halved")
    assert result.step_halving_used


def test_exhausted_halving(tmp_path) -> None:
    result, _ = _golden("halving_exhausted", tmp_path)
    # The full step and all 25 halvings were read; none was acceptable.
    assert rs.SolverStep("irls", "iteration", 26, False, "exhausted") in result.solver_trace
    assert result.solver_trace[-1] == rs.SolverStep(
        "irls", "result", result.iterations, False, "step_halving_no_improvement"
    )
    assert len(_steps(result, "iteration")) == result.iterations + 1


def test_constrained_best_restored(tmp_path) -> None:
    glm = {
        "response": "y_pois",
        "terms": {
            "x1": {"type": "linear", "monotonicity": "decreasing"},
            "cat_a": {"type": "categorical"},
            "x2": {"type": "linear"},
        },
        "data": design_frame(3000, seed=1),
        "family": "poisson",
    }
    result, _ = _both(glm, {}, tmp_path)
    assert rs.SolverStep("irls", "restore", 0, True, "constrained_best") in result.solver_trace
    # The restore decides before the final extraction.
    kinds = [step.kind for step in result.solver_trace]
    assert kinds.index("restore") < kinds.index("extraction")


def test_final_extraction_rejected_and_skipped(tmp_path) -> None:
    result, _ = _golden("poisson_linear_cat_exposure", tmp_path)
    assert _steps(result, "extraction") == [
        rs.SolverStep("irls", "extraction", 1, False, "rejected")
    ]
    skipped, _ = _golden("no_covariance", tmp_path)
    assert _steps(skipped, "extraction") == []


def test_robust_types_change_no_step(tmp_path) -> None:
    plain, _ = _golden("robust_hc_all", tmp_path)
    robust, _ = _golden("robust_hc_all", tmp_path, robust=["HC0", "HC1", "HC2", "HC3"])
    assert robust.solver_trace == plain.solver_trace


def test_coordinate_descent_with_one_iteration_cold_and_warm(tmp_path) -> None:
    cold, _ = _golden("lasso_fixed", tmp_path, max_iter=1)
    assert [step.kind for step in cold.solver_trace] == ["iteration", "result"]
    assert cold.solver_trace[-1] == rs.SolverStep("cd", "result", 1, False, "max_iterations")
    # In CV each fold fit after the first alpha is warm-started; each is one step.
    warm, _ = _golden("cv_lasso_min", tmp_path, cv_max_iter=1)
    folds = _steps(warm, "fold_alpha")
    assert folds and all(step.evaluations == 1 for step in folds)
    assert {step.status for step in folds} == {"max_iterations"}
    assert [step.driver for step in warm.solver_trace[:2]] == ["alpha_max", "alpha_max"]
    assert warm.solver_trace[-1].driver == "cd"


# ── T16: penalised smooth terms ─────────────────────────────────────────────


def test_smooth_gcv_and_fixed_penalty(tmp_path) -> None:
    gcv, _ = _golden("smooth_gcv", tmp_path)
    updates = _steps(gcv, "lambda_update")
    assert updates and updates[0].accepted and {s.status for s in updates} == {"gcv"}
    assert gcv.solver_trace[-1] == rs.SolverStep(
        "smooth", "result", gcv.iterations, True, "converged"
    )

    fixed, _ = _golden("smooth_fixed_penalty", tmp_path)
    # λ is pinned: the update leaves it where it is.
    assert all(not step.accepted for step in _steps(fixed, "lambda_update"))
    assert fixed.solver_trace[-1].kind == "result"


def test_intercept_refresh_and_a_stalled_fit(tmp_path) -> None:
    """A loose tolerance stops the smooth loop with an unpenalized linear term's
    score far from zero: the refresh restores the intercept, and the fit reports
    stalled_nonstationary."""
    from test_termination_contract import make_data

    glm = {
        "response": "y",
        "terms": {
            "x_q2i": {"type": "linear"},
            "x_net": {"type": "ns"},
            "x_qts": {"type": "bs"},
        },
        "data": make_data(n=3000, seed=5),
        "family": "binomial",
    }
    result, _ = _both(glm, {"tol": 0.1}, tmp_path)
    refresh = _steps(result, "refresh")
    assert len(refresh) == 1 and refresh[0].status == "applied" and refresh[0].accepted
    # The score at 0, at least one bracket pair, the bisection, and the deviance at
    # the shift.
    assert refresh[0].evaluations > 4
    assert result.solver_trace[-1] == rs.SolverStep(
        "smooth", "result", result.iterations, False, "stalled_nonstationary"
    )


def test_monotone_fit_records_its_phases(tmp_path) -> None:
    result, _ = _golden("monotone_smooth", tmp_path)
    kinds = [step.kind for step in result.solver_trace]
    assert kinds[0] == "lambda_update" and "warm_start" in kinds and "alpha_reset" in kinds
    assert {step.driver for step in result.solver_trace} == {"monotone"}
    assert kinds[-1] == "result"


def test_theta_estimation_records_its_rounds(tmp_path) -> None:
    result, _ = _golden("negbinomial_theta_estimate", tmp_path)
    rounds = _steps(result, "round")
    assert _steps(result, "initial_theta") == [
        rs.SolverStep("nb_theta", "initial_theta", 1, True, "moments")
    ]
    assert len(rounds) == result.solver_settings["theta_rounds"]
    assert rounds[-1].status == "converged" and all(step.evaluations > 0 for step in rounds)


# ── Thread independence ─────────────────────────────────────────────────────

_THREAD_SCRIPT = """
import sys, tempfile, warnings
sys.path.insert(0, "tests/python")
warnings.simplefilter("ignore")
import rustystats as rs
from test_solver_trace import _CASES, _source
for name in ("halving_first_iteration", "smooth_gcv", "cv_lasso_min", "target_encoding_cv"):
    glm, fit, _ = _CASES[name].build()
    frame = glm.pop("data")
    print(name, rs.glm_dict(data=frame, **glm).fit(**fit).solver_trace)
    with tempfile.TemporaryDirectory() as scratch:
        options = rs.OutOfCoreOptions(encode_rows=4096, scratch_dir=scratch)
        result = rs.glm_dict(data=_source(frame), **glm).fit(out_of_core=options, **fit)
    print(name, result.solver_trace)
"""


@pytest.mark.slow
def test_the_trace_does_not_depend_on_thread_count() -> None:
    outputs = set()
    for threads in ("1", "4"):
        completed = subprocess.run(
            [sys.executable, "-c", _THREAD_SCRIPT],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "RAYON_NUM_THREADS": threads},
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        )
        outputs.add(completed.stdout)
    assert len(outputs) == 1
    lines = outputs.pop().splitlines()
    # Each case's in-memory and out-of-core traces agree too.
    assert lines[0::2] == lines[1::2]
