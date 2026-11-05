"""An out-of-core fit gives the in-memory fit bit for bit (``specs/out-of-core-fit.md`` §3).

Each golden case the out-of-core path runs is fitted twice: in memory from the
DataFrame, and from a ``BatchSource`` over the same rows in uneven batches. Every
recorded field, the robust covariances and the holdout predictions must agree
exactly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest
import rustystats as rs
from _golden_cases import CASES, RESULT_FIELDS, GoldenCase, _read
from rustystats.exceptions import (
    BatchSourceChangedError,
    OutOfCoreOptionError,
    OutOfCoreResultError,
)

#: Fields a rowless result serves (§4.5); the rest must refuse with OutOfCoreResultError.
ROWLESS_FIELDS = [field for field in RESULT_FIELDS if field not in ("cv_deviance",)]


def _source(frame: pl.DataFrame, sizes=(997, 4096, 1, 3001)) -> rs.BatchSource:
    """A source over ``frame`` whose batches cycle through ``sizes`` rows."""
    opened = {"passes": 0}

    def open_batches(columns):
        opened["passes"] += 1
        selected = frame.select(columns)
        start, i = 0, 0
        while start < selected.height:
            size = sizes[i % len(sizes)]
            yield selected.slice(start, size)
            start += size
            i += 1

    source = rs.BatchSource(open_batches, frame.schema)
    source.opened = opened
    return source


#: Every golden case runs out of core (§1); array inputs are compared with their
#: columns in ``test_out_of_core_contract.py``.
SUPPORTED = list(CASES)


def _bits(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"unavailable"}:
        return value
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, np.ndarray) and value.dtype.kind == "f":
        return [float(v).hex() for v in value.ravel()]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [_bits(v) for v in value]
    if isinstance(value, dict):
        return {k: _bits(v) for k, v in value.items()}
    return value


@pytest.mark.parametrize("case", SUPPORTED, ids=[case.name for case in SUPPORTED])
def test_out_of_core_fit_equals_in_memory_fit_bitwise(case: GoldenCase, tmp_path) -> None:
    glm, fit, holdout = case.build()
    fit = {k: v for k, v in fit.items() if k != "store_design_matrix"}
    robust = ["HC0", "HC1", "HC2", "HC3"] if fit.get("compute_covariance", True) else []
    in_memory = rs.glm_dict(**glm).fit(robust=robust, **fit)

    glm, _fit, _holdout = case.build()
    frame = glm.pop("data")
    out_of_core = rs.glm_dict(data=_source(frame), **glm).fit(
        robust=robust,
        out_of_core=rs.OutOfCoreOptions(encode_rows=4096, scratch_dir=tmp_path),
        **fit,
    )
    assert list(tmp_path.iterdir()) == [], "spill files are removed"

    for field in ROWLESS_FIELDS:
        assert _bits(_read(out_of_core, field)) == _bits(_read(in_memory, field)), field
    # §4.5: the drivers' steps are the same in both modes (T3).
    assert out_of_core.solver_trace == in_memory.solver_trace
    assert in_memory.solver_trace and all(
        isinstance(step, rs.SolverStep) for step in in_memory.solver_trace
    )
    for hc_type in robust:
        assert _bits(out_of_core.bse_robust(hc_type)) == _bits(in_memory.bse_robust(hc_type))
    if holdout is not None:
        assert _bits(np.asarray(out_of_core.predict(holdout))) == _bits(
            np.asarray(in_memory.predict(holdout))
        )
    with pytest.raises(OutOfCoreResultError):
        out_of_core._result.fittedvalues  # noqa: B018
    assert out_of_core.memory_plan.encode_rows == 4096


def test_progress_reports_every_pass_and_the_source_is_fingerprinted() -> None:
    case = next(case for case in SUPPORTED if case.name.startswith("poisson"))
    glm, fit, _holdout = case.build()
    frame = glm.pop("data")
    seen: list[rs.BatchProgress] = []
    model = rs.glm_dict(data=_source(frame), **glm)
    result = model.fit(out_of_core=rs.OutOfCoreOptions(on_batch=seen.append), **fit)
    kinds = [progress.pass_kind for progress in seen]
    assert kinds[0] == "freeze" and "candidate" in kinds and kinds[-1] == "final"
    assert seen[0].n_rows is None and seen[-1].n_rows == frame.height
    assert sum(result.passes.values()) == len({p.pass_index for p in seen})

    # A source whose rows change after the freeze pass is refused.
    calls = {"n": 0}

    def changing(columns):
        calls["n"] += 1
        data = frame if calls["n"] == 1 else frame.with_columns(pl.col(glm["response"]) * 2)
        yield data.select(columns)

    with pytest.raises(BatchSourceChangedError, match="changed between passes"):
        rs.glm_dict(data=rs.BatchSource(changing, frame.schema), **glm).fit(
            out_of_core=rs.OutOfCoreOptions(), **fit
        )


def test_an_exception_from_on_batch_propagates_unchanged() -> None:
    case = SUPPORTED[0]
    glm, fit, _holdout = case.build()
    frame = glm.pop("data")

    class Stop(Exception):
        pass

    def on_batch(progress):
        if progress.pass_kind == "candidate":
            raise Stop("cancelled")

    with pytest.raises(Stop, match="cancelled"):
        rs.glm_dict(data=_source(frame), **glm).fit(
            out_of_core=rs.OutOfCoreOptions(on_batch=on_batch), **fit
        )


def test_out_of_core_option_refusals() -> None:
    case = SUPPORTED[0]
    glm, _fit, _holdout = case.build()
    frame = glm.pop("data")
    model = rs.glm_dict(data=_source(frame), **glm)
    with pytest.raises(OutOfCoreOptionError, match="store_design_matrix"):
        model.fit(store_design_matrix=True, out_of_core=rs.OutOfCoreOptions())
    in_memory = rs.glm_dict(data=frame, **glm)
    with pytest.raises(OutOfCoreOptionError, match="BatchSource"):
        in_memory.fit(out_of_core=rs.OutOfCoreOptions())


def test_observed_category_labels_from_a_source_match_the_frame() -> None:
    frame = pl.DataFrame(
        {
            "region": ["b", None, "a", "c", "a"] * 900,
            "band": pl.Series([2, None, 1, 3, 2] * 900, dtype=pl.Int64),
            "kind": pl.Series(["x", "y", "x", None, "z"] * 900, dtype=pl.Categorical),
            "unused": list(range(4500)),
        }
    )
    columns = ["region", "band", "kind"]
    read: list[list[str]] = []

    def open_batches(cols):
        read.append(list(cols))
        for start in range(0, frame.height, 1000):
            yield frame.select(cols).slice(start, 1000)

    progress: list[rs.BatchProgress] = []
    from_source = rs.observed_category_labels(
        rs.BatchSource(open_batches, frame.schema),
        columns,
        out_of_core=rs.OutOfCoreOptions(on_batch=progress.append),
    )
    assert from_source == rs.observed_category_labels(frame, columns)
    assert from_source["band"] == ["1.0", "2.0", "3.0"]
    assert read == [columns]
    assert {p.pass_kind for p in progress} == {"labels"}


def test_theta_estimation_needs_a_scratch_dir() -> None:
    case = next(case for case in SUPPORTED if case.name == "negbinomial_theta_estimate")
    glm, fit, _holdout = case.build()
    frame = glm.pop("data")
    source = _source(frame)
    with pytest.raises(OutOfCoreOptionError, match="scratch_dir"):
        rs.glm_dict(data=source, **glm).fit(out_of_core=rs.OutOfCoreOptions(), **fit)
    assert source.opened["passes"] == 0


def test_a_model_complement_equals_in_memory_bitwise() -> None:
    """A fitted GLMModel complement is predicted per slice; with predict's row kernel
    its values do not depend on the slicing (§6.8)."""
    from _golden_cases import design_frame

    frame = design_frame(9000, seed=7)
    prior = rs.glm_dict(
        response="y_pois",
        terms={"x1": {"type": "linear"}, "cat_a": {"type": "categorical"}},
        data=frame,
        family="poisson",
        exposure="exposure",
    ).fit()
    glm = {
        "response": "y_pois",
        "terms": {"x2": {"type": "bs", "df": 4}, "cat_b": {"type": "categorical"}},
        "family": "poisson",
        "exposure": "exposure",
        "complement": prior,
    }
    in_memory = rs.glm_dict(data=frame, **glm).fit(alpha=0.5)
    out_of_core = rs.glm_dict(data=_source(frame), **glm).fit(
        alpha=0.5, out_of_core=rs.OutOfCoreOptions(encode_rows=4096)
    )
    for field in ("params", "deviance", "llf", "bse"):
        assert _bits(_read(out_of_core, field)) == _bits(_read(in_memory, field)), field


def _high_cardinality_frame(rows: int = 400_000) -> pl.DataFrame:
    rng = np.random.default_rng(12)
    return pl.DataFrame(
        {
            "key": [f"k{index:029d}" for index in range(rows)],
            "x": rng.normal(size=rows),
            "y": rng.normal(size=rows),
        }
    )


@pytest.mark.parametrize("route", ["labels", "freeze"])
def test_max_bytes_alone_caps_learned_state(route: str) -> None:
    """§6.4: with max_learned_state_bytes unset, learned state still fits what max_bytes
    leaves; 400,000 distinct 30-byte labels do not fit beside 100 MiB's other state."""
    from rustystats.exceptions import LearnedStateTooLargeError

    frame = _high_cardinality_frame()
    options = rs.OutOfCoreOptions(max_bytes=100 * 1024**2)
    with pytest.raises(LearnedStateTooLargeError, match="set by max_bytes = 104857600"):
        if route == "labels":
            rs.observed_category_labels(_source(frame), ["key"], out_of_core=options)
        else:
            rs.glm_dict(
                response="y",
                terms={"x": {"type": "linear"}, "key": {"type": "categorical"}},
                data=_source(frame),
                family="gaussian",
            ).fit(out_of_core=options)
    # Unbounded, the same labels are learned.
    labels = rs.observed_category_labels(_source(frame), ["key"])
    assert len(labels["key"]) == frame.height


def _metadata_frame(rows: int = 6_000) -> pl.DataFrame:
    rng = np.random.default_rng(21)
    x = rng.uniform(0.0, 3.0, size=rows)
    return pl.DataFrame(
        {"x": x, "z": rng.normal(size=rows), "y": rng.poisson(np.exp(0.2 + 0.4 * x)).astype(float)}
    )


@pytest.mark.parametrize(
    ("terms", "fit", "max_iter"),
    [
        ({"x": {"type": "linear"}, "z": {"type": "linear"}}, {}, 25),
        ({"x": {"type": "linear"}}, {"max_iter": 40, "tol": 1e-6}, 40),
        ({"x": {"type": "bs", "k": 8, "monotonicity": "increasing"}}, {}, 2_000),
    ],
    ids=["defaults", "configured", "monotone-smooth"],
)
def test_both_modes_report_how_the_fit_ran(terms, fit, max_iter) -> None:
    """§4.5: fit_mode, the I/O passes per kind, and the caps the solve used."""
    from rustystats.out_of_core import GRAM_BLOCK_ROWS

    frame = _metadata_frame()
    glm = {"response": "y", "terms": terms, "family": "poisson"}
    in_memory = rs.glm_dict(data=frame, **glm).fit(**fit)
    out_of_core = rs.glm_dict(data=_source(frame), **glm).fit(**fit)

    assert in_memory.fit_mode == "in_memory"
    assert out_of_core.fit_mode == "out_of_core"
    assert in_memory.passes == {}
    assert out_of_core.passes["freeze"] == 1
    assert out_of_core.passes["candidate"] >= 1
    assert out_of_core.solver_settings == in_memory.solver_settings
    assert in_memory.solver_settings == {
        "max_iter": max_iter,
        "tol": fit.get("tol", 1e-8),
        "max_half_steps": 25,
        "gram_block_rows": GRAM_BLOCK_ROWS,
        "target_encoding_seed": None,
    }


def test_an_estimated_theta_records_its_rounds(tmp_path) -> None:
    frame = _metadata_frame()
    glm = {"response": "y", "terms": {"x": {"type": "linear"}}, "family": "negbinomial"}
    in_memory = rs.glm_dict(data=frame, theta="estimate", **glm).fit()
    out_of_core = rs.glm_dict(data=_source(frame), theta="estimate", **glm).fit(
        out_of_core=rs.OutOfCoreOptions(scratch_dir=tmp_path)
    )

    settings = in_memory.solver_settings
    assert out_of_core.solver_settings == settings
    assert settings["theta_rounds"] == in_memory.theta_metadata["theta_iterations"] >= 1
    assert settings["theta_tol"] == in_memory.theta_metadata["theta_tol"]
    assert settings["max_theta_iter"] == in_memory.theta_metadata["max_theta_iter"]
