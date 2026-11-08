"""The contract of an out-of-core fit around its passes (``specs/out-of-core-fit.md``).

- T10: an exception ``on_batch`` raises in any pass kind reaches the caller unchanged;
  the reader is closed, no later batch is requested and no spill file is left.
- T11: refusals the options and the schema decide come before the source is opened;
  ``max_design_columns`` and ``max_learned_state_bytes`` stop learning while the state
  grows, the long-label cases included; ``max_bytes`` refuses each phase before its
  passes (§4.3, §6.4, §11).
- T12: an out-of-core result refuses every per-row method, naming it, serves the rest,
  and round-trips through ``to_bytes`` (§4.5).
- §6.8: exposure, offset, weights and a complement given as arrays equal the same
  columns; a non-finite complement is refused as in memory.
"""

from __future__ import annotations

import pickle
import re
import tracemalloc
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import pytest
import rustystats as rs
from _golden_cases import CASES, RESULT_FIELDS, _read, design_frame
from rustystats import exceptions
from rustystats.exceptions import (
    DesignTooWideError,
    LearnedStateTooLargeError,
    MemoryPlanExceededError,
    OutOfCoreOptionError,
    OutOfCoreResultError,
    SpillError,
    ValidationError,
)
from rustystats.out_of_core import PASS_KINDS

# ── A recording source ──────────────────────────────────────────────────────


@dataclass
class _Log:
    opens: int = 0
    batches: int = 0
    closed: int = 0
    #: Set by an ``on_batch`` that raises; a batch produced after it is counted.
    aborted: bool = False
    after_abort: int = 0


class _Reader:
    """An owning reader: ``__iter__`` returns a generator, and only the reader's own
    ``close()`` releases it (§4.1)."""

    def __init__(self, log: _Log, frame: pl.DataFrame, sizes: tuple[int, ...]) -> None:
        self.log = log
        self.frame = frame
        self.sizes = sizes

    def __iter__(self):
        start, i = 0, 0
        while start < self.frame.height:
            if self.log.aborted:
                self.log.after_abort += 1
            self.log.batches += 1
            size = self.sizes[i % len(self.sizes)]
            yield self.frame.slice(start, size)
            start += size
            i += 1

    def close(self) -> None:
        self.log.closed += 1


def _source(
    frame: pl.DataFrame, sizes: tuple[int, ...] = (997, 2048, 1, 1501)
) -> tuple[rs.BatchSource, _Log]:
    log = _Log()

    def open_batches(columns):
        log.opens += 1
        return _Reader(log, frame.select(columns), sizes)

    return rs.BatchSource(open_batches, frame.schema), log


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


FIELDS = [field for field in RESULT_FIELDS if field != "cv_deviance"]


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    return design_frame(6000, seed=11)


# ── T10: an abort in every pass kind ────────────────────────────────────────


class Sentinel(ValueError):
    """A ValueError ``on_batch`` raises: never a failed alpha or a numerical failure."""


def _model(name: str, frame: pl.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    if name == "learning":
        glm = {
            "response": "y_pois",
            "terms": {
                "x1": {"type": "linear"},
                "x2": {"type": "bs", "df": 6},
                "cat_a": {"type": "categorical"},
                "brand": {"type": "target_encoding"},
            },
            "family": "poisson",
            "exposure": "exposure",
            "seed": 3,
        }
        return {**glm, "data": frame}, {"alpha": 0.1}
    if name == "cv":
        glm = {
            "response": "y_pois",
            "terms": {"x1": {"type": "linear"}, "cat_a": {"type": "categorical"}},
            "family": "poisson",
            "exposure": "exposure",
        }
        return {**glm, "data": frame}, {"cv": 3, "regularization": "lasso", "n_alphas": 3}
    if name == "theta":
        glm = {"response": "y_nb", "terms": {"x1": {"type": "linear"}}, "family": "negbinomial"}
        return {**glm, "theta": "estimate", "data": frame}, {}
    if name == "halving":
        case = next(case for case in CASES if case.name == "halving_first_iteration")
        glm, fit, _holdout = case.build()
        return glm, {k: v for k, v in fit.items() if k != "store_design_matrix"}
    if name == "refresh":
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
        return glm, {"tol": 0.1}
    raise AssertionError(name)


#: The model whose passes include each pass kind.
KIND_MODEL = {
    "freeze": "learning",
    "knots": "learning",
    "te_stats": "learning",
    "te_sort": "learning",
    "te_merge": "learning",
    "te_walk": "learning",
    "te_average": "learning",
    "moments": "learning",
    "start": "learning",
    "candidate": "learning",
    "reconstruct": "learning",
    "final": "learning",
    "gram": "theta",
    "fold": "cv",
    "null_fit": "cv",
    "theta": "theta",
    "trials": "halving",
    "refresh": "refresh",
}


def test_every_pass_kind_is_covered() -> None:
    # The labels pass belongs to observed_category_labels (tested below).
    assert set(KIND_MODEL) == set(PASS_KINDS) - {"labels"}


@pytest.mark.parametrize("kind", sorted(KIND_MODEL))
def test_an_abort_in_any_pass_kind_reaches_the_caller_unchanged(frame, tmp_path, kind) -> None:
    glm, fit = _model(KIND_MODEL[kind], frame)
    data = glm.pop("data")
    source, log = _source(data)
    raised: list[BaseException] = []
    seen: list[str] = []

    def on_batch(progress: rs.BatchProgress) -> None:
        seen.append(progress.pass_kind)
        if progress.pass_kind == kind:
            log.aborted = True
            raised.append(Sentinel(kind))
            raise raised[-1]

    with pytest.raises(Sentinel) as caught:
        rs.glm_dict(data=source, **glm).fit(
            out_of_core=rs.OutOfCoreOptions(
                scratch_dir=tmp_path, te_run_records=2_000, on_batch=on_batch
            ),
            **fit,
        )
    assert caught.value is raised[0]
    assert seen[-1] == kind
    assert log.after_abort == 0, "no batch is requested after the abort"
    assert log.closed == log.opens, "every reader is closed"
    assert list(tmp_path.iterdir()) == [], "no spill file is left"


def test_an_abort_in_the_labels_pass(frame) -> None:
    source, log = _source(frame)
    raised = Sentinel("labels")

    def on_batch(progress: rs.BatchProgress) -> None:
        assert progress.pass_kind == "labels"
        log.aborted = True
        raise raised

    with pytest.raises(Sentinel) as caught:
        rs.observed_category_labels(
            source, ["brand"], out_of_core=rs.OutOfCoreOptions(on_batch=on_batch)
        )
    assert caught.value is raised
    assert log.after_abort == 0 and log.closed == log.opens == 1


# ── T11: refusals before any I/O ────────────────────────────────────────────

_LINEAR = {"x1": {"type": "linear"}}
_POISSON = {"response": "y_pois", "family": "poisson"}

REFUSED_BEFORE_IO = [
    (
        "nested categorical (DEC-5)",
        {"terms": {"tags": {"type": "categorical"}}},
        {},
        {},
        ValidationError,
        "cannot be a categorical",
    ),
    ("unknown link", {"link": "probit"}, {}, {}, ValidationError, "Unknown link"),
    (
        "one fold",
        {},
        {"cv": 1, "regularization": "ridge"},
        {},
        ValidationError,
        "at least 2 folds",
    ),
    (
        "unknown selection",
        {},
        {"cv": 3, "regularization": "ridge", "selection": "max"},
        {},
        ValidationError,
        "selection",
    ),
    ("negative max_iter", {}, {"max_iter": -1}, {}, ValidationError, "max_iter"),
    (
        "a Tweedie power without a distribution",
        {"response": "y_tweedie", "family": "tweedie", "var_power": 0.5},
        {},
        {},
        ValidationError,
        "no Tweedie distribution",
    ),
    (
        "L1 with a smooth term",
        {"terms": {"x2": {"type": "bs", "k": 6}}},
        {"alpha": 0.1, "l1_ratio": 0.5},
        {},
        ValidationError,
        "L1/elastic-net",
    ),
    (
        "cross-validation with a smooth term",
        {"terms": {"x2": {"type": "bs", "k": 6}}},
        {"cv": 3, "regularization": "ridge"},
        {},
        ValidationError,
        "CV-based regularization",
    ),
    (
        "θ estimation with a smooth term",
        {
            "response": "y_nb",
            "family": "negbinomial",
            "theta": "estimate",
            "terms": {"x2": {"type": "bs", "k": 6}},
        },
        {},
        {"scratch_dir": "TMP"},
        ValidationError,
        "theta estimation is not supported for smooth",
    ),
    (
        "θ estimation with a monotone linear term",
        {
            "response": "y_nb",
            "family": "negbinomial",
            "theta": "estimate",
            "terms": {"x1": {"type": "linear", "monotonicity": "increasing"}},
        },
        {},
        {"scratch_dir": "TMP"},
        ValidationError,
        "theta estimation is not supported for sign-constrained",
    ),
    (
        "θ estimation with a fixed-df monotone spline",
        {
            "response": "y_nb",
            "family": "negbinomial",
            "theta": "estimate",
            "terms": {"x2": {"type": "bs", "df": 5, "monotonicity": "decreasing"}},
        },
        {},
        {"scratch_dir": "TMP"},
        ValidationError,
        "theta estimation is not supported for sign-constrained",
    ),
    (
        "robust without the covariance",
        {},
        {"robust": ["HC1"], "compute_covariance": False},
        {},
        ValidationError,
        "compute_covariance=False",
    ),
    (
        "a stored design",
        {},
        {"store_design_matrix": True},
        {},
        OutOfCoreOptionError,
        "store_design_matrix",
    ),
    (
        "target encoding without a scratch directory",
        {"terms": {"brand": {"type": "target_encoding"}}},
        {},
        {},
        OutOfCoreOptionError,
        "scratch_dir",
    ),
    (
        "θ estimation without a scratch directory",
        {"response": "y_nb", "family": "negbinomial", "theta": "estimate"},
        {},
        {},
        OutOfCoreOptionError,
        "scratch_dir",
    ),
    (
        "a scratch directory that does not exist",
        {"terms": {"brand": {"type": "target_encoding"}}},
        {},
        {"scratch_dir": "MISSING"},
        SpillError,
        "not a directory",
    ),
    (
        "knots beyond max_learned_state_bytes",
        {"terms": {"x2": {"type": "bs", "df": 6}}},
        {},
        {"max_learned_state_bytes": 1},
        LearnedStateTooLargeError,
        "knots of x2",
    ),
    (
        "columns that need no data beyond max_design_columns",
        {"terms": {"x1": {"type": "linear"}, "x3": {"type": "linear"}, "x4": {"type": "linear"}}},
        {},
        {"max_design_columns": 3},
        DesignTooWideError,
        "x4",
    ),
]


@pytest.mark.parametrize(
    ("glm", "fit", "options", "error", "match"),
    [case[1:] for case in REFUSED_BEFORE_IO],
    ids=[case[0] for case in REFUSED_BEFORE_IO],
)
def test_refusals_come_before_the_source_is_opened(
    frame, tmp_path, glm, fit, options, error, match
) -> None:
    data = frame.with_columns(tags=pl.col("cat_a").map_elements(lambda v: [v], pl.List(pl.String)))
    source, log = _source(data)
    options = {
        name: (
            tmp_path if value == "TMP" else tmp_path / "missing" if value == "MISSING" else value
        )
        for name, value in options.items()
    }
    spec = {**_POISSON, "terms": _LINEAR, **glm}
    with pytest.raises(error, match=re.escape(match) if "(" in match else match):
        rs.glm_dict(data=source, **spec).fit(out_of_core=rs.OutOfCoreOptions(**options), **fit)
    assert log.opens == 0


# ── T11: too few folds in both modes ────────────────────────────────────────


@pytest.mark.parametrize("cv", [1, 0, -3])
def test_too_few_folds_are_refused_alike_in_both_modes(frame, cv, monkeypatch) -> None:
    """A fold count below 2 raises the same ValidationError in memory as out of core,
    before any fit runs (in memory) or the source is opened (out of core)."""
    import rustystats._rustystats as rust

    def no_fit(*args, **kwargs):
        raise AssertionError("a fit ran")

    monkeypatch.setattr(rust, "fit_cv_path_py", no_fit)
    monkeypatch.setattr(rust, "fit_glm_py", no_fit)
    glm = {"response": "y_pois", "terms": _LINEAR, "family": "poisson"}
    fit = {"cv": cv, "regularization": "ridge"}
    with pytest.raises(ValidationError, match="at least 2 folds") as in_memory:
        rs.glm_dict(data=frame, **glm).fit(**fit)
    source, log = _source(frame)
    with pytest.raises(ValidationError, match="at least 2 folds") as out_of_core:
        rs.glm_dict(data=source, **glm).fit(**fit)
    assert str(out_of_core.value) == str(in_memory.value)
    assert log.opens == 0


# ── T11: DEC-5 in both modes ────────────────────────────────────────────────


def _nested(kind: str, rows: int) -> pl.Series:
    if kind == "List":
        return pl.Series("c", [[i % 3] for i in range(rows)], dtype=pl.List(pl.Int64))
    if kind == "Array":
        return pl.Series("c", [[i % 3, 1] for i in range(rows)], dtype=pl.Array(pl.Int64, 2))
    if kind == "Struct":
        return pl.Series("c", [{"a": i % 3} for i in range(rows)])
    if kind == "Object":
        return pl.Series("c", [object() for _ in range(rows)], dtype=pl.Object)
    return pl.Series("c", [None] * rows, dtype=pl.Null)


@pytest.mark.parametrize("term", ["categorical", "frequency_encoding", "target_encoding"])
@pytest.mark.parametrize("dtype", ["List", "Array", "Struct", "Object", "Null"])
def test_nested_category_inputs_are_refused_alike_in_both_modes(dtype, term) -> None:
    """DEC-5: a List, Array, Struct, Object or Null column as a categorical,
    frequency-encoded or target-encoded input raises the same ValidationError in
    memory (before any design is built) and out of core (before the source is opened)."""
    rng = np.random.default_rng(5)
    data = pl.DataFrame({"y": rng.poisson(1.0, 60).astype(float), "x": rng.normal(size=60)})
    data = data.with_columns(_nested(dtype, data.height))
    glm = {
        "response": "y",
        "terms": {"x": {"type": "linear"}, "c": {"type": term}},
        "family": "poisson",
    }
    message = "Column 'c' has dtype .* cannot be a categorical, frequency-encoded or target-encoded"
    with pytest.raises(ValidationError, match=message) as in_memory:
        rs.glm_dict(data=data, **glm)
    source, log = _source(data)
    with pytest.raises(ValidationError, match=message) as out_of_core:
        rs.glm_dict(data=source, **glm)
    assert str(out_of_core.value) == str(in_memory.value)
    assert log.opens == 0


# ── T11: limits while learned state grows ───────────────────────────────────


def _labels_frame(rows: int = 120) -> pl.DataFrame:
    rng = np.random.default_rng(4)
    return pl.DataFrame(
        {
            "c": [f"level{i:03d}" for i in range(rows)],
            "d": [f"d{i % 60:02d}" for i in range(rows)],
            "x": rng.normal(size=rows),
            "y": rng.normal(size=rows),
        }
    )


def test_max_design_columns_stops_while_levels_grow() -> None:
    """120 distinct labels in six batches of 20: the first batch already crosses two
    columns, so learning stops there (§6.4)."""
    data = _labels_frame()
    source, log = _source(data, sizes=(20,))
    with pytest.raises(DesignTooWideError, match="'c' crossed the limit"):
        rs.glm_dict(
            response="y", terms={"c": {"type": "categorical"}}, data=source, family="gaussian"
        ).fit(out_of_core=rs.OutOfCoreOptions(max_design_columns=2))
    assert log.opens == 1 and log.batches == 1


def test_max_design_columns_judges_an_interaction_from_running_levels() -> None:
    """Two factors of 60 and 120 levels: their cells cross 500 columns long before
    the source ends, and before any layout is built."""
    data = _labels_frame(1200)
    source, log = _source(data, sizes=(40,))
    with pytest.raises(DesignTooWideError, match="'d:c' crossed the limit"):
        rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}},
            interactions=[{"d": {"type": "categorical"}, "c": {"type": "categorical"}}],
            data=source,
            family="gaussian",
        ).fit(out_of_core=rs.OutOfCoreOptions(max_design_columns=500))
    assert log.batches == 1, log.batches


def test_learned_state_counts_knots_and_standardization(frame) -> None:
    """§11: knots, centres and scales are learned state under the shared budget."""
    source, log = _source(frame)
    spline = rs.glm_dict(
        response="y_pois", terms={"x2": {"type": "bs", "df": 8}}, data=source, family="poisson"
    )
    with pytest.raises(LearnedStateTooLargeError, match="knots of x2"):
        spline.fit(out_of_core=rs.OutOfCoreOptions(max_learned_state_bytes=1))
    assert log.opens == 0, "knots are charged before any pass"

    source, log = _source(frame)
    ridge = rs.glm_dict(
        response="y_pois", terms={"x1": {"type": "linear"}}, data=source, family="poisson"
    )
    with pytest.raises(LearnedStateTooLargeError, match="standardization"):
        ridge.fit(alpha=0.1, out_of_core=rs.OutOfCoreOptions(max_learned_state_bytes=1))
    assert log.opens == 1, "only the freeze pass ran"

    # Unlimited, the plan's persistent state holds them.
    plain = rs.glm_dict(
        response="y_pois",
        terms={"x1": {"type": "linear"}},
        data=_source(frame)[0],
        family="poisson",
    ).fit(alpha=0.1)
    standardized = rs.glm_dict(
        response="y_pois",
        terms={"x1": {"type": "linear"}},
        data=_source(frame)[0],
        family="poisson",
    ).fit(alpha=0.1, standardize=False)
    assert (
        plain.memory_plan.persistent_bytes - standardized.memory_plan.persistent_bytes == 3 * 8 * 2
    )


def test_high_cardinality_frequency_encoding_is_refused_while_it_grows() -> None:
    data = _labels_frame(20_000).with_columns(c=pl.format("k{}", pl.int_range(pl.len())))
    source, log = _source(data, sizes=(2_000,))
    with pytest.raises(LearnedStateTooLargeError, match="'c'"):
        rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}, "c": {"type": "frequency_encoding"}},
            data=source,
            family="gaussian",
        ).fit(out_of_core=rs.OutOfCoreOptions(max_learned_state_bytes=200_000))
    assert log.batches < 10


_MiB = 1024 * 1024
LONG_LABELS = {
    "one 16 KiB string among one-byte strings": ["a", "b"] * 500 + ["x" * 16_384],
    "a 1 MiB ASCII label": ["a", "b"] * 500 + ["x" * _MiB],
    "a 1 MiB ASCII label ending in a non-BMP character": ["a", "b"] * 500
    + ["x" * (_MiB - 4) + "\U0001f600"],
}


@pytest.mark.parametrize("values", list(LONG_LABELS.values()), ids=list(LONG_LABELS))
def test_long_labels_are_reserved_before_they_are_formed(values) -> None:
    """A label's Python string can take four bytes per character: it is reserved
    before it is formed, so a budget a little short of it refuses, and the labels
    pass's own allocations stay within its reservation (§6.3)."""
    data = pl.DataFrame({"c": values})
    longest = max(len(value.encode()) for value in values)
    with pytest.raises(LearnedStateTooLargeError, match="'c'"):
        rs.observed_category_labels(
            _source(data)[0],
            ["c"],
            out_of_core=rs.OutOfCoreOptions(max_learned_state_bytes=4 * longest),
        )
    tracemalloc.start()
    try:
        labels = rs.observed_category_labels(_source(data)[0], ["c"])
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(labels["c"]) == 3
    assert peak > 4 * longest, "the long label was formed"
    # The pass reserved at least what it allocated: a budget one byte short of its
    # traced peak (NumPy arrays and Python strings) refuses it.
    with pytest.raises(LearnedStateTooLargeError, match="'c'"):
        rs.observed_category_labels(
            _source(data)[0],
            ["c"],
            out_of_core=rs.OutOfCoreOptions(max_learned_state_bytes=peak - 1),
        )


def test_an_oversized_row_and_a_binary_label_are_charged() -> None:
    """A row larger than the learning slice forms its own slice; a 1 MiB binary value
    of 0xff bytes renders to 4 MiB of label text under ``str()`` (§6.3)."""
    rng = np.random.default_rng(0)
    data = pl.DataFrame(
        {
            "b": pl.Series([b"a", b"b"] * 500 + [b"\xff" * _MiB], dtype=pl.Binary),
            "y": rng.normal(size=1001),
        }
    )
    spec = {"response": "y", "terms": {"b": {"type": "categorical"}}, "family": "gaussian"}
    with pytest.raises(LearnedStateTooLargeError, match="'b'"):
        rs.glm_dict(data=_source(data)[0], **spec).fit(
            out_of_core=rs.OutOfCoreOptions(max_learned_state_bytes=8 * _MiB)
        )
    result = rs.glm_dict(data=_source(data)[0], **spec).fit()
    assert len(result.params) == 3


# ── T11: max_bytes refuses each phase before its passes ─────────────────────


def _numeric_frame(rows: int = 4_000, levels: int = 300) -> pl.DataFrame:
    rng = np.random.default_rng(9)
    level = rng.integers(0, levels, rows)
    return pl.DataFrame(
        {
            "x": rng.normal(size=rows),
            "z": rng.uniform(0.0, 10.0, rows),
            "level": pl.Series(level, dtype=pl.Int64),
            "y": rng.poisson(np.exp(0.1 + 0.002 * level)).astype(float),
        }
    )


def _learning_threshold() -> int:
    """The smallest ``max_bytes`` (in MiB steps) whose freeze phase is admitted."""
    from rustystats import out_of_core_fit as fit_module

    columns = 4
    limit = 0
    while True:
        limit += _MiB
        learn = fit_module._learning_peak(fit_module._learn_slice_bytes(limit))
        if fit_module._ALLOCATOR_ALLOWANCE + 64 * columns + learn <= limit:
            return limit


LEARNING_PHASES = {
    # Many interior knots: 60 ranks of 512 KiB histograms.
    "knots": ({"terms": {"z": {"type": "bs", "df": 64}}}, {}, {}),
    # A sort run of two million records (160 MB).
    "te_sort": (
        {"terms": {"level": {"type": "target_encoding"}}},
        {},
        {"te_run_records": 2_000_000},
    ),
    # The merge's 65 MiB of buffers beside a small sort run.
    "te_merge": ({"terms": {"level": {"type": "target_encoding"}}}, {}, {"te_run_records": 100}),
}


@pytest.mark.parametrize("phase", ["freeze", *LEARNING_PHASES])
def test_max_bytes_refuses_a_learning_phase_before_its_passes(tmp_path, phase) -> None:
    data = _numeric_frame()
    source, log = _source(data)
    if phase == "freeze":
        glm, fit, options = {"terms": {"x": {"type": "linear"}}}, {}, {}
        limit = _learning_threshold() - _MiB
    else:
        # Room for the learned state beside the learning passes, not for the phase.
        glm, fit, options = LEARNING_PHASES[phase]
        limit = _learning_threshold() + 4 * _MiB
    kinds: list[str] = []
    with pytest.raises(MemoryPlanExceededError, match=f"The {phase} phase"):
        rs.glm_dict(data=source, response="y", family="poisson", seed=1, **glm).fit(
            out_of_core=rs.OutOfCoreOptions(
                max_bytes=limit,
                scratch_dir=tmp_path,
                on_batch=lambda progress: kinds.append(progress.pass_kind),
                **options,
            ),
            **fit,
        )
    first = {"freeze": 0, "knots": 1, "te_sort": 2, "te_merge": 3}[phase]
    assert log.opens <= first, (phase, log.opens)
    assert phase not in kinds
    assert list(tmp_path.iterdir()) == []


SOLVER_PHASES = {
    # Ten lockstep folds' Grams.
    "cv": (
        {"terms": {"level": {"type": "categorical"}}},
        {"cv": 10, "regularization": "ridge", "n_alphas": 2},
    ),
    "solve": ({"terms": {"level": {"type": "categorical"}}}, {}),
    # θ's per-row blocks beside the solve's.
    "theta": (
        {"terms": {"level": {"type": "categorical"}}, "family": "negbinomial", "theta": "estimate"},
        {},
    ),
    # Four robust meats beside the covariance.
    "final": (
        {"terms": {"level": {"type": "categorical"}}},
        {"robust": ["HC0", "HC1", "HC2", "HC3"]},
    ),
}


@pytest.mark.parametrize("phase", list(SOLVER_PHASES))
def test_max_bytes_refuses_a_solver_phase_before_its_passes(tmp_path, phase) -> None:
    """With encode_rows and the threads fixed a solver phase's peak does not depend on
    max_bytes: the plan's own figure admits the fit, and a byte less refuses it before
    any solver pass."""
    glm, fit = SOLVER_PHASES[phase]
    spec = {"response": "y", "family": "poisson", **glm}
    options = {"encode_rows": 16_384, "threads": 2, "scratch_dir": tmp_path}
    data = _numeric_frame()
    plan = (
        rs.glm_dict(data=_source(data)[0], **spec)
        .fit(out_of_core=rs.OutOfCoreOptions(**options), **fit)
        .memory_plan
    )
    needed = plan.persistent_bytes + plan.phases[phase]
    assert needed >= _learning_threshold()
    assert plan.phases[phase] == max(
        peak for name, peak in plan.phases.items() if name in SOLVER_PHASES or name == "moments"
    )
    admitted = rs.glm_dict(data=_source(data)[0], **spec).fit(
        out_of_core=rs.OutOfCoreOptions(max_bytes=needed, **options), **fit
    )
    assert admitted.memory_plan.phases[phase] == plan.phases[phase]

    source, _log = _source(data)
    kinds: list[str] = []
    with pytest.raises(MemoryPlanExceededError, match=f"The {phase} phase"):
        rs.glm_dict(data=source, **spec).fit(
            out_of_core=rs.OutOfCoreOptions(
                max_bytes=needed - 1, on_batch=lambda p: kinds.append(p.pass_kind), **options
            ),
            **fit,
        )
    assert set(kinds) <= {
        "freeze",
        "knots",
        "te_stats",
        "te_sort",
        "te_merge",
        "te_walk",
        "te_average",
    }


def test_max_bytes_refuses_the_labels_pass() -> None:
    source, log = _source(_labels_frame())
    with pytest.raises(MemoryPlanExceededError, match="The labels phase"):
        rs.observed_category_labels(
            source, ["c"], out_of_core=rs.OutOfCoreOptions(max_bytes=_learning_threshold() - _MiB)
        )
    assert log.opens == 0


def test_every_planned_phase_is_reported(frame, tmp_path) -> None:
    glm, fit = _model("learning", frame)
    data = glm.pop("data")
    result = rs.glm_dict(data=_source(data)[0], **glm).fit(
        out_of_core=rs.OutOfCoreOptions(scratch_dir=tmp_path), **fit
    )
    assert set(result.memory_plan.phases) == {
        "freeze",
        "knots",
        "te_stats",
        "te_sort",
        "te_merge",
        "moments",
        "solve",
        "final",
    }
    cv = rs.glm_dict(
        data=_source(frame)[0], response="y_pois", terms=_LINEAR, family="poisson"
    ).fit(cv=10, regularization="ridge", n_alphas=2)
    assert "cv" in cv.memory_plan.phases


# ── T12: the result surface ─────────────────────────────────────────────────


def test_an_out_of_core_result_refuses_each_per_row_method(frame) -> None:
    holdout = design_frame(500, seed=3)
    glm = {
        "response": "y_pois",
        "terms": {"x1": {"type": "linear"}, "cat_a": {"type": "categorical"}},
        "family": "poisson",
        "exposure": "exposure",
    }
    in_memory = rs.glm_dict(data=frame, **glm).fit(robust=["HC1"])
    result = rs.glm_dict(data=_source(frame)[0], **glm).fit(robust=["HC1"])
    refused = {
        "fittedvalues": lambda: result.fittedvalues,
        "linear_predictor": lambda: result.linear_predictor,
        "resid_response": lambda: result.resid_response(),
        "resid_pearson": lambda: result.resid_pearson(),
        "resid_deviance": lambda: result.resid_deviance(),
        "resid_working": lambda: result.resid_working(),
        "fit_diagnostics": lambda: result.fit_diagnostics(),
        "relevel": lambda: result.relevel(frame),
        "diagnostics": lambda: result.diagnostics(),
        "HC3": lambda: result.bse_robust("HC3"),
    }
    for name, call in refused.items():
        with pytest.raises(OutOfCoreResultError, match=name):
            call()
    for call in (
        lambda: result.cov_robust("HC0"),
        lambda: result.tvalues_robust("HC2"),
        lambda: result.pvalues_robust("HC3"),
        lambda: result.conf_int_robust(cov_type="HC0"),
    ):
        with pytest.raises(OutOfCoreResultError, match="not listed in robust="):
            call()

    # What it serves equals the in-memory fit's, and survives to_bytes.
    for field in FIELDS:
        assert _bits(_read(result, field)) == _bits(_read(in_memory, field)), field
    assert _bits(result.bse_robust("HC1")) == _bits(in_memory.bse_robust("HC1"))
    restored = rs.GLMModel.from_bytes(result.to_bytes())
    assert _bits(np.asarray(restored.predict(holdout))) == _bits(
        np.asarray(in_memory.predict(holdout))
    )
    # Diagnostics on data predict on it.
    diagnostics = result.diagnostics(
        frame, categorical_factors=["cat_a"], continuous_factors=["x1"]
    )
    assert diagnostics is not None


# ── §4.5: the fit record through to_bytes ───────────────────────────────────


RECORD_GLM = {
    "response": "y_pois",
    "family": "poisson",
    "terms": {"x1": {"type": "linear"}, "cat_a": {"type": "categorical"}},
}


def _record(result: rs.GLMModel) -> tuple:
    return result.fit_mode, result.passes, result.solver_settings, result.solver_trace


@pytest.mark.parametrize("out_of_core", [False, True], ids=["in_memory", "out_of_core"])
def test_a_restored_result_reports_how_the_fit_ran(frame, out_of_core: bool) -> None:
    if out_of_core:
        result = rs.glm_dict(data=_source(frame)[0], **RECORD_GLM).fit(
            out_of_core=rs.OutOfCoreOptions()
        )
    else:
        result = rs.glm_dict(data=frame, **RECORD_GLM).fit()

    restored = rs.GLMModel.from_bytes(result.to_bytes())

    assert _record(restored) == _record(result)
    assert restored.solver_trace
    assert all(isinstance(step, rs.SolverStep) for step in restored.solver_trace)
    if out_of_core:
        assert restored.memory_plan == result.memory_plan
        assert restored.encode_rows == result.encode_rows


def test_a_restored_out_of_core_result_still_refuses_per_row_methods(frame) -> None:
    result = rs.glm_dict(data=_source(frame)[0], **RECORD_GLM).fit(
        out_of_core=rs.OutOfCoreOptions()
    )
    restored = rs.GLMModel.from_bytes(result.to_bytes())

    with pytest.raises(OutOfCoreResultError, match="relevel"):
        restored.relevel(frame)


def test_a_payload_without_the_fit_record_reads_as_an_in_memory_fit(frame) -> None:
    """0.9.0 wrote the same schema version without these fields."""
    state = pickle.loads(rs.glm_dict(data=frame, **RECORD_GLM).fit().to_bytes())
    for key in (
        "fit_mode",
        "passes",
        "solver_settings",
        "solver_trace",
        "memory_plan",
        "encode_rows",
    ):
        del state["result_state"][key]

    restored = rs.GLMModel.from_bytes(pickle.dumps(state))

    assert _record(restored) == ("in_memory", {}, {}, ())


def test_every_exception_can_be_caught_from_the_package() -> None:
    for name in exceptions.__all__:
        assert getattr(rs, name) is getattr(exceptions, name), name
        assert name in rs.__all__, name


# ── §6.8: array inputs ──────────────────────────────────────────────────────


ARRAY_CASES = {
    "exposure, offset and weights": {
        "response": "y_pois",
        "family": "poisson",
        "exposure": "exposure",
        "offset": "offset",
        "weights": "weight",
    },
    "complement with exposure": {
        "response": "y_pois",
        "family": "poisson",
        "exposure": "exposure",
        "complement": "comp",
    },
}


@pytest.mark.parametrize("glm", list(ARRAY_CASES.values()), ids=list(ARRAY_CASES))
def test_array_inputs_equal_their_columns_bitwise(frame, glm) -> None:
    holdout = design_frame(400, seed=5)
    terms = {
        "x1": {"type": "linear"},
        "x2": {"type": "bs", "df": 4},
        "cat_b": {"type": "categorical"},
    }
    arrays = {
        name: frame[glm[name]].to_numpy()
        for name in ("exposure", "offset", "weights", "complement")
        if name in glm
    }
    fit = {"alpha": 0.2, "robust": ["HC1"]}
    in_memory = rs.glm_dict(data=frame, terms=terms, **{**glm, **arrays}).fit(**fit)
    from_arrays = rs.glm_dict(data=_source(frame)[0], terms=terms, **{**glm, **arrays}).fit(**fit)
    from_columns = rs.glm_dict(data=_source(frame)[0], terms=terms, **glm).fit(**fit)
    for field in FIELDS:
        expected = _bits(_read(in_memory, field))
        assert _bits(_read(from_arrays, field)) == expected, field
        assert _bits(_read(from_columns, field)) == expected, field
    assert from_arrays.has_complement == in_memory.has_complement == ("complement" in glm)
    # A model fitted with arrays is given the new data's arrays to predict.
    new = {name: holdout[glm[name]].to_numpy() for name in ("exposure", "offset") if name in glm}
    assert _bits(np.asarray(from_arrays.predict(holdout, **new))) == _bits(
        np.asarray(in_memory.predict(holdout, **new))
    )


def test_array_weights_estimate_theta_equal_in_memory(frame, tmp_path) -> None:
    glm = {"response": "y_nb", "terms": _LINEAR, "family": "negbinomial", "theta": "estimate"}
    weights = frame["weight"].to_numpy()
    in_memory = rs.glm_dict(data=frame, weights=weights, **glm).fit()
    out_of_core = rs.glm_dict(data=_source(frame)[0], weights=weights, **glm).fit(
        out_of_core=rs.OutOfCoreOptions(scratch_dir=tmp_path)
    )
    for field in ("params", "theta", "deviance", "bse"):
        assert _bits(_read(out_of_core, field)) == _bits(_read(in_memory, field)), field


@pytest.mark.parametrize("length", [5_999, 6_001])
def test_an_array_must_have_one_value_per_source_row(frame, length) -> None:
    weights = np.ones(length)
    with pytest.raises(ValidationError, match="weights array length"):
        rs.glm_dict(
            data=_source(frame)[0],
            response="y_pois",
            terms=_LINEAR,
            family="poisson",
            weights=weights,
        ).fit()


def test_a_two_dimensional_array_is_refused_at_construction(frame) -> None:
    source, log = _source(frame)
    with pytest.raises(ValidationError, match="one-dimensional"):
        rs.glm_dict(
            data=source,
            response="y_pois",
            terms=_LINEAR,
            family="poisson",
            exposure=np.ones((frame.height, 2)),
        )
    assert log.opens == 0


# ── §6.8: a non-finite complement ───────────────────────────────────────────


@pytest.mark.parametrize("bad", [np.nan, np.inf])
@pytest.mark.parametrize("as_array", [False, True], ids=["column", "array"])
def test_a_non_finite_complement_is_refused_as_in_memory(frame, bad, as_array) -> None:
    comp = frame["comp"].to_numpy().copy()
    comp[17] = bad
    data = frame.with_columns(comp=pl.Series(comp))
    glm = {
        "response": "y_gauss",
        "terms": _LINEAR,
        "family": "gaussian",
        "complement": comp if as_array else "comp",
    }
    fit = {"max_iter": 0, "compute_covariance": False}
    with pytest.raises(ValidationError, match="offset contains 1") as in_memory:
        rs.glm_dict(data=data, **glm).fit(**fit)
    with pytest.raises(ValidationError, match="offset contains 1") as out_of_core:
        rs.glm_dict(data=_source(data)[0], **glm).fit(**fit)
    assert "source rows 0..996" in str(out_of_core.value)
    assert str(in_memory.value).split(" values")[0] == str(out_of_core.value).split(" values")[0]


# ── T18 (reduced): lockstep ten-fold CV within its plan ─────────────────────

_RSS_SCRIPT = r"""
import json, sys
import numpy as np, polars as pl, rustystats as rs

def status(field):
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    raise KeyError(field)

levels, rows, max_bytes = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
rng = np.random.default_rng(18)
level = rng.integers(0, levels, rows)
frame = pl.DataFrame({
    "x": rng.normal(size=rows),
    "level": pl.Series(level, dtype=pl.Int64),
    "y": rng.poisson(np.exp(0.1 + 0.001 * level)).astype(float),
})
source = rs.BatchSource(
    lambda columns: (frame.select(columns).slice(s, 2048) for s in range(0, rows, 2048)),
    frame.schema,
)
model = rs.glm_dict(
    response="y",
    terms={"x": {"type": "linear"}, "level": {"type": "categorical"}},
    data=source,
    family="poisson",
)
options = {"threads": 2, "encode_rows": 4096}
if max_bytes != "none":
    options["max_bytes"] = int(max_bytes)
# The peak from here on: the fit's own allocations.
with open("/proc/self/clear_refs", "w") as handle:
    handle.write("5")
baseline = status("VmRSS")
result = model.fit(
    cv=10, regularization="ridge", n_alphas=3, out_of_core=rs.OutOfCoreOptions(**options)
)
plan = result.memory_plan
print(json.dumps({
    "rise": status("VmHWM") - baseline,
    "persistent": plan.persistent_bytes,
    "phases": plan.phases,
    "folds": result.n_cv_folds,
}))
"""


def _measured_cv(levels: int, rows: int, max_bytes: int | None) -> dict[str, Any]:
    import json
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "-c", _RSS_SCRIPT, str(levels), str(rows), str(max_bytes or "none")],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not __import__("os").path.exists("/proc/self/clear_refs"), reason="Linux /proc")
def test_lockstep_cv_at_ten_folds_is_admitted_by_its_plan_and_stays_within_it() -> None:
    """T18, reduced for CI (p ≈ 150; the p ≈ 1,000 run is in
    ``benchmarks/bench_out_of_core_memory.py``): a fresh process's peak RSS rise over
    the fit stays within the plan it was admitted under, at exactly that plan."""
    planned = _measured_cv(150, 4_000, None)
    needed = planned["persistent"] + max(planned["phases"].values())
    measured = _measured_cv(150, 4_000, needed)
    assert measured["folds"] == 10
    assert measured["phases"]["cv"] == planned["phases"]["cv"]
    admitted = measured["persistent"] + max(measured["phases"].values())
    assert admitted <= needed
    assert measured["rise"] <= admitted, (measured["rise"], admitted)


def test_the_plan_counts_the_threads_the_fit_runs_on(frame) -> None:
    """§11: T is ``threads``, or the global pool's size; the Gram partials scale with it."""
    from rustystats._rustystats import rayon_threads_py

    glm = {"response": "y_pois", "terms": _LINEAR, "family": "poisson"}
    plans = {
        threads: rs.glm_dict(data=_source(frame)[0], **glm)
        .fit(out_of_core=rs.OutOfCoreOptions(threads=threads, encode_rows=4_096))
        .memory_plan
        for threads in (1, 3, None)
    }
    assert plans[1].threads == 1 and plans[3].threads == 3
    assert plans[None].threads == rayon_threads_py()
    p = 2
    partial = 8 * (p * (p + 1) // 2) + 8 * p + 16 + 8 * p
    assert plans[3].phases["solve"] - plans[1].phases["solve"] == 2 * partial


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_design_widths_count_the_layouts_columns(case) -> None:
    """§6.4: the width ``max_design_columns`` is judged by equals the frozen design's p,
    from a fresh formula (knots not yet placed) and from the learned one."""
    from rustystats.formula import dict_to_parsed_formula
    from rustystats.frozen_design import (
        design_term_widths,
        in_memory_target_inputs,
        learn_frozen_design,
    )

    glm, _fit, _holdout = case.build()

    def parse():
        return dict_to_parsed_formula(
            response=glm["response"],
            terms=glm["terms"],
            interactions=glm.get("interactions"),
            intercept=glm.get("intercept", True),
        )

    parsed = parse()
    target = (
        in_memory_target_inputs(glm["response"], None, 1) if parsed.target_encoding_terms else None
    )
    design = learn_frozen_design(parsed, [glm["data"]], target=target)

    def labels(column: str) -> int:
        return len(design.categories[column].labels)

    for formula in (parse(), parsed):
        p = sum(width for _term, width in design_term_widths(formula, labels))
        assert max(p, 1) == design.p

    # §5.2: the structural nonzeros per row bound every encoded row's.
    x = np.empty((glm["data"].height, design.p))
    design.encode_into(glm["data"], x, first_row=0)
    assert int(np.count_nonzero(x, axis=1).max()) <= design.max_row_nonzeros <= design.p


def _wide_sparse_frame(rows: int, levels: int = 100) -> pl.DataFrame:
    rng = np.random.default_rng(52)
    level = rng.integers(0, levels, rows)
    return pl.DataFrame(
        {
            "x": rng.normal(size=rows),
            "level": pl.Series(level, dtype=pl.Int64),
            "y": rng.poisson(np.exp(0.2 + 0.004 * level)).astype(float),
        }
    )


def test_sparse_rows_are_chosen_from_the_structure_and_planned() -> None:
    """§5.2, §11: a wide categorical design (3 nonzeros of 102 columns) is served sparse
    rows once n·p reaches 10⁷; the plan then holds each thread's block CSR, and the fit
    equals the in-memory one bit for bit."""
    from rustystats._rustystats import sparse_rows_py
    from rustystats.out_of_core import GRAM_BLOCK_ROWS

    glm = {
        "response": "y",
        "terms": {"x": {"type": "linear"}, "level": {"type": "categorical"}},
        "family": "poisson",
    }
    options = {"threads": 2, "encode_rows": 8_192}
    plans = {}
    for rows in (60_000, 100_000):
        data = _wide_sparse_frame(rows)
        result = rs.glm_dict(data=_source(data, sizes=(25_000,))[0], **glm).fit(
            out_of_core=rs.OutOfCoreOptions(**options)
        )
        plans[rows] = result.memory_plan
        p = len(result.feature_names)
        assert sparse_rows_py(rows, p, 3) == (rows * p >= 10_000_000)
        if rows == 100_000:
            in_memory = rs.glm_dict(data=data, **glm).fit()
            assert _bits(result.params) == _bits(np.asarray(in_memory.params))
            assert _bits(result.bse()) == _bits(np.asarray(in_memory.bse()))
    block = GRAM_BLOCK_ROWS * 12 * 3 + 8 * (GRAM_BLOCK_ROWS + 1)
    csr = 2 * block + 8 * (p + 1)
    for phase in ("solve", "final"):
        assert plans[100_000].phases[phase] - plans[60_000].phases[phase] == csr, phase


# ── Review 2: arrays, oversized rows, penalties, rowless results ────────────


def test_an_array_changed_between_passes_is_refused(frame) -> None:
    """§9: an array input is fingerprinted by global row in every pass, like a source
    column; changing a weight during the reconstruct pass is caught."""
    weights = frame["weight"].to_numpy().copy()

    def on_batch(progress: rs.BatchProgress) -> None:
        if progress.pass_kind == "reconstruct":
            weights[5] = 1e6

    with pytest.raises(rs.exceptions.BatchSourceChangedError, match="the weights array"):
        rs.glm_dict(
            data=_source(frame)[0],
            response="y_pois",
            terms=_LINEAR,
            family="poisson",
            weights=weights,
        ).fit(out_of_core=rs.OutOfCoreOptions(on_batch=on_batch))


_OVERSIZED_SCRIPT = r"""
import json, sys
import polars as pl, rustystats as rs

def status(field):
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024

big = 40 * 1024 * 1024
frame = pl.DataFrame({
    "b": pl.Series([b"\xff" * big] + [b"a", b"b"] * 500, dtype=pl.Binary),
    "y": [float(i % 7) for i in range(1001)],
})
source = rs.BatchSource(lambda columns: [frame.select(columns)], frame.schema)
model = rs.glm_dict(response="y", terms={"b": {"type": "categorical"}}, data=source,
                    family="gaussian")
with open("/proc/self/clear_refs", "w") as handle:
    handle.write("5")
baseline = status("VmRSS")
try:
    model.fit(out_of_core=rs.OutOfCoreOptions(max_bytes=100 * 1024 * 1024,
                                              max_learned_state_bytes=1024 * 1024))
    error = None
except Exception as exc:
    error = f"{type(exc).__name__}: {exc}"
print(json.dumps({"rise": status("VmHWM") - baseline, "error": error}))
"""


@pytest.mark.skipif(not __import__("os").path.exists("/proc/self/clear_refs"), reason="Linux /proc")
def test_an_oversized_row_is_refused_before_it_is_touched() -> None:
    """§4.3: a 40 MiB binary value is larger than a learning slice; it is charged before
    it is fingerprinted or learned, so a small budget refuses it before memory rises
    beyond what the freeze phase admitted."""
    import json
    import subprocess
    import sys

    from rustystats import out_of_core_fit as fit_module

    completed = subprocess.run(
        [sys.executable, "-c", _OVERSIZED_SCRIPT], capture_output=True, text=True, check=True
    )
    measured = json.loads(completed.stdout.strip().splitlines()[-1])
    assert measured["error"].startswith("LearnedStateTooLargeError: A row of column 'b'")
    assert "holds 41,943,048 bytes" in measured["error"]  # the value and y's 8 bytes
    # Admitted before the refusal: the freeze phase's working set; persistent state is
    # the allocator allowance the process already holds.
    learning = fit_module._learning_peak(fit_module._learn_slice_bytes(100 * _MiB))
    assert measured["rise"] <= learning, (measured["rise"], learning)


def test_penalty_matrices_are_built_after_the_solve_is_admitted(frame, monkeypatch) -> None:
    """§11: smooth penalties are planned from their dimensions; the matrices are built
    only once the solver phases are admitted, and not at all when they are refused."""
    from rustystats import out_of_core_fit as fit_module
    from rustystats.splines import SplineTerm

    events: list[str] = []
    build = SplineTerm.compute_penalty_matrix
    admit = fit_module._Run._admit

    def penalty(self, *args, **kwargs):
        events.append("penalty")
        return build(self, *args, **kwargs)

    def record(self, phase, peak):
        events.append(phase)
        return admit(self, phase, peak)

    monkeypatch.setattr(SplineTerm, "compute_penalty_matrix", penalty)
    monkeypatch.setattr(fit_module._Run, "_admit", record)
    glm = {"response": "y_pois", "terms": {"x2": {"type": "bs", "k": 8}}, "family": "poisson"}
    rs.glm_dict(data=_source(frame)[0], **glm).fit()
    assert "penalty" in events
    assert events.index("final") < events.index("penalty")

    def refuse(self, phase, peak):
        events.append(phase)
        if phase == "solve":
            raise MemoryPlanExceededError("The solve phase plans too much.")
        return admit(self, phase, peak)

    events.clear()
    monkeypatch.setattr(fit_module._Run, "_admit", refuse)
    with pytest.raises(MemoryPlanExceededError):
        rs.glm_dict(data=_source(frame)[0], **glm).fit()
    assert "solve" in events and "penalty" not in events


def _array_fit(rows: int) -> tuple[Any, list[Any]]:
    data = design_frame(rows, seed=31)
    arrays = {
        "exposure": data["exposure"].to_numpy().copy(),
        "offset": data["offset"].to_numpy().copy(),
        "weights": data["weight"].to_numpy().copy(),
        "complement": data["comp"].to_numpy().copy(),
    }
    import weakref

    refs = [weakref.ref(array) for array in arrays.values()]
    result = rs.glm_dict(
        data=_source(data)[0],
        response="y_pois",
        terms={"x1": {"type": "linear"}, "x3": {"type": "linear"}},
        family="poisson",
        **arrays,
    ).fit(robust=["HC1"])
    return result, refs


def test_an_out_of_core_result_keeps_no_training_arrays() -> None:
    """§4.5: the result keeps column specs and, for arrays, only that prediction must
    be given new ones; it holds no reference to the arrays and serializes to the same
    size whatever the row count."""
    import gc

    small, small_refs = _array_fit(300)
    large, large_refs = _array_fit(10_000)
    gc.collect()
    assert all(ref() is None for ref in small_refs + large_refs)
    assert len(small.to_bytes()) == len(large.to_bytes())

    holdout = design_frame(50, seed=2)
    with pytest.raises(rs.exceptions.PredictionError, match="exposure"):
        large.predict(holdout)
    with pytest.raises(rs.exceptions.PredictionError, match="offset array"):
        large.predict(holdout, exposure=holdout["exposure"].to_numpy())
    restored = rs.GLMModel.from_bytes(large.to_bytes())
    new = {"exposure": holdout["exposure"].to_numpy(), "offset": holdout["offset"].to_numpy()}
    assert _bits(np.asarray(restored.predict(holdout, **new))) == _bits(
        np.asarray(large.predict(holdout, **new))
    )
    assert large.has_complement


def _named_frame(rows: int = 3_000) -> pl.DataFrame:
    rng = np.random.default_rng(41)
    x = rng.normal(size=rows)
    return pl.DataFrame(
        {"x": x, "": np.full(rows, 2.0), "y": rng.poisson(np.exp(0.2 + 0.3 * x)).astype(float)}
    )


@pytest.mark.parametrize("role", ["exposure", "offset"])
def test_an_empty_string_column_name_is_a_column_in_both_modes(role: str) -> None:
    """A column named "" is a column spec, not an array: an out-of-core result keeps it
    and predicts exactly as the in-memory result, before and after serialization."""
    frame = _named_frame()
    glm = {"response": "y", "terms": {"x": {"type": "linear"}}, "family": "poisson", role: ""}
    in_memory = rs.glm_dict(data=frame, **glm).fit()
    out_of_core = rs.glm_dict(data=_source(frame)[0], **glm).fit(out_of_core=rs.OutOfCoreOptions())
    restored = rs.GLMModel.from_bytes(out_of_core.to_bytes())

    expected = np.asarray(in_memory.predict(frame))
    np.testing.assert_array_equal(np.asarray(out_of_core.predict(frame)), expected)
    np.testing.assert_array_equal(np.asarray(restored.predict(frame)), expected)


def test_a_column_named_like_an_array_key_keeps_its_own_fingerprint() -> None:
    """Fingerprint keys are tagged, so a source column whose name looks like an array's
    key never shares that array's fingerprint across passes that read different columns."""
    frame = _named_frame().drop("").rename({"y": "<weights array>"})
    weights = np.random.default_rng(5).uniform(0.5, 2.0, size=frame.height)
    glm = {
        "response": "<weights array>",
        "terms": {"x": {"type": "bs", "df": 5}},
        "family": "poisson",
        "weights": weights,
    }
    in_memory = rs.glm_dict(data=frame, **glm).fit()
    out_of_core = rs.glm_dict(data=_source(frame)[0], **glm).fit(out_of_core=rs.OutOfCoreOptions())

    np.testing.assert_array_equal(np.asarray(out_of_core.params), np.asarray(in_memory.params))
