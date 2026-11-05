"""Refusals, edge cases and failure paths of out-of-core fitting, each checked by what
the caller sees: the error and its message, or the value (``specs/out-of-core-fit.md``).

Sources and options (§4.1, §4.3), category keys and the learned-state budget (§6.3,
§6.4), the freeze pass's checks (§6.1, §6.8), the θ spill and target-encoding spill
files (§10), the memory plan (§11), and the labels pass (§4.6).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import polars as pl
import pytest
import rustystats as rs
from _golden_cases import RESULT_FIELDS, _read, design_frame
from rustystats import out_of_core_fit as fit_module
from rustystats.category_keys import CategoryKeyLearner, LearnedStateBudget
from rustystats.exceptions import (
    BatchSourceChangedError,
    LearnedStateTooLargeError,
    MemoryPlanExceededError,
    OutOfCoreOptionError,
    SpillError,
    UnseenLabelError,
    ValidationError,
)
from rustystats.out_of_core import BatchSource, OutOfCoreOptions, slice_rows

_MiB = 1024 * 1024
FIELDS = [field for field in RESULT_FIELDS if field != "cv_deviance"]
_LINEAR = {"x1": {"type": "linear"}}


def _source(frame: pl.DataFrame, sizes: tuple[int, ...] = (997, 2048, 1, 1501)) -> BatchSource:
    def open_batches(columns):
        selected = frame.select(columns)
        start, i = 0, 0
        while start < selected.height:
            yield selected.slice(start, sizes[i % len(sizes)])
            start += sizes[i % len(sizes)]
            i += 1

    return BatchSource(open_batches, frame.schema)


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


def _assert_equal(in_memory: Any, out_of_core: Any) -> None:
    for field in FIELDS:
        assert _bits(_read(out_of_core, field)) == _bits(_read(in_memory, field)), field


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    return design_frame(4000, seed=17)


# ── Sources, options and slicing (§4.1, §4.3) ───────────────────────────────


def test_options_refuse_a_non_positive_run_size() -> None:
    with pytest.raises(ValidationError, match="te_run_records must be positive, got 0"):
        OutOfCoreOptions(te_run_records=0)


def test_a_source_needs_a_callable_and_dataframes(frame) -> None:
    with pytest.raises(ValidationError, match="callable `open"):
        BatchSource("not callable", frame.schema)  # type: ignore[arg-type]
    source = BatchSource(lambda columns: [{"x1": [1.0]}], frame.schema)
    with pytest.raises(ValidationError, match="must yield Polars DataFrames, got dict"):
        list(source.batches(["x1"]))


def test_slicing_refuses_no_rows_per_slice_and_yields_nothing_for_an_empty_batch() -> None:
    empty = pl.DataFrame({"s": pl.Series([], dtype=pl.String)})
    with pytest.raises(ValidationError, match="max_rows must be at least 1, got 0"):
        list(slice_rows(empty, max_rows=0))
    assert list(slice_rows(empty, max_rows=10, max_bytes=100)) == []


# ── Category keys and the budget (§6.3, §6.4) ───────────────────────────────


def test_a_looser_cap_keeps_the_tighter_limit_and_its_source() -> None:
    budget = LearnedStateBudget(100)
    budget.cap(200, source="max_bytes")
    assert (budget.limit, budget.source) == (100, "max_learned_state_bytes")
    budget.cap(50, source="max_bytes = 1")
    assert (budget.limit, budget.source) == (50, "max_bytes = 1")


def test_an_oversized_row_fits_or_is_refused_before_it_is_learned() -> None:
    budget = LearnedStateBudget(1_000)
    budget.charge(400, term="c", cardinality=1)
    budget.admit_row("c", 100, transient=600)  # 400 + 600 fits exactly
    with pytest.raises(
        LearnedStateTooLargeError,
        match=r"A row of column 'c' holds 150 bytes.* needs 601 bytes beside 400 bytes",
    ):
        budget.admit_row("c", 150, transient=601)
    assert budget.used == 400, "admitting a row charges nothing lasting"
    LearnedStateBudget(None).admit_row("c", 10**12, transient=10**12)


def test_a_learner_reports_its_dtype_once_it_has_seen_a_slice() -> None:
    learner = CategoryKeyLearner("c")
    assert learner.dtype is None
    learner.observe(pl.Series("c", [3, 1], dtype=pl.Int32))
    assert learner.dtype == pl.Int32


@pytest.mark.parametrize(
    ("values", "floor", "labels"),
    [
        # Integers beyond ±2**53 may share a float64 label once a null arrives: the
        # two wide keys count as one, the null as one more.
        ([2**53 + 1, 2**53 + 2, 5, None], 3, 4),
        ([2**53 + 1, -(2**53) - 1, 5], 2, 3),
        # A float column's null may share "nan" with its NaN.
        ([1.5, float("nan"), None], 2, 2),
        (["a", "b", None], 3, 3),
    ],
)
def test_the_label_floor_never_exceeds_the_labels(values, floor, labels) -> None:
    learner = CategoryKeyLearner("c")
    assert learner.label_floor == 0
    learner.observe(pl.Series("c", values[:0], dtype=pl.Series(values).dtype))
    assert learner.label_floor == 0, "an empty slice adds no key"
    learner.observe(pl.Series("c", values))
    assert learner.label_floor == floor
    assert len(learner.finish().labels) == labels >= floor


def test_an_empty_column_has_no_labels() -> None:
    empty = pl.DataFrame({"c": pl.Series([], dtype=pl.String)})
    assert rs.observed_category_labels(empty, ["c"]) == {"c": []}


def test_an_unseen_value_beside_nulls_is_named() -> None:
    learner = CategoryKeyLearner("c")
    learner.observe(pl.Series("c", ["a", None]))
    with pytest.raises(UnseenLabelError, match="has the value 'z'"):
        learner.finish().encode(pl.Series("c", ["a", "z"]))


# ── Construction and fit options (§4.2, §4.3) ───────────────────────────────


def test_construction_refuses_inputs_the_source_cannot_serve(frame) -> None:
    source = _source(frame)
    with pytest.raises(ValidationError, match="weights= takes a column name or an array"):
        rs.glm_dict(data=source, response="y_pois", terms=_LINEAR, weights=3.0)
    with pytest.raises(ValidationError, match="`exposure=` is only supported for log-link"):
        rs.glm_dict(data=source, response="y_gauss", terms=_LINEAR, exposure="exposure")
    with pytest.raises(ValidationError, match=r"no column\(s\) \['nope'\]"):
        rs.glm_dict(data=source, response="y_pois", terms={"nope": {"type": "linear"}})


def test_a_nested_list_of_weights_is_refused_when_read(frame) -> None:
    weights = [[1.0]] * frame.height
    model = rs.glm_dict(
        data=_source(frame), response="y_pois", terms=_LINEAR, family="poisson", weights=weights
    )
    with pytest.raises(ValidationError, match="weights must be a one-dimensional array"):
        model.fit()


@pytest.mark.parametrize(
    ("fit", "error", "match"),
    [
        ({"out_of_core": "fast"}, OutOfCoreOptionError, "takes OutOfCoreOptions, got str"),
        ({"cv": 3}, ValidationError, "regularization' must be set"),
        ({"regularization": "group"}, ValidationError, "Unknown regularization type: group"),
        ({"alpha": -0.5}, ValidationError, "alpha must be >= 0, got -0.5"),
    ],
)
def test_fit_options_are_refused(frame, fit, error, match) -> None:
    model = rs.glm_dict(data=_source(frame), response="y_pois", terms=_LINEAR, family="poisson")
    with pytest.raises(error, match=match):
        model.fit(**fit)


@pytest.mark.parametrize(
    ("theta", "fit", "match"),
    [
        (None, {}, "requires an explicit theta"),
        ("guess", {}, "positive number or 'estimate', got 'guess'"),
        ("estimate", {"alpha": 0.1}, "not supported for regularized models"),
        (-1.0, {}, "theta must be > 0 for negative binomial, got -1.0"),
    ],
)
def test_negative_binomial_theta_is_checked(frame, theta, fit, match) -> None:
    model = rs.glm_dict(
        data=_source(frame), response="y_nb", terms=_LINEAR, family="negbinomial", theta=theta
    )
    with pytest.raises(ValidationError, match=match):
        model.fit(**fit)


def test_regularization_alone_cross_validates_five_folds(frame) -> None:
    glm = {"response": "y_pois", "terms": _LINEAR, "family": "poisson"}
    fit = {"regularization": "ridge", "n_alphas": 2}
    in_memory = rs.glm_dict(data=frame, **glm).fit(**fit)
    out_of_core = rs.glm_dict(data=_source(frame), **glm).fit(**fit)
    assert out_of_core.n_cv_folds == in_memory.n_cv_folds == 5
    _assert_equal(in_memory, out_of_core)


def test_cv_without_standardizing_or_the_unregularized_fit_equals_in_memory(frame) -> None:
    glm = {
        "response": "y_pois",
        "terms": {"x1": {"type": "linear"}, "cat_a": {"type": "categorical"}},
        "family": "poisson",
    }
    fit = {
        "cv": 3,
        "regularization": "lasso",
        "n_alphas": 3,
        "standardize": False,
        "include_unregularized": False,
    }
    in_memory = rs.glm_dict(data=frame, **glm).fit(**fit)
    out_of_core = rs.glm_dict(data=_source(frame), **glm).fit(**fit)
    assert 0.0 not in [entry["alpha"] for entry in out_of_core.regularization_path]
    _assert_equal(in_memory, out_of_core)


@pytest.mark.parametrize("target_encoded", [False, True], ids=["plain", "target-encoded"])
def test_cross_validation_reads_link_names_in_any_case(frame, tmp_path, target_encoded) -> None:
    """A plain fit accepts ``link="LOG"``; so does cross-validation's alpha_max, in both
    modes, with bitwise the results of ``link="log"``."""
    terms = {"x1": {"type": "linear"}, "cat_a": {"type": "categorical"}}
    if target_encoded:
        terms["brand"] = {"type": "target_encoding"}
    glm = {"response": "y_pois", "terms": terms, "family": "poisson", "seed": 5}
    fit = {"cv": 3, "regularization": "ridge", "n_alphas": 3, "cv_seed": 2}
    options = OutOfCoreOptions(scratch_dir=tmp_path, te_run_records=900)
    upper = rs.glm_dict(data=frame, link="LOG", **glm).fit(**fit)
    lower = rs.glm_dict(data=frame, link="log", **glm).fit(**fit)
    _assert_equal(lower, upper)
    upper = rs.glm_dict(data=_source(frame), link="LOG", **glm).fit(out_of_core=options, **fit)
    lower = rs.glm_dict(data=_source(frame), link="log", **glm).fit(out_of_core=options, **fit)
    _assert_equal(lower, upper)


_SMOOTH_INTERACTION = [{"cat_a": {"type": "categorical"}, "x2": {"type": "bs", "k": 6}}]


@pytest.mark.parametrize(
    ("glm", "fit", "options", "match"),
    [
        ({}, {"alpha": 0.1, "l1_ratio": 0.5}, {}, "L1/elastic-net"),
        ({}, {"cv": 3, "regularization": "ridge"}, {}, "CV-based regularization"),
        (
            {"response": "y_nb", "family": "negbinomial", "theta": "estimate"},
            {},
            {"scratch_dir": "TMP"},
            "theta estimation is not supported for smooth models",
        ),
    ],
    ids=["l1", "cv", "theta"],
)
def test_a_smooth_interaction_is_refused_once_its_columns_are_known(
    frame, tmp_path, glm, fit, options, match
) -> None:
    """An interaction's smooth columns depend on its categorical's levels, so the
    refusal follows the freeze pass (main-effect smooths are refused before it)."""
    opened = {"n": 0}
    base = _source(frame)

    def open_batches(columns):
        opened["n"] += 1
        return base._open(columns)

    source = BatchSource(open_batches, frame.schema)
    options = {k: tmp_path if v == "TMP" else v for k, v in options.items()}
    spec = {"response": "y_pois", "family": "poisson", **glm}
    with pytest.raises(ValidationError, match=match):
        rs.glm_dict(data=source, terms=_LINEAR, interactions=_SMOOTH_INTERACTION, **spec).fit(
            out_of_core=OutOfCoreOptions(**options), **fit
        )
    assert opened["n"] >= 1


# ── The freeze pass's checks (§6.1, §6.8) ───────────────────────────────────


def _both_raise(data: pl.DataFrame, glm: dict, match: str, **fit: Any) -> tuple[str, str]:
    with pytest.raises(ValidationError, match=match) as in_memory:
        rs.glm_dict(data=data, **glm).fit(**fit)
    with pytest.raises(ValidationError, match=match) as out_of_core:
        rs.glm_dict(data=_source(data), **glm).fit(**fit)
    return str(in_memory.value), str(out_of_core.value)


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("exposure", np.nan, "exposure must be finite"),
        ("exposure", 0.0, "exposure must be strictly positive"),
        ("weight", -1.0, "weights contains 1 negative values"),
    ],
)
def test_the_freeze_pass_refuses_bad_exposure_and_weights_as_in_memory(
    frame, column, value, match
) -> None:
    data = frame.with_columns(
        pl.when(pl.int_range(pl.len()) == 7).then(value).otherwise(pl.col(column)).alias(column)
    )
    glm = {
        "response": "y_pois",
        "terms": _LINEAR,
        "family": "poisson",
        "exposure": "exposure",
        "weights": "weight",
    }
    in_memory, out_of_core = _both_raise(data, glm, match)
    assert out_of_core.startswith(in_memory.split(".")[0])
    assert "(source rows 0..996)" in out_of_core


def test_a_constant_response_and_zero_weights_are_refused(frame) -> None:
    constant = frame.with_columns(y_pois=pl.lit(2.0))
    with pytest.raises(ValidationError, match=r"response \(y\) is constant \(all values = 2.0\)"):
        rs.glm_dict(data=_source(constant), response="y_pois", terms=_LINEAR).fit()
    zero = frame.with_columns(weight=pl.lit(0.0))
    with pytest.raises(ValidationError, match="weights sum to zero"):
        rs.glm_dict(data=_source(zero), response="y_pois", terms=_LINEAR, weights="weight").fit()


def test_an_empty_source_is_refused_for_its_empty_response(frame) -> None:
    empty = frame.clear()
    with pytest.raises(ValidationError, match=r"response \(y\) is empty"):
        rs.glm_dict(data=_source(empty), response="y_pois", terms=_LINEAR).fit()


def test_a_non_finite_spline_input_cannot_be_learned_out_of_core(frame) -> None:
    data = frame.with_columns(
        x2=pl.when(pl.int_range(pl.len()) == 3).then(np.inf).otherwise(pl.col("x2"))
    )
    with pytest.raises(OutOfCoreOptionError, match="spline input 'x2' is not finite"):
        rs.glm_dict(
            data=_source(data), response="y_pois", terms={"x2": {"type": "bs", "df": 5}}
        ).fit()


def test_a_non_finite_design_column_is_named(frame) -> None:
    data = frame.with_columns(
        x1=pl.when(pl.int_range(pl.len()) == 3000).then(np.inf).otherwise(pl.col("x1"))
    )
    with pytest.raises(ValidationError, match="The design column 'x1' has non-finite values"):
        rs.glm_dict(data=_source(data), response="y_pois", terms=_LINEAR, family="poisson").fit()


@pytest.mark.parametrize(
    ("family", "response", "complement", "match"),
    [
        ("poisson", "y_pois", 0.0, "must be strictly positive for poisson family with log link"),
        ("binomial", "y_bin", 1.5, r"must be in \(0, 1\) for binomial family with logit link"),
    ],
)
def test_a_complement_outside_the_links_domain_is_refused(
    frame, family, response, complement, match
) -> None:
    data = frame.with_columns(comp=pl.lit(0.5))
    data = data.with_columns(
        comp=pl.when(pl.int_range(pl.len()) == 9).then(complement).otherwise(pl.col("comp"))
    )
    glm = {"response": response, "terms": _LINEAR, "family": family, "complement": "comp"}
    with pytest.raises(ValidationError, match=match):
        rs.glm_dict(data=data, **glm)
    with pytest.raises(ValidationError, match=match):
        rs.glm_dict(data=_source(data), **glm).fit()


def test_a_logit_complement_inside_the_domain_equals_in_memory(frame) -> None:
    data = frame.with_columns(comp=(pl.col("x1").abs() / (1.0 + pl.col("x1").abs())) * 0.8 + 0.1)
    glm = {"response": "y_bin", "terms": _LINEAR, "family": "binomial", "complement": "comp"}
    in_memory = rs.glm_dict(data=data, **glm).fit(alpha=0.2)
    out_of_core = rs.glm_dict(data=_source(data), **glm).fit(alpha=0.2)
    _assert_equal(in_memory, out_of_core)


def test_a_model_complement_without_exposure_equals_in_memory(frame) -> None:
    prior = rs.glm_dict(data=frame, response="y_pois", terms=_LINEAR, family="poisson").fit()
    glm = {
        "response": "y_pois",
        "terms": {"cat_a": {"type": "categorical"}},
        "family": "poisson",
        "complement": prior,
    }
    in_memory = rs.glm_dict(data=frame, **glm).fit(alpha=0.3)
    out_of_core = rs.glm_dict(data=_source(frame), **glm).fit(alpha=0.3)
    _assert_equal(in_memory, out_of_core)


# ── Input transforms beside target encoding and θ estimation ────────────────

_CENTER = [
    {
        "type": "center",
        "name": "x1_center",
        "sources": ["x1"],
        "output": "x1c",
        "center": 0.25,
        "metadata": None,
    }
]


def test_input_transforms_with_target_encoding_and_theta_equal_in_memory(frame, tmp_path) -> None:
    """Target encoding's passes and θ's passes read every source column when the model
    transforms its inputs."""
    te = {
        "response": "y_pois",
        "terms": {"x1c": {"type": "linear"}, "brand": {"type": "target_encoding"}},
        "family": "poisson",
        "seed": 4,
        "input_transforms": _CENTER,
    }
    in_memory = rs.glm_dict(data=frame, **te).fit()
    out_of_core = rs.glm_dict(data=_source(frame), **te).fit(
        out_of_core=OutOfCoreOptions(scratch_dir=tmp_path, te_run_records=900)
    )
    _assert_equal(in_memory, out_of_core)

    nb = {
        "response": "y_nb",
        "terms": {"x1c": {"type": "linear"}},
        "family": "negbinomial",
        "theta": "estimate",
        "input_transforms": _CENTER,
    }
    in_memory = rs.glm_dict(data=frame, **nb).fit()
    out_of_core = rs.glm_dict(data=_source(frame), **nb).fit(
        out_of_core=OutOfCoreOptions(scratch_dir=tmp_path)
    )
    assert _bits(out_of_core.params) == _bits(np.asarray(in_memory.params))
    assert out_of_core.theta == in_memory.theta
    assert list(tmp_path.iterdir()) == []


# ── Spill files (§10) ───────────────────────────────────────────────────────


def _nb_model(frame: pl.DataFrame, source: BatchSource | None = None) -> Any:
    return rs.glm_dict(
        data=source or _source(frame),
        response="y_nb",
        terms=_LINEAR,
        family="negbinomial",
        theta="estimate",
    )


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs POSIX permissions")
def test_a_scratch_directory_without_write_access_raises_spill_error(frame, tmp_path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        with pytest.raises(SpillError, match="Cannot create a spill file in"):
            _nb_model(frame).fit(out_of_core=OutOfCoreOptions(scratch_dir=locked))
    finally:
        locked.chmod(0o700)
    assert list(locked.iterdir()) == []


@pytest.mark.parametrize(("mode", "match"), [("wb", "Cannot write"), ("rb", "Cannot read")])
def test_a_theta_spill_that_cannot_be_written_or_read_raises_spill_error(
    frame, tmp_path, monkeypatch, mode, match
) -> None:
    real_open = open

    def failing_open(path, flags="r", *args, **kwargs):
        if "rustystats-nb-mu-" in str(path) and flags == mode:
            raise PermissionError(13, "denied", str(path))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(fit_module, "open", failing_open, raising=False)
    with pytest.raises(SpillError, match=f"{match} the spill file"):
        _nb_model(frame).fit(out_of_core=OutOfCoreOptions(scratch_dir=tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_removing_a_spill_file_already_gone_is_not_an_error(frame, tmp_path, monkeypatch) -> None:
    real_remove = os.remove

    def remove_then_report_missing(path):
        real_remove(path)
        raise FileNotFoundError(2, "gone", path)

    monkeypatch.setattr(fit_module.os, "remove", remove_then_report_missing)
    nb = _nb_model(frame).fit(out_of_core=OutOfCoreOptions(scratch_dir=tmp_path))
    te = rs.glm_dict(
        data=_source(frame),
        response="y_pois",
        terms={"brand": {"type": "target_encoding"}},
        seed=2,
    ).fit(out_of_core=OutOfCoreOptions(scratch_dir=tmp_path, te_run_records=900))
    assert nb.theta > 0 and len(te.params) == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("kind", ["theta", "target"])
def test_a_spill_file_that_cannot_be_removed_raises_spill_error(
    frame, tmp_path, monkeypatch, kind
) -> None:
    real_remove = os.remove

    def refuse(path):
        raise PermissionError(13, "denied", path)

    monkeypatch.setattr(fit_module.os, "remove", refuse)
    with pytest.raises(SpillError, match="Cannot remove the spill file"):
        if kind == "theta":
            _nb_model(frame).fit(out_of_core=OutOfCoreOptions(scratch_dir=tmp_path))
        else:
            rs.glm_dict(
                data=_source(frame),
                response="y_pois",
                terms={"brand": {"type": "target_encoding"}},
                seed=2,
            ).fit(out_of_core=OutOfCoreOptions(scratch_dir=tmp_path, te_run_records=900))
    for leftover in tmp_path.iterdir():
        real_remove(leftover)


def test_a_theta_pass_with_more_rows_than_were_spilled_is_refused(frame, tmp_path) -> None:
    extra = frame.slice(0, 1)

    def open_batches(columns):
        yield frame.select(columns)
        if list(columns) == ["y_nb"]:  # a θ pass reads the response alone
            yield extra.select(columns)

    source = BatchSource(open_batches, frame.schema)
    with pytest.raises(BatchSourceChangedError, match="more rows on a θ pass than were spilled"):
        _nb_model(frame, source).fit(out_of_core=OutOfCoreOptions(scratch_dir=tmp_path))
    assert list(tmp_path.iterdir()) == []


# ── The memory plan (§11) ───────────────────────────────────────────────────


def test_target_encodings_sort_is_sized_from_max_bytes(frame, tmp_path) -> None:
    glm = {"response": "y_pois", "terms": {"brand": {"type": "target_encoding"}}, "seed": 2}
    roomy = rs.glm_dict(data=_source(frame), **glm).fit(
        out_of_core=OutOfCoreOptions(scratch_dir=tmp_path, max_bytes=512 * _MiB)
    )
    plan = roomy.memory_plan
    assert plan.phases["te_sort"] + plan.persistent_bytes <= 512 * _MiB
    assert plan.phases["te_merge"] >= 65 * _MiB
    with pytest.raises(MemoryPlanExceededError, match="Target encoding's sort needs at least 80"):
        rs.glm_dict(data=_source(frame), **glm).fit(
            out_of_core=OutOfCoreOptions(scratch_dir=tmp_path, max_bytes=100 * _MiB)
        )
    assert list(tmp_path.iterdir()) == []


def test_too_little_room_for_an_encode_block_is_refused() -> None:
    rng = np.random.default_rng(3)
    data = pl.DataFrame(
        {
            "level": pl.Series(rng.integers(0, 1000, 3000), dtype=pl.Int64),
            "y": rng.poisson(1.0, 3000).astype(float),
        }
    )
    with pytest.raises(MemoryPlanExceededError, match="Even 4096 encoded rows of"):
        rs.glm_dict(data=_source(data), response="y", terms={"level": {"type": "categorical"}}).fit(
            out_of_core=OutOfCoreOptions(max_bytes=100 * _MiB, threads=1)
        )


def test_sparse_rows_leave_the_moments_dense() -> None:
    """The CSR buffers join every solver phase but the moments, which run dense."""
    rng = np.random.default_rng(52)
    plans = {}
    for rows in (60_000, 100_000):
        level = rng.integers(0, 100, rows)
        data = pl.DataFrame(
            {
                "x": rng.normal(size=rows),
                "level": pl.Series(level, dtype=pl.Int64),
                "y": rng.poisson(np.exp(0.2 + 0.004 * level)).astype(float),
            }
        )
        plans[rows] = (
            rs.glm_dict(
                data=_source(data, sizes=(25_000,)),
                response="y",
                terms={"x": {"type": "linear"}, "level": {"type": "categorical"}},
                family="poisson",
            )
            .fit(alpha=0.1, out_of_core=OutOfCoreOptions(threads=2, encode_rows=8_192))
            .memory_plan
        )
    dense, sparse = plans[60_000].phases, plans[100_000].phases
    assert sparse["moments"] == dense["moments"]
    assert sparse["solve"] > dense["solve"]


def test_an_oversized_row_is_charged_before_it_is_learned() -> None:
    """A row larger than a learning slice is refused when its copies do not fit beside
    the learned state; when they do, it is learned, and its label is reserved next."""
    rng = np.random.default_rng(8)
    rows = 2_000
    long = "x" * (20 * _MiB)
    values = [long] + [f"k{i % 5}" for i in range(rows - 1)]
    data = pl.DataFrame({"c": values, "y": rng.normal(size=rows)})
    glm = {"response": "y", "terms": {"c": {"type": "frequency_encoding"}}}
    with pytest.raises(LearnedStateTooLargeError, match="A row of column 'c' holds 20,971,528"):
        rs.glm_dict(data=_source(data), **glm).fit(
            out_of_core=OutOfCoreOptions(max_bytes=256 * _MiB, max_learned_state_bytes=4 * _MiB)
        )
    # 4 × 20 MiB fits what 256 MiB leaves for learned state; the label (up to 4 bytes a
    # character, and its conversion) does not.
    with pytest.raises(LearnedStateTooLargeError, match=r"Learning 'c' \(6 distinct values\)"):
        rs.glm_dict(data=_source(data), **glm).fit(
            out_of_core=OutOfCoreOptions(max_bytes=256 * _MiB)
        )


# ── A pass closed early, and the labels pass (§4.4, §4.6) ────────────────────


def test_a_pass_closed_early_closes_the_reader(frame) -> None:
    closed = []

    class Reader:
        def __init__(self, columns):
            self.columns = columns

        def __iter__(self):
            yield from (frame.select(self.columns).slice(s, 1000) for s in range(0, 4000, 1000))

        def close(self):
            closed.append(True)

    model = rs.glm_dict(
        data=BatchSource(Reader, frame.schema), response="y_pois", terms=_LINEAR, family="poisson"
    )
    run = fit_module._Run(model, OutOfCoreOptions(encode_rows=4_096))
    (design,) = run.learn()
    run.plan_solve(design, robust=0)
    opened = len(closed)
    blocks = run.solver_passes(design)("candidate")
    x, y, weights, offset = next(blocks)
    assert x.shape == (1000, design.p) and y.shape == weights.shape == offset.shape == (1000,)
    blocks.close()
    assert len(closed) == opened + 1


def test_the_labels_pass_refuses_a_missing_column_charges_long_rows_and_reads_no_rows() -> None:
    data = pl.DataFrame({"c": ["a", "x" * (8 * _MiB), "b"]})
    with pytest.raises(ValidationError, match=r"no column\(s\) \['d'\]"):
        rs.observed_category_labels(_source(data), ["d"])
    with pytest.raises(LearnedStateTooLargeError, match="A row of column 'c' holds 8,388,608"):
        rs.observed_category_labels(
            _source(data),
            ["c"],
            out_of_core=OutOfCoreOptions(max_bytes=100 * _MiB, max_learned_state_bytes=_MiB),
        )
    assert rs.observed_category_labels(_source(data.clear()), ["c"]) == {"c": []}
