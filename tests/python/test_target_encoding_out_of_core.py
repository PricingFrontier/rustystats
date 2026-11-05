"""Target encoding in DEC-1's keyed order, in memory and out of core (spec §6.6, §7.9, §10).

- T1-TE: a training column equals a pure-NumPy reference of §6.6 (sort by key, walk,
  average), single variable, exposure-weighted and interaction terms.
- T20: an out-of-core fit equals the in-memory fit bit for bit whatever the number of
  sort runs: one, two, and enough to force an intermediate merge stage.
- T22 / T10: spill files are removed on success, on error and when ``on_batch`` raises
  in each spill stage; an exception raised mid-path in cross-validation with target
  encoding reaches the caller unchanged; an unusable scratch directory raises
  ``SpillError``.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

import numpy as np
import polars as pl
import pytest
import rustystats as rs
from _golden_cases import RESULT_FIELDS, _read, design_frame
from rustystats.exceptions import OutOfCoreOptionError, SpillError

FIELDS = [field for field in RESULT_FIELDS if field != "cv_deviance"]


def _source(frame: pl.DataFrame, sizes=(997, 4096, 1, 3001)) -> rs.BatchSource:
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


def _bits(value: Any) -> Any:
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


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    return design_frame(3000, seed=20261105)


def _glm(**overrides: Any) -> dict[str, Any]:
    glm = {
        "response": "y_pois",
        "terms": {"brand": {"type": "target_encoding"}, "x1": {"type": "linear"}},
        "family": "poisson",
        "exposure": "exposure",
        "seed": 7,
    }
    glm.update(overrides)
    return glm


def _fit_both(frame, tmp_path, glm, fit=None, **options):
    fit = fit or {}
    in_memory = rs.glm_dict(data=frame, **glm).fit(**fit)
    out_of_core = rs.glm_dict(data=_source(frame), **glm).fit(
        out_of_core=rs.OutOfCoreOptions(encode_rows=4096, scratch_dir=tmp_path, **options), **fit
    )
    assert list(tmp_path.iterdir()) == [], "spill files are removed"
    return in_memory, out_of_core


def _assert_equal(left, right, holdout=None) -> None:
    for field in FIELDS:
        assert _bits(_read(left, field)) == _bits(_read(right, field)), field
    assert left.solver_trace == right.solver_trace
    if holdout is not None:
        assert _bits(np.asarray(left.predict(holdout))) == _bits(np.asarray(right.predict(holdout)))


# ── T1-TE: the keyed order's reference ──────────────────────────────────────

_U64 = np.uint64


def _mix64(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=_U64)
    z = (z ^ (z >> _U64(30))) * _U64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> _U64(27))) * _U64(0x94D049BB133111EB)
    return z ^ (z >> _U64(31))


def _reference(codes, target, exposure, seed, permutations, prior, prior_weight) -> np.ndarray:
    """§6.6 in NumPy and Python: sort rows by (mix64(mix64(seed + m) ^ r), r), walk with
    running per-label totals, average the permutations in order."""
    n = len(codes)
    rows = np.arange(n, dtype=_U64)
    columns = []
    for m in range(permutations):
        keys = _mix64(_mix64(np.array([seed + m], dtype=_U64))[0] ^ rows)
        order = np.lexsort((rows, keys))
        sums: dict[int, float] = defaultdict(float)
        totals: dict[int, float] = defaultdict(float)
        counts: dict[int, int] = defaultdict(int)
        values = np.empty(n)
        for r in order:
            c = int(codes[r])
            if exposure is None:
                values[r] = (sums[c] + prior * prior_weight) / (float(counts[c]) + prior_weight)
                counts[c] += 1
            else:
                values[r] = (sums[c] + prior * prior_weight) / (totals[c] + prior_weight)
                totals[c] += float(exposure[r])
            sums[c] += float(target[r])
        columns.append(values)
    if permutations == 1:
        return columns[0]
    return np.array([sum(column[i] for column in columns) / permutations for i in range(n)])


@pytest.mark.parametrize(
    ("permutations", "exposure", "interaction"),
    [(1, False, False), (4, False, False), (4, True, False), (3, True, True)],
)
def test_training_values_equal_the_reference(frame, permutations, exposure, interaction) -> None:
    variables = ["brand", "cat_a"] if interaction else ["brand"]
    if interaction:
        terms = {"x1": {"type": "linear"}}
        interactions = [
            {
                "brand": {"type": "categorical"},
                "cat_a": {"type": "categorical"},
                "target_encoding": True,
                "n_permutations": permutations,
            }
        ]
    else:
        terms = {"brand": {"type": "target_encoding", "n_permutations": permutations}}
        interactions = None
    model = rs.glm_dict(
        response="y_pois",
        terms=terms,
        interactions=interactions,
        data=frame,
        family="poisson",
        exposure="exposure" if exposure else None,
        seed=11,
    )
    name = "TE(" + ":".join(variables) + ")"
    stats = model._builder._te_stats[":".join(variables)]
    labels = [
        ":".join(str(v) for v in row)
        for row in zip(*(frame[variable].to_numpy() for variable in variables))
    ]
    level = {label: code for code, label in enumerate(sorted(stats["stats"]))}
    codes = np.array([level[label] for label in labels])
    y = frame["y_pois"].to_numpy()
    raw_exposure = frame["exposure"].to_numpy() if exposure else None
    expected = _reference(
        codes, y, raw_exposure, 11, permutations, stats["prior"], stats["prior_weight"]
    )
    column = model.X[:, model.feature_names.index(name)]
    assert _bits(column) == _bits(expected)

    # Prediction's statistics: each label's totals added in row order.
    sums: dict[str, float] = defaultdict(float)
    second: dict[str, float] = defaultdict(float)
    for i, label in enumerate(labels):
        sums[label] += y[i]
        second[label] += raw_exposure[i] if exposure else 1
    for label, (total, other) in stats["stats"].items():
        assert total == sums[label]
        assert other == second[label]
    assert stats["prior"] == pytest.approx(
        y.sum() / (raw_exposure.sum() if exposure else len(y)), rel=1e-12
    )


# ── T20: run counts ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("run_records", [10**6, 1500, 23])
def test_out_of_core_equals_in_memory_for_every_run_count(frame, tmp_path, run_records) -> None:
    """1 run, 2 runs, and 131 runs (an intermediate merge stage at fan-in 64)."""
    holdout = design_frame(500, seed=3)
    progress: list[rs.BatchProgress] = []
    glm = _glm(terms={"brand": {"type": "target_encoding"}, "cat_b": {"type": "target_encoding"}})
    in_memory = rs.glm_dict(data=frame, **glm).fit()
    out_of_core = rs.glm_dict(data=_source(frame), **glm).fit(
        out_of_core=rs.OutOfCoreOptions(
            encode_rows=4096,
            scratch_dir=tmp_path,
            te_run_records=run_records,
            on_batch=progress.append,
        )
    )
    assert list(tmp_path.iterdir()) == []
    _assert_equal(in_memory, out_of_core, holdout)
    assert out_of_core.solver_settings["target_encoding_seed"] == 7
    assert in_memory.solver_settings["target_encoding_seed"] == 7

    counts = out_of_core.passes
    # Per term and permutation: one source pass, one walk, the final merge by row, and
    # with more than 64 runs one intermediate merge by key and one by row; per term
    # one average.
    assert counts["te_stats"] == 1
    assert counts["te_sort"] == counts["te_walk"] == 2 * 4
    assert counts["te_average"] == 2
    runs = -(-frame.height // run_records)
    assert counts["te_merge"] == 2 * 4 * (3 if runs > 64 else 1)
    spill = [p for p in progress if p.pass_kind in ("te_merge", "te_walk", "te_average")]
    assert spill and all(p.n_rows == frame.height for p in spill)
    assert all(p.rows_done == frame.height for p in spill)
    assert sum(counts.values()) == len({p.pass_index for p in progress})


def test_target_encoded_interactions_equal_in_memory(frame, tmp_path) -> None:
    """A three-way TE interaction and continuous × TE factors (x2:TE(brand), TE(cat_b))."""
    glm = _glm(
        terms={"x1": {"type": "linear"}},
        interactions=[
            {
                "brand": {"type": "categorical"},
                "cat_a": {"type": "categorical"},
                "flag": {"type": "categorical"},
                "target_encoding": True,
            },
            {"x2": {"type": "linear"}, "brand": {"type": "target_encoding"}},
            {"cat_b": {"type": "target_encoding"}, "x3": {"type": "linear"}},
        ],
    )
    in_memory, out_of_core = _fit_both(frame, tmp_path, glm, te_run_records=700)
    assert "x2:TE(brand)" in in_memory.feature_names
    _assert_equal(in_memory, out_of_core, design_frame(400, seed=5))


def test_a_drawn_seed_is_recorded_and_reproduces_the_fit(frame) -> None:
    glm = _glm(seed=None)
    first = rs.glm_dict(data=frame, **glm).fit()
    seed = first.solver_settings["target_encoding_seed"]
    assert isinstance(seed, int)
    again = rs.glm_dict(data=frame, **_glm(seed=seed)).fit()
    assert _bits(first.params) == _bits(again.params)


# ── §7.9: cross-validation with target encoding ─────────────────────────────


def test_cv_with_target_encoding_equals_in_memory(frame, tmp_path) -> None:
    fit = {
        "cv": 4,
        "regularization": "elastic_net",
        "l1_ratio": 0.3,
        "n_alphas": 3,
        "cv_seed": 5,
    }
    in_memory, out_of_core = _fit_both(
        frame, tmp_path, _glm(weights="weight"), fit, te_run_records=400
    )
    _assert_equal(in_memory, out_of_core)
    assert in_memory._regularization_path_info.fold_safe_target_encoding


class Sentinel(ValueError):
    """A ValueError an on_batch raises: never a failed alpha."""


@pytest.mark.parametrize("kind", ["fold", "moments", "te_sort"])
def test_an_on_batch_exception_in_cv_reaches_the_caller(frame, tmp_path, kind) -> None:
    seen = {"n": 0}

    def on_batch(progress):
        if progress.pass_kind == kind:
            seen["n"] += 1
            if seen["n"] == 3:
                raise Sentinel("stop")

    with pytest.raises(Sentinel, match="stop"):
        rs.glm_dict(data=_source(frame), **_glm()).fit(
            cv=3,
            regularization="ridge",
            n_alphas=3,
            out_of_core=rs.OutOfCoreOptions(
                scratch_dir=tmp_path, te_run_records=500, on_batch=on_batch
            ),
        )
    assert list(tmp_path.iterdir()) == []


# ── T22: spill files and their errors ───────────────────────────────────────


@pytest.mark.parametrize("kind", ["te_sort", "te_merge", "te_walk", "te_average", "candidate"])
def test_an_on_batch_abort_removes_every_spill_file(frame, tmp_path, kind) -> None:
    class Stop(Exception):
        pass

    def on_batch(progress):
        if progress.pass_kind == kind:
            raise Stop(kind)

    with pytest.raises(Stop, match=kind):
        rs.glm_dict(data=_source(frame), **_glm()).fit(
            out_of_core=rs.OutOfCoreOptions(
                scratch_dir=tmp_path, te_run_records=300, on_batch=on_batch
            )
        )
    assert list(tmp_path.iterdir()) == []


def test_an_error_after_target_encoding_removes_the_spill(frame, tmp_path) -> None:
    calls = {"n": 0}

    def changing(columns):
        calls["n"] += 1
        data = frame if calls["n"] < 8 else frame.with_columns(pl.col("x1") * 2)
        yield data.select(columns)

    with pytest.raises(rs.exceptions.BatchSourceChangedError):
        rs.glm_dict(data=rs.BatchSource(changing, frame.schema), **_glm()).fit(
            out_of_core=rs.OutOfCoreOptions(scratch_dir=tmp_path, te_run_records=1000)
        )
    assert list(tmp_path.iterdir()) == []


def test_target_encoding_needs_a_scratch_dir_before_reading(frame) -> None:
    source = _source(frame)
    with pytest.raises(OutOfCoreOptionError, match="scratch_dir"):
        rs.glm_dict(data=source, **_glm()).fit(out_of_core=rs.OutOfCoreOptions())
    assert source.opened["passes"] == 0


def test_an_unusable_scratch_dir_raises_spill_error(frame, tmp_path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(SpillError, match="missing"):
        rs.glm_dict(data=_source(frame), **_glm()).fit(
            out_of_core=rs.OutOfCoreOptions(scratch_dir=missing, te_run_records=1000)
        )
    assert not missing.exists()


@pytest.mark.parametrize("cv", [None, 10])
def test_the_spill_peak_stays_within_the_plan(frame, tmp_path, cv) -> None:
    """Two TE terms with four permutations, alone and in CV with target encoding at
    ten folds (T22)."""
    peak = {"bytes": 0}

    def on_batch(progress):
        size = sum(entry.stat().st_size for entry in os.scandir(tmp_path))
        peak["bytes"] = max(peak["bytes"], size)

    glm = _glm(terms={"brand": {"type": "target_encoding"}, "cat_b": {"type": "target_encoding"}})
    fit = {} if cv is None else {"cv": cv, "regularization": "ridge", "n_alphas": 2}
    result = rs.glm_dict(data=_source(frame, sizes=(500,)), **glm).fit(
        out_of_core=rs.OutOfCoreOptions(
            scratch_dir=tmp_path, te_run_records=100, on_batch=on_batch
        ),
        **fit,
    )
    assert peak["bytes"] > 0
    assert peak["bytes"] <= result.memory_plan.spill_bytes
    assert list(tmp_path.iterdir()) == []
