"""The frozen design: a model's data-dependent design state, learned from slices.

``specs/out-of-core-fit.md`` §6 splits building a design matrix into learning and
encoding. :func:`learn_frozen_design` reads slices of the frame and learns everything
a design takes from its data (§6.1):

- the labels and codes of every discovered categorical, frequency-encoded and
  ``levels=`` column (:class:`~rustystats.category_keys.CategoryKeyLearner`, §6.3);
- frequency-encoding counts per label, and ``max_count``;
- spline boundary knots, and interior knots as exact order statistics over four
  passes (``OrderStatistics``, §6.5);
- target encoding (§6.6): the prior and "auto" prior weight (freeze pass), each
  label's totals in row order (te_stats), and the training values in DEC-1's keyed
  order (te_sort; in RAM, or spilled to a scratch directory out of core).

From that state alone, before any row is encoded, the design lays out its columns in
the term order of ``InteractionBuilder._build_design_matrix_core``: the names, the
:class:`~rustystats.interactions.TermSlot` ranges and p. It also declares the encoder
scratch per row (§6.8, §11). :meth:`FrozenDesign.encode_into` then writes every term
of a slice straight into its columns of a caller's block, with today's column
arithmetic. :func:`build_design_in_memory` allocates X once and encodes consecutive
slices into its rows. X, the names and every piece of builder state that fitting,
prediction and serialisation read equal what
``InteractionBuilder.build_design_matrix_from_parsed`` produces.

:func:`learn_frozen_designs` learns several designs in the same passes: the full
data's, and each cross-validation fold's from its training rows (§7.9). A fold's
design encodes every row: training rows with the fold's training values, validation
rows as new data (target encoding from the fold's full statistics, unseen labels as
prediction treats them).
"""

from __future__ import annotations

import inspect
import itertools
import re
import secrets
import weakref
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from numbers import Real
from typing import Protocol

import numpy as np
import polars as pl

from rustystats._rustystats import (
    OrderStatistics,
    TargetEncodingLevelTotals,
    TargetEncodingPrior,
    TargetEncodingTerm,
    cv_fold_ids_py,
)
from rustystats.category_keys import (
    ENTRY_OVERHEAD_BYTES,
    CategoryKeyLearner,
    FrozenCategories,
    LearnedStateBudget,
    check_key_dtype,
    training_keys,
)
from rustystats.exceptions import (
    DesignMatrixError,
    DesignTooWideError,
    SpillError,
    ValidationError,
)
from rustystats.interactions import (
    CategoricalEncoding,
    CategoricalTermSpec,
    ConstraintTermSpec,
    FrequencyEncodingTermSpec,
    IdentityTermSpec,
    InteractionBuilder,
    InteractionTerm,
    ParsedFormula,
    TargetEncodingTermSpec,
    TermSlot,
    check_monotone_by_category,
    interaction_spline_monotonicity,
    refuse_fixed_df_monotone_interaction,
    refuse_monotone_tensor_interaction,
)
from rustystats.splines import SplineTerm

__all__ = [
    "E_LEARN",
    "FoldRows",
    "FrozenDesign",
    "FrozenDesignUnsupported",
    "FrozenFrequency",
    "FrozenTarget",
    "TargetInputs",
    "build_design_in_memory",
    "build_design_matrix",
    "draw_target_encoding_seed",
    "learn_frozen_design",
    "learn_frozen_designs",
    "learn_target_values",
]

#: Rows per learning and encoding slice in memory (§4.3).
E_LEARN = 65_536

#: Slices of one frame: a re-iterable sequence, or a callable opening a fresh pass
#: (optionally taking the pass kind: "freeze", "knots", "te_stats" or "te_sort").
Slices = Callable[..., Iterable[pl.DataFrame]] | Sequence[pl.DataFrame]


class FrozenDesignUnsupported(Exception):
    """The model needs ``InteractionBuilder``; raised before any learned state is applied.

    For a dtype other than float64, a spline whose knots today's code places from
    non-finite values or with a ``df``/``degree`` the knot functions do not accept, a
    categorical column whose dtype has no training key, an empty frame, and a
    target-encoded model learned without :class:`TargetInputs`.
    """


@dataclass(frozen=True)
class FoldRows:
    """Cross-validation fold ``fold``'s training rows: the rows whose DEC-2 fold
    ``mix64(seed ^ mix64(r)) mod folds`` is another fold (§7.8, §7.9)."""

    fold: int
    folds: int
    seed: int

    def training(self, first_row: int, n: int) -> np.ndarray:
        """Whether each of the global rows ``first_row..first_row + n`` is a training row."""
        return cv_fold_ids_py(first_row, n, self.folds, self.seed) != self.fold


@dataclass(frozen=True)
class TargetInputs:
    """What target-encoded terms learn from besides the slices (§6.6)."""

    #: A slice's response as float64, given the slice and its first global row.
    target: Callable[[pl.DataFrame, int], np.ndarray]
    #: A slice's raw exposure as float64; None when the terms are not exposure-weighted.
    exposure: Callable[[pl.DataFrame, int], np.ndarray] | None
    #: The keyed order's seed (DEC-1) of a design learned from every row; a fold's
    #: design keys its order by the fold's CV seed.
    seed: int
    #: Spill sort runs of ``run_records`` records under ``scratch_dir``; without it every
    #: record stays in RAM. Both give the same values. A callable is asked once, when
    #: the te_sort passes start (the sort is sized after the freeze pass, §11).
    scratch_dir: str | None = None
    run_records: int | Callable[[], int] | None = None
    #: Called as a spill stage runs: (pass kind, records done, records total).
    progress: Callable[[str, int, int], None] | None = None
    #: Takes ownership of a spilled term column file; it must outlive the fit's passes.
    keep: Callable[[str], None] | None = None


def draw_target_encoding_seed() -> int:
    """A seed for a fit whose model gives none (DEC-3): drawn once and recorded."""
    return secrets.randbits(63)


# ── What a parsed formula learns from ───────────────────────────────────────


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _discovered_columns(parsed: ParsedFormula) -> list[str]:
    """Columns encoded with discovered levels, in the order today's build first meets them."""
    main = [var for var in parsed.main_effects if var in parsed.categorical_vars]
    factors = [
        factor
        for interaction in parsed.interactions
        for factor, is_cat in zip(interaction.factors, interaction.categorical_flags)
        if is_cat
    ]
    return _unique(main + factors)


def _frequency_variables(term: FrequencyEncodingTermSpec) -> list[str]:
    if term.interaction_vars is not None and len(term.interaction_vars) >= 2:
        return list(term.interaction_vars)
    return [term.var_name]


def _explicit_columns(parsed: ParsedFormula) -> list[str]:
    return _unique(term.var_name for term in parsed.categorical_terms if term.levels)


def _spline_terms(parsed: ParsedFormula) -> list[SplineTerm]:
    """Every spline object a build can transform: main effects, then interaction-local."""
    splines: dict[int, SplineTerm] = {}
    for spline in parsed.spline_terms:
        splines.setdefault(id(spline), spline)
    for interaction in parsed.interactions:
        for spline in (interaction.spline_terms or {}).values():
            splines.setdefault(id(spline), spline)
    return list(splines.values())


def _slice_passes(slices: Slices) -> Callable[[str], Iterator[pl.DataFrame]]:
    """``open(kind)`` for each learning pass; a callable may take the pass kind."""
    if callable(slices):
        try:
            takes_kind = bool(inspect.signature(slices).parameters)
        except (TypeError, ValueError):
            takes_kind = False
        if takes_kind:
            return lambda kind: iter(slices(kind))
        return lambda kind: iter(slices())
    if iter(slices) is slices:
        raise ValidationError(
            "slices must be re-iterable (a sequence) or a callable opening a new pass; "
            "learning reads them more than once."
        )
    return lambda kind: iter(slices)


def _float_column(rows: pl.DataFrame, column: str) -> np.ndarray:
    """A slice's column as the training builder reads a continuous input."""
    return np.ascontiguousarray(rows[column].to_numpy().astype(np.float64, copy=False))


# ── Spline knots ────────────────────────────────────────────────────────────


def knot_rank(i: int, m: int, n: int) -> int:
    """0-based rank of interior knot ``i`` (1-based) of ``m`` among ``n`` sorted values.

    ``floor(i / (m + 1) · (n − 1))`` in float64, as ``splines::knots::knot_rank``.
    """
    return int((i / (m + 1)) * float(max(n - 1, 0)))


def _is_count(value: object) -> bool:
    return isinstance(value, int | np.integer) and value >= 0


@dataclass
class _KnotPlan:
    spline: SplineTerm
    column: str
    #: Interior knots to place, or None when they were given.
    n_interior: int | None
    #: Boundary knots, or None when they are learned from the column.
    boundary: tuple[float, float] | None

    @property
    def reads_column(self) -> bool:
        return self.boundary is None or bool(self.n_interior)


def _knot_plan(spline: SplineTerm) -> _KnotPlan | None:
    """What the first ``transform`` would learn from the column, or None when nothing."""
    if spline._computed_boundary_knots is not None:
        if spline._computed_internal_knots is None:
            raise FrozenDesignUnsupported(f"spline {spline!r} has boundary but no interior knots")
        return None
    given = spline.boundary_knots
    if spline._computed_internal_knots is not None:
        boundary = None if given is None else (float(given[0]), float(given[1]))
        return _KnotPlan(spline, spline.var_name, None, boundary)

    # compute_knots_py / compute_knots_natural_py take usize df and degree, and an
    # optional (f64, f64) boundary; anything else fails there today.
    uses_degree = spline.spline_type in ("bs", "ms")
    if not _is_count(spline.df) or (uses_degree and not _is_count(spline.degree)):
        raise FrozenDesignUnsupported(f"spline {spline!r} has a df or degree knots cannot use")
    if given is not None:
        if len(given) != 2 or not all(isinstance(bound, Real) for bound in given):
            raise FrozenDesignUnsupported(f"spline {spline!r} has unusable boundary knots")
        given = (float(given[0]), float(given[1]))
    df = int(spline.df)
    if uses_degree:
        degree = int(spline.degree)
        n_interior = df - degree - 1 if df > degree + 1 else 0
    else:
        n_interior = max(df - 1, 0)
    return _KnotPlan(spline, spline.var_name, n_interior, given)


def _total_min(a: float, b: float) -> float:
    """The smaller of two finite floats, with −0.0 below +0.0."""
    return a if (a < b or (a == b and np.signbit(a))) else b


def _total_max(a: float, b: float) -> float:
    """The larger of two finite floats, with +0.0 above −0.0."""
    return a if (a > b or (a == b and not np.signbit(a))) else b


class _ColumnRange:
    """Running minimum and maximum of a spline column, refusing non-finite values.

    Today's minimum is a SIMD fold whose choice between −0.0 and +0.0 depends on the
    lanes the values fall in; here a zero extreme is −0.0 for the minimum and +0.0 for
    the maximum whenever that sign occurs, so the result does not depend on slicing.
    """

    def __init__(self, column: str) -> None:
        self.column = column
        self.low = np.inf
        self.high = -np.inf

    def observe(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        if not np.isfinite(values).all():
            raise FrozenDesignUnsupported(f"spline input {self.column!r} is not finite")
        low = float(values.min())
        high = float(values.max())
        if low == 0.0 or high == 0.0:
            zeros = np.signbit(values[values == 0.0])
            if low == 0.0:
                low = -0.0 if zeros.any() else 0.0
            if high == 0.0:
                high = 0.0 if not zeros.all() else -0.0
        self.low = _total_min(self.low, low)
        self.high = _total_max(self.high, high)


#: One radix-select histogram: 65,536 counts of 8 bytes (§6.5).
KNOT_HISTOGRAM_BYTES = 65_536 * 8


def _interior_knots(
    learners: list[_DesignLearner],
    open_slices: Callable[[str], Iterator[pl.DataFrame]],
    admit: Callable[[str, int], None] | None = None,
) -> list[dict[int, list[float]]]:
    """Each design's interior knots per plan index, as exact order statistics over its
    training rows, in four passes the designs share (§6.5).

    ``admit("knots", bytes)`` is told the passes' histograms before any is allocated:
    one per selector in the first pass and at most one per rank afterwards, so at most
    one per rank (§11).
    """
    ranks: list[dict[str, list[tuple[int, list[int]]]]] = []
    for learner in learners:
        design_ranks: dict[str, list[tuple[int, list[int]]]] = {}
        for index, plan in enumerate(learner.plans):
            if plan.n_interior:
                m = plan.n_interior
                plan_ranks = [knot_rank(i, m, learner.n) for i in range(1, m + 1)]
                design_ranks.setdefault(plan.column, []).append((index, plan_ranks))
        ranks.append(design_ranks)
    if not any(ranks):
        return [{} for _ in learners]
    if admit is not None:
        total = sum(
            len(plan_ranks)
            for design_ranks in ranks
            for entries in design_ranks.values()
            for _index, plan_ranks in entries
        )
        admit("knots", total * KNOT_HISTOGRAM_BYTES)

    selectors = [
        {
            column: OrderStatistics(
                [rank for _, plan_ranks in entries for rank in plan_ranks], learner.n
            )
            for column, entries in design_ranks.items()
        }
        for learner, design_ranks in zip(learners, ranks)
    ]
    every = [selector for design in selectors for selector in design.values()]
    passes = max(selector.passes_needed() for selector in every)
    for pass_index in range(passes):
        for selector in every:
            selector.begin_pass(pass_index)
        first_row = 0
        masks = _FoldMasks()
        for rows in open_slices("knots"):
            for learner, design in zip(learners, selectors):
                if not design:
                    continue
                mask = masks(learner.fold, first_row, rows.height)
                training = rows if mask is None else rows.filter(pl.Series(mask))
                for column, selector in design.items():
                    selector.observe(_float_column(training, column))
            first_row += rows.height
        for selector in every:
            selector.end_pass()

    knots: list[dict[int, list[float]]] = []
    for design_ranks, design in zip(ranks, selectors):
        design_knots: dict[int, list[float]] = {}
        for column, entries in design_ranks.items():
            values = list(design[column].results())
            start = 0
            for index, plan_ranks in entries:
                design_knots[index] = values[start : start + len(plan_ranks)]
                start += len(plan_ranks)
        knots.append(design_knots)
    return knots


# ── Frequency encoding ──────────────────────────────────────────────────────

#: Bytes of a learned count (UInt64).
_COUNT_BYTES = 8


def _key_bytes(keys: pl.Series) -> int:
    """Payload bytes of a training-key column: string and binary lengths, else its buffer."""
    if keys.dtype == pl.String:
        return int(keys.str.len_bytes().sum())
    if keys.dtype == pl.Binary:
        return int(keys.bin.size().sum())
    return int(keys.estimated_size())


class _FrequencyLearner:
    """Counts of each distinct training-key tuple of a frequency-encoded term.

    Each slice's counts come from a Polars ``group_by`` over the key columns, with one
    representative value per key so its label can be formed once categories freeze.
    The tuples a slice adds to the table are charged to the budget before the table
    grows (§6.4).
    """

    def __init__(
        self,
        term: FrequencyEncodingTermSpec,
        budget: LearnedStateBudget,
    ) -> None:
        self.term = term
        self.variables = _frequency_variables(term)
        self._budget = budget
        self._keys = [f"k{j}" for j in range(len(self.variables))]
        self._values = [f"v{j}" for j in range(len(self.variables))]
        self._table: pl.DataFrame | None = None
        self._entries = 0
        self._nbytes = 0

    def _aggregate(self, frame: pl.DataFrame, count: pl.Expr) -> pl.DataFrame:
        return frame.group_by(self._keys).agg(
            *(pl.col(value).first() for value in self._values), count.alias("count")
        )

    def observe(self, rows: pl.DataFrame) -> None:
        """Count one slice's key tuples.

        Raises
        ------
        LearnedStateTooLargeError
            When the slice's new tuples would cross the budget; the table is unchanged.
        """
        columns = {}
        for key, value, variable in zip(self._keys, self._values, self.variables):
            columns[key] = training_keys(rows[variable])
            columns[value] = rows[variable]
        # The slice's distinct tuples, a working set bounded by the slice.
        counts = self._aggregate(pl.DataFrame(columns), pl.len().cast(pl.UInt64))
        new = counts
        if self._table is not None:
            # A struct key compares null fields as equal, as ``group_by`` groups them.
            tuples = pl.struct(self._keys).alias("tuple")
            new = counts.with_columns(tuples).join(
                self._table.select(tuples), on="tuple", how="anti"
            )
        if new.height:
            # Each new tuple keeps its keys and one representative value of each
            # variable, whose payload is the key's, and its count.
            payload = sum(_key_bytes(new[key]) for key in self._keys)
            cost = 2 * payload + new.height * (ENTRY_OVERHEAD_BYTES + _COUNT_BYTES)
            self._budget.charge(
                cost, term=self.term.var_name, cardinality=self._entries + new.height
            )
            self._entries += new.height
            self._nbytes += cost
        if self._table is None:
            self._table = counts
        else:
            self._table = self._aggregate(pl.concat([self._table, counts]), pl.col("count").sum())

    def finish(self, categories: dict[str, FrozenCategories]) -> FrozenFrequency:
        table = self._table
        assert table is not None  # at least one slice was observed
        counts = table["count"].to_numpy().astype(np.int64)
        codes = [
            categories[variable].encode(table[value].alias(variable))
            for value, variable in zip(self._values, self.variables)
        ]
        if len(self.variables) == 1:
            labels = categories[self.variables[0]].labels
            by_code = np.zeros(len(labels), dtype=np.int64)
            np.add.at(by_code, codes[0], counts)
            level_counts = {label: int(count) for label, count in zip(labels, by_code) if count > 0}
            return FrozenFrequency(
                term=self.term,
                variables=tuple(self.variables),
                level_counts=level_counts,
                max_count=max(level_counts.values(), default=1),
                counts_by_code=by_code.astype(np.float64),
                code_table=None,
            )

        # Today's label is each variable's discovered label joined with ":", so
        # code tuples that join to one string share its count.
        label_lists = [categories[variable].labels for variable in self.variables]
        joined = [
            ":".join(labels[code] for labels, code in zip(label_lists, row))
            for row in zip(*(code.tolist() for code in codes))
        ]
        level_counts: dict[str, int] = {}
        for label, count in zip(joined, counts.tolist()):
            level_counts[label] = level_counts.get(label, 0) + count
        code_columns = {f"c{j}": code for j, code in enumerate(codes)}
        code_table = (
            pl.DataFrame(code_columns)
            .with_columns(
                pl.Series("count", [float(level_counts[label]) for label in joined], pl.Float64)
            )
            .unique(subset=list(code_columns), keep="first", maintain_order=True)
        )
        return FrozenFrequency(
            term=self.term,
            variables=tuple(self.variables),
            level_counts=dict(sorted(level_counts.items())),
            max_count=max(level_counts.values(), default=1),
            counts_by_code=None,
            code_table=code_table,
        )


@dataclass(frozen=True)
class FrozenFrequency:
    """A frequency-encoded term's frozen counts per label, and how to encode a slice."""

    term: FrequencyEncodingTermSpec
    variables: tuple[str, ...]
    level_counts: dict[str, int]
    max_count: int
    #: Single variable: count per discovered label code, as float64.
    counts_by_code: np.ndarray | None
    #: Interaction: each learned code tuple (``c0``, ``c1``, ...) and its label's count.
    code_table: pl.DataFrame | None

    def stats(self) -> dict:
        """The ``_fe_stats`` entry today's build stores for prediction."""
        return {
            "level_counts": dict(self.level_counts),
            "max_count": self.max_count,
            "interaction_vars": self.term.interaction_vars,
        }

    def encode(
        self,
        codes: Sequence[np.ndarray],
        out: np.ndarray | None = None,
        *,
        lenient: bool = False,
    ) -> np.ndarray:
        """``count / max_count`` per row, with the arithmetic of ``frequency_encode_py``.

        ``codes`` holds each variable's discovered-label codes for the rows. The result
        is written into ``out`` when it is given. With ``lenient``, a label (code -1)
        or combination that was not learned counts 0, as prediction encodes it.
        """
        if self.counts_by_code is not None:
            level = codes[0]
            if lenient and (level < 0).any():
                counts = np.where(level >= 0, self.counts_by_code[np.maximum(level, 0)], 0.0)
            else:
                counts = self.counts_by_code[level]
        else:
            assert self.code_table is not None
            keys = list(self.code_table.columns[:-1])
            found = (
                pl.DataFrame(dict(zip(keys, codes)))
                .join(self.code_table, on=keys, how="left", maintain_order="left")
                .get_column("count")
            )
            if found.null_count():
                if not lenient:
                    raise DesignMatrixError(
                        f"FE({self.term.var_name}) met a combination of labels that was "
                        f"not learned; the data changed after it was learned."
                    )
                found = found.fill_null(0.0)
            counts = found.to_numpy()
        # Rust: `count as f64 / max_count as f64`, one correctly rounded division.
        return np.divide(counts, np.float64(self.max_count), out=out)


def _frequency_for(
    frequency: Iterable[FrozenFrequency], term: FrequencyEncodingTermSpec
) -> FrozenFrequency:
    """The learned state of ``term``: learning finishes one per frequency-encoded term."""
    return next(frozen for frozen in frequency if frozen.term is term)


# ── Target encoding (§6.6) ──────────────────────────────────────────────────

#: Bytes per learned target-encoding label beyond its text: its three totals.
_TOTALS_BYTES = 24
#: Rows per fold-id chunk when a spilled column is read from an unexpected row.
_ORDINAL_CHUNK = 1 << 20


def _target_variables(term: TargetEncodingTermSpec) -> list[str]:
    if term.interaction_vars is not None and len(term.interaction_vars) >= 2:
        return list(term.interaction_vars)
    return [term.var_name]


def _n_permutations(term: TargetEncodingTermSpec) -> int:
    try:
        n = int(term.n_permutations)
    except (TypeError, ValueError) as exc:
        raise ValidationError("target_encoding n_permutations must be a positive integer.") from exc
    if n < 1:
        raise ValidationError("target_encoding n_permutations must be a positive integer.")
    return n


def _prior_weight(spec: float | str, prior: TargetEncodingPrior) -> float:
    """``_resolve_target_encoding_prior_weight``, "auto" from the block-ordered sums."""
    message = "target_encoding prior_weight must be numeric, non-negative, or 'auto'."
    if isinstance(spec, str):
        if spec.lower() != "auto":
            raise ValidationError(message)
        return float(prior.auto_prior_weight())
    try:
        resolved = float(spec)
    except (TypeError, ValueError) as exc:
        raise ValidationError(message) from exc
    if not np.isfinite(resolved) or resolved < 0.0:
        raise ValidationError("target_encoding prior_weight must be finite and non-negative.")
    return resolved


class _FoldMasks:
    """A slice's training mask for each fold design, from one fold-id pass per slice."""

    def __init__(self) -> None:
        self._key: tuple[int, int, int, int] | None = None
        self._ids: np.ndarray | None = None

    def __call__(self, fold: FoldRows | None, first_row: int, n: int) -> np.ndarray | None:
        if fold is None:
            return None
        key = (first_row, n, fold.folds, fold.seed)
        if key != self._key:
            self._ids = cv_fold_ids_py(first_row, n, fold.folds, fold.seed)
            self._key = key
        return self._ids != fold.fold


class _TrainingValues:
    """A term's training values, one per training row in row order: in RAM, or a
    spilled column file read slice by slice (seeking only when a pass restarts)."""

    def __init__(
        self,
        fold: FoldRows | None,
        *,
        array: np.ndarray | None = None,
        path: str | None = None,
    ) -> None:
        self._fold = fold
        self._array = array
        self.path = path
        self._handle = None
        self._position = 0
        self._next_row = 0
        self._next_ordinal = 0

    def _ordinal(self, first_row: int) -> int:
        """Training rows before global row ``first_row``."""
        if first_row == self._next_row:
            return self._next_ordinal
        if first_row == 0 or self._fold is None:
            return first_row
        count = 0
        for start in range(0, first_row, _ORDINAL_CHUNK):
            n = min(_ORDINAL_CHUNK, first_row - start)
            count += int(np.count_nonzero(self._fold.training(start, n)))
        return count

    def read(self, first_row: int, n: int, training: np.ndarray | None) -> np.ndarray:
        """The values of the training rows among global rows ``first_row..first_row + n``."""
        ordinal = self._ordinal(first_row)
        count = n if training is None else int(np.count_nonzero(training))
        if self._array is not None:
            values = self._array[ordinal : ordinal + count]
            if values.shape[0] != count:
                raise DesignMatrixError(
                    f"Rows {first_row}..{first_row + n - 1} go past the "
                    f"{self._array.shape[0]} rows target encoding learned from."
                )
        else:
            values = self._read_file(ordinal, count)
        self._next_row, self._next_ordinal = first_row + n, ordinal + count
        return values

    def _read_file(self, ordinal: int, count: int) -> np.ndarray:
        assert self.path is not None
        try:
            if self._handle is None:
                self._handle = open(self.path, "rb")  # noqa: SIM115
                self._position = 0
            if self._position != ordinal:
                self._handle.seek(8 * ordinal)
            data = self._handle.read(8 * count)
        except OSError as exc:
            raise SpillError(f"Cannot read the spill file {self.path!r}: {exc}") from exc
        if len(data) != 8 * count:
            raise SpillError(
                f"The spill file {self.path!r} ends before row {ordinal + count}: it was "
                "changed or truncated during the fit."
            )
        self._position = ordinal + count
        return np.frombuffer(data, dtype="<f8")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


@dataclass(eq=False)
class FrozenTarget:
    """A target-encoded term's learned state (§6.6).

    Its labels (each variable's discovered label, joined with ":" for an
    interaction), the prior and resolved prior weight, each label's totals over the
    training rows in row order (the statistics prediction encodes with), and the
    training values. A fold's design (``fold``) encodes its validation rows from the
    full statistics, a label it did not learn from as the prior.
    """

    term: TargetEncodingTermSpec
    variables: tuple[str, ...]
    #: Label of each level index, sorted.
    labels: list[str]
    #: Interaction: each learned code tuple (``c0``, ``c1``, ...) and its level index.
    code_table: pl.DataFrame | None
    prior: float
    prior_weight: float
    exposure_weighted: bool
    n_permutations: int
    #: The keyed order's seed: the model's, or for a fold's design the CV seed (as
    #: today's fold builder encoded with it).
    seed: int = 0
    fold: FoldRows | None = None
    sums: np.ndarray | None = None
    counts: np.ndarray | None = None
    exposures: np.ndarray | None = None
    values: _TrainingValues | None = None
    _full: np.ndarray | None = None
    _cache: tuple[int, int, np.ndarray] | None = None

    @property
    def name(self) -> str:
        return f"TE({self.term.var_name})"

    def levels(self, codes: Sequence[np.ndarray]) -> np.ndarray:
        """Each row's level index from its variables' codes; -1 for one not learned."""
        if self.code_table is None:
            return np.ascontiguousarray(codes[0], dtype=np.int32)
        keys = list(self.code_table.columns[:-1])
        found = (
            pl.DataFrame(dict(zip(keys, codes)))
            .join(self.code_table, on=keys, how="left", maintain_order="left")
            .get_column("level")
            .fill_null(-1)
        )
        return np.ascontiguousarray(found.to_numpy(), dtype=np.int32)

    def stats(self) -> dict:
        """The ``_te_stats`` entry prediction and serialisation read."""
        assert self.sums is not None and self.counts is not None and self.exposures is not None
        if self.exposure_weighted:
            level_stats = {
                label: (float(total), float(exposure))
                for label, total, exposure in zip(self.labels, self.sums, self.exposures)
            }
        else:
            level_stats = {
                label: (float(total), int(count))
                for label, total, count in zip(self.labels, self.sums, self.counts)
            }
        return {
            "prior": self.prior,
            "stats": level_stats,
            "prior_weight": self.prior_weight,
            "prior_weight_spec": self.term.prior_weight,
            "used_exposure_weighted": self.exposure_weighted,
            "interaction_vars": self.term.interaction_vars,
        }

    def _full_values(self) -> np.ndarray:
        """Each level's value from its full statistics, as ``apply_target_encoding``."""
        if self._full is None:
            assert self.sums is not None and self.counts is not None
            shrink = self.prior * self.prior_weight
            if self.exposure_weighted:
                assert self.exposures is not None
                self._full = (self.sums + shrink) / (self.exposures + self.prior_weight)
            else:
                self._full = (self.sums + shrink) / (
                    self.counts.astype(np.float64) + self.prior_weight
                )
        return self._full

    def column(self, codes: _SliceCodes, n: int) -> np.ndarray:
        """The term's value for each row of a slice (kept until the next slice)."""
        # ``encode_into`` refuses a target-encoded design's slice without its first row.
        first_row = codes.first_row
        assert first_row is not None
        if self._cache is not None and self._cache[:2] == (first_row, n):
            return self._cache[2]
        if self.values is None:
            raise DesignMatrixError(f"{self.name} has no training values yet.")
        training = codes.training
        if training is None:
            column = self.values.read(first_row, n, None)
        else:
            column = np.empty(n, dtype=np.float64)
            column[training] = self.values.read(first_row, n, training)
            held_out = ~training
            if held_out.any():
                levels = self.levels([codes(variable) for variable in self.variables])[held_out]
                full = self._full_values()
                column[held_out] = np.where(levels >= 0, full[np.maximum(levels, 0)], self.prior)
        self._cache = (first_row, n, column)
        return column

    def close(self) -> None:
        self._cache = None
        if self.values is not None:
            self.values.close()


class _TargetLearner:
    """A target-encoded term's freeze-pass learning: an interaction's code tuples."""

    def __init__(self, term: TargetEncodingTermSpec, budget: LearnedStateBudget) -> None:
        self.term = term
        self.variables = _target_variables(term)
        self.n_permutations = _n_permutations(term)
        self._budget = budget
        self._tuples = _FrequencyLearner(term, budget) if len(self.variables) > 1 else None

    def observe(self, rows: pl.DataFrame) -> None:
        if self._tuples is not None:
            self._tuples.observe(rows)

    def finish(
        self,
        categories: dict[str, FrozenCategories],
        prior: TargetEncodingPrior,
        exposure_weighted: bool,
        fold: FoldRows | None,
        seed: int,
    ) -> FrozenTarget:
        code_table = None
        if self._tuples is None:
            labels = categories[self.variables[0]].labels
        else:
            table = self._tuples._table
            assert table is not None
            codes = [
                categories[variable].encode(table[value].alias(variable))
                for value, variable in zip(self._tuples._values, self.variables)
            ]
            # Today's label joins each variable's discovered label with ":", so code
            # tuples that join to one string share a level.
            label_lists = [categories[variable].labels for variable in self.variables]
            joined = [
                ":".join(labels[code] for labels, code in zip(label_lists, row))
                for row in zip(*(code.tolist() for code in codes))
            ]
            labels = sorted(set(joined))
            index = {label: level for level, label in enumerate(labels)}
            code_table = (
                pl.DataFrame({f"c{j}": code for j, code in enumerate(codes)})
                .with_columns(
                    pl.Series("level", [index[label] for label in joined], dtype=pl.Int32)
                )
                .unique(subset=[f"c{j}" for j in range(len(codes))], maintain_order=True)
            )
        self._budget.charge(
            sum(len(label) for label in labels)
            + len(labels) * (ENTRY_OVERHEAD_BYTES + _TOTALS_BYTES),
            term=f"TE({self.term.var_name})",
            cardinality=len(labels),
        )
        return FrozenTarget(
            term=self.term,
            variables=tuple(self.variables),
            labels=list(labels),
            code_table=code_table,
            prior=float(prior.prior()),
            prior_weight=_prior_weight(self.term.prior_weight, prior),
            exposure_weighted=exposure_weighted,
            n_permutations=self.n_permutations,
            seed=seed,
            fold=fold,
        )


class _DesignLearner:
    """One design's freeze-pass learners: the full data's, or a fold's over its
    training rows."""

    def __init__(
        self,
        parsed: ParsedFormula,
        fold: FoldRows | None,
        budget: LearnedStateBudget,
        target: TargetInputs | None,
    ) -> None:
        if parsed.target_encoding_terms and target is None:
            raise FrozenDesignUnsupported("target-encoded terms learn from TargetInputs")
        self.parsed = parsed
        self.fold = fold
        self.discovered = _discovered_columns(parsed)
        self.frequency = [
            _FrequencyLearner(term, budget) for term in parsed.frequency_encoding_terms
        ]
        self.targets = [_TargetLearner(term, budget) for term in parsed.target_encoding_terms]
        self.exposure_weighted = target is not None and target.exposure is not None
        self.category_columns = _unique(
            self.discovered
            + [variable for learner in self.frequency for variable in learner.variables]
            + [variable for learner in self.targets for variable in learner.variables]
            + _explicit_columns(parsed)
        )
        self.plans = [plan for plan in map(_knot_plan, _spline_terms(parsed)) if plan is not None]
        # §11: learned knots are persistent learned state, charged before the passes
        # that place them (a learned boundary pair, and each interior knot).
        for plan in self.plans:
            knots = (2 if plan.boundary is None else 0) + (plan.n_interior or 0)
            if knots:
                budget.charge(_F64 * knots, term=f"knots of {plan.column}", cardinality=knots)
        self.ranges = {
            plan.column: _ColumnRange(plan.column) for plan in self.plans if plan.reads_column
        }
        self.learners = {
            column: CategoryKeyLearner(column, budget=budget) for column in self.category_columns
        }
        self.prior = TargetEncodingPrior(self.exposure_weighted) if self.targets else None
        #: Training rows seen.
        self.n = 0

    def label_floor(self, column: str) -> int:
        """The fewest discovered labels ``column`` can end with, from its keys so far."""
        learner = self.learners.get(column)
        return 0 if learner is None else learner.label_floor

    def observe(
        self,
        rows: pl.DataFrame,
        mask: np.ndarray | None,
        target: np.ndarray | None,
        exposure: np.ndarray | None,
    ) -> None:
        training = rows if mask is None else rows.filter(pl.Series(mask))
        self.n += training.height
        for column, learner in self.learners.items():
            learner.observe(training[column])
        for learner in self.frequency:
            learner.observe(training)
        for learner in self.targets:
            learner.observe(training)
        for column, column_range in self.ranges.items():
            column_range.observe(_float_column(training, column))
        if self.prior is not None:
            assert target is not None
            self.prior.update(target, exposure, mask)


def _target_arrays(
    inputs: TargetInputs, rows: pl.DataFrame, first_row: int
) -> tuple[np.ndarray, np.ndarray | None]:
    target = np.ascontiguousarray(inputs.target(rows, first_row), dtype=np.float64)
    exposure = (
        None
        if inputs.exposure is None
        else np.ascontiguousarray(inputs.exposure(rows, first_row), dtype=np.float64)
    )
    return target, exposure


def _target_levels(
    frozen: FrozenTarget, categories: dict[str, FrozenCategories], rows: pl.DataFrame
) -> np.ndarray:
    """A slice's level index per row; validation rows of a fold may be unseen (-1)."""
    lenient = frozen.fold is not None
    return frozen.levels(
        [
            categories[variable].encode(rows[variable], lenient=lenient)
            for variable in frozen.variables
        ]
    )


def _target_statistics(
    learners: list[_DesignLearner],
    targets: list[list[FrozenTarget]],
    categories: list[dict[str, FrozenCategories]],
    open_slices: Callable[[str], Iterator[pl.DataFrame]],
    inputs: TargetInputs,
) -> None:
    """The te_stats pass: each label's totals over each design's training rows, in
    row order."""
    totals = [
        [
            TargetEncodingLevelTotals(len(frozen.labels), frozen.exposure_weighted)
            for frozen in design
        ]
        for design in targets
    ]
    first_row = 0
    masks = _FoldMasks()
    for rows in open_slices("te_stats"):
        target, exposure = _target_arrays(inputs, rows, first_row)
        for learner, design, design_totals, cats in zip(learners, targets, totals, categories):
            mask = masks(learner.fold, first_row, rows.height)
            for frozen, total in zip(design, design_totals):
                total.update(_target_levels(frozen, cats, rows), target, exposure, mask)
        first_row += rows.height
    for design, design_totals in zip(targets, totals):
        for frozen, total in zip(design, design_totals):
            frozen.sums, frozen.counts, frozen.exposures = total.totals()


def _target_values(
    targets: list[list[FrozenTarget]],
    categories: list[dict[str, FrozenCategories]],
    open_slices: Callable[[str], Iterator[pl.DataFrame]],
    inputs: TargetInputs,
) -> None:
    """Each term's training values: per design, term and permutation one te_sort
    pass, then the spill stages; one term at a time (§10)."""
    run_records = inputs.run_records
    if inputs.scratch_dir is None:
        run_records = None
    elif callable(run_records):
        run_records = run_records()
    for design, cats in zip(targets, categories):
        for frozen in design:
            fold = frozen.fold
            encoder = TargetEncodingTerm(
                frozen.seed,
                frozen.n_permutations,
                len(frozen.labels),
                frozen.prior,
                frozen.prior_weight,
                frozen.exposure_weighted,
                scratch_dir=inputs.scratch_dir,
                run_records=run_records,
            )
            try:
                for permutation in range(frozen.n_permutations):
                    encoder.begin(permutation)
                    first_row = 0
                    masks = _FoldMasks()
                    for rows in open_slices("te_sort"):
                        target, exposure = _target_arrays(inputs, rows, first_row)
                        mask = masks(fold, first_row, rows.height)
                        encoder.push(
                            first_row, _target_levels(frozen, cats, rows), target, exposure, mask
                        )
                        first_row += rows.height
                    encoder.end(inputs.progress)
                values = encoder.finish(inputs.progress)
            finally:
                encoder.close()
            if isinstance(values, str):
                if inputs.keep is None:
                    raise DesignMatrixError("A spilled target-encoding column needs an owner.")
                inputs.keep(values)
                frozen.values = _TrainingValues(fold, path=values)
            else:
                frozen.values = _TrainingValues(fold, array=values)


def learn_target_values(
    designs: Sequence[FrozenDesign], slices: Slices, target: TargetInputs
) -> None:
    """The te_sort passes and spill stages of designs learned without their
    target-encoding training values (``learn_frozen_designs(..., values=...)``)."""
    pending = [design for design in designs if any(t.values is None for t in design.target)]
    _target_values(
        [[t for t in design.target if t.values is None] for design in pending],
        [design.categories for design in pending],
        _slice_passes(slices),
        target,
    )


# ── Learning ────────────────────────────────────────────────────────────────


@dataclass
class FrozenDesign:
    """A model's learned design state and its column layout.

    Spline knots live on the formula's ``SplineTerm`` objects: while laying out the
    columns, :func:`learn_frozen_design` places them as today's first ``transform``
    would. ``knots`` records what was learned.
    """

    parsed: ParsedFormula
    dtype: np.dtype
    n_rows: int
    #: Columns of the frame the design was learned from, in order.
    columns: tuple[str, ...]
    #: Frozen labels per column: discovered categoricals, frequency-encoded and ``levels=``.
    categories: dict[str, FrozenCategories]
    #: Columns encoded as discovered categoricals, in the order today's build meets them.
    discovered: tuple[str, ...]
    #: One entry per ``parsed.frequency_encoding_terms``, in order.
    frequency: tuple[FrozenFrequency, ...]
    #: Learned spline knots: (spline, interior or None when given, boundary).
    knots: tuple[tuple[SplineTerm, list[float] | None, tuple[float, float]], ...]
    #: The columns, their terms and their writers, laid out before any row is encoded.
    layout: _Layout
    #: One entry per ``parsed.target_encoding_terms``, in order.
    target: tuple[FrozenTarget, ...] = ()
    #: The cross-validation fold whose training rows the design was learned from.
    fold: FoldRows | None = None
    #: Rows the design was learned from (``n_rows``, or the fold's training rows).
    training_rows: int = 0

    @property
    def names(self) -> tuple[str, ...]:
        """Design column names, in order."""
        return self.layout.names

    @property
    def term_slots(self) -> tuple[TermSlot, ...]:
        """Each source term's column range, as ``InteractionBuilder._term_slots``."""
        return self.layout.term_slots

    @property
    def p(self) -> int:
        """Design columns."""
        return len(self.layout.names)

    @property
    def scratch_bytes_per_row(self) -> int:
        """Encoder scratch per encoded row (§11).

        The codes kept for the whole slice, plus the most any one term holds while it
        writes its columns.
        """
        return self.layout.scratch_bytes_per_row

    @property
    def max_row_nonzeros(self) -> int:
        """The most nonzeros any encoded row can hold, judged from the terms and never
        from data values (§5.2's structural density is this over p): one for an
        intercept, a numeric column, a categorical's dummies or an interaction's
        cells, a B-spline's degree + 1, a natural or I-spline basis's width, and
        products and sums of these for interactions."""
        return self.layout.max_row_nonzeros

    def encode_into(
        self, rows: pl.DataFrame, out: np.ndarray, first_row: int | None = None
    ) -> list[str]:
        """Encode ``rows`` into ``out``, a C-order float64 block of ``len(rows)`` × p.

        Every term writes its own columns of ``out``; beyond them a term allocates at
        most :attr:`scratch_bytes_per_row` per row. Returns the column names.
        ``first_row`` is the slice's first global row, which target-encoded terms and
        a fold's design need: a training row takes its learned value, and a fold's
        validation rows are encoded as new data.
        """
        shape = (rows.height, self.p)
        if out.shape != shape or out.dtype != np.float64 or not out.flags.c_contiguous:
            order = "C-order" if out.flags.c_contiguous else "not C-order"
            raise DesignMatrixError(
                f"encode_into needs a C-order float64 block of shape {shape}; out is "
                f"{out.dtype} with shape {out.shape}, {order}."
            )
        if first_row is None and (self.target or self.fold is not None):
            raise DesignMatrixError(
                "encode_into needs first_row, the slice's first row, for a design with "
                "target-encoded terms or learned from a fold."
            )
        training = None
        if self.fold is not None:
            assert first_row is not None
            training = self.fold.training(first_row, rows.height)
        codes = _SliceCodes(
            rows,
            self.categories,
            self.layout.shared_codes,
            lenient=self.fold is not None,
            first_row=first_row,
            training=training,
        )
        for writer in self.layout.writers:
            writer.write(rows, out, codes)
        return list(self.names)

    def close(self) -> None:
        """Close the target-encoding columns this design reads (files stay on disk)."""
        for frozen in self.target:
            frozen.close()


def learn_frozen_design(
    parsed: ParsedFormula,
    slices: Slices,
    dtype: np.dtype = np.float64,
    *,
    budget: LearnedStateBudget | int | None = None,
    target: TargetInputs | None = None,
) -> FrozenDesign:
    """Learn every piece of data-dependent design state from slices of the frame.

    One freeze pass learns category keys, frequency counts, spline ranges and the
    target-encoding prior; when any spline places interior knots, four more passes
    read only spline columns; target encoding adds a te_stats pass and the te_sort
    passes (§6.2). The column layout follows from the learned state, before any row
    is encoded.

    Raises
    ------
    FrozenDesignUnsupported
        Before any state is applied, for a model that needs ``InteractionBuilder``.
    LearnedStateTooLargeError
        When learned state crosses ``budget``.
    """
    return learn_frozen_designs([parsed], slices, [None], dtype, budget=budget, target=target)[0]


def learn_frozen_designs(
    parsed_formulas: Sequence[ParsedFormula],
    slices: Slices,
    folds: Sequence[FoldRows | None],
    dtype: np.dtype = np.float64,
    *,
    budget: LearnedStateBudget | int | None = None,
    target: TargetInputs | None = None,
    values: Sequence[bool] | None = None,
    max_design_columns: int | None = None,
    admit: Callable[[str, int], None] | None = None,
) -> list[FrozenDesign]:
    """Learn several designs in the same passes over the slices (§6.2, §7.9).

    Design i learns from the rows ``folds[i]`` selects (every row when None). Each
    design needs its own ``ParsedFormula``, whose splines take its knots. A design
    whose ``values`` entry is False is learned without its target-encoding training
    values; :func:`learn_target_values` sorts them later.

    ``max_design_columns`` is judged before the first slice is read, after every
    slice from the running label counts, and from the final labels before any column
    is laid out (§6.4). ``admit(phase, bytes)`` is told a learning phase's own
    allocations before it starts: ``"knots"`` (the histograms) and ``"te_stats"``.

    Raises
    ------
    FrozenDesignUnsupported
        Before any state is applied, for a model that needs ``InteractionBuilder``.
    LearnedStateTooLargeError
        When learned state crosses ``budget``.
    DesignTooWideError
        When a design's columns cross ``max_design_columns``.
    """
    # The builder keeps ``dtype`` as given (``to_bytes`` stores it), so only check it.
    if np.dtype(dtype) != np.float64:
        raise FrozenDesignUnsupported(f"a frozen design encodes float64, not {np.dtype(dtype)}")
    if len(parsed_formulas) != len(folds):
        raise ValidationError("learn_frozen_designs takes one fold (or None) per formula.")
    open_slices = _slice_passes(slices)
    budget = budget if isinstance(budget, LearnedStateBudget) else LearnedStateBudget(budget)

    learners = [
        _DesignLearner(parsed, fold, budget, target) for parsed, fold in zip(parsed_formulas, folds)
    ]
    category_columns = _unique(
        column for learner in learners for column in learner.category_columns
    )
    any_target = any(learner.targets for learner in learners)

    def check_width(learner: _DesignLearner, labels: Callable[[str], int]) -> None:
        if max_design_columns is not None:
            check_design_width(learner.parsed, labels, max_design_columns)

    # Columns that need no data can cross the limit before anything is read.
    for learner in learners:
        check_width(learner, lambda column: 0)
    n = 0
    first = True
    columns: tuple[str, ...] = ()
    masks = _FoldMasks()
    for rows in open_slices("freeze"):
        if first:
            _check_key_dtypes(rows, category_columns)
            columns = tuple(rows.columns)
            first = False
        response, exposure = (
            _target_arrays(target, rows, n) if any_target and target else (None, None)
        )
        for learner in learners:
            learner.observe(rows, masks(learner.fold, n, rows.height), response, exposure)
            check_width(learner, learner.label_floor)
        n += rows.height
    if n == 0:
        raise FrozenDesignUnsupported("there are no rows to learn from")
    for learner in learners:
        if learner.n == 0:
            assert learner.fold is not None
            raise ValidationError(
                f"Cross-validation fold {learner.fold.fold} has no training rows; use fewer folds."
            )

    categories = [
        {column: category.finish() for column, category in learner.learners.items()}
        for learner in learners
    ]
    # The final labels, before any learned state or column is built from them.
    for learner, cats in zip(learners, categories):
        check_width(learner, lambda column, cats=cats: len(cats[column].labels))
    frequency = [
        tuple(frequency_learner.finish(cats) for frequency_learner in learner.frequency)
        for learner, cats in zip(learners, categories)
    ]
    targets = [
        [
            target_learner.finish(
                cats,
                learner.prior,
                learner.exposure_weighted,
                learner.fold,
                learner.fold.seed if learner.fold is not None else target.seed,
            )
            for target_learner in learner.targets
        ]
        if target is not None
        else []
        for learner, cats in zip(learners, categories)
    ]
    interior = _interior_knots(learners, open_slices, admit)
    if any_target and target is not None:
        if admit is not None:
            admit("te_stats", 0)
        _target_statistics(learners, targets, categories, open_slices, target)
        wanted = [True] * len(learners) if values is None else list(values)
        _target_values(
            [design for design, want in zip(targets, wanted) if want],
            [cats for cats, want in zip(categories, wanted) if want],
            open_slices,
            target,
        )

    designs = []
    for learner, cats, design_frequency, design_targets, design_interior in zip(
        learners, categories, frequency, targets, interior
    ):
        knots = []
        for index, plan in enumerate(learner.plans):
            if plan.boundary is not None:
                boundary = plan.boundary
            else:
                column_range = learner.ranges[plan.column]
                boundary = (column_range.low, column_range.high)
            knots_interior = None if plan.n_interior is None else design_interior.get(index, [])
            knots.append((plan.spline, knots_interior, boundary))
        # Nothing above changes the formula; from here its splines hold the learned knots.
        for spline, knots_interior, boundary in knots:
            spline.freeze_knots(knots_interior, boundary)
        try:
            layout = _Planner(
                learner.parsed, cats, design_frequency, columns, tuple(design_targets)
            ).layout()
        finally:
            # A spline the layout never reaches keeps no knots, as today's build never
            # places them (an interaction whose categoricals give no columns).
            for spline, _interior, _boundary in knots:
                spline.discard_frozen_knots()

        designs.append(
            FrozenDesign(
                parsed=learner.parsed,
                dtype=dtype,
                n_rows=n,
                columns=columns,
                categories=cats,
                discovered=tuple(learner.discovered),
                frequency=design_frequency,
                knots=tuple(knots),
                layout=layout,
                target=tuple(design_targets),
                fold=learner.fold,
                training_rows=learner.n,
            )
        )
    return designs


def key_columns(parsed: ParsedFormula) -> list[str]:
    """Columns a model labels as categories: discovered categoricals, frequency- and
    target-encoded variables, and ``levels=`` columns (§6.3)."""
    return _unique(
        _discovered_columns(parsed)
        + [v for term in parsed.frequency_encoding_terms for v in _frequency_variables(term)]
        + [v for term in parsed.target_encoding_terms for v in _target_variables(term)]
        + _explicit_columns(parsed)
    )


def check_key_dtypes(parsed: ParsedFormula, schema: Mapping[str, pl.DataType]) -> None:
    """Refuse, in either fit mode and before any design is built, a category column
    whose dtype has no training key: List, Array, Struct, Object or Null (DEC-5), or
    Int128 and UInt128 (§6.3). Columns ``schema`` does not hold are judged when their
    first slice arrives.

    Raises
    ------
    ValidationError
        Naming the column and its dtype.
    """
    for column in key_columns(parsed):
        dtype = schema.get(column)
        if dtype is not None:
            check_key_dtype(dtype, column)


def _check_key_dtypes(rows: pl.DataFrame, columns: list[str]) -> None:
    """The same refusal for columns that first appear in a slice (an input transform's
    output)."""
    schema = rows.schema
    for column in columns:
        check_key_dtype(schema[column], column)


# ── Encoder scratch (§6.8, §11) ─────────────────────────────────────────────
#
# Each writer declares the bytes per row of the temporaries it holds while it writes a
# slice: NumPy arrays, and the copies Rust kernels make of their inputs. A writer drops
# them before the next one starts, so the design declares the largest, plus the codes
# of the columns several terms read, which are kept while the slice is written.

_F64 = 8
#: An int32 categorical code.
_CODE = 4
#: A column read as float64: Polars' NumPy conversion (a view for float64 without
#: nulls) and the float64 cast.
_READ = 2 * _F64
#: Forming one categorical's codes (``FrozenCategories.encode``): float keys' bit
#: patterns and NaN mask, the int32 codes, and the null and unseen masks.
_CODES = 24
#: A scatter's active rows and cells (int64 each), and the gathered codes forming them.
_CELLS = 2 * _F64 + _CODE
#: A product of continuous columns: the running product, the next factor read as
#: float64, and the new product.
_PRODUCT = 2 * _F64 + _READ
#: Polars' join of code tuples against the learned table: the key frame, the joined
#: counts and the hash table, beyond the codes themselves.
_JOIN = 4 * _F64


def _kernel_scratch(width: int) -> int:
    """A spline kernel call: its float64 input, the kernel's copy, and its output.

    The output has one column more than the basis keeps for ``bs`` and ``ms``.
    """
    return _F64 * (2 + width + 1)


# ── Writing a slice ─────────────────────────────────────────────────────────


def _active_cells(codes: list[np.ndarray], sizes: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Rows whose every factor is off its reference level, and the cell each falls in.

    Cells number each factor's non-reference levels (``sizes``) row-major, the order
    today's interaction kernels give their columns.
    """
    active = codes[0] > 0
    for factor_codes in codes[1:]:
        active &= factor_codes > 0
    rows = np.flatnonzero(active)
    del active
    cells = codes[0][rows].astype(np.intp)
    cells -= 1
    for factor_codes, size in zip(codes[1:], sizes[1:]):
        cells *= size
        cells += factor_codes[rows]
        cells -= 1
    return rows, cells


def _grouped(block: np.ndarray, groups: int, width: int) -> np.ndarray:
    """``block`` (rows × groups·width) as rows × groups × width, sharing its memory."""
    row_stride, column_stride = block.strides
    return np.lib.stride_tricks.as_strided(
        block,
        shape=(block.shape[0], groups, width),
        strides=(row_stride, width * column_stride, column_stride),
        writeable=True,
    )


def _product(rows: pl.DataFrame, factors: Sequence[str]) -> np.ndarray:
    """The product of continuous columns, multiplied left to right as today's builder does."""
    product = _float_column(rows, factors[0])
    if len(factors) > 1:
        product = product * _float_column(rows, factors[1])
        for factor in factors[2:]:
            np.multiply(product, _float_column(rows, factor), out=product)
    return product


def _as_rust() -> np.errstate:
    """Floating-point warnings off, for NumPy arithmetic that stands in for a Rust kernel.

    The kernels raise none; NumPy warns on ``0 * inf`` and on overflow.
    """
    return np.errstate(all="ignore")


def _basis(spline: SplineTerm, rows: pl.DataFrame, width: int) -> np.ndarray:
    """A slice's spline basis, checked against the width the layout declared."""
    basis = spline.basis(_float_column(rows, spline.var_name))
    if basis.shape[1] != width:
        raise DesignMatrixError(
            f"{spline!r} evaluated to {basis.shape[1]} columns; its layout declares {width}."
        )
    return basis


class _SliceCodes:
    """A slice's discovered-label codes per categorical column.

    Codes of a column more than one term reads (``shared``) are formed once and kept
    until the slice is written; others are formed for the term that reads them.
    """

    def __init__(
        self,
        rows: pl.DataFrame,
        categories: dict[str, FrozenCategories],
        shared: frozenset[str],
        *,
        lenient: bool = False,
        first_row: int | None = None,
        training: np.ndarray | None = None,
    ) -> None:
        self._rows = rows
        self._categories = categories
        self._shared = shared
        self._kept: dict[str, np.ndarray] = {}
        #: A fold's design encodes a label it did not learn as -1 (new data).
        self.lenient = lenient
        #: The slice's first global row, and its training rows in a fold's design.
        self.first_row = first_row
        self.training = training

    def __call__(self, column: str) -> np.ndarray:
        codes = self._kept.get(column)
        if codes is None:
            codes = self._categories[column].encode(self._rows[column], lenient=self.lenient)
            if column in self._shared:
                self._kept[column] = codes
        return codes

    def explicit(self, column: str) -> np.ndarray:
        """Codes of the explicit-level labels (``levels=``), formed for the caller."""
        return self._categories[column].encode(
            self._rows[column], explicit=True, lenient=self.lenient
        )


class _Writer(Protocol):
    """Writes one term's columns of a slice's block."""

    @property
    def scratch(self) -> int:
        """Bytes per row this writer holds beyond its columns of the block."""
        ...

    @property
    def categoricals(self) -> tuple[str, ...]:
        """Columns whose discovered-label codes it reads."""
        ...

    def write(self, rows: pl.DataFrame, out: np.ndarray, codes: _SliceCodes) -> None: ...


@dataclass(frozen=True, eq=False)
class _Intercept:
    start: int
    scratch = 0
    categoricals = ()

    def write(self, rows, out, codes) -> None:
        out[:, self.start] = 1.0


@dataclass(frozen=True, eq=False)
class _Linear:
    start: int
    column: str
    #: Polars' NumPy conversion; the float64 cast writes into the column.
    scratch = _F64
    categoricals = ()

    def write(self, rows, out, codes) -> None:
        # ``copyto`` casts as ``astype`` does: both assign with unsafe casting.
        np.copyto(out[:, self.start], rows[self.column].to_numpy(), casting="unsafe")


@dataclass(frozen=True, eq=False)
class _Dummies:
    """Treatment dummies of a discovered categorical: a one for each non-reference row."""

    start: int
    end: int
    column: str
    scratch = _CODES + _CELLS

    @property
    def categoricals(self) -> tuple[str, ...]:
        return (self.column,)

    def write(self, rows, out, codes) -> None:
        level = codes(self.column)
        if self.end == self.start:
            return
        block = out[:, self.start : self.end]
        block.fill(0.0)
        hit, cells = _active_cells([level], [self.end - self.start])
        del level
        block[hit, cells] = 1.0


@dataclass(frozen=True, eq=False)
class _Spline:
    start: int
    end: int
    spline: SplineTerm
    categoricals = ()

    @property
    def scratch(self) -> int:
        return _kernel_scratch(self.end - self.start)

    def write(self, rows, out, codes) -> None:
        out[:, self.start : self.end] = _basis(self.spline, rows, self.end - self.start)


@dataclass(frozen=True, eq=False)
class _Frequency:
    start: int
    frozen: FrozenFrequency

    @property
    def categoricals(self) -> tuple[str, ...]:
        return self.frozen.variables

    @property
    def scratch(self) -> int:
        variables = len(self.frozen.variables)
        if variables == 1:
            return _CODES + _F64  # the codes, then the gathered counts
        return _CODE * variables + _CODES + _JOIN + _F64

    def write(self, rows, out, codes) -> None:
        variables = [codes(variable) for variable in self.frozen.variables]
        self.frozen.encode(variables, out=out[:, self.start], lenient=codes.lenient)


@dataclass(frozen=True, eq=False)
class _ContinuousProduct:
    """A continuous interaction without splines: one column, the factors' product."""

    start: int
    factors: tuple[str, ...]
    scratch = _F64 + _READ
    categoricals = ()

    def write(self, rows, out, codes) -> None:
        column = out[:, self.start]
        first = _float_column(rows, self.factors[0])
        # An interaction has two factors or more (the parser refuses fewer).
        # ``build_cont_cont_interaction_py`` multiplies elementwise, pair by pair.
        with _as_rust():
            np.multiply(first, _float_column(rows, self.factors[1]), out=column)
        del first
        for factor in self.factors[2:]:
            with _as_rust():
                np.multiply(column, _float_column(rows, factor), out=column)


@dataclass(frozen=True, eq=False)
class _SplineProduct:
    """A continuous interaction with splines: their bases' tensor product × the rest."""

    start: int
    end: int
    #: Each spline and its basis width, in factor order.
    splines: tuple[tuple[SplineTerm, int], ...]
    continuous: tuple[str, ...]
    categoricals = ()

    @property
    def scratch(self) -> int:
        # Every basis is held at once, then the continuous product.
        return sum(_kernel_scratch(width) for _spline, width in self.splines) + _PRODUCT

    def write(self, rows, out, codes) -> None:
        bases = [_basis(spline, rows, width) for spline, width in self.splines]
        product = _product(rows, self.continuous) if self.continuous else None
        block = out[:, self.start : self.end]
        if len(bases) == 1:
            if product is None:
                block[...] = bases[0]
            else:
                np.multiply(bases[0], product[:, None], out=block)
            return
        combos = itertools.product(*(range(basis.shape[1]) for basis in bases))
        for index, combo in enumerate(combos):
            column = block[:, index]
            column[...] = 1.0
            for basis, j in zip(bases, combo):
                np.multiply(column, basis[:, j], out=column)
            if product is not None:
                np.multiply(column, product, out=column)


@dataclass(frozen=True, eq=False)
class _Cells:
    """Categorical cells of an interaction, ones or a continuous product.

    A pure categorical interaction writes a one in each row's cell. With continuous
    factors, one or two categoricals copy the product into the cell, as the Rust
    kernels do; three or more multiply every cell by it, zeros too, as
    ``multiply_matrix_by_continuous_py`` does.
    """

    start: int
    end: int
    categoricals: tuple[str, ...]
    sizes: tuple[int, ...]
    continuous: tuple[str, ...] = ()
    multiply: bool = False

    @property
    def scratch(self) -> int:
        product = _PRODUCT + _F64 if self.continuous else 0  # the product, and its gather
        return _CODE * len(self.categoricals) + _CODES + _CELLS + product

    def write(self, rows, out, codes) -> None:
        levels = [codes(column) for column in self.categoricals]
        if self.end == self.start:
            return
        block = out[:, self.start : self.end]
        block.fill(0.0)
        hit, cells = _active_cells(levels, self.sizes)
        del levels
        if not self.continuous:
            block[hit, cells] = 1.0
        elif self.multiply:
            block[hit, cells] = 1.0
            del hit, cells
            product = _product(rows, self.continuous)
            with _as_rust():
                np.multiply(block, product[:, None], out=block)
        else:
            block[hit, cells] = _product(rows, self.continuous)[hit]


@dataclass(frozen=True, eq=False)
class _BasisBlock:
    """One spline's columns in a categorical × spline interaction."""

    spline: SplineTerm
    #: First column, relative to the term.
    offset: int
    #: Categorical groups: the cells, or every level for a monotone curve per category.
    groups: int
    width: int
    monotone: bool


@dataclass(frozen=True, eq=False)
class _CellBasis:
    """A categorical × spline interaction, category-major, × any continuous factors.

    One categorical copies the basis into the row's group, as
    ``build_cat_basis_interaction_py`` does; more multiply the cells' dummies by the
    basis, as ``multiply_matrix_by_continuous_py`` does. A monotone smooth copies it
    into the group of the row's own level, the reference included.
    """

    start: int
    end: int
    categoricals: tuple[str, ...]
    sizes: tuple[int, ...]
    blocks: tuple[_BasisBlock, ...]
    continuous: tuple[str, ...]

    @property
    def scratch(self) -> int:
        # Codes and cells are held while each basis is written (with a gathered column,
        # or the row and level indices of a monotone block); the product comes last.
        spline = max(_kernel_scratch(block.width) for block in self.blocks) + 2 * _F64
        return _CODE * len(self.categoricals) + _CODES + _CELLS + max(spline, _PRODUCT)

    def write(self, rows, out, codes) -> None:
        levels = [codes(column) for column in self.categoricals]
        term = out[:, self.start : self.end]
        term.fill(0.0)
        hit, cells = _active_cells(levels, self.sizes)
        level = levels[0] if any(block.monotone for block in self.blocks) else None
        del levels
        single = len(self.categoricals) == 1
        for block in self.blocks:
            basis = _basis(block.spline, rows, block.width)
            columns = term[:, block.offset : block.offset + block.groups * block.width]
            cube = _grouped(columns, block.groups, block.width)
            if block.monotone:
                if (level < 0).any():
                    # A fold design's unseen label (new data) has no curve.
                    seen = np.flatnonzero(level >= 0)
                    cube[seen, level[seen]] = basis[seen]
                else:
                    cube[np.arange(rows.height), level] = basis
            elif single:
                for j in range(block.width):
                    cube[hit, cells, j] = basis[hit, j]
            else:
                cube[hit, cells] = 1.0
                with _as_rust():
                    np.multiply(cube, basis[:, None, :], out=cube)
            del basis, columns, cube
        del hit, cells, level
        if self.continuous:
            np.multiply(term, _product(rows, self.continuous)[:, None], out=term)


@dataclass(frozen=True, eq=False)
class _TargetEncoded:
    """A target-encoded term's column: training values, or a fold's full statistics
    for its validation rows."""

    start: int
    frozen: FrozenTarget

    @property
    def categoricals(self) -> tuple[str, ...]:
        return self.frozen.variables if self.frozen.fold is not None else ()

    @property
    def scratch(self) -> int:
        # The column (kept for interactions), and for a fold the levels it reads.
        levels = _CODE * len(self.frozen.variables) + _CODES + _JOIN
        return 2 * _F64 + (levels if self.frozen.fold is not None else 0)

    def write(self, rows, out, codes) -> None:
        out[:, self.start] = self.frozen.column(codes, rows.height)


@dataclass(frozen=True, eq=False)
class _TargetProduct:
    """A continuous interaction with target-encoded factors: one column per
    target-encoded factor, the continuous factors' product times its values, as
    ``_build_continuous_interaction`` gives it."""

    start: int
    continuous: tuple[str, ...]
    targets: tuple[FrozenTarget, ...]
    categoricals = ()

    @property
    def scratch(self) -> int:
        return _PRODUCT + _F64

    def write(self, rows, out, codes) -> None:
        if self.continuous:
            product = _product(rows, self.continuous)
        else:
            product = np.ones(rows.height, dtype=np.float64)
        for offset, frozen in enumerate(self.targets):
            with _as_rust():
                np.multiply(
                    product, frozen.column(codes, rows.height), out=out[:, self.start + offset]
                )


@dataclass(frozen=True, eq=False)
class _Expression:
    """``I(expression)``, from today's column kernel."""

    start: int
    identity: IdentityTermSpec
    #: Polars' result, its NumPy conversion and the kernel's float64 copy.
    scratch = _READ + _F64
    categoricals = ()

    def write(self, rows, out, codes) -> None:
        values, _name = InteractionBuilder(rows)._build_identity_columns(self.identity, rows)
        out[:, self.start] = values


@dataclass(frozen=True, eq=False)
class _Constraint:
    """``pos(...)`` or ``neg(...)``, from today's column kernel."""

    start: int
    constraint: ConstraintTermSpec
    scratch = _READ + _F64
    categoricals = ()

    def write(self, rows, out, codes) -> None:
        builder = InteractionBuilder(rows)
        values, _name = builder._build_constraint_columns(self.constraint, rows)
        out[:, self.start] = values


@dataclass(frozen=True, eq=False)
class _LevelIndicators:
    """``levels=`` indicators: whether each row's explicit-level label is the level."""

    start: int
    column: str
    #: Per listed level, whether each explicit-label code is that level.
    matches: tuple[np.ndarray, ...]
    scratch = _CODES + 1
    categoricals = ()

    def write(self, rows, out, codes) -> None:
        explicit = codes.explicit(self.column)
        unseen = explicit < 0
        for offset, matches in enumerate(self.matches):
            if unseen.any():
                # A fold design's unseen label (new data) matches no level.
                out[:, self.start + offset] = np.where(
                    unseen, False, matches[np.maximum(explicit, 0)]
                )
            else:
                out[:, self.start + offset] = matches[explicit]


# ── Layout ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False)
class _Layout:
    names: tuple[str, ...]
    term_slots: tuple[TermSlot, ...]
    writers: tuple[_Writer, ...]
    #: Builder state today's build leaves behind, by attribute.
    state: dict[str, object]
    #: Categorical columns more than one term reads; their codes are kept per slice.
    shared_codes: frozenset[str]
    scratch_bytes_per_row: int
    #: The most nonzeros any encoded row can hold, from the terms alone (§5.2).
    max_row_nonzeros: int = 0


def _spline_nonzeros(spline: SplineTerm, width: int) -> int:
    """A spline basis's nonzeros per row: a B-spline's degree + 1 (inside and beyond
    its boundary knots); natural and I-spline bases are not local, so their width."""
    if spline.spline_type == "bs":
        return min(width, int(spline.degree) + 1)
    return width


def _writer_nonzeros(writer: _Writer) -> int:
    """The most nonzeros a term writes into one row, from its structure (§5.2)."""
    if isinstance(writer, _Dummies | _Cells):
        return int(writer.end > writer.start)  # one cell, or none for the reference
    if isinstance(writer, _Spline):
        return _spline_nonzeros(writer.spline, writer.end - writer.start)
    if isinstance(writer, _SplineProduct):
        product = 1
        for spline, width in writer.splines:
            product *= _spline_nonzeros(spline, width)
        return min(writer.end - writer.start, product)
    if isinstance(writer, _CellBasis):
        # Each basis block fills one group: the row's cell, or its own level.
        per_row = sum(_spline_nonzeros(block.spline, block.width) for block in writer.blocks)
        return min(writer.end - writer.start, per_row)
    if isinstance(writer, _TargetProduct):
        return len(writer.targets)
    if isinstance(writer, _LevelIndicators):
        # A label matches every listed level equal to it.
        return int(np.sum(np.vstack(writer.matches), axis=0).max()) if writer.matches else 0
    return 1


def _dummy_names(column: str, levels: Sequence[str]) -> list[str]:
    """Treatment dummy names, as ``encode_categorical_indices_py`` gives them."""
    return [f"{column}[T.{level}]" for level in levels[1:]]


def _expression_columns(expression: str, columns: Sequence[str]) -> list[str]:
    """Frame columns an expression reads, as today's build lists a term's factors."""
    return sorted(
        {token for token in re.findall(r"\b([A-Za-z_]\w*)\b", expression) if token in columns}
    )


def _place(spline: SplineTerm) -> tuple[int, list[str]]:
    """Place a spline's learned knots, as its first ``transform`` would; its width and names.

    Learning froze knots for every spline the layout reaches without placed ones.
    """
    if spline._computed_boundary_knots is None:
        spline.adopt_frozen_knots()
    width = spline.basis_width()
    return width, spline.column_names(width)


# ── Design width (§6.4) ─────────────────────────────────────────────────────


def _resolve_interaction_spline(
    parsed: ParsedFormula, interaction: InteractionTerm, factor: str
) -> SplineTerm | None:
    """``InteractionBuilder._resolve_interaction_spline``: local first, then main."""
    if interaction.spline_terms is not None:
        spline = interaction.spline_terms.get(factor)
        if spline is not None:
            return spline
    return parsed.spline_terms_by_var.get(factor)


def _resolve_target_factor(parsed: ParsedFormula, factor: str) -> TargetEncodingTermSpec | None:
    """``InteractionBuilder._parse_te_factor``."""
    result = parsed.te_terms_by_var.get(factor)
    if result is None and factor.startswith("TE(") and factor.endswith(")"):
        result = parsed.te_terms_by_var.get(factor[3:-1])
    return result


def _spline_width(spline: SplineTerm) -> int:
    """A spline's basis width, known before its knots are learned."""
    if spline._computed_boundary_knots is not None:
        return spline.basis_width()
    plan = _knot_plan(spline)
    assert plan is not None  # no boundary knots: the first transform learns them
    if plan.n_interior is None:
        return spline.width_with_interior(len(spline._computed_internal_knots or ()))
    return spline.width_with_interior(plan.n_interior)


def _interaction_width(
    parsed: ParsedFormula, interaction: InteractionTerm, labels: Callable[[str], int]
) -> int:
    """An interaction's columns, as :class:`_Planner` lays them out."""
    force_linear = interaction.force_linear or set()
    if interaction.is_pure_continuous:
        widths = []
        targets = 0
        for factor in interaction.factors:
            if factor in force_linear:
                continue
            spline = _resolve_interaction_spline(parsed, interaction, factor)
            if spline is not None:
                widths.append(_spline_width(spline))
            elif _resolve_target_factor(parsed, factor) is not None:
                targets += 1
        if not widths:
            return targets or 1
        return int(np.prod(widths, dtype=object))
    sizes = [
        max(labels(factor) - 1, 0)
        for factor, is_cat in zip(interaction.factors, interaction.categorical_flags)
        if is_cat
    ]
    cells = int(np.prod(sizes, dtype=object)) if sizes else 1
    if interaction.is_pure_categorical:
        return cells
    splines = [
        spline
        for factor, is_cat in zip(interaction.factors, interaction.categorical_flags)
        if not is_cat and factor not in force_linear
        for spline in [_resolve_interaction_spline(parsed, interaction, factor)]
        if spline is not None
    ]
    if not splines:
        return cells
    if not cells:
        return 0
    first = next(
        factor
        for factor, is_cat in zip(interaction.factors, interaction.categorical_flags)
        if is_cat
    )
    width = 0
    for spline in splines:
        groups = labels(first) if interaction_spline_monotonicity(spline) else cells
        width += groups * _spline_width(spline)
    return width


def design_term_widths(
    parsed: ParsedFormula, labels: Callable[[str], int]
) -> list[tuple[str, int]]:
    """Each term's column count, in layout order, from each discovered column's label
    count. Every count only grows with the labels, so counts taken while level sets
    grow bound the final design from below."""
    widths: list[tuple[str, int]] = []
    if parsed.has_intercept:
        widths.append(("Intercept", 1))
    for var in parsed.main_effects:
        width = max(labels(var) - 1, 0) if var in parsed.categorical_vars else 1
        widths.append((var, width))
    for spline in parsed.spline_terms:
        widths.append((spline.var_name, _spline_width(spline)))
    widths.extend((f"TE({term.var_name})", 1) for term in parsed.target_encoding_terms)
    widths.extend((f"FE({term.var_name})", 1) for term in parsed.frequency_encoding_terms)
    for interaction in parsed.interactions:
        widths.append(
            (":".join(interaction.factors), _interaction_width(parsed, interaction, labels))
        )
    widths.extend((f"I({term.expression})", 1) for term in parsed.identity_terms)
    widths.extend((f"{term.constraint}({term.var_name})", 1) for term in parsed.constraint_terms)
    widths.extend((term.var_name, len(term.levels or [])) for term in parsed.categorical_terms)
    return widths


def check_design_width(parsed: ParsedFormula, labels: Callable[[str], int], limit: int) -> None:
    """Refuse a design whose columns (counted from ``labels``) exceed ``limit``.

    Raises
    ------
    DesignTooWideError
        Naming p, the limit and the term whose columns crossed it.
    """
    widths = design_term_widths(parsed, labels)
    p = sum(width for _term, width in widths)
    if p <= limit:
        return
    running = itertools.accumulate(width for _term, width in widths)
    term, width = next(entry for entry, total in zip(widths, running) if total > limit)
    raise DesignTooWideError(
        f"The design has at least p = {p:,} columns, more than max_design_columns = "
        f"{limit:,}; term {term!r} crossed the limit with {width:,} columns."
    )


class _Planner:
    """Lays out a design's columns in ``_build_design_matrix_core``'s term order.

    It reads the parsed formula, the learned state and the frame's column names, and
    places each spline's learned knots where today's build first transforms it. It
    records the builder state that build leaves behind, and raises its errors.
    """

    def __init__(
        self,
        parsed: ParsedFormula,
        categories: dict[str, FrozenCategories],
        frequency: tuple[FrozenFrequency, ...],
        columns: tuple[str, ...],
        targets: tuple[FrozenTarget, ...] = (),
    ) -> None:
        self.parsed = parsed
        self.categories = categories
        self.frequency = frequency
        self.targets = targets
        self.te_stats: dict[str, dict] = {}
        self.columns = columns
        self.names: list[str] = []
        self.slots: list[TermSlot] = []
        self.writers: list[_Writer] = []
        self.fitted_splines: dict[str, SplineTerm] = {}
        self.smooth_terms: list[SplineTerm] = []
        self.smooth_columns: list[tuple[int, int]] = []
        self.spline_terms: list[SplineTerm] = []
        self.spline_columns: list[tuple[int, int]] = []
        self.fe_stats: dict[str, dict] = {}
        #: The last interaction's smooth blocks; None until an interaction is laid out.
        self.pending: list[tuple[SplineTerm, int, int]] | None = None

    @property
    def n_cols(self) -> int:
        return len(self.names)

    def layout(self) -> _Layout:
        parsed = self.parsed
        if parsed.has_intercept:
            self._add(_Intercept(0), ["Intercept"], "Intercept", "intercept", [])
        for var in parsed.main_effects:
            if var in parsed.categorical_vars:
                self._main_categorical(var)
            else:
                self._add(_Linear(self.n_cols, var), [var], var, "linear", [var])
        for spline in parsed.spline_terms:
            self._main_spline(spline)
        # Today's build adds target encoding before frequency encoding and the
        # interactions (which may read its values).
        for te_term in parsed.target_encoding_terms:
            self._target(te_term)
        for fe_term in parsed.frequency_encoding_terms:
            self._frequency(fe_term)
        for interaction in parsed.interactions:
            self._interaction(interaction)
        for identity in parsed.identity_terms:
            self._expression(identity)
        for constraint in parsed.constraint_terms:
            self._constraint(constraint)
        for cat_term in parsed.categorical_terms:
            self._level_indicators(cat_term)

        names, writers = self.names, self.writers
        if not writers:
            # Today's stack of no terms is one column of ones, without a slot.
            names, writers = ["Intercept"], [_Intercept(0)]
        state: dict[str, object] = {
            "_term_slots": self.slots,
            "_fitted_splines": self.fitted_splines,
            "_smooth_terms": self.smooth_terms,
            "_smooth_col_indices": self.smooth_columns,
            "_all_spline_terms": self.spline_terms,
            "_all_spline_col_indices": self.spline_columns,
            "_te_stats": self.te_stats,
            "_fe_stats": self.fe_stats,
        }
        if self.pending is not None:
            state["_pending_interaction_smooth_blocks"] = self.pending
        reads = Counter(column for writer in writers for column in set(writer.categoricals))
        shared = frozenset(column for column, count in reads.items() if count > 1)
        return _Layout(
            names=tuple(names),
            term_slots=tuple(self.slots),
            writers=tuple(writers),
            state=state,
            shared_codes=shared,
            # Shared codes are kept while every term writes; each term's own
            # temporaries are dropped before the next starts.
            scratch_bytes_per_row=_CODE * len(shared) + max(w.scratch for w in writers),
            max_row_nonzeros=min(len(names), sum(_writer_nonzeros(w) for w in writers)),
        )

    def _add(
        self,
        writer: _Writer,
        names: list[str],
        term_name: str,
        term_type: str,
        factors: Sequence[str],
        extra: dict | None = None,
    ) -> None:
        start = self.n_cols
        self.names.extend(names)
        self.writers.append(writer)
        self.slots.append(
            TermSlot(
                term_name=term_name,
                term_type=term_type,
                factors=list(factors),
                col_start=start,
                col_end=start + len(names),
                design_column_names=list(names),
                extra={} if extra is None else extra,
            )
        )

    # Main effects ----------------------------------------------------------

    def _main_categorical(self, var: str) -> None:
        levels = self.categories[var].labels
        names = _dummy_names(var, levels)
        start = self.n_cols
        writer = _Dummies(start, start + len(names), var)
        self._add(writer, names, var, "categorical", [var], {"levels": list(levels)})

    def _main_spline(self, spline: SplineTerm) -> None:
        start = self.n_cols
        width, names = _place(spline)
        end = start + width
        self.fitted_splines[spline.var_name] = spline
        self.spline_terms.append(spline)
        self.spline_columns.append((start, end))
        is_smooth = bool(getattr(spline, "_is_smooth", False))
        if is_smooth:
            self.smooth_terms.append(spline)
            self.smooth_columns.append((start, end))
        extra = {"is_smooth": is_smooth}
        writer = _Spline(start, end, spline)
        self._add(writer, names, spline.var_name, spline.spline_type, [spline.var_name], extra)

    def _frozen_target(self, te_term: TargetEncodingTermSpec) -> FrozenTarget:
        """The learned state of ``te_term``: learning finishes one per target-encoded term."""
        return next(frozen for frozen in self.targets if frozen.term is te_term)

    def _target(self, te_term: TargetEncodingTermSpec) -> None:
        frozen = self._frozen_target(te_term)
        stats = frozen.stats()
        self.te_stats[te_term.var_name] = stats
        extra = {
            "var_name": te_term.var_name,
            "interaction_vars": te_term.interaction_vars,
            "prior_weight": stats["prior_weight"],
            "prior_weight_spec": stats["prior_weight_spec"],
        }
        factors = list(te_term.interaction_vars or [te_term.var_name])
        writer = _TargetEncoded(self.n_cols, frozen)
        self._add(writer, [frozen.name], frozen.name, "target_encoding", factors, extra)

    def _frequency(self, fe_term: FrequencyEncodingTermSpec) -> None:
        frozen = _frequency_for(self.frequency, fe_term)
        name = f"FE({fe_term.var_name})"
        self.fe_stats[fe_term.var_name] = frozen.stats()
        factors = list(fe_term.interaction_vars or [fe_term.var_name])
        extra = {"var_name": fe_term.var_name, "interaction_vars": fe_term.interaction_vars}
        self._add(
            _Frequency(self.n_cols, frozen), [name], name, "frequency_encoding", factors, extra
        )

    def _expression(self, identity: IdentityTermSpec) -> None:
        name = f"I({identity.expression})"
        factors = _expression_columns(identity.expression, self.columns)
        extra = {"expression": identity.expression}
        self._add(_Expression(self.n_cols, identity), [name], name, "expression", factors, extra)

    def _constraint(self, constraint: ConstraintTermSpec) -> None:
        name = f"{constraint.constraint}({constraint.var_name})"
        inner = constraint.var_name
        if inner.startswith("I(") and inner.endswith(")"):
            factors = _expression_columns(inner[2:-1], self.columns)
        else:
            factors = [inner] if inner in self.columns else []
        extra = {"constraint": constraint.constraint, "var_name": constraint.var_name}
        writer = _Constraint(self.n_cols, constraint)
        self._add(writer, [name], name, "constraint", factors, extra)

    def _level_indicators(self, cat_term: CategoricalTermSpec) -> None:
        # The parser keeps a categorical term here only when it lists levels.
        levels = cat_term.levels or []
        var = cat_term.var_name
        # Today: ``column.to_numpy().astype(str) == level`` over the whole column.
        labels = np.array(self.categories[var].explicit_labels)
        matches = tuple(labels == level for level in levels)
        names = [f"{var}[{level}]" for level in levels]
        extra = {"levels": list(cat_term.levels or [])}
        writer = _LevelIndicators(self.n_cols, var, matches)
        self._add(writer, names, var, "categorical_indicator", [var], extra)

    # Interactions ----------------------------------------------------------

    def _interaction_spline(self, interaction: InteractionTerm, factor: str) -> SplineTerm | None:
        return _resolve_interaction_spline(self.parsed, interaction, factor)

    def _target_encoded(self, factor: str) -> TargetEncodingTermSpec | None:
        return _resolve_target_factor(self.parsed, factor)

    def _interaction_spline_placed(self, spline: SplineTerm) -> tuple[int, list[str]]:
        width, names = _place(spline)
        # An interaction-local spline keeps its knots on itself; registering it under
        # its variable would replace a same-named main-effect spline.
        if not getattr(spline, "_interaction_local", False):
            self.fitted_splines[spline.var_name] = spline
        return width, names

    def _cells(self, categoricals: Sequence[str]) -> tuple[tuple[int, ...], list[str]]:
        """Non-reference level counts and cell names, row-major over the factors."""
        dummies = [_dummy_names(column, self.categories[column].labels) for column in categoricals]
        sizes = tuple(len(names) for names in dummies)
        return sizes, [":".join(cell) for cell in itertools.product(*dummies)]

    def _interaction(self, interaction: InteractionTerm) -> None:
        start = self.n_cols
        if interaction.is_pure_continuous:
            writer, names, pending = self._continuous_interaction(interaction, start)
        elif interaction.is_pure_categorical:
            categoricals = tuple(interaction.factors)
            sizes, names = self._cells(categoricals)
            writer, pending = _Cells(start, start + len(names), categoricals, sizes), []
        else:
            writer, names, pending = self._mixed_interaction(interaction, start)
        self.pending = pending
        # Every pending spline is a smooth with columns: an interaction refuses a
        # fixed-df monotone spline, and one without categorical columns has none.
        for spline, local_start, local_end in pending:
            columns = (start + local_start, start + local_end)
            registered = copy(spline)
            registered._interaction_smooth_endpoint_constraint = True
            self.spline_terms.append(registered)
            self.spline_columns.append(columns)
            self.smooth_terms.append(registered)
            self.smooth_columns.append(columns)
        extra = {
            "categorical_flags": list(interaction.categorical_flags),
            "force_linear": list(interaction.force_linear or []),
        }
        term_name = ":".join(interaction.factors)
        self._add(writer, names, term_name, "interaction", interaction.factors, extra)

    def _continuous_interaction(
        self, interaction: InteractionTerm, start: int
    ) -> tuple[_Writer, list[str], list[tuple[SplineTerm, int, int]]]:
        force_linear = interaction.force_linear or set()
        splines: list[SplineTerm] = []
        continuous: list[str] = []
        targets: list[FrozenTarget] = []
        for factor in interaction.factors:
            if factor in force_linear:
                continuous.append(factor)
                continue
            spline = self._interaction_spline(interaction, factor)
            target = self._target_encoded(factor)
            if spline is not None:
                splines.append(spline)
            elif target is not None:
                # With a spline factor too, today's build leaves this factor out (D14).
                targets.append(self._frozen_target(target))
            else:
                continuous.append(factor)
        if not splines and targets:
            prefix = ":".join(continuous)
            names = [f"{prefix}:{frozen.name}" if prefix else frozen.name for frozen in targets]
            return _TargetProduct(start, tuple(continuous), tuple(targets)), names, []
        if not splines:
            factors = tuple(interaction.factors)
            return _ContinuousProduct(start, factors), [":".join(factors)], []

        placed = [(spline, *self._interaction_spline_placed(spline)) for spline in splines]
        prefix = ":".join(continuous) if continuous else None
        pending: list[tuple[SplineTerm, int, int]] = []
        if len(placed) == 1:
            spline, width, spline_names = placed[0]
            refuse_fixed_df_monotone_interaction(spline)
            names = [f"{prefix}:{name}" if prefix else name for name in spline_names]
            if getattr(spline, "_is_smooth", False):
                pending = [(spline, 0, width)]
        else:
            refuse_monotone_tensor_interaction(splines)
            names = []
            for combo in itertools.product(*(spline_names for _s, _w, spline_names in placed)):
                names.append(":".join([prefix, *combo] if prefix else combo))
        widths = tuple((spline, width) for spline, width, _names in placed)
        writer = _SplineProduct(start, start + len(names), widths, tuple(continuous))
        return writer, names, pending

    def _mixed_interaction(
        self, interaction: InteractionTerm, start: int
    ) -> tuple[_Writer, list[str], list[tuple[SplineTerm, int, int]]]:
        force_linear = interaction.force_linear or set()
        categoricals: list[str] = []
        continuous: list[str] = []
        splines: list[SplineTerm] = []
        for factor, is_cat in zip(interaction.factors, interaction.categorical_flags):
            if is_cat:
                categoricals.append(factor)
            elif factor in force_linear:
                continuous.append(factor)
            else:
                spline = self._interaction_spline(interaction, factor)
                if spline is not None:
                    splines.append(spline)
                else:
                    continuous.append(factor)
        cats = tuple(categoricals)
        sizes, cells = self._cells(cats)
        if not splines:
            suffix = ":".join(continuous)
            names = [f"{cell}:{suffix}" for cell in cells]
            writer = _Cells(
                start, start + len(names), cats, sizes, tuple(continuous), multiply=len(cats) > 2
            )
            return writer, names, []
        if not cells:
            # No categorical columns: today returns before transforming any spline.
            return _Cells(start, start, cats, sizes), [], []

        blocks: list[_BasisBlock] = []
        names: list[str] = []
        pending: list[tuple[SplineTerm, int, int]] = []
        offset = 0
        for spline in splines:
            width, spline_names = self._interaction_spline_placed(spline)
            monotone = bool(interaction_spline_monotonicity(spline))
            refuse_fixed_df_monotone_interaction(spline)
            if monotone:
                # One full curve per category, the reference included.
                check_monotone_by_category(
                    self.parsed, categoricals, spline, single_categorical=len(cats) == 1
                )
                levels = self.categories[cats[0]].labels
                groups = len(levels)
                names.extend(
                    f"{cats[0]}[{level}]:{name}" for level in levels for name in spline_names
                )
            else:
                groups = len(cells)
                names.extend(f"{cell}:{name}" for cell in cells for name in spline_names)
            if monotone or getattr(spline, "_is_smooth", False):
                pending.extend(
                    (spline, offset + group * width, offset + (group + 1) * width)
                    for group in range(groups)
                )
            blocks.append(_BasisBlock(spline, offset, groups, width, monotone))
            offset += groups * width
        if continuous:
            suffix = ":".join(continuous)
            names = [f"{name}:{suffix}" for name in names]
        writer = _CellBasis(start, start + offset, cats, sizes, tuple(blocks), tuple(continuous))
        return writer, names, pending


# ── In-memory build ─────────────────────────────────────────────────────────


def _training_builder(
    data: pl.DataFrame,
    design: FrozenDesign,
    X: np.ndarray,
    names: list[str],
) -> InteractionBuilder:
    """The training frame's builder, holding the state today's build leaves behind.

    The categorical cache keeps levels and names, without the n-row codes and dummy
    matrices that ``clear_caches`` drops after fitting.
    """
    builder = InteractionBuilder(data, dtype=design.dtype)
    for column in design.discovered:
        levels = design.categories[column].labels
        builder._cat_encoding_cache[f"{column}_True"] = CategoricalEncoding(
            encoding=None, names=_dummy_names(column, levels), indices=None, levels=levels
        )
    for attr, value in design.layout.state.items():
        setattr(builder, attr, copy(value))
    builder._parsed_formula = design.parsed
    builder._last_X = X
    builder._last_names = names
    _BUILDER_NONZEROS[builder] = design.max_row_nonzeros
    return builder


#: Each training builder's structural nonzeros per row, which the fits' kernel choice
#: reads (§5.2). Kept beside the builder so its own state stays what today's build
#: leaves behind.
_BUILDER_NONZEROS: weakref.WeakKeyDictionary[InteractionBuilder, int] = weakref.WeakKeyDictionary()


def builder_max_row_nonzeros(builder: object) -> int | None:
    """The structural nonzeros per row of the frozen design a training builder came
    from; None for a builder from today's ``InteractionBuilder`` path."""
    try:
        return _BUILDER_NONZEROS.get(builder)  # type: ignore[arg-type]
    except TypeError:
        return None


def in_memory_target_inputs(response: str, exposure: np.ndarray | None, seed: int) -> TargetInputs:
    """Target encoding's inputs from an in-memory frame: the response column, and the
    raw exposure array's rows of each slice (every record kept in RAM)."""

    def target(rows: pl.DataFrame, first_row: int) -> np.ndarray:
        return rows[response].to_numpy().astype(np.float64, copy=False)

    raw = None if exposure is None else np.asarray(exposure, dtype=np.float64)

    def exposure_of(rows: pl.DataFrame, first_row: int) -> np.ndarray:
        assert raw is not None
        return raw[first_row : first_row + rows.height]

    return TargetInputs(target=target, exposure=None if raw is None else exposure_of, seed=seed)


def build_design_in_memory(
    data: pl.DataFrame,
    parsed: ParsedFormula,
    *,
    dtype: np.dtype = np.float64,
    slice_rows: int = E_LEARN,
    budget: LearnedStateBudget | int | None = None,
    exposure: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[InteractionBuilder, np.ndarray, np.ndarray, list[str]]:
    """Learn a frozen design, then encode consecutive slices into one preallocated X.

    Returns ``(builder, y, X, names)`` equal to an ``InteractionBuilder`` build: X is
    an n × p C-order float64 array, allocated once from the design's layout. Target
    encoding reads ``exposure`` (raw, per row) and keys its order by ``seed`` (drawn
    when None).

    Raises
    ------
    FrozenDesignUnsupported
        Before the formula is changed, for a model that needs ``InteractionBuilder``.
    """
    if slice_rows < 1:
        raise ValidationError(f"slice_rows must be at least 1, got {slice_rows}.")
    n = data.height
    starts = range(0, n, slice_rows)
    slices = [data.slice(start, slice_rows) for start in starts]
    target = None
    if parsed.target_encoding_terms:
        if seed is None:
            seed = draw_target_encoding_seed()
        target = in_memory_target_inputs(parsed.response, exposure, seed)
    design = learn_frozen_design(parsed, slices, dtype, budget=budget, target=target)
    X = np.empty((n, design.p), dtype=np.float64)
    try:
        for start, rows in zip(starts, slices):
            design.encode_into(rows, X[start : start + rows.height], first_row=start)
    finally:
        design.close()
    names = list(design.names)
    builder = _training_builder(data, design, X, names)
    if target is not None:
        builder._te_seed = target.seed
    y = builder._get_column(parsed.response)
    return builder, y, X, names


def build_design_matrix(
    data: pl.DataFrame,
    parsed: ParsedFormula,
    *,
    exposure: np.ndarray | None = None,
    seed: int | None = None,
    dtype: np.dtype = np.float64,
) -> tuple[InteractionBuilder, np.ndarray, np.ndarray, list[str]]:
    """Build ``(builder, y, X, names)``: through a frozen design when the model allows.

    A model :class:`FrozenDesignUnsupported` names keeps
    ``InteractionBuilder.build_design_matrix_from_parsed``. A target-encoded model
    without ``seed`` draws one here (DEC-3), so both routes key the same order; the
    builder records it as ``_te_seed``.
    """
    # DEC-5: refused here as out of core, never handed to today's build.
    check_key_dtypes(parsed, data.schema)
    if parsed.target_encoding_terms and seed is None:
        seed = draw_target_encoding_seed()
    try:
        return build_design_in_memory(data, parsed, dtype=dtype, exposure=exposure, seed=seed)
    except FrozenDesignUnsupported:
        pass
    builder = InteractionBuilder(data, dtype=dtype)
    y, X, names = builder.build_design_matrix_from_parsed(parsed, exposure=exposure, seed=seed)
    return builder, y, X, names
