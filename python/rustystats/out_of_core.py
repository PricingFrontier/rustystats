"""Out-of-core fitting: sources, options and the pass loop (``specs/out-of-core-fit.md`` §4).

An out-of-core fit reads its rows from a :class:`BatchSource` once per pass and never
holds them all. Rust drives the passes (the same driver and per-row arithmetic as an
in-memory fit), and this module serves each pass: it opens the source, cuts every
batch into slices, encodes them and hands Rust one block at a time.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from rustystats.exceptions import BatchSourceChangedError, ValidationError

#: Rows per learning slice (§4.3); slices are also bounded by bytes.
E_LEARN = 65_536
#: Rows per reduction block; ``encode_rows`` is a multiple of it.
GRAM_BLOCK_ROWS = 4_096
#: Default byte bound of one learning slice (§4.3).
DEFAULT_LEARN_SLICE_BYTES = 64 * 1024 * 1024
#: Default byte budget of one encoded block (§4.3).
DEFAULT_ENCODE_BLOCK_BYTES = 256 * 1024 * 1024

PASS_KINDS = frozenset(
    {
        "freeze",
        "knots",
        "te_stats",
        "te_sort",
        "te_merge",
        "te_walk",
        "te_average",
        "moments",
        "start",
        "candidate",
        "trials",
        "gram",
        "reconstruct",
        "final",
        "fold",
        "null_fit",
        "theta",
        "refresh",
        "labels",
    }
)


@dataclass(frozen=True)
class BatchProgress:
    """Where a pass is, reported to ``on_batch`` after each source batch (§4.4)."""

    pass_index: int
    pass_kind: str
    rows_done: int
    n_rows: int | None


@dataclass(frozen=True)
class OutOfCoreOptions:
    """Limits and hooks of an out-of-core fit (§4.3)."""

    max_bytes: int | None = None
    encode_rows: int | None = None
    threads: int | None = None
    max_learned_state_bytes: int | None = None
    max_design_columns: int | None = None
    scratch_dir: str | os.PathLike[str] | None = None
    on_batch: Callable[[BatchProgress], None] | None = None
    te_run_records: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_bytes", "max_learned_state_bytes", "max_design_columns", "threads"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValidationError(f"OutOfCoreOptions.{name} must be positive, got {value}.")
        if self.encode_rows is not None and self.encode_rows <= 0:
            raise ValidationError(
                f"OutOfCoreOptions.encode_rows must be positive, got {self.encode_rows}."
            )
        if self.te_run_records is not None and self.te_run_records <= 0:
            raise ValidationError(
                f"OutOfCoreOptions.te_run_records must be positive, got {self.te_run_records}."
            )


class BatchSource:
    """A re-openable source of row batches (§4.1).

    ``open(columns)`` returns an iterable of Polars DataFrames that together hold every
    row, in the same order every time, with at least ``columns``. ``schema`` is the
    source's column dtypes, known without reading rows (for example from parquet
    metadata). The caller owns the reader and its memory.
    """

    def __init__(
        self,
        open: Callable[[Sequence[str]], Iterable[pl.DataFrame]],
        schema: Mapping[str, pl.DataType],
    ) -> None:
        if not callable(open):
            raise ValidationError("BatchSource needs a callable `open(columns)`.")
        self._open = open
        self.schema = dict(schema)

    def batches(self, columns: Sequence[str]) -> Iterator[pl.DataFrame]:
        """The source's batches projected to ``columns``, dtype-checked against the schema.

        The reader is closed when the returned iterator ends, fails or is closed: its
        iterator's ``close()`` and, when ``open`` returned a different object, the
        iterable's own ``close()`` (§4.1).
        """
        missing = [column for column in columns if column not in self.schema]
        if missing:
            raise ValidationError(
                f"The source has no column(s) {missing}; its schema has {sorted(self.schema)}."
            )
        iterable = self._open(list(columns))
        iterator: Any = None
        try:
            iterator = iter(iterable)
            for batch in iterator:
                if not isinstance(batch, pl.DataFrame):
                    raise ValidationError(
                        f"A BatchSource must yield Polars DataFrames, got {type(batch).__name__}."
                    )
                projected = batch.select(columns)
                for column in columns:
                    if projected.schema[column] != self.schema[column]:
                        raise BatchSourceChangedError(
                            f"Column {column!r} arrived as {projected.schema[column]}, but the "
                            f"source's schema says {self.schema[column]}: the source changed."
                        )
                yield projected
        except BaseException:
            # The exception on its way out (an abort, an error, or GeneratorExit when
            # the pass is closed early) wins over any error from closing.
            _close_reader(iterator, iterable, quiet=True)
            raise
        _close_reader(iterator, iterable, quiet=False)


def _close_reader(iterator: Any, iterable: Any, *, quiet: bool) -> None:
    """Close a pass's iterator and, when it is another object, the iterable ``open``
    returned: a reader whose ``__iter__`` returns a generator owns resources the
    generator's ``close()`` does not release."""
    closers = []
    for owner in (iterator, iterable):
        if owner is None or any(owner is seen for seen, _close in closers):
            continue
        close = getattr(owner, "close", None)
        if callable(close):
            closers.append((owner, close))
    failure: BaseException | None = None
    for _owner, close in closers:
        try:
            close()
        except BaseException as exc:  # every closer runs; the first error wins
            failure = failure or exc
    if failure is not None and not quiet:
        raise failure


class _PassAborted(BaseException):
    """Carries an exception raised while serving a pass to the fit's outer boundary.

    It derives from ``BaseException`` so that recovery handlers inside the fit
    (``except Exception``, ``except ValueError``) cannot swallow it (§7.1).
    """

    def __init__(self, original: BaseException) -> None:
        super().__init__(repr(original))
        self.original = original


def slice_rows(
    batch: pl.DataFrame,
    max_rows: int,
    max_bytes: int | None = None,
    *,
    fixed_bytes: int = 8,
    admit: Callable[[str, int], None] | None = None,
) -> Iterator[pl.DataFrame]:
    """Cut ``batch`` into consecutive zero-copy slices of at most ``max_rows`` rows.

    With ``max_bytes``, a slice also ends before the row that would take its payload past
    ``max_bytes`` (``fixed_bytes`` per fixed-width value, String and Binary by their
    actual lengths); a single larger row forms its own slice (§4.3). Row sizes are
    computed for at most ``max_rows`` rows at a time, so the scratch is
    ``8 · max_rows`` bytes whatever the batch's size.

    ``admit(column, row_bytes)`` is called for a row larger than ``max_bytes`` before
    the row is sliced out (Polars may copy a lone large value when it is), naming the
    column that holds most of its payload; it raises to refuse the row.
    """
    if max_rows < 1:
        raise ValidationError(f"max_rows must be at least 1, got {max_rows}.")
    n = batch.height
    if n == 0:
        return
    if max_bytes is None:
        for start in range(0, n, max_rows):
            yield batch.slice(start, max_rows)
        return
    # ``cumulative`` holds the running payload of rows window..window + len - 1, and
    # ``widest`` the column holding most of each row's.
    cumulative = np.empty(0, dtype=np.int64)
    widest = np.empty(0, dtype=np.int64)
    window = 0
    start = 0  # the open slice's first row
    used = 0  # its bytes
    row = 0
    while row < n:
        if row >= window + len(cumulative):
            window = row
            sizes, widest = _row_payload(batch, row, max_rows, fixed_bytes)
            cumulative = np.cumsum(sizes)
            del sizes
        local = row - window
        base = int(cumulative[local - 1]) if local else 0
        remaining = len(cumulative) - local
        fit = int(np.searchsorted(cumulative, base + (max_bytes - used), side="right")) - local
        if fit <= 0 and row == start:
            fit = 1  # an oversized row forms its own slice
            if admit is not None:
                admit(batch.columns[int(widest[local])], int(cumulative[local]) - base)
        take = max(0, min(fit, max_rows - (row - start), remaining))
        if take:
            used += int(cumulative[local + take - 1]) - base
            row += take
        if take < remaining or row >= n:
            # Closed by its byte or row bound, or by the batch's end; a slice that only
            # reached the end of the sized window continues into the next one.
            yield batch.slice(start, row - start)
            start, used = row, 0


def _row_payload(
    batch: pl.DataFrame, start: int, length: int, fixed_bytes: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Bytes per row of rows ``start .. start + length`` of ``batch``, and the index of
    the column holding most of each row's: ``fixed_bytes`` per fixed-width value (8 is
    an upper bound), String and Binary by their payload, categoricals by their string's.

    The lengths come from expressions that slice the column themselves: a length kernel
    run on a sliced-out Series can copy a large value first."""
    rows = min(length, max(batch.height - start, 0))
    sizes = np.zeros(rows, dtype=np.int64)
    largest = np.full(rows, -1, dtype=np.int64)
    widest = np.zeros(rows, dtype=np.int64)
    for index, (name, dtype) in enumerate(batch.schema.items()):
        column = pl.col(name).slice(start, length)
        if dtype == pl.String:
            expression = column.str.len_bytes()
        elif dtype == pl.Binary:
            expression = column.bin.size()
        elif isinstance(dtype, (pl.Categorical, pl.Enum)):
            expression = column.cat.len_bytes()  # without forming the strings
        else:
            expression = None
        if expression is None:
            lengths = np.full(rows, fixed_bytes, dtype=np.int64)
        else:
            lengths = batch.select(expression).to_series().fill_null(0).to_numpy().astype(np.int64)
        sizes += lengths
        more = lengths > largest
        largest[more] = lengths[more]
        widest[more] = index
    return sizes, widest


#: Rows a fingerprint call converts at a time: its scratch is bounded by this many rows,
#: whatever the slice it is given.
FINGERPRINT_ROWS = 4_096


def fingerprint_update(fingerprint: Any, values: pl.Series) -> None:
    """Add a slice of a column to its fingerprint (§9).

    Every value is fed to the 128-bit hash in a tagged encoding (``fingerprint::kind``):
    floats by their float64 bits; integers, booleans and temporals by their 64-bit
    (physical) patterns; strings, categoricals and enums by their UTF-8 bytes; binaries
    by their bytes; other scalars by their text. Byte values are length-prefixed and
    nulls contribute their tag only. The hash is streamed, so the digest does not depend
    on how a column is cut into slices.

    The slice is converted ``FINGERPRINT_ROWS`` rows at a time. A chunk of fixed-width
    values takes at most 25 bytes per row. A chunk of byte values takes its values'
    bytes once as Python ``bytes`` objects (twice for categoricals, enums and other
    scalars, whose labels are formed first) plus 64 bytes per row. Nothing is joined or
    re-encoded.
    """
    # Kind 1 (float) is the one update_f64 writes.
    integer_kind, boolean_kind, temporal_kind = 2, 3, 4
    string_kind, binary_kind, text_kind = 5, 6, 7
    dtype = values.dtype
    wide = (pl.Int128, pl.UInt128)

    def valid(chunk: pl.Series) -> np.ndarray:
        return np.ascontiguousarray((~chunk.is_null()).to_numpy(), dtype=bool)

    def words(kind: int, chunk: pl.Series, physical: pl.Series) -> None:
        bits = np.ascontiguousarray(physical.fill_null(0).to_numpy()).view(np.uint64)
        fingerprint.update_words(kind, bits, valid(chunk))

    for start in range(0, values.len(), FINGERPRINT_ROWS):
        chunk = values.slice(start, FINGERPRINT_ROWS)
        if dtype.is_float():
            floats = chunk.cast(pl.Float64).fill_null(0.0).to_numpy()
            fingerprint.update_f64(np.ascontiguousarray(floats, dtype=np.float64), valid(chunk))
        elif dtype == pl.Boolean:
            words(boolean_kind, chunk, chunk.cast(pl.UInt64))
        elif dtype.is_integer() and dtype not in wide:
            physical = chunk.cast(pl.UInt64 if dtype.is_unsigned_integer() else pl.Int64)
            words(integer_kind, chunk, physical)
        elif dtype.is_temporal():
            words(temporal_kind, chunk, chunk.to_physical().cast(pl.Int64))
        elif dtype == pl.Binary:
            fingerprint.update_bytes(binary_kind, chunk.to_list())
        elif dtype == pl.String:
            fingerprint.update_bytes(string_kind, chunk.cast(pl.Binary).to_list())
        elif dtype in (pl.Categorical, pl.Enum):
            labels = chunk.cast(pl.String).cast(pl.Binary)
            fingerprint.update_bytes(string_kind, labels.to_list())
        else:
            fingerprint.update_bytes(text_kind, chunk.cast(pl.String).cast(pl.Binary).to_list())
