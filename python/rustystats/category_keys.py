"""Learn a categorical column's labels from slices, exactly as training labels them.

Training labels a categorical value with ``str()`` of its element in
``column.to_numpy()`` over the whole column, and matches explicit ``levels=``
against ``column.to_numpy().astype(str)``. Both depend on the whole column's
NumPy dtype: an Int64 column with any null converts to float64, so its labels
read ``"1.0"`` and ``"nan"``.

This module reproduces those labels and codes from slices of the column, as an
out-of-core pass meets them, without a Python object per row
(``specs/out-of-core-fit.md`` §6.3, §6.4):

1. Every value has a lossless *training key* (:func:`training_keys`). A slice's
   distinct keys come from Polars ``unique`` over the key form.
2. :class:`CategoryKeyLearner` merges each slice's new keys into its table and
   records whether a null was seen. It charges their bytes to a
   :class:`LearnedStateBudget` before inserting them.
3. :meth:`CategoryKeyLearner.finish` freezes the table. Labels are formed once
   per key, converted with the whole column's NumPy dtype, and sorted the way
   training factorises them. Keys that render to one label share its code.
4. :meth:`FrozenCategories.encode` maps a slice to codes with Polars
   ``replace_strict`` over the key form.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import polars as pl

from rustystats.exceptions import (
    EncodingError,
    LearnedStateTooLargeError,
    UnseenLabelError,
    ValidationError,
)

__all__ = [
    "ENTRY_OVERHEAD_BYTES",
    "CategoryKeyLearner",
    "FrozenCategories",
    "LearnedStateBudget",
    "check_key_dtype",
    "observed_category_labels",
    "training_keys",
]

#: Bytes charged per key on top of its payload: the key's slot (a string view
#: or a fixed-width value), its validity bit, its code and its share of a hash table.
ENTRY_OVERHEAD_BYTES = 32

# Label reservation (§6.3). A Python ``str`` stores at most 4 bytes per character
# behind an object header of at most 80 bytes.
_LABEL_OBJECT_BYTES = 80
# Longest label any fixed-width dtype renders to (a float64 repr is at most 24
# characters, a datetime64 or timedelta64 at most 40, a 38-digit decimal 42).
_FIXED_LABEL_CHARS = 48
# Keys are converted to NumPy in chunks of at most this many keys and label characters.
_RENDER_CHUNK_KEYS = 256
_RENDER_CHUNK_CHARS = 1 << 16
# Transient references per key while labels are sorted and indexed.
_FACTORISE_BYTES_PER_KEY = 192

_STRING = "string"
_BOOLEAN = "boolean"
_INTEGER = "integer"
_FLOAT = "float"
_TEMPORAL = "temporal"
_DECIMAL = "decimal"
_BINARY = "binary"

_DISCOVERED = "discovered"
_EXPLICIT = "explicit"

# Bit width of a float, and the one NaN every NaN payload maps to.
_FLOAT_BITS: dict[np.dtype, tuple[type[np.unsignedinteger], int]] = {
    np.dtype(np.float16): (np.uint16, 0x7E00),
    np.dtype(np.float32): (np.uint32, 0x7FC0_0000),
    np.dtype(np.float64): (np.uint64, 0x7FF8_0000_0000_0000),
}
_FLOAT_OF_BITS = {
    np.dtype(np.uint16): np.float16,
    np.dtype(np.uint32): np.float32,
    np.dtype(np.uint64): np.float64,
}

_FIXED_WIDTHS: tuple[tuple[type[pl.DataType], int], ...] = (
    (pl.Boolean, 1),
    (pl.Int8, 1),
    (pl.UInt8, 1),
    (pl.Int16, 2),
    (pl.UInt16, 2),
    (pl.Int32, 4),
    (pl.UInt32, 4),
    (pl.Int64, 8),
    (pl.UInt64, 8),
    (pl.Decimal, 16),
)

_WIDE_INTEGERS = (pl.Int128, pl.UInt128)


# ── Keys ────────────────────────────────────────────────────────────────────


def _key_kind(dtype: pl.DataType, column: str) -> str:
    """Return the key form for ``dtype``, refusing dtypes that cannot be categories."""
    if dtype == pl.String or isinstance(dtype, (pl.Categorical, pl.Enum)):
        return _STRING
    if dtype == pl.Boolean:
        return _BOOLEAN
    if any(dtype == wide for wide in _WIDE_INTEGERS):
        raise ValidationError(
            f"Column {column!r} has dtype {dtype}, which has no NumPy conversion to label "
            f"categories with. Cast it to Int64 or String first."
        )
    if dtype.is_integer():
        return _INTEGER
    if dtype.is_float():
        return _FLOAT
    if dtype.is_temporal():
        return _TEMPORAL
    if dtype.is_decimal():
        return _DECIMAL
    if dtype == pl.Binary:
        return _BINARY
    raise ValidationError(
        f"Column {column!r} has dtype {dtype}, which cannot be a categorical, "
        f"frequency-encoded or target-encoded input. Use a scalar column (string, "
        f"categorical, boolean, integer, float, temporal, decimal or binary)."
    )


def _float_bits(values: pl.Series) -> pl.Series:
    """IEEE bit patterns of a float series, with every NaN mapped to one NaN."""
    nulls = values.null_count()
    floats = np.ascontiguousarray((values.fill_null(0.0) if nulls else values).to_numpy())
    uint_type, canonical_nan = _FLOAT_BITS[floats.dtype]
    bits = np.where(np.isnan(floats), uint_type(canonical_nan), floats.view(uint_type))
    keys = pl.Series(values.name, bits)
    if nulls:
        keys = pl.select(pl.when(values.is_not_null()).then(keys)).to_series().alias(values.name)
    return keys


def _to_keys(values: pl.Series, kind: str) -> pl.Series:
    if kind == _STRING:
        return values if values.dtype == pl.String else values.cast(pl.String)
    if kind == _FLOAT:
        return _float_bits(values)
    if kind == _TEMPORAL:
        # The physical integer; the unit and time zone live in the column's dtype.
        return values.to_physical()
    # Booleans, integers, decimals (the unscaled integer under the dtype's scale)
    # and binaries are their own keys.
    return values


def _from_keys(keys: pl.Series, kind: str, dtype: pl.DataType) -> pl.Series:
    """Turn non-null keys back into values of the column's dtype."""
    if kind == _FLOAT:
        bits = keys.to_numpy()
        return pl.Series(keys.name, bits.view(_FLOAT_OF_BITS[bits.dtype])).cast(dtype)
    if kind in (_STRING, _TEMPORAL):
        return keys.cast(dtype)
    return keys


def check_key_dtype(dtype: pl.DataType, column: str) -> None:
    """Refuse a dtype that cannot be a categorical, frequency-encoded or target-encoded
    input, from the dtype alone: List, Array, Struct, Object and Null (DEC-5), and
    Int128 and UInt128, which have no NumPy conversion to label with (§6.3).

    Raises
    ------
    ValidationError
        Naming the column and its dtype.
    """
    _key_kind(dtype, column)


def training_keys(values: pl.Series) -> pl.Series:
    """Return the lossless training key of every value, nulls kept as nulls.

    Two values share a key exactly when training could not tell them apart:
    strings, categoricals and enums by their text; floats by their IEEE bits
    (``-0.0`` and ``0.0`` differ, every NaN is one key); temporals by their
    physical integer under the column's unit and time zone; decimals by their
    unscaled integer under the column's scale; binaries by their bytes.

    Raises
    ------
    ValidationError
        For List, Array, Struct, Object, Null, Int128 and UInt128 columns.
    """
    return _to_keys(values, _key_kind(values.dtype, values.name))


def _wide_integer_keys(keys: pl.Series) -> int:
    """Integer keys a float64 cannot hold apart: those beyond ±2**53."""
    limit = 2**53
    if keys.dtype == pl.UInt64:
        return int((keys > limit).sum())
    if keys.dtype == pl.Int64:
        return int(((keys > limit) | (keys < -limit)).sum())
    return 0


def _fixed_width(dtype: pl.DataType) -> int:
    for fixed, width in _FIXED_WIDTHS:
        if dtype == fixed:
            return width
    raise AssertionError(f"no fixed width for key dtype {dtype}")  # pragma: no cover


def _payload(keys: pl.Series, kind: str) -> tuple[int, int]:
    """Total and largest payload bytes of non-null keys, without Python objects per key."""
    if keys.is_empty():
        return 0, 0
    if kind == _STRING:
        sizes = keys.str.len_bytes()
    elif kind == _BINARY:
        sizes = keys.bin.size()
    else:
        width = _fixed_width(keys.dtype)
        return width * len(keys), width
    return int(sizes.sum()), int(sizes.max())


# ── Budget ──────────────────────────────────────────────────────────────────


class LearnedStateBudget:
    """Running byte count of learned state, checked against an optional limit.

    One budget can be shared by every column's learner, so the limit covers
    the run's learned state as a whole. ``limit=None`` means no limit.
    """

    def __init__(
        self, limit: int | None = None, *, source: str = "max_learned_state_bytes"
    ) -> None:
        if limit is not None and limit < 0:
            raise ValidationError(f"The learned-state limit must be non-negative, got {limit}.")
        self.limit = limit
        #: The option the limit comes from, named when it is crossed.
        self.source = source
        self.used = 0

    def cap(self, limit: int, *, source: str) -> None:
        """Lower the limit to ``limit`` when that is tighter (§6.4: the smaller of
        ``max_learned_state_bytes`` and what ``max_bytes`` leaves)."""
        if self.limit is None or limit < self.limit:
            self.limit = max(0, limit)
            self.source = source

    def charge(self, nbytes: int, *, term: str, cardinality: int, transient: int = 0) -> None:
        """Add ``nbytes`` of lasting state, after checking it fits with ``transient`` more.

        Raises
        ------
        LearnedStateTooLargeError
            Before anything is charged, when the state would cross the limit.
        """
        needed = self.used + nbytes + transient
        if self.limit is not None and needed > self.limit:
            raise LearnedStateTooLargeError(
                f"Learning {term!r} ({cardinality:,} distinct values) would bring learned "
                f"state to {needed:,} bytes, over the limit of {self.limit:,} bytes set by "
                f"{self.source}. Raise it, or reduce the cardinality of {term!r}."
            )
        self.used += nbytes

    def admit_row(self, column: str, row_bytes: int, transient: int) -> None:
        """Check that a row larger than a learning slice can be processed: its
        ``transient`` working bytes beside the learned state must fit the limit before
        the row is fingerprinted, transformed or learned (§4.3).

        Raises
        ------
        LearnedStateTooLargeError
            Naming the column and the row's size.
        """
        needed = self.used + transient
        if self.limit is not None and needed > self.limit:
            raise LearnedStateTooLargeError(
                f"A row of column {column!r} holds {row_bytes:,} bytes, more than a learning "
                f"slice; processing it needs {transient:,} bytes beside {self.used:,} bytes "
                f"of learned state, over the limit of {self.limit:,} bytes set by "
                f"{self.source}."
            )


def _as_budget(budget: LearnedStateBudget | int | None) -> LearnedStateBudget:
    return budget if isinstance(budget, LearnedStateBudget) else LearnedStateBudget(budget)


def _check_dtype(column: str, expected: pl.DataType, values: pl.Series) -> None:
    if values.dtype != expected:
        raise ValidationError(
            f"Column {column!r} changed dtype from {expected} to {values.dtype} between "
            f"slices; every slice of a column must have the same dtype."
        )


# ── Learner ─────────────────────────────────────────────────────────────────


class CategoryKeyLearner:
    """Learn one column's distinct training keys from its slices.

    Parameters
    ----------
    column : str
        The column's name, used in errors.
    budget : LearnedStateBudget or int, optional
        Byte budget for learned state; an int creates a private budget. Each
        new key costs its payload plus :data:`ENTRY_OVERHEAD_BYTES`, charged
        before the key is inserted.
    """

    def __init__(self, column: str, *, budget: LearnedStateBudget | int | None = None) -> None:
        self.column = column
        self._budget = _as_budget(budget)
        self._dtype: pl.DataType | None = None
        self._kind = ""
        self._keys: pl.Series | None = None
        self._has_null = False
        self._payload_bytes = 0
        self._payload_max = 0
        self._nbytes = 0
        #: Integer keys beyond ±2**53, which may share a float64 label (§6.3).
        self._wide_keys = 0
        self._finished = False

    @property
    def dtype(self) -> pl.DataType | None:
        """The column's dtype, fixed by the first slice."""
        return self._dtype

    @property
    def has_null(self) -> bool:
        return self._has_null

    @property
    def n_keys(self) -> int:
        """Distinct keys learned so far, the null key included."""
        return (0 if self._keys is None else len(self._keys)) + self._has_null

    @property
    def nbytes(self) -> int:
        """Bytes charged for this column's key table."""
        return self._nbytes

    @property
    def label_floor(self) -> int:
        """The fewest distinct discovered labels the keys so far can render to.

        Keys render to distinct labels except where the whole-column conversion
        merges them: integers beyond ±2**53 when a null makes the column float64, and
        a float column's null with its NaN (both ``"nan"``). Labels only grow as keys
        do, so this bounds the final count from below (§6.4, ``max_design_columns``).
        """
        if self._keys is None:
            return int(self._has_null)
        keys = len(self._keys)
        if self._kind == _INTEGER:
            return keys - self._wide_keys + min(self._wide_keys, 1) + self._has_null
        if self._kind == _FLOAT:
            return keys
        return keys + self._has_null

    def observe(self, values: pl.Series) -> None:
        """Merge one slice's distinct keys into the table.

        Raises
        ------
        LearnedStateTooLargeError
            When the slice's new keys would cross the budget; nothing is inserted.
        ValidationError
            For a refused dtype, or a dtype different from earlier slices.
        """
        if self._finished:
            raise ValidationError(f"Categories of {self.column!r} are already frozen.")
        if self._dtype is None:
            self._kind = _key_kind(values.dtype, self.column)
            self._dtype = values.dtype
        else:
            _check_dtype(self.column, self._dtype, values)

        new_null = not self._has_null and values.null_count() > 0
        slice_keys = _to_keys(values.drop_nulls(), self._kind).unique()
        if self._keys is not None and not slice_keys.is_empty():
            slice_keys = pl.DataFrame({"key": slice_keys}).join(
                pl.DataFrame({"key": self._keys}), on="key", how="anti"
            )["key"]
        n_new = len(slice_keys) + new_null
        if n_new == 0:
            return

        payload, payload_max = _payload(slice_keys, self._kind)
        cost = payload + n_new * ENTRY_OVERHEAD_BYTES
        self._budget.charge(cost, term=self.column, cardinality=self.n_keys + n_new)

        if self._kind == _INTEGER and not slice_keys.is_empty():
            self._wide_keys += _wide_integer_keys(slice_keys)
        if self._keys is None:
            self._keys = slice_keys.alias(self.column)
        elif not slice_keys.is_empty():
            self._keys.append(slice_keys.alias(self.column))
        self._has_null = self._has_null or new_null
        self._payload_bytes += payload
        self._payload_max = max(self._payload_max, payload_max)
        self._nbytes += cost

    def _freeze(self) -> FrozenCategories:
        if self._dtype is None:
            raise ValidationError(f"No slices of {self.column!r} were observed.")
        self._finished = True
        keys = (
            self._keys
            if self._keys is not None
            else _to_keys(pl.Series(dtype=self._dtype), self._kind)
        )
        return FrozenCategories(
            column=self.column,
            dtype=self._dtype,
            kind=self._kind,
            keys=keys.clone(),
            has_null=self._has_null,
            payload_bytes=self._payload_bytes,
            payload_max=self._payload_max,
            budget=self._budget,
        )

    def finish(self) -> FrozenCategories:
        """Freeze the table and form its discovered labels.

        Raises
        ------
        LearnedStateTooLargeError
            When the labels' reservation does not fit the budget; no label is formed.
        """
        frozen = self._freeze()
        frozen._rendering(_DISCOVERED)
        return frozen


# ── Frozen table ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Rendering:
    labels: list[str]
    key_codes: np.ndarray  # int32, aligned with the key table
    null_code: int | None


def _render_discovered(column: str, array: np.ndarray, i: int) -> str:
    return str(array[i])


def _render_explicit(column: str, array: np.ndarray, i: int) -> str:
    try:
        return str(array[i : i + 1].astype(str)[0])
    except UnicodeDecodeError as exc:
        raise EncodingError(
            f"Column {column!r} has a binary value that is not ASCII, so it has no "
            f"level label to match: {array[i]!r}."
        ) from exc


class FrozenCategories:
    """One column's frozen key table, its labels and their codes.

    Built by :meth:`CategoryKeyLearner.finish`. ``labels`` and ``encode`` use the
    discovered renderer (``str()`` of each NumPy element), which training uses
    to discover levels. ``explicit_labels``, ``encode(..., explicit=True)`` and
    ``observed_labels`` use the explicit-level renderer (NumPy ``astype(str)``),
    which training uses to match ``levels=``.
    """

    def __init__(
        self,
        *,
        column: str,
        dtype: pl.DataType,
        kind: str,
        keys: pl.Series,
        has_null: bool,
        payload_bytes: int,
        payload_max: int,
        budget: LearnedStateBudget,
    ) -> None:
        self.column = column
        self.dtype = dtype
        self.has_null = has_null
        self._kind = kind
        self._keys = keys
        self._payload_bytes = payload_bytes
        self._payload_max = payload_max
        self._budget = budget
        self._renderings: dict[str, _Rendering] = {}

    @property
    def n_keys(self) -> int:
        """Distinct keys in the table, the null key included."""
        return len(self._keys) + self.has_null

    @property
    def labels(self) -> list[str]:
        """Discovered labels in code order (today's sorted factorisation)."""
        return list(self._rendering(_DISCOVERED).labels)

    @property
    def explicit_labels(self) -> list[str]:
        """Explicit-level labels in code order, the null's label included."""
        return list(self._rendering(_EXPLICIT).labels)

    def observed_labels(self) -> list[str]:
        """Sorted distinct explicit-level labels of the non-null values."""
        rendering = self._rendering(_EXPLICIT)
        return [rendering.labels[code] for code in np.unique(rendering.key_codes)]

    def encode(
        self, values: pl.Series, *, explicit: bool = False, lenient: bool = False
    ) -> np.ndarray:
        """Map each value of a slice to its label's code.

        With ``lenient``, a value whose key is not in the table gets code -1 instead
        of an error: a cross-validation fold's design encodes its validation rows as
        new data (``specs/out-of-core-fit.md`` §7.9).

        Raises
        ------
        UnseenLabelError
            When a value's key is not in the table (unless ``lenient``).
        """
        _check_dtype(self.column, self.dtype, values)
        rendering = self._rendering(_EXPLICIT if explicit else _DISCOVERED)
        if self._keys.is_empty():
            codes = np.full(len(values), -1, dtype=np.int32)
        else:
            replaced = _to_keys(values, self._kind).replace_strict(
                self._keys,
                pl.Series(rendering.key_codes),
                default=-1,
                return_dtype=pl.Int32,
            )
            if lenient:
                replaced = replaced.fill_null(-1)
            codes = replaced.to_numpy(writable=True)
        if rendering.null_code is not None and values.null_count():
            codes[values.is_null().to_numpy()] = rendering.null_code
        if lenient:
            return codes
        unseen = np.flatnonzero(codes < 0)
        if unseen.size:
            label = self._label_of(values.slice(int(unseen[0]), 1), explicit=explicit)
            raise UnseenLabelError(
                f"Column {self.column!r} has the value {label!r}, which was not among the "
                f"categories learned from it. The data changed after its categories were "
                f"learned; fit again on unchanged data."
            )
        return codes

    # Label formation ---------------------------------------------------------

    def _label_chars(self) -> tuple[int, int]:
        """Bounds on label characters: the sum over keys, and the largest key's."""
        n = len(self._keys)
        if self._kind == _STRING:
            total, largest = self._payload_bytes, self._payload_max
        elif self._kind == _BINARY:
            # ``str()`` escapes each byte to at most 4 characters inside ``b''``.
            total, largest = 4 * self._payload_bytes + 3 * n, 4 * self._payload_max + 3
        else:
            total, largest = _FIXED_LABEL_CHARS * n, _FIXED_LABEL_CHARS if n else 0
        if self.has_null:
            total += _FIXED_LABEL_CHARS
            largest = max(largest, _FIXED_LABEL_CHARS)
        return total, largest

    def _chunk_size(self, largest: int) -> int:
        return max(1, min(_RENDER_CHUNK_KEYS, _RENDER_CHUNK_CHARS // max(largest, 1)))

    def _reserve_labels(self) -> None:
        """Charge the labels' bytes, and check the transient peak, before forming any."""
        total, largest = self._label_chars()
        n = self.n_keys
        persistent = 4 * total + (_LABEL_OBJECT_BYTES + 4) * n
        chunk = self._chunk_size(largest)
        transient = (
            (4 * largest + _LABEL_OBJECT_BYTES) * chunk  # one chunk's converted elements
            + 8 * largest  # NumPy's fixed-width array and the label being formed
            + _LABEL_OBJECT_BYTES
            + _FACTORISE_BYTES_PER_KEY * n
        )
        self._budget.charge(persistent, term=self.column, cardinality=n, transient=transient)

    def _to_numpy(self, keys: pl.Series) -> np.ndarray:
        """Convert keys as the whole column converts: a null makes the column's dtype nullable."""
        values = _from_keys(keys, self._kind, self.dtype)
        if self.has_null:
            values = values.extend_constant(None, 1)
        return values.to_numpy()

    def _converted_chunks(self) -> Iterator[tuple[np.ndarray, int]]:
        """Yield each chunk of keys converted to NumPy (null last, if any), with its key count."""
        n = len(self._keys)
        if n == 0:
            if self.has_null:
                yield self._to_numpy(self._keys), 0
            return
        size = self._chunk_size(self._label_chars()[1])
        for start in range(0, n, size):
            chunk = self._keys.slice(start, size)
            yield self._to_numpy(chunk), len(chunk)

    def _rendering(self, renderer: str) -> _Rendering:
        cached = self._renderings.get(renderer)
        if cached is not None:
            return cached
        self._reserve_labels()
        render = _render_discovered if renderer == _DISCOVERED else _render_explicit
        key_labels: list[str] = []
        null_label: str | None = None
        for array, n_chunk in self._converted_chunks():
            key_labels.extend(render(self.column, array, i) for i in range(n_chunk))
            if self.has_null and null_label is None:
                null_label = render(self.column, array, n_chunk)
        if null_label is not None:
            key_labels.append(null_label)
        # Training sorts labels as Rust ``String``s: by UTF-8 bytes, which is code-point order.
        labels = sorted(set(key_labels))
        index = {label: code for code, label in enumerate(labels)}
        codes = np.fromiter(
            (index[label] for label in key_labels), dtype=np.int32, count=len(key_labels)
        )
        rendering = _Rendering(
            labels=labels,
            key_codes=codes[: len(self._keys)],
            null_code=int(codes[-1]) if self.has_null else None,
        )
        self._renderings[renderer] = rendering
        return rendering

    def _label_of(self, value: pl.Series, *, explicit: bool) -> str:
        """Render one value, for an error message."""
        if self.has_null:
            value = value.extend_constant(None, 1)
        render = _render_explicit if explicit else _render_discovered
        return render(self.column, value.to_numpy(), 0)


# ── Observed labels ─────────────────────────────────────────────────────────


def observed_category_labels(
    values: pl.Series | Iterable[pl.Series],
    *,
    column: str | None = None,
    budget: LearnedStateBudget | int | None = None,
) -> list[str]:
    """Sorted distinct labels of a column's non-null values, as ``levels=`` matches them.

    ``values`` is the whole column or an iterable of its slices. Labels use the
    whole-column conversion and NumPy ``astype(str)``, so an Int64 column with
    nulls gives ``"2.0"``; null rows are left out.
    """
    slices = [values] if isinstance(values, pl.Series) else values
    learner: CategoryKeyLearner | None = None
    for part in slices:
        if learner is None:
            learner = CategoryKeyLearner(column or part.name, budget=budget)
        learner.observe(part)
    if learner is None:
        return []
    return learner._freeze().observed_labels()
