"""
Optimized interaction term support for RustyStats.

This module provides high-performance interaction term handling for GLMs.
All heavy computation is done in Rust for maximum speed:
- Categorical encoding (Rust parallel construction)
- Interaction terms (Rust parallel for large data)
- Spline basis functions (Rust with Rayon)

The Python layer handles only:
- DataFrame column extraction
- Orchestration of Rust calls

Example
-------
>>> from rustystats.interactions import InteractionBuilder
>>>
>>> builder = InteractionBuilder(data)
>>> y, X, names = builder.build_design_matrix_from_parsed(parsed)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from rustystats._rustystats import (
    apply_exposure_weighted_target_encoding_py as _apply_exposure_weighted_te_rust,
)
from rustystats._rustystats import (
    apply_frequency_encoding_py as _apply_frequency_encoding_rust,
)
from rustystats._rustystats import (
    apply_target_encoding_py as _apply_target_encoding_rust,
)
from rustystats._rustystats import (
    build_cat_cat_interaction_py as _build_cat_cat_rust,
)
from rustystats._rustystats import (
    build_cat_cont_interaction_py as _build_cat_cont_rust,
)
from rustystats._rustystats import (
    build_cont_cont_interaction_py as _build_cont_cont_rust,
)
from rustystats._rustystats import (
    build_two_cat_cont_interaction_py as _build_two_cat_cont_rust,
)
from rustystats._rustystats import (
    encode_categorical_indices_py as _encode_categorical_indices_rust,
)

# Import Rust implementations for heavy computation
from rustystats._rustystats import (
    encode_categorical_py as _encode_categorical_rust,
)
from rustystats._rustystats import (
    factorize_strings_py as _factorize_strings_rust,
)
from rustystats._rustystats import (
    multiply_matrix_by_continuous_py as _multiply_matrix_cont_rust,
)
from rustystats._rustystats import (
    stack_columns_horizontal_py as _stack_columns_rust,
)
from rustystats._rustystats import (
    target_encode_py as _target_encode_rust,
)
from rustystats.constants import (
    CONDITION_NUMBER_THRESHOLD,
    DEFAULT_CORRELATION_THRESHOLD,
    DEFAULT_N_PERMUTATIONS,
    DEFAULT_PRIOR_WEIGHT,
    ZERO_VARIANCE_THRESHOLD,
)
from rustystats.exceptions import (
    DesignMatrixError,
    EncodingError,
    PredictionError,
    ValidationError,
)

if TYPE_CHECKING:
    import polars as pl


@dataclass
class InteractionTerm:
    """Represents a single interaction term like x1:x2 or C(cat1):x2."""

    factors: list[str]  # Variables involved (e.g., ['x1', 'x2'] or ['cat1', 'x2'])
    categorical_flags: list[bool]  # Which factors are categorical
    force_linear: set[str] | None = None  # Factors that must stay linear (no spline expansion)

    @property
    def order(self) -> int:
        """Order of interaction (2 for pairwise, 3 for three-way, etc.)."""
        return len(self.factors)

    @property
    def is_pure_continuous(self) -> bool:
        """True if all factors are continuous."""
        return not any(self.categorical_flags)

    @property
    def is_pure_categorical(self) -> bool:
        """True if all factors are categorical."""
        return all(self.categorical_flags)

    @property
    def is_mixed(self) -> bool:
        """True if mixture of categorical and continuous."""
        return any(self.categorical_flags) and not all(self.categorical_flags)


# Import SplineTerm from splines module (canonical implementation)
from rustystats.splines import SplineTerm


@dataclass
class CategoricalEncoding:
    """Cached categorical encoding data for a variable."""

    encoding: np.ndarray  # (n, k-1) dummy matrix
    names: list[str]  # Column names like ['var[T.B]', 'var[T.C]']
    indices: np.ndarray  # (n,) level indices (int32)
    levels: list[str]  # All categorical levels


@dataclass
class TargetEncodingTermSpec:
    """Parsed target encoding term specification from formula."""

    var_name: str
    prior_weight: float = DEFAULT_PRIOR_WEIGHT
    n_permutations: int = DEFAULT_N_PERMUTATIONS
    interaction_vars: list[str] | None = None  # For TE(a:b) interactions


@dataclass
class FrequencyEncodingTermSpec:
    """Parsed frequency encoding term specification from formula."""

    var_name: str
    interaction_vars: list[str] | None = None  # For FE(a:b) interactions


@dataclass
class IdentityTermSpec:
    """Parsed identity term specification from formula (I() expressions)."""

    expression: str  # The raw expression inside I(), e.g., "x ** 2" or "x + y"


@dataclass
class CategoricalTermSpec:
    """Parsed categorical term specification with optional level selection.

    C(var) - all levels (standard treatment coding) -> levels=None
    C(var, level='Paris') - single level indicator -> levels=['Paris']
    C(var, levels=['Paris', 'Lyon']) - multiple specific levels
    """

    var_name: str
    levels: list[str] | None = None  # None = all levels, list = specific levels only


@dataclass
class ConstraintTermSpec:
    """Parsed coefficient constraint term specification.

    pos(var) - coefficient must be >= 0
    neg(var) - coefficient must be <= 0
    """

    var_name: str
    constraint: str  # "pos" or "neg"


@dataclass
class TermSlot:
    """Maps one source term to its design-matrix column range.

    Used by predict_contributions to group design columns (spline bases,
    categorical dummies, TE columns, interaction tensor products) back to
    their originating user-facing term so the contribution ladder shows
    factor-level rows instead of per-basis rows.
    """

    # User-facing name: "VehAge", "Region", "VehAge:Region", "TE(brand)", "Intercept"
    term_name: str
    term_type: str  # "intercept" | "linear" | "categorical" | "bs" | "ns" | "ms" |
    #              "target_encoding" | "frequency_encoding" | "expression" |
    #              "interaction" | "constraint" | "categorical_indicator"
    factors: list[str]  # Raw column names this term depends on
    col_start: int  # Inclusive design-matrix column index
    col_end: int  # Exclusive
    design_column_names: list[str]  # Per-column names (same as feature_names slice)
    extra: dict = field(default_factory=dict)  # Term-type-specific bag (levels, expression, etc.)


@dataclass
class ParsedFormula:
    """Parsed formula with identified terms."""

    response: str
    main_effects: list[str]  # Main effect variables
    interactions: list[InteractionTerm]  # Interaction terms
    categorical_vars: set[str]  # Variables marked as categorical with C()
    spline_terms: list[SplineTerm] = field(default_factory=list)  # Spline terms
    target_encoding_terms: list[TargetEncodingTermSpec] = field(default_factory=list)  # TE() terms
    frequency_encoding_terms: list[FrequencyEncodingTermSpec] = field(
        default_factory=list
    )  # FE() terms
    identity_terms: list[IdentityTermSpec] = field(default_factory=list)  # I() terms
    categorical_terms: list[CategoricalTermSpec] = field(
        default_factory=list
    )  # C(var, level='...') terms
    constraint_terms: list[ConstraintTermSpec] = field(default_factory=list)  # pos()/neg() terms
    has_intercept: bool = True

    # Cached lookup dicts (built lazily)
    _spline_by_var: dict[str, SplineTerm] | None = field(default=None, repr=False)
    _te_by_var: dict[str, TargetEncodingTermSpec] | None = field(default=None, repr=False)

    @property
    def spline_terms_by_var(self) -> dict[str, SplineTerm]:
        """Lookup dict: var_name → SplineTerm for O(1) access during interaction building."""
        if self._spline_by_var is None:
            self._spline_by_var = {s.var_name: s for s in self.spline_terms}
        return self._spline_by_var

    @property
    def te_terms_by_var(self) -> dict[str, TargetEncodingTermSpec]:
        """Lookup dict: var_name → TargetEncodingTermSpec for O(1) access during interaction building."""
        if self._te_by_var is None:
            self._te_by_var = {t.var_name: t for t in self.target_encoding_terms}
        return self._te_by_var


class InteractionBuilder:
    """
    Efficiently builds design matrices with interaction terms.

    Optimizations:
    1. Continuous × Continuous: Single vectorized multiplication
    2. Categorical × Continuous: Sparse-aware dummy encoding
    3. Categorical × Categorical: Direct index-based construction

    Parameters
    ----------
    data : pl.DataFrame
        Polars DataFrame
    dtype : numpy dtype, default=np.float64
        Data type for output arrays

    Example
    -------
    >>> builder = InteractionBuilder(df)
    >>> y, X, names = builder.build_design_matrix_from_parsed(parsed)
    """

    def __init__(
        self,
        data: pl.DataFrame,
        dtype: np.dtype = np.float64,
    ):
        self.data = data
        self.dtype = dtype
        self._n = len(data)

        # Consolidated cache for categorical encodings (keyed by "varname_dropfirst")
        self._cat_encoding_cache: dict[str, CategoricalEncoding] = {}
        self._cont_cache: dict[str, np.ndarray] = {}
        # Store spline terms with fitted knots for prediction
        self._fitted_splines: dict[str, SplineTerm] = {}
        # Store parsed formula for prediction
        self._parsed_formula: ParsedFormula | None = None
        # Per-term column ranges populated during _build_design_matrix_core
        # (consumed by predict_contributions to group design columns back to terms)
        self._term_slots: list[TermSlot] = []

    def get_spline_info(self) -> dict[str, dict]:
        """
        Get knot information for all fitted spline terms.

        Returns
        -------
        dict
            Dictionary mapping variable names to their spline info:
            {
                "VehAge": {
                    "type": "ms",
                    "df": 4,
                    "knots": [2.0, 5.0, 8.0],
                    "boundary_knots": [0.0, 20.0]
                },
                ...
            }
        """
        return {
            var_name: spline.get_knot_info() for var_name, spline in self._fitted_splines.items()
        }

    def get_smooth_terms(self) -> tuple:
        """
        Get smooth term information for penalized fitting.

        Returns
        -------
        smooth_terms : list[SplineTerm]
            List of SplineTerm objects that are marked as smooth (s() terms)
        smooth_col_indices : list[tuple]
            List of (start, end) column indices for each smooth term
        """
        return (getattr(self, "_smooth_terms", []), getattr(self, "_smooth_col_indices", []))

    def get_all_spline_terms(self) -> tuple:
        """
        Get ALL spline term information (including fixed-df).

        Used for D'D penalty routing when alpha > 0 on spline models.

        Returns
        -------
        all_spline_terms : list[SplineTerm]
            All SplineTerm objects (smooth and fixed-df)
        all_spline_col_indices : list[tuple]
            List of (start, end) column indices for each spline term
        """
        return (
            getattr(self, "_all_spline_terms", []),
            getattr(self, "_all_spline_col_indices", []),
        )

    def clear_caches(self) -> None:
        """
        Clear internal caches to free memory.

        This is called automatically after design matrix construction.
        Keeps:
        - Categorical levels (needed for encoding new data)
        - Target encoding stats (_te_stats)
        - Fitted splines (knot positions)
        """
        # Preserve categorical levels but clear the large encoding matrices
        # We need levels for transform_new_data() to work
        for _key, cached in self._cat_encoding_cache.items():
            # Keep the levels list but clear the large encoding matrix
            cached.encoding = None
            cached.indices = None
        # Clear any continuous value caches
        if hasattr(self, "_cont_cache"):
            self._cont_cache.clear()
        # Clear last X/names (can be large)
        self._last_X = None
        self._last_names = None

    def _parse_spline_factor(self, factor: str) -> SplineTerm | None:
        """Look up a pre-parsed spline term by variable name."""
        if self._parsed_formula is not None:
            return self._parsed_formula.spline_terms_by_var.get(factor)
        return None

    def _parse_te_factor(self, factor: str) -> TargetEncodingTermSpec | None:
        """Look up a pre-parsed TE term by variable name."""
        if self._parsed_formula is not None:
            result = self._parsed_formula.te_terms_by_var.get(factor)
            if result is not None:
                return result
            # Handle TE(...) wrapped names from interaction specs
            if factor.startswith("TE(") and factor.endswith(")"):
                var_name = factor[3:-1]
                return self._parsed_formula.te_terms_by_var.get(var_name)
        return None

    def _get_column(self, name: str) -> np.ndarray:
        """Extract column as numpy array."""
        cached = self._cont_cache.get(name)
        if cached is not None:
            return cached
        values = self.data[name].to_numpy().astype(self.dtype, copy=False)
        self._cont_cache[name] = values
        return values

    def _get_categorical_indices(self, name: str) -> tuple[np.ndarray, list[str]]:
        """Get cached categorical indices and levels for a variable."""
        cache_key = f"{name}_True"  # Always use drop_first=True for indices
        if (
            cache_key not in self._cat_encoding_cache
            or self._cat_encoding_cache[cache_key].indices is None
        ):
            self._get_categorical_indices_and_names(name)  # Populate index-only cache
        cached = self._cat_encoding_cache[cache_key]
        return cached.indices, cached.levels

    def _get_categorical_indices_and_names(
        self, name: str, drop_first: bool = True
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """Get categorical indices/levels/names without materializing dummy columns."""
        cache_key = f"{name}_{drop_first}"
        cached = self._cat_encoding_cache.get(cache_key)
        if cached is not None and cached.indices is not None:
            return cached.indices, cached.levels, cached.names

        col = self.data[name].to_numpy()
        values = [str(v) for v in col]
        levels, codes = _factorize_strings_rust(values)
        indices = np.asarray(codes, dtype=np.int32)
        start_idx = 1 if drop_first else 0
        names = [f"{name}[T.{level}]" for level in levels[start_idx:]]
        self._cat_encoding_cache[cache_key] = CategoricalEncoding(
            encoding=None,
            names=names,
            indices=indices,
            levels=list(levels),
        )
        return indices, list(levels), names

    def _get_categorical_levels(self, name: str) -> list[str]:
        """Get cached categorical levels for a variable."""
        cache_key = f"{name}_True"
        if cache_key not in self._cat_encoding_cache:
            raise EncodingError(f"Categorical variable '{name}' was not seen during training.")
        return self._cat_encoding_cache[cache_key].levels

    def _get_categorical_encoding(
        self, name: str, drop_first: bool = True
    ) -> tuple[np.ndarray, list[str]]:
        """
        Get dummy encoding for a categorical variable.

        Uses Rust for factorization and parallel matrix construction.
        Pure Rust implementation.

        Returns
        -------
        encoding : np.ndarray
            (n, k-1) dummy matrix where k is number of levels
        names : list[str]
            Column names like ['var[T.B]', 'var[T.C]', ...]
        """
        cache_key = f"{name}_{drop_first}"
        if cache_key in self._cat_encoding_cache:
            cached = self._cat_encoding_cache[cache_key]
            if cached.encoding is not None:
                return cached.encoding, cached.names
            if cached.indices is not None:
                encoding, names = _encode_categorical_indices_rust(
                    cached.indices,
                    len(cached.levels),
                    list(cached.levels),
                    name,
                    drop_first,
                )
                cached.encoding = encoding
                cached.names = names
                return cached.encoding, cached.names

        col = self.data[name].to_numpy()

        # Convert to string list for Rust factorization
        values = [str(v) for v in col]

        # Use Rust for factorization + matrix construction
        encoding, names, indices, levels = _encode_categorical_rust(values, name, drop_first)

        # Cache all encoding data in a single consolidated object
        self._cat_encoding_cache[cache_key] = CategoricalEncoding(
            encoding=encoding,
            names=names,
            indices=np.array(indices, dtype=np.int32),
            levels=levels,
        )

        return encoding, names

    def build_interaction_columns(
        self,
        interaction: InteractionTerm,
        te_encodings: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Build columns for a single interaction term.

        Optimized for different interaction types:
        - Pure continuous: Single O(n) element-wise multiply
        - Mixed: Broadcast multiply continuous with each dummy column
        - Pure categorical: Sparse index-based construction

        Parameters
        ----------
        te_encodings : dict, optional
            Pre-computed TE encodings for use in interactions like X:TE(Y)

        Returns
        -------
        columns : np.ndarray
            (n, k) interaction columns
        names : list[str]
            Column names
        """
        if interaction.is_pure_continuous:
            return self._build_continuous_interaction(interaction, te_encodings)
        elif interaction.is_pure_categorical:
            return self._build_categorical_interaction(interaction)
        else:
            return self._build_mixed_interaction(interaction)

    def _build_continuous_interaction(
        self,
        interaction: InteractionTerm,
        te_encodings: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Build continuous × continuous interaction, including spline and TE terms."""
        factors = interaction.factors
        te_encodings = te_encodings or {}

        # Separate spline, TE, and regular continuous factors
        spline_factors = []
        te_factors = []
        cont_factors = []
        force_linear = interaction.force_linear or set()
        for factor in factors:
            if factor in force_linear:
                cont_factors.append(factor)
                continue
            spline = self._parse_spline_factor(factor)
            te = self._parse_te_factor(factor)
            if spline is not None:
                spline_factors.append((factor, spline))
            elif te is not None:
                te_factors.append((factor, te))
            else:
                cont_factors.append(factor)

        # Handle continuous × spline interactions
        if spline_factors:
            all_columns = []
            all_names = []

            # Build spline basis for each spline factor
            spline_bases = []
            spline_name_lists = []
            for _spline_str, spline in spline_factors:
                x = self._get_column(spline.var_name)
                basis, names = spline.transform(x)
                self._fitted_splines[spline.var_name] = spline
                spline_bases.append(basis)
                spline_name_lists.append(names)

            # Build continuous product if any
            if cont_factors:
                cont_product = self._get_column(cont_factors[0])
                for factor in cont_factors[1:]:
                    cont_product = cont_product * self._get_column(factor)
                cont_name = ":".join(cont_factors)
            else:
                cont_product = None
                cont_name = None

            # Combine: multiply each spline column by continuous factors
            # For multiple splines, create cross-product of all spline columns
            if len(spline_bases) == 1:
                for j, spl_name in enumerate(spline_name_lists[0]):
                    col = spline_bases[0][:, j]
                    if cont_product is not None:
                        col = col * cont_product
                        all_names.append(f"{cont_name}:{spl_name}")
                    else:
                        all_names.append(spl_name)
                    all_columns.append(col)
            else:
                # Multiple splines: cross-product (rare case)
                from itertools import product as cartesian_product

                indices = [range(b.shape[1]) for b in spline_bases]
                for idx_combo in cartesian_product(*indices):
                    col = np.ones(self._n, dtype=self.dtype)
                    name_parts = []
                    for i, j in enumerate(idx_combo):
                        col = col * spline_bases[i][:, j]
                        name_parts.append(spline_name_lists[i][j])
                    if cont_product is not None:
                        col = col * cont_product
                        name_parts.insert(0, cont_name)
                    all_names.append(":".join(name_parts))
                    all_columns.append(col)

            if all_columns:
                return np.column_stack(all_columns), all_names
            return np.zeros((self._n, 0), dtype=self.dtype), []

        # Handle continuous × TE interactions
        if te_factors:
            all_columns = []
            all_names = []

            # Get TE encoded values from pre-computed encodings
            te_values = []
            te_names_list = []
            for _te_str, te_spec in te_factors:
                te_name = f"TE({te_spec.var_name})"
                if te_name in te_encodings:
                    te_values.append(te_encodings[te_name])
                    te_names_list.append(te_name)
                else:
                    raise EncodingError(
                        f"TE encoding for '{te_spec.var_name}' not found. "
                        f"Ensure TE({te_spec.var_name}) is included as a main effect."
                    )

            # Build continuous product if any
            if cont_factors:
                cont_product = self._get_column(cont_factors[0])
                for factor in cont_factors[1:]:
                    cont_product = cont_product * self._get_column(factor)
                cont_name = ":".join(cont_factors)
            else:
                cont_product = np.ones(self._n, dtype=self.dtype)
                cont_name = None

            # Multiply continuous by each TE encoding
            for te_val, te_name in zip(te_values, te_names_list):
                col = cont_product * te_val
                if cont_name:
                    all_names.append(f"{cont_name}:{te_name}")
                else:
                    all_names.append(te_name)
                all_columns.append(col)

            if all_columns:
                return np.column_stack(all_columns), all_names
            return np.zeros((self._n, 0), dtype=self.dtype), []

        # Standard continuous × continuous (no splines or TE)
        if len(factors) == 2:
            # Optimized 2-way: direct Rust call
            x1 = self._get_column(factors[0])
            x2 = self._get_column(factors[1])
            result, name = _build_cont_cont_rust(x1, x2, factors[0], factors[1])
            return result.reshape(-1, 1), [name]
        else:
            # N-way: chain pairwise Rust calls
            result = self._get_column(factors[0])
            current_name = factors[0]

            for factor in factors[1:]:
                x2 = self._get_column(factor)
                result, current_name = _build_cont_cont_rust(result, x2, current_name, factor)

            return result.reshape(-1, 1), [current_name]

    def _build_categorical_interaction(
        self, interaction: InteractionTerm
    ) -> tuple[np.ndarray, list[str]]:
        """
        Build categorical × categorical interaction efficiently.

        Uses index-based construction instead of materializing full matrices.
        """
        # Get encodings for each categorical factor
        encodings = []
        all_names = []

        for factor in interaction.factors:
            enc, names = self._get_categorical_encoding(factor)
            encodings.append(enc)
            all_names.append(names)

        if len(interaction.factors) == 2:
            # Optimized 2-way interaction
            return self._build_2way_categorical(encodings, all_names, interaction.factors)
        else:
            # General n-way interaction (slower)
            return self._build_nway_categorical(encodings, all_names, interaction.factors)

    def _build_2way_categorical(
        self,
        encodings: list[np.ndarray],
        all_names: list[list[str]],
        factors: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        """
        Optimized 2-way categorical interaction using index-based construction.

        Instead of multiplying dense matrices, we use the fact that for any row,
        at most one column in each encoding is 1. So the interaction column
        corresponding to (level_i, level_j) is 1 only if both encodings are 1.
        """
        # Get original indices (from cache or compute via encoding)
        cat1, cat2 = factors

        # Get indices and levels using consolidated cache
        idx1, levels1 = self._get_categorical_indices(cat1)
        idx2, levels2 = self._get_categorical_indices(cat2)

        # Number of non-reference levels
        n1 = len(levels1) - 1
        n2 = len(levels2) - 1

        if n1 * n2 == 0:
            return np.zeros((self._n, 0), dtype=self.dtype), []

        # Use Rust for fast parallel construction
        names1, names2 = all_names
        result, col_names = _build_cat_cat_rust(
            idx1.astype(np.int32), n1, idx2.astype(np.int32), n2, list(names1), list(names2)
        )

        return result, col_names

    def _build_nway_categorical(
        self,
        encodings: list[np.ndarray],
        all_names: list[list[str]],
        factors: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        """
        General n-way categorical interaction using recursive 2-way Rust calls.

        For 3+ way interactions, we recursively combine pairs using the
        optimized 2-way Rust implementation.
        """
        if len(factors) == 2:
            # Base case - use optimized 2-way
            return self._build_2way_categorical(encodings, all_names, factors)

        # Recursive case: combine first two factors, then combine with rest
        # Build first two factors' interaction
        first_two_enc = encodings[:2]
        first_two_names = all_names[:2]
        first_two_factors = factors[:2]

        combined, combined_names = self._build_2way_categorical(
            first_two_enc, first_two_names, first_two_factors
        )

        # Recursively combine with remaining factors
        remaining_enc = [combined] + encodings[2:]
        remaining_names = [combined_names] + all_names[2:]
        remaining_factors = [f"{first_two_factors[0]}:{first_two_factors[1]}"] + factors[2:]

        return self._build_nway_categorical(remaining_enc, remaining_names, remaining_factors)

    def _build_mixed_interaction(
        self, interaction: InteractionTerm
    ) -> tuple[np.ndarray, list[str]]:
        """Build categorical × continuous interaction using Rust."""
        # Separate categorical and continuous factors
        cat_factors = []
        cont_factors = []
        spline_factors = []  # Spline terms need special handling

        force_linear = interaction.force_linear or set()
        for factor, is_cat in zip(interaction.factors, interaction.categorical_flags):
            if is_cat:
                cat_factors.append(factor)
            elif factor in force_linear:
                cont_factors.append(factor)
            else:
                # Check if this is a spline term
                spline = self._parse_spline_factor(factor)
                if spline is not None:
                    spline_factors.append((factor, spline))
                else:
                    cont_factors.append(factor)

        # Handle spline × categorical interactions
        if spline_factors:
            if len(cat_factors) == 1:
                cat_name = cat_factors[0]
                cat_encoding, cat_names = self._get_categorical_encoding(cat_name)
            else:
                cat_interaction = InteractionTerm(
                    factors=cat_factors, categorical_flags=[True] * len(cat_factors)
                )
                cat_encoding, cat_names = self._build_categorical_interaction(cat_interaction)

            if cat_encoding.shape[1] == 0:
                return np.zeros((self._n, 0), dtype=self.dtype), []

            # Build spline basis for each spline factor
            all_columns = []
            all_names = []

            for _spline_str, spline in spline_factors:
                x = self._get_column(spline.var_name)
                spline_basis, spline_names = spline.transform(x)
                # Store fitted spline for prediction
                self._fitted_splines[spline.var_name] = spline

                # Multiply each spline column by each categorical column
                for j, spl_name in enumerate(spline_names):
                    for i, cat_name in enumerate(cat_names):
                        col = cat_encoding[:, i] * spline_basis[:, j]
                        all_columns.append(col)
                        all_names.append(f"{cat_name}:{spl_name}")

            # Also include any regular continuous factors
            if cont_factors:
                cont_product = self._get_column(cont_factors[0])
                for factor in cont_factors[1:]:
                    cont_product = cont_product * self._get_column(factor)
                cont_name = ":".join(cont_factors)

                # Multiply by continuous
                final_columns = []
                final_names = []
                for col, name in zip(all_columns, all_names):
                    final_columns.append(col * cont_product)
                    final_names.append(f"{name}:{cont_name}")
                all_columns = final_columns
                all_names = final_names

            if all_columns:
                return np.column_stack(all_columns), all_names
            return np.zeros((self._n, 0), dtype=self.dtype), []

        # Standard continuous × categorical (no splines)
        cont_product = self._get_column(cont_factors[0])
        for factor in cont_factors[1:]:
            cont_product = cont_product * self._get_column(factor)
        cont_name = ":".join(cont_factors)

        # Build categorical part and use Rust for interaction
        if len(cat_factors) == 1:
            # Single categorical - use Rust directly
            cat_name = cat_factors[0]

            # Get indices and levels using consolidated cache
            cat_indices, levels = self._get_categorical_indices(cat_name)
            n_levels = len(levels) - 1  # Excluding reference

            if n_levels == 0:
                return np.zeros((self._n, 0), dtype=self.dtype), []

            # Get category names from encoding
            _, cat_names = self._get_categorical_encoding(cat_name)

            # Use Rust for fast parallel construction
            result, col_names = _build_cat_cont_rust(
                cat_indices.astype(np.int32),
                n_levels,
                cont_product.astype(np.float64),
                list(cat_names),
                cont_name,
            )
            return result, col_names
        elif len(cat_factors) == 2:
            cat1, cat2 = cat_factors
            idx1, levels1 = self._get_categorical_indices(cat1)
            idx2, levels2 = self._get_categorical_indices(cat2)
            n1 = len(levels1) - 1
            n2 = len(levels2) - 1
            if n1 * n2 == 0:
                return np.zeros((self._n, 0), dtype=self.dtype), []
            _idx1, _levels1, names1 = self._get_categorical_indices_and_names(cat1)
            _idx2, _levels2, names2 = self._get_categorical_indices_and_names(cat2)
            result, col_names = _build_two_cat_cont_rust(
                idx1.astype(np.int32),
                n1,
                idx2.astype(np.int32),
                n2,
                cont_product.astype(np.float64),
                list(names1),
                list(names2),
                cont_name,
            )
            return result, col_names
        else:
            # Fallback for rare 3+ categorical mixed interactions.
            cat_interaction = InteractionTerm(
                factors=cat_factors, categorical_flags=[True] * len(cat_factors)
            )
            cat_encoding, cat_names = self._build_categorical_interaction(cat_interaction)
            if cat_encoding.shape[1] == 0:
                return np.zeros((self._n, 0), dtype=self.dtype), []
            result, col_names = _multiply_matrix_cont_rust(
                cat_encoding.astype(np.float64),
                cont_product.astype(np.float64),
                list(cat_names),
                cont_name,
            )
            return result, col_names

    def _build_spline_columns(
        self,
        spline: SplineTerm,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Build columns for a spline term.

        Uses SplineTerm.transform() which calls the fast Rust implementation.
        """
        x = self._get_column(spline.var_name)
        return spline.transform(x)

    def _build_target_encoding_columns(
        self,
        te_term: TargetEncodingTermSpec,
        target: np.ndarray,
        seed: int | None = None,
        exposure: np.ndarray | None = None,
    ) -> tuple[np.ndarray, str, dict]:
        """
        Build target-encoded column for a categorical variable.

        Uses ordered target statistics to prevent target leakage.

        For frequency models with exposure, uses exposure-weighted encoding:
        cumulative claims / cumulative exposure. This aligns with actuarial
        credibility theory where high-exposure observations contribute more.

        Parameters
        ----------
        te_term : TargetEncodingTermSpec
            Target encoding term specification
        target : np.ndarray
            Target variable values (e.g., ClaimCount)
        seed : int, optional
            Random seed for reproducibility
        exposure : np.ndarray, optional
            Exposure values. If provided, uses exposure-weighted target encoding
            (sum_claims / sum_exposure) instead of observation-weighted encoding.

        Returns
        -------
        encoded : np.ndarray
            Target-encoded values (n,)
        name : str
            Column name like "TE(brand)"
        stats : dict
            Level statistics for prediction on new data
        """
        has_exposure = exposure is not None
        target_f64 = target.astype(np.float64)
        exposure_f64 = exposure.astype(np.float64) if has_exposure else None

        # Check if this is a TE interaction (e.g., TE(brand:region))
        if te_term.interaction_vars is not None and len(te_term.interaction_vars) >= 2:
            cols = [self.data[var].to_numpy() for var in te_term.interaction_vars]
            cat1 = [str(v) for v in cols[0]]
            cat2 = [str(v) for v in cols[1]]

            # Select the appropriate Rust function and build shared arg tuples
            if has_exposure:
                from rustystats._rustystats import (
                    target_encode_interaction_with_exposure_py as _te_interact,
                )

                target_args = (target_f64, exposure_f64)
            else:
                from rustystats._rustystats import target_encode_interaction_py as _te_interact

                target_args = (target_f64,)
            tail_args = (te_term.prior_weight, te_term.n_permutations, seed)

            encoded, name, prior, stats = _te_interact(
                cat1,
                cat2,
                *target_args,
                te_term.interaction_vars[0],
                te_term.interaction_vars[1],
                *tail_args,
            )

            # For 3+ way interactions, combine first two then continue
            for i in range(2, len(te_term.interaction_vars)):
                combined = [f"{a}:{b}" for a, b in zip(cat1, cat2)]
                cat1 = combined
                cat2 = [str(v) for v in cols[i]]
                encoded, name, prior, stats = _te_interact(
                    cat1,
                    cat2,
                    *target_args,
                    ":".join(te_term.interaction_vars[:i]),
                    te_term.interaction_vars[i],
                    *tail_args,
                )
        else:
            # Single variable target encoding
            col = self.data[te_term.var_name].to_numpy()
            categories = [str(v) for v in col]

            if has_exposure:
                from rustystats._rustystats import target_encode_with_exposure_py

                encoded, name, prior, stats = target_encode_with_exposure_py(
                    categories,
                    target_f64,
                    exposure_f64,
                    te_term.var_name,
                    te_term.prior_weight,
                    te_term.n_permutations,
                    seed,
                )
            else:
                encoded, name, prior, stats = _target_encode_rust(
                    categories,
                    target_f64,
                    te_term.var_name,
                    te_term.prior_weight,
                    te_term.n_permutations,
                    seed,
                )

        return (
            encoded,
            name,
            {
                "prior": prior,
                "stats": stats,
                "prior_weight": te_term.prior_weight,
                "used_exposure_weighted": has_exposure,
                "interaction_vars": te_term.interaction_vars,
            },
        )

    def _build_frequency_encoding_columns(
        self,
        fe_term: FrequencyEncodingTermSpec,
    ) -> tuple[np.ndarray, str, dict]:
        """
        Build frequency-encoded column for a categorical variable.

        Encodes categories by their frequency (count / max_count).
        No target variable needed - purely based on category prevalence.

        Parameters
        ----------
        fe_term : FrequencyEncodingTermSpec
            Frequency encoding term specification

        Returns
        -------
        encoded : np.ndarray
            Frequency-encoded values (n,)
        name : str
            Column name like "FE(brand)"
        stats : dict
            Level counts for prediction on new data
        """
        from rustystats._rustystats import frequency_encode_py

        # Handle FE interactions (e.g., FE(brand:region))
        if fe_term.interaction_vars is not None and len(fe_term.interaction_vars) >= 2:
            cols = [self.data[var].to_numpy() for var in fe_term.interaction_vars]
            categories = [
                ":".join(str(cols[j][i]) for j in range(len(cols))) for i in range(len(cols[0]))
            ]
        else:
            col = self.data[fe_term.var_name].to_numpy()
            categories = [str(v) for v in col]

        encoded, name, level_counts, max_count, _n_obs = frequency_encode_py(
            categories, fe_term.var_name
        )

        return (
            encoded,
            name,
            {
                "level_counts": level_counts,
                "max_count": max_count,
                "interaction_vars": fe_term.interaction_vars,
            },
        )

    def _build_constraint_columns(
        self,
        constraint: ConstraintTermSpec,
        data: pl.DataFrame,
    ) -> tuple[np.ndarray, str]:
        """
        Build column for a constraint term (pos() or neg()).

        The column is just the variable values - the constraint is enforced during fitting.
        Supports nested expressions like pos(I(x ** 2)) or neg(I(age ** 2)).

        Parameters
        ----------
        constraint : ConstraintTermSpec
            Constraint term specification with var_name and constraint type
        data : pl.DataFrame
            DataFrame containing the column

        Returns
        -------
        values : np.ndarray
            Variable values (n,)
        name : str
            Column name like "pos(age)" or "neg(I(age ** 2))"
        """
        var_name = constraint.var_name
        name = f"{constraint.constraint}({var_name})"

        # Check if var_name is an I() expression (identity/polynomial term)
        if var_name.startswith("I(") and var_name.endswith(")"):
            # Extract expression from I(...)
            expression = var_name[2:-1]
            identity = IdentityTermSpec(expression=expression)
            values, _ = self._build_identity_columns(identity, data)
            return values, name

        # Simple variable name
        if var_name not in data.columns:
            raise ValidationError(f"Variable '{var_name}' not found in data for {name}")

        values = data[var_name].to_numpy().astype(self.dtype)
        return values, name

    def _build_identity_columns(
        self,
        identity: IdentityTermSpec,
        data: pl.DataFrame,
    ) -> tuple[np.ndarray, str]:
        """
        Build column for an identity term (I() expression).

        Evaluates expressions like I(x ** 2), I(x + y), I(x * y) against DataFrame columns.

        Parameters
        ----------
        identity : IdentityTermSpec
            Identity term specification with the expression
        data : pl.DataFrame
            DataFrame containing the columns referenced in the expression

        Returns
        -------
        values : np.ndarray
            Evaluated expression values (n,)
        name : str
            Column name like "I(x ** 2)"
        """

        expr = identity.expression
        name = f"I({expr})"

        # Convert Python ** to Polars pow() and evaluate
        # Common patterns: x ** 2, x ** 3, x + y, x * y, x / y
        try:
            # Use Polars eval with SQL-like syntax
            # Convert ** to .pow() for polars
            polars_expr = self._convert_expression_to_polars(expr)
            result = data.select(polars_expr.alias("__result__"))["__result__"].to_numpy()
            return result.astype(self.dtype), name
        except Exception as e:
            raise ValidationError(
                f"Failed to evaluate I() expression '{expr}': {e}\n"
                f"Supported operations: +, -, *, /, ** (power)\n"
                f"Example: I(x ** 2), I(x + y), I(x * y)"
            ) from e

    def _convert_expression_to_polars(self, expr: str) -> pl.Expr:
        """
        Convert a Python-style expression to a Polars expression.

        Handles:
        - x ** 2 -> col("x").pow(2)
        - x + y -> col("x") + col("y")
        - x * y -> col("x") * col("y")
        - x / y -> col("x") / col("y")
        - x - y -> col("x") - col("y")
        """
        import re

        import polars as pl

        expr = expr.strip()

        # Handle power operator: var ** num or var ** var
        power_match = re.match(r"^(\w+)\s*\*\*\s*(\d+(?:\.\d+)?|\w+)$", expr)
        if power_match:
            var_name = power_match.group(1)
            power = power_match.group(2)
            try:
                # Try to parse as number
                power_val = float(power)
                return pl.col(var_name).pow(power_val)
            except ValueError:
                # It's a column name
                return pl.col(var_name).pow(pl.col(power))

        # Handle binary operations: var op var or var op num
        binary_ops = [
            (r"^(\w+)\s*\+\s*(\w+|\d+(?:\.\d+)?)$", lambda a, b: a + b),
            (r"^(\w+)\s*-\s*(\w+|\d+(?:\.\d+)?)$", lambda a, b: a - b),
            (r"^(\w+)\s*\*\s*(\w+|\d+(?:\.\d+)?)$", lambda a, b: a * b),
            (r"^(\w+)\s*/\s*(\w+|\d+(?:\.\d+)?)$", lambda a, b: a / b),
        ]

        for pattern, op_func in binary_ops:
            match = re.match(pattern, expr)
            if match:
                left = match.group(1)
                right = match.group(2)
                left_expr = pl.col(left)
                try:
                    right_val = float(right)
                    right_expr = pl.lit(right_val)
                except ValueError:
                    right_expr = pl.col(right)
                return op_func(left_expr, right_expr)

        # If no pattern matched, try direct column reference (simple case)
        # This handles cases like I(x) which is just the column itself
        if re.match(r"^\w+$", expr):
            return pl.col(expr)

        raise ValidationError(
            f"Cannot parse expression '{expr}'. "
            f"Supported formats: 'x ** 2', 'x + y', 'x * y', 'x / y', 'x - y'"
        )

    def _build_categorical_level_indicators(
        self,
        cat_term: CategoricalTermSpec,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Build indicator columns for specific categorical levels.

        C(var, level='Paris') creates a 0/1 indicator for that level.
        C(var, levels=['Paris', 'Lyon']) creates indicators for multiple levels.

        Parameters
        ----------
        cat_term : CategoricalTermSpec
            Categorical term with level selection

        Returns
        -------
        columns : np.ndarray
            (n, k) indicator columns where k is number of specified levels
        names : list[str]
            Column names like "Region[Paris]" or "Region[Lyon]"
        """
        col = self.data[cat_term.var_name].to_numpy()
        levels = cat_term.levels or []

        if not levels:
            # No levels specified - shouldn't happen, but return empty
            return np.zeros((self._n, 0), dtype=self.dtype), []

        # Build indicator columns for each specified level
        columns = []
        names = []

        for level in levels:
            # Create 0/1 indicator for this level
            indicator = (col.astype(str) == level).astype(self.dtype)
            columns.append(indicator.reshape(-1, 1))
            names.append(f"{cat_term.var_name}[{level}]")

        if columns:
            return np.hstack(columns), names
        return np.zeros((self._n, 0), dtype=self.dtype), []

    @staticmethod
    def _stack_columns(columns: list[np.ndarray], n_rows: int, dtype: np.dtype) -> np.ndarray:
        """Stack a list of 1-D and 2-D column arrays into a single design matrix.

        Delegates the horizontal stack to a Rust kernel
        (``stack_columns_horizontal_py``) which performs a parallel,
        per-block memcpy with the GIL released. Saves ~200-400 ms on the
        1M × 30 ``result.predict()`` path versus the previous numpy loop.

        Parameters
        ----------
        columns : list of np.ndarray
            Mix of 1-D (single column) and 2-D arrays to stack horizontally.
        n_rows : int
            Number of rows (observations).
        dtype : np.dtype
            Output data type. The Rust kernel always produces float64; if
            ``dtype`` differs, the result is cast on return.

        Returns
        -------
        np.ndarray
            Stacked (n_rows, total_cols) design matrix.
        """
        if not columns:
            return np.ones((n_rows, 1), dtype=dtype)

        # Normalize: every block must be 2-D float64 for the Rust kernel.
        # ``copy=False`` avoids an extra alloc when the array is already f64.
        normalized = [
            (c if c.ndim == 2 else c.reshape(-1, 1)).astype(np.float64, copy=False) for c in columns
        ]
        out = _stack_columns_rust(normalized)
        if out.dtype != dtype:
            out = out.astype(dtype, copy=False)
        return out

    def _build_design_matrix_core(
        self,
        parsed: ParsedFormula,
        exposure: np.ndarray | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Core implementation for building design matrix from parsed formula.

        Parameters
        ----------
        parsed : ParsedFormula
            Parsed formula specification
        exposure : np.ndarray, optional
            Exposure values for target encoding
        seed : int, optional
            Random seed for deterministic target encoding

        Returns
        -------
        y : np.ndarray
            Response variable
        X : np.ndarray
            Design matrix
        names : list[str]
            Column names
        """
        columns = []
        names = []
        self._term_slots = []
        n_cols = 0  # Running column count, kept in sync with columns/names

        # Add intercept
        if parsed.has_intercept:
            columns.append(np.ones(self._n, dtype=self.dtype))
            names.append("Intercept")
            self._term_slots.append(
                TermSlot(
                    term_name="Intercept",
                    term_type="intercept",
                    factors=[],
                    col_start=n_cols,
                    col_end=n_cols + 1,
                    design_column_names=["Intercept"],
                )
            )
            n_cols += 1

        # Add main effects
        for var in parsed.main_effects:
            if var in parsed.categorical_vars:
                enc, enc_names = self._get_categorical_encoding(var)
                columns.append(enc)
                names.extend(enc_names)
                n_added = enc.shape[1]
                self._term_slots.append(
                    TermSlot(
                        term_name=var,
                        term_type="categorical",
                        factors=[var],
                        col_start=n_cols,
                        col_end=n_cols + n_added,
                        design_column_names=list(enc_names),
                        extra={"levels": list(self._cat_encoding_cache[f"{var}_True"].levels)},
                    )
                )
                n_cols += n_added
            else:
                columns.append(self._get_column(var).reshape(-1, 1))
                names.append(var)
                self._term_slots.append(
                    TermSlot(
                        term_name=var,
                        term_type="linear",
                        factors=[var],
                        col_start=n_cols,
                        col_end=n_cols + 1,
                        design_column_names=[var],
                    )
                )
                n_cols += 1

        # Add spline terms (tracking smooth term column indices for penalized fitting)
        self._smooth_terms = []  # SplineTerm objects marked as smooth
        self._smooth_col_indices = []  # (start, end) column indices
        self._all_spline_terms = []  # ALL spline terms (including fixed-df)
        self._all_spline_col_indices = []  # (start, end) for all spline terms

        for spline in parsed.spline_terms:
            col_start = n_cols
            spline_cols, spline_names = self._build_spline_columns(spline)
            col_end = col_start + spline_cols.shape[1]

            columns.append(spline_cols)
            names.extend(spline_names)
            # Store fitted spline for prediction
            self._fitted_splines[spline.var_name] = spline

            # Track ALL spline terms for D'D penalty routing
            self._all_spline_terms.append(spline)
            self._all_spline_col_indices.append((col_start, col_end))

            # Track smooth terms (those with _is_smooth flag)
            if getattr(spline, "_is_smooth", False):
                self._smooth_terms.append(spline)
                self._smooth_col_indices.append((col_start, col_end))

            self._term_slots.append(
                TermSlot(
                    term_name=spline.var_name,
                    term_type=spline.spline_type,
                    factors=[spline.var_name],
                    col_start=col_start,
                    col_end=col_end,
                    design_column_names=list(spline_names),
                    extra={"is_smooth": bool(getattr(spline, "_is_smooth", False))},
                )
            )
            n_cols = col_end

        # Store parsed formula for prediction
        self._parsed_formula = parsed

        # Get response (needed for target encoding)
        y = self._get_column(parsed.response)

        # Add target encoding terms BEFORE interactions (so TE values are available for X:TE(Y))
        # Store stats for prediction on new data
        # When exposure is provided, use rate (y/exposure) for encoding
        self._te_stats: dict[str, dict] = {}
        te_encodings: dict[str, np.ndarray] = {}  # For use in interactions
        for te_term in parsed.target_encoding_terms:
            te_col, te_name, te_stats = self._build_target_encoding_columns(
                te_term, y, seed=seed, exposure=exposure
            )
            columns.append(te_col.reshape(-1, 1))
            names.append(te_name)
            self._te_stats[te_term.var_name] = te_stats
            te_encodings[te_name] = te_col  # Store for interactions
            self._term_slots.append(
                TermSlot(
                    term_name=te_name,
                    term_type="target_encoding",
                    factors=list(te_term.interaction_vars or [te_term.var_name]),
                    col_start=n_cols,
                    col_end=n_cols + 1,
                    design_column_names=[te_name],
                    extra={
                        "var_name": te_term.var_name,
                        "interaction_vars": te_term.interaction_vars,
                        "prior_weight": te_term.prior_weight,
                    },
                )
            )
            n_cols += 1

        # Add frequency encoding terms
        self._fe_stats: dict[str, dict] = {}
        for fe_term in parsed.frequency_encoding_terms:
            fe_col, fe_name, fe_stats = self._build_frequency_encoding_columns(fe_term)
            columns.append(fe_col.reshape(-1, 1))
            names.append(fe_name)
            self._fe_stats[fe_term.var_name] = fe_stats
            self._term_slots.append(
                TermSlot(
                    term_name=fe_name,
                    term_type="frequency_encoding",
                    factors=list(fe_term.interaction_vars or [fe_term.var_name]),
                    col_start=n_cols,
                    col_end=n_cols + 1,
                    design_column_names=[fe_name],
                    extra={
                        "var_name": fe_term.var_name,
                        "interaction_vars": fe_term.interaction_vars,
                    },
                )
            )
            n_cols += 1

        # Add interactions (now with TE encodings available)
        for interaction in parsed.interactions:
            int_cols, int_names = self.build_interaction_columns(interaction, te_encodings)
            if int_cols.ndim == 1:
                int_cols = int_cols.reshape(-1, 1)
            columns.append(int_cols)
            names.extend(int_names)
            n_added = int_cols.shape[1]
            self._term_slots.append(
                TermSlot(
                    term_name=":".join(interaction.factors),
                    term_type="interaction",
                    factors=list(interaction.factors),
                    col_start=n_cols,
                    col_end=n_cols + n_added,
                    design_column_names=list(int_names),
                    extra={
                        "categorical_flags": list(interaction.categorical_flags),
                        "force_linear": list(interaction.force_linear or []),
                    },
                )
            )
            n_cols += n_added

        # Add identity terms (I() expressions like I(x ** 2))
        for identity in parsed.identity_terms:
            id_col, id_name = self._build_identity_columns(identity, self.data)
            columns.append(id_col.reshape(-1, 1))
            names.append(id_name)
            # Extract referenced columns from the expression for `factors`
            import re as _re

            id_factors = sorted(
                {
                    tok
                    for tok in _re.findall(r"\b([A-Za-z_]\w*)\b", identity.expression)
                    if tok in self.data.columns
                }
            )
            self._term_slots.append(
                TermSlot(
                    term_name=id_name,
                    term_type="expression",
                    factors=id_factors,
                    col_start=n_cols,
                    col_end=n_cols + 1,
                    design_column_names=[id_name],
                    extra={"expression": identity.expression},
                )
            )
            n_cols += 1

        # Add constraint terms (pos() / neg() for coefficient sign constraints)
        for constraint in parsed.constraint_terms:
            con_col, con_name = self._build_constraint_columns(constraint, self.data)
            columns.append(con_col.reshape(-1, 1))
            names.append(con_name)
            # Constraint may wrap a raw var or an I() expression
            inner = constraint.var_name
            if inner.startswith("I(") and inner.endswith(")"):
                import re as _re

                con_factors = sorted(
                    {
                        tok
                        for tok in _re.findall(r"\b([A-Za-z_]\w*)\b", inner[2:-1])
                        if tok in self.data.columns
                    }
                )
            else:
                con_factors = [inner] if inner in self.data.columns else []
            self._term_slots.append(
                TermSlot(
                    term_name=con_name,
                    term_type="constraint",
                    factors=con_factors,
                    col_start=n_cols,
                    col_end=n_cols + 1,
                    design_column_names=[con_name],
                    extra={"constraint": constraint.constraint, "var_name": constraint.var_name},
                )
            )
            n_cols += 1

        # Add categorical terms with level selection (C(var, level='value'))
        for cat_term in parsed.categorical_terms:
            cat_cols, cat_names = self._build_categorical_level_indicators(cat_term)
            columns.append(cat_cols)
            names.extend(cat_names)
            n_added = cat_cols.shape[1]
            if n_added > 0:
                self._term_slots.append(
                    TermSlot(
                        term_name=cat_term.var_name,
                        term_type="categorical_indicator",
                        factors=[cat_term.var_name],
                        col_start=n_cols,
                        col_end=n_cols + n_added,
                        design_column_names=list(cat_names),
                        extra={"levels": list(cat_term.levels or [])},
                    )
                )
            n_cols += n_added

        # Stack all columns using pre-allocated helper
        X = self._stack_columns(columns, self._n, self.dtype)
        if not columns:
            names = ["Intercept"]

        # Store for validation
        self._last_X = X
        self._last_names = names

        return y, X, names

    def build_design_matrix_from_parsed(
        self,
        parsed: ParsedFormula,
        exposure: np.ndarray | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Build design matrix from a pre-parsed ParsedFormula.

        This is used by the dict-based API which constructs ParsedFormula directly.

        Parameters
        ----------
        parsed : ParsedFormula
            Pre-parsed formula specification
        exposure : np.ndarray, optional
            Exposure values for target encoding
        seed : int, optional
            Random seed for deterministic target encoding

        Returns
        -------
        y : np.ndarray
            Response variable
        X : np.ndarray
            Design matrix
        names : list[str]
            Column names
        """
        return self._build_design_matrix_core(parsed, exposure=exposure, seed=seed)

    def validate_design_matrix(
        self,
        X: np.ndarray = None,
        names: list[str] | None = None,
        corr_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
        verbose: bool = True,
    ) -> dict:
        """
        Validate design matrix for common issues that cause fitting failures.

        Parameters
        ----------
        X : np.ndarray, optional
            Design matrix to validate. If None, uses last built matrix.
        names : list of str, optional
            Feature names. If None, uses last built names.
        corr_threshold : float, default=0.999
            Correlation threshold above which columns are flagged as problematic.
        verbose : bool, default=True
            Print diagnostic messages.

        Returns
        -------
        dict
            Validation results with keys:
            - 'valid': bool, True if matrix is suitable for fitting
            - 'rank': int, matrix rank
            - 'expected_rank': int, number of columns
            - 'condition_number': float, condition number (large = ill-conditioned)
            - 'problematic_columns': list of tuples (col1, col2, correlation)
            - 'zero_variance_columns': list of column names with zero variance
            - 'suggestions': list of actionable fix suggestions
        """
        if X is None:
            X = getattr(self, "_last_X", None)
            names = getattr(self, "_last_names", None)
        if X is None:
            raise DesignMatrixError(
                "No design matrix to validate. Call build_design_matrix_from_parsed() first."
            )

        n_rows, n_cols = X.shape
        results = {
            "valid": True,
            "rank": None,
            "expected_rank": n_cols,
            "condition_number": None,
            "problematic_columns": [],
            "zero_variance_columns": [],
            "suggestions": [],
        }

        # Check for NaN/Inf
        if np.isnan(X).any():
            results["valid"] = False
            nan_cols = [names[i] for i in range(n_cols) if np.isnan(X[:, i]).any()]
            results["suggestions"].append(f"Columns contain NaN values: {nan_cols}")

        if np.isinf(X).any():
            results["valid"] = False
            inf_cols = [names[i] for i in range(n_cols) if np.isinf(X[:, i]).any()]
            results["suggestions"].append(f"Columns contain Inf values: {inf_cols}")

        # Check for zero variance columns (exclude Intercept which is supposed to be constant)
        variances = np.var(X, axis=0)
        zero_var_idx = np.where(variances < ZERO_VARIANCE_THRESHOLD)[0]
        if len(zero_var_idx) > 0:
            zero_var_cols = [
                names[i] for i in zero_var_idx if i < len(names) and names[i] != "Intercept"
            ]
            if zero_var_cols:
                results["zero_variance_columns"] = zero_var_cols
                results["valid"] = False
                results["suggestions"].append(
                    f"Columns have zero/near-zero variance: {zero_var_cols}. "
                    "This often happens with splines on highly skewed data where most values are identical."
                )

        # Check matrix rank
        try:
            results["rank"] = np.linalg.matrix_rank(X)
            if results["rank"] < n_cols:
                results["valid"] = False
                results["suggestions"].append(
                    f"Matrix is rank-deficient: rank={results['rank']}, expected={n_cols}. "
                    f"{n_cols - results['rank']} columns are linearly dependent."
                )
        except Exception as e:
            raise DesignMatrixError(f"Failed to compute matrix rank: {e}") from e

        # Check condition number
        try:
            results["condition_number"] = np.linalg.cond(X)
            if results["condition_number"] > CONDITION_NUMBER_THRESHOLD:
                results["valid"] = False
                results["suggestions"].append(
                    f"Matrix is ill-conditioned (condition number={results['condition_number']:.2e}). "
                    "This indicates near-linear dependence between columns."
                )
        except Exception as e:
            raise DesignMatrixError(f"Failed to compute condition number: {e}") from e

        # Check for highly correlated columns (skip intercept)
        try:
            # Compute correlations only for non-constant columns
            non_const_idx = [i for i in range(n_cols) if variances[i] > ZERO_VARIANCE_THRESHOLD]
            if len(non_const_idx) > 1:
                X_subset = X[:, non_const_idx]
                corr_matrix = np.corrcoef(X_subset.T)

                for i in range(len(non_const_idx)):
                    for j in range(i + 1, len(non_const_idx)):
                        corr = abs(corr_matrix[i, j])
                        if corr > corr_threshold:
                            col1 = names[non_const_idx[i]]
                            col2 = names[non_const_idx[j]]
                            results["problematic_columns"].append((col1, col2, corr))

                if results["problematic_columns"]:
                    results["valid"] = False
                    pairs = [
                        f"'{c1}' <-> '{c2}' (r={r:.4f})"
                        for c1, c2, r in results["problematic_columns"]
                    ]
                    results["suggestions"].append(
                        "Highly correlated column pairs detected:\n  " + "\n  ".join(pairs) + "\n"
                        "This often happens with natural splines (ns) on skewed data. Fixes:\n"
                        "  1. Use B-splines instead: bs(VarName, df=4) - more robust to skewed data\n"
                        "  2. Use log transform: ns(log_VarName, df=4) for skewed variables\n"
                        "  3. Reduce degrees of freedom: ns(VarName, df=2)\n"
                        "  4. Use linear term instead: just 'VarName' without spline"
                    )
        except Exception as e:
            raise DesignMatrixError(f"Failed to compute column correlations: {e}") from e

        if verbose:
            print("=" * 60)
            print("DESIGN MATRIX VALIDATION")
            print("=" * 60)
            print(f"Shape: {n_rows} rows × {n_cols} columns")
            print(f"Rank: {results['rank']} / {n_cols}")
            if results["condition_number"]:
                print(f"Condition number: {results['condition_number']:.2e}")
            print(f"Status: {'✓ VALID' if results['valid'] else '✗ INVALID'}")

            if not results["valid"]:
                print("\nPROBLEMS DETECTED:")
                for i, suggestion in enumerate(results["suggestions"], 1):
                    print(f"\n{i}. {suggestion}")
            print("=" * 60)

        return results

    def transform_new_data(
        self,
        new_data: pl.DataFrame,
    ) -> np.ndarray:
        """
        Transform new data using the encoding state from training.

        This method applies the same transformations learned during
        build_design_matrix_from_parsed() to new data for prediction.

        Parameters
        ----------
        new_data : pl.DataFrame
            New data to transform. Must have same columns as training data.

        Returns
        -------
        X : np.ndarray
            Design matrix for new data

        Raises
        ------
        ValueError
            If build_design_matrix_from_parsed() was not called first, or if new data
            contains unseen categorical levels.
        """
        if self._parsed_formula is None:
            raise PredictionError(
                "Must call build_design_matrix_from_parsed() before transform_new_data(). "
                "No parsed formula has been fitted yet."
            )

        parsed = self._parsed_formula
        n_new = len(new_data)
        columns = []

        # Add intercept
        if parsed.has_intercept:
            columns.append(np.ones(n_new, dtype=self.dtype))

        # Add main effects
        for var in parsed.main_effects:
            if var in parsed.categorical_vars:
                enc = self._encode_categorical_new(new_data, var)
                columns.append(enc)
            else:
                col = new_data[var].to_numpy().astype(self.dtype)
                columns.append(col.reshape(-1, 1))

        # Add spline terms using fitted knots
        for spline in parsed.spline_terms:
            x = new_data[spline.var_name].to_numpy().astype(self.dtype)
            # Use the fitted spline which has the same knots as training
            fitted_spline = self._fitted_splines.get(spline.var_name, spline)
            spline_cols, _ = fitted_spline.transform(x)
            columns.append(spline_cols)

        # Add target encoding terms BEFORE interactions (must match build order)
        for te_term in parsed.target_encoding_terms:
            te_col = self._encode_target_new(new_data, te_term)
            columns.append(te_col.reshape(-1, 1))

        # Add frequency encoding terms
        for fe_term in parsed.frequency_encoding_terms:
            fe_col = self._encode_frequency_new(new_data, fe_term)
            columns.append(fe_col.reshape(-1, 1))

        # Add interactions (after TE terms to match build order)
        for interaction in parsed.interactions:
            int_cols = self._build_interaction_new(new_data, interaction, n_new)
            if int_cols.ndim == 1:
                int_cols = int_cols.reshape(-1, 1)
            columns.append(int_cols)

        # Add identity terms (I() expressions) - same evaluation on new data
        for identity in parsed.identity_terms:
            id_col, _ = self._build_identity_columns(identity, new_data)
            columns.append(id_col.reshape(-1, 1))

        # Add constraint terms (pos() / neg()) - same variable values on new data
        for constraint in parsed.constraint_terms:
            con_col, _ = self._build_constraint_columns(constraint, new_data)
            columns.append(con_col.reshape(-1, 1))

        # Add categorical terms with level selection (C(var, level='value'))
        for cat_term in parsed.categorical_terms:
            cat_cols, _ = self._build_categorical_level_indicators_new(cat_term, new_data)
            columns.append(cat_cols)

        # Stack all columns using pre-allocated helper
        return self._stack_columns(columns, n_new, self.dtype)

    def _map_to_training_indices(
        self,
        new_data: pl.DataFrame,
        var_name: str,
    ) -> tuple[np.ndarray, list[str]]:
        """Map new data values to training-level integer indices.

        Returns indices and levels from training. Unknown levels map to 0
        (reference level), producing all-zero dummy columns.
        """
        import polars as pl

        levels = self._get_categorical_levels(var_name)
        level_to_idx = {level: i for i, level in enumerate(levels)}

        # Polars-native mapping is ~4x faster than a Python list comprehension
        # over to_numpy() (avoids per-element str() and dict.get()).
        indices = (
            new_data[var_name]
            .cast(pl.Utf8)
            .replace_strict(level_to_idx, default=0, return_dtype=pl.Int32)
            .to_numpy()
        )
        return indices, levels

    def _encode_categorical_new(
        self,
        new_data: pl.DataFrame,
        var_name: str,
    ) -> np.ndarray:
        """Encode categorical variable using levels from training (Rust)."""
        indices, levels = self._map_to_training_indices(new_data, var_name)
        encoding, _ = _encode_categorical_indices_rust(
            indices, len(levels), list(levels), var_name, True
        )
        return np.asarray(encoding)

    def _build_interaction_new(
        self,
        new_data: pl.DataFrame,
        interaction: InteractionTerm,
        n: int,
    ) -> np.ndarray:
        """Build interaction columns for new data.

        Delegates to sub-methods matching the fit-time structure in
        build_interaction_columns.
        """
        if interaction.is_pure_continuous:
            return self._build_continuous_interaction_new(new_data, interaction, n)
        elif interaction.is_pure_categorical:
            return self._build_categorical_interaction_new(new_data, interaction, n)
        else:
            return self._build_mixed_interaction_new(new_data, interaction, n)

    def _resolve_factor_new(
        self,
        new_data: pl.DataFrame,
        factor: str,
        force_linear: set[str] | None = None,
    ) -> np.ndarray:
        """Resolve a single interaction factor to column(s) for new data.

        Returns 1-D array for scalar factors, 2-D for spline basis.
        """
        _force_linear = force_linear or set()
        te = self._parse_te_factor(factor)
        if te is not None:
            return self._encode_target_new(new_data, te)
        if factor not in _force_linear:
            spline = self._parse_spline_factor(factor)
            if spline is not None:
                x = new_data[spline.var_name].to_numpy().astype(self.dtype)
                fitted_spline = self._fitted_splines.get(spline.var_name, spline)
                basis, _ = fitted_spline.transform(x)
                return basis  # 2-D
        return new_data[factor].to_numpy().astype(self.dtype)

    def _build_continuous_interaction_new(
        self,
        new_data: pl.DataFrame,
        interaction: InteractionTerm,
        n: int,
    ) -> np.ndarray:
        """Build continuous × continuous interaction for new data (may include spline/TE)."""
        resolved = [
            self._resolve_factor_new(new_data, f, interaction.force_linear)
            for f in interaction.factors
        ]

        # If any factor is multi-column (spline basis), build cross-product
        has_multi = any(r.ndim == 2 and r.shape[1] > 1 for r in resolved)
        if not has_multi:
            # All scalar columns — element-wise product
            result = resolved[0].ravel()
            for r in resolved[1:]:
                result = result * r.ravel()
            return result.reshape(-1, 1)
        else:
            # Expand multi-column factors via outer product
            from itertools import product as cartesian_product

            bases = [r if r.ndim == 2 else r.reshape(-1, 1) for r in resolved]
            indices = [range(b.shape[1]) for b in bases]
            all_columns = []
            for idx_combo in cartesian_product(*indices):
                col = np.ones(n, dtype=self.dtype)
                for i, j in enumerate(idx_combo):
                    col = col * bases[i][:, j]
                all_columns.append(col)
            return np.column_stack(all_columns)

    def _get_categorical_names(self, var_name: str) -> list[str]:
        """Get cached categorical column names from training."""
        cache_key = f"{var_name}_True"
        cached = self._cat_encoding_cache.get(cache_key)
        if cached is not None:
            return cached.names
        return []

    def _build_categorical_interaction_new(
        self,
        new_data: pl.DataFrame,
        interaction: InteractionTerm,
        n: int,
    ) -> np.ndarray:
        """Build categorical × categorical interaction for new data (Rust)."""
        factors = interaction.factors

        if len(factors) == 2:
            # Optimized 2-way: use Rust index-based construction
            idx1, levels1 = self._map_to_training_indices(new_data, factors[0])
            idx2, levels2 = self._map_to_training_indices(new_data, factors[1])
            n1 = len(levels1) - 1
            n2 = len(levels2) - 1
            if n1 * n2 == 0:
                return np.zeros((n, 0), dtype=self.dtype)
            names1 = self._get_categorical_names(factors[0])
            names2 = self._get_categorical_names(factors[1])
            result, _ = _build_cat_cat_rust(idx1, n1, idx2, n2, list(names1), list(names2))
            return np.asarray(result)

        # N-way: start with Rust 2-way, then extend with numpy broadcasting
        idx1, levels1 = self._map_to_training_indices(new_data, factors[0])
        idx2, levels2 = self._map_to_training_indices(new_data, factors[1])
        n1, n2 = len(levels1) - 1, len(levels2) - 1
        if n1 * n2 == 0:
            return np.zeros((n, 0), dtype=self.dtype)
        names1 = self._get_categorical_names(factors[0])
        names2 = self._get_categorical_names(factors[1])
        combined, _ = _build_cat_cat_rust(idx1, n1, idx2, n2, list(names1), list(names2))
        combined = np.asarray(combined)

        for factor in factors[2:]:
            enc = self._encode_categorical_new(new_data, factor)
            # Outer product via broadcasting: combined[:, :, None] * enc[:, None, :]
            combined = (combined[:, :, None] * enc[:, None, :]).reshape(n, -1)

        return combined

    def _build_mixed_interaction_new(
        self,
        new_data: pl.DataFrame,
        interaction: InteractionTerm,
        n: int,
    ) -> np.ndarray:
        """Build categorical × continuous interaction for new data (Rust)."""
        cat_factors = []
        cont_factors = []
        spline_factors = []

        force_linear = interaction.force_linear or set()
        for factor, is_cat in zip(interaction.factors, interaction.categorical_flags):
            if is_cat:
                cat_factors.append(factor)
            elif factor in force_linear:
                cont_factors.append(factor)
            else:
                spline = self._parse_spline_factor(factor)
                if spline is not None:
                    spline_factors.append((factor, spline))
                else:
                    cont_factors.append(factor)

        # Handle spline × categorical interactions
        if spline_factors:
            if len(cat_factors) == 1:
                cat_enc = self._encode_categorical_new(new_data, cat_factors[0])
                cat_names = self._get_categorical_names(cat_factors[0])
            else:
                # Multi-categorical: build interaction via Rust
                cat_interaction = InteractionTerm(
                    factors=cat_factors, categorical_flags=[True] * len(cat_factors)
                )
                cat_enc = self._build_categorical_interaction_new(new_data, cat_interaction, n)
                cat_names = []  # Names not needed for further multiplication

            if cat_enc.shape[1] == 0:
                return np.zeros((n, 0), dtype=self.dtype)

            all_columns = []

            for _spline_str, spline in spline_factors:
                x = new_data[spline.var_name].to_numpy().astype(self.dtype)
                fitted_spline = self._fitted_splines.get(spline.var_name, spline)
                spline_basis, _ = fitted_spline.transform(x)

                # Use Rust to multiply categorical matrix by each spline column
                for j in range(spline_basis.shape[1]):
                    result, _ = _multiply_matrix_cont_rust(
                        cat_enc.astype(np.float64),
                        spline_basis[:, j].astype(np.float64),
                        list(cat_names)
                        if cat_names
                        else [f"c{i}" for i in range(cat_enc.shape[1])],
                        f"{spline.var_name}[{j}]",
                    )
                    all_columns.append(np.asarray(result))

            if cont_factors:
                cont_product = new_data[cont_factors[0]].to_numpy().astype(self.dtype)
                for factor in cont_factors[1:]:
                    cont_product = cont_product * new_data[factor].to_numpy().astype(self.dtype)
                all_columns = [col * cont_product.reshape(-1, 1) for col in all_columns]

            if all_columns:
                return np.column_stack(all_columns)
            return np.zeros((n, 0), dtype=self.dtype)

        # Standard continuous × categorical (no splines) — use Rust
        cont_product = new_data[cont_factors[0]].to_numpy().astype(self.dtype)
        for factor in cont_factors[1:]:
            cont_product = cont_product * new_data[factor].to_numpy().astype(self.dtype)
        cont_name = ":".join(cont_factors)

        if len(cat_factors) == 1:
            # Single categorical: use fast index-based Rust construction
            cat_indices, levels = self._map_to_training_indices(new_data, cat_factors[0])
            n_levels = len(levels) - 1
            if n_levels == 0:
                return np.zeros((n, 0), dtype=self.dtype)
            cat_names = self._get_categorical_names(cat_factors[0])
            result, _ = _build_cat_cont_rust(
                cat_indices.astype(np.int32),
                n_levels,
                cont_product.astype(np.float64),
                list(cat_names),
                cont_name,
            )
            return np.asarray(result)
        elif len(cat_factors) == 2:
            cat1, cat2 = cat_factors
            idx1, levels1 = self._map_to_training_indices(new_data, cat1)
            idx2, levels2 = self._map_to_training_indices(new_data, cat2)
            n1 = len(levels1) - 1
            n2 = len(levels2) - 1
            if n1 * n2 == 0:
                return np.zeros((n, 0), dtype=self.dtype)
            result, _ = _build_two_cat_cont_rust(
                idx1.astype(np.int32),
                n1,
                idx2.astype(np.int32),
                n2,
                cont_product.astype(np.float64),
                list(self._get_categorical_names(cat1)),
                list(self._get_categorical_names(cat2)),
                cont_name,
            )
            return np.asarray(result)
        else:
            # Multi-categorical: use matrix × continuous Rust
            cat_interaction = InteractionTerm(
                factors=cat_factors, categorical_flags=[True] * len(cat_factors)
            )
            cat_enc = self._build_categorical_interaction_new(new_data, cat_interaction, n)
            if cat_enc.shape[1] == 0:
                return np.zeros((n, 0), dtype=self.dtype)
            result, _ = _multiply_matrix_cont_rust(
                cat_enc.astype(np.float64),
                cont_product.astype(np.float64),
                [f"c{i}" for i in range(cat_enc.shape[1])],
                cont_name,
            )
            return np.asarray(result)

    def _encode_target_new(
        self,
        new_data: pl.DataFrame,
        te_term: TargetEncodingTermSpec,
    ) -> np.ndarray:
        """Encode using target statistics from training (Rust)."""
        if te_term.var_name not in self._te_stats:
            raise EncodingError(
                f"Target encoding for '{te_term.var_name}' was not fitted during training."
            )

        stats = self._te_stats[te_term.var_name]
        prior = stats["prior"]
        level_stats = stats["stats"]
        prior_weight = stats["prior_weight"]
        used_exposure_weighted = stats.get("used_exposure_weighted", False)
        interaction_vars = stats.get("interaction_vars")

        # Build category strings for Rust using polars-native ops to avoid
        # per-element Python str() overhead (saves ~30-80% on 1M rows).
        import polars as pl

        if interaction_vars is not None and len(interaction_vars) >= 2:
            # Polars concat_str is ~5x faster than building strings in Python
            categories = new_data.select(
                pl.concat_str(
                    [pl.col(var).cast(pl.Utf8) for var in interaction_vars],
                    separator=":",
                ).alias("__te_combined__")
            )["__te_combined__"].to_list()
        else:
            # cast(Utf8).to_list() is ~30% faster than [str(v) for v in to_numpy()]
            categories = new_data[te_term.var_name].cast(pl.Utf8).to_list()

        if used_exposure_weighted:
            encoded = _apply_exposure_weighted_te_rust(categories, level_stats, prior, prior_weight)
        else:
            encoded = _apply_target_encoding_rust(categories, level_stats, prior, prior_weight)

        return np.asarray(encoded, dtype=self.dtype)

    def _encode_frequency_new(
        self,
        new_data: pl.DataFrame,
        fe_term: FrequencyEncodingTermSpec,
    ) -> np.ndarray:
        """Encode using frequency statistics from training (Rust)."""
        if fe_term.var_name not in self._fe_stats:
            raise EncodingError(
                f"Frequency encoding for '{fe_term.var_name}' was not fitted during training."
            )

        stats = self._fe_stats[fe_term.var_name]

        # Handle FE interactions (e.g., FE(brand:region))
        interaction_vars = stats.get("interaction_vars")
        if interaction_vars is not None and len(interaction_vars) >= 2:
            cols = [new_data[var].to_numpy() for var in interaction_vars]
            categories = [
                ":".join(str(cols[j][i]) for j in range(len(cols))) for i in range(len(cols[0]))
            ]
        else:
            categories = [str(v) for v in new_data[fe_term.var_name].to_numpy()]

        encoded = _apply_frequency_encoding_rust(
            categories, stats["level_counts"], stats["max_count"]
        )
        return np.asarray(encoded, dtype=self.dtype)

    def _build_categorical_level_indicators_new(
        self,
        cat_term: CategoricalTermSpec,
        new_data: pl.DataFrame,
    ) -> tuple[np.ndarray, list[str]]:
        """Build indicator columns for specific categorical levels on new data."""
        col = new_data[cat_term.var_name].to_numpy()
        levels = cat_term.levels or []
        n = len(col)

        if not levels:
            return np.zeros((n, 0), dtype=self.dtype), []

        columns = []
        names = []

        for level in levels:
            indicator = (col.astype(str) == level).astype(self.dtype)
            columns.append(indicator.reshape(-1, 1))
            names.append(f"{cat_term.var_name}[{level}]")

        if columns:
            return np.hstack(columns), names
        return np.zeros((n, 0), dtype=self.dtype), []
