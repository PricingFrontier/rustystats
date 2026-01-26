"""
Optimized interaction term support for RustyStats.

This module provides high-performance interaction term handling for GLMs.
All heavy computation is done in Rust for maximum speed:
- Categorical encoding (Rust parallel construction)
- Interaction terms (Rust parallel for large data)
- Spline basis functions (Rust with Rayon)

The Python layer handles only:
- Formula parsing (string manipulation)
- DataFrame column extraction
- Orchestration of Rust calls

Example
-------
>>> from rustystats.interactions import InteractionBuilder
>>> 
>>> builder = InteractionBuilder(data)
>>> y, X, names = builder.build_design_matrix('y ~ x1*x2 + C(cat) + bs(age, df=5)')
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict, Set, TYPE_CHECKING

import numpy as np

# Import Rust implementations for heavy computation
from rustystats._rustystats import (
    encode_categorical_py as _encode_categorical_rust,
    build_cat_cat_interaction_py as _build_cat_cat_rust,
    build_cat_cont_interaction_py as _build_cat_cont_rust,
    build_cont_cont_interaction_py as _build_cont_cont_rust,
    multiply_matrix_by_continuous_py as _multiply_matrix_cont_rust,
    parse_formula_py as _parse_formula_rust,
    target_encode_py as _target_encode_rust,
)

if TYPE_CHECKING:
    import polars as pl


@dataclass
class InteractionTerm:
    """Represents a single interaction term like x1:x2 or C(cat1):x2."""
    
    factors: List[str]  # Variables involved (e.g., ['x1', 'x2'] or ['cat1', 'x2'])
    categorical_flags: List[bool]  # Which factors are categorical
    
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
    names: List[str]  # Column names like ['var[T.B]', 'var[T.C]']
    indices: np.ndarray  # (n,) level indices (int32)
    levels: List[str]  # All categorical levels


@dataclass
class TargetEncodingTermSpec:
    """Parsed target encoding term specification from formula."""
    var_name: str
    prior_weight: float = 1.0
    n_permutations: int = 4


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
    levels: Optional[List[str]] = None  # None = all levels, list = specific levels only


@dataclass
class ConstraintTermSpec:
    """Parsed coefficient constraint term specification.
    
    pos(var) - coefficient must be >= 0
    neg(var) - coefficient must be <= 0
    """
    var_name: str
    constraint: str  # "pos" or "neg"


@dataclass 
class ParsedFormula:
    """Parsed formula with identified terms."""
    
    response: str
    main_effects: List[str]  # Main effect variables
    interactions: List[InteractionTerm]  # Interaction terms
    categorical_vars: Set[str]  # Variables marked as categorical with C()
    spline_terms: List[SplineTerm] = field(default_factory=list)  # Spline terms
    target_encoding_terms: List[TargetEncodingTermSpec] = field(default_factory=list)  # TE() terms
    identity_terms: List[IdentityTermSpec] = field(default_factory=list)  # I() terms
    categorical_terms: List[CategoricalTermSpec] = field(default_factory=list)  # C(var, level='...') terms
    constraint_terms: List[ConstraintTermSpec] = field(default_factory=list)  # pos()/neg() terms
    has_intercept: bool = True


def parse_formula_interactions(formula: str) -> ParsedFormula:
    """
    Parse a formula string and extract interaction terms.
    
    Uses Rust for fast parsing of:
    - Main effects: x1, x2, C(cat)
    - Two-way interactions: x1:x2, x1*x2, C(cat):x
    - Higher-order: x1:x2:x3
    - Intercept removal: 0 + ... or -1
    - Spline terms: bs(x, df=5), ns(x, df=4)
    
    Parameters
    ----------
    formula : str
        R-style formula like "y ~ x1*x2 + C(cat) + bs(age, df=5)"
        
    Returns
    -------
    ParsedFormula
        Parsed structure with all terms identified
    """
    # Use Rust parser
    parsed = _parse_formula_rust(formula)
    
    # Convert to Python dataclasses
    interactions = [
        InteractionTerm(
            factors=i['factors'],
            categorical_flags=i['categorical_flags']
        )
        for i in parsed['interactions']
    ]
    
    spline_terms = []
    for s in parsed['spline_terms']:
        # Convert legacy 'increasing' to 'monotonicity' if present
        monotonicity = s.get('monotonicity')
        if monotonicity is None and s.get('monotonic', False):
            monotonicity = 'increasing' if s.get('increasing', True) else 'decreasing'
        
        term = SplineTerm(
            var_name=s['var_name'],
            spline_type=s['spline_type'],
            df=s['df'],
            degree=s['degree'],
            monotonicity=monotonicity
        )
        # Set smooth flag for penalized splines with auto lambda selection
        term._is_smooth = s.get('is_smooth', False)
        spline_terms.append(term)
    
    # Parse target encoding terms
    target_encoding_terms = [
        TargetEncodingTermSpec(
            var_name=t['var_name'],
            prior_weight=t['prior_weight'],
            n_permutations=t['n_permutations']
        )
        for t in parsed.get('target_encoding_terms', [])
    ]
    
    # Parse identity terms (I() expressions)
    identity_terms = [
        IdentityTermSpec(expression=i['expression'])
        for i in parsed.get('identity_terms', [])
    ]
    
    # Parse categorical terms with level selection (C(var, level='...'))
    categorical_terms = [
        CategoricalTermSpec(
            var_name=c['var_name'],
            levels=c['levels']
        )
        for c in parsed.get('categorical_terms', [])
    ]
    
    # Parse constraint terms (pos() / neg())
    constraint_terms = [
        ConstraintTermSpec(
            var_name=c['var_name'],
            constraint=c['constraint']
        )
        for c in parsed.get('constraint_terms', [])
    ]
    
    # Filter out "1" from main effects (it's just an explicit intercept indicator)
    main_effects = [m for m in parsed['main_effects'] if m != '1']
    
    return ParsedFormula(
        response=parsed['response'],
        main_effects=main_effects,
        interactions=interactions,
        categorical_vars=set(parsed['categorical_vars']),
        spline_terms=spline_terms,
        target_encoding_terms=target_encoding_terms,
        identity_terms=identity_terms,
        categorical_terms=categorical_terms,
        constraint_terms=constraint_terms,
        has_intercept=parsed['has_intercept'],
    )


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
    >>> X, names = builder.build_matrix('y ~ x1*x2 + C(area):age')
    """
    
    def __init__(
        self,
        data: "pl.DataFrame",
        dtype: np.dtype = np.float64,
    ):
        self.data = data
        self.dtype = dtype
        self._n = len(data)
        
        # Consolidated cache for categorical encodings (keyed by "varname_dropfirst")
        self._cat_encoding_cache: Dict[str, CategoricalEncoding] = {}
        # Store spline terms with fitted knots for prediction
        self._fitted_splines: Dict[str, SplineTerm] = {}
        # Store parsed formula for prediction
        self._parsed_formula: Optional[ParsedFormula] = None
    
    def get_spline_info(self) -> Dict[str, dict]:
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
            var_name: spline.get_knot_info()
            for var_name, spline in self._fitted_splines.items()
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
        return (
            getattr(self, '_smooth_terms', []),
            getattr(self, '_smooth_col_indices', [])
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
        for key, cached in self._cat_encoding_cache.items():
            # Keep the levels list but clear the large encoding matrix
            cached.encoding = None
            cached.indices = None
        # Clear any continuous value caches
        if hasattr(self, '_cont_cache'):
            self._cont_cache.clear()
        # Clear last X/names (can be large)
        self._last_X = None
        self._last_names = None
    
    def _parse_spline_factor(self, factor: str) -> Optional[SplineTerm]:
        """Parse a spline term from a factor name like 'bs(VehAge, df=4)' or 'ns(age, df=3)'."""
        factor_lower = factor.strip().lower()
        if factor_lower.startswith('bs(') or factor_lower.startswith('ns('):
            spline_type = 'bs' if factor_lower.startswith('bs(') else 'ns'
            # Extract content inside parentheses
            content = factor[3:-1] if factor.endswith(')') else factor[3:]
            parts = [p.strip() for p in content.split(',')]
            var_name = parts[0]
            df = None  # default: penalized smooth
            k = None
            degree = 3  # default for B-splines
            monotonicity = None
            for part in parts[1:]:
                if '=' in part:
                    key, val = part.split('=', 1)
                    key = key.strip().lower()
                    val = val.strip()
                    if key == 'df':
                        df = int(val)
                    elif key == 'k':
                        k = int(val)
                    elif key == 'degree':
                        degree = int(val)
                    elif key == 'monotonicity':
                        monotonicity = val.strip("'\"").lower()
            
            # Determine effective df and whether this is a smooth term
            is_smooth = False
            if df is None and k is None:
                # No df or k specified: default to penalized smooth with k=10
                effective_df = 10
                is_smooth = True
            elif k is not None:
                # k parameter specified: penalized smooth
                effective_df = k
                is_smooth = True
            else:
                # df parameter specified: fixed df, no penalty
                effective_df = df
            
            term = SplineTerm(var_name=var_name, spline_type=spline_type, df=effective_df, 
                              degree=degree, monotonicity=monotonicity)
            if is_smooth:
                term._is_smooth = True
            return term
        
        return None
    
    def _parse_te_factor(self, factor: str) -> Optional[TargetEncodingTermSpec]:
        """Parse a TE term from a factor name like 'TE(Region)' or 'TE(Brand, pw=2)'."""
        factor_stripped = factor.strip()
        if factor_stripped.upper().startswith('TE(') and factor_stripped.endswith(')'):
            content = factor_stripped[3:-1]
            parts = [p.strip() for p in content.split(',')]
            var_name = parts[0]
            prior_weight = 1.0
            n_permutations = 4
            for part in parts[1:]:
                if '=' in part:
                    key, val = part.split('=', 1)
                    key = key.strip().lower()
                    val = val.strip()
                    if key in ('pw', 'prior_weight'):
                        prior_weight = float(val)
                    elif key in ('n', 'n_permutations'):
                        n_permutations = int(val)
            return TargetEncodingTermSpec(var_name=var_name, prior_weight=prior_weight, n_permutations=n_permutations)
        return None
    
    def _get_column(self, name: str) -> np.ndarray:
        """Extract column as numpy array."""
        return self.data[name].to_numpy().astype(self.dtype)
    
    def _get_categorical_indices(self, name: str) -> Tuple[np.ndarray, List[str]]:
        """Get cached categorical indices and levels for a variable."""
        cache_key = f"{name}_True"  # Always use drop_first=True for indices
        if cache_key not in self._cat_encoding_cache:
            self._get_categorical_encoding(name)  # Populate cache
        cached = self._cat_encoding_cache[cache_key]
        return cached.indices, cached.levels
    
    def _get_categorical_levels(self, name: str) -> List[str]:
        """Get cached categorical levels for a variable."""
        cache_key = f"{name}_True"
        if cache_key not in self._cat_encoding_cache:
            raise ValueError(f"Categorical variable '{name}' was not seen during training.")
        return self._cat_encoding_cache[cache_key].levels
    
    def _get_categorical_encoding(
        self, 
        name: str,
        drop_first: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
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
        te_encodings: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
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
        te_encodings: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Build continuous × continuous interaction, including spline and TE terms."""
        factors = interaction.factors
        te_encodings = te_encodings or {}
        
        # Separate spline, TE, and regular continuous factors
        spline_factors = []
        te_factors = []
        cont_factors = []
        for factor in factors:
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
            for spline_str, spline in spline_factors:
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
                cont_name = ':'.join(cont_factors)
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
                    all_names.append(':'.join(name_parts))
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
            for te_str, te_spec in te_factors:
                te_name = f"TE({te_spec.var_name})"
                if te_name in te_encodings:
                    te_values.append(te_encodings[te_name])
                    te_names_list.append(te_name)
                else:
                    raise ValueError(f"TE encoding for '{te_spec.var_name}' not found. "
                                   f"Ensure TE({te_spec.var_name}) is included as a main effect.")
            
            # Build continuous product if any
            if cont_factors:
                cont_product = self._get_column(cont_factors[0])
                for factor in cont_factors[1:]:
                    cont_product = cont_product * self._get_column(factor)
                cont_name = ':'.join(cont_factors)
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
        self,
        interaction: InteractionTerm
    ) -> Tuple[np.ndarray, List[str]]:
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
        encodings: List[np.ndarray],
        all_names: List[List[str]],
        factors: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
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
            idx1.astype(np.int32), n1,
            idx2.astype(np.int32), n2,
            list(names1), list(names2)
        )
        
        return result, col_names
    
    def _build_nway_categorical(
        self,
        encodings: List[np.ndarray],
        all_names: List[List[str]],
        factors: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
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
        self,
        interaction: InteractionTerm
    ) -> Tuple[np.ndarray, List[str]]:
        """Build categorical × continuous interaction using Rust."""
        # Separate categorical and continuous factors
        cat_factors = []
        cont_factors = []
        spline_factors = []  # Spline terms need special handling
        
        for factor, is_cat in zip(interaction.factors, interaction.categorical_flags):
            if is_cat:
                cat_factors.append(factor)
            else:
                # Check if this is a spline term
                spline = self._parse_spline_factor(factor)
                if spline is not None:
                    spline_factors.append((factor, spline))
                else:
                    cont_factors.append(factor)
        
        # Build categorical encoding first
        if len(cat_factors) == 1:
            cat_name = cat_factors[0]
            cat_encoding, cat_names = self._get_categorical_encoding(cat_name)
        else:
            cat_interaction = InteractionTerm(
                factors=cat_factors,
                categorical_flags=[True] * len(cat_factors)
            )
            cat_encoding, cat_names = self._build_categorical_interaction(cat_interaction)
        
        if cat_encoding.shape[1] == 0:
            return np.zeros((self._n, 0), dtype=self.dtype), []
        
        # Handle spline × categorical interactions
        if spline_factors:
            # Build spline basis for each spline factor
            all_columns = []
            all_names = []
            
            for spline_str, spline in spline_factors:
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
                cont_name = ':'.join(cont_factors)
                
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
        cont_name = ':'.join(cont_factors)
        
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
                cont_name
            )
            return result, col_names
        else:
            # Multiple categorical - build their interaction first, then multiply using Rust
            cat_interaction = InteractionTerm(
                factors=cat_factors,
                categorical_flags=[True] * len(cat_factors)
            )
            cat_encoding, cat_names = self._build_categorical_interaction(cat_interaction)
            
            # Use Rust to multiply categorical matrix by continuous
            result, col_names = _multiply_matrix_cont_rust(
                cat_encoding.astype(np.float64),
                cont_product.astype(np.float64),
                list(cat_names),
                cont_name
            )
            return result, col_names
    
    def _build_spline_columns(
        self,
        spline: SplineTerm,
    ) -> Tuple[np.ndarray, List[str]]:
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
        seed: Optional[int] = None,
        exposure: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, str, dict]:
        """
        Build target-encoded column for a categorical variable.
        
        Uses ordered target statistics to prevent target leakage.
        
        For frequency models with exposure, uses claim rate (target/exposure)
        instead of raw counts to produce more meaningful encoded values.
        
        Parameters
        ----------
        te_term : TargetEncodingTermSpec
            Target encoding term specification
        target : np.ndarray
            Target variable values (e.g., ClaimCount)
        seed : int, optional
            Random seed for reproducibility
        exposure : np.ndarray, optional
            Exposure values. If provided, target encoding uses rate (target/exposure)
            instead of raw target values. This prevents collapse to near-constant
            values for low-frequency count data.
            
        Returns
        -------
        encoded : np.ndarray
            Target-encoded values (n,)
        name : str
            Column name like "TE(brand)"
        stats : dict
            Level statistics for prediction on new data
        """
        col = self.data[te_term.var_name].to_numpy()
        categories = [str(v) for v in col]
        
        # Use rate (target/exposure) for encoding when exposure is available
        # This prevents near-constant encoded values for low-frequency count data
        if exposure is not None:
            encoding_target = (target / np.maximum(exposure, 1e-10)).astype(np.float64)
        else:
            encoding_target = target.astype(np.float64)
        
        encoded, name, prior, stats = _target_encode_rust(
            categories,
            encoding_target,
            te_term.var_name,
            te_term.prior_weight,
            te_term.n_permutations,
            seed,
        )
        
        # Store whether we used rate encoding for prediction
        return encoded, name, {
            'prior': prior, 
            'stats': stats, 
            'prior_weight': te_term.prior_weight,
            'used_rate_encoding': exposure is not None,
        }
    
    def _build_constraint_columns(
        self,
        constraint: ConstraintTermSpec,
        data: "pl.DataFrame",
    ) -> Tuple[np.ndarray, str]:
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
            raise ValueError(f"Variable '{var_name}' not found in data for {name}")
        
        values = data[var_name].to_numpy().astype(self.dtype)
        return values, name
    
    def _build_identity_columns(
        self,
        identity: IdentityTermSpec,
        data: "pl.DataFrame",
    ) -> Tuple[np.ndarray, str]:
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
        import polars as pl
        
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
            raise ValueError(
                f"Failed to evaluate I() expression '{expr}': {e}\n"
                f"Supported operations: +, -, *, /, ** (power)\n"
                f"Example: I(x ** 2), I(x + y), I(x * y)"
            ) from e
    
    def _convert_expression_to_polars(self, expr: str) -> "pl.Expr":
        """
        Convert a Python-style expression to a Polars expression.
        
        Handles:
        - x ** 2 -> col("x").pow(2)
        - x + y -> col("x") + col("y")
        - x * y -> col("x") * col("y")
        - x / y -> col("x") / col("y")
        - x - y -> col("x") - col("y")
        """
        import polars as pl
        import re
        
        expr = expr.strip()
        
        # Handle power operator: var ** num or var ** var
        power_match = re.match(r'^(\w+)\s*\*\*\s*(\d+(?:\.\d+)?|\w+)$', expr)
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
            (r'^(\w+)\s*\+\s*(\w+|\d+(?:\.\d+)?)$', lambda a, b: a + b),
            (r'^(\w+)\s*-\s*(\w+|\d+(?:\.\d+)?)$', lambda a, b: a - b),
            (r'^(\w+)\s*\*\s*(\w+|\d+(?:\.\d+)?)$', lambda a, b: a * b),
            (r'^(\w+)\s*/\s*(\w+|\d+(?:\.\d+)?)$', lambda a, b: a / b),
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
        if re.match(r'^\w+$', expr):
            return pl.col(expr)
        
        raise ValueError(
            f"Cannot parse expression '{expr}'. "
            f"Supported formats: 'x ** 2', 'x + y', 'x * y', 'x / y', 'x - y'"
        )
    
    def _build_categorical_level_indicators(
        self,
        cat_term: CategoricalTermSpec,
    ) -> Tuple[np.ndarray, List[str]]:
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
    
    def _build_design_matrix_core(
        self,
        parsed: ParsedFormula,
        exposure: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Core implementation for building design matrix from parsed formula.
        
        This is the shared implementation used by both build_design_matrix()
        and build_design_matrix_from_parsed().
        
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
        
        # Add intercept
        if parsed.has_intercept:
            columns.append(np.ones(self._n, dtype=self.dtype))
            names.append('Intercept')
        
        # Add main effects
        for var in parsed.main_effects:
            if var in parsed.categorical_vars:
                enc, enc_names = self._get_categorical_encoding(var)
                columns.append(enc)
                names.extend(enc_names)
            else:
                columns.append(self._get_column(var).reshape(-1, 1))
                names.append(var)
        
        # Add spline terms (tracking smooth term column indices for penalized fitting)
        self._smooth_terms = []  # SplineTerm objects marked as smooth
        self._smooth_col_indices = []  # (start, end) column indices
        
        for spline in parsed.spline_terms:
            col_start = sum(c.shape[1] if c.ndim == 2 else 1 for c in columns)
            spline_cols, spline_names = self._build_spline_columns(spline)
            col_end = col_start + spline_cols.shape[1]
            
            columns.append(spline_cols)
            names.extend(spline_names)
            # Store fitted spline for prediction
            self._fitted_splines[spline.var_name] = spline
            
            # Track smooth terms (those with _is_smooth flag)
            if getattr(spline, '_is_smooth', False):
                self._smooth_terms.append(spline)
                self._smooth_col_indices.append((col_start, col_end))
        
        # Store parsed formula for prediction
        self._parsed_formula = parsed
        
        # Get response (needed for target encoding)
        y = self._get_column(parsed.response)
        
        # Add target encoding terms BEFORE interactions (so TE values are available for X:TE(Y))
        # Store stats for prediction on new data
        # When exposure is provided, use rate (y/exposure) for encoding
        self._te_stats: Dict[str, dict] = {}
        te_encodings: Dict[str, np.ndarray] = {}  # For use in interactions
        for te_term in parsed.target_encoding_terms:
            te_col, te_name, te_stats = self._build_target_encoding_columns(
                te_term, y, seed=seed, exposure=exposure
            )
            columns.append(te_col.reshape(-1, 1))
            names.append(te_name)
            self._te_stats[te_term.var_name] = te_stats
            te_encodings[te_name] = te_col  # Store for interactions
        
        # Add interactions (now with TE encodings available)
        for interaction in parsed.interactions:
            int_cols, int_names = self.build_interaction_columns(interaction, te_encodings)
            if int_cols.ndim == 1:
                int_cols = int_cols.reshape(-1, 1)
            columns.append(int_cols)
            names.extend(int_names)
        
        # Add identity terms (I() expressions like I(x ** 2))
        for identity in parsed.identity_terms:
            id_col, id_name = self._build_identity_columns(identity, self.data)
            columns.append(id_col.reshape(-1, 1))
            names.append(id_name)
        
        # Add constraint terms (pos() / neg() for coefficient sign constraints)
        for constraint in parsed.constraint_terms:
            con_col, con_name = self._build_constraint_columns(constraint, self.data)
            columns.append(con_col.reshape(-1, 1))
            names.append(con_name)
        
        # Add categorical terms with level selection (C(var, level='value'))
        for cat_term in parsed.categorical_terms:
            cat_cols, cat_names = self._build_categorical_level_indicators(cat_term)
            columns.append(cat_cols)
            names.extend(cat_names)
        
        # Stack all columns using pre-allocation (more memory efficient than np.hstack)
        if columns:
            # Calculate total columns and pre-allocate
            total_cols = 0
            for c in columns:
                total_cols += c.shape[1] if c.ndim == 2 else 1
            
            X = np.empty((self._n, total_cols), dtype=self.dtype)
            
            # Fill in place
            col_idx = 0
            for c in columns:
                if c.ndim == 1:
                    X[:, col_idx] = c
                    col_idx += 1
                else:
                    width = c.shape[1]
                    X[:, col_idx:col_idx + width] = c
                    col_idx += width
        else:
            X = np.ones((self._n, 1), dtype=self.dtype)
            names = ['Intercept']
        
        # Store for validation
        self._last_X = X
        self._last_names = names
        
        return y, X, names
    
    def build_design_matrix(
        self,
        formula: str,
        exposure: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build complete design matrix from formula.
        
        Parameters
        ----------
        formula : str
            R-style formula like "y ~ x1*x2 + C(cat) + bs(age, df=5)"
        exposure : np.ndarray, optional
            Exposure values. If provided, target encoding (TE) will use
            rate (y/exposure) instead of raw y values. This is important
            for frequency models to prevent TE values collapsing to near-constant.
        seed : int, optional
            Random seed for deterministic target encoding. If None, TE uses
            random permutations (non-deterministic).
            
        Returns
        -------
        y : np.ndarray
            Response variable
        X : np.ndarray
            Design matrix
        names : list[str]
            Column names
        """
        parsed = parse_formula_interactions(formula)
        return self._build_design_matrix_core(parsed, exposure=exposure, seed=seed)
    
    def build_design_matrix_from_parsed(
        self,
        parsed: ParsedFormula,
        exposure: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
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
        names: List[str] = None,
        corr_threshold: float = 0.999,
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
            X = getattr(self, '_last_X', None)
            names = getattr(self, '_last_names', None)
        if X is None:
            raise ValueError("No design matrix to validate. Call build_design_matrix() first.")
        
        n_rows, n_cols = X.shape
        results = {
            'valid': True,
            'rank': None,
            'expected_rank': n_cols,
            'condition_number': None,
            'problematic_columns': [],
            'zero_variance_columns': [],
            'suggestions': [],
        }
        
        # Check for NaN/Inf
        if np.isnan(X).any():
            results['valid'] = False
            nan_cols = [names[i] for i in range(n_cols) if np.isnan(X[:, i]).any()]
            results['suggestions'].append(f"Columns contain NaN values: {nan_cols}")
        
        if np.isinf(X).any():
            results['valid'] = False
            inf_cols = [names[i] for i in range(n_cols) if np.isinf(X[:, i]).any()]
            results['suggestions'].append(f"Columns contain Inf values: {inf_cols}")
        
        # Check for zero variance columns (exclude Intercept which is supposed to be constant)
        variances = np.var(X, axis=0)
        zero_var_idx = np.where(variances < 1e-10)[0]
        if len(zero_var_idx) > 0:
            zero_var_cols = [names[i] for i in zero_var_idx if i < len(names) and names[i] != 'Intercept']
            if zero_var_cols:
                results['zero_variance_columns'] = zero_var_cols
                results['valid'] = False
                results['suggestions'].append(
                    f"Columns have zero/near-zero variance: {zero_var_cols}. "
                    "This often happens with splines on highly skewed data where most values are identical."
                )
        
        # Check matrix rank
        try:
            results['rank'] = np.linalg.matrix_rank(X)
            if results['rank'] < n_cols:
                results['valid'] = False
                results['suggestions'].append(
                    f"Matrix is rank-deficient: rank={results['rank']}, expected={n_cols}. "
                    f"{n_cols - results['rank']} columns are linearly dependent."
                )
        except Exception as e:
            raise RuntimeError(f"Failed to compute matrix rank: {e}") from e
        
        # Check condition number
        try:
            results['condition_number'] = np.linalg.cond(X)
            if results['condition_number'] > 1e10:
                results['valid'] = False
                results['suggestions'].append(
                    f"Matrix is ill-conditioned (condition number={results['condition_number']:.2e}). "
                    "This indicates near-linear dependence between columns."
                )
        except Exception as e:
            raise RuntimeError(f"Failed to compute condition number: {e}") from e
        
        # Check for highly correlated columns (skip intercept)
        try:
            # Compute correlations only for non-constant columns
            non_const_idx = [i for i in range(n_cols) if variances[i] > 1e-10]
            if len(non_const_idx) > 1:
                X_subset = X[:, non_const_idx]
                corr_matrix = np.corrcoef(X_subset.T)
                
                for i in range(len(non_const_idx)):
                    for j in range(i + 1, len(non_const_idx)):
                        corr = abs(corr_matrix[i, j])
                        if corr > corr_threshold:
                            col1 = names[non_const_idx[i]]
                            col2 = names[non_const_idx[j]]
                            results['problematic_columns'].append((col1, col2, corr))
                            
                if results['problematic_columns']:
                    results['valid'] = False
                    pairs = [f"'{c1}' <-> '{c2}' (r={r:.4f})" for c1, c2, r in results['problematic_columns']]
                    results['suggestions'].append(
                        f"Highly correlated column pairs detected:\n  " + "\n  ".join(pairs) + "\n"
                        "This often happens with natural splines (ns) on skewed data. Fixes:\n"
                        "  1. Use B-splines instead: bs(VarName, df=4) - more robust to skewed data\n"
                        "  2. Use log transform: ns(log_VarName, df=4) for skewed variables\n"
                        "  3. Reduce degrees of freedom: ns(VarName, df=2)\n"
                        "  4. Use linear term instead: just 'VarName' without spline"
                    )
        except Exception as e:
            raise RuntimeError(f"Failed to compute column correlations: {e}") from e
        
        if verbose:
            print("=" * 60)
            print("DESIGN MATRIX VALIDATION")
            print("=" * 60)
            print(f"Shape: {n_rows} rows × {n_cols} columns")
            print(f"Rank: {results['rank']} / {n_cols}")
            if results['condition_number']:
                print(f"Condition number: {results['condition_number']:.2e}")
            print(f"Status: {'✓ VALID' if results['valid'] else '✗ INVALID'}")
            
            if not results['valid']:
                print("\nPROBLEMS DETECTED:")
                for i, suggestion in enumerate(results['suggestions'], 1):
                    print(f"\n{i}. {suggestion}")
            print("=" * 60)
        
        return results
    
    def transform_new_data(
        self,
        new_data: "pl.DataFrame",
    ) -> np.ndarray:
        """
        Transform new data using the encoding state from training.
        
        This method applies the same transformations learned during
        build_design_matrix() to new data for prediction.
        
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
            If build_design_matrix() was not called first, or if new data
            contains unseen categorical levels.
        """
        if self._parsed_formula is None:
            raise ValueError(
                "Must call build_design_matrix() before transform_new_data(). "
                "No formula has been fitted yet."
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
        
        # Add target encoding terms BEFORE interactions (must match build_design_matrix order)
        for te_term in parsed.target_encoding_terms:
            te_col = self._encode_target_new(new_data, te_term)
            columns.append(te_col.reshape(-1, 1))
        
        # Add interactions (after TE terms to match build_design_matrix order)
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
        
        # Stack all columns
        if columns:
            X = np.hstack([c if c.ndim == 2 else c.reshape(-1, 1) for c in columns])
        else:
            X = np.ones((n_new, 1), dtype=self.dtype)
        
        return X
    
    def _encode_categorical_new(
        self,
        new_data: "pl.DataFrame",
        var_name: str,
    ) -> np.ndarray:
        """Encode categorical variable using levels from training."""
        levels = self._get_categorical_levels(var_name)
        col = new_data[var_name].to_numpy()
        n = len(col)
        
        # Create level to index mapping (reference level is index 0)
        level_to_idx = {level: i for i, level in enumerate(levels)}
        
        # Number of dummy columns (excluding reference level)
        n_dummies = len(levels) - 1
        encoding = np.zeros((n, n_dummies), dtype=self.dtype)
        
        for i, val in enumerate(col):
            val_str = str(val)
            if val_str in level_to_idx:
                idx = level_to_idx[val_str]
                if idx > 0:  # Skip reference level
                    encoding[i, idx - 1] = 1.0
            # Unknown levels get all zeros (mapped to reference)
        
        return encoding
    
    def _build_interaction_new(
        self,
        new_data: "pl.DataFrame",
        interaction: InteractionTerm,
        n: int,
    ) -> np.ndarray:
        """Build interaction columns for new data."""
        if interaction.is_pure_continuous:
            # Continuous × continuous
            result = new_data[interaction.factors[0]].to_numpy().astype(self.dtype)
            for factor in interaction.factors[1:]:
                result = result * new_data[factor].to_numpy().astype(self.dtype)
            return result.reshape(-1, 1)
        
        elif interaction.is_pure_categorical:
            # Categorical × categorical
            encodings = []
            for factor in interaction.factors:
                enc = self._encode_categorical_new(new_data, factor)
                encodings.append(enc)
            
            # Build interaction by taking outer product
            result = encodings[0]
            for enc in encodings[1:]:
                # Kronecker-style expansion
                n_cols1, n_cols2 = result.shape[1], enc.shape[1]
                new_result = np.zeros((n, n_cols1 * n_cols2), dtype=self.dtype)
                for i in range(n_cols1):
                    for j in range(n_cols2):
                        new_result[:, i * n_cols2 + j] = result[:, i] * enc[:, j]
                result = new_result
            return result
        
        else:
            # Mixed: categorical × continuous (may include splines)
            cat_factors = []
            cont_factors = []
            spline_factors = []
            
            for factor, is_cat in zip(interaction.factors, interaction.categorical_flags):
                if is_cat:
                    cat_factors.append(factor)
                else:
                    # Check if this is a spline term
                    spline = self._parse_spline_factor(factor)
                    if spline is not None:
                        spline_factors.append((factor, spline))
                    else:
                        cont_factors.append(factor)
            
            # Build categorical encoding
            if len(cat_factors) == 1:
                cat_enc = self._encode_categorical_new(new_data, cat_factors[0])
            else:
                # Multiple categorical - build their interaction
                cat_enc = self._encode_categorical_new(new_data, cat_factors[0])
                for factor in cat_factors[1:]:
                    enc = self._encode_categorical_new(new_data, factor)
                    n_cols1, n_cols2 = cat_enc.shape[1], enc.shape[1]
                    new_enc = np.zeros((n, n_cols1 * n_cols2), dtype=self.dtype)
                    for i in range(n_cols1):
                        for j in range(n_cols2):
                            new_enc[:, i * n_cols2 + j] = cat_enc[:, i] * enc[:, j]
                    cat_enc = new_enc
            
            # Handle spline × categorical interactions
            if spline_factors:
                all_columns = []
                
                for spline_str, spline in spline_factors:
                    x = new_data[spline.var_name].to_numpy().astype(self.dtype)
                    # Use the fitted spline which has the same knots as training
                    fitted_spline = self._fitted_splines.get(spline.var_name, spline)
                    spline_basis, _ = fitted_spline.transform(x)
                    
                    # Multiply each spline column by each categorical column
                    for j in range(spline_basis.shape[1]):
                        for i in range(cat_enc.shape[1]):
                            col = cat_enc[:, i] * spline_basis[:, j]
                            all_columns.append(col)
                
                # Also include any regular continuous factors
                if cont_factors:
                    cont_product = new_data[cont_factors[0]].to_numpy().astype(self.dtype)
                    for factor in cont_factors[1:]:
                        cont_product = cont_product * new_data[factor].to_numpy().astype(self.dtype)
                    
                    # Multiply by continuous
                    all_columns = [col * cont_product for col in all_columns]
                
                if all_columns:
                    return np.column_stack(all_columns)
                return np.zeros((n, 0), dtype=self.dtype)
            
            # Standard continuous × categorical (no splines)
            cont_product = new_data[cont_factors[0]].to_numpy().astype(self.dtype)
            for factor in cont_factors[1:]:
                cont_product = cont_product * new_data[factor].to_numpy().astype(self.dtype)
            
            # Multiply categorical dummies by continuous
            result = cat_enc * cont_product.reshape(-1, 1)
            return result
    
    def _encode_target_new(
        self,
        new_data: "pl.DataFrame",
        te_term: TargetEncodingTermSpec,
    ) -> np.ndarray:
        """Encode using target statistics from training."""
        if te_term.var_name not in self._te_stats:
            raise ValueError(
                f"Target encoding for '{te_term.var_name}' was not fitted during training."
            )
        
        stats = self._te_stats[te_term.var_name]
        prior = stats['prior']
        level_stats = stats['stats']  # Dict[str, (sum, count)]
        prior_weight = stats['prior_weight']
        
        col = new_data[te_term.var_name].to_numpy()
        n = len(col)
        encoded = np.zeros(n, dtype=self.dtype)
        
        for i, val in enumerate(col):
            val_str = str(val)
            if val_str in level_stats:
                level_sum, level_count = level_stats[val_str]
                # Use full training statistics for prediction
                encoded[i] = (level_sum + prior * prior_weight) / (level_count + prior_weight)
            else:
                # Unknown level - use global prior
                encoded[i] = prior
        
        return encoded
    
    def _build_categorical_level_indicators_new(
        self,
        cat_term: CategoricalTermSpec,
        new_data: "pl.DataFrame",
    ) -> Tuple[np.ndarray, List[str]]:
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


def build_design_matrix(
    formula: str,
    data: "pl.DataFrame",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build design matrix with optimized interaction handling.
    
    This is a drop-in replacement for formulaic's model_matrix that is
    optimized for:
    - Large datasets (uses vectorized operations)
    - High-cardinality categoricals (sparse intermediate representations)
    - Many interaction terms
    
    Parameters
    ----------
    formula : str
        R-style formula
    data : pl.DataFrame
        Polars DataFrame
        
    Returns
    -------
    y : np.ndarray
        Response variable
    X : np.ndarray
        Design matrix
    feature_names : list[str]
        Column names
        
    Example
    -------
    >>> y, X, names = build_design_matrix(
    ...     "claims ~ age*C(region) + C(brand)*C(fuel)",
    ...     data
    ... )
    """
    builder = InteractionBuilder(data)
    return builder.build_design_matrix(formula)
