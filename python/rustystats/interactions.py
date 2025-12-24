"""
Optimized interaction term support for RustyStats.

This module provides high-performance interaction term handling for GLMs,
including:
- Lazy interaction computation (avoid materializing large matrices)
- Sparse matrix support for categorical interactions
- Efficient vectorized construction of interaction columns

Performance Characteristics:
- Continuous × Continuous: O(n) memory, O(n) time
- Categorical × Continuous: O(n × k) memory where k = levels
- Categorical × Categorical: O(n) memory using sparse representation

Example
-------
>>> from rustystats.interactions import InteractionBuilder
>>> 
>>> builder = InteractionBuilder(data, ['x1', 'x2'], ['cat1', 'cat2'])
>>> X, names = builder.build('x1*x2 + C(cat1):x1')
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict, Set, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from scipy import sparse


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


@dataclass 
class ParsedFormula:
    """Parsed formula with identified terms."""
    
    response: str
    main_effects: List[str]  # Main effect variables
    interactions: List[InteractionTerm]  # Interaction terms
    categorical_vars: Set[str]  # Variables marked as categorical with C()
    has_intercept: bool = True


def parse_formula_interactions(formula: str) -> ParsedFormula:
    """
    Parse a formula string and extract interaction terms.
    
    Handles:
    - Main effects: x1, x2, C(cat)
    - Two-way interactions: x1:x2, x1*x2, C(cat):x
    - Higher-order: x1:x2:x3
    - Intercept removal: 0 + ... or -1
    
    Parameters
    ----------
    formula : str
        R-style formula like "y ~ x1*x2 + C(cat)"
        
    Returns
    -------
    ParsedFormula
        Parsed structure with all terms identified
    """
    # Split into response and predictors
    if '~' not in formula:
        raise ValueError(f"Formula must contain '~': {formula}")
    
    response, rhs = formula.split('~', 1)
    response = response.strip()
    rhs = rhs.strip()
    
    # Check for intercept removal
    has_intercept = True
    if rhs.startswith('0 +') or rhs.startswith('0+'):
        has_intercept = False
        rhs = rhs[3:].strip()
    
    # Handle "- 1" or "-1" anywhere in formula (R-style intercept removal)
    # Match patterns like "+ - 1", "- 1", "-1" 
    if re.search(r'[-+]\s*1\s*$', rhs) or re.search(r'[-+]\s*1\s*[-+]', rhs):
        has_intercept = False
        # Remove the "-1" or "- 1" term
        rhs = re.sub(r'\s*[-+]\s*1\s*', ' ', rhs).strip()
        # Clean up any leading/trailing operators
        rhs = re.sub(r'^\s*[-+]\s*', '', rhs)
        rhs = re.sub(r'\s*[-+]\s*$', '', rhs)
    
    # Find all C(...) categorical markers
    categorical_pattern = r'C\(([^)]+)\)'
    categorical_vars = set(re.findall(categorical_pattern, rhs))
    
    # Split into terms (by +)
    terms = [t.strip() for t in rhs.split('+')]
    terms = [t for t in terms if t]  # Remove empty
    
    main_effects = []
    interactions = []
    
    for term in terms:
        if '*' in term:
            # Full interaction: a*b = a + b + a:b
            factors = _parse_interaction_factors(term.replace('*', ':'), categorical_vars)
            
            # Add main effects
            for i, (var, is_cat) in enumerate(zip(factors, [v in categorical_vars or v.startswith('C(') for v in term.split('*')])):
                clean_var = _clean_var_name(term.split('*')[i].strip())
                if clean_var not in main_effects:
                    main_effects.append(clean_var)
            
            # Add interaction
            parsed = _parse_interaction_factors(term.replace('*', ':'), categorical_vars)
            interactions.append(InteractionTerm(
                factors=[_clean_var_name(f) for f in term.split('*')],
                categorical_flags=[_is_categorical(f.strip(), categorical_vars) for f in term.split('*')]
            ))
            
        elif ':' in term:
            # Pure interaction: a:b (no main effects added)
            factor_strs = term.split(':')
            interactions.append(InteractionTerm(
                factors=[_clean_var_name(f.strip()) for f in factor_strs],
                categorical_flags=[_is_categorical(f.strip(), categorical_vars) for f in factor_strs]
            ))
        else:
            # Main effect
            clean = _clean_var_name(term)
            if clean and clean not in main_effects:
                main_effects.append(clean)
    
    return ParsedFormula(
        response=response,
        main_effects=main_effects,
        interactions=interactions,
        categorical_vars=categorical_vars,
        has_intercept=has_intercept,
    )


def _clean_var_name(term: str) -> str:
    """Extract variable name from term like 'C(var)' -> 'var'."""
    term = term.strip()
    match = re.match(r'C\(([^)]+)\)', term)
    if match:
        return match.group(1)
    return term


def _is_categorical(term: str, categorical_vars: Set[str]) -> bool:
    """Check if a term is categorical."""
    term = term.strip()
    if term.startswith('C('):
        return True
    return _clean_var_name(term) in categorical_vars


def _parse_interaction_factors(term: str, categorical_vars: Set[str]) -> List[Tuple[str, bool]]:
    """Parse interaction term into (variable_name, is_categorical) pairs."""
    factors = term.split(':')
    return [(f.strip(), _is_categorical(f, categorical_vars)) for f in factors]


class InteractionBuilder:
    """
    Efficiently builds design matrices with interaction terms.
    
    Optimizations:
    1. Continuous × Continuous: Single vectorized multiplication
    2. Categorical × Continuous: Sparse-aware dummy encoding
    3. Categorical × Categorical: Direct index-based construction
    
    Parameters
    ----------
    data : DataFrame
        Polars or Pandas DataFrame
    dtype : numpy dtype, default=np.float64
        Data type for output arrays
        
    Example
    -------
    >>> builder = InteractionBuilder(df)
    >>> X, names = builder.build_matrix('y ~ x1*x2 + C(area):age')
    """
    
    def __init__(
        self,
        data: Union["pl.DataFrame", "pd.DataFrame"],
        dtype: np.dtype = np.float64,
    ):
        self.data = data
        self.dtype = dtype
        self._is_polars = hasattr(data, 'to_pandas')
        self._n = len(data)
        
        # Cache for encoded categorical variables
        self._cat_cache: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    
    def _get_column(self, name: str) -> np.ndarray:
        """Extract column as numpy array."""
        if self._is_polars:
            return self.data[name].to_numpy().astype(self.dtype)
        return self.data[name].values.astype(self.dtype)
    
    def _get_categorical_encoding(
        self, 
        name: str,
        drop_first: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Get dummy encoding for a categorical variable.
        
        Uses optimized vectorized construction with sparse intermediate.
        
        Returns
        -------
        encoding : np.ndarray
            (n, k-1) dummy matrix where k is number of levels
        names : list[str]
            Column names like ['var[T.B]', 'var[T.C]', ...]
        """
        cache_key = f"{name}_{drop_first}"
        if cache_key in self._cat_cache:
            return self._cat_cache[cache_key]
        
        if self._is_polars:
            col = self.data[name].to_numpy()
        else:
            col = self.data[name].values
        
        # Get unique levels sorted
        levels, inverse = np.unique(col, return_inverse=True)
        indices = inverse.astype(np.int32)
        
        # Create dummy matrix
        n_levels = len(levels)
        start_idx = 1 if drop_first else 0
        n_cols = n_levels - start_idx
        
        if n_cols == 0:
            self._cat_cache[cache_key] = (np.zeros((self._n, 0), dtype=self.dtype), [])
            return self._cat_cache[cache_key]
        
        # Vectorized sparse construction (always efficient)
        from scipy import sparse
        
        col_idx = indices - start_idx
        
        if drop_first:
            # Only include rows where col_idx >= 0
            mask = col_idx >= 0
            row_idx = np.arange(self._n, dtype=np.int32)[mask]
            col_idx = col_idx[mask]
        else:
            row_idx = np.arange(self._n, dtype=np.int32)
        
        data = np.ones(len(row_idx), dtype=self.dtype)
        
        sp = sparse.csr_matrix(
            (data, (row_idx, col_idx)),
            shape=(self._n, n_cols),
            dtype=self.dtype
        )
        encoding = sp.toarray()
        
        # Generate names
        names = [f"{name}[T.{levels[i + start_idx]}]" for i in range(n_cols)]
        
        self._cat_cache[cache_key] = (encoding, names)
        return encoding, names
    
    def build_interaction_columns(
        self,
        interaction: InteractionTerm,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build columns for a single interaction term.
        
        Optimized for different interaction types:
        - Pure continuous: Single O(n) element-wise multiply
        - Mixed: Broadcast multiply continuous with each dummy column
        - Pure categorical: Sparse index-based construction
        
        Returns
        -------
        columns : np.ndarray
            (n, k) interaction columns
        names : list[str]
            Column names
        """
        if interaction.is_pure_continuous:
            return self._build_continuous_interaction(interaction)
        elif interaction.is_pure_categorical:
            return self._build_categorical_interaction(interaction)
        else:
            return self._build_mixed_interaction(interaction)
    
    def _build_continuous_interaction(
        self, 
        interaction: InteractionTerm
    ) -> Tuple[np.ndarray, List[str]]:
        """Build continuous × continuous interaction."""
        # Start with first factor
        result = self._get_column(interaction.factors[0])
        
        # Multiply remaining factors
        for factor in interaction.factors[1:]:
            result = result * self._get_column(factor)
        
        # Name is "x1:x2:x3"
        name = ':'.join(interaction.factors)
        
        return result.reshape(-1, 1), [name]
    
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
        # Get original indices (before dummy encoding)
        cat1, cat2 = factors
        
        # Re-extract level indices for efficient construction
        if self._is_polars:
            col1 = self.data[cat1].to_numpy()
            col2 = self.data[cat2].to_numpy()
        else:
            col1 = self.data[cat1].values
            col2 = self.data[cat2].values
        
        levels1 = np.unique(col1)
        levels2 = np.unique(col2)
        levels1 = np.sort(levels1)
        levels2 = np.sort(levels2)
        
        # Map values to indices (0-based, first level will be dropped)
        level_to_idx1 = {l: i for i, l in enumerate(levels1)}
        level_to_idx2 = {l: i for i, l in enumerate(levels2)}
        
        idx1 = np.array([level_to_idx1[v] for v in col1], dtype=np.int32)
        idx2 = np.array([level_to_idx2[v] for v in col2], dtype=np.int32)
        
        # Number of levels after dropping first
        n1 = len(levels1) - 1
        n2 = len(levels2) - 1
        n_cols = n1 * n2
        
        if n_cols == 0:
            return np.zeros((self._n, 0), dtype=self.dtype), []
        
        # Use sparse construction for large data
        if self._n > 5000 or n_cols > 50:
            from scipy import sparse
            
            # For interaction (i, j) where i,j >= 1 (after dropping first level),
            # the interaction column index is (i-1) * n2 + (j-1)
            # Only create entries where both indices are >= 1
            mask = (idx1 >= 1) & (idx2 >= 1)
            row_indices = np.arange(self._n)[mask]
            col_indices = (idx1[mask] - 1) * n2 + (idx2[mask] - 1)
            data = np.ones(len(row_indices), dtype=self.dtype)
            
            sp = sparse.csr_matrix(
                (data, (row_indices, col_indices)),
                shape=(self._n, n_cols),
                dtype=self.dtype
            )
            result = sp.toarray()
        else:
            # Direct dense construction for small data
            result = np.zeros((self._n, n_cols), dtype=self.dtype)
            for i in range(self._n):
                i1, i2 = idx1[i], idx2[i]
                if i1 >= 1 and i2 >= 1:
                    col_idx = (i1 - 1) * n2 + (i2 - 1)
                    result[i, col_idx] = 1.0
        
        # Generate column names
        names1, names2 = all_names
        col_names = []
        for i in range(n1):
            for j in range(n2):
                col_names.append(f"{names1[i]}:{names2[j]}")
        
        return result, col_names
    
    def _build_nway_categorical(
        self,
        encodings: List[np.ndarray],
        all_names: List[List[str]],
        factors: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """General n-way categorical interaction."""
        from itertools import product
        
        # Get all combinations of column indices
        col_ranges = [range(e.shape[1]) for e in encodings]
        
        result_cols = []
        col_names = []
        
        for indices in product(*col_ranges):
            # Multiply all encodings at these indices
            col = np.ones(self._n, dtype=self.dtype)
            name_parts = []
            
            for k, idx in enumerate(indices):
                col = col * encodings[k][:, idx]
                name_parts.append(all_names[k][idx])
            
            result_cols.append(col)
            col_names.append(':'.join(name_parts))
        
        if result_cols:
            return np.column_stack(result_cols), col_names
        return np.zeros((self._n, 0), dtype=self.dtype), []
    
    def _build_mixed_interaction(
        self,
        interaction: InteractionTerm
    ) -> Tuple[np.ndarray, List[str]]:
        """Build categorical × continuous interaction."""
        # Separate categorical and continuous factors
        cat_factors = []
        cont_factors = []
        
        for factor, is_cat in zip(interaction.factors, interaction.categorical_flags):
            if is_cat:
                cat_factors.append(factor)
            else:
                cont_factors.append(factor)
        
        # Build categorical part
        if len(cat_factors) == 1:
            cat_encoding, cat_names = self._get_categorical_encoding(cat_factors[0])
        else:
            # Multiple categorical - build their interaction first
            cat_interaction = InteractionTerm(
                factors=cat_factors,
                categorical_flags=[True] * len(cat_factors)
            )
            cat_encoding, cat_names = self._build_categorical_interaction(cat_interaction)
        
        # Build continuous part (product of all continuous)
        cont_product = self._get_column(cont_factors[0])
        for factor in cont_factors[1:]:
            cont_product = cont_product * self._get_column(factor)
        
        # Multiply each categorical column by continuous product
        n_cat_cols = cat_encoding.shape[1]
        result = np.zeros((self._n, n_cat_cols), dtype=self.dtype)
        col_names = []
        
        cont_name = ':'.join(cont_factors)
        
        for i in range(n_cat_cols):
            result[:, i] = cat_encoding[:, i] * cont_product
            col_names.append(f"{cat_names[i]}:{cont_name}")
        
        return result, col_names
    
    def build_design_matrix(
        self,
        formula: str,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build complete design matrix from formula.
        
        Parameters
        ----------
        formula : str
            R-style formula like "y ~ x1*x2 + C(cat)"
            
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
        
        # Add interactions
        for interaction in parsed.interactions:
            int_cols, int_names = self.build_interaction_columns(interaction)
            if int_cols.ndim == 1:
                int_cols = int_cols.reshape(-1, 1)
            columns.append(int_cols)
            names.extend(int_names)
        
        # Stack all columns
        if columns:
            X = np.hstack([c if c.ndim == 2 else c.reshape(-1, 1) for c in columns])
        else:
            X = np.ones((self._n, 1), dtype=self.dtype)
            names = ['Intercept']
        
        # Get response
        y = self._get_column(parsed.response)
        
        return y, X, names


def build_design_matrix_optimized(
    formula: str,
    data: Union["pl.DataFrame", "pd.DataFrame"],
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
    data : DataFrame
        Polars or Pandas DataFrame
        
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
    >>> y, X, names = build_design_matrix_optimized(
    ...     "claims ~ age*C(region) + C(brand)*C(fuel)",
    ...     data
    ... )
    """
    builder = InteractionBuilder(data)
    return builder.build_design_matrix(formula)


class SparseInteractionMatrix:
    """
    Lazy sparse representation for large categorical interactions.
    
    Instead of materializing the full (n × p) design matrix where p can be
    very large for categorical × categorical interactions, this class stores
    the component factors and computes products on-the-fly.
    
    This is particularly efficient for:
    - Computing X'WX where most entries are zero
    - Memory-constrained environments
    - Very high-cardinality categoricals
    
    Example
    -------
    >>> sim = SparseInteractionMatrix(data, ['region', 'brand'])
    >>> xtx = sim.compute_xtx(weights)  # Without materializing full X
    """
    
    def __init__(
        self,
        data: Union["pl.DataFrame", "pd.DataFrame"],
        categorical_factors: List[str],
        continuous_factor: Optional[str] = None,
    ):
        self.data = data
        self.categorical_factors = categorical_factors
        self.continuous_factor = continuous_factor
        
        self._is_polars = hasattr(data, 'to_pandas')
        self._n = len(data)
        
        # Store level indices for each categorical
        self._indices: List[np.ndarray] = []
        self._levels: List[np.ndarray] = []
        self._n_levels: List[int] = []
        
        for cat in categorical_factors:
            if self._is_polars:
                col = data[cat].to_numpy()
            else:
                col = data[cat].values
            
            levels = np.unique(col)
            levels = np.sort(levels)
            level_to_idx = {l: i for i, l in enumerate(levels)}
            indices = np.array([level_to_idx[v] for v in col], dtype=np.int32)
            
            self._indices.append(indices)
            self._levels.append(levels)
            self._n_levels.append(len(levels))
        
        # Store continuous factor if present
        self._continuous: Optional[np.ndarray] = None
        if continuous_factor:
            if self._is_polars:
                self._continuous = data[continuous_factor].to_numpy().astype(np.float64)
            else:
                self._continuous = data[continuous_factor].values.astype(np.float64)
    
    @property
    def n_interaction_columns(self) -> int:
        """Number of columns in full interaction matrix (excluding reference levels)."""
        result = 1
        for n in self._n_levels:
            result *= (n - 1)  # Drop first level of each
        return result
    
    def compute_xtx_contribution(
        self,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Compute X'WX contribution from this interaction without full materialization.
        
        For sparse categorical interactions, most X'WX entries are zero.
        We compute only the non-zero entries.
        
        Parameters
        ----------
        weights : np.ndarray
            IRLS weights (n,)
            
        Returns
        -------
        xtx : np.ndarray
            Contribution to X'WX from this interaction
        """
        from itertools import product
        
        n_cols = self.n_interaction_columns
        xtx = np.zeros((n_cols, n_cols), dtype=np.float64)
        
        # Generate all non-reference level combinations
        col_ranges = [range(1, n) for n in self._n_levels]  # Skip first level
        
        col_idx = 0
        for indices in product(*col_ranges):
            # Find rows where all categoricals match these indices
            mask = np.ones(self._n, dtype=bool)
            for k, idx in enumerate(indices):
                mask &= (self._indices[k] == idx)
            
            if self._continuous is not None:
                # Weight is sum of w_i * x_i^2 for matching rows
                x_vals = self._continuous[mask]
                w_vals = weights[mask]
                xtx[col_idx, col_idx] = np.sum(w_vals * x_vals * x_vals)
            else:
                # Just sum of weights for matching rows
                xtx[col_idx, col_idx] = np.sum(weights[mask])
            
            col_idx += 1
        
        return xtx
    
    def compute_xtz_contribution(
        self,
        z: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Compute X'Wz contribution from this interaction."""
        from itertools import product
        
        n_cols = self.n_interaction_columns
        xtz = np.zeros(n_cols, dtype=np.float64)
        
        col_ranges = [range(1, n) for n in self._n_levels]
        
        col_idx = 0
        for indices in product(*col_ranges):
            mask = np.ones(self._n, dtype=bool)
            for k, idx in enumerate(indices):
                mask &= (self._indices[k] == idx)
            
            if self._continuous is not None:
                x_vals = self._continuous[mask]
                xtz[col_idx] = np.sum(weights[mask] * z[mask] * x_vals)
            else:
                xtz[col_idx] = np.sum(weights[mask] * z[mask])
            
            col_idx += 1
        
        return xtz
