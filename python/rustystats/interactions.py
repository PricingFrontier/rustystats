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
class TargetEncodingTermSpec:
    """Parsed target encoding term specification from formula."""
    var_name: str
    prior_weight: float = 1.0
    n_permutations: int = 4


@dataclass 
class ParsedFormula:
    """Parsed formula with identified terms."""
    
    response: str
    main_effects: List[str]  # Main effect variables
    interactions: List[InteractionTerm]  # Interaction terms
    categorical_vars: Set[str]  # Variables marked as categorical with C()
    spline_terms: List[SplineTerm] = field(default_factory=list)  # Spline terms
    target_encoding_terms: List[TargetEncodingTermSpec] = field(default_factory=list)  # TE() terms
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
    
    spline_terms = [
        SplineTerm(
            var_name=s['var_name'],
            spline_type=s['spline_type'],
            df=s['df'],
            degree=s['degree']
        )
        for s in parsed['spline_terms']
    ]
    
    # Parse target encoding terms
    target_encoding_terms = [
        TargetEncodingTermSpec(
            var_name=t['var_name'],
            prior_weight=t['prior_weight'],
            n_permutations=t['n_permutations']
        )
        for t in parsed.get('target_encoding_terms', [])
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
        
        # Cache for encoded categorical variables
        self._cat_cache: Dict[str, Tuple[np.ndarray, List[str]]] = {}
        # Cache for categorical indices (for interaction building)
        self._cat_indices_cache: Dict[str, Tuple[np.ndarray, List[str]]] = {}
        # Store categorical levels for prediction on new data
        self._cat_levels: Dict[str, List[str]] = {}
        # Store spline terms with fitted knots for prediction
        self._fitted_splines: Dict[str, SplineTerm] = {}
        # Store parsed formula for prediction
        self._parsed_formula: Optional[ParsedFormula] = None
    
    def _get_column(self, name: str) -> np.ndarray:
        """Extract column as numpy array."""
        return self.data[name].to_numpy().astype(self.dtype)
    
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
        if cache_key in self._cat_cache:
            return self._cat_cache[cache_key]
        
        col = self.data[name].to_numpy()
        
        # Convert to string list for Rust factorization
        values = [str(v) for v in col]
        
        # Use Rust for factorization + matrix construction
        encoding, names, indices, levels = _encode_categorical_rust(values, name, drop_first)
        
        # Cache both the encoding and the indices/levels for interaction building
        self._cat_cache[cache_key] = (encoding, names)
        self._cat_indices_cache[name] = (np.array(indices, dtype=np.int32), levels)
        # Store levels for prediction on new data
        self._cat_levels[name] = levels
        
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
        # Get original indices (from cache or compute via encoding)
        cat1, cat2 = factors
        
        # Ensure we have indices cached (this will populate cache if not already)
        if cat1 not in self._cat_indices_cache:
            self._get_categorical_encoding(cat1)
        if cat2 not in self._cat_indices_cache:
            self._get_categorical_encoding(cat2)
        
        idx1, levels1 = self._cat_indices_cache[cat1]
        idx2, levels2 = self._cat_indices_cache[cat2]
        
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
        
        for factor, is_cat in zip(interaction.factors, interaction.categorical_flags):
            if is_cat:
                cat_factors.append(factor)
            else:
                cont_factors.append(factor)
        
        # Build continuous part (product of all continuous)
        cont_product = self._get_column(cont_factors[0])
        for factor in cont_factors[1:]:
            cont_product = cont_product * self._get_column(factor)
        cont_name = ':'.join(cont_factors)
        
        # Build categorical part and use Rust for interaction
        if len(cat_factors) == 1:
            # Single categorical - use Rust directly
            cat_name = cat_factors[0]
            
            # Ensure indices are cached
            if cat_name not in self._cat_indices_cache:
                self._get_categorical_encoding(cat_name)
            
            cat_indices, levels = self._cat_indices_cache[cat_name]
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
    ) -> Tuple[np.ndarray, str, dict]:
        """
        Build target-encoded column for a categorical variable.
        
        Uses CatBoost-style ordered target statistics to prevent target leakage.
        
        Parameters
        ----------
        te_term : TargetEncodingTermSpec
            Target encoding term specification
        target : np.ndarray
            Target variable values
        seed : int, optional
            Random seed for reproducibility
            
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
        
        encoded, name, prior, stats = _target_encode_rust(
            categories,
            target.astype(np.float64),
            te_term.var_name,
            te_term.prior_weight,
            te_term.n_permutations,
            seed,
        )
        
        return encoded, name, {'prior': prior, 'stats': stats, 'prior_weight': te_term.prior_weight}
    
    def build_design_matrix(
        self,
        formula: str,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build complete design matrix from formula.
        
        Parameters
        ----------
        formula : str
            R-style formula like "y ~ x1*x2 + C(cat) + bs(age, df=5)"
            
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
        
        # Add spline terms
        for spline in parsed.spline_terms:
            spline_cols, spline_names = self._build_spline_columns(spline)
            columns.append(spline_cols)
            names.extend(spline_names)
            # Store fitted spline for prediction
            self._fitted_splines[spline.var_name] = spline
        
        # Add interactions
        for interaction in parsed.interactions:
            int_cols, int_names = self.build_interaction_columns(interaction)
            if int_cols.ndim == 1:
                int_cols = int_cols.reshape(-1, 1)
            columns.append(int_cols)
            names.extend(int_names)
        
        # Store parsed formula for prediction
        self._parsed_formula = parsed
        
        # Get response (needed for target encoding)
        y = self._get_column(parsed.response)
        
        # Add target encoding terms (CatBoost-style)
        # Store stats for prediction on new data
        self._te_stats: Dict[str, dict] = {}
        for te_term in parsed.target_encoding_terms:
            te_col, te_name, te_stats = self._build_target_encoding_columns(te_term, y)
            columns.append(te_col.reshape(-1, 1))
            names.append(te_name)
            self._te_stats[te_term.var_name] = te_stats
        
        # Stack all columns
        if columns:
            X = np.hstack([c if c.ndim == 2 else c.reshape(-1, 1) for c in columns])
        else:
            X = np.ones((self._n, 1), dtype=self.dtype)
            names = ['Intercept']
        
        return y, X, names
    
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
        
        # Add interactions
        for interaction in parsed.interactions:
            int_cols = self._build_interaction_new(new_data, interaction, n_new)
            if int_cols.ndim == 1:
                int_cols = int_cols.reshape(-1, 1)
            columns.append(int_cols)
        
        # Add target encoding terms using stored statistics
        for te_term in parsed.target_encoding_terms:
            te_col = self._encode_target_new(new_data, te_term)
            columns.append(te_col.reshape(-1, 1))
        
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
        if var_name not in self._cat_levels:
            raise ValueError(
                f"Categorical variable '{var_name}' was not seen during training."
            )
        
        levels = self._cat_levels[var_name]
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
            # Mixed: categorical × continuous
            cat_factors = []
            cont_factors = []
            for factor, is_cat in zip(interaction.factors, interaction.categorical_flags):
                if is_cat:
                    cat_factors.append(factor)
                else:
                    cont_factors.append(factor)
            
            # Build continuous product
            cont_product = new_data[cont_factors[0]].to_numpy().astype(self.dtype)
            for factor in cont_factors[1:]:
                cont_product = cont_product * new_data[factor].to_numpy().astype(self.dtype)
            
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
