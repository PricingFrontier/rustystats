"""
Target Encoding (CatBoost-style Ordered Target Statistics)
===========================================================

Implements CatBoost's ordered target statistics for categorical encoding.
This prevents target leakage during training by using only "past" observations
in the permutation order to compute statistics.

Reference: https://arxiv.org/abs/1706.09516 (CatBoost paper)

Key Features
------------
- **Ordered statistics**: For training, each observation is encoded using only
  observations that appear before it in a random permutation order
- **Multiple permutations**: Average across several permutations to reduce variance
- **Regularization**: Prior weight controls smoothing toward global mean
- **No target leakage**: The observation's own target is never used in its encoding

Usage
-----
Direct API:
    >>> import rustystats as rs
    >>> encoded, name, prior, stats = rs.target_encode(categories, target, "var")
    >>> # For prediction on new data:
    >>> new_encoded = rs.apply_target_encoding(new_categories, stats, prior)

Dict API:
    >>> result = rs.glm_dict(response="y", terms={"brand": {"type": "target_encoding"}, "age": {"type": "linear"}}, data=data, family="poisson").fit()
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from . import _rustystats
from .constants import DEFAULT_PRIOR_WEIGHT, DEFAULT_N_PERMUTATIONS
from .exceptions import EncodingError


def target_encode(
    categories: Union[List[str], np.ndarray],
    target: np.ndarray,
    var_name: str = "x",
    prior_weight: float = DEFAULT_PRIOR_WEIGHT,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, str, float, Dict[str, Tuple[float, int]]]:
    """
    Target encode categorical variables using CatBoost-style ordered target statistics.
    
    This encoding prevents target leakage during training by computing statistics
    using only "past" observations in a random permutation order.
    
    Parameters
    ----------
    categories : list[str] or numpy.ndarray
        Categorical values as strings
    target : numpy.ndarray
        Target variable (continuous or binary)
    var_name : str, optional
        Variable name for output column (default: "x")
    prior_weight : float, optional
        Regularization strength toward global mean (default: 1.0).
        Higher values = more regularization for rare categories.
    n_permutations : int, optional
        Number of random permutations to average (default: 4).
        More permutations = lower variance but slower.
    seed : int, optional
        Random seed for reproducibility (default: None = random)
    
    Returns
    -------
    encoded : numpy.ndarray
        Encoded values (shape: n_samples,)
    name : str
        Column name like "TE(var_name)"
    prior : float
        Global prior (mean of target) - needed for prediction
    level_stats : dict
        Mapping of level -> (sum_target, count) for prediction on new data
    
    Examples
    --------
    >>> import rustystats as rs
    >>> import numpy as np
    >>> 
    >>> categories = ["A", "B", "A", "B", "A", "B"]
    >>> target = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    >>> 
    >>> encoded, name, prior, stats = rs.target_encode(categories, target, "cat")
    >>> print(f"Column: {name}, Prior: {prior:.3f}")
    Column: TE(cat), Prior: 0.500
    >>> 
    >>> # For new data:
    >>> new_cats = ["A", "B", "C"]  # C is unseen
    >>> new_encoded = rs.apply_target_encoding(new_cats, stats, prior)
    >>> print(new_encoded)  # C gets the prior
    
    Notes
    -----
    The algorithm:
    1. Shuffle data with random permutation
    2. For each observation i in permutation order:
       encoded[i] = (sum_target_before + prior * prior_weight) / (count_before + prior_weight)
    3. Average across multiple permutations to reduce variance
    
    For prediction on new data, use `apply_target_encoding()` which uses the full
    training statistics (no ordering needed).
    """
    # Convert to list of strings if numpy array
    if isinstance(categories, np.ndarray):
        categories = [str(x) for x in categories]
    else:
        categories = [str(x) for x in categories]
    
    target = np.asarray(target, dtype=np.float64)
    
    return _rustystats.target_encode_py(
        categories, target, var_name, prior_weight, n_permutations, seed
    )


def apply_target_encoding(
    categories: Union[List[str], np.ndarray],
    level_stats: Dict[str, Tuple[float, int]],
    prior: float,
    prior_weight: float = DEFAULT_PRIOR_WEIGHT,
) -> np.ndarray:
    """
    Apply target encoding to new data using pre-computed statistics.
    
    For prediction: uses full training statistics (no ordering needed).
    Unseen categories get the prior (global mean).
    
    Parameters
    ----------
    categories : list[str] or numpy.ndarray
        Categorical values for new data
    level_stats : dict
        Mapping of level -> (sum_target, count) from training.
        Returned by `target_encode()`.
    prior : float
        Global prior (mean of training target).
        Returned by `target_encode()`.
    prior_weight : float, optional
        Prior weight (should match training, default: 1.0)
    
    Returns
    -------
    numpy.ndarray
        Encoded values for new data
    
    Examples
    --------
    >>> # Train
    >>> encoded, name, prior, stats = rs.target_encode(train_cats, train_y, "brand")
    >>> 
    >>> # Predict
    >>> test_encoded = rs.apply_target_encoding(test_cats, stats, prior)
    """
    # Convert to list of strings if numpy array
    if isinstance(categories, np.ndarray):
        categories = [str(x) for x in categories]
    else:
        categories = [str(x) for x in categories]
    
    return _rustystats.apply_target_encoding_py(
        categories, level_stats, prior, prior_weight
    )


class TargetEncoder:
    """
    Scikit-learn style target encoder with CatBoost-style ordered target statistics.
    
    Fits on training data and transforms both training and test data.
    
    Parameters
    ----------
    prior_weight : float, optional
        Regularization strength toward global mean (default: 1.0)
    n_permutations : int, optional
        Number of random permutations to average (default: 4)
    seed : int, optional
        Random seed for reproducibility
    
    Attributes
    ----------
    prior_ : float
        Global prior (mean of training target)
    level_stats_ : dict
        Mapping of level -> (sum_target, count)
    
    Examples
    --------
    >>> encoder = rs.TargetEncoder(prior_weight=1.0, n_permutations=4)
    >>> train_encoded = encoder.fit_transform(train_categories, train_y)
    >>> test_encoded = encoder.transform(test_categories)
    """
    
    def __init__(
        self,
        prior_weight: float = DEFAULT_PRIOR_WEIGHT,
        n_permutations: int = DEFAULT_N_PERMUTATIONS,
        seed: Optional[int] = None,
    ):
        self.prior_weight = prior_weight
        self.n_permutations = n_permutations
        self.seed = seed
        self.prior_: Optional[float] = None
        self.level_stats_: Optional[Dict[str, Tuple[float, int]]] = None
    
    def fit(
        self,
        categories: Union[List[str], np.ndarray],
        target: np.ndarray,
    ) -> "TargetEncoder":
        """
        Fit the encoder on training data.
        
        Parameters
        ----------
        categories : list[str] or numpy.ndarray
            Categorical values
        target : numpy.ndarray
            Target variable
        
        Returns
        -------
        self
        """
        _, _, self.prior_, self.level_stats_ = target_encode(
            categories, target, "x",
            self.prior_weight, self.n_permutations, self.seed
        )
        return self
    
    def transform(
        self,
        categories: Union[List[str], np.ndarray],
    ) -> np.ndarray:
        """
        Transform categories using fitted statistics.
        
        For test/validation data, uses full training statistics.
        
        Parameters
        ----------
        categories : list[str] or numpy.ndarray
            Categorical values
        
        Returns
        -------
        numpy.ndarray
            Encoded values
        """
        if self.level_stats_ is None or self.prior_ is None:
            raise EncodingError("Encoder not fitted. Call fit() first.")
        
        return apply_target_encoding(
            categories, self.level_stats_, self.prior_, self.prior_weight
        )
    
    def fit_transform(
        self,
        categories: Union[List[str], np.ndarray],
        target: np.ndarray,
    ) -> np.ndarray:
        """
        Fit and transform training data using ordered target statistics.
        
        Uses CatBoost-style ordering to prevent target leakage.
        
        Parameters
        ----------
        categories : list[str] or numpy.ndarray
            Categorical values
        target : numpy.ndarray
            Target variable
        
        Returns
        -------
        numpy.ndarray
            Encoded values (with ordered statistics for training)
        """
        encoded, _, self.prior_, self.level_stats_ = target_encode(
            categories, target, "x",
            self.prior_weight, self.n_permutations, self.seed
        )
        return encoded


# =============================================================================
# FREQUENCY ENCODING
# =============================================================================


def frequency_encode(
    categories: Union[List[str], np.ndarray],
    var_name: str = "x",
) -> Tuple[np.ndarray, str, Dict[str, int], int, int]:
    """
    Frequency encode categorical variables.
    
    Encodes categories by their frequency (count / max_count).
    No target variable involved - purely based on category prevalence.
    Useful when category frequency itself is predictive.
    
    Parameters
    ----------
    categories : list[str] or numpy.ndarray
        Categorical values as strings
    var_name : str, optional
        Variable name for output column (default: "x")
    
    Returns
    -------
    encoded : numpy.ndarray
        Encoded values (shape: n_samples,)
    name : str
        Column name like "FE(var_name)"
    level_counts : dict
        Mapping of level -> count for prediction on new data
    max_count : int
        Maximum count (for normalization)
    n_obs : int
        Total number of observations
    
    Examples
    --------
    >>> import rustystats as rs
    >>> import numpy as np
    >>> 
    >>> categories = ["A", "B", "A", "A", "B", "C"]
    >>> encoded, name, counts, max_count, n_obs = rs.frequency_encode(categories, "cat")
    >>> print(f"Column: {name}, Max count: {max_count}")
    Column: FE(cat), Max count: 3
    """
    # Convert to list of strings if numpy array
    if isinstance(categories, np.ndarray):
        categories = [str(x) for x in categories]
    else:
        categories = [str(x) for x in categories]
    
    return _rustystats.frequency_encode_py(categories, var_name)


def apply_frequency_encoding(
    categories: Union[List[str], np.ndarray],
    level_counts: Dict[str, int],
    max_count: int,
) -> np.ndarray:
    """
    Apply frequency encoding to new data using pre-computed statistics.
    
    Unseen categories get 0.0 (zero frequency in training).
    
    Parameters
    ----------
    categories : list[str] or numpy.ndarray
        Categorical values for new data
    level_counts : dict
        Mapping of level -> count from training.
        Returned by `frequency_encode()`.
    max_count : int
        Maximum count from training (for normalization).
        Returned by `frequency_encode()`.
    
    Returns
    -------
    numpy.ndarray
        Encoded values for new data
    
    Examples
    --------
    >>> # Train
    >>> encoded, name, counts, max_count, _ = rs.frequency_encode(train_cats, "brand")
    >>> 
    >>> # Predict
    >>> test_encoded = rs.apply_frequency_encoding(test_cats, counts, max_count)
    """
    # Convert to list of strings if numpy array
    if isinstance(categories, np.ndarray):
        categories = [str(x) for x in categories]
    else:
        categories = [str(x) for x in categories]
    
    return _rustystats.apply_frequency_encoding_py(categories, level_counts, max_count)


class FrequencyEncoder:
    """
    Scikit-learn style frequency encoder.
    
    Encodes categories by their frequency (count / max_count).
    No target variable needed.
    
    Attributes
    ----------
    level_counts_ : dict
        Mapping of level -> count
    max_count_ : int
        Maximum count for normalization
    
    Examples
    --------
    >>> encoder = rs.FrequencyEncoder()
    >>> train_encoded = encoder.fit_transform(train_categories)
    >>> test_encoded = encoder.transform(test_categories)
    """
    
    def __init__(self):
        self.level_counts_: Optional[Dict[str, int]] = None
        self.max_count_: Optional[int] = None
    
    def fit(
        self,
        categories: Union[List[str], np.ndarray],
    ) -> "FrequencyEncoder":
        """Fit the encoder on training data."""
        _, _, self.level_counts_, self.max_count_, _ = frequency_encode(categories, "x")
        return self
    
    def transform(
        self,
        categories: Union[List[str], np.ndarray],
    ) -> np.ndarray:
        """Transform categories using fitted statistics."""
        if self.level_counts_ is None or self.max_count_ is None:
            raise EncodingError("Encoder not fitted. Call fit() first.")
        return apply_frequency_encoding(categories, self.level_counts_, self.max_count_)
    
    def fit_transform(
        self,
        categories: Union[List[str], np.ndarray],
    ) -> np.ndarray:
        """Fit and transform in one step."""
        encoded, _, self.level_counts_, self.max_count_, _ = frequency_encode(categories, "x")
        return encoded


# =============================================================================
# TARGET ENCODING FOR INTERACTIONS
# =============================================================================


def target_encode_interaction(
    cat1: Union[List[str], np.ndarray],
    cat2: Union[List[str], np.ndarray],
    target: np.ndarray,
    var_name1: str = "x1",
    var_name2: str = "x2",
    prior_weight: float = DEFAULT_PRIOR_WEIGHT,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, str, float, Dict[str, Tuple[float, int]]]:
    """
    Target encode a categorical interaction (two variables combined).
    
    Creates combined categories like "brand:region" and applies
    ordered target statistics encoding. Useful for capturing interaction
    effects between high-cardinality categoricals.
    
    Parameters
    ----------
    cat1 : list[str] or numpy.ndarray
        First categorical variable values
    cat2 : list[str] or numpy.ndarray
        Second categorical variable values
    target : numpy.ndarray
        Target variable (continuous or binary)
    var_name1 : str, optional
        Name of first variable (default: "x1")
    var_name2 : str, optional
        Name of second variable (default: "x2")
    prior_weight : float, optional
        Regularization strength toward global mean (default: 1.0)
    n_permutations : int, optional
        Number of random permutations to average (default: 4)
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    encoded : numpy.ndarray
        Encoded values (shape: n_samples,)
    name : str
        Column name like "TE(var1:var2)"
    prior : float
        Global prior (mean of target)
    level_stats : dict
        Mapping of combined level -> (sum_target, count)
    
    Examples
    --------
    >>> import rustystats as rs
    >>> 
    >>> brands = ["Nike", "Adidas", "Nike", "Adidas"]
    >>> regions = ["North", "North", "South", "South"]
    >>> sales = np.array([100, 80, 60, 90])
    >>> 
    >>> encoded, name, prior, stats = rs.target_encode_interaction(
    ...     brands, regions, sales, "brand", "region"
    ... )
    >>> print(f"Column: {name}")
    Column: TE(brand:region)
    """
    # Convert to list of strings
    if isinstance(cat1, np.ndarray):
        cat1 = [str(x) for x in cat1]
    else:
        cat1 = [str(x) for x in cat1]
    
    if isinstance(cat2, np.ndarray):
        cat2 = [str(x) for x in cat2]
    else:
        cat2 = [str(x) for x in cat2]
    
    target = np.asarray(target, dtype=np.float64)
    
    return _rustystats.target_encode_interaction_py(
        cat1, cat2, target, var_name1, var_name2,
        prior_weight, n_permutations, seed
    )


# =============================================================================
# FORMULA TERM CLASSES
# =============================================================================


class TargetEncodingTerm:
    """
    Represents a target encoding term in a formula.
    
    Used internally by the formula parser to handle TE(var) syntax.
    
    Parameters
    ----------
    var_name : str
        Variable name to encode
    prior_weight : float, optional
        Prior weight for regularization
    n_permutations : int, optional
        Number of permutations
    """
    
    def __init__(
        self,
        var_name: str,
        prior_weight: float = DEFAULT_PRIOR_WEIGHT,
        n_permutations: int = DEFAULT_N_PERMUTATIONS,
    ):
        self.var_name = var_name
        self.prior_weight = prior_weight
        self.n_permutations = n_permutations
        self.encoder: Optional[TargetEncoder] = None
    
    def fit_transform(
        self,
        data,  # DataFrame (Polars or Pandas)
        target: np.ndarray,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, str]:
        """
        Fit and transform the column from a DataFrame.
        
        Returns
        -------
        values : numpy.ndarray
            Encoded values
        name : str
            Column name
        """
        # Extract column
        if hasattr(data, 'to_numpy'):
            # Polars DataFrame
            col = data[self.var_name].to_numpy()
        else:
            # Pandas DataFrame
            col = data[self.var_name].values
        
        categories = [str(x) for x in col]
        
        self.encoder = TargetEncoder(
            prior_weight=self.prior_weight,
            n_permutations=self.n_permutations,
            seed=seed,
        )
        encoded = self.encoder.fit_transform(categories, target)
        
        return encoded, f"TE({self.var_name})"
    
    def transform(self, data) -> Tuple[np.ndarray, str]:
        """
        Transform new data using fitted encoder.
        
        Returns
        -------
        values : numpy.ndarray
            Encoded values
        name : str
            Column name
        """
        if self.encoder is None:
            raise EncodingError("Term not fitted. Call fit_transform() first.")
        
        # Extract column
        if hasattr(data, 'to_numpy'):
            col = data[self.var_name].to_numpy()
        else:
            col = data[self.var_name].values
        
        categories = [str(x) for x in col]
        encoded = self.encoder.transform(categories)
        
        return encoded, f"TE({self.var_name})"
