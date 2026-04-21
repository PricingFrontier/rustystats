"""Examples of dictionary-style model configuration.

This file is intentionally written as importable example code. Python dictionaries
cannot contain the same key more than once, so alternative configurations for the
same source column are shown in separate dictionaries.
"""

import rustystats as rs

BASE_TERMS = {
    # Linear continuous term.
    "VehAge": {
        "type": "linear",
    },
    # Categorical term, optionally restricted to known levels.
    "Region": {
        "type": "categorical",
        "levels": ["Paris", "Lyon"],
    },
    # Spline terms.
    "DrivAge": {
        "type": "bs",
        "df": 5,
        "degree": 2,
    },
    "Income": {
        "type": "ns",
        "df": 4,
    },
    # Monotonic terms.
    "BonusMalus": {
        "type": "bs",
        "df": 4,
        "monotonicity": "increasing",
    },
    "Risk": {
        "type": "ms",
        "df": 4,
        "monotonicity": "decreasing",
    },
    # Target encoding.
    "Brand": {
        "type": "target_encoding",
        "prior_weight": 5,
    },
    # Expressions and constrained coefficients.
    "Age2": {
        "type": "expression",
        "expr": "DrivAge**2",
    },
    "LogIncome": {
        "type": "expression",
        "expr": "np.log(Income)",
    },
    "Premium": {
        "type": "linear",
        "monotonicity": "increasing",
    },
    "Discount": {
        "type": "linear",
        "monotonicity": "decreasing",
    },
    "Age2Pos": {
        "type": "expression",
        "expr": "DrivAge**2",
        "monotonicity": "increasing",
    },
    "DiscNeg": {
        "type": "expression",
        "expr": "Discount**2",
        "monotonicity": "decreasing",
    },
}


ALTERNATIVE_TERM_OPTIONS = {
    "region_all_levels": {
        "Region": {
            "type": "categorical",
        },
    },
    "driver_age_default_bs": {
        "DrivAge": {
            "type": "bs",
            "df": 5,
        },
    },
    "risk_bs_decreasing": {
        "Risk": {
            "type": "bs",
            "df": 4,
            "monotonicity": "decreasing",
        },
    },
    "risk_ms_increasing": {
        "Risk": {
            "type": "ms",
            "df": 4,
            "monotonicity": "increasing",
        },
    },
    "brand_default_te": {
        "Brand": {
            "type": "target_encoding",
        },
    },
}


INTERACTIONS = [
    # Standard interaction: spline x target encoding product terms.
    {
        "DrivAge": {
            "type": "bs",
            "df": 5,
        },
        "Brand": {
            "type": "target_encoding",
        },
        "include_main": True,
    },
    # Standard interaction: continuous x categorical.
    {
        "VehAge": {
            "type": "linear",
        },
        "Region": {
            "type": "categorical",
        },
        "include_main": False,
    },
    # Combined target encoding for Brand:Region.
    {
        "Brand": {
            "type": "categorical",
        },
        "Region": {
            "type": "categorical",
        },
        "target_encoding": True,
        "prior_weight": 1.0,
    },
    # Combined frequency encoding for Brand:Region.
    {
        "Brand": {
            "type": "categorical",
        },
        "Region": {
            "type": "categorical",
        },
        "frequency_encoding": True,
    },
]


def build_frequency_model(data):
    """Build a Poisson frequency model from a Polars DataFrame."""
    return rs.glm_dict(
        response="ClaimCount",
        terms=BASE_TERMS,
        interactions=INTERACTIONS,
        intercept=True,
        data=data,
        family="poisson",
        offset="Exposure",
        weights=None,
        seed=42,
    )
