rs.glm_dict(
    response="ClaimCount",
    terms={
        # === Linear (continuous) ===
        "VehAge": {"type": "linear"},
        
        # === Categorical ===
        "Region": {"type": "categorical"},
        "Region": {"type": "categorical", "levels": ["Paris", "Lyon"]},  # specific levels only
        
        # === B-spline ===
        "DrivAge": {"type": "bs", "df": 5},
        "DrivAge": {"type": "bs", "df": 5, "degree": 2},  # quadratic
        
        # === Natural spline ===
        "Income": {"type": "ns", "df": 4},
        
        # === Monotonic spline ===
        "BonusMalus": {"type": "bs", "df": 4, "monotonicity": "increasing"},                    # increasing (default)
        "Risk": {"type": "bs", "df": 4, "monotonicity": "decreasing"},     # decreasing 
        "Risk": {"type": "ms", "df": 4, "monotonicity": "decreasing"},     # decreasing
        "Risk": {"type": "ms", "df": 4, "monotonicity": "increasing"},    # increasing
        
        # === Target encoding ===
        "Brand": {"type": "target_encoding"},
        "Brand": {"type": "target_encoding", "prior_weight": 5},
        
        # === Identity/Expression ===
        "Age2": {"type": "expression", "expr": "DrivAge**2"},
        "LogIncome": {"type": "expression", "expr": "np.log(Income)"},
        
        # === Constrained coefficients ===
        "Premium": {"type": "linear", "monotonicity": "decreasing"},                          # β ≥ 0
        "Discount": {"type": "linear", "monotonicity": "decreasing"},                         # β ≤ 0
        "Age2Pos": {"type": "expression", "expr": "DrivAge**2", "monotonicity": "increasing"},    # pos(I(x²))
        "DiscNeg": {"type": "expression", "expr": "Discount**2", "monotonicity": "decreasing"},   # neg(I(x²))
    },
    interactions=[
        {
            "DrivAge": {"type": "bs", "df": 5}, 
            "Brand": {"type": "target_encoding"},
            "include_main": True
        },
        {
            "VehAge": {"type": "linear"}, 
            "Region": {"type": "categorical"}, 
            "include_main": False
        }
    ],
    intercept=True,                     # default True
    data=data,
    family="poisson",
    offset="Exposure",
    weights=None,
    seed=42,
)