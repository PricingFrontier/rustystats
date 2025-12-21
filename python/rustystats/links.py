"""
Link Functions for GLMs
=======================

Link functions connect the linear predictor (η = Xβ) to the mean (μ).
They're written as:

    η = g(μ)    or equivalently    μ = g⁻¹(η)

Why Do We Need Link Functions?
------------------------------

Different types of responses need different transformations:

1. **Continuous data**: Use identity link (η = μ)
   - No transformation needed
   - Predictions can be any real value

2. **Count data**: Use log link (η = log(μ))
   - Ensures predictions are always positive (μ = exp(η) > 0)
   - Gives multiplicative interpretation to coefficients

3. **Binary data**: Use logit link (η = log(μ/(1-μ)))
   - Ensures predictions are probabilities (0 < μ < 1)
   - Coefficients are log-odds ratios

Choosing a Link Function
------------------------

+------------------+-------------------+-------------------+--------------------+
| Family           | Canonical Link    | Common Alternative| Interpretation     |
+==================+===================+===================+====================+
| Gaussian         | Identity          | Log               | Additive effects   |
| Poisson          | Log               | -                 | Multiplicative     |
| Binomial         | Logit             | Probit, Cloglog   | Odds ratios        |
| Gamma            | Inverse (1/μ)     | Log               | Multiplicative     |
+------------------+-------------------+-------------------+--------------------+

In actuarial practice, the **log link** is extremely common because:
- It ensures positive predictions (important for counts and amounts)
- Coefficients have multiplicative interpretation (rate relativities!)
- It's consistent across frequency and severity models

Examples
--------
>>> import rustystats as rs
>>> import numpy as np
>>>
>>> # Log link example
>>> log_link = rs.links.Log()
>>> eta = np.array([0.0, 0.5, 1.0])  # Linear predictor values
>>> mu = log_link.inverse(eta)
>>> print(mu)  # [1.0, 1.649, 2.718] - always positive!
>>>
>>> # Logit link example  
>>> logit_link = rs.links.Logit()
>>> eta = np.array([-2.0, 0.0, 2.0])
>>> mu = logit_link.inverse(eta)
>>> print(mu)  # [0.119, 0.5, 0.881] - always between 0 and 1!
"""

# Import the Rust implementations
from rustystats._rustystats import (
    IdentityLink as _IdentityLink,
    LogLink as _LogLink,
    LogitLink as _LogitLink,
)


def Identity():
    """
    Identity link function: η = μ
    
    The simplest link - no transformation at all.
    
    Properties
    ----------
    - Link: η = μ
    - Inverse: μ = η  
    - Derivative: dη/dμ = 1
    
    When to Use
    -----------
    - Gaussian family (standard linear regression)
    - When you want to model the mean directly
    - When predictions can be any real value
    
    Interpretation
    --------------
    Coefficients have an additive interpretation:
    
    - If β = 10 for variable X
    - Then a 1-unit increase in X increases the predicted mean by 10
    
    Example
    -------
    >>> link = rs.links.Identity()
    >>> mu = np.array([1.0, 2.0, 3.0])
    >>> eta = link.link(mu)  # [1.0, 2.0, 3.0] - unchanged
    """
    return _IdentityLink()


def Log():
    """
    Log link function: η = log(μ)
    
    Ensures predictions are always positive. The workhorse of actuarial GLMs.
    
    Properties
    ----------
    - Link: η = log(μ)
    - Inverse: μ = exp(η)
    - Derivative: dη/dμ = 1/μ
    
    When to Use
    -----------
    - Poisson family (claim frequency)
    - Gamma family (claim severity)
    - Whenever the response must be positive
    
    Multiplicative Interpretation (Important!)
    ------------------------------------------
    Coefficients represent MULTIPLICATIVE effects (rate relativities):
    
    - If β = 0.2 for "young driver" indicator
    - Then exp(0.2) ≈ 1.22
    - Young drivers have 1.22× the expected count/amount (22% higher)
    
    This is why log link is standard in insurance pricing:
    - Base rate × relativity_1 × relativity_2 × ...
    - On log scale: log(base) + β₁ + β₂ + ...
    
    Combining Frequency and Severity
    --------------------------------
    If both models use log link:
    - Frequency: log(μ_freq) = X β_freq
    - Severity: log(μ_sev) = X β_sev
    - Pure Premium: log(μ_freq × μ_sev) = X (β_freq + β_sev)
    
    The pure premium coefficients are just the SUM!
    
    Example
    -------
    >>> link = rs.links.Log()
    >>> 
    >>> # If linear predictor η = 0, predicted count/amount = exp(0) = 1
    >>> # If η increases by 0.1, prediction multiplied by exp(0.1) ≈ 1.105
    >>> 
    >>> eta = np.array([0.0, 0.1, 0.2])
    >>> mu = link.inverse(eta)
    >>> print(mu)  # [1.0, 1.105, 1.221]
    """
    return _LogLink()


def Logit():
    """
    Logit link function: η = log(μ/(1-μ))
    
    Transforms probabilities to the log-odds scale.
    The foundation of logistic regression.
    
    Properties
    ----------
    - Link: η = log(μ/(1-μ)) [the "log-odds"]
    - Inverse: μ = 1/(1+exp(-η)) [the "sigmoid" or "logistic" function]
    - Derivative: dη/dμ = 1/(μ(1-μ))
    
    When to Use
    -----------
    - Binomial family (binary outcomes)
    - Modeling probabilities
    - Yes/no, claim/no-claim type questions
    
    Understanding Log-Odds
    ----------------------
    If μ = 0.8 (80% probability):
    - Odds = μ/(1-μ) = 0.8/0.2 = 4 ("4-to-1 odds")
    - Log-odds = log(4) ≈ 1.39
    
    The logit function maps:
    - μ = 0.5 → η = 0 (even odds)
    - μ → 0 maps to η → -∞
    - μ → 1 maps to η → +∞
    
    Odds Ratio Interpretation
    -------------------------
    Coefficients represent LOG odds ratios:
    
    - If β = 0.5 for "previous claims" indicator
    - Then exp(0.5) ≈ 1.65
    - People with previous claims have 1.65× the ODDS of claiming
    - This is NOT the same as 1.65× the probability!
    
    Converting to Probability Change
    --------------------------------
    The effect on probability depends on the baseline probability:
    
    - At baseline μ=0.1: a β=0.5 coefficient changes probability to ~0.15
    - At baseline μ=0.5: the same β changes probability to ~0.62
    
    This is why we report odds ratios, not probability ratios.
    
    Example
    -------
    >>> link = rs.links.Logit()
    >>> 
    >>> # η = 0 means 50% probability
    >>> # η = 2 means high probability (about 88%)
    >>> # η = -2 means low probability (about 12%)
    >>> 
    >>> eta = np.array([-2.0, 0.0, 2.0])
    >>> mu = link.inverse(eta)
    >>> print(mu)  # [0.119, 0.5, 0.881]
    """
    return _LogitLink()


# For backwards compatibility and convenience
__all__ = ["Identity", "Log", "Logit"]
