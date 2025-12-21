"""
Distribution Families for GLMs
==============================

This module provides distribution families that specify:
1. What type of data you're modeling (counts, binary, continuous, etc.)
2. How variance relates to the mean (the variance function)
3. The default link function to use

Choosing the Right Family
-------------------------

+------------------+-------------------+------------------+-------------------+
| Data Type        | Example           | Family           | Typical Link      |
+==================+===================+==================+===================+
| Continuous       | Temperature       | Gaussian         | Identity          |
| Strictly positive| Claim amounts     | Gamma            | Log               |
| Counts (0,1,2,..)| Claim frequency   | Poisson          | Log               |
| Binary (0 or 1)  | Did they claim?   | Binomial         | Logit             |
| Proportions      | % who claimed     | Binomial         | Logit             |
+------------------+-------------------+------------------+-------------------+

Understanding Variance Functions
--------------------------------

The variance function V(μ) tells us how the variance of Y relates to its mean:

    Var(Y) = φ × V(μ)

where φ is the dispersion parameter (φ=1 for Poisson and Binomial).

- **Gaussian**: V(μ) = 1
  Variance is constant. A $100 claim varies the same as a $10,000 claim.
  This is usually unrealistic for monetary amounts.

- **Poisson**: V(μ) = μ  
  Variance equals mean. If average claims = 0.1, variance = 0.1.
  Good for counts, but real data is often overdispersed.

- **Gamma**: V(μ) = μ²
  Variance is proportional to mean squared, so coefficient of variation (CV)
  is constant. A $1,000 claim varies proportionally the same as a $100,000 claim.
  Very appropriate for insurance claim amounts.

- **Binomial**: V(μ) = μ(1-μ)
  Maximum variance at μ=0.5, zero variance at μ=0 or μ=1.
  Makes sense: if something always (or never) happens, there's no variation.

Examples
--------
>>> import rustystats as rs
>>> import numpy as np
>>>
>>> # Check variance function values
>>> poisson = rs.families.Poisson()
>>> mu = np.array([1.0, 2.0, 5.0])
>>> print(poisson.variance(mu))  # [1.0, 2.0, 5.0] - same as mu!
>>>
>>> gamma = rs.families.Gamma()
>>> print(gamma.variance(mu))  # [1.0, 4.0, 25.0] - mu squared!
"""

# Import the Rust implementations
from rustystats._rustystats import (
    GaussianFamily as _GaussianFamily,
    PoissonFamily as _PoissonFamily,
    BinomialFamily as _BinomialFamily,
    GammaFamily as _GammaFamily,
)


def Gaussian():
    """
    Gaussian (Normal) family for continuous response data.
    
    Use this for standard linear regression where the response can be
    any real value (positive, negative, or zero).
    
    Properties
    ----------
    - Variance function: V(μ) = 1 (constant variance)
    - Default link: Identity (η = μ)
    - Dispersion: σ² (estimated from residuals)
    
    When to Use
    -----------
    - Continuous data with approximately constant variance
    - When you'd normally use ordinary least squares
    - When residuals are roughly normally distributed
    
    When NOT to Use
    ---------------
    - For strictly positive data (use Gamma instead)
    - For count data (use Poisson instead)
    - For binary outcomes (use Binomial instead)
    
    Example
    -------
    >>> family = rs.families.Gaussian()
    >>> print(family.name())  # "Gaussian"
    >>> print(family.variance(np.array([1.0, 100.0])))  # [1.0, 1.0]
    """
    return _GaussianFamily()


def Poisson():
    """
    Poisson family for count data (0, 1, 2, 3, ...).
    
    This is the standard family for claim FREQUENCY modeling.
    
    Properties
    ----------
    - Variance function: V(μ) = μ (variance equals the mean)
    - Default link: Log (η = log(μ))
    - Dispersion: φ = 1 (fixed)
    
    Key Assumption: Equidispersion
    ------------------------------
    Poisson assumes variance = mean. This is often violated in practice
    ("overdispersion"). Check by looking at:
    
        Pearson χ² / degrees of freedom
    
    If this is much greater than 1, you have overdispersion. Options:
    - Use quasi-Poisson (adjust standard errors)
    - Use Negative Binomial family (once implemented)
    
    When to Use
    -----------
    - Claim counts per policy
    - Number of accidents
    - Event counts in a fixed period
    
    Exposure Adjustment
    -------------------
    Often used with an "exposure" offset. If modeling annual claim counts
    but some policies are only 6 months:
    
        E(claims) = exposure × exp(Xβ)
        log(E(claims)) = log(exposure) + Xβ
    
    The log(exposure) term is an "offset" with coefficient fixed at 1.
    
    Example
    -------
    >>> family = rs.families.Poisson()
    >>> mu = np.array([0.5, 1.0, 2.0])
    >>> print(family.variance(mu))  # [0.5, 1.0, 2.0] - same as mu!
    """
    return _PoissonFamily()


def Binomial():
    """
    Binomial family for binary or proportion data.
    
    This is the foundation of LOGISTIC REGRESSION.
    
    Properties
    ----------
    - Variance function: V(μ) = μ(1-μ)
    - Default link: Logit (η = log(μ/(1-μ)))
    - Dispersion: φ = 1 (fixed)
    
    Understanding the Variance Function
    -----------------------------------
    V(μ) = μ(1-μ) means variance is:
    - Maximum at μ = 0.5 (most uncertainty)
    - Zero at μ = 0 or μ = 1 (certain outcomes)
    
    This makes intuitive sense: if something almost always (or never)
    happens, there's not much variation in outcomes.
    
    Interpreting Coefficients
    -------------------------
    With logit link, coefficients are on the log-odds scale:
    
    - If β = 0.5 for variable X, then exp(0.5) ≈ 1.65
    - This means: "1.65 times the odds for each 1-unit increase in X"
    - OR: "65% higher odds"
    
    When to Use
    -----------
    - Binary outcomes (claim/no claim, lapse/retain)
    - Conversion rates
    - Any yes/no question
    
    Example
    -------
    >>> family = rs.families.Binomial()
    >>> mu = np.array([0.2, 0.5, 0.8])
    >>> print(family.variance(mu))  # [0.16, 0.25, 0.16]
    >>> # Note: max variance at μ=0.5
    """
    return _BinomialFamily()


def Gamma():
    """
    Gamma family for positive continuous data.
    
    This is the standard family for claim SEVERITY (amount) modeling.
    
    Properties
    ----------
    - Variance function: V(μ) = μ² (variance proportional to mean squared)
    - Default link: Log (η = log(μ)) - note: canonical is inverse, but log is standard
    - Dispersion: φ = 1/shape (estimated from residuals)
    
    Key Insight: Constant Coefficient of Variation
    ----------------------------------------------
    Since V(μ) = μ², the standard deviation is proportional to the mean:
    
        SD(Y) = √(φ × μ²) = √φ × μ
        CV = SD/mean = √φ (constant!)
    
    This is very realistic for monetary amounts:
    - A $1,000 claim might vary by ±$500 (CV = 50%)
    - A $100,000 claim might vary by ±$50,000 (same CV = 50%)
    
    Why Gamma for Claim Amounts?
    ----------------------------
    - Gaussian assumes constant variance (unrealistic for money)
    - Gamma's constant CV matches observed behavior of claim amounts
    - Log link ensures predictions are always positive
    - Coefficients have multiplicative interpretation
    
    Combining with Poisson (Pure Premium)
    -------------------------------------
    Pure premium = Frequency × Severity
    
    If you model:
    - Frequency: Poisson with log link
    - Severity: Gamma with log link
    
    Then pure premium coefficients are the SUM of the two models' coefficients!
    (Because log(Freq × Sev) = log(Freq) + log(Sev))
    
    Example
    -------
    >>> family = rs.families.Gamma()
    >>> mu = np.array([100.0, 1000.0, 10000.0])
    >>> print(family.variance(mu))  # [10000, 1000000, 100000000]
    >>> # Variance grows with the square of the mean
    """
    return _GammaFamily()


# For backwards compatibility and convenience
__all__ = ["Gaussian", "Poisson", "Binomial", "Gamma"]
