"""Deterministic synthetic datasets for the actuarial-hardening test suite.

These generators are engineered to exercise the specific failure modes the
RS-ACT items target, and are deliberately reproducible (seeded ``default_rng``)
so they can anchor characterization and golden tests.

Designed properties (asserted in ``test_characterization.py``):

* ``make_freq_frame`` -- ``Exposure`` is drawn from an independent stream, so it
  is uncorrelated with the true risk rate. Ranking policies by predicted count
  ``mu`` and by predicted rate ``mu / Exposure`` therefore genuinely disagree
  (needed by RS-ACT-004 / RS-ACT-009). It also contains a deliberately rare
  ``Brand`` level (:data:`RARE_BRAND`) confined to a contiguous tail block, so a
  CV split can be constructed in which the level is unseen in training
  (needed by RS-ACT-001).
* ``make_severity_frame`` -- strictly positive Gamma severity with no exposure
  (RS-ACT-004 negative case / RS-ACT-006).
* ``make_overdispersed_counts`` -- Negative-Binomial counts with a known
  ``theta`` (RS-ACT-010 / RS-ACT-008).

The generators depend only on numpy + polars (never on rustystats) so they can
also be imported by helper scripts.
"""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = [
    "OVERDISP_THETA",
    "RARE_BRAND",
    "make_freq_frame",
    "make_overdispersed_counts",
    "make_severity_frame",
]

#: The deliberately rare ``Brand`` level injected by :func:`make_freq_frame`.
RARE_BRAND = "RARE"

#: True dispersion parameter baked into :func:`make_overdispersed_counts`.
OVERDISP_THETA = 0.7


def make_freq_frame(n: int = 4000, seed: int = 0, n_rare: int = 8) -> pl.DataFrame:
    """Poisson claim-frequency frame with exposure independent of risk.

    Columns: ``ClaimCount`` (response), ``DrivAge``/``VehAge`` (continuous),
    ``Region`` (categorical), ``Brand`` (high-cardinality, incl. a rare level),
    ``Exposure`` (positive, risk-independent), and ``true_rate`` (an oracle
    column used by tests, never a model input).
    """
    rng = np.random.default_rng(seed)
    driv_age = rng.uniform(18.0, 80.0, n)
    veh_age = rng.uniform(0.0, 20.0, n)
    region = rng.choice(["A", "B", "C", "D"], size=n)
    brand = rng.choice([f"B{i}" for i in range(1, 6)], size=n).astype(object)

    # Confine a rare level to the tail so a single CV fold can hold it out.
    if n_rare > 0:
        brand[-n_rare:] = RARE_BRAND

    region_effect = np.select(
        [region == "A", region == "B", region == "C"],
        [0.30, 0.10, -0.05],
        default=0.0,
    )
    log_rate = -3.0 + 0.03 * driv_age - 0.02 * veh_age + region_effect
    rate = np.exp(log_rate)  # per-unit-exposure claim rate

    # Exposure is an independent draw => exposure is uncorrelated with risk.
    exposure = rng.uniform(0.05, 1.0, n)
    claim_count = rng.poisson(rate * exposure).astype(float)

    return pl.DataFrame(
        {
            "ClaimCount": claim_count,
            "DrivAge": driv_age,
            "VehAge": veh_age,
            "Region": region,
            "Brand": pl.Series("Brand", brand.tolist(), dtype=pl.String),
            "Exposure": exposure,
            "true_rate": rate,
        }
    )


def make_severity_frame(n: int = 3000, seed: int = 1) -> pl.DataFrame:
    """Strictly positive Gamma severity frame (no exposure).

    Columns: ``ClaimAmt`` (response, > 0), ``DrivAge``/``VehAge`` (continuous).
    """
    rng = np.random.default_rng(seed)
    driv_age = rng.uniform(18.0, 80.0, n)
    veh_age = rng.uniform(0.0, 20.0, n)
    mu = np.exp(6.0 + 0.01 * driv_age + 0.02 * veh_age)
    shape = 2.0
    claim_amt = rng.gamma(shape, mu / shape)  # mean == mu, strictly positive
    return pl.DataFrame(
        {
            "ClaimAmt": claim_amt,
            "DrivAge": driv_age,
            "VehAge": veh_age,
        }
    )


def make_overdispersed_counts(
    n: int = 3000, seed: int = 2, theta: float = OVERDISP_THETA
) -> pl.DataFrame:
    """Negative-Binomial counts with a known dispersion ``theta``.

    Uses the ``Var = mu + mu**2 / theta`` parameterization (the same one
    RustyStats' NB family uses). Columns: ``y`` (response), ``x`` (continuous).
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    mu = np.exp(0.5 + 0.4 * x)
    # numpy parameterization: mean = theta * (1 - p) / p; with p = theta/(theta+mu)
    # this yields mean == mu and Var == mu + mu**2 / theta.
    p = theta / (theta + mu)
    y = rng.negative_binomial(theta, p).astype(float)
    return pl.DataFrame({"y": y, "x": x})
