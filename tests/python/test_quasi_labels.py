"""RS-ACT-008 (PR14): honest labels for quasi-likelihood diagnostics.

Quasi-Poisson and quasi-Binomial don't have ordinary full likelihoods — only a
quasi-likelihood with an estimated dispersion ``φ`` (Pearson χ² / df_resid).
Reporting AIC/BIC as if they were proper likelihood values can mislead model
selection. This PR:

* Flags quasi-likelihood fits via ``GLMModel.is_quasi_likelihood``.
* Returns ``None`` from ``aic()`` / ``bic()`` on quasi models — no documented
  QIC is implemented yet, so we don't surface a fake value.
* Relabels the log-likelihood-like value in ``summary()`` as
  ``Quasi-Log-Likelihood`` and prints AIC/BIC as ``NA``.
* Round-trips the quasi flag through serialization.

Tests:
* 008.1 — quasi-Poisson summary does not present AIC/BIC as ordinary
  likelihood values; ``result.aic()`` / ``result.bic()`` return ``None``.
* 008.2 — same for quasi-Binomial.
* 008.3 — serialization preserves the quasi diagnostic flag.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import rustystats as rs


def _quasipoisson_frame(seed: int = 0, n: int = 600) -> pl.DataFrame:
    """Overdispersed count data — Negative-Binomial draws fit by quasi-Poisson.

    NB-distributed counts have Var > μ, so a Poisson fit is mis-specified and
    a quasi-Poisson is the natural choice. This shape exposes the quasi path
    without leaning on the actual NB family.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    mu = np.exp(0.5 + 0.3 * x1 - 0.2 * x2)
    # NB2 parameterisation: Var = μ + μ²/θ; θ=2 → moderate overdispersion.
    theta = 2.0
    p = theta / (theta + mu)
    y = rng.negative_binomial(theta, p).astype(float)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2})


def _quasibinomial_frame(seed: int = 1, n: int = 500) -> pl.DataFrame:
    """Binary data with mild overdispersion via a noisy linear predictor."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # Inject extra noise into η so the realised variance exceeds μ(1-μ).
    eta = 0.1 + 0.4 * x1 - 0.3 * x2 + 0.5 * rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(size=n) < p).astype(float)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2})


def _fit(df: pl.DataFrame, family: str):
    return rs.glm_dict(
        response="y",
        terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
        data=df,
        family=family,
    ).fit()


# --------------------------------------------------------------------------
# 008.1: quasi-Poisson — no ordinary-likelihood AIC/BIC.
# --------------------------------------------------------------------------


class TestQuasiPoisson:
    def test_is_quasi_likelihood_flag(self):
        result = _fit(_quasipoisson_frame(), "quasipoisson")
        assert result.is_quasi_likelihood is True

    def test_aic_returns_none(self):
        """008.1: aic() must not present a Poisson-style AIC for quasi-Poisson."""
        result = _fit(_quasipoisson_frame(), "quasipoisson")
        assert result.aic() is None

    def test_bic_returns_none(self):
        result = _fit(_quasipoisson_frame(), "quasipoisson")
        assert result.bic() is None

    def test_summary_labels_quasi_likelihood(self):
        """008.1: summary text labels the log-likelihood-like value as quasi
        and omits a numeric AIC/BIC (prints NA)."""
        result = _fit(_quasipoisson_frame(), "quasipoisson")
        text = result.summary()
        assert "Quasi-Log-Likelihood" in text or "Quasi-Likelihood" in text, (
            "summary should relabel the loglik-like value as quasi"
        )
        # AIC / BIC lines exist but show NA, not a number.
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("AIC"):
                assert "NA" in line, f"AIC line should be NA for quasi: {line!r}"
            if stripped.startswith("BIC"):
                assert "NA" in line, f"BIC line should be NA for quasi: {line!r}"

    def test_non_quasi_poisson_still_reports_aic(self):
        """Guard: the suppression is *only* for quasi families."""
        df = _quasipoisson_frame()
        result = _fit(df, "poisson")
        assert result.is_quasi_likelihood is False
        # Plain Poisson still has a numeric AIC/BIC.
        assert result.aic() is not None
        assert result.bic() is not None
        assert np.isfinite(result.aic())
        assert np.isfinite(result.bic())


# --------------------------------------------------------------------------
# 008.2: quasi-Binomial — same treatment.
# --------------------------------------------------------------------------


class TestQuasiBinomial:
    def test_is_quasi_likelihood_flag(self):
        result = _fit(_quasibinomial_frame(), "quasibinomial")
        assert result.is_quasi_likelihood is True

    def test_aic_returns_none(self):
        result = _fit(_quasibinomial_frame(), "quasibinomial")
        assert result.aic() is None

    def test_bic_returns_none(self):
        result = _fit(_quasibinomial_frame(), "quasibinomial")
        assert result.bic() is None

    def test_non_quasi_binomial_still_reports_aic(self):
        df = _quasibinomial_frame()
        result = _fit(df, "binomial")
        assert result.is_quasi_likelihood is False
        assert result.aic() is not None
        assert result.bic() is not None


# --------------------------------------------------------------------------
# 008.3: serialization preserves the quasi flag.
# --------------------------------------------------------------------------


class TestSerializationRoundTrip:
    @pytest.mark.parametrize("family", ["quasipoisson", "quasibinomial"])
    def test_round_trip_preserves_quasi_flag_and_aic_behaviour(self, family):
        """008.3: a reloaded quasi model still flags as quasi-likelihood and
        still returns None from ``aic()`` / ``bic()``.

        The quasi flag is derived from ``self.family`` and ``family`` is
        already preserved by ``to_bytes`` / ``from_bytes``, so the round-trip
        only needs to assert the derived behaviour stays intact — no extra
        serialisation state is needed.

        Summary text is not checked here because deserialised results lack
        inference state (``bse`` / ``tvalues``) by design — that's an
        orthogonal limitation that ``GLMModel`` warns about elsewhere.
        """
        if family == "quasipoisson":
            df = _quasipoisson_frame()
        else:
            df = _quasibinomial_frame()
        original = _fit(df, family)
        loaded = rs.GLMModel.from_bytes(original.to_bytes())

        # The family string itself round-trips, so the derived flag follows.
        assert loaded.family.lower().startswith("quasi"), (
            f"reloaded model lost its quasi family tag: {loaded.family!r}"
        )
        assert loaded.is_quasi_likelihood is True
        assert loaded.aic() is None
        assert loaded.bic() is None
