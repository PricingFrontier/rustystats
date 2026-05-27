"""Characterization tests for the actuarial-hardening work.

These pin *current correct behaviour* before the refactors land, so that later
changes are provably behaviour-preserving for the cases they must not affect.
They deliberately anchor against statsmodels (a version-stable oracle) rather
than frozen magic numbers -- matching the philosophy of ``test_comparison.py``
-- and assert the designed properties of the shared fixtures in ``_fixtures.py``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import rustystats as rs
from _fixtures import RARE_BRAND, make_freq_frame, make_severity_frame

sm = pytest.importorskip("statsmodels.api")
smf = pytest.importorskip("statsmodels.genmod.families")

# Cross-library tolerances mirror tests/python/test_comparison.py.
COEF_ATOL = 0.02
COEF_RTOL = 0.01
DEVIANCE_RTOL = 0.005


# ---------------------------------------------------------------------------
# Fixture property guards: a vacuous fixture would make later tests meaningless
# ---------------------------------------------------------------------------


class TestFixtureProperties:
    def test_exposure_independent_of_risk(self):
        df = make_freq_frame()
        corr = np.corrcoef(df["Exposure"].to_numpy(), df["true_rate"].to_numpy())[0, 1]
        assert abs(corr) < 0.05

    def test_rate_and_count_orderings_disagree(self):
        # mu and mu/exposure must induce materially different deciles, otherwise
        # the RS-ACT-004 rate-ranking tests would be vacuous.
        df = make_freq_frame()
        rate = df["true_rate"].to_numpy()
        exposure = df["Exposure"].to_numpy()
        mu = rate * exposure

        def deciles(v):
            ranks = np.argsort(np.argsort(v))
            return ranks * 10 // len(v)

        disagree = np.mean(deciles(mu) != deciles(mu / exposure))
        assert disagree > 0.2

    def test_rare_brand_present_and_small(self):
        df = make_freq_frame()
        assert RARE_BRAND in df["Brand"].unique().to_list()
        n_rare = df.filter(pl.col("Brand") == RARE_BRAND).height
        assert 0 < n_rare <= 20


# ---------------------------------------------------------------------------
# Determinism: the whole pipeline must be reproducible for golden tests to hold
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_poisson_offset_fit_is_reproducible(self):
        df = make_freq_frame()
        terms = {"DrivAge": {"type": "linear"}, "VehAge": {"type": "linear"}}
        r1 = rs.glm_dict(
            response="ClaimCount", terms=terms, data=df, family="poisson", offset="Exposure"
        ).fit()
        r2 = rs.glm_dict(
            response="ClaimCount", terms=terms, data=df, family="poisson", offset="Exposure"
        ).fit()
        np.testing.assert_array_equal(r1.params, r2.params)

    def test_target_encoding_fit_is_reproducible(self):
        df = make_freq_frame()
        terms = {"DrivAge": {"type": "linear"}, "Brand": {"type": "target_encoding"}}
        r1 = rs.glm_dict(
            response="ClaimCount",
            terms=terms,
            data=df,
            family="poisson",
            offset="Exposure",
            seed=42,
        ).fit()
        r2 = rs.glm_dict(
            response="ClaimCount",
            terms=terms,
            data=df,
            family="poisson",
            offset="Exposure",
            seed=42,
        ).fit()
        np.testing.assert_array_equal(r1.params, r2.params)


# ---------------------------------------------------------------------------
# statsmodels anchors: guard exposure/offset semantics across RS-ACT-002/004
# ---------------------------------------------------------------------------


class TestPoissonOffsetAnchor:
    """Poisson + log-link offset must match statsmodels.

    This is the regression guard for RS-ACT-002: after `exposure=` is added and
    legacy `offset="Exposure"` is normalized, BOTH spellings must still match
    this anchor (and each other).
    """

    @staticmethod
    def _design(df):
        return sm.add_constant(np.column_stack([df["DrivAge"].to_numpy(), df["VehAge"].to_numpy()]))

    def _fit_rustystats(self, df):
        terms = {"DrivAge": {"type": "linear"}, "VehAge": {"type": "linear"}}
        return rs.glm_dict(
            response="ClaimCount", terms=terms, data=df, family="poisson", offset="Exposure"
        ).fit()

    def _fit_statsmodels(self, df):
        y = df["ClaimCount"].to_numpy()
        offset = np.log(df["Exposure"].to_numpy())
        return sm.GLM(y, self._design(df), family=smf.Poisson(), offset=offset).fit()

    def test_coefficients_match_statsmodels(self):
        df = make_freq_frame()
        rs_res = self._fit_rustystats(df)
        sm_res = self._fit_statsmodels(df)
        # Order: [Intercept, DrivAge, VehAge] vs [const, DrivAge, VehAge].
        np.testing.assert_allclose(rs_res.params, sm_res.params, atol=COEF_ATOL, rtol=COEF_RTOL)

    def test_deviance_matches_statsmodels(self):
        df = make_freq_frame()
        rs_res = self._fit_rustystats(df)
        sm_res = self._fit_statsmodels(df)
        np.testing.assert_allclose(rs_res.deviance, sm_res.deviance, rtol=DEVIANCE_RTOL)


class TestGammaSeverityAnchor:
    """Gamma + log link (no exposure) must match statsmodels.

    Guards that the severity path (no exposure) is unaffected by the exposure
    work in RS-ACT-002/004 and the Tweedie/family work in RS-ACT-006.
    """

    def test_coefficients_match_statsmodels(self):
        df = make_severity_frame()
        terms = {"DrivAge": {"type": "linear"}, "VehAge": {"type": "linear"}}
        rs_res = rs.glm_dict(response="ClaimAmt", terms=terms, data=df, family="gamma").fit()

        y = df["ClaimAmt"].to_numpy()
        X = sm.add_constant(np.column_stack([df["DrivAge"].to_numpy(), df["VehAge"].to_numpy()]))
        sm_res = sm.GLM(y, X, family=smf.Gamma(link=smf.links.Log())).fit()
        np.testing.assert_allclose(rs_res.params, sm_res.params, atol=COEF_ATOL, rtol=COEF_RTOL)
