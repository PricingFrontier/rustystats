"""RS-ACT-002: explicit raw exposure vs link-scale offset.

PR2a scope: the `exposure=` keyword, validation, and the rule that a link-scale
array offset must never be used as the target-encoding denominator. Prediction,
diagnostics and serialization parity for exposure models are covered in PR4.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import rustystats as rs
from _fixtures import make_freq_frame


def _te_column(model, needle: str = "Brand") -> np.ndarray:
    """Return the design column for the single term whose name contains `needle`."""
    idx = [i for i, name in enumerate(model.feature_names) if needle in name]
    assert len(idx) == 1, f"expected one '{needle}' column, found {idx}: {model.feature_names}"
    return np.asarray(model.X[:, idx[0]])


class TestExposureKwarg:
    def test_string_exposure_matches_legacy_offset(self):
        """002.1: exposure="Exposure" is identical to legacy offset="Exposure"."""
        df = make_freq_frame()
        terms = {"DrivAge": {"type": "linear"}, "Region": {"type": "categorical"}}
        legacy = rs.glm_dict(
            response="ClaimCount", terms=terms, data=df, family="poisson", offset="Exposure"
        ).fit()
        explicit = rs.glm_dict(
            response="ClaimCount", terms=terms, data=df, family="poisson", exposure="Exposure"
        ).fit()
        np.testing.assert_array_equal(explicit.params, legacy.params)
        np.testing.assert_array_equal(explicit.fittedvalues, legacy.fittedvalues)


class TestExposureValidation:
    def test_non_positive_exposure_raises(self):
        df = make_freq_frame()
        exp = df["Exposure"].to_numpy().copy()
        exp[0] = 0.0
        df = df.with_columns(pl.Series("Exposure", exp))
        with pytest.raises(rs.ValidationError):
            rs.glm_dict(
                response="ClaimCount",
                terms={"DrivAge": {"type": "linear"}},
                data=df,
                family="poisson",
                exposure="Exposure",
            )

    def test_wrong_length_array_exposure_raises(self):
        df = make_freq_frame()
        with pytest.raises(rs.ValidationError):
            rs.glm_dict(
                response="ClaimCount",
                terms={"DrivAge": {"type": "linear"}},
                data=df,
                family="poisson",
                exposure=np.ones(df.height - 1),
            )

    def test_non_finite_exposure_raises(self):
        df = make_freq_frame()
        exp = df["Exposure"].to_numpy().copy()
        exp[0] = np.inf
        with pytest.raises(rs.ValidationError):
            rs.glm_dict(
                response="ClaimCount",
                terms={"DrivAge": {"type": "linear"}},
                data=df,
                family="poisson",
                exposure=exp,
            )

    def test_exposure_on_non_log_link_raises(self):
        df = make_freq_frame()
        with pytest.raises(rs.ValidationError):
            rs.glm_dict(
                response="ClaimCount",
                terms={"DrivAge": {"type": "linear"}},
                data=df,
                family="gaussian",
                exposure="Exposure",
            )


class TestTargetEncodingExposureSource:
    def test_array_offset_not_used_as_te_denominator(self):
        """002.3: a link-scale array offset must NOT drive exposure-weighted TE.

        With the fix it falls back to unweighted encoding (and warns), so the TE
        column equals the no-exposure (unweighted) encoding rather than the old
        ``sum(y)/sum(log(exposure))`` garbage.
        """
        df = make_freq_frame()
        terms = {"Brand": {"type": "target_encoding"}}
        log_exp = np.log(df["Exposure"].to_numpy())

        m_unweighted = rs.glm_dict(
            response="ClaimCount", terms=terms, data=df, family="poisson", seed=42
        )
        with pytest.warns(UserWarning, match="exposure"):
            m_array_offset = rs.glm_dict(
                response="ClaimCount",
                terms=terms,
                data=df,
                family="poisson",
                offset=log_exp,
                seed=42,
            )

        np.testing.assert_array_equal(_te_column(m_array_offset), _te_column(m_unweighted))

    def test_exposure_weighted_te_differs_from_unweighted(self):
        """002.4: exposure= produces exposure-weighted TE, distinct from unweighted."""
        df = make_freq_frame()
        terms = {"Brand": {"type": "target_encoding"}}
        m_unweighted = rs.glm_dict(
            response="ClaimCount", terms=terms, data=df, family="poisson", seed=42
        )
        m_exposure = rs.glm_dict(
            response="ClaimCount",
            terms=terms,
            data=df,
            family="poisson",
            exposure="Exposure",
            seed=42,
        )
        assert not np.allclose(_te_column(m_exposure), _te_column(m_unweighted))
