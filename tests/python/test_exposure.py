"""RS-ACT-002: explicit raw exposure vs link-scale offset.

Covers the `exposure=` keyword and validation plus the rule that a link-scale
array offset is never used as the target-encoding denominator (PR2a), and
prediction, serialization, and diagnostics threading for exposure models (PR4).
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

    def test_explicit_exposure_plus_string_offset_treats_offset_as_link_scale(self):
        """002.2: when exposure= is explicit, offset= remains link-scale."""
        df = make_freq_frame()
        adjustment = np.linspace(-0.2, 0.2, df.height)
        df = df.with_columns(pl.Series("Adj", adjustment))
        terms = {"DrivAge": {"type": "linear"}, "VehAge": {"type": "linear"}}

        string_offset = rs.glm_dict(
            response="ClaimCount",
            terms=terms,
            data=df,
            family="poisson",
            exposure="Exposure",
            offset="Adj",
        ).fit()
        array_offset = rs.glm_dict(
            response="ClaimCount",
            terms=terms,
            data=df,
            family="poisson",
            exposure="Exposure",
            offset=adjustment,
        ).fit()

        np.testing.assert_allclose(string_offset.params, array_offset.params, rtol=0, atol=0)
        np.testing.assert_allclose(
            string_offset.fittedvalues, array_offset.fittedvalues, rtol=0, atol=0
        )


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


class TestExposurePrediction:
    def _fit(self, df, **kw):
        terms = {"DrivAge": {"type": "linear"}, "Region": {"type": "categorical"}}
        return rs.glm_dict(
            response="ClaimCount", terms=terms, data=df, family="poisson", **kw
        ).fit()

    def test_predict_with_stored_exposure_column(self):
        """002.5: predict re-derives log(exposure) from the stored column."""
        df = make_freq_frame()
        result = self._fit(df, exposure="Exposure")
        np.testing.assert_allclose(result.predict(df), result.fittedvalues, rtol=1e-9, atol=1e-9)

    def test_predict_with_new_exposure_array(self):
        """002.6: exposure may be supplied at predict time."""
        df = make_freq_frame()
        result = self._fit(df, exposure="Exposure")
        head = df.head(20)
        new_exp = head["Exposure"].to_numpy()
        pred_array = result.predict(head.drop("Exposure"), exposure=new_exp)
        pred_col = result.predict(head)
        np.testing.assert_allclose(pred_array, pred_col, rtol=1e-9, atol=1e-9)

    def test_predict_errors_when_exposure_required_but_missing(self):
        """002.7: clear error when exposure is needed but unavailable."""
        df = make_freq_frame()
        result = self._fit(df, exposure="Exposure")
        with pytest.raises(rs.PredictionError, match="exposure='Exposure'"):
            result.predict(df.head(20).drop("Exposure"))

    def test_predict_with_stored_exposure_and_string_offset_on_row_subset(self):
        """002.9: stored exposure+offset metadata resolves against prediction rows."""
        df = make_freq_frame()
        adjustment = np.linspace(-0.2, 0.2, df.height)
        df = df.with_columns(pl.Series("Adj", adjustment))
        result = self._fit(df, exposure="Exposure", offset="Adj")

        head = df.head(20)
        pred_stored = result.predict(head)
        pred_override = result.predict(
            head.drop(["Exposure", "Adj"]),
            exposure=head["Exposure"].to_numpy(),
            offset=head["Adj"].to_numpy(),
        )
        assert {"Exposure", "Adj"}.issubset(result.required_columns)
        np.testing.assert_allclose(pred_stored, pred_override, rtol=1e-9, atol=1e-9)

    def test_predict_accepts_exposure_and_offset_overrides_together(self):
        """002.10: predict-time exposure and link-scale offset are additive."""
        df = make_freq_frame()
        result = self._fit(df, exposure="Exposure")
        head = df.head(20)
        offset = np.linspace(-0.1, 0.1, head.height)

        pred_both = result.predict(
            head.drop("Exposure"),
            exposure=head["Exposure"].to_numpy(),
            offset=offset,
        )
        pred_without_offset = result.predict(
            head.drop("Exposure"),
            exposure=head["Exposure"].to_numpy(),
        )
        np.testing.assert_allclose(
            pred_both,
            pred_without_offset * np.exp(offset),
            rtol=1e-9,
            atol=1e-9,
        )

    def test_predict_exposure_array_length_must_match_prediction_rows(self):
        df = make_freq_frame()
        result = self._fit(df, exposure="Exposure")
        with pytest.raises(rs.PredictionError, match="exposure array length"):
            result.predict(df.head(20).drop("Exposure"), exposure=np.ones(19))

    def test_exposure_plus_offset_combines_additively(self):
        """002.2: eta = eta_terms + log(exposure) + offset."""
        df = make_freq_frame()
        adj = np.linspace(-0.2, 0.2, df.height)
        terms = {"DrivAge": {"type": "linear"}, "VehAge": {"type": "linear"}}
        combined = rs.glm_dict(
            response="ClaimCount",
            terms=terms,
            data=df,
            family="poisson",
            exposure="Exposure",
            offset=adj,
        ).fit()
        single = rs.glm_dict(
            response="ClaimCount",
            terms=terms,
            data=df,
            family="poisson",
            offset=np.log(df["Exposure"].to_numpy()) + adj,
        ).fit()
        np.testing.assert_allclose(combined.params, single.params, rtol=0, atol=0)


class TestExposureSerialization:
    def test_serialization_roundtrips_exposure_model(self):
        """002.8: exposure_spec round-trips and predictions match after load."""
        df = make_freq_frame()
        terms = {"DrivAge": {"type": "linear"}, "Region": {"type": "categorical"}}
        result = rs.glm_dict(
            response="ClaimCount", terms=terms, data=df, family="poisson", exposure="Exposure"
        ).fit()
        loaded = rs.GLMModel.from_bytes(result.to_bytes())
        np.testing.assert_array_almost_equal(loaded.params, result.params)
        np.testing.assert_allclose(loaded.predict(df), result.predict(df), rtol=1e-9, atol=1e-9)

    def test_serialization_roundtrips_split_exposure_and_offset_metadata(self):
        df = make_freq_frame()
        adjustment = np.linspace(-0.2, 0.2, df.height)
        df = df.with_columns(pl.Series("Adj", adjustment))
        terms = {"DrivAge": {"type": "linear"}, "Region": {"type": "categorical"}}
        result = rs.glm_dict(
            response="ClaimCount",
            terms=terms,
            data=df,
            family="poisson",
            exposure="Exposure",
            offset="Adj",
        ).fit()

        loaded = rs.GLMModel.from_bytes(result.to_bytes())
        assert loaded._exposure_spec == "Exposure"
        assert loaded._offset_spec == "Adj"
        np.testing.assert_allclose(
            loaded.predict(df.head(20)),
            result.predict(df.head(20)),
            rtol=1e-9,
            atol=1e-9,
        )

    def test_deserializes_legacy_offset_exposure_payload_as_exposure_model(self):
        import pickle

        df = make_freq_frame()
        terms = {"DrivAge": {"type": "linear"}, "Region": {"type": "categorical"}}
        result = rs.glm_dict(
            response="ClaimCount", terms=terms, data=df, family="poisson", exposure="Exposure"
        ).fit()
        state = pickle.loads(result.to_bytes())
        state.pop("exposure_spec", None)
        state["offset_spec"] = "Exposure"
        state["offset_is_exposure"] = True

        loaded = rs.GLMModel.from_bytes(pickle.dumps(state))
        assert loaded._exposure_spec == "Exposure"
        assert loaded._offset_spec is None
        assert not loaded._offset_is_exposure
        np.testing.assert_allclose(
            loaded.predict(df.head(20)),
            result.predict(df.head(20)),
            rtol=1e-9,
            atol=1e-9,
        )


class TestExposureDiagnostics:
    def test_diagnostics_sources_exposure_from_exposure_spec(self):
        from rustystats.diagnostics.api import _resolve_offset_and_response

        df = make_freq_frame()
        df = df.with_columns(pl.Series("Adj", np.linspace(-0.1, 0.1, df.height)))
        result = rs.glm_dict(
            response="ClaimCount",
            terms={"DrivAge": {"type": "linear"}},
            data=df,
            family="poisson",
            exposure="Exposure",
            offset="Adj",
        ).fit()
        _response_col, exposure_col, exposure = _resolve_offset_and_response(result, df)
        assert exposure_col == "Exposure"
        assert exposure is not None
        np.testing.assert_array_equal(exposure, df["Exposure"].to_numpy())

    def test_diagnostics_runs_for_exposure_model(self):
        df = make_freq_frame()
        result = rs.glm_dict(
            response="ClaimCount",
            terms={"DrivAge": {"type": "linear"}, "Region": {"type": "categorical"}},
            data=df,
            family="poisson",
            exposure="Exposure",
        ).fit()
        diag = result.diagnostics(
            train_data=df, categorical_factors=["Region"], continuous_factors=["DrivAge"]
        )
        assert diag.calibration is not None
