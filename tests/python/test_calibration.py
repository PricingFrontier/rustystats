"""RS-ACT-009 (PR11): explicit calibration diagnostics and primitives.

Covers:
* ``rs.calibration_summary`` array primitive (009.1-4).
* ``rs.GlobalCalibration`` / ``rs.fit_global_calibration`` (009.5).
* Log-link ``result.relevel()`` invariants (009.6-7).
* Relevel rejects non-log links (009.8).
* ``rs.IsotonicCalibration`` is monotone, opt-in, serialized separately, and
  raw vs calibrated predictions are both accessible (009.9-10).
* Weighted relevel balances ``Σ(w·μ_new) == Σ(w·y)`` and reduces to the
  unweighted case when ``w ≡ 1`` (009.11).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import rustystats as rs
from _fixtures import make_freq_frame, make_severity_frame
from rustystats.exceptions import ValidationError

# --------------------------------------------------------------------------
# 009.1-2: Overall A/E (unweighted and weighted) via rs.calibration_summary.
# --------------------------------------------------------------------------


class TestOverallAE:
    def test_unweighted_overall_ae_equals_sum_ratio(self):
        """009.1: overall A/E == Σy / Σpred."""
        rng = np.random.default_rng(42)
        y = rng.uniform(0.0, 5.0, 500)
        pred = rng.uniform(0.1, 5.0, 500)
        summary = rs.calibration_summary(y, pred)
        expected = float(np.sum(y)) / float(np.sum(pred))
        np.testing.assert_allclose(summary["overall"]["ae_ratio"], expected, rtol=1e-12)
        np.testing.assert_allclose(summary["overall"]["actual"], float(np.sum(y)), rtol=1e-12)
        np.testing.assert_allclose(summary["overall"]["expected"], float(np.sum(pred)), rtol=1e-12)
        assert summary["overall"]["n_obs"] == 500

    def test_weighted_overall_ae_uses_weighted_sums(self):
        """009.2: weighted overall A/E == Σ(w·y) / Σ(w·pred)."""
        rng = np.random.default_rng(7)
        n = 400
        y = rng.uniform(0.0, 5.0, n)
        pred = rng.uniform(0.1, 5.0, n)
        weights = rng.uniform(0.5, 3.0, n)
        summary = rs.calibration_summary(y, pred, weights=weights)
        expected = float(np.sum(weights * y)) / float(np.sum(weights * pred))
        np.testing.assert_allclose(summary["overall"]["ae_ratio"], expected, rtol=1e-12)
        np.testing.assert_allclose(
            summary["overall"]["actual"], float(np.sum(weights * y)), rtol=1e-12
        )
        np.testing.assert_allclose(
            summary["overall"]["expected"], float(np.sum(weights * pred)), rtol=1e-12
        )
        np.testing.assert_allclose(
            summary["overall"]["total_weight"], float(np.sum(weights)), rtol=1e-12
        )

    def test_weights_one_reduces_to_unweighted(self):
        """sanity: weights==1 produces the unweighted result."""
        rng = np.random.default_rng(11)
        y = rng.uniform(0.0, 5.0, 200)
        pred = rng.uniform(0.1, 5.0, 200)
        s_unw = rs.calibration_summary(y, pred)
        s_w1 = rs.calibration_summary(y, pred, weights=np.ones_like(y))
        np.testing.assert_allclose(
            s_unw["overall"]["ae_ratio"], s_w1["overall"]["ae_ratio"], rtol=1e-12
        )

    def test_calibration_line_reports_intercept_and_slope(self):
        pred = np.linspace(1.0, 10.0, 200)
        y = 0.25 + 1.5 * pred
        summary = rs.calibration_summary(y, pred)
        np.testing.assert_allclose(summary["calibration_intercept"], 0.25, rtol=1e-12)
        np.testing.assert_allclose(summary["calibration_slope"], 1.5, rtol=1e-12)


class TestCalibrationValidation:
    @pytest.mark.parametrize(
        "exposure",
        [
            np.array([1.0, 0.0, 2.0]),
            np.array([1.0, -1.0, 2.0]),
            np.array([1.0, np.nan, 2.0]),
            np.array([1.0, np.inf, 2.0]),
        ],
        ids=["zero", "negative", "nan", "inf"],
    )
    def test_exposure_must_be_finite_and_positive(self, exposure):
        y = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValidationError, match="exposure must be finite and positive"):
            rs.calibration_summary(y, pred, exposure=exposure)

    @pytest.mark.parametrize(
        "pred",
        [np.array([1.0, np.nan]), np.array([1.0, np.inf])],
        ids=["nan", "inf"],
    )
    def test_pred_must_be_finite(self, pred):
        with pytest.raises(ValidationError, match="pred must contain only finite values"):
            rs.fit_global_calibration(np.array([1.0, 2.0]), pred)

    def test_response_must_be_finite(self):
        with pytest.raises(ValidationError, match="y must contain only finite values"):
            rs.calibration_summary(np.array([1.0, np.nan]), np.array([1.0, 2.0]))


# --------------------------------------------------------------------------
# 009.3: rate-ranked bins when exposure is present (reuses RS-ACT-004 helper).
# --------------------------------------------------------------------------


class TestBinning:
    def test_auto_rate_ranks_with_exposure(self):
        """009.3: bins are rate-ranked when exposure is present and ranking='auto'."""
        # Construct conflict between count-ranking and rate-ranking. The first
        # half is high-count / low-rate; the second half is low-count / high-rate.
        # Exposure-balanced binning won't split exactly 10/10, but the *rate*
        # of the first bin must still be below the rate of the last bin.
        y = np.arange(1, 21, dtype=np.float64)
        mu = np.array(
            [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            dtype=np.float64,
        )
        exposure = np.array([1000.0] * 10 + [1.0] * 10, dtype=np.float64)
        summary = rs.calibration_summary(y, mu, exposure=exposure, n_bins=2)
        bins = summary["bins"]
        assert len(bins) >= 1
        # The strict rate-monotonicity across bins is the actual invariant; the
        # legacy code, which ranked on raw ``mu``, would invert this ordering
        # because high-exposure low-rate rows have the largest ``mu``.
        rates = [b["predicted_rate_mean"] for b in bins]
        assert rates == sorted(rates), f"bins not rate-monotone: {rates}"
        # Bin 0 must consist entirely of rate-low (first-half) rows, since
        # rate-ranking puts them before every rate-high row regardless of size.
        # We verify this by checking that bin 0's count of rows is at most the
        # number of rate-low rows (10) — under count-ranking it would include
        # at least some rate-high rows to keep exposure balanced.
        assert bins[0]["count"] <= 10
        # And bin 0's per-row rate ceiling stays inside the rate-low band.
        assert bins[0]["predicted_rate_max"] <= 0.19 + 1e-9

    def test_mean_ranking_ignores_exposure(self):
        """ranking='mean' ranks by raw mu even when exposure is present."""
        y = np.arange(1, 11, dtype=np.float64)
        mu = np.linspace(1.0, 10.0, 10)
        exposure = np.full(10, 1.0)
        # ranking='mean' should produce the same first/last as rate when exposure is constant.
        s = rs.calibration_summary(y, mu, exposure=exposure, n_bins=2, ranking="mean")
        assert s["bins"][0]["expected"] < s["bins"][1]["expected"]

    def test_rate_requires_exposure(self):
        with pytest.raises(ValidationError):
            rs.calibration_summary(
                np.arange(5.0), np.arange(1.0, 6.0), exposure=None, ranking="rate"
            )

    def test_aggregates_are_on_count_scale(self):
        """Bin aggregates are sums of actual/expected/exposure (count-scale)."""
        rng = np.random.default_rng(3)
        n = 300
        mu = rng.uniform(0.1, 5.0, n)
        exposure = rng.uniform(0.1, 2.0, n)
        y = rng.poisson(mu).astype(float)
        s = rs.calibration_summary(y, mu, exposure=exposure, n_bins=5)
        total_actual = sum(b["actual"] for b in s["bins"])
        total_expected = sum(b["expected"] for b in s["bins"])
        total_exposure = sum(b["exposure"] for b in s["bins"])
        np.testing.assert_allclose(total_actual, float(np.sum(y)), rtol=1e-12)
        np.testing.assert_allclose(total_expected, float(np.sum(mu)), rtol=1e-12)
        np.testing.assert_allclose(total_exposure, float(np.sum(exposure)), rtol=1e-12)


# --------------------------------------------------------------------------
# 009.4: per-factor calibration with min-exposure suppression.
# --------------------------------------------------------------------------


class TestFactorCalibration:
    def test_by_factor_aggregates(self):
        """009.4: by= produces per-level actual/expected/exposure/A/E."""
        rng = np.random.default_rng(5)
        n = 600
        levels = np.array(["A", "B", "C"])
        idx = rng.integers(0, 3, n)
        region = levels[idx]
        mu = rng.uniform(0.1, 5.0, n)
        exposure = rng.uniform(0.1, 2.0, n)
        y = rng.poisson(mu).astype(float)
        s = rs.calibration_summary(y, mu, exposure=exposure, by={"Region": region})
        assert "by_factor" in s
        assert "Region" in s["by_factor"]
        rows = s["by_factor"]["Region"]
        # Every level represented; totals reconcile with overall.
        by_level = {row["level"]: row for row in rows}
        assert set(by_level) == {"A", "B", "C"}
        total_actual = sum(row["actual"] for row in rows)
        np.testing.assert_allclose(total_actual, float(np.sum(y)), rtol=1e-12)
        for row in rows:
            mask = region == row["level"]
            np.testing.assert_allclose(row["actual"], float(np.sum(y[mask])), rtol=1e-12)
            np.testing.assert_allclose(row["expected"], float(np.sum(mu[mask])), rtol=1e-12)
            np.testing.assert_allclose(row["exposure"], float(np.sum(exposure[mask])), rtol=1e-12)

    def test_min_exposure_suppression(self):
        """Low-exposure cells flagged ``suppressed=True`` when below min_exposure."""
        rng = np.random.default_rng(6)
        n = 500
        region = np.array(["common"] * 495 + ["rare"] * 5)
        mu = rng.uniform(0.5, 2.0, n)
        exposure = np.where(region == "rare", 0.01, 1.0)
        y = rng.poisson(mu).astype(float)
        s = rs.calibration_summary(
            y, mu, exposure=exposure, by={"Region": region}, min_exposure=0.5
        )
        by_level = {row["level"]: row for row in s["by_factor"]["Region"]}
        assert by_level["common"]["suppressed"] is False
        assert by_level["rare"]["suppressed"] is True


# --------------------------------------------------------------------------
# 009.5: GlobalCalibration object — predict and round-trip.
# --------------------------------------------------------------------------


class TestGlobalCalibration:
    def test_predict_is_factor_times_pred(self):
        """009.5: GlobalCalibration.predict(pred) == factor * pred."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([0.5, 1.0, 1.5, 2.0])
        cal = rs.fit_global_calibration(y, pred)
        # factor == Σy/Σpred == 10/5 == 2.0
        np.testing.assert_allclose(cal.factor, 2.0, rtol=1e-12)
        np.testing.assert_allclose(cal.predict(pred), 2.0 * pred, rtol=1e-12)

    def test_weighted_factor(self):
        rng = np.random.default_rng(13)
        y = rng.uniform(0.0, 5.0, 200)
        pred = rng.uniform(0.1, 5.0, 200)
        w = rng.uniform(0.5, 2.0, 200)
        cal = rs.fit_global_calibration(y, pred, weights=w)
        expected = float(np.sum(w * y) / np.sum(w * pred))
        np.testing.assert_allclose(cal.factor, expected, rtol=1e-12)

    def test_roundtrip_to_dict_from_dict(self):
        cal = rs.fit_global_calibration(
            np.array([2.0, 4.0]), np.array([1.0, 2.0]), weights=np.array([1.0, 2.0])
        )
        state = cal.to_dict()
        cal2 = rs.GlobalCalibration.from_dict(state)
        assert cal2.factor == cal.factor
        assert cal2.method == "global"
        np.testing.assert_allclose(
            cal2.predict(np.linspace(0, 5, 20)),
            cal.predict(np.linspace(0, 5, 20)),
            rtol=1e-12,
        )


# --------------------------------------------------------------------------
# 009.6-7: log-link relevel invariants.
# --------------------------------------------------------------------------


def _fit_log_link_poisson(seed: int = 0):
    df = make_freq_frame(n=2000, seed=seed)
    terms = {
        "DrivAge": {"type": "linear"},
        "VehAge": {"type": "linear"},
        "Region": {"type": "categorical"},
    }
    result = rs.glm_dict(
        response="ClaimCount",
        terms=terms,
        data=df,
        family="poisson",
        exposure="Exposure",
    ).fit()
    return df, result


class TestReleveLogLink:
    def test_non_intercept_params_bit_identical(self):
        """009.6(a): β[1:] is bit-identical after relevel."""
        df, result = _fit_log_link_poisson()
        releveled = result.relevel(data=df)
        # Intercept stays at index 0
        assert result.feature_names[0] == "Intercept"
        assert releveled.feature_names[0] == "Intercept"
        np.testing.assert_array_equal(result.params[1:], releveled.params[1:])

    def test_predictions_balance_unweighted(self):
        """009.6(b): Σμ_new == Σy within tight tolerance."""
        df, result = _fit_log_link_poisson(seed=1)
        releveled = result.relevel(data=df)
        y = df["ClaimCount"].to_numpy()
        mu_new = releveled.predict(df)
        np.testing.assert_allclose(np.sum(mu_new), np.sum(y), rtol=1e-10)

    def test_relativities_unchanged(self):
        """009.7: non-intercept relativities (exp(β_j)) are unchanged."""
        df, result = _fit_log_link_poisson(seed=2)
        releveled = result.relevel(data=df)
        rel_orig = result.relativities()
        rel_new = releveled.relativities()
        # Skip Intercept (row 0); other relativities unchanged.
        np.testing.assert_array_equal(
            rel_orig["Relativity"].to_numpy()[1:],
            rel_new["Relativity"].to_numpy()[1:],
        )

    def test_intercept_shifted_by_log_factor(self):
        """Intercept moves by exactly log(c) where c = Σy/Σμ on the calibration data.

        Use a held-out seed for the calibration frame — fitting Poisson with
        MLE forces ``Σμ == Σy`` on the training set, so a same-seed relevel
        produces only float-noise shifts and can't witness the equation.
        """
        _df_train, result = _fit_log_link_poisson(seed=3)
        df_holdout = make_freq_frame(n=2000, seed=303)
        y = df_holdout["ClaimCount"].to_numpy()
        mu = result.predict(df_holdout)
        c = float(np.sum(y) / np.sum(mu))
        # Confirm the holdout actually exposes a non-trivial calibration gap;
        # otherwise the next assert tests nothing.
        assert abs(c - 1.0) > 1e-4, f"holdout c={c} too close to 1; pick a different seed"
        releveled = result.relevel(data=df_holdout)
        delta = releveled.params[0] - result.params[0]
        np.testing.assert_allclose(delta, float(np.log(c)), rtol=1e-12)

    def test_relevel_serialization_preserves_shift_and_metadata(self):
        """Releveled models must not revert to the raw intercept after to_bytes()."""
        df, result = _fit_log_link_poisson(seed=17)
        calibration_df = df.with_columns((pl.col("ClaimCount") + 1).alias("ClaimCount"))

        releveled = result.relevel(data=calibration_df)
        loaded = rs.GLMModel.from_bytes(releveled.to_bytes())

        np.testing.assert_allclose(loaded.params, releveled.params, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(loaded.predict(df), releveled.predict(df), rtol=1e-12)
        np.testing.assert_allclose(
            loaded.intercept_delta,
            releveled.intercept_delta,
            rtol=0.0,
            atol=0.0,
        )
        assert loaded.relevel_history == releveled.relevel_history

        entry = loaded.relevel_history[-1]
        np.testing.assert_allclose(
            entry["new_intercept"] - entry["original_intercept"],
            entry["log_shift"],
            rtol=1e-12,
        )
        np.testing.assert_allclose(entry["new_intercept"], loaded.params[0], rtol=1e-12)

    def test_returns_new_object_when_not_inplace(self):
        df, result = _fit_log_link_poisson(seed=4)
        releveled = result.relevel(data=df)
        assert releveled is not result
        # Original is untouched.
        y = df["ClaimCount"].to_numpy()
        mu_orig = result.predict(df)
        np.testing.assert_allclose(mu_orig.sum(), result.fittedvalues.sum(), rtol=1e-6)
        assert abs(np.sum(mu_orig) - np.sum(y)) >= 0  # not necessarily balanced

    def test_inplace_mutates_self(self):
        df, result = _fit_log_link_poisson(seed=5)
        intercept_before = result.params[0]
        returned = result.relevel(data=df, inplace=True)
        assert returned is result
        assert result.params[0] != intercept_before

    def test_relevel_updates_cached_fit_evidence(self):
        df, result = _fit_log_link_poisson(seed=6)
        calibration_df = df.with_columns((pl.col("ClaimCount") * 1.2).alias("ClaimCount"))
        releveled = result.relevel(data=calibration_df)
        np.testing.assert_allclose(releveled.fittedvalues, result.fittedvalues * 1.2, rtol=1e-7)
        assert releveled.deviance != pytest.approx(result.deviance)
        assert releveled.llf() != pytest.approx(result.llf())
        assert f"{releveled.deviance:.4f}" in releveled.summary()


# --------------------------------------------------------------------------
# 009.8: non-log link → relevel raises.
# --------------------------------------------------------------------------


class TestReleveNonLogLink:
    def test_gaussian_identity_rejected(self):
        """009.8: relevel() raises clearly for non-log links."""
        df = make_severity_frame(n=500)
        result = rs.glm_dict(
            response="ClaimAmt",
            terms={"DrivAge": {"type": "linear"}, "VehAge": {"type": "linear"}},
            data=df,
            family="gaussian",  # identity link by default
        ).fit()
        assert result.link == "identity"
        with pytest.raises(ValidationError, match="log"):
            result.relevel(data=df)


# --------------------------------------------------------------------------
# 009.9-10: IsotonicCalibration monotonicity, serialization, opt-in.
# --------------------------------------------------------------------------


class TestIsotonicCalibration:
    def test_predictions_are_monotone(self):
        """009.9: isotonic calibration is monotone in pred."""
        rng = np.random.default_rng(21)
        n = 500
        pred = rng.uniform(0.0, 5.0, n)
        # Noisy but monotone-on-average relationship.
        y = pred + rng.normal(0.0, 1.0, n)
        cal = rs.fit_isotonic_calibration(y, pred, increasing=True)
        xs = np.linspace(pred.min(), pred.max(), 100)
        ys = cal.predict(xs)
        diffs = np.diff(ys)
        assert np.all(diffs >= -1e-9), (
            f"isotonic not monotone non-decreasing: min diff = {diffs.min()}"
        )

    def test_serializable_roundtrip(self):
        """009.9: isotonic round-trips through to_dict / from_dict."""
        rng = np.random.default_rng(22)
        pred = rng.uniform(0.0, 5.0, 200)
        y = pred + rng.normal(0.0, 1.0, 200)
        cal = rs.fit_isotonic_calibration(y, pred)
        cal2 = rs.IsotonicCalibration.from_dict(cal.to_dict())
        xs = np.linspace(0.0, 5.0, 50)
        np.testing.assert_allclose(cal.predict(xs), cal2.predict(xs), rtol=1e-12)
        assert cal2.method == "isotonic"
        assert cal.to_dict()["scale"] == "response"

    def test_duplicate_predictions_are_order_invariant(self):
        """Duplicate prediction scores are pooled before PAV, so row order does not matter."""
        pred = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        y = np.array([0.5, 1.5, 1.0, 3.0, 2.0, 4.0])
        order = np.array([1, 0, 3, 2, 5, 4])

        cal_a = rs.fit_isotonic_calibration(y, pred)
        cal_b = rs.fit_isotonic_calibration(y[order], pred[order])

        xs = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        np.testing.assert_allclose(cal_a.predict(xs), cal_b.predict(xs), rtol=1e-12)

    def test_not_applied_implicitly_raw_and_calibrated_both_accessible(self):
        """009.10: fitting a calibration does not modify model.predict()."""
        df, result = _fit_log_link_poisson(seed=8)
        raw = result.predict(df)
        cal = result.fit_calibration(df, method="isotonic")
        calibrated = cal.predict(raw)
        # Raw predictions still come straight off the model.
        raw_again = result.predict(df)
        np.testing.assert_array_equal(raw, raw_again)
        # Calibrated predictions are a separate array of the same shape.
        assert calibrated.shape == raw.shape

    def test_global_calibration_attached_via_fit_calibration(self):
        df, result = _fit_log_link_poisson(seed=9)
        cal = result.fit_calibration(df, method="global")
        assert isinstance(cal, rs.GlobalCalibration)
        # factor balances Σy with Σ(factor·μ).
        y = df["ClaimCount"].to_numpy()
        mu = result.predict(df)
        np.testing.assert_allclose(cal.factor, float(np.sum(y) / np.sum(mu)), rtol=1e-12)


# --------------------------------------------------------------------------
# 009.11: weighted relevel balances and reduces to unweighted when w≡1.
# --------------------------------------------------------------------------


class TestWeightedRelevel:
    def test_weighted_relevel_balances(self):
        """009.11: weighted relevel makes Σ(w·μ_new) == Σ(w·y)."""
        df, result = _fit_log_link_poisson(seed=12)
        weights = np.linspace(0.5, 2.0, df.height)
        releveled = result.relevel(data=df, weights=weights)
        y = df["ClaimCount"].to_numpy()
        mu_new = releveled.predict(df)
        np.testing.assert_allclose(np.sum(weights * mu_new), np.sum(weights * y), rtol=1e-10)

    def test_weights_one_reduces_to_unweighted(self):
        """w ≡ 1 produces the same intercept shift as no weights at all."""
        df, result = _fit_log_link_poisson(seed=13)
        r_unw = result.relevel(data=df)
        r_w1 = result.relevel(data=df, weights=np.ones(df.height))
        np.testing.assert_allclose(r_unw.params, r_w1.params, rtol=1e-12)


# --------------------------------------------------------------------------
# Result-level calibration_summary wrapper.
# --------------------------------------------------------------------------


class TestResultCalibrationSummary:
    def test_result_calibration_summary_matches_predict(self):
        """result.calibration_summary(data) uses result.predict(data) internally."""
        df, result = _fit_log_link_poisson(seed=15)
        summary = result.calibration_summary(df)
        y = df["ClaimCount"].to_numpy()
        mu = result.predict(df)
        np.testing.assert_allclose(
            summary["overall"]["ae_ratio"], float(np.sum(y) / np.sum(mu)), rtol=1e-12
        )

    def test_result_calibration_summary_by_factor(self):
        df, result = _fit_log_link_poisson(seed=16)
        summary = result.calibration_summary(df, by="Region")
        assert "Region" in summary["by_factor"]
        levels = {row["level"] for row in summary["by_factor"]["Region"]}
        assert levels == set(df["Region"].unique().to_list())


# --------------------------------------------------------------------------
# PR11 review — extended coverage.
# --------------------------------------------------------------------------


class TestReleveComposition:
    def test_relevel_twice_accumulates_delta_and_history(self):
        """Composition: applying relevel() twice accumulates intercept_delta
        and grows relevel_history; the cumulative shift is reflected in
        predict(); β[1:] stays bit-identical to the original.
        """
        _df_train, result = _fit_log_link_poisson(seed=31)
        original_params = result.params.copy()

        # Two distinct holdout frames force two non-trivial calibration shifts.
        df_cal1 = make_freq_frame(n=2000, seed=331)
        df_cal2 = make_freq_frame(n=2000, seed=332)

        first = result.relevel(data=df_cal1)
        delta1 = first.intercept_delta
        assert len(first.relevel_history) == 1
        assert first.relevel_history[0]["log_shift"] == pytest.approx(delta1, rel=1e-12)

        second = first.relevel(data=df_cal2)
        delta2 = second.relevel_history[-1]["log_shift"]

        # Cumulative delta == delta1 + delta2 (float-add isn't bit-exact
        # vs. the running accumulator inside relevel, so rtol=1e-10).
        np.testing.assert_allclose(second.intercept_delta, delta1 + delta2, rtol=1e-10)
        # History has length 2 with the correct factor recorded each step.
        assert len(second.relevel_history) == 2
        assert second.relevel_history[0]["log_shift"] == pytest.approx(delta1, rel=1e-12)
        assert second.relevel_history[1]["log_shift"] == pytest.approx(delta2, rel=1e-12)
        # Original is untouched.
        np.testing.assert_array_equal(result.params, original_params)
        # β[1:] still bit-identical to original on both reliveled models.
        np.testing.assert_array_equal(first.params[1:], original_params[1:])
        np.testing.assert_array_equal(second.params[1:], original_params[1:])
        # The intercept on ``second`` reflects the cumulative shift.
        np.testing.assert_allclose(
            second.params[0] - original_params[0],
            delta1 + delta2,
            rtol=1e-10,
        )
        # And predict() on a third frame reflects that cumulative shift.
        df_eval = make_freq_frame(n=500, seed=999)
        mu_orig = result.predict(df_eval)
        mu_second = second.predict(df_eval)
        np.testing.assert_allclose(mu_second, mu_orig * np.exp(delta1 + delta2), rtol=1e-10)


class TestReleveInvariantProperty:
    @pytest.mark.parametrize("seed", [101, 202, 303, 404, 505])
    def test_weighted_balance_invariant_per_seed(self, seed):
        """Property: Σ(w·μ_new) ≈ Σ(w·y) on the calibration frame, for 5 seeds.

        Uses non-uniform weights so the invariant exercises the weighted
        balance rather than the unweighted reduction case.
        """
        df_train, result = _fit_log_link_poisson(seed=seed)
        rng = np.random.default_rng(seed + 1)
        weights = rng.uniform(0.5, 2.0, df_train.height)
        releveled = result.relevel(data=df_train, weights=weights)
        y = df_train["ClaimCount"].to_numpy()
        mu_new = releveled.predict(df_train)
        np.testing.assert_allclose(np.sum(weights * mu_new), np.sum(weights * y), rtol=1e-10)


class TestIsotonicDecreasing:
    def test_decreasing_predictions_are_monotone_non_increasing(self):
        """``IsotonicCalibration(increasing=False)`` produces non-increasing
        predictions over the input range.
        """
        rng = np.random.default_rng(41)
        n = 500
        pred = rng.uniform(0.0, 5.0, n)
        # y trends DOWN with pred: large pred → small y on average.
        y = (5.0 - pred) + rng.normal(0.0, 0.5, n)
        cal = rs.fit_isotonic_calibration(y, pred, increasing=False)
        assert cal.increasing is False
        # Predictions at increasing input must be non-increasing in output.
        xs = np.linspace(pred.min(), pred.max(), 100)
        ys = cal.predict(xs)
        diffs = np.diff(ys)
        assert np.all(diffs <= 1e-9), (
            f"isotonic(increasing=False) not monotone non-increasing: max diff = {diffs.max()}"
        )


class TestGlobalCalibrationPredict:
    def test_fit_calibration_global_predict_matches_factor(self):
        """``result.fit_calibration(method="global").predict(raw)`` ==
        (Σy/Σmu) * raw, within numerical tolerance."""
        df, result = _fit_log_link_poisson(seed=51)
        cal = result.fit_calibration(df, method="global")
        # Predict on raw model predictions.
        raw = result.predict(df)
        calibrated = cal.predict(raw)
        y = df["ClaimCount"].to_numpy()
        expected_factor = float(np.sum(y) / np.sum(raw))
        np.testing.assert_allclose(cal.factor, expected_factor, rtol=1e-12)
        np.testing.assert_allclose(calibrated, expected_factor * raw, rtol=1e-12)


class TestGlobalCalibrationScaleMetadata:
    def test_to_dict_includes_scale_response(self):
        """``to_dict()`` carries ``scale="response"`` (forward-compatibility)."""
        cal = rs.fit_global_calibration(np.array([2.0, 4.0]), np.array([1.0, 2.0]))
        state = cal.to_dict()
        assert state["scale"] == "response"
        # Round-trip preserves the predictions.
        cal2 = rs.GlobalCalibration.from_dict(state)
        np.testing.assert_allclose(
            cal2.predict(np.linspace(0, 5, 20)),
            cal.predict(np.linspace(0, 5, 20)),
            rtol=1e-12,
        )

    def test_from_dict_rejects_unknown_scale(self):
        """Unknown ``scale`` values are rejected with a clear error."""
        state = {
            "method": "global",
            "scale": "link",  # future variant, not yet supported
            "factor": 1.5,
            "n_obs": 100,
            "total_weight": 100.0,
        }
        with pytest.raises(ValidationError, match="scale"):
            rs.GlobalCalibration.from_dict(state)


class TestNoExposureBinKeys:
    def test_no_exposure_bins_emit_predicted_score_not_predicted_rate(self):
        """Without exposure, bin keys are ``predicted_score_*`` (raw mu),
        not ``predicted_rate_*`` — there is no rate denominator to honour
        (RS-ACT-009 review).
        """
        rng = np.random.default_rng(81)
        n = 200
        mu = rng.uniform(0.1, 5.0, n)
        y = rng.poisson(mu).astype(float)
        s = rs.calibration_summary(y, mu, n_bins=4)
        for b in s["bins"]:
            assert "predicted_rate_min" not in b
            assert "predicted_rate_max" not in b
            assert "predicted_rate_mean" not in b
            assert "predicted_score_min" in b
            assert "predicted_score_max" in b
            assert "predicted_score_mean" in b

    def test_with_exposure_bins_emit_predicted_rate(self):
        """With exposure, bin keys retain the ``predicted_rate_*`` set so
        downstream readers keep working unchanged.
        """
        rng = np.random.default_rng(82)
        n = 200
        mu = rng.uniform(0.1, 5.0, n)
        exposure = rng.uniform(0.1, 2.0, n)
        y = rng.poisson(mu).astype(float)
        s = rs.calibration_summary(y, mu, exposure=exposure, n_bins=4)
        for b in s["bins"]:
            assert "predicted_rate_min" in b
            assert "predicted_rate_max" in b
            assert "predicted_rate_mean" in b
            assert "predicted_score_min" not in b
