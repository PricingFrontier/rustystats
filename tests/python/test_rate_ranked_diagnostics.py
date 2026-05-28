"""RS-ACT-004: rate-ranked diagnostics.

When exposure is present, decile / calibration / lift / discrimination
diagnostics rank observations by predicted RATE (mu/exposure), not raw expected
count (mu). Aggregates stay on the count scale. Without exposure the ordering is
unchanged (rate reduces to mu). A ``ranking`` knob exposes mean/rate explicitly.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import rustystats as rs
from _fixtures import make_freq_frame
from rustystats.diagnostics.api import _extract_test_arrays
from rustystats.diagnostics.computer import DiagnosticsComputer, rank_sort_idx


def _computer(with_exposure: bool, seed: int = 0, n: int = 500) -> DiagnosticsComputer:
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.5, 10.0, n)
    # Exposure independent of mu, so mu-order and (mu/exposure)-order differ.
    exposure = rng.uniform(0.1, 20.0, n) if with_exposure else None
    y = rng.poisson(mu).astype(float)
    return DiagnosticsComputer(
        y=y,
        mu=mu,
        linear_predictor=np.log(mu),
        family="poisson",
        n_params=2,
        deviance=100.0,
        exposure=exposure,
    )


def _decile_keys(lift) -> list[tuple]:
    return [(d.actual, d.predicted) for d in lift.deciles]


def _conflicting_mu_exposure() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Raw expected counts sort the first half high; predicted rates sort the
    # second half high. This catches accidental fallback to count-ranking.
    y = np.arange(1, 21, dtype=np.float64)
    mu = np.array(
        [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        dtype=np.float64,
    )
    exposure = np.array([1000.0] * 10 + [1.0] * 10, dtype=np.float64)
    return y, mu, exposure


def _manual_exposure_weighted_gini(
    y: np.ndarray, exposure: np.ndarray, sort_idx: np.ndarray
) -> float:
    total_actual = float(np.sum(y))
    total_exposure = float(np.sum(exposure))
    cum_actual = 0.0
    cum_exposure = 0.0
    prev_actual_pct = 0.0
    prev_exposure_pct = 0.0
    area = 0.0
    for idx in sort_idx[::-1]:
        cum_actual += float(y[idx])
        cum_exposure += float(exposure[idx])
        actual_pct = cum_actual / total_actual
        exposure_pct = cum_exposure / total_exposure
        area += (exposure_pct - prev_exposure_pct) * (actual_pct + prev_actual_pct) / 2.0
        prev_actual_pct = actual_pct
        prev_exposure_pct = exposure_pct
    return 2.0 * area - 1.0


class TestRankingModes:
    def test_auto_equals_rate_when_exposure_present(self):
        """004.1/004.3: auto ranks by rate when exposure exists."""
        comp = _computer(with_exposure=True)
        assert _decile_keys(comp.compute_lift_chart(ranking="auto")) == _decile_keys(
            comp.compute_lift_chart(ranking="rate")
        )

    def test_rate_ranking_differs_from_mean_with_exposure(self):
        """004.1: mu-order and rate-order produce materially different deciles."""
        comp = _computer(with_exposure=True)
        assert _decile_keys(comp.compute_lift_chart(ranking="rate")) != _decile_keys(
            comp.compute_lift_chart(ranking="mean")
        )

    def test_rate_ranked_predicted_rate_increases(self):
        """004.3: ranking by rate orders deciles by increasing predicted rate."""
        comp = _computer(with_exposure=True)
        predicted_rates = [d.predicted for d in comp.compute_lift_chart(ranking="rate").deciles]
        assert predicted_rates[0] < predicted_rates[-1]
        # Mean-ranking does not order by rate, so its first/last need not separate.
        mean_rates = [d.predicted for d in comp.compute_lift_chart(ranking="mean").deciles]
        assert predicted_rates != mean_rates

    def test_no_exposure_auto_equals_mean(self):
        """004.4: without exposure the ordering is unchanged (rate == mu)."""
        comp = _computer(with_exposure=False)
        assert _decile_keys(comp.compute_lift_chart(ranking="auto")) == _decile_keys(
            comp.compute_lift_chart(ranking="mean")
        )

    def test_ranking_rate_requires_exposure(self):
        comp = _computer(with_exposure=False)
        with pytest.raises(rs.ValidationError, match=r"(?i)exposure"):
            comp.compute_lift_chart(ranking="rate")

    def test_invalid_ranking_raises(self):
        comp = _computer(with_exposure=True)
        with pytest.raises(rs.ValidationError, match=r"(?i)ranking"):
            comp.compute_lift_chart(ranking="nonsense")

    def test_rank_sort_idx_ties_match_rust_index_tiebreak(self):
        """004.4: Python pre-sorts preserve Rust's score-then-index ordering."""
        mu = np.array([2.0, 1.0, 1.0, 2.0, 1.0])
        assert rank_sort_idx(mu, ranking="mean").tolist() == [1, 2, 4, 0, 3]

    def test_mean_ranking_gini_uses_mean_rank_order(self):
        """004.3: lift-chart Gini follows the requested ranking mode."""
        y, mu, exposure = _conflicting_mu_exposure()
        comp = DiagnosticsComputer(
            y=y,
            mu=mu,
            linear_predictor=np.log(mu),
            family="poisson",
            n_params=2,
            deviance=100.0,
            exposure=exposure,
        )
        mean_chart = comp.compute_lift_chart(ranking="mean")
        rate_chart = comp.compute_lift_chart(ranking="rate")
        mean_sort = rank_sort_idx(mu, exposure, has_exposure=True, ranking="mean")
        expected_mean_gini = _manual_exposure_weighted_gini(y, exposure, mean_sort)

        assert mean_chart.gini == pytest.approx(round(expected_mean_gini, 3))
        assert mean_chart.gini != rate_chart.gini

    def test_test_data_extract_uses_rate_order_when_exposure_present(self):
        """004.2: train/test diagnostics pre-sort held-out rows by rate, not raw mu."""
        y, mu, exposure = _conflicting_mu_exposure()
        test_data = pl.DataFrame({"y": y, "Exposure": exposure})

        class _PredictOnly:
            def predict(self, data):
                return mu

        _, _, _, sort_idx = _extract_test_arrays(
            test_data,
            _PredictOnly(),
            response_col="y",
            exposure_col="Exposure",
            ranking="auto",
        )

        assert (
            sort_idx.tolist()
            == rank_sort_idx(mu, exposure, has_exposure=True, ranking="auto").tolist()
        )
        assert sort_idx.tolist() != np.argsort(mu, kind="stable").tolist()


class TestRateRankedDiagnosticsEndToEnd:
    def test_diagnostics_run_rate_ranked_with_exposure(self):
        """End-to-end: a model with exposure produces rate-ranked diagnostics."""
        df = make_freq_frame()
        result = rs.glm_dict(
            response="ClaimCount",
            terms={"DrivAge": {"type": "linear"}, "VehAge": {"type": "linear"}},
            data=df,
            family="poisson",
            exposure="Exposure",
        ).fit()
        diag = result.diagnostics(df, compute_lift=True)
        assert diag is not None
        lift = diag.lift_chart
        assert lift is not None
        assert len(lift.deciles) == 10
        # Aggregates stay on the count scale: per-decile exposure sums to total.
        assert sum(d.exposure for d in lift.deciles) == pytest.approx(
            float(df["Exposure"].sum()), abs=0.1
        )

    def test_public_diagnostics_exposes_ranking_mode(self):
        """004.R: public diagnostics accepts explicit ranking modes."""
        df = make_freq_frame()
        result = rs.glm_dict(
            response="ClaimCount",
            terms={"DrivAge": {"type": "linear"}, "VehAge": {"type": "linear"}},
            data=df,
            family="poisson",
            exposure="Exposure",
        ).fit()
        common = {
            "compute_vif": False,
            "compute_coefficients": False,
            "compute_deviance_by_level": False,
            "compute_partial_dep": False,
            "compute_robust_se": False,
            "compute_score_tests": False,
        }
        mean_diag = result.diagnostics(df, ranking="mean", **common)
        rate_diag = result.diagnostics(df, ranking="rate", **common)

        assert mean_diag.lift_chart is not None
        assert rate_diag.lift_chart is not None
        assert _decile_keys(mean_diag.lift_chart) != _decile_keys(rate_diag.lift_chart)


# --------------------------------------------------------------------------
# 004.5 — weighted-aggregates: bin sums match Σy / Σmu / Σexposure (treating
# exposure as the prior weight per the spec convention).
# --------------------------------------------------------------------------


class TestWeightedAggregates:
    def test_bins_aggregate_to_hand_computed_sums(self):
        """004.5: per-bin actual/expected/exposure sums to Σy / Σmu / Σexposure.

        We don't pass ``weights=`` explicitly because :mod:`calibration` treats
        exposure as the prior weight (per the spec note: "the implementation
        does not thread prior weights; treat exposure as the weight"). The
        invariant is that the bin partitions the rows, so summing each
        column over all bins recovers the totals.
        """
        rng = np.random.default_rng(123)
        n = 200
        # Non-uniform exposure to ensure exposure-balanced binning is doing
        # actual work.
        exposure = rng.uniform(0.05, 5.0, n)
        # mu independent of exposure so rate (mu/exposure) is well-spread.
        mu = rng.uniform(0.1, 10.0, n)
        y = rng.poisson(mu).astype(np.float64)

        summary = rs.calibration_summary(y, mu, exposure=exposure, n_bins=10)
        bins = summary["bins"]
        assert len(bins) >= 1

        total_actual = sum(b["actual"] for b in bins)
        total_expected = sum(b["expected"] for b in bins)
        total_exposure = sum(b["exposure"] for b in bins)

        # Sums reconcile with Σy, Σmu, Σexposure — every row is placed in
        # exactly one bin and aggregates stay on the count scale.
        np.testing.assert_allclose(total_actual, float(np.sum(y)), rtol=1e-12)
        np.testing.assert_allclose(total_expected, float(np.sum(mu)), rtol=1e-12)
        np.testing.assert_allclose(total_exposure, float(np.sum(exposure)), rtol=1e-12)

        # Overall A/E ratio is also Σy / Σmu (exposure-weighted under the
        # convention).
        overall = summary["overall"]
        np.testing.assert_allclose(overall["actual"], float(np.sum(y)), rtol=1e-12)
        np.testing.assert_allclose(overall["expected"], float(np.sum(mu)), rtol=1e-12)
        np.testing.assert_allclose(overall["ae_ratio"], float(np.sum(y) / np.sum(mu)), rtol=1e-12)


# --------------------------------------------------------------------------
# 004.3 — Lorenz curve rate-ranking. compute_lorenz_curve historically sorted
# on raw mu; with RS-ACT-004 it sorts on mu/exposure when exposure is present.
# --------------------------------------------------------------------------


class TestLorenzRateRanking:
    def test_lorenz_curve_uses_rate_with_exposure(self):
        """004.3: Lorenz curve sorts ascending by mu/exposure when exposure is
        supplied — so its path through (cumulative_exposure_pct,
        cumulative_actual_pct) space differs from the raw-mu ordering.
        """
        from rustystats._rustystats import compute_lorenz_curve_py

        # High-exposure rows have large mu but low rate; low-exposure rows
        # have small mu but high rate (rate-ranking inverts mu-ranking).
        y, mu, exposure = _conflicting_mu_exposure()

        rate_curve = compute_lorenz_curve_py(y, mu, exposure, 10)
        # Pass exposure=None to force the mean-ranking baseline.
        mean_curve = compute_lorenz_curve_py(y, mu, None, 10)

        # Both must close at (1, 1, 1); but the path differs because the
        # ordering of rows along the cumulative exposure axis is reversed.
        assert rate_curve[-1]["cumulative_exposure_pct"] == pytest.approx(1.0, abs=1e-9)
        assert mean_curve[-1]["cumulative_exposure_pct"] == pytest.approx(1.0, abs=1e-9)

        # Pick a mid-point and confirm divergence: under rate ranking the
        # high-exposure rows come first (cumulative_exposure_pct grows fast
        # but cumulative_actual_pct stays low); under mean ranking the small
        # mu rows come first (cumulative_actual_pct stays low *and*
        # cumulative_exposure_pct stays low, since each row has exposure 1).
        rate_predicted = [p["cumulative_predicted_pct"] for p in rate_curve]
        mean_predicted = [p["cumulative_predicted_pct"] for p in mean_curve]
        assert rate_predicted != mean_predicted, (
            "Lorenz curve should respond to exposure-vs-mean ranking; "
            "if these match, the rate ranking is not being applied."
        )

    def test_lorenz_curve_falls_back_to_mu_without_exposure(self):
        """Sanity: without exposure, the Lorenz curve uses raw mu (rate == mu)."""
        from rustystats._rustystats import compute_lorenz_curve_py

        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        mu = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64)

        curve = compute_lorenz_curve_py(y, mu, None, 5)
        # Endpoints are (0,0,0) and (1,1,1).
        assert curve[0]["cumulative_exposure_pct"] == pytest.approx(0.0, abs=1e-9)
        assert curve[-1]["cumulative_exposure_pct"] == pytest.approx(1.0, abs=1e-9)
        assert curve[-1]["cumulative_actual_pct"] == pytest.approx(1.0, abs=1e-9)
        assert curve[-1]["cumulative_predicted_pct"] == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------
# 004 review: zero-exposure guard. ``rank_sort_idx`` must not produce inf/nan
# rank keys when an exposure row is zero (or non-positive); it should fall
# back to mu[i] for those rows, mirroring the Rust ``predicted_rate`` helper.
# --------------------------------------------------------------------------


class TestZeroExposureGuard:
    def test_zero_exposure_falls_back_to_mu(self):
        """e[i] <= 0 → rank key uses mu[i] instead of inf/nan."""
        mu = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        exposure = np.array([1.0, 0.0, 2.0, 1.0], dtype=np.float64)
        # Should not raise (no division-by-zero warning, no inf/nan).
        idx = rank_sort_idx(mu, exposure, has_exposure=True, ranking="auto")
        # Resulting order must be a permutation of [0..n).
        assert sorted(idx.tolist()) == [0, 1, 2, 3]
        # And the rank keys are finite — verify by reconstructing the key.
        safe_exp = np.where(exposure > 0.0, exposure, 1.0)
        key = np.where(exposure > 0.0, mu / safe_exp, mu)
        assert np.all(np.isfinite(key))
        # Sorted-by-key reproduces ``idx``.
        np.testing.assert_array_equal(idx, np.argsort(key, kind="stable"))

    def test_rate_ranking_with_zero_exposure_is_finite(self):
        """ranking='rate' with a zero-exposure row stays finite."""
        mu = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        exposure = np.array([1.0, 0.0, 2.0], dtype=np.float64)
        idx = rank_sort_idx(mu, exposure, has_exposure=True, ranking="rate")
        assert sorted(idx.tolist()) == [0, 1, 2]
