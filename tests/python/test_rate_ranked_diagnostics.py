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
