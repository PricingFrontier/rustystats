"""RS-ACT-004: rate-ranked diagnostics.

When exposure is present, decile / calibration / lift / discrimination
diagnostics rank observations by predicted RATE (mu/exposure), not raw expected
count (mu). Aggregates stay on the count scale. Without exposure the ordering is
unchanged (rate reduces to mu). A ``ranking`` knob exposes mean/rate explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest
import rustystats as rs
from _fixtures import make_freq_frame
from rustystats.diagnostics.computer import DiagnosticsComputer


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
