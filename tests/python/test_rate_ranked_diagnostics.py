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
    y: np.ndarray,
    exposure: np.ndarray,
    sort_idx: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=np.float64)
    total_actual = float(np.sum(w * y))
    total_exposure = float(np.sum(w * exposure))
    cum_actual = 0.0
    cum_exposure = 0.0
    prev_actual_pct = 0.0
    prev_exposure_pct = 0.0
    area = 0.0
    for idx in sort_idx[::-1]:
        cum_actual += float(w[idx] * y[idx])
        cum_exposure += float(w[idx] * exposure[idx])
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
            def predict(self, data, **kwargs):
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
        """004.5 (via the PR11 ``calibration_summary`` primitive): per-bin
        actual/expected/exposure sums to Σy / Σmu / Σexposure.

        This exercises ``rs.calibration_summary`` (RS-ACT-009), the
        weighted-aware primitive. We don't pass ``weights=`` here; calibration
        treats exposure as the rate denominator. The invariant is that the bins
        partition the rows, so summing each column over all bins recovers the
        totals.
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


class TestWeightedDecileLiftAggregates:
    """Backlog #1 (RS-ACT-004): prior weights give Σw·y / Σw·μ / Σw·exposure
    decile/lift aggregates; uniform weights reproduce the unweighted path."""

    def _comp(self, weights, seed: int = 0, n: int = 600) -> DiagnosticsComputer:
        rng = np.random.default_rng(seed)
        mu = rng.uniform(0.5, 10.0, n)
        exposure = rng.uniform(0.1, 20.0, n)
        y = rng.poisson(mu).astype(float)
        return DiagnosticsComputer(
            y=y,
            mu=mu,
            linear_predictor=np.log(mu),
            family="poisson",
            n_params=2,
            deviance=100.0,
            exposure=exposure,
            weights=weights,
        )

    def test_uniform_weights_match_unweighted(self):
        # Same seed -> identical y/mu/exposure; weights=ones must reproduce the
        # unweighted lift chart, gini, and (Python vs Rust) A/E-by-decile.
        unw = self._comp(None)
        ones = self._comp(np.ones(600))
        l0, l1 = unw.compute_lift_chart(), ones.compute_lift_chart()
        assert _decile_keys(l0) == _decile_keys(l1)
        assert l0.gini == pytest.approx(l1.gini)
        idx = unw._rank_sort_idx("auto")
        d0 = unw._compute_ae_by_decile(unw.y, unw.mu, unw.exposure, sort_idx=idx)
        d1 = ones._compute_ae_by_decile(
            ones.y, ones.mu, ones.exposure, sort_idx=idx, weights=np.ones(600)
        )
        for a, b in zip(d0, d1):
            assert a.actual == pytest.approx(b.actual)
            assert a.predicted == pytest.approx(b.predicted)
            assert a.exposure == pytest.approx(b.exposure)

    def test_weights_change_aggregates_to_hand_computed(self):
        rng = np.random.default_rng(1)
        w = rng.uniform(0.1, 5.0, 600)
        comp = self._comp(w)
        lift = comp.compute_lift_chart()
        # First decile's weighted exposure on the same equal-count bins.
        idx = comp._rank_sort_idx("auto")
        first = idx[: 600 // 10]
        exp_hand = float(np.sum((w * comp.exposure)[first]))
        assert lift.deciles[0].exposure == pytest.approx(round(exp_hand, 2))
        # ...and the weighted chart differs from the unweighted one.
        unw = self._comp(None)
        assert any(
            abs(a.exposure - b.exposure) > 1e-6
            for a, b in zip(lift.deciles, unw.compute_lift_chart().deciles)
        )
        expected_gini = _manual_exposure_weighted_gini(comp.y, comp.exposure, idx, weights=w)
        assert lift.gini == pytest.approx(round(expected_gini, 3))

    def test_fitted_weights_auto_propagate_through_diagnostics(self):
        rng = np.random.default_rng(2)
        df = make_freq_frame(n=2000, seed=7)
        w = rng.uniform(0.1, 5.0, df.height)
        df = df.with_columns(pl.Series("w", w))
        result = rs.glm_dict(
            response="ClaimCount",
            terms={"DrivAge": {"type": "linear"}, "Region": {"type": "categorical"}},
            data=df,
            family="poisson",
            exposure="Exposure",
            weights="w",
        ).fit()
        common = dict(
            continuous_factors=["DrivAge"],
            compute_vif=False,
            compute_deviance_by_level=False,
            compute_partial_dep=False,
            compute_robust_se=False,
            compute_score_tests=False,
            compute_coefficients=False,
        )
        auto = result.diagnostics(df, **common)  # fitted "w" auto-propagates
        explicit = result.diagnostics(df, weights=w, **common)  # same weights, explicit
        forced_unit = result.diagnostics(df, weights=np.ones(df.height), **common)  # override
        for a, b in zip(auto.lift_chart.deciles, explicit.lift_chart.deciles):
            assert a.actual == pytest.approx(b.actual)
            assert a.exposure == pytest.approx(b.exposure)
        assert any(
            abs(a.exposure - b.exposure) > 1e-6
            for a, b in zip(auto.lift_chart.deciles, forced_unit.lift_chart.deciles)
        )

    def test_dataset_gini_matches_weighted_lift_gini(self):
        """Weighted train DatasetDiagnostics.gini and lift_chart.gini share the
        same prior-weighted Lorenz calculation."""
        rng = np.random.default_rng(3)
        df = make_freq_frame(n=2000, seed=8)
        df = df.with_columns(pl.Series("w", rng.uniform(0.1, 5.0, df.height)))
        result = rs.glm_dict(
            response="ClaimCount",
            terms={"DrivAge": {"type": "linear"}, "Region": {"type": "categorical"}},
            data=df,
            family="poisson",
            exposure="Exposure",
            weights="w",
        ).fit()
        diag = result.diagnostics(
            df,
            continuous_factors=["DrivAge"],
            compute_vif=False,
            compute_deviance_by_level=False,
            compute_partial_dep=False,
            compute_robust_se=False,
            compute_score_tests=False,
            compute_coefficients=False,
        )

        assert diag.lift_chart is not None
        assert diag.train_test is not None
        assert diag.train_test.train.gini == pytest.approx(diag.lift_chart.gini, abs=5e-4)

    def test_test_data_uses_test_weights_column(self):
        """Held-out diagnostics use the held-out weights column, not train weights."""
        train = make_freq_frame(n=200, seed=1).with_columns(
            pl.Series("w", np.linspace(1.0, 2.0, 200))
        )
        test = make_freq_frame(n=250, seed=2).with_columns(
            pl.Series("w", np.linspace(3.0, 4.0, 250))
        )
        result = rs.glm_dict(
            response="ClaimCount",
            terms={"DrivAge": {"type": "linear"}, "Region": {"type": "categorical"}},
            data=train,
            family="poisson",
            exposure="Exposure",
            weights="w",
        ).fit()
        diag = result.diagnostics(
            train,
            test_data=test,
            continuous_factors=["DrivAge"],
            compute_vif=False,
            compute_deviance_by_level=False,
            compute_lift=False,
            compute_partial_dep=False,
            compute_robust_se=False,
            compute_score_tests=False,
            compute_coefficients=False,
        )

        assert diag.train_test is not None
        assert diag.train_test.test is not None
        mu_test = np.asarray(result.predict(test), dtype=np.float64)
        exposure_test = test["Exposure"].to_numpy().astype(np.float64)
        weights_test = test["w"].to_numpy().astype(np.float64)
        idx = rank_sort_idx(mu_test, exposure_test, has_exposure=True, ranking="auto")
        first = idx[: test.height // 10]
        expected_exposure = float(np.sum(weights_test[first] * exposure_test[first]))

        assert diag.train_test.test.ae_by_decile[0].exposure == pytest.approx(
            round(expected_exposure, 2)
        )


class TestBasePredictionDiagnostics:
    def test_base_predictions_by_role_use_prior_weighted_totals(self):
        train = make_freq_frame(n=300, seed=11).with_columns(
            pl.Series("w", np.linspace(0.5, 2.0, 300))
        )
        test = make_freq_frame(n=240, seed=12).with_columns(
            pl.Series("w", np.linspace(1.0, 3.0, 240))
        )
        result = rs.glm_dict(
            response="ClaimCount",
            terms={"DrivAge": {"type": "linear"}, "Region": {"type": "categorical"}},
            data=train,
            family="poisson",
            exposure="Exposure",
            weights="w",
        ).fit()
        train_base = np.asarray(result.predict(train), dtype=np.float64) * 1.1
        test_base = np.asarray(result.predict(test), dtype=np.float64) * 0.9
        train = train.with_columns(pl.Series("base_mu", train_base))
        test = test.with_columns(pl.Series("base_mu", test_base))

        diag = result.diagnostics(
            train,
            test_data=test,
            categorical_factors=["Region"],
            continuous_factors=["DrivAge"],
            base_predictions="base_mu",
            weights="w",
            compute_vif=False,
            compute_deviance_by_level=False,
            compute_lift=False,
            compute_partial_dep=False,
            compute_robust_se=False,
            compute_score_tests=False,
            compute_coefficients=False,
        )

        assert diag.base_predictions_by_role is not None
        assert diag.base_predictions_by_role.train is not None
        assert diag.base_predictions_by_role.test is not None
        expected_train_base = float(np.sum(train["w"].to_numpy() * train_base))
        expected_test_base = float(np.sum(test["w"].to_numpy() * test_base))
        assert diag.base_predictions_by_role.train.base_metrics.total_predicted == pytest.approx(
            round(expected_train_base, 2)
        )
        assert diag.base_predictions_by_role.test.base_metrics.total_predicted == pytest.approx(
            round(expected_test_base, 2)
        )

    def test_factor_bins_include_weighted_totals_and_base_overlay(self):
        df = make_freq_frame(n=320, seed=13).with_columns(
            pl.Series("w", np.linspace(0.25, 2.5, 320))
        )
        result = rs.glm_dict(
            response="ClaimCount",
            terms={"DrivAge": {"type": "linear"}, "Region": {"type": "categorical"}},
            data=df,
            family="poisson",
            exposure="Exposure",
            weights="w",
        ).fit()
        mu = np.asarray(result.predict(df), dtype=np.float64)
        base = mu * 1.2
        df = df.with_columns(pl.Series("base_mu", base))

        diag = result.diagnostics(
            df,
            categorical_factors=["Region"],
            base_predictions="base_mu",
            weights="w",
            compute_vif=False,
            compute_deviance_by_level=False,
            compute_lift=False,
            compute_partial_dep=False,
            compute_robust_se=False,
            compute_score_tests=False,
            compute_coefficients=False,
        )

        region = next(f for f in diag.factors if f.name == "Region")
        # The rare bucket is labeled "_Other" (matching the Rust kernel); skip it
        # so ``first`` is a real Region level we can mask the source rows by.
        first = next(b for b in region.actual_vs_expected if b.bin != "_Other")
        mask = df["Region"].to_numpy() == first.bin
        w = df["w"].to_numpy().astype(np.float64)
        y = df["ClaimCount"].to_numpy().astype(np.float64)
        exposure = df["Exposure"].to_numpy().astype(np.float64)
        assert first.actual_total == pytest.approx(float(np.sum(w[mask] * y[mask])))
        assert first.expected_total == pytest.approx(float(np.sum(w[mask] * mu[mask])))
        assert first.base_expected_total == pytest.approx(float(np.sum(w[mask] * base[mask])))
        assert first.exposure == pytest.approx(round(float(np.sum(w[mask] * exposure[mask])), 2))

    def test_role_specific_base_prediction_columns_and_missing_test_warning(self):
        train = make_freq_frame(n=180, seed=14)
        test = make_freq_frame(n=160, seed=15)
        result = rs.glm_dict(
            response="ClaimCount",
            terms={"DrivAge": {"type": "linear"}, "Region": {"type": "categorical"}},
            data=train,
            family="poisson",
            exposure="Exposure",
        ).fit()
        train = train.with_columns(
            pl.Series("base_train", np.asarray(result.predict(train)) * 1.05)
        )
        test = test.with_columns(pl.Series("base_test", np.asarray(result.predict(test)) * 0.95))

        diag = result.diagnostics(
            train,
            test_data=test,
            base_predictions={"train": "base_train", "test": "base_test"},
            compute_vif=False,
            compute_deviance_by_level=False,
            compute_lift=False,
            compute_partial_dep=False,
            compute_robust_se=False,
            compute_score_tests=False,
            compute_coefficients=False,
        )
        assert diag.base_predictions_by_role is not None
        assert diag.base_predictions_by_role.test is not None

        missing = result.diagnostics(
            train,
            test_data=test.drop("base_test"),
            base_predictions="base_train",
            compute_vif=False,
            compute_deviance_by_level=False,
            compute_lift=False,
            compute_partial_dep=False,
            compute_robust_se=False,
            compute_score_tests=False,
            compute_coefficients=False,
        )
        assert missing.base_predictions_by_role is not None
        assert missing.base_predictions_by_role.test is None
        assert any(w["type"] == "base_predictions_unavailable" for w in missing.warnings)


# Common compute_* flags that strip diagnostics down to the factor A/E tables,
# keeping the parity tests below fast and focused.
_AE_ONLY = dict(
    compute_vif=False,
    compute_deviance_by_level=False,
    compute_lift=False,
    compute_partial_dep=False,
    compute_robust_se=False,
    compute_score_tests=False,
    compute_coefficients=False,
)


class TestBasePredictionBinningParity:
    """Regression guards for the weighted/base A/E path. Both the unweighted and
    the weighted/base cases now flow through the same Rust kernel, so supplying
    ``base_predictions=`` or ``weights=`` must not change the model's own A/E
    binning, labels, or confidence intervals.
    """

    def test_zero_claim_bin_gets_proper_one_sided_ci(self):
        """A populated zero-claim bin must surface a Wilson-Poisson interval with
        a positive upper bound, not a degenerate ``[0, 0]``."""
        from rustystats._rustystats import compute_ae_continuous_py

        # The lowest-x bin has all-zero claims against a positive expectation.
        x = np.arange(20.0)
        y = np.where(x < 4, 0.0, 1.0)
        mu = np.full(20, 0.5)
        exposure = np.ones(20)
        first = compute_ae_continuous_py(x, y, mu, exposure, 5, "poisson")[0]
        assert first["actual_sum"] == 0.0 and first["predicted_sum"] > 0.0
        assert first["ae_ci_lower"] == 0.0
        assert first["ae_ci_upper"] > 0.0

    def test_base_predictions_does_not_change_factor_binning(self):
        """The Rust path (no base) and Python fallback (with base) must produce
        the same bin labels, counts, and A/E ratios for every factor."""
        df = make_freq_frame(n=400, seed=7)
        result = rs.glm_dict(
            response="ClaimCount",
            terms={"DrivAge": {"type": "linear"}, "Region": {"type": "categorical"}},
            data=df,
            family="poisson",
            exposure="Exposure",
        ).fit()
        df = df.with_columns(pl.Series("base_mu", np.asarray(result.predict(df)) * 1.1))

        shared = dict(categorical_factors=["Region"], continuous_factors=["DrivAge"], **_AE_ONLY)
        rust_path = result.diagnostics(df, **shared)  # no base -> Rust kernel
        py_path = result.diagnostics(df, base_predictions="base_mu", **shared)  # base -> Python

        def bins(diag, name):
            f = next(f for f in diag.factors if f.name == name)
            return f.actual_vs_expected

        for name in ("DrivAge", "Region"):
            rust_bins, py_bins = bins(rust_path, name), bins(py_path, name)
            assert [(b.bin, b.n) for b in rust_bins] == [(b.bin, b.n) for b in py_bins]
            for rb, pb in zip(rust_bins, py_bins):
                assert pb.ae_ratio == pytest.approx(rb.ae_ratio, abs=0.01)
                assert pb.ae_ci == pytest.approx(rb.ae_ci, abs=0.01)

    def test_weighted_categorical_rare_bucket_uses_underscore_other(self):
        """The folded rare bucket is labeled ``_Other`` (Rust convention), not
        ``Other``, on the weighted/base Python path."""
        rng = np.random.default_rng(99)
        n = 1500
        cat = rng.integers(0, 30, n).astype(str)  # 30 levels > max_levels=20 -> rare bucket
        exposure = rng.uniform(0.5, 1.5, n)
        y = rng.poisson(np.exp(-2.0) * exposure)
        df = pl.DataFrame({"y": y, "cat": cat, "exposure": exposure})
        result = rs.glm_dict(
            response="y",
            terms={"cat": {"type": "categorical"}},
            data=df,
            family="poisson",
            exposure="exposure",
        ).fit()
        df = df.with_columns(pl.Series("base_mu", np.asarray(result.predict(df))))

        diag = result.diagnostics(
            df, categorical_factors=["cat"], base_predictions="base_mu", **_AE_ONLY
        )
        labels = [
            b.bin for b in next(f for f in diag.factors if f.name == "cat").actual_vs_expected
        ]
        assert "_Other" in labels
        assert "Other" not in labels

    def test_constant_weights_preserve_continuous_partial_dep(self):
        """``partial_dep`` is a weighted mean (Σwμ/Σw); a constant weight must
        leave it (and the band ranges) equal to the unweighted Rust path."""
        df = make_freq_frame(n=400, seed=8)
        result = rs.glm_dict(
            response="ClaimCount",
            terms={"DrivAge": {"type": "linear"}},
            data=df,
            family="poisson",
            exposure="Exposure",
        ).fit()
        unweighted = result.diagnostics(df, continuous_factors=["DrivAge"], **_AE_ONLY)
        df_w = df.with_columns(pl.Series("w", np.full(df.height, 3.0)))
        weighted = result.diagnostics(df_w, continuous_factors=["DrivAge"], weights="w", **_AE_ONLY)

        rb = unweighted.train_test.train.continuous_diagnostics["DrivAge"]
        wb = weighted.train_test.train.continuous_diagnostics["DrivAge"]
        assert [b.band for b in rb] == [b.band for b in wb]
        for r, w in zip(rb, wb):
            assert (w.range_min, w.range_max) == (r.range_min, r.range_max)
            assert w.partial_dep == pytest.approx(r.partial_dep, rel=1e-9)
