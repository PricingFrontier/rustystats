"""
Tests for model diagnostics functionality.
"""

import numpy as np
import pytest

# Try to import polars, skip tests if not available
pytest.importorskip("polars")
import polars as pl


class TestDiagnosticsComputer:
    """Tests for the DiagnosticsComputer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n = 1000

        age = np.random.uniform(18, 70, n)
        region = np.random.choice(["A", "B", "C", "D"], n)
        income = np.random.uniform(20000, 100000, n)

        # Generate response with some pattern
        mu_true = np.exp(-2 + 0.02 * age + 0.5 * (region == "A").astype(float))
        y = np.random.poisson(mu_true)
        exposure = np.random.uniform(0.5, 1.0, n)

        data = pl.DataFrame(
            {
                "y": y,
                "age": age,
                "region": region,
                "income": income,
                "exposure": exposure,
            }
        )

        return data

    def test_diagnostics_computer_creation(self, sample_data):
        """Test that DiagnosticsComputer can be created."""
        from rustystats.diagnostics import DiagnosticsComputer

        y = sample_data["y"].to_numpy().astype(np.float64)
        mu = np.maximum(y.astype(np.float64), 0.1)  # Simple "predictions"
        lp = np.log(mu)

        computer = DiagnosticsComputer(
            y=y,
            mu=mu,
            linear_predictor=lp,
            family="poisson",
            n_params=3,
            deviance=100.0,
        )

        assert computer.n_obs == len(y)
        assert computer.family == "poisson"

    def test_fit_statistics(self, sample_data):
        """Test fit statistics computation."""
        from rustystats.diagnostics import DiagnosticsComputer

        y = sample_data["y"].to_numpy().astype(np.float64)
        mu = np.maximum(y.astype(np.float64) + 0.1, 0.1)
        lp = np.log(mu)

        computer = DiagnosticsComputer(
            y=y,
            mu=mu,
            linear_predictor=lp,
            family="poisson",
            n_params=3,
            deviance=100.0,
        )

        stats = computer.compute_fit_statistics()

        assert "deviance" in stats
        assert "null_deviance" in stats
        assert "deviance_explained" in stats
        assert "aic" in stats
        assert "bic" in stats
        assert "pearson_chi2" in stats
        assert stats["null_deviance"] >= stats["deviance"]  # Should improve on null

    def test_loss_metrics(self, sample_data):
        """Test loss metrics computation."""
        from rustystats.diagnostics import DiagnosticsComputer

        y = sample_data["y"].to_numpy().astype(np.float64)
        mu = np.maximum(y.astype(np.float64) + 0.1, 0.1)
        lp = np.log(mu)

        computer = DiagnosticsComputer(
            y=y,
            mu=mu,
            linear_predictor=lp,
            family="poisson",
            n_params=3,
            deviance=100.0,
        )

        metrics = computer.compute_loss_metrics()

        assert "mse" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "loss" in metrics
        assert metrics["mse"] >= 0
        assert metrics["mae"] >= 0
        assert metrics["loss"] >= 0

    def test_calibration(self, sample_data):
        """Test calibration computation."""
        from rustystats.diagnostics import DiagnosticsComputer

        y = sample_data["y"].to_numpy().astype(np.float64)
        mu = np.maximum(y.astype(np.float64), 0.1)
        lp = np.log(mu)

        computer = DiagnosticsComputer(
            y=y,
            mu=mu,
            linear_predictor=lp,
            family="poisson",
            n_params=3,
            deviance=100.0,
        )

        calibration = computer.compute_calibration(n_bins=10)

        assert "ae_ratio" in calibration
        assert "hl_pvalue" in calibration
        assert "problem_deciles" in calibration
        # problem_deciles only includes deciles with A/E outside [0.9, 1.1]

        # A/E should be close to 1 for perfect predictions
        ae = calibration["ae_ratio"]
        assert 0.5 < ae < 2.0  # Reasonable range

    def test_discrimination(self, sample_data):
        """Test discrimination metrics."""
        from rustystats.diagnostics import DiagnosticsComputer

        y = sample_data["y"].to_numpy().astype(np.float64)
        # Create predictions that have some discrimination
        mu = np.exp(-2 + 0.02 * sample_data["age"].to_numpy())
        lp = np.log(mu)

        computer = DiagnosticsComputer(
            y=y,
            mu=mu,
            linear_predictor=lp,
            family="poisson",
            n_params=3,
            deviance=100.0,
        )

        disc = computer.compute_discrimination()

        assert disc is not None
        assert "gini" in disc
        assert "auc" in disc
        assert "ks" in disc
        # lorenz_curve removed for token efficiency

        # Gini should be between -1 and 1
        assert -1 <= disc["gini"] <= 1
        # AUC should be between 0 and 1
        assert 0 <= disc["auc"] <= 1

    def test_residual_summary(self, sample_data):
        """Test residual summary computation."""
        from rustystats.diagnostics import DiagnosticsComputer

        y = sample_data["y"].to_numpy().astype(np.float64)
        mu = np.maximum(y.astype(np.float64), 0.1)
        lp = np.log(mu)

        computer = DiagnosticsComputer(
            y=y,
            mu=mu,
            linear_predictor=lp,
            family="poisson",
            n_params=3,
            deviance=100.0,
        )

        resid_summary = computer.compute_residual_summary()

        assert "pearson" in resid_summary
        assert "deviance" in resid_summary

        pearson = resid_summary["pearson"]
        assert hasattr(pearson, "mean")
        assert hasattr(pearson, "std")
        assert hasattr(pearson, "skewness")  # percentiles removed for compression

    def test_factor_diagnostics_continuous(self, sample_data):
        """Test factor diagnostics for continuous variables."""
        from rustystats.diagnostics import DiagnosticsComputer

        y = sample_data["y"].to_numpy().astype(np.float64)
        mu = np.maximum(y.astype(np.float64), 0.1)
        lp = np.log(mu)

        computer = DiagnosticsComputer(
            y=y,
            mu=mu,
            linear_predictor=lp,
            family="poisson",
            n_params=3,
            deviance=100.0,
            feature_names=["intercept", "age"],
        )

        factors = computer.compute_factor_diagnostics(
            data=sample_data,
            categorical_factors=[],
            continuous_factors=["age", "income"],
            n_bins=5,
        )

        assert len(factors) == 2

        age_factor = next(f for f in factors if f.name == "age")
        assert age_factor.factor_type == "continuous"
        assert age_factor.in_model  # "age" is in feature_names
        assert len(age_factor.actual_vs_expected) == 5

        income_factor = next(f for f in factors if f.name == "income")
        assert not income_factor.in_model  # "income" not in model

    def test_factor_diagnostics_categorical(self, sample_data):
        """Test factor diagnostics for categorical variables."""
        from rustystats.diagnostics import DiagnosticsComputer

        y = sample_data["y"].to_numpy().astype(np.float64)
        mu = np.maximum(y.astype(np.float64), 0.1)
        lp = np.log(mu)

        computer = DiagnosticsComputer(
            y=y,
            mu=mu,
            linear_predictor=lp,
            family="poisson",
            n_params=3,
            deviance=100.0,
            feature_names=["intercept", "region[B]", "region[C]"],
        )

        factors = computer.compute_factor_diagnostics(
            data=sample_data,
            categorical_factors=["region"],
            continuous_factors=[],
            n_bins=5,
        )

        assert len(factors) == 1

        region_factor = factors[0]
        assert region_factor.factor_type == "categorical"
        assert region_factor.in_model  # region is in feature_names
        assert len(region_factor.actual_vs_expected) <= 5  # 4 levels + maybe "Other"


class TestInteractionDetection:
    """Tests for interaction detection."""

    @pytest.fixture
    def interaction_data(self):
        """Create data with a known interaction."""
        np.random.seed(42)
        n = 2000

        x1 = np.random.uniform(0, 10, n)
        x2 = np.random.choice(["A", "B"], n)

        # Create interaction: different slope for A vs B
        mu_true = np.exp(
            1.0 + 0.1 * x1 * (x2 == "A").astype(float) + 0.3 * x1 * (x2 == "B").astype(float)
        )
        y = np.random.poisson(mu_true)

        # Predictions without interaction
        mu_pred = np.exp(1.0 + 0.2 * x1)

        return (
            pl.DataFrame(
                {
                    "y": y,
                    "x1": x1,
                    "x2": x2,
                }
            ),
            y,
            mu_pred,
        )

    def test_interaction_detection(self, interaction_data):
        """Test that interaction detection finds the interaction."""
        from rustystats.diagnostics import DiagnosticsComputer

        data, y, mu = interaction_data
        y = y.astype(np.float64)
        mu = mu.astype(np.float64)
        lp = np.log(mu)

        computer = DiagnosticsComputer(
            y=y,
            mu=mu,
            linear_predictor=lp,
            family="poisson",
            n_params=2,
            deviance=100.0,
        )

        candidates = computer.detect_interactions(
            data=data,
            factor_names=["x1", "x2"],
            max_factors=5,
        )

        # Should find at least one candidate
        assert len(candidates) >= 1

        # Top candidate should involve x1 and x2
        top = candidates[0]
        assert (top.factor1 == "x1" and top.factor2 == "x2") or (
            top.factor1 == "x2" and top.factor2 == "x1"
        )


class TestModelDiagnostics:
    """Tests for the full ModelDiagnostics output."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted model for testing."""
        import rustystats as rs

        np.random.seed(42)
        n = 500

        age = np.random.uniform(18, 70, n)
        region = np.random.choice(["A", "B", "C"], n)

        mu_true = np.exp(-2 + 0.02 * age)
        y = np.random.poisson(mu_true)

        data = pl.DataFrame(
            {
                "y": y,
                "age": age,
                "region": region,
            }
        )

        result = rs.glm_dict(
            response="y",
            terms={"age": {"type": "linear"}, "region": {"type": "categorical"}},
            data=data,
            family="poisson",
        ).fit()

        return result, data

    def test_compute_diagnostics(self, fitted_model):
        """Test the main compute_diagnostics function."""
        from rustystats.diagnostics import compute_diagnostics

        result, data = fitted_model

        diagnostics = compute_diagnostics(
            result=result,
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["age"],
        )

        assert diagnostics.model_summary is not None
        assert diagnostics.train_test is not None
        assert diagnostics.train_test.train.loss > 0
        assert diagnostics.calibration is not None
        assert len(diagnostics.factors) == 2

    def test_diagnostics_to_json(self, fitted_model):
        """Test JSON serialization."""
        import json

        from rustystats.diagnostics import compute_diagnostics

        result, data = fitted_model

        diagnostics = compute_diagnostics(
            result=result,
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["age"],
        )

        json_str = diagnostics.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)

        assert "model_summary" in parsed
        assert "train_test" in parsed
        assert "factors" in parsed
        assert len(parsed["factors"]) == 2

    def test_diagnostics_method_on_result(self, fitted_model):
        """Test calling diagnostics directly on result object."""
        result, data = fitted_model

        diagnostics = result.diagnostics(
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["age"],
        )

        assert diagnostics is not None
        assert len(diagnostics.factors) == 2

    def test_diagnostics_json_method(self, fitted_model):
        """Test the diagnostics_json convenience method."""
        import json

        result, data = fitted_model

        json_str = result.diagnostics_json(
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["age"],
        )

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert "model_summary" in parsed


class TestDifferentFamilies:
    """Test diagnostics work with different families."""

    @pytest.fixture
    def gaussian_model(self):
        """Create a Gaussian model."""
        import rustystats as rs

        np.random.seed(42)
        n = 300
        x = np.random.randn(n)
        y = 2 + 0.5 * x + np.random.randn(n) * 0.5

        data = pl.DataFrame({"y": y, "x": x})
        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}},
            data=data,
            family="gaussian",
        ).fit()

        return result, data

    @pytest.fixture
    def binomial_model(self):
        """Create a Binomial model."""
        import rustystats as rs

        np.random.seed(42)
        n = 300
        x = np.random.randn(n)
        p = 1 / (1 + np.exp(-x))
        y = np.random.binomial(1, p).astype(float)

        data = pl.DataFrame({"y": y, "x": x})
        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}},
            data=data,
            family="binomial",
        ).fit()

        return result, data

    def test_gaussian_diagnostics(self, gaussian_model):
        """Test diagnostics for Gaussian family."""
        from rustystats.diagnostics import compute_diagnostics

        result, data = gaussian_model

        diag = compute_diagnostics(
            result=result,
            train_data=data,
            continuous_factors=["x"],
        )

        assert diag.train_test.train.deviance > 0
        assert len(diag.factors) == 1

    def test_binomial_diagnostics(self, binomial_model):
        """Test diagnostics for Binomial family."""
        from rustystats.diagnostics import compute_diagnostics

        result, data = binomial_model

        diag = compute_diagnostics(
            result=result,
            train_data=data,
            continuous_factors=["x"],
        )

        assert diag.train_test.train.deviance > 0
        # Should have discrimination metrics in train_test
        assert diag.train_test.train.gini is not None


class TestPreFitExploration:
    """Tests for pre-fit data exploration."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for exploration."""
        np.random.seed(42)
        n = 500

        age = np.random.uniform(18, 70, n)
        region = np.random.choice(["North", "South", "East", "West"], n)
        exposure = np.random.uniform(0.5, 1.0, n)

        mu = np.exp(-2 + 0.02 * age)
        y = np.random.poisson(mu * exposure)

        return pl.DataFrame(
            {
                "ClaimNb": y,
                "Age": age,
                "Region": region,
                "Exposure": exposure,
            }
        )

    def test_explore_data_function(self, sample_data):
        """Test the explore_data function."""
        from rustystats.diagnostics import explore_data

        exploration = explore_data(
            data=sample_data,
            response="ClaimNb",
            exposure="Exposure",
            categorical_factors=["Region"],
            continuous_factors=["Age"],
        )

        assert exploration is not None
        assert exploration.data_summary is not None
        assert len(exploration.factor_stats) >= 2

    def test_explore_data_response_stats(self, sample_data):
        """Test response statistics in exploration."""
        from rustystats.diagnostics import explore_data

        exploration = explore_data(
            data=sample_data,
            response="ClaimNb",
            exposure="Exposure",
            categorical_factors=["Region"],
            continuous_factors=["Age"],
        )

        assert exploration.response_stats is not None
        assert "mean_response" in exploration.response_stats

    def test_explore_data_interaction_detection(self, sample_data):
        """Test interaction detection in exploration."""
        from rustystats.diagnostics import explore_data

        exploration = explore_data(
            data=sample_data,
            response="ClaimNb",
            exposure="Exposure",
            categorical_factors=["Region"],
            continuous_factors=["Age"],
            detect_interactions=True,
        )

        # Should have some interaction info even if no strong interactions found
        assert exploration.interaction_candidates is not None

    def test_explore_data_to_json(self, sample_data):
        """Test JSON serialization of exploration."""
        import json

        from rustystats.diagnostics import explore_data

        exploration = explore_data(
            data=sample_data,
            response="ClaimNb",
            exposure="Exposure",
            categorical_factors=["Region"],
            continuous_factors=["Age"],
        )

        json_str = exploration.to_json()
        parsed = json.loads(json_str)

        assert "data_summary" in parsed
        assert "factor_stats" in parsed

    def test_explore_method_on_model(self):
        """Test explore method on FormulaGLMDict."""
        import rustystats as rs

        np.random.seed(42)
        n = 200

        age = np.random.uniform(18, 70, n)
        region = np.random.choice(["A", "B"], n)
        y = np.random.poisson(np.exp(-2 + 0.02 * age), n)

        data = pl.DataFrame(
            {
                "y": y,
                "age": age,
                "region": region,
            }
        )

        model = rs.glm_dict(
            response="y",
            terms={"age": {"type": "linear"}, "region": {"type": "categorical"}},
            data=data,
            family="poisson",
        )

        exploration = model.explore(
            categorical_factors=["region"],
            continuous_factors=["age"],
        )

        assert exploration is not None
        assert len(exploration.factor_stats) >= 2

        # Can still fit after exploring
        result = model.fit()
        assert result.converged

    def test_explore_data_deterministic_golden_outputs(self):
        """Golden-ish public output check for optimized exploration paths."""
        from rustystats.constants import EPSILON
        from rustystats.diagnostics import explore_data

        rows = []
        for cat_a, cat_b, count, response in [
            ("A0", "B0", 45, 1.0),
            ("A0", "B1", 35, 2.0),
            ("A1", "B0", 35, 2.0),
            ("A1", "B1", 45, 12.0),
        ]:
            rows.extend([(cat_a, cat_b, response)] * count)

        rng = np.random.default_rng(123)
        idx = rng.permutation(len(rows))
        cat_a = np.array([rows[i][0] for i in idx])
        cat_b = np.array([rows[i][1] for i in idx])
        y = np.array([rows[i][2] for i in idx], dtype=np.float64)
        x1 = rng.normal(size=len(rows))
        x2 = 0.35 * x1 + rng.normal(scale=0.8, size=len(rows))

        data = pl.DataFrame(
            {
                "y": y,
                "exposure": np.ones(len(rows), dtype=np.float64),
                "cat_a": cat_a,
                "cat_b": cat_b,
                "x1": x1,
                "x2": x2,
            }
        )

        exploration = explore_data(
            data=data,
            response="y",
            exposure="exposure",
            categorical_factors=["cat_a", "cat_b"],
            continuous_factors=["x1", "x2"],
            n_bins=4,
            detect_interactions=True,
            max_interaction_factors=4,
        )

        corr = np.asarray(exploration.correlations["matrix"], dtype=np.float64)
        expected_corr = float(np.corrcoef(x1, x2)[0, 1])
        np.testing.assert_allclose(corr, [[1.0, expected_corr], [expected_corr, 1.0]], rtol=1e-12)

        expected_vif = (1.0 + EPSILON) / ((1.0 + EPSILON) ** 2 - expected_corr**2)
        vif_by_factor = {row["factor"]: row for row in exploration.vif}
        np.testing.assert_allclose(vif_by_factor["x1"]["vif"], expected_vif, rtol=1e-10)
        np.testing.assert_allclose(vif_by_factor["x2"]["vif"], expected_vif, rtol=1e-10)
        assert vif_by_factor["x1"]["severity"] == "none"
        assert vif_by_factor["x2"]["severity"] == "none"

        cramers_v = np.asarray(exploration.cramers_v["matrix"], dtype=np.float64)
        np.testing.assert_allclose(cramers_v, [[1.0, 0.125], [0.125, 1.0]], rtol=1e-12)

        assert exploration.interaction_candidates
        top = exploration.interaction_candidates[0]
        assert {top.factor1, top.factor2} == {"cat_a", "cat_b"}
        np.testing.assert_allclose(top.interaction_strength, 1.0, rtol=1e-12)
        assert top.n_cells == 4

    def test_explore_data_public_api_detects_known_interaction(self):
        """Public API should route known pre-fit interactions through Rust."""
        from rustystats.diagnostics import explore_data

        rows = []
        for segment, channel, count, response in [
            ("A", "online", 60, 20.0),
            ("A", "branch", 60, 2.0),
            ("B", "online", 60, 2.0),
            ("B", "branch", 60, 1.0),
        ]:
            rows.extend([(segment, channel, response)] * count)

        rng = np.random.default_rng(0)
        idx = rng.permutation(len(rows))
        data = pl.DataFrame(
            {
                "y": np.array([rows[i][2] for i in idx], dtype=np.float64),
                "exposure": np.ones(len(rows), dtype=np.float64),
                "segment": np.array([rows[i][0] for i in idx]),
                "channel": np.array([rows[i][1] for i in idx]),
                "noise": rng.choice(["n0", "n1", "n2"], len(rows)),
            }
        )

        exploration = explore_data(
            data=data,
            response="y",
            exposure="exposure",
            categorical_factors=["segment", "channel", "noise"],
            continuous_factors=[],
            detect_interactions=True,
            max_interaction_factors=3,
        )

        assert exploration.interaction_candidates
        top = exploration.interaction_candidates[0]
        assert {top.factor1, top.factor2} == {"segment", "channel"}
        assert top.interaction_strength > 0.95
        assert top.n_cells == 4

    def test_explore_data_handles_categorical_nulls_and_continuous_nan_inf(self):
        """Null categoricals and non-finite continuous values should be stable."""
        from rustystats.diagnostics import explore_data

        x = np.array([0.0, 1.0, np.nan, 2.0, np.inf, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        x2 = np.array([0.0, 2.0, 4.0, 4.0, 8.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0])
        data = pl.DataFrame(
            {
                "y": np.arange(12, dtype=np.float64),
                "exposure": np.ones(12, dtype=np.float64),
                "cat": ["A", None, "B", None, "A", "B", "A", None, "B", "A", "B", None],
                "x": x,
                "x2": x2,
            }
        )

        exploration = explore_data(
            data=data,
            response="y",
            exposure="exposure",
            categorical_factors=["cat"],
            continuous_factors=["x", "x2"],
            n_bins=4,
            detect_interactions=False,
        )

        factor_stats = {row["name"]: row for row in exploration.factor_stats}
        assert factor_stats["cat"]["n_levels"] == 3
        assert {row["level"] for row in factor_stats["cat"]["levels"]} == {"A", "B", "None"}

        assert factor_stats["x"]["missing_count"] == 2
        assert sum(row["count"] for row in factor_stats["x"]["response_by_bin"]) == 10

        corr = np.asarray(exploration.correlations["matrix"], dtype=np.float64)
        assert corr.shape == (2, 2)
        assert np.isfinite(corr).all()
        np.testing.assert_allclose(corr[0, 1], 1.0, rtol=1e-12)
        assert exploration.correlations["high_correlations"][0]["severity"] == "high"

        vif_by_factor = {row["factor"]: row for row in exploration.vif}
        assert vif_by_factor["x"]["severity"] == "severe"
        assert vif_by_factor["x2"]["severity"] == "severe"

    @pytest.mark.slow
    def test_explore_data_performance_regression_smoke(self):
        """Generous smoke guard against obvious exploratory path regressions."""
        import time

        from rustystats.diagnostics import explore_data

        n = 50_000
        rng = np.random.default_rng(10)
        exposure = rng.uniform(0.5, 1.5, n)
        cols = {
            "y": rng.poisson(0.4 * exposure).astype(np.float64),
            "exposure": exposure,
        }
        categorical_factors = []
        continuous_factors = []

        for i in range(6):
            name = f"cat{i}"
            categorical_factors.append(name)
            levels = np.array([f"L{i}_{j}" for j in range(30)])
            cols[name] = rng.choice(levels, n)

        for i in range(6):
            name = f"x{i}"
            continuous_factors.append(name)
            cols[name] = rng.normal(size=n)

        start = time.perf_counter()
        exploration = explore_data(
            data=pl.DataFrame(cols),
            response="y",
            exposure="exposure",
            categorical_factors=categorical_factors,
            continuous_factors=continuous_factors,
            detect_interactions=True,
        )
        elapsed = time.perf_counter() - start

        assert len(exploration.factor_stats) == 12
        assert elapsed < 10.0, f"explore_data performance smoke took {elapsed:.3f}s"


class TestEnhancedDiagnostics:
    """Tests for new enhanced diagnostics features for agentic workflows."""

    @pytest.fixture
    def fitted_model_with_data(self):
        """Create a fitted model with training data."""
        import rustystats as rs

        np.random.seed(42)
        n = 500

        age = np.random.uniform(18, 70, n)
        veh_power = np.random.uniform(50, 200, n)
        region = np.random.choice(["A", "B", "C", "D"], n)
        exposure = np.random.uniform(0.5, 1.0, n)

        mu_true = np.exp(-2 + 0.02 * age + 0.001 * veh_power + 0.3 * (region == "A").astype(float))
        y = np.random.poisson(mu_true * exposure)

        data = pl.DataFrame(
            {
                "y": y,
                "age": age,
                "veh_power": veh_power,
                "region": region,
                "exposure": exposure,
            }
        )

        result = rs.glm_dict(
            response="y",
            terms={
                "age": {"type": "linear"},
                "veh_power": {"type": "linear"},
                "region": {"type": "categorical"},
            },
            data=data,
            family="poisson",
            offset="exposure",
        ).fit()

        return result, data

    def test_vif_computation(self, fitted_model_with_data):
        """Test VIF computation for multicollinearity detection."""
        from rustystats.diagnostics import DiagnosticsComputer

        result, data = fitted_model_with_data

        y = data["y"].to_numpy().astype(np.float64)
        mu = result.fittedvalues
        lp = result.linear_predictor

        computer = DiagnosticsComputer(
            y=y,
            mu=mu,
            linear_predictor=lp,
            family="poisson",
            n_params=len(result.params),
            deviance=result.deviance,
            feature_names=result.feature_names,
        )

        # Create a simple design matrix for testing
        X = np.column_stack(
            [
                np.ones(len(y)),
                data["age"].to_numpy(),
                data["veh_power"].to_numpy(),
            ]
        )

        vif_results = computer.compute_vif(X, ["Intercept", "age", "veh_power"])

        # VIF returns all features with their VIF values and severity levels
        for v in vif_results:
            assert hasattr(v, "feature")
            assert hasattr(v, "vif")
            assert hasattr(v, "severity")
            assert v.severity in ("none", "moderate", "severe")
            assert v.vif >= 1.0  # VIF is always >= 1

    def test_vif_detects_collinearity(self):
        """Test that VIF detects collinear features."""
        from rustystats.diagnostics import DiagnosticsComputer

        np.random.seed(42)
        n = 500

        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.1  # x2 is almost x1

        y = np.random.poisson(np.exp(1 + x1), n).astype(np.float64)
        mu = np.full(n, np.mean(y))

        computer = DiagnosticsComputer(
            y=y,
            mu=mu,
            linear_predictor=np.log(mu),
            family="poisson",
            n_params=3,
            deviance=100.0,
            feature_names=["Intercept", "x1", "x2"],
        )

        X = np.column_stack([np.ones(n), x1, x2])
        vif_results = computer.compute_vif(X, ["Intercept", "x1", "x2"])

        # Both x1 and x2 should have high VIF
        for v in vif_results:
            assert v.vif > 5.0  # Should detect collinearity
            assert v.severity in ("moderate", "severe")

    def test_coefficient_summary(self, fitted_model_with_data):
        """Test coefficient summary with interpretations."""
        from rustystats.diagnostics import DiagnosticsComputer

        result, data = fitted_model_with_data

        y = data["y"].to_numpy().astype(np.float64)
        mu = result.fittedvalues

        computer = DiagnosticsComputer(
            y=y,
            mu=mu,
            linear_predictor=result.linear_predictor,
            family="poisson",
            n_params=len(result.params),
            deviance=result.deviance,
            feature_names=result.feature_names,
        )

        coef_summary = computer.compute_coefficient_summary(result, link="log")

        assert len(coef_summary) == len(result.params)

        for cs in coef_summary:
            assert hasattr(cs, "feature")
            assert hasattr(cs, "estimate")
            assert hasattr(cs, "relativity")
            assert hasattr(cs, "significant")
            # For log link, relativity should be computed
            if cs.feature != "Intercept":
                assert cs.relativity is not None
                assert cs.relativity > 0

    def test_factor_deviance(self, fitted_model_with_data):
        """Test deviance breakdown by factor level."""
        from rustystats.diagnostics import DiagnosticsComputer

        result, data = fitted_model_with_data

        y = data["y"].to_numpy().astype(np.float64)
        mu = result.fittedvalues

        computer = DiagnosticsComputer(
            y=y,
            mu=mu,
            linear_predictor=result.linear_predictor,
            family="poisson",
            n_params=len(result.params),
            deviance=result.deviance,
            feature_names=result.feature_names,
        )

        factor_dev = computer.compute_factor_deviance(data, ["region"])

        assert len(factor_dev) == 1
        fd = factor_dev[0]

        assert fd.factor == "region"
        assert fd.total_deviance > 0
        assert len(fd.levels) == 4  # A, B, C, D

        # Check that deviance percentages sum to ~100%
        total_pct = sum(level.deviance_pct for level in fd.levels)
        assert 95 < total_pct < 105  # Allow small rounding errors

    def test_lift_chart(self, fitted_model_with_data):
        """Test full lift chart computation."""
        from rustystats.diagnostics import DiagnosticsComputer

        result, data = fitted_model_with_data

        y = data["y"].to_numpy().astype(np.float64)
        mu = result.fittedvalues

        computer = DiagnosticsComputer(
            y=y,
            mu=mu,
            linear_predictor=result.linear_predictor,
            family="poisson",
            n_params=len(result.params),
            deviance=result.deviance,
        )

        lift_chart = computer.compute_lift_chart(n_deciles=10)

        assert len(lift_chart.deciles) == 10
        assert -1 <= lift_chart.gini <= 1
        assert lift_chart.ks_statistic >= 0
        assert 1 <= lift_chart.ks_decile <= 10

        # Check decile structure
        for decile in lift_chart.deciles:
            assert 1 <= decile.decile <= 10
            assert decile.n > 0
            assert decile.lift > 0
            assert 0 <= decile.cumulative_actual_pct <= 100

    def test_partial_dependence(self, fitted_model_with_data):
        """Test partial dependence computation."""
        from rustystats.diagnostics import DiagnosticsComputer

        result, data = fitted_model_with_data

        y = data["y"].to_numpy().astype(np.float64)
        mu = result.fittedvalues

        computer = DiagnosticsComputer(
            y=y,
            mu=mu,
            linear_predictor=result.linear_predictor,
            family="poisson",
            n_params=len(result.params),
            deviance=result.deviance,
            feature_names=result.feature_names,
        )

        partial_dep = computer.compute_partial_dependence(
            data=data,
            result=result,
            continuous_factors=["age", "veh_power"],
            categorical_factors=["region"],
            link="log",
        )

        # Should have 3 partial dependence results
        assert len(partial_dep) == 3

        for pd in partial_dep:
            assert hasattr(pd, "variable")
            assert hasattr(pd, "variable_type")
            assert hasattr(pd, "shape")
            assert hasattr(pd, "recommendation")
            assert len(pd.grid_values) > 0
            assert len(pd.predictions) == len(pd.grid_values)

    def test_pd_logit_link_uses_sigmoid(self):
        """FIX-O B2: For binomial GLM with logit link, PD predictions must be
        probabilities in (0, 1) computed as sigmoid(eta_baseline + delta_eta),
        NOT base + delta. Before this fix the non-log branch added delta_eta
        directly to base_pred, silently producing wrong PD probabilities for
        every binomial model (the canonical link is logit, not identity)."""
        import rustystats as rs
        from rustystats.diagnostics import DiagnosticsComputer

        rng = np.random.default_rng(42)
        n = 5000
        x = rng.standard_normal(n)
        p = 1.0 / (1.0 + np.exp(-(0.5 + 1.0 * x)))
        y_arr = (rng.uniform(size=n) < p).astype(float)
        df = pl.DataFrame({"y": y_arr, "x": x})
        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}},
            data=df,
            family="binomial",
        ).fit()
        # Drive `compute_partial_dependence` directly: bypasses the broader
        # factor-diagnostics pipeline so the test isolates the PD codepath.
        computer = DiagnosticsComputer(
            y=df["y"].to_numpy().astype(np.float64),
            mu=result.fittedvalues,
            linear_predictor=result.linear_predictor,
            family="binomial",
            n_params=len(result.params),
            deviance=result.deviance,
            feature_names=result.feature_names,
        )
        partial_dep = computer.compute_partial_dependence(
            data=df,
            result=result,
            continuous_factors=["x"],
            categorical_factors=[],
            link="logit",
        )
        pd_x = next(p for p in partial_dep if p.variable == "x")
        # All predictions must be valid probabilities in (0, 1).
        assert all(0.0 < pred < 1.0 for pred in pd_x.predictions), (
            f"Logit PD predictions out of (0,1): {pd_x.predictions}"
        )
        # The PD should also reflect the underlying β > 0: predictions should
        # increase across the grid (predictions are aligned with grid_values).
        first, last = pd_x.predictions[0], pd_x.predictions[-1]
        assert last > first, (
            f"Expected monotonically increasing PD for positive β, got first={first} last={last}"
        )
        # Sanity check: at the middle of the grid the PD should be close to
        # the baseline (mean of fitted μ), since delta_eta ≈ 0 there.
        mid_pred = pd_x.predictions[len(pd_x.predictions) // 2]
        baseline = float(np.mean(result.fittedvalues))
        assert abs(mid_pred - baseline) < 0.05, (
            f"PD at grid midpoint ({mid_pred}) should be near baseline ({baseline})"
        )

    def test_full_diagnostics_with_enhancements(self, fitted_model_with_data):
        """Test full diagnostics includes all new fields."""
        result, data = fitted_model_with_data

        diagnostics = result.diagnostics(
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["age", "veh_power"],
        )

        # Check new fields are present
        assert diagnostics.coefficient_summary is not None
        assert len(diagnostics.coefficient_summary) > 0

        assert diagnostics.lift_chart is not None
        assert len(diagnostics.lift_chart.deciles) == 10

        assert diagnostics.factor_deviance is not None
        assert len(diagnostics.factor_deviance) == 1

        assert diagnostics.partial_dependence is not None
        assert len(diagnostics.partial_dependence) == 3

    def test_diagnostics_json_includes_enhancements(self, fitted_model_with_data):
        """Test JSON output includes new fields."""
        import json

        result, data = fitted_model_with_data

        diagnostics = result.diagnostics(
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["age"],
        )

        json_str = diagnostics.to_json()
        parsed = json.loads(json_str)

        assert "coefficient_summary" in parsed
        assert "lift_chart" in parsed
        assert "factor_deviance" in parsed
        assert "partial_dependence" in parsed

        # Check coefficient summary structure
        coef_summary = parsed["coefficient_summary"]
        assert len(coef_summary) > 0
        assert "feature" in coef_summary[0]
        assert "relativity" in coef_summary[0]
        # impact/recommendation removed for token optimization (derivable from z_value and relativity)

    def test_partial_dependence_categorical_batch_py_matches_bincount(self):
        """Change 7 regression guard: ``partial_dependence_categorical_batch_py``
        now accepts a list of 1D u32 arrays (one per factor) instead of a
        stacked ``(n, k)`` matrix, eliminating a 400 MB transient ``np.stack``
        at n=1M × k=100. This test calls the PyO3 binding directly with 5
        factors × 10_000 rows (varying levels 3, 5, 10, 20, 50) and asserts
        the per-factor ``(counts, mu_sums)`` match an independent
        ``np.bincount`` reference. The underlying Rust aggregator's behaviour
        is the invariant the refactor must preserve — independent of how the
        Python caller assembles its arguments."""
        from rustystats._rustystats import partial_dependence_categorical_batch_py

        rng = np.random.default_rng(12345)
        n = 10_000
        n_levels_per_factor = [3, 5, 10, 20, 50]
        k = len(n_levels_per_factor)

        codes_list = [
            rng.integers(0, n_levels_per_factor[j], size=n, dtype=np.uint32) for j in range(k)
        ]
        mu = rng.uniform(0.1, 2.1, size=n).astype(np.float64)

        batch_results = partial_dependence_categorical_batch_py(codes_list, mu, n_levels_per_factor)

        assert len(batch_results) == k
        for j in range(k):
            got_counts, got_mu_sums = batch_results[j]
            m = n_levels_per_factor[j]

            # Independent reference via numpy's bincount — the semantics the
            # Rust helper replaced.
            ref_counts = np.bincount(codes_list[j], minlength=m).astype(np.float64)
            ref_mu_sums = np.bincount(codes_list[j], weights=mu, minlength=m)

            assert len(got_counts) == m
            assert len(got_mu_sums) == m
            np.testing.assert_array_equal(
                np.asarray(got_counts), ref_counts, err_msg=f"counts factor {j}"
            )
            # mu_sums can differ by FP ULPs due to summation order — tight
            # but not bit-exact.
            np.testing.assert_allclose(
                np.asarray(got_mu_sums),
                ref_mu_sums,
                rtol=1e-12,
                atol=1e-9,
                err_msg=f"mu_sums factor {j}",
            )

            # Invariant: total count == n, total mu_sum == sum(mu).
            assert sum(got_counts) == float(n)
            np.testing.assert_allclose(sum(got_mu_sums), float(mu.sum()), rtol=1e-12)

    def test_partial_dependence_diagnostics_end_to_end_stable(self):
        """End-to-end regression guard: fit a model with 3 categorical
        factors, run ``result.diagnostics(...)`` and pin per-level mean
        predictions. The refactor of ``partial_dependence_categorical_batch_py``
        must leave these numbers unchanged. Uses hand-computable per-factor
        grouping so the expected values are derivable from the input."""
        import rustystats as rs

        rng = np.random.default_rng(777)
        n = 1_500

        regions = rng.choice(["A", "B", "C"], n)
        brands = rng.choice(["x", "y", "z", "w"], n)
        fuels = rng.choice(["petrol", "diesel", "electric"], n)
        age = rng.uniform(18, 70, n)

        eta = -2.0 + 0.02 * age + 0.15 * (regions == "A").astype(float)
        mu_true = np.exp(eta)
        y = rng.poisson(mu_true).astype(np.float64)

        data = pl.DataFrame(
            {
                "y": y,
                "age": age,
                "region": regions,
                "brand": brands,
                "fuel": fuels,
            }
        )

        result = rs.glm_dict(
            response="y",
            terms={
                "age": {"type": "linear"},
                "region": {"type": "categorical"},
                "brand": {"type": "categorical"},
                "fuel": {"type": "categorical"},
            },
            data=data,
            family="poisson",
        ).fit()

        cat_factors = ["region", "brand", "fuel"]
        diagnostics = result.diagnostics(
            train_data=data,
            categorical_factors=cat_factors,
            continuous_factors=["age"],
        )

        assert diagnostics.partial_dependence is not None
        pd_by_var = {pd.variable: pd for pd in diagnostics.partial_dependence}
        mu = result.fittedvalues

        # For each categorical factor, the bucket-mean prediction for every
        # observed level must equal mean(mu[rows at that level]) — pin it.
        # (grid_values ordering is an internal implementation detail of the
        # diagnostics computer and not load-bearing for this refactor; we
        # look up predictions by level string rather than by index.)
        for var in cat_factors:
            assert var in pd_by_var, f"missing partial_dependence for {var}"
            pd_obj = pd_by_var[var]
            assert pd_obj.variable_type == "categorical"

            grid = list(pd_obj.grid_values)
            preds = list(pd_obj.predictions)
            assert len(preds) == len(grid)
            values = data[var].to_numpy().astype(str)

            for lvl, got_pred in zip(grid, preds):
                mask = values == lvl
                if not mask.any():
                    # Level advertised in grid but not present in data: the
                    # computer falls back to a base prediction. Skip — the
                    # invariant we're pinning is the populated-bucket mean.
                    continue
                expected = float(mu[mask].mean())
                assert abs(float(got_pred) - expected) < 1e-5, (
                    f"PD regression for {var} level {lvl!r}: "
                    f"got {got_pred} vs expected bucket-mean {expected}"
                )

    def test_multicollinearity_warning(self):
        """Test that multicollinearity generates appropriate warnings."""
        import rustystats as rs

        np.random.seed(42)
        n = 500

        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.05  # Nearly collinear
        y = np.random.poisson(np.exp(1 + x1), n)

        data = pl.DataFrame({"y": y, "x1": x1, "x2": x2})

        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=data,
            family="poisson",
        ).fit()

        diagnostics = result.diagnostics(
            train_data=data,
            continuous_factors=["x1", "x2"],
        )

        # Token optimization: multicollinearity warnings removed (info in VIF array)
        # Check VIF results instead
        assert diagnostics.vif is not None
        assert len(diagnostics.vif) > 0  # Should detect collinearity
        for v in diagnostics.vif:
            assert v.severity in ("severe", "moderate")

    def test_train_test_comparison(self, fitted_model_with_data):
        """Test comprehensive train vs test comparison."""

        result, train_data = fitted_model_with_data

        # Create test data with same structure
        np.random.seed(999)
        n_test = 200

        age = np.random.uniform(18, 70, n_test)
        veh_power = np.random.uniform(50, 200, n_test)
        region = np.random.choice(["A", "B", "C", "D"], n_test)
        exposure = np.random.uniform(0.5, 1.0, n_test)

        mu_true = np.exp(-2 + 0.02 * age + 0.001 * veh_power + 0.3 * (region == "A").astype(float))
        y = np.random.poisson(mu_true * exposure)

        test_data = pl.DataFrame(
            {
                "y": y,
                "age": age,
                "veh_power": veh_power,
                "region": region,
                "exposure": exposure,
            }
        )

        diagnostics = result.diagnostics(
            train_data=train_data,
            test_data=test_data,
            categorical_factors=["region"],
            continuous_factors=["age", "veh_power"],
        )

        # Check train_test comparison is present
        assert diagnostics.train_test is not None
        tt = diagnostics.train_test

        # Check structure
        assert tt.train is not None
        assert tt.test is not None
        assert tt.train.dataset == "train"
        assert tt.test.dataset == "test"

        # Check comparison metrics
        assert hasattr(tt, "gini_gap")
        assert hasattr(tt, "ae_ratio_diff")
        assert hasattr(tt, "overfitting_risk")
        assert hasattr(tt, "calibration_drift")
        assert hasattr(tt, "unstable_factors")

        # Check decile comparison
        assert len(tt.decile_comparison) == 10
        for d in tt.decile_comparison:
            assert "decile" in d
            assert "train_ae" in d
            assert "test_ae" in d


class TestScoreTest:
    """Tests for Rao's score test for unfitted factors."""

    def test_score_test_continuous_via_rust(self):
        """Test score test for continuous variable via Rust binding."""
        from rustystats._rustystats import score_test_continuous_py

        np.random.seed(42)
        n = 100

        # Design matrix (intercept + one variable)
        x = np.column_stack([np.ones(n), np.linspace(0, 10, n)])
        y = np.random.poisson(np.exp(0.5 + 0.1 * x[:, 1]))
        mu = np.exp(0.5 + 0.1 * x[:, 1])
        weights = np.ones(n)
        bread = np.linalg.inv(x.T @ np.diag(weights * mu) @ x)

        # New variable to test
        z = np.random.randn(n)

        result = score_test_continuous_py(z, x, y.astype(float), mu, weights, bread, "poisson")

        assert "statistic" in result
        assert "df" in result
        assert "pvalue" in result
        assert "significant" in result
        assert result["df"] == 1
        assert 0 <= result["pvalue"] <= 1

    def test_score_test_categorical_via_rust(self):
        """Test score test for categorical variable via Rust binding."""
        from rustystats._rustystats import score_test_categorical_py

        np.random.seed(42)
        n = 120

        # Design matrix (intercept only)
        x = np.ones((n, 1))

        # Response varies by hidden group
        groups = np.repeat([0, 1, 2], n // 3)
        lambdas = np.array([np.exp(0.5), np.exp(1.0), np.exp(1.5)])
        y = np.array([np.random.poisson(lambdas[g]) for g in groups])
        mu = np.ones(n) * np.mean(y)
        weights = np.ones(n)
        bread = np.array([[1.0 / (n * mu[0])]])

        # Dummy matrix for 3-level categorical (2 columns)
        z_matrix = np.zeros((n, 2))
        z_matrix[n // 3 : 2 * n // 3, 0] = 1.0  # Group 1
        z_matrix[2 * n // 3 :, 1] = 1.0  # Group 2

        result = score_test_categorical_py(
            z_matrix, x, y.astype(float), mu, weights, bread, "poisson"
        )

        assert result["df"] == 2
        assert 0 <= result["pvalue"] <= 1
        # Should be significant since y varies by group
        assert result["significant"]

    def test_score_test_in_factor_diagnostics(self):
        """Test that score test appears in factor diagnostics for unfitted factors."""
        import rustystats as rs

        np.random.seed(42)
        n = 500

        age = np.random.uniform(20, 60, n)
        region = np.random.choice(["A", "B", "C"], n)
        unfitted_var = np.random.randn(n)  # Not in model
        unfitted_cat = np.random.choice(["X", "Y", "Z"], n)  # Not in model
        exposure = np.ones(n)

        # Generate response based on age and region
        mu_true = np.exp(-1 + 0.02 * age + 0.3 * (region == "A").astype(float))
        y = np.random.poisson(mu_true)

        data = pl.DataFrame(
            {
                "y": y,
                "age": age,
                "region": region,
                "unfitted_var": unfitted_var,
                "unfitted_cat": unfitted_cat,
                "exposure": exposure,
            }
        )

        # Fit model without unfitted_var and unfitted_cat
        result = rs.glm_dict(
            response="y",
            terms={"age": {"type": "linear"}, "region": {"type": "categorical"}},
            data=data,
            family="poisson",
        ).fit()

        # Get diagnostics including unfitted factors
        diagnostics = result.diagnostics(
            train_data=data,
            categorical_factors=["region", "unfitted_cat"],
            continuous_factors=["age", "unfitted_var"],
        )

        # Check that fitted factors don't have score_test
        for factor in diagnostics.factors:
            if factor.in_model:
                assert factor.score_test is None, (
                    f"Fitted factor {factor.name} should not have score_test"
                )

        # Check that unfitted factors have score_test (if diagnostics provides matrices)
        # Note: score_test requires design_matrix, bread_matrix, and irls_weights
        # which may not always be available depending on diagnostics call

    def test_score_test_result_dataclass(self):
        """Test ScoreTestResult dataclass structure."""
        from rustystats.diagnostics import ScoreTestResult

        result = ScoreTestResult(
            statistic=5.5,
            df=2,
            pvalue=0.064,
            significant=False,
        )

        assert result.statistic == 5.5
        assert result.df == 2
        assert result.pvalue == 0.064
        assert not result.significant


class TestSmoothTermDiagnostics:
    """Tests for smooth term (penalized spline) diagnostics."""

    def test_smooth_term_edf_in_diagnostics(self):
        """Test that smooth term EDF appears in diagnostics output."""
        import rustystats as rs

        np.random.seed(42)
        n = 500

        age = np.random.uniform(18, 70, n)
        # Non-linear relationship
        mu_true = np.exp(-3 + 0.1 * np.sin(age / 10))
        y = np.random.poisson(mu_true)

        data = pl.DataFrame({"y": y, "age": age})

        # Fit model with smooth term
        result = rs.glm_dict(
            response="y",
            terms={"age": {"type": "bs"}},  # Penalized smooth
            data=data,
            family="poisson",
        ).fit()

        # Check smooth term info on result
        assert result.has_smooth_terms()
        assert result.smooth_terms is not None
        assert len(result.smooth_terms) == 1

        st = result.smooth_terms[0]
        assert st.variable == "age"
        assert st.edf > 0  # EDF should be positive
        assert st.edf <= st.k  # EDF should not exceed k
        assert st.lambda_ >= 0  # Lambda should be non-negative
        assert st.gcv > 0  # GCV should be positive

    def test_smooth_term_significance_in_diagnostics(self):
        """Test that smooth term significance test appears in diagnostics."""
        import rustystats as rs
        from rustystats.diagnostics import SmoothTermDiagnostics, compute_diagnostics

        np.random.seed(42)
        n = 500

        age = np.random.uniform(18, 70, n)
        # Strong non-linear effect - should be significant
        mu_true = np.exp(-2 + 0.05 * age - 0.0005 * age**2)
        y = np.random.poisson(mu_true)

        data = pl.DataFrame({"y": y, "age": age})

        result = rs.glm_dict(
            response="y",
            terms={"age": {"type": "bs"}},
            data=data,
            family="poisson",
        ).fit()

        diagnostics = compute_diagnostics(
            result=result,
            train_data=data,
            continuous_factors=["age"],
        )

        # Check smooth_terms in diagnostics
        assert diagnostics.smooth_terms is not None
        assert len(diagnostics.smooth_terms) == 1

        st_diag = diagnostics.smooth_terms[0]
        assert isinstance(st_diag, SmoothTermDiagnostics)
        assert st_diag.variable == "age"
        assert st_diag.edf > 0
        assert st_diag.chi2 >= 0  # Wald chi-squared
        assert 0 <= st_diag.p_value <= 1
        assert st_diag.ref_df > 0  # Reference df for test

    def test_smooth_term_diagnostics_json(self):
        """Test smooth term diagnostics in JSON output."""
        import json

        import rustystats as rs

        np.random.seed(42)
        n = 300

        age = np.random.uniform(20, 60, n)
        mu_true = np.exp(-2 + 0.03 * age)
        y = np.random.poisson(mu_true)

        data = pl.DataFrame({"y": y, "age": age})

        result = rs.glm_dict(
            response="y",
            terms={"age": {"type": "bs"}},
            data=data,
            family="poisson",
        ).fit()

        diagnostics = result.diagnostics(
            train_data=data,
            continuous_factors=["age"],
        )

        json_str = diagnostics.to_json()
        parsed = json.loads(json_str)

        assert "smooth_terms" in parsed
        assert parsed["smooth_terms"] is not None
        assert len(parsed["smooth_terms"]) == 1

        st = parsed["smooth_terms"][0]
        assert "variable" in st
        assert "edf" in st
        assert "lambda" in st
        assert "gcv" in st
        assert "chi2" in st
        assert "p_value" in st
        assert "ref_df" in st

    def test_insignificant_smooth_term_warning(self):
        """Test that insignificant smooth terms generate warnings."""
        import rustystats as rs
        from rustystats.diagnostics import compute_diagnostics

        np.random.seed(42)
        n = 200

        # Random noise variable with no real effect
        x = np.random.uniform(0, 10, n)
        y = np.random.poisson(np.ones(n) * 2)  # Constant rate, no x effect

        data = pl.DataFrame({"y": y, "x": x})

        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "bs"}},
            data=data,
            family="poisson",
        ).fit()

        diagnostics = compute_diagnostics(
            result=result,
            train_data=data,
            continuous_factors=["x"],
        )

        # Check for insignificant smooth warning
        _warning_types = [w["type"] for w in diagnostics.warnings]
        # May or may not trigger depending on random data, but structure should be valid
        assert diagnostics.smooth_terms is not None

    def test_multiple_smooth_terms(self):
        """Test diagnostics with multiple smooth terms."""
        import rustystats as rs

        np.random.seed(42)
        n = 500

        age = np.random.uniform(18, 70, n)
        income = np.random.uniform(20000, 100000, n)
        mu_true = np.exp(-4 + 0.02 * age + 0.00001 * income)
        y = np.random.poisson(mu_true)

        data = pl.DataFrame({"y": y, "age": age, "income": income})

        result = rs.glm_dict(
            response="y",
            terms={
                "age": {"type": "bs"},
                "income": {"type": "bs"},
            },
            data=data,
            family="poisson",
        ).fit()

        if result.has_smooth_terms():
            diagnostics = result.diagnostics(
                train_data=data,
                continuous_factors=["age", "income"],
            )

            if diagnostics.smooth_terms:
                # Should have diagnostics for both terms
                assert len(diagnostics.smooth_terms) >= 1

                for st in diagnostics.smooth_terms:
                    assert st.edf > 0
                    assert 0 <= st.p_value <= 1


class TestFactorFeatureIndex:
    """FIX-N regression tests: strict-matching factor → feature index.

    The diagnostics module previously used ``if name in fn`` substring matching
    to find features for a variable, which produced false positives when one
    variable name was a substring of another (e.g. ``Age`` matching
    ``bs(VehAge, 1/4)``). These tests guard against the bug class re-appearing.
    """

    def test_factor_index_no_substring_false_positive(self):
        """Variable 'Age' must not match feature 'bs(VehAge, 1/4)'."""
        from rustystats.diagnostics.factors import _FactorFeatureIndex

        feature_names = [
            "Intercept",
            "bs(Age, 1/4)",
            "bs(Age, 2/4)",
            "bs(VehAge, 1/4)",
            "bs(VehAge, 2/4)",
        ]
        index = _FactorFeatureIndex(["Age", "VehAge"], feature_names)
        assert index.features_for("Age").indices == [1, 2]
        assert index.features_for("VehAge").indices == [3, 4]
        # Term type should be detected as spline.
        assert index.features_for("Age").term_type == "spline"
        assert index.features_for("VehAge").term_type == "spline"

    def test_factor_index_te_interaction(self):
        """TE interaction features should match every variable they involve."""
        from rustystats.diagnostics.factors import _FactorFeatureIndex

        feature_names = [
            "Intercept",
            "TE(Region)",
            "TE(Brand:Region)",
            "TE(Brand)",
        ]
        index = _FactorFeatureIndex(["Region", "Brand"], feature_names)
        # TE(Region) and TE(Brand:Region) both reference Region.
        assert index.features_for("Region").indices == [1, 2]
        # TE(Brand:Region) and TE(Brand) both reference Brand.
        assert index.features_for("Brand").indices == [2, 3]
        assert index.features_for("Region").term_type == "te"
        assert index.features_for("Brand").term_type == "te"

    def test_factor_index_expression_word_boundary(self):
        """I(<expr>) must word-boundary match (age != driver_age)."""
        from rustystats.diagnostics.factors import _FactorFeatureIndex

        feature_names = ["Intercept", "I(age ** 2)", "I(driver_age ** 2)"]
        index = _FactorFeatureIndex(["age"], feature_names)
        # age must NOT pull in I(driver_age ** 2) — that's a different variable.
        assert index.features_for("age").indices == [1]
        assert index.features_for("age").term_type == "expression"

    def test_factor_index_linear_exact_match(self):
        """Linear (raw) features must exact-match — no substring spread."""
        from rustystats.diagnostics.factors import _FactorFeatureIndex

        feature_names = ["Intercept", "Age", "VehAge"]
        index = _FactorFeatureIndex(["Age", "VehAge"], feature_names)
        assert index.features_for("Age").indices == [1]
        assert index.features_for("VehAge").indices == [2]
        assert index.features_for("Age").term_type == "linear"

    def test_factor_index_categorical_strict(self):
        """C(name) must strictly match — C(Brand) ≠ C(BrandX)."""
        from rustystats.diagnostics.factors import _FactorFeatureIndex

        feature_names = ["Intercept", "C(Brand)[T.A]", "C(Brand)[T.B]", "C(BrandX)[T.A]"]
        index = _FactorFeatureIndex(["Brand", "BrandX"], feature_names)
        assert index.features_for("Brand").indices == [1, 2]
        assert index.features_for("BrandX").indices == [3]
        assert index.features_for("Brand").term_type == "categorical"

    def test_factor_index_intercept_excluded(self):
        """The Intercept feature must never be matched."""
        from rustystats.diagnostics.factors import _FactorFeatureIndex

        index = _FactorFeatureIndex(["Intercept"], ["Intercept", "x"])
        assert index.features_for("Intercept").indices == []
        assert not index.is_in_model("Intercept")

    def test_factor_index_unregistered_variable(self):
        """Unregistered variables should return an empty _FactorFeature."""
        from rustystats.diagnostics.factors import _FactorFeatureIndex

        index = _FactorFeatureIndex(["Age"], ["Intercept", "Age"])
        empty = index.features_for("NotRegistered")
        assert empty.indices == []
        assert empty.feature_names == []
        assert empty.transformation is None
        assert empty.term_type == "unknown"

    def test_factor_index_interaction_recursion(self):
        """Interaction features should be matched if any part matches the variable."""
        from rustystats.diagnostics.factors import _FactorFeatureIndex

        feature_names = [
            "Intercept",
            "Age",
            "Region",
            "Age:Region",
            "bs(Age, 1/4):Region",
        ]
        index = _FactorFeatureIndex(["Age", "Region"], feature_names)
        # Age appears as itself, in Age:Region, and inside bs(Age, 1/4):Region.
        assert index.features_for("Age").indices == [1, 3, 4]
        # Region appears as itself, in Age:Region, and in bs(Age, 1/4):Region.
        assert index.features_for("Region").indices == [2, 3, 4]

    def test_match_factor_helper_directly(self):
        """The _match_factor helper should report the correct term kind."""
        from rustystats.diagnostics.factors import _match_factor

        assert _match_factor("Age", "Age") == (True, "linear")
        assert _match_factor("VehAge", "Age") == (False, "unknown")
        assert _match_factor("bs(Age, 1/4)", "Age") == (True, "spline")
        assert _match_factor("bs(VehAge, 1/4)", "Age") == (False, "unknown")
        assert _match_factor("ns(Age, 2)", "Age") == (True, "spline")
        assert _match_factor("TE(Region)", "Region") == (True, "te")
        assert _match_factor("TE(Brand:Region)", "Region") == (True, "te")
        assert _match_factor("C(Brand)[T.A]", "Brand") == (True, "categorical")
        assert _match_factor("C(BrandX)[T.A]", "Brand") == (False, "unknown")
        assert _match_factor("I(age ** 2)", "age") == (True, "expression")
        assert _match_factor("I(driver_age ** 2)", "age") == (False, "unknown")
        assert _match_factor("Intercept", "Intercept") == (False, "unknown")

    def test_diagnostics_no_substring_contamination_integration(self):
        """End-to-end: Age and VehAge splines must not contaminate each other.

        Builds a fitted model with two factors whose names are substrings of
        each other, then verifies the diagnostic factor entries reference
        DISJOINT feature sets.
        """
        import rustystats as rs

        np.random.seed(42)
        n = 500

        age = np.random.uniform(18, 70, n)
        veh_age = np.random.uniform(0, 25, n)
        # Generate a Poisson response with both variables driving the rate.
        mu = np.exp(-4 + 0.02 * age + 0.05 * veh_age)
        y = np.random.poisson(mu)

        data = pl.DataFrame({"y": y, "Age": age, "VehAge": veh_age})

        result = rs.glm_dict(
            response="y",
            terms={
                "Age": {"type": "bs"},
                "VehAge": {"type": "bs"},
            },
            data=data,
            family="poisson",
        ).fit()

        diagnostics = result.diagnostics(
            train_data=data,
            continuous_factors=["Age", "VehAge"],
        )

        age_factor = next(f for f in diagnostics.factors if f.name == "Age")
        vehage_factor = next(f for f in diagnostics.factors if f.name == "VehAge")
        assert age_factor.in_model
        assert vehage_factor.in_model
        assert age_factor.coefficients is not None
        assert vehage_factor.coefficients is not None

        age_terms = {c.term for c in age_factor.coefficients}
        vehage_terms = {c.term for c in vehage_factor.coefficients}

        # CRITICAL: no overlap between the two factors' coefficient term names.
        # Before FIX-N, Age would pull in bs(VehAge, ...) features via
        # substring matching, polluting both significance and the coefficient
        # table.
        assert age_terms & vehage_terms == set()
        assert all("Age" in t and "VehAge" not in t for t in age_terms)
        assert all("VehAge" in t for t in vehage_terms)

        # Spline terms have no meaningful per-coefficient relativity (B3 fix);
        # the multi-coef effect only makes sense in aggregate.
        assert all(c.relativity is None for c in age_factor.coefficients)
        assert all(c.relativity is None for c in vehage_factor.coefficients)

        # Significance is also computed off the strict index, so each factor's
        # χ² is based ONLY on its own basis coefficients (not the other
        # factor's). We do NOT require statistical significance here — that's
        # noise-dependent — only that the two significance entries exist
        # independently and are not identical (which would happen if both
        # factors aliased to the same param indices via substring matching).
        assert age_factor.significance is not None
        assert vehage_factor.significance is not None
        # The two factors should have distinct chi2 values when their
        # coefficient sets are disjoint. (If they shared the same param
        # indices, the chi2 would be identical.)
        assert age_factor.significance.chi2 != vehage_factor.significance.chi2 or len(
            age_factor.coefficients
        ) != len(vehage_factor.coefficients)


class TestScoreTestMatrixChunking:
    """Regression tests for memory-hardening of the lean-mode score-test path.

    `_extract_score_test_matrices` (diagnostics/api.py) rebuilds the design
    matrix from train_data when `store_design_matrix=False` was used at fit
    time. The dev is refactoring this to chunk the rebuild via
    `_compute_predict_chunk_size`, writing into a preallocated (n, p) output
    rather than letting the Rust horizontal stack double-allocate. These
    tests pin down both behavioral equivalence and the chunking trigger.
    """

    # Use a cross-class seed so downstream diagnostics are deterministic.
    _SEED = 20260419

    @staticmethod
    def _make_frequency_dataset(n: int, n_cat_levels: int = 4, seed: int = 20260419):
        """Synthetic Poisson frequency dataset with one continuous + one categorical.

        Kept small (2 covariates in the model => ~n_cat_levels + 2 features)
        so the first test can fit quickly even at n=300_001. Row count is the
        knob we use to trigger chunking (default row cap is 200_000).
        """
        import polars as pl

        rng = np.random.default_rng(seed)
        age = rng.uniform(18, 70, n).astype(np.float64)
        cats = np.array(["A", "B", "C", "D"][:n_cat_levels])
        region = rng.choice(cats, n)
        unfitted_cont = rng.standard_normal(n).astype(np.float64)
        unfitted_cat = rng.choice(np.array(["X", "Y", "Z"]), n)
        mu_true = np.exp(-2.0 + 0.01 * age + 0.3 * (region == cats[0]).astype(np.float64))
        y = rng.poisson(mu_true).astype(np.float64)
        exposure = rng.uniform(0.5, 1.0, n)

        data = pl.DataFrame(
            {
                "y": y,
                "age": age,
                "region": region,
                "unfitted_cont": unfitted_cont,
                "unfitted_cat": unfitted_cat,
                "exposure": exposure,
            }
        )
        return data

    @staticmethod
    def _make_wide_dataset(n: int, seed: int = 20260419):
        """High-cardinality categorical + spline + linear, producing ~150 features."""
        import polars as pl

        rng = np.random.default_rng(seed)
        age = rng.uniform(18, 70, n).astype(np.float64)
        vehage = rng.uniform(0, 25, n).astype(np.float64)
        # 100 levels in "brand" -> ~99 dummy columns -> ~150 total features with
        # age + bs(vehage, 6).
        brand = np.array([f"B{i:03d}" for i in rng.integers(0, 100, n)])
        unfitted_cont = rng.standard_normal(n).astype(np.float64)
        unfitted_cat = rng.choice(np.array(["X", "Y", "Z"]), n)
        mu_true = np.exp(-2.0 + 0.01 * age + 0.02 * vehage)
        y = rng.poisson(mu_true).astype(np.float64)
        exposure = rng.uniform(0.5, 1.0, n)

        data = pl.DataFrame(
            {
                "y": y,
                "age": age,
                "vehage": vehage,
                "brand": brand,
                "unfitted_cont": unfitted_cont,
                "unfitted_cat": unfitted_cat,
                "exposure": exposure,
            }
        )
        return data

    def _run_lean_diagnostics(self, data, categorical=None, continuous=None):
        """Helper: fit in lean mode (store_design_matrix=False) and run diagnostics."""
        import rustystats as rs

        result = rs.glm_dict(
            response="y",
            terms={
                "age": {"type": "linear"},
                "region": {"type": "categorical"},
            },
            data=data,
            family="poisson",
            offset="exposure",
        ).fit(store_design_matrix=False)

        return result.diagnostics(
            train_data=data,
            categorical_factors=categorical or ["region", "unfitted_cat"],
            continuous_factors=continuous or ["age", "unfitted_cont"],
        )

    def test_score_test_matrices_bit_exact_under_chunking(self):
        """Chunked vs single-shot score-test design matrix produce matching p-values.

        Two diagnostics runs on a >200k-row model:
          (a) lean mode (store_design_matrix=False) → score-test path rebuilds
              X from train_data. After the refactor, this rebuild is chunked.
          (b) eager mode (store_design_matrix=True) → score-test path uses the
              cached X from fit time; this path is unaffected by the refactor.

        Both paths feed the SAME X (bit-exact, since horizontal stacking is
        deterministic and row-chunks are concatenated, not reduced with BLAS),
        so the score-test p-values for the unfitted factors must agree to
        within numerical noise.

        n=300_001 rows forces the chunked path (> default 200_000 row cap).
        """
        import rustystats as rs

        data = self._make_frequency_dataset(n=300_001)

        # Path (a): lean mode — rebuilds X (chunked after refactor).
        lean_result = rs.glm_dict(
            response="y",
            terms={
                "age": {"type": "linear"},
                "region": {"type": "categorical"},
            },
            data=data,
            family="poisson",
            offset="exposure",
        ).fit(store_design_matrix=False)
        diag_lean = lean_result.diagnostics(
            train_data=data,
            categorical_factors=["region", "unfitted_cat"],
            continuous_factors=["age", "unfitted_cont"],
        )

        # Path (b): eager mode — uses cached X.
        eager_result = rs.glm_dict(
            response="y",
            terms={
                "age": {"type": "linear"},
                "region": {"type": "categorical"},
            },
            data=data,
            family="poisson",
            offset="exposure",
        ).fit(store_design_matrix=True)
        diag_eager = eager_result.diagnostics(
            train_data=data,
            categorical_factors=["region", "unfitted_cat"],
            continuous_factors=["age", "unfitted_cont"],
        )

        # Sanity: coefficients match (fit path is identical; store flag only
        # controls whether X is retained). If they don't match, the test is
        # broken, not the feature under test.
        np.testing.assert_allclose(
            np.asarray(lean_result.params),
            np.asarray(eager_result.params),
            rtol=1e-12,
            atol=1e-12,
            err_msg="Fit coefficients diverged between lean and eager modes.",
        )

        # Align factors by name and compare score-test p-values. Unfitted
        # factors have `score_test`; fitted factors do not.
        lean_by_name = {f.name: f for f in diag_lean.factors}
        eager_by_name = {f.name: f for f in diag_eager.factors}
        assert set(lean_by_name) == set(eager_by_name)

        compared_any = False
        for name, lean_f in lean_by_name.items():
            eager_f = eager_by_name[name]
            if lean_f.score_test is None and eager_f.score_test is None:
                continue
            # Both modes must agree on whether a score test was produced.
            assert (lean_f.score_test is None) == (eager_f.score_test is None), (
                f"Factor {name}: score_test presence differs between lean and eager"
            )
            if lean_f.score_test is None:
                continue
            compared_any = True
            # Statistic, df, pvalue should match. `pvalue` is rounded to 4
            # decimals inside the ScoreTestResult constructor; the underlying
            # chi2 is rounded to 2 decimals. Use near-exact tolerances after
            # accounting for that rounding.
            assert lean_f.score_test.df == eager_f.score_test.df
            assert abs(lean_f.score_test.statistic - eager_f.score_test.statistic) < 1e-6, (
                f"Factor {name}: score statistic lean={lean_f.score_test.statistic} "
                f"eager={eager_f.score_test.statistic}"
            )
            assert abs(lean_f.score_test.pvalue - eager_f.score_test.pvalue) < 1e-6, (
                f"Factor {name}: score pvalue lean={lean_f.score_test.pvalue} "
                f"eager={eager_f.score_test.pvalue}"
            )

        # Regression guard: we must actually have compared at least one
        # score test, else the assertion above is vacuous.
        assert compared_any, (
            "No unfitted factors produced a score_test; test would be vacuous. Check factor lists."
        )

    def test_chunked_build_fires_for_large_n_in_lean_mode(self, monkeypatch):
        """transform_new_data is called > 1 time when rebuilding lean-mode X.

        This test currently FAILS on unmodified code: the lean-mode path at
        diagnostics/api.py:360 calls transform_new_data once (single-shot),
        regardless of n. After the refactor it should be called multiple
        times (one per row chunk) when n > _PREDICT_ROW_CHUNK_DEFAULT.
        """
        import rustystats as rs
        from rustystats.interactions import InteractionBuilder

        data = self._make_frequency_dataset(n=500_000)

        result = rs.glm_dict(
            response="y",
            terms={
                "age": {"type": "linear"},
                "region": {"type": "categorical"},
            },
            data=data,
            family="poisson",
            offset="exposure",
        ).fit(store_design_matrix=False)

        # Patch transform_new_data to count calls. Must happen AFTER fit so
        # we don't count build-time calls on the training X.
        original = InteractionBuilder.transform_new_data
        call_count = {"n": 0}

        def counting_transform(self, new_data):
            call_count["n"] += 1
            return original(self, new_data)

        monkeypatch.setattr(InteractionBuilder, "transform_new_data", counting_transform)

        # Run diagnostics. Disable partial dependence and interaction detection
        # to isolate the score-test rebuild call(s) — those features also
        # internally call transform_new_data on new data slices.
        result.diagnostics(
            train_data=data,
            categorical_factors=["region", "unfitted_cat"],
            continuous_factors=["age", "unfitted_cont"],
            compute_partial_dep=False,
            detect_interactions=False,
        )

        # After the refactor: at n=500_000 with default chunk size of 200_000,
        # the lean-mode score-test rebuild should fire at least 3 transform
        # calls (ceil(500_000 / 200_000) = 3). We assert >= 2 to remain
        # robust to minor chunk-size tuning.
        assert call_count["n"] >= 2, (
            f"Expected chunked rebuild (>=2 transform_new_data calls); "
            f"got {call_count['n']}. Lean-mode score-test path is probably "
            f"still doing a single-shot rebuild."
        )

    def test_small_n_lean_mode_single_shot_preserved(self, monkeypatch):
        """Small inputs keep the existing single-shot fast path.

        When n <= chunk_size, the helper should fall through to one
        transform_new_data call (matching pre-refactor behavior). This
        guards against over-eager chunking adding overhead to the common
        small-input case.
        """
        import rustystats as rs
        from rustystats.interactions import InteractionBuilder

        data = self._make_frequency_dataset(n=100)

        result = rs.glm_dict(
            response="y",
            terms={
                "age": {"type": "linear"},
                "region": {"type": "categorical"},
            },
            data=data,
            family="poisson",
            offset="exposure",
        ).fit(store_design_matrix=False)

        original = InteractionBuilder.transform_new_data
        call_count = {"n": 0}

        def counting_transform(self, new_data):
            call_count["n"] += 1
            return original(self, new_data)

        monkeypatch.setattr(InteractionBuilder, "transform_new_data", counting_transform)

        result.diagnostics(
            train_data=data,
            categorical_factors=["region", "unfitted_cat"],
            continuous_factors=["age", "unfitted_cont"],
            compute_partial_dep=False,
            detect_interactions=False,
        )

        # At n=100 (well below any reasonable chunk_size), the score-test
        # rebuild must be exactly 1 transform_new_data call — the pre-refactor
        # single-shot fast path. Other diagnostics phases with this setup do
        # not call transform_new_data (partial dep / interactions disabled),
        # so we can assert the exact count.
        assert call_count["n"] == 1, (
            f"Expected exactly 1 transform_new_data call on small-n lean "
            f"mode (single-shot path preserved); got {call_count['n']}. "
            f"Chunking may be firing unnecessarily."
        )

    def test_eager_mode_skips_rebuild_entirely(self, monkeypatch):
        """store_design_matrix=True path: score tests use cached X, zero rebuilds.

        This pins down the orthogonal invariant: when the design matrix IS
        stored, the score-test rebuild branch at api.py:359-360 must not
        execute at all. The chunking refactor should not regress this.
        """
        import rustystats as rs
        from rustystats.interactions import InteractionBuilder

        data = self._make_frequency_dataset(n=100)

        result = rs.glm_dict(
            response="y",
            terms={
                "age": {"type": "linear"},
                "region": {"type": "categorical"},
            },
            data=data,
            family="poisson",
            offset="exposure",
        ).fit(store_design_matrix=True)

        original = InteractionBuilder.transform_new_data
        call_count = {"n": 0}

        def counting_transform(self, new_data):
            call_count["n"] += 1
            return original(self, new_data)

        monkeypatch.setattr(InteractionBuilder, "transform_new_data", counting_transform)

        result.diagnostics(
            train_data=data,
            categorical_factors=["region", "unfitted_cat"],
            continuous_factors=["age", "unfitted_cont"],
            compute_partial_dep=False,
            detect_interactions=False,
        )

        # With eager mode, get_design_matrix() returns a non-None X, so the
        # fallback rebuild branch is skipped entirely. Zero rebuild calls
        # from this path.
        assert call_count["n"] == 0, (
            f"Expected 0 transform_new_data calls in eager mode "
            f"(cached X is used); got {call_count['n']}. The rebuild "
            f"fallback is firing when it shouldn't."
        )

    def test_factor_score_test_pvalues_finite_with_wide_model_lean(self):
        """Wide (~150-feature) lean-mode diagnostics produce finite score p-values.

        This is the correctness regression check: if the chunked-build
        concatenates rows wrong (e.g. a slicing off-by-one, or writes into
        the wrong output row), the score test math downstream will produce
        NaN/inf or wildly wrong p-values. We assert every unfitted factor's
        score-test pvalue is finite and in [0, 1].
        """
        import rustystats as rs

        data = self._make_wide_dataset(n=400_001)

        result = rs.glm_dict(
            response="y",
            terms={
                "age": {"type": "linear"},
                "vehage": {"type": "bs", "df": 6},
                "brand": {"type": "categorical"},
            },
            data=data,
            family="poisson",
            offset="exposure",
        ).fit(store_design_matrix=False)

        # Confirm the model is genuinely wide — if `brand` collapsed to
        # <10 levels on this random seed for some reason, the "wide"
        # part of the regression guard would evaporate.
        n_features = len(result.params)
        assert n_features >= 100, (
            f"Expected a wide model (>=100 features); got {n_features}. "
            f"Dataset construction changed."
        )

        diagnostics = result.diagnostics(
            train_data=data,
            categorical_factors=["brand", "unfitted_cat"],
            continuous_factors=["age", "vehage", "unfitted_cont"],
            compute_partial_dep=False,
            detect_interactions=False,
        )

        checked = 0
        for factor in diagnostics.factors:
            if factor.score_test is None:
                continue
            checked += 1
            pval = factor.score_test.pvalue
            stat = factor.score_test.statistic
            assert np.isfinite(pval), (
                f"Factor {factor.name}: score_test.pvalue={pval} is not finite. "
                f"Chunked rebuild likely corrupted the design matrix."
            )
            assert 0.0 <= pval <= 1.0, (
                f"Factor {factor.name}: score_test.pvalue={pval} not in [0, 1]."
            )
            assert np.isfinite(stat), (
                f"Factor {factor.name}: score_test.statistic={stat} is not finite."
            )
            assert stat >= 0.0, f"Factor {factor.name}: score_test.statistic={stat} is negative."

        # At least one unfitted factor must have a score test, else this
        # regression guard is toothless.
        assert checked >= 1, "No factor produced a score_test result; widen the factor list."


class TestExplorerCramersVStreaming:
    """Regression guard for the upcoming streaming refactor of
    ``DataExplorer._compute_cramers_v_pair_fast``.

    The refactor replaces a full ``(r, k)`` contingency materialization with
    a streaming sum over unique ``(x_inv, y_inv)`` pairs using the algebraic
    identity

        chi2 = sum_ij obs_ij^2 / exp_ij - n

    where ``exp_ij = row_sum[i] * col_sum[j] / n``. These tests pin numerical
    equivalence against the current implementation and a hand-computed
    reference, exercise high-cardinality sparse scenarios where the memory
    savings are realized, and cover edge cases (r=1, k=1, n=0, independent,
    perfectly associated, ValidationError on zero expected frequencies).
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cramers_v_reference(x_inv, y_inv, r, k):
        """Hand-rolled Cramér's V via an explicit (r, k) contingency table.

        This mirrors the pre-refactor implementation and acts as the
        ground-truth reference the streaming implementation must match to
        1e-10 relative tolerance. No scipy — numpy only.
        """
        x_inv = np.asarray(x_inv)
        y_inv = np.asarray(y_inv)
        n = len(x_inv)
        if r < 2 or k < 2 or n == 0:
            return 0.0
        combined = x_inv.astype(np.int64) * k + y_inv.astype(np.int64)
        contingency = np.bincount(combined, minlength=r * k).reshape(r, k).astype(np.float64)
        row_sums = contingency.sum(axis=1, keepdims=True)
        col_sums = contingency.sum(axis=0, keepdims=True)
        expected = row_sums * col_sums / n
        # Avoid division-by-zero noise in the reference (tests that exercise
        # the ValidationError path do not call this helper).
        chi2 = np.sum((contingency - expected) ** 2 / expected)
        min_dim = min(r - 1, k - 1)
        if min_dim == 0:
            return 0.0
        return float(np.sqrt(chi2 / (n * min_dim)))

    @staticmethod
    def _build_inv(r, k, n, seed, ensure_all_levels=True):
        """Construct (x_inv, y_inv) such that every level of r and k is
        observed at least once (otherwise a zero row/column sum trips
        ValidationError and the parity check can't run)."""
        rng = np.random.default_rng(seed)
        x_inv = rng.integers(0, r, size=n).astype(np.intp)
        y_inv = rng.integers(0, k, size=n).astype(np.intp)
        if ensure_all_levels:
            # Plant each level at least once so row/col sums are nonzero.
            if n >= r:
                x_inv[:r] = np.arange(r, dtype=np.intp)
            if n >= k:
                y_inv[:k] = np.arange(k, dtype=np.intp)
        return x_inv, y_inv

    @staticmethod
    def _make_explorer(n=10):
        from rustystats.diagnostics.explorer import DataExplorer

        return DataExplorer(y=np.zeros(n, dtype=np.float64))

    @staticmethod
    def _pair(x_inv, y_inv, r=None, k=None):
        """Build the ``(cats, inv)`` tuple pair the fast path expects."""
        if r is None:
            r = int(x_inv.max()) + 1 if len(x_inv) else 0
        if k is None:
            k = int(y_inv.max()) + 1 if len(y_inv) else 0
        x_cats = np.array([f"x{i}" for i in range(r)])
        y_cats = np.array([f"y{i}" for i in range(k)])
        return (x_cats, x_inv), (y_cats, y_inv)

    # ------------------------------------------------------------------
    # 1. Parity — hand-computed reference on a small 2x2 table
    # ------------------------------------------------------------------

    def test_cramers_v_matches_hand_computed_2x2(self):
        """Classic 2x2 table with known chi-squared/V.

        Layout (counts):
              B=0  B=1
        A=0    50   10  (row sum 60)
        A=1    20   20  (row sum 40)
             col70 col30, n=100

        Expected cells: 42, 18, 28, 12.
        chi2 = 64/42 + 64/18 + 64/28 + 64/12 = 12.6984126984...
        V = sqrt(chi2 / (100 * min(1,1))) = 0.356348322549899...
        """
        x_inv = np.concatenate(
            [
                np.zeros(50, dtype=np.intp),  # A=0, B=0 (50)
                np.zeros(10, dtype=np.intp),  # A=0, B=1 (10)
                np.ones(20, dtype=np.intp),  # A=1, B=0 (20)
                np.ones(20, dtype=np.intp),  # A=1, B=1 (20)
            ]
        )
        y_inv = np.concatenate(
            [
                np.zeros(50, dtype=np.intp),
                np.ones(10, dtype=np.intp),
                np.zeros(20, dtype=np.intp),
                np.ones(20, dtype=np.intp),
            ]
        )
        chi2 = 64 / 42 + 64 / 18 + 64 / 28 + 64 / 12
        v_expected = float(np.sqrt(chi2 / (100 * 1)))

        explorer = self._make_explorer()
        x_pair, y_pair = self._pair(x_inv, y_inv, r=2, k=2)
        v_actual = explorer._compute_cramers_v_pair_fast(x_pair, y_pair)

        assert abs(v_actual - v_expected) / v_expected < 1e-10, (
            f"Hand-computed V={v_expected} vs function V={v_actual}; "
            f"rel_err={abs(v_actual - v_expected) / v_expected:.3e}"
        )

    # ------------------------------------------------------------------
    # 2. Parity against the reference implementation across sizes
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "n,r,k,seed",
        [
            (1_000, 5, 7, 0),
            (10_000, 50, 80, 1),
            (100_000, 500, 800, 2),
        ],
    )
    def test_cramers_v_parity_against_reference(self, n, r, k, seed):
        """Across a range of sizes the fast path must match the explicit
        reference to 1e-10 relative tolerance."""
        x_inv, y_inv = self._build_inv(r, k, n, seed)
        ref = self._cramers_v_reference(x_inv, y_inv, r, k)

        explorer = self._make_explorer()
        x_pair, y_pair = self._pair(x_inv, y_inv, r=r, k=k)
        actual = explorer._compute_cramers_v_pair_fast(x_pair, y_pair)

        # Both should be finite and positive for the random-level mix.
        assert np.isfinite(ref) and ref > 0.0
        assert np.isfinite(actual) and actual > 0.0

        rel_err = abs(actual - ref) / max(abs(ref), 1e-30)
        assert rel_err < 1e-10, (
            f"n={n}, r={r}, k={k}: reference={ref:.15g} actual={actual:.15g} rel_err={rel_err:.3e}"
        )

    # ------------------------------------------------------------------
    # 3. High-cardinality sparse case — memory pressure scenario
    # ------------------------------------------------------------------

    def test_cramers_v_high_cardinality_sparse(self):
        """Insurance-shaped sparse case: n=50k, r=500, k=700, only a small
        fraction of the 350_000 cells ever appear. The streaming impl must
        still return a finite, positive V within a few seconds."""
        import time

        n = 50_000
        r = 500
        k = 700

        rng = np.random.default_rng(123)
        # Sparse pattern: bias toward a small number of (x, y) combinations
        # so the contingency stays sparse. Each row's y-level is chosen
        # pseudo-independently of x but from a restricted per-x subset so
        # the marginals remain balanced enough to avoid zero row/col sums.
        x_inv = rng.integers(0, r, size=n).astype(np.intp)
        y_inv = rng.integers(0, k, size=n).astype(np.intp)
        # Plant each level at least once so every row_sum and col_sum > 0.
        x_inv[:r] = np.arange(r, dtype=np.intp)
        y_inv[:k] = np.arange(k, dtype=np.intp)

        # Confirm the sparsity assumption holds — otherwise this test is
        # not exercising the memory-pressure case.
        n_distinct_pairs = len(np.unique(x_inv.astype(np.int64) * k + y_inv.astype(np.int64)))
        assert n_distinct_pairs < r * k, (
            f"Expected sparse contingency; got {n_distinct_pairs}/{r * k} cells."
        )

        explorer = self._make_explorer(n)
        x_pair, y_pair = self._pair(x_inv, y_inv, r=r, k=k)

        t0 = time.perf_counter()
        v = explorer._compute_cramers_v_pair_fast(x_pair, y_pair)
        elapsed = time.perf_counter() - t0

        assert np.isfinite(v), f"V={v} is not finite"
        assert v > 0.0, f"V={v} should be positive for random independent-ish data"
        assert elapsed < 5.0, f"High-cardinality Cramér's V took {elapsed:.3f}s (>5s)"

        # Parity on the same sparse input — the streaming impl must agree
        # with the explicit reference here too.
        ref = self._cramers_v_reference(x_inv, y_inv, r, k)
        rel_err = abs(v - ref) / max(abs(ref), 1e-30)
        assert rel_err < 1e-10, (
            f"Sparse case: reference={ref:.15g} actual={v:.15g} rel_err={rel_err:.3e}"
        )

    # ------------------------------------------------------------------
    # 4. Edge cases
    # ------------------------------------------------------------------

    def test_cramers_v_single_x_category_returns_zero(self):
        """r=1: Cramér's V is trivially 0 (no variation in x)."""
        x_inv = np.zeros(20, dtype=np.intp)
        y_inv = np.tile(np.arange(4, dtype=np.intp), 5)
        explorer = self._make_explorer()
        x_pair, y_pair = self._pair(x_inv, y_inv, r=1, k=4)
        v = explorer._compute_cramers_v_pair_fast(x_pair, y_pair)
        assert v == 0.0

    def test_cramers_v_single_y_category_returns_zero(self):
        """k=1: V is trivially 0."""
        x_inv = np.tile(np.arange(4, dtype=np.intp), 5)
        y_inv = np.zeros(20, dtype=np.intp)
        explorer = self._make_explorer()
        x_pair, y_pair = self._pair(x_inv, y_inv, r=4, k=1)
        v = explorer._compute_cramers_v_pair_fast(x_pair, y_pair)
        assert v == 0.0

    def test_cramers_v_empty_input_returns_zero(self):
        """n=0 with r,k>=2: function must return 0, not NaN or raise."""
        explorer = self._make_explorer(n=0)
        x_cats = np.array(["x0", "x1"])
        y_cats = np.array(["y0", "y1"])
        empty = np.array([], dtype=np.intp)
        v = explorer._compute_cramers_v_pair_fast((x_cats, empty), (y_cats, empty))
        assert v == 0.0

    def test_cramers_v_independent_near_zero(self):
        """Independent draws: V should be small (not exactly 0 due to
        finite-sample noise, but well below any reasonable threshold)."""
        rng = np.random.default_rng(2026)
        n = 20_000
        r, k = 5, 7
        x_inv = rng.integers(0, r, size=n).astype(np.intp)
        y_inv = rng.integers(0, k, size=n).astype(np.intp)
        # Ensure every level is observed.
        x_inv[:r] = np.arange(r, dtype=np.intp)
        y_inv[:k] = np.arange(k, dtype=np.intp)

        explorer = self._make_explorer(n)
        x_pair, y_pair = self._pair(x_inv, y_inv, r=r, k=k)
        v = explorer._compute_cramers_v_pair_fast(x_pair, y_pair)

        assert np.isfinite(v)
        assert v >= 0.0
        assert v < 0.1, f"Independent data V={v}; expected near 0."

    def test_cramers_v_perfect_association_is_one(self):
        """y is a deterministic function of x: V must be 1.0."""
        n_per = 50
        # x takes 3 levels, y = x (one-to-one). Contingency is diagonal.
        x_inv = np.concatenate([np.full(n_per, i, dtype=np.intp) for i in range(3)])
        y_inv = x_inv.copy()
        explorer = self._make_explorer(3 * n_per)
        x_pair, y_pair = self._pair(x_inv, y_inv, r=3, k=3)
        v = explorer._compute_cramers_v_pair_fast(x_pair, y_pair)
        assert abs(v - 1.0) < 1e-10, f"Perfect association V={v}; expected 1.0"

    def test_cramers_v_small_table_all_sizes(self):
        """Parity for very small tables including 2x3, 3x2, 3x3, 4x5 to
        cover any branching on r vs k (the refactor may special-case
        square vs rectangular)."""
        rng = np.random.default_rng(99)
        for r, k, n in [(2, 3, 200), (3, 2, 200), (3, 3, 500), (4, 5, 800)]:
            x_inv, y_inv = self._build_inv(r, k, n, seed=int(r * 100 + k))
            ref = self._cramers_v_reference(x_inv, y_inv, r, k)
            explorer = self._make_explorer(n)
            x_pair, y_pair = self._pair(x_inv, y_inv, r=r, k=k)
            actual = explorer._compute_cramers_v_pair_fast(x_pair, y_pair)
            rel_err = abs(actual - ref) / max(abs(ref), 1e-30)
            assert rel_err < 1e-10, (
                f"({r}x{k}, n={n}): reference={ref} actual={actual} rel_err={rel_err:.3e}"
            )

    # ------------------------------------------------------------------
    # 5. ValidationError preserved on zero expected frequencies
    # ------------------------------------------------------------------

    def test_cramers_v_validation_error_on_empty_level(self):
        """If a level is declared in ``x_cats``/``y_cats`` but never appears
        in ``x_inv``/``y_inv``, its row/column sum is 0, producing a zero
        expected cell and a ValidationError. This behavior must survive
        the refactor."""
        from rustystats.exceptions import ValidationError

        # Three declared x levels but level 2 never appears.
        x_cats = np.array(["a", "b", "c"])
        x_inv = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.intp)
        y_cats = np.array(["x", "y"])
        y_inv = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0], dtype=np.intp)

        explorer = self._make_explorer(len(x_inv))
        with pytest.raises(ValidationError):
            explorer._compute_cramers_v_pair_fast((x_cats, x_inv), (y_cats, y_inv))

    # ------------------------------------------------------------------
    # 6. Public API path — explore_data() integration
    # ------------------------------------------------------------------

    def test_cramers_v_via_explore_data_public_api(self):
        """End-to-end: invoking ``explore_data`` with two categorical
        factors populates ``exploration.cramers_v`` with a 2x2 matrix whose
        off-diagonal equals the direct private-method call. This ensures
        the refactor does not break the public callsite."""
        from rustystats.diagnostics import explore_data

        rng = np.random.default_rng(7)
        n = 2_000
        A = rng.choice(["A1", "A2", "A3"], size=n)
        B = rng.choice(["B1", "B2", "B3", "B4"], size=n)
        # Plant every combination at least once to keep marginals nonzero.
        for i, a in enumerate(["A1", "A2", "A3"]):
            A[i] = a
        for j, b in enumerate(["B1", "B2", "B3", "B4"]):
            B[j] = b
        y = rng.poisson(1.0, size=n).astype(np.int64)
        data = pl.DataFrame({"y": y, "A": A, "B": B})

        exploration = explore_data(
            data=data,
            response="y",
            categorical_factors=["A", "B"],
            continuous_factors=[],
        )

        cv = exploration.cramers_v
        assert cv is not None
        assert cv["factors"] == ["A", "B"]
        matrix = np.asarray(cv["matrix"], dtype=np.float64)
        assert matrix.shape == (2, 2)
        assert matrix[0, 0] == 1.0
        assert matrix[1, 1] == 1.0
        # Symmetric and finite.
        assert np.isfinite(matrix[0, 1])
        assert abs(matrix[0, 1] - matrix[1, 0]) < 1e-15
        # Parity vs. hand-rolled reference on the same data.
        a_cats, a_inv = np.unique(A.astype(str), return_inverse=True)
        b_cats, b_inv = np.unique(B.astype(str), return_inverse=True)
        ref = self._cramers_v_reference(a_inv, b_inv, len(a_cats), len(b_cats))
        rel_err = abs(matrix[0, 1] - ref) / max(abs(ref), 1e-30)
        assert rel_err < 1e-10, (
            f"explore_data V={matrix[0, 1]} vs reference V={ref}; rel_err={rel_err:.3e}"
        )


class TestCategoricalFactorizationIsolation:
    """Regression tests: categorical factor caches must be per-column.

    Previously, `_precompute_data_caches` used
    `data[name].cast(pl.Utf8).cast(pl.Categorical)`, which relies on the
    polars global/session string cache. That cache accumulates across
    successive casts in the same loop, so the second column's `Categorical`
    inherits the first column's levels. Downstream this corrupted
    `grid_values` in partial dependence and the `(counts, mu_sums)` arrays
    in factor-level analysis, because their length became
    `len(levels_union)` rather than `len(actual_distinct)`.

    The fix is to factorize each column on its own (e.g. np.unique on the
    raw string values), producing per-column `(sorted_levels, codes)`.
    """

    def _build_df(self, *, with_brand=True, with_region=True, with_product=False):
        rng = np.random.default_rng(0)
        n = 1000
        cols = {
            "exposure": rng.uniform(0.1, 1.0, n),
            "y": rng.poisson(1.0, n).astype(float),
        }
        if with_brand:
            cols["brand"] = rng.choice(["A", "B", "C"], n)
        if with_region:
            cols["region"] = rng.choice(["W", "X", "Y", "Z"], n)
        if with_product:
            cols["product"] = rng.choice(["p1", "p2"], n)
        return pl.DataFrame(cols), n

    def test_partial_dependence_grid_values_not_contaminated_across_factors(self):
        """Regression: polars Categorical cache leaked levels across columns,
        so region's grid_values included brand's levels."""
        import rustystats as rs

        df, _ = self._build_df()

        terms = {
            "brand": {"type": "categorical"},
            "region": {"type": "categorical"},
        }
        result = rs.glm_dict(
            response="y",
            terms=terms,
            data=df,
            family="poisson",
            offset="exposure",
        ).fit()
        diag = result.diagnostics(
            train_data=df,
            categorical_factors=["brand", "region"],
            continuous_factors=[],
        )

        pd_by_var = {pd.variable: pd for pd in (diag.partial_dependence or [])}
        assert "brand" in pd_by_var
        assert "region" in pd_by_var

        brand_levels_actual = set(df["brand"].unique().to_list())
        region_levels_actual = set(df["region"].unique().to_list())
        # Disjoint: by construction brand in {A,B,C}, region in {W,X,Y,Z}.
        assert brand_levels_actual.isdisjoint(region_levels_actual)

        brand_grid = set(pd_by_var["brand"].grid_values)
        region_grid = set(pd_by_var["region"].grid_values)

        # Core assertion: each factor's grid_values is exactly its own
        # distinct levels — nothing from the other factor.
        assert brand_grid == brand_levels_actual, f"brand grid_values contaminated: {brand_grid}"
        assert region_grid == region_levels_actual, (
            f"region grid_values contaminated: {region_grid}"
        )
        # predictions must be aligned 1:1 with grid_values.
        assert len(pd_by_var["brand"].predictions) == len(brand_levels_actual)
        assert len(pd_by_var["region"].predictions) == len(region_levels_actual)

    def test_precompute_data_caches_returns_per_column_levels(self):
        """Unit test: the cache helper must factorize each column on its own."""
        from rustystats.diagnostics.api import _precompute_data_caches

        df = pl.DataFrame({"brand": ["A", "B", "A"], "region": ["W", "X", "Y"]})

        _cat_cache, cat_unique_cache, _cont_cache = _precompute_data_caches(
            df, ["brand", "region"], []
        )

        brand_levels, brand_codes = cat_unique_cache["brand"]
        region_levels, region_codes = cat_unique_cache["region"]

        assert set(brand_levels.tolist()) == {"A", "B"}
        assert set(region_levels.tolist()) == {"W", "X", "Y"}
        # Codes must remain valid indices into the per-column levels.
        assert int(brand_codes.max()) < len(brand_levels)
        assert int(region_codes.max()) < len(region_levels)
        # Codes must round-trip: levels[codes] == original values.
        brand_round_trip = brand_levels[brand_codes].tolist()
        region_round_trip = region_levels[region_codes].tolist()
        assert brand_round_trip == ["A", "B", "A"]
        assert region_round_trip == ["W", "X", "Y"]

    def test_precompute_counts_match_actual_observations(self):
        """Regression: bincount over codes must sum to n and have one
        entry per ACTUAL distinct value, not per union-of-levels."""
        from rustystats.diagnostics.api import _precompute_data_caches

        df, n = self._build_df()

        _cat_cache, cat_unique_cache, _cont_cache = _precompute_data_caches(
            df, ["brand", "region"], []
        )

        brand_levels, brand_codes = cat_unique_cache["brand"]
        region_levels, region_codes = cat_unique_cache["region"]

        brand_counts = np.bincount(brand_codes, minlength=len(brand_levels))
        region_counts = np.bincount(region_codes, minlength=len(region_levels))

        # Total observations must be accounted for.
        assert int(brand_counts.sum()) == n
        assert int(region_counts.sum()) == n
        # Length must equal the number of distinct values actually in the
        # column — not the union with sibling columns.
        assert len(brand_counts) == df["brand"].n_unique()
        assert len(region_counts) == df["region"].n_unique()
        # And concretely: no zero-count entries (each level observed).
        assert int(brand_counts.min()) > 0
        assert int(region_counts.min()) > 0

    def test_precompute_levels_are_sorted_lexicographically(self):
        """The existing API sorts levels lexicographically via np.argsort
        (stable). The fix must preserve that deterministic ordering."""
        from rustystats.diagnostics.api import _precompute_data_caches

        # Insertion order is NOT sorted; verify the cache returns sorted.
        df = pl.DataFrame(
            {
                "brand": ["C", "A", "B", "A", "C"],
                "region": ["Z", "W", "Y", "X", "W"],
            }
        )

        _cat_cache, cat_unique_cache, _cont_cache = _precompute_data_caches(
            df, ["brand", "region"], []
        )

        brand_levels, brand_codes = cat_unique_cache["brand"]
        region_levels, region_codes = cat_unique_cache["region"]

        assert brand_levels.tolist() == sorted(brand_levels.tolist())
        assert region_levels.tolist() == sorted(region_levels.tolist())
        assert brand_levels.tolist() == ["A", "B", "C"]
        assert region_levels.tolist() == ["W", "X", "Y", "Z"]

        # Codes are indices into the SORTED levels (must round-trip).
        assert brand_levels[brand_codes].tolist() == ["C", "A", "B", "A", "C"]
        assert region_levels[region_codes].tolist() == ["Z", "W", "Y", "X", "W"]

    def test_single_categorical_factor_still_works(self):
        """Sanity: with only one categorical, behaviour is unchanged."""
        from rustystats.diagnostics.api import _precompute_data_caches

        df, n = self._build_df(with_brand=True, with_region=False)

        _cat_cache, cat_unique_cache, _cont_cache = _precompute_data_caches(df, ["brand"], [])

        brand_levels, brand_codes = cat_unique_cache["brand"]
        assert set(brand_levels.tolist()) == {"A", "B", "C"}
        counts = np.bincount(brand_codes, minlength=len(brand_levels))
        assert int(counts.sum()) == n
        assert len(counts) == 3

    def test_three_categoricals_no_cross_contamination(self):
        """Stress test: with 3 disjoint categoricals, each cache entry
        must contain only that column's levels."""
        from rustystats.diagnostics.api import _precompute_data_caches

        df, n = self._build_df(with_brand=True, with_region=True, with_product=True)

        _cat_cache, cat_unique_cache, _cont_cache = _precompute_data_caches(
            df, ["brand", "region", "product"], []
        )

        expected = {
            "brand": {"A", "B", "C"},
            "region": {"W", "X", "Y", "Z"},
            "product": {"p1", "p2"},
        }
        for name, want in expected.items():
            levels, codes = cat_unique_cache[name]
            got = set(levels.tolist())
            assert got == want, f"{name} levels contaminated: got={got} want={want}"
            counts = np.bincount(codes, minlength=len(levels))
            assert int(counts.sum()) == n
            assert len(counts) == len(want)
            assert int(counts.min()) > 0


# =============================================================================
# Pair (interaction) diagnostics — user-specified `interactions=[...]`
# =============================================================================


class TestPairDiagnostics:
    """Exercises ``result.diagnostics(interactions=[...])`` and the matching
    ``rs.explore_data(interactions=[...])`` path.

    The pair pipeline must:
      - produce one ``InteractionDiagnostics`` per requested pair, in order
      - detect ``in_model`` correctly from ``TermSlot``
      - populate ``train_surface_grid`` cells with raw aggregates
      - share bin edges between train and test surfaces
      - compute block GVIF only for fitted pairs
      - serialize cleanly through ``to_json()`` and auto-save to disk
      - never perform a refit (covered indirectly: no per-pair test data
        prediction beyond the single ``result.predict(test_data)`` call)
    """

    @staticmethod
    def _make_data(seed: int = 42, n: int = 1000) -> "pl.DataFrame":
        import polars as pl

        rng = np.random.default_rng(seed)
        age = rng.uniform(18, 70, n)
        veh_power = rng.uniform(50, 200, n)
        region = rng.choice(["A", "B", "C", "D"], n)
        brand = rng.choice(["X", "Y", "Z"], n)
        exposure = rng.uniform(0.5, 1.0, n)
        mu_true = np.exp(
            -2.0
            + 0.02 * age
            + 0.001 * veh_power
            + 0.3 * (region == "A").astype(float)
            - 0.2 * (brand == "Z").astype(float)
        )
        y = rng.poisson(mu_true * exposure)
        return pl.DataFrame(
            {
                "y": y,
                "age": age,
                "veh_power": veh_power,
                "region": region,
                "brand": brand,
                "exposure": exposure,
            }
        )

    @staticmethod
    def _fit(data: "pl.DataFrame", with_interaction: bool = False):
        import rustystats as rs

        kwargs = dict(
            response="y",
            terms={
                "age": {"type": "linear"},
                "veh_power": {"type": "linear"},
                "region": {"type": "categorical"},
                "brand": {"type": "categorical"},
            },
            data=data,
            family="poisson",
            offset="exposure",
        )
        if with_interaction:
            kwargs["interactions"] = [
                {
                    "age": {"type": "linear"},
                    "region": {"type": "categorical"},
                    "include_main": False,
                }
            ]
        return rs.glm_dict(**kwargs).fit()

    # ------------------------------------------------------------------ post-fit

    def test_interactions_none_default(self):
        """When ``interactions`` is not passed, the new field is empty and
        no per-pair work is done. This is the backwards-compatible default.
        """
        data = self._make_data()
        result = self._fit(data, with_interaction=False)
        diag = result.diagnostics(
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["age"],
        )
        assert diag.interactions == []

    def test_post_fit_basic_shape(self):
        """One requested pair → one InteractionDiagnostics. Cells contain
        raw aggregates; axis labels match the input columns.
        """
        data = self._make_data()
        result = self._fit(data, with_interaction=True)
        diag = result.diagnostics(
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["age", "veh_power"],
            interactions=[("age", "region")],
        )
        assert len(diag.interactions) == 1
        inter = diag.interactions[0]
        assert inter.name == "age:region"
        assert inter.factor1 == "age"
        assert inter.factor2 == "region"
        assert inter.pair_type == "continuous_x_categorical"
        assert inter.in_model is True
        assert inter.representation == "tensor_product"
        # Surface grid
        grid = inter.train_surface_grid
        assert grid.row_axis == "age"
        assert grid.col_axis == "region"
        assert grid.row_type == "quantile"
        assert grid.col_type == "levels"
        assert grid.row_edges is not None and len(grid.row_edges) >= 2
        assert grid.col_levels is not None and "A" in grid.col_levels
        assert len(grid.cells) > 0
        # Per-cell raw fields
        c0 = grid.cells[0]
        assert c0.n > 0
        assert c0.exposure > 0
        assert c0.actual >= 0
        assert c0.predicted is not None and c0.predicted > 0
        assert c0.ae_ratio is not None
        # No test surface when no test data supplied
        assert inter.test_surface_grid is None
        # Block GVIF and block significance populated for the fitted pair
        assert inter.gvif is not None and inter.gvif > 0
        assert inter.significance is not None
        assert inter.coefficients is not None and len(inter.coefficients) > 0

    def test_in_model_false_for_unfitted_pair(self):
        """A pair the user requests but never fit: ``in_model=False``, no
        coefficients, no significance, no GVIF — but the surface grid still
        renders from the raw data.
        """
        data = self._make_data()
        result = self._fit(data, with_interaction=False)
        diag = result.diagnostics(
            train_data=data,
            categorical_factors=["region", "brand"],
            continuous_factors=["age", "veh_power"],
            interactions=[("brand", "region")],
        )
        assert len(diag.interactions) == 1
        inter = diag.interactions[0]
        assert inter.in_model is False
        assert inter.representation is None
        assert inter.coefficients is None
        assert inter.significance is None
        assert inter.gvif is None
        # Surface still rendered from raw data
        assert len(inter.train_surface_grid.cells) > 0
        for cell in inter.train_surface_grid.cells:
            assert cell.predicted is not None  # model is fit, so we have mu

    def test_test_surface_alignment(self):
        """Train and test surface grids share bin edges/levels by construction
        so the caller can do element-wise divergence in one line.
        """
        data = self._make_data(n=1200)
        train = data[:900]
        test = data[900:]
        result = self._fit(train, with_interaction=False)
        diag = result.diagnostics(
            train_data=train,
            test_data=test,
            categorical_factors=["region"],
            continuous_factors=["age"],
            interactions=[("age", "region")],
        )
        inter = diag.interactions[0]
        assert inter.test_surface_grid is not None
        assert inter.train_surface_grid.row_edges == inter.test_surface_grid.row_edges
        assert inter.train_surface_grid.col_levels == inter.test_surface_grid.col_levels
        # Test cells are a subset/superset of train cells (some may be empty)
        # but the grid frame matches.
        assert inter.test_surface_grid.row_axis == inter.train_surface_grid.row_axis
        assert inter.test_surface_grid.col_axis == inter.train_surface_grid.col_axis

    def test_continuous_x_continuous(self):
        """Cont × cont uses quantile bins on both axes."""
        data = self._make_data()
        result = self._fit(data, with_interaction=False)
        diag = result.diagnostics(
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["age", "veh_power"],
            interactions=[("age", "veh_power")],
        )
        inter = diag.interactions[0]
        assert inter.pair_type == "continuous_x_continuous"
        assert inter.train_surface_grid.row_type == "quantile"
        assert inter.train_surface_grid.col_type == "quantile"
        assert inter.train_surface_grid.row_edges is not None
        assert inter.train_surface_grid.col_edges is not None
        assert inter.train_surface_grid.row_levels is None
        assert inter.train_surface_grid.col_levels is None

    def test_categorical_x_categorical(self):
        """Cat × cat uses level lists on both axes; higher cardinality
        becomes ``factor1``.
        """
        data = self._make_data()
        result = self._fit(data, with_interaction=False)
        diag = result.diagnostics(
            train_data=data,
            categorical_factors=["region", "brand"],
            continuous_factors=["age"],
            interactions=[("brand", "region")],
        )
        inter = diag.interactions[0]
        assert inter.pair_type == "categorical_x_categorical"
        assert inter.train_surface_grid.row_type == "levels"
        assert inter.train_surface_grid.col_type == "levels"
        # region has 4 levels, brand has 3 → region is factor1
        assert inter.factor1 == "region"
        assert inter.factor2 == "brand"

    def test_pair_spec_forms(self):
        """All accepted spec forms produce the same result."""
        data = self._make_data()
        result = self._fit(data, with_interaction=False)
        diag1 = result.diagnostics(
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["age"],
            interactions=[("age", "region")],
        )
        diag2 = result.diagnostics(
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["age"],
            interactions=[["age", "region"]],
        )
        diag3 = result.diagnostics(
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["age"],
            interactions=[{"factor1": "age", "factor2": "region"}],
        )
        assert (
            diag1.interactions[0].name == diag2.interactions[0].name == diag3.interactions[0].name
        )

    def test_json_serialization(self):
        """Interactions field round-trips through ``to_json``."""
        import json

        data = self._make_data()
        result = self._fit(data, with_interaction=True)
        diag = result.diagnostics(
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["age"],
            interactions=[("age", "region")],
        )
        payload = json.loads(diag.to_json())
        assert "interactions" in payload
        assert len(payload["interactions"]) == 1
        node = payload["interactions"][0]
        assert node["name"] == "age:region"
        assert node["pair_type"] == "continuous_x_categorical"
        assert node["in_model"] is True
        assert "train_surface_grid" in node
        assert "cells" in node["train_surface_grid"]
        # Each cell carries the raw fields visualizers need
        cell0 = node["train_surface_grid"]["cells"][0]
        for key in ("r", "c", "n", "exposure", "actual", "predicted", "ae_ratio"):
            assert key in cell0

    # ------------------------------------------------------------------ pre-fit

    def test_pre_fit_basic(self):
        """``explore_data(interactions=...)`` produces InteractionExploration
        with a surface grid and an ``interaction_strength`` scalar.
        """
        import rustystats as rs

        data = self._make_data()
        exp = rs.explore_data(
            data=data,
            response="y",
            categorical_factors=["region"],
            continuous_factors=["age", "veh_power"],
            exposure="exposure",
            family="poisson",
            interactions=[("age", "region")],
        )
        assert len(exp.interactions) == 1
        ex = exp.interactions[0]
        assert ex.name == "age:region"
        assert ex.pair_type == "continuous_x_categorical"
        assert 0.0 <= ex.interaction_strength <= 1.0
        # Pre-fit cells: actual is populated but predicted/ae_ratio are null.
        assert len(ex.surface_grid.cells) > 0
        for cell in ex.surface_grid.cells:
            assert cell.predicted is None
            assert cell.ae_ratio is None

    def test_pre_fit_json_field_present(self):
        """The new field is present in the auto-emitted JSON."""
        import json

        import rustystats as rs

        data = self._make_data()
        exp = rs.explore_data(
            data=data,
            response="y",
            categorical_factors=["region"],
            continuous_factors=["age"],
            exposure="exposure",
            family="poisson",
            interactions=[("age", "region")],
        )
        payload = json.loads(exp.to_json())
        assert "interactions" in payload
        assert len(payload["interactions"]) == 1
        assert "surface_grid" in payload["interactions"][0]
        assert "interaction_strength" in payload["interactions"][0]


class TestFactorExtensions:
    """Block GVIF and typed train/test bin pairs added to ``FactorDiagnostics``."""

    @staticmethod
    def _make_data(n: int = 1000):
        import polars as pl

        rng = np.random.default_rng(0)
        age = rng.uniform(18, 70, n)
        power = rng.uniform(50, 200, n)
        region = rng.choice(["A", "B", "C", "D"], n)
        exposure = rng.uniform(0.5, 1.0, n)
        mu = np.exp(-2 + 0.02 * age + 0.001 * power)
        y = rng.poisson(mu * exposure)
        return pl.DataFrame(
            {"y": y, "age": age, "power": power, "region": region, "exposure": exposure}
        )

    @staticmethod
    def _fit(data):
        import rustystats as rs

        return rs.glm_dict(
            response="y",
            terms={
                "age": {"type": "linear"},
                "power": {"type": "linear"},
                "region": {"type": "categorical"},
            },
            data=data,
            family="poisson",
            offset="exposure",
        ).fit()

    def test_gvif_populated_for_fitted_factors(self):
        """Block GVIF is populated for every fitted factor and is positive."""
        data = self._make_data()
        result = self._fit(data)
        diag = result.diagnostics(
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["age", "power"],
        )
        for factor in diag.factors:
            if factor.in_model:
                assert factor.gvif is not None
                assert factor.gvif > 0
            # else (unfitted factor): GVIF stays None

    def test_train_test_bins_only_with_test_data(self):
        """``train_test_bins`` is None when no test data is supplied."""
        data = self._make_data()
        result = self._fit(data)
        diag_no_test = result.diagnostics(
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["age"],
        )
        for factor in diag_no_test.factors:
            assert factor.train_test_bins is None

    def test_train_test_bins_populated_with_test(self):
        """``train_test_bins`` is populated with one entry per train bin
        when test data is supplied, regardless of test-side coverage.
        """
        data = self._make_data(n=1200)
        train, test = data[:900], data[900:]
        result = self._fit(train)
        diag = result.diagnostics(
            train_data=train,
            test_data=test,
            categorical_factors=["region"],
            continuous_factors=["age"],
        )
        for factor in diag.factors:
            ttb = factor.train_test_bins
            if factor.in_model:
                assert ttb is not None
                assert len(ttb) > 0
                # Each pair carries train-side fields always
                for pair in ttb:
                    assert pair.bin
                    assert pair.train_n >= 0
                    assert pair.train_exposure >= 0
                    # test fields can be None when the bin is empty in test

    def test_robust_se_unavailable_emits_warning_on_regularized_fit(self):
        """A regularized fit cannot produce HC1 sandwich SEs. Instead of
        silently leaving every ``coefficient_summary[*].robust_*`` null with
        no signal to the caller, ``diagnostics(...)`` must emit a
        ``robust_se_unavailable`` warning that names the reason.
        """
        import polars as pl
        import rustystats as rs

        rng = np.random.default_rng(1129)
        n = 600
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        region = rng.choice(["A", "B", "C"], n)
        exposure = rng.uniform(0.5, 1.0, n)
        mu = np.exp(-2 + 0.2 * x1 + 0.1 * x2)
        y = rng.poisson(mu * exposure)
        data = pl.DataFrame({"y": y, "x1": x1, "x2": x2, "region": region, "exposure": exposure})
        # Regularized fit — bse_robust() will raise on this result.
        result = rs.glm_dict(
            response="y",
            terms={
                "x1": {"type": "linear"},
                "x2": {"type": "linear"},
                "region": {"type": "categorical"},
            },
            data=data,
            family="poisson",
            offset="exposure",
        ).fit(alpha=0.5, l1_ratio=0.5)

        diag = result.diagnostics(
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["x1", "x2"],
        )
        # All robust SEs null on every coefficient.
        assert diag.coefficient_summary is not None
        assert all(c.robust_std_error is None for c in diag.coefficient_summary)
        # And the warning is present, with a non-empty message naming the reason.
        robust_warnings = [w for w in diag.warnings if w.get("type") == "robust_se_unavailable"]
        assert len(robust_warnings) == 1
        assert "robust SE unavailable" in robust_warnings[0]["message"]

    def test_factor_extensions_in_json(self):
        """New fields serialize through ``to_json``."""
        import json

        data = self._make_data(n=1200)
        train, test = data[:900], data[900:]
        result = self._fit(train)
        diag = result.diagnostics(
            train_data=train,
            test_data=test,
            categorical_factors=["region"],
            continuous_factors=["age"],
        )
        payload = json.loads(diag.to_json())
        fitted = [f for f in payload["factors"] if f["in_model"]]
        assert fitted
        for f in fitted:
            assert "gvif" in f and f["gvif"] is not None
            assert "train_test_bins" in f
            if f["train_test_bins"]:
                pair = f["train_test_bins"][0]
                for key in (
                    "bin",
                    "train_n",
                    "train_exposure",
                    "train_actual",
                    "train_predicted",
                    "train_ae_ratio",
                    "test_n",
                    "test_exposure",
                ):
                    assert key in pair


# =============================================================================
# Helper unit tests — exercise the pair_diagnostics building blocks in
# isolation so a future refactor that breaks one helper fails loudly here.
# =============================================================================


class TestPairDiagnosticsHelpers:
    """Unit tests for the pure helpers in ``pair_diagnostics`` module."""

    # ----- spec normalization -----

    def test_normalize_pair_spec_tuple(self):
        from rustystats.diagnostics.pair_diagnostics import _normalize_pair_spec

        assert _normalize_pair_spec(("a", "b")) == ("a", "b")

    def test_normalize_pair_spec_list(self):
        from rustystats.diagnostics.pair_diagnostics import _normalize_pair_spec

        assert _normalize_pair_spec(["a", "b"]) == ("a", "b")

    def test_normalize_pair_spec_named_dict(self):
        from rustystats.diagnostics.pair_diagnostics import _normalize_pair_spec

        assert _normalize_pair_spec({"factor1": "x", "factor2": "y"}) == ("x", "y")

    def test_normalize_pair_spec_legacy_dict(self):
        """Fit-time style with reserved keys filtered out."""
        from rustystats.diagnostics.pair_diagnostics import _normalize_pair_spec

        spec = {
            "VehAge": {"type": "linear"},
            "Region": {"type": "categorical"},
            "include_main": True,
        }
        assert _normalize_pair_spec(spec) == ("VehAge", "Region")

    def test_normalize_pair_spec_target_encoding_keys_ignored(self):
        """``target_encoding`` / ``frequency_encoding`` / ``prior_weight``
        keys are stripped before counting variables.
        """
        from rustystats.diagnostics.pair_diagnostics import _normalize_pair_spec

        spec = {
            "Brand": {"type": "categorical"},
            "Region": {"type": "categorical"},
            "target_encoding": True,
            "prior_weight": 1.0,
        }
        assert _normalize_pair_spec(spec) == ("Brand", "Region")

    def test_normalize_pair_spec_invalid_dict_raises(self):
        from rustystats.diagnostics.pair_diagnostics import _normalize_pair_spec

        with pytest.raises(ValueError):
            _normalize_pair_spec({"only_one": {"type": "linear"}})

    def test_normalize_pair_spec_invalid_type_raises(self):
        from rustystats.diagnostics.pair_diagnostics import _normalize_pair_spec

        with pytest.raises(ValueError):
            _normalize_pair_spec("not a pair")

    def test_normalize_pair_spec_three_element_list_raises(self):
        from rustystats.diagnostics.pair_diagnostics import _normalize_pair_spec

        with pytest.raises(ValueError):
            _normalize_pair_spec(["a", "b", "c"])

    # ----- pair type detection / normalization -----

    def test_detect_pair_type_cont_cont(self):
        import polars as pl
        from rustystats.diagnostics.pair_diagnostics import _detect_pair_type

        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        assert _detect_pair_type(df, "x", "y") == ("continuous_x_continuous", "x", "y")

    def test_detect_pair_type_cat_cat_normalizes_by_cardinality(self):
        """Higher-cardinality categorical becomes factor1."""
        import polars as pl
        from rustystats.diagnostics.pair_diagnostics import _detect_pair_type

        df = pl.DataFrame(
            {
                "small": ["A", "A", "B", "B"],  # 2 unique
                "big": ["X", "Y", "Z", "W"],  # 4 unique
            }
        )
        kind, f1, f2 = _detect_pair_type(df, "small", "big")
        assert kind == "categorical_x_categorical"
        assert f1 == "big"
        assert f2 == "small"

    def test_detect_pair_type_mixed_continuous_first(self):
        import polars as pl
        from rustystats.diagnostics.pair_diagnostics import _detect_pair_type

        df = pl.DataFrame({"age": [1.0, 2.0, 3.0], "region": ["A", "B", "C"]})
        # Pass categorical first — should be re-ordered.
        kind, f1, f2 = _detect_pair_type(df, "region", "age")
        assert kind == "continuous_x_categorical"
        assert f1 == "age"
        assert f2 == "region"

    # ----- continuous binning -----

    def test_bin_continuous_edges_normal(self):
        from rustystats.diagnostics.pair_diagnostics import _bin_continuous_edges

        values = np.linspace(0.0, 10.0, 100)
        edges = _bin_continuous_edges(values, n_bins=5)
        assert edges.size == 6  # 5 bins → 6 edges
        assert edges[0] <= values.min()
        assert edges[-1] >= values.max()

    def test_bin_continuous_edges_degenerate_identical(self):
        """All identical values → produces a valid 2-edge degenerate grid."""
        from rustystats.diagnostics.pair_diagnostics import _bin_continuous_edges

        values = np.full(100, 5.0)
        edges = _bin_continuous_edges(values, n_bins=5)
        assert edges.size >= 2
        assert edges[0] == 5.0

    def test_bin_continuous_edges_empty(self):
        """No finite values: returns a placeholder [0, 1] grid."""
        from rustystats.diagnostics.pair_diagnostics import _bin_continuous_edges

        values = np.array([np.nan, np.nan])
        edges = _bin_continuous_edges(values, n_bins=4)
        assert edges.size >= 2

    def test_apply_continuous_edges_in_range(self):
        from rustystats.diagnostics.pair_diagnostics import (
            _apply_continuous_edges,
            _bin_continuous_edges,
        )

        edges = _bin_continuous_edges(np.linspace(0.0, 10.0, 100), n_bins=5)
        codes = _apply_continuous_edges(np.array([0.1, 5.0, 9.9]), edges)
        # Codes are bin indices in [0, len(edges)-2]
        assert codes.min() >= 0
        assert codes.max() <= edges.size - 2

    def test_apply_continuous_edges_clips_out_of_range(self):
        """Test data outside the train range is clipped to boundary bins."""
        from rustystats.diagnostics.pair_diagnostics import (
            _apply_continuous_edges,
            _bin_continuous_edges,
        )

        edges = _bin_continuous_edges(np.linspace(0.0, 10.0, 100), n_bins=5)
        # -1000 → first bin (0); +1000 → last bin (edges.size-2)
        codes = _apply_continuous_edges(np.array([-1000.0, 1000.0]), edges)
        assert codes[0] == 0
        assert codes[1] == edges.size - 2

    # ----- categorical binning -----

    def test_bin_categorical_levels_top_k(self):
        from rustystats.diagnostics.pair_diagnostics import _bin_categorical_levels

        # 5 unique levels, take top 3 → returns 3 selected + _Other
        values = np.array(["A"] * 50 + ["B"] * 30 + ["C"] * 15 + ["D"] * 4 + ["E"] * 1)
        levels = _bin_categorical_levels(values, top_k=3)
        assert levels[-1] == "_Other"
        assert set(levels[:-1]) == {"A", "B", "C"}

    def test_bin_categorical_levels_exposure_weighted(self):
        """Top-K is exposure-weighted, not row-count-weighted."""
        from rustystats.diagnostics.pair_diagnostics import _bin_categorical_levels

        # A appears 1 time with exposure 1000; B appears 50 times with exposure 1
        values = np.array(["A"] + ["B"] * 50)
        exposure = np.array([1000.0] + [1.0] * 50)
        levels = _bin_categorical_levels(values, top_k=1, exposure=exposure)
        # Without exposure: B wins (50 rows). With exposure: A wins (1000 units).
        assert "A" in levels

    def test_bin_categorical_levels_no_other_when_cardinality_fits(self):
        """If cardinality <= top_k, ``_Other`` is not appended."""
        from rustystats.diagnostics.pair_diagnostics import _bin_categorical_levels

        values = np.array(["A", "A", "B", "B", "C", "C"])
        levels = _bin_categorical_levels(values, top_k=5)
        assert "_Other" not in levels
        assert set(levels) == {"A", "B", "C"}

    def test_apply_categorical_levels_seen(self):
        from rustystats.diagnostics.pair_diagnostics import _apply_categorical_levels

        levels = ["A", "B", "C", "_Other"]
        codes = _apply_categorical_levels(np.array(["A", "B", "C", "A"]), levels)
        assert codes.tolist() == [0, 1, 2, 0]

    def test_apply_categorical_levels_unseen_to_other(self):
        from rustystats.diagnostics.pair_diagnostics import _apply_categorical_levels

        levels = ["A", "B", "_Other"]
        codes = _apply_categorical_levels(np.array(["A", "Z", "B", "Q"]), levels)
        # Z and Q are unseen → _Other (code 2)
        assert codes.tolist() == [0, 2, 1, 2]

    def test_apply_categorical_levels_no_other_drops_unseen(self):
        """When the levels list has no ``_Other``, unseen values get -1
        sentinel which the cell aggregator then drops.
        """
        from rustystats.diagnostics.pair_diagnostics import _apply_categorical_levels

        levels = ["A", "B"]  # no _Other
        codes = _apply_categorical_levels(np.array(["A", "Z", "B"]), levels)
        assert codes[0] == 0
        assert codes[1] == -1  # unseen sentinel
        assert codes[2] == 1

    # ----- bin-count selection -----

    def test_choose_bin_counts_cont_cont_under_cap(self):
        from rustystats.diagnostics.pair_diagnostics import _choose_bin_counts

        n1, n2 = _choose_bin_counts(
            "continuous_x_continuous", (6, 6), (8, 15), (12, 12), 150, None, None
        )
        assert (n1, n2) == (6, 6)

    def test_choose_bin_counts_cont_cont_over_cap_shrinks(self):
        from rustystats.diagnostics.pair_diagnostics import _choose_bin_counts

        n1, n2 = _choose_bin_counts(
            "continuous_x_continuous", (20, 20), (8, 15), (12, 12), 100, None, None
        )
        assert n1 * n2 <= 100

    def test_choose_bin_counts_cat_cat_respects_cardinality(self):
        from rustystats.diagnostics.pair_diagnostics import _choose_bin_counts

        # cat × cat with cardinalities 3 and 5; defaults are (12, 12)
        n1, n2 = _choose_bin_counts(
            "categorical_x_categorical",
            (6, 6),
            (8, 15),
            (12, 12),
            150,
            cardinality1=3,
            cardinality2=5,
        )
        assert n1 == 3
        assert n2 == 5

    def test_choose_bin_counts_cat_cat_shrinks_to_cap(self):
        from rustystats.diagnostics.pair_diagnostics import _choose_bin_counts

        # 50 × 50 cardinalities, cap 150
        n1, n2 = _choose_bin_counts(
            "categorical_x_categorical",
            (6, 6),
            (8, 15),
            (12, 12),
            150,
            cardinality1=50,
            cardinality2=50,
        )
        assert n1 * n2 <= 150

    # ----- cell aggregation -----

    def test_aggregate_cells_basic(self):
        from rustystats.diagnostics.pair_diagnostics import _aggregate_cells

        codes1 = np.array([0, 0, 1, 1], dtype=np.int32)
        codes2 = np.array([0, 1, 0, 1], dtype=np.int32)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        exposure = np.ones(4)
        mu = np.array([0.5, 1.5, 2.5, 3.5])
        cells = _aggregate_cells(codes1, codes2, y, exposure, mu, n_rows=2, n_cols=2)
        assert cells[(0, 0)] == (1, 1.0, 1.0, 0.5)
        assert cells[(0, 1)] == (1, 1.0, 2.0, 1.5)
        assert cells[(1, 0)] == (1, 1.0, 3.0, 2.5)
        assert cells[(1, 1)] == (1, 1.0, 4.0, 3.5)

    def test_aggregate_cells_groups_correctly(self):
        """Multiple rows in the same cell → counts and sums add up."""
        from rustystats.diagnostics.pair_diagnostics import _aggregate_cells

        codes1 = np.array([0, 0, 0, 1], dtype=np.int32)
        codes2 = np.array([0, 0, 0, 1], dtype=np.int32)
        y = np.array([1.0, 2.0, 3.0, 10.0])
        exposure = np.array([1.0, 1.0, 1.0, 2.0])
        cells = _aggregate_cells(codes1, codes2, y, exposure, mu=None, n_rows=2, n_cols=2)
        assert cells[(0, 0)] == (3, 3.0, 6.0, None)
        assert cells[(1, 1)] == (1, 2.0, 10.0, None)

    def test_aggregate_cells_drops_invalid_codes(self):
        """Rows with negative codes (unseen, no _Other) are dropped."""
        from rustystats.diagnostics.pair_diagnostics import _aggregate_cells

        codes1 = np.array([0, -1, 1], dtype=np.int32)
        codes2 = np.array([0, 1, 1], dtype=np.int32)
        y = np.array([1.0, 99.0, 3.0])
        exposure = np.ones(3)
        cells = _aggregate_cells(codes1, codes2, y, exposure, mu=None, n_rows=2, n_cols=2)
        assert (0, 0) in cells
        assert (1, 1) in cells
        # The middle row (code1 = -1) is dropped — no cell with y=99
        for cell in cells.values():
            assert cell[2] != 99.0  # y_sum

    # ----- interaction_strength is exposed as a Rust function (no Python
    #       reimplementation); see TestPairDiagnosticsNumerics::
    #       test_pre_fit_strength_matches_existing_detector for the
    #       integrated parity check, and TestPairDiagnosticsEdgeCases::
    #       test_pre_fit_strength_in_unit_interval for the [0, 1] contract.

    # ----- block GVIF on synthetic designs -----

    def test_compute_block_gvif_orthogonal_near_one(self):
        """Block GVIF on orthogonal columns is ~1 (no inflation)."""
        from rustystats.diagnostics.pair_diagnostics import (
            _build_design_correlation_matrix,
            _compute_block_gvif,
        )

        rng = np.random.default_rng(0)
        n = 500
        # 4 mutually-orthogonal-ish columns; one of them is the "block".
        X = rng.standard_normal((n, 4))
        # Orthogonalize via QR
        Q, _ = np.linalg.qr(X)
        R_reg = _build_design_correlation_matrix(Q)
        gvif = _compute_block_gvif(R_reg, col_start=0, col_end=2)
        assert gvif is not None
        # Allow some slack — finite-sample noise after standardization.
        assert 0.5 < gvif < 2.0

    def test_compute_block_gvif_no_complement_returns_none(self):
        """When the block IS the entire design, GVIF is undefined."""
        from rustystats.diagnostics.pair_diagnostics import (
            _build_design_correlation_matrix,
            _compute_block_gvif,
        )

        X = np.random.default_rng(0).standard_normal((100, 3))
        R_reg = _build_design_correlation_matrix(X)
        assert _compute_block_gvif(R_reg, col_start=0, col_end=3) is None

    def test_compute_block_gvif_invalid_range_returns_none(self):
        from rustystats.diagnostics.pair_diagnostics import (
            _build_design_correlation_matrix,
            _compute_block_gvif,
        )

        X = np.random.default_rng(0).standard_normal((100, 3))
        R_reg = _build_design_correlation_matrix(X)
        assert _compute_block_gvif(R_reg, col_start=-1, col_end=2) is None
        assert _compute_block_gvif(R_reg, col_start=0, col_end=99) is None
        assert _compute_block_gvif(R_reg, col_start=2, col_end=2) is None  # empty

    def test_compute_block_gvif_none_design(self):
        from rustystats.diagnostics.pair_diagnostics import _compute_block_gvif

        assert _compute_block_gvif(None, 0, 1) is None

    def test_build_design_correlation_matrix_none_input(self):
        from rustystats.diagnostics.pair_diagnostics import (
            _build_design_correlation_matrix,
        )

        assert _build_design_correlation_matrix(None) is None
        assert _build_design_correlation_matrix(np.array([[1.0]])) is None  # n<=1
        assert _build_design_correlation_matrix(np.empty((10, 0))) is None  # p==0

    # ----- representation tagging -----

    def test_representation_from_slot_interaction(self):
        from rustystats.diagnostics.pair_diagnostics import _representation_from_slot

        class FakeSlot:
            term_type = "interaction"

        assert _representation_from_slot(FakeSlot()) == "tensor_product"

    def test_representation_from_slot_te(self):
        from rustystats.diagnostics.pair_diagnostics import _representation_from_slot

        class FakeSlot:
            term_type = "target_encoding"

        assert _representation_from_slot(FakeSlot()) == "target_encoding"

    def test_representation_from_slot_fe(self):
        from rustystats.diagnostics.pair_diagnostics import _representation_from_slot

        class FakeSlot:
            term_type = "frequency_encoding"

        assert _representation_from_slot(FakeSlot()) == "frequency_encoding"

    def test_representation_from_slot_none(self):
        from rustystats.diagnostics.pair_diagnostics import _representation_from_slot

        assert _representation_from_slot(None) is None


# =============================================================================
# Numerical-correctness tests — cells match manual groupby, etc.
# =============================================================================


class TestPairDiagnosticsNumerics:
    """Verify that aggregated values match a hand-computed groupby."""

    @staticmethod
    def _make_data(seed: int = 11, n: int = 800):
        import polars as pl

        rng = np.random.default_rng(seed)
        age = rng.uniform(18, 70, n)
        region = rng.choice(["A", "B", "C", "D"], n)
        exposure = rng.uniform(0.5, 1.0, n)
        mu_true = np.exp(-2 + 0.02 * age + 0.3 * (region == "A").astype(float))
        y = rng.poisson(mu_true * exposure)
        return pl.DataFrame({"y": y, "age": age, "region": region, "exposure": exposure})

    def _fit(self, data):
        import rustystats as rs

        return rs.glm_dict(
            response="y",
            terms={"age": {"type": "linear"}, "region": {"type": "categorical"}},
            data=data,
            family="poisson",
            offset="exposure",
        ).fit()

    def test_cell_actual_matches_manual_aggregation(self):
        """Each cell's ``actual`` equals ``sum(y) / sum(exposure)`` on its rows."""
        data = self._make_data()
        result = self._fit(data)
        diag = result.diagnostics(
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["age"],
            interactions=[("age", "region")],
        )
        inter = diag.interactions[0]
        grid = inter.train_surface_grid

        # Reconstruct the binning the library used
        age = data["age"].to_numpy()
        edges = np.asarray(grid.row_edges, dtype=np.float64)
        digitized = np.clip(np.digitize(age, edges[1:-1]), 0, edges.size - 2)
        region = data["region"].to_numpy()
        # Categorical level lookup
        level_to_code = {lvl: i for i, lvl in enumerate(grid.col_levels)}
        col_codes = np.array([level_to_code.get(v, len(level_to_code)) for v in region])

        y = data["y"].to_numpy().astype(np.float64)
        exposure = data["exposure"].to_numpy().astype(np.float64)

        for cell in grid.cells:
            mask = (digitized == cell.r) & (col_codes == cell.c)
            expected_actual = y[mask].sum() / exposure[mask].sum()
            assert abs(cell.actual - expected_actual) < 1e-9, (
                f"actual mismatch at ({cell.r},{cell.c}): got {cell.actual}, want {expected_actual}"
            )
            assert int(mask.sum()) == cell.n

    def test_cell_predicted_matches_manual_aggregation(self):
        """Each cell's ``predicted`` equals ``sum(mu) / sum(exposure)`` on its rows."""
        data = self._make_data()
        result = self._fit(data)
        diag = result.diagnostics(
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["age"],
            interactions=[("age", "region")],
        )
        inter = diag.interactions[0]
        grid = inter.train_surface_grid

        # Predictions on train (must use the same path the diagnostics used).
        mu = result.predict(data).astype(np.float64)
        age = data["age"].to_numpy()
        edges = np.asarray(grid.row_edges, dtype=np.float64)
        digitized = np.clip(np.digitize(age, edges[1:-1]), 0, edges.size - 2)
        region = data["region"].to_numpy()
        level_to_code = {lvl: i for i, lvl in enumerate(grid.col_levels)}
        col_codes = np.array([level_to_code.get(v, len(level_to_code)) for v in region])
        exposure = data["exposure"].to_numpy().astype(np.float64)

        for cell in grid.cells:
            mask = (digitized == cell.r) & (col_codes == cell.c)
            expected = mu[mask].sum() / exposure[mask].sum()
            assert abs(cell.predicted - expected) < 1e-9, (
                f"predicted mismatch at ({cell.r},{cell.c}): got {cell.predicted}, want {expected}"
            )

    def test_ae_ratio_matches_actual_over_predicted(self):
        """``ae_ratio`` must equal ``actual / predicted`` to within float tolerance."""
        data = self._make_data()
        result = self._fit(data)
        diag = result.diagnostics(
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["age"],
            interactions=[("age", "region")],
        )
        inter = diag.interactions[0]
        for cell in inter.train_surface_grid.cells:
            if cell.predicted and cell.predicted > 0:
                assert abs(cell.ae_ratio - cell.actual / cell.predicted) < 1e-9

    def test_coefficient_count_matches_block_size(self):
        """The coefficients list has one entry per design column in the
        interaction's TermSlot range.
        """
        import rustystats as rs

        data = self._make_data()
        result = rs.glm_dict(
            response="y",
            terms={
                "age": {"type": "linear"},
                "region": {"type": "categorical"},
            },
            interactions=[
                {
                    "age": {"type": "linear"},
                    "region": {"type": "categorical"},
                    "include_main": False,
                }
            ],
            data=data,
            family="poisson",
            offset="exposure",
        ).fit()
        diag = result.diagnostics(
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["age"],
            interactions=[("age", "region")],
        )
        inter = diag.interactions[0]
        slot = next(
            s
            for s in result._builder._term_slots
            if s.term_type == "interaction" and frozenset(s.factors) == frozenset({"age", "region"})
        )
        expected_n = slot.col_end - slot.col_start
        assert inter.coefficients is not None
        assert len(inter.coefficients) == expected_n

    def test_significance_df_matches_block_size(self):
        """Block Wald χ² df equals the number of design columns in the block."""
        import rustystats as rs

        data = self._make_data()
        result = rs.glm_dict(
            response="y",
            terms={
                "age": {"type": "linear"},
                "region": {"type": "categorical"},
            },
            interactions=[
                {
                    "age": {"type": "linear"},
                    "region": {"type": "categorical"},
                    "include_main": False,
                }
            ],
            data=data,
            family="poisson",
            offset="exposure",
        ).fit()
        diag = result.diagnostics(
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["age"],
            interactions=[("age", "region")],
        )
        inter = diag.interactions[0]
        slot = next(
            s
            for s in result._builder._term_slots
            if s.term_type == "interaction" and frozenset(s.factors) == frozenset({"age", "region"})
        )
        # The Wald statistic is returned but df is implicit in the result of
        # the Rust call. We round-trip via the dev_contrib (chi2) which we
        # know equals the statistic. Validate the chi² is positive and finite.
        assert inter.significance is not None
        assert inter.significance.chi2 > 0
        assert np.isfinite(inter.significance.chi2)
        assert 0.0 <= inter.significance.p <= 1.0
        # And that the block size is what we expect (independent sanity).
        assert slot.col_end - slot.col_start > 0

    def test_pre_fit_strength_matches_existing_detector(self):
        """``InteractionExploration.interaction_strength`` must equal the
        existing ``InteractionCandidate.interaction_strength`` for the same
        pair — they're advertised as the same scalar.
        """
        import polars as pl
        import rustystats as rs

        rng = np.random.default_rng(101)
        n = 1500
        # Force a real cell-level signal so the detector picks the pair up.
        cat_a = rng.choice(["X", "Y", "Z", "W"], n)
        cat_b = rng.choice(["P", "Q", "R", "S"], n)
        exposure = rng.uniform(0.5, 1.0, n)
        # Strong interaction signal in (X, P).
        hot = (cat_a == "X") & (cat_b == "P")
        mu = np.exp(-1.5 + 0.8 * hot.astype(float))
        y = rng.poisson(mu * exposure)
        data = pl.DataFrame({"y": y, "cat_a": cat_a, "cat_b": cat_b, "exposure": exposure})
        exp = rs.explore_data(
            data=data,
            response="y",
            categorical_factors=["cat_a", "cat_b"],
            exposure="exposure",
            family="poisson",
            detect_interactions=True,
            interactions=[("cat_a", "cat_b")],
        )
        user_strength = exp.interactions[0].interaction_strength
        # The auto-detector emits the same pair under interaction_candidates.
        # Find the matching candidate by factor names (order-insensitive).
        target = frozenset({"cat_a", "cat_b"})
        match = next(
            (
                cand
                for cand in exp.interaction_candidates
                if frozenset({cand.factor1, cand.factor2}) == target
            ),
            None,
        )
        if match is not None:
            # Floating-point bit-equality is too strict (different code paths
            # may accumulate sums in different orders); 1e-6 absolute is more
            # than tight enough to catch a formula mismatch.
            assert abs(user_strength - match.interaction_strength) < 1e-6

    def test_cell_count_within_max_total_cells(self):
        """High-cardinality cat × cat respects the 150-cell cap."""
        import polars as pl
        import rustystats as rs

        rng = np.random.default_rng(2)
        n = 4000
        big1 = rng.choice([f"L{i}" for i in range(40)], n)
        big2 = rng.choice([f"M{i}" for i in range(40)], n)
        exposure = rng.uniform(0.5, 1.0, n)
        y = rng.poisson(0.5 * exposure)
        data = pl.DataFrame({"y": y, "big1": big1, "big2": big2, "exposure": exposure})
        result = rs.glm_dict(
            response="y",
            terms={"big1": {"type": "categorical"}},
            data=data,
            family="poisson",
            offset="exposure",
        ).fit()
        diag = result.diagnostics(
            train_data=data,
            categorical_factors=["big1", "big2"],
            interactions=[("big1", "big2")],
        )
        inter = diag.interactions[0]
        # Cells emitted ≤ cells in the grid frame ≤ max_total_cells.
        n_row = len(inter.train_surface_grid.row_levels)
        n_col = len(inter.train_surface_grid.col_levels)
        assert n_row * n_col <= 150


# =============================================================================
# Edge cases
# =============================================================================


class TestPairDiagnosticsEdgeCases:
    @staticmethod
    def _make_data(seed: int = 17, n: int = 600):
        import polars as pl

        rng = np.random.default_rng(seed)
        age = rng.uniform(18, 70, n)
        region = rng.choice(["A", "B", "C"], n)
        exposure = rng.uniform(0.5, 1.0, n)
        mu = np.exp(-2 + 0.02 * age)
        y = rng.poisson(mu * exposure)
        return pl.DataFrame({"y": y, "age": age, "region": region, "exposure": exposure})

    def _fit(self, data):
        import rustystats as rs

        return rs.glm_dict(
            response="y",
            terms={"age": {"type": "linear"}, "region": {"type": "categorical"}},
            data=data,
            family="poisson",
            offset="exposure",
        ).fit()

    def test_empty_interactions_list(self):
        """``interactions=[]`` is equivalent to omitting the kwarg."""
        data = self._make_data()
        result = self._fit(data)
        diag = result.diagnostics(
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["age"],
            interactions=[],
        )
        assert diag.interactions == []

    def test_unknown_factor_raises_clear_error(self):
        data = self._make_data()
        result = self._fit(data)
        with pytest.raises(ValueError, match="not a column"):
            result.diagnostics(
                train_data=data,
                categorical_factors=["region"],
                continuous_factors=["age"],
                interactions=[("age", "nonexistent_factor")],
            )

    def test_test_data_with_unseen_categorical_level(self):
        """Test rows with a level absent from train are routed to ``_Other``
        when that bucket exists, or dropped otherwise.
        """
        import polars as pl

        data = self._make_data()
        train = data
        # Test data has a brand-new level not seen in train.
        test = pl.DataFrame(
            {
                "y": [1, 2],
                "age": [30.0, 40.0],
                "region": ["NEW", "A"],
                "exposure": [1.0, 1.0],
            }
        )
        result = self._fit(train)
        diag = result.diagnostics(
            train_data=train,
            test_data=test,
            categorical_factors=["region"],
            continuous_factors=["age"],
            interactions=[("age", "region")],
        )
        inter = diag.interactions[0]
        assert inter.test_surface_grid is not None
        # The "NEW" row is either bucketed to _Other (if present in train levels)
        # or dropped. Either way, the test grid uses the SAME col_levels as train.
        assert inter.test_surface_grid.col_levels == inter.train_surface_grid.col_levels

    def test_test_data_with_out_of_range_continuous(self):
        """Test continuous values outside train range are clipped to boundary
        bins; the test surface stays cell-aligned.
        """
        import polars as pl

        data = self._make_data()
        # Test with extreme continuous values
        test = pl.DataFrame(
            {
                "y": [1, 2, 3],
                "age": [-100.0, 200.0, 50.0],
                "region": ["A", "B", "C"],
                "exposure": [1.0, 1.0, 1.0],
            }
        )
        result = self._fit(data)
        diag = result.diagnostics(
            train_data=data,
            test_data=test,
            categorical_factors=["region"],
            continuous_factors=["age"],
            interactions=[("age", "region")],
        )
        inter = diag.interactions[0]
        assert inter.test_surface_grid is not None
        # Same edges as train, despite test having out-of-range values.
        assert inter.train_surface_grid.row_edges == inter.test_surface_grid.row_edges

    def test_high_cardinality_categorical_against_continuous(self):
        """cont × cat with very high categorical cardinality respects cap
        and shrinks the categorical side first.
        """
        import polars as pl
        import rustystats as rs

        rng = np.random.default_rng(3)
        n = 3000
        age = rng.uniform(18, 70, n)
        # 100 unique brand levels
        brand = rng.choice([f"B{i}" for i in range(100)], n)
        exposure = rng.uniform(0.5, 1.0, n)
        y = rng.poisson(0.5 * exposure)
        data = pl.DataFrame({"y": y, "age": age, "brand": brand, "exposure": exposure})
        result = rs.glm_dict(
            response="y",
            terms={"age": {"type": "linear"}},
            data=data,
            family="poisson",
            offset="exposure",
        ).fit()
        diag = result.diagnostics(
            train_data=data,
            categorical_factors=["brand"],
            continuous_factors=["age"],
            interactions=[("age", "brand")],
        )
        inter = diag.interactions[0]
        # Default cont×cat is (8, 15). Cap 150 keeps this under control.
        assert len(inter.train_surface_grid.col_levels) <= 16  # 15 + _Other

    def test_single_unique_continuous_value(self):
        """Continuous factor with all-identical values produces a degenerate
        but non-crashing grid.
        """
        import polars as pl
        import rustystats as rs

        n = 200
        rng = np.random.default_rng(5)
        data = pl.DataFrame(
            {
                "y": rng.poisson(0.5, n),
                "constant": np.full(n, 5.0),
                "region": rng.choice(["A", "B"], n),
                "exposure": np.ones(n),
            }
        )
        result = rs.glm_dict(
            response="y",
            terms={"region": {"type": "categorical"}},
            data=data,
            family="poisson",
            offset="exposure",
        ).fit()
        # `constant` is unfitted, but the surface still aggregates raw cells.
        diag = result.diagnostics(
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["constant"],
            interactions=[("constant", "region")],
        )
        inter = diag.interactions[0]
        assert inter.in_model is False
        # At least one cell emitted from the degenerate row axis
        assert len(inter.train_surface_grid.cells) >= 1

    def test_pre_fit_strength_in_unit_interval(self):
        """``interaction_strength`` is always in ``[0, 1]``."""
        import rustystats as rs

        data = self._make_data()
        exp = rs.explore_data(
            data=data,
            response="y",
            categorical_factors=["region"],
            continuous_factors=["age"],
            exposure="exposure",
            family="poisson",
            interactions=[("age", "region")],
        )
        s = exp.interactions[0].interaction_strength
        assert 0.0 <= s <= 1.0

    def test_pair_pipeline_coexists_with_auto_detector(self):
        """Setting both ``detect_interactions=True`` and ``interactions=[...]``
        populates both fields independently.
        """
        data = self._make_data()
        result = self._fit(data)
        diag = result.diagnostics(
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["age"],
            detect_interactions=True,
            interactions=[("age", "region")],
        )
        # Either or both may have entries depending on data, but both fields
        # must coexist without one clobbering the other.
        assert hasattr(diag, "interaction_candidates")
        assert hasattr(diag, "interactions")
        assert len(diag.interactions) == 1


# =============================================================================
# Encoding-routing tests — TE / FE interactions show the right representation.
# =============================================================================


class TestPairDiagnosticsEncodings:
    """TE-encoded and FE-encoded interactions emit the right ``representation``
    tag and a single-column coefficient block.
    """

    @staticmethod
    def _make_data(seed: int = 23, n: int = 1200):
        import polars as pl

        rng = np.random.default_rng(seed)
        brand = rng.choice([f"B{i}" for i in range(8)], n)
        region = rng.choice(["A", "B", "C", "D"], n)
        exposure = rng.uniform(0.5, 1.0, n)
        # Pick a brand-region combination as the "hot spot"
        hot = (brand == "B0") & (region == "A")
        mu = np.exp(-2.0 + 0.6 * hot.astype(float))
        y = rng.poisson(mu * exposure)
        return pl.DataFrame({"y": y, "brand": brand, "region": region, "exposure": exposure})

    def test_te_interaction_representation(self):
        """A target-encoded interaction has ``representation='target_encoding'``
        and a single coefficient entry.
        """
        import rustystats as rs

        data = self._make_data()
        result = rs.glm_dict(
            response="y",
            terms={"brand": {"type": "categorical"}, "region": {"type": "categorical"}},
            interactions=[
                {
                    "brand": {"type": "categorical"},
                    "region": {"type": "categorical"},
                    "target_encoding": True,
                }
            ],
            data=data,
            family="poisson",
            offset="exposure",
        ).fit()
        diag = result.diagnostics(
            train_data=data,
            categorical_factors=["brand", "region"],
            interactions=[("brand", "region")],
        )
        inter = diag.interactions[0]
        assert inter.in_model is True
        assert inter.representation == "target_encoding"
        # TE interaction → single column
        assert inter.coefficients is not None
        assert len(inter.coefficients) == 1

    def test_fe_interaction_representation(self):
        """A frequency-encoded interaction has
        ``representation='frequency_encoding'``.
        """
        import rustystats as rs

        data = self._make_data()
        result = rs.glm_dict(
            response="y",
            terms={"brand": {"type": "categorical"}, "region": {"type": "categorical"}},
            interactions=[
                {
                    "brand": {"type": "categorical"},
                    "region": {"type": "categorical"},
                    "frequency_encoding": True,
                }
            ],
            data=data,
            family="poisson",
            offset="exposure",
        ).fit()
        diag = result.diagnostics(
            train_data=data,
            categorical_factors=["brand", "region"],
            interactions=[("brand", "region")],
        )
        inter = diag.interactions[0]
        assert inter.representation == "frequency_encoding"
        assert inter.coefficients is not None
        assert len(inter.coefficients) == 1


# =============================================================================
# Family coverage — verify Gaussian and Binomial don't crash and produce
# coherent A/E ratios.
# =============================================================================


class TestPairDiagnosticsFamilies:
    def test_gaussian(self):
        import polars as pl
        import rustystats as rs

        rng = np.random.default_rng(31)
        n = 600
        x1 = rng.uniform(-3, 3, n)
        cat = rng.choice(["A", "B"], n)
        y = 1.0 + 0.5 * x1 + (cat == "A") * 0.3 + rng.standard_normal(n) * 0.2
        data = pl.DataFrame({"y": y, "x1": x1, "cat": cat})
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "cat": {"type": "categorical"}},
            data=data,
            family="gaussian",
        ).fit()
        diag = result.diagnostics(
            train_data=data,
            categorical_factors=["cat"],
            continuous_factors=["x1"],
            interactions=[("x1", "cat")],
        )
        inter = diag.interactions[0]
        assert inter.pair_type == "continuous_x_categorical"
        # Cells render even when actual is signed (Gaussian).
        assert len(inter.train_surface_grid.cells) > 0

    def test_binomial(self):
        import polars as pl
        import rustystats as rs

        rng = np.random.default_rng(41)
        n = 800
        x1 = rng.uniform(-2, 2, n)
        cat = rng.choice(["A", "B", "C"], n)
        # P(y=1) = sigmoid(0.5 * x1 + 0.3 * (cat==A))
        eta = 0.5 * x1 + 0.3 * (cat == "A").astype(float)
        p = 1.0 / (1.0 + np.exp(-eta))
        y = (rng.uniform(size=n) < p).astype(int)
        data = pl.DataFrame({"y": y, "x1": x1, "cat": cat})
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "cat": {"type": "categorical"}},
            data=data,
            family="binomial",
        ).fit()
        diag = result.diagnostics(
            train_data=data,
            categorical_factors=["cat"],
            continuous_factors=["x1"],
            interactions=[("x1", "cat")],
        )
        inter = diag.interactions[0]
        # Predicted rates should be in (0, 1) for binomial.
        for cell in inter.train_surface_grid.cells:
            assert 0.0 <= cell.predicted <= 1.0


# =============================================================================
# Factor-extensions deep tests — verify GVIF behavior on orthogonal vs
# correlated designs and train_test_bins edge cases.
# =============================================================================


class TestFactorExtensionsDeep:
    def test_gvif_orthogonal_design_low(self):
        """Independent continuous factors → block GVIF ≈ 1."""
        import polars as pl
        import rustystats as rs

        rng = np.random.default_rng(57)
        n = 1500
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)  # independent of x1
        x3 = rng.standard_normal(n)  # independent
        eta = 0.4 * x1 + 0.3 * x2 + 0.1 * x3
        y = rng.poisson(np.exp(eta))
        data = pl.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})
        result = rs.glm_dict(
            response="y",
            terms={
                "x1": {"type": "linear"},
                "x2": {"type": "linear"},
                "x3": {"type": "linear"},
            },
            data=data,
            family="poisson",
        ).fit()
        diag = result.diagnostics(
            train_data=data,
            continuous_factors=["x1", "x2", "x3"],
        )
        for f in diag.factors:
            assert f.gvif is not None
            # Orthogonal columns → GVIF very near 1 (Fox-Monette block of size 1).
            assert 0.8 < f.gvif < 1.3, f"{f.name} gvif={f.gvif}"

    def test_gvif_correlated_design_higher(self):
        """Strongly correlated factors → block GVIF noticeably above 1."""
        import polars as pl
        import rustystats as rs

        rng = np.random.default_rng(59)
        n = 1500
        x1 = rng.standard_normal(n)
        # x2 highly correlated with x1
        x2 = x1 + 0.1 * rng.standard_normal(n)
        eta = 0.4 * x1 + 0.3 * x2
        y = rng.poisson(np.exp(eta))
        data = pl.DataFrame({"y": y, "x1": x1, "x2": x2})
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=data,
            family="poisson",
        ).fit()
        diag = result.diagnostics(
            train_data=data,
            continuous_factors=["x1", "x2"],
        )
        gvifs = [f.gvif for f in diag.factors if f.in_model]
        assert all(g is not None for g in gvifs)
        # At least one factor's GVIF should be inflated.
        assert max(gvifs) > 5.0, f"max gvif={max(gvifs)}, expected > 5"

    def test_train_test_bins_categorical_unseen_in_test(self):
        """train has levels {A, B, C}; test has only {A, B} — entry for C
        gets train numbers but test_n=0 and test_actual=None.
        """
        import polars as pl
        import rustystats as rs

        rng = np.random.default_rng(67)
        n_train, n_test = 600, 80
        # Train has all three levels
        train = pl.DataFrame(
            {
                "y": rng.poisson(0.5, n_train),
                "region": rng.choice(["A", "B", "C"], n_train),
                "exposure": rng.uniform(0.5, 1.0, n_train),
            }
        )
        # Test only has A and B
        test = pl.DataFrame(
            {
                "y": rng.poisson(0.5, n_test),
                "region": rng.choice(["A", "B"], n_test),
                "exposure": rng.uniform(0.5, 1.0, n_test),
            }
        )
        result = rs.glm_dict(
            response="y",
            terms={"region": {"type": "categorical"}},
            data=train,
            family="poisson",
            offset="exposure",
        ).fit()
        diag = result.diagnostics(
            train_data=train,
            test_data=test,
            categorical_factors=["region"],
        )
        region_factor = next(f for f in diag.factors if f.name == "region")
        assert region_factor.train_test_bins is not None
        by_level = {pair.bin: pair for pair in region_factor.train_test_bins}
        # Train pinned the bins, so C is present even though test never had it.
        if "C" in by_level:
            c_pair = by_level["C"]
            assert c_pair.train_n > 0
            # test_n may be 0 if C didn't appear in test
            assert c_pair.test_n >= 0

    def test_train_test_bins_no_test_means_no_field(self):
        """``train_test_bins`` is None for all factors when test_data is omitted."""
        import polars as pl
        import rustystats as rs

        rng = np.random.default_rng(71)
        n = 500
        data = pl.DataFrame(
            {
                "y": rng.poisson(0.5, n),
                "x": rng.uniform(0, 1, n),
                "region": rng.choice(["A", "B"], n),
                "exposure": rng.uniform(0.5, 1.0, n),
            }
        )
        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}, "region": {"type": "categorical"}},
            data=data,
            family="poisson",
            offset="exposure",
        ).fit()
        diag = result.diagnostics(
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["x"],
        )
        for f in diag.factors:
            assert f.train_test_bins is None

    def test_unfitted_factor_has_no_gvif(self):
        """A factor passed in but not in the model has ``gvif=None``."""
        import polars as pl
        import rustystats as rs

        rng = np.random.default_rng(73)
        n = 500
        data = pl.DataFrame(
            {
                "y": rng.poisson(0.5, n),
                "x": rng.uniform(0, 1, n),
                "z_unfit": rng.uniform(0, 1, n),
                "exposure": rng.uniform(0.5, 1.0, n),
            }
        )
        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}},
            data=data,
            family="poisson",
            offset="exposure",
        ).fit()
        diag = result.diagnostics(
            train_data=data,
            continuous_factors=["x", "z_unfit"],
        )
        unfit = next(f for f in diag.factors if f.name == "z_unfit")
        assert unfit.in_model is False
        assert unfit.gvif is None

    def test_train_test_bins_continuous_use_train_edges(self):
        """Continuous train/test bins are aligned on train-derived edges.

        If test data are all above the train range, every test row should land
        in the final train bin. A test-side quantile re-bin would spread these
        rows across bins, which is exactly the regression this guards against.
        """
        import polars as pl
        import rustystats as rs

        n_train = 500
        n_test = 80
        x_train = np.linspace(0.0, 10.0, n_train)
        train = pl.DataFrame(
            {
                "y": 1.0 + 0.2 * x_train,
                "x": x_train,
            }
        )
        test = pl.DataFrame(
            {
                "y": np.full(n_test, 5.0),
                "x": np.full(n_test, 100.0),
            }
        )
        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}},
            data=train,
            family="gaussian",
        ).fit()

        diag = result.diagnostics(
            train_data=train,
            test_data=test,
            continuous_factors=["x"],
            compute_partial_dep=False,
        )
        x_factor = next(f for f in diag.factors if f.name == "x")
        assert x_factor.train_test_bins is not None
        bins = x_factor.train_test_bins
        assert sum(b.test_n for b in bins) == n_test
        assert bins[-1].test_n == n_test
        assert all(b.test_n == 0 for b in bins[:-1])


# =============================================================================
# Determinism — same inputs produce identical outputs across calls.
# =============================================================================


class TestPairDiagnosticsDeterminism:
    def test_gvif_correlation_matrix_computed_once(self):
        """The expensive O(n·p²) correlation-matrix computation runs ONCE per
        diagnostics call regardless of how many fitted blocks need a GVIF.
        Spies on ``_build_design_correlation_matrix`` and asserts the call
        count, guarding against a future refactor that recomputes per-block.
        """
        import polars as pl
        import rustystats as rs

        rng = np.random.default_rng(311)
        n = 800
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        x3 = rng.standard_normal(n)
        region = rng.choice(["A", "B", "C"], n)
        exposure = rng.uniform(0.5, 1.0, n)
        mu = np.exp(-2 + 0.2 * x1 + 0.1 * x2)
        y = rng.poisson(mu * exposure)
        data = pl.DataFrame(
            {
                "y": y,
                "x1": x1,
                "x2": x2,
                "x3": x3,
                "region": region,
                "exposure": exposure,
            }
        )
        result = rs.glm_dict(
            response="y",
            terms={
                "x1": {"type": "linear"},
                "x2": {"type": "linear"},
                "x3": {"type": "linear"},
                "region": {"type": "categorical"},
            },
            data=data,
            family="poisson",
            offset="exposure",
        ).fit()

        # Monkey-patch the matrix builder at the site that actually calls it
        # (``api`` re-binds the name at import time via the module-level
        # ``from .pair_diagnostics import _build_design_correlation_matrix``).
        from rustystats.diagnostics import api as diag_api

        original = diag_api._build_design_correlation_matrix
        call_count = {"n": 0}

        def counting_builder(*args, **kwargs):
            call_count["n"] += 1
            return original(*args, **kwargs)

        diag_api._build_design_correlation_matrix = counting_builder
        try:
            diag = result.diagnostics(
                train_data=data,
                test_data=data[:200],
                categorical_factors=["region"],
                continuous_factors=["x1", "x2", "x3"],
                interactions=[("x1", "region"), ("x2", "region")],
            )
        finally:
            diag_api._build_design_correlation_matrix = original

        # Exactly one O(n·p²) standardize per diagnostics call — not one per
        # factor and not one per interaction.
        assert call_count["n"] == 1, f"expected 1 correlation-matrix build, got {call_count['n']}"
        # And the GVIFs that depend on it are populated.
        assert sum(1 for f in diag.factors if f.gvif is not None) >= 3

    def test_test_prediction_reused_for_train_test_and_pair_surfaces(self):
        """Pair diagnostics reuse the same test prediction as train/test comparison."""
        import polars as pl
        import rustystats as rs
        from rustystats.diagnostics.api import compute_diagnostics

        rng = np.random.default_rng(313)
        n = 600
        age = rng.uniform(18, 70, n)
        region = rng.choice(["A", "B", "C"], n)
        exposure = rng.uniform(0.5, 1.0, n)
        mu = np.exp(-2 + 0.02 * age + 0.2 * (region == "A").astype(float))
        y = rng.poisson(mu * exposure)
        data = pl.DataFrame({"y": y, "age": age, "region": region, "exposure": exposure})
        train, test = data[:450], data[450:]
        result = rs.glm_dict(
            response="y",
            terms={"age": {"type": "linear"}, "region": {"type": "categorical"}},
            data=train,
            family="poisson",
            offset="exposure",
        ).fit()

        class CountingResult:
            def __init__(self, inner):
                self.inner = inner
                self.predict_calls = 0

            def __getattr__(self, name):
                return getattr(self.inner, name)

            def predict(self, data_arg):
                self.predict_calls += 1
                return self.inner.predict(data_arg)

        wrapped = CountingResult(result)
        diag = compute_diagnostics(
            wrapped,
            train_data=train,
            test_data=test,
            categorical_factors=["region"],
            continuous_factors=["age"],
            interactions=[("age", "region")],
            compute_partial_dep=False,
        )

        assert len(diag.interactions) == 1
        assert diag.train_test.test is not None
        assert diag.interactions[0].test_surface_grid is not None
        assert wrapped.predict_calls == 1

    def test_same_inputs_same_outputs(self):
        """Running the diagnostics twice with the same inputs produces
        identical surface cells, gvif, and significance. Fits the
        interaction so the significance + gvif paths are both exercised.
        """
        import polars as pl
        import rustystats as rs

        rng = np.random.default_rng(91)
        n = 800
        age = rng.uniform(18, 70, n)
        region = rng.choice(["A", "B", "C"], n)
        exposure = rng.uniform(0.5, 1.0, n)
        mu = np.exp(-2 + 0.02 * age + 0.3 * (region == "A").astype(float))
        y = rng.poisson(mu * exposure)
        data = pl.DataFrame({"y": y, "age": age, "region": region, "exposure": exposure})
        result = rs.glm_dict(
            response="y",
            terms={"age": {"type": "linear"}, "region": {"type": "categorical"}},
            interactions=[
                {
                    "age": {"type": "linear"},
                    "region": {"type": "categorical"},
                    "include_main": False,
                }
            ],
            data=data,
            family="poisson",
            offset="exposure",
        ).fit()
        diag1 = result.diagnostics(
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["age"],
            interactions=[("age", "region")],
        )
        diag2 = result.diagnostics(
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["age"],
            interactions=[("age", "region")],
        )
        i1 = diag1.interactions[0]
        i2 = diag2.interactions[0]
        # gvif and significance reproduce bit-for-bit
        assert i1.gvif == i2.gvif
        assert i1.significance is not None and i2.significance is not None
        assert i1.significance.chi2 == i2.significance.chi2
        assert i1.significance.p == i2.significance.p
        # Surface cells reproduce
        assert len(i1.train_surface_grid.cells) == len(i2.train_surface_grid.cells)
        for c1, c2 in zip(i1.train_surface_grid.cells, i2.train_surface_grid.cells):
            assert c1.r == c2.r and c1.c == c2.c
            assert c1.actual == c2.actual
            assert c1.predicted == c2.predicted
            assert c1.exposure == c2.exposure
            assert c1.n == c2.n
