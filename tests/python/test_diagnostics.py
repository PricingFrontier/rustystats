"""
Tests for model diagnostics functionality.
"""

import pytest
import numpy as np

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
        
        data = pl.DataFrame({
            "y": y,
            "age": age,
            "region": region,
            "income": income,
            "exposure": exposure,
        })
        
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
            y=y, mu=mu, linear_predictor=lp,
            family="poisson", n_params=3, deviance=100.0,
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
            y=y, mu=mu, linear_predictor=lp,
            family="poisson", n_params=3, deviance=100.0,
        )
        
        metrics = computer.compute_loss_metrics()
        
        assert "mse" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "family_deviance_loss" in metrics
        assert metrics["mse"] >= 0
        assert metrics["mae"] >= 0
    
    def test_calibration(self, sample_data):
        """Test calibration computation."""
        from rustystats.diagnostics import DiagnosticsComputer
        
        y = sample_data["y"].to_numpy().astype(np.float64)
        mu = np.maximum(y.astype(np.float64), 0.1)
        lp = np.log(mu)
        
        computer = DiagnosticsComputer(
            y=y, mu=mu, linear_predictor=lp,
            family="poisson", n_params=3, deviance=100.0,
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
            y=y, mu=mu, linear_predictor=lp,
            family="poisson", n_params=3, deviance=100.0,
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
            y=y, mu=mu, linear_predictor=lp,
            family="poisson", n_params=3, deviance=100.0,
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
            y=y, mu=mu, linear_predictor=lp,
            family="poisson", n_params=3, deviance=100.0,
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
        assert age_factor.in_model == True  # "age" is in feature_names
        assert len(age_factor.actual_vs_expected) == 5
        
        income_factor = next(f for f in factors if f.name == "income")
        assert income_factor.in_model == False  # "income" not in model
    
    def test_factor_diagnostics_categorical(self, sample_data):
        """Test factor diagnostics for categorical variables."""
        from rustystats.diagnostics import DiagnosticsComputer
        
        y = sample_data["y"].to_numpy().astype(np.float64)
        mu = np.maximum(y.astype(np.float64), 0.1)
        lp = np.log(mu)
        
        computer = DiagnosticsComputer(
            y=y, mu=mu, linear_predictor=lp,
            family="poisson", n_params=3, deviance=100.0,
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
        assert region_factor.in_model == True  # region is in feature_names
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
        mu_true = np.exp(1.0 + 0.1 * x1 * (x2 == "A").astype(float) 
                        + 0.3 * x1 * (x2 == "B").astype(float))
        y = np.random.poisson(mu_true)
        
        # Predictions without interaction
        mu_pred = np.exp(1.0 + 0.2 * x1)
        
        return pl.DataFrame({
            "y": y,
            "x1": x1,
            "x2": x2,
        }), y, mu_pred
    
    def test_interaction_detection(self, interaction_data):
        """Test that interaction detection finds the interaction."""
        from rustystats.diagnostics import DiagnosticsComputer
        
        data, y, mu = interaction_data
        y = y.astype(np.float64)
        mu = mu.astype(np.float64)
        lp = np.log(mu)
        
        computer = DiagnosticsComputer(
            y=y, mu=mu, linear_predictor=lp,
            family="poisson", n_params=2, deviance=100.0,
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
        assert (top.factor1 == "x1" and top.factor2 == "x2") or \
               (top.factor1 == "x2" and top.factor2 == "x1")


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
        
        data = pl.DataFrame({
            "y": y,
            "age": age,
            "region": region,
        })
        
        result = rs.glm(
            "y ~ age + C(region)",
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
        assert diagnostics.fit_statistics is not None
        assert diagnostics.loss_metrics is not None
        assert diagnostics.calibration is not None
        assert len(diagnostics.factors) == 2
    
    def test_diagnostics_to_json(self, fitted_model):
        """Test JSON serialization."""
        from rustystats.diagnostics import compute_diagnostics
        import json
        
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
        assert "fit_statistics" in parsed
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
        result = rs.glm("y ~ x", data, family="gaussian").fit()
        
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
        result = rs.glm("y ~ x", data, family="binomial").fit()
        
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
        
        assert diag.fit_statistics["deviance"] > 0
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
        
        assert diag.fit_statistics["deviance"] > 0
        # Should have discrimination metrics for binomial (compressed field names)
        assert diag.discrimination is not None
        assert "gini" in diag.discrimination


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
        
        return pl.DataFrame({
            "ClaimNb": y,
            "Age": age,
            "Region": region,
            "Exposure": exposure,
        })
    
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
        from rustystats.diagnostics import explore_data
        import json
        
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
        """Test explore method on FormulaGLM."""
        import rustystats as rs
        
        np.random.seed(42)
        n = 200
        
        age = np.random.uniform(18, 70, n)
        region = np.random.choice(["A", "B"], n)
        y = np.random.poisson(np.exp(-2 + 0.02 * age), n)
        
        data = pl.DataFrame({
            "y": y,
            "age": age,
            "region": region,
        })
        
        model = rs.glm(
            "y ~ age + C(region)",
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
        
        data = pl.DataFrame({
            "y": y,
            "age": age,
            "veh_power": veh_power,
            "region": region,
            "exposure": exposure,
        })
        
        result = rs.glm(
            "y ~ age + veh_power + C(region)",
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
            y=y, mu=mu, linear_predictor=lp,
            family="poisson", n_params=len(result.params), deviance=result.deviance,
            feature_names=result.feature_names,
        )
        
        # Create a simple design matrix for testing
        X = np.column_stack([
            np.ones(len(y)),
            data["age"].to_numpy(),
            data["veh_power"].to_numpy(),
        ])
        
        vif_results = computer.compute_vif(X, ["Intercept", "age", "veh_power"])
        
        assert len(vif_results) == 2  # Excludes intercept
        for v in vif_results:
            assert hasattr(v, 'feature')
            assert hasattr(v, 'vif')
            assert hasattr(v, 'severity')
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
            y=y, mu=mu, linear_predictor=np.log(mu),
            family="poisson", n_params=3, deviance=100.0,
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
            y=y, mu=mu, linear_predictor=result.linear_predictor,
            family="poisson", n_params=len(result.params), deviance=result.deviance,
            feature_names=result.feature_names,
        )
        
        coef_summary = computer.compute_coefficient_summary(result, link="log")
        
        assert len(coef_summary) == len(result.params)
        
        for cs in coef_summary:
            assert hasattr(cs, 'feature')
            assert hasattr(cs, 'estimate')
            assert hasattr(cs, 'relativity')
            assert hasattr(cs, 'impact')
            assert hasattr(cs, 'significant')
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
            y=y, mu=mu, linear_predictor=result.linear_predictor,
            family="poisson", n_params=len(result.params), deviance=result.deviance,
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
            y=y, mu=mu, linear_predictor=result.linear_predictor,
            family="poisson", n_params=len(result.params), deviance=result.deviance,
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
            y=y, mu=mu, linear_predictor=result.linear_predictor,
            family="poisson", n_params=len(result.params), deviance=result.deviance,
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
            assert hasattr(pd, 'variable')
            assert hasattr(pd, 'variable_type')
            assert hasattr(pd, 'shape')
            assert hasattr(pd, 'recommendation')
            assert len(pd.grid_values) > 0
            assert len(pd.predictions) == len(pd.grid_values)
    
    def test_train_test_metrics(self, fitted_model_with_data):
        """Test train vs test metrics computation."""
        from rustystats.diagnostics import DiagnosticsComputer
        
        result, data = fitted_model_with_data
        
        y = data["y"].to_numpy().astype(np.float64)
        mu = result.fittedvalues
        
        computer = DiagnosticsComputer(
            y=y, mu=mu, linear_predictor=result.linear_predictor,
            family="poisson", n_params=len(result.params), deviance=result.deviance,
        )
        
        # Create fake test data
        np.random.seed(123)
        n_test = 100
        y_test = np.random.poisson(1.0, n_test).astype(np.float64)
        mu_test = np.random.uniform(0.5, 1.5, n_test)
        
        train_test = computer.compute_train_test_metrics(y_test, mu_test)
        
        assert "train" in train_test
        assert "test" in train_test
        
        train = train_test["train"]
        test = train_test["test"]
        
        assert train.dataset == "train"
        assert test.dataset == "test"
        assert train.n_obs == len(y)
        assert test.n_obs == n_test
        assert train.gini is not None
        assert test.gini is not None
    
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
        assert "impact" in coef_summary[0]
    
    def test_multicollinearity_warning(self):
        """Test that multicollinearity generates appropriate warnings."""
        import rustystats as rs
        
        np.random.seed(42)
        n = 500
        
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.05  # Nearly collinear
        y = np.random.poisson(np.exp(1 + x1), n)
        
        data = pl.DataFrame({"y": y, "x1": x1, "x2": x2})
        
        result = rs.glm("y ~ x1 + x2", data, family="poisson").fit()
        
        diagnostics = result.diagnostics(
            train_data=data,
            continuous_factors=["x1", "x2"],
        )
        
        # Should have multicollinearity warning
        warning_types = [w["type"] for w in diagnostics.warnings]
        assert "multicollinearity" in warning_types or "multicollinearity_moderate" in warning_types
    
    def test_train_test_comparison(self, fitted_model_with_data):
        """Test comprehensive train vs test comparison."""
        import rustystats as rs
        
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
        
        test_data = pl.DataFrame({
            "y": y,
            "age": age,
            "veh_power": veh_power,
            "region": region,
            "exposure": exposure,
        })
        
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
        assert hasattr(tt, 'gini_gap')
        assert hasattr(tt, 'ae_ratio_diff')
        assert hasattr(tt, 'overfitting_risk')
        assert hasattr(tt, 'calibration_drift')
        assert hasattr(tt, 'unstable_factors')
        
        # Check decile comparison
        assert len(tt.decile_comparison) == 10
        for d in tt.decile_comparison:
            assert "decile" in d
            assert "train_ae" in d
            assert "test_ae" in d
