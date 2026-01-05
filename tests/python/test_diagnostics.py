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
            data=data,
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
            data=data,
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
            data=data,
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
            data=data,
            categorical_factors=["region"],
            continuous_factors=["age"],
        )
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert "model_summary" in parsed


class TestWarnings:
    """Tests for diagnostic warnings."""
    
    def test_high_dispersion_warning(self):
        """Test that high dispersion generates a warning."""
        from rustystats.diagnostics import DiagnosticsComputer
        
        np.random.seed(42)
        n = 500
        
        # Create overdispersed data
        y = np.random.negative_binomial(n=1, p=0.5, size=n).astype(np.float64)
        mu = np.full(n, np.mean(y))
        
        computer = DiagnosticsComputer(
            y=y, mu=mu, linear_predictor=np.log(mu),
            family="poisson", n_params=1, deviance=1000.0,
        )
        
        fit_stats = computer.compute_fit_statistics()
        calibration = computer.compute_calibration()
        
        warnings = computer.generate_warnings(fit_stats, calibration, [])
        
        # Should have at least one warning about dispersion
        warning_types = [w["type"] for w in warnings]
        assert "high_dispersion" in warning_types or fit_stats["dispersion_pearson"] < 1.5


class TestCalibrationBins:
    """Tests for calibration bin computation."""
    
    def test_calibration_bins_count(self):
        """Test that calibration returns compressed format with problem_deciles."""
        from rustystats.diagnostics import DiagnosticsComputer
        
        np.random.seed(42)
        n = 1000
        
        y = np.random.poisson(1.0, n).astype(np.float64)
        mu = np.random.uniform(0.5, 1.5, n)
        
        computer = DiagnosticsComputer(
            y=y, mu=mu, linear_predictor=np.log(mu),
            family="poisson", n_params=1, deviance=100.0,
        )
        
        calibration = computer.compute_calibration(n_bins=10)
        # Compressed format: problem_deciles only contains deciles with A/E outside [0.9, 1.1]
        assert "ae_ratio" in calibration
        assert "problem_deciles" in calibration
        assert isinstance(calibration["problem_deciles"], list)
    
    def test_calibration_bins_coverage(self):
        """Test that calibration returns problem_deciles in compressed format."""
        from rustystats.diagnostics import DiagnosticsComputer
        
        np.random.seed(42)
        n = 1000
        
        y = np.random.poisson(1.0, n).astype(np.float64)
        mu = np.random.uniform(0.5, 1.5, n)
        
        computer = DiagnosticsComputer(
            y=y, mu=mu, linear_predictor=np.log(mu),
            family="poisson", n_params=1, deviance=100.0,
        )
        
        calibration = computer.compute_calibration(n_bins=10)
        
        # Compressed format only includes problem deciles (A/E outside [0.9, 1.1])
        assert "problem_deciles" in calibration
        for decile in calibration["problem_deciles"]:
            assert "decile" in decile
            assert "ae" in decile
            assert "n" in decile


class TestDiscriminationMetrics:
    """Tests for discrimination metrics (Lorenz curve removed for compression)."""
    
    def test_discrimination_compressed_format(self):
        """Test that discrimination uses compressed format without lorenz_curve."""
        from rustystats.diagnostics import DiagnosticsComputer
        
        np.random.seed(42)
        n = 500
        
        y = np.random.poisson(1.0, n).astype(np.float64)
        mu = np.random.uniform(0.5, 1.5, n)
        
        computer = DiagnosticsComputer(
            y=y, mu=mu, linear_predictor=np.log(mu),
            family="poisson", n_params=1, deviance=100.0,
        )
        
        disc = computer.compute_discrimination()
        
        # Compressed format: no lorenz_curve, shortened field names
        assert "gini" in disc
        assert "auc" in disc
        assert "ks" in disc
        assert "lift_10pct" in disc
        assert "lift_20pct" in disc
        assert "lorenz_curve" not in disc  # Removed for token efficiency


class TestPreFitExploration:
    """Tests for pre-fit data exploration."""
    
    @pytest.fixture
    def exploration_data(self):
        """Create sample data for exploration."""
        np.random.seed(42)
        n = 1000
        
        age = np.random.uniform(18, 70, n)
        region = np.random.choice(["A", "B", "C", "D"], n)
        income = np.random.uniform(20000, 100000, n)
        exposure = np.random.uniform(0.5, 1.0, n)
        
        # Generate response with some pattern
        mu_true = np.exp(-2 + 0.02 * age + 0.5 * (region == "A").astype(float))
        y = np.random.poisson(mu_true * exposure)
        
        data = pl.DataFrame({
            "y": y,
            "age": age,
            "region": region,
            "income": income,
            "exposure": exposure,
        })
        
        return data
    
    def test_explore_data_function(self, exploration_data):
        """Test the explore_data function."""
        from rustystats.diagnostics import explore_data
        
        result = explore_data(
            data=exploration_data,
            response="y",
            categorical_factors=["region"],
            continuous_factors=["age", "income"],
            exposure="exposure",
        )
        
        assert result is not None
        assert result.data_summary is not None
        assert result.response_stats is not None
        assert len(result.factor_stats) == 3  # region, age, income
    
    def test_explore_data_response_stats(self, exploration_data):
        """Test response statistics in exploration."""
        from rustystats.diagnostics import explore_data
        
        result = explore_data(
            data=exploration_data,
            response="y",
            exposure="exposure",
        )
        
        assert "n_observations" in result.response_stats
        assert "total_exposure" in result.response_stats
        assert "mean_rate" in result.response_stats
        assert result.response_stats["n_observations"] == len(exploration_data)
    
    def test_explore_data_interaction_detection(self, exploration_data):
        """Test interaction detection in pre-fit exploration."""
        from rustystats.diagnostics import explore_data
        
        result = explore_data(
            data=exploration_data,
            response="y",
            categorical_factors=["region"],
            continuous_factors=["age", "income"],
            exposure="exposure",
            detect_interactions=True,
        )
        
        # Should find at least one interaction candidate
        # (depends on data, so just check structure)
        assert isinstance(result.interaction_candidates, list)
    
    def test_explore_data_to_json(self, exploration_data):
        """Test JSON serialization of exploration results."""
        from rustystats.diagnostics import explore_data
        import json
        
        result = explore_data(
            data=exploration_data,
            response="y",
            categorical_factors=["region"],
            continuous_factors=["age"],
            exposure="exposure",
        )
        
        json_str = result.to_json()
        parsed = json.loads(json_str)
        
        assert "data_summary" in parsed
        assert "response_stats" in parsed
        assert "factor_stats" in parsed
    
    def test_explore_method_on_model(self, exploration_data):
        """Test the explore method on FormulaGLM."""
        import rustystats as rs
        
        model = rs.glm(
            "y ~ age + C(region)",
            exploration_data,
            family="poisson",
            offset="exposure",
        )
        
        # Explore before fitting
        exploration = model.explore(
            categorical_factors=["region"],
            continuous_factors=["age", "income"],
        )
        
        assert exploration is not None
        assert len(exploration.factor_stats) >= 2
        
        # Can still fit after exploring
        result = model.fit()
        assert result.converged


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
            data=data,
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
            data=data,
            continuous_factors=["x"],
        )
        
        assert diag.fit_statistics["deviance"] > 0
        # Should have discrimination metrics for binomial (compressed field names)
        assert diag.discrimination is not None
        assert "gini" in diag.discrimination
