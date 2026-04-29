"""
Dict API tests for RustyStats.

Comprehensive tests for rs.glm_dict() covering construction, fitting,
model results, serialization, interactions, regularization, diagnostics,
splines, and prediction consistency.
"""

import numpy as np
import polars as pl
import pytest
import rustystats as rs

# =============================================================================
# Construction Tests
# =============================================================================


class TestDictConstruction:
    """Test FormulaGLMDict construction."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        return pl.DataFrame(
            {
                "y": np.random.poisson(1, n),
                "x1": np.random.uniform(0, 10, n),
                "x2": np.random.uniform(0, 10, n),
                "cat": np.random.choice(["A", "B", "C"], n),
                "exposure": np.random.uniform(0.5, 1.5, n),
                "weight": np.random.uniform(0.5, 2.0, n),
            }
        )

    def test_basic_construction(self, sample_data):
        model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=sample_data,
        )
        assert model.family == "gaussian"
        assert model.n_obs == 100
        assert model.n_params == 3  # Intercept + x1 + x2

    def test_poisson_construction(self, sample_data):
        model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=sample_data,
            family="poisson",
        )
        assert model.family == "poisson"

    def test_binomial_construction(self):
        np.random.seed(42)
        data = pl.DataFrame(
            {
                "y": np.random.binomial(1, 0.5, 100),
                "x1": np.random.uniform(0, 10, 100),
            }
        )
        model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=data,
            family="binomial",
        )
        assert model.family == "binomial"

    def test_gamma_construction(self):
        np.random.seed(42)
        data = pl.DataFrame(
            {
                "y": np.random.gamma(2, 2, 100),
                "x1": np.random.uniform(0, 10, 100),
            }
        )
        model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=data,
            family="gamma",
        )
        assert model.family == "gamma"

    def test_offset_as_column_name(self, sample_data):
        model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=sample_data,
            family="poisson",
            offset="exposure",
        )
        assert model.offset is not None
        assert len(model.offset) == 100

    def test_offset_as_array(self, sample_data):
        offset_arr = np.log(sample_data["exposure"].to_numpy())
        model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=sample_data,
            family="poisson",
            offset=offset_arr,
        )
        assert model.offset is not None

    def test_weights_as_column_name(self, sample_data):
        model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=sample_data,
            weights="weight",
        )
        assert model.weights is not None
        assert len(model.weights) == 100

    def test_weights_as_array(self, sample_data):
        weights_arr = sample_data["weight"].to_numpy()
        model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=sample_data,
            weights=weights_arr,
        )
        assert model.weights is not None
        np.testing.assert_array_almost_equal(model.weights, weights_arr)


# =============================================================================
# Mirrors: test_formula_glm.py → TestGLMFitting
# =============================================================================


class TestDictFitting:
    """Test GLM fitting for various families (mirrors TestGLMFitting)."""

    def test_fit_gaussian(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 100)
        y = 2 + 3 * x + np.random.normal(0, 1, 100)
        data = pl.DataFrame({"y": y, "x": x})

        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}},
            data=data,
            family="gaussian",
        ).fit()

        assert result.converged
        assert len(result.params) == 2
        assert abs(result.params[0] - 2) < 1.0
        assert abs(result.params[1] - 3) < 0.5

    def test_fit_poisson(self):
        np.random.seed(42)
        n = 200
        x = np.random.uniform(0, 2, n)
        mu = np.exp(0.5 + 0.5 * x)
        y = np.random.poisson(mu)
        data = pl.DataFrame({"y": y, "x": x})

        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}},
            data=data,
            family="poisson",
        ).fit()

        assert result.converged
        assert len(result.params) == 2

    def test_fit_binomial(self):
        np.random.seed(42)
        n = 200
        x = np.random.uniform(-2, 2, n)
        p = 1 / (1 + np.exp(-(0.5 + x)))
        y = np.random.binomial(1, p)
        data = pl.DataFrame({"y": y.astype(float), "x": x})

        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}},
            data=data,
            family="binomial",
        ).fit()

        assert result.converged
        assert len(result.params) == 2

    def test_fit_gamma(self):
        np.random.seed(42)
        n = 200
        x = np.random.uniform(1, 5, n)
        mu = np.exp(1 + 0.3 * x)
        y = np.random.gamma(2, mu / 2, n)
        data = pl.DataFrame({"y": y, "x": x})

        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}},
            data=data,
            family="gamma",
        ).fit()

        assert result.converged
        assert len(result.params) == 2

    def test_fit_quasipoisson(self):
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 5, n)
        y = np.random.poisson(np.exp(0.5 + 0.3 * x))
        data = pl.DataFrame({"y": y, "x": x})

        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}},
            data=data,
            family="quasipoisson",
        ).fit()

        assert result.converged

    def test_fit_negbinomial(self):
        np.random.seed(42)
        n = 200
        x = np.random.uniform(0, 2, n)
        mu = np.exp(0.5 + 0.3 * x)
        y = np.random.negative_binomial(2, 2 / (2 + mu))
        data = pl.DataFrame({"y": y.astype(float), "x": x})

        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}},
            data=data,
            family="negbinomial",
        ).fit()

        assert result.converged
        assert "NegativeBinomial" in result.family

    def test_fit_with_offset(self):
        np.random.seed(42)
        n = 100
        exposure = np.random.uniform(0.5, 2, n)
        x = np.random.uniform(0, 5, n)
        y = np.random.poisson(exposure * np.exp(0.5 + 0.2 * x))
        data = pl.DataFrame({"y": y, "x": x, "exposure": exposure})

        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}},
            data=data,
            family="poisson",
            offset="exposure",
        ).fit()

        assert result.converged

    def test_fit_with_weights(self):
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 10, n)
        y = 2 + 3 * x + np.random.normal(0, 1, n)
        weights = np.random.uniform(0.5, 2, n)
        data = pl.DataFrame({"y": y, "x": x, "w": weights})

        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}},
            data=data,
            weights="w",
        ).fit()

        assert result.converged


# =============================================================================
# Mirrors: test_formula_glm.py → TestGLMModel
# =============================================================================


class TestDictModel:
    """Test GLMModel attributes and methods (mirrors TestGLMModel)."""

    @pytest.fixture
    def fitted_result(self):
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 10, n)
        y = 2 + 3 * x + np.random.normal(0, 1, n)
        data = pl.DataFrame({"y": y, "x": x})
        return rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}},
            data=data,
        ).fit()

    def test_params_shape(self, fitted_result):
        assert len(fitted_result.params) == 2

    def test_feature_names(self, fitted_result):
        assert "Intercept" in fitted_result.feature_names
        assert "x" in fitted_result.feature_names

    def test_fittedvalues_shape(self, fitted_result):
        assert len(fitted_result.fittedvalues) == 100

    def test_linear_predictor(self, fitted_result):
        assert len(fitted_result.linear_predictor) == 100

    def test_deviance(self, fitted_result):
        assert fitted_result.deviance >= 0

    def test_bse(self, fitted_result):
        bse = fitted_result.bse()
        assert len(bse) == 2
        assert np.all(bse > 0)

    def test_tvalues(self, fitted_result):
        tvals = fitted_result.tvalues()
        assert len(tvals) == 2
        assert np.all(np.isfinite(tvals))

    def test_pvalues(self, fitted_result):
        pvals = fitted_result.pvalues()
        assert len(pvals) == 2
        assert np.all(pvals >= 0)
        assert np.all(pvals <= 1)

    def test_conf_int(self, fitted_result):
        ci = fitted_result.conf_int(alpha=0.05)
        assert ci.shape == (2, 2)
        assert np.all(ci[:, 0] < ci[:, 1])

    def test_significance_codes(self, fitted_result):
        codes = fitted_result.significance_codes()
        assert len(codes) == 2
        valid_codes = ["***", "**", "*", ".", ""]
        for code in codes:
            assert code in valid_codes

    def test_summary(self, fitted_result):
        summary = fitted_result.summary()
        assert isinstance(summary, str)
        assert "Intercept" in summary
        assert "x" in summary


# =============================================================================
# Mirrors: test_formula_glm.py → TestRobustStandardErrors
# =============================================================================


class TestDictRobustSE:
    """Test robust standard errors (mirrors TestRobustStandardErrors)."""

    @pytest.fixture
    def heteroscedastic_result(self):
        np.random.seed(42)
        n = 200
        x = np.random.uniform(1, 10, n)
        y = 2 + 3 * x + np.random.normal(0, x, n)
        data = pl.DataFrame({"y": y, "x": x})
        return rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}},
            data=data,
        ).fit(store_design_matrix=True)

    def test_bse_robust_hc1(self, heteroscedastic_result):
        bse_robust = heteroscedastic_result.bse_robust("HC1")
        assert len(bse_robust) == 2
        assert np.all(bse_robust > 0)

    def test_bse_robust_hc0(self, heteroscedastic_result):
        bse_robust = heteroscedastic_result.bse_robust("HC0")
        assert len(bse_robust) == 2
        assert np.all(bse_robust > 0)

    def test_bse_robust_hc3(self, heteroscedastic_result):
        bse_robust = heteroscedastic_result.bse_robust("HC3")
        assert len(bse_robust) == 2
        assert np.all(bse_robust > 0)

    def test_robust_vs_model_se(self, heteroscedastic_result):
        bse_model = heteroscedastic_result.bse()
        bse_robust = heteroscedastic_result.bse_robust("HC1")
        assert not np.allclose(bse_model, bse_robust, rtol=0.01)

    def test_tvalues_robust(self, heteroscedastic_result):
        tvals = heteroscedastic_result.tvalues_robust("HC1")
        assert len(tvals) == 2
        assert np.all(np.isfinite(tvals))

    def test_pvalues_robust(self, heteroscedastic_result):
        pvals = heteroscedastic_result.pvalues_robust("HC1")
        assert len(pvals) == 2
        assert np.all(pvals >= 0)
        assert np.all(pvals <= 1)

    def test_conf_int_robust(self, heteroscedastic_result):
        ci = heteroscedastic_result.conf_int_robust(alpha=0.05, cov_type="HC1")
        assert ci.shape == (2, 2)
        assert np.all(ci[:, 0] < ci[:, 1])


# =============================================================================
# Mirrors: test_formula_glm.py → TestResiduals
# =============================================================================


class TestDictResiduals:
    """Test residual methods (mirrors TestResiduals)."""

    @pytest.fixture
    def fitted_result(self):
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 10, n)
        y = 2 + 3 * x + np.random.normal(0, 2, n)
        data = pl.DataFrame({"y": y, "x": x})
        return rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}},
            data=data,
        ).fit()

    def test_resid_response(self, fitted_result):
        resid = fitted_result.resid_response()
        assert len(resid) == 100
        assert abs(np.mean(resid)) < 1.0

    def test_resid_pearson(self, fitted_result):
        resid = fitted_result.resid_pearson()
        assert len(resid) == 100
        assert np.all(np.isfinite(resid))

    def test_resid_deviance(self, fitted_result):
        resid = fitted_result.resid_deviance()
        assert len(resid) == 100
        assert np.all(np.isfinite(resid))


# =============================================================================
# Mirrors: test_formula_glm.py → TestRegularization
# =============================================================================


class TestDictRegularization:
    """Test regularization options (mirrors TestRegularization)."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 200
        data = {"y": np.random.poisson(2, n)}
        for i in range(10):
            data[f"x{i}"] = np.random.uniform(0, 10, n)
        data["exposure"] = np.ones(n)
        return pl.DataFrame(data)

    def _terms_x(self, n):
        return {f"x{i}": {"type": "linear"} for i in range(n)}

    def test_ridge_regularization(self, sample_data):
        result = rs.glm_dict(
            response="y",
            terms=self._terms_x(10),
            data=sample_data,
            family="poisson",
            offset="exposure",
        ).fit(alpha=0.1, l1_ratio=0.0)

        assert result.converged
        assert result.is_regularized

    def test_lasso_regularization(self, sample_data):
        result = rs.glm_dict(
            response="y",
            terms=self._terms_x(10),
            data=sample_data,
            family="poisson",
            offset="exposure",
        ).fit(alpha=0.1, l1_ratio=1.0)

        assert result.converged
        assert result.is_regularized

    def test_elastic_net_regularization(self, sample_data):
        result = rs.glm_dict(
            response="y",
            terms=self._terms_x(10),
            data=sample_data,
            family="poisson",
            offset="exposure",
        ).fit(alpha=0.1, l1_ratio=0.5)

        assert result.converged
        assert result.is_regularized

    def test_no_regularization(self, sample_data):
        result = rs.glm_dict(
            response="y",
            terms={"x0": {"type": "linear"}, "x1": {"type": "linear"}},
            data=sample_data,
            family="poisson",
            offset="exposure",
        ).fit(alpha=0.0)

        assert result.converged
        assert not result.is_regularized


# =============================================================================
# Mirrors: test_formula_glm.py → TestPrediction
# =============================================================================


class TestDictPrediction:
    """Test prediction functionality (mirrors TestPrediction)."""

    def test_predict_on_training_data(self):
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 10, n)
        y = 2 + 3 * x + np.random.normal(0, 1, n)
        data = pl.DataFrame({"y": y, "x": x})

        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}},
            data=data,
        ).fit()
        predictions = result.fittedvalues

        assert len(predictions) == n
        assert np.corrcoef(predictions, y)[0, 1] > 0.9

    def test_predict_on_new_data(self):
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 10, n)
        y = 2 + 3 * x + np.random.normal(0, 1, n)
        train_data = pl.DataFrame({"y": y, "x": x})

        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}},
            data=train_data,
        ).fit()

        new_x = np.array([1.0, 5.0, 9.0])
        new_data = pl.DataFrame({"x": new_x})

        predictions = result.predict(new_data)

        assert len(predictions) == 3
        expected = 2 + 3 * new_x
        np.testing.assert_array_almost_equal(predictions, expected, decimal=0)


# =============================================================================
# Mirrors: test_formula_glm.py → TestEdgeCases
# =============================================================================


class TestDictEdgeCases:
    """Test edge cases (mirrors TestEdgeCases)."""

    def test_intercept_only_model(self):
        np.random.seed(42)
        data = pl.DataFrame({"y": np.random.normal(5, 1, 100)})

        result = rs.glm_dict(
            response="y",
            terms={},
            data=data,
        ).fit()

        assert result.converged
        assert len(result.params) == 1
        assert abs(result.params[0] - 5) < 0.5

    def test_no_intercept_model(self):
        np.random.seed(42)
        x = np.random.uniform(1, 10, 100)
        y = 3 * x + np.random.normal(0, 1, 100)
        data = pl.DataFrame({"y": y, "x": x})

        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}},
            data=data,
            intercept=False,
        ).fit()

        assert result.converged
        assert "Intercept" not in result.feature_names

    def test_single_observation_fails(self):
        data = pl.DataFrame({"y": [1.0], "x": [1.0]})
        with pytest.raises(rs.exceptions.ValidationError):
            rs.glm_dict(
                response="y",
                terms={"x": {"type": "linear"}},
                data=data,
            ).fit()

    def test_missing_variable_fails(self):
        data = pl.DataFrame({"y": [1.0, 2.0], "x": [1.0, 2.0]})
        with pytest.raises(
            (KeyError, rs.exceptions.ValidationError, pl.exceptions.ColumnNotFoundError)
        ):
            rs.glm_dict(
                response="y",
                terms={"z": {"type": "linear"}},
                data=data,
            ).fit()


# =============================================================================
# Mirrors: test_serialization.py
# =============================================================================


class TestDictSerialization:
    """Test serialization roundtrip (mirrors TestBasicSerialization)."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 500
        return pl.DataFrame(
            {
                "y": np.random.poisson(2, n).astype(float),
                "x1": np.random.randn(n),
                "x2": np.random.randn(n),
                "cat": np.random.choice(["A", "B", "C"], n),
                "exposure": np.random.uniform(0.5, 2.0, n),
            }
        )

    def test_simple_model_roundtrip(self, sample_data):
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=sample_data,
            family="poisson",
        ).fit()

        model_bytes = result.to_bytes()
        assert isinstance(model_bytes, bytes)
        assert len(model_bytes) > 0

        loaded = rs.GLMModel.from_bytes(model_bytes)

        assert loaded.family == result.family
        assert loaded.link == result.link
        assert len(loaded.params) == len(result.params)
        np.testing.assert_array_almost_equal(loaded.params, result.params)

    def test_categorical_model_roundtrip(self, sample_data):
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "cat": {"type": "categorical"}},
            data=sample_data,
            family="poisson",
        ).fit()

        model_bytes = result.to_bytes()
        loaded = rs.GLMModel.from_bytes(model_bytes)

        np.testing.assert_array_almost_equal(loaded.params, result.params)
        assert loaded.feature_names == result.feature_names

    def test_offset_model_roundtrip(self, sample_data):
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "cat": {"type": "categorical"}},
            data=sample_data,
            family="poisson",
            offset="exposure",
        ).fit()

        model_bytes = result.to_bytes()
        loaded = rs.GLMModel.from_bytes(model_bytes)

        np.testing.assert_array_almost_equal(loaded.params, result.params)


class TestDictSerializationPrediction:
    """Test predictions after load (mirrors TestPredictionAfterLoad)."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 500
        return pl.DataFrame(
            {
                "y": np.random.poisson(2, n).astype(float),
                "x1": np.random.randn(n),
                "x2": np.random.randn(n),
                "cat": np.random.choice(["A", "B", "C"], n),
                "exposure": np.random.uniform(0.5, 2.0, n),
            }
        )

    def test_predict_after_load(self, sample_data):
        train = sample_data.head(400)
        test = sample_data.tail(100)

        result = rs.glm_dict(
            response="y",
            terms={
                "x1": {"type": "linear"},
                "x2": {"type": "linear"},
                "cat": {"type": "categorical"},
            },
            data=train,
            family="poisson",
        ).fit()
        original_pred = result.predict(test)

        loaded = rs.GLMModel.from_bytes(result.to_bytes())
        loaded_pred = loaded.predict(test)

        np.testing.assert_array_almost_equal(original_pred, loaded_pred)

    def test_predict_with_offset_after_load(self, sample_data):
        train = sample_data.head(400)
        test = sample_data.tail(100)

        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "cat": {"type": "categorical"}},
            data=train,
            family="poisson",
            offset="exposure",
        ).fit()
        original_pred = result.predict(test)

        loaded = rs.GLMModel.from_bytes(result.to_bytes())
        loaded_pred = loaded.predict(test)

        np.testing.assert_array_almost_equal(original_pred, loaded_pred)


class TestDictSplineSerialization:
    """Test serialization with splines (mirrors TestSplineSerialization)."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 500
        return pl.DataFrame(
            {
                "y": np.random.poisson(2, n).astype(float),
                "x1": np.random.randn(n),
                "x2": np.random.randn(n),
                "cat": np.random.choice(["A", "B", "C"], n),
                "exposure": np.random.uniform(0.5, 2.0, n),
            }
        )

    def test_spline_model_roundtrip(self, sample_data):
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "bs", "df": 4}, "x2": {"type": "linear"}},
            data=sample_data,
            family="poisson",
        ).fit()

        model_bytes = result.to_bytes()
        loaded = rs.GLMModel.from_bytes(model_bytes)

        np.testing.assert_array_almost_equal(loaded.params, result.params)

    def test_spline_prediction_after_load(self, sample_data):
        train = sample_data.head(400)
        test = sample_data.tail(100)

        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "bs", "df": 4}, "cat": {"type": "categorical"}},
            data=train,
            family="poisson",
        ).fit()
        original_pred = result.predict(test)

        loaded = rs.GLMModel.from_bytes(result.to_bytes())
        loaded_pred = loaded.predict(test)

        np.testing.assert_array_almost_equal(original_pred, loaded_pred)

    def test_monotonic_bs_serialization_roundtrip(self):
        np.random.seed(42)
        n = 500
        x = np.random.uniform(0, 10, n)
        y = np.random.poisson(np.exp(0.1 * x), n).astype(float)
        data = pl.DataFrame({"y": y, "x": x})
        train = data.head(400)
        test = data.tail(100)

        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "bs", "df": 5, "monotonicity": "increasing"}},
            data=train,
            family="poisson",
        ).fit()
        original_pred = result.predict(test)

        model_bytes = result.to_bytes()
        loaded = rs.GLMModel.from_bytes(model_bytes)
        loaded_pred = loaded.predict(test)

        np.testing.assert_array_almost_equal(original_pred, loaded_pred)

    def test_monotonic_bs_decreasing_serialization(self):
        np.random.seed(42)
        n = 500
        x = np.random.uniform(0, 10, n)
        y = np.random.poisson(np.exp(2.0 - 0.15 * x), n).astype(float)
        data = pl.DataFrame({"y": y, "x": x})
        train = data.head(400)
        test = data.tail(100)

        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "bs", "df": 5, "monotonicity": "decreasing"}},
            data=train,
            family="poisson",
        ).fit()
        original_pred = result.predict(test)

        model_bytes = result.to_bytes()
        loaded = rs.GLMModel.from_bytes(model_bytes)
        loaded_pred = loaded.predict(test)

        np.testing.assert_array_almost_equal(original_pred, loaded_pred)

    def test_ms_type_serialization_roundtrip(self):
        np.random.seed(42)
        n = 500
        x = np.random.uniform(0, 10, n)
        y = np.random.poisson(np.exp(0.1 * x), n).astype(float)
        data = pl.DataFrame({"y": y, "x": x})
        train = data.head(400)
        test = data.tail(100)

        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "ms", "df": 5}},
            data=train,
            family="poisson",
        ).fit()
        original_pred = result.predict(test)

        model_bytes = result.to_bytes()
        loaded = rs.GLMModel.from_bytes(model_bytes)
        loaded_pred = loaded.predict(test)

        np.testing.assert_array_almost_equal(original_pred, loaded_pred)

    def test_monotonic_serialization_roundtrip_works(self):
        np.random.seed(42)
        n = 200
        x = np.random.uniform(0, 10, n)
        y = np.random.poisson(np.exp(0.1 * x), n).astype(float)
        data = pl.DataFrame({"y": y, "x": x})

        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "bs", "df": 5, "monotonicity": "increasing"}},
            data=data,
            family="poisson",
        ).fit()

        model_bytes = result.to_bytes()
        loaded = rs.GLMModel.from_bytes(model_bytes)
        new_data = pl.DataFrame({"x": np.linspace(0, 10, 50)})
        original_pred = result.predict(new_data)
        loaded_pred = loaded.predict(new_data)
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)


class TestDictInteractionSerialization:
    """Test serialization with interactions (mirrors TestInteractionSerialization)."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 500
        return pl.DataFrame(
            {
                "y": np.random.poisson(2, n).astype(float),
                "x1": np.random.randn(n),
                "x2": np.random.randn(n),
                "cat": np.random.choice(["A", "B", "C"], n),
                "exposure": np.random.uniform(0.5, 2.0, n),
            }
        )

    def test_interaction_model_roundtrip(self, sample_data):
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            interactions=[
                {"x1": {"type": "linear"}, "x2": {"type": "linear"}, "include_main": False},
            ],
            data=sample_data,
            family="poisson",
        ).fit()

        model_bytes = result.to_bytes()
        loaded = rs.GLMModel.from_bytes(model_bytes)

        np.testing.assert_array_almost_equal(loaded.params, result.params)

    def test_cat_interaction_prediction(self, sample_data):
        train = sample_data.head(400)
        test = sample_data.tail(100)

        result = rs.glm_dict(
            response="y",
            terms={"cat": {"type": "categorical"}, "x1": {"type": "linear"}},
            interactions=[
                {"cat": {"type": "categorical"}, "x1": {"type": "linear"}, "include_main": False},
            ],
            data=train,
            family="poisson",
        ).fit()
        original_pred = result.predict(test)

        loaded = rs.GLMModel.from_bytes(result.to_bytes())
        loaded_pred = loaded.predict(test)

        np.testing.assert_array_almost_equal(original_pred, loaded_pred)


class TestDictSerializationProperties:
    """Test model properties preserved (mirrors TestModelProperties)."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 500
        return pl.DataFrame(
            {
                "y": np.random.poisson(2, n).astype(float),
                "x1": np.random.randn(n),
                "x2": np.random.randn(n),
                "cat": np.random.choice(["A", "B", "C"], n),
                "exposure": np.random.uniform(0.5, 2.0, n),
            }
        )

    def test_properties_preserved(self, sample_data):
        result = rs.glm_dict(
            response="y",
            terms={
                "x1": {"type": "linear"},
                "x2": {"type": "linear"},
                "cat": {"type": "categorical"},
            },
            data=sample_data,
            family="poisson",
        ).fit()
        loaded = rs.GLMModel.from_bytes(result.to_bytes())

        assert loaded.deviance == result.deviance
        assert loaded.converged == result.converged
        assert loaded.iterations == result.iterations
        assert loaded.nobs == result.nobs
        assert loaded.df_resid == result.df_resid
        assert loaded.df_model == result.df_model

    def test_different_families(self, sample_data):
        for family in ["gaussian", "poisson", "gamma"]:
            data = sample_data.with_columns(pl.col("y").abs() + 0.1)
            result = rs.glm_dict(
                response="y",
                terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
                data=data,
                family=family,
            ).fit()
            loaded = rs.GLMModel.from_bytes(result.to_bytes())

            assert loaded.family == family
            np.testing.assert_array_almost_equal(loaded.params, result.params)

    def test_intercept_only_model(self, sample_data):
        result = rs.glm_dict(
            response="y",
            terms={},
            data=sample_data,
            family="poisson",
        ).fit()
        loaded = rs.GLMModel.from_bytes(result.to_bytes())

        np.testing.assert_array_almost_equal(loaded.params, result.params)


# =============================================================================
# Mirrors: test_interactions.py → TestGLMInteractions
# =============================================================================


class TestDictInteractions:
    """Test GLM fitting with interactions (mirrors TestGLMInteractions)."""

    @pytest.fixture
    def insurance_data(self):
        np.random.seed(42)
        n = 1000
        age = np.random.uniform(20, 70, n)
        power = np.random.uniform(50, 200, n)
        area = np.random.choice(["Urban", "Suburban", "Rural"], n)
        log_rate = -3.0 + 0.02 * age + 0.01 * power - 0.0001 * age * power
        log_rate += np.where(area == "Urban", 0.3, np.where(area == "Suburban", 0.1, 0.0))
        claims = np.random.poisson(np.exp(log_rate))
        exposure = np.random.uniform(0.5, 1.0, n)
        return pl.DataFrame(
            {
                "claims": claims,
                "age": age,
                "power": power,
                "area": area,
                "exposure": exposure,
            }
        )

    def test_fit_continuous_interaction(self, insurance_data):
        result = rs.glm_dict(
            response="claims",
            terms={"age": {"type": "linear"}, "power": {"type": "linear"}},
            interactions=[
                {"age": {"type": "linear"}, "power": {"type": "linear"}, "include_main": False},
            ],
            data=insurance_data,
            family="poisson",
            offset="exposure",
        ).fit()

        assert len(result.params) == 4  # Intercept, age, power, age:power
        assert result.converged
        summary = result.summary()
        assert "age:power" in summary

    def test_fit_categorical_continuous_interaction(self, insurance_data):
        result = rs.glm_dict(
            response="claims",
            terms={"area": {"type": "categorical"}, "age": {"type": "linear"}},
            interactions=[
                {"area": {"type": "categorical"}, "age": {"type": "linear"}, "include_main": False},
            ],
            data=insurance_data,
            family="poisson",
            offset="exposure",
        ).fit()

        # Intercept + 2 area dummies + age + 2 interactions = 6
        assert len(result.params) == 6
        assert result.converged

    def test_fit_categorical_categorical_interaction(self, insurance_data):
        insurance_data = insurance_data.with_columns(
            pl.Series("fuel", np.random.choice(["Petrol", "Diesel"], len(insurance_data)))
        )

        result = rs.glm_dict(
            response="claims",
            terms={"area": {"type": "categorical"}, "fuel": {"type": "categorical"}},
            interactions=[
                {
                    "area": {"type": "categorical"},
                    "fuel": {"type": "categorical"},
                    "include_main": False,
                },
            ],
            data=insurance_data,
            family="poisson",
            offset="exposure",
        ).fit()

        # area: 2 dummies, fuel: 1 dummy, Total: 1 + 2 + 1 + 2×1 = 6
        assert len(result.params) == 6
        assert result.converged

    def test_regularized_interaction_model(self, insurance_data):
        result = rs.glm_dict(
            response="claims",
            terms={
                "age": {"type": "linear"},
                "power": {"type": "linear"},
                "area": {"type": "categorical"},
            },
            interactions=[
                {"age": {"type": "linear"}, "power": {"type": "linear"}, "include_main": False},
            ],
            data=insurance_data,
            family="poisson",
            offset="exposure",
        ).fit(alpha=0.1, l1_ratio=0.0)

        assert result.is_regularized
        assert result.converged

    def test_predictions_with_interactions(self, insurance_data):
        result = rs.glm_dict(
            response="claims",
            terms={"age": {"type": "linear"}, "power": {"type": "linear"}},
            interactions=[
                {"age": {"type": "linear"}, "power": {"type": "linear"}, "include_main": False},
            ],
            data=insurance_data,
            family="poisson",
            offset="exposure",
        ).fit()

        fv = result.fittedvalues
        assert np.all(fv >= 0)
        assert len(fv) == len(insurance_data)

    def test_large_categorical_interaction(self):
        np.random.seed(42)
        n = 50_000
        df = pl.DataFrame(
            {
                "y": np.random.poisson(1, n),
                "cat1": np.random.choice([f"A{i}" for i in range(10)], n),
                "cat2": np.random.choice([f"B{i}" for i in range(8)], n),
                "exposure": np.random.uniform(0.5, 1.0, n),
            }
        )

        import time

        t0 = time.time()
        result = rs.glm_dict(
            response="y",
            terms={"cat1": {"type": "categorical"}, "cat2": {"type": "categorical"}},
            interactions=[
                {
                    "cat1": {"type": "categorical"},
                    "cat2": {"type": "categorical"},
                    "include_main": False,
                },
            ],
            data=df,
            family="poisson",
            offset="exposure",
        ).fit()
        t_opt = time.time() - t0

        assert t_opt < 30.0, f"Took {t_opt:.1f}s (expected < 30s)"
        # cat1: 9 dummies, cat2: 7 dummies, Total: 1 + 9 + 7 + 63 = 80
        assert len(result.params) == 80
        assert result.converged

    def test_fit_specific_levels_categorical(self):
        np.random.seed(42)
        data = pl.DataFrame(
            {
                "claims": np.random.poisson(0.1, 100),
                "Region": np.random.choice(["Paris", "Lyon", "Marseille"], 100),
                "age": np.random.uniform(20, 60, 100),
                "exposure": np.random.uniform(0.5, 1.5, 100),
            }
        )

        result = rs.glm_dict(
            response="claims",
            terms={
                "Region": {"type": "categorical", "levels": ["Paris"]},
                "age": {"type": "linear"},
            },
            data=data,
            family="poisson",
            offset="exposure",
        ).fit()

        assert result.converged
        assert "Region[Paris]" in result.feature_names
        assert len(result.params) == 3  # Intercept + Paris + age


# =============================================================================
# Mirrors: test_regularization_path.py → TestCVRegularizationFit
# =============================================================================


class TestDictCVPathNonLogLink:
    """Regression tests for CV path with non-log link functions (Phase 1 fix)."""

    def test_cv_path_gamma_identity_link(self):
        """CV path must use the link inverse, not hardcoded exp()."""
        np.random.seed(42)
        n = 300
        x = np.random.uniform(1, 5, n)
        mu = 2.0 + 0.5 * x  # identity link: mu = eta
        y = np.random.gamma(5, mu / 5, n)
        data = pl.DataFrame({"y": y, "x": x})

        model = rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}},
            data=data,
            family="gamma",
            link="identity",
        )
        result = model.fit(cv=3, regularization="ridge", n_alphas=5, verbose=False)

        assert result.cv_deviance is not None
        assert np.isfinite(result.cv_deviance)
        assert result.converged

    def test_cv_path_binomial_logit_link(self):
        """CV path must use logit inverse (sigmoid), not exp()."""
        np.random.seed(42)
        n = 500
        x = np.random.uniform(-2, 2, n)
        p = 1 / (1 + np.exp(-(0.5 + x)))
        y = np.random.binomial(1, p).astype(float)
        data = pl.DataFrame({"y": y, "x": x})

        model = rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}},
            data=data,
            family="binomial",
        )
        result = model.fit(cv=3, regularization="ridge", n_alphas=5, verbose=False)

        assert result.cv_deviance is not None
        assert np.isfinite(result.cv_deviance)
        assert result.converged


class TestDictCVRegularization:
    """Test CV-based regularization (mirrors TestCVRegularizationFit)."""

    @pytest.fixture
    def simple_data(self):
        np.random.seed(42)
        n = 500
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        x3 = np.random.randn(n)
        eta = 0.5 + 0.3 * x1 - 0.2 * x2 + 0.1 * x3
        y = np.random.poisson(np.exp(eta))
        return pl.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})

    @pytest.fixture
    def insurance_like_data(self):
        np.random.seed(123)
        n = 1000
        age = np.random.uniform(18, 80, n)
        bonus_malus = np.random.uniform(50, 200, n)
        exposure = np.random.uniform(0.1, 1.0, n)
        eta = -2.0 + 0.01 * age + 0.005 * bonus_malus
        claims = np.random.poisson(np.exp(eta) * exposure)
        return pl.DataFrame(
            {
                "ClaimCount": claims,
                "Age": age,
                "BonusMalus": bonus_malus,
                "Exposure": exposure,
            }
        )

    def test_ridge_cv_basic(self, simple_data):
        model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}, "x3": {"type": "linear"}},
            data=simple_data,
            family="poisson",
        )
        result = model.fit(cv=3, regularization="ridge", n_alphas=10, verbose=False)

        assert result.cv_deviance is not None
        assert result.cv_deviance_se is not None
        assert result.regularization_type in ("ridge", "none")
        assert result.regularization_path is not None

    def test_lasso_cv_basic(self, simple_data):
        model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}, "x3": {"type": "linear"}},
            data=simple_data,
            family="poisson",
        )
        result = model.fit(cv=3, regularization="lasso", n_alphas=10, verbose=False)

        assert result.cv_deviance is not None
        assert result.regularization_type in ("lasso", "none")

    def test_1se_selection_more_regularized(self, simple_data):
        model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}, "x3": {"type": "linear"}},
            data=simple_data,
            family="poisson",
        )
        result_min = model.fit(
            cv=3,
            regularization="ridge",
            n_alphas=20,
            selection="min",
            cv_seed=42,
            verbose=False,
        )
        result_1se = model.fit(
            cv=3,
            regularization="ridge",
            n_alphas=20,
            selection="1se",
            cv_seed=42,
            verbose=False,
        )
        assert result_1se.alpha >= result_min.alpha

    def test_cv_requires_regularization_type(self, simple_data):
        model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=simple_data,
            family="poisson",
        )
        with pytest.raises(rs.ValidationError, match="regularization"):
            model.fit(cv=5)

    def test_explicit_alpha_no_cv(self, simple_data):
        model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=simple_data,
            family="poisson",
        )
        result = model.fit(alpha=0.1, l1_ratio=0.0)
        assert result.alpha == pytest.approx(0.1)
        assert result.cv_deviance is None

    def test_poisson_with_offset(self, insurance_like_data):
        model = rs.glm_dict(
            response="ClaimCount",
            terms={"Age": {"type": "linear"}, "BonusMalus": {"type": "linear"}},
            data=insurance_like_data,
            family="poisson",
            offset="Exposure",
        )
        result = model.fit(cv=3, regularization="ridge", n_alphas=10, verbose=False)

        assert result.cv_deviance is not None
        assert result.converged

    def test_cv_reproducibility(self, insurance_like_data):
        model = rs.glm_dict(
            response="ClaimCount",
            terms={"Age": {"type": "linear"}, "BonusMalus": {"type": "linear"}},
            data=insurance_like_data,
            family="poisson",
            offset="Exposure",
        )
        result1 = model.fit(cv=3, regularization="ridge", n_alphas=10, cv_seed=42, verbose=False)
        result2 = model.fit(cv=3, regularization="ridge", n_alphas=10, cv_seed=42, verbose=False)

        assert result1.alpha == pytest.approx(result2.alpha)
        assert result1.cv_deviance == pytest.approx(result2.cv_deviance)

    def test_all_cv_attributes_present(self, simple_data):
        model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=simple_data,
            family="poisson",
        )
        result = model.fit(cv=3, regularization="ridge", n_alphas=10, verbose=False)

        assert result.cv_deviance is not None
        assert result.cv_deviance_se is not None
        assert result.regularization_type is not None
        assert result.regularization_path is not None
        assert result.cv_selection_method is not None
        assert result.n_cv_folds == 3

    def test_path_structure(self, simple_data):
        model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=simple_data,
            family="poisson",
        )
        result = model.fit(cv=3, regularization="ridge", n_alphas=10, verbose=False)
        path = result.regularization_path

        for entry in path:
            assert "alpha" in entry
            assert "l1_ratio" in entry
            assert "cv_deviance_mean" in entry
            assert "cv_deviance_se" in entry
            assert "n_nonzero" in entry
            assert "max_coef" in entry


# =============================================================================
# Mirrors: test_diagnostics.py → TestModelDiagnostics, TestDifferentFamilies,
#          TestPreFitExploration, TestEnhancedDiagnostics, TestScoreTest
# =============================================================================


class TestDictDiagnostics:
    """Test diagnostics (mirrors TestModelDiagnostics)."""

    @pytest.fixture
    def fitted_model(self):
        np.random.seed(42)
        n = 500
        age = np.random.uniform(18, 70, n)
        region = np.random.choice(["A", "B", "C"], n)
        mu_true = np.exp(-2 + 0.02 * age)
        y = np.random.poisson(mu_true)
        data = pl.DataFrame({"y": y, "age": age, "region": region})

        result = rs.glm_dict(
            response="y",
            terms={"age": {"type": "linear"}, "region": {"type": "categorical"}},
            data=data,
            family="poisson",
        ).fit()

        return result, data

    def test_compute_diagnostics(self, fitted_model):
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
        parsed = json.loads(json_str)

        assert "model_summary" in parsed
        assert "train_test" in parsed
        assert "factors" in parsed
        assert len(parsed["factors"]) == 2

    def test_diagnostics_method_on_result(self, fitted_model):
        result, data = fitted_model
        diagnostics = result.diagnostics(
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["age"],
        )
        assert diagnostics is not None
        assert len(diagnostics.factors) == 2

    def test_diagnostics_json_method(self, fitted_model):
        import json

        result, data = fitted_model
        json_str = result.diagnostics_json(
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["age"],
        )
        parsed = json.loads(json_str)
        assert "model_summary" in parsed


class TestDictDiagnosticsFamilies:
    """Test diagnostics with different families (mirrors TestDifferentFamilies)."""

    def test_gaussian_diagnostics(self):
        from rustystats.diagnostics import compute_diagnostics

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

        diag = compute_diagnostics(result=result, train_data=data, continuous_factors=["x"])
        assert diag.train_test.train.deviance > 0
        assert len(diag.factors) == 1

    def test_binomial_diagnostics(self):
        from rustystats.diagnostics import compute_diagnostics

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

        diag = compute_diagnostics(result=result, train_data=data, continuous_factors=["x"])
        assert diag.train_test.train.deviance > 0
        assert diag.train_test.train.gini is not None


class TestDictEnhancedDiagnostics:
    """Test enhanced diagnostics (mirrors TestEnhancedDiagnostics)."""

    @pytest.fixture
    def fitted_model_with_data(self):
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

    def test_full_diagnostics_with_enhancements(self, fitted_model_with_data):
        result, data = fitted_model_with_data

        diagnostics = result.diagnostics(
            train_data=data,
            categorical_factors=["region"],
            continuous_factors=["age", "veh_power"],
        )

        assert diagnostics.coefficient_summary is not None
        assert len(diagnostics.coefficient_summary) > 0
        assert diagnostics.lift_chart is not None
        assert len(diagnostics.lift_chart.deciles) == 10
        assert diagnostics.factor_deviance is not None
        assert len(diagnostics.factor_deviance) == 1
        assert diagnostics.partial_dependence is not None
        assert len(diagnostics.partial_dependence) == 3

    def test_diagnostics_json_includes_enhancements(self, fitted_model_with_data):
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

    def test_multicollinearity_warning(self):
        np.random.seed(42)
        n = 500
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.05
        y = np.random.poisson(np.exp(1 + x1), n)
        data = pl.DataFrame({"y": y, "x1": x1, "x2": x2})

        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=data,
            family="poisson",
        ).fit()

        diagnostics = result.diagnostics(train_data=data, continuous_factors=["x1", "x2"])

        assert diagnostics.vif is not None
        assert len(diagnostics.vif) > 0
        for v in diagnostics.vif:
            assert v.severity in ("severe", "moderate")

    def test_train_test_comparison(self, fitted_model_with_data):
        result, train_data = fitted_model_with_data

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

        tt = diagnostics.train_test
        assert tt.train is not None
        assert tt.test is not None
        assert tt.train.dataset == "train"
        assert tt.test.dataset == "test"
        assert hasattr(tt, "gini_gap")
        assert hasattr(tt, "overfitting_risk")
        assert len(tt.decile_comparison) == 10

    def test_score_test_in_factor_diagnostics(self):
        np.random.seed(42)
        n = 500
        age = np.random.uniform(20, 60, n)
        region = np.random.choice(["A", "B", "C"], n)
        unfitted_var = np.random.randn(n)
        unfitted_cat = np.random.choice(["X", "Y", "Z"], n)
        mu_true = np.exp(-1 + 0.02 * age + 0.3 * (region == "A").astype(float))
        y = np.random.poisson(mu_true)

        data = pl.DataFrame(
            {
                "y": y,
                "age": age,
                "region": region,
                "unfitted_var": unfitted_var,
                "unfitted_cat": unfitted_cat,
            }
        )

        result = rs.glm_dict(
            response="y",
            terms={"age": {"type": "linear"}, "region": {"type": "categorical"}},
            data=data,
            family="poisson",
        ).fit()

        diagnostics = result.diagnostics(
            train_data=data,
            categorical_factors=["region", "unfitted_cat"],
            continuous_factors=["age", "unfitted_var"],
        )

        for factor in diagnostics.factors:
            if factor.in_model:
                assert factor.score_test is None


# =============================================================================
# Mirrors: test_splines.py → TestSplineFormula, TestMonotonicSplineFormula
# =============================================================================


class TestDictSplineFormula:
    """Test splines via dict API (mirrors TestSplineFormula)."""

    def test_dict_with_bs(self):
        np.random.seed(42)
        n = 100
        data = pl.DataFrame(
            {
                "y": np.random.poisson(3, n),
                "age": np.random.uniform(20, 70, n),
            }
        )

        result = rs.glm_dict(
            response="y",
            terms={"age": {"type": "bs", "df": 5}},
            data=data,
            family="poisson",
        ).fit()

        assert len(result.params) >= 2

    def test_dict_with_ns(self):
        np.random.seed(42)
        n = 100
        data = pl.DataFrame(
            {
                "y": np.random.normal(0, 1, n),
                "x": np.random.uniform(0, 10, n),
            }
        )

        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "ns", "df": 4}},
            data=data,
            family="gaussian",
        ).fit()

        assert result.converged
        assert len(result.params) >= 2

    def test_dict_spline_with_categorical(self):
        np.random.seed(42)
        n = 200
        data = pl.DataFrame(
            {
                "y": np.random.poisson(2, n),
                "age": np.random.uniform(20, 70, n),
                "region": np.random.choice(["A", "B", "C"], n),
            }
        )

        result = rs.glm_dict(
            response="y",
            terms={"age": {"type": "bs", "df": 4}, "region": {"type": "categorical"}},
            data=data,
            family="poisson",
        ).fit()

        assert result.converged
        assert len(result.params) >= 4

    def test_dict_multiple_splines(self):
        np.random.seed(42)
        n = 150
        data = pl.DataFrame(
            {
                "y": np.random.poisson(3, n),
                "age": np.random.uniform(20, 70, n),
                "income": np.random.uniform(30000, 150000, n),
            }
        )

        result = rs.glm_dict(
            response="y",
            terms={"age": {"type": "bs", "df": 4}, "income": {"type": "ns", "df": 3}},
            data=data,
            family="poisson",
        ).fit()

        assert result.converged


class TestDictMonotonicSplineFormula:
    """Test monotonic splines via dict API (mirrors TestMonotonicSplineFormula)."""

    def test_dict_monotonic_bs_basic(self):
        np.random.seed(42)
        n = 200
        age = np.random.uniform(20, 70, n)
        rate = np.exp(0.5 + 0.02 * age)
        y = np.random.poisson(rate)
        data = pl.DataFrame({"y": y, "age": age})

        result = rs.glm_dict(
            response="y",
            terms={"age": {"type": "bs", "df": 5, "monotonicity": "increasing"}},
            data=data,
            family="poisson",
        ).fit(max_iter=100)

        assert len(result.params) >= 2
        assert result.converged

    def test_dict_monotonic_bs_decreasing(self):
        np.random.seed(42)
        n = 200
        vehicle_age = np.random.uniform(0, 20, n)
        rate = np.exp(1.5 - 0.05 * vehicle_age)
        y = np.random.poisson(rate)
        data = pl.DataFrame({"y": y, "vehicle_age": vehicle_age})

        result = rs.glm_dict(
            response="y",
            terms={"vehicle_age": {"type": "bs", "df": 3, "monotonicity": "decreasing"}},
            data=data,
            family="poisson",
        ).fit(max_iter=100, alpha=1e-4)

        assert len(result.params) >= 2
        assert result.converged

    def test_dict_monotonic_bs_decreasing_direction(self):
        """Verify that a decreasing monotonic spline actually produces
        decreasing predictions as x increases (regression test for
        double-negation bug in I-spline basis)."""
        np.random.seed(42)
        n = 500
        x = np.random.uniform(0, 20, n)
        rate = np.exp(2.0 - 0.1 * x)
        y = np.random.poisson(rate)
        data = pl.DataFrame({"y": y, "x": x})

        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "bs", "df": 5, "monotonicity": "decreasing"}},
            data=data,
            family="poisson",
        ).fit(max_iter=100)

        assert result.converged
        # Predict on a grid from low to high
        grid = pl.DataFrame({"x": np.linspace(1, 19, 20).tolist()})
        preds = result.predict(grid)
        # Predictions should be non-increasing (decreasing or flat)
        diffs = np.diff(preds)
        assert np.all(diffs <= 1e-10), (
            f"Decreasing monotonic spline produced increasing predictions: "
            f"max positive diff = {diffs.max():.6f}"
        )

    def test_dict_monotonic_bs_with_other_terms(self):
        np.random.seed(42)
        n = 200
        age = np.random.uniform(20, 70, n)
        income = np.random.uniform(30000, 150000, n)
        region = np.random.choice(["A", "B", "C"], n)
        rate = np.exp(0.3 + 0.01 * age + 0.000005 * income)
        y = np.random.poisson(rate)
        data = pl.DataFrame(
            {
                "y": y,
                "age": age,
                "income": income,
                "region": region,
            }
        )

        result = rs.glm_dict(
            response="y",
            terms={
                "age": {"type": "bs", "df": 4, "monotonicity": "increasing"},
                "income": {"type": "bs", "df": 3},
                "region": {"type": "categorical"},
            },
            data=data,
            family="poisson",
        ).fit(max_iter=100)

        assert len(result.params) >= 5
        assert result.converged


class TestMonotonicSplineBoundary:
    """Test monotonic spline boundary/tail behavior — regression tests for
    bugs that previously hid in extrapolation and tail regions."""

    def test_monotonic_increasing_at_boundaries(self):
        """Fit bs(k=10, monotonicity='increasing') on data with a real
        increasing relationship. Verify predictions are monotonically
        non-decreasing within the training range."""
        np.random.seed(42)
        n = 500
        x = np.random.uniform(10, 60, n)
        rate = np.exp(0.5 + 0.03 * x)
        y = np.random.poisson(rate)
        data = pl.DataFrame({"y": y, "x": x})

        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "bs", "k": 10, "monotonicity": "increasing"}},
            data=data,
            family="poisson",
        ).fit(max_iter=100)

        # Predict within the training range (B-splines don't guarantee
        # monotonicity during extrapolation beyond boundary knots)
        grid_values = np.linspace(15, 55, 50)
        grid = pl.DataFrame({"x": grid_values.tolist()})
        preds = result.predict(grid)

        # Predictions must be non-decreasing within the training range
        diffs = np.diff(preds)
        assert np.all(diffs >= -1e-4), (
            f"Increasing monotonic spline produced decreasing predictions: "
            f"min diff = {diffs.min():.6f}"
        )

    def test_monotonic_decreasing_at_boundaries(self):
        """Fit bs(k=10, monotonicity='decreasing') and verify predictions
        are monotonically non-increasing within the training range."""
        np.random.seed(42)
        n = 500
        x = np.random.uniform(0, 20, n)
        rate = np.exp(2.0 - 0.1 * x)
        y = np.random.poisson(rate)
        data = pl.DataFrame({"y": y, "x": x})

        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "bs", "k": 10, "monotonicity": "decreasing"}},
            data=data,
            family="poisson",
        ).fit(max_iter=100)

        # Predict within the training range
        grid_values = np.linspace(0.5, 19.5, 50)
        grid = pl.DataFrame({"x": grid_values.tolist()})
        preds = result.predict(grid)

        # Predictions must be non-increasing within the training range
        diffs = np.diff(preds)
        assert np.all(diffs <= 1e-4), (
            f"Decreasing monotonic spline produced increasing predictions: "
            f"max positive diff = {diffs.max():.6f}"
        )

    def test_monotonic_smooth_at_boundaries(self):
        """Fit penalized smooth bs(k=10, monotonicity='decreasing') and
        verify predictions are monotone within the training range."""
        np.random.seed(42)
        n = 500
        x = np.random.uniform(0, 20, n)
        rate = np.exp(2.5 - 0.08 * x)
        y = np.random.poisson(rate)
        data = pl.DataFrame({"y": y, "x": x})

        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "bs", "k": 10, "monotonicity": "decreasing"}},
            data=data,
            family="poisson",
        ).fit(max_iter=100)

        # Predict within the training range
        grid_values = np.linspace(0.5, 19.5, 40)
        grid = pl.DataFrame({"x": grid_values.tolist()})
        preds = result.predict(grid)

        # Predictions must be non-increasing
        diffs = np.diff(preds)
        assert np.all(diffs <= 1e-4), (
            f"Penalized decreasing monotonic spline produced increasing "
            f"predictions: max positive diff = {diffs.max():.6f}"
        )

    def test_monotonic_coefficients_sign(self):
        """After fitting monotonic splines, verify that the spline
        coefficients respect sign constraints: all non-negative for
        increasing, all non-positive for decreasing."""
        np.random.seed(42)
        n = 500
        x = np.random.uniform(10, 60, n)

        # --- Increasing: verify B-spline coefficients are non-decreasing ---
        rate_inc = np.exp(0.5 + 0.03 * x)
        y_inc = np.random.poisson(rate_inc)
        data_inc = pl.DataFrame({"y": y_inc, "x": x})

        result_inc = rs.glm_dict(
            response="y",
            terms={"x": {"type": "bs", "k": 10, "monotonicity": "increasing"}},
            data=data_inc,
            family="poisson",
        ).fit(max_iter=100)

        # With exp reparam, B-spline coefficients should be non-decreasing
        spline_mask_inc = [
            i for i, name in enumerate(result_inc.feature_names) if "bs(" in name and ", +)" in name
        ]
        assert len(spline_mask_inc) > 0, "No increasing-constrained features found"
        spline_coefs_inc = np.array(result_inc.params)[spline_mask_inc]
        coef_diffs = np.diff(spline_coefs_inc)
        assert np.all(
            coef_diffs >= -1e-4
        ), f"Increasing monotonic spline has non-monotone coefficients: diffs = {coef_diffs}"

        # --- Decreasing: verify B-spline coefficients are non-increasing ---
        np.random.seed(42)
        rate_dec = np.exp(2.0 - 0.1 * x)
        y_dec = np.random.poisson(rate_dec)
        data_dec = pl.DataFrame({"y": y_dec, "x": x})

        result_dec = rs.glm_dict(
            response="y",
            terms={"x": {"type": "bs", "k": 10, "monotonicity": "decreasing"}},
            data=data_dec,
            family="poisson",
        ).fit(max_iter=100)

        # With exp reparam, B-spline coefficients should be non-increasing
        spline_mask_dec = [
            i for i, name in enumerate(result_dec.feature_names) if "bs(" in name and ", -)" in name
        ]
        assert len(spline_mask_dec) > 0, "No decreasing-constrained features found"
        spline_coefs_dec = np.array(result_dec.params)[spline_mask_dec]
        coef_diffs_dec = np.diff(spline_coefs_dec)
        assert np.all(
            coef_diffs_dec <= 1e-4
        ), f"Decreasing monotonic spline has non-monotone coefficients: diffs = {coef_diffs_dec}"

    def test_ms_type_via_glm_dict(self):
        """Verify that type='ms' works through glm_dict() and produces a
        monotonically increasing curve (the default direction for ms)."""
        np.random.seed(42)
        n = 500
        x = np.random.uniform(0, 30, n)
        rate = np.exp(0.2 + 0.05 * x)
        y = np.random.poisson(rate)
        data = pl.DataFrame({"y": y, "x": x})

        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "ms", "df": 5}},
            data=data,
            family="poisson",
        ).fit(max_iter=100)

        assert result.converged

        # Predict on a grid spanning and slightly beyond the training range
        grid_values = np.linspace(-2, 35, 50)
        grid = pl.DataFrame({"x": grid_values.tolist()})
        preds = result.predict(grid)

        # ms() defaults to increasing — predictions must be non-decreasing
        diffs = np.diff(preds)
        assert np.all(
            diffs >= -1e-10
        ), f"ms() monotonic spline produced decreasing predictions: min diff = {diffs.min():.6f}"


# =============================================================================
# Mirrors: test_train_predict_consistency.py → formula-based tests
# =============================================================================


class TestDictTrainPredictConsistency:
    """Verify dict API transformations are consistent (mirrors formula tests)."""

    def test_ns_knots_reused_on_new_data(self):
        np.random.seed(42)
        train_x = np.random.uniform(0, 100, 1000)
        train_y = 5 + 0.1 * train_x + np.random.normal(0, 1, 1000)
        train_data = pl.DataFrame({"x": train_x, "y": train_y})

        test_x = np.random.uniform(20, 80, 500)
        test_data = pl.DataFrame({"x": test_x, "y": np.zeros(500)})

        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "ns", "df": 5}},
            data=train_data,
            family="gaussian",
        ).fit()

        train_pred = result.fittedvalues
        test_pred = result.predict(test_data)

        assert test_pred.min() > 0
        assert test_pred.max() < 20
        assert abs(test_pred.mean() - train_pred.mean()) < 5
        assert np.all(np.isfinite(test_pred))

    def test_bs_knots_reused_on_new_data(self):
        np.random.seed(42)
        train_x = np.random.uniform(0, 100, 1000)
        train_y = 5 + 0.1 * train_x + np.random.normal(0, 1, 1000)
        train_data = pl.DataFrame({"x": train_x, "y": train_y})

        test_x = np.random.uniform(20, 80, 500)
        test_data = pl.DataFrame({"x": test_x, "y": np.zeros(500)})

        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "bs", "df": 5}},
            data=train_data,
            family="gaussian",
        ).fit()

        train_pred = result.fittedvalues
        test_pred = result.predict(test_data)

        assert test_pred.min() > 0
        assert test_pred.max() < 20
        assert abs(test_pred.mean() - train_pred.mean()) < 5
        assert np.all(np.isfinite(test_pred))

    def test_monotonic_spline_knots_reused(self):
        np.random.seed(42)
        train_x = np.random.uniform(0, 100, 1000)
        train_y = 5 + 0.1 * train_x + np.random.normal(0, 0.5, 1000)
        train_data = pl.DataFrame({"x": train_x, "y": train_y})

        test_x = np.random.uniform(20, 80, 500)
        test_data = pl.DataFrame({"x": test_x, "y": np.zeros(500)})

        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "bs", "df": 5, "monotonicity": "increasing"}},
            data=train_data,
            family="gaussian",
        ).fit()

        train_pred = result.fittedvalues
        test_pred = result.predict(test_data)

        assert test_pred.min() > 0
        assert test_pred.max() < 20
        assert abs(test_pred.mean() - train_pred.mean()) < 5
        assert np.all(np.isfinite(test_pred))

    def test_categorical_levels_preserved(self):
        np.random.seed(42)
        train_cat = np.random.choice(["A", "B", "C", "D"], 1000)
        train_y = np.random.normal(10, 1, 1000)
        train_data = pl.DataFrame({"cat": train_cat, "y": train_y})

        test_cat = np.random.choice(["A", "B"], 500)
        test_y = np.random.normal(10, 1, 500)
        test_data = pl.DataFrame({"cat": test_cat, "y": test_y})

        result = rs.glm_dict(
            response="y",
            terms={"cat": {"type": "categorical"}},
            data=train_data,
            family="gaussian",
        ).fit()

        train_pred = result.fittedvalues
        test_pred = result.predict(test_data)

        assert np.all(np.isfinite(test_pred))
        assert abs(test_pred.mean() - train_pred.mean()) < 1

    def test_unseen_categorical_level_handled(self):
        np.random.seed(42)
        train_cat = np.random.choice(["A", "B", "C"], 1000)
        train_y = np.random.normal(10, 1, 1000)
        train_data = pl.DataFrame({"cat": train_cat, "y": train_y})

        test_cat = np.array(["A", "B", "X", "Y", "Z"])
        test_y = np.random.normal(10, 1, 5)
        test_data = pl.DataFrame({"cat": test_cat, "y": test_y})

        result = rs.glm_dict(
            response="y",
            terms={"cat": {"type": "categorical"}},
            data=train_data,
            family="gaussian",
        ).fit()
        test_pred = result.predict(test_data)

        assert np.all(np.isfinite(test_pred))


# =============================================================================
# LazyFrame Tests
# =============================================================================

from rustystats.formula import _collect_lazyframe, _extract_needed_columns


class TestExtractNeededColumns:
    """Unit tests for _extract_needed_columns column detection."""

    def test_terms_only(self):
        cols = _extract_needed_columns(terms={})
        assert cols == set()

    def test_response_included_when_provided(self):
        cols = _extract_needed_columns(terms={}, response="y")
        assert cols == {"y"}

    def test_response_omitted_for_prediction(self):
        cols = _extract_needed_columns(
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
        )
        assert cols == {"x1", "x2"}

    def test_linear_terms(self):
        cols = _extract_needed_columns(
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            response="y",
        )
        assert cols == {"y", "x1", "x2"}

    def test_categorical_terms(self):
        cols = _extract_needed_columns(
            terms={"region": {"type": "categorical"}},
            response="y",
        )
        assert cols == {"y", "region"}

    def test_spline_terms(self):
        cols = _extract_needed_columns(
            terms={"age": {"type": "bs", "df": 5}, "income": {"type": "ns"}},
            response="y",
        )
        assert cols == {"y", "age", "income"}

    def test_expression_extracts_referenced_columns(self):
        cols = _extract_needed_columns(
            terms={"x1_sq": {"type": "expression", "expr": "x1 ** 2"}},
            response="y",
        )
        assert cols == {"y", "x1"}

    def test_expression_binary_op_extracts_both_columns(self):
        cols = _extract_needed_columns(
            terms={"ratio": {"type": "expression", "expr": "weight / height"}},
            response="y",
        )
        assert cols == {"y", "weight", "height"}

    def test_expression_missing_expr_key_raises(self):
        with pytest.raises(KeyError):
            _extract_needed_columns(
                terms={"bad": {"type": "expression"}},
            )

    def test_expression_does_not_include_numeric_literals(self):
        cols = _extract_needed_columns(
            terms={"scaled": {"type": "expression", "expr": "income / 1000"}},
            response="y",
        )
        assert "1000" not in cols
        assert cols == {"y", "income"}

    def test_interaction_columns(self):
        cols = _extract_needed_columns(
            terms={},
            response="y",
            interactions=[
                {"x1": {"type": "linear"}, "cat": {"type": "categorical"}, "include_main": True},
            ],
        )
        assert cols == {"y", "x1", "cat"}

    def test_interaction_flags_excluded(self):
        """include_main, target_encoding, frequency_encoding, prior_weight are not column names."""
        cols = _extract_needed_columns(
            terms={},
            response="y",
            interactions=[
                {
                    "a": {"type": "categorical"},
                    "b": {"type": "categorical"},
                    "target_encoding": True,
                    "prior_weight": 1.0,
                    "include_main": False,
                    "frequency_encoding": False,
                },
            ],
        )
        assert cols == {"y", "a", "b"}

    def test_offset_string(self):
        cols = _extract_needed_columns(
            terms={"x": {"type": "linear"}},
            response="y",
            offset="exposure",
        )
        assert "exposure" in cols

    def test_offset_array_not_included(self):
        cols = _extract_needed_columns(
            terms={"x": {"type": "linear"}},
            response="y",
            offset=np.array([1.0]),
        )
        assert cols == {"y", "x"}

    def test_weights_string(self):
        cols = _extract_needed_columns(
            terms={"x": {"type": "linear"}},
            response="y",
            weights="w",
        )
        assert "w" in cols

    def test_complement_string(self):
        cols = _extract_needed_columns(
            terms={"x": {"type": "linear"}},
            response="y",
            complement="prior_rate",
        )
        assert "prior_rate" in cols

    def test_all_sources_combined(self):
        """Every source of columns works together without duplication."""
        cols = _extract_needed_columns(
            terms={
                "x1": {"type": "linear"},
                "cat": {"type": "categorical"},
                "x1_sq": {"type": "expression", "expr": "x1 ** 2"},
                "age": {"type": "bs", "df": 4},
            },
            response="y",
            interactions=[
                {"x1": {"type": "linear"}, "region": {"type": "categorical"}, "include_main": True},
            ],
            offset="exposure",
            weights="w",
            complement="prior",
        )
        assert cols == {"y", "x1", "cat", "age", "region", "exposure", "w", "prior"}


class TestCollectLazyFrame:
    """Unit tests for _collect_lazyframe."""

    def test_dataframe_passes_through(self):
        df = pl.DataFrame({"a": [1], "b": [2]})
        result = _collect_lazyframe(df, {"a"})
        assert result is df  # exact same object, no copy

    def test_lazyframe_selects_only_needed_columns(self):
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        result = _collect_lazyframe(df.lazy(), {"a", "c"})
        assert sorted(result.columns) == ["a", "c"]
        assert result.shape == (2, 2)

    def test_lazyframe_missing_column_raises(self):
        df = pl.DataFrame({"a": [1]})
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            _collect_lazyframe(df.lazy(), {"a", "nonexistent"})

    def test_lazyframe_empty_needed_collects_all(self):
        df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
        result = _collect_lazyframe(df.lazy(), set())
        assert sorted(result.columns) == ["a", "b", "c"]


class TestLazyFrameIntegration:
    """Integration tests: LazyFrame through glm_dict produces identical results to DataFrame."""

    @pytest.fixture
    def wide_data(self):
        np.random.seed(42)
        n = 200
        return pl.DataFrame(
            {
                "y": np.random.poisson(1, n),
                "x1": np.random.uniform(0, 10, n),
                "x2": np.random.uniform(0, 10, n),
                "cat": np.random.choice(["A", "B", "C"], n),
                "exposure": np.random.uniform(0.5, 1.5, n),
                "weight": np.random.uniform(0.5, 2.0, n),
                "unused1": np.random.normal(0, 1, n),
                "unused2": np.random.normal(0, 1, n),
                "unused3": np.random.choice(["X", "Y"], n),
            }
        )

    def _fit_both(self, wide_data, **kwargs):
        """Fit with DataFrame and LazyFrame, return both results."""
        result_df = rs.glm_dict(data=wide_data, **kwargs).fit()
        result_lf = rs.glm_dict(data=wide_data.lazy(), **kwargs).fit()
        return result_df, result_lf

    def test_linear_terms(self, wide_data):
        r_df, r_lf = self._fit_both(
            wide_data,
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            family="poisson",
            offset="exposure",
        )
        np.testing.assert_allclose(r_df.params, r_lf.params)

    def test_categorical(self, wide_data):
        r_df, r_lf = self._fit_both(
            wide_data,
            response="y",
            terms={"x1": {"type": "linear"}, "cat": {"type": "categorical"}},
            family="poisson",
        )
        np.testing.assert_allclose(r_df.params, r_lf.params)

    def test_spline(self, wide_data):
        r_df, r_lf = self._fit_both(
            wide_data,
            response="y",
            terms={"x1": {"type": "bs", "df": 4}},
            family="poisson",
        )
        np.testing.assert_allclose(r_df.params, r_lf.params)

    def test_expression(self, wide_data):
        r_df, r_lf = self._fit_both(
            wide_data,
            response="y",
            terms={
                "x1": {"type": "linear"},
                "x1_sq": {"type": "expression", "expr": "x1 ** 2"},
            },
            family="poisson",
        )
        np.testing.assert_allclose(r_df.params, r_lf.params)

    def test_interaction(self, wide_data):
        r_df, r_lf = self._fit_both(
            wide_data,
            response="y",
            terms={"x1": {"type": "linear"}, "cat": {"type": "categorical"}},
            interactions=[
                {"x1": {"type": "linear"}, "cat": {"type": "categorical"}, "include_main": False},
            ],
            family="poisson",
        )
        np.testing.assert_allclose(r_df.params, r_lf.params)

    def test_weights(self, wide_data):
        r_df, r_lf = self._fit_both(
            wide_data,
            response="y",
            terms={"x1": {"type": "linear"}},
            family="poisson",
            weights="weight",
        )
        np.testing.assert_allclose(r_df.params, r_lf.params)

    def test_predict_identical(self, wide_data):
        """predict() with LazyFrame gives identical results to DataFrame."""
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "cat": {"type": "categorical"}},
            data=wide_data,
            family="poisson",
            offset="exposure",
        ).fit()

        pred_df = result.predict(wide_data)
        pred_lf = result.predict(wide_data.lazy())
        np.testing.assert_allclose(pred_df, pred_lf)

    def test_predict_with_explicit_offset(self, wide_data):
        """predict() with LazyFrame and string offset resolves correctly."""
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=wide_data,
            family="poisson",
            offset="exposure",
        ).fit()

        pred_df = result.predict(wide_data, offset="exposure")
        pred_lf = result.predict(wide_data.lazy(), offset="exposure")
        np.testing.assert_allclose(pred_df, pred_lf)

    def test_missing_column_in_lazyframe_raises(self, wide_data):
        """LazyFrame missing a needed column raises immediately, not deep in the pipeline."""
        lf_missing = wide_data.select("y", "x1").lazy()
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            rs.glm_dict(
                response="y",
                terms={"x1": {"type": "linear"}, "cat": {"type": "categorical"}},
                data=lf_missing,
                family="poisson",
            )

    def test_unused_columns_not_collected(self, wide_data):
        """Verify the LazyFrame only materialises needed columns via scan_parquet simulation."""
        # We can't intercept Polars' internal read, but we can verify by giving
        # a LazyFrame that only has the needed columns — it should work fine,
        # proving unused columns weren't required.
        needed_only = wide_data.select("y", "x1", "exposure").lazy()
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=needed_only,
            family="poisson",
            offset="exposure",
        ).fit()
        assert result.converged

    def test_expression_column_not_in_terms_still_collected(self, wide_data):
        """Expression referencing a column not in terms dict still works."""
        # x2 is not a term key, but referenced in the expression
        result = rs.glm_dict(
            response="y",
            terms={"ratio": {"type": "expression", "expr": "x1 / x2"}},
            data=wide_data.lazy(),
            family="poisson",
        ).fit()
        assert result.converged

    def test_deserialized_model_predict_with_lazyframe(self, wide_data):
        """Deserialized model (from_bytes) still prunes columns on predict with LazyFrame."""
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "cat": {"type": "categorical"}},
            data=wide_data,
            family="poisson",
            offset="exposure",
        ).fit()

        loaded = rs.GLMModel.from_bytes(result.to_bytes())

        # Predict with full DataFrame
        pred_df = loaded.predict(wide_data)

        # Predict with LazyFrame containing only needed columns
        needed_only = wide_data.select("x1", "cat", "exposure").lazy()
        pred_lf = loaded.predict(needed_only)

        np.testing.assert_allclose(pred_df, pred_lf)

    def test_deserialized_model_predict_lazyframe_missing_col_raises(self, wide_data):
        """Deserialized model raises on LazyFrame missing a needed column."""
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "cat": {"type": "categorical"}},
            data=wide_data,
            family="poisson",
        ).fit()

        loaded = rs.GLMModel.from_bytes(result.to_bytes())

        lf_missing = wide_data.select("x1").lazy()  # missing "cat"
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            loaded.predict(lf_missing)


class TestRequiredColumns:
    """GLMModel.required_columns exposes raw input columns for predict-time pipelines."""

    @pytest.fixture
    def data(self):
        np.random.seed(42)
        n = 200
        return pl.DataFrame(
            {
                "y": np.random.poisson(2, n).astype(float),
                "x1": np.random.randn(n),
                "x2": np.random.randn(n),
                "cat": np.random.choice(["A", "B", "C"], n),
                "region": np.random.choice(["N", "S"], n),
                "exposure": np.random.uniform(0.5, 2.0, n),
                "prior_rate": np.random.uniform(0.1, 0.5, n),
            }
        )

    def test_basic_terms_excludes_response(self, data):
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "cat": {"type": "categorical"}},
            data=data,
            family="poisson",
        ).fit()
        assert result.required_columns == ["cat", "x1"]

    def test_returns_sorted_list(self, data):
        result = rs.glm_dict(
            response="y",
            terms={"x2": {"type": "linear"}, "x1": {"type": "linear"}},
            data=data,
            family="poisson",
        ).fit()
        cols = result.required_columns
        assert isinstance(cols, list)
        assert cols == sorted(cols)

    def test_expression_pulls_source_columns(self, data):
        result = rs.glm_dict(
            response="y",
            terms={"sq": {"type": "expression", "expr": "x1 ** 2"}},
            data=data,
            family="poisson",
        ).fit()
        assert result.required_columns == ["x1"]

    def test_includes_offset_column(self, data):
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=data,
            family="poisson",
            offset="exposure",
        ).fit()
        assert "exposure" in result.required_columns

    def test_includes_complement_column(self, data):
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=data,
            family="poisson",
            offset="exposure",
            complement="prior_rate",
        ).fit(regularization="lasso")
        cols = result.required_columns
        assert "prior_rate" in cols
        assert "exposure" in cols

    def test_interaction_columns_included(self, data):
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=data,
            family="poisson",
            interactions=[
                {"x2": {"type": "linear"}, "region": {"type": "categorical"}, "include_main": True},
            ],
        ).fit()
        cols = result.required_columns
        assert "x2" in cols
        assert "region" in cols

    def test_survives_serialization(self, data):
        result = rs.glm_dict(
            response="y",
            terms={
                "x1": {"type": "linear"},
                "cat": {"type": "categorical"},
                "sq": {"type": "expression", "expr": "x2 ** 2"},
            },
            data=data,
            family="poisson",
            offset="exposure",
        ).fit()
        loaded = rs.GLMModel.from_bytes(result.to_bytes())
        assert loaded.required_columns == result.required_columns

    def test_raises_when_terms_dict_missing(self, data):
        """Models without a stored dict spec (e.g. older serialized formats) fail loudly."""
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=data,
            family="poisson",
        ).fit()
        result._terms_dict = None
        with pytest.raises(RuntimeError, match="glm_dict"):
            _ = result.required_columns

    def test_predict_with_lazyframe_select(self, data):
        """The advertised pipeline-optimization use case actually works."""
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "cat": {"type": "categorical"}},
            data=data,
            family="poisson",
            offset="exposure",
        ).fit()
        full_pred = result.predict(data)
        projected = data.lazy().select(result.required_columns).collect()
        np.testing.assert_array_almost_equal(result.predict(projected), full_pred)


# =============================================================================
# Constraint Enforcement Tests
# =============================================================================


class TestConstraintEnforcement:
    """Test that sign constraints are correctly enforced across different code paths."""

    def test_coord_descent_respects_sign_constraints(self):
        """Lasso (coordinate descent) must respect monotonicity sign constraints on spline coefficients."""
        np.random.seed(42)
        n = 500
        x = np.random.uniform(0, 10, n)
        mu = np.exp(0.5 + 0.1 * x)
        y = np.random.poisson(mu)
        data = pl.DataFrame({"y": y, "x": x})

        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "bs", "df": 4, "monotonicity": "increasing"}},
            data=data,
            family="poisson",
        ).fit(regularization="lasso", verbose=False)

        assert result.converged
        # Spline coefficients have feature names like "bs(x, +)[1]"
        spline_indices = [
            i
            for i, name in enumerate(result.feature_names)
            if name.startswith("bs(") and ", +)" in name
        ]
        assert len(spline_indices) > 0, "Expected monotonic spline feature names"
        spline_coefs = result.params[spline_indices]
        assert np.all(
            spline_coefs >= -1e-10
        ), f"Coordinate descent violated non-negative constraint: {spline_coefs}"

    def test_smooth_plus_constrained_linear(self):
        """A model with both a smooth term and a constrained linear term must enforce the linear constraint."""
        np.random.seed(42)
        n = 500
        x1 = np.random.uniform(0, 10, n)
        x2 = np.random.uniform(0, 5, n)
        mu = np.exp(0.2 + 0.05 * x1 + 0.1 * x2)
        y = np.random.poisson(mu)
        data = pl.DataFrame({"y": y, "x1": x1, "x2": x2})

        result = rs.glm_dict(
            response="y",
            terms={
                "x1": {"type": "bs", "k": 10},
                "x2": {"type": "linear", "monotonicity": "increasing"},
            },
            data=data,
            family="poisson",
        ).fit()

        assert result.converged
        # The constrained linear term has feature name "pos(x2)"
        pos_indices = [i for i, name in enumerate(result.feature_names) if name == "pos(x2)"]
        assert len(pos_indices) == 1, f"Expected pos(x2) in feature names: {result.feature_names}"
        assert (
            result.params[pos_indices[0]] >= -1e-10
        ), f"Linear constraint violated: pos(x2) coefficient = {result.params[pos_indices[0]]}"

    def test_ns_monotonicity_rejected(self):
        """Natural splines (ns) must reject monotonicity constraints."""
        np.random.seed(42)
        n = 100
        data = pl.DataFrame(
            {
                "y": np.random.poisson(2, n),
                "x": np.random.uniform(0, 10, n),
            }
        )

        with pytest.raises((ValueError, rs.ValidationError)):
            rs.glm_dict(
                response="y",
                terms={"x": {"type": "ns", "df": 4, "monotonicity": "increasing"}},
                data=data,
                family="poisson",
            )

    def test_cv_path_with_constraints(self):
        """CV-based ridge path must converge and respect monotonicity constraints."""
        np.random.seed(42)
        n = 500
        x = np.random.uniform(0, 10, n)
        mu = np.exp(0.5 + 0.08 * x)
        y = np.random.poisson(mu)
        data = pl.DataFrame({"y": y, "x": x})

        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "bs", "df": 4, "monotonicity": "increasing"}},
            data=data,
            family="poisson",
        ).fit(cv=3, regularization="ridge", n_alphas=5, verbose=False)

        assert result.converged
        # Spline coefficients must respect non-negative constraint
        spline_indices = [
            i
            for i, name in enumerate(result.feature_names)
            if name.startswith("bs(") and ", +)" in name
        ]
        assert len(spline_indices) > 0, "Expected monotonic spline feature names"
        spline_coefs = result.params[spline_indices]
        assert np.all(
            spline_coefs >= -1e-10
        ), f"CV path violated non-negative constraint: {spline_coefs}"


# =============================================================================
# Explicit Knots Tests
# =============================================================================


class TestExplicitKnots:
    """Test explicit knot placement for bs() and ns() splines."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 200
        x = np.random.uniform(0, 10, n)
        return pl.DataFrame(
            {
                "y": np.random.poisson(np.exp(0.5 + 0.1 * x), n),
                "x1": x,
                "x2": np.random.uniform(0, 10, n),
            }
        )

    def test_bs_explicit_knots(self, sample_data):
        """bs with explicit knots converges and has correct param count."""
        knots = [2.0, 5.0, 8.0]
        # degree=3 (default), so df = len(knots) + degree + 1 = 7
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "bs", "knots": knots}},
            data=sample_data,
            family="poisson",
        ).fit()
        assert result.converged
        # 6 spline columns + intercept = 7 params (len(knots) + degree = 3 + 3 = 6)
        assert len(result.params) == 7

    def test_bs_explicit_knots_monotonicity(self, sample_data):
        """bs with explicit knots and monotonicity constraint converges."""
        knots = [3.0, 6.0]
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "bs", "knots": knots, "monotonicity": "increasing"}},
            data=sample_data,
            family="poisson",
        ).fit()
        assert result.converged

    def test_ns_explicit_knots(self, sample_data):
        """ns with explicit knots converges and has correct param count."""
        knots = [2.0, 5.0, 8.0]
        # ns: df = len(knots) + 1 = 4
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "ns", "knots": knots}},
            data=sample_data,
            family="poisson",
        ).fit()
        assert result.converged
        # 3 spline columns + intercept = 4 params (len(knots) = 3)
        assert len(result.params) == 4

    def test_ns_explicit_knots_with_boundary_knots(self, sample_data):
        """ns with explicit knots and boundary_knots converges."""
        knots = [3.0, 7.0]
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "ns", "knots": knots, "boundary_knots": (0.0, 10.0)}},
            data=sample_data,
            family="poisson",
        ).fit()
        assert result.converged
        assert len(result.params) == 3  # 2 spline cols + intercept

    def test_explicit_knots_rejects_df(self, sample_data):
        """Cannot specify both knots and df."""
        with pytest.raises(rs.ValidationError, match="Cannot specify both"):
            rs.glm_dict(
                response="y",
                terms={"x1": {"type": "bs", "knots": [3.0, 6.0], "df": 5}},
                data=sample_data,
            )

    def test_explicit_knots_rejects_k(self, sample_data):
        """Cannot specify both knots and k."""
        with pytest.raises(rs.ValidationError, match="Cannot specify both"):
            rs.glm_dict(
                response="y",
                terms={"x1": {"type": "ns", "knots": [3.0, 6.0], "k": 10}},
                data=sample_data,
            )

    def test_explicit_knots_empty(self, sample_data):
        """Empty knots list raises ValidationError."""
        with pytest.raises(rs.ValidationError, match="non-empty"):
            rs.glm_dict(
                response="y",
                terms={"x1": {"type": "bs", "knots": []}},
                data=sample_data,
            )

    def test_explicit_knots_unsorted(self, sample_data):
        """Unsorted knots raises ValidationError."""
        with pytest.raises(rs.ValidationError, match="sorted"):
            rs.glm_dict(
                response="y",
                terms={"x1": {"type": "bs", "knots": [5.0, 2.0, 8.0]}},
                data=sample_data,
            )

    def test_explicit_knots_duplicated(self, sample_data):
        """Duplicate knots raises ValidationError."""
        with pytest.raises(rs.ValidationError, match="unique"):
            rs.glm_dict(
                response="y",
                terms={"x1": {"type": "ns", "knots": [3.0, 3.0, 6.0]}},
                data=sample_data,
            )

    def test_explicit_knots_prediction_consistency(self, sample_data):
        """Predictions on new data reuse the same knots."""
        knots = [2.0, 5.0, 8.0]
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "bs", "knots": knots}},
            data=sample_data,
            family="poisson",
        ).fit()

        # Predict on new data within the same range
        new_data = pl.DataFrame({"x1": np.linspace(0, 10, 50)})
        preds = result.predict(new_data)
        assert len(preds) == 50
        assert np.all(np.isfinite(preds))

    def test_explicit_knots_serialization(self, sample_data):
        """to_bytes/from_bytes roundtrip preserves explicit knots model."""
        knots = [2.0, 5.0, 8.0]
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "bs", "knots": knots}},
            data=sample_data,
            family="poisson",
        ).fit()

        # Roundtrip
        loaded = rs.GLMModel.from_bytes(result.to_bytes())

        # Predictions should match
        new_data = pl.DataFrame({"x1": np.linspace(0, 10, 50)})
        np.testing.assert_allclose(
            result.predict(new_data),
            loaded.predict(new_data),
        )


class TestPredictChunked:
    """Pin the row-chunked predict path against the single-shot path.

    `predict()` switches to chunked design-matrix construction when
    n_rows > _PREDICT_ROW_CHUNK_DEFAULT. The chunked path is
    FP-equivalent (not bit-exact) because BLAS gemv reduces row
    contributions in a different order across chunks. We verify the
    two agree to ~1 ULP relative.
    """

    def test_predict_chunked_matches_singleshot(self):
        from rustystats import formula as _formula

        rng = np.random.default_rng(7)
        n_fit = 5_000
        fit_df = pl.DataFrame(
            {
                "x": rng.standard_normal(n_fit),
                "cat": rng.integers(0, 4, n_fit).astype(str),
                "exposure": rng.uniform(0.1, 1.0, n_fit),
            }
        )
        eta = 0.5 * fit_df["x"].to_numpy() + 0.3
        mu = np.exp(eta) * fit_df["exposure"].to_numpy()
        fit_df = fit_df.with_columns(pl.Series("y", rng.poisson(mu).astype(float)))

        result = rs.glm_dict(
            response="y",
            terms={
                "x": {"type": "bs", "df": 4},
                "cat": {"type": "categorical"},
            },
            data=fit_df,
            family="poisson",
            offset="exposure",
        ).fit()

        # n deliberately NOT a multiple of any reasonable chunk size.
        n_pred = 350_001
        pred_df = pl.DataFrame(
            {
                "x": rng.standard_normal(n_pred),
                "cat": rng.integers(0, 4, n_pred).astype(str),
                "exposure": rng.uniform(0.1, 1.0, n_pred),
            }
        )

        original_chunk = _formula._PREDICT_ROW_CHUNK_DEFAULT
        try:
            _formula._PREDICT_ROW_CHUNK_DEFAULT = n_pred + 1
            single = result.predict(pred_df)
            _formula._PREDICT_ROW_CHUNK_DEFAULT = 50_000
            chunked = result.predict(pred_df)
        finally:
            _formula._PREDICT_ROW_CHUNK_DEFAULT = original_chunk

        assert single.shape == chunked.shape == (n_pred,)
        np.testing.assert_allclose(chunked, single, rtol=1e-12, atol=0)


class TestPredictAdaptiveChunk:
    """Pin adaptive chunk sizing for the chunked predict path.

    The current implementation uses a fixed chunk_size of
    `_PREDICT_ROW_CHUNK_DEFAULT = 200_000` rows regardless of model
    width (n_features). For a model with many categorical levels,
    each chunk materializes a (chunk_size × n_features) float64
    design matrix — at 200_000 × 10_000 that's 16 GB per chunk, OOM.

    The expected fix is an adaptive chunk size:
        chunk_size = min(_PREDICT_ROW_CHUNK_DEFAULT,
                         BUDGET_BYTES // (n_features * 8))
    exposed as a pure helper `_compute_predict_chunk_size(n_features)`
    so it can be unit-tested in isolation.

    The tests pin the memory-hardening behavior while keeping the
    integration cases bounded enough for routine CI runs.
    """

    @staticmethod
    def _make_wide_fit_data(n_fit: int, n_levels: int, seed: int = 11):
        """Build a fitting DataFrame with a high-cardinality categorical.

        A categorical with `n_levels` levels + 3 linear terms +
        intercept yields ~(n_levels + 3) features. Cheap way to get a
        wide model without needing many input columns.
        """
        rng = np.random.default_rng(seed)
        cats = np.array([f"L{i}" for i in range(n_levels)])
        fit_df = pl.DataFrame(
            {
                "cat": rng.choice(cats, n_fit),
                "x1": rng.standard_normal(n_fit),
                "x2": rng.standard_normal(n_fit),
                "x3": rng.standard_normal(n_fit),
                "exposure": rng.uniform(0.1, 1.0, n_fit),
            }
        )
        eta = (
            0.1 * fit_df["x1"].to_numpy()
            + 0.05 * fit_df["x2"].to_numpy()
            - 0.05 * fit_df["x3"].to_numpy()
            + 0.2
        )
        mu = np.exp(eta) * fit_df["exposure"].to_numpy()
        fit_df = fit_df.with_columns(
            pl.Series("y", rng.poisson(np.clip(mu, 1e-6, None)).astype(float))
        )
        return fit_df

    @staticmethod
    def _fit_wide_model(n_fit: int, n_levels: int, seed: int = 11):
        fit_df = TestPredictAdaptiveChunk._make_wide_fit_data(
            n_fit=n_fit, n_levels=n_levels, seed=seed
        )
        # Ridge keeps high-cardinality categorical well-conditioned.
        return rs.glm_dict(
            response="y",
            terms={
                "cat": {"type": "categorical"},
                "x1": {"type": "linear"},
                "x2": {"type": "linear"},
                "x3": {"type": "linear"},
            },
            data=fit_df,
            family="poisson",
            offset="exposure",
        ).fit(alpha=1e-3, l1_ratio=0.0)

    # ---- Test 1: Wide-model predict produces correct output --------------

    def test_wide_model_chunked_predict_matches_singleshot(self):
        """Correctness: chunked predict on a wide model agrees with single-shot.

        Monkey-patches `_PREDICT_ROW_CHUNK_DEFAULT` to force each
        branch. This test does NOT depend on the adaptive-chunk fix
        and should pass on both current and post-fix code.
        """
        from rustystats import formula as _formula

        # ~200 features: 150 categorical levels → 149 dummies + intercept +
        # 3 linear terms = 153 columns. Bump to ~200 by adding splines on x1.
        # We instead use 200 categorical levels which gives ~203 columns.
        n_levels = 200
        result = self._fit_wide_model(n_fit=4_000, n_levels=n_levels)
        n_features = len(result.params)
        # Sanity check on model width.
        assert n_features >= 150, f"expected ~200 features for the wide test, got {n_features}"

        rng = np.random.default_rng(23)
        n_pred = 400_001  # > current chunk default (200_000) to force chunking
        cats = np.array([f"L{i}" for i in range(n_levels)])
        pred_df = pl.DataFrame(
            {
                "cat": rng.choice(cats, n_pred),
                "x1": rng.standard_normal(n_pred),
                "x2": rng.standard_normal(n_pred),
                "x3": rng.standard_normal(n_pred),
                "exposure": rng.uniform(0.1, 1.0, n_pred),
            }
        )

        original = _formula._PREDICT_ROW_CHUNK_DEFAULT
        try:
            # Force single-shot by bumping the threshold above n_pred.
            _formula._PREDICT_ROW_CHUNK_DEFAULT = n_pred + 1
            single = result.predict(pred_df)
            # Force chunking with a small chunk size.
            _formula._PREDICT_ROW_CHUNK_DEFAULT = 50_000
            chunked = result.predict(pred_df)
        finally:
            _formula._PREDICT_ROW_CHUNK_DEFAULT = original

        assert single.shape == chunked.shape == (n_pred,)
        assert np.all(np.isfinite(single))
        assert np.all(np.isfinite(chunked))
        # Reasonable response-scale values (poisson + log link + exposure).
        assert (single >= 0).all()
        # FP-equivalent, not bit-exact (BLAS gemv reduction order).
        np.testing.assert_allclose(chunked, single, rtol=1e-10, atol=0)

    # ---- Test 2: Chunked predict with wide model completes (no-OOM smoke) -

    @pytest.mark.slow
    def test_wide_model_large_predict_completes(self):
        """Smoke test: wide model + large predict completes in reasonable time.

        We can't easily assert peak RSS in a unit test. This test
        just verifies the call returns a correct-shape finite array
        under a generous timeout. With the adaptive-chunk fix the
        per-chunk allocation is bounded; without it, a bigger model
        would OOM on CI.
        """
        import time

        # ~300 features: 300 categorical levels → 299 dummies + intercept +
        # 3 linear terms ≈ 303 columns.
        n_levels = 300
        result = self._fit_wide_model(n_fit=4_000, n_levels=n_levels)
        n_features = len(result.params)
        assert n_features >= 250, f"expected wide model, got {n_features} features"

        rng = np.random.default_rng(37)
        n_pred = 500_000
        cats = np.array([f"L{i}" for i in range(n_levels)])
        pred_df = pl.DataFrame(
            {
                "cat": rng.choice(cats, n_pred),
                "x1": rng.standard_normal(n_pred),
                "x2": rng.standard_normal(n_pred),
                "x3": rng.standard_normal(n_pred),
                "exposure": rng.uniform(0.1, 1.0, n_pred),
            }
        )

        t0 = time.perf_counter()
        preds = result.predict(pred_df)
        elapsed = time.perf_counter() - t0

        assert preds.shape == (n_pred,)
        assert np.all(np.isfinite(preds))
        assert elapsed < 60.0, f"predict too slow: {elapsed:.1f}s"

    # ---- Test 3: _compute_predict_chunk_size helper invariants -----------

    def test_compute_predict_chunk_size_narrow_model_unchanged(self):
        """Small/narrow models keep the default chunk size.

        For a tiny model (p=10) the memory budget is a non-binding
        constraint, so chunk_size should equal the default (200_000).
        """
        from rustystats.formula import (
            _PREDICT_ROW_CHUNK_DEFAULT,
            _compute_predict_chunk_size,
        )

        assert _compute_predict_chunk_size(10) == _PREDICT_ROW_CHUNK_DEFAULT
        assert _compute_predict_chunk_size(50) == _PREDICT_ROW_CHUNK_DEFAULT

    def test_compute_predict_chunk_size_wide_model_shrinks(self):
        """Wide models get a chunk size below the default.

        For p=1000 the default-sized chunk would materialize
        200_000 × 1000 × 8B = 1.6 GB per chunk. The adaptive logic
        should cap this below the default.
        """
        from rustystats.formula import (
            _PREDICT_ROW_CHUNK_DEFAULT,
            _compute_predict_chunk_size,
        )

        chunk = _compute_predict_chunk_size(1000)
        assert chunk < _PREDICT_ROW_CHUNK_DEFAULT
        assert chunk > 0
        # Sanity: per-chunk float64 bytes should not blow past a few
        # hundred MB. At p=1000, 50_000 rows × 1000 × 8B = 400 MB which
        # is a rough upper bound — if the budget is ~200 MB we'd expect
        # chunk ≤ 25_000, but we leave headroom for implementation choice.
        per_chunk_bytes = chunk * 1000 * 8
        assert (
            per_chunk_bytes <= 1_000_000_000
        ), f"per-chunk allocation too large: {per_chunk_bytes:,} bytes"

    def test_compute_predict_chunk_size_very_wide_model_positive(self):
        """Very wide models get a small but strictly positive chunk size.

        For p=10_000 the chunk size must be small — but NEVER zero,
        or the predict loop would divide-by-zero / spin forever.
        """
        from rustystats.formula import _compute_predict_chunk_size

        chunk = _compute_predict_chunk_size(10_000)
        assert chunk > 0, "chunk size must be strictly positive"
        # And strictly smaller than for p=1000 (monotonicity is pinned
        # separately below; this is just a sanity cap).
        assert chunk < 200_000

    def test_compute_predict_chunk_size_monotonic_in_width(self):
        """Chunk size is non-increasing as n_features grows.

        chunk_size(p1) >= chunk_size(p2) whenever p1 <= p2.
        """
        from rustystats.formula import _compute_predict_chunk_size

        widths = [1, 10, 50, 200, 500, 1_000, 5_000, 10_000, 50_000]
        chunks = [_compute_predict_chunk_size(p) for p in widths]
        for p1, p2, c1, c2 in zip(widths, widths[1:], chunks, chunks[1:]):
            assert c1 >= c2, f"non-monotonic: chunk_size({p1})={c1} < chunk_size({p2})={c2}"
        # All strictly positive.
        assert all(c > 0 for c in chunks)

    def test_compute_predict_chunk_size_budget_cap_observed(self):
        """Per-chunk bytes stay within a reasonable memory budget.

        For any moderately wide model, `chunk_size * n_features * 8`
        should fit in a few hundred MB. Exact budget is the dev's call
        (spec targets ~200 MB ≈ 25M float64 values); we pin a loose
        upper bound of 1 GB so the test is robust to reasonable choices.
        """
        from rustystats.formula import _compute_predict_chunk_size

        for p in (100, 500, 1_000, 5_000, 10_000):
            chunk = _compute_predict_chunk_size(p)
            per_chunk_bytes = chunk * p * 8
            assert (
                per_chunk_bytes <= 1_000_000_000
            ), f"p={p}: per-chunk {per_chunk_bytes:,} bytes exceeds 1 GB"

    # ---- Test 3b: Observable chunk-count scaling (integration) ----------

    def test_chunk_count_scales_with_model_width(self):
        """Wider models produce MORE chunks for the same n_rows.

        Counts calls to `InteractionBuilder.transform_new_data` via
        monkey-patching. For the same n_rows, a wider model must
        trigger at least as many chunks as a narrower model (and
        strictly more once the width forces the budget below the
        default). This is the user-visible effect of the adaptive
        chunk sizing.

        This guards against regressing to a fixed 200k-row chunk size.
        """
        from unittest.mock import patch

        from rustystats.interactions import InteractionBuilder

        narrow = self._fit_wide_model(n_fit=500, n_levels=20)
        wide = self._fit_wide_model(n_fit=800, n_levels=220)

        # Must be wide enough that the adaptive budget bites.
        assert len(wide.params) >= 150, (
            f"wide model only has {len(wide.params)} features; "
            "test cannot distinguish the adaptive-chunk effect"
        )

        rng = np.random.default_rng(101)
        n_pred = 400_000  # > default chunk to force chunking on both

        def _make_pred_df(n_levels_: int) -> pl.DataFrame:
            cats = np.array([f"L{i}" for i in range(n_levels_)])
            return pl.DataFrame(
                {
                    "cat": rng.choice(cats, n_pred),
                    "x1": rng.standard_normal(n_pred),
                    "x2": rng.standard_normal(n_pred),
                    "x3": rng.standard_normal(n_pred),
                    "exposure": rng.uniform(0.1, 1.0, n_pred),
                }
            )

        pred_narrow = _make_pred_df(20)
        pred_wide = _make_pred_df(220)

        original = InteractionBuilder.transform_new_data

        def count_calls(model, pred_df):
            counter = {"n": 0}

            def counting_transform(self, data):
                counter["n"] += 1
                return original(self, data)

            with patch.object(InteractionBuilder, "transform_new_data", counting_transform):
                model.predict(pred_df)
            return counter["n"]

        n_chunks_narrow = count_calls(narrow, pred_narrow)
        n_chunks_wide = count_calls(wide, pred_wide)

        # Narrow model at n=400k with default chunk=200k uses at least 2 chunks.
        assert n_chunks_narrow >= 2
        # Wide model should produce *more* chunks once adaptive sizing
        # shrinks per-chunk rows below the default.
        assert n_chunks_wide > n_chunks_narrow, (
            f"adaptive chunking not active: narrow={n_chunks_narrow} "
            f"chunks, wide={n_chunks_wide} chunks (expected wide > narrow)"
        )

    # ---- Test 4: Small-n fast path unaffected ----------------------------

    def test_small_n_fast_path_unaffected(self):
        """Small n_rows uses the single-shot path regardless of width.

        For n=100 rows, no chunking should happen: the predict output
        must equal what a single-shot call produces (bit-exact, since
        it IS the same call path). Verified by counting calls to
        `transform_new_data` — must be exactly 1.

        This test should pass on both current and post-fix code.
        """
        from unittest.mock import patch

        from rustystats.interactions import InteractionBuilder

        result = self._fit_wide_model(n_fit=2_000, n_levels=100)

        rng = np.random.default_rng(55)
        n_pred = 100
        cats = np.array([f"L{i}" for i in range(100)])
        pred_df = pl.DataFrame(
            {
                "cat": rng.choice(cats, n_pred),
                "x1": rng.standard_normal(n_pred),
                "x2": rng.standard_normal(n_pred),
                "x3": rng.standard_normal(n_pred),
                "exposure": rng.uniform(0.1, 1.0, n_pred),
            }
        )

        original = InteractionBuilder.transform_new_data
        counter = {"n": 0}

        def counting_transform(self, data):
            counter["n"] += 1
            return original(self, data)

        with patch.object(InteractionBuilder, "transform_new_data", counting_transform):
            preds = result.predict(pred_df)

        assert (
            counter["n"] == 1
        ), f"small-n should use single-shot path, got {counter['n']} calls to transform_new_data"
        assert preds.shape == (n_pred,)
        assert np.all(np.isfinite(preds))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
