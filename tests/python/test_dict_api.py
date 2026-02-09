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
        return pl.DataFrame({
            'y': np.random.poisson(1, n),
            'x1': np.random.uniform(0, 10, n),
            'x2': np.random.uniform(0, 10, n),
            'cat': np.random.choice(['A', 'B', 'C'], n),
            'exposure': np.random.uniform(0.5, 1.5, n),
            'weight': np.random.uniform(0.5, 2.0, n),
        })

    def test_basic_construction(self, sample_data):
        model = rs.glm_dict(
            response='y',
            terms={'x1': {'type': 'linear'}, 'x2': {'type': 'linear'}},
            data=sample_data,
        )
        assert model.family == 'gaussian'
        assert model.n_obs == 100
        assert model.n_params == 3  # Intercept + x1 + x2

    def test_poisson_construction(self, sample_data):
        model = rs.glm_dict(
            response='y',
            terms={'x1': {'type': 'linear'}},
            data=sample_data,
            family='poisson',
        )
        assert model.family == 'poisson'

    def test_binomial_construction(self):
        np.random.seed(42)
        data = pl.DataFrame({
            'y': np.random.binomial(1, 0.5, 100),
            'x1': np.random.uniform(0, 10, 100),
        })
        model = rs.glm_dict(
            response='y',
            terms={'x1': {'type': 'linear'}},
            data=data,
            family='binomial',
        )
        assert model.family == 'binomial'

    def test_gamma_construction(self):
        np.random.seed(42)
        data = pl.DataFrame({
            'y': np.random.gamma(2, 2, 100),
            'x1': np.random.uniform(0, 10, 100),
        })
        model = rs.glm_dict(
            response='y',
            terms={'x1': {'type': 'linear'}},
            data=data,
            family='gamma',
        )
        assert model.family == 'gamma'

    def test_offset_as_column_name(self, sample_data):
        model = rs.glm_dict(
            response='y',
            terms={'x1': {'type': 'linear'}},
            data=sample_data,
            family='poisson',
            offset='exposure',
        )
        assert model.offset is not None
        assert len(model.offset) == 100

    def test_offset_as_array(self, sample_data):
        offset_arr = np.log(sample_data['exposure'].to_numpy())
        model = rs.glm_dict(
            response='y',
            terms={'x1': {'type': 'linear'}},
            data=sample_data,
            family='poisson',
            offset=offset_arr,
        )
        assert model.offset is not None

    def test_weights_as_column_name(self, sample_data):
        model = rs.glm_dict(
            response='y',
            terms={'x1': {'type': 'linear'}},
            data=sample_data,
            weights='weight',
        )
        assert model.weights is not None
        assert len(model.weights) == 100

    def test_weights_as_array(self, sample_data):
        weights_arr = sample_data['weight'].to_numpy()
        model = rs.glm_dict(
            response='y',
            terms={'x1': {'type': 'linear'}},
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
        data = pl.DataFrame({'y': y, 'x': x})

        result = rs.glm_dict(
            response='y', terms={'x': {'type': 'linear'}},
            data=data, family='gaussian',
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
        data = pl.DataFrame({'y': y, 'x': x})

        result = rs.glm_dict(
            response='y', terms={'x': {'type': 'linear'}},
            data=data, family='poisson',
        ).fit()

        assert result.converged
        assert len(result.params) == 2

    def test_fit_binomial(self):
        np.random.seed(42)
        n = 200
        x = np.random.uniform(-2, 2, n)
        p = 1 / (1 + np.exp(-(0.5 + x)))
        y = np.random.binomial(1, p)
        data = pl.DataFrame({'y': y.astype(float), 'x': x})

        result = rs.glm_dict(
            response='y', terms={'x': {'type': 'linear'}},
            data=data, family='binomial',
        ).fit()

        assert result.converged
        assert len(result.params) == 2

    def test_fit_gamma(self):
        np.random.seed(42)
        n = 200
        x = np.random.uniform(1, 5, n)
        mu = np.exp(1 + 0.3 * x)
        y = np.random.gamma(2, mu / 2, n)
        data = pl.DataFrame({'y': y, 'x': x})

        result = rs.glm_dict(
            response='y', terms={'x': {'type': 'linear'}},
            data=data, family='gamma',
        ).fit()

        assert result.converged
        assert len(result.params) == 2

    def test_fit_quasipoisson(self):
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 5, n)
        y = np.random.poisson(np.exp(0.5 + 0.3 * x))
        data = pl.DataFrame({'y': y, 'x': x})

        result = rs.glm_dict(
            response='y', terms={'x': {'type': 'linear'}},
            data=data, family='quasipoisson',
        ).fit()

        assert result.converged

    def test_fit_negbinomial(self):
        np.random.seed(42)
        n = 200
        x = np.random.uniform(0, 2, n)
        mu = np.exp(0.5 + 0.3 * x)
        y = np.random.negative_binomial(2, 2 / (2 + mu))
        data = pl.DataFrame({'y': y.astype(float), 'x': x})

        result = rs.glm_dict(
            response='y', terms={'x': {'type': 'linear'}},
            data=data, family='negbinomial',
        ).fit()

        assert result.converged
        assert 'NegativeBinomial' in result.family

    def test_fit_with_offset(self):
        np.random.seed(42)
        n = 100
        exposure = np.random.uniform(0.5, 2, n)
        x = np.random.uniform(0, 5, n)
        y = np.random.poisson(exposure * np.exp(0.5 + 0.2 * x))
        data = pl.DataFrame({'y': y, 'x': x, 'exposure': exposure})

        result = rs.glm_dict(
            response='y', terms={'x': {'type': 'linear'}},
            data=data, family='poisson', offset='exposure',
        ).fit()

        assert result.converged

    def test_fit_with_weights(self):
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 10, n)
        y = 2 + 3 * x + np.random.normal(0, 1, n)
        weights = np.random.uniform(0.5, 2, n)
        data = pl.DataFrame({'y': y, 'x': x, 'w': weights})

        result = rs.glm_dict(
            response='y', terms={'x': {'type': 'linear'}},
            data=data, weights='w',
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
        data = pl.DataFrame({'y': y, 'x': x})
        return rs.glm_dict(
            response='y', terms={'x': {'type': 'linear'}}, data=data,
        ).fit()

    def test_params_shape(self, fitted_result):
        assert len(fitted_result.params) == 2

    def test_feature_names(self, fitted_result):
        assert 'Intercept' in fitted_result.feature_names
        assert 'x' in fitted_result.feature_names

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
        valid_codes = ['***', '**', '*', '.', '']
        for code in codes:
            assert code in valid_codes

    def test_summary(self, fitted_result):
        summary = fitted_result.summary()
        assert isinstance(summary, str)
        assert 'Intercept' in summary
        assert 'x' in summary


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
        data = pl.DataFrame({'y': y, 'x': x})
        return rs.glm_dict(
            response='y', terms={'x': {'type': 'linear'}}, data=data,
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
        data = pl.DataFrame({'y': y, 'x': x})
        return rs.glm_dict(
            response='y', terms={'x': {'type': 'linear'}}, data=data,
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
        data = {'y': np.random.poisson(2, n)}
        for i in range(10):
            data[f'x{i}'] = np.random.uniform(0, 10, n)
        data['exposure'] = np.ones(n)
        return pl.DataFrame(data)

    def _terms_x(self, n):
        return {f'x{i}': {'type': 'linear'} for i in range(n)}

    def test_ridge_regularization(self, sample_data):
        result = rs.glm_dict(
            response='y', terms=self._terms_x(10),
            data=sample_data, family='poisson', offset='exposure',
        ).fit(alpha=0.1, l1_ratio=0.0)

        assert result.converged
        assert result.is_regularized

    def test_lasso_regularization(self, sample_data):
        result = rs.glm_dict(
            response='y', terms=self._terms_x(10),
            data=sample_data, family='poisson', offset='exposure',
        ).fit(alpha=0.1, l1_ratio=1.0)

        assert result.converged
        assert result.is_regularized

    def test_elastic_net_regularization(self, sample_data):
        result = rs.glm_dict(
            response='y', terms=self._terms_x(10),
            data=sample_data, family='poisson', offset='exposure',
        ).fit(alpha=0.1, l1_ratio=0.5)

        assert result.converged
        assert result.is_regularized

    def test_no_regularization(self, sample_data):
        result = rs.glm_dict(
            response='y', terms={'x0': {'type': 'linear'}, 'x1': {'type': 'linear'}},
            data=sample_data, family='poisson', offset='exposure',
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
        data = pl.DataFrame({'y': y, 'x': x})

        result = rs.glm_dict(
            response='y', terms={'x': {'type': 'linear'}}, data=data,
        ).fit()
        predictions = result.fittedvalues

        assert len(predictions) == n
        assert np.corrcoef(predictions, y)[0, 1] > 0.9

    def test_predict_on_new_data(self):
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 10, n)
        y = 2 + 3 * x + np.random.normal(0, 1, n)
        train_data = pl.DataFrame({'y': y, 'x': x})

        result = rs.glm_dict(
            response='y', terms={'x': {'type': 'linear'}}, data=train_data,
        ).fit()

        new_x = np.array([1.0, 5.0, 9.0])
        new_data = pl.DataFrame({'x': new_x})

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
        data = pl.DataFrame({'y': np.random.normal(5, 1, 100)})

        result = rs.glm_dict(
            response='y', terms={}, data=data,
        ).fit()

        assert result.converged
        assert len(result.params) == 1
        assert abs(result.params[0] - 5) < 0.5

    def test_no_intercept_model(self):
        np.random.seed(42)
        x = np.random.uniform(1, 10, 100)
        y = 3 * x + np.random.normal(0, 1, 100)
        data = pl.DataFrame({'y': y, 'x': x})

        result = rs.glm_dict(
            response='y', terms={'x': {'type': 'linear'}},
            data=data, intercept=False,
        ).fit()

        assert result.converged
        assert 'Intercept' not in result.feature_names

    def test_single_observation_fails(self):
        data = pl.DataFrame({'y': [1.0], 'x': [1.0]})
        with pytest.raises(Exception):
            rs.glm_dict(
                response='y', terms={'x': {'type': 'linear'}}, data=data,
            ).fit()

    def test_missing_variable_fails(self):
        data = pl.DataFrame({'y': [1.0, 2.0], 'x': [1.0, 2.0]})
        with pytest.raises((KeyError, Exception)):
            rs.glm_dict(
                response='y', terms={'z': {'type': 'linear'}}, data=data,
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
        return pl.DataFrame({
            "y": np.random.poisson(2, n).astype(float),
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "cat": np.random.choice(["A", "B", "C"], n),
            "exposure": np.random.uniform(0.5, 2.0, n),
        })

    def test_simple_model_roundtrip(self, sample_data):
        result = rs.glm_dict(
            response="y", terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=sample_data, family="poisson",
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
            data=sample_data, family="poisson",
        ).fit()

        model_bytes = result.to_bytes()
        loaded = rs.GLMModel.from_bytes(model_bytes)

        np.testing.assert_array_almost_equal(loaded.params, result.params)
        assert loaded.feature_names == result.feature_names

    def test_offset_model_roundtrip(self, sample_data):
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "cat": {"type": "categorical"}},
            data=sample_data, family="poisson", offset="exposure",
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
        return pl.DataFrame({
            "y": np.random.poisson(2, n).astype(float),
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "cat": np.random.choice(["A", "B", "C"], n),
            "exposure": np.random.uniform(0.5, 2.0, n),
        })

    def test_predict_after_load(self, sample_data):
        train = sample_data.head(400)
        test = sample_data.tail(100)

        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}, "cat": {"type": "categorical"}},
            data=train, family="poisson",
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
            data=train, family="poisson", offset="exposure",
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
        return pl.DataFrame({
            "y": np.random.poisson(2, n).astype(float),
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "cat": np.random.choice(["A", "B", "C"], n),
            "exposure": np.random.uniform(0.5, 2.0, n),
        })

    def test_spline_model_roundtrip(self, sample_data):
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "bs", "df": 4}, "x2": {"type": "linear"}},
            data=sample_data, family="poisson",
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
            data=train, family="poisson",
        ).fit()
        original_pred = result.predict(test)

        loaded = rs.GLMModel.from_bytes(result.to_bytes())
        loaded_pred = loaded.predict(test)

        np.testing.assert_array_almost_equal(original_pred, loaded_pred)


class TestDictInteractionSerialization:
    """Test serialization with interactions (mirrors TestInteractionSerialization)."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 500
        return pl.DataFrame({
            "y": np.random.poisson(2, n).astype(float),
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "cat": np.random.choice(["A", "B", "C"], n),
            "exposure": np.random.uniform(0.5, 2.0, n),
        })

    def test_interaction_model_roundtrip(self, sample_data):
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            interactions=[
                {"x1": {"type": "linear"}, "x2": {"type": "linear"}, "include_main": False},
            ],
            data=sample_data, family="poisson",
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
            data=train, family="poisson",
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
        return pl.DataFrame({
            "y": np.random.poisson(2, n).astype(float),
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "cat": np.random.choice(["A", "B", "C"], n),
            "exposure": np.random.uniform(0.5, 2.0, n),
        })

    def test_properties_preserved(self, sample_data):
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}, "cat": {"type": "categorical"}},
            data=sample_data, family="poisson",
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
                data=data, family=family,
            ).fit()
            loaded = rs.GLMModel.from_bytes(result.to_bytes())

            assert loaded.family == family
            np.testing.assert_array_almost_equal(loaded.params, result.params)

    def test_intercept_only_model(self, sample_data):
        result = rs.glm_dict(
            response="y", terms={}, data=sample_data, family="poisson",
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
        area = np.random.choice(['Urban', 'Suburban', 'Rural'], n)
        log_rate = -3.0 + 0.02 * age + 0.01 * power - 0.0001 * age * power
        log_rate += np.where(area == 'Urban', 0.3, np.where(area == 'Suburban', 0.1, 0.0))
        claims = np.random.poisson(np.exp(log_rate))
        exposure = np.random.uniform(0.5, 1.0, n)
        return pl.DataFrame({
            'claims': claims, 'age': age, 'power': power,
            'area': area, 'exposure': exposure,
        })

    def test_fit_continuous_interaction(self, insurance_data):
        result = rs.glm_dict(
            response='claims',
            terms={'age': {'type': 'linear'}, 'power': {'type': 'linear'}},
            interactions=[
                {'age': {'type': 'linear'}, 'power': {'type': 'linear'}, 'include_main': False},
            ],
            data=insurance_data, family='poisson', offset='exposure',
        ).fit()

        assert len(result.params) == 4  # Intercept, age, power, age:power
        assert result.converged
        summary = result.summary()
        assert 'age:power' in summary

    def test_fit_categorical_continuous_interaction(self, insurance_data):
        result = rs.glm_dict(
            response='claims',
            terms={'area': {'type': 'categorical'}, 'age': {'type': 'linear'}},
            interactions=[
                {'area': {'type': 'categorical'}, 'age': {'type': 'linear'}, 'include_main': False},
            ],
            data=insurance_data, family='poisson', offset='exposure',
        ).fit()

        # Intercept + 2 area dummies + age + 2 interactions = 6
        assert len(result.params) == 6
        assert result.converged

    def test_fit_categorical_categorical_interaction(self, insurance_data):
        insurance_data = insurance_data.with_columns(
            pl.Series('fuel', np.random.choice(['Petrol', 'Diesel'], len(insurance_data)))
        )

        result = rs.glm_dict(
            response='claims',
            terms={'area': {'type': 'categorical'}, 'fuel': {'type': 'categorical'}},
            interactions=[
                {'area': {'type': 'categorical'}, 'fuel': {'type': 'categorical'}, 'include_main': False},
            ],
            data=insurance_data, family='poisson', offset='exposure',
        ).fit()

        # area: 2 dummies, fuel: 1 dummy, Total: 1 + 2 + 1 + 2×1 = 6
        assert len(result.params) == 6
        assert result.converged

    def test_regularized_interaction_model(self, insurance_data):
        result = rs.glm_dict(
            response='claims',
            terms={
                'age': {'type': 'linear'}, 'power': {'type': 'linear'},
                'area': {'type': 'categorical'},
            },
            interactions=[
                {'age': {'type': 'linear'}, 'power': {'type': 'linear'}, 'include_main': False},
            ],
            data=insurance_data, family='poisson', offset='exposure',
        ).fit(alpha=0.1, l1_ratio=0.0)

        assert result.is_regularized
        assert result.converged

    def test_predictions_with_interactions(self, insurance_data):
        result = rs.glm_dict(
            response='claims',
            terms={'age': {'type': 'linear'}, 'power': {'type': 'linear'}},
            interactions=[
                {'age': {'type': 'linear'}, 'power': {'type': 'linear'}, 'include_main': False},
            ],
            data=insurance_data, family='poisson', offset='exposure',
        ).fit()

        fv = result.fittedvalues
        assert np.all(fv >= 0)
        assert len(fv) == len(insurance_data)

    def test_large_categorical_interaction(self):
        np.random.seed(42)
        n = 50_000
        df = pl.DataFrame({
            'y': np.random.poisson(1, n),
            'cat1': np.random.choice([f'A{i}' for i in range(10)], n),
            'cat2': np.random.choice([f'B{i}' for i in range(8)], n),
            'exposure': np.random.uniform(0.5, 1.0, n),
        })

        import time
        t0 = time.time()
        result = rs.glm_dict(
            response='y',
            terms={'cat1': {'type': 'categorical'}, 'cat2': {'type': 'categorical'}},
            interactions=[
                {'cat1': {'type': 'categorical'}, 'cat2': {'type': 'categorical'}, 'include_main': False},
            ],
            data=df, family='poisson', offset='exposure',
        ).fit()
        t_opt = time.time() - t0

        assert t_opt < 30.0, f"Took {t_opt:.1f}s (expected < 30s)"
        # cat1: 9 dummies, cat2: 7 dummies, Total: 1 + 9 + 7 + 63 = 80
        assert len(result.params) == 80
        assert result.converged

    def test_fit_specific_levels_categorical(self):
        np.random.seed(42)
        data = pl.DataFrame({
            'claims': np.random.poisson(0.1, 100),
            'Region': np.random.choice(['Paris', 'Lyon', 'Marseille'], 100),
            'age': np.random.uniform(20, 60, 100),
            'exposure': np.random.uniform(0.5, 1.5, 100),
        })

        result = rs.glm_dict(
            response='claims',
            terms={
                'Region': {'type': 'categorical', 'levels': ['Paris']},
                'age': {'type': 'linear'},
            },
            data=data, family='poisson', offset='exposure',
        ).fit()

        assert result.converged
        assert "Region[Paris]" in result.feature_names
        assert len(result.params) == 3  # Intercept + Paris + age


# =============================================================================
# Mirrors: test_regularization_path.py → TestCVRegularizationFit
# =============================================================================

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
        return pl.DataFrame({
            "ClaimCount": claims, "Age": age, "BonusMalus": bonus_malus, "Exposure": exposure,
        })

    def test_ridge_cv_basic(self, simple_data):
        model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}, "x3": {"type": "linear"}},
            data=simple_data, family="poisson",
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
            data=simple_data, family="poisson",
        )
        result = model.fit(cv=3, regularization="lasso", n_alphas=10, verbose=False)

        assert result.cv_deviance is not None
        assert result.regularization_type in ("lasso", "none")

    def test_1se_selection_more_regularized(self, simple_data):
        model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}, "x3": {"type": "linear"}},
            data=simple_data, family="poisson",
        )
        result_min = model.fit(
            cv=3, regularization="ridge", n_alphas=20,
            selection="min", cv_seed=42, verbose=False,
        )
        result_1se = model.fit(
            cv=3, regularization="ridge", n_alphas=20,
            selection="1se", cv_seed=42, verbose=False,
        )
        assert result_1se.alpha >= result_min.alpha

    def test_cv_requires_regularization_type(self, simple_data):
        model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=simple_data, family="poisson",
        )
        with pytest.raises(rs.ValidationError, match="regularization"):
            model.fit(cv=5)

    def test_explicit_alpha_no_cv(self, simple_data):
        model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=simple_data, family="poisson",
        )
        result = model.fit(alpha=0.1, l1_ratio=0.0)
        assert result.alpha == pytest.approx(0.1)
        assert result.cv_deviance is None

    def test_poisson_with_offset(self, insurance_like_data):
        model = rs.glm_dict(
            response="ClaimCount",
            terms={"Age": {"type": "linear"}, "BonusMalus": {"type": "linear"}},
            data=insurance_like_data, family="poisson", offset="Exposure",
        )
        result = model.fit(cv=3, regularization="ridge", n_alphas=10, verbose=False)

        assert result.cv_deviance is not None
        assert result.converged

    def test_cv_reproducibility(self, insurance_like_data):
        model = rs.glm_dict(
            response="ClaimCount",
            terms={"Age": {"type": "linear"}, "BonusMalus": {"type": "linear"}},
            data=insurance_like_data, family="poisson", offset="Exposure",
        )
        result1 = model.fit(cv=3, regularization="ridge", n_alphas=10, cv_seed=42, verbose=False)
        result2 = model.fit(cv=3, regularization="ridge", n_alphas=10, cv_seed=42, verbose=False)

        assert result1.alpha == pytest.approx(result2.alpha)
        assert result1.cv_deviance == pytest.approx(result2.cv_deviance)

    def test_all_cv_attributes_present(self, simple_data):
        model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=simple_data, family="poisson",
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
            data=simple_data, family="poisson",
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
            data=data, family="poisson",
        ).fit()

        return result, data

    def test_compute_diagnostics(self, fitted_model):
        from rustystats.diagnostics import compute_diagnostics
        result, data = fitted_model

        diagnostics = compute_diagnostics(
            result=result, train_data=data,
            categorical_factors=["region"], continuous_factors=["age"],
        )

        assert diagnostics.model_summary is not None
        assert diagnostics.train_test is not None
        assert diagnostics.train_test.train.loss > 0
        assert diagnostics.calibration is not None
        assert len(diagnostics.factors) == 2

    def test_diagnostics_to_json(self, fitted_model):
        from rustystats.diagnostics import compute_diagnostics
        import json
        result, data = fitted_model

        diagnostics = compute_diagnostics(
            result=result, train_data=data,
            categorical_factors=["region"], continuous_factors=["age"],
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
            categorical_factors=["region"], continuous_factors=["age"],
        )
        assert diagnostics is not None
        assert len(diagnostics.factors) == 2

    def test_diagnostics_json_method(self, fitted_model):
        import json
        result, data = fitted_model
        json_str = result.diagnostics_json(
            train_data=data, categorical_factors=["region"], continuous_factors=["age"],
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
            response="y", terms={"x": {"type": "linear"}},
            data=data, family="gaussian",
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
            response="y", terms={"x": {"type": "linear"}},
            data=data, family="binomial",
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

        data = pl.DataFrame({
            "y": y, "age": age, "veh_power": veh_power,
            "region": region, "exposure": exposure,
        })

        result = rs.glm_dict(
            response="y",
            terms={
                "age": {"type": "linear"}, "veh_power": {"type": "linear"},
                "region": {"type": "categorical"},
            },
            data=data, family="poisson", offset="exposure",
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
            train_data=data, categorical_factors=["region"], continuous_factors=["age"],
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
            data=data, family="poisson",
        ).fit()

        diagnostics = result.diagnostics(train_data=data, continuous_factors=["x1", "x2"])

        assert diagnostics.vif is not None
        assert len(diagnostics.vif) > 0
        for v in diagnostics.vif:
            assert v.severity in ('severe', 'moderate')

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

        test_data = pl.DataFrame({
            "y": y, "age": age, "veh_power": veh_power,
            "region": region, "exposure": exposure,
        })

        diagnostics = result.diagnostics(
            train_data=train_data, test_data=test_data,
            categorical_factors=["region"], continuous_factors=["age", "veh_power"],
        )

        tt = diagnostics.train_test
        assert tt.train is not None
        assert tt.test is not None
        assert tt.train.dataset == "train"
        assert tt.test.dataset == "test"
        assert hasattr(tt, 'gini_gap')
        assert hasattr(tt, 'overfitting_risk')
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

        data = pl.DataFrame({
            "y": y, "age": age, "region": region,
            "unfitted_var": unfitted_var, "unfitted_cat": unfitted_cat,
        })

        result = rs.glm_dict(
            response="y",
            terms={"age": {"type": "linear"}, "region": {"type": "categorical"}},
            data=data, family="poisson",
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
        data = pl.DataFrame({
            "y": np.random.poisson(3, n),
            "age": np.random.uniform(20, 70, n),
        })

        result = rs.glm_dict(
            response="y", terms={"age": {"type": "bs", "df": 5}},
            data=data, family="poisson",
        ).fit()

        assert len(result.params) >= 2

    def test_dict_with_ns(self):
        np.random.seed(42)
        n = 100
        data = pl.DataFrame({
            "y": np.random.normal(0, 1, n),
            "x": np.random.uniform(0, 10, n),
        })

        result = rs.glm_dict(
            response="y", terms={"x": {"type": "ns", "df": 4}},
            data=data, family="gaussian",
        ).fit()

        assert result.converged
        assert len(result.params) >= 2

    def test_dict_spline_with_categorical(self):
        np.random.seed(42)
        n = 200
        data = pl.DataFrame({
            "y": np.random.poisson(2, n),
            "age": np.random.uniform(20, 70, n),
            "region": np.random.choice(["A", "B", "C"], n),
        })

        result = rs.glm_dict(
            response="y",
            terms={"age": {"type": "bs", "df": 4}, "region": {"type": "categorical"}},
            data=data, family="poisson",
        ).fit()

        assert result.converged
        assert len(result.params) >= 4

    def test_dict_multiple_splines(self):
        np.random.seed(42)
        n = 150
        data = pl.DataFrame({
            "y": np.random.poisson(3, n),
            "age": np.random.uniform(20, 70, n),
            "income": np.random.uniform(30000, 150000, n),
        })

        result = rs.glm_dict(
            response="y",
            terms={"age": {"type": "bs", "df": 4}, "income": {"type": "ns", "df": 3}},
            data=data, family="poisson",
        ).fit()

        assert result.converged


class TestDictMonotonicSplineFormula:
    """Test monotonic splines via dict API (mirrors TestMonotonicSplineFormula)."""

    def test_dict_monotonic_bs_basic(self):
        np.random.seed(42)
        n = 100
        data = pl.DataFrame({
            "y": np.random.poisson(3, n),
            "age": np.random.uniform(20, 70, n),
        })

        result = rs.glm_dict(
            response="y",
            terms={"age": {"type": "bs", "df": 5, "monotonicity": "increasing"}},
            data=data, family="poisson",
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
            data=data, family="poisson",
        ).fit(max_iter=100, alpha=1e-4)

        assert len(result.params) >= 2
        assert result.converged

    def test_dict_monotonic_bs_with_other_terms(self):
        np.random.seed(42)
        n = 200
        data = pl.DataFrame({
            "y": np.random.poisson(2, n),
            "age": np.random.uniform(20, 70, n),
            "income": np.random.uniform(30000, 150000, n),
            "region": np.random.choice(["A", "B", "C"], n),
        })

        result = rs.glm_dict(
            response="y",
            terms={
                "age": {"type": "bs", "df": 4, "monotonicity": "increasing"},
                "income": {"type": "bs", "df": 3},
                "region": {"type": "categorical"},
            },
            data=data, family="poisson",
        ).fit(max_iter=100)

        assert len(result.params) >= 5
        assert result.converged


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
            response="y", terms={"x": {"type": "ns", "df": 5}},
            data=train_data, family="gaussian",
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
            response="y", terms={"x": {"type": "bs", "df": 5}},
            data=train_data, family="gaussian",
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
            data=train_data, family="gaussian",
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
            response="y", terms={"cat": {"type": "categorical"}},
            data=train_data, family="gaussian",
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
            response="y", terms={"cat": {"type": "categorical"}},
            data=train_data, family="gaussian",
        ).fit()
        test_pred = result.predict(test_data)

        assert np.all(np.isfinite(test_pred))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
