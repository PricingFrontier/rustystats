"""Direct tests for diagnostics helper utilities and dataclass serialization."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest
import rustystats.diagnostics.api as diagnostics_api
import rustystats.diagnostics.components as components_mod
import rustystats.diagnostics.computer as computer_mod
import rustystats.diagnostics.explorer as explorer_mod
import rustystats.diagnostics.factors as factors_mod
import rustystats.diagnostics.pair_diagnostics as pair_mod
from rustystats.diagnostics.computer import DiagnosticsComputer, _gini_for_arrays, rank_sort_idx
from rustystats.diagnostics.explorer import DataExplorer
from rustystats.diagnostics.factors import (
    _FactorDiagnosticsComputer,
    _FactorFeature,
    _FactorFeatureIndex,
    _inference_is_valid,
)
from rustystats.diagnostics.interactions import _InteractionDetector
from rustystats.diagnostics.types import (
    DataExploration,
    InteractionCandidate,
    Percentiles,
    SmoothTermDiagnostics,
    _extract_base_variable,
    _json_default,
    _round_float,
    _to_dict_recursive,
)
from rustystats.diagnostics.utils import discretize, validate_factor_in_data
from rustystats.exceptions import DesignMatrixError, FittingError, ValidationError


def _basic_computer(**kwargs) -> DiagnosticsComputer:
    y = np.asarray(kwargs.pop("y", [1.0, 0.0, 2.0, 1.0]), dtype=np.float64)
    mu = np.asarray(kwargs.pop("mu", [1.1, 0.9, 1.8, 1.2]), dtype=np.float64)
    return DiagnosticsComputer(
        y=y,
        mu=mu,
        linear_predictor=np.log(np.maximum(mu, 1e-9)),
        family=kwargs.pop("family", "poisson"),
        n_params=kwargs.pop("n_params", 2),
        deviance=kwargs.pop("deviance", 1.0),
        exposure=kwargs.pop("exposure", np.ones_like(y)),
        weights=kwargs.pop("weights", None),
        base_mu=kwargs.pop("base_mu", None),
        **kwargs,
    )


class TestDiagnosticsUtils:
    def test_discretize_continuous_uses_quantile_bins_and_invalid_bin(self):
        values = np.array([0.0, 1.0, 2.0, 3.0, np.nan, np.inf])

        bins = discretize(values, n_bins=2)

        np.testing.assert_array_equal(bins, np.array([0, 0, 1, 1, 2, 2]))

    def test_discretize_all_invalid_continuous_values_go_to_zero(self):
        bins = discretize(np.array([np.nan, np.inf, -np.inf]), n_bins=3)

        np.testing.assert_array_equal(bins, np.array([0, 0, 0]))

    def test_discretize_categorical_assigns_sorted_levels_and_null_bin(self):
        values = np.array(["b", "a", None, "b"], dtype=object)

        bins = discretize(values, n_bins=10)

        np.testing.assert_array_equal(bins, np.array([1, 0, 2, 1]))

    def test_discretize_all_null_categorical_values_go_to_zero(self, monkeypatch):
        values = np.array([None, None], dtype=object)

        original_asarray = np.asarray

        def fail_float_coercion(candidate, dtype=None, *args, **kwargs):
            if dtype is np.float64:
                raise TypeError("force categorical fallback")
            return original_asarray(candidate, *args, dtype=dtype, **kwargs)

        monkeypatch.setattr("rustystats.diagnostics.utils.np.asarray", fail_float_coercion)

        bins = discretize(values, n_bins=10)

        np.testing.assert_array_equal(bins, np.array([0, 0]))

    def test_discretize_numeric_object_values_use_continuous_path(self):
        values = np.array(["1", "2", "3", "4"], dtype=object)

        bins = discretize(values, n_bins=2)

        np.testing.assert_array_equal(bins, np.array([0, 0, 1, 1]))

    def test_validate_factor_in_data_passes_and_raises_with_context(self):
        data = pl.DataFrame({"age": [1, 2, 3]})

        validate_factor_in_data("age", data, "Continuous factor")
        with pytest.raises(ValidationError, match="Continuous factor 'missing'"):
            validate_factor_in_data("missing", data, "Continuous factor")


class TestDiagnosticsComputerHelperContracts:
    def test_rank_and_gini_validation_fail_closed(self):
        with pytest.raises(ValidationError, match="ranking"):
            rank_sort_idx(np.array([1.0, 2.0]), ranking="other")
        with pytest.raises(ValidationError, match="requires exposure"):
            rank_sort_idx(np.array([1.0, 2.0]), ranking="rate")

        assert (
            _gini_for_arrays(
                np.zeros(2),
                np.ones(2),
                np.array([0, 1]),
            )
            == 0.0
        )
        assert (
            _gini_for_arrays(
                np.ones(2),
                np.zeros(2),
                np.array([0, 1]),
            )
            == 0.0
        )
        with pytest.raises(ValidationError, match="weights length"):
            _gini_for_arrays(np.ones(2), np.ones(2), np.array([0, 1]), weights=np.ones(3))

    def test_constructor_and_loss_validation_contracts(self):
        with pytest.raises(ValidationError, match="mu of length"):
            _basic_computer(mu=[1.0, 2.0])
        with pytest.raises(ValidationError, match="exposure of length"):
            _basic_computer(exposure=np.ones(3))
        with pytest.raises(ValidationError, match="weights of length"):
            _basic_computer(weights=np.ones(3))
        with pytest.raises(ValidationError, match="base_mu of length"):
            _basic_computer(base_mu=np.ones(3))

        computer = _basic_computer()
        assert computer._compute_loss(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == pytest.approx(
            0.0
        )

    def test_calibration_component_validation_and_empty_bins(self):
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.0, 2.0, 3.0])
        unit_exposure = np.ones_like(y)
        comp = components_mod._CalibrationComputer(y, mu, unit_exposure)

        with pytest.raises(ValidationError, match="requires exposure"):
            comp._compute_bins(3, ranking="rate")

        with pytest.raises(ValidationError, match="ranking"):
            comp._compute_bins(3, ranking="bad")

        assert comp._compute_bins(0) == []

        zero_exposure = components_mod._CalibrationComputer(
            y,
            mu,
            np.zeros_like(y),
        )
        assert zero_exposure._compute_bins(3) == []

    def test_generate_warnings_covers_negbinomial_and_factor_branches(self):
        computer = _basic_computer()
        missing_factor = SimpleNamespace(
            name="territory",
            in_model=False,
            residual_pattern=SimpleNamespace(var_explained=0.031),
        )

        warnings = computer.generate_warnings(
            {"dispersion": 1.8},
            {"ae_ratio": 0.9},
            [missing_factor],
            family="NegativeBinomial(theta=100.0)",
        )

        warning_types = {w["type"] for w in warnings}
        assert "negbinomial_regularization" in warning_types
        assert "negbinomial_large_theta" in warning_types
        assert "high_dispersion" in warning_types
        assert "poor_calibration" in warning_types
        assert "missing_factor" in warning_types

        small_theta = computer.generate_warnings(
            {"dispersion": 1.0},
            {"ae_ratio": 1.0},
            [],
            family="NegativeBinomial(theta=0.05)",
        )
        assert any(w["type"] == "negbinomial_small_theta" for w in small_theta)

        with pytest.raises(ValidationError, match="Failed to parse theta"):
            computer.generate_warnings(
                {"dispersion": 1.0},
                {"ae_ratio": 1.0},
                [],
                family="NegativeBinomial(theta=not-a-number)",
            )

    def test_vif_context_and_direct_vif_edge_contracts(self, monkeypatch):
        computer = _basic_computer()

        assert computer.compute_vif_from_correlation_context(SimpleNamespace(), ["x"]) is None

        single = computer.compute_vif_from_correlation_context(
            SimpleNamespace(matrix=np.array([[1.0]]), vif_values=np.array([1.0])),
            ["x"],
        )
        assert single is not None
        assert single[0].feature == "x"
        assert single[0].vif == 1.0

        no_intercept = computer.compute_vif_from_correlation_context(
            SimpleNamespace(
                matrix=np.array([[1.0, 0.9], [0.9, 1.0]]),
                vif_values=np.array([5.0, 5.0]),
            ),
            ["x1", "x2"],
            threshold_moderate=2.0,
            threshold_severe=10.0,
        )
        assert no_intercept is not None
        assert {v.feature for v in no_intercept} == {"x1", "x2"}

        stripped_context = computer.compute_vif_from_correlation_context(
            SimpleNamespace(
                matrix=np.array([[1.0, 0.25], [0.25, 1.0]]),
                vif_values=np.array([1.1, 1.1]),
            ),
            ["Intercept", "x1", "x2"],
        )
        assert stripped_context is not None
        assert [v.feature for v in stripped_context] == ["x1", "x2"]

        direct = computer.compute_vif(np.ones((4, 2)), ["Intercept", "x"])
        assert direct[0].feature == "x"
        assert direct[0].severity == "none"

        monkeypatch.setattr(
            "rustystats.diagnostics.computer._rust_correlation_and_vif",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(np.linalg.LinAlgError("rust boom")),
        )
        singular = np.array(
            [
                [1.0, 1.0, 2.0],
                [1.0, 2.0, 4.0],
                [1.0, 3.0, 6.0],
                [1.0, 4.0, 8.0],
            ]
        )
        with pytest.raises(DesignMatrixError, match="VIF computation failed"):
            computer.compute_vif(singular, ["Intercept", "x", "x2"])

    def test_base_prediction_comparison_validates_lengths_and_weights(self):
        computer = _basic_computer()
        y = np.array([1.0, 2.0, 3.0])
        mu_model = np.array([1.1, 2.1, 3.1])
        mu_base = np.array([0.9, 2.2, 2.8])
        exposure = np.ones(3)

        with pytest.raises(ValidationError, match="matching lengths"):
            computer.compute_base_predictions_comparison(y, mu_model[:-1], mu_base, exposure)
        with pytest.raises(ValidationError, match="weights length"):
            computer.compute_base_predictions_comparison(
                y,
                mu_model,
                mu_base,
                exposure,
                weights=np.ones(2),
            )

        comparison = computer.compute_base_predictions_comparison(
            y,
            mu_model,
            mu_base,
            exposure,
            n_deciles=3,
            weights=np.array([1.0, 2.0, 3.0]),
            ranking="mean",
        )

        assert comparison.model_metrics.total_actual == pytest.approx(14.0)
        assert len(comparison.model_vs_base_deciles) == 3
        assert comparison.loss_improvement_pct is not None

    def test_eta_contribution_handles_splines_expressions_and_silent_fallbacks(self):
        grid = np.array([1.0, 2.0, 3.0])

        class Spline:
            def transform(self, values):
                return np.column_stack([values + 1.0]), ["bs(x, 1/2)"]

        class Builder:
            def __init__(self):
                self._fitted_splines = {"x": Spline()}

            def _convert_expression_to_polars(self, expr):
                if "missing" in expr:
                    raise ValueError("bad expression")
                return pl.col("x") ** 2

        feature_names = [
            "Intercept",
            "x",
            "bs(x, 1/2)",
            "I(x ** 2)",
            "I(x * 0)",
            "I(x / missing)",
            "TE(x)",
        ]
        computer = _basic_computer(feature_names=feature_names, n_params=len(feature_names))
        result = SimpleNamespace(
            params=np.array([0.0, 1.0, 2.0, 0.5, 0.0, 1.0, 0.0]),
            _builder=Builder(),
        )
        feature_to_idx = {name: i for i, name in enumerate(feature_names)}

        eta = computer._compute_eta_contribution("x", grid, result, feature_to_idx)

        np.testing.assert_allclose(eta, grid + 2.0 * (grid + 1.0) + 0.5 * grid**2)

        missing_index = computer._compute_eta_contribution(
            "x",
            grid,
            SimpleNamespace(params=np.array([0.5]), _builder=Builder()),
            {},
        )
        np.testing.assert_allclose(missing_index, np.zeros_like(grid))

        no_builder = _basic_computer(feature_names=["I(x ** 2)"], n_params=1)
        no_builder_eta = no_builder._compute_eta_contribution(
            "x",
            grid,
            SimpleNamespace(params=np.array([0.5])),
            {"I(x ** 2)": 0},
        )
        np.testing.assert_allclose(no_builder_eta, np.zeros_like(grid))

        class BadSpline:
            def transform(self, _values):
                raise ValueError("cannot transform")

        bad_spline = _basic_computer(feature_names=["bs(x, 1/2)"], n_params=1)
        bad_eta = bad_spline._compute_eta_contribution(
            "x",
            grid,
            SimpleNamespace(
                params=np.array([1.0]),
                _builder=SimpleNamespace(_fitted_splines={"x": BadSpline()}),
            ),
            {"bs(x, 1/2)": 0},
        )
        np.testing.assert_allclose(bad_eta, np.zeros_like(grid))

    def test_partial_dependence_skips_missing_and_sparse_inputs(self):
        n = 12
        mu = np.linspace(1.0, 2.1, n)
        computer = _basic_computer(
            y=np.linspace(1.0, 2.0, n),
            mu=mu,
            exposure=np.ones(n),
            feature_names=["Intercept", "x"],
            n_params=2,
        )
        data = pl.DataFrame(
            {
                "x": np.arange(n, dtype=np.float64),
                "short": [1.0] * 9 + [np.nan] * 3,
                "cat": ["A"] * n,
                "one": ["Only"] * n,
            }
        )
        result = SimpleNamespace(
            params=np.array([0.0, 0.1]),
            _builder=SimpleNamespace(
                get_spline_info=lambda: {},
                _term_slots=[
                    SimpleNamespace(factors=["x"], term_type="linear"),
                    SimpleNamespace(factors=["cat"], term_type="categorical"),
                    SimpleNamespace(factors=["one"], term_type="categorical"),
                ],
            ),
        )

        partial = computer.compute_partial_dependence(
            data,
            result,
            continuous_factors=["missing", "short", "x"],
            categorical_factors=["missing_cat", "cat", "one"],
            link="identity",
            n_grid=5,
            cat_unique_cache={
                "cat": (np.array(["A", "B"]), np.zeros(n, dtype=np.uint32)),
            },
            cont_column_cache={"short": np.array([1.0] * 9 + [np.nan] * 3)},
        )

        by_name = {item.variable: item for item in partial}
        assert set(by_name) == {"x", "cat", "one"}
        assert by_name["x"].term_type == "linear"
        assert by_name["cat"].grid_values == ["A", "B"]
        assert by_name["one"].shape == "single_level"

    def test_factor_deviance_cache_routes_and_empty_inputs(self):
        computer = _basic_computer()
        data = pl.DataFrame({"region": ["A", "B", "A", "B"]})

        assert computer.compute_factor_deviance(data, ["missing"]) == []

        with_codes = computer.compute_factor_deviance(
            data,
            ["missing", "region"],
            cat_unique_cache={
                "region": (
                    np.array(["A", "B"]),
                    np.array([0, 1, 0, 1], dtype=np.uint32),
                )
            },
        )
        assert with_codes[0].factor == "region"

        slow_path = computer.compute_factor_deviance(
            data,
            ["region"],
            cat_column_cache={"region": np.array(["A", "B", "A", "B"])},
            cat_unique_cache={"region": None},
        )
        assert slow_path[0].factor == "region"

    def test_dataset_and_band_metric_helpers_validate_and_handle_empty_bins(self, monkeypatch):
        n = 12
        y = np.linspace(1.0, 2.1, n)
        mu = y + 0.1
        exposure = np.ones(n)
        computer = _basic_computer(y=y, mu=mu, exposure=exposure, n_params=2)
        data = pl.DataFrame({"region": ["A", "B"] * 6, "empty": [np.nan] * n, "x": y})

        with pytest.raises(ValidationError, match="weights length"):
            computer.compute_dataset_diagnostics(
                y,
                mu,
                exposure,
                data,
                [],
                [],
                "test",
                weights=np.ones(2),
            )
        with pytest.raises(ValidationError, match="base_mu length"):
            computer.compute_dataset_diagnostics(
                y,
                mu,
                exposure,
                data,
                [],
                [],
                "test",
                base_mu=np.ones(2),
            )

        weighted = computer.compute_dataset_diagnostics(
            y,
            mu,
            exposure,
            data,
            [],
            [],
            "test",
            weights=np.linspace(1.0, 2.0, n),
        )
        assert weighted.n_obs == n

        monkeypatch.setattr(
            computer_mod,
            "_rust_ae_categorical_batch",
            lambda *_args, **_kwargs: [
                [
                    {
                        "count": 0,
                        "bin_label": "zero",
                        "exposure": 0.0,
                        "actual_sum": 0.0,
                        "predicted_sum": 0.0,
                        "base_sum": None,
                    },
                    {
                        "count": 2,
                        "bin_label": "A",
                        "exposure": 2.0,
                        "actual_sum": 3.0,
                        "predicted_sum": 2.5,
                        "base_sum": None,
                    },
                ]
            ],
        )
        factor_metrics = computer._compute_factor_level_metrics(
            y[:4],
            mu[:4],
            exposure[:4],
            pl.DataFrame({"region": ["A", "B", "A", "B"]}),
            ["missing", "region"],
            deviance_residuals=None,
            cat_unique_cache=None,
        )
        assert factor_metrics["region"][0].level == "A"
        assert factor_metrics["region"][0].residual_mean == 0.0

        monkeypatch.setattr(
            computer_mod,
            "_rust_ae_continuous_batch",
            lambda *_args, **_kwargs: [
                [],
                [
                    {
                        "count": 0,
                        "bin_lower": 0.0,
                        "bin_upper": 1.0,
                        "exposure": 0.0,
                        "actual_sum": 0.0,
                        "predicted_sum": 0.0,
                        "weight_sum": 0.0,
                        "base_sum": None,
                    },
                    {
                        "count": 2,
                        "bin_lower": 1.0,
                        "bin_upper": 3.0,
                        "exposure": 2.0,
                        "actual_sum": 3.0,
                        "predicted_sum": 2.5,
                        "weight_sum": 2.0,
                        "base_sum": None,
                    },
                ],
            ],
        )
        continuous_metrics = computer._compute_continuous_band_metrics(
            y[:4],
            mu[:4],
            exposure[:4],
            pl.DataFrame({"empty": [np.nan] * 4, "x": [1.0, 2.0, 3.0, 4.0]}),
            ["missing", "empty", "x"],
            deviance_residuals=None,
            n_bands=2,
            cont_column_cache=None,
        )
        assert continuous_metrics["empty"] == []
        assert continuous_metrics["x"][0].band == 2

    def test_train_test_comparison_flags_and_divergence_from_dataset_contracts(self, monkeypatch):
        computer = _basic_computer(weights=np.ones(4))
        train_data = pl.DataFrame({"region": ["A", "B", "A", "B"]})
        test_data = pl.DataFrame({"region": ["A", "B", "A", "B"]})

        def fake_dataset(
            _y,
            _mu,
            _exposure,
            _data,
            _categorical_factors,
            _continuous_factors,
            dataset_name,
            _result,
            **kwargs,
        ):
            if dataset_name == "train":
                assert kwargs["weights"] is not None
                return SimpleNamespace(
                    gini=0.20,
                    ae_ratio=1.0,
                    ae_by_decile=[SimpleNamespace(ae_ratio=1.0), SimpleNamespace(ae_ratio=1.2)],
                    factor_diagnostics={
                        "region": [
                            SimpleNamespace(level="A", ae_ratio=1.0),
                            SimpleNamespace(level="B", ae_ratio=1.25),
                        ]
                    },
                )
            assert kwargs.get("weights") is None
            return SimpleNamespace(
                gini=0.10,
                ae_ratio=1.12,
                ae_by_decile=[SimpleNamespace(ae_ratio=0.9), SimpleNamespace(ae_ratio=0.8)],
                factor_diagnostics={
                    "region": [
                        SimpleNamespace(level="A", ae_ratio=1.25),
                        SimpleNamespace(level="C", ae_ratio=0.8),
                    ]
                },
            )

        monkeypatch.setattr(computer, "compute_dataset_diagnostics", fake_dataset)

        comparison = computer.compute_train_test_comparison(
            train_data,
            test_data,
            y_train=np.ones(4),
            mu_train=np.ones(4),
            exposure_train=np.ones(4),
            y_test=np.ones(4),
            mu_test=np.ones(4),
            exposure_test=np.ones(4),
            categorical_factors=["region"],
            continuous_factors=[],
            result=SimpleNamespace(),
        )

        assert comparison.gini_gap == pytest.approx(0.1)
        assert comparison.overfitting_risk is True
        assert comparison.calibration_drift is True
        assert comparison.decile_comparison[0]["ae_diff"] == pytest.approx(0.1)
        assert comparison.factor_divergence["region"][0]["level"] in {"A", "B"}
        assert comparison.unstable_factors


class TestInteractionDetectorContracts:
    def test_detect_interactions_skips_sparse_continuous_inputs_and_missing_factors(self):
        detector = _InteractionDetector(
            pearson_residuals=np.linspace(-1.0, 1.0, 12),
            feature_names=["x", "z"],
        )
        data = pl.DataFrame(
            {
                "x": [1.0, 2.0, np.nan, np.inf, 3.0, 4.0, 5.0, 6.0, np.nan, 7.0, 8.0, 9.0],
                "z": np.linspace(0.0, 1.0, 12),
            }
        )

        assert detector.detect_interactions(data, ["x", "z"], min_correlation=2.0) == []
        assert (
            detector.detect_interactions(
                data,
                ["x", "z"],
                min_correlation=2.0,
                cont_column_cache={"x": data["x"].to_numpy(), "z": data["z"].to_numpy()},
            )
            == []
        )

        with pytest.raises(ValidationError, match="Factor 'missing' not found"):
            detector.detect_interactions(data, ["missing"])

    def test_interaction_recommendations_cover_factor_type_combinations(self):
        detector = _InteractionDetector(np.arange(12, dtype=float), feature_names=[])
        cat = np.array(["A", "B", "A"], dtype=object)
        cont = np.array([1.0, 2.0, 3.0])

        assert detector._generate_interaction_recommendation(
            "brand", "region", [], [], cat, cat
        ) == ("Consider C(brand):C(region) interaction term")
        assert detector._generate_interaction_recommendation(
            "brand", "age", [], ["bs(age)"], cat, cont
        ) == ("Consider C(brand):age or separate splines by brand level")
        assert detector._generate_interaction_recommendation("brand", "age", [], [], cat, cont) == (
            "Consider C(brand):age interaction term"
        )
        assert detector._generate_interaction_recommendation(
            "age", "brand", ["s(age)"], [], cont, cat
        ) == ("Consider age:C(brand) or separate splines by brand level")
        assert detector._generate_interaction_recommendation("age", "brand", [], [], cont, cat) == (
            "Consider age:C(brand) interaction term"
        )
        assert detector._generate_interaction_recommendation(
            "age", "score", ["I(age ** 2)"], [], cont, cont
        ) == ("Consider age:score or tensor product spline")
        assert detector._generate_interaction_recommendation(
            "age", "score", [], [], cont, cont
        ) == ("Consider age:score interaction or joint spline")

    def test_eta_and_interaction_strength_degenerate_cases_fail_closed(self):
        constant_detector = _InteractionDetector(np.ones(8), feature_names=[])
        assert constant_detector._compute_eta_squared(np.array(["A", "B", "A", "B"])) == 0.0
        assert (
            constant_detector._compute_interaction_strength(
                "a",
                np.array([0, 0, 1, 1, 0, 0, 1, 1]),
                "b",
                np.array([0, 0, 0, 0, 1, 1, 1, 1]),
                min_cell_count=2,
            )
            is None
        )

        detector = _InteractionDetector(np.array([0.0, 1.0, 0.0, 1.0]), feature_names=[])
        assert (
            detector._compute_interaction_strength(
                "a",
                np.array([0, 1, 2, 3]),
                "b",
                np.array([0, 0, 0, 0]),
                min_cell_count=2,
            )
            is None
        )

        saturated = detector._compute_interaction_strength(
            "a",
            np.array([0, 1, 2, 3]),
            "b",
            np.array([0, 0, 0, 0]),
            min_cell_count=1,
        )
        assert saturated is not None
        assert saturated.n_cells == 4
        assert math.isnan(saturated.pvalue)


class TestDiagnosticsApiHelperContracts:
    def test_metadata_family_and_spline_helpers_fail_closed(self):
        assert diagnostics_api._compute_smooth_term_diagnostics(SimpleNamespace(), []) == []
        assert diagnostics_api._parse_family_params("Tweedie(p=1.7)") == (1.7, 1.0)
        assert diagnostics_api._parse_family_params("NegativeBinomial(theta=2.5)") == (1.5, 2.5)
        assert diagnostics_api._resolve_null_deviance(SimpleNamespace()) is None
        assert (
            diagnostics_api._resolve_null_deviance(SimpleNamespace(null_deviance=lambda: 12.3))
            == 12.3
        )
        assert diagnostics_api._extract_spline_info(SimpleNamespace()) is None
        assert (
            diagnostics_api._extract_spline_info(
                SimpleNamespace(_builder=SimpleNamespace(get_spline_info=lambda: []))
            )
            is None
        )

        with pytest.raises(ValidationError, match="missing 'family'"):
            diagnostics_api._extract_model_metadata(SimpleNamespace())
        with pytest.raises(ValidationError, match="missing 'link'"):
            diagnostics_api._extract_model_metadata(SimpleNamespace(family="poisson"))
        with pytest.raises(ValidationError, match="missing 'feature_names'"):
            diagnostics_api._extract_model_metadata(SimpleNamespace(family="poisson", link="log"))
        assert diagnostics_api._extract_model_metadata(
            SimpleNamespace(
                family="poisson",
                link="log",
                feature_names=["Intercept", "x"],
                params=np.array([0.1, 0.2]),
                deviance=3.4,
            )
        ) == ("poisson", "log", 2, 3.4, ["Intercept", "x"])

        with pytest.raises(ValidationError, match="train_data has 1 rows"):
            diagnostics_api._validate_data_length(pl.DataFrame({"x": [1.0]}), np.ones(2))

    def test_smooth_term_diagnostics_cover_covariance_sources_and_fallback(self, monkeypatch):
        term = SimpleNamespace(
            variable="age",
            k=5,
            edf=4.0,
            lambda_=0.2,
            gcv=1.1,
            col_start=0,
            col_end=2,
        )

        warnings: list[dict[str, str]] = []
        insignificant = diagnostics_api._compute_smooth_term_diagnostics(
            SimpleNamespace(
                smooth_terms=[term],
                params=np.array([0.01, -0.01]),
                get_bread_matrix=lambda: np.eye(2) * 1_000.0,
            ),
            warnings,
        )
        assert insignificant[0].p_value > 0.05
        assert any(w["type"] == "insignificant_smooth" for w in warnings)

        warnings = []
        undersmoothed = diagnostics_api._compute_smooth_term_diagnostics(
            SimpleNamespace(
                smooth_terms=[term],
                params=np.array([2.0, -1.0]),
                _result=SimpleNamespace(cov_params_unscaled=np.eye(2) * 0.001),
            ),
            warnings,
        )
        assert undersmoothed[0].chi2 > 0.0
        assert any(w["type"] == "undersmoothed" for w in warnings)

        for cov_attr in ("covariance_unscaled", "cov_params"):
            result = SimpleNamespace(smooth_terms=[term], params=np.array([0.5, -0.25]))
            if cov_attr == "covariance_unscaled":
                result._result = SimpleNamespace(covariance_unscaled=np.eye(2))
            else:
                result.cov_params = lambda: np.eye(2)
            assert diagnostics_api._compute_smooth_term_diagnostics(result, [])[0].chi2 > 0.0

        def raise_value_error(_matrix):
            raise ValueError("singular")

        monkeypatch.setattr(np.linalg, "pinv", raise_value_error)
        warnings = []
        fallback = diagnostics_api._compute_smooth_term_diagnostics(
            SimpleNamespace(
                smooth_terms=[term],
                params=np.array([1.0, 2.0]),
                _result=SimpleNamespace(cov_params_unscaled=np.eye(2)),
            ),
            warnings,
        )
        assert fallback[0].chi2 == pytest.approx(5.0)
        assert fallback[0].ref_df == pytest.approx(2.0)
        assert any(w["type"] == "smooth_significance_fallback" for w in warnings)

    def test_prediction_and_test_array_helpers_validate_exposure_contracts(self):
        train = pl.DataFrame({"y": [1.0, 2.0], "expo": [2.0, 4.0]})
        fallback_result = SimpleNamespace(
            formula="missing ~ x",
            fittedvalues=np.array([1.1, 1.8]),
            resid_response=lambda: np.array([-0.1, 0.2]),
            linear_predictor=np.array([0.1, 0.2]),
        )

        y, mu, lp = diagnostics_api._extract_response_and_predictions(fallback_result, train)

        np.testing.assert_allclose(y, [1.0, 2.0])
        np.testing.assert_allclose(mu, [1.1, 1.8])
        np.testing.assert_allclose(lp, [0.1, 0.2])

        test = pl.DataFrame({"y": [3.0, 4.0], "expo": [1.5, 2.5]})
        assert diagnostics_api._extract_test_arrays(None, fallback_result, "y", None) == (
            None,
            None,
            None,
            None,
        )
        assert diagnostics_api._extract_test_arrays(test, fallback_result, None, None) == (
            None,
            None,
            None,
            None,
        )
        with pytest.raises(ValidationError, match="Response column"):
            diagnostics_api._extract_test_arrays(test, fallback_result, "missing", None)
        with pytest.raises(ValidationError, match="does not support prediction"):
            diagnostics_api._extract_test_arrays(test, SimpleNamespace(), "y", None)
        with pytest.raises(ValidationError, match="one-dimensional"):
            diagnostics_api._extract_test_arrays(
                test,
                SimpleNamespace(predict=lambda *_args, **_kwargs: np.ones(2)),
                "y",
                None,
                exposure_override=np.ones((2, 1)),
            )
        with pytest.raises(ValidationError, match="array exposure require"):
            diagnostics_api._extract_test_arrays(
                test,
                SimpleNamespace(
                    _exposure_spec=np.ones(2),
                    predict=lambda *_args, **_kwargs: np.ones(2),
                ),
                "y",
                None,
            )

        calls: list[Any] = []

        def predict(_frame, exposure=None):
            calls.append(exposure)
            return np.array([2.9, 4.1])

        model = SimpleNamespace(predict=predict, _exposure_spec=None)
        y_test, mu_test, exposure_test, sort_idx = diagnostics_api._extract_test_arrays(
            test,
            model,
            "y",
            None,
            exposure_override="expo",
        )

        np.testing.assert_allclose(y_test, [3.0, 4.0])
        np.testing.assert_allclose(mu_test, [2.9, 4.1])
        np.testing.assert_allclose(exposure_test, [1.5, 2.5])
        assert sort_idx.shape == (2,)
        assert calls[-1] == "expo"

        override = np.array([10.0, 20.0])
        _, _, exposure_test, _ = diagnostics_api._extract_test_arrays(
            test,
            model,
            "y",
            None,
            exposure_override=override,
        )
        np.testing.assert_allclose(calls[-1], override)
        np.testing.assert_allclose(exposure_test, override)

        _, _, exposure_test, _ = diagnostics_api._extract_test_arrays(
            test,
            model,
            "y",
            None,
            exposure_override=np.array([99.0]),
        )
        assert calls[-1] is None
        np.testing.assert_allclose(exposure_test, np.ones(2))

    def test_resolution_helpers_cover_exposure_base_predictions_and_weights(self):
        train = pl.DataFrame(
            {
                "y": [1.0, 2.0, 3.0],
                "expo": [1.0, 1.5, 2.0],
                "w": [0.5, 1.0, 2.0],
                "base": [1.1, 2.1, 3.1],
            }
        )
        result = SimpleNamespace(formula="y ~ x", _exposure_spec="expo")

        response_col, exposure_col, exposure = diagnostics_api._resolve_offset_and_response(
            result, train
        )

        assert response_col == "y"
        assert exposure_col == "expo"
        np.testing.assert_allclose(exposure, [1.0, 1.5, 2.0])

        with pytest.raises(ValidationError, match="array exposure"):
            diagnostics_api._resolve_offset_and_response(
                SimpleNamespace(_array_exposure_requires_prediction_override=True),
                train,
            )
        with pytest.raises(ValidationError, match="one-dimensional"):
            diagnostics_api._resolve_offset_and_response(
                result, train, exposure_override=np.ones((3, 1))
            )
        with pytest.raises(ValidationError, match="Stored exposure length"):
            diagnostics_api._resolve_offset_and_response(
                result, train, exposure_override=np.ones(2)
            )

        assert diagnostics_api._resolve_base_predictions_column(None, "train") is None
        assert diagnostics_api._resolve_base_predictions_column("base", "test") == "base"
        assert (
            diagnostics_api._resolve_base_predictions_column({"test": "base_test"}, "train") is None
        )
        with pytest.raises(ValidationError, match="base_predictions must be"):
            diagnostics_api._resolve_base_predictions_column(123, "train")

        warnings: list[dict[str, str]] = []
        assert (
            diagnostics_api._extract_base_predictions_array(
                None, "train", train, 3, warnings, False
            )
            is None
        )
        assert (
            diagnostics_api._extract_base_predictions_array(
                {"train": None}, "train", train, 3, warnings, False
            )
            is None
        )
        assert (
            diagnostics_api._extract_base_predictions_array(
                "missing", "train", train, 3, warnings, False
            )
            is None
        )
        assert warnings[-1]["type"] == "base_predictions_unavailable"
        with pytest.raises(ValidationError, match="not found"):
            diagnostics_api._extract_base_predictions_array(
                "missing", "train", train, 3, warnings, True
            )
        with pytest.raises(ValidationError, match="rows but expected"):
            diagnostics_api._extract_base_predictions_array(
                "base", "train", train, 2, warnings, True
            )
        np.testing.assert_allclose(
            diagnostics_api._extract_base_predictions_array(
                "base", "train", train, 3, warnings, True
            ),
            [1.1, 2.1, 3.1],
        )

        assert (
            diagnostics_api._resolve_weights(SimpleNamespace(_weights_spec=None), train, None)
            is None
        )
        with pytest.raises(ValidationError, match="weights column"):
            diagnostics_api._resolve_weights(SimpleNamespace(), train, "missing")
        assert (
            diagnostics_api._resolve_weights(SimpleNamespace(_weights_spec="missing"), train, None)
            is None
        )
        np.testing.assert_allclose(
            diagnostics_api._resolve_weights(SimpleNamespace(_weights_spec="w"), train, None),
            [0.5, 1.0, 2.0],
        )
        with pytest.raises(ValidationError, match="weights length"):
            diagnostics_api._resolve_weights(SimpleNamespace(), train, np.ones(2))
        assert (
            diagnostics_api._resolve_weights(SimpleNamespace(_weights_spec=np.ones(2)), train, None)
            is None
        )
        np.testing.assert_allclose(
            diagnostics_api._resolve_weights(SimpleNamespace(), train, np.array([1.0, 2.0, 3.0])),
            [1.0, 2.0, 3.0],
        )

        test_warnings: list[dict[str, str]] = []
        assert (
            diagnostics_api._resolve_test_weights(SimpleNamespace(), None, "w", test_warnings)
            is None
        )
        with pytest.raises(ValidationError, match="weights column"):
            diagnostics_api._resolve_test_weights(
                SimpleNamespace(), train, "missing", test_warnings
            )
        assert (
            diagnostics_api._resolve_test_weights(
                SimpleNamespace(_weights_spec="missing"), train, None, test_warnings
            )
            is None
        )
        assert test_warnings[-1]["type"] == "test_weights_unavailable"
        assert (
            diagnostics_api._resolve_test_weights(
                SimpleNamespace(_weights_spec=np.ones(3)), train, None, test_warnings
            )
            is None
        )
        assert "Array weights" in test_warnings[-1]["message"]

    def test_encoding_and_interaction_spec_helpers(self):
        train = pl.DataFrame(
            {
                "region": ["A", "B", "A"],
                "brand": ["X", "X", "Y"],
                "region_grp": ["north", "south", "north"],
            }
        )
        test = pl.DataFrame(
            {
                "region": ["A", "C"],
                "brand": ["X", "Z"],
                "region_grp": ["north", "east"],
            }
        )
        slot = SimpleNamespace(
            term_name="region",
            term_type="categorical",
            factors=["region"],
            design_column_names=["region[B]"],
        )

        assert diagnostics_api._level_set(None, ["region"]) == set()
        assert diagnostics_api._level_set(train, ["missing"]) == set()
        assert diagnostics_api._level_set(train, ["region"]) == {"A", "B"}
        assert diagnostics_api._level_set(train, ["brand", "region"]) == {"X:A", "X:B", "Y:A"}
        assert diagnostics_api._encoding_kind_from_slot(slot) == "categorical"
        assert (
            diagnostics_api._encoding_kind_from_slot(SimpleNamespace(term_type="target_encoding"))
            == "target_encoding"
        )
        assert (
            diagnostics_api._encoding_kind_from_slot(
                SimpleNamespace(term_type="frequency_encoding")
            )
            == "frequency_encoding"
        )
        assert (
            diagnostics_api._encoding_kind_from_slot(SimpleNamespace(term_type="linear"))
            == "unknown"
        )
        notes: list[str] = []
        assert (
            diagnostics_api._maybe_grouped_kind("region_grp", "unknown", notes)
            == "grouped_categorical"
        )
        assert notes

        entries = diagnostics_api._compute_encoding_diagnostics(
            SimpleNamespace(_builder=SimpleNamespace(_term_slots=[slot])),
            train,
            test,
            ["region", "region_grp"],
        )

        assert entries is not None
        assert entries[0].name == "region"
        assert entries[0].unseen_levels_test == 1
        assert entries[1].name == "region_grp"
        assert entries[1].kind == "grouped_categorical"

        assert diagnostics_api._extract_interaction_spec_factors(
            {"factor1": "a", "factor2": "b", "factor3": "c"}
        ) == ["a", "b", "c"]
        assert diagnostics_api._extract_interaction_spec_factors(
            {"a": {"type": "linear"}, "b": {"type": "linear"}}
        ) == ["a", "b"]
        assert diagnostics_api._extract_interaction_spec_factors(("a", "b")) == ["a", "b"]
        assert diagnostics_api._extract_interaction_spec_factors("a:b") == []
        pairs, blocks = diagnostics_api._split_interaction_specs(
            [("a", "b"), ("a", "b", "c"), "bad"]
        )
        assert pairs == [("a", "b"), "bad"]
        assert blocks == [["a", "b", "c"]]

    def test_interaction_slot_and_model_summary_helpers(self):
        main = SimpleNamespace(
            term_name="age",
            term_type="linear",
            factors=["age"],
            col_start=1,
            col_end=2,
            design_column_names=["age"],
        )
        inter = SimpleNamespace(
            term_name="age:region",
            term_type="interaction",
            factors=["age", "region"],
            col_start=1,
            col_end=3,
            design_column_names=["age:region[B]", "age:region[C]"],
        )
        builder = SimpleNamespace(_term_slots=[main, inter])
        model = SimpleNamespace(
            _builder=builder,
            _interactions_spec=[("age", "region")],
            params=np.array([0.1, 0.2, -0.3]),
            bse=lambda: np.array([0.01, 0.1, 0.2]),
            pvalues=lambda: np.array([0.0, 0.04, 0.2]),
        )

        assert diagnostics_api._find_main_effect_slot(SimpleNamespace(), "age") is None
        assert (
            diagnostics_api._find_main_effect_slot(
                SimpleNamespace(_builder=SimpleNamespace()), "age"
            )
            is None
        )
        assert diagnostics_api._find_main_effect_slot(model, "age") is main
        assert diagnostics_api._find_interaction_slot(SimpleNamespace(), ["age", "region"]) is None
        assert diagnostics_api._find_interaction_slot(model, ["region", "age"]) is inter
        assert diagnostics_api._interaction_representation(None) is None
        assert diagnostics_api._interaction_representation(inter) == "tensor_product"
        assert (
            diagnostics_api._interaction_representation(
                SimpleNamespace(term_type="target_encoding")
            )
            == "target_encoding"
        )
        assert (
            diagnostics_api._interaction_representation(
                SimpleNamespace(term_type="frequency_encoding")
            )
            == "frequency_encoding"
        )
        assert (
            diagnostics_api._interaction_representation(SimpleNamespace(term_type="other"))
            == "unknown"
        )

        coeffs = diagnostics_api._coefficients_for_slot(model, inter)
        assert coeffs is not None
        assert [c.term for c in coeffs] == ["age:region[B]", "age:region[C]"]
        assert coeffs[0].z_value == pytest.approx(2.0)
        assert diagnostics_api._coefficients_for_slot(SimpleNamespace(), inter) is None
        assert (
            diagnostics_api._significance_for_slot(
                model, SimpleNamespace(col_start=1, col_end=1), None
            )
            is None
        )
        sig = diagnostics_api._significance_for_slot(model, inter, np.eye(3))
        assert sig is not None
        assert sig.chi2 > 0.0
        assert (
            diagnostics_api._significance_for_slot(SimpleNamespace(params=[]), inter, None) is None
        )
        fallback_sig = diagnostics_api._significance_for_slot(model, inter, None)
        assert fallback_sig is not None
        assert fallback_sig.chi2 > 0.0

        assert diagnostics_api._resolve_diagnostics_interactions(model, None, False) is None
        explicit = [("x", "z")]
        assert (
            diagnostics_api._resolve_diagnostics_interactions(
                SimpleNamespace(_interactions_spec=[]), explicit, True
            )
            is explicit
        )
        assert diagnostics_api._resolve_diagnostics_interactions(model, None, True) == [
            ("age", "region")
        ]
        assert diagnostics_api._resolve_diagnostics_interactions(model, [("x", "z")], True) == [
            ("x", "z"),
            ("age", "region"),
        ]

        train = pl.DataFrame({"x": [1, 2], "raw": [10, 20]})
        test = pl.DataFrame({"x": [3], "raw": [30]})

        def prepare_input(frame):
            return frame.with_columns((pl.col("raw") * 2).alias("derived"))

        prepared_train, prepared_test = diagnostics_api._interaction_diagnostics_data(
            SimpleNamespace(_input_transforms=[{"output": "derived"}], prepare_input=prepare_input),
            train,
            test,
            [("derived", "x")],
        )
        assert "derived" in prepared_train.columns
        assert prepared_test is not None
        assert "derived" in prepared_test.columns
        same_train = train.with_columns((pl.col("raw") * 2).alias("derived"))
        same_test = test.with_columns((pl.col("raw") * 2).alias("derived"))
        returned_train, returned_test = diagnostics_api._interaction_diagnostics_data(
            SimpleNamespace(_input_transforms=[{"output": "derived"}], prepare_input=prepare_input),
            same_train,
            same_test,
            [("derived", "x")],
        )
        assert returned_train is same_train
        assert returned_test is same_test
        returned_train, returned_test = diagnostics_api._interaction_diagnostics_data(
            SimpleNamespace(_input_transforms=[{"output": "derived"}], prepare_input=prepare_input),
            train,
            test,
            [("raw", "x")],
        )
        assert returned_train is train
        assert returned_test is test

        warnings: list[dict[str, str]] = []
        blocks = diagnostics_api._compute_interaction_block_diagnostics(
            [["missing", "region", "brand"]],
            model,
            bread_matrix=None,
            correlation_matrix=None,
            warnings=warnings,
        )
        assert blocks[0].in_model is False
        assert warnings[-1]["type"] == "interaction_score_test_unavailable"

        result = SimpleNamespace(
            formula="y ~ x",
            converged=True,
            iterations=5,
            alpha=0.2,
            l1_ratio=1.0,
            cv_deviance=1.23456,
            n_cv_folds=3,
            cv_selection_method="min",
            boundary_active_coefficients=lambda: [{"feature": "x"}],
            scale=lambda: 1.25,
            scale_pearson=lambda: 1.5,
        )
        summary = diagnostics_api._build_model_summary(
            result,
            family="poisson",
            link="log",
            computer=SimpleNamespace(n_obs=100, df_resid=98),
            n_params=2,
            null_deviance=12.345,
            robust_se_enriched=True,
        )
        assert summary["regularization"]["type"] == "lasso"
        assert summary["regularization"]["cv_folds"] == 3
        assert summary["robust_se_type"] == "HC1"
        assert summary["boundary_active_coefficients"] == [{"feature": "x"}]

        with pytest.raises(ValidationError, match="missing 'converged'"):
            diagnostics_api._build_model_summary(
                SimpleNamespace(),
                "poisson",
                "log",
                SimpleNamespace(n_obs=1, df_resid=0),
                1,
                None,
                False,
            )
        with pytest.raises(ValidationError, match="missing 'iterations'"):
            diagnostics_api._build_model_summary(
                SimpleNamespace(converged=True),
                "poisson",
                "log",
                SimpleNamespace(n_obs=1, df_resid=0),
                1,
                None,
                False,
            )
        with pytest.raises(ValidationError, match="missing 'formula'"):
            diagnostics_api._build_model_summary(
                SimpleNamespace(converged=True, iterations=1),
                "poisson",
                "log",
                SimpleNamespace(n_obs=1, df_resid=0),
                1,
                None,
                False,
            )

        ridge_summary = diagnostics_api._build_model_summary(
            SimpleNamespace(
                formula="y ~ x",
                converged=True,
                iterations=3,
                alpha=0.4,
                l1_ratio=0.0,
                regularization_type=None,
                boundary_active_coefficients=[{"feature": "z"}],
            ),
            "poisson",
            "log",
            SimpleNamespace(n_obs=10, df_resid=8),
            2,
            None,
            False,
        )
        assert ridge_summary["regularization"]["type"] == "ridge"
        assert ridge_summary["scale"] is None
        assert ridge_summary["boundary_active_coefficients"] == [{"feature": "z"}]

    def test_design_correlation_context_fallback_edges(self, monkeypatch):
        monkeypatch.setattr(diagnostics_api, "_iter_design_matrix_chunks", lambda *_args: None)
        assert (
            diagnostics_api._build_design_correlation_context_from_model(
                SimpleNamespace(),
                pl.DataFrame({"x": [1.0]}),
                include_inverse=False,
            )
            is None
        )

        monkeypatch.setattr(diagnostics_api, "_iter_design_matrix_chunks", lambda *_args: iter([]))
        assert (
            diagnostics_api._build_design_correlation_context_from_model(
                SimpleNamespace(),
                pl.DataFrame({"x": [1.0]}),
                include_inverse=False,
            )
            is None
        )

        monkeypatch.setattr(
            diagnostics_api,
            "_iter_design_matrix_chunks",
            lambda *_args: iter([np.ones((2, 2)), np.ones((2, 3))]),
        )
        assert (
            diagnostics_api._build_design_correlation_context_from_model(
                SimpleNamespace(),
                pl.DataFrame({"x": [1.0, 2.0]}),
                include_inverse=False,
            )
            is None
        )

    def test_significance_handles_linear_algebra_failure(self, monkeypatch):
        model = SimpleNamespace(
            params=np.array([0.1, 0.2, 0.3]),
            bse=lambda: np.array([0.1, 0.1, 0.1]),
        )
        slot = SimpleNamespace(col_start=1, col_end=3)
        monkeypatch.setattr(
            np.linalg,
            "pinv",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(np.linalg.LinAlgError("pinv")),
        )

        assert diagnostics_api._significance_for_slot(model, slot, np.eye(3)) is None

    def test_score_vif_and_pair_api_helpers_reuse_available_contexts(self):
        assert not diagnostics_api._score_tests_need_design_matrix(
            SimpleNamespace(),
            SimpleNamespace(),
            ["region"],
            [],
            compute_score_tests=False,
        )
        assert diagnostics_api._score_tests_need_design_matrix(
            SimpleNamespace(),
            SimpleNamespace(),
            ["region"],
            [],
            compute_score_tests=True,
        )

        refreshed: list[Any] = []
        factor_computer = SimpleNamespace(
            _refresh_transform_source_aliases=lambda result: refreshed.append(result)
        )
        assert diagnostics_api._score_tests_need_design_matrix(
            SimpleNamespace(_factors=factor_computer),
            SimpleNamespace(label="model"),
            ["region"],
            [],
            compute_score_tests=True,
        )
        assert refreshed == [SimpleNamespace(label="model")]

        factor_computer._get_feature_for = lambda name: SimpleNamespace(
            indices=[1] if name == "region" else []
        )
        assert diagnostics_api._score_tests_need_design_matrix(
            SimpleNamespace(_factors=factor_computer),
            SimpleNamespace(),
            ["region"],
            ["age"],
            compute_score_tests=True,
        )

        assert (
            diagnostics_api._maybe_compute_interactions(
                True,
                SimpleNamespace(),
                pl.DataFrame({"x": [1.0]}),
                ["region"],
                [],
                2,
                {},
                {},
            )
            == []
        )

        vif_computer = SimpleNamespace(
            compute_vif_from_correlation_context=lambda _ctx, _names: ["ctx"],
            compute_vif=lambda _matrix, _names: ["matrix"],
        )
        assert diagnostics_api._maybe_compute_vif(False, vif_computer, np.eye(2), ["x"]) is None
        assert diagnostics_api._maybe_compute_vif(
            True, vif_computer, None, ["x"], correlation_context=object()
        ) == ["ctx"]

        fallback_vif_computer = SimpleNamespace(
            compute_vif_from_correlation_context=lambda _ctx, _names: None,
            compute_vif=lambda _matrix, _names: ["matrix"],
        )
        assert (
            diagnostics_api._maybe_compute_vif(
                True, fallback_vif_computer, None, ["x"], correlation_context=object()
            )
            is None
        )
        assert diagnostics_api._maybe_compute_vif(
            True, fallback_vif_computer, np.eye(2), ["x"]
        ) == ["matrix"]

        captured: dict[str, Any] = {}

        class PairComputer:
            def compute_pair_diagnostics(self, **kwargs):
                captured.clear()
                captured.update(kwargs)
                return ["ok"]

        test_data = pl.DataFrame({"y": [1.0, 2.0], "expo": [1.5, 2.5]})
        model = SimpleNamespace(
            params=np.array([0.0, 0.1]),
            predict=lambda frame: np.asarray(frame["y"], dtype=np.float64) + 0.5,
        )

        assert diagnostics_api._compute_pair_diagnostics(
            PairComputer(),
            [("age", "region")],
            pl.DataFrame({"age": [1.0, 2.0], "region": ["A", "B"]}),
            model,
            response_col="y",
            exposure_col="expo",
            score_test_design_matrix=None,
            score_test_bread_matrix=None,
            test_data=test_data,
            link="log",
        ) == ["ok"]
        assert captured["params"] is None
        assert captured["bse"] is None
        np.testing.assert_allclose(captured["test_y"], [1.0, 2.0])
        np.testing.assert_allclose(captured["test_mu"], [1.5, 2.5])
        np.testing.assert_allclose(captured["test_exposure"], [1.5, 2.5])

        diagnostics_api._compute_pair_diagnostics(
            PairComputer(),
            [("age", "region")],
            pl.DataFrame({"age": [1.0, 2.0], "region": ["A", "B"]}),
            model,
            response_col="y",
            exposure_col=None,
            score_test_design_matrix=None,
            score_test_bread_matrix=None,
            test_data=test_data,
            link="log",
        )
        np.testing.assert_allclose(captured["test_exposure"], [1.0, 1.0])

    def test_factor_bin_pair_helpers_join_categorical_and_continuous_metrics(self):
        train_factor_bins = [
            diagnostics_api.FactorLevelMetrics(
                "A",
                n=2,
                exposure=3.0,
                actual=1.5,
                predicted=1.25,
                ae_ratio=1.2,
                residual_mean=0.1,
                actual_total=4.5,
                predicted_total=3.75,
                base_predicted=1.4,
                base_predicted_total=4.2,
                base_ae_ratio=1.07,
            ),
            diagnostics_api.FactorLevelMetrics(
                "B",
                n=1,
                exposure=1.0,
                actual=2.0,
                predicted=1.0,
                ae_ratio=2.0,
                residual_mean=1.0,
            ),
        ]
        test_factor_bins = [
            diagnostics_api.FactorLevelMetrics(
                "A",
                n=1,
                exposure=2.0,
                actual=1.0,
                predicted=1.2,
                ae_ratio=0.83,
                residual_mean=-0.2,
                actual_total=2.0,
                predicted_total=2.4,
                base_predicted=1.1,
                base_predicted_total=2.2,
                base_ae_ratio=0.91,
            )
        ]
        train_diag = SimpleNamespace(
            factor_diagnostics={"region": train_factor_bins},
            continuous_diagnostics={},
        )
        test_diag = SimpleNamespace(
            factor_diagnostics={"region": test_factor_bins},
            continuous_diagnostics={},
        )

        pairs = diagnostics_api._build_factor_bin_pairs(
            "region", "categorical", train_diag, test_diag
        )

        assert pairs is not None
        assert [p.bin for p in pairs] == ["A", "B"]
        assert pairs[0].test_actual_total == pytest.approx(2.0)
        assert pairs[0].test_base_ae_ratio == pytest.approx(0.91)
        assert pairs[1].test_n == 0
        assert pairs[1].test_actual is None
        assert (
            diagnostics_api._build_factor_bin_pairs(
                "missing",
                "categorical",
                SimpleNamespace(factor_diagnostics={}, continuous_diagnostics={}),
                SimpleNamespace(factor_diagnostics={}, continuous_diagnostics={}),
            )
            is None
        )

        train_bands = [
            SimpleNamespace(
                band=0,
                range_min=0.0,
                range_max=1.0,
                n=2,
                exposure=2.0,
                actual=1.5,
                predicted=1.0,
                ae_ratio=1.5,
                actual_total=3.0,
                predicted_total=2.0,
                base_predicted=1.2,
                base_predicted_total=2.4,
                base_ae_ratio=1.25,
            ),
            SimpleNamespace(
                band=1,
                range_min=1.0,
                range_max=2.0,
                n=2,
                exposure=2.0,
                actual=2.0,
                predicted=2.5,
                ae_ratio=0.8,
            ),
        ]
        test_bands = [
            SimpleNamespace(
                band=0,
                range_min=0.0,
                range_max=1.0,
                n=1,
                exposure=1.0,
                actual=1.0,
                predicted=1.1,
                ae_ratio=0.91,
                actual_total=1.0,
                predicted_total=1.1,
                base_predicted=1.05,
                base_predicted_total=1.05,
                base_ae_ratio=0.95,
            )
        ]
        train_diag = SimpleNamespace(
            factor_diagnostics={},
            continuous_diagnostics={"age": train_bands},
        )
        test_diag = SimpleNamespace(
            factor_diagnostics={},
            continuous_diagnostics={"age": test_bands},
        )

        fallback_pairs = diagnostics_api._build_factor_bin_pairs(
            "age", "continuous", train_diag, test_diag
        )
        assert fallback_pairs is not None
        assert fallback_pairs[0].bin == "0-1"
        assert fallback_pairs[0].test_base_predicted_total == pytest.approx(1.05)
        assert fallback_pairs[1].test_n == 0

        data_pairs = diagnostics_api._build_factor_bin_pairs(
            "age",
            "continuous",
            train_diag,
            test_diag,
            test_data=pl.DataFrame({"age": [0.2, 0.8, 1.2, np.nan]}),
            test_y=np.array([1.0, 3.0, 2.0, 10.0]),
            test_mu=np.array([1.1, 2.7, 2.5, 8.0]),
            test_exposure=np.array([1.0, 2.0, 1.0, 1.0]),
            test_weights=np.array([1.0, 2.0, 1.0, 1.0]),
            test_base_mu=np.array([1.0, 2.0, 2.2, 7.5]),
        )
        assert data_pairs is not None
        assert data_pairs[0].test_n == 2
        assert data_pairs[0].test_exposure == pytest.approx(5.0)
        assert data_pairs[0].test_base_predicted_total == pytest.approx(5.0)
        assert data_pairs[1].test_n == 1

        assert (
            diagnostics_api._build_continuous_train_test_bin_pairs(
                "age",
                [],
                pl.DataFrame({"age": [1.0]}),
                np.ones(1),
                np.ones(1),
                np.ones(1),
            )
            is None
        )
        assert (
            diagnostics_api._build_continuous_train_test_bin_pairs(
                "age",
                [SimpleNamespace(band=0, range_min=np.nan, range_max=1.0)],
                pl.DataFrame({"age": [1.0]}),
                np.ones(1),
                np.ones(1),
                np.ones(1),
            )
            is None
        )


class TestPairDiagnosticsHelperContracts:
    def test_correlation_context_and_slot_helpers_fail_closed(self):
        ctx = pair_mod._GVIFCorrelationContext(
            matrix=np.array([[1.0, 0.2], [0.2, 1.0]]),
            inverse=np.array([[1.1, -0.2], [-0.2, 1.1]]),
            vif_values=np.array([1.1, 1.1]),
        )

        assert ctx.shape == (2, 2)
        np.testing.assert_allclose(np.asarray(ctx), ctx.matrix)
        assert ctx[0, 1] == pytest.approx(0.2)
        assert pair_mod._find_termslot_for_pair(SimpleNamespace(), "a", "b") is None
        assert (
            pair_mod._find_termslot_for_pair(
                SimpleNamespace(_builder=SimpleNamespace(_term_slots=None)),
                "a",
                "b",
            )
            is None
        )
        assert (
            pair_mod._find_termslot_for_pair(
                SimpleNamespace(
                    _builder=SimpleNamespace(
                        _term_slots=[
                            SimpleNamespace(term_type="linear", factors=["a", "b"]),
                            SimpleNamespace(term_type="interaction", factors=["a"]),
                        ]
                    )
                ),
                "a",
                "b",
            )
            is None
        )

    def test_binning_helpers_cover_degenerate_inputs(self):
        data = pl.DataFrame(
            {
                "x": [0.0, 1.0, 2.0, 3.0],
                "segment": ["A", "B", "C", None],
                "few": ["A", "A", "B", "B"],
            }
        )

        assert pair_mod._apply_continuous_edges(np.array([1.0, 2.0]), np.array([1.0])).tolist() == [
            0,
            0,
        ]
        assert pair_mod._bin_categorical_levels(np.array([], dtype=object), top_k=3) == [
            pair_mod._OTHER_LABEL
        ]
        np.testing.assert_array_equal(
            pair_mod._apply_categorical_levels(np.array(["A", "B"]), []),
            [-1, -1],
        )
        np.testing.assert_array_equal(
            pair_mod._apply_categorical_levels(pl.Series(["A", "missing"]), ["A"]),
            [0, -1],
        )
        np.testing.assert_array_equal(pair_mod._to_string_array(np.array(["a", "b"])), ["a", "b"])
        np.testing.assert_array_equal(pair_mod._to_string_array(np.array([1, 2])), ["1", "2"])

        assert pair_mod._choose_bin_counts(
            "continuous_x_continuous",
            (20, 20),
            (10, 5),
            (5, 5),
            max_total_cells=100,
            cardinality1=None,
            cardinality2=None,
        ) == (10, 10)
        assert pair_mod._choose_bin_counts(
            "continuous_x_categorical",
            (5, 5),
            (10, 8),
            (8, 8),
            max_total_cells=30,
            cardinality1=None,
            cardinality2=None,
        ) == (10, 2)
        assert pair_mod._choose_bin_counts(
            "categorical_x_categorical",
            (5, 5),
            (5, 5),
            (8, 7),
            max_total_cells=9,
            cardinality1=None,
            cardinality2=None,
        ) == (2, 2)

        cat_codes, edges, levels = pair_mod._bin_factor(
            data,
            "segment",
            "row",
            "categorical_x_categorical",
            2,
            exposure=np.ones(data.height),
        )
        assert edges is None
        assert levels is not None and levels[-1] == pair_mod._OTHER_LABEL
        assert cat_codes.shape == (data.height,)

        cont_codes, edges, levels = pair_mod._bin_factor(
            data,
            "x",
            "row",
            "continuous_x_categorical",
            2,
            exposure=None,
        )
        assert levels is None
        assert edges is not None
        assert cont_codes.shape == (data.height,)

    def test_surface_grid_and_matrix_helpers_cover_failure_modes(self):
        assert (
            pair_mod._aggregate_cells(
                np.array([-1], dtype=np.int32),
                np.array([0], dtype=np.int32),
                y=np.array([1.0]),
                exposure=np.array([1.0]),
                mu=np.array([1.0]),
                n_rows=1,
                n_cols=1,
            )
            == {}
        )

        empty_grid = pair_mod._build_surface_grid(
            "a",
            "b",
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            None,
            [],
            None,
            [],
            y=np.array([1.0]),
            exposure=np.array([0.0]),
            mu=np.array([2.0]),
        )
        assert empty_grid.row_levels == []
        assert empty_grid.col_levels == []
        assert empty_grid.cells[0].actual == 0.0
        assert empty_grid.cells[0].predicted is None
        assert empty_grid.cells[0].ae_ratio is None

        assert pair_mod._build_design_correlation_matrix(None) is None
        assert pair_mod._build_design_correlation_matrix(np.ones((1, 2))) is None
        assert (
            pair_mod._build_design_correlation_matrix_from_moments(1, np.ones(2), np.eye(2)) is None
        )
        assert (
            pair_mod._build_design_correlation_matrix_from_moments(3, np.ones((1, 2)), np.eye(2))
            is None
        )
        assert (
            pair_mod._build_design_correlation_matrix_from_moments(3, np.ones(2), np.ones((3, 3)))
            is None
        )

        singular = pair_mod._build_design_correlation_matrix_from_moments(
            3,
            np.array([3.0, 6.0]),
            np.array([[3.0, 6.0], [0.0, 12.0]]),
            epsilon=0.0,
            include_inverse=True,
        )
        assert singular is not None
        assert singular.inverse is None
        assert np.all(np.isnan(singular.vif_values))

        n_rows, sums, gram = pair_mod._correlation_moments_for_design_chunk(
            np.array([[1.0, 2.0], [3.0, 4.0]])
        )
        assert n_rows == 2
        np.testing.assert_allclose(sums, [4.0, 6.0])
        assert gram.shape == (2, 2)

        assert pair_mod._compute_block_gvif(None, 0, 1) is None
        assert pair_mod._compute_block_gvif(np.ones((2, 2)), 0, 1) is None
        assert pair_mod._compute_block_gvif(singular, 0, 1) is None
        assert pair_mod._compute_block_gvif(np.eye(2), 0, 2) is None
        assert pair_mod._compute_block_gvif(np.array([[-1.0, 0.0], [0.0, 1.0]]), 0, 1) is None

    def test_pair_matrix_helpers_handle_rust_kernel_failures(self, monkeypatch):
        monkeypatch.setattr(
            pair_mod,
            "_rust_correlation_and_vif",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert pair_mod._build_design_correlation_matrix(np.eye(3)) is None

    def test_block_significance_and_coefficient_helpers_fail_closed(self):
        slot = SimpleNamespace(
            col_start=1,
            col_end=3,
            design_column_names=["a:b[B]", "a:b[C]"],
        )
        assert pair_mod._compute_block_significance(None, np.ones(3), np.ones(3), np.eye(3)) is None
        assert (
            pair_mod._compute_block_significance(
                SimpleNamespace(col_start=1, col_end=1),
                np.ones(3),
                np.ones(3),
                np.eye(3),
            )
            is None
        )
        assert pair_mod._compute_block_significance(slot, np.ones(2), np.ones(2), np.eye(3)) is None
        sig = pair_mod._compute_block_significance(
            slot, np.array([0.0, 0.2, -0.3]), np.ones(3), np.eye(3)
        )
        assert sig is not None
        assert sig.chi2 >= 0.0

        assert (
            pair_mod._extract_block_coefficients(None, np.ones(3), np.ones(3), None, "log") is None
        )
        coeffs = pair_mod._extract_block_coefficients(
            SimpleNamespace(col_start=1, col_end=4, design_column_names=["named"]),
            np.array([0.0, 0.2, -0.3]),
            np.array([1.0, 0.1, 0.0]),
            feature_names=["Intercept", "fallback"],
            link="log",
        )
        assert coeffs is not None
        assert [coef.term for coef in coeffs] == ["named", "col_2"]
        assert coeffs[0].z_value == pytest.approx(2.0)
        assert coeffs[1].z_value == 0.0

    def test_block_significance_handles_rust_failures_and_empty_results(self, monkeypatch):
        slot = SimpleNamespace(col_start=0, col_end=1)

        monkeypatch.setattr(
            pair_mod,
            "_rust_factor_significance_batch",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert pair_mod._compute_block_significance(slot, np.ones(1), np.ones(1), np.eye(1)) is None

        monkeypatch.setattr(pair_mod, "_rust_factor_significance_batch", lambda *args, **kwargs: [])
        assert pair_mod._compute_block_significance(slot, np.ones(1), np.ones(1), np.eye(1)) is None

        monkeypatch.setattr(
            pair_mod, "_rust_factor_significance_batch", lambda *args, **kwargs: [None]
        )
        assert pair_mod._compute_block_significance(slot, np.ones(1), np.ones(1), np.eye(1)) is None

    def test_pair_computer_validation_and_apply_bins_errors(self):
        computer = pair_mod._PairDiagnosticsComputer(
            y=np.array([1.0, 2.0, 3.0]),
            mu=None,
            exposure=None,
            family="poisson",
            feature_names=[],
        )
        data = pl.DataFrame({"x": [1.0, 2.0, 3.0], "cat": ["A", "B", "C"]})

        with pytest.raises(ValueError, match="not a column"):
            computer.compute_pair_diagnostics([("x", "missing")], data)
        with pytest.raises(ValueError, match="not a column"):
            computer.compute_pair_exploration([("x", "missing")], data)
        with pytest.raises(ValueError, match="Either edges or levels"):
            computer._apply_bins(pl.Series([1.0, 2.0]), None, None)

    def test_pair_computer_reuses_train_and_test_bin_caches(self):
        y = np.array([1.0, 2.0, 1.5, 2.5])
        exposure = np.ones(4)
        data = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "cat": ["A", "B", "A", "B"]})
        computer = pair_mod._PairDiagnosticsComputer(
            y=y,
            mu=np.array([1.1, 1.9, 1.4, 2.4]),
            exposure=exposure,
            family="poisson",
            feature_names=[],
        )

        diagnostics = computer.compute_pair_diagnostics(
            [("x", "cat"), ("x", "cat")],
            data,
            design_matrix=np.column_stack([np.ones(4), np.arange(4.0), [0.0, 1.0, 0.0, 1.0]]),
            test_data=data,
            test_y=y,
            test_mu=np.array([1.0, 2.0, 1.5, 2.5]),
            test_exposure=exposure,
            binning_cont_cat=(2, 2),
        )

        assert len(diagnostics) == 2
        assert all(item.test_surface_grid is not None for item in diagnostics)
        assert diagnostics[0].name == diagnostics[1].name == "x:cat"


class TestDiagnosticsTypes:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.0, 0.0),
            (1234.5678, 1234.57),
            (12.345678, 12.3457),
            (0.123456789, 0.123457),
        ],
    )
    def test_round_float_uses_token_efficient_precision_policy(self, value, expected):
        assert _round_float(value) == expected

    def test_json_default_converts_non_json_specials(self):
        assert _json_default(float("nan")) is None
        assert _json_default(float("inf")) is None
        assert _json_default(Percentiles.from_values(1, 5, 10, 25, 50, 75, 90, 95, 99)) == {
            "values": [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }
        assert _json_default(object()).startswith("<object object")

    def test_to_dict_recursive_handles_dataclasses_arrays_and_special_floats(self):
        payload = {
            "smooth": SmoothTermDiagnostics(
                variable="age",
                k=6,
                edf=3.456,
                lambda_=0.012345,
                gcv=1.23456,
                ref_df=2.3456,
                chi2=12.345,
                p_value=0.004321,
            ),
            "array": np.array([1.23456789, np.nan]),
            "int": np.int64(7),
            "float": np.float64(123.456),
            "bad": math.inf,
        }

        converted = _to_dict_recursive(payload)

        assert converted["smooth"]["lambda"] == 0.0123
        assert converted["smooth"]["edf"] == 3.46
        assert converted["array"] == [1.2346, None]
        assert converted["int"] == 7
        assert converted["float"] == 123.46
        assert converted["bad"] is None

    @pytest.mark.parametrize(
        ("feature_name", "expected"),
        [
            ("BonusMalus", "BonusMalus"),
            ("pos(BonusMalus)", "BonusMalus"),
            ("neg(BonusMalus)", "BonusMalus"),
            ("C(Region)[T.A]", "Region"),
            ("ms(age, 1/5)", "age"),
            ("bs(age, 2/5)", "age"),
            ("ns(age, 2/5)", "age"),
            ("s(age)", "age"),
            ("I(BonusMalus ** 2)", "BonusMalus"),
            ("np.log(Exposure)", "Exposure"),
            ("age:region", "age"),
        ],
    )
    def test_extract_base_variable_for_supported_feature_name_patterns(
        self, feature_name, expected
    ):
        assert _extract_base_variable(feature_name) == expected

    def test_data_exploration_serializes_nested_interaction_candidates_to_json(self):
        exploration = DataExploration(
            data_summary={"n": np.int64(10)},
            factor_stats=[{"mean": np.float64(1.23456)}],
            missing_values={},
            univariate_tests=[],
            correlations={},
            cramers_v={},
            vif=[],
            zero_inflation={},
            overdispersion={"ratio": math.nan},
            interaction_candidates=[
                InteractionCandidate(
                    factor1="age",
                    factor2="region",
                    interaction_strength=0.123456,
                    pvalue=0.00123456,
                    n_cells=12,
                )
            ],
            response_stats={},
        )

        as_dict = exploration.to_dict()
        as_json = exploration.to_json()

        assert as_dict["data_summary"]["n"] == 10
        assert as_dict["factor_stats"][0]["mean"] == 1.2346
        assert as_dict["overdispersion"]["ratio"] is None
        assert as_dict["interaction_candidates"][0]["interaction_strength"] == 0.123456
        assert '"factor1": "age"' in as_json


class TestDataExplorerContracts:
    def test_shape_and_ordinal_hints_cover_modeling_recommendations(self):
        explorer = DataExplorer(y=np.arange(5, dtype=np.float64))

        assert explorer._compute_shape_hint([1.0, 2.0]) == {
            "shape": "insufficient_data",
            "recommendation": "linear",
        }
        assert explorer._compute_shape_hint([1.0, 2.0, 3.0])["shape"] == "monotonic_increasing"
        assert explorer._compute_shape_hint([3.0, 2.0, 1.0])["shape"] == "monotonic_decreasing"
        assert explorer._compute_shape_hint([3.0, 2.0, 1.0, 2.0, 3.0])["shape"] == "u_shaped"
        assert explorer._compute_shape_hint([1.0, 2.0, 3.0, 2.0, 1.0])["shape"] == "inverted_u"
        assert explorer._compute_shape_hint([1.0, 1.0, 1.0, 10.0, 10.0])["shape"] == "step_function"
        assert explorer._compute_shape_hint([1.0, 1.0, 1.0, 1.0])["shape"] == "flat"

        assert explorer._detect_ordinal_pattern(np.array(["1", "2"])) == {
            "is_ordinal": True,
            "pattern": "numeric",
        }
        assert explorer._detect_ordinal_pattern(np.array(["A", "B"]))["pattern"] == "alphabetic"
        assert (
            explorer._detect_ordinal_pattern(np.array(["low", "medium", "high"]))["pattern"]
            == "low_medium_high"
        )
        assert (
            explorer._detect_ordinal_pattern(np.array(["Region1", "Region2"]))["pattern"]
            == "prefix_numeric"
        )
        assert explorer._detect_ordinal_pattern(np.array(["red", "blue"])) == {
            "is_ordinal": False,
            "pattern": None,
        }

    def test_factor_stats_use_caches_group_rare_levels_and_skip_invalid_continuous(self):
        y = np.array([0.0, 1.0, 2.0, 3.0, 10.0, 0.0])
        exposure = np.array([100.0, 100.0, 100.0, 100.0, 1.0, 1.0])
        data = pl.DataFrame(
            {
                "cat": ["A", "A", "B", "B", "Rare1", "Rare2"],
                "x": [0.0, 1.0, 2.0, 3.0, np.nan, np.inf],
                "all_bad": [np.nan, np.inf, np.nan, np.inf, np.nan, np.inf],
            }
        )
        explorer = DataExplorer(y=y, exposure=exposure)

        stats = explorer.compute_factor_stats(
            data,
            categorical_factors=["cat"],
            continuous_factors=["x", "all_bad"],
            n_bins=2,
            rare_threshold_pct=1.0,
            max_categorical_levels=3,
            cat_column_cache={"cat": np.array(["A", "A", "B", "B", "Rare1", "Rare2"])},
            cont_column_cache={
                "x": np.array([0.0, 1.0, 2.0, 3.0, np.nan, np.inf]),
                "all_bad": np.array([np.nan, np.inf, np.nan, np.inf, np.nan, np.inf]),
            },
        )

        by_name = {item["name"]: item for item in stats}
        assert set(by_name) == {"x", "cat"}
        assert by_name["x"]["missing_count"] == 2
        assert by_name["x"]["response_by_bin"]
        assert by_name["cat"]["levels"][-1]["level"] == "_Other"
        assert by_name["cat"]["modeling_hints"]["suggested_base_level"] == "A"

    def test_factor_stats_mark_thin_cells_and_cached_levels(self):
        y = np.array([1.0] * 10 + [50.0], dtype=np.float64)
        exposure = np.array([100.0] * 10 + [0.1], dtype=np.float64)
        data = pl.DataFrame(
            {
                "x": list(range(11)),
                "cat": ["A"] * 10 + ["Thin"],
            }
        )
        explorer = DataExplorer(y=y, exposure=exposure)

        stats = explorer.compute_factor_stats(
            data,
            categorical_factors=["cat"],
            continuous_factors=["x"],
            n_bins=11,
            rare_threshold_pct=0.0,
            max_categorical_levels=10,
            cat_unique_cache={
                "cat": (
                    np.array(["A", "Thin"]),
                    np.array([0] * 10 + [1], dtype=np.uint32),
                )
            },
        )

        by_name = {item["name"]: item for item in stats}
        assert by_name["x"]["modeling_hints"]["thin_cells"] == [10]
        assert by_name["cat"]["modeling_hints"]["thin_levels"] == ["Thin"]

    def test_univariate_correlations_vif_missing_and_count_diagnostics(self):
        data = pl.DataFrame(
            {
                "cat": ["A", "B", "A", "B", None],
                "x1": [1.0, 2.0, 3.0, 4.0, None],
                "x2": [2.0, 4.0, 6.0, 8.0, 10.0],
                "x3": [1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )
        y = np.array([0.0, 4.0, 0.0, 4.0, 20.0])
        explorer = DataExplorer(y=y, exposure=np.ones_like(y))

        missing = explorer.compute_missing_values(data, ["cat"], ["x1", "x2"])
        assert missing["summary"] == "Missing values present"
        assert missing["factors_with_missing"][0]["severity"] == "high"

        corr = explorer.compute_correlations(
            data,
            ["x1", "x2"],
            cont_column_cache={
                "x1": np.array([1.0, 2.0, 3.0, 4.0, np.nan]),
                "x2": np.array([2.0, 4.0, 6.0, 8.0, 10.0]),
            },
        )
        assert corr["high_correlations"][0]["severity"] == "high"

        assert explorer.compute_correlations(data, ["x1"]) == {
            "factors": ["x1"],
            "matrix": [],
            "high_correlations": [],
        }

        vif_single = explorer.compute_vif(data, ["x1"])
        assert vif_single == [{"factor": "x1", "vif": 1.0, "severity": "none"}]

        vif_unknown = explorer.compute_vif(
            data,
            ["x1", "x2", "x3"],
            cont_column_cache={
                "x1": np.array([1.0, np.nan, np.nan, np.nan, np.nan]),
                "x2": np.array([2.0, 4.0, 6.0, 8.0, 10.0]),
                "x3": np.ones(5),
            },
        )
        assert {row["severity"] for row in vif_unknown} == {"unknown"}

        zero = explorer.compute_zero_inflation()
        assert zero["severity"] == "severe"

        over = explorer.compute_overdispersion()
        assert over["severity"] in {"moderate", "severe"}

        assert DataExplorer(y=np.zeros(3)).compute_zero_inflation()["severity"] == "none"
        assert DataExplorer(y=np.zeros(3)).compute_overdispersion()["severity"] == "none"

    def test_univariate_vif_and_count_diagnostic_edge_contracts(self, monkeypatch):
        data = pl.DataFrame(
            {
                "short": [1.0] * 9 + [np.nan] * 3,
                "x1": [1.0, 2.0] + [np.nan] * 10,
                "x2": [2.0, 4.0] + [np.nan] * 10,
                "x3": [3.0, 6.0] + [np.nan] * 10,
                "cat_one": ["A"] * 12,
                "cat_many": ["A", "B", "C", "D"] * 3,
            }
        )
        y = np.array([0.0, 1.0, 3.0, 6.0] * 3, dtype=np.float64)
        explorer = DataExplorer(y=y, exposure=np.ones_like(y))

        tests = explorer.compute_univariate_tests(
            data,
            categorical_factors=["cat_one", "cat_many"],
            continuous_factors=["short"],
            cat_unique_cache={
                "cat_one": (np.array(["A"]), np.zeros(12, dtype=np.uint32)),
                "cat_many": (np.array(["A", "B", "C", "D"]), np.tile(np.arange(4), 3)),
            },
            cont_column_cache={"short": np.array([1.0] * 9 + [np.nan] * 3)},
        )
        assert [row["factor"] for row in tests] == ["cat_one", "cat_many"]
        assert tests[0]["f_statistic"] == 0.0
        assert tests[0]["pvalue"] == 1.0

        low_info_corr = explorer.compute_correlations(data, ["x1", "x2", "x3"])
        assert math.isnan(low_info_corr["matrix"][0][2])

        monkeypatch.setattr(
            explorer_mod,
            "_compute_correlation_and_vif",
            lambda *_args: (_ for _ in ()).throw(ValueError("singular")),
        )
        with pytest.raises(FittingError, match="Failed to compute VIF"):
            explorer.compute_vif(
                pl.DataFrame(
                    {
                        "x1": np.arange(5, dtype=np.float64),
                        "x2": np.arange(5, dtype=np.float64) + 1.0,
                    }
                ),
                ["x1", "x2"],
            )

        moderate_zero = DataExplorer(y=np.array([0.0, 0.0, 0.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]))
        assert moderate_zero.compute_zero_inflation()["severity"] == "moderate"

        mild_over = DataExplorer(y=np.array([1.0, 1.0, 1.0, 5.0]))
        assert mild_over.compute_overdispersion()["severity"] == "mild"

    def test_cramers_v_and_interaction_detection_contracts(self, monkeypatch):
        data = pl.DataFrame(
            {
                "cat1": ["A", "A", "B", "B", "A", "B"] * 2,
                "cat2": ["X", "X", "Y", "Y", "X", "Y"] * 2,
                "constant": ["only"] * 12,
                "x": np.arange(12, dtype=np.float64),
            }
        )
        y = np.array([0.0, 0.0, 4.0, 4.0, 1.0, 5.0] * 2, dtype=np.float64)
        explorer = DataExplorer(y=y, exposure=np.ones_like(y))

        one_factor = explorer.compute_cramers_v(data, ["cat1"])
        assert one_factor == {"factors": ["cat1"], "matrix": [], "high_associations": []}

        cramers = explorer.compute_cramers_v(data, ["cat1", "cat2"])
        assert cramers["high_associations"][0]["severity"] == "high"

        calls = {}

        def fake_detect(y_arg, exposure_arg, names, codes, n_levels, *args):
            calls["names"] = list(names)
            calls["n_levels"] = list(n_levels)
            calls["code_shapes"] = [code.shape for code in codes]
            np.testing.assert_allclose(y_arg, y)
            np.testing.assert_allclose(exposure_arg, np.ones_like(y))
            return [
                {
                    "factor1": names[0],
                    "factor2": names[1],
                    "interaction_strength": 0.25,
                    "pvalue": 0.01,
                    "n_cells": 4,
                }
            ]

        monkeypatch.setattr(explorer_mod, "_detect_exploratory_interactions", fake_detect)
        candidates = explorer.detect_interactions(
            data,
            ["cat1", "x", "constant"],
            cat_unique_cache={"cat1": np.unique(data["cat1"].to_numpy(), return_inverse=True)},
            cont_column_cache={"x": data["x"].to_numpy().astype(float)},
            min_cell_count=1,
        )

        assert calls["names"] == ["cat1", "x"]
        assert calls["n_levels"] == [2, 5]
        assert calls["code_shapes"] == [(12,), (12,)]
        assert candidates[0].factor1 == "cat1"
        assert candidates[0].interaction_strength == pytest.approx(0.25)

    def test_cramers_v_and_interaction_detection_fallback_paths(self, monkeypatch):
        data = pl.DataFrame(
            {
                "cat1": ["A", "B"] * 6,
                "cat2": ["X", "Y", "Z"] * 4,
                "cat_cache": ["L", "M"] * 6,
                "x_sparse": [1.0] * 9 + [np.nan] * 3,
                "x_dense": np.arange(12, dtype=np.float64),
                "x_raw": np.linspace(0.0, 1.0, 12),
                "constant": ["only"] * 12,
            }
        )
        explorer = DataExplorer(y=np.arange(12, dtype=np.float64), exposure=np.ones(12))

        monkeypatch.setattr(
            explorer_mod,
            "_compute_cramers_v_matrix_from_codes",
            lambda *_args: (_ for _ in ()).throw(ValueError("bad cramers")),
        )
        with pytest.raises(ValidationError, match="bad cramers"):
            explorer.compute_cramers_v(data, ["cat1", "cat2"])

        calls: dict[str, Any] = {}

        def fake_detect(_y, _exposure, names, codes, n_levels, *_args):
            calls["names"] = list(names)
            calls["n_levels"] = list(n_levels)
            calls["max_codes"] = [int(np.max(code)) for code in codes]
            return []

        monkeypatch.setattr(explorer_mod, "_detect_exploratory_interactions", fake_detect)
        assert (
            explorer.detect_interactions(
                data,
                ["cat_cache", "x_sparse", "x_dense", "x_raw", "constant"],
                cat_column_cache={"cat_cache": np.array(["L", "M"] * 6)},
                cont_column_cache={
                    "x_sparse": np.array([1.0] * 9 + [np.nan] * 3),
                    "x_dense": np.arange(12, dtype=np.float64),
                },
                min_cell_count=1,
            )
            == []
        )
        assert calls["names"] == ["cat_cache", "x_dense", "x_raw"]
        assert calls["n_levels"] == [2, 5, 5]

        assert explorer.detect_interactions(data, ["constant", "x_sparse"]) == []

    def test_eta_squared_and_interaction_strength_edge_cases(self):
        explorer = DataExplorer(
            y=np.array([0.0, 1.0, 10.0, 9.0, 1.0, 0.0, 9.0, 10.0]),
            exposure=np.ones(8),
        )

        assert explorer._compute_eta_squared_response_codes(np.array([], dtype=int), 0) == 0.0
        assert (
            DataExplorer(y=np.ones(4))._compute_eta_squared_response(np.array(["A", "B", "A", "B"]))
            == 0.0
        )
        assert (
            explorer._compute_eta_squared_response(
                np.array(["A", "A", "B", "B", "A", "A", "B", "B"])
            )
            > 0.9
        )

        assert (
            explorer._compute_interaction_strength_response(
                "a",
                np.array([0, 0, 1, 1, 0, 0, 1, 1]),
                "b",
                np.array([0, 0, 0, 0, 1, 1, 1, 1]),
                min_cell_count=3,
            )
            is None
        )

        candidate = explorer._compute_interaction_strength_response(
            "a",
            np.array([0, 0, 1, 1, 0, 0, 1, 1]),
            "b",
            np.array([0, 0, 0, 0, 1, 1, 1, 1]),
            min_cell_count=2,
        )
        assert candidate is not None
        assert candidate.n_cells == 4
        assert candidate.interaction_strength >= 0.0

        zero_var = DataExplorer(y=np.ones(8), exposure=np.ones(8))
        assert (
            zero_var._compute_interaction_strength_response(
                "a",
                np.array([0, 0, 1, 1, 0, 0, 1, 1]),
                "b",
                np.array([0, 0, 0, 0, 1, 1, 1, 1]),
                min_cell_count=2,
            )
            is None
        )

        saturated_explorer = DataExplorer(y=np.array([0.0, 1.0, 2.0, 3.0]), exposure=np.ones(4))
        saturated = saturated_explorer._compute_interaction_strength_response(
            "a",
            np.array([0, 1, 2, 3]),
            "b",
            np.array([0, 0, 0, 0]),
            min_cell_count=1,
        )
        assert saturated is not None
        assert math.isnan(saturated.pvalue)


class TestFactorDiagnosticsHelperContracts:
    def _computer(self):
        return _FactorDiagnosticsComputer(
            y=np.array([1.0, 2.0, 3.0, 4.0]),
            mu=np.array([1.1, 1.9, 3.2, 3.8]),
            exposure=np.ones(4),
            pearson_residuals=np.array([-0.1, 0.1, -0.2, 0.2]),
            feature_names=["Intercept", "age", "bs(age, 1/2)", "region[T.B]", "age:region[T.B]"],
            family="poisson",
            weights=np.array([1.0, 2.0, 1.0, 2.0]),
            base_mu=np.array([1.0, 2.0, 3.0, 4.0]),
        )

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (None, True),
            ("valid", True),
            ("unavailable", True),
            ("constrained_boundary", True),
            ("covariance_skipped", False),
            ("naive_after_regularization", False),
            ("naive_after_selection", False),
            ("naive_after_cv_selection", False),
        ],
    )
    def test_inference_validity_policy_matches_public_summary_honesty(self, status, expected):
        assert _inference_is_valid(SimpleNamespace(inference_status=status)) is expected

    def test_feature_alias_refresh_merges_single_source_transform_outputs(self):
        computer = self._computer()
        computer.feature_names = ["Intercept", "age_log"]
        computer._feature_names_list = list(computer.feature_names)
        computer._var_feature_cache["age"] = _FactorFeature(indices=[99], feature_names=["stale"])

        result = SimpleNamespace(
            _input_transforms=[
                {"sources": ["age"], "output": "age_log"},
                {"sources": ["age"], "output": "unused_alias"},
                {"sources": ["age", "territory"], "output": "bad_multi"},
                {"sources": ["age"], "output": 123},
            ]
        )

        computer._refresh_transform_source_aliases(None)
        assert computer._var_feature_cache["age"].feature_names == ["stale"]

        computer._refresh_transform_source_aliases(result)
        feature = computer._get_feature_for("age")

        assert feature.indices == [1]
        assert feature.feature_names == ["age_log"]
        assert computer._is_in_model("age")
        assert computer._get_factor_terms("age") == ["age_log"]
        assert computer._get_transformation("age") == "age_log"
        assert not computer._is_in_model("unused")

    def test_merge_transform_alias_features_keeps_existing_indices_unique(self):
        feature = _FactorFeature(
            indices=[1], feature_names=["age"], term_type="linear", transformation="age"
        )
        merged = _FactorDiagnosticsComputer._merge_transform_alias_features(
            feature,
            aliases=["age", "age_log"],
            feature_names=["Intercept", "age", "age_log"],
        )

        assert merged.indices == [1, 2]
        assert merged.feature_names == ["age", "age_log"]
        assert merged.term_type == "linear"

        index = _FactorFeatureIndex(["missing"], ["Intercept", "age"])
        assert index.features_for("unregistered") == _FactorFeature()
        assert not index.is_in_model("missing")

    def test_get_feature_for_repairs_partially_initialized_instances(self):
        computer = _FactorDiagnosticsComputer.__new__(_FactorDiagnosticsComputer)
        computer.feature_names = ["Intercept", "age", "C(region)[B]"]

        age = computer._get_feature_for("age")
        region = computer._get_feature_for("region")

        assert age.indices == [1]
        assert region.indices == [2]
        assert computer._feature_names_list == ["Intercept", "age", "C(region)[B]"]
        assert "age" in computer._var_feature_cache

    def test_factor_coefficients_respect_inference_policy_and_relativity_scope(self):
        computer = self._computer()
        result = SimpleNamespace(
            params=lambda: np.array([0.0, 0.2, 0.3, -0.1, 0.4]),
            bse=lambda: np.array([0.0, 0.1, 0.2, 0.05, 0.1]),
            pvalues=lambda: np.array([1.0, 0.03, 0.04, 0.05, 0.06]),
            feature_names=computer.feature_names,
            link="log",
            inference_status="valid",
        )

        age_coeffs = computer._get_factor_coefficients("age", result)
        region_coeffs = computer._get_factor_coefficients("region", result)
        missing_coeffs = computer._get_factor_coefficients("missing", result)

        assert age_coeffs is not None
        assert [coef.term for coef in age_coeffs] == ["age", "bs(age, 1/2)"]
        assert age_coeffs[0].relativity == pytest.approx(round(math.exp(0.2), 4))
        assert age_coeffs[1].relativity is None
        assert region_coeffs is not None
        assert region_coeffs[0].term == "region[T.B]"
        assert missing_coeffs is None

        result.inference_status = "covariance_skipped"
        suppressed = computer._get_factor_coefficients("region", result)
        assert suppressed is not None
        assert math.isnan(suppressed[0].std_error)
        assert math.isnan(suppressed[0].p_value)

        assert computer._get_factor_coefficients("age", None) is None

        std_error_result = SimpleNamespace(
            params=np.array([0.0, 0.25, 0.3, -0.2, 0.4]),
            std_errors=lambda: np.array([0.0, 0.05, 0.2, 0.1, 0.1]),
            feature_names=computer.feature_names,
            link="log",
            inference_status="valid",
        )
        coeffs = computer._get_factor_coefficients("region", std_error_result)
        assert coeffs is not None
        assert coeffs[0].std_error == pytest.approx(0.1)
        assert coeffs[0].z_value == pytest.approx(-2.0)

        with pytest.raises(FittingError, match="Failed to extract coefficient"):
            computer._get_factor_coefficients(
                "region",
                SimpleNamespace(
                    params=np.array([0.0, 0.2, 0.3, "not-float", 0.4], dtype=object),
                    bse=lambda: np.ones(5),
                    feature_names=computer.feature_names,
                    inference_status="valid",
                ),
            )

    def test_factor_significance_uses_joint_wald_and_suppresses_invalid_inference(self):
        computer = self._computer()
        result = SimpleNamespace(
            params=np.array([0.0, 0.2, 0.3, -0.1, 0.4]),
            bse=lambda: np.array([0.0, 0.1, 0.2, 0.05, 0.1]),
            feature_names=computer.feature_names,
            inference_status="valid",
        )

        significance = computer.compute_factor_significance("age", result, bread_matrix=np.eye(5))

        assert significance is not None
        assert significance.chi2 > 0.0
        assert 0.0 <= significance.p <= 1.0

        result.inference_status = "naive_after_selection"
        assert computer.compute_factor_significance("age", result, bread_matrix=np.eye(5)) is None
        assert (
            computer.compute_factor_significance("missing", result, bread_matrix=np.eye(5)) is None
        )
        assert (
            computer.compute_factor_significance("age", SimpleNamespace(params=np.ones(5))) is None
        )

        result.inference_status = "valid"
        single = computer.compute_factor_significance("region", result, bread_matrix=None)
        assert single is not None
        assert single.chi2 == pytest.approx(20.0)
        assert computer.compute_factor_significance("missing", result, bread_matrix=None) is None

    def test_factor_significance_raises_on_numeric_covariance_failures(self, monkeypatch):
        computer = self._computer()
        result = SimpleNamespace(
            params=np.array([0.0, 0.2, 0.3, -0.1, 0.4]),
            bse=lambda: np.array([0.0, 0.1, 0.2, 0.05, 0.1]),
            feature_names=computer.feature_names,
            inference_status="valid",
        )
        monkeypatch.setattr(
            np.linalg,
            "inv",
            lambda _matrix: (_ for _ in ()).throw(np.linalg.LinAlgError("singular")),
        )

        with pytest.raises(FittingError, match="Failed to compute factor significance"):
            computer.compute_factor_significance("age", result, bread_matrix=np.eye(5))

    def test_ae_and_residual_pattern_formatters_include_base_overlay_and_cache_moments(self):
        computer = self._computer()
        bins = _FactorDiagnosticsComputer._format_ae_bins(
            [
                {"count": 0},
                {
                    "count": 10,
                    "bin_label": "A",
                    "exposure": 5.0,
                    "actual_sum": 15.0,
                    "predicted_sum": 10.0,
                    "base_sum": 12.5,
                    "actual_expected_ratio": 1.5,
                    "ae_ci_lower": 1.1,
                    "ae_ci_upper": 2.0,
                },
            ]
        )

        assert len(bins) == 1
        assert bins[0].actual == pytest.approx(3.0)
        assert bins[0].expected == pytest.approx(2.0)
        assert bins[0].base_expected == pytest.approx(2.5)
        assert bins[0].base_ae_ratio == pytest.approx(1.2)

        assert computer._compute_residual_pattern_continuous(np.array([np.nan, np.inf]), 3) == (
            factors_mod.ResidualPattern(resid_corr=0.0, var_explained=0.0)
        )
        pattern = _FactorDiagnosticsComputer._format_residual_pattern_continuous(
            {"correlation_with_residuals": np.nan}
        )
        assert pattern.resid_corr == 0.0

        categorical = computer._compute_residual_pattern_categorical(
            np.array(["A", "A", "B", "B"]),
            precomputed_unique_inverse=(
                np.array(["A", "B"]),
                np.array([0, 0, 1, 1]),
            ),
        )
        assert categorical.var_explained >= 0.0
        assert hasattr(computer, "_pearson_resid_moments_cache")

        direct_ae = computer._compute_ae_continuous(np.array([0.0, 1.0, 2.0, 3.0]), n_bins=2)
        assert direct_ae
        direct_pattern = computer._compute_residual_pattern_continuous(
            np.array([0.0, 1.0, 2.0, 3.0]),
            n_bins=2,
        )
        assert direct_pattern.var_explained >= 0.0

        no_precompute = computer._compute_residual_pattern_categorical(
            np.array(["A", "A", "B", "B"])
        )
        assert no_precompute.var_explained >= 0.0

    def test_score_tests_and_linear_trend_fail_closed_on_degenerate_inputs(self, monkeypatch):
        computer = self._computer()

        assert (
            computer._compute_score_test_categorical(
                np.array(["A", "A", "A", "A"]),
                np.ones((4, 1)),
                np.eye(1),
                np.ones(4),
            )
            is None
        )

        def raise_value_error(*_args, **_kwargs):
            raise ValueError("singular")

        monkeypatch.setattr(factors_mod, "_rust_score_test_continuous", raise_value_error)
        assert (
            computer._compute_score_test_continuous(
                np.array([1.0, np.nan, 2.0, np.inf]),
                np.ones((4, 1)),
                np.eye(1),
                np.ones(4),
            )
            is None
        )
        assert (
            computer._compute_score_test_categorical(
                np.array(["A", "B", "A", "B"]),
                np.ones((4, 1)),
                np.eye(1),
                np.ones(4),
            )
            is None
        )

        slope, pvalue = computer._linear_trend_test(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert math.isnan(slope)
        assert math.isnan(pvalue)

        assert computer._linear_trend_test(np.ones(4), np.arange(4, dtype=float)) == (0.0, 1.0)
        slope, pvalue = computer._linear_trend_test(
            np.arange(5, dtype=float),
            np.array([1.0, 1.8, 3.2, 3.9, 5.1]),
        )
        assert slope > 0.0
        assert 0.0 <= pvalue <= 1.0

        slope, pvalue = computer._linear_trend_test(
            np.arange(4, dtype=float),
            np.array([1.0, 3.0, 5.0, 7.0]),
        )
        assert slope == pytest.approx(2.0)
        assert math.isnan(pvalue)

    def test_score_tests_format_successful_rust_results_and_scrub_invalid_inputs(
        self,
        monkeypatch,
    ):
        captured: dict[str, np.ndarray] = {}

        def fake_score_test(z, *_args):
            captured["z"] = np.asarray(z, dtype=np.float64)
            return {"statistic": 3.456, "df": 1, "pvalue": 0.01234, "significant": True}

        monkeypatch.setattr(factors_mod, "_rust_score_test_continuous", fake_score_test)
        computer = self._computer()
        score = computer._compute_score_test_continuous(
            np.array([0.0, np.nan, 2.0, np.inf]),
            np.ones((4, 1)),
            np.eye(1),
            np.ones(4),
        )
        assert score is not None
        assert score.statistic == pytest.approx(3.46)
        assert np.all(np.isfinite(captured["z"]))

        no_exposure_computer = _FactorDiagnosticsComputer(
            y=np.array([1.0, np.inf, 2.0, 4.0]),
            mu=np.ones(4),
            exposure=None,
            pearson_residuals=np.zeros(4),
            feature_names=[],
            family="poisson",
        )
        score = no_exposure_computer._compute_score_test_categorical(
            np.array(["A", "B", "B", "A"]),
            np.ones((4, 1)),
            np.eye(1),
            np.ones(4),
        )
        assert score is not None
        assert score.significant is True
        assert np.all(np.isfinite(captured["z"]))
        assert (
            no_exposure_computer._compute_score_test_categorical(
                np.array(["A", "A"]),
                np.ones((2, 1)),
                np.eye(1),
                np.ones(2),
            )
            is None
        )

    def test_factor_diagnostics_fall_back_from_batch_kernels(self, monkeypatch):
        computer = self._computer()
        data = pl.DataFrame(
            {
                "age": [0.0, 1.0, 2.0, 3.0],
                "new_x": [0.0, np.nan, 2.0, np.inf],
            }
        )
        result = SimpleNamespace(
            params=np.array([0.0, 0.2, 0.3, -0.1, 0.4]),
            bse=lambda: np.array([0.0, 0.1, 0.2, 0.05, 0.1]),
            feature_names=computer.feature_names,
            inference_status="valid",
            link="log",
        )

        monkeypatch.setattr(
            factors_mod,
            "_rust_factor_significance_batch",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("batch significance failed")),
        )
        monkeypatch.setattr(
            factors_mod,
            "_rust_score_test_continuous_batch",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("batch score failed")),
        )
        monkeypatch.setattr(
            factors_mod,
            "_rust_score_test_continuous",
            lambda *_args: {
                "statistic": 2.5,
                "df": 1,
                "pvalue": 0.04,
                "significant": True,
            },
        )

        diagnostics = computer.compute_factor_diagnostics(
            data,
            categorical_factors=[],
            continuous_factors=["age", "new_x"],
            result=result,
            design_matrix=np.ones((4, 1)),
            bread_matrix=np.eye(5),
            irls_weights=np.ones(4),
            compute_score_tests=True,
        )

        by_name = {diag.name: diag for diag in diagnostics}
        assert by_name["age"].significance is not None
        assert by_name["new_x"].score_test is not None
        assert by_name["new_x"].score_test.significant is True
