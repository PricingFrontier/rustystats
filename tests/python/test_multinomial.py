import pickle

import numpy as np
import polars as pl
import pytest
import rustystats as rs
import rustystats.multinomial as multinomial_module
from rustystats._rustystats import fit_multinomial_py
from rustystats.exceptions import PredictionError, ValidationError


def _tier_frame(n: int = 240, seed: int = 123) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    channel = rng.choice(["direct", "agent"], size=n)
    channel_agent = (channel == "agent").astype(float)
    eta = np.column_stack(
        [
            np.zeros(n),
            -0.2 + 0.7 * x - 0.3 * channel_agent,
            -0.5 - 0.2 * x + 0.5 * channel_agent,
            -1.0 + 0.4 * x + 0.2 * channel_agent,
        ]
    )
    exp_eta = np.exp(eta - eta.max(axis=1, keepdims=True))
    probs = exp_eta / exp_eta.sum(axis=1, keepdims=True)
    classes = np.array(["none", "basic", "standard", "premium"], dtype=object)
    y = [rng.choice(classes, p=row) for row in probs]
    return pl.DataFrame({"tier": y, "x": x, "channel": channel})


def _tier_price_frame(n: int = 1200, seed: int = 8642) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    channel = rng.choice(["direct", "agent"], size=n)
    channel_agent = (channel == "agent").astype(float)
    price_basic = np.exp(np.log(320.0) + 0.08 * x + rng.normal(scale=0.25, size=n))
    price_standard = np.exp(np.log(460.0) + 0.06 * x + rng.normal(scale=0.25, size=n))
    price_premium = np.exp(np.log(640.0) + 0.04 * x + rng.normal(scale=0.25, size=n))
    richness_basic = 1.0 + 0.10 * x + rng.normal(scale=0.12, size=n)
    richness_standard = 1.4 + 0.08 * x + rng.normal(scale=0.12, size=n)
    richness_premium = 1.9 + 0.06 * x + rng.normal(scale=0.12, size=n)

    eta = np.column_stack(
        [
            np.zeros(n),
            6.0
            + 0.55 * x
            - 0.20 * channel_agent
            - 1.05 * np.log(price_basic)
            + 0.30 * richness_basic,
            6.5
            + 0.20 * x
            + 0.25 * channel_agent
            - 1.05 * np.log(price_standard)
            + 0.55 * richness_standard,
            7.1
            + 0.10 * x
            + 0.10 * channel_agent
            - 1.05 * np.log(price_premium)
            + 0.85 * richness_premium,
        ]
    )
    exp_eta = np.exp(eta - eta.max(axis=1, keepdims=True))
    probs = exp_eta / exp_eta.sum(axis=1, keepdims=True)
    classes = np.array(["none", "basic", "standard", "premium"], dtype=object)
    y = [rng.choice(classes, p=row) for row in probs]
    premium_available = (x > -0.6) | (np.asarray(y, dtype=object) == "premium")
    return pl.DataFrame(
        {
            "tier": y,
            "x": x,
            "channel": channel,
            "price_basic": price_basic,
            "price_standard": price_standard,
            "price_premium": price_premium,
            "richness_basic": richness_basic,
            "richness_standard": richness_standard,
            "richness_premium": richness_premium,
            "premium_available": premium_available,
            "w": np.where(channel == "agent", 1.5, 1.0),
        }
    )


def _tier_target_encoding_frame(n: int = 360, seed: int = 2468) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    brand = rng.choice(["A", "B", "C", "D", "E"], size=n, p=[0.20, 0.24, 0.22, 0.18, 0.16])
    region = rng.choice(["north", "south", "west"], size=n)
    brand_basic = np.select(
        [brand == "B", brand == "C", brand == "D", brand == "E"],
        [0.35, -0.25, 0.15, -0.10],
        default=0.0,
    )
    brand_premium = np.select(
        [brand == "B", brand == "C", brand == "D", brand == "E"],
        [-0.20, 0.50, 0.10, -0.35],
        default=0.0,
    )
    region_premium = np.where(region == "north", 0.25, np.where(region == "west", -0.15, 0.0))
    eta = np.column_stack(
        [
            np.zeros(n),
            -0.2 + 0.35 * x + brand_basic,
            -0.55 - 0.15 * x + brand_premium + region_premium,
        ]
    )
    exp_eta = np.exp(eta - eta.max(axis=1, keepdims=True))
    probs = exp_eta / exp_eta.sum(axis=1, keepdims=True)
    classes = np.array(["none", "basic", "premium"], dtype=object)
    y = [rng.choice(classes, p=row) for row in probs]
    return pl.DataFrame(
        {
            "tier": y,
            "x": x,
            "brand": brand,
            "region": region,
            "w": np.where(region == "north", 1.7, 1.0),
        }
    )


def test_direct_rust_binding_shapes_and_skip_covariance():
    x = np.array(
        [[1.0, -1.0], [1.0, -0.5], [1.0, 0.2], [1.0, 0.7], [1.0, 1.2], [1.0, 1.8]],
        dtype=np.float64,
    )
    y = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)

    result = fit_multinomial_py(y, x, 3, 0, skip_covariance=True)

    assert result.params.shape == (2, 2)
    assert result.fitted_probabilities.shape == (6, 3)
    assert result.cov_params_unscaled is None
    np.testing.assert_allclose(result.fitted_probabilities.sum(axis=1), 1.0)


def test_direct_rust_binding_with_alternative_tensors():
    rng = np.random.default_rng(4567)
    n = 90
    z = rng.normal(size=n)
    x = np.column_stack([np.ones(n), z]).astype(np.float64)
    alternative_generic = np.zeros((n, 3, 1), dtype=np.float64)
    alternative_generic[:, 1, 0] = rng.normal(loc=1.0, scale=0.3, size=n)
    alternative_generic[:, 2, 0] = rng.normal(loc=1.4, scale=0.3, size=n)
    alternative_specific = np.zeros((n, 3, 1), dtype=np.float64)
    alternative_specific[:, 1, 0] = rng.normal(loc=0.6, scale=0.2, size=n)
    alternative_specific[:, 2, 0] = rng.normal(loc=1.0, scale=0.2, size=n)
    eta = np.column_stack(
        [
            np.zeros(n),
            -0.2
            + 0.4 * z
            - 0.5 * alternative_generic[:, 1, 0]
            + 0.4 * alternative_specific[:, 1, 0],
            0.1
            - 0.3 * z
            - 0.5 * alternative_generic[:, 2, 0]
            + 0.7 * alternative_specific[:, 2, 0],
        ]
    )
    exp_eta = np.exp(eta - eta.max(axis=1, keepdims=True))
    probs = exp_eta / exp_eta.sum(axis=1, keepdims=True)
    y = np.asarray([rng.choice(3, p=row) for row in probs], dtype=np.int64)

    result = fit_multinomial_py(
        y,
        x,
        3,
        0,
        skip_covariance=True,
        alternative_generic=alternative_generic,
        alternative_specific=alternative_specific,
    )

    assert result.params.shape == (2, 2)
    assert result.alternative_generic_coefficients.shape == (1,)
    assert result.alternative_specific_coefficients.shape == (2, 1)
    assert result.n_params == 7
    np.testing.assert_allclose(result.fitted_probabilities.sum(axis=1), 1.0)


def test_direct_rust_binding_accepts_regularized_alternative_tensors():
    x = np.array(
        [[1.0, -1.0], [1.0, -0.5], [1.0, 0.2], [1.0, 0.7], [1.0, 1.2], [1.0, 1.8]],
        dtype=np.float64,
    )
    y = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    alternative_generic = np.zeros((len(y), 3, 1), dtype=np.float64)
    alternative_generic[:, 1, 0] = 1.0
    alternative_generic[:, 2, 0] = 1.5

    result = fit_multinomial_py(
        y,
        x,
        3,
        0,
        alpha=0.1,
        skip_covariance=True,
        alternative_generic=alternative_generic,
        alternative_generic_center=np.array([0.0]),
        alternative_generic_scale=np.array([2.0]),
    )

    assert result.alpha == 0.1
    assert result.alternative_generic_coefficients.shape == (1,)
    np.testing.assert_allclose(result.fitted_probabilities.sum(axis=1), 1.0)


def test_multinomial_dict_fit_predict_and_coef_table():
    data = _tier_frame()

    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}, "channel": {"type": "categorical"}},
        data=data,
        classes=["none", "basic", "standard", "premium"],
        reference="none",
    ).fit()

    assert result.converged
    assert result.params.shape == (3, 3)
    assert result.intercepts.shape == (4,)
    probabilities = result.predict_proba(data)
    assert probabilities.shape == (data.height, 4)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)
    assert set(result.predict(data)).issubset(set(result.classes_))

    table = result.coef_table()
    assert table.height == 9
    assert {"class", "feature", "estimate", "std_error"}.issubset(set(table.columns))


def test_multinomial_phase3_alternative_terms_and_scenario_engine():
    data = _tier_price_frame()
    classes = ["none", "basic", "standard", "premium"]
    alternative_terms = {
        "log_price": {
            "columns": {
                "basic": "price_basic",
                "standard": "price_standard",
                "premium": "price_premium",
            },
            "coefficient": "generic",
            "transform": "log",
        },
        "richness": {
            "columns": {
                "basic": "richness_basic",
                "standard": "richness_standard",
                "premium": "richness_premium",
            },
            "coefficient": "class_specific",
        },
    }

    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}, "channel": {"type": "categorical"}},
        alternative_terms=alternative_terms,
        data=data,
        classes=classes,
        reference="none",
        availability={"premium": "premium_available"},
        weights="w",
    ).fit(compute_covariance=False)

    assert result.converged
    assert result.alternative_generic_feature_names == ["log_price"]
    assert result.alternative_specific_feature_names == ["richness"]
    assert result.alternative_generic_coefficients.shape == (1,)
    assert result.alternative_specific_coefficients.shape == (3, 1)
    assert result.n_params == result.params.size + 4
    assert result.alternative_generic_coefficients[0] < 0.0

    table = result.coef_table()
    assert table.height == result.n_params
    assert {"alternative_generic", "alternative_class_specific"}.issubset(
        set(table["coefficient_type"].to_list())
    )
    summary = result.summary()
    assert "alt_gen" in summary
    assert "alt_class" in summary
    assert "log_price" in summary
    assert "richness" in summary

    probabilities = result.predict_proba(data)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)
    unavailable = ~data["premium_available"].to_numpy()
    assert np.all(probabilities[unavailable, classes.index("premium")] == 0.0)

    scenario = result.scenario(
        data,
        changes={"price_premium": 1.15},
        weights="w",
        value_columns={
            "basic": "price_basic",
            "standard": "price_standard",
            "premium": "price_premium",
        },
        categorical_factors=["channel"],
    )

    assert isinstance(scenario, rs.MultinomialScenario)
    assert scenario.class_mix_delta["premium"] < 0.0
    assert scenario.scenario_class_mix["premium"] < scenario.base_class_mix["premium"]
    assert scenario.expected_value is not None
    weights = data["w"].to_numpy()
    base_values = np.zeros_like(scenario.base_probabilities)
    scenario_values = np.zeros_like(scenario.scenario_probabilities)
    for class_label, column in {
        "basic": "price_basic",
        "standard": "price_standard",
        "premium": "price_premium",
    }.items():
        class_idx = classes.index(class_label)
        base_values[:, class_idx] = data[column].to_numpy()
        multiplier = 1.15 if column == "price_premium" else 1.0
        scenario_values[:, class_idx] = data[column].to_numpy() * multiplier
    expected_base = np.sum(
        weights * np.sum(scenario.base_probabilities * base_values, axis=1)
    ) / np.sum(weights)
    expected_scenario = np.sum(
        weights * np.sum(scenario.scenario_probabilities * scenario_values, axis=1)
    ) / np.sum(weights)
    assert scenario.expected_value["base"] == pytest.approx(expected_base)
    assert scenario.expected_value["scenario"] == pytest.approx(expected_scenario)
    assert scenario.expected_value["delta"] == pytest.approx(expected_scenario - expected_base)
    assert {row["factor"] for row in scenario.segment_mix} == {"channel"}
    scenario_dict = scenario.to_dict()
    assert "scenario_class_mix" in scenario.to_json(indent=None)
    assert "probability_delta_by_class" not in scenario_dict

    loaded = rs.MultinomialModel.from_bytes(result.to_bytes())
    loaded_scenario = loaded.scenario(data, changes={"price_premium": 1.15}, weights="w")
    np.testing.assert_allclose(
        scenario.scenario_probabilities, loaded_scenario.scenario_probabilities
    )


def test_multinomial_chunked_prediction_matches_single_shot(monkeypatch):
    data = _tier_price_frame(n=260, seed=7777)
    classes = ["none", "basic", "standard", "premium"]
    alternative_terms = {
        "log_price": {
            "columns": {
                "basic": "price_basic",
                "standard": "price_standard",
                "premium": "price_premium",
            },
            "coefficient": "generic",
            "transform": "log",
        },
        "richness": {
            "columns": {
                "basic": "richness_basic",
                "standard": "richness_standard",
                "premium": "richness_premium",
            },
            "coefficient": "class_specific",
        },
    }
    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}, "channel": {"type": "categorical"}},
        alternative_terms=alternative_terms,
        data=data,
        classes=classes,
        reference="none",
        availability={"premium": "premium_available"},
        weights="w",
    ).fit(compute_covariance=False)
    calibration = result.fit_calibration(data)
    offset = np.zeros((data.height, len(classes)), dtype=np.float64)
    offset[:, classes.index("basic")] = np.linspace(-0.15, 0.15, data.height)
    offset[:, classes.index("premium")] = np.where(data["premium_available"].to_numpy(), 0.2, -0.2)

    single_logits = result.decision_function(data, offset=offset)
    single_proba = result.predict_proba(data, offset=offset, calibration=calibration)
    single_log_proba = result.predict_log_proba(data, offset=offset, calibration=calibration)
    single_top2 = result.predict_top_k(data, k=2, offset=offset, calibration=calibration)

    chunk_sizes: list[int] = []
    original_transform = result._builder.transform_new_data

    def record_transform(chunk):
        chunk_sizes.append(len(chunk))
        return original_transform(chunk)

    monkeypatch.setattr(result._builder, "transform_new_data", record_transform)
    monkeypatch.setattr(multinomial_module, "_compute_predict_chunk_size", lambda n_features: 17)

    chunked_logits = result.decision_function(data, offset=offset)
    assert len(chunk_sizes) > 1
    assert max(chunk_sizes) <= 17
    np.testing.assert_allclose(chunked_logits, single_logits, atol=1e-12)
    np.testing.assert_allclose(
        result.predict_proba(data, offset=offset, calibration=calibration),
        single_proba,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.predict_log_proba(data, offset=offset, calibration=calibration),
        single_log_proba,
        atol=1e-12,
    )
    chunked_top2 = result.predict_top_k(data, k=2, offset=offset, calibration=calibration)
    assert chunked_top2["class_1"].to_list() == single_top2["class_1"].to_list()
    assert chunked_top2["class_2"].to_list() == single_top2["class_2"].to_list()
    np.testing.assert_allclose(
        chunked_top2["prob_1"].to_numpy(), single_top2["prob_1"].to_numpy(), atol=1e-12
    )
    np.testing.assert_allclose(
        chunked_top2["prob_2"].to_numpy(), single_top2["prob_2"].to_numpy(), atol=1e-12
    )


def test_multinomial_phase4_ridge_supports_alternative_terms():
    data = _tier_price_frame(n=500, seed=2468)
    classes = ["none", "basic", "standard", "premium"]
    alternative_terms = {
        "log_price": {
            "columns": {
                "basic": "price_basic",
                "standard": "price_standard",
                "premium": "price_premium",
            },
            "coefficient": "generic",
            "transform": "log",
        },
        "richness": {
            "columns": {
                "basic": "richness_basic",
                "standard": "richness_standard",
                "premium": "richness_premium",
            },
            "coefficient": "class_specific",
        },
    }

    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}, "channel": {"type": "categorical"}},
        alternative_terms=alternative_terms,
        data=data,
        classes=classes,
        reference="none",
        availability={"premium": "premium_available"},
        weights="w",
    ).fit(alpha=1.0, compute_covariance=False)

    assert result.converged
    assert result.alpha == 1.0
    assert "naive_after_regularization" in result.inference_status
    assert result.alternative_generic_coefficients[0] < 0.0
    assert result.alternative_specific_coefficients.shape == (3, 1)
    probabilities = result.predict_proba(data)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)


def test_multinomial_lasso_zeroes_penalized_terms_and_suppresses_se():
    data = _tier_price_frame(n=520, seed=9281)
    classes = ["none", "basic", "standard", "premium"]
    alternative_terms = {
        "log_price": {
            "columns": {
                "basic": "price_basic",
                "standard": "price_standard",
                "premium": "price_premium",
            },
            "coefficient": "generic",
            "transform": "log",
        },
        "richness": {
            "columns": {
                "basic": "richness_basic",
                "standard": "richness_standard",
                "premium": "richness_premium",
            },
            "coefficient": "class_specific",
        },
    }
    model = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}, "channel": {"type": "categorical"}},
        alternative_terms=alternative_terms,
        data=data,
        classes=classes,
        reference="none",
        availability={"premium": "premium_available"},
        weights="w",
    )
    alpha = rs.multinomial_alpha_max(model, l1_ratio=1.0) * 1.05

    result = model.fit(alpha=alpha, regularization="lasso")

    assert result.converged
    assert result.l1_ratio == 1.0
    assert "naive_after_selection" in result.inference_status
    assert "covariance_unavailable" in result.inference_status
    np.testing.assert_allclose(result.params[:, 1:], 0.0, atol=1e-10)
    np.testing.assert_allclose(result.alternative_generic_coefficients, 0.0, atol=1e-10)
    np.testing.assert_allclose(result.alternative_specific_coefficients, 0.0, atol=1e-10)
    assert np.any(np.abs(result.intercepts) > 1e-8)
    table = result.coef_table(return_format="records")
    assert all(np.isnan(row["std_error"]) for row in table)
    np.testing.assert_allclose(result.predict_proba(data).sum(axis=1), 1.0, atol=1e-10)


def test_multinomial_elastic_net_direct_fit_is_finite():
    data = _tier_frame(n=360, seed=2027)

    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}, "channel": {"type": "categorical"}},
        data=data,
        classes=["none", "basic", "standard", "premium"],
        reference="none",
    ).fit(alpha=0.75, regularization="elastic_net", l1_ratio=0.25, compute_covariance=False)

    assert result.converged
    assert result.l1_ratio == 0.25
    assert np.all(np.isfinite(result.params))
    assert "naive_after_selection" in result.inference_status
    np.testing.assert_allclose(result.fitted_probabilities.sum(axis=1), 1.0, atol=1e-10)
    assert "Newton + Elastic Net" in result.summary()


def test_multinomial_l1_ratio_boundaries_match_ridge_and_lasso():
    data = _tier_frame(n=300, seed=4312)
    classes = ["none", "basic", "standard", "premium"]

    def model() -> rs.MultinomialDict:
        return rs.multinomial_dict(
            response="tier",
            terms={"x": {"type": "linear"}, "channel": {"type": "categorical"}},
            data=data,
            classes=classes,
            reference="none",
        )

    ridge = model().fit(alpha=0.6, regularization="ridge", compute_covariance=False)
    l1_zero = model().fit(alpha=0.6, l1_ratio=0.0, compute_covariance=False)
    lasso = model().fit(alpha=0.6, regularization="lasso", compute_covariance=False)
    l1_one = model().fit(alpha=0.6, l1_ratio=1.0, compute_covariance=False)

    np.testing.assert_allclose(l1_zero.params, ridge.params, atol=1e-10)
    np.testing.assert_allclose(l1_zero.predict_proba(data), ridge.predict_proba(data), atol=1e-10)
    np.testing.assert_allclose(l1_one.params, lasso.params, atol=1e-10)
    np.testing.assert_allclose(l1_one.predict_proba(data), lasso.predict_proba(data), atol=1e-10)


def test_multinomial_lasso_is_reference_dependent_by_design():
    data = _tier_frame(n=360, seed=2027)
    classes = ["none", "basic", "standard", "premium"]
    terms = {"x": {"type": "linear"}, "channel": {"type": "categorical"}}
    ref_none = rs.multinomial_dict(
        response="tier",
        terms=terms,
        data=data,
        classes=classes,
        reference="none",
    ).fit(alpha=0.5, regularization="lasso", compute_covariance=False)
    ref_basic = rs.multinomial_dict(
        response="tier",
        terms=terms,
        data=data,
        classes=classes,
        reference="basic",
    ).fit(alpha=0.5, regularization="lasso", compute_covariance=False)

    max_prediction_delta = np.max(
        np.abs(ref_none.predict_proba(data) - ref_basic.predict_proba(data))
    )
    assert max_prediction_delta > 1e-3
    assert "reference-dependent" in ref_none.summary()


def test_multinomial_validation_deviance_matches_manual_weighted_oracle():
    probabilities = np.array(
        [
            [0.80, 0.15, 0.05],
            [0.20, 0.60, 0.20],
            [0.10, 0.25, 0.65],
            [0.30, 0.55, 0.15],
        ],
        dtype=np.float64,
    )
    y_codes = np.array([0, 1, 2, 1], dtype=np.int64)
    weights = np.array([1.0, 2.5, 0.5, 3.0], dtype=np.float64)

    selected = probabilities[np.arange(len(y_codes)), y_codes]
    expected_weighted = -2.0 * np.sum(weights * np.log(selected)) / np.sum(weights)
    expected_unweighted = -2.0 * np.mean(np.log(selected))

    assert multinomial_module._multinomial_validation_deviance(
        probabilities, y_codes, weights
    ) == pytest.approx(expected_weighted)
    assert multinomial_module._multinomial_validation_deviance(
        probabilities, y_codes, None
    ) == pytest.approx(expected_unweighted)


def test_standardized_multinomial_warm_start_preserves_ridge_solution_speedup():
    data = _tier_price_frame(n=420, seed=613)
    classes = ["none", "basic", "standard", "premium"]
    alternative_terms = {
        "log_price": {
            "columns": {
                "basic": "price_basic",
                "standard": "price_standard",
                "premium": "price_premium",
            },
            "coefficient": "generic",
            "transform": "log",
        },
        "richness": {
            "columns": {
                "basic": "richness_basic",
                "standard": "richness_standard",
                "premium": "richness_premium",
            },
            "coefficient": "class_specific",
        },
    }
    model = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}, "channel": {"type": "categorical"}},
        alternative_terms=alternative_terms,
        data=data,
        classes=classes,
        reference="none",
        availability={"premium": "premium_available"},
        weights="w",
    )

    fit_kwargs = {
        "y_codes": model.y_codes,
        "x": model.X,
        "n_classes": len(model.classes_),
        "reference_index": model.reference_index_,
        "availability": model.availability,
        "offset": model.offset,
        "weights": model.weights,
        "alpha": 0.8,
        "l1_ratio": 0.0,
        "max_iter": 100,
        "tol": 1e-8,
        "fit_intercept": model.intercept,
        "standardize": True,
        "compute_covariance": False,
        "store_design_matrix": False,
        "verbose": False,
        "hessian_memory_limit_bytes": 256 * 1024 * 1024,
        "max_dense_parameters": 5000,
        "alternative_generic": model.alternative_generic,
        "alternative_specific": model.alternative_specific,
    }

    cold = multinomial_module._fit_multinomial_arrays(**fit_kwargs)
    warm = multinomial_module._fit_multinomial_arrays(**fit_kwargs, initial_result=cold)

    assert cold.converged
    assert warm.converged
    assert cold.iterations > 1
    assert warm.iterations <= 1
    np.testing.assert_allclose(warm.params, cold.params, atol=1e-8)
    np.testing.assert_allclose(
        warm.alternative_generic_coefficients,
        cold.alternative_generic_coefficients,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        warm.alternative_specific_coefficients,
        cold.alternative_specific_coefficients,
        atol=1e-8,
    )
    np.testing.assert_allclose(warm.fitted_probabilities, cold.fitted_probabilities, atol=1e-10)


def test_multinomial_cv_ridge_selects_from_grid_and_is_reproducible():
    data = _tier_frame(n=180, seed=773)
    kwargs = {
        "response": "tier",
        "terms": {"x": {"type": "linear"}, "channel": {"type": "categorical"}},
        "data": data,
        "classes": ["none", "basic", "standard", "premium"],
        "reference": "none",
    }
    alphas = [2.0, 0.5, 0.1, 0.0]

    first = rs.multinomial_dict(**kwargs).fit(
        cv=3,
        regularization="ridge",
        alphas=alphas,
        cv_seed=99,
        compute_covariance=False,
    )
    second = rs.multinomial_dict(**kwargs).fit(
        cv=3,
        regularization="ridge",
        alphas=alphas,
        cv_seed=99,
        compute_covariance=False,
    )

    path_alphas = [row["alpha"] for row in first.regularization_path]
    assert first.alpha in path_alphas
    assert first.n_cv_folds == 3
    assert first.cv_selection_method == "min"
    assert first.cv_scoring_objective == "weighted_mean_multinomial_deviance"
    assert first.cv_convergence == {"max_iter": 100, "tol": 1e-8}
    assert first.cv_profile["candidate_fit_count"] == 12
    assert first.cv_profile["within_fold_warm_start"] is True
    assert first.cv_profile["final_refit_warm_start"] is True
    assert "naive_after_cv_selection" in first.inference_status
    np.testing.assert_allclose(first.alpha, second.alpha)
    np.testing.assert_allclose(first.cv_deviance, second.cv_deviance)
    np.testing.assert_allclose(first.predict_proba(data), second.predict_proba(data))


def test_multinomial_cv_owns_collected_lazyframe_data():
    data = _tier_frame(n=120, seed=8765)

    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}, "channel": {"type": "categorical"}},
        data=data.lazy(),
        classes=["none", "basic", "standard", "premium"],
        reference="none",
    ).fit(
        cv=2,
        regularization="ridge",
        alphas=[0.5, 0.1],
        cv_seed=17,
        compute_covariance=False,
    )

    assert result.n_cv_folds == 2
    assert result.cv_profile["candidate_fit_count"] == 4


def test_multinomial_cv_one_se_is_at_least_as_regularized_as_min():
    data = _tier_frame(n=180, seed=1234)
    kwargs = {
        "response": "tier",
        "terms": {"x": {"type": "linear"}, "channel": {"type": "categorical"}},
        "data": data,
        "classes": ["none", "basic", "standard", "premium"],
        "reference": "none",
    }
    fit_min = rs.multinomial_dict(**kwargs).fit(
        cv=3,
        regularization="ridge",
        selection="min",
        alphas=[4.0, 1.0, 0.25, 0.0],
        cv_seed=7,
        compute_covariance=False,
    )
    fit_1se = rs.multinomial_dict(**kwargs).fit(
        cv=3,
        regularization="ridge",
        selection="1se",
        alphas=[4.0, 1.0, 0.25, 0.0],
        cv_seed=7,
        compute_covariance=False,
    )

    assert fit_1se.alpha >= fit_min.alpha


def test_multinomial_cv_lasso_suppresses_covariance_and_records_path():
    data = _tier_frame(n=220, seed=991)

    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}, "channel": {"type": "categorical"}},
        data=data,
        classes=["none", "basic", "standard", "premium"],
        reference="none",
    ).fit(
        cv=3,
        regularization="lasso",
        alphas=[0.5, 0.1],
        include_unregularized=False,
    )

    assert result.alpha in {0.5, 0.1}
    assert result.l1_ratio == 1.0
    assert result.regularization_type == "lasso"
    assert "naive_after_cv_selection" in result.inference_status
    assert "covariance_unavailable" in result.inference_status
    assert len(result.regularization_path) == 2
    assert all(len(scores) == 3 for scores in result.cv_fold_scores.values())
    table = result.coef_table(return_format="records")
    assert all(np.isnan(row["std_error"]) for row in table)


def test_multinomial_cv_target_encoding_uses_fold_safe_path_and_serializes():
    data = _tier_target_encoding_frame(n=180, seed=303)

    result = rs.multinomial_dict(
        response="tier",
        terms={
            "x": {"type": "linear"},
            "brand": {"type": "target_encoding", "n_permutations": 2},
        },
        data=data,
        classes=["none", "basic", "premium"],
        reference="none",
        seed=123,
    ).fit(
        cv=3,
        regularization="ridge",
        n_alphas=3,
        cv_seed=21,
        compute_covariance=False,
    )

    assert result.fold_safe_target_encoding is True
    assert result.cv_profile["n_folds"] == 3
    assert result.cv_profile["cv_seed"] == 21
    assert result.regularization_path is not None
    loaded = rs.MultinomialModel.from_bytes(result.to_bytes())
    assert loaded.fold_safe_target_encoding is True
    np.testing.assert_allclose(loaded.predict_proba(data), result.predict_proba(data))
    assert loaded.regularization_path == result.regularization_path


def test_multinomial_cv_rare_class_fails_before_core_empty_class_error():
    data = pl.DataFrame(
        {
            "tier": ["none", "none", "none", "basic", "basic", "premium"],
            "x": [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0],
        }
    )

    with pytest.raises(ValidationError, match="could not construct CV folds"):
        rs.multinomial_dict(
            response="tier",
            terms={"x": {"type": "linear"}},
            data=data,
            classes=["none", "basic", "premium"],
            reference="none",
        ).fit(cv=2, regularization="ridge", alphas=[1.0, 0.0], compute_covariance=False)


def test_lasso_standardization_keeps_shared_and_alternative_support_invariant():
    data = _tier_price_frame(n=600, seed=7631)
    factor = 17.0
    scaled = data.with_columns(
        (pl.col("x") * factor).alias("x"),
        (pl.col("price_basic") * factor).alias("price_basic"),
        (pl.col("price_standard") * factor).alias("price_standard"),
        (pl.col("price_premium") * factor).alias("price_premium"),
    )
    classes = ["none", "basic", "standard", "premium"]
    alternative_terms = {
        "price": {
            "columns": {
                "basic": "price_basic",
                "standard": "price_standard",
                "premium": "price_premium",
            },
            "coefficient": "generic",
        }
    }

    model_a = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}, "channel": {"type": "categorical"}},
        alternative_terms=alternative_terms,
        data=data,
        classes=classes,
        reference="none",
        weights="w",
    )
    model_b = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}, "channel": {"type": "categorical"}},
        alternative_terms=alternative_terms,
        data=scaled,
        classes=classes,
        reference="none",
        weights="w",
    )
    alpha_a = rs.multinomial_alpha_max(model_a, l1_ratio=1.0) * 0.20
    alpha_b = rs.multinomial_alpha_max(model_b, l1_ratio=1.0) * 0.20

    fit_a = model_a.fit(alpha=alpha_a, regularization="lasso", compute_covariance=False)
    fit_b = model_b.fit(alpha=alpha_b, regularization="lasso", compute_covariance=False)

    support_a = np.concatenate(
        [
            np.abs(fit_a.params[:, 1:]).ravel() > 1e-8,
            np.abs(fit_a.alternative_generic_coefficients).ravel() > 1e-8,
        ]
    )
    support_b = np.concatenate(
        [
            np.abs(fit_b.params[:, 1:]).ravel() > 1e-8,
            np.abs(fit_b.alternative_generic_coefficients).ravel() > 1e-8,
        ]
    )
    np.testing.assert_array_equal(support_a, support_b)
    np.testing.assert_allclose(fit_a.predict_proba(data), fit_b.predict_proba(scaled), atol=1e-7)


def test_ridge_standardizes_generic_alternative_terms():
    data = _tier_price_frame(n=420, seed=1357)
    classes = ["none", "basic", "standard", "premium"]
    factor = 11.0
    scaled = data.with_columns(
        (pl.col("price_basic") * factor).alias("price_basic"),
        (pl.col("price_standard") * factor).alias("price_standard"),
        (pl.col("price_premium") * factor).alias("price_premium"),
    )
    alternative_terms = {
        "price": {
            "columns": {
                "basic": "price_basic",
                "standard": "price_standard",
                "premium": "price_premium",
            },
            "coefficient": "generic",
        }
    }

    fit_a = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}, "channel": {"type": "categorical"}},
        alternative_terms=alternative_terms,
        data=data,
        classes=classes,
        reference="none",
    ).fit(alpha=2.0)
    fit_b = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}, "channel": {"type": "categorical"}},
        alternative_terms=alternative_terms,
        data=scaled,
        classes=classes,
        reference="none",
    ).fit(alpha=2.0)

    np.testing.assert_allclose(
        fit_a.predict_proba(data),
        fit_b.predict_proba(scaled),
        atol=2e-7,
    )
    np.testing.assert_allclose(
        fit_b.alternative_generic_coefficients[0] * factor,
        fit_a.alternative_generic_coefficients[0],
        rtol=1e-5,
        atol=1e-10,
    )
    table_a = fit_a.coef_table().filter(pl.col("coefficient_type") == "alternative_generic")
    table_b = fit_b.coef_table().filter(pl.col("coefficient_type") == "alternative_generic")
    assert np.isfinite(table_a["std_error"][0])
    np.testing.assert_allclose(
        table_b["std_error"][0] * factor,
        table_a["std_error"][0],
        rtol=1e-5,
        atol=1e-10,
    )


def test_multinomial_diagnostics_summary_and_json():
    data = _tier_frame()

    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}, "channel": {"type": "categorical"}},
        data=data,
        classes=["none", "basic", "standard", "premium"],
        reference="none",
    ).fit(compute_covariance=False)

    diagnostics = result.diagnostics()
    assert isinstance(diagnostics, rs.MultinomialDiagnostics)
    assert diagnostics.confusion_matrix.shape == (4, 4)
    assert np.isclose(diagnostics.confusion_matrix.sum(), data.height)
    assert diagnostics.log_loss >= 0.0
    assert 0.0 <= diagnostics.accuracy <= 1.0
    assert 0.0 <= diagnostics.top_2_accuracy <= 1.0
    assert np.isclose(sum(diagnostics.actual_class_mix.values()), 1.0)
    assert np.isclose(sum(diagnostics.predicted_class_mix.values()), 1.0)
    assert diagnostics.aic == result.aic()
    assert diagnostics.bic == result.bic()
    assert result.aic() is not None
    assert result.bic() is not None
    assert result.n_params == (len(result.classes_) - 1) * len(result.feature_names)
    np.testing.assert_allclose(result.aic(), result.deviance + 2.0 * result.n_params)
    np.testing.assert_allclose(
        result.bic(), result.deviance + result.n_params * np.log(result.nobs)
    )

    payload = diagnostics.to_dict()
    assert payload["classes"] == result.classes_
    assert "confusion_matrix" in result.diagnostics_json()
    summary = result.summary()
    assert "Log Loss" in summary
    assert "Top-2 Accuracy" in summary
    assert "Class Mix:" in summary
    assert "Actual" in summary
    assert "Predicted" in summary
    assert "Coefficients:" in summary
    assert "P>|z|" in summary
    for class_label in result.classes_:
        assert class_label in summary
    for non_reference in [label for label in result.classes_ if label != result.reference_]:
        assert any(
            line.startswith(non_reference) and " Intercept " in line
            for line in summary.splitlines()
        )


def test_multinomial_level1_pmml_onnx_export_and_fail_closed(tmp_path):
    import xml.etree.ElementTree as ET

    data = _tier_frame(n=80)
    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}},
        data=data,
        classes=["none", "basic", "standard", "premium"],
        reference="none",
    ).fit(compute_covariance=False)

    onnx_bytes = result.to_onnx()
    assert isinstance(onnx_bytes, bytes)
    assert b"Softmax" in onnx_bytes
    assert b"baseline_category_multinomial_logit" in onnx_bytes
    assert b"prebuilt_shared_design_without_intercept" in onnx_bytes
    assert b"probabilities" in onnx_bytes
    assert rs.to_onnx(result) == onnx_bytes
    onnx_path = tmp_path / "tier.onnx"
    assert result.to_onnx(path=str(onnx_path)) == onnx_path.read_bytes()
    with pytest.raises(ValidationError, match="full raw-data export"):
        result.to_onnx(mode="full")

    pmml = result.to_pmml()
    assert isinstance(pmml, str)
    assert "RegressionModel" in pmml
    assert 'normalizationMethod="softmax"' in pmml
    assert "RegressionTable" in pmml
    assert "NumericPredictor" in pmml
    standalone_pmml = rs.to_pmml(result)
    assert 'normalizationMethod="softmax"' in standalone_pmml
    pmml_path = tmp_path / "tier.pmml"
    assert result.to_pmml(path=str(pmml_path)) == pmml_path.read_text()

    root = ET.fromstring(pmml)
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    regression_model = root.find(f".//{ns}RegressionModel")
    assert regression_model is not None
    assert regression_model.get("normalizationMethod") == "softmax"
    tables = root.findall(f".//{ns}RegressionTable")
    assert [table.get("targetCategory") for table in tables] == result.classes_

    feature_names = [name for name in result.feature_names if name != "Intercept"]
    block_by_class = {}
    block = 0
    for class_idx, class_label in enumerate(result.classes_):
        if class_idx == result.reference_index_:
            continue
        block_by_class[class_label] = block
        block += 1
    for table in tables:
        class_label = table.get("targetCategory")
        if class_label == result.reference_:
            assert float(table.get("intercept")) == pytest.approx(0.0)
            assert table.findall(f"{ns}NumericPredictor") == []
            continue
        block = block_by_class[class_label]
        assert float(table.get("intercept")) == pytest.approx(result.params[block, 0])
        predictors = {
            predictor.get("name"): float(predictor.get("coefficient"))
            for predictor in table.findall(f"{ns}NumericPredictor")
        }
        assert set(predictors) == set(feature_names)
        for feature in feature_names:
            feature_idx = result.feature_names.index(feature)
            assert predictors[feature] == pytest.approx(result.params[block, feature_idx])

    alt_result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}},
        alternative_terms={
            "price": {
                "coefficient": "generic",
                "transform": "log",
                "columns": {
                    "basic": "price_basic",
                    "standard": "price_standard",
                    "premium": "price_premium",
                },
            }
        },
        data=_tier_price_frame(n=120, seed=808),
        classes=["none", "basic", "standard", "premium"],
        reference="none",
    ).fit(compute_covariance=False)
    with pytest.raises(ValidationError, match="alternative-generic"):
        alt_result.to_onnx()
    with pytest.raises(ValidationError, match="alternative-generic"):
        alt_result.to_pmml()

    availability = np.ones((len(data), 4), dtype=bool)
    availability_result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}},
        data=data,
        classes=["none", "basic", "standard", "premium"],
        reference="none",
        availability=availability,
    ).fit(compute_covariance=False)
    with pytest.raises(ValidationError, match="availability"):
        availability_result.to_onnx()
    with pytest.raises(ValidationError, match="availability"):
        availability_result.to_pmml()

    target_encoding_result = rs.multinomial_dict(
        response="tier",
        terms={
            "x": {"type": "linear"},
            "brand": {"type": "target_encoding", "n_permutations": 1},
        },
        data=_tier_target_encoding_frame(n=120, seed=909),
        classes=["none", "basic", "premium"],
        reference="none",
        seed=909,
    ).fit(compute_covariance=False)
    with pytest.raises(ValidationError, match="alternative-specific"):
        target_encoding_result.to_onnx()
    with pytest.raises(ValidationError, match="alternative-specific"):
        target_encoding_result.to_pmml()

    offset_data = data.with_columns(pl.lit(0.0).alias("offset_premium"))
    offset_result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}},
        data=offset_data,
        classes=["none", "basic", "standard", "premium"],
        reference="none",
        offset={"premium": "offset_premium"},
    ).fit(compute_covariance=False)
    with pytest.raises(ValidationError, match="offset"):
        offset_result.to_onnx()
    with pytest.raises(ValidationError, match="offset"):
        offset_result.to_pmml()


def test_multinomial_level1_onnx_numeric_parity_nonzero_reference():
    import xml.etree.ElementTree as ET

    data = _tier_frame(n=160, seed=24601)
    spec = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}, "channel": {"type": "categorical"}},
        data=data,
        classes=["none", "basic", "standard", "premium"],
        reference="standard",
    )
    result = spec.fit(compute_covariance=False)

    onnx_bytes = result.to_onnx()
    assert b"rustystats_version" in onnx_bytes
    assert result.reference_index_ == 2

    feature_indices = [idx for idx, name in enumerate(result.feature_names) if name != "Intercept"]
    x_scoring = np.ascontiguousarray(spec.X[:, feature_indices], dtype=np.float64)
    coefficients = np.zeros((len(feature_indices), len(result.classes_)), dtype=np.float64)
    intercepts = np.zeros(len(result.classes_), dtype=np.float64)
    intercept_idx = result.feature_names.index("Intercept")
    block = 0
    for class_idx in range(len(result.classes_)):
        if class_idx == result.reference_index_:
            continue
        intercepts[class_idx] = result.params[block, intercept_idx]
        coefficients[:, class_idx] = result.params[block, feature_indices]
        block += 1

    assert intercepts[result.reference_index_] == pytest.approx(0.0)
    np.testing.assert_array_equal(coefficients[:, result.reference_index_], 0.0)

    logits = x_scoring @ coefficients + intercepts
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    manual_probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    expected_probabilities = result.predict_proba(data)
    np.testing.assert_allclose(
        manual_probabilities,
        expected_probabilities,
        rtol=1e-12,
        atol=1e-12,
    )

    root = ET.fromstring(result.to_pmml())
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    scoring_feature_names = [
        name for idx, name in enumerate(result.feature_names) if idx in feature_indices
    ]
    feature_position = {name: idx for idx, name in enumerate(scoring_feature_names)}
    pmml_logits = np.zeros((len(data), len(result.classes_)), dtype=np.float64)
    for table in root.findall(f".//{ns}RegressionTable"):
        class_idx = result.classes_.index(table.get("targetCategory"))
        pmml_logits[:, class_idx] = float(table.get("intercept"))
        for predictor in table.findall(f"{ns}NumericPredictor"):
            pmml_logits[:, class_idx] += x_scoring[
                :, feature_position[predictor.get("name")]
            ] * float(predictor.get("coefficient"))
    exp_pmml_logits = np.exp(pmml_logits - pmml_logits.max(axis=1, keepdims=True))
    pmml_probabilities = exp_pmml_logits / exp_pmml_logits.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(
        pmml_probabilities,
        expected_probabilities,
        rtol=1e-9,
        atol=1e-9,
    )

    try:
        import onnxruntime as ort
    except ImportError:
        ort = None
    if ort is not None:
        session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
        [onnx_probabilities] = session.run(None, {"X": x_scoring})
        np.testing.assert_allclose(
            onnx_probabilities,
            expected_probabilities,
            rtol=1e-12,
            atol=1e-12,
        )


def test_multinomial_phase6_intercept_calibration_matches_class_mix_and_serializes():
    train = _tier_frame(n=420, seed=97531)
    calibration_data = _tier_frame(n=260, seed=86420)
    classes = ["none", "basic", "standard", "premium"]
    y = np.asarray(calibration_data["tier"].to_list(), dtype=object)
    for source, target, count in [
        ("none", "premium", 18),
        ("standard", "basic", 10),
    ]:
        source_idx = np.flatnonzero(y == source)
        take = source_idx[: max(0, min(count, len(source_idx) - 3))]
        y[take] = target
    calibration_data = calibration_data.with_columns(pl.Series("tier", y.tolist()))

    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}, "channel": {"type": "categorical"}},
        data=train,
        classes=classes,
        reference="none",
    ).fit(compute_covariance=False)

    params_before = result.params.copy()
    calibration = result.fit_calibration(calibration_data)

    assert isinstance(calibration, rs.MultinomialInterceptCalibration)
    assert calibration.converged
    assert calibration.shifts["none"] == pytest.approx(0.0)
    assert calibration.classes == classes
    assert "vector_intercept" in calibration.to_json(indent=None)
    np.testing.assert_allclose(result.params, params_before)

    class_to_code = {label: idx for idx, label in enumerate(classes)}
    y_codes = np.asarray([class_to_code[label] for label in y], dtype=np.int64)
    actual_mix = np.bincount(y_codes, minlength=len(classes)) / len(y_codes)
    base_probabilities = result.predict_proba(calibration_data)
    calibrated_probabilities = result.predict_proba(calibration_data, calibration=calibration)
    base_mix = base_probabilities.mean(axis=0)
    calibrated_mix = calibrated_probabilities.mean(axis=0)

    np.testing.assert_allclose(calibrated_mix, actual_mix, atol=1e-8)
    assert np.abs(calibrated_mix - actual_mix).sum() < np.abs(base_mix - actual_mix).sum()
    np.testing.assert_allclose(
        [calibration.calibrated_class_mix[label] for label in classes],
        calibrated_mix,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        [result.tier_mix(calibration_data, calibration=calibration)[label] for label in classes],
        calibrated_mix,
        atol=1e-8,
    )

    round_tripped = rs.MultinomialInterceptCalibration.from_dict(calibration.to_dict())
    np.testing.assert_allclose(
        result.predict_proba(calibration_data, calibration=round_tripped),
        calibrated_probabilities,
    )
    np.testing.assert_allclose(
        round_tripped.predict_proba(result, calibration_data),
        calibrated_probabilities,
    )


def test_multinomial_intercept_calibration_respects_availability_and_weights():
    data = _tier_price_frame(n=520, seed=112233)
    classes = ["none", "basic", "standard", "premium"]
    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}, "channel": {"type": "categorical"}},
        data=data,
        classes=classes,
        reference="none",
        availability={"premium": "premium_available"},
        weights="w",
    ).fit(compute_covariance=False)

    calibration = result.fit_calibration(data)
    probabilities = result.predict_proba(data, calibration=calibration)

    unavailable = ~data["premium_available"].to_numpy()
    assert np.all(probabilities[unavailable, classes.index("premium")] == 0.0)
    weights = data["w"].to_numpy()
    class_to_code = {label: idx for idx, label in enumerate(classes)}
    y_codes = np.asarray([class_to_code[label] for label in data["tier"].to_list()])
    actual_mix = np.bincount(y_codes, weights=weights, minlength=len(classes)) / weights.sum()
    calibrated_mix = (probabilities * weights[:, None]).sum(axis=0) / weights.sum()
    np.testing.assert_allclose(calibrated_mix, actual_mix, atol=1e-8)
    np.testing.assert_allclose(
        [result.tier_mix(data, weights="w", calibration=calibration)[label] for label in classes],
        calibrated_mix,
        atol=1e-8,
    )


def test_multinomial_intercept_calibration_validation_errors():
    data = _tier_frame(n=120, seed=54321)
    classes = ["none", "basic", "standard", "premium"]
    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}},
        data=data,
        classes=classes,
        reference="none",
    ).fit(compute_covariance=False)

    with pytest.raises(ValidationError, match="method='intercept'"):
        result.fit_calibration(data, method="temperature")

    missing_premium = data.filter(pl.col("tier") != "premium")
    with pytest.raises(ValidationError, match="positive weighted observations"):
        result.fit_calibration(missing_premium)

    wrong_classes = rs.MultinomialInterceptCalibration(
        classes=["none", "basic"],
        reference="none",
        shifts={"none": 0.0, "basic": 0.1},
    )
    with pytest.raises(ValidationError, match="classes"):
        result.predict_proba(data, calibration=wrong_classes)


def test_multinomial_phase2_diagnostics_train_test_calibration_and_factors():
    data = _tier_frame(n=420, seed=1357).with_columns(
        w=pl.when(pl.col("channel") == "agent").then(2.0).otherwise(1.0)
    )
    train = data.slice(0, 300)
    test = data.slice(300)
    classes = ["none", "basic", "standard", "premium"]

    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}, "channel": {"type": "categorical"}},
        data=train,
        classes=classes,
        reference="none",
        weights="w",
    ).fit(compute_covariance=False)

    diagnostics = result.diagnostics(
        train_data=train.lazy(),
        test_data=test,
        categorical_factors=["channel"],
        continuous_factors=["x"],
    )

    assert isinstance(diagnostics.train, rs.MultinomialDatasetDiagnostics)
    assert isinstance(diagnostics, rs.MultinomialDiagnostics)
    assert diagnostics.test is not None
    assert diagnostics.train_test_comparison is not None
    assert set(diagnostics.per_class_metrics) == set(classes)
    assert set(diagnostics.class_calibration) == set(classes)
    assert set(diagnostics.expected_calibration_error_by_class) == set(classes)
    assert 0.0 <= diagnostics.multiclass_expected_calibration_error <= 1.0
    assert diagnostics.mcfadden_pseudo_r2 is not None
    assert diagnostics.balanced_accuracy is not None
    assert diagnostics.macro_f1 is not None
    for class_label in classes:
        np.testing.assert_allclose(
            diagnostics.class_mix_error[class_label],
            diagnostics.actual_class_mix[class_label]
            - diagnostics.predicted_class_mix[class_label],
        )

    np.testing.assert_allclose(diagnostics.train.confusion_matrix.sum(), train["w"].sum())
    np.testing.assert_allclose(diagnostics.test.confusion_matrix.sum(), test["w"].sum())
    assert diagnostics.train_test_comparison["class_mix_mae_test"] >= 0.0

    factor_rows = diagnostics.factor_diagnostics
    assert {row["dataset"] for row in factor_rows} == {"train", "test"}
    assert {"channel", "x"}.issubset({row["factor"] for row in factor_rows})
    channel_agent = next(
        row
        for row in factor_rows
        if row["dataset"] == "train" and row["factor"] == "channel" and row["level"] == "agent"
    )
    np.testing.assert_allclose(sum(channel_agent["actual_class_mix"].values()), 1.0)
    np.testing.assert_allclose(sum(channel_agent["predicted_class_mix"].values()), 1.0)
    for class_label in classes:
        np.testing.assert_allclose(
            channel_agent["class_mix_error"][class_label],
            channel_agent["actual_class_mix"][class_label]
            - channel_agent["predicted_class_mix"][class_label],
        )
    assert channel_agent["observed_winning_class"] in classes
    assert channel_agent["predicted_winning_class"] in classes
    assert channel_agent["chi_square_class_mix"] is not None

    payload = diagnostics.to_dict()
    assert payload["train"]["name"] == "train"
    assert payload["test"]["name"] == "test"
    assert "factor_diagnostics" in result.diagnostics_json(
        train_data=train,
        test_data=test,
        categorical_factors=["channel"],
        continuous_factors=["x"],
        indent=None,
    )


def test_multinomial_class_weighted_diagnostics_are_labelled_naive():
    data = _tier_frame(n=180, seed=246)

    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}},
        data=data,
        classes=["none", "basic", "standard", "premium"],
        reference="none",
        class_weights={"premium": 3.0},
    ).fit(compute_covariance=False)

    assert "naive_class_weighted" in result.inference_status
    assert result.aic() is None
    assert result.bic() is None
    diagnostics = result.diagnostics()
    assert diagnostics.aic is None
    # Diagnostics report the row-weighted data distribution, not the
    # class-reweighted training objective: confusion totals the raw rows, and
    # the no-train_data path agrees with the supplied-data path.
    assert np.isclose(diagnostics.confusion_matrix.sum(), data.height)
    supplied = result.diagnostics(train_data=data)
    for class_label in result.classes_:
        np.testing.assert_allclose(
            diagnostics.actual_class_mix[class_label],
            supplied.actual_class_mix[class_label],
            atol=1e-12,
        )


def test_availability_affects_fit_not_just_prediction():
    data = _tier_frame(n=180, seed=321).with_columns(premium_available=(pl.col("x") > -0.2))
    classes = ["none", "basic", "standard", "premium"]
    # Ensure observed premium rows stay available for a valid likelihood.
    data = data.with_columns(
        premium_available=pl.when(pl.col("tier") == "premium")
        .then(True)
        .otherwise(pl.col("premium_available"))
    )

    all_available = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}},
        data=data,
        classes=classes,
        reference="none",
    ).fit(compute_covariance=False)
    masked = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}},
        data=data,
        classes=classes,
        reference="none",
        availability={"premium": "premium_available"},
    ).fit(compute_covariance=False)

    diff = np.abs(all_available.fitted_probabilities - masked.fitted_probabilities).sum()
    assert diff > 1e-4
    unavailable = ~data["premium_available"].to_numpy()
    assert np.all(masked.fitted_probabilities[unavailable, classes.index("premium")] == 0.0)


def test_unpenalized_predictions_are_reference_invariant():
    data = _tier_frame(n=320, seed=456)
    classes = ["none", "basic", "standard", "premium"]
    terms = {"x": {"type": "linear"}, "channel": {"type": "categorical"}}

    none_ref = rs.multinomial_dict(
        response="tier",
        terms=terms,
        data=data,
        classes=classes,
        reference="none",
    ).fit(compute_covariance=False)
    basic_ref = rs.multinomial_dict(
        response="tier",
        terms=terms,
        data=data,
        classes=classes,
        reference="basic",
    ).fit(compute_covariance=False)

    np.testing.assert_allclose(
        none_ref.predict_proba(data),
        basic_ref.predict_proba(data),
        atol=5e-7,
        rtol=5e-7,
    )


def test_unpenalized_predictions_match_statsmodels_mnlogit():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(987)
    n = 260
    x = rng.normal(size=n)
    x2 = rng.normal(size=n)
    eta = np.column_stack(
        [
            np.zeros(n),
            -0.1 + 0.8 * x - 0.2 * x2,
            -0.4 - 0.3 * x + 0.6 * x2,
        ]
    )
    exp_eta = np.exp(eta - eta.max(axis=1, keepdims=True))
    probs = exp_eta / exp_eta.sum(axis=1, keepdims=True)
    y = np.asarray([rng.choice(3, p=row) for row in probs], dtype=np.int64)
    data = pl.DataFrame({"y": y, "x": x, "x2": x2})

    rust_result = rs.multinomial_dict(
        response="y",
        terms={"x": {"type": "linear"}, "x2": {"type": "linear"}},
        data=data,
        classes=[0, 1, 2],
        reference="0",
    ).fit(compute_covariance=False)

    x_sm = np.column_stack([np.ones(n), x, x2])
    sm_result = sm.MNLogit(y, x_sm).fit(method="newton", disp=False, maxiter=100)

    np.testing.assert_allclose(
        rust_result.predict_proba(data),
        sm_result.predict(x_sm),
        atol=5e-6,
        rtol=5e-6,
    )


def test_serialization_round_trip_preserves_predictions():
    data = _tier_frame(n=120, seed=789)
    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}, "channel": {"type": "categorical"}},
        data=data,
        classes=["none", "basic", "standard", "premium"],
        reference="none",
    ).fit(compute_covariance=False)

    payload = result.to_bytes()
    state = pickle.loads(payload)
    assert state["schema_version"] == 3
    loaded = rs.MultinomialModel.from_bytes(payload)

    np.testing.assert_allclose(result.predict_proba(data), loaded.predict_proba(data))
    assert loaded.classes_ == result.classes_
    assert loaded.reference_ == result.reference_
    assert loaded.diagnostics(train_data=data).nobs == data.height


def test_serialization_schema_mismatch_fails_clearly():
    data = _tier_frame(n=80, seed=2026)
    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}},
        data=data,
        classes=["none", "basic", "standard", "premium"],
        reference="none",
    ).fit(compute_covariance=False)
    state = pickle.loads(result.to_bytes())
    state["schema_version"] = -1

    with pytest.raises(ValidationError, match="serialized schema_version"):
        rs.MultinomialModel.from_bytes(pickle.dumps(state))


def test_array_availability_requires_prediction_override():
    data = _tier_frame(n=80, seed=42)
    classes = ["none", "basic", "standard", "premium"]
    availability = np.ones((data.height, len(classes)), dtype=bool)
    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}},
        data=data,
        classes=classes,
        reference="none",
        availability=availability,
    ).fit(compute_covariance=False)

    with pytest.raises(PredictionError, match="array availability"):
        result.predict_proba(data)

    probs = result.predict_proba(data, availability=availability)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


def test_array_offset_requires_prediction_override_after_serialization():
    data = _tier_frame(n=90, seed=4242)
    classes = ["none", "basic", "standard", "premium"]
    offset = np.zeros((data.height, len(classes)), dtype=np.float64)
    offset[:, classes.index("premium")] = np.linspace(-0.4, 0.4, data.height)
    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}},
        data=data,
        classes=classes,
        reference="none",
        offset=offset,
    ).fit(compute_covariance=False)
    loaded = rs.MultinomialModel.from_bytes(result.to_bytes())

    with pytest.raises(PredictionError, match="array offsets"):
        loaded.predict_proba(data)

    offset_before = offset.copy()
    np.testing.assert_allclose(
        result.predict_proba(data, offset=offset),
        loaded.predict_proba(data, offset=offset),
    )
    np.testing.assert_allclose(offset, offset_before)


def test_multinomial_target_encoding_direct_fit_predict_and_serializes():
    data = _tier_target_encoding_frame()
    result = rs.multinomial_dict(
        response="tier",
        terms={
            "x": {"type": "linear"},
            "brand": {
                "type": "target_encoding",
                "prior_weight": 2.0,
                "n_permutations": 2,
            },
        },
        data=data,
        classes=["none", "basic", "premium"],
        reference="none",
        seed=123,
    ).fit(compute_covariance=False)

    assert "TE(brand)" not in result.feature_names
    assert result.alternative_specific_feature_names == ["TE(brand)"]
    assert result.alternative_specific_coefficients.shape == (2, 1)
    assert result.n_params == (len(result.classes_) - 1) * (
        len(result.feature_names) + len(result.alternative_specific_feature_names)
    )

    rows = result.coef_table(return_format="records")
    te_rows = [row for row in rows if row["feature"] == "TE(brand)"]
    assert len(te_rows) == 2
    assert {row["coefficient_type"] for row in te_rows} == {"alternative_class_specific"}
    assert {row["class"] for row in te_rows} == {"basic", "premium"}

    prediction_data = data.drop("tier")
    probs = result.predict_proba(prediction_data)
    assert probs.shape == (data.height, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)

    loaded = rs.MultinomialModel.from_bytes(result.to_bytes())
    assert loaded.alternative_specific_feature_names == ["TE(brand)"]
    np.testing.assert_allclose(loaded.predict_proba(prediction_data), probs)

    ridged = rs.multinomial_dict(
        response="tier",
        terms={
            "x": {"type": "linear"},
            "brand": {
                "type": "target_encoding",
                "prior_weight": 2.0,
                "n_permutations": 2,
            },
        },
        data=data,
        classes=["none", "basic", "premium"],
        reference="none",
        seed=123,
    ).fit(alpha=0.1, regularization="ridge", compute_covariance=False)
    assert ridged.alternative_specific_coefficients.shape == (2, 1)
    assert "naive_after_regularization" in ridged.inference_status
    np.testing.assert_allclose(ridged.predict_proba(prediction_data).sum(axis=1), 1.0)


def test_multinomial_target_encoding_interaction_direct_fit():
    data = _tier_target_encoding_frame(seed=1357)
    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}},
        interactions=[
            {
                "brand": {"type": "categorical"},
                "region": {"type": "categorical"},
                "target_encoding": True,
                "prior_weight": 1.5,
                "n_permutations": 2,
            }
        ],
        data=data,
        classes=["none", "basic", "premium"],
        reference="none",
        seed=456,
    ).fit(compute_covariance=False)

    assert "TE(brand:region)" in result.alternative_specific_feature_names
    assert all("TE(" not in name for name in result.feature_names)
    probs = result.predict_proba(data.drop("tier"))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


def test_multinomial_target_encoding_stats_use_availability_and_row_weights_only():
    data = pl.DataFrame(
        {
            "tier": [
                "none",
                "premium",
                "basic",
                "premium",
                "none",
                "basic",
                "premium",
                "none",
            ],
            "brand": ["A", "A", "A", "B", "B", "B", "B", "A"],
            "w": [1.0, 2.0, 3.0, 4.0, 1.5, 0.5, 2.5, 1.0],
            "premium_available": [False, True, True, True, False, True, True, True],
        }
    )
    classes = ["none", "basic", "premium"]
    weighted = rs.multinomial_dict(
        response="tier",
        terms={"brand": {"type": "target_encoding", "n_permutations": 1}},
        data=data,
        classes=classes,
        reference="none",
        availability={"premium": "premium_available"},
        weights="w",
        seed=99,
    )
    state = weighted._target_encoding_state
    assert state is not None
    assert state.terms[0].prior_weight_spec == "auto"
    assert state.terms[0].class_stats["premium"]["prior_weight"] == pytest.approx(20.0)
    premium_stats = state.terms[0].class_stats["premium"]["stats"]
    assert premium_stats["A"] == (2.0, 6.0)
    assert premium_stats["B"] == (6.5, 7.0)

    class_weighted = rs.multinomial_dict(
        response="tier",
        terms={"brand": {"type": "target_encoding", "n_permutations": 1}},
        data=data,
        classes=classes,
        reference="none",
        availability={"premium": "premium_available"},
        weights="w",
        class_weights={"premium": 10.0, "basic": 2.0},
        seed=99,
    )
    assert class_weighted._target_encoding_state is not None
    assert class_weighted._target_encoding_state.terms[0].class_stats == state.terms[0].class_stats

    unweighted = rs.multinomial_dict(
        response="tier",
        terms={"brand": {"type": "target_encoding", "n_permutations": 1}},
        data=data,
        classes=classes,
        reference="none",
        availability={"premium": "premium_available"},
        seed=99,
    )
    assert unweighted._target_encoding_state is not None
    unweighted_stats = unweighted._target_encoding_state.terms[0].class_stats["premium"]["stats"]
    assert unweighted_stats["A"] == (1.0, 3.0)
    assert unweighted_stats["B"] == (2.0, 3.0)


def test_multinomial_target_encoding_fit_uses_ordered_no_leakage():
    data = pl.DataFrame(
        {
            "tier": ["none", "basic", "premium", "none", "basic", "premium"],
            "brand": ["A", "B", "C", "D", "E", "F"],
        }
    )
    classes = ["none", "basic", "premium"]
    model = rs.multinomial_dict(
        response="tier",
        terms={"brand": {"type": "target_encoding", "n_permutations": 1}},
        data=data,
        classes=classes,
        reference="none",
        seed=7,
    )
    assert model._target_encoding_state is not None
    state = model._target_encoding_state.terms[0].class_stats

    for class_label in ["basic", "premium"]:
        class_idx = classes.index(class_label)
        prior = state[class_label]["prior"]
        np.testing.assert_allclose(model.alternative_specific[:, class_idx, 0], prior)


def test_multinomial_target_encoding_ridge_standardizes_alternative_tensor(monkeypatch):
    data = _tier_target_encoding_frame(seed=31415)
    calls = []
    original = multinomial_module._alternative_standardization

    def recording_standardization(
        alternative_generic,
        alternative_specific,
        availability,
        weights,
        reference_index,
    ):
        calls.append((alternative_generic.shape, alternative_specific.shape))
        return original(
            alternative_generic,
            alternative_specific,
            availability,
            weights,
            reference_index,
        )

    monkeypatch.setattr(
        multinomial_module,
        "_alternative_standardization",
        recording_standardization,
    )

    result = rs.multinomial_dict(
        response="tier",
        terms={
            "x": {"type": "linear"},
            "brand": {"type": "target_encoding", "n_permutations": 2},
        },
        data=data,
        classes=["none", "basic", "premium"],
        reference="none",
        seed=123,
    ).fit(alpha=0.1, regularization="ridge", compute_covariance=False)

    assert calls
    generic_shape, specific_shape = calls[0]
    assert generic_shape == (data.height, 3, 0)
    assert specific_shape == (data.height, 3, 1)
    assert result.alternative_specific_feature_names == ["TE(brand)"]
    assert "TE(brand)" not in result.feature_names


def test_multinomial_fold_design_matches_manual_components_and_is_deterministic():
    from rustystats.formula import dict_to_parsed_formula
    from rustystats.interactions import InteractionBuilder

    data = _tier_target_encoding_frame(n=120, seed=2025)
    model = rs.multinomial_dict(
        response="tier",
        terms={
            "x": {"type": "linear"},
            "brand": {"type": "target_encoding", "n_permutations": 2},
        },
        data=data,
        classes=["none", "basic", "premium"],
        reference="none",
        seed=22,
    )
    train_idx = np.arange(0, 84)
    val_idx = np.arange(84, data.height)

    fold = multinomial_module.build_multinomial_fold_design(model, train_idx, val_idx, seed=22)
    repeated = multinomial_module.build_multinomial_fold_design(model, train_idx, val_idx, seed=22)

    for attr in [
        "x_train",
        "x_val",
        "alternative_generic_train",
        "alternative_generic_val",
        "alternative_specific_train",
        "alternative_specific_val",
        "availability_train",
        "availability_val",
        "offset_train",
        "offset_val",
        "y_train",
        "y_val",
    ]:
        np.testing.assert_allclose(getattr(fold, attr), getattr(repeated, attr))
    assert fold.feature_names == repeated.feature_names
    assert fold.preprocessing_state.alternative_specific_feature_names == ["TE(brand)"]

    train_data = model.data[train_idx]
    val_data = model.data[val_idx]
    dummy_response = "__manual_fold_y__"
    parsed = dict_to_parsed_formula(
        response=dummy_response,
        terms=model.terms,
        interactions=model.interactions_spec,
        intercept=model.intercept,
    )
    shared_parsed = multinomial_module._fold_local_parsed_formula(
        multinomial_module._parsed_without_target_encoding(parsed)
    )
    train_build = train_data.with_columns(pl.Series(dummy_response, np.zeros(len(train_data))))
    val_build = val_data.with_columns(pl.Series(dummy_response, np.zeros(len(val_data))))
    builder = InteractionBuilder(train_build)
    _manual_y, manual_x_train, manual_names = builder.build_design_matrix_from_parsed(
        shared_parsed,
        seed=22,
    )
    manual_x_val = builder.transform_new_data(val_build)
    (
        manual_generic_train,
        manual_specific_train,
        manual_generic_names,
        manual_specific_names,
    ) = multinomial_module._resolve_alternative_arrays(
        train_data, model.classes_, model.alternative_terms
    )
    manual_generic_val, manual_specific_val, _generic_val_names, _specific_val_names = (
        multinomial_module._resolve_alternative_arrays(
            val_data, model.classes_, model.alternative_terms
        )
    )
    manual_te_train, manual_te_state = multinomial_module._build_multinomial_target_encoding(
        train_data,
        parsed.target_encoding_terms,
        classes=model.classes_,
        reference=model.reference_,
        y_codes=model.y_codes[train_idx],
        availability=model.availability[train_idx],
        row_weights=None,
        seed=22,
    )
    manual_specific_train = multinomial_module._append_alternative_specific_terms(
        manual_specific_train,
        manual_te_train,
    )
    assert manual_te_state is not None
    manual_specific_val = multinomial_module._append_alternative_specific_terms(
        manual_specific_val,
        manual_te_state.transform(val_data),
    )

    np.testing.assert_allclose(fold.x_train, manual_x_train)
    np.testing.assert_allclose(fold.x_val, manual_x_val)
    np.testing.assert_allclose(fold.alternative_generic_train, manual_generic_train)
    np.testing.assert_allclose(fold.alternative_generic_val, manual_generic_val)
    np.testing.assert_allclose(fold.alternative_specific_train, manual_specific_train)
    np.testing.assert_allclose(fold.alternative_specific_val, manual_specific_val)
    assert fold.feature_names == manual_names
    assert fold.preprocessing_state.alternative_generic_feature_names == manual_generic_names
    assert fold.preprocessing_state.alternative_specific_feature_names == [
        *manual_specific_names,
        *manual_te_state.feature_names,
    ]


def test_multinomial_fold_design_rejects_duplicate_or_overlapping_indices():
    data = _tier_target_encoding_frame(n=90, seed=2026)
    model = rs.multinomial_dict(
        response="tier",
        terms={
            "x": {"type": "linear"},
            "brand": {"type": "target_encoding", "n_permutations": 2},
        },
        data=data,
        classes=["none", "basic", "premium"],
        reference="none",
        seed=33,
    )

    with pytest.raises(ValidationError, match="duplicate"):
        multinomial_module.build_multinomial_fold_design(
            model,
            np.array([0, 1, 1, 2], dtype=np.int64),
            np.array([3, 4, 5], dtype=np.int64),
        )
    with pytest.raises(ValidationError, match="duplicate"):
        multinomial_module.build_multinomial_fold_design(
            model,
            np.array([0, 1, 2], dtype=np.int64),
            np.array([3, 4, 4], dtype=np.int64),
        )
    with pytest.raises(ValidationError, match="disjoint"):
        multinomial_module.build_multinomial_fold_design(
            model,
            np.array([0, 1, 2], dtype=np.int64),
            np.array([2, 3, 4], dtype=np.int64),
        )


def test_multinomial_fold_design_validation_labels_do_not_affect_validation_te():
    data = pl.DataFrame(
        {
            "tier": [
                "none",
                "basic",
                "premium",
                "none",
                "basic",
                "premium",
                "none",
                "basic",
                "premium",
                "none",
            ],
            # Validation includes one seen brand ("A") and one unseen brand
            # ("Z"), so label-invariance covers both lookup paths.
            "brand": ["A", "B", "C", "A", "B", "C", "A", "B", "A", "Z"],
        }
    )
    mutated_tiers = data["tier"].to_list()
    mutated_tiers[8:] = ["none", "basic"]
    mutated = data.with_columns(pl.Series("tier", mutated_tiers))
    classes = ["none", "basic", "premium"]
    train_idx = np.arange(0, 8)
    val_idx = np.arange(8, 10)

    base = rs.multinomial_dict(
        response="tier",
        terms={"brand": {"type": "target_encoding", "n_permutations": 1}},
        data=data,
        classes=classes,
        reference="none",
        seed=5,
    )
    changed_val_y = rs.multinomial_dict(
        response="tier",
        terms={"brand": {"type": "target_encoding", "n_permutations": 1}},
        data=mutated,
        classes=classes,
        reference="none",
        seed=5,
    )

    base_fold = multinomial_module.build_multinomial_fold_design(base, train_idx, val_idx, seed=5)
    changed_fold = multinomial_module.build_multinomial_fold_design(
        changed_val_y, train_idx, val_idx, seed=5
    )

    np.testing.assert_array_equal(base_fold.y_train, changed_fold.y_train)
    assert not np.array_equal(base_fold.y_val, changed_fold.y_val)
    np.testing.assert_allclose(
        base_fold.alternative_specific_val,
        changed_fold.alternative_specific_val,
    )


def test_multinomial_fold_design_unseen_levels_use_fold_training_priors():
    data = pl.DataFrame(
        {
            "tier": [
                "none",
                "basic",
                "premium",
                "none",
                "basic",
                "premium",
                "none",
                "basic",
                "premium",
                "none",
            ],
            "brand": ["A", "B", "C", "A", "B", "C", "A", "B", "Z", "Z"],
        }
    )
    classes = ["none", "basic", "premium"]
    model = rs.multinomial_dict(
        response="tier",
        terms={"brand": {"type": "target_encoding", "n_permutations": 1}},
        data=data,
        classes=classes,
        reference="none",
        seed=17,
    )
    fold = multinomial_module.build_multinomial_fold_design(
        model,
        np.arange(0, 8),
        np.arange(8, 10),
        seed=17,
    )

    state = fold.preprocessing_state.target_encoding_state
    assert state is not None
    for class_label in ["basic", "premium"]:
        class_idx = classes.index(class_label)
        prior = state.terms[0].class_stats[class_label]["prior"]
        np.testing.assert_allclose(
            fold.alternative_specific_val[:, class_idx, 0],
            prior,
        )


def test_multinomial_fold_design_availability_changes_te_values():
    data = pl.DataFrame(
        {
            "tier": [
                "none",
                "basic",
                "premium",
                "none",
                "basic",
                "premium",
                "none",
                "basic",
                "premium",
                "none",
            ],
            "brand": ["A", "A", "B", "A", "B", "B", "A", "B", "A", "A"],
            "premium_available": [
                False,
                False,
                True,
                False,
                True,
                True,
                False,
                True,
                True,
                True,
            ],
        }
    )
    classes = ["none", "basic", "premium"]
    all_available = rs.multinomial_dict(
        response="tier",
        terms={"brand": {"type": "target_encoding", "n_permutations": 1}},
        data=data,
        classes=classes,
        reference="none",
        seed=12,
    )
    masked = rs.multinomial_dict(
        response="tier",
        terms={"brand": {"type": "target_encoding", "n_permutations": 1}},
        data=data,
        classes=classes,
        reference="none",
        availability={"premium": "premium_available"},
        seed=12,
    )
    train_idx = np.arange(0, 8)
    val_idx = np.arange(8, 10)
    all_fold = multinomial_module.build_multinomial_fold_design(
        all_available,
        train_idx,
        val_idx,
        seed=12,
    )
    masked_fold = multinomial_module.build_multinomial_fold_design(
        masked,
        train_idx,
        val_idx,
        seed=12,
    )

    premium_idx = classes.index("premium")
    assert not np.allclose(
        all_fold.alternative_specific_val[:, premium_idx, 0],
        masked_fold.alternative_specific_val[:, premium_idx, 0],
    )


def test_multinomial_rejects_unsupported_te_options_and_smooth_interactions():
    data = _tier_frame(n=80)

    with pytest.raises(ValidationError, match="mode"):
        rs.multinomial_dict(
            response="tier",
            terms={"channel": {"type": "target_encoding", "mode": "full_block"}},
            data=data,
        )

    with pytest.raises(ValidationError, match="Unknown key"):
        rs.multinomial_dict(
            response="tier",
            terms={"channel": {"type": "target_encoding", "surprise": True}},
            data=data,
        )

    with pytest.raises(ValidationError, match="mode"):
        rs.multinomial_dict(
            response="tier",
            terms={"x": {"type": "linear"}},
            interactions=[
                {
                    "channel": {"type": "categorical"},
                    "x": {"type": "linear"},
                    "target_encoding": True,
                    "mode": "generic",
                }
            ],
            data=data,
        )

    with pytest.raises(ValidationError, match="main effects"):
        rs.multinomial_dict(
            response="tier",
            terms={"channel": {"type": "categorical"}},
            interactions=[
                {
                    "x": {"type": "bs", "k": 5},
                    "channel": {"type": "categorical"},
                }
            ],
            data=data,
        )

    with pytest.raises(ValidationError, match="target_encoding factors"):
        rs.multinomial_dict(
            response="tier",
            terms={"x": {"type": "linear"}},
            interactions=[
                {
                    "x": {"type": "linear"},
                    "channel": {"type": "target_encoding"},
                }
            ],
            data=data,
        )


def test_multinomial_relevel_raises_clear_error():
    data = _tier_frame(n=80)
    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}},
        data=data,
        classes=["none", "basic", "standard", "premium"],
        reference="none",
    ).fit(compute_covariance=False)

    with pytest.raises(ValidationError, match="relevel"):
        result.relevel()


def test_string_weights_must_be_finite():
    data = _tier_frame(n=60)
    bad = np.ones(data.height)
    bad[0] = np.inf
    data = data.with_columns(w=pl.Series(bad))
    with pytest.raises(ValidationError, match="finite"):
        rs.multinomial_dict(
            response="tier",
            terms={"x": {"type": "linear"}},
            data=data,
            classes=["none", "basic", "standard", "premium"],
            reference="none",
            weights="w",
        )


def test_multinomial_hessian_memory_limit_rejects_at_public_fit_boundary():
    data = _tier_frame(n=80, seed=707)
    with pytest.raises(ValueError, match="dense Hessian"):
        rs.multinomial_dict(
            response="tier",
            terms={"x": {"type": "linear"}, "channel": {"type": "categorical"}},
            data=data,
            classes=["none", "basic", "standard", "premium"],
            reference="none",
        ).fit(
            compute_covariance=False,
            hessian_memory_limit_bytes=64,
            max_dense_parameters=1_000,
        )


def test_multinomial_covariance_workspace_memory_guard_rejects_peak_allocation():
    rng = np.random.default_rng(708)
    n = 90
    data = pl.DataFrame(
        {
            "tier": np.tile(["a", "b", "c"], n // 3),
            "x0": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
        }
    )

    with pytest.raises(ValueError, match="workspace"):
        rs.multinomial_dict(
            response="tier",
            terms={f"x{idx}": {"type": "linear"} for idx in range(4)},
            data=data,
            classes=["a", "b", "c"],
            reference="a",
        ).fit(
            compute_covariance=True,
            hessian_memory_limit_bytes=1_000,
            max_dense_parameters=100,
        )


def test_singular_design_degrades_covariance_and_surfaces_warning():
    # A constant feature duplicates the intercept, making the Hessian singular.
    # With balanced classes the warm start already sits at the optimum so the
    # fit converges immediately, but the covariance cannot be inverted. The fit
    # must still succeed, drop standard errors, and surface a warning rather
    # than raising.
    data = pl.DataFrame({"y": ["a", "a", "b", "b", "c", "c"], "k": [1.0] * 6})
    result = rs.multinomial_dict(
        response="y",
        terms={"k": {"type": "expression", "expr": "k"}},
        data=data,
        classes=["a", "b", "c"],
        reference="a",
    ).fit()  # compute_covariance defaults to True

    assert result.converged
    assert result._covariance() is None
    assert "covariance_unavailable" in result.inference_status
    assert result.aic() is not None
    assert result.bic() is not None
    assert any("could not invert" in message for message in result.warnings)
    assert "Warnings:" in result.summary()
    table = result.coef_table()
    assert np.all(np.isnan(table["std_error"].to_numpy()))


def test_separation_is_stabilized_by_ridge():
    # "z" perfectly predicts the "premium" tier (premium iff z == 1), which
    # drives the unpenalized premium coefficient toward infinity. Both fits must
    # still return usable models with valid probabilities, and ridge must clearly
    # shrink the separated coefficient.
    rng = np.random.default_rng(2024)
    z = np.concatenate([np.ones(40), np.zeros(200)])
    tier = np.concatenate(
        [np.full(40, "premium"), rng.choice(["none", "basic", "standard"], size=200)]
    )
    data = pl.DataFrame({"tier": tier, "z": z})
    classes = ["none", "basic", "standard", "premium"]
    terms = {"z": {"type": "linear"}}

    unpenalized = rs.multinomial_dict(
        response="tier", terms=terms, data=data, classes=classes, reference="none"
    ).fit(compute_covariance=False)
    ridged = rs.multinomial_dict(
        response="tier", terms=terms, data=data, classes=classes, reference="none"
    ).fit(alpha=1.0, compute_covariance=False)

    for model in (unpenalized, ridged):
        probabilities = model.predict_proba(data)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-8)

    premium_block = [label for label in classes if label != "none"].index("premium")
    z_index = unpenalized.feature_names.index("z")
    separated = abs(unpenalized.params[premium_block, z_index])
    stabilized = abs(ridged.params[premium_block, z_index])
    assert separated > 3.0 * stabilized
    assert ridged.converged


def test_unpenalized_covariance_matches_statsmodels_bse():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(2718)
    n = 500
    x = rng.normal(size=n)
    x2 = rng.normal(size=n)
    eta = np.column_stack(
        [
            np.zeros(n),
            -0.1 + 0.8 * x - 0.2 * x2,
            -0.4 - 0.3 * x + 0.6 * x2,
        ]
    )
    exp_eta = np.exp(eta - eta.max(axis=1, keepdims=True))
    probs = exp_eta / exp_eta.sum(axis=1, keepdims=True)
    y = np.asarray([rng.choice(3, p=row) for row in probs], dtype=np.int64)
    data = pl.DataFrame({"y": y, "x": x, "x2": x2})

    result = rs.multinomial_dict(
        response="y",
        terms={"x": {"type": "linear"}, "x2": {"type": "linear"}},
        data=data,
        classes=[0, 1, 2],
        reference="0",
    ).fit()  # compute_covariance defaults to True

    x_sm = np.column_stack([np.ones(n), x, x2])
    sm_result = sm.MNLogit(y, x_sm).fit(method="newton", disp=False, maxiter=200)
    # bse shape (p, K-1): rows [const, x, x2], cols [class 1, class 2].
    sm_bse = np.asarray(sm_result.bse)
    feature_row = {"Intercept": 0, "x": 1, "x2": 2}
    class_col = {"1": 0, "2": 1}

    table = result.coef_table()
    assert table.height == 6
    for row in table.iter_rows(named=True):
        expected = sm_bse[feature_row[row["feature"]], class_col[row["class"]]]
        np.testing.assert_allclose(row["std_error"], expected, rtol=2e-3, atol=2e-4)


def test_ridge_standardized_covariance_back_transform():
    # Standardized ridge must be invariant to rescaling a feature: predictions,
    # the rescaled coefficient's estimate/SE (scaled by the factor), and every
    # other coefficient's estimate/SE (invariant) must all line up. The "other"
    # checks exercise the per-block intercept cross-term in the covariance
    # back-transform, which a naive one-shot scalar reuse would corrupt.
    data = _tier_frame(n=320, seed=2024)
    classes = ["none", "basic", "standard", "premium"]
    terms = {"x": {"type": "linear"}, "channel": {"type": "categorical"}}
    factor = 7.0
    scaled = data.with_columns((pl.col("x") * factor).alias("x"))

    fit_a = rs.multinomial_dict(
        response="tier", terms=terms, data=data, classes=classes, reference="none"
    ).fit(alpha=3.0)
    fit_b = rs.multinomial_dict(
        response="tier", terms=terms, data=scaled, classes=classes, reference="none"
    ).fit(alpha=3.0)

    np.testing.assert_allclose(fit_a.predict_proba(data), fit_b.predict_proba(scaled), atol=1e-7)

    ta = {(r["class"], r["feature"]): r for r in fit_a.coef_table().iter_rows(named=True)}
    tb = {(r["class"], r["feature"]): r for r in fit_b.coef_table().iter_rows(named=True)}
    assert all(np.isfinite(r["std_error"]) for r in ta.values())

    for (cls, feature), row_a in ta.items():
        row_b = tb[(cls, feature)]
        if feature == "x":
            np.testing.assert_allclose(
                row_b["estimate"] * factor, row_a["estimate"], rtol=1e-5, atol=1e-8
            )
            np.testing.assert_allclose(
                row_b["std_error"] * factor, row_a["std_error"], rtol=1e-5, atol=1e-8
            )
        else:
            np.testing.assert_allclose(row_b["estimate"], row_a["estimate"], rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(row_b["std_error"], row_a["std_error"], rtol=1e-5, atol=1e-8)

    cov = fit_a._covariance()
    assert cov is not None
    assert cov.shape == (9, 9)  # q = p(3) * (K-1)(3)
    np.testing.assert_allclose(cov, cov.T, atol=1e-10)
    assert "naive_after_regularization" in fit_a.inference_status


def test_prediction_methods_are_consistent():
    data = _tier_frame()
    classes = ["none", "basic", "standard", "premium"]
    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}, "channel": {"type": "categorical"}},
        data=data,
        classes=classes,
        reference="none",
    ).fit(compute_covariance=False)

    proba = result.predict_proba(data)

    # decision_function: reference column included with logit 0 (no offset), and
    # the softmax of the returned logits reproduces predict_proba.
    logits = result.decision_function(data)
    assert logits.shape == (data.height, 4)
    np.testing.assert_allclose(logits[:, classes.index("none")], 0.0, atol=1e-12)
    softmax = np.exp(logits - logits.max(axis=1, keepdims=True))
    softmax /= softmax.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(softmax, proba, atol=1e-10)
    assert result.decision_function(data, include_reference=False).shape == (data.height, 3)

    # predict_log_proba
    log_proba = result.predict_log_proba(data)
    np.testing.assert_allclose(log_proba, np.log(proba), atol=1e-10)
    np.testing.assert_allclose(np.exp(log_proba).sum(axis=1), 1.0, atol=1e-10)
    log_proba_pl = result.predict_log_proba(data, return_format="polars")
    assert log_proba_pl.columns == [f"log_prob_{label}" for label in classes]

    # predict_top_k
    top2 = result.predict_top_k(data, k=2)
    assert top2.columns == ["class_1", "prob_1", "class_2", "prob_2"]
    assert list(top2["class_1"]) == list(result.predict(data))
    p1 = top2["prob_1"].to_numpy()
    p2 = top2["prob_2"].to_numpy()
    assert np.all(p1 >= p2 - 1e-12)
    np.testing.assert_allclose(p1, proba.max(axis=1), atol=1e-12)
    with pytest.raises(ValidationError):
        result.predict_top_k(data, k=0)
    with pytest.raises(ValidationError):
        result.predict_top_k(data, k=len(classes) + 1)

    # tier_mix: unweighted equals mean probability; weighted matches manual mean.
    mix = result.tier_mix(data)
    assert set(mix) == set(classes)
    assert np.isclose(sum(mix.values()), 1.0)
    np.testing.assert_allclose([mix[label] for label in classes], proba.mean(axis=0), atol=1e-10)
    weights = np.linspace(0.5, 2.0, data.height)
    weighted_mix = result.tier_mix(data, weights=weights)
    expected = (proba * weights[:, None]).sum(axis=0) / weights.sum()
    np.testing.assert_allclose([weighted_mix[label] for label in classes], expected, atol=1e-10)
    mix_pl = result.tier_mix(data, return_format="polars")
    assert set(mix_pl["class"].to_list()) == set(classes)


def test_class_specific_alternative_term_rejects_reference_column():
    data = _tier_price_frame(n=120)
    with pytest.raises(ValidationError, match="reference class"):
        rs.multinomial_dict(
            response="tier",
            terms={"x": {"type": "linear"}},
            alternative_terms={
                "richness": {
                    "columns": {
                        "none": "x",  # reference-class column on a class_specific term
                        "basic": "richness_basic",
                        "standard": "richness_standard",
                        "premium": "richness_premium",
                    },
                    "coefficient": "class_specific",
                }
            },
            data=data,
            classes=["none", "basic", "standard", "premium"],
            reference="none",
        )


def test_array_fit_weights_diagnostics_requires_column_override():
    data = _tier_frame(n=160, seed=99)
    weights = np.linspace(0.5, 2.0, data.height)
    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}},
        data=data,
        classes=["none", "basic", "standard", "premium"],
        reference="none",
        weights=weights,
    ).fit(compute_covariance=False)

    # No-train_data diagnostics use the fitted (array) row weights.
    diag = result.diagnostics()
    np.testing.assert_allclose(diag.train.total_weight, weights.sum())

    # Supplied-data diagnostics cannot reconstruct array weights -> loud error.
    with pytest.raises(PredictionError, match="array weight"):
        result.diagnostics(train_data=data)

    # ... unless a weights column is provided for the supplied data.
    data_with_weights = data.with_columns(wcol=pl.Series(weights))
    diag2 = result.diagnostics(train_data=data_with_weights, weights="wcol")
    np.testing.assert_allclose(diag2.train.total_weight, weights.sum())
    for class_label in result.classes_:
        np.testing.assert_allclose(
            diag.actual_class_mix[class_label],
            diag2.actual_class_mix[class_label],
            atol=1e-12,
        )


def test_mcfadden_pseudo_r2_is_self_consistent():
    data = _tier_frame(n=240, seed=11)
    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}, "channel": {"type": "categorical"}},
        data=data,
        classes=["none", "basic", "standard", "premium"],
        reference="none",
    ).fit(compute_covariance=False)
    no_arg = result.diagnostics().mcfadden_pseudo_r2
    supplied = result.diagnostics(train_data=data).mcfadden_pseudo_r2
    assert no_arg is not None and supplied is not None
    assert supplied <= 1.0
    np.testing.assert_allclose(no_arg, supplied, atol=1e-8)


def test_weighted_fit_matches_row_replication():
    rng = np.random.default_rng(303)
    n = 70
    x = rng.normal(size=n)
    tier = rng.choice(["a", "b", "c"], size=n)
    base = pl.DataFrame({"tier": tier, "x": x})
    reps = rng.integers(1, 4, size=n)
    weighted = base.with_columns(w=pl.Series(reps.astype(np.float64)))
    replicated = base[np.repeat(np.arange(n), reps).tolist()]
    classes = ["a", "b", "c"]

    weighted_fit = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}},
        data=weighted,
        classes=classes,
        reference="a",
        weights="w",
    ).fit(compute_covariance=False)
    replicated_fit = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}},
        data=replicated,
        classes=classes,
        reference="a",
    ).fit(compute_covariance=False)

    np.testing.assert_allclose(weighted_fit.params, replicated_fit.params, rtol=1e-5, atol=1e-6)


def test_multinomial_fixed_spline_and_interaction_fit():
    data = _tier_frame(n=320, seed=77)
    classes = ["none", "basic", "standard", "premium"]

    bs_result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "bs", "df": 4}, "channel": {"type": "categorical"}},
        interactions=[
            {"x": {"type": "linear"}, "channel": {"type": "categorical"}, "include_main": False}
        ],
        data=data,
        classes=classes,
        reference="none",
    ).fit(compute_covariance=False)
    assert bs_result.converged
    np.testing.assert_allclose(bs_result.predict_proba(data).sum(axis=1), 1.0, atol=1e-10)

    ns_result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "ns", "df": 3}, "channel": {"type": "categorical"}},
        data=data,
        classes=classes,
        reference="none",
    ).fit(compute_covariance=False)
    assert ns_result.converged
    np.testing.assert_allclose(ns_result.predict_proba(data).sum(axis=1), 1.0, atol=1e-10)
    assert bs_result.regularization_type == "none"
    assert bs_result.smooth_terms == []
    assert bs_result.total_edf is None
    assert bs_result.gcv is None


def test_multinomial_linear_and_expression_monotonic_constraints_apply_to_all_blocks():
    data = _tier_frame(n=360, seed=771).with_columns(
        pl.Series("z", np.sin(np.arange(360, dtype=np.float64) * 0.17))
    )
    classes = ["none", "basic", "standard", "premium"]

    result = rs.multinomial_dict(
        response="tier",
        terms={
            "x": {"type": "linear", "monotonicity": "increasing"},
            "z": {"type": "linear", "monotonicity": "decreasing"},
            "z2": {"type": "expression", "expr": "z * z", "monotonicity": "increasing"},
        },
        data=data,
        classes=classes,
        reference="none",
    ).fit(compute_covariance=False)

    assert result.converged
    assert "constrained_boundary" in result.inference_status
    assert result.aic() is None
    assert result.bic() is None
    assert result.feature_names[result.feature_names.index("pos(x)")] == "pos(x)"
    assert result.feature_names[result.feature_names.index("neg(z)")] == "neg(z)"
    expr_idx = next(
        idx
        for idx, feature in enumerate(result.feature_names)
        if feature.startswith("pos(I(z * z))")
    )
    pos_idx = result.feature_names.index("pos(x)")
    neg_idx = result.feature_names.index("neg(z)")
    assert np.all(result.params[:, pos_idx] >= -1e-8)
    assert np.all(result.params[:, expr_idx] >= -1e-8)
    assert np.all(result.params[:, neg_idx] <= 1e-8)
    assert {item["feature"] for item in result.constraint_metadata} >= {
        "pos(x)",
        "neg(z)",
        "pos(I(z * z))",
    }
    assert all(
        item["target"] == "class_utility_vs_reference" for item in result.constraint_metadata
    )
    assert all(item["probability_monotonicity"] is False for item in result.constraint_metadata)
    assert "class utilities versus the reference class" in result.summary()

    loaded = rs.MultinomialModel.from_bytes(result.to_bytes())
    assert loaded.constraint_metadata == result.constraint_metadata
    np.testing.assert_allclose(loaded.predict_proba(data), result.predict_proba(data))


def test_multinomial_fixed_bs_monotonic_constraints_use_generated_basis_columns():
    data = _tier_frame(n=300, seed=772)
    classes = ["none", "basic", "standard", "premium"]

    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "bs", "df": 5, "monotonicity": "increasing"}},
        data=data,
        classes=classes,
        reference="none",
    ).fit(compute_covariance=False)

    spline_indices = [
        idx
        for idx, feature in enumerate(result.feature_names)
        if feature.startswith("bs(x,") and ", +)" in feature
    ]
    assert spline_indices
    assert len(result.constraint_metadata) == len(spline_indices)
    np.testing.assert_array_less(-1e-8, result.params[:, spline_indices] + 1e-12)


def test_multinomial_monotonic_rejections_are_fail_closed():
    data = _tier_frame(n=120, seed=773)
    classes = ["none", "basic", "standard", "premium"]

    with pytest.raises(ValidationError, match="monotonic smooth bs terms"):
        rs.multinomial_dict(
            response="tier",
            terms={"x": {"type": "bs", "k": 5, "monotonicity": "increasing"}},
            data=data,
            classes=classes,
            reference="none",
        )
    with pytest.raises(ValidationError, match="natural splines"):
        rs.multinomial_dict(
            response="tier",
            terms={"x": {"type": "ns", "df": 4, "monotonicity": "increasing"}},
            data=data,
            classes=classes,
            reference="none",
        )
    with pytest.raises(ValidationError, match="target_encoding terms"):
        rs.multinomial_dict(
            response="tier",
            terms={"channel": {"type": "target_encoding", "monotonicity": "increasing"}},
            data=data,
            classes=classes,
            reference="none",
        )
    with pytest.raises(ValidationError, match="main effects"):
        rs.multinomial_dict(
            response="tier",
            terms={"x": {"type": "linear"}, "channel": {"type": "categorical"}},
            interactions=[
                {
                    "x": {"type": "linear", "monotonicity": "increasing"},
                    "channel": {"type": "categorical"},
                    "include_main": False,
                }
            ],
            data=data,
            classes=classes,
            reference="none",
        )


def test_multinomial_cv_preserves_monotonic_constraints_in_fold_refits():
    data = _tier_frame(n=360, seed=774)
    classes = ["none", "basic", "standard", "premium"]

    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear", "monotonicity": "increasing"}},
        data=data,
        classes=classes,
        reference="none",
    ).fit(
        regularization="ridge",
        cv=3,
        n_alphas=3,
        include_unregularized=False,
        compute_covariance=False,
    )

    pos_idx = result.feature_names.index("pos(x)")
    assert np.all(result.params[:, pos_idx] >= -1e-8)
    assert "constrained_boundary" in result.inference_status
    assert result.regularization_path is not None
    assert result.cv_profile["n_folds"] == 3


def test_multinomial_smooth_terms_compose_with_separate_monotonic_terms():
    rng = np.random.default_rng(775)
    data = _tier_frame(n=260, seed=775).with_columns(pl.Series("z", rng.normal(size=260)))
    classes = ["none", "basic", "standard", "premium"]

    result = rs.multinomial_dict(
        response="tier",
        terms={
            "x": {"type": "bs"},
            "z": {"type": "linear", "monotonicity": "increasing"},
        },
        data=data,
        classes=classes,
        reference="none",
    ).fit(
        n_lambda=2,
        lambda_min=0.1,
        lambda_max=1.0,
        max_lambda_iter=1,
        compute_covariance=False,
    )

    assert result.converged
    assert result.regularization_type == "smooth"
    assert len(result.smooth_terms) == 1
    assert result.smooth_terms[0]["variable"] == "x"
    pos_idx = result.feature_names.index("pos(z)")
    assert np.all(result.params[:, pos_idx] >= -1e-8)
    assert "naive_after_regularization" in result.inference_status
    assert "constrained_boundary" in result.inference_status
    assert result.aic() is None
    assert any(item["feature"] == "pos(z)" for item in result.constraint_metadata)
    np.testing.assert_allclose(result.predict_proba(data).sum(axis=1), 1.0, atol=1e-10)


def test_multinomial_default_bs_smooth_fit_metadata_and_serialization():
    data = _tier_frame(n=260, seed=606)
    classes = ["none", "basic", "standard", "premium"]

    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "bs"}, "channel": {"type": "categorical"}},
        data=data,
        classes=classes,
        reference="none",
    ).fit(
        n_lambda=3,
        lambda_min=0.1,
        lambda_max=10.0,
        max_lambda_iter=2,
        compute_covariance=False,
    )

    assert result.converged
    assert result.regularization_type == "smooth"
    assert result.smooth_profile["candidate_fit_count"] >= 3
    assert result.smooth_profile["warm_start"] is True
    assert len(result.smooth_terms) == 1
    smooth = result.smooth_terms[0]
    assert smooth["variable"] == "x"
    assert smooth["type"] == "bs"
    assert smooth["k"] == 10
    assert any(smooth["lambda"] == pytest.approx(value) for value in [0.1, 1.0, 10.0])
    assert 0.0 < smooth["edf"] <= (len(classes) - 1) * smooth["k"]
    assert result.smooth_lambdas == [smooth["lambda"]]
    assert result.smooth_edfs == [smooth["edf"]]
    assert result.total_edf is not None
    assert result.total_edf >= smooth["edf"]
    assert np.isfinite(result.gcv)
    assert "naive_after_regularization" in result.inference_status
    assert result.aic() == pytest.approx(-2.0 * result.log_likelihood + 2.0 * result.total_edf)
    np.testing.assert_allclose(result.predict_proba(data).sum(axis=1), 1.0, atol=1e-10)
    summary = result.summary()
    assert "Newton + Smooth" in summary
    assert "Smooth Terms:" in summary
    assert "Total EDF" in summary

    loaded = rs.MultinomialModel.from_bytes(result.to_bytes())
    assert loaded.regularization_type == "smooth"
    assert loaded.smooth_terms == result.smooth_terms
    assert loaded.total_edf == pytest.approx(result.total_edf)
    assert loaded.gcv == pytest.approx(result.gcv)
    np.testing.assert_allclose(loaded.predict_proba(data), result.predict_proba(data))


def test_multinomial_ns_k_smooth_sets_basis_width_and_rejects_deferred_combinations():
    data = _tier_frame(n=220, seed=607)
    classes = ["none", "basic", "standard", "premium"]

    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "ns", "k": 5}},
        data=data,
        classes=classes,
        reference="none",
    ).fit(
        n_lambda=2,
        lambda_min=0.1,
        lambda_max=1.0,
        max_lambda_iter=1,
        compute_covariance=False,
    )

    assert result.converged
    assert result.regularization_type == "smooth"
    assert result.smooth_terms[0]["type"] == "ns"
    assert result.smooth_terms[0]["k"] == 5
    assert result.smooth_terms[0]["col_end"] - result.smooth_terms[0]["col_start"] == 4
    assert result.params.shape[1] == result.smooth_terms[0]["col_end"]
    np.testing.assert_allclose(result.predict_proba(data).sum(axis=1), 1.0, atol=1e-10)

    with pytest.raises(ValidationError, match="do not yet support cv"):
        rs.multinomial_dict(
            response="tier",
            terms={"x": {"type": "bs", "k": 5}},
            data=data,
            classes=classes,
            reference="none",
        ).fit(cv=2, regularization="ridge")

    with pytest.raises(ValidationError, match="cannot yet be combined"):
        rs.multinomial_dict(
            response="tier",
            terms={"x": {"type": "bs", "k": 5}},
            data=data,
            classes=classes,
            reference="none",
        ).fit(alpha=0.1, regularization="ridge")


def test_ridge_alt_class_specific_se_scale_invariant():
    data = _tier_price_frame(n=400, seed=909)
    classes = ["none", "basic", "standard", "premium"]
    alt = {
        "richness": {
            "columns": {
                "basic": "richness_basic",
                "standard": "richness_standard",
                "premium": "richness_premium",
            },
            "coefficient": "class_specific",
        }
    }
    factor = 9.0
    scaled = data.with_columns(
        (pl.col("richness_basic") * factor).alias("richness_basic"),
        (pl.col("richness_standard") * factor).alias("richness_standard"),
        (pl.col("richness_premium") * factor).alias("richness_premium"),
    )
    fit_a = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}},
        alternative_terms=alt,
        data=data,
        classes=classes,
        reference="none",
    ).fit(alpha=2.0)
    fit_b = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}},
        alternative_terms=alt,
        data=scaled,
        classes=classes,
        reference="none",
    ).fit(alpha=2.0)

    np.testing.assert_allclose(fit_a.predict_proba(data), fit_b.predict_proba(scaled), atol=1e-7)
    ta = {
        (r["class"], r["coefficient_type"]): r
        for r in fit_a.coef_table(return_format="records")
        if r["coefficient_type"] == "alternative_class_specific"
    }
    tb = {
        (r["class"], r["coefficient_type"]): r
        for r in fit_b.coef_table(return_format="records")
        if r["coefficient_type"] == "alternative_class_specific"
    }
    for class_label in ["basic", "standard", "premium"]:
        key = (class_label, "alternative_class_specific")
        assert np.isfinite(ta[key]["std_error"])
        np.testing.assert_allclose(
            tb[key]["estimate"] * factor, ta[key]["estimate"], rtol=1e-5, atol=1e-8
        )
        np.testing.assert_allclose(
            tb[key]["std_error"] * factor, ta[key]["std_error"], rtol=1e-5, atol=1e-8
        )


def test_multinomial_cv_min_selects_argmin_of_cv_deviance():
    data = _tier_frame(n=200, seed=4242)
    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}, "channel": {"type": "categorical"}},
        data=data,
        classes=["none", "basic", "standard", "premium"],
        reference="none",
    ).fit(
        cv=3,
        regularization="ridge",
        alphas=[3.0, 1.0, 0.3, 0.1, 0.0],
        cv_seed=11,
        compute_covariance=False,
    )

    path = result.regularization_path
    assert path is not None
    best = min(path, key=lambda row: row["cv_deviance_mean"])
    assert result.alpha == pytest.approx(best["alpha"])
    assert result.cv_deviance == pytest.approx(best["cv_deviance_mean"])


def test_multinomial_cv_one_se_selects_most_regularized_within_band():
    kwargs = {
        "response": "tier",
        "terms": {"x": {"type": "linear"}, "channel": {"type": "categorical"}},
        "data": _tier_frame(n=200, seed=4242),
        "classes": ["none", "basic", "standard", "premium"],
        "reference": "none",
    }
    alphas = [3.0, 1.0, 0.3, 0.1, 0.0]
    fit_1se = rs.multinomial_dict(**kwargs).fit(
        cv=3,
        regularization="ridge",
        selection="1se",
        alphas=alphas,
        cv_seed=11,
        compute_covariance=False,
    )

    path = fit_1se.regularization_path
    best_min = min(path, key=lambda row: row["cv_deviance_mean"])
    threshold = best_min["cv_deviance_mean"] + best_min["cv_deviance_se"]
    within_band = [row for row in path if row["cv_deviance_mean"] <= threshold]
    expected_alpha = max(within_band, key=lambda row: row["alpha"])["alpha"]
    assert fit_1se.alpha == pytest.approx(expected_alpha)
    assert fit_1se.alpha >= best_min["alpha"]


def test_multinomial_cv_drops_alpha_when_a_fold_fails(monkeypatch):
    data = _tier_frame(n=180, seed=515)
    n_rows = data.height
    failing_alpha = 0.5
    real_fit = multinomial_module._fit_multinomial_arrays

    def flaky_fit(*args, **kwargs):
        # Fail only for CV fold fits (subset rows), never the full-data refit or
        # warm-start, so the selected model still fits cleanly.
        if kwargs.get("alpha") == failing_alpha and np.asarray(kwargs["x"]).shape[0] < n_rows:
            raise RuntimeError("induced CV fold failure")
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(multinomial_module, "_fit_multinomial_arrays", flaky_fit)

    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}, "channel": {"type": "categorical"}},
        data=data,
        classes=["none", "basic", "standard", "premium"],
        reference="none",
    ).fit(
        cv=3,
        regularization="ridge",
        alphas=[2.0, 0.5, 0.0],
        cv_seed=3,
        compute_covariance=False,
    )

    path_alphas = {row["alpha"] for row in result.regularization_path}
    assert failing_alpha not in path_alphas
    assert result.alpha in path_alphas
    assert result.alpha != failing_alpha
    # The dropped alpha still appears in the raw fold scores with non-finite entries.
    assert any(not np.isfinite(score) for score in result.cv_fold_scores[failing_alpha])


def test_multinomial_predict_rejects_all_unavailable_row():
    data = _tier_frame(n=120, seed=77)
    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}},
        data=data,
        classes=["none", "basic", "standard", "premium"],
        reference="none",
    ).fit(compute_covariance=False)

    new_data = data.head(3)
    availability = np.ones((3, len(result.classes_)), dtype=bool)
    availability[0, :] = False
    with pytest.raises(PredictionError, match="no classes"):
        result.predict_proba(new_data, availability=availability)


def test_multinomial_smooth_lambda_is_gcv_argmin():
    data = _tier_frame(n=320, seed=909)
    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "bs", "k": 6}},
        data=data,
        classes=["none", "basic", "standard", "premium"],
        reference="none",
    ).fit(compute_covariance=False)

    profile = result.smooth_profile
    assert profile is not None
    candidate_gcvs = profile["candidate_gcvs"]
    assert len(candidate_gcvs) == 1  # single smooth term
    (candidates,) = candidate_gcvs.values()
    assert len(candidates) >= 2
    argmin_lambda = min(candidates, key=lambda lam_gcv: lam_gcv[1])[0]
    # The reported selected lambda must be the GCV argmin of its final candidate sweep.
    assert result.smooth_lambdas[0] == pytest.approx(argmin_lambda)
