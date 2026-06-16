import numpy as np
import polars as pl
import pytest
import rustystats as rs
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

    payload = diagnostics.to_dict()
    assert payload["classes"] == result.classes_
    assert "confusion_matrix" in result.diagnostics_json()
    summary = result.summary()
    assert "Log Loss" in summary
    assert "Top-2 Accuracy" in summary


def test_multinomial_export_fails_explicitly():
    data = _tier_frame(n=80)
    result = rs.multinomial_dict(
        response="tier",
        terms={"x": {"type": "linear"}},
        data=data,
        classes=["none", "basic", "standard", "premium"],
        reference="none",
    ).fit(compute_covariance=False)

    with pytest.raises(ValidationError, match="to_pmml does not yet support MultinomialModel"):
        result.to_pmml()
    with pytest.raises(ValidationError, match="to_onnx does not yet support MultinomialModel"):
        result.to_onnx()
    with pytest.raises(ValidationError, match="to_pmml does not yet support MultinomialModel"):
        rs.to_pmml(result)
    with pytest.raises(ValidationError, match="to_onnx does not yet support MultinomialModel"):
        rs.to_onnx(result)


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
    assert diagnostics.confusion_matrix.sum() > data.height


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

    loaded = rs.MultinomialModel.from_bytes(result.to_bytes())

    np.testing.assert_allclose(result.predict_proba(data), loaded.predict_proba(data))
    assert loaded.classes_ == result.classes_
    assert loaded.reference_ == result.reference_
    assert loaded.diagnostics(train_data=data).nobs == data.height


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


def test_rejects_unsupported_target_encoding_and_smooth_defaults():
    data = _tier_frame(n=80)

    with pytest.raises(ValidationError, match="target_encoding"):
        rs.multinomial_dict(
            response="tier",
            terms={"channel": {"type": "target_encoding"}},
            data=data,
        )

    with pytest.raises(ValidationError, match="fixed-degree"):
        rs.multinomial_dict(
            response="tier",
            terms={"x": {"type": "bs"}},
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
