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
