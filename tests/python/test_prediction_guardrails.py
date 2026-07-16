import numpy as np
import polars as pl
import pytest
import rustystats as rs
from rustystats.exceptions import PredictionError, ValidationError
from rustystats.formula import _MU_CEILING_FACTOR, _family_has_unbounded_mean


def test_log_link_prediction_fails_closed_on_extreme_eta():
    data = pl.DataFrame(
        {
            "y": [1.0, 0.0, 1.0, 2.0, 1.0, 0.0],
            "x": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "exposure": [1.0] * 6,
        }
    )
    model = rs.glm_dict(
        response="y",
        terms={"x": {"type": "linear"}},
        data=data,
        family="poisson",
        exposure="exposure",
    ).fit()

    new_data = pl.DataFrame({"x": [0.5]})
    huge_exposure = np.array([np.exp(60.0)])

    eta = model.predict_linear(new_data, exposure=huge_exposure)
    assert eta[0] > 50.0
    with pytest.raises(PredictionError, match="extreme linear predictors"):
        model.predict(new_data, exposure=huge_exposure)

    clipped = model.predict(new_data, exposure=huge_exposure, on_extreme_eta="clip")
    assert np.isfinite(clipped).all()
    assert clipped[0] <= np.exp(50.0)

    # "warn" surfaces the problem but returns the honest un-clipped mean.
    with pytest.warns(RuntimeWarning, match="extreme linear predictors"):
        warned = model.predict(new_data, exposure=huge_exposure, on_extreme_eta="warn")
    assert warned[0] > np.exp(50.0)

    diag = model.predict_diagnostics(new_data, exposure=huge_exposure)
    assert diag["extreme_eta_count"] == 1
    assert diag["eta"]["max"] > 50.0


def test_alpha_positive_smooth_routes_to_fixed_spline_penalty():
    rng = np.random.default_rng(123)
    x = rng.uniform(0.0, 3.0, 300)
    y = rng.poisson(np.exp(0.1 + 0.2 * np.sin(x))).astype(float)
    data = pl.DataFrame({"y": y, "x": x})

    model = rs.glm_dict(
        response="y",
        terms={"x": {"type": "bs", "k": 6}},
        data=data,
        family="poisson",
    ).fit(alpha=0.05)

    assert model.optimizer_route == "fixed_spline_penalty"
    assert model.smooth_terms
    assert np.isfinite(model.deviance)


def test_l1_regularization_rejected_for_smooth_terms():
    data = pl.DataFrame({"y": [1.0, 2.0, 1.0, 3.0, 2.0, 1.0], "x": np.linspace(0, 1, 6)})
    spec = rs.glm_dict(
        response="y",
        terms={"x": {"type": "bs", "k": 5}},
        data=data,
        family="poisson",
    )
    with pytest.raises(ValidationError, match="L1/elastic-net"):
        spec.fit(alpha=0.1, l1_ratio=0.5)


def test_glm_dict_target_encoding_default_resolves_auto_prior():
    data = pl.DataFrame(
        {
            "y": [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            "brand": ["A", "A", "B", "B", "C", "C", "C", "D"],
        }
    )
    model = rs.glm_dict(
        response="y",
        terms={"brand": {"type": "target_encoding"}},
        data=data,
        family="gaussian",
    ).fit()

    te_diag = model.target_encoding_diagnostics()
    assert te_diag[0]["prior_weight_spec"] == "auto"
    assert te_diag[0]["prior_weight"] >= 20.0


# ---------------------------------------------------------------------------
# Data-driven response ceiling for unbounded-mean families.
# ---------------------------------------------------------------------------


def test_family_has_unbounded_mean_classification():
    for fam in ("poisson", "Poisson", "gamma", "tweedie", "tweedie(p=1.5)", "nb", "negbinomial"):
        assert _family_has_unbounded_mean(fam), fam
    for fam in ("gaussian", "binomial", "quasibinomial", None, "", "mystery"):
        assert not _family_has_unbounded_mean(fam), fam


def _fit_poisson_rate_model():
    rng = np.random.default_rng(7)
    x = rng.uniform(0.0, 1.0, 400)
    y = rng.poisson(np.exp(0.3 + 0.5 * x)).astype(float)
    data = pl.DataFrame({"y": y, "x": x, "exposure": np.ones_like(x)})
    return rs.glm_dict(
        response="y",
        terms={"x": {"type": "linear"}},
        data=data,
        family="poisson",
        exposure="exposure",
    ).fit()


def test_response_scale_is_recorded():
    model = _fit_poisson_rate_model()
    scale = model._response_scale
    assert scale is not None
    assert np.isfinite(scale) and scale > 0.0
    # It is exactly the largest observed |training response| -- anchored to the
    # response (Rust-computed), NOT the fitted values, so a non-converged fit
    # cannot inflate it.
    assert scale == pytest.approx(float(np.max(np.abs(model._result.response))))
    assert scale == pytest.approx(float(model._result.response_scale))


def test_response_ceiling_is_opt_in_and_auto_caps_unbounded_family():
    model = _fit_poisson_rate_model()
    scale = model._response_scale
    ceiling = scale * _MU_CEILING_FACTOR

    # Large exposure inflates eta by log(exposure); exp(20) keeps eta below the
    # extreme-eta fail-closed threshold (50) so predict() reaches the response
    # ceiling instead of raising, while the raw mean would still explode.
    new_data = pl.DataFrame({"x": [0.5]})
    huge_exposure = np.array([np.exp(20.0)])

    eta = model.predict_linear(new_data, exposure=huge_exposure)
    assert eta[0] < _EXTREME_LOG_ETA_THRESHOLD_FOR_TEST  # no fail-closed raise

    # Default predict is the honest inverse-link: NO cap unless opted in.
    default_mu = model.predict(new_data, exposure=huge_exposure)
    assert default_mu[0] > ceiling
    assert default_mu[0] == pytest.approx(
        model.predict(new_data, exposure=huge_exposure, response_ceiling=None)[0]
    )

    # Opting in caps -- and must never engage silently.
    with pytest.warns(UserWarning, match="response_ceiling='auto' capped 1 of 1"):
        capped = model.predict(new_data, exposure=huge_exposure, response_ceiling="auto")
    assert capped[0] == pytest.approx(ceiling)


def test_response_ceiling_leaves_in_range_predictions_untouched():
    model = _fit_poisson_rate_model()
    new_data = pl.DataFrame({"x": np.linspace(0.0, 1.0, 25)})
    exposure = np.ones(25)
    auto = model.predict(new_data, exposure=exposure, response_ceiling="auto")
    off = model.predict(new_data, exposure=exposure)
    # Ordinary in-range predictions live far below the cap and must be identical.
    assert np.allclose(auto, off)


def test_response_ceiling_explicit_float_overrides_and_applies_to_any_family():
    # Gaussian identity link is a bounded/well-scaled family: "auto" is a no-op,
    # but an explicit float cap is honored regardless of family.
    rng = np.random.default_rng(11)
    x = rng.uniform(0.0, 1.0, 200)
    y = 10.0 + 3.0 * x + rng.normal(0.0, 0.1, 200)
    data = pl.DataFrame({"y": y, "x": x})
    model = rs.glm_dict(
        response="y", terms={"x": {"type": "linear"}}, data=data, family="gaussian"
    ).fit()

    new_data = pl.DataFrame({"x": [0.5]})
    offset = np.array([1_000_000.0])  # identity link -> mean ~ 1e6

    auto = model.predict(new_data, offset=offset, response_ceiling="auto")
    assert auto[0] > 1e5  # "auto" does not cap a bounded-mean family

    explicit = model.predict(new_data, offset=offset, response_ceiling=42.0)
    assert explicit[0] == pytest.approx(42.0)


def test_response_ceiling_survives_serialization_round_trip():
    model = _fit_poisson_rate_model()
    loaded = rs.GLMModel.from_bytes(model.to_bytes())
    # The reference scale is persisted, so an opted-in "auto" cap behaves
    # identically in-session and after deployment via to_bytes/from_bytes.
    assert loaded._response_scale == pytest.approx(model._response_scale)

    new_data = pl.DataFrame({"x": [0.5]})
    huge_exposure = np.array([np.exp(20.0)])
    with pytest.warns(UserWarning, match="response_ceiling='auto' capped"):
        capped_live = model.predict(new_data, exposure=huge_exposure, response_ceiling="auto")
    with pytest.warns(UserWarning, match="response_ceiling='auto' capped"):
        capped_loaded = loaded.predict(new_data, exposure=huge_exposure, response_ceiling="auto")
    assert capped_loaded[0] == pytest.approx(capped_live[0])


def test_internal_scoring_uses_uncapped_mu():
    from rustystats._rustystats import compute_loss_metrics_py

    model = _fit_poisson_rate_model()
    # Eval frame whose exposure pushes mu far beyond the auto cap: reported loss
    # must reflect the model's true mean, not the guardrail-clipped one.
    eval_data = pl.DataFrame(
        {"y": np.ones(4), "x": np.full(4, 0.5), "exposure": np.full(4, np.exp(20.0))}
    )
    y = eval_data["y"].to_numpy()
    mu_uncapped = np.asarray(model.predict(eval_data))
    with pytest.warns(UserWarning, match="response_ceiling='auto' capped"):
        mu_capped = np.asarray(model.predict(eval_data, response_ceiling="auto"))
    assert mu_capped[0] < mu_uncapped[0]  # the cap genuinely engages here

    loss = model.compute_loss(eval_data)
    loss_uncapped = compute_loss_metrics_py(y, mu_uncapped, model.family)["family_loss"]
    loss_capped = compute_loss_metrics_py(y, mu_capped, model.family)["family_loss"]
    assert loss == pytest.approx(loss_uncapped)
    assert loss != pytest.approx(loss_capped)


def test_response_ceiling_rejects_invalid_values():
    model = _fit_poisson_rate_model()
    new_data = pl.DataFrame({"x": [0.5]})
    exposure = np.ones(1)
    for bad in ("tight", -1.0, 0.0, float("inf"), float("nan")):
        with pytest.raises(ValidationError):
            model.predict(new_data, exposure=exposure, response_ceiling=bad)


# ---------------------------------------------------------------------------
# Fit- and prediction-health diagnostics (fit_diagnostics / predict_diagnostics).
# ---------------------------------------------------------------------------


def test_fit_diagnostics_reports_fit_health():
    model = _fit_poisson_rate_model()
    diag = model.fit_diagnostics()

    assert diag["nobs"] == 400
    assert diag["family"] == "poisson"
    assert diag["link"] == "log"
    assert diag["converged"] is True
    assert diag["iterations"] > 0
    assert np.isfinite(diag["deviance"])
    assert diag["nonfinite_eta_count"] == 0
    assert diag["nonfinite_fitted_value_count"] == 0
    assert diag["extreme_log_eta_count"] == 0
    assert diag["warnings"] == []
    for key in ("min", "p50", "p95", "p99", "max"):
        assert np.isfinite(diag["eta"][key])
        assert np.isfinite(diag["fitted_value"][key])


def test_predict_diagnostics_deviance_rows_per_family():
    rng = np.random.default_rng(3)
    n = 60
    x = np.linspace(0.0, 1.0, n)
    cases = {
        "poisson": rng.poisson(1.0 + x).astype(float),
        "gamma": rng.gamma(2.0, 1.0 + x),
        "binomial": rng.binomial(1, 0.3 + 0.4 * x).astype(float),
        "tweedie": np.where(rng.uniform(size=n) < 0.3, 0.0, rng.gamma(2.0, 1.0 + x)),
        "gaussian": 1.0 + 2.0 * x + rng.normal(0.0, 0.1, n),
    }
    for family, y in cases.items():
        data = pl.DataFrame({"y": y, "x": x})
        model = rs.glm_dict(
            response="y", terms={"x": {"type": "linear"}}, data=data, family=family
        ).fit()
        diag = model.predict_diagnostics(data, y=y, weights=np.ones(n), top_n=3)

        assert diag["n"] == n, family
        dev_quantiles = diag["deviance"]
        for key in ("min", "p50", "max"):
            assert np.isfinite(dev_quantiles[key]), family
        rows = diag["top_deviance_rows"]
        assert len(rows) == 3, family
        assert all(set(row) == {"row", "y", "prediction", "eta", "deviance"} for row in rows), (
            family
        )
        # Rows are ranked worst-first.
        devs = [row["deviance"] for row in rows]
        assert devs == sorted(devs, reverse=True), family


def test_prediction_deviance_values_validation():
    model = _fit_poisson_rate_model()
    with pytest.raises(ValidationError, match="same length as predictions"):
        model._prediction_deviance_values(np.ones(3), np.ones(4), None)
    with pytest.raises(ValidationError, match="weights must have the same length"):
        model._prediction_deviance_values(np.ones(4), np.ones(4), np.ones(3))


def test_predict_accepts_complement_column_override():
    model = _fit_poisson_rate_model()
    new_data = pl.DataFrame({"x": [0.2, 0.8], "prior": [1.5, 2.5]})
    exposure = np.ones(2)
    with_prior = model.predict(new_data, complement="prior", exposure=exposure)
    base = model.predict(new_data.drop("prior"), exposure=exposure)
    # For a log link the complement is added on the link scale, so the prior
    # multiplies the predicted mean.
    assert np.allclose(with_prior, base * np.array([1.5, 2.5]))


def test_predict_accepts_complement_model_override_with_exposure():
    model = _fit_poisson_rate_model()
    rng = np.random.default_rng(13)
    x = rng.uniform(0.0, 1.0, 100)
    prior_data = pl.DataFrame({"y": rng.poisson(np.exp(0.1 + x)).astype(float), "x": x})
    prior_model = rs.glm_dict(
        response="y", terms={"x": {"type": "linear"}}, data=prior_data, family="poisson"
    ).fit()

    new_data = pl.DataFrame({"x": [0.2, 0.8]})
    exposure = np.ones(2)
    with_prior = model.predict(new_data, complement=prior_model, exposure=exposure)
    base = model.predict(new_data, exposure=exposure)
    prior_pred = prior_model.predict(new_data)
    # With unit exposure, a model complement contributes its rate multiplicatively.
    assert np.allclose(with_prior, base * prior_pred)


def test_compute_loss_requires_response_column():
    model = _fit_poisson_rate_model()
    with pytest.raises(ValidationError, match="not found in data"):
        model.compute_loss(pl.DataFrame({"x": [0.1, 0.2]}))


def test_explore_supports_array_exposure():
    rng = np.random.default_rng(23)
    n = 80
    x = rng.uniform(0.0, 1.0, n)
    exposure = rng.uniform(0.5, 2.0, n)
    data = pl.DataFrame({"y": rng.poisson(exposure * np.exp(0.2 * x)).astype(float), "x": x})
    spec = rs.glm_dict(
        response="y",
        terms={"x": {"type": "linear"}},
        data=data,
        family="poisson",
        exposure=exposure,
    )
    # Array exposure is materialized into a synthetic column for exploration.
    # (Regression: this path raised NameError until the polars import was fixed.)
    exploration = spec.explore()
    assert exploration is not None


# Local mirror of the library threshold so the test asserts against a literal.
_EXTREME_LOG_ETA_THRESHOLD_FOR_TEST = 50.0
