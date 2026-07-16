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

    # Even after deliberate eta clipping, exp(50) still exceeds the auto
    # response ceiling, which engages loudly rather than silently.
    with pytest.warns(UserWarning, match="response_ceiling='auto' capped"):
        clipped = model.predict(new_data, exposure=huge_exposure, on_extreme_eta="clip")
    assert np.isfinite(clipped).all()
    assert clipped[0] <= np.exp(50.0)

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


def test_response_ceiling_caps_unbounded_family_by_default():
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

    # The auto cap must never engage silently.
    with pytest.warns(UserWarning, match="response_ceiling='auto' capped 1 of 1"):
        capped = model.predict(new_data, exposure=huge_exposure)
    assert capped[0] == pytest.approx(ceiling)

    uncapped = model.predict(new_data, exposure=huge_exposure, response_ceiling=None)
    assert uncapped[0] > ceiling  # the raw mean genuinely exceeds the cap


def test_response_ceiling_leaves_in_range_predictions_untouched():
    model = _fit_poisson_rate_model()
    new_data = pl.DataFrame({"x": np.linspace(0.0, 1.0, 25)})
    exposure = np.ones(25)
    auto = model.predict(new_data, exposure=exposure)
    off = model.predict(new_data, exposure=exposure, response_ceiling=None)
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

    auto = model.predict(new_data, offset=offset)
    assert auto[0] > 1e5  # "auto" does not cap a bounded-mean family

    explicit = model.predict(new_data, offset=offset, response_ceiling=42.0)
    assert explicit[0] == pytest.approx(42.0)


def test_response_ceiling_survives_serialization_round_trip():
    model = _fit_poisson_rate_model()
    loaded = rs.GLMModel.from_bytes(model.to_bytes())
    # The reference scale is persisted, so the default "auto" cap behaves
    # identically in-session and after deployment via to_bytes/from_bytes.
    assert loaded._response_scale == pytest.approx(model._response_scale)

    new_data = pl.DataFrame({"x": [0.5]})
    huge_exposure = np.array([np.exp(20.0)])
    with pytest.warns(UserWarning, match="response_ceiling='auto' capped"):
        capped_live = model.predict(new_data, exposure=huge_exposure)
    with pytest.warns(UserWarning, match="response_ceiling='auto' capped"):
        capped_loaded = loaded.predict(new_data, exposure=huge_exposure)
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
    mu_uncapped = np.asarray(model.predict(eval_data, response_ceiling=None))
    with pytest.warns(UserWarning, match="response_ceiling='auto' capped"):
        mu_capped = np.asarray(model.predict(eval_data))
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


# Local mirror of the library threshold so the test asserts against a literal.
_EXTREME_LOG_ETA_THRESHOLD_FOR_TEST = 50.0
