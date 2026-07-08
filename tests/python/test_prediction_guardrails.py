import numpy as np
import polars as pl
import pytest
import rustystats as rs
from rustystats.exceptions import PredictionError, ValidationError


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
