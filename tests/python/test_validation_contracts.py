import numpy as np
import pytest
from rustystats.exceptions import ValidationError
from rustystats.validation import (
    coerce_to_float64,
    validate_design_matrix,
    validate_glm_inputs,
    validate_offset,
    validate_residual_inputs,
    validate_response,
    validate_weights,
)


def test_coerce_to_float64_reports_non_numeric_values():
    with pytest.raises(ValidationError, match="cannot be converted"):
        coerce_to_float64(np.array([1.0, "bad"], dtype=object), name="premium")


def test_coerce_to_float64_nonfinite_modes_are_explicit():
    with pytest.raises(ValidationError, match="1 NaN"):
        coerce_to_float64(np.array([1.0, np.nan]), name="loss")

    with pytest.raises(ValidationError, match="1 infinite"):
        coerce_to_float64(np.array([1.0, np.inf]), name="loss", allow_nan=True)

    allowed = coerce_to_float64(
        np.array([1.0, np.nan, np.inf]),
        name="loss",
        allow_nan=True,
        allow_inf=True,
    )
    assert np.isnan(allowed[1])
    assert np.isposinf(allowed[2])

    with pytest.raises(ValidationError, match="1 NaN"):
        coerce_to_float64(np.array([1.0, np.nan, np.inf]), name="loss", allow_inf=True)

    with pytest.raises(ValidationError, match="1 infinite"):
        coerce_to_float64(np.array([1.0, np.inf]), name="loss", allow_nan=False)


def test_response_empty_constant_and_family_domain_errors_are_explicit():
    with pytest.raises(ValidationError, match="empty"):
        validate_response(np.array([]), "gaussian")

    with pytest.raises(ValidationError, match="constant"):
        validate_response(np.array([2.0, 2.0]), "gaussian")

    with pytest.raises(ValidationError, match="in \\[0, 1\\]"):
        validate_response(np.array([-0.1, 0.5, 1.2]), "binomial", require_variation=False)

    with pytest.raises(ValidationError, match="strictly positive"):
        validate_response(np.array([0.0, 1.0]), "gamma", require_variation=False)

    with pytest.raises(ValidationError, match="non-negative"):
        validate_response(np.array([0.0, -1.0]), "nb", require_variation=False)

    with pytest.raises(ValidationError, match="Inverse Gaussian"):
        validate_response(np.array([0.0, 1.0]), "inverse_gaussian", require_variation=False)


def test_response_warnings_are_explicit_contracts():
    with pytest.warns(UserWarning, match="non-integer"):
        validate_response(np.array([0.0, 1.5, 2.0]), "poisson")

    with pytest.warns(UserWarning, match="not exactly"):
        validate_response(np.array([0.2, 0.8, 0.2, 0.8]), "binomial")


def test_design_matrix_underdetermined_warning_is_explicit():
    with pytest.warns(UserWarning, match="fewer observations"):
        validated = validate_design_matrix(
            np.array([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]]),
            feature_names=["Intercept", "x1", "x2"],
        )
    assert validated.shape == (2, 3)


def test_design_matrix_shape_errors_are_actionable():
    with pytest.raises(ValidationError, match="2-dimensional"):
        validate_design_matrix(np.array([1.0, 2.0]))

    with pytest.raises(ValidationError, match="no observations"):
        validate_design_matrix(np.empty((0, 2)))

    with pytest.raises(ValidationError, match="no features"):
        validate_design_matrix(np.empty((2, 0)))


def test_weight_zero_mass_contracts():
    assert validate_weights(None, n_obs=4) is None

    with pytest.raises(ValidationError, match="length"):
        validate_weights(np.ones(3), n_obs=4)

    with pytest.raises(ValidationError, match="negative"):
        validate_weights(np.array([1.0, -1.0]), n_obs=2)

    with pytest.raises(ValidationError, match="sum to zero"):
        validate_weights(np.zeros(4), n_obs=4)

    with pytest.warns(UserWarning, match="75.0%"):
        validated = validate_weights(np.array([0.0, 0.0, 0.0, 1.0]), n_obs=4)
    np.testing.assert_allclose(validated, [0.0, 0.0, 0.0, 1.0])


def test_glm_input_validation_distinguishes_exposure_from_link_offset():
    y = np.array([0.0, 1.0])
    x = np.array([[1.0], [2.0]])
    offset = np.array([0.0, 1.0])

    _, _, _, link_offset = validate_glm_inputs(
        y,
        x,
        "poisson",
        offset=offset,
        is_exposure_offset=False,
    )
    np.testing.assert_allclose(link_offset, offset)

    with pytest.raises(ValidationError, match="Exposure must be strictly positive"):
        validate_glm_inputs(y, x, "poisson", offset=offset, is_exposure_offset=True)


def test_offset_validation_checks_lengths_and_log_link_exposure_aliases():
    assert validate_offset(None, n_obs=2, family="poisson") is None

    with pytest.raises(ValidationError, match="length"):
        validate_offset(np.ones(3), n_obs=2, family="poisson")

    for family in ["quasipoisson", "gamma", "negativebinomial", "negative_binomial", "nb"]:
        with pytest.raises(ValidationError, match="strictly positive"):
            validate_offset(np.array([1.0, 0.0]), n_obs=2, family=family, is_exposure=True)

    validated = validate_offset(np.array([0.0, -1.0]), n_obs=2, family="gaussian")
    np.testing.assert_allclose(validated, [0.0, -1.0])


def test_glm_input_validation_rejects_response_design_length_mismatch():
    with pytest.raises(ValidationError, match="design matrix has 3 rows"):
        validate_glm_inputs(
            np.array([0.0, 1.0]),
            np.ones((3, 1)),
            "poisson",
        )


def test_residual_validation_strips_embedded_family_and_checks_eta_shape():
    y = np.array([0.0, 2.0, 3.0])
    eta = np.array([0.1, 0.2, 0.3])
    validated_y, validated_eta, _, _ = validate_residual_inputs(
        y,
        eta,
        "negativebinomial(theta=2.5)",
    )
    np.testing.assert_allclose(validated_y, y)
    np.testing.assert_allclose(validated_eta, eta)

    with pytest.raises(ValidationError, match="eta must be 1-dimensional"):
        validate_residual_inputs(y, eta.reshape(-1, 1), "poisson")

    with pytest.raises(ValidationError, match="eta length"):
        validate_residual_inputs(y, eta[:2], "poisson")


def test_tweedie_extended_boundary_contracts():
    y = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValidationError, match="Use family='gamma' directly"):
        validate_response(y, "tweedie", var_power=2.0, allow_extended_tweedie=False)

    with pytest.raises(ValidationError, match="strictly positive"):
        validate_response(y, "tweedie", var_power=2.0, allow_extended_tweedie=True)

    validated = validate_response(
        np.array([0.1, 1.0, 2.0]),
        "tweedie",
        var_power=2.0,
        allow_extended_tweedie=True,
    )
    np.testing.assert_allclose(validated, [0.1, 1.0, 2.0])


def test_tweedie_regime_contracts_cover_invalid_and_extended_regions():
    with pytest.raises(ValidationError, match="non-negative"):
        validate_response(
            np.array([-0.1, 1.0]),
            "tweedie",
            require_variation=False,
            var_power=None,
        )

    with pytest.raises(ValidationError, match="finite"):
        validate_response(
            np.array([0.0, 1.0]),
            "tweedie",
            require_variation=False,
            var_power=np.inf,
        )

    with pytest.raises(ValidationError, match="open interval"):
        validate_response(
            np.array([0.0, 1.0]),
            "tweedie",
            require_variation=False,
            var_power=0.5,
            allow_extended_tweedie=True,
        )

    with pytest.raises(ValidationError, match="Use family='poisson' directly"):
        validate_response(
            np.array([0.0, 1.0]),
            "tweedie",
            require_variation=False,
            var_power=1.0,
        )

    with pytest.raises(ValidationError, match="non-negative"):
        validate_response(
            np.array([-1.0, 1.0]),
            "tweedie",
            require_variation=False,
            var_power=1.0,
            allow_extended_tweedie=True,
        )

    validated = validate_response(
        np.array([0.0, 1.0]),
        "tweedie(p=1.5)",
        require_variation=False,
        var_power=1.5,
    )
    np.testing.assert_allclose(validated, [0.0, 1.0])
