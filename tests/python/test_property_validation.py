import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from rustystats.exceptions import ValidationError
from rustystats.validation import validate_glm_inputs, validate_response, validate_weights

FINITE_POSITIVE = st.floats(
    min_value=1e-9,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(st.lists(FINITE_POSITIVE, min_size=1, max_size=50))
def test_positive_weights_round_trip(values):
    weights = np.asarray(values, dtype=np.float64)
    validated = validate_weights(weights, n_obs=len(weights))
    np.testing.assert_allclose(validated, weights)


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(st.lists(st.integers(min_value=0, max_value=100), min_size=2, max_size=50))
def test_integer_poisson_responses_round_trip(values):
    if len(set(values)) == 1:
        values[0] += 1
    y = np.asarray(values, dtype=np.float64)
    validated = validate_response(y, "poisson")
    np.testing.assert_allclose(validated, y)


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(
    st.lists(st.floats(min_value=0.0, max_value=100.0, allow_nan=False), min_size=1, max_size=50),
    st.floats(min_value=-1e6, max_value=-1e-9, allow_nan=False, allow_infinity=False),
)
def test_poisson_responses_reject_any_negative_value(nonnegative_values, negative_value):
    y = np.asarray([negative_value, *nonnegative_values], dtype=np.float64)
    with pytest.raises(ValidationError, match="non-negative"):
        validate_response(y, "poisson")


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(st.lists(FINITE_POSITIVE, min_size=2, max_size=50))
def test_positive_exposures_round_trip_through_glm_validation(exposure_values):
    exposure = np.asarray(exposure_values, dtype=np.float64)
    n = len(exposure)
    y = (np.arange(n) % 2).astype(np.float64)
    x = np.ones((n, 1), dtype=np.float64)

    _, _, _, validated_exposure = validate_glm_inputs(
        y,
        x,
        "poisson",
        offset=exposure,
        is_exposure_offset=True,
    )
    np.testing.assert_allclose(validated_exposure, exposure)


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(
    st.lists(FINITE_POSITIVE, min_size=1, max_size=49),
    st.one_of(
        st.just(0.0),
        st.floats(min_value=-1e6, max_value=-1e-9, allow_nan=False, allow_infinity=False),
    ),
)
def test_exposure_validation_rejects_any_nonpositive_value(positive_tail, bad_exposure):
    exposure = np.asarray([bad_exposure, *positive_tail], dtype=np.float64)
    n = len(exposure)
    y = (np.arange(n) % 2).astype(np.float64)
    x = np.ones((n, 1), dtype=np.float64)

    with pytest.raises(ValidationError, match="Exposure must be strictly positive"):
        validate_glm_inputs(
            y,
            x,
            "poisson",
            offset=exposure,
            is_exposure_offset=True,
        )
