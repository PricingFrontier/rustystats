"""Edge cases of the fit options and the training builder's target encoding.

Each test pins an observable outcome: the error a refusal raises, the value an
option resolves to, or parity between today's builder and the frozen design.
"""

from __future__ import annotations

import pickle

import numpy as np
import polars as pl
import pytest
import rustystats as rs
import rustystats.frozen_design as frozen_design
from _design_cases import LIN, _frame, _spec
from rustystats.exceptions import ValidationError
from rustystats.formula import (
    ArrayInputRequired,
    _resolve_family_spec,
    _resolve_robust_types,
    dict_to_parsed_formula,
)
from rustystats.frozen_design import build_design_in_memory
from rustystats.interactions import InteractionBuilder, check_monotone_by_category
from rustystats.splines import SplineTerm


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    return _frame(n=400, seed=13)


def _parse(spec: dict) -> object:
    return dict_to_parsed_formula(
        response=spec["response"],
        terms=spec["terms"],
        interactions=spec.get("interactions"),
        intercept=spec.get("intercept", True),
    )


@pytest.mark.parametrize("cv", [2.5, True, "3"])
def test_a_fold_count_must_be_a_whole_number(frame, cv) -> None:
    model = rs.glm_dict(
        response="y_pois", terms={"x1": LIN}, data=frame, family="poisson", exposure="exposure"
    )
    with pytest.raises(ValidationError, match="cv must be a whole number of folds"):
        model.fit(cv=cv, regularization="ridge")


def test_robust_types_are_normalised_and_listed_once() -> None:
    assert _resolve_robust_types(["hc1", "HC1", "hc3"], True) == ["HC1", "HC3"]
    assert _resolve_robust_types([], True) is None
    with pytest.raises(ValidationError, match="Unknown robust covariance type 'HC9'"):
        _resolve_robust_types(["HC9"], True)


def test_a_theta_in_the_family_name_is_the_fits_theta() -> None:
    assert _resolve_family_spec("negbinomial(theta=2)", 1.5, None) == ("negbinomial", 1.5, 2.0)
    assert _resolve_family_spec("negbinomial(theta=2)", 1.5, 2.0)[2] == 2.0
    with pytest.raises(ValidationError, match="Conflicting Negative Binomial theta values"):
        _resolve_family_spec("negbinomial(theta=2)", 1.5, 3.0)


def test_an_array_input_placeholder_names_its_input_and_pickles_small() -> None:
    placeholder = ArrayInputRequired("offset")
    assert repr(placeholder) == "ArrayInputRequired('offset')"
    restored = pickle.loads(pickle.dumps(placeholder))
    assert isinstance(restored, ArrayInputRequired) and restored.name == "offset"


def test_diagnostics_need_the_training_data_in_memory_too(frame) -> None:
    result = rs.glm_dict(
        response="y_pois", terms={"x1": LIN}, data=frame, family="poisson", exposure="exposure"
    ).fit()
    with pytest.raises(TypeError, match="missing 1 required argument: 'train_data'"):
        result.diagnostics()


def test_a_monotone_check_without_a_formula_only_counts_categoricals() -> None:
    spline = SplineTerm("x2", spline_type="bs", df=5, monotonicity="increasing")
    # Without the parsed formula there are no main effects to check against.
    assert check_monotone_by_category(None, ["cat_a"], spline, single_categorical=True) is None
    with pytest.raises(ValidationError, match="single categorical factor per interaction"):
        check_monotone_by_category(None, ["cat_a", "cat_b"], spline, single_categorical=False)


def test_target_encoding_with_exposure_equals_the_frozen_design(frame) -> None:
    spec = _spec({"brand": {"type": "target_encoding"}, "x1": LIN})
    exposure = frame["exposure"].to_numpy()
    builder = InteractionBuilder(frame)
    y, X, names = builder.build_design_matrix_from_parsed(_parse(spec), exposure=exposure, seed=9)
    frozen, new_y, new_X, new_names = build_design_in_memory(
        frame, _parse(spec), exposure=exposure, seed=9
    )
    np.testing.assert_array_equal(new_X, X)
    np.testing.assert_array_equal(new_y, y)
    assert new_names == names
    # Exposure-weighted statistics keep each label's total exposure, not its count.
    stats = builder._te_stats["brand"]["stats"]
    assert stats == frozen._te_stats["brand"]["stats"]
    label, (total, weight) = next(iter(stats.items()))
    rows = frame.filter(pl.col("brand").cast(pl.String) == label)
    assert weight == pytest.approx(float(rows["exposure"].sum()), rel=1e-12)
    assert total == pytest.approx(float(rows["y_pois"].sum()), rel=1e-12)


def test_an_unseeded_training_build_draws_its_seed(frame, monkeypatch) -> None:
    spec = _spec({"brand": {"type": "target_encoding"}, "x1": LIN})
    monkeypatch.setattr(frozen_design, "draw_target_encoding_seed", lambda: 77)
    _y, drawn, _names = InteractionBuilder(frame).build_design_matrix_from_parsed(_parse(spec))
    _y, seeded, _names = InteractionBuilder(frame).build_design_matrix_from_parsed(
        _parse(spec), seed=77
    )
    np.testing.assert_array_equal(drawn, seeded)
