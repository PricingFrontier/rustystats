"""Direct contracts for multinomial helper utilities."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import rustystats.multinomial as multinomial_mod
from rustystats.exceptions import PredictionError, ValidationError
from rustystats.interactions import TargetEncodingTermSpec
from rustystats.multinomial import (
    ALPHA_MAX_FLOOR,
    MultinomialFoldDesign,
    MultinomialInterceptCalibration,
    MultinomialModel,
    _alternative_columns_raw,
    _alternative_names_by_kind,
    _alternative_needed_columns,
    _append_alternative_specific_terms,
    _apply_alternative_transform,
    _as_bool_array,
    _as_float_array,
    _continuous_factor_bins,
    _derive_classes,
    _extra_needed_columns,
    _masked_softmax,
    _mean_defined,
    _multinomial_alpha_max_from_arrays,
    _multinomial_bound_indices,
    _multinomial_null_probabilities,
    _multinomial_parameter_count,
    _multinomial_probabilities_from_result,
    _multinomial_smooth_penalty_inputs,
    _multinomial_validation_deviance,
    _normalize_alternative_terms,
    _normalize_multinomial_cv_alphas,
    _optional_f1,
    _optional_ratio,
    _resolve_class_matrix,
    _resolve_classes,
    _resolve_regularization,
    _resolve_weights,
    _share_based_null_deviance,
    _slice_class_matrix_override,
    _spec_column_names,
    _spec_has_array,
    _stratified_multinomial_cv_folds,
    _stringify_factor_values,
    _target_encoding_categories,
    _target_encoding_feature_name,
    _target_encoding_options,
    _target_encoding_source_columns,
    _unique_temp_column,
    _validate_multinomial_dense_fit_size,
    _validate_supported_term_spec,
    _validate_supported_terms,
    _weighted_multinomial_gcv,
    _weighted_probability_mix,
    _weighted_standardization_scale,
)
from rustystats.regularization_path import RegularizationPathInfo, RegularizationPathResult


class _DummyMultinomialResult:
    def __init__(self, *, alpha: float = 0.0, l1_ratio: float = 0.0):
        self.params = np.array([[0.0, 0.0], [0.2, -0.3]], dtype=np.float64)
        self.fitted_probabilities = np.array(
            [[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]],
            dtype=np.float64,
        )
        self.linear_predictor = np.zeros((2, 3), dtype=np.float64)
        self.log_likelihood = -3.0
        self.deviance = 6.0
        self.null_deviance = 7.0
        self.iterations = 5
        self.converged = True
        self.solver_status = "converged"
        self.warnings = []
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.prior_weights = np.ones(2, dtype=np.float64)
        self.y_codes = np.array([0, 1], dtype=np.int64)
        self.cov_params_unscaled = np.diag([1.0, 1.0, 1e-4, 1e-4])


class _TinyPredictionBuilder:
    def transform_new_data(self, data: pl.DataFrame) -> np.ndarray:
        return np.column_stack(
            [
                np.ones(len(data), dtype=np.float64),
                data["x"].to_numpy().astype(np.float64),
            ]
        )


def _dummy_multinomial_model(**kwargs) -> MultinomialModel:
    return MultinomialModel(
        result=kwargs.pop("result", _DummyMultinomialResult()),
        response=kwargs.pop("response", "choice"),
        classes=kwargs.pop("classes", ["none", "basic", "premium"]),
        reference=kwargs.pop("reference", "none"),
        feature_names=kwargs.pop("feature_names", ["Intercept", "x"]),
        builder=kwargs.pop("builder", object()),
        terms=kwargs.pop("terms", {"x": {"type": "linear"}}),
        alternative_terms=kwargs.pop("alternative_terms", None),
        interactions=kwargs.pop("interactions", None),
        input_transforms=kwargs.pop("input_transforms", None),
        compiled_input_transforms=kwargs.pop("compiled_input_transforms", None),
        target_encoding_state=kwargs.pop("target_encoding_state", None),
        availability_spec=kwargs.pop("availability_spec", None),
        offset_spec=kwargs.pop("offset_spec", None),
        weights_spec=kwargs.pop("weights_spec", None),
        array_availability_requires_prediction_override=kwargs.pop(
            "array_availability_requires_prediction_override", False
        ),
        array_offset_requires_prediction_override=kwargs.pop(
            "array_offset_requires_prediction_override", False
        ),
        inference_status=kwargs.pop("inference_status", "available"),
        **kwargs,
    )


def test_multinomial_basic_validation_helpers():
    assert _resolve_regularization(0.0, 0.7, None) == (0.0, 0.0)
    assert _resolve_regularization(1.0, 0.7, "ridge") == (1.0, 0.0)
    assert _resolve_regularization(1.0, 0.0, "elastic_net") == (1.0, 0.5)
    with pytest.raises(ValidationError, match="alpha"):
        _resolve_regularization(float("nan"), 0.0, None)
    with pytest.raises(ValidationError, match="l1_ratio"):
        _resolve_regularization(1.0, 2.0, None)
    with pytest.raises(ValidationError, match="regularization"):
        _resolve_regularization(1.0, 0.5, "enet")

    np.testing.assert_allclose(_as_float_array([1, 2], name="x", length=2), [1.0, 2.0])
    with pytest.raises(ValidationError, match="one-dimensional"):
        _as_float_array([[1.0]], name="x")
    with pytest.raises(ValidationError, match="expected 3"):
        _as_float_array([1.0, 2.0], name="x", length=3)
    with pytest.raises(ValidationError, match="finite"):
        _as_float_array([1.0, np.inf], name="x")

    np.testing.assert_array_equal(_as_bool_array([1, 0], name="mask", length=2), [True, False])
    with pytest.raises(ValidationError, match="one-dimensional"):
        _as_bool_array([[True]], name="mask")
    with pytest.raises(ValidationError, match="expected 3"):
        _as_bool_array([True], name="mask", length=3)

    assert _unique_temp_column({"tmp", "tmp_1"}, "tmp") == "tmp_2"
    assert _spec_has_array(None) is False
    assert _spec_has_array(np.ones(2)) is True
    assert _spec_has_array({"a": np.ones(2)}) is True
    assert _spec_has_array("col") is False
    assert _spec_column_names(None) == set()
    assert _spec_column_names("col") == {"col"}
    assert _spec_column_names({"a": "col", "b": 1.0}) == {"col"}
    assert _extra_needed_columns(["raw", {"x": "produced"}], [{"output": "produced"}]) == {"raw"}


def test_multinomial_class_and_weight_resolution_contracts():
    data = pl.DataFrame(
        {
            "choice": ["b", "a", "b", "c"],
            "w": [1.0, 2.0, 0.0, 3.0],
        }
    )
    assert _derive_classes(data["choice"]) == ["a", "b", "c"]
    categorical = pl.Series("choice", ["medium", "low"], dtype=pl.Categorical)
    assert set(_derive_classes(categorical)) == {"low", "medium"}

    classes, reference, class_to_code, y_codes = _resolve_classes(
        ["b", "a", "c"],
        data["choice"],
        ["a", "b", "c"],
        "b",
    )
    assert classes == ["a", "b", "c"]
    assert reference == "b"
    assert class_to_code == {"a": 0, "b": 1, "c": 2}
    np.testing.assert_array_equal(y_codes, [1, 0, 2])

    with pytest.raises(ValidationError, match="at least two classes"):
        _resolve_classes(["a"], data["choice"], ["a"], None)
    with pytest.raises(ValidationError, match="unique"):
        _resolve_classes(["a"], data["choice"], ["a", "a"], None)
    with pytest.raises(ValidationError, match="observed response labels"):
        _resolve_classes(["missing"], data["choice"], ["a", "b"], None)
    with pytest.raises(ValidationError, match="no observed rows"):
        _resolve_classes(["a", "b"], data["choice"], ["a", "b", "c"], None)
    with pytest.raises(ValidationError, match="reference"):
        _resolve_classes(["a", "b"], data["choice"], ["a", "b"], "c")

    effective, is_class_weighted, row_weights = _resolve_weights(
        data,
        "w",
        {"a": 2.0, "c": 0.5},
        ["b", "a", "b", "c"],
    )
    assert is_class_weighted is True
    np.testing.assert_allclose(row_weights, [1.0, 2.0, 0.0, 3.0])
    np.testing.assert_allclose(effective, [1.0, 4.0, 0.0, 1.5])

    row_only, is_class_weighted, original = _resolve_weights(data, np.ones(4), None, ["b"] * 4)
    assert is_class_weighted is False
    np.testing.assert_allclose(row_only, np.ones(4))
    np.testing.assert_allclose(original, np.ones(4))
    with pytest.raises(ValidationError, match="weights"):
        _resolve_weights(data, np.array([1.0, -1.0, 1.0, 1.0]), None, ["b"] * 4)
    with pytest.raises(ValidationError, match="class_weights"):
        _resolve_weights(data, None, {"a": -1.0}, ["a", "b"])


def test_alternative_term_contracts_and_resolution():
    data = pl.DataFrame(
        {
            "price_basic": [100.0, 120.0],
            "price_premium": [200.0, 240.0],
            "rich_basic": [1.0, 2.0],
        }
    )
    raw = {
        "price": {
            "coefficient": "generic",
            "transform": "log",
            "columns": {"basic": "price_basic", "premium": "price_premium"},
        },
        "richness": {
            "coefficient": "class-specific",
            "columns": {"basic": "rich_basic"},
        },
    }
    normalized = _normalize_alternative_terms(raw, ["none", "basic", "premium"], reference="none")

    assert _alternative_columns_raw(raw) == {"price_basic", "price_premium", "rich_basic"}
    assert _alternative_needed_columns(raw, [{"output": "price_basic"}]) == {
        "price_premium",
        "rich_basic",
    }
    assert _alternative_names_by_kind(normalized) == (["price"], ["richness"])

    generic, specific, generic_names, specific_names = __import__(
        "rustystats.multinomial", fromlist=["_resolve_alternative_arrays"]
    )._resolve_alternative_arrays(data, ["none", "basic", "premium"], normalized)

    assert generic_names == ["price"]
    assert specific_names == ["richness"]
    np.testing.assert_allclose(generic[:, 1, 0], np.log([100.0, 120.0]))
    np.testing.assert_allclose(generic[:, 2, 0], np.log([200.0, 240.0]))
    np.testing.assert_allclose(specific[:, 1, 0], [1.0, 2.0])

    with pytest.raises(ValidationError, match="alternative_terms must be a dict"):
        _alternative_columns_raw([])
    with pytest.raises(ValidationError, match="must be a dict"):
        _alternative_columns_raw({"bad": []})
    with pytest.raises(ValidationError, match="non-empty columns"):
        _alternative_columns_raw({"bad": {"columns": {}}})
    with pytest.raises(ValidationError, match="column name"):
        _alternative_columns_raw({"bad": {"columns": {"a": 1}}})
    with pytest.raises(ValidationError, match="duplicate"):
        _normalize_alternative_terms(
            {1: {"columns": {"a": "x"}}, "1": {"columns": {"a": "y"}}}, ["a"]
        )
    with pytest.raises(ValidationError, match="coefficient"):
        _normalize_alternative_terms(
            {"bad": {"coefficient": "other", "columns": {"a": "x"}}}, ["a"]
        )
    with pytest.raises(ValidationError, match="transform"):
        _normalize_alternative_terms({"bad": {"transform": "sqrt", "columns": {"a": "x"}}}, ["a"])
    with pytest.raises(ValidationError, match="unknown class"):
        _normalize_alternative_terms({"bad": {"columns": {"missing": "x"}}}, ["a"])
    with pytest.raises(ValidationError, match="reference class"):
        _normalize_alternative_terms(
            {"bad": {"coefficient": "class_specific", "columns": {"a": "x"}}},
            ["a", "b"],
            reference="a",
        )
    with pytest.raises(ValidationError, match="non-positive"):
        _apply_alternative_transform(np.array([1.0, 0.0]), term_name="price", transform="log")
    with pytest.raises(AssertionError, match="unsupported"):
        _apply_alternative_transform(np.array([1.0]), term_name="price", transform="sqrt")
    with pytest.raises(ValidationError, match="requires column"):
        __import__(
            "rustystats.multinomial", fromlist=["_resolve_alternative_arrays"]
        )._resolve_alternative_arrays(
            data,
            ["none", "basic"],
            {
                "x": {
                    "coefficient": "generic",
                    "transform": "identity",
                    "columns": {"basic": "missing"},
                }
            },
        )


def test_target_encoding_and_supported_term_contracts():
    data = pl.DataFrame({"brand": ["A", "B"], "region": ["N", "S"], "grp": ["A:N", "B:S"]})
    te = TargetEncodingTermSpec(var_name="brand", interaction_vars=["brand", "region"])

    assert _target_encoding_feature_name(te) == "TE(brand:region)"
    assert _target_encoding_source_columns(
        {"grp": {"type": "target_encoding", "variable": "brand_score"}},
        [{"brand": {}, "region": {}, "target_encoding": True}],
        [{"output": "brand_score"}],
    ) == {"brand", "region"}
    assert _target_encoding_categories(data, te) == ["A:N", "B:S"]
    with pytest.raises(PredictionError, match="missing columns"):
        _target_encoding_categories(
            data, TargetEncodingTermSpec("brand", interaction_vars=["missing", "region"])
        )
    with pytest.raises(PredictionError, match="requires column"):
        _target_encoding_categories(data, TargetEncodingTermSpec("missing"))

    assert _target_encoding_options({"prior_weight": 2.0, "n_permutations": 3}, context="main") == (
        2.0,
        3,
    )
    with pytest.raises(ValidationError, match="mode"):
        _target_encoding_options({"mode": "other"}, context="main")
    with pytest.raises(ValidationError, match="Unknown key"):
        _target_encoding_options(
            {"type": "target_encoding", "extra": 1}, context="main", allowed_keys={"type"}
        )
    with pytest.raises(ValidationError, match="prior_weight must be numeric"):
        _target_encoding_options({"prior_weight": object()}, context="main")
    with pytest.raises(ValidationError, match="n_permutations must be an integer"):
        _target_encoding_options({"n_permutations": object()}, context="main")
    with pytest.raises(ValidationError, match="finite and non-negative"):
        _target_encoding_options({"prior_weight": -1.0}, context="main")
    with pytest.raises(ValidationError, match="positive"):
        _target_encoding_options({"n_permutations": 0}, context="main")

    _validate_supported_term_spec(
        "x", {"type": "bs", "df": 4, "monotonicity": "increasing"}, context="main"
    )
    with pytest.raises(ValidationError, match="not yet supported"):
        _validate_supported_term_spec("x", {"type": "s"}, context="main")
    with pytest.raises(ValidationError, match="monotonicity"):
        _validate_supported_term_spec(
            "x", {"type": "linear", "monotonicity": "flat"}, context="main"
        )
    with pytest.raises(ValidationError, match="only supported"):
        _validate_supported_term_spec(
            "x", {"type": "linear", "monotonicity": "increasing"}, context="interaction"
        )
    with pytest.raises(ValidationError, match="monotonic smooth"):
        _validate_supported_term_spec(
            "x", {"type": "bs", "k": 5, "monotonicity": "increasing"}, context="main"
        )
    with pytest.raises(ValidationError, match="natural splines"):
        _validate_supported_term_spec(
            "x", {"type": "ns", "df": 4, "monotonicity": "increasing"}, context="main"
        )
    with pytest.raises(ValidationError, match="categorical"):
        _validate_supported_term_spec(
            "x", {"type": "categorical", "monotonicity": "increasing"}, context="main"
        )
    with pytest.raises(ValidationError, match="term type"):
        _validate_supported_term_spec(
            "x", {"type": "unknown", "monotonicity": "increasing"}, context="main"
        )
    with pytest.raises(ValidationError, match="automatic smooth penalties"):
        _validate_supported_term_spec("x", {"type": "bs", "k": 5}, context="interaction")
    with pytest.raises(ValidationError, match="target_encoding factors"):
        _validate_supported_terms({}, [{"brand": {"type": "target_encoding"}}])
    with pytest.raises(ValidationError, match="must be a dict"):
        _validate_supported_terms({}, [{"brand": "bad"}])


def test_class_matrix_probability_and_metric_helpers():
    data = pl.DataFrame({"a": [1.0, 2.0], "avail": [True, False]})
    classes = ["none", "basic"]

    np.testing.assert_array_equal(
        _resolve_class_matrix(
            data, None, classes, default=True, dtype=bool, name="availability", allow_arrays=True
        ),
        [[True, True], [True, True]],
    )
    with pytest.raises(PredictionError, match="fit-time data"):
        _resolve_class_matrix(
            data,
            np.ones((2, 2), dtype=bool),
            classes,
            default=True,
            dtype=bool,
            name="availability",
            allow_arrays=False,
        )
    with pytest.raises(ValidationError, match="shape"):
        _resolve_class_matrix(
            data,
            np.ones((2, 1), dtype=bool),
            classes,
            default=True,
            dtype=bool,
            name="availability",
            allow_arrays=True,
        )
    with pytest.raises(ValidationError, match="dict"):
        _resolve_class_matrix(
            data, "bad", classes, default=True, dtype=bool, name="availability", allow_arrays=True
        )
    with pytest.raises(PredictionError, match="not present"):
        _resolve_class_matrix(
            data,
            {"basic": "missing"},
            classes,
            default=True,
            dtype=bool,
            name="availability",
            allow_arrays=True,
        )
    np.testing.assert_array_equal(
        _resolve_class_matrix(
            data,
            {"none": False, "basic": "avail"},
            classes,
            default=True,
            dtype=bool,
            name="availability",
            allow_arrays=True,
        ),
        [[False, True], [False, False]],
    )
    np.testing.assert_allclose(
        _resolve_class_matrix(
            data,
            {"basic": np.array([0.1, 0.2])},
            classes,
            default=0.0,
            dtype=float,
            name="offset",
            allow_arrays=True,
        ),
        [[0.0, 0.1], [0.0, 0.2]],
    )
    with pytest.raises(PredictionError, match="fit-time data"):
        _resolve_class_matrix(
            data,
            {"basic": np.array([0.1, 0.2])},
            classes,
            default=0.0,
            dtype=float,
            name="offset",
            allow_arrays=False,
        )
    with pytest.raises(ValidationError, match="length 2"):
        _resolve_class_matrix(
            data,
            {"basic": np.ones(3)},
            classes,
            default=0.0,
            dtype=float,
            name="offset",
            allow_arrays=True,
        )

    assert (
        _slice_class_matrix_override(None, start=0, stop=1, n_rows=2, n_classes=2, name="offset")
        is None
    )
    np.testing.assert_allclose(
        _slice_class_matrix_override(
            np.arange(4).reshape(2, 2), start=1, stop=2, n_rows=2, n_classes=2, name="offset"
        ),
        [[2, 3]],
    )
    assert (
        _slice_class_matrix_override("col", start=0, stop=1, n_rows=2, n_classes=2, name="offset")
        == "col"
    )
    sliced = _slice_class_matrix_override(
        {"basic": np.array([1.0, 2.0]), "none": False, "other": 1.5},
        start=1,
        stop=2,
        n_rows=2,
        n_classes=2,
        name="offset",
    )
    np.testing.assert_allclose(sliced["basic"], [2.0])
    with pytest.raises(ValidationError, match="shape"):
        _slice_class_matrix_override(
            np.ones((2, 1)), start=0, stop=1, n_rows=2, n_classes=2, name="offset"
        )
    with pytest.raises(ValidationError, match="length 2"):
        _slice_class_matrix_override(
            {"basic": np.ones(3)}, start=0, stop=1, n_rows=2, n_classes=2, name="offset"
        )

    logits = np.array([[0.0, 1.0], [3.0, -3.0]])
    probs = _masked_softmax(logits, np.array([[True, True], [True, False]]))
    np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0])
    np.testing.assert_allclose(probs[1], [1.0, 0.0])
    with pytest.raises(PredictionError, match="shape"):
        _masked_softmax(logits, np.ones((2, 3), dtype=bool))
    with pytest.raises(PredictionError, match="no classes"):
        _masked_softmax(logits, np.array([[False, False], [True, True]]))

    with pytest.raises(ValidationError, match="positive total"):
        _weighted_probability_mix(np.ones((2, 2)), np.zeros(2))
    np.testing.assert_allclose(
        _weighted_probability_mix(np.array([[0.2, 0.8], [0.6, 0.4]]), np.array([1.0, 3.0])),
        [0.5, 0.5],
    )
    assert _share_based_null_deviance({"a": 0.0}, 0.0) is None
    assert _share_based_null_deviance({"a": 2.0, "b": 2.0}, 4.0) == pytest.approx(5.545177444)
    assert _optional_ratio(1.0, 0.0) is None
    assert _optional_ratio(1.0, 2.0) == 0.5
    assert _optional_f1(None, 0.5) is None
    assert _optional_f1(0.0, 0.0) is None
    assert _optional_f1(0.5, 0.25) == pytest.approx(1.0 / 3.0)
    assert _mean_defined([None, float("nan")]) is None
    assert _mean_defined([1.0, None, 3.0]) == 2.0


def test_multinomial_numerical_helpers_and_cv_contracts():
    existing = np.zeros((2, 3, 0))
    extra = np.ones((2, 3, 1))
    assert _append_alternative_specific_terms(existing, extra) is extra
    assert _append_alternative_specific_terms(extra, existing) is extra
    assert _append_alternative_specific_terms(extra, extra).shape == (2, 3, 2)
    assert _weighted_standardization_scale(np.array([1.0, 2.0]), np.zeros(2)) == 1.0
    assert _weighted_standardization_scale(np.array([1.0, 1.0]), np.ones(2)) == 1.0
    assert _weighted_standardization_scale(np.array([1.0, 3.0]), np.ones(2)) == 1.0

    with pytest.raises(ValidationError, match="l1_ratio"):
        _multinomial_alpha_max_from_arrays(
            np.array([0, 1]),
            np.ones((2, 1)),
            np.zeros((2, 2, 0)),
            np.zeros((2, 2, 0)),
            np.ones((2, 2), dtype=bool),
            np.zeros((2, 2)),
            None,
            2,
            0,
            l1_ratio=float("nan"),
            fit_intercept=True,
        )
    assert (
        _multinomial_alpha_max_from_arrays(
            np.array([0, 1]),
            np.ones((2, 1)),
            np.zeros((2, 2, 0)),
            np.zeros((2, 2, 0)),
            np.ones((2, 2), dtype=bool),
            np.zeros((2, 2)),
            None,
            2,
            0,
            l1_ratio=1.0,
            fit_intercept=True,
        )
        == ALPHA_MAX_FLOOR
    )
    assert (
        _multinomial_alpha_max_from_arrays(
            np.array([0, 1]),
            np.ones((2, 1)),
            np.ones((2, 2, 1)),
            np.zeros((2, 2, 0)),
            np.ones((2, 2), dtype=bool),
            np.zeros((2, 2)),
            None,
            2,
            0,
            l1_ratio=0.0,
            fit_intercept=True,
        )
        >= ALPHA_MAX_FLOOR
    )
    assert (
        _multinomial_alpha_max_from_arrays(
            np.array([0, 1]),
            np.ones((2, 1)),
            np.zeros((2, 2, 0)),
            np.zeros((2, 2, 0)),
            np.ones((2, 2), dtype=bool),
            np.zeros((2, 2)),
            None,
            2,
            0,
            l1_ratio=0.0,
            fit_intercept=True,
        )
        == ALPHA_MAX_FLOOR
    )

    np.testing.assert_allclose(
        _multinomial_null_probabilities(
            np.array([0, 1]),
            n_classes=2,
            reference_index=0,
            availability=np.ones((2, 2), dtype=bool),
            offset=np.array([[0.0, 1.0], [0.0, 0.0]]),
            weights=None,
            fit_intercept=False,
        ).sum(axis=1),
        [1.0, 1.0],
    )
    assert _multinomial_parameter_count(2, 3, 1, 4) == 13
    with pytest.raises(ValidationError, match="q=12"):
        _validate_multinomial_dense_fit_size(
            n_shared=4,
            n_classes=3,
            n_alt_generic=0,
            n_alt_specific=2,
            hessian_memory_limit_bytes=10_000,
            max_dense_parameters=10,
            context="fit",
        )
    with pytest.raises(ValidationError, match="dense Hessian"):
        _validate_multinomial_dense_fit_size(
            n_shared=4,
            n_classes=3,
            n_alt_generic=0,
            n_alt_specific=2,
            hessian_memory_limit_bytes=100,
            max_dense_parameters=100,
            context="fit",
        )

    assert _weighted_multinomial_gcv(1.0, 10.0, None) == float("inf")
    assert _weighted_multinomial_gcv(float("nan"), 10.0, 2.0) == float("inf")
    assert _weighted_multinomial_gcv(1.0, 2.0, 2.0) == float("inf")
    assert _weighted_multinomial_gcv(1.0, 10.0, 2.0) == pytest.approx(10.0 / 64.0)

    mismatch_model = SimpleNamespace(
        _builder=SimpleNamespace(get_smooth_terms=lambda: ([object()], []))
    )
    with pytest.raises(ValidationError, match="smooth metadata"):
        _multinomial_smooth_penalty_inputs(mismatch_model)

    monotone_model = SimpleNamespace(
        _builder=SimpleNamespace(
            get_smooth_terms=lambda: (
                [SimpleNamespace(monotonicity="increasing", spline_type="bs", var_name="x")],
                [(0, 2)],
            )
        )
    )
    with pytest.raises(ValidationError, match="monotonic smooth"):
        _multinomial_smooth_penalty_inputs(monotone_model)

    empty_range_model = SimpleNamespace(
        _builder=SimpleNamespace(
            get_smooth_terms=lambda: (
                [SimpleNamespace(monotonicity=None, spline_type="bs", var_name="x")],
                [(2, 2)],
            )
        )
    )
    with pytest.raises(ValidationError, match="empty"):
        _multinomial_smooth_penalty_inputs(empty_range_model)

    with pytest.raises(ValidationError, match="at least 2"):
        _stratified_multinomial_cv_folds(np.array([0, 1]), None, 2, cv=1, seed=1)
    with pytest.raises(ValidationError, match="cannot exceed"):
        _stratified_multinomial_cv_folds(np.array([0, 1]), None, 2, cv=3, seed=1)
    with pytest.raises(ValidationError, match="finite and non-negative"):
        _stratified_multinomial_cv_folds(np.array([0, 1]), np.array([1.0, -1.0]), 2, cv=2, seed=1)
    with pytest.raises(ValidationError, match="zero positive effective weight"):
        _stratified_multinomial_cv_folds(
            np.array([0, 0, 1, 1]), np.array([1.0, 1.0, 0.0, 0.0]), 2, cv=2, seed=1
        )
    folds = _stratified_multinomial_cv_folds(np.array([0, 0, 1, 1]), None, 2, cv=2, seed=1)
    assert len(folds) == 2
    assert all(set(np.array([0, 0, 1, 1])[train]) == {0, 1} for train, _ in folds)

    result = type(
        "R",
        (),
        {
            "alternative_generic_coefficients": np.array([0.1]),
            "alternative_specific_coefficients": np.array([[0.2]]),
            "params": np.array([[0.5, -0.25]]),
        },
    )()
    probs = _multinomial_probabilities_from_result(
        result,
        x=np.array([[1.0, 2.0]]),
        alternative_generic=np.ones((1, 2, 1)),
        alternative_specific=np.ones((1, 2, 1)),
        availability=np.ones((1, 2), dtype=bool),
        offset=np.zeros((1, 2)),
        n_classes=2,
        reference_index=0,
    )
    np.testing.assert_allclose(probs.sum(axis=1), [1.0])
    assert _multinomial_validation_deviance(probs, np.array([1]), None) >= 0.0
    assert _multinomial_validation_deviance(probs, np.array([1]), np.zeros(1)) == float("inf")
    assert _multinomial_bound_indices(
        ["Intercept", "pos(x)", "neg(z)"], n_classes=3, reference_index=1
    ) == (
        [1, 4],
        [2, 5],
        [1],
        [2],
    )

    with pytest.raises(ValidationError, match="expects a MultinomialDict"):
        multinomial_mod.multinomial_alpha_max(object())


def test_multinomial_cv_alpha_normalization_edge_contracts(monkeypatch):
    model = type(
        "M",
        (),
        {
            "classes_": ["none", "basic"],
            "reference_index_": 0,
            "intercept": True,
            "_target_encoding_state": None,
        },
    )()
    fold = MultinomialFoldDesign(
        x_train=np.ones((2, 1)),
        x_val=np.ones((1, 1)),
        alternative_generic_train=np.zeros((2, 2, 0)),
        alternative_generic_val=np.zeros((1, 2, 0)),
        alternative_specific_train=np.zeros((2, 2, 0)),
        alternative_specific_val=np.zeros((1, 2, 0)),
        availability_train=np.ones((2, 2), dtype=bool),
        availability_val=np.ones((1, 2), dtype=bool),
        offset_train=np.zeros((2, 2)),
        offset_val=np.zeros((1, 2)),
        weights_train=None,
        weights_val=None,
        y_train=np.array([0, 1]),
        y_val=np.array([1]),
        feature_names=["Intercept"],
        preprocessing_state=object(),
    )

    with pytest.raises(ValidationError, match="alphas"):
        _normalize_multinomial_cv_alphas(
            model,
            [fold],
            1.0,
            alphas=np.ones((1, 1)),
            n_alphas=3,
            alpha_min_ratio=0.1,
            include_unregularized=False,
            standardize=True,
        )
    with pytest.raises(ValidationError, match="finite and non-negative"):
        _normalize_multinomial_cv_alphas(
            model,
            [fold],
            1.0,
            alphas=[1.0, -0.1],
            n_alphas=3,
            alpha_min_ratio=0.1,
            include_unregularized=False,
            standardize=True,
        )
    with pytest.raises(ValidationError, match="n_alphas"):
        _normalize_multinomial_cv_alphas(
            model,
            [fold],
            1.0,
            alphas=None,
            n_alphas=0,
            alpha_min_ratio=0.1,
            include_unregularized=False,
            standardize=True,
        )
    with pytest.raises(ValidationError, match="alpha_min_ratio"):
        _normalize_multinomial_cv_alphas(
            model,
            [fold],
            1.0,
            alphas=None,
            n_alphas=3,
            alpha_min_ratio=1.0,
            include_unregularized=False,
            standardize=True,
        )

    monkeypatch.setattr(
        multinomial_mod,
        "_multinomial_alpha_max_from_arrays",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(ValidationError, match="failed to compute alpha_max"):
        _normalize_multinomial_cv_alphas(
            model,
            [fold],
            1.0,
            alphas=None,
            n_alphas=3,
            alpha_min_ratio=0.1,
            include_unregularized=False,
            standardize=True,
        )

    model._target_encoding_state = object()
    path = _normalize_multinomial_cv_alphas(
        model,
        [fold],
        1.0,
        alphas=None,
        n_alphas=3,
        alpha_min_ratio=0.1,
        include_unregularized=True,
        standardize=True,
    )
    assert path[-1] == 0.0


def test_multinomial_calibration_object_validation_and_prediction_contracts():
    valid = MultinomialInterceptCalibration(
        classes=["none", "basic"],
        reference="none",
        shifts={"none": 0.0, "basic": 0.2},
        actual_class_mix=None,
    )
    assert valid.actual_class_mix == {}
    np.testing.assert_allclose(valid.adjust_logits(np.zeros((2, 2))), [[0.0, 0.2], [0.0, 0.2]])
    np.testing.assert_allclose(
        valid.predict_proba_from_logits(np.zeros((1, 2))).sum(axis=1),
        [1.0],
    )

    invalid_kwargs = [
        {"method": "other"},
        {"classes": ["only"], "reference": "only", "shifts": {"only": 0.0}},
        {"classes": ["a", "a"], "reference": "a", "shifts": {"a": 0.0}},
        {"reference": "missing"},
        {"shifts": {"none": 1.0, "basic": 0.2}},
        {"nobs": -1},
        {"total_weight": -1.0},
        {"iterations": -1},
        {"shifts": []},
        {"shifts": {"none": 0.0}},
        {"shifts": {"none": 0.0, "basic": np.inf}},
        {"actual_class_mix": []},
        {"actual_class_mix": {"other": 0.1}},
        {"actual_class_mix": {"basic": np.nan}},
    ]
    for kwargs in invalid_kwargs:
        payload = {
            "classes": ["none", "basic"],
            "reference": "none",
            "shifts": {"none": 0.0, "basic": 0.2},
        }
        payload.update(kwargs)
        with pytest.raises(ValidationError):
            MultinomialInterceptCalibration(**payload)

    with pytest.raises(ValidationError, match="shape"):
        valid.adjust_logits(np.zeros((2, 3)))
    with pytest.raises(ValidationError, match="payload"):
        MultinomialInterceptCalibration.from_dict([])

    model = type("Model", (), {"predict_proba": lambda self, *_args, **kwargs: kwargs})()
    assert valid.predict_proba(model, pl.DataFrame({"x": [1]}))["calibration"] is valid


def test_multinomial_prediction_formats_calibration_and_tier_mix_edges():
    model = _dummy_multinomial_model(builder=_TinyPredictionBuilder())
    data = pl.DataFrame({"x": [0.0, 1.0, 2.0], "w": [1.0, 2.0, 3.0]})

    logits = model.decision_function(data, include_reference=False)
    assert logits.shape == (3, 2)
    probabilities = model.predict_proba(data)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    proba_frame = model.predict_proba(data, return_format="polars")
    assert proba_frame.columns == ["prob_none", "prob_basic", "prob_premium"]
    with pytest.raises(ValidationError, match="return_format"):
        model.predict_proba(data, return_format="records")

    log_frame = model.predict_log_proba(data, return_format="polars")
    assert log_frame.columns == ["log_prob_none", "log_prob_basic", "log_prob_premium"]
    with pytest.raises(ValidationError, match="return_format"):
        model.predict_log_proba(data, return_format="records")

    assert model.predict(data).shape == (3,)
    top_k = model.predict_top_k(data, k=2)
    assert top_k.columns == ["class_1", "prob_1", "class_2", "prob_2"]
    with pytest.raises(ValidationError, match="k must be"):
        model.predict_top_k(data, k=0)

    mix = model.tier_mix(data, weights="w")
    assert set(mix) == set(model.classes_)
    mix_frame = model.tier_mix(data, weights=np.array([1.0, 1.0, 1.0]), return_format="polars")
    assert mix_frame.columns == ["class", "probability"]
    with pytest.raises(ValidationError, match="positive total"):
        model.tier_mix(data, weights=np.zeros(3))
    with pytest.raises(ValidationError, match="return_format"):
        model.tier_mix(data, return_format="records")
    with pytest.raises(PredictionError, match="weights column"):
        model.tier_mix(pl.DataFrame({"x": [1.0]}), weights="w")

    calibration = MultinomialInterceptCalibration(
        classes=model.classes_,
        reference=model.reference_,
        shifts={"none": 0.0, "basic": 0.1, "premium": -0.2},
    )
    shifted = model.predict_proba(data, calibration=calibration)
    assert shifted.shape == probabilities.shape
    with pytest.raises(ValidationError, match="calibration must"):
        model._calibration_shift_vector(object())
    with pytest.raises(ValidationError, match="classes"):
        model._calibration_shift_vector(
            MultinomialInterceptCalibration(
                classes=["none", "basic"],
                reference="none",
                shifts={"none": 0.0, "basic": 0.1},
            )
        )
    with pytest.raises(ValidationError, match="reference"):
        model._calibration_shift_vector(
            MultinomialInterceptCalibration(
                classes=model.classes_,
                reference="basic",
                shifts={"none": 0.1, "basic": 0.0, "premium": -0.1},
            )
        )

    no_builder = _dummy_multinomial_model(builder=None)
    with pytest.raises(PredictionError, match="no stored design builder"):
        no_builder.predict_proba(data)

    transformed = _dummy_multinomial_model(
        builder=object(),
        terms={"x_ctr": {"type": "linear"}},
        input_transforms=[
            {
                "type": "center",
                "name": "x_center",
                "sources": ["x"],
                "output": "x_ctr",
                "center": 1.5,
                "metadata": None,
            }
        ],
    )
    prepared = transformed._prepare_prediction_data(
        pl.DataFrame({"x": [1.0, 2.0], "x_ctr": [99.0, 99.0]}),
        availability=None,
        offset=None,
    )
    np.testing.assert_allclose(prepared["x_ctr"].to_numpy(), [-0.5, 0.5])

    transformed_diag = _dummy_multinomial_model(
        input_transforms=[
            {
                "type": "center",
                "name": "x_center",
                "sources": ["x"],
                "output": "x_ctr",
                "center": 1.5,
                "metadata": None,
            }
        ],
        weights_spec="w",
    )
    diagnostic_data = transformed_diag._collect_diagnostic_data(
        pl.DataFrame(
            {
                "choice": ["none", "basic"],
                "x": [1.0, 2.0],
                "x_ctr": [99.0, 99.0],
                "w": [1.0, 1.0],
            }
        ),
        categorical_factors=[],
        continuous_factors=["x_ctr"],
    )
    np.testing.assert_allclose(diagnostic_data["x_ctr"].to_numpy(), [-0.5, 0.5])


def test_multinomial_diagnostic_and_fit_calibration_failure_edges(monkeypatch):
    model = _dummy_multinomial_model(builder=_TinyPredictionBuilder())
    data = pl.DataFrame(
        {
            "choice": ["none", "basic", "premium"],
            "x": [0.0, 1.0, 2.0],
            "w": [1.0, 2.0, 3.0],
            "bad_w": [1.0, -1.0, 1.0],
        }
    )

    with pytest.raises(ValidationError, match="labels not present"):
        model._class_codes_from_values(["missing"], name="choice")
    with pytest.raises(ValidationError, match="column name"):
        model._diagnostic_weight_array(data, weights_override=np.ones(3))
    with pytest.raises(PredictionError, match="weights column"):
        model._diagnostic_weight_array(data, weights_override="missing")
    with pytest.raises(ValidationError, match="finite and non-negative"):
        model._diagnostic_weight_array(data, weights_override="bad_w")

    array_weighted = _dummy_multinomial_model(array_weighted=True)
    with pytest.raises(PredictionError, match="array weight"):
        array_weighted._diagnostic_weight_array(data)

    no_response = _dummy_multinomial_model(response=None)
    with pytest.raises(PredictionError, match="no stored response"):
        no_response._collect_diagnostic_data(
            data,
            categorical_factors=[],
            continuous_factors=[],
        )

    with pytest.raises(PredictionError, match="response length mismatch"):
        model._dataset_diagnostics_from_arrays(
            name="bad",
            probabilities=np.ones((2, 3)) / 3.0,
            y_codes=np.array([0]),
            weights=np.ones(2),
        )
    with pytest.raises(PredictionError, match="weight length mismatch"):
        model._dataset_diagnostics_from_arrays(
            name="bad",
            probabilities=np.ones((2, 3)) / 3.0,
            y_codes=np.array([0, 1]),
            weights=np.ones(1),
        )
    with pytest.raises(ValidationError, match="finite and non-negative"):
        model._dataset_diagnostics_from_arrays(
            name="bad",
            probabilities=np.ones((2, 3)) / 3.0,
            y_codes=np.array([0, 1]),
            weights=np.array([1.0, -1.0]),
        )
    with pytest.raises(PredictionError, match="zero total"):
        model._dataset_diagnostics_from_arrays(
            name="bad",
            probabilities=np.ones((2, 3)) / 3.0,
            y_codes=np.array([0, 1]),
            weights=np.zeros(2),
        )

    with pytest.raises(PredictionError, match="diagnostic factor"):
        model._factor_diagnostics(
            dataset_name="train",
            data=data,
            y_codes=np.array([0, 1, 2]),
            probabilities=np.ones((3, 3)) / 3.0,
            weights=np.ones(3),
            categorical_factors=["missing"],
            continuous_factors=[],
        )
    skipped = model._factor_diagnostics(
        dataset_name="train",
        data=data,
        y_codes=np.array([0, 1, 2]),
        probabilities=np.ones((3, 3)) / 3.0,
        weights=np.array([1.0, 0.0, 1.0]),
        categorical_factors=["choice"],
        continuous_factors=[],
    )
    assert {row["level"] for row in skipped} == {"none", "premium"}

    row_mismatch = _dummy_multinomial_model(builder=_TinyPredictionBuilder())
    row_mismatch.predict_proba = lambda *_args, **_kwargs: np.ones((4, 3)) / 3.0
    with pytest.raises(PredictionError, match="row counts"):
        row_mismatch._diagnostic_arrays_from_data(
            data,
            categorical_factors=[],
            continuous_factors=[],
        )

    with pytest.raises(PredictionError, match="weights column"):
        model._calibration_weights(data, "missing")
    with pytest.raises(ValidationError, match="finite and non-negative"):
        model._calibration_weights(data, np.array([1.0, -1.0, 1.0]))
    with pytest.raises(ValidationError, match="positive total"):
        model._calibration_weights(data, np.zeros(3))

    with pytest.raises(ValidationError, match="marked unavailable"):
        model.fit_calibration(
            data,
            availability={"basic": np.array([True, False, True])},
        )

    shape_mismatch = _dummy_multinomial_model(builder=_TinyPredictionBuilder())
    monkeypatch.setattr(
        shape_mismatch,
        "decision_function",
        lambda *_args, **_kwargs: np.zeros((3, 2)),
    )
    with pytest.raises(PredictionError, match="availability shapes"):
        shape_mismatch.fit_calibration(data)


def test_multinomial_scenario_helper_edge_contracts():
    model = _dummy_multinomial_model(builder=_TinyPredictionBuilder())
    base = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "segment": ["A", "B", "B"],
            "value_none": [0.0, 0.0, 0.0],
            "value_basic": [1.0, 2.0, 3.0],
        }
    )
    scenario = base.with_columns(pl.Series("value_basic", [1.5, 2.5, 3.5]))
    probs = np.ones((3, 3), dtype=np.float64) / 3.0
    weights = np.array([1.0, 0.0, 2.0])

    with pytest.raises(PredictionError, match="value column"):
        model._scenario_expected_value(
            base,
            scenario,
            probs,
            probs,
            weights,
            {"basic": "missing"},
        )
    with pytest.raises(PredictionError, match="scenario data"):
        model._scenario_expected_value(
            base,
            scenario.drop("value_basic"),
            probs,
            probs,
            weights,
            {"basic": "value_basic"},
        )

    expected_value = model._scenario_expected_value(
        base,
        scenario,
        probs,
        probs,
        weights,
        {"none": "value_none", "basic": "value_basic"},
    )
    assert expected_value is not None
    assert expected_value["scenario"] > expected_value["base"]

    with pytest.raises(PredictionError, match="scenario factor"):
        model._scenario_segment_mix(
            base,
            probs,
            probs,
            weights,
            categorical_factors=["missing"],
            continuous_factors=[],
        )
    rows = model._scenario_segment_mix(
        base,
        probs,
        probs,
        weights,
        categorical_factors=["segment"],
        continuous_factors=[],
    )
    assert {row["level"] for row in rows} == {"A", "B"}

    scenario_model = _dummy_multinomial_model(
        builder=_TinyPredictionBuilder(),
        alternative_terms={
            "price": {
                "columns": {"basic": "price_basic"},
                "coefficient": "generic",
                "transform": "identity",
            }
        },
    )
    scenario_data = pl.DataFrame({"x": [1.0, 2.0], "price_basic": [10.0, 20.0]})
    with pytest.raises(PredictionError, match="scenario change column"):
        scenario_model.scenario(
            scenario_data.drop("price_basic"),
            {"price_basic": 1.1},
        )
    with pytest.raises(ValidationError, match="must be finite"):
        scenario_model.scenario(scenario_data, {"price_basic": float("inf")})
    array_change = scenario_model.scenario(
        scenario_data,
        {"price_basic": np.array([11.0, 21.0])},
    )
    assert array_change.nobs == 2


def test_multinomial_model_metadata_scenario_and_calibration_guards():
    model = _dummy_multinomial_model()

    np.testing.assert_allclose(model.coef_matrix, model.params)
    assert model.llf() == pytest.approx(-3.0)
    assert "MultinomialModel" in repr(model)
    np.testing.assert_allclose(model.alternative_generic_coefficients, np.zeros(0))
    np.testing.assert_allclose(model.alternative_specific_coefficients, np.zeros((2, 0)))
    np.testing.assert_allclose(model.intercepts, [0.0, 0.0, 0.2])
    assert _dummy_multinomial_model(feature_names=["x"]).intercepts.tolist() == [0.0, 0.0, 0.0]

    assert model.regularization_type == "none"
    assert (
        _dummy_multinomial_model(
            result=_DummyMultinomialResult(alpha=0.1, l1_ratio=1.0)
        ).regularization_type
        == "lasso"
    )
    assert (
        _dummy_multinomial_model(
            result=_DummyMultinomialResult(alpha=0.1, l1_ratio=0.5)
        ).regularization_type
        == "elastic_net"
    )
    assert (
        _dummy_multinomial_model(
            result=_DummyMultinomialResult(alpha=0.1, l1_ratio=0.0)
        ).regularization_type
        == "ridge"
    )
    assert model.regularization_path is None
    assert model.cv_deviance is None
    assert model.cv_deviance_se is None
    assert model.cv_selection_method is None
    assert model.n_cv_folds is None
    assert model.cv_convergence is None
    assert model.cv_fold_scores is None
    assert model.cv_scoring_objective is None
    assert model.cv_profile is None
    assert model.fold_safe_target_encoding is None

    path_info = RegularizationPathInfo(
        selected_alpha=0.2,
        selected_l1_ratio=0.0,
        cv_deviance=1.5,
        cv_deviance_se=0.25,
        selection_method="min",
        regularization_type="ridge",
        path=[RegularizationPathResult(0.2, 0.0, 1.5, 0.25, 2, 0.3)],
        n_folds=3,
    )
    cv_model = _dummy_multinomial_model(regularization_path_info=path_info)
    assert cv_model.cv_deviance == pytest.approx(1.5)
    assert cv_model.cv_deviance_se == pytest.approx(0.25)
    assert cv_model.cv_fold_scores is None

    constrained = _dummy_multinomial_model(
        constraint_metadata=[{"name": "mono", "parameter_indices": [0, 2, 999]}]
    )
    assert constrained.active_constraints == [
        {"name": "mono", "parameter_indices": [0, 2, 999], "active_parameter_indices": [0]}
    ]

    data = pl.DataFrame({"w": [1.0, 2.0]})
    np.testing.assert_allclose(model._scenario_weights(data, None), [1.0, 1.0])
    np.testing.assert_allclose(model._scenario_weights(data, "w"), [1.0, 2.0])
    np.testing.assert_allclose(model._scenario_weights(data, np.array([2.0, 3.0])), [2.0, 3.0])
    with pytest.raises(PredictionError, match="weights column"):
        model._scenario_weights(data, "missing")
    with pytest.raises(ValidationError, match="non-negative"):
        model._scenario_weights(data, np.array([1.0, -1.0]))
    with pytest.raises(ValidationError, match="positive total"):
        model._scenario_weights(data, np.zeros(2))

    with pytest.raises(ValidationError, match="alternative_terms"):
        model.scenario(pl.DataFrame({"x": [1.0]}), {"price": 1.1})
    scenario_model = _dummy_multinomial_model(
        alternative_terms={
            "price": {
                "columns": {"basic": "price_basic"},
                "coefficient": "generic",
                "transform": "identity",
            }
        }
    )
    with pytest.raises(ValidationError, match="non-empty dict"):
        scenario_model.scenario(pl.DataFrame({"price_basic": [1.0]}), {})
    with pytest.raises(ValidationError, match="not used"):
        scenario_model.scenario(pl.DataFrame({"price_basic": [1.0]}), {"other": 1.1})

    no_response = _dummy_multinomial_model(response=None)
    with pytest.raises(PredictionError, match="no stored response"):
        no_response.fit_calibration(pl.DataFrame({"choice": ["none", "basic"]}))
    with pytest.raises(ValidationError, match="max_iter"):
        model.fit_calibration(pl.DataFrame({"choice": ["none", "basic"]}), max_iter=0)
    with pytest.raises(ValidationError, match="tol"):
        model.fit_calibration(pl.DataFrame({"choice": ["none", "basic"]}), tol=0.0)

    ridge_summary = _dummy_multinomial_model(
        result=_DummyMultinomialResult(alpha=0.1, l1_ratio=0.0)
    ).summary()
    assert "Newton + Ridge" in ridge_summary
    assert "<0.0001" in ridge_summary
    assert "1.0000" in ridge_summary
    with pytest.raises(ValidationError, match="return_format"):
        model.coef_table(return_format="bad")

    deserialized = multinomial_mod._DeserializedMultinomialResult(
        params=np.array([[1.0, 2.0]]),
        alternative_generic_coefficients=np.array([0.1]),
        alternative_specific_coefficients=np.array([[0.2]]),
        fitted_probabilities=np.array([[0.6, 0.4], [0.7, 0.3]]),
        linear_predictor=np.array([[0.0, 0.5], [0.0, 0.2]]),
        log_likelihood=-2.0,
        deviance=4.0,
        null_deviance=5.0,
        iterations=3,
        converged=True,
        covariance_unscaled=np.eye(2),
        prior_weights=np.ones(2),
        y_codes=np.array([0, 1]),
        reference_index=0,
        warnings=[],
        solver_status="converged",
        alpha=0.0,
        l1_ratio=0.0,
        fit_intercept=True,
    )
    np.testing.assert_allclose(deserialized.coefficients, [[1.0, 2.0]])
    np.testing.assert_allclose(deserialized.coef_matrix, [[1.0, 2.0]])
    np.testing.assert_allclose(deserialized.fittedvalues, [[0.6, 0.4], [0.7, 0.3]])
    np.testing.assert_allclose(deserialized.cov_params_unscaled, np.eye(2))
    assert deserialized.nobs == 2


def test_factor_value_stringification_and_continuous_bins():
    labels = _stringify_factor_values([None, np.nan, "A", 2])
    assert labels.tolist() == ["<null>", "<null>", "A", "2"]
    assert _continuous_factor_bins([np.nan, np.inf]).tolist() == ["<null>", "<null>"]
    assert _continuous_factor_bins([1.0, 2.0, 2.0], max_bins=5).tolist() == ["1", "2", "2"]
    assert _continuous_factor_bins([1.0, 2.0], max_bins=0).tolist() == ["1", "2"]
    many = _continuous_factor_bins(np.arange(20.0), max_bins=4)
    assert all(label.startswith("[") for label in many)
