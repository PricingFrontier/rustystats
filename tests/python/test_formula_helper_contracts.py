"""Direct contracts for formula parser and fitted-model helper utilities."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import rustystats as rs
import rustystats.formula as formula
from rustystats.exceptions import FittingError, PredictionError, ValidationError
from rustystats.regularization_path import RegularizationPathInfo, RegularizationPathResult


class _DummyResult:
    def __init__(self, warnings=None, params=None):
        self.params = (
            np.array([0.0, 0.0, 0.0]) if params is None else np.asarray(params, dtype=np.float64)
        )
        self.linear_predictor = np.array([0.0, 0.1, 0.2])
        self.fittedvalues = np.exp(self.linear_predictor)
        self.deviance = 12.0
        self.iterations = 4
        self.converged = True
        self.nobs = 3
        self.df_resid = 1
        self.df_model = 2
        self.alpha = 0.0
        self.l1_ratio = 0.0
        self.is_regularized = False
        self.penalty_type = "none"
        self.warnings = warnings

    def bse(self):
        return np.array([0.1, 0.2, 0.3])

    def tvalues(self):
        return np.array([0.0, 0.0, 0.0])

    def pvalues(self):
        return np.array([1.0, 1.0, 1.0])

    def conf_int(self, alpha=0.05):
        assert alpha == 0.05
        return np.array([[-0.2, 0.2], [-0.3, 0.3], [-0.4, 0.4]])

    def significance_codes(self):
        return ["", "", ""]

    def selected_features(self):
        return [1, 2]

    def scale(self):
        return 1.5

    def llf(self):
        return -7.0

    def aic(self):
        return 20.0

    def bic(self):
        return 21.0

    def resid_response(self):
        return np.array([0.0, 0.0, 0.0])


def _dummy_model(**kwargs) -> formula.GLMModel:
    return formula.GLMModel(
        result=kwargs.pop("result", _DummyResult()),
        feature_names=kwargs.pop("feature_names", ["Intercept", "pos(age)", "neg(score)"]),
        formula=kwargs.pop("formula", "y ~ age + score"),
        family=kwargs.pop("family", "poisson"),
        link=kwargs.pop("link", "log"),
        builder=kwargs.pop("builder", None),
        **kwargs,
    )


def test_link_and_column_helper_edge_contracts():
    with pytest.raises(ValidationError, match="Unknown family"):
        formula.get_default_link("not-a-family")

    with pytest.raises(ValidationError, match="Unknown link"):
        formula.apply_inverse_link(np.array([0.0]), "not-a-link")

    assert formula._compute_predict_chunk_size(0) == formula._PREDICT_ROW_CHUNK_DEFAULT

    complement = _dummy_model(
        terms_dict={"comp_x": {"type": "linear"}},
        offset_spec="comp_offset",
        exposure_spec="comp_exposure",
    )
    complement._complement_spec = complement
    needed = formula._extract_needed_columns(
        terms={
            "lookup_out": {"type": "linear"},
            "I(score)": {"type": "expression", "expr": "score + lookup_out"},
        },
        response="y",
        interactions=[{"region": {"type": "categorical"}, "include_main": True}],
        input_transforms=[
            "not-a-transform",
            {"sources": ["raw_lookup"], "output": "lookup_out"},
        ],
        offset="offset",
        weights="weights",
        exposure="exposure",
        complement=complement,
    )

    assert needed == {
        "y",
        "score",
        "raw_lookup",
        "region",
        "offset",
        "weights",
        "exposure",
        "comp_x",
        "comp_offset",
        "comp_exposure",
    }


def test_embedded_family_parameter_parser_fails_closed():
    assert formula._parse_embedded_numeric_param("tweedie(p=1.7)", "p") == ("tweedie", 1.7)
    assert formula._parse_embedded_numeric_param("poisson", "p") == ("poisson", None)

    with pytest.raises(ValidationError, match="Malformed family parameter"):
        formula._split_embedded_family_param("tweedie(p=1.7")
    with pytest.raises(ValidationError, match="expected p"):
        formula._parse_embedded_numeric_param("tweedie(1.7)", "p")
    with pytest.raises(ValidationError, match="unsupported parameter"):
        formula._parse_embedded_numeric_param("tweedie(theta=1.7)", "p")
    with pytest.raises(ValidationError, match="non-numeric"):
        formula._parse_embedded_numeric_param("tweedie(p=bad)", "p")
    with pytest.raises(ValidationError, match="non-finite"):
        formula._parse_embedded_numeric_param("tweedie(p=inf)", "p")


def test_dict_term_parser_covers_spline_encoding_and_constraint_variants():
    parsed = formula.dict_to_parsed_formula(
        response="y",
        terms={
            "age": {"type": "linear", "monotonicity": "increasing"},
            "score": {"type": "expression", "expr": "score ** 2", "monotonicity": "decreasing"},
            "region": {"type": "categorical", "levels": ["North", "South"]},
            "veh_age": {
                "type": "bs",
                "knots": [1.0, 2.0],
                "degree": 2,
                "boundary_knots": [0.0, 3.0],
                "monotonicity": "increasing",
            },
            "bonus": {"type": "ns", "knots": [10.0, 20.0], "boundary_knots": [0.0, 30.0]},
            "natural_default": {"type": "ns"},
            "mileage": {"type": "ms", "k": 4},
            "mileage_default": {"type": "ms"},
            "mileage_knots": {
                "type": "ms",
                "knots": [1.0, 3.0],
                "degree": 2,
                "boundary_knots": [0.0, 4.0],
            },
            "brand_score": {
                "type": "target_encoding",
                "variable": "brand",
                "prior_weight": 2.5,
                "n_permutations": 3,
            },
            "zip_score": {"type": "frequency_encoding", "variable": "zip"},
        },
        intercept=False,
    )

    assert parsed.has_intercept is False
    assert {(c.var_name, c.constraint) for c in parsed.constraint_terms} == {
        ("age", "pos"),
        ("I(score ** 2)", "neg"),
    }
    assert parsed.categorical_terms[0].levels == ["North", "South"]
    assert "region" in parsed.categorical_vars
    assert parsed.target_encoding_terms[0].var_name == "brand"
    assert parsed.target_encoding_terms[0].n_permutations == 3
    assert parsed.frequency_encoding_terms[0].var_name == "zip"

    splines = {term.var_name: term for term in parsed.spline_terms}
    assert splines["veh_age"].df == 4
    assert splines["veh_age"]._computed_internal_knots == [1.0, 2.0]
    assert splines["veh_age"].boundary_knots == (0.0, 3.0)
    assert splines["veh_age"].monotonicity == "increasing"
    assert splines["bonus"].df == 2
    assert splines["natural_default"].df == formula.DEFAULT_SPLINE_DF
    assert splines["natural_default"]._is_smooth is True
    assert splines["mileage"]._is_smooth is True
    assert splines["mileage"].monotonicity == "increasing"
    assert splines["mileage_default"].df == formula.DEFAULT_SPLINE_DF
    assert splines["mileage_default"]._is_smooth is True
    assert splines["mileage_knots"].df == 4
    assert splines["mileage_knots"]._computed_internal_knots == [1.0, 3.0]
    assert splines["mileage_knots"].monotonicity == "increasing"

    with pytest.raises(ValidationError, match="did you mean 'monotonicity'"):
        formula.dict_to_parsed_formula("y", {"x": {"type": "linear", "monoticity": "increasing"}})
    with pytest.raises(ValidationError, match="Unknown key"):
        formula.dict_to_parsed_formula("y", {"x": {"type": "ns", "monotonicity": "increasing"}})
    with pytest.raises(ValidationError, match="frequency_encoding type not supported"):
        formula._parse_term_spec(
            "zip",
            {"type": "frequency_encoding"},
            set(),
            [],
            [],
            [],
            [],
            [],
            [],
            None,
        )
    with pytest.raises(ValidationError, match="Cannot specify both 'knots'"):
        formula.dict_to_parsed_formula("y", {"x": {"type": "bs", "df": 4, "knots": [1.0]}})
    with pytest.raises(ValidationError, match="non-empty"):
        formula.dict_to_parsed_formula("y", {"x": {"type": "bs", "knots": []}})
    with pytest.raises(ValidationError, match="sorted"):
        formula.dict_to_parsed_formula("y", {"x": {"type": "bs", "knots": [2.0, 1.0]}})
    with pytest.raises(ValidationError, match="unique"):
        formula.dict_to_parsed_formula("y", {"x": {"type": "bs", "knots": [1.0, 1.0]}})


def test_dict_interaction_parser_covers_encoding_and_spline_variants():
    parsed = formula.dict_to_parsed_formula(
        response="y",
        terms={},
        interactions=[
            {
                "brand": {"type": "categorical"},
                "region": {"type": "categorical"},
                "target_encoding": True,
                "include_main": True,
                "prior_weight": 4.0,
                "n_permutations": 5,
            },
            {
                "brand": {"type": "categorical"},
                "region": {"type": "categorical"},
                "frequency_encoding": True,
            },
            {
                "age": {"type": "s", "k": 6, "monotonicity": "increasing"},
                "score": {"type": "linear"},
                "brand": {"type": "target_encoding", "prior_weight": 3.0},
            },
            {
                "claim_age": {
                    "type": "bs",
                    "knots": [1.0, 2.0],
                    "degree": 2,
                    "boundary_knots": [0.0, 3.0],
                },
                "territory": {"type": "categorical"},
            },
            {
                "mileage": {"type": "ms"},
                "region": {"type": "categorical"},
            },
        ],
    )

    te_interaction = next(
        te for te in parsed.target_encoding_terms if te.var_name == "brand:region"
    )
    assert te_interaction.interaction_vars == ["brand", "region"]
    assert te_interaction.prior_weight == 4.0
    assert te_interaction.n_permutations == 5
    assert any(
        te.var_name == "brand" and te.prior_weight == 3.0 for te in parsed.target_encoding_terms
    )
    assert parsed.frequency_encoding_terms[0].var_name == "brand:region"
    assert parsed.frequency_encoding_terms[0].interaction_vars == ["brand", "region"]
    assert any(inter.factors == ["age", "score", "TE(brand)"] for inter in parsed.interactions)
    assert any(inter.factors == ["claim_age", "territory"] for inter in parsed.interactions)
    assert any(inter.factors == ["mileage", "region"] for inter in parsed.interactions)
    assert any(inter.force_linear == {"score"} for inter in parsed.interactions)
    assert "brand" in parsed.main_effects
    assert "region" in parsed.main_effects

    with pytest.raises(ValidationError, match="Cannot specify both"):
        formula.dict_to_parsed_formula(
            "y",
            {},
            interactions=[
                {
                    "a": {"type": "categorical"},
                    "b": {"type": "categorical"},
                    "target_encoding": True,
                    "frequency_encoding": True,
                }
            ],
        )
    with pytest.raises(ValidationError, match="at least 2 variables"):
        formula.dict_to_parsed_formula("y", {}, interactions=[{"a": {"type": "linear"}}])
    with pytest.raises(ValidationError, match="frequency_encoding interaction not supported"):
        formula._parse_interaction_spec(
            {
                "a": {"type": "categorical"},
                "b": {"type": "categorical"},
                "frequency_encoding": True,
            },
            [],
            set(),
            [],
            [],
            [],
            [],
            [],
            [],
            None,
        )


def test_prediction_auxiliary_resolvers_validate_shapes_and_transform_scales():
    data = pl.DataFrame({"offset": [0.0, 0.2], "expo": [1.0, 2.0], "prior": [0.5, 0.75]})

    raw, link, col = formula._resolve_predict_offset(data, None, "offset")
    assert col == "offset"
    np.testing.assert_allclose(raw, [0.0, 0.2])
    np.testing.assert_allclose(link, [0.0, 0.2])
    np.testing.assert_allclose(
        formula._resolve_predict_offset(data, np.array([0.1, 0.2]), "offset")[1],
        [0.1, 0.2],
    )
    with pytest.raises(PredictionError, match="not present"):
        formula._resolve_predict_offset(data, None, "missing")
    with pytest.raises(PredictionError, match="offset array length"):
        formula._resolve_predict_offset(data, np.ones((2, 1)), None)

    exposure, exposure_link, exposure_col = formula._resolve_predict_exposure(data, "expo")
    assert exposure_col == "expo"
    np.testing.assert_allclose(exposure, [1.0, 2.0])
    np.testing.assert_allclose(exposure_link, np.log([1.0, 2.0]))
    with pytest.raises(PredictionError, match="not present"):
        formula._resolve_predict_exposure(data, "missing")
    with pytest.raises(PredictionError, match="one-dimensional"):
        formula._resolve_predict_exposure(data, np.ones((2, 1)))
    with pytest.raises(PredictionError, match="length"):
        formula._resolve_predict_exposure(data, np.ones(3))
    with pytest.raises(PredictionError, match="finite and strictly positive"):
        formula._resolve_predict_exposure(data, np.array([1.0, 0.0]))

    comp_response, comp_link = formula._resolve_predict_complement(
        data, None, "prior", None, "logit"
    )
    np.testing.assert_allclose(comp_response, [0.5, 0.75])
    np.testing.assert_allclose(comp_link, [0.0, np.log(3.0)])
    comp_response, comp_link = formula._resolve_predict_complement(
        data, np.array([1.0, 2.0]), None, None, "log"
    )
    np.testing.assert_allclose(comp_response, [1.0, 2.0])
    np.testing.assert_allclose(comp_link, np.log([1.0, 2.0]))


def test_glm_base_helper_branches_and_cv_path(monkeypatch, capsys):
    base = object.__new__(formula.FormulaGLMDict)
    data = pl.DataFrame({"x": [1.0, 2.0], "offset": [0.1, 0.2]})
    base._data_ref = lambda: data
    base.family = "poisson"
    base.link = "log"
    base.terms = {}
    base.interactions_spec = None
    base._builder = SimpleNamespace(_parsed_formula=SimpleNamespace(target_encoding_terms=[]))
    base._seed = 123

    assert base.data is data
    assert base._uses_log_link() is True
    base.link = None
    base.family = "not-a-family"
    assert base._uses_log_link() is False

    collected = object.__new__(formula.FormulaGLMDict)
    collected._data_ref = lambda: None
    with pytest.raises(ValidationError, match="garbage collected"):
        _ = collected.data

    base.family = "poisson"
    base.link = "log"
    complement = _dummy_model()
    complement.predict = lambda frame, **kwargs: np.array([2.0, 8.0])
    comp_link = base._process_complement(complement, raw_exposure=np.array([1.0, 4.0]))
    np.testing.assert_allclose(comp_link, np.log([2.0, 2.0]))

    import rustystats.regularization_path as reg_path

    def fake_cv_path(**_kwargs):
        return SimpleNamespace(selected_alpha=0.07, selected_l1_ratio=0.5)

    monkeypatch.setattr(reg_path, "fit_cv_regularization_path", fake_cv_path)
    alpha, l1_ratio, info = base._resolve_cv_path(
        alpha=1.0,
        l1_ratio=0.0,
        max_iter=10,
        tol=1e-6,
        cv=3,
        selection="min",
        regularization="elastic_net",
        n_alphas=4,
        alpha_min_ratio=0.01,
        cv_seed=None,
        cv_max_iter=None,
        cv_tol=None,
        include_unregularized=False,
        verbose=True,
        standardize=True,
    )
    assert (alpha, l1_ratio, info) == (0.07, 0.5, info)
    assert "Refitting on full data" in capsys.readouterr().out

    with pytest.raises(ValidationError, match="Unknown regularization"):
        base._resolve_cv_path(
            alpha=1.0,
            l1_ratio=0.0,
            max_iter=10,
            tol=1e-6,
            cv=3,
            selection="min",
            regularization="unknown",
            n_alphas=4,
            alpha_min_ratio=0.01,
            cv_seed=None,
            cv_max_iter=None,
            cv_tol=None,
            include_unregularized=False,
            verbose=False,
            standardize=True,
        )


def test_glm_model_metadata_properties_and_relevel_helpers():
    model = _dummy_model(
        result=_DummyResult(warnings=lambda: ("one", "two")),
        terms_dict={"age": {"type": "linear"}, "score": {"type": "linear"}},
        offset_spec="offset",
        complement_spec="prior",
    )

    assert model.warnings == ["one", "two"]
    assert model.selected_features() == ["pos(age)", "neg(score)"]
    assert model.required_columns == sorted(["age", "score", "offset", "prior"])
    assert model.boundary_active_coefficients == [
        {"feature": "pos(age)", "constraint": "nonnegative", "coefficient": 0.0},
        {"feature": "neg(score)", "constraint": "nonpositive", "coefficient": 0.0},
    ]
    assert model.cv_fold_scores is None
    assert model.cv_scoring_objective is None
    assert model.cv_profile is None
    assert model.regularization_path is None
    assert model.n_cv_folds is None
    assert model.cv_convergence is None
    assert model.get_design_matrix() is None
    assert model.fold_safe_target_encoding is None
    assert model._family_unit_variance(np.array([2.0, 3.0])).tolist() == [2.0, 3.0]
    assert _dummy_model(family="gamma")._family_unit_variance(np.array([2.0])).tolist() == [4.0]
    np.testing.assert_allclose(
        _dummy_model(family="tweedie(p=1.5)")._family_unit_variance(np.array([4.0])),
        [8.0],
    )
    np.testing.assert_allclose(
        _dummy_model(family="NegativeBinomial(theta=2.0)")._family_unit_variance(np.array([4.0])),
        [12.0],
    )

    model._intercept_delta = np.log(2.0)
    model._intercept_delta_var = 0.03
    assert model._intercept_releveled() is True
    assert model.params[0] == pytest.approx(np.log(2.0))
    assert model.linear_predictor[0] == pytest.approx(np.log(2.0))
    assert model.fittedvalues[0] == pytest.approx(2.0)
    corrected = model._releveled_intercept_inference(0.1, (-0.2, 0.2))
    assert corrected is not None
    assert corrected["se"] > 0.1
    assert corrected["ci_lo"] < model.params[0] < corrected["ci_hi"]
    zero_se = model._releveled_intercept_inference(0.0)
    assert zero_se is not None
    assert np.isnan(zero_se["z"])
    assert model._with_releveled_intercept_ci(np.empty((0, 2)), np.array([])).shape == (0, 2)

    assert (
        model._relevel_log_factor_variance(np.array([0.0, 0.0]), np.array([1.0, 2.0]), None) == 0.0
    )

    class RaisingScaleResult(_DummyResult):
        def scale(self):
            raise RuntimeError("scale unavailable")

    class BadScaleResult(_DummyResult):
        def scale(self):
            return -1.0

    np.testing.assert_allclose(
        _dummy_model(result=RaisingScaleResult())._relevel_log_factor_variance(
            np.array([1.0, 2.0]), np.array([1.0, 2.0]), None
        ),
        1.0 / 3.0,
    )
    assert (
        _dummy_model(result=BadScaleResult())._relevel_log_factor_variance(
            np.array([1.0, 2.0]), np.array([1.0, 2.0]), None
        )
        > 0.0
    )

    unavailable = _dummy_model(covariance_available=False)
    with pytest.raises(FittingError, match="Covariance was skipped"):
        unavailable.bse()

    with pytest.raises(ValidationError, match="Relativities only meaningful"):
        _dummy_model(link="identity").relativities()


def test_glm_model_fallback_metrics_and_prediction_guard(monkeypatch):
    model = _dummy_model()

    with pytest.raises(PredictionError, match="Cannot predict"):
        model.predict(pl.DataFrame({"age": [1.0], "score": [2.0]}))

    model._intercept_delta = np.log(1.5)

    import rustystats._rustystats as rust_core
    import rustystats.regularization_path as reg_path

    monkeypatch.setattr(
        reg_path,
        "compute_deviance",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        rust_core,
        "compute_dataset_metrics_py",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert model.deviance == pytest.approx(12.0)
    assert model.llf() == pytest.approx(-7.0)


class _RecordingPredictionBuilder:
    def __init__(self):
        self._term_slots = [SimpleNamespace(col_start=0, col_end=1)]
        self.transform_batches: list[tuple[str, ...]] = []
        self.termwise_batches: list[tuple[str, ...]] = []

    def _design(self, data: pl.DataFrame) -> np.ndarray:
        return np.column_stack(
            [
                np.ones(len(data), dtype=np.float64),
                data["x"].to_numpy().astype(np.float64),
                data["z"].to_numpy().astype(np.float64),
            ]
        )

    def transform_new_data(self, data: pl.DataFrame) -> np.ndarray:
        self.transform_batches.append(tuple(data.columns))
        return self._design(data)

    def linear_predict_new_data(self, data: pl.DataFrame, params: np.ndarray) -> np.ndarray:
        self.termwise_batches.append(tuple(data.columns))
        return self._design(data) @ params


def test_predict_direct_edge_paths_for_no_terms_termwise_and_aux_projection(monkeypatch):
    params = np.array([0.2, 0.1, -0.05])
    data = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "z": [0.5, 0.0, -0.5, 1.0, 2.0],
            "raw": [0.0, 1.0, 0.0, 1.0, 0.0],
            "expo": [1.0, 2.0, 1.5, 3.0, 0.5],
            "unused": [99.0, 98.0, 97.0, 96.0, 95.0],
        }
    )

    no_terms_builder = _RecordingPredictionBuilder()
    no_terms = _dummy_model(
        result=_DummyResult(params=params),
        feature_names=["Intercept", "x", "z"],
        builder=no_terms_builder,
        terms_dict=None,
    )
    no_terms_pred = no_terms.predict(data.lazy())
    np.testing.assert_allclose(
        no_terms_pred,
        np.exp(params[0] + params[1] * data["x"].to_numpy() + params[2] * data["z"].to_numpy()),
    )
    assert no_terms_builder.transform_batches == [("x", "z", "raw", "expo", "unused")]

    non_log_exposure = _dummy_model(
        result=_DummyResult(params=params),
        feature_names=["Intercept", "x", "z"],
        link="identity",
        builder=_RecordingPredictionBuilder(),
        terms_dict={"x": {"type": "linear"}, "z": {"type": "linear"}},
        exposure_spec="expo",
    )
    with pytest.raises(ValidationError, match="only meaningful for log-link"):
        non_log_exposure.predict(data)

    monkeypatch.setattr(formula, "_PREDICT_TERMWISE_FEATURE_THRESHOLD", 3)
    monkeypatch.setattr(formula, "_compute_predict_chunk_size", lambda _n: 2)
    monkeypatch.setattr(formula, "_compute_termwise_predict_chunk_size", lambda *_args: 10)
    termwise_builder = _RecordingPredictionBuilder()
    termwise = _dummy_model(
        result=_DummyResult(params=params),
        feature_names=["Intercept", "x", "z"],
        builder=termwise_builder,
        terms_dict={"x": {"type": "linear"}, "z": {"type": "linear"}},
    )
    termwise_pred = termwise.predict(data.head(4).lazy())
    np.testing.assert_allclose(
        termwise_pred,
        np.exp(
            params[0] + params[1] * data["x"].to_numpy()[:4] + params[2] * data["z"].to_numpy()[:4]
        ),
    )
    assert termwise_builder.transform_batches == []
    assert termwise_builder.termwise_batches == [("x", "z")]

    monkeypatch.setattr(formula, "_compute_predict_chunk_size", lambda _n: 1)
    monkeypatch.setattr(formula, "_compute_termwise_predict_chunk_size", lambda *_args: 3)
    aux_builder = _RecordingPredictionBuilder()
    aux_model = _dummy_model(
        result=_DummyResult(params=params),
        feature_names=["Intercept", "x", "z"],
        builder=aux_builder,
        terms_dict={"x": {"type": "linear"}, "z": {"type": "linear"}},
        input_transforms=[
            {
                "type": "lookup",
                "name": "expo_lookup",
                "sources": ["raw"],
                "output": "expo",
                "output_dtype": "float64",
                "keys": [[0.0], [1.0]],
                "values": [1.0, 2.0],
                "default": 1.5,
            }
        ],
        compiled_input_transforms=[],
        exposure_spec="expo",
    )
    prepared_columns: list[tuple[str, ...]] = []

    def record_prepare(frame: pl.DataFrame) -> pl.DataFrame:
        prepared_columns.append(tuple(frame.columns))
        return frame

    aux_model._apply_model_input_transforms = record_prepare
    chunked_pred = aux_model.predict(data)
    linear = params[0] + params[1] * data["x"].to_numpy() + params[2] * data["z"].to_numpy()
    np.testing.assert_allclose(chunked_pred, np.exp(linear) * data["expo"].to_numpy())
    assert aux_builder.transform_batches == []
    assert aux_builder.termwise_batches == [
        ("x", "z", "raw", "expo", "unused"),
        ("x", "z", "raw", "expo", "unused"),
    ]
    assert prepared_columns == [("raw", "expo")]


def test_formula_dict_constructor_and_formula_string_edge_contracts():
    data = pl.DataFrame({"y": [1.0, 2.0], "x": [1.0, 2.0]})

    with pytest.raises(ValidationError, match="Conflicting Tweedie"):
        formula.FormulaGLMDict(
            response="y",
            terms={"x": {"type": "linear"}},
            data=data,
            family="tweedie(p=1.2)",
            var_power=1.6,
        )

    with pytest.raises(ValidationError, match="Conflicting Negative Binomial"):
        formula.FormulaGLMDict(
            response="y",
            terms={"x": {"type": "linear"}},
            data=data,
            family="NegativeBinomial(theta=2.0)",
            theta=3.0,
        )

    builder = object.__new__(formula.FormulaGLMDict)
    builder.response = "y"
    builder.intercept = False
    builder.terms = {
        "age": {"type": "ms", "knots": [1.0, 2.0], "monotonicity": "decreasing"},
        "brand_score": {"type": "target_encoding", "interaction": ["brand", "region"]},
    }

    assert builder._build_formula_string() == (
        "y ~ 0 + ms(age, knots=[2], decreasing) + TE(brand:region)"
    )


def test_glm_model_regularization_path_and_serialized_result_contracts():
    path_info = RegularizationPathInfo(
        selected_alpha=0.2,
        selected_l1_ratio=0.5,
        cv_deviance=1.2,
        cv_deviance_se=0.03,
        selection_method="1se",
        regularization_type="elastic_net",
        path=[RegularizationPathResult(0.2, 0.5, 1.2, 0.03, 2, 0.4)],
        n_folds=4,
        cv_max_iter=55,
        cv_tol=1e-7,
        fold_safe_target_encoding=True,
        cv_fold_scores={0.2: [1.1, 1.3]},
        cv_scoring_objective="oracle",
        cv_profile={"mode": "fast"},
    )
    model = _dummy_model(regularization_path_info=path_info)

    assert model.regularization_type == "elastic_net"
    assert model.regularization_path == [
        {
            "alpha": 0.2,
            "l1_ratio": 0.5,
            "cv_deviance_mean": 1.2,
            "cv_deviance_se": 0.03,
            "n_nonzero": 2,
            "max_coef": 0.4,
        }
    ]
    assert model.cv_selection_method == "1se"
    assert model.n_cv_folds == 4
    assert model.cv_convergence == {"max_iter": 55, "tol": 1e-7}
    assert model.cv_fold_scores == {0.2: [1.1, 1.3]}
    assert model.cv_scoring_objective == "oracle"
    assert model.cv_profile == {"mode": "fast"}
    assert model.fold_safe_target_encoding is True

    deserialized = formula._DeserializedResult(
        params=np.array([1.0]),
        deviance=1.0,
        iterations=1,
        converged=True,
        nobs=2,
        df_resid=1,
        df_model=0,
        alpha=0.0,
        l1_ratio=0.0,
        is_regularized=False,
        penalty_type="none",
    )
    with pytest.raises(AttributeError, match="fittedvalues not available"):
        _ = deserialized.fittedvalues
    with pytest.raises(AttributeError, match="linear_predictor not available"):
        _ = deserialized.linear_predictor


def test_calibration_extract_arrays_materializes_response_prediction_exposure_and_weights():
    model = _dummy_model(formula="claims ~ x", exposure_spec="expo")
    model.predict = lambda data, exposure=None, **kwargs: np.array([1.0, 2.0, 3.0])
    data = pl.DataFrame(
        {
            "claims": [1.0, 3.0, 2.0],
            "x": [0.0, 1.0, 2.0],
            "expo": [1.0, 1.5, 2.0],
            "w": [0.5, 1.0, 2.0],
        }
    )

    out_data, y, mu, exposure, weights = model._calibration_extract_arrays(
        data.lazy(), exposure=None, weights="w"
    )

    assert out_data.shape == data.shape
    np.testing.assert_allclose(y, [1.0, 3.0, 2.0])
    np.testing.assert_allclose(mu, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(exposure, [1.0, 1.5, 2.0])
    np.testing.assert_allclose(weights, [0.5, 1.0, 2.0])

    with pytest.raises(ValidationError, match="response column"):
        model._calibration_extract_arrays(pl.DataFrame({"x": [1.0]}), exposure=None, weights=None)
    with pytest.raises(ValidationError, match="weights column"):
        model._calibration_extract_arrays(data, exposure=None, weights="missing")
    _, _, _, exposure_array, weights_array = model._calibration_extract_arrays(
        data,
        exposure=np.array([2.0, 2.0, 2.0]),
        weights=np.array([1.0, 1.0, 1.0]),
    )
    np.testing.assert_allclose(exposure_array, [2.0, 2.0, 2.0])
    np.testing.assert_allclose(weights_array, [1.0, 1.0, 1.0])

    bare_model = _dummy_model(formula="claims ~ x")
    bare_model.predict = lambda data, exposure=None, **kwargs: np.array([1.0, 2.0, 3.0])
    _, _, _, no_exposure, no_weights = bare_model._calibration_extract_arrays(
        data.drop("expo"),
        exposure=None,
        weights=None,
    )
    assert no_exposure is None
    assert no_weights is None
    _, _, _, missing_exposure, _ = bare_model._calibration_extract_arrays(
        data.drop("expo"),
        exposure="expo",
        weights=None,
    )
    assert missing_exposure is None

    with pytest.raises(ValidationError, match="unknown calibration method"):
        model.fit_calibration(data, method="not-a-method")


def test_relevel_failure_edges_for_intercept_and_invalid_global_factor(monkeypatch):
    data = pl.DataFrame({"claims": [1.0, 2.0], "x": [0.0, 1.0]})

    no_intercept = _dummy_model(
        formula="claims ~ x",
        feature_names=["x"],
        result=_DummyResult(params=[0.1]),
    )
    with pytest.raises(ValidationError, match="requires a model fitted with an intercept"):
        no_intercept.relevel(data)

    import rustystats.calibration as calibration

    bad_factor = _dummy_model(formula="claims ~ x")
    bad_factor._calibration_extract_arrays = lambda *_args, **_kwargs: (
        data,
        np.array([1.0, 2.0]),
        np.array([1.0, 1.0]),
        None,
        None,
    )
    monkeypatch.setattr(
        calibration,
        "fit_global_calibration",
        lambda *_args, **_kwargs: SimpleNamespace(factor=0.0),
    )

    with pytest.raises(ValidationError, match="not finite/positive"):
        bad_factor.relevel(data)


def test_cv_path_info_properties_exposed():
    rng = np.random.default_rng(31)
    n = 120
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + rng.normal(0.0, 0.2, n)
    data = pl.DataFrame({"y": y, "x": x})
    model = rs.glm_dict(
        response="y",
        terms={"x": {"type": "linear"}},
        data=data,
        family="gaussian",
    ).fit(cv=2, regularization="ridge", n_alphas=3, cv_seed=7)

    assert model.cv_selection_method in {"min", "1se"}
    se = model.cv_deviance_se
    assert se is None or np.isfinite(float(se))
    assert model.terms_dict == {"x": {"type": "linear"}}
    assert model.interactions_spec is None


def test_spline_spec_df_is_fixed_basis_and_k_is_penalized():
    rng = np.random.default_rng(5)
    n = 150
    x = rng.uniform(0.0, 2.0, n)
    y = rng.poisson(np.exp(0.2 + 0.3 * np.sin(x))).astype(float)
    data = pl.DataFrame({"y": y, "x": x})

    fixed = rs.glm_dict(
        response="y", terms={"x": {"type": "bs", "df": 4}}, data=data, family="poisson"
    ).fit()
    assert np.isfinite(fixed.deviance)
    assert not fixed.smooth_terms  # df= requests a fixed unpenalized basis

    penalized = rs.glm_dict(
        response="y", terms={"x": {"type": "bs", "k": 5}}, data=data, family="poisson"
    ).fit()
    assert np.isfinite(penalized.deviance)
    assert penalized.smooth_terms  # k= requests a penalized smooth


def test_parse_spline_spec_df_k_and_error_branches():
    with pytest.raises(ValidationError, match="Expected spline term"):
        formula._parse_spline_spec("x", {"type": "linear"})
    with pytest.raises(ValidationError, match="not supported for natural splines"):
        formula._parse_spline_spec("x", {"type": "ns", "monotonicity": "increasing"})

    # s(k=) and bare specs are penalized smooths.
    smooth = formula._parse_spline_spec("x", {"type": "s", "k": 5})
    assert smooth.df == 5 and smooth._is_smooth is True
    default = formula._parse_spline_spec("x", {"type": "bs"})
    assert default.df == formula.DEFAULT_SPLINE_DF and default._is_smooth is True

    # k= requests a penalized smooth; df= a fixed unpenalized basis.
    k_spec = formula._parse_spline_spec("x", {"type": "bs", "k": 4})
    assert k_spec.df == 4 and k_spec._is_smooth is True
    df_spec = formula._parse_spline_spec("x", {"type": "bs", "df": 4})
    assert df_spec.df == 4 and not getattr(df_spec, "_is_smooth", False)


def test_prediction_deviance_embeds_family_parameters():
    """negbinomial theta and tweedie var_power reach the Rust unit-deviance
    helper via the embedded family string (a bare family name would silently
    score with theta=1 / p=1.5 defaults)."""
    y = np.array([0.0, 1.0, 3.0])
    mu = np.array([0.5, 1.2, 2.5])

    nb = _dummy_model(family="negbinomial")
    nb.theta = 1.7
    dev_theta = nb._prediction_deviance_values(y, mu, None)
    nb.theta = 25.0
    dev_other = nb._prediction_deviance_values(y, mu, None)
    assert dev_theta.shape == (3,)
    assert not np.allclose(dev_theta, dev_other)  # theta genuinely flows through

    tw = _dummy_model(family="tweedie")
    tw.var_power = 1.3
    dev_p13 = tw._prediction_deviance_values(y, mu, np.ones(3))
    tw.var_power = 1.8
    dev_p18 = tw._prediction_deviance_values(y, mu, np.ones(3))
    assert not np.allclose(dev_p13, dev_p18)  # var_power genuinely flows through
