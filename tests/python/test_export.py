"""
Tests for PMML and ONNX export functionality.

Covers:
- PMML XML structure and correctness
- ONNX protobuf output (scoring and full modes)
- Round-trip: model → export → file → reload bytes
- Term types: linear, categorical, spline, TE, FE
"""

import os
import tempfile
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import rustystats as rs

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def simple_poisson_data():
    np.random.seed(42)
    n = 300
    return pl.DataFrame(
        {
            "y": np.random.poisson(1.0, n).astype(float),
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "cat": np.random.choice(["A", "B", "C"], n),
            "exposure": np.ones(n),
        }
    )


@pytest.fixture
def simple_model(simple_poisson_data):
    return rs.glm_dict(
        response="y",
        terms={
            "x1": {"type": "linear"},
            "x2": {"type": "linear"},
            "cat": {"type": "categorical"},
        },
        data=simple_poisson_data,
        family="poisson",
        offset="exposure",
    ).fit(store_design_matrix=True)


@pytest.fixture
def gaussian_data():
    np.random.seed(123)
    n = 200
    x = np.random.randn(n)
    return pl.DataFrame(
        {
            "y": (2.0 + 0.5 * x + np.random.randn(n) * 0.3),
            "x": x,
        }
    )


@pytest.fixture
def gaussian_model(gaussian_data):
    return rs.glm_dict(
        response="y",
        terms={"x": {"type": "linear"}},
        data=gaussian_data,
        family="gaussian",
    ).fit(store_design_matrix=True)


@pytest.fixture
def full_export_model(simple_poisson_data):
    # ONNX full raw-data export does not embed offset/exposure into the graph,
    # so a model exercised through full mode must not carry either.
    return rs.glm_dict(
        response="y",
        terms={
            "x1": {"type": "linear"},
            "x2": {"type": "linear"},
            "cat": {"type": "categorical"},
        },
        data=simple_poisson_data,
        family="poisson",
    ).fit(store_design_matrix=True)


def _onnx_runtime_predictions(onnx_bytes: bytes, inputs: np.ndarray) -> np.ndarray:
    if os.environ.get("RUSTYSTATS_REQUIRE_EXPORT_RUNTIMES") == "1":
        try:
            import onnxruntime as ort
        except ImportError as exc:
            pytest.fail(f"onnxruntime is required for deployment parity: {exc}")
    else:
        ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: np.ascontiguousarray(inputs, dtype=np.float64)})
    assert len(outputs) == 1
    return np.asarray(outputs[0], dtype=np.float64).reshape(-1)


def _pmml_runtime_predictions(pmml_xml: str, data: pl.DataFrame, tmp_path) -> np.ndarray:
    if os.environ.get("RUSTYSTATS_RUN_PMML_RUNTIME") != "1":
        pytest.skip("set RUSTYSTATS_RUN_PMML_RUNTIME=1 to execute PMML runtime parity")
    try:
        from pypmml import Model
    except ImportError as exc:
        pytest.fail(f"pypmml is required for deployment parity: {exc}")

    path = tmp_path / "model.pmml"
    path.write_text(pmml_xml, encoding="utf-8")
    model = Model.fromFile(str(path))
    scored = model.predict(data.to_pandas())
    prediction_column = next(col for col in scored.columns if col.startswith("predicted_"))
    return np.asarray(scored[prediction_column], dtype=np.float64)


def _design_matrix_without_intercept(model) -> np.ndarray:
    design = model.get_design_matrix()
    assert design is not None
    columns = [idx for idx, name in enumerate(model.feature_names) if name != "Intercept"]
    return np.asarray(design[:, columns], dtype=np.float64)


def _fit_onnx_runtime_model(family: str):
    rng = np.random.default_rng(7309)
    n = 220
    x1 = rng.normal(size=n)
    x2 = rng.uniform(-1.0, 1.0, size=n)
    eta = 0.2 + 0.35 * x1 - 0.25 * x2

    if family == "gaussian":
        y = eta + rng.normal(0.0, 0.15, size=n)
        kwargs = {}
    elif family == "poisson":
        y = rng.poisson(np.exp(eta)).astype(float)
        kwargs = {}
    elif family == "binomial":
        p = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, p).astype(float)
        kwargs = {}
    elif family == "gamma":
        mu = np.exp(eta)
        y = rng.gamma(shape=8.0, scale=mu / 8.0)
        kwargs = {"link": "log"}
    elif family == "negbinomial":
        theta = 2.0
        mu = np.exp(eta)
        p = theta / (theta + mu)
        y = rng.negative_binomial(theta, p).astype(float)
        kwargs = {"theta": theta}
    elif family == "tweedie":
        y = rng.poisson(np.exp(eta)).astype(float)
        kwargs = {"var_power": 1.5}
    else:
        raise AssertionError(f"unexpected family: {family}")

    data = pl.DataFrame({"y": y, "x1": x1, "x2": x2})
    return rs.glm_dict(
        response="y",
        terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
        data=data,
        family=family,
        **kwargs,
    ).fit(store_design_matrix=True)


def _full_mode_input_matrix(model, data: pl.DataFrame) -> np.ndarray:
    cache = model._builder._cat_encoding_cache["cat_True"]
    cat_lookup = {level: idx for idx, level in enumerate(cache.levels)}
    cat_codes = np.asarray([cat_lookup[value] for value in data["cat"]], dtype=np.float64)
    return np.column_stack(
        [
            data["x1"].to_numpy(),
            data["x2"].to_numpy(),
            cat_codes,
        ]
    ).astype(np.float64)


def _fake_export_model(
    *,
    family: str = "gaussian",
    link: str = "identity",
    feature_names: list[str] | None = None,
    params: list[float] | None = None,
    builder=None,
    offset_spec=None,
    exposure_spec=None,
    input_transforms: list[dict] | None = None,
):
    if feature_names is None:
        feature_names = ["Intercept", "x"]
    if params is None:
        params = [1.0, 0.25]
    return SimpleNamespace(
        feature_names=feature_names,
        params=np.asarray(params, dtype=np.float64),
        family=family,
        link=link,
        formula="y ~ x",
        _builder=builder,
        _offset_spec=offset_spec,
        _exposure_spec=exposure_spec,
        _input_transforms=[] if input_transforms is None else input_transforms,
    )


class _FakeSpline:
    def get_knot_info(self):
        return {"boundary_knots": [0.0, 10.0], "knots": [5.0]}

    def transform(self, values):
        values = np.asarray(values, dtype=np.float64)
        basis = np.column_stack([np.ones_like(values), values / 10.0])
        return basis, ["bs(age, 1/2)", "bs(age, 2/2)"]


# ── PMML Tests ───────────────────────────────────────────────────────────────


class TestPMMLInternalHelpers:
    def test_split_interaction_respects_nested_syntax(self):
        from rustystats.export_pmml import _split_interaction

        assert _split_interaction("cat[T.A]:bs(age, 2/5, intercept=False):I(income ** 2)") == [
            "cat[T.A]",
            "bs(age, 2/5, intercept=False)",
            "I(income ** 2)",
        ]

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("Intercept", {"type": "intercept"}),
            ("cat[T.B]", {"type": "categorical", "variable": "cat", "level": "B"}),
            (
                "bs(age, 3/7, intercept=False)",
                {
                    "type": "spline",
                    "spline_type": "bs",
                    "variable": "age",
                    "basis_idx": 3,
                    "basis_total": 7,
                    "flags": "intercept=False",
                },
            ),
            ("TE(region)", {"type": "te", "variable": "region"}),
            ("FE(region)", {"type": "fe", "variable": "region"}),
            ("I(age ** 2)", {"type": "expression", "expr": "age ** 2"}),
            ("pos(loss)", {"type": "constraint", "variable": "loss", "sign": "pos"}),
            ("neg(loss)", {"type": "constraint", "variable": "loss", "sign": "neg"}),
            ("severity", {"type": "linear", "variable": "severity"}),
        ],
    )
    def test_classify_feature_components(self, name, expected):
        from rustystats.export_pmml import _classify

        assert _classify(name) == expected

    def test_classify_feature_returns_interaction_components(self):
        from rustystats.export_pmml import _classify_feature

        info = _classify_feature("cat[T.B]:I(age ** 2):FE(region)")
        assert info["type"] == "interaction"
        assert [component["type"] for component in info["components"]] == [
            "categorical",
            "expression",
            "fe",
        ]
        assert [component["name"] for component in info["components"]] == [
            "cat[T.B]",
            "I(age ** 2)",
            "FE(region)",
        ]

    @pytest.mark.parametrize(
        ("link", "expected_parameter"),
        [("inverse", "-1"), ("sqrt", "0.5")],
    )
    def test_pmml_power_links_include_link_parameter(self, link, expected_parameter):
        from rustystats.export_pmml import PMMLExporter

        xml = PMMLExporter(_fake_export_model(family="gamma", link=link)).export()

        assert 'linkFunction="power"' in xml
        assert f'linkParameter="{expected_parameter}"' in xml

    def test_pmml_unknown_family_and_link_fail_closed_to_portable_defaults(self):
        from rustystats.export_pmml import PMMLExporter

        xml = PMMLExporter(_fake_export_model(family="unexpected", link="unknown")).export()

        assert 'distributionName="normal"' in xml
        assert 'linkFunction="log"' in xml
        assert "linkParameter=" not in xml

    def test_pmml_emits_encoded_expression_interaction_and_spline_fallback_terms(self):
        from rustystats.export_pmml import PMMLExporter

        builder = SimpleNamespace(
            _cat_encoding_cache={"cat_True": SimpleNamespace(levels=["A", "B"])},
            _te_stats={
                "brand": {
                    "prior": 0.25,
                    "prior_weight": 2.0,
                    "stats": {"A": (2.0, 4), "B": (1.0, 2)},
                    "used_exposure_weighted": False,
                }
            },
            _fe_stats={"region": {"level_counts": {"North": 2, "South": 4}, "max_count": 4}},
            _fitted_splines={},
        )
        model = _fake_export_model(
            family="poisson",
            link="log",
            builder=builder,
            feature_names=[
                "Intercept",
                "TE(brand)",
                "FE(region)",
                "I(age ** 2)",
                "I(age + income)",
                "cat[T.B]:x",
                "TE(brand):FE(region):I(age ** 2)",
                "bs(age, 1/2)",
            ],
            params=[0.1, 0.5, 0.25, 0.01, -0.02, 0.3, 0.4, 0.05],
        )

        xml = PMMLExporter(model).export()

        assert 'name="TE_brand"' in xml
        assert 'name="FE_region"' in xml
        assert "<MapValues" in xml
        assert '<Apply function="pow">' in xml
        assert 'name="rustystats_expression" value="age + income"' in xml
        assert 'label="cat[T.B]:x"' in xml
        assert 'label="TE(brand):FE(region):I(age ** 2)"' in xml
        # No fitted spline metadata means spline columns fall back to ordinary
        # continuous covariates rather than silently approximating with bogus knots.
        assert 'label="bs(age, 1/2)"' in xml

    def test_pmml_exposure_weighted_target_encoding_and_zero_frequency_encoding(self):
        from rustystats.export_pmml import PMMLExporter

        builder = SimpleNamespace(
            _cat_encoding_cache={},
            _te_stats={
                "brand": {
                    "prior": 0.5,
                    "prior_weight": 2.0,
                    "stats": {"A": (3.0, 4.0)},
                    "used_exposure_weighted": True,
                }
            },
            _fe_stats={"region": {"level_counts": {"North": 3}, "max_count": 0}},
            _fitted_splines={},
        )
        model = _fake_export_model(
            family="poisson",
            link="log",
            builder=builder,
            feature_names=["TE(brand)", "FE(region)"],
            params=[1.0, 2.0],
        )

        xml = PMMLExporter(model).export()

        assert "<original>A</original>" in xml
        assert "<encoded>0.6666666667</encoded>" in xml
        assert "<original>North</original>" in xml
        assert "<encoded>0</encoded>" in xml

    def test_pmml_collapses_fitted_spline_and_spline_interaction(self):
        from rustystats.export_pmml import PMMLExporter

        builder = SimpleNamespace(
            _cat_encoding_cache={"cat_True": SimpleNamespace(levels=["A", "B"])},
            _te_stats={},
            _fe_stats={},
            _fitted_splines={"age": _FakeSpline()},
        )
        model = _fake_export_model(
            family="poisson",
            link="log",
            builder=builder,
            feature_names=[
                "bs(age, 1/2)",
                "bs(age, 2/2)",
                "cat[T.B]:bs(age, 1/2)",
                "cat[T.B]:bs(age, 2/2)",
            ],
            params=[0.5, 0.25, 0.1, 0.2],
        )

        xml = PMMLExporter(model, n_grid_points=4).export()

        assert 'name="spline_effect_age"' in xml
        assert 'name="int_spline_age_cat_T_B_"' in xml
        assert "<NormContinuous" in xml
        assert xml.count("<LinearNorm") == 8
        assert 'predictorName="cat" parameterName=' in xml

    def test_pmml_rejects_input_transforms_before_exporting_raw_data(self):
        from rustystats.export_pmml import to_pmml

        model = _fake_export_model(input_transforms=[{"name": "territory_lookup"}])

        with pytest.raises(rs.ValidationError, match="input_transforms"):
            to_pmml(model)


class TestPMMLExport:
    def test_pmml_returns_string(self, simple_model):
        result = simple_model.to_pmml()
        assert isinstance(result, str)

    def test_pmml_valid_xml_header(self, simple_model):
        xml = simple_model.to_pmml()
        assert xml.startswith("<?xml")

    def test_pmml_contains_general_regression_model(self, simple_model):
        xml = simple_model.to_pmml()
        assert "GeneralRegressionModel" in xml

    def test_pmml_contains_data_dictionary(self, simple_model):
        xml = simple_model.to_pmml()
        assert "DataDictionary" in xml

    def test_pmml_contains_mining_schema(self, simple_model):
        xml = simple_model.to_pmml()
        assert "MiningSchema" in xml

    def test_pmml_contains_parameter_list(self, simple_model):
        xml = simple_model.to_pmml()
        assert "ParameterList" in xml

    def test_pmml_contains_predictor_variables(self, simple_model):
        xml = simple_model.to_pmml()
        assert "x1" in xml
        assert "x2" in xml
        assert "cat" in xml

    def test_pmml_contains_intercept(self, simple_model):
        xml = simple_model.to_pmml()
        # Intercept should appear as a parameter
        assert "Intercept" in xml or "p0" in xml

    def test_pmml_distribution_link(self, simple_model):
        xml = simple_model.to_pmml()
        assert "poisson" in xml.lower() or "Poisson" in xml
        assert "log" in xml.lower()

    def test_pmml_write_to_file(self, simple_model):
        with tempfile.NamedTemporaryFile(suffix=".pmml", delete=False) as f:
            path = f.name
        try:
            result = simple_model.to_pmml(path=path)
            assert os.path.exists(path)
            size = os.path.getsize(path)
            assert size > 0
            with open(path) as f:
                content = f.read()
            assert content == result
        finally:
            os.unlink(path)

    def test_pmml_gaussian(self, gaussian_model):
        xml = gaussian_model.to_pmml()
        assert isinstance(xml, str)
        assert "GeneralRegressionModel" in xml
        assert "x" in xml

    def test_pmml_to_pmml_function(self, simple_model):
        """Test the standalone to_pmml function."""
        from rustystats.export_pmml import to_pmml

        xml = to_pmml(simple_model)
        assert isinstance(xml, str)
        assert "<?xml" in xml

    def test_pmml_uses_log_exposure_offset_variable(self, simple_poisson_data):
        model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=simple_poisson_data,
            family="poisson",
            exposure="exposure",
        ).fit()
        xml = model.to_pmml()
        assert 'offsetVariable="ln_exposure"' in xml
        assert 'function="ln"' in xml

    def test_pmml_combines_explicit_exposure_and_link_offset(self, simple_poisson_data):
        data = simple_poisson_data.with_columns(pl.Series("adj", np.linspace(-0.1, 0.1, 300)))
        model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=data,
            family="poisson",
            exposure="exposure",
            offset="adj",
        ).fit()
        xml = model.to_pmml()
        assert 'offsetVariable="adj_plus_ln_exposure"' in xml
        assert 'function="+"' in xml

    def test_pmml_rejects_array_exposure(self, simple_poisson_data):
        model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=simple_poisson_data,
            family="poisson",
            exposure=simple_poisson_data["exposure"].to_numpy(),
        ).fit()
        with pytest.raises(rs.ValidationError, match="exposure"):
            model.to_pmml()

    def test_pmml_rejects_array_offset(self, simple_poisson_data):
        model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=simple_poisson_data,
            family="poisson",
            offset=np.zeros(len(simple_poisson_data), dtype=np.float64),
        ).fit()
        with pytest.raises(rs.ValidationError, match="offset"):
            model.to_pmml()

    @pytest.mark.assurance
    def test_pmml_runtime_matches_native_predict_with_exposure_and_categorical(
        self, simple_poisson_data, tmp_path
    ):
        data = simple_poisson_data.with_columns(
            pl.Series("adj", np.linspace(-0.2, 0.2, len(simple_poisson_data)))
        )
        model = rs.glm_dict(
            response="y",
            terms={
                "x1": {"type": "linear"},
                "x2": {"type": "linear"},
                "cat": {"type": "categorical"},
            },
            data=data,
            family="poisson",
            exposure="exposure",
            offset="adj",
        ).fit()

        pmml_pred = _pmml_runtime_predictions(model.to_pmml(), data.head(25), tmp_path)
        native_pred = np.asarray(model.predict(data.head(25)), dtype=np.float64)
        np.testing.assert_allclose(pmml_pred, native_pred, rtol=1e-8, atol=1e-8)

    @pytest.mark.assurance
    def test_pmml_runtime_matches_native_predict_for_gaussian_identity(
        self, gaussian_model, gaussian_data, tmp_path
    ):
        pmml_pred = _pmml_runtime_predictions(gaussian_model.to_pmml(), gaussian_data, tmp_path)
        native_pred = np.asarray(gaussian_model.predict(gaussian_data), dtype=np.float64)
        np.testing.assert_allclose(pmml_pred, native_pred, rtol=1e-8, atol=1e-8)


# ── ONNX Tests ───────────────────────────────────────────────────────────────


class TestONNXInternalHelpers:
    @pytest.mark.parametrize(
        ("link", "expected"),
        [
            ("identity", "identity"),
            ("log", "exp"),
            ("logit", "sigmoid"),
            ("inverse", "1/x"),
            ("sqrt", "square"),
            ("cloglog", "cloglog_inv"),
            ("probit", "probit_inv"),
            ("custom", "custom"),
        ],
    )
    def test_inverse_link_metadata_names(self, link, expected):
        from rustystats.export_onnx import _inverse_link_name

        assert _inverse_link_name(link) == expected

    def test_graph_accumulator_preserves_parallel_serializer_lists(self):
        from rustystats.export_onnx import _GraphAccumulator

        graph = _GraphAccumulator()
        graph.add_init_f64("weights", np.array([[1.0, 2.0], [3.0, 4.0]]))
        graph.add_init_i64("axis", np.array([1]))
        graph.add_node(
            "ReduceSum",
            ["X", "axis"],
            ["Y"],
            [("keepdims", "int", 1), ("noop_with_empty_axes", "float", 0.0)],
        )

        assert graph.uid("tmp") == "tmp_1"
        assert graph.init_names_f64 == ["weights"]
        assert graph.init_shapes_f64 == [[2, 2]]
        assert graph.init_data_f64 == [[1.0, 2.0, 3.0, 4.0]]
        assert graph.init_names_i64 == ["axis"]
        assert graph.init_shapes_i64 == [[1]]
        assert graph.node_ops == ["ReduceSum"]
        assert graph.node_attr_names == [["keepdims", "noop_with_empty_axes"]]
        assert graph.node_attr_types == [["int", "float"]]
        assert graph.node_attr_ints == [[1, 0]]
        assert graph.node_attr_floats == [[0.0, 0.0]]

    def test_piecewise_linear_nodes_encode_expected_interpolation_graph(self):
        from rustystats.export_onnx import _GraphAccumulator, _pwl_nodes

        graph = _GraphAccumulator()
        _pwl_nodes(
            graph,
            "x",
            np.array([0.0, 1.0, 2.0], dtype=np.float64),
            np.array([0.0, 10.0, 30.0], dtype=np.float64),
            "effect",
        )

        assert graph.node_ops[:5] == ["Clip", "Sub", "Div", "Floor", "Clip"]
        assert graph.node_ops.count("Gather") == 2
        assert graph.node_ops[-1] == "Unsqueeze"
        assert graph.node_outputs[-1] == ["effect"]
        assert [0.0, 10.0, 30.0] in graph.init_data_f64

    def test_onnx_full_fake_model_covers_encoded_and_fallback_term_builders(self):
        from rustystats.export_onnx import to_onnx

        builder = SimpleNamespace(
            _cat_encoding_cache={
                "cat_True": SimpleNamespace(levels=["A", "B"]),
                "brand_True": SimpleNamespace(levels=["A", "B"]),
                "region_True": SimpleNamespace(levels=["North", "South"]),
            },
            _te_stats={
                "brand": {
                    "prior": 0.25,
                    "prior_weight": 1.0,
                    "stats": {"A": (2.0, 4), "B": (1.0, 2)},
                    "used_exposure_weighted": False,
                }
            },
            _fe_stats={"region": {"level_counts": {"North": 2, "South": 4}, "max_count": 4}},
            _fitted_splines={},
        )
        model = _fake_export_model(
            family="poisson",
            link="log",
            builder=builder,
            feature_names=[
                "Intercept",
                "x",
                "pos(z)",
                "cat[T.B]",
                "bs(age, 1/2)",
                "TE(brand)",
                "FE(region)",
            ],
            params=[0.1, 0.2, 0.3, 0.4, 0.05, 0.6, 0.7],
        )

        onnx_bytes = to_onnx(model, mode="full")

        assert isinstance(onnx_bytes, bytes)
        assert b"input_names" in onnx_bytes
        assert b"cat_level_maps" in onnx_bytes

    @pytest.mark.parametrize("link", ["identity", "inverse", "sqrt", "custom"])
    def test_onnx_full_intercept_only_graph_covers_inverse_link_branches(self, link):
        from rustystats.export_onnx import to_onnx

        model = _fake_export_model(
            family="gaussian",
            link=link,
            feature_names=["Intercept"],
            params=[2.0],
        )

        onnx_bytes = to_onnx(model, mode="full")

        assert isinstance(onnx_bytes, bytes)
        assert b"RustyStats" in onnx_bytes

    def test_onnx_full_rejects_input_transforms_before_raw_export(self):
        from rustystats.export_onnx import to_onnx

        model = _fake_export_model(input_transforms=[{"name": "territory_lookup"}])

        with pytest.raises(rs.ValidationError, match="input_transforms"):
            to_onnx(model, mode="full")


class TestONNXExport:
    def test_onnx_scoring_returns_bytes(self, simple_model):
        result = simple_model.to_onnx(mode="scoring")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_onnx_full_returns_bytes(self, full_export_model):
        result = full_export_model.to_onnx(mode="full")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_onnx_full_rejects_offset_exposure(self, simple_model, simple_poisson_data):
        # simple_model is fit with offset="exposure"; full mode would silently
        # drop that term, so it must fail closed.
        with pytest.raises(rs.ValidationError, match="offset/exposure"):
            simple_model.to_onnx(mode="full")
        exposure_model = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=simple_poisson_data,
            family="poisson",
            exposure="exposure",
        ).fit()
        with pytest.raises(rs.ValidationError, match="offset/exposure"):
            exposure_model.to_onnx(mode="full")
        # scoring mode (caller supplies the design matrix) is unaffected.
        assert isinstance(exposure_model.to_onnx(mode="scoring"), bytes)

    def test_onnx_scoring_write_to_file(self, simple_model):
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name
        try:
            result = simple_model.to_onnx(path=path, mode="scoring")
            assert os.path.exists(path)
            size = os.path.getsize(path)
            assert size > 0
            with open(path, "rb") as f:
                content = f.read()
            assert content == result
        finally:
            os.unlink(path)

    def test_onnx_full_write_to_file(self, full_export_model):
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name
        try:
            result = full_export_model.to_onnx(path=path, mode="full")
            assert os.path.exists(path)
            with open(path, "rb") as f:
                content = f.read()
            assert content == result
        finally:
            os.unlink(path)

    def test_onnx_default_mode_is_scoring(self, simple_model):
        result_default = simple_model.to_onnx()
        result_scoring = simple_model.to_onnx(mode="scoring")
        assert result_default == result_scoring

    def test_onnx_scoring_vs_full_different(self, full_export_model):
        scoring = full_export_model.to_onnx(mode="scoring")
        full = full_export_model.to_onnx(mode="full")
        assert scoring != full
        # Full model should be larger (has preprocessing nodes)
        assert len(full) > len(scoring)

    def test_onnx_gaussian(self, gaussian_model):
        scoring = gaussian_model.to_onnx(mode="scoring")
        assert isinstance(scoring, bytes)
        assert len(scoring) > 0

    def test_onnx_to_onnx_function(self, simple_model):
        """Test the standalone to_onnx function."""
        from rustystats.export_onnx import to_onnx

        result = to_onnx(simple_model, mode="scoring")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_onnx_rejects_unknown_mode(self, simple_model):
        with pytest.raises(rs.ValidationError, match="mode"):
            simple_model.to_onnx(mode="unknown")

    def test_onnx_contains_rustystats_producer(self, simple_model):
        """Check that RustyStats is embedded as producer name."""
        onnx_bytes = simple_model.to_onnx(mode="scoring")
        assert b"RustyStats" in onnx_bytes

    def test_onnx_contains_metadata(self, simple_model):
        """Check that metadata is embedded in the ONNX model."""
        onnx_bytes = simple_model.to_onnx(mode="scoring")
        assert b"feature_names" in onnx_bytes
        assert b"inverse_link" in onnx_bytes
        assert b"exp" in onnx_bytes  # log link → exp inverse

    def test_onnx_full_contains_input_metadata(self, full_export_model):
        onnx_bytes = full_export_model.to_onnx(mode="full")
        assert b"input_names" in onnx_bytes
        assert b"input_types" in onnx_bytes

    @pytest.mark.assurance
    @pytest.mark.parametrize(
        "family",
        ["gaussian", "poisson", "binomial", "gamma", "negbinomial", "tweedie"],
    )
    def test_onnx_scoring_runtime_matches_fitted_values(self, family):
        model = _fit_onnx_runtime_model(family)
        onnx_pred = _onnx_runtime_predictions(
            model.to_onnx(mode="scoring"),
            _design_matrix_without_intercept(model),
        )
        np.testing.assert_allclose(
            onnx_pred,
            np.asarray(model.fittedvalues, dtype=np.float64),
            rtol=1e-10,
            atol=1e-10,
        )

    @pytest.mark.assurance
    def test_onnx_full_runtime_matches_predict_for_raw_inputs(
        self, full_export_model, simple_poisson_data
    ):
        onnx_pred = _onnx_runtime_predictions(
            full_export_model.to_onnx(mode="full"),
            _full_mode_input_matrix(full_export_model, simple_poisson_data),
        )
        np.testing.assert_allclose(
            onnx_pred,
            np.asarray(full_export_model.predict(simple_poisson_data), dtype=np.float64),
            rtol=1e-10,
            atol=1e-10,
        )


# ── Rust protobuf serializer direct tests ────────────────────────────────────


class TestRustProtobuf:
    def test_build_onnx_glm_scoring_py(self):
        """Test the Rust scoring builder directly."""
        from rustystats._rustystats import build_onnx_glm_scoring_py

        coefs = np.array([0.5, -0.3, 0.1], dtype=np.float64)
        result = build_onnx_glm_scoring_py(
            coefs,
            0.2,
            3,
            "log",
            "poisson",
            ["test_key"],
            ["test_value"],
        )
        assert isinstance(result, bytes)
        assert len(result) > 0
        assert b"RustyStats" in result
        assert b"test_key" in result
        assert b"test_value" in result

    def test_serialize_onnx_graph_py(self):
        """Test the Rust generic serializer directly."""
        from rustystats._rustystats import serialize_onnx_graph_py

        result = serialize_onnx_graph_py(
            # Nodes: single Identity node
            node_ops=["Identity"],
            node_inputs=[["input"]],
            node_outputs=[["output"]],
            node_attr_names=[[]],
            node_attr_types=[[]],
            node_attr_ints=[[]],
            node_attr_floats=[[]],
            # No initializers
            init_names_f64=[],
            init_data_f64=[],
            init_shapes_f64=[],
            init_names_i64=[],
            init_data_i64=[],
            init_shapes_i64=[],
            # Input/output
            input_names=["input"],
            input_types=[11],
            input_shapes=[[-1, 1]],
            output_names=["output"],
            output_types=[11],
            output_shapes=[[-1, 1]],
            # Model info
            ir_version=8,
            opset_version=18,
            producer="test",
            doc_string="test model",
            meta_keys=["k"],
            meta_values=["v"],
        )
        assert isinstance(result, bytes)
        assert len(result) > 0
        assert b"test" in result

    def test_scoring_different_links(self):
        """Test that different link functions produce different ONNX bytes."""
        from rustystats._rustystats import build_onnx_glm_scoring_py

        coefs = np.array([1.0], dtype=np.float64)
        log_bytes = build_onnx_glm_scoring_py(
            coefs,
            0.0,
            1,
            "log",
            "poisson",
            [],
            [],
        )
        logit_bytes = build_onnx_glm_scoring_py(
            coefs,
            0.0,
            1,
            "logit",
            "binomial",
            [],
            [],
        )
        identity_bytes = build_onnx_glm_scoring_py(
            coefs,
            0.0,
            1,
            "identity",
            "gaussian",
            [],
            [],
        )
        assert log_bytes != logit_bytes
        assert log_bytes != identity_bytes
        # Each should contain the respective op name
        assert b"Exp" in log_bytes
        assert b"Sigmoid" in logit_bytes
        assert b"Identity" in identity_bytes


# ── Spline model tests ──────────────────────────────────────────────────────


class TestSplineExport:
    @pytest.fixture
    def spline_model(self):
        np.random.seed(99)
        n = 300
        x = np.random.randn(n)
        data = pl.DataFrame(
            {
                "y": np.random.poisson(np.exp(0.5 * x), n).astype(float),
                "x": x,
            }
        )
        # No offset/exposure: full-mode ONNX export does not embed those terms.
        return rs.glm_dict(
            response="y",
            terms={"x": {"type": "ns", "df": 4}},
            data=data,
            family="poisson",
        ).fit()

    def test_pmml_spline(self, spline_model):
        xml = spline_model.to_pmml()
        assert isinstance(xml, str)
        assert "NormContinuous" in xml or "DerivedField" in xml

    def test_onnx_scoring_spline(self, spline_model):
        result = spline_model.to_onnx(mode="scoring")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_onnx_full_spline(self, spline_model):
        result = spline_model.to_onnx(mode="full")
        assert isinstance(result, bytes)
        # Full model with splines should be larger (piecewise linear nodes)
        scoring = spline_model.to_onnx(mode="scoring")
        assert len(result) > len(scoring)


# ── PMML XML Structure Tests ─────────────────────────────────────────────────


class TestPMMLStructure:
    """Validate PMML XML structure, coefficients, and feature count."""

    def test_pmml_is_valid_xml(self, simple_model):
        """PMML output should parse as valid XML without errors."""
        import xml.etree.ElementTree as ET

        xml_str = simple_model.to_pmml()
        root = ET.fromstring(xml_str)
        assert root.tag.endswith("PMML") or "PMML" in root.tag

    def test_pmml_coefficients_match_model(self, simple_model):
        """PCell beta values in ParamMatrix should match model.params."""
        import xml.etree.ElementTree as ET

        xml_str = simple_model.to_pmml()
        root = ET.fromstring(xml_str)

        # Handle PMML namespace
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"

        # Find all PCell elements in ParamMatrix
        pcells = root.findall(f".//{ns}PCell")
        assert len(pcells) > 0, "PMML should contain PCell elements"

        betas = [float(pc.get("beta")) for pc in pcells]
        np.testing.assert_allclose(
            sorted(betas),
            sorted(simple_model.params),
            atol=1e-6,
            err_msg="PMML PCell betas should match model params",
        )

    def test_pmml_feature_count_matches(self, simple_model):
        """Number of Parameter elements should match len(model.params)."""
        import xml.etree.ElementTree as ET

        xml_str = simple_model.to_pmml()
        root = ET.fromstring(xml_str)

        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"

        parameters = root.findall(f".//{ns}Parameter")
        assert len(parameters) == len(simple_model.params), (
            f"Expected {len(simple_model.params)} Parameter elements, got {len(parameters)}"
        )
