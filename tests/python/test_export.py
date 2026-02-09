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
import numpy as np
import polars as pl
import pytest

import rustystats as rs


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_poisson_data():
    np.random.seed(42)
    n = 300
    return pl.DataFrame({
        'y': np.random.poisson(1.0, n).astype(float),
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'cat': np.random.choice(['A', 'B', 'C'], n),
        'exposure': np.ones(n),
    })


@pytest.fixture
def simple_model(simple_poisson_data):
    return rs.glm_dict(
        response='y',
        terms={
            'x1': {'type': 'linear'},
            'x2': {'type': 'linear'},
            'cat': {'type': 'categorical'},
        },
        data=simple_poisson_data,
        family='poisson',
        offset='exposure',
    ).fit()


@pytest.fixture
def gaussian_data():
    np.random.seed(123)
    n = 200
    x = np.random.randn(n)
    return pl.DataFrame({
        'y': (2.0 + 0.5 * x + np.random.randn(n) * 0.3),
        'x': x,
    })


@pytest.fixture
def gaussian_model(gaussian_data):
    return rs.glm_dict(
        response='y',
        terms={'x': {'type': 'linear'}},
        data=gaussian_data,
        family='gaussian',
    ).fit()


# ── PMML Tests ───────────────────────────────────────────────────────────────

class TestPMMLExport:

    def test_pmml_returns_string(self, simple_model):
        result = simple_model.to_pmml()
        assert isinstance(result, str)

    def test_pmml_valid_xml_header(self, simple_model):
        xml = simple_model.to_pmml()
        assert xml.startswith('<?xml')

    def test_pmml_contains_general_regression_model(self, simple_model):
        xml = simple_model.to_pmml()
        assert 'GeneralRegressionModel' in xml

    def test_pmml_contains_data_dictionary(self, simple_model):
        xml = simple_model.to_pmml()
        assert 'DataDictionary' in xml

    def test_pmml_contains_mining_schema(self, simple_model):
        xml = simple_model.to_pmml()
        assert 'MiningSchema' in xml

    def test_pmml_contains_parameter_list(self, simple_model):
        xml = simple_model.to_pmml()
        assert 'ParameterList' in xml

    def test_pmml_contains_predictor_variables(self, simple_model):
        xml = simple_model.to_pmml()
        assert 'x1' in xml
        assert 'x2' in xml
        assert 'cat' in xml

    def test_pmml_contains_intercept(self, simple_model):
        xml = simple_model.to_pmml()
        # Intercept should appear as a parameter
        assert 'Intercept' in xml or 'p0' in xml

    def test_pmml_distribution_link(self, simple_model):
        xml = simple_model.to_pmml()
        assert 'poisson' in xml.lower() or 'Poisson' in xml
        assert 'log' in xml.lower()

    def test_pmml_write_to_file(self, simple_model):
        with tempfile.NamedTemporaryFile(suffix='.pmml', delete=False) as f:
            path = f.name
        try:
            result = simple_model.to_pmml(path=path)
            assert os.path.exists(path)
            size = os.path.getsize(path)
            assert size > 0
            with open(path, 'r') as f:
                content = f.read()
            assert content == result
        finally:
            os.unlink(path)

    def test_pmml_gaussian(self, gaussian_model):
        xml = gaussian_model.to_pmml()
        assert isinstance(xml, str)
        assert 'GeneralRegressionModel' in xml
        assert 'x' in xml

    def test_pmml_to_pmml_function(self, simple_model):
        """Test the standalone to_pmml function."""
        from rustystats.export_pmml import to_pmml
        xml = to_pmml(simple_model)
        assert isinstance(xml, str)
        assert '<?xml' in xml


# ── ONNX Tests ───────────────────────────────────────────────────────────────

class TestONNXExport:

    def test_onnx_scoring_returns_bytes(self, simple_model):
        result = simple_model.to_onnx(mode='scoring')
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_onnx_full_returns_bytes(self, simple_model):
        result = simple_model.to_onnx(mode='full')
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_onnx_scoring_write_to_file(self, simple_model):
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            path = f.name
        try:
            result = simple_model.to_onnx(path=path, mode='scoring')
            assert os.path.exists(path)
            size = os.path.getsize(path)
            assert size > 0
            with open(path, 'rb') as f:
                content = f.read()
            assert content == result
        finally:
            os.unlink(path)

    def test_onnx_full_write_to_file(self, simple_model):
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            path = f.name
        try:
            result = simple_model.to_onnx(path=path, mode='full')
            assert os.path.exists(path)
            with open(path, 'rb') as f:
                content = f.read()
            assert content == result
        finally:
            os.unlink(path)

    def test_onnx_default_mode_is_scoring(self, simple_model):
        result_default = simple_model.to_onnx()
        result_scoring = simple_model.to_onnx(mode='scoring')
        assert result_default == result_scoring

    def test_onnx_scoring_vs_full_different(self, simple_model):
        scoring = simple_model.to_onnx(mode='scoring')
        full = simple_model.to_onnx(mode='full')
        assert scoring != full
        # Full model should be larger (has preprocessing nodes)
        assert len(full) > len(scoring)

    def test_onnx_gaussian(self, gaussian_model):
        scoring = gaussian_model.to_onnx(mode='scoring')
        assert isinstance(scoring, bytes)
        assert len(scoring) > 0

    def test_onnx_to_onnx_function(self, simple_model):
        """Test the standalone to_onnx function."""
        from rustystats.export_onnx import to_onnx
        result = to_onnx(simple_model, mode='scoring')
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_onnx_contains_rustystats_producer(self, simple_model):
        """Check that RustyStats is embedded as producer name."""
        onnx_bytes = simple_model.to_onnx(mode='scoring')
        assert b'RustyStats' in onnx_bytes

    def test_onnx_contains_metadata(self, simple_model):
        """Check that metadata is embedded in the ONNX model."""
        onnx_bytes = simple_model.to_onnx(mode='scoring')
        assert b'feature_names' in onnx_bytes
        assert b'inverse_link' in onnx_bytes
        assert b'exp' in onnx_bytes  # log link → exp inverse

    def test_onnx_full_contains_input_metadata(self, simple_model):
        onnx_bytes = simple_model.to_onnx(mode='full')
        assert b'input_names' in onnx_bytes
        assert b'input_types' in onnx_bytes


# ── Rust protobuf serializer direct tests ────────────────────────────────────

class TestRustProtobuf:

    def test_build_onnx_glm_scoring_py(self):
        """Test the Rust scoring builder directly."""
        from rustystats._rustystats import build_onnx_glm_scoring_py

        coefs = np.array([0.5, -0.3, 0.1], dtype=np.float64)
        result = build_onnx_glm_scoring_py(
            coefs, 0.2, 3, 'log', 'poisson',
            ['test_key'], ['test_value'],
        )
        assert isinstance(result, bytes)
        assert len(result) > 0
        assert b'RustyStats' in result
        assert b'test_key' in result
        assert b'test_value' in result

    def test_serialize_onnx_graph_py(self):
        """Test the Rust generic serializer directly."""
        from rustystats._rustystats import serialize_onnx_graph_py

        result = serialize_onnx_graph_py(
            # Nodes: single Identity node
            node_ops=['Identity'],
            node_inputs=[['input']],
            node_outputs=[['output']],
            node_attr_names=[[]],
            node_attr_types=[[]],
            node_attr_ints=[[]],
            node_attr_floats=[[]],
            # No initializers
            init_names_f64=[], init_data_f64=[], init_shapes_f64=[],
            init_names_i64=[], init_data_i64=[], init_shapes_i64=[],
            # Input/output
            input_names=['input'], input_types=[11], input_shapes=[[-1, 1]],
            output_names=['output'], output_types=[11], output_shapes=[[-1, 1]],
            # Model info
            ir_version=8, opset_version=18,
            producer='test', doc_string='test model',
            meta_keys=['k'], meta_values=['v'],
        )
        assert isinstance(result, bytes)
        assert len(result) > 0
        assert b'test' in result

    def test_scoring_different_links(self):
        """Test that different link functions produce different ONNX bytes."""
        from rustystats._rustystats import build_onnx_glm_scoring_py

        coefs = np.array([1.0], dtype=np.float64)
        log_bytes = build_onnx_glm_scoring_py(
            coefs, 0.0, 1, 'log', 'poisson', [], [],
        )
        logit_bytes = build_onnx_glm_scoring_py(
            coefs, 0.0, 1, 'logit', 'binomial', [], [],
        )
        identity_bytes = build_onnx_glm_scoring_py(
            coefs, 0.0, 1, 'identity', 'gaussian', [], [],
        )
        assert log_bytes != logit_bytes
        assert log_bytes != identity_bytes
        # Each should contain the respective op name
        assert b'Exp' in log_bytes
        assert b'Sigmoid' in logit_bytes
        assert b'Identity' in identity_bytes


# ── Spline model tests ──────────────────────────────────────────────────────

class TestSplineExport:

    @pytest.fixture
    def spline_model(self):
        np.random.seed(99)
        n = 300
        x = np.random.randn(n)
        data = pl.DataFrame({
            'y': np.random.poisson(np.exp(0.5 * x), n).astype(float),
            'x': x,
            'exposure': np.ones(n),
        })
        return rs.glm_dict(
            response='y',
            terms={'x': {'type': 'ns', 'df': 4}},
            data=data,
            family='poisson',
            offset='exposure',
        ).fit()

    def test_pmml_spline(self, spline_model):
        xml = spline_model.to_pmml()
        assert isinstance(xml, str)
        assert 'NormContinuous' in xml or 'DerivedField' in xml

    def test_onnx_scoring_spline(self, spline_model):
        result = spline_model.to_onnx(mode='scoring')
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_onnx_full_spline(self, spline_model):
        result = spline_model.to_onnx(mode='full')
        assert isinstance(result, bytes)
        # Full model with splines should be larger (piecewise linear nodes)
        scoring = spline_model.to_onnx(mode='scoring')
        assert len(result) > len(scoring)
