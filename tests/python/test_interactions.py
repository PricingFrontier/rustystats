"""
Tests for interaction term support in RustyStats.

Tests cover:
- Continuous × Continuous interactions
- Categorical × Continuous interactions
- Categorical × Categorical interactions
- Pure interaction terms
- InteractionTerm dataclass properties
- Design matrix construction via dict API
"""

import numpy as np
import polars as pl
import pytest
import rustystats as rs
import rustystats.interactions as interactions_mod
from rustystats.exceptions import DesignMatrixError, EncodingError, PredictionError, ValidationError
from rustystats.interactions import (
    CategoricalEncoding,
    CategoricalTermSpec,
    ConstraintTermSpec,
    FrequencyEncodingTermSpec,
    IdentityTermSpec,
    InteractionBuilder,
    InteractionTerm,
    ParsedFormula,
    TargetEncodingTermSpec,
)
from rustystats.splines import SplineTerm

# =============================================================================
# Design Matrix Construction Tests
# =============================================================================


class TestInteractionBuilder:
    """Test the InteractionBuilder class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        np.random.seed(42)
        n = 100
        return pl.DataFrame(
            {
                "y": np.random.poisson(1, n),
                "x1": np.random.uniform(0, 10, n),
                "x2": np.random.uniform(0, 10, n),
                "cat1": np.random.choice(["A", "B", "C"], n),
                "cat2": np.random.choice(["X", "Y"], n),
            }
        )

    def test_continuous_continuous(self, sample_data):
        """Test continuous × continuous interaction."""
        parsed = ParsedFormula(
            response="y",
            main_effects=["x1", "x2"],
            interactions=[InteractionTerm(factors=["x1", "x2"], categorical_flags=[False, False])],
            categorical_vars=set(),
            has_intercept=True,
        )
        builder = InteractionBuilder(sample_data)
        _y, X, names = builder.build_design_matrix_from_parsed(parsed)

        assert "Intercept" in names
        assert "x1" in names
        assert "x2" in names
        assert "x1:x2" in names
        assert X.shape[1] == 4

        # Verify interaction column is product of x1 and x2
        x1_col = names.index("x1")
        x2_col = names.index("x2")
        int_col = names.index("x1:x2")
        np.testing.assert_allclose(X[:, int_col], X[:, x1_col] * X[:, x2_col])

    def test_categorical_continuous(self, sample_data):
        """Test categorical × continuous interaction."""
        parsed = ParsedFormula(
            response="y",
            main_effects=["cat1", "x1"],
            interactions=[InteractionTerm(factors=["cat1", "x1"], categorical_flags=[True, False])],
            categorical_vars={"cat1"},
            has_intercept=True,
        )
        builder = InteractionBuilder(sample_data)
        _y, X, names = builder.build_design_matrix_from_parsed(parsed)

        assert "Intercept" in names
        assert "cat1[T.B]" in names
        assert "cat1[T.C]" in names
        assert "x1" in names
        assert "cat1[T.B]:x1" in names
        assert "cat1[T.C]:x1" in names
        assert X.shape[1] == 6

        # Verify interaction: cat1[T.B]:x1 should be cat1[T.B] * x1
        cat_b_col = names.index("cat1[T.B]")
        x1_col = names.index("x1")
        int_col = names.index("cat1[T.B]:x1")
        np.testing.assert_allclose(X[:, int_col], X[:, cat_b_col] * X[:, x1_col])

    def test_categorical_categorical(self, sample_data):
        """Test categorical × categorical interaction."""
        parsed = ParsedFormula(
            response="y",
            main_effects=["cat1", "cat2"],
            interactions=[
                InteractionTerm(factors=["cat1", "cat2"], categorical_flags=[True, True])
            ],
            categorical_vars={"cat1", "cat2"},
            has_intercept=True,
        )
        builder = InteractionBuilder(sample_data)
        _y, X, names = builder.build_design_matrix_from_parsed(parsed)

        # cat1 has 3 levels (2 dummies), cat2 has 2 levels (1 dummy)
        # Interaction: 2 × 1 = 2 columns
        # Total: 1 + 2 + 1 + 2 = 6
        assert X.shape[1] == 6

        # Check interaction column names
        assert "cat1[T.B]:cat2[T.Y]" in names
        assert "cat1[T.C]:cat2[T.Y]" in names

    def test_two_categorical_continuous_interaction(self):
        """Test direct two-categorical × continuous interaction construction."""
        df = pl.DataFrame(
            {
                "y": [0, 1, 2, 3, 4, 5],
                "x": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                "cat1": ["A", "B", "C", "B", "C", "A"],
                "cat2": ["X", "Y", "Y", "Z", "X", "Z"],
            }
        )
        parsed = ParsedFormula(
            response="y",
            main_effects=["x"],
            interactions=[
                InteractionTerm(
                    factors=["cat1", "cat2", "x"],
                    categorical_flags=[True, True, False],
                )
            ],
            categorical_vars={"cat1", "cat2"},
            has_intercept=True,
        )

        builder = InteractionBuilder(df)
        _y, X, names = builder.build_design_matrix_from_parsed(parsed)

        expected_names = [
            "cat1[T.B]:cat2[T.Y]:x",
            "cat1[T.B]:cat2[T.Z]:x",
            "cat1[T.C]:cat2[T.Y]:x",
            "cat1[T.C]:cat2[T.Z]:x",
        ]
        for name in expected_names:
            assert name in names

        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [20.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 30.0, 0.0],
                [0.0, 40.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        actual = X[:, [names.index(name) for name in expected_names]]
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(builder.transform_new_data(df), X)

    def test_linear_predict_new_data_matches_dense_transform_with_interactions(self):
        """Prediction-only scorer preserves dense design column order."""
        rng = np.random.default_rng(123)
        n = 200
        df = pl.DataFrame(
            {
                "y": rng.poisson(1.0, n),
                "x": rng.normal(size=n),
                "age": rng.uniform(18.0, 80.0, n),
                "cat1": rng.choice([f"A{i}" for i in range(6)], n),
                "cat2": rng.choice([f"B{i}" for i in range(5)], n),
            }
        )
        spline = SplineTerm(var_name="age", spline_type="bs", df=5, degree=3)
        parsed = ParsedFormula(
            response="y",
            main_effects=["cat1", "x"],
            interactions=[
                InteractionTerm(factors=["cat1", "cat2"], categorical_flags=[True, True]),
                InteractionTerm(
                    factors=["cat1", "cat2", "x"],
                    categorical_flags=[True, True, False],
                ),
                InteractionTerm(factors=["cat1", "age"], categorical_flags=[True, False]),
                InteractionTerm(factors=["age", "x"], categorical_flags=[False, False]),
            ],
            categorical_vars={"cat1"},
            spline_terms=[spline],
            has_intercept=True,
        )

        builder = InteractionBuilder(df)
        _y, X, _names = builder.build_design_matrix_from_parsed(parsed)
        params = rng.normal(size=X.shape[1])

        dense_eta = builder.transform_new_data(df) @ params
        direct_eta = builder.linear_predict_new_data(df, params)
        np.testing.assert_allclose(direct_eta, dense_eta, rtol=1e-12, atol=1e-12)

    def test_linear_predict_new_data_matches_dense_transform_for_three_way_mixed_fallback(self):
        """Prediction scorer preserves dense order for rare 3+ categorical mixed terms."""
        df = pl.DataFrame(
            {
                "y": [0, 1, 2, 3, 4, 5, 6, 7],
                "x": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
                "cat1": ["A", "B", "B", "A", "C", "C", "B", "A"],
                "cat2": ["U", "U", "V", "V", "U", "V", "V", "U"],
                "cat3": ["M", "N", "M", "N", "M", "N", "N", "M"],
            }
        )
        parsed = ParsedFormula(
            response="y",
            main_effects=[],
            interactions=[
                InteractionTerm(
                    factors=["cat1", "cat2", "cat3", "x"],
                    categorical_flags=[True, True, True, False],
                )
            ],
            categorical_vars=set(),
            has_intercept=True,
        )

        builder = InteractionBuilder(df)
        _y, x, names = builder.build_design_matrix_from_parsed(parsed)
        params = np.linspace(-0.3, 0.7, x.shape[1])

        assert any("cat1" in name and "cat2" in name and "cat3" in name for name in names)
        np.testing.assert_allclose(
            builder.linear_predict_new_data(df, params),
            builder.transform_new_data(df) @ params,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_linear_predict_new_data_matches_dense_transform_for_target_encoded_interaction(self):
        """Prediction scorer uses fitted TE statistics for TE(...) interaction factors."""
        df = pl.DataFrame(
            {
                "y": [0.0, 1.0, 3.0, 2.0, 4.0, 1.0, 5.0, 2.0],
                "x": [1.0, 2.0, 1.5, 2.5, 3.0, 1.2, 2.8, 3.5],
                "brand": ["A", "B", "A", "C", "B", "C", "B", "D"],
            }
        )
        parsed = ParsedFormula(
            response="y",
            main_effects=["x"],
            target_encoding_terms=[TargetEncodingTermSpec(var_name="brand")],
            interactions=[
                InteractionTerm(
                    factors=["TE(brand)", "x"],
                    categorical_flags=[False, False],
                )
            ],
            categorical_vars=set(),
            has_intercept=True,
        )

        builder = InteractionBuilder(df)
        _y, x, _names = builder.build_design_matrix_from_parsed(parsed, seed=123)
        params = np.linspace(0.1, 0.5, x.shape[1])
        score = pl.DataFrame(
            {
                "y": [0.0, 0.0, 0.0],
                "x": [1.0, 4.0, 2.0],
                "brand": ["A", "unseen", "B"],
            }
        )

        np.testing.assert_allclose(
            builder.linear_predict_new_data(score, params),
            builder.transform_new_data(score) @ params,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_pure_interaction(self, sample_data):
        """Test pure interaction without main effects for some variables."""
        parsed = ParsedFormula(
            response="y",
            main_effects=["x1"],
            interactions=[InteractionTerm(factors=["cat1", "x2"], categorical_flags=[True, False])],
            categorical_vars={"cat1"},
            has_intercept=True,
        )
        builder = InteractionBuilder(sample_data)
        _y, _X, names = builder.build_design_matrix_from_parsed(parsed)

        # Should have: Intercept, x1, cat1[T.B]:x2, cat1[T.C]:x2
        assert "Intercept" in names
        assert "x1" in names
        assert "cat1[T.B]:x2" in names
        assert "cat1[T.C]:x2" in names
        assert "cat1[T.B]" not in names
        assert "cat1[T.C]" not in names
        assert "x2" not in names


class TestInteractionBuilderHelperContracts:
    def test_parser_lookup_and_cache_clearing_contracts(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "x": [1.0, 2.0, 3.0]})
        spline = SplineTerm("x", "bs", df=4)
        parsed = ParsedFormula(
            response="y",
            main_effects=[],
            interactions=[],
            categorical_vars=set(),
            spline_terms=[spline],
            target_encoding_terms=[TargetEncodingTermSpec("brand")],
            has_intercept=True,
        )
        builder = InteractionBuilder(df)

        assert builder._parse_spline_factor("x") is None
        assert builder._parse_te_factor("TE(brand)") is None

        builder._parsed_formula = parsed
        assert builder._parse_spline_factor("x") is spline
        assert builder._parse_te_factor("TE(brand)").var_name == "brand"
        assert builder._parse_te_factor("brand").var_name == "brand"
        assert builder._parse_te_factor("missing") is None
        assert builder.get_all_spline_terms() == ([], [])
        builder._all_spline_terms = [spline]
        builder._all_spline_col_indices = [(0, 3)]
        assert builder.get_all_spline_terms() == ([spline], [(0, 3)])

        builder._cat_encoding_cache["cat_True"] = CategoricalEncoding(
            encoding=np.ones((3, 1)),
            names=["cat[T.B]"],
            indices=np.array([0, 1, 0], dtype=np.int32),
            levels=["A", "B"],
        )
        builder._cont_cache["x"] = np.array([1.0, 2.0, 3.0])
        builder._last_X = np.ones((3, 2))
        builder._last_names = ["Intercept", "x"]

        builder.clear_caches()

        cached = builder._cat_encoding_cache["cat_True"]
        assert cached.encoding is None
        assert cached.indices is None
        assert builder._cont_cache == {}
        assert builder._last_X is None
        assert builder._last_names is None

    def test_categorical_encoding_cache_can_rebuild_matrix_from_indices(self):
        df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "cat": ["A", "B", "A"]})
        builder = InteractionBuilder(df)
        builder._cat_encoding_cache["cat_True"] = CategoricalEncoding(
            encoding=None,
            names=[],
            indices=np.array([0, 1, 0], dtype=np.int32),
            levels=["A", "B"],
        )

        encoding, names = builder._get_categorical_encoding("cat")

        np.testing.assert_allclose(encoding, [[0.0], [1.0], [0.0]])
        assert names == ["cat[T.B]"]

    def test_validate_design_matrix_reports_numerical_pathologies(self, capsys, monkeypatch):
        builder = InteractionBuilder(pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0]}))

        with pytest.raises(DesignMatrixError, match="No design matrix"):
            builder.validate_design_matrix(verbose=False)

        X_with_bad_values = np.array(
            [
                [1.0, 1.0, 2.0, 1.0, 10.0],
                [1.0, 1.0, 4.0, np.nan, 20.0],
                [1.0, 1.0, 6.0, np.inf, 30.0],
                [1.0, 1.0, 8.0, 4.0, 40.0],
            ]
        )
        names = ["Intercept", "constant", "x", "bad", "x_clone"]

        monkeypatch.setattr(np.linalg, "matrix_rank", lambda _x: 4)
        monkeypatch.setattr(np.linalg, "cond", lambda _x: 1.0)
        bad_value_result = builder.validate_design_matrix(
            X_with_bad_values, names, corr_threshold=0.9, verbose=False
        )
        assert bad_value_result["valid"] is False
        assert any("NaN" in suggestion for suggestion in bad_value_result["suggestions"])
        assert any("Inf" in suggestion for suggestion in bad_value_result["suggestions"])
        monkeypatch.undo()

        X = np.array(
            [
                [1.0, 1.0, 1.0, 2.0],
                [1.0, 1.0, 2.0, 4.0],
                [1.0, 1.0, 3.0, 6.0],
                [1.0, 1.0, 4.0, 8.0],
            ]
        )
        names = ["Intercept", "constant", "x", "x_clone"]

        result = builder.validate_design_matrix(X, names, corr_threshold=0.9, verbose=True)

        assert result["valid"] is False
        assert "constant" in result["zero_variance_columns"]
        assert result["rank"] < X.shape[1]
        assert result["problematic_columns"]
        assert "DESIGN MATRIX VALIDATION" in capsys.readouterr().out

    def test_validate_design_matrix_wraps_linalg_failures(self, monkeypatch):
        builder = InteractionBuilder(pl.DataFrame({"y": [1.0, 2.0, 3.0]}))
        X = np.eye(3)
        names = ["a", "b", "c"]

        def fail_rank(_x):
            raise RuntimeError("rank boom")

        monkeypatch.setattr(np.linalg, "matrix_rank", fail_rank)
        with pytest.raises(DesignMatrixError, match="Failed to compute matrix rank"):
            builder.validate_design_matrix(X, names, verbose=False)

        monkeypatch.setattr(np.linalg, "matrix_rank", lambda _x: 3)

        def fail_cond(_x):
            raise RuntimeError("cond boom")

        monkeypatch.setattr(np.linalg, "cond", fail_cond)
        with pytest.raises(DesignMatrixError, match="Failed to compute condition number"):
            builder.validate_design_matrix(X, names, verbose=False)

        monkeypatch.setattr(np.linalg, "cond", lambda _x: 1.0)

        def fail_corrcoef(_x):
            raise RuntimeError("corr boom")

        monkeypatch.setattr(np, "corrcoef", fail_corrcoef)
        with pytest.raises(DesignMatrixError, match="Failed to compute column correlations"):
            builder.validate_design_matrix(
                np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [1.0, 1.0, 2.0],
                        [1.0, 2.0, 4.0],
                    ]
                ),
                names,
                verbose=False,
            )

    def test_prediction_categorical_cache_handles_empty_constant_and_vectorized_paths(self):
        train = pl.DataFrame(
            {
                "y": [1.0, 2.0, 3.0, 4.0],
                "cat": ["A", "B", "C", "A"],
                "cat2": ["X", "Y", "X", "Y"],
            }
        )
        parsed = ParsedFormula(
            response="y",
            main_effects=["cat"],
            interactions=[InteractionTerm(["cat", "cat2"], [True, True])],
            categorical_vars={"cat"},
            has_intercept=True,
        )
        builder = InteractionBuilder(train)
        builder.build_design_matrix_from_parsed(parsed)

        empty = pl.DataFrame({"y": [], "cat": [], "cat2": []}, schema=train.schema)
        cache = builder._prepare_prediction_categorical_index_cache(empty, parsed)
        assert cache["cat"][0].size == 0
        assert cache["cat2"][0].size == 0

        constant = pl.DataFrame({"y": [0.0, 0.0], "cat": ["B", "B"], "cat2": ["Y", "Y"]})
        cache = builder._prepare_prediction_categorical_index_cache(constant, parsed)
        np.testing.assert_array_equal(cache["cat"][0], np.array([1, 1], dtype=np.int32))
        assert builder._map_to_training_indices_cached(constant, "cat", cache) is cache["cat"]

        varied = pl.DataFrame(
            {"y": [0.0, 0.0, 0.0], "cat": ["A", "C", "unknown"], "cat2": ["X", "Y", "X"]}
        )
        cache = builder._prepare_prediction_categorical_index_cache(varied, parsed)
        np.testing.assert_array_equal(cache["cat"][0], [0, 2, 0])
        uncached: dict[str, tuple[np.ndarray, list[str]]] = {}
        first = builder._map_to_training_indices_cached(varied, "cat", uncached)
        second = builder._map_to_training_indices_cached(varied, "cat", uncached)
        assert first is second
        np.testing.assert_array_equal(
            builder._map_to_training_indices_cached(varied, "cat", None)[0],
            first[0],
        )

    def test_spline_prediction_cache_and_constant_basis_contracts(self, monkeypatch):
        df = pl.DataFrame({"y": np.arange(8.0), "age": np.linspace(20.0, 80.0, 8)})
        spline = SplineTerm("age", "bs", df=4)
        parsed = ParsedFormula(
            response="y",
            main_effects=[],
            interactions=[],
            categorical_vars=set(),
            spline_terms=[spline],
            has_intercept=True,
        )
        builder = InteractionBuilder(df)
        builder.build_design_matrix_from_parsed(parsed)

        assert builder._constant_spline_basis_new(pl.DataFrame({"age": []}), spline) is None
        constant_basis = builder._constant_spline_basis_new(
            pl.DataFrame({"age": [30.0, 30.0]}), spline
        )
        assert constant_basis is not None
        assert constant_basis.shape[0] == 2
        assert constant_basis.strides[0] == 0

        monkeypatch.setattr(interactions_mod, "_PREDICT_SPLINE_CACHE_MIN_ROWS", 2)
        monkeypatch.setattr(interactions_mod, "_PREDICT_SPLINE_CACHE_MAX_BYTES", 1)
        values = np.linspace(20.0, 80.0, 4)

        first = builder._cached_spline_basis_from_values(spline, values)
        second = builder._cached_spline_basis_from_values(spline, values)

        np.testing.assert_allclose(first, second)
        assert hasattr(builder, "_spline_basis_predict_cache")
        assert builder._spline_basis_predict_cache_bytes == 0

    def test_rare_interaction_construction_paths_have_explicit_contracts(self):
        df = pl.DataFrame(
            {
                "y": [1.0, 2.0, 3.0, 4.0],
                "x": [1.0, 2.0, 3.0, 4.0],
                "z": [2.0, 3.0, 4.0, 5.0],
                "plain": [1.5, 2.5, 3.5, 4.5],
                "brand": ["A", "B", "A", "B"],
                "region": ["N", "S", "N", "S"],
                "single": ["only", "only", "only", "only"],
            }
        )
        age_spline = SplineTerm("x", "bs", df=4)
        z_spline = SplineTerm("z", "bs", df=4)
        parsed = ParsedFormula(
            response="y",
            main_effects=[],
            interactions=[],
            categorical_vars={"brand", "region", "single"},
            spline_terms=[age_spline, z_spline],
            target_encoding_terms=[TargetEncodingTermSpec("brand")],
            has_intercept=False,
        )
        builder = InteractionBuilder(df)
        builder._parsed_formula = parsed

        one_spline_cols, one_spline_names = builder._build_continuous_interaction(
            InteractionTerm(["x"], [False])
        )
        assert one_spline_cols.shape[1] == len(one_spline_names)
        assert all(name.startswith("bs(x") for name in one_spline_names)

        multi_spline_cols, multi_spline_names = builder._build_continuous_interaction(
            InteractionTerm(["x", "z"], [False, False])
        )
        assert multi_spline_cols.shape[1] == len(multi_spline_names)
        assert multi_spline_cols.shape[1] == one_spline_cols.shape[1] * one_spline_cols.shape[1]

        with pytest.raises(EncodingError, match="TE encoding"):
            builder._build_continuous_interaction(
                InteractionTerm(["TE(brand)", "plain"], [False, False])
            )

        te_cols, te_names = builder._build_continuous_interaction(
            InteractionTerm(["TE(brand)"], [False]),
            te_encodings={"TE(brand)": np.array([0.1, 0.2, 0.3, 0.4])},
        )
        assert te_names == ["TE(brand)"]
        np.testing.assert_allclose(te_cols.ravel(), [0.1, 0.2, 0.3, 0.4])

        zero_cols, zero_names = builder._build_mixed_interaction(
            InteractionTerm(["single", "x"], [True, False])
        )
        assert zero_cols.shape == (df.height, 0)
        assert zero_names == []

        mixed_cols, mixed_names = builder._build_mixed_interaction(
            InteractionTerm(["brand", "region", "x", "z"], [True, True, False, False])
        )
        assert mixed_cols.shape[1] == len(mixed_names)
        assert any(":bs(x," in name for name in mixed_names)
        assert any(":bs(z," in name for name in mixed_names)

        cont_spline_cols, cont_spline_names = builder._build_continuous_interaction(
            InteractionTerm(
                ["plain", "x", "z"],
                [False, False, False],
                force_linear={"plain"},
            )
        )
        assert cont_spline_cols.shape[1] == len(cont_spline_names)
        assert all(name.startswith("plain:") for name in cont_spline_names)

        three_way_cont, three_way_name = builder._build_continuous_interaction(
            InteractionTerm(
                ["x", "z", "plain"],
                [False, False, False],
                force_linear={"x", "z", "plain"},
            )
        )
        np.testing.assert_allclose(
            three_way_cont.ravel(),
            df["x"].to_numpy() * df["z"].to_numpy() * df["plain"].to_numpy(),
        )
        assert three_way_name == ["x:z:plain"]

        spline_cont_cols, spline_cont_names = builder._build_mixed_interaction(
            InteractionTerm(
                ["brand", "region", "x", "plain"],
                [True, True, False, False],
                force_linear={"plain"},
            )
        )
        assert spline_cont_cols.shape[1] == len(spline_cont_names)
        assert all(name.endswith(":plain") for name in spline_cont_names)

        zero_standard, zero_standard_names = builder._build_mixed_interaction(
            InteractionTerm(["single", "plain"], [True, False])
        )
        assert zero_standard.shape == (df.height, 0)
        assert zero_standard_names == []

        zero_two_cat, zero_two_cat_names = builder._build_mixed_interaction(
            InteractionTerm(["single", "region", "plain"], [True, True, False])
        )
        assert zero_two_cat.shape == (df.height, 0)
        assert zero_two_cat_names == []

        zero_three_cat, zero_three_cat_names = builder._build_mixed_interaction(
            InteractionTerm(["single", "brand", "region", "plain"], [True, True, True, False])
        )
        assert zero_three_cat.shape == (df.height, 0)
        assert zero_three_cat_names == []

    def test_target_encoding_three_way_and_spline_cache_contracts(self, monkeypatch):
        df = pl.DataFrame(
            {
                "y": [1.0, 0.0, 2.0, 1.0, 3.0],
                "brand": ["A", "B", "A", "C", "B"],
                "region": ["N", "N", "S", "S", "N"],
                "channel": ["web", "agent", "web", "agent", "web"],
                "age": [20.0, 30.0, 40.0, 50.0, 60.0],
            }
        )
        builder = InteractionBuilder(df)
        encoded, name, stats = builder._build_target_encoding_columns(
            TargetEncodingTermSpec(
                "brand:region:channel",
                interaction_vars=["brand", "region", "channel"],
                prior_weight=1.0,
                n_permutations=1,
            ),
            df["y"].to_numpy(),
            seed=123,
        )

        assert encoded.shape == (df.height,)
        assert name.startswith("TE(")
        assert stats["interaction_vars"] == ["brand", "region", "channel"]

        spline = SplineTerm("age", "bs", df=4)
        spline.transform(df["age"].to_numpy())
        builder._fitted_splines["age"] = spline
        monkeypatch.setattr(interactions_mod, "_PREDICT_SPLINE_CACHE_MIN_ROWS", 2)
        monkeypatch.setattr(interactions_mod, "_PREDICT_SPLINE_CACHE_MAX_BYTES", 10_000_000)

        first = builder._cached_spline_basis_from_values(spline, df["age"].to_numpy())
        second = builder._cached_spline_basis_from_values(spline, df["age"].to_numpy())

        assert second is first
        assert builder._spline_basis_predict_cache_bytes >= first.nbytes

    def test_expression_stack_and_unfitted_prediction_edge_contracts(self):
        df = pl.DataFrame(
            {
                "y": [1.0, 2.0, 3.0],
                "x": [2.0, 3.0, 4.0],
                "power": [2.0, 3.0, 2.0],
                "cat": ["A", "B", "A"],
            }
        )
        builder = InteractionBuilder(df)

        powered = df.select(builder._convert_expression_to_polars("x ** power").alias("out"))[
            "out"
        ].to_numpy()
        np.testing.assert_allclose(powered, [4.0, 27.0, 16.0])

        with pytest.raises(ValidationError, match="Cannot parse expression"):
            builder._convert_expression_to_polars("x // 2")

        empty_cat, empty_names = builder._build_categorical_level_indicators(
            CategoricalTermSpec("cat", levels=[])
        )
        assert empty_cat.shape == (df.height, 0)
        assert empty_names == []

        empty_stack = InteractionBuilder._stack_columns([], n_rows=3, dtype=np.float32)
        assert empty_stack.dtype == np.float32
        np.testing.assert_allclose(empty_stack, np.ones((3, 1), dtype=np.float32))

        cast_stack = InteractionBuilder._stack_columns([np.array([1.0, 2.0, 3.0])], 3, np.float32)
        assert cast_stack.dtype == np.float32

        with pytest.raises(PredictionError, match="build_design_matrix"):
            builder.transform_new_data(df)
        with pytest.raises(PredictionError, match="build_design_matrix"):
            builder.linear_predict_new_data(df, np.ones(1))

    def test_transform_and_linear_predict_cover_fe_identity_constraints_and_level_terms(self):
        df = pl.DataFrame(
            {
                "y": [1.0, 2.0, 3.0, 4.0],
                "x": [1.0, 2.0, 3.0, 4.0],
                "cat": ["A", "B", "A", "C"],
                "brand": ["foo", "bar", "foo", "baz"],
            }
        )
        parsed = ParsedFormula(
            response="y",
            main_effects=["x"],
            interactions=[],
            categorical_vars=set(),
            frequency_encoding_terms=[FrequencyEncodingTermSpec("brand")],
            identity_terms=[IdentityTermSpec("x ** 2")],
            constraint_terms=[ConstraintTermSpec("x", "pos")],
            categorical_terms=[CategoricalTermSpec("cat", levels=["A", "C"])],
            has_intercept=True,
        )
        builder = InteractionBuilder(df)
        _y, x_train, names = builder.build_design_matrix_from_parsed(parsed)
        params = np.linspace(0.1, 0.7, x_train.shape[1])
        score = df.with_columns((pl.col("x") + 1.0).alias("x"))

        dense = builder.transform_new_data(score)
        eta = builder.linear_predict_new_data(score, params)

        assert "FE(brand)" in names
        assert "I(x ** 2)" in names
        assert "pos(x)" in names
        assert "cat[A]" in names
        assert dense.shape[1] == params.shape[0]
        np.testing.assert_allclose(eta, dense @ params, rtol=1e-12)

        builder._last_names = []
        with pytest.raises(PredictionError, match="produced"):
            builder.linear_predict_new_data(score, np.r_[params, 99.0])

        empty_new, empty_names = builder._build_categorical_level_indicators_new(
            CategoricalTermSpec("cat", levels=[]),
            score,
        )
        assert empty_new.shape == (score.height, 0)
        assert empty_names == []

    def test_low_level_prediction_helpers_cover_zero_width_and_missing_state(self):
        train = pl.DataFrame(
            {
                "y": [1.0, 2.0, 3.0],
                "x": [1.0, 2.0, 3.0],
                "single": ["only", "only", "only"],
                "single2": ["ref", "ref", "ref"],
                "brand": ["A", "B", "A"],
            }
        )
        parsed = ParsedFormula(
            response="y",
            main_effects=["single"],
            interactions=[
                InteractionTerm(["single", "single2"], [True, True]),
                InteractionTerm(["single", "x"], [True, False]),
            ],
            categorical_vars={"single", "single2"},
            has_intercept=False,
        )
        builder = InteractionBuilder(train)
        builder.build_design_matrix_from_parsed(parsed)

        eta = np.zeros(train.height, dtype=np.float64)
        assert (
            builder._accumulate_linear_prediction_block(
                eta,
                np.zeros((train.height, 0)),
                np.array([], dtype=np.float64),
                0,
            )
            == 0
        )
        np.testing.assert_allclose(eta, 0.0)

        eta = np.zeros(train.height, dtype=np.float64)
        assert (
            builder._accumulate_categorical_prediction_new(
                eta,
                train,
                "single",
                np.array([], dtype=np.float64),
                0,
                None,
            )
            == 0
        )
        np.testing.assert_allclose(eta, 0.0)

        eta = np.zeros(train.height, dtype=np.float64)
        assert (
            builder._accumulate_categorical_interaction_prediction_new(
                eta,
                train,
                parsed.interactions[0],
                np.array([], dtype=np.float64),
                0,
                None,
            )
            == 0
        )
        np.testing.assert_allclose(eta, 0.0)

        eta = np.zeros(train.height, dtype=np.float64)
        assert (
            builder._accumulate_mixed_interaction_prediction_new(
                eta,
                train,
                parsed.interactions[1],
                train.height,
                np.array([], dtype=np.float64),
                0,
                None,
                {},
            )
            == 0
        )
        np.testing.assert_allclose(eta, 0.0)

        np.testing.assert_allclose(
            builder._continuous_product_new(train, ["x", "x"], builder.dtype),
            train["x"].to_numpy() ** 2,
        )
        assert builder._get_categorical_names("not_fit") == []

        with pytest.raises(EncodingError, match="Target encoding"):
            builder._encode_target_new(train, TargetEncodingTermSpec("brand"))
        with pytest.raises(EncodingError, match="Frequency encoding"):
            builder._encode_frequency_new(train, FrequencyEncodingTermSpec("brand"))

        empty_builder = InteractionBuilder(pl.DataFrame({"y": [1.0, 2.0]}))
        _y, empty_x, empty_names = empty_builder.build_design_matrix_from_parsed(
            ParsedFormula(
                response="y",
                main_effects=[],
                interactions=[],
                categorical_vars=set(),
                has_intercept=False,
            )
        )
        np.testing.assert_allclose(empty_x, np.ones((2, 1)))
        assert empty_names == ["Intercept"]

    def test_mixed_prediction_accumulator_with_constant_spline_matches_dense_block(self):
        df = pl.DataFrame(
            {
                "y": [1.0, 2.0, 3.0, 4.0, 5.0],
                "brand": ["A", "B", "B", "A", "B"],
                "region": ["N", "S", "N", "S", "S"],
                "x": [1.0, 1.5, 2.0, 2.5, 3.0],
                "plain": [2.0, 1.0, 3.0, 4.0, 2.5],
            }
        )
        spline = SplineTerm("x", "bs", df=4)
        interaction = InteractionTerm(
            ["brand", "region", "x", "plain"],
            [True, True, False, False],
            force_linear={"plain"},
        )
        parsed = ParsedFormula(
            response="y",
            main_effects=[],
            interactions=[interaction],
            categorical_vars={"brand", "region"},
            spline_terms=[spline],
            has_intercept=False,
        )
        builder = InteractionBuilder(df)
        builder.build_design_matrix_from_parsed(parsed)

        const_new = df.with_columns(pl.lit(2.0).alias("x"))
        dense = builder._build_mixed_interaction_new(const_new, interaction, const_new.height)
        params = np.linspace(-0.2, 0.4, dense.shape[1])
        eta = np.zeros(const_new.height, dtype=np.float64)
        cache = builder._prepare_prediction_categorical_index_cache(const_new, parsed)
        col = builder._accumulate_mixed_interaction_prediction_new(
            eta,
            const_new,
            interaction,
            const_new.height,
            params,
            0,
            cache,
            {},
        )

        assert dense.shape[1] > 0
        assert col == dense.shape[1]
        np.testing.assert_allclose(eta, dense @ params, rtol=1e-12, atol=1e-12)

    def test_new_data_zero_width_interaction_paths(self):
        df = pl.DataFrame(
            {
                "y": [1.0, 2.0, 3.0],
                "x": [1.0, 2.0, 3.0],
                "brand": ["A", "B", "A"],
                "region": ["N", "S", "N"],
                "single": ["only", "only", "only"],
            }
        )
        parsed = ParsedFormula(
            response="y",
            main_effects=["brand", "region", "single"],
            interactions=[],
            categorical_vars={"brand", "region", "single"},
            has_intercept=True,
        )
        builder = InteractionBuilder(df)
        builder.build_design_matrix_from_parsed(parsed)

        assert builder._build_categorical_interaction_new(
            df,
            InteractionTerm(["single", "brand"], [True, True]),
            df.height,
        ).shape == (df.height, 0)
        assert builder._build_categorical_interaction_new(
            df,
            InteractionTerm(["single", "brand", "region"], [True, True, True]),
            df.height,
        ).shape == (df.height, 0)
        assert builder._build_mixed_interaction_new(
            df,
            InteractionTerm(["single", "x"], [True, False]),
            df.height,
        ).shape == (df.height, 0)
        assert builder._build_mixed_interaction_new(
            df,
            InteractionTerm(["single", "brand", "x"], [True, True, False]),
            df.height,
        ).shape == (df.height, 0)
        assert builder._build_mixed_interaction_new(
            df,
            InteractionTerm(["single", "brand", "region", "x"], [True, True, True, False]),
            df.height,
        ).shape == (df.height, 0)

    def test_continuous_interaction_prediction_accumulator_branches(self, monkeypatch):
        df = pl.DataFrame({"y": [0.0, 1.0, 2.0]})
        builder = InteractionBuilder(df)
        n = df.height

        def run_accumulator(
            factors: list[str],
            bases: list[np.ndarray],
            params: np.ndarray,
        ) -> tuple[np.ndarray, int]:
            mapping = dict(zip(factors, bases, strict=True))
            monkeypatch.setattr(
                builder,
                "_resolve_factor_new",
                lambda _new_data, factor, _force_linear, _cache: mapping[factor],
            )
            eta = np.zeros(n, dtype=np.float64)
            col = builder._accumulate_continuous_interaction_prediction_new(
                eta,
                df,
                InteractionTerm(factors, [False] * len(factors)),
                n,
                params,
                0,
                {},
            )
            return eta, col

        const_left = np.broadcast_to(np.array([[2.0, 3.0]]), (n, 2))
        const_right = np.broadcast_to(np.array([[5.0, 7.0]]), (n, 2))
        params = np.array([1.0, 2.0, 3.0, 4.0])
        eta, col = run_accumulator(["a", "b"], [const_left, const_right], params)
        assert col == 4
        expected_const = 1.0 * 2.0 * 5.0 + 2.0 * 2.0 * 7.0 + 3.0 * 3.0 * 5.0 + 4.0 * 3.0 * 7.0
        np.testing.assert_allclose(eta, np.full(n, expected_const))

        variable = np.column_stack([np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])])
        eta, col = run_accumulator(
            ["const", "variable"],
            [const_left, variable],
            np.array([0.0, 1.0, 2.0, 3.0]),
        )
        assert col == 4
        np.testing.assert_allclose(eta, 6.0 * variable[:, 0] + 11.0 * variable[:, 1])

        left = np.column_stack([np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])])
        right = np.column_stack([np.array([0.5, 1.0, 1.5]), np.array([1.5, 2.0, 2.5])])
        coef = np.array([[1.0, 2.0], [3.0, 4.0]])
        eta, col = run_accumulator(["left", "right"], [left, right], coef.ravel())
        assert col == 4
        np.testing.assert_allclose(eta, np.einsum("ij,ik,jk->i", left, right, coef))

        third = np.column_stack([np.ones(n), np.array([2.0, 3.0, 4.0])])
        params = np.arange(1.0, 9.0)
        eta, col = run_accumulator(["left", "right", "third"], [left, right, third], params)
        assert col == 8
        expected = np.zeros(n)
        p = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    expected += left[:, i] * right[:, j] * third[:, k] * params[p]
                    p += 1
        np.testing.assert_allclose(eta, expected)


# =============================================================================
# GLM Fitting with Interactions Tests
# =============================================================================


class TestGLMInteractions:
    """Test GLM fitting with interaction terms."""

    @pytest.fixture
    def insurance_data(self):
        """Create simulated insurance data."""
        np.random.seed(42)
        n = 1000

        age = np.random.uniform(20, 70, n)
        power = np.random.uniform(50, 200, n)
        area = np.random.choice(["Urban", "Suburban", "Rural"], n)

        # Create claims with some interaction effects
        log_rate = -3.0 + 0.02 * age + 0.01 * power - 0.0001 * age * power
        log_rate += np.where(area == "Urban", 0.3, np.where(area == "Suburban", 0.1, 0.0))

        claims = np.random.poisson(np.exp(log_rate))
        exposure = np.random.uniform(0.5, 1.0, n)

        return pl.DataFrame(
            {
                "claims": claims,
                "age": age,
                "power": power,
                "area": area,
                "exposure": exposure,
            }
        )

    def test_fit_continuous_interaction(self, insurance_data):
        """Fit GLM with continuous × continuous interaction."""
        result = rs.glm_dict(
            response="claims",
            terms={"age": {"type": "linear"}, "power": {"type": "linear"}},
            interactions=[{"age": {"type": "linear"}, "power": {"type": "linear"}}],
            data=insurance_data,
            family="poisson",
            offset="exposure",
        ).fit()

        assert len(result.params) == 4  # Intercept, age, power, age:power
        assert result.converged

    def test_fit_categorical_continuous_interaction(self, insurance_data):
        """Fit GLM with categorical × continuous interaction."""
        result = rs.glm_dict(
            response="claims",
            terms={"area": {"type": "categorical"}, "age": {"type": "linear"}},
            interactions=[{"area": {"type": "categorical"}, "age": {"type": "linear"}}],
            data=insurance_data,
            family="poisson",
            offset="exposure",
        ).fit()

        # Intercept + 2 area dummies + age + 2 interactions = 6
        assert len(result.params) == 6
        assert result.converged

    def test_fit_categorical_categorical_interaction(self, insurance_data):
        """Fit GLM with categorical × categorical interaction."""
        insurance_data = insurance_data.with_columns(
            pl.Series("fuel", np.random.choice(["Petrol", "Diesel"], len(insurance_data)))
        )

        result = rs.glm_dict(
            response="claims",
            terms={"area": {"type": "categorical"}, "fuel": {"type": "categorical"}},
            interactions=[{"area": {"type": "categorical"}, "fuel": {"type": "categorical"}}],
            data=insurance_data,
            family="poisson",
            offset="exposure",
        ).fit()

        # area: 2 dummies, fuel: 1 dummy
        # Total: 1 + 2 + 1 + 2×1 = 6
        assert len(result.params) == 6
        assert result.converged

    def test_regularized_interaction_model(self, insurance_data):
        """Fit regularized model with interactions."""
        result = rs.glm_dict(
            response="claims",
            terms={
                "age": {"type": "linear"},
                "power": {"type": "linear"},
                "area": {"type": "categorical"},
            },
            interactions=[{"age": {"type": "linear"}, "power": {"type": "linear"}}],
            data=insurance_data,
            family="poisson",
            offset="exposure",
        ).fit(alpha=0.1, l1_ratio=0.0)  # Ridge

        assert result.is_regularized
        assert result.converged

    def test_predictions_with_interactions(self, insurance_data):
        """Verify predictions work with interaction models."""
        result = rs.glm_dict(
            response="claims",
            terms={"age": {"type": "linear"}, "power": {"type": "linear"}},
            interactions=[{"age": {"type": "linear"}, "power": {"type": "linear"}}],
            data=insurance_data,
            family="poisson",
            offset="exposure",
        ).fit()

        # Check fitted values are reasonable
        fv = result.fittedvalues
        assert np.all(fv >= 0)  # Poisson predictions should be non-negative
        assert len(fv) == len(insurance_data)


# =============================================================================
# Performance Tests
# =============================================================================


class TestInteractionPerformance:
    """Test performance of interaction handling."""

    def test_large_categorical_interaction(self):
        """Test performance with high-cardinality categorical interaction."""
        np.random.seed(42)
        n = 50_000

        df = pl.DataFrame(
            {
                "y": np.random.poisson(1, n),
                "cat1": np.random.choice([f"A{i}" for i in range(10)], n),
                "cat2": np.random.choice([f"B{i}" for i in range(8)], n),
                "exposure": np.random.uniform(0.5, 1.0, n),
            }
        )

        import time

        t0 = time.time()
        result = rs.glm_dict(
            response="y",
            terms={"cat1": {"type": "categorical"}, "cat2": {"type": "categorical"}},
            interactions=[{"cat1": {"type": "categorical"}, "cat2": {"type": "categorical"}}],
            data=df,
            family="poisson",
            offset="exposure",
        ).fit()
        t_opt = time.time() - t0

        # Should complete in reasonable time
        assert t_opt < 30.0, f"Optimized backend took {t_opt:.1f}s (expected < 30s)"

        # Verify correct number of features
        # cat1: 9 dummies, cat2: 7 dummies
        # Total: 1 + 9 + 7 + 63 = 80
        assert len(result.params) == 80
        assert result.converged


# =============================================================================
# Edge Cases
# =============================================================================


class TestInteractionEdgeCases:
    """Test edge cases and error handling."""

    def test_single_level_categorical(self):
        """Handle categorical with single level (no variation)."""
        df = pl.DataFrame(
            {
                "y": [1, 2, 3, 4],
                "x": [1.0, 2.0, 3.0, 4.0],
                "cat": ["A", "A", "A", "A"],  # Only one level
            }
        )

        parsed = ParsedFormula(
            response="y",
            main_effects=["x", "cat"],
            interactions=[],
            categorical_vars={"cat"},
            has_intercept=True,
        )
        builder = InteractionBuilder(df)
        _y, _X, names = builder.build_design_matrix_from_parsed(parsed)

        # Only Intercept and x (no cat dummies since it's constant)
        assert "x" in names

    def test_categorical_levels_must_be_fit_before_prediction_mapping(self):
        builder = InteractionBuilder(pl.DataFrame({"y": [1.0], "cat": ["A"]}))

        with pytest.raises(rs.EncodingError, match="not seen during training"):
            builder._get_categorical_levels("cat")

    def test_categorical_encoding_matches_training_indices_with_and_without_reference(self):
        df = pl.DataFrame({"y": [0.0, 1.0, 2.0, 3.0], "cat": ["B", "A", "C", "B"]})
        builder = InteractionBuilder(df)

        full, full_names = builder._get_categorical_encoding("cat", drop_first=False)
        idx, levels, idx_names = builder._get_categorical_indices_and_names("cat", drop_first=False)
        expected_full = np.eye(len(levels), dtype=np.float64)[idx]
        np.testing.assert_allclose(full, expected_full)
        assert full_names == idx_names == [f"cat[T.{level}]" for level in levels]

        dropped, dropped_names = builder._get_categorical_encoding("cat", drop_first=True)
        np.testing.assert_allclose(dropped, expected_full[:, 1:])
        assert dropped_names == [f"cat[T.{level}]" for level in levels[1:]]

    def test_clear_caches_preserves_training_levels_for_new_data_transform(self):
        train = pl.DataFrame(
            {
                "y": [0.0, 1.0, 2.0, 3.0],
                "x": [10.0, 20.0, 30.0, 40.0],
                "cat": ["A", "B", "C", "B"],
            }
        )
        parsed = ParsedFormula(
            response="y",
            main_effects=["cat", "x"],
            interactions=[],
            categorical_vars={"cat"},
            has_intercept=True,
        )
        builder = InteractionBuilder(train)
        _y, x_train, _names = builder.build_design_matrix_from_parsed(parsed)
        builder.clear_caches()

        cached = builder._cat_encoding_cache["cat_True"]
        assert cached.encoding is None
        assert cached.indices is None
        assert cached.levels

        transformed = builder.transform_new_data(train)
        np.testing.assert_allclose(transformed, x_train)

    def test_unseen_and_null_categorical_values_map_to_reference_level(self):
        train = pl.DataFrame({"y": [1.0, 2.0, 3.0], "cat": ["A", "B", "C"]})
        new = pl.DataFrame({"cat": ["B", "unseen", None]})
        builder = InteractionBuilder(train)
        builder._get_categorical_encoding("cat")

        idx, _levels = builder._map_to_training_indices(new, "cat")

        assert idx[0] > 0
        assert idx[1] == 0
        assert idx[2] == 0


# =============================================================================
# InteractionTerm Property Tests
# =============================================================================


class TestInteractionTermProperties:
    """Test InteractionTerm dataclass properties."""

    def test_order(self):
        """Test order property returns number of factors."""
        term = InteractionTerm(factors=["x1", "x2"], categorical_flags=[False, False])
        assert term.order == 2

        term3 = InteractionTerm(factors=["x1", "x2", "x3"], categorical_flags=[False, False, False])
        assert term3.order == 3

    def test_is_pure_continuous(self):
        """Test pure continuous detection."""
        term = InteractionTerm(factors=["x1", "x2"], categorical_flags=[False, False])
        assert term.is_pure_continuous
        assert not term.is_pure_categorical
        assert not term.is_mixed

    def test_is_pure_categorical(self):
        """Test pure categorical detection."""
        term = InteractionTerm(factors=["cat1", "cat2"], categorical_flags=[True, True])
        assert term.is_pure_categorical
        assert not term.is_pure_continuous
        assert not term.is_mixed

    def test_is_mixed(self):
        """Test mixed detection."""
        term = InteractionTerm(factors=["cat1", "x1"], categorical_flags=[True, False])
        assert term.is_mixed
        assert not term.is_pure_continuous
        assert not term.is_pure_categorical


# =============================================================================
# InteractionBuilder Additional Tests
# =============================================================================


class TestInteractionBuilderAdvanced:
    """Additional tests for InteractionBuilder."""

    @pytest.fixture
    def spline_data(self):
        """Create data for spline tests."""
        np.random.seed(42)
        n = 100
        return pl.DataFrame(
            {
                "y": np.random.poisson(1, n),
                "age": np.random.uniform(20, 70, n),
                "income": np.random.uniform(30000, 150000, n),
                "region": np.random.choice(["North", "South", "East", "West"], n),
            }
        )

    def test_build_with_bspline(self, spline_data):
        """Build design matrix with B-spline term."""
        spline = SplineTerm(var_name="age", spline_type="bs", df=5, degree=3)
        parsed = ParsedFormula(
            response="y",
            main_effects=[],
            interactions=[],
            categorical_vars=set(),
            spline_terms=[spline],
            has_intercept=True,
        )
        builder = InteractionBuilder(spline_data)
        _y, X, names = builder.build_design_matrix_from_parsed(parsed)

        # Intercept + spline columns
        assert X.shape[1] >= 5
        assert any("bs(age" in name for name in names)

    def test_build_with_natural_spline(self, spline_data):
        """Build design matrix with natural spline term."""
        spline = SplineTerm(var_name="age", spline_type="ns", df=4)
        parsed = ParsedFormula(
            response="y",
            main_effects=[],
            interactions=[],
            categorical_vars=set(),
            spline_terms=[spline],
            has_intercept=True,
        )
        builder = InteractionBuilder(spline_data)
        _y, X, names = builder.build_design_matrix_from_parsed(parsed)

        assert X.shape[1] >= 4
        assert any("ns(age" in name for name in names)

    def test_get_spline_info(self, spline_data):
        """Test get_spline_info returns knot information."""
        spline = SplineTerm(var_name="age", spline_type="ns", df=4)
        parsed = ParsedFormula(
            response="y",
            main_effects=[],
            interactions=[],
            categorical_vars=set(),
            spline_terms=[spline],
            has_intercept=True,
        )
        builder = InteractionBuilder(spline_data)
        _y, _X, _names = builder.build_design_matrix_from_parsed(parsed)

        info = builder.get_spline_info()
        assert "age" in info
        assert "type" in info["age"]
        assert "df" in info["age"]

    def test_prediction_spline_cache_reuses_large_basis(self, spline_data, monkeypatch):
        spline = SplineTerm(var_name="age", spline_type="bs", df=5, degree=3)
        parsed = ParsedFormula(
            response="y",
            main_effects=[],
            interactions=[],
            categorical_vars=set(),
            spline_terms=[spline],
            has_intercept=True,
        )
        builder = InteractionBuilder(spline_data)
        builder.build_design_matrix_from_parsed(parsed)
        score = pl.DataFrame(
            {
                "y": np.zeros(12),
                "age": np.linspace(20.0, 70.0, 12),
                "income": np.linspace(30000.0, 150000.0, 12),
                "region": ["North", "South", "East", "West"] * 3,
            }
        )

        monkeypatch.setattr(interactions_mod, "_PREDICT_SPLINE_CACHE_MIN_ROWS", 5)
        first = builder._spline_basis_new_cached(score, spline, cache=None)
        second = builder._spline_basis_new_cached(score, spline, cache=None)

        assert first is second
        np.testing.assert_allclose(first, second)

    def test_linear_predict_new_data_matches_dense_transform_for_constant_spline_score_data(
        self, spline_data
    ):
        age_spline = SplineTerm(var_name="age", spline_type="bs", df=5, degree=3)
        income_spline = SplineTerm(var_name="income", spline_type="bs", df=5, degree=3)
        parsed = ParsedFormula(
            response="y",
            main_effects=[],
            interactions=[
                InteractionTerm(factors=["age", "income"], categorical_flags=[False, False]),
                InteractionTerm(factors=["region", "age"], categorical_flags=[True, False]),
            ],
            categorical_vars=set(),
            spline_terms=[age_spline, income_spline],
            has_intercept=True,
        )
        builder = InteractionBuilder(spline_data)
        _y, x, _names = builder.build_design_matrix_from_parsed(parsed)
        params = np.linspace(-0.2, 0.4, x.shape[1])
        score = pl.DataFrame(
            {
                "y": np.zeros(8),
                "age": np.repeat(42.0, 8),
                "income": np.linspace(40000.0, 90000.0, 8),
                "region": ["North", "South", "East", "West", "North", "South", "East", "West"],
            }
        )

        np.testing.assert_allclose(
            builder.linear_predict_new_data(score, params),
            builder.transform_new_data(score) @ params,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_no_intercept(self, spline_data):
        """Test design matrix without intercept."""
        parsed = ParsedFormula(
            response="y",
            main_effects=["age", "income"],
            interactions=[],
            categorical_vars=set(),
            has_intercept=False,
        )
        builder = InteractionBuilder(spline_data)
        _y, _X, names = builder.build_design_matrix_from_parsed(parsed)

        assert "Intercept" not in names
        assert "age" in names
        assert "income" in names

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("age ** 2", lambda df: df["age"].to_numpy() ** 2),
            ("age + income", lambda df: df["age"].to_numpy() + df["income"].to_numpy()),
            ("income - age", lambda df: df["income"].to_numpy() - df["age"].to_numpy()),
            ("age * 2", lambda df: df["age"].to_numpy() * 2.0),
            ("income / 2", lambda df: df["income"].to_numpy() / 2.0),
            ("age", lambda df: df["age"].to_numpy()),
        ],
    )
    def test_identity_expression_columns_match_polars_oracle(
        self, spline_data, expression, expected
    ):
        builder = InteractionBuilder(spline_data)

        values, name = builder._build_identity_columns(
            IdentityTermSpec(expression=expression),
            spline_data,
        )

        assert name == f"I({expression})"
        np.testing.assert_allclose(values, expected(spline_data))

    def test_identity_expression_rejects_unsupported_syntax(self, spline_data):
        builder = InteractionBuilder(spline_data)

        with pytest.raises(rs.ValidationError, match="Failed to evaluate"):
            builder._build_identity_columns(IdentityTermSpec(expression="age // 2"), spline_data)

    def test_constraint_columns_support_raw_and_identity_terms(self, spline_data):
        builder = InteractionBuilder(spline_data)

        raw, raw_name = builder._build_constraint_columns(
            ConstraintTermSpec(var_name="age", constraint="pos"),
            spline_data,
        )
        squared, squared_name = builder._build_constraint_columns(
            ConstraintTermSpec(var_name="I(age ** 2)", constraint="neg"),
            spline_data,
        )

        assert raw_name == "pos(age)"
        assert squared_name == "neg(I(age ** 2))"
        np.testing.assert_allclose(raw, spline_data["age"].to_numpy())
        np.testing.assert_allclose(squared, spline_data["age"].to_numpy() ** 2)

    def test_constraint_columns_reject_missing_variable(self, spline_data):
        builder = InteractionBuilder(spline_data)

        with pytest.raises(rs.ValidationError, match="not found"):
            builder._build_constraint_columns(
                ConstraintTermSpec(var_name="missing", constraint="pos"),
                spline_data,
            )

    def test_specific_categorical_level_indicators_match_oracle(self, spline_data):
        builder = InteractionBuilder(spline_data)
        term = CategoricalTermSpec(var_name="region", levels=["North", "West", "Missing"])

        cols, names = builder._build_categorical_level_indicators(term)
        new_cols, new_names = builder._build_categorical_level_indicators_new(term, spline_data)

        expected = np.column_stack(
            [
                (spline_data["region"].to_numpy() == "North").astype(float),
                (spline_data["region"].to_numpy() == "West").astype(float),
                np.zeros(len(spline_data), dtype=float),
            ]
        )
        assert names == new_names == ["region[North]", "region[West]", "region[Missing]"]
        np.testing.assert_allclose(cols, expected)
        np.testing.assert_allclose(new_cols, expected)

    def test_term_slots_cover_design_columns_without_gaps(self, spline_data):
        parsed = ParsedFormula(
            response="y",
            main_effects=["age", "region"],
            interactions=[
                InteractionTerm(factors=["region", "income"], categorical_flags=[True, False])
            ],
            categorical_vars={"region"},
            frequency_encoding_terms=[FrequencyEncodingTermSpec(var_name="region")],
            identity_terms=[IdentityTermSpec(expression="age ** 2")],
            categorical_terms=[CategoricalTermSpec(var_name="region", levels=["North"])],
            has_intercept=True,
        )
        builder = InteractionBuilder(spline_data)
        _y, x, names = builder.build_design_matrix_from_parsed(parsed)

        covered_names = []
        next_col = 0
        for slot in builder._term_slots:
            assert slot.col_start == next_col
            assert slot.col_end >= slot.col_start
            covered_names.extend(slot.design_column_names)
            next_col = slot.col_end

        assert next_col == x.shape[1] == len(names)
        assert covered_names == names

    def test_linear_predict_new_data_rejects_coefficient_length_mismatch(self, spline_data):
        parsed = ParsedFormula(
            response="y",
            main_effects=["age", "income"],
            interactions=[],
            categorical_vars=set(),
            has_intercept=True,
        )
        builder = InteractionBuilder(spline_data)
        _y, x, _names = builder.build_design_matrix_from_parsed(parsed)

        with pytest.raises(rs.PredictionError, match="coefficients"):
            builder.linear_predict_new_data(spline_data, np.ones(x.shape[1] - 1))

    def test_validate_design_matrix_reports_cached_invalid_structure(self, spline_data, capsys):
        builder = InteractionBuilder(spline_data)

        with pytest.raises(rs.DesignMatrixError, match="No design matrix"):
            builder.validate_design_matrix(verbose=False)

        x = np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 2.0],
                [1.0, 1.0, 2.0, 4.0],
                [1.0, 1.0, 3.0, 6.0],
            ]
        )
        names = ["Intercept", "constant", "linear", "linear_x2"]
        results = builder.validate_design_matrix(x, names, corr_threshold=0.95, verbose=True)

        assert not results["valid"]
        assert results["zero_variance_columns"] == ["constant"]
        assert results["rank"] < results["expected_rank"]
        assert ("linear", "linear_x2", pytest.approx(1.0)) in results["problematic_columns"]
        assert "INVALID" in capsys.readouterr().out

    def test_two_categorical_lookup_indices_encode_reference_as_zero(self):
        idx1 = np.array([0, 1, 2, 3, 2])
        idx2 = np.array([1, 0, 2, 2, 3])

        flat = InteractionBuilder._two_cat_lookup_indices(idx1, 2, idx2, 2)

        np.testing.assert_array_equal(flat, np.array([0, 0, 4, 0, 0]))


# =============================================================================
# Combined Workflow Tests
# =============================================================================


class TestIntegrationWorkflows:
    """Test complete workflows combining multiple features."""

    def test_spline_with_categorical(self):
        """Fit model with spline and categorical terms."""
        np.random.seed(42)
        n = 200
        df = pl.DataFrame(
            {
                "y": np.random.poisson(1, n),
                "age": np.random.uniform(20, 70, n),
                "region": np.random.choice(["A", "B"], n),
                "exposure": np.random.uniform(0.5, 1.5, n),
            }
        )

        result = rs.glm_dict(
            response="y",
            terms={
                "age": {"type": "ns", "df": 3},
                "region": {"type": "categorical"},
            },
            data=df,
            family="poisson",
            offset="exposure",
        ).fit()

        assert result.converged
        # 1 intercept + 2 spline + 1 region dummy = 4
        assert len(result.params) >= 4

    def test_multiple_interactions(self):
        """Fit model with multiple different interaction types."""
        np.random.seed(42)
        n = 200
        df = pl.DataFrame(
            {
                "y": np.random.poisson(1, n),
                "x1": np.random.uniform(0, 10, n),
                "x2": np.random.uniform(0, 10, n),
                "cat": np.random.choice(["A", "B", "C"], n),
                "exposure": np.ones(n),
            }
        )

        result = rs.glm_dict(
            response="y",
            terms={
                "x1": {"type": "linear"},
                "x2": {"type": "linear"},
                "cat": {"type": "categorical"},
            },
            interactions=[
                {"x1": {"type": "linear"}, "x2": {"type": "linear"}, "include_main": False},
                {"cat": {"type": "categorical"}, "x1": {"type": "linear"}, "include_main": False},
            ],
            data=df,
            family="poisson",
            offset="exposure",
        ).fit()

        assert result.converged


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
