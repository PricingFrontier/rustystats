"""Tests for ``GLMModel.predict_contributions``.

Covers per-term decomposition for every supported term type, all three link
families used in the codebase (identity / log / logit), grouping behaviour,
offset and complement-of-credibility plumbing, regularised-model handling,
serialization round-trips, and additivity validation.
"""

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import rustystats as rs
from rustystats.interactions import TermSlot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assert_additivity(rows: list[dict], predict_values: np.ndarray, tol: float = 1e-9) -> None:
    """Verify base_value + sum(contributions) = eta and inverse_link(eta) = predict."""
    for i, row in enumerate(rows):
        sum_c = sum(c["contribution"] for c in row["contributions"])
        eta = row["base_value"] + sum_c
        assert abs(eta - row["prediction_from_contributions"]) < tol, (
            f"row {i}: base+sum={eta}, prediction_from_contributions={row['prediction_from_contributions']}"
        )
        assert abs(row["prediction_value"] - predict_values[i]) < 1e-9, (
            f"row {i}: predict_value={row['prediction_value']}, predict()={predict_values[i]}"
        )


class TestContributionHelpers:
    def test_compute_contributions_rejects_models_without_formula_metadata(self):
        from rustystats.contributions import compute_contributions

        model = SimpleNamespace(_builder=None)

        with pytest.raises(rs.PredictionError, match="formula API"):
            compute_contributions(model, pl.DataFrame({"x": [1.0]}))

    def test_validate_term_slots_accepts_exact_contiguous_coverage(self):
        from rustystats.contributions import _validate_term_slots

        _validate_term_slots(
            [
                TermSlot("Intercept", "intercept", [], 0, 1, ["Intercept"]),
                TermSlot("offset", "offset", [], -1, -1, []),
                TermSlot("x", "linear", ["x"], 1, 2, ["x"]),
                TermSlot("cat", "categorical", ["cat"], 2, 4, ["cat[T.B]", "cat[T.C]"]),
            ],
            n_params=4,
        )
        _validate_term_slots([], n_params=0)

    def test_validate_term_slots_rejects_missing_or_inconsistent_coverage(self):
        from rustystats.contributions import _validate_term_slots

        with pytest.raises(rs.PredictionError, match="lacks term-slot metadata"):
            _validate_term_slots([], n_params=1)

        with pytest.raises(rs.PredictionError, match="inconsistent"):
            _validate_term_slots(
                [TermSlot("bad", "linear", ["x"], 2, 1, ["x"])],
                n_params=2,
            )

        with pytest.raises(rs.PredictionError, match="does not cover"):
            _validate_term_slots(
                [TermSlot("Intercept", "intercept", [], 0, 1, ["Intercept"])],
                n_params=2,
            )

    def test_extract_feature_values_covers_interaction_and_synthetic_terms(self):
        from rustystats.contributions import _extract_feature_values

        data = pl.DataFrame(
            {
                "x": [1.5, 2.5],
                "cat": ["A", "B"],
                "brand": ["a", "b"],
                "region": ["r1", "r2"],
            }
        )
        x_new = np.array([[1.0, 1.5, 0.2], [1.0, 2.5, 0.8]])

        interaction = TermSlot(
            "TE(brand:region):x",
            "interaction",
            ["TE(brand:region)", "x"],
            1,
            2,
            ["TE(brand:region):x"],
            extra={"categorical_flags": [False, False]},
        )
        values = _extract_feature_values(interaction, data, x_new)
        assert values == [
            {"brand:region": "a:r1", "x": 1.5},
            {"brand:region": "b:r2", "x": 2.5},
        ]

        exposure = TermSlot(
            "exposure",
            "exposure",
            ["expo"],
            -1,
            -1,
            [],
            extra={"raw": np.array([1.0, 2.0])},
        )
        np.testing.assert_array_equal(_extract_feature_values(exposure, data, x_new), [1.0, 2.0])

        fallback = TermSlot("column", "custom_column", [], 2, 3, ["column"])
        np.testing.assert_allclose(_extract_feature_values(fallback, data, x_new), [0.2, 0.8])

        linear_fallback = TermSlot("I(x ** 2)", "linear", ["missing"], 1, 2, ["I(x ** 2)"])
        np.testing.assert_allclose(
            _extract_feature_values(linear_fallback, data, x_new),
            [1.5, 2.5],
        )

        target_encoded_interaction = TermSlot(
            "TE(brand:region)",
            "target_encoding",
            ["brand", "region"],
            2,
            3,
            ["TE(brand:region)"],
            extra={"interaction_vars": ["brand", "region"]},
        )
        assert _extract_feature_values(target_encoded_interaction, data, x_new) == [
            {"raw": "a:r1", "encoded": 0.2},
            {"raw": "b:r2", "encoded": 0.8},
        ]

        indicator = TermSlot("cat[B]", "categorical_indicator", ["cat"], 2, 3, ["cat[B]"])
        np.testing.assert_array_equal(_extract_feature_values(indicator, data, x_new), ["A", "B"])

        exposure_without_raw = TermSlot("exposure", "exposure", [], -1, -1, [], extra={})
        np.testing.assert_allclose(
            _extract_feature_values(exposure_without_raw, data, x_new), [1.0, 1.0]
        )

        negative_fallback = TermSlot("synthetic", "synthetic", [], -1, -1, [])
        np.testing.assert_allclose(
            _extract_feature_values(negative_fallback, data, x_new), [0.0, 0.0]
        )

    def test_row_value_and_builders_cover_empty_and_scalar_feature_values(self):
        from rustystats.contributions import _build_dataframe, _build_records, _row_value

        model = SimpleNamespace(family="gaussian", link="identity")
        base = np.array([1.0, 2.0])
        zeros = np.zeros((2, 0))
        empty_records = _build_records(
            model=model,
            slots=[],
            contribs_matrix=zeros,
            feature_values_per_slot=[],
            base_value=base,
            sum_contribs=np.zeros(2),
            eta_from_contribs=base,
            mu_from_contribs=base,
            output_space="response",
            prediction_space="response",
            X_new=zeros,
            params=np.array([]),
            include_design_columns=False,
        )
        assert empty_records[0]["contributions"] == []

        empty_df = _build_dataframe(
            model=model,
            slots=[],
            contribs_matrix=zeros,
            feature_values_per_slot=[],
            base_value=base,
            sum_contribs=np.zeros(2),
            eta_from_contribs=base,
            mu_from_contribs=base,
            output_space="response",
            prediction_space="response",
        )
        assert empty_df.height == 2
        assert "term" not in empty_df.columns

        assert _row_value(np.array([np.bool_(True)]), 0) is True
        assert _row_value("constant", 0) == "constant"

        slots = [
            TermSlot("list_term", "custom", [], 0, 1, ["list_term"]),
            TermSlot("scalar_term", "custom", [], 1, 2, ["scalar_term"]),
        ]
        df = _build_dataframe(
            model=model,
            slots=slots,
            contribs_matrix=np.array([[0.5, -0.25], [1.5, -0.75]]),
            feature_values_per_slot=[["a", "b"], "same"],
            base_value=base,
            sum_contribs=np.array([0.25, 0.75]),
            eta_from_contribs=np.array([1.25, 2.75]),
            mu_from_contribs=np.array([1.25, 2.75]),
            output_space="response",
            prediction_space="response",
        )
        assert df["feature_value"].to_list() == ["a", "same", "b", "same"]


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 400
    return pl.DataFrame(
        {
            "y": np.random.poisson(2.0, n).astype(float),
            "y_bin": np.random.binomial(1, 0.3, n).astype(float),
            "y_cont": np.random.randn(n) * 3 + 5,
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "age": np.random.uniform(18, 75, n),
            "income": np.random.uniform(20000, 100000, n),
            "region": np.random.choice(["North", "South", "East", "West"], n),
            "brand": np.random.choice(["A", "B", "C", "D", "E"], n),
            "exposure": np.random.uniform(0.5, 2.0, n),
            "diff_to_market": np.random.uniform(-30, 30, n),
        }
    )


class TestContributionPublicErrorContracts:
    def test_invalid_return_format_fails_loud(self, sample_data):
        result = rs.glm_dict(
            response="y_cont",
            terms={"x1": {"type": "linear"}},
            data=sample_data,
            family="gaussian",
        ).fit()

        with pytest.raises(ValueError, match="return_format"):
            result.predict_contributions(sample_data.head(2), return_format="wide")

    def test_exposure_requires_log_link(self, sample_data):
        result = rs.glm_dict(
            response="y_cont",
            terms={"x1": {"type": "linear"}},
            data=sample_data,
            family="gaussian",
        ).fit()

        with pytest.raises(rs.ValidationError, match="log-link rate models"):
            result.predict_contributions(sample_data.head(2), exposure="exposure")

    def test_ungrouped_spline_expands_design_column_rows(self, sample_data):
        result = rs.glm_dict(
            response="y_bin",
            terms={"diff_to_market": {"type": "ns", "df": 5}},
            data=sample_data,
            family="binomial",
        ).fit()

        rows = result.predict_contributions(sample_data.head(1), group_terms=False)
        terms = [c["term"] for c in rows[0]["contributions"]]

        assert "diff_to_market" not in terms
        assert sum(term.startswith("ns(diff_to_market") for term in terms) >= 2


# ---------------------------------------------------------------------------
# 1. Identity-link reconstruction
# ---------------------------------------------------------------------------


class TestIdentityLink:
    def test_gaussian_linear_reconstructs_prediction(self, sample_data):
        result = rs.glm_dict(
            response="y_cont",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=sample_data,
            family="gaussian",
        ).fit()
        new_d = sample_data.head(5)
        preds = result.predict(new_d)
        rows = result.predict_contributions(new_d)

        for i, row in enumerate(rows):
            assert row["output_space"] == "response"
            assert row["prediction_space"] == "response"
            assert row["link"] == "identity"
            assert abs(row["prediction_value"] - preds[i]) < 1e-12

    def test_identity_base_plus_sum_matches_prediction(self, sample_data):
        result = rs.glm_dict(
            response="y_cont",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=sample_data,
            family="gaussian",
        ).fit()
        rows = result.predict_contributions(sample_data.head(10))
        preds = result.predict(sample_data.head(10))
        assert_additivity(rows, preds)


# ---------------------------------------------------------------------------
# 2. Logit-link reconstruction (mirrors conversion_scoring)
# ---------------------------------------------------------------------------


class TestLogitLink:
    def test_binomial_ns_spline_reconstructs(self, sample_data):
        result = rs.glm_dict(
            response="y_bin",
            terms={"diff_to_market": {"type": "ns", "df": 5}},
            data=sample_data,
            family="binomial",
        ).fit()
        new_d = sample_data.head(5)
        preds = result.predict(new_d)
        rows = result.predict_contributions(new_d)

        for row in rows:
            assert row["output_space"] == "linear_predictor"
            assert row["prediction_space"] == "response"
            assert row["family"] == "binomial"
            assert row["link"] == "logit"

        assert_additivity(rows, preds)

    def test_spline_collapses_to_single_grouped_row(self, sample_data):
        result = rs.glm_dict(
            response="y_bin",
            terms={"diff_to_market": {"type": "ns", "df": 5}},
            data=sample_data,
            family="binomial",
        ).fit()
        rows = result.predict_contributions(sample_data.head(1))
        terms = [c["term"] for c in rows[0]["contributions"]]
        # exactly one row per source term: diff_to_market (the spline)
        assert "diff_to_market" in terms
        spline_rows = [c for c in rows[0]["contributions"] if c["term"] == "diff_to_market"]
        assert len(spline_rows) == 1
        assert spline_rows[0]["term_type"] == "ns"

    def test_include_design_columns_expands_spline(self, sample_data):
        result = rs.glm_dict(
            response="y_bin",
            terms={"diff_to_market": {"type": "ns", "df": 5}},
            data=sample_data,
            family="binomial",
        ).fit()
        rows = result.predict_contributions(sample_data.head(1), include_design_columns=True)
        spline = next(c for c in rows[0]["contributions"] if c["term"] == "diff_to_market")
        assert "design_columns" in spline
        # Natural-spline basis can collapse a column at the boundary, so just
        # verify there is more than one design column and the contributions add up.
        assert len(spline["design_columns"]) >= 2
        dc_sum = sum(d["contribution"] for d in spline["design_columns"])
        assert abs(dc_sum - spline["contribution"]) < 1e-12


# ---------------------------------------------------------------------------
# 3. Each term type isolation
# ---------------------------------------------------------------------------


class TestTermTypes:
    def test_linear_only(self, sample_data):
        result = rs.glm_dict(
            response="y_cont",
            terms={"x1": {"type": "linear"}},
            data=sample_data,
            family="gaussian",
        ).fit()
        rows = result.predict_contributions(sample_data.head(3))
        for row in rows:
            x1_row = next(c for c in row["contributions"] if c["term"] == "x1")
            assert x1_row["term_type"] == "linear"

    def test_categorical_reference_level_zero(self, sample_data):
        result = rs.glm_dict(
            response="y",
            terms={"region": {"type": "categorical"}},
            data=sample_data,
            family="poisson",
        ).fit()
        rows = result.predict_contributions(sample_data.head(20))

        # Reference level is alphabetically first ("East") given drop_first
        ref_level = sorted(sample_data["region"].unique().to_list())[0]
        for row in rows:
            cat = next(c for c in row["contributions"] if c["term"] == "region")
            if cat["feature_value"] == ref_level:
                assert abs(cat["contribution"]) < 1e-12, (
                    f"reference level {ref_level} should have contribution 0, got {cat['contribution']}"
                )

    def test_bs_spline_grouped(self, sample_data):
        result = rs.glm_dict(
            response="y",
            terms={"age": {"type": "bs", "df": 4}},
            data=sample_data,
            family="poisson",
        ).fit()
        rows = result.predict_contributions(sample_data.head(2))
        for row in rows:
            spline_rows = [c for c in row["contributions"] if c["term"] == "age"]
            assert len(spline_rows) == 1
            assert spline_rows[0]["term_type"] == "bs"

    def test_target_encoding_feature_value_shape(self, sample_data):
        result = rs.glm_dict(
            response="y",
            terms={"brand": {"type": "target_encoding"}},
            data=sample_data,
            family="poisson",
        ).fit()
        rows = result.predict_contributions(sample_data.head(3))
        for row in rows:
            te = next(c for c in row["contributions"] if c["term"] == "TE(brand)")
            assert te["term_type"] == "target_encoding"
            assert isinstance(te["feature_value"], dict)
            assert "raw" in te["feature_value"]
            assert "encoded" in te["feature_value"]

    def test_expression_feature_value_shape(self, sample_data):
        result = rs.glm_dict(
            response="y_cont",
            terms={
                "x1": {"type": "linear"},
                "x1_sq": {"type": "expression", "expr": "x1 ** 2"},
            },
            data=sample_data,
            family="gaussian",
        ).fit()
        rows = result.predict_contributions(sample_data.head(2))
        expr = next(c for c in rows[0]["contributions"] if c["term_type"] == "expression")
        assert isinstance(expr["feature_value"], dict)
        assert expr["feature_value"]["expr"] == "x1 ** 2"


# ---------------------------------------------------------------------------
# 4. Interaction grouping
# ---------------------------------------------------------------------------


class TestInteractions:
    def test_cat_cont_interaction_single_grouped_row(self, sample_data):
        result = rs.glm_dict(
            response="y",
            terms={"income": {"type": "linear"}, "region": {"type": "categorical"}},
            interactions=[
                {
                    "income": {"type": "linear"},
                    "region": {"type": "categorical"},
                    "include_main": False,
                }
            ],
            data=sample_data,
            family="poisson",
        ).fit()
        rows = result.predict_contributions(sample_data.head(3))
        for row in rows:
            inter_rows = [c for c in row["contributions"] if c["term_type"] == "interaction"]
            assert len(inter_rows) == 1
            assert inter_rows[0]["term"] == "income:region"
            fv = inter_rows[0]["feature_value"]
            assert isinstance(fv, dict)
            assert set(fv.keys()) == {"income", "region"}

    def test_cat_cat_interaction_dict_feature_value(self, sample_data):
        result = rs.glm_dict(
            response="y",
            terms={"region": {"type": "categorical"}, "brand": {"type": "categorical"}},
            interactions=[
                {
                    "region": {"type": "categorical"},
                    "brand": {"type": "categorical"},
                    "include_main": False,
                }
            ],
            data=sample_data,
            family="poisson",
        ).fit()
        rows = result.predict_contributions(sample_data.head(3))
        inter = next(c for c in rows[0]["contributions"] if c["term_type"] == "interaction")
        assert isinstance(inter["feature_value"], dict)
        assert set(inter["feature_value"].keys()) == {"region", "brand"}

    def test_main_effects_unaffected_by_interaction(self, sample_data):
        """Verify main-effect rows are independent of any interaction term."""
        result = rs.glm_dict(
            response="y_cont",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            interactions=[
                {
                    "x1": {"type": "linear"},
                    "x2": {"type": "linear"},
                    "include_main": False,
                }
            ],
            data=sample_data,
            family="gaussian",
        ).fit()
        rows = result.predict_contributions(sample_data.head(2))
        # We get x1, x2, x1:x2: three separate rows
        terms = [c["term"] for c in rows[0]["contributions"]]
        assert "x1" in terms
        assert "x2" in terms
        assert "x1:x2" in terms

    def test_target_encoded_interaction_feature_values(self, sample_data):
        result = rs.glm_dict(
            response="y",
            terms={},
            interactions=[
                {
                    "x1": {"type": "linear"},
                    "brand": {"type": "target_encoding"},
                    "include_main": False,
                }
            ],
            data=sample_data,
            family="poisson",
        ).fit()

        rows = result.predict_contributions(sample_data.head(3))
        preds = result.predict(sample_data.head(3))
        assert_additivity(rows, preds)

        interaction = next(c for c in rows[0]["contributions"] if c["term"] == "x1:TE(brand)")
        assert interaction["term_type"] == "interaction"
        assert set(interaction["feature_value"].keys()) == {"x1", "brand"}


# ---------------------------------------------------------------------------
# 5. Offset row
# ---------------------------------------------------------------------------


class TestOffset:
    def test_log_link_exposure_offset_row(self, sample_data):
        # RS-ACT-002: raw rate denominators use ``exposure=`` and appear as an
        # exposure row in the contribution ladder.
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=sample_data,
            family="poisson",
            exposure="exposure",
        ).fit()
        rows = result.predict_contributions(sample_data.head(3))
        for i, row in enumerate(rows):
            exposure_row = next(c for c in row["contributions"] if c["term_type"] == "exposure")
            raw = sample_data["exposure"].to_numpy()[i]
            # contribution is log(exposure) for log-link models
            assert abs(exposure_row["contribution"] - np.log(raw)) < 1e-12
            assert abs(exposure_row["feature_value"] - raw) < 1e-12

    def test_identity_link_array_offset(self, sample_data):
        result = rs.glm_dict(
            response="y_cont",
            terms={"x1": {"type": "linear"}},
            data=sample_data,
            family="gaussian",
        ).fit()
        new_d = sample_data.head(3)
        offset_arr = np.array([1.0, 2.0, 3.0])
        rows = result.predict_contributions(new_d, offset=offset_arr)
        preds = result.predict(new_d, offset=offset_arr)
        assert_additivity(rows, preds)
        for i, row in enumerate(rows):
            offset_row = next(c for c in row["contributions"] if c["term_type"] == "offset")
            # identity link: offset contribution equals raw offset
            assert abs(offset_row["contribution"] - offset_arr[i]) < 1e-12

    def test_explicit_exposure_plus_link_offset_rows(self, sample_data):
        adjustment = np.linspace(-0.2, 0.2, len(sample_data))
        data = sample_data.with_columns(pl.Series("adj", adjustment))
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=data,
            family="poisson",
            exposure="exposure",
            offset="adj",
        ).fit()

        new_d = data.head(3)
        rows = result.predict_contributions(new_d)
        preds = result.predict(new_d)
        assert_additivity(rows, preds)
        for i, row in enumerate(rows):
            # RS-ACT-002b: exposure and link-scale offset are now distinct
            # ladder rows — exposure carries term_type="exposure" and link
            # offset keeps term_type="offset" — so consumers identify them by
            # the visible column rather than by position.
            exposure_rows = [c for c in row["contributions"] if c["term_type"] == "exposure"]
            offset_rows = [c for c in row["contributions"] if c["term_type"] == "offset"]
            raw = sample_data["exposure"].to_numpy()[i]
            assert len(exposure_rows) == 1
            assert len(offset_rows) == 1
            assert exposure_rows[0]["term"] == "exposure"
            assert offset_rows[0]["term"] == "offset"
            assert abs(exposure_rows[0]["contribution"] - np.log(raw)) < 1e-12
            assert abs(exposure_rows[0]["feature_value"] - raw) < 1e-12
            assert abs(offset_rows[0]["contribution"] - adjustment[i]) < 1e-12
            assert abs(offset_rows[0]["feature_value"] - adjustment[i]) < 1e-12


# ---------------------------------------------------------------------------
# 6. Complement of credibility: per-row base
# ---------------------------------------------------------------------------


class TestComplement:
    def test_string_complement_per_row_base(self, sample_data):
        # Use a varying complement column so per-row base value is non-trivial
        cw = np.random.uniform(0.5, 2.5, len(sample_data))
        data = sample_data.with_columns(pl.lit(cw).alias("cw_rate"))
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "region": {"type": "categorical"}},
            data=data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        ).fit()

        new_d = data.head(5)
        preds = result.predict(new_d)
        rows = result.predict_contributions(new_d)
        assert_additivity(rows, preds)

        for i, row in enumerate(rows):
            expected_base = np.log(cw[i])
            assert abs(row["base_value"] - expected_base) < 1e-12, (
                f"row {i}: expected base {expected_base}, got {row['base_value']}"
            )

    def test_complement_intercept_in_contributions(self, sample_data):
        cw = np.random.uniform(0.8, 1.5, len(sample_data))
        data = sample_data.with_columns(pl.lit(cw).alias("cw_rate"))
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=data,
            family="poisson",
            offset="exposure",
            complement="cw_rate",
        ).fit()
        rows = result.predict_contributions(data.head(2))
        for row in rows:
            terms = [c["term"] for c in row["contributions"]]
            assert "Intercept" in terms

    def test_glmmodel_complement(self, sample_data):
        # Train a prior model
        prior = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=sample_data,
            family="poisson",
            offset="exposure",
        ).fit()

        # State model with prior as complement
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "region": {"type": "categorical"}},
            data=sample_data,
            family="poisson",
            offset="exposure",
            complement=prior,
        ).fit()

        new_d = sample_data.head(3)
        preds = result.predict(new_d)
        rows = result.predict_contributions(new_d)
        assert_additivity(rows, preds)


# ---------------------------------------------------------------------------
# 7. Regularised model with zeroed coefficients
# ---------------------------------------------------------------------------


class TestRegularization:
    def test_lasso_zeroed_terms_still_emit_rows(self, sample_data):
        result = rs.glm_dict(
            response="y",
            terms={
                "x1": {"type": "linear"},
                "x2": {"type": "linear"},
                "age": {"type": "linear"},
                "income": {"type": "linear"},
            },
            data=sample_data,
            family="poisson",
        ).fit(alpha=10.0, l1_ratio=1.0)

        rows = result.predict_contributions(sample_data.head(2))
        # Every linear term should appear regardless of coefficient magnitude
        terms = {c["term"] for c in rows[0]["contributions"]}
        assert {"x1", "x2", "age", "income"}.issubset(terms)

        # Heavy lasso likely zeroed at least one; verify they all sum up correctly
        preds = result.predict(sample_data.head(2))
        assert_additivity(rows, preds)


# ---------------------------------------------------------------------------
# 8. Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_from_bytes_preserves_contributions(self, sample_data):
        result = rs.glm_dict(
            response="y",
            terms={
                "x1": {"type": "linear"},
                "region": {"type": "categorical"},
                "age": {"type": "ns", "df": 4},
            },
            data=sample_data,
            family="poisson",
            offset="exposure",
        ).fit()
        loaded = rs.GLMModel.from_bytes(result.to_bytes())

        new_d = sample_data.head(3)
        rows_orig = result.predict_contributions(new_d)
        rows_loaded = loaded.predict_contributions(new_d)

        for r1, r2 in zip(rows_orig, rows_loaded):
            assert abs(r1["base_value"] - r2["base_value"]) < 1e-12
            assert abs(r1["prediction_value"] - r2["prediction_value"]) < 1e-12
            assert len(r1["contributions"]) == len(r2["contributions"])
            for c1, c2 in zip(r1["contributions"], r2["contributions"]):
                assert c1["term"] == c2["term"]
                assert abs(c1["contribution"] - c2["contribution"]) < 1e-12


# ---------------------------------------------------------------------------
# 9. Validation failure mode
# ---------------------------------------------------------------------------


class TestValidation:
    def test_validate_false_skips_check(self, sample_data):
        result = rs.glm_dict(
            response="y_cont",
            terms={"x1": {"type": "linear"}},
            data=sample_data,
            family="gaussian",
        ).fit()
        rows = result.predict_contributions(sample_data.head(2), validate=False)
        assert len(rows) == 2

    def test_default_tolerance_passes_clean_models(self, sample_data):
        # All standard models should pass validation under default tolerance
        result = rs.glm_dict(
            response="y",
            terms={
                "x1": {"type": "linear"},
                "region": {"type": "categorical"},
                "age": {"type": "bs", "df": 4},
            },
            data=sample_data,
            family="poisson",
            offset="exposure",
        ).fit()
        # Should not raise
        rows = result.predict_contributions(sample_data.head(50), validate=True)
        assert len(rows) == 50


# ---------------------------------------------------------------------------
# 10. return_format=dataframe
# ---------------------------------------------------------------------------


class TestDataframeFormat:
    def test_dataframe_long_format_shape(self, sample_data):
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "region": {"type": "categorical"}},
            data=sample_data,
            family="poisson",
            offset="exposure",
        ).fit()
        n = 5
        df = result.predict_contributions(sample_data.head(n), return_format="dataframe")
        assert isinstance(df, pl.DataFrame)
        # x1 (linear) + region (categorical) + offset = 3 terms per row
        expected_n_terms = 3
        assert len(df) == n * expected_n_terms
        assert df["row_index"].n_unique() == n
        assert {
            "row_index",
            "term",
            "term_type",
            "feature_value",
            "contribution",
            "rank",
            "base_value",
            "sum_contributions",
            "prediction_from_contributions",
            "prediction_value",
            "output_space",
            "prediction_space",
            "family",
            "link",
        }.issubset(set(df.columns))

    def test_dataframe_contributions_match_records(self, sample_data):
        result = rs.glm_dict(
            response="y_cont",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=sample_data,
            family="gaussian",
        ).fit()
        rows = result.predict_contributions(sample_data.head(3))
        df = result.predict_contributions(sample_data.head(3), return_format="dataframe")

        # For each (row, term) the contribution should agree
        for i, row in enumerate(rows):
            for c in row["contributions"]:
                df_row = df.filter((pl.col("row_index") == i) & (pl.col("term") == c["term"]))
                assert len(df_row) == 1
                assert abs(df_row["contribution"][0] - c["contribution"]) < 1e-12


# ---------------------------------------------------------------------------
# 11. No-intercept model
# ---------------------------------------------------------------------------


class TestNoIntercept:
    def test_no_intercept_base_value_is_zero(self, sample_data):
        result = rs.glm_dict(
            response="y_cont",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=sample_data,
            family="gaussian",
            intercept=False,
        ).fit()
        new_d = sample_data.head(5)
        preds = result.predict(new_d)
        rows = result.predict_contributions(new_d)
        for row in rows:
            assert row["base_value"] == 0.0
            assert "Intercept" not in [c["term"] for c in row["contributions"]]
        assert_additivity(rows, preds)


# ---------------------------------------------------------------------------
# 12. Lazy frame input
# ---------------------------------------------------------------------------


class TestLazyFrame:
    def test_lazy_frame_input_works(self, sample_data):
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=sample_data,
            family="poisson",
            offset="exposure",
        ).fit()
        new_lazy = sample_data.head(3).lazy()
        rows = result.predict_contributions(new_lazy)
        assert len(rows) == 3

    def test_lazy_frame_glmmodel_complement_keeps_prior_columns(self, sample_data):
        prior = rs.glm_dict(
            response="y",
            terms={"x2": {"type": "linear"}},
            data=sample_data,
            family="poisson",
            offset="exposure",
        ).fit()
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}},
            data=sample_data,
            family="poisson",
            offset="exposure",
            complement=prior,
        ).fit()

        new_lazy = sample_data.head(3).lazy()
        rows = result.predict_contributions(new_lazy)
        preds = result.predict(new_lazy)
        assert len(rows) == 3
        assert_additivity(rows, preds)
