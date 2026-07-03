"""Contract tests for resolved rate-table export helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import rustystats as rs
from rustystats import rate_tables as rate_tables_mod
from rustystats.interactions import TermSlot
from rustystats.rate_tables import to_rate_tables


def _fake_model(*, slots: list[TermSlot], params: list[float], link: str = "log", builder=None):
    if builder is None:
        builder = SimpleNamespace(_term_slots=slots)
    else:
        builder._term_slots = slots
    return SimpleNamespace(
        family="poisson",
        link=link,
        params=np.asarray(params, dtype=np.float64),
        feature_names=["Intercept", *[name for slot in slots for name in slot.design_column_names]],
        _builder=builder,
        _input_transforms=[],
    )


def _lookup_transform(
    *,
    output: str,
    output_dtype: str,
    values,
    default=None,
    on_unseen: str = "default",
    on_null: str = "default",
):
    return {
        "name": f"{output}_lookup",
        "type": "lookup",
        "sources": ["raw"],
        "output": output,
        "output_dtype": output_dtype,
        "keys": [["B"], ["A"], [None]],
        "values": values,
        "default": default,
        "on_unseen": on_unseen,
        "on_null": on_null,
    }


class FakeSpline:
    def transform(self, values):
        values = np.asarray(values, dtype=np.float64)
        basis = np.column_stack([np.ones_like(values), values])
        return basis, ["bs(age, 1/2)", "bs(age, 2/2)"]

    def get_knot_info(self):
        return {"boundary_knots": [0.0, 10.0], "knots": [5.0]}


class TestRateTableValidation:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"style": "raw"}, "style"),
            ({"format": "json"}, "format"),
            ({"spline_strategy": "bad"}, "spline_strategy"),
            ({"spline_interpolation": "nearest"}, "spline_interpolation"),
            ({"spline_extrapolation": "extend"}, "spline_extrapolation"),
        ],
    )
    def test_to_rate_tables_rejects_unsupported_options(self, kwargs, message):
        model = _fake_model(
            slots=[TermSlot("x", "linear", ["x"], 1, 2, ["x"])],
            params=[0.1, 0.2],
        )

        with pytest.raises(rs.ValidationError, match=message):
            to_rate_tables(model, **kwargs)

    def test_stacked_csv_requires_path_and_writes_blocks(self, tmp_path):
        slot = TermSlot(
            "grp",
            "categorical",
            ["grp"],
            1,
            3,
            ["grp[T.B]", "grp[T.C]"],
            extra={"levels": ["A", "B", "C"]},
        )
        model = _fake_model(slots=[slot], params=[0.0, 0.2, -0.3])

        with pytest.raises(rs.ValidationError, match="path is required"):
            to_rate_tables(model, format="stacked_csv")

        path = tmp_path / "rates.csv"
        artifact = to_rate_tables(model, path=path, format="stacked_csv")
        text = path.read_text()
        assert "_base" in text
        assert "_manifest" in text
        assert "grp" in text
        assert artifact["manifest"][0]["row_count"] == 4

    def test_deployment_false_collects_unrepresentable_term_warning(self):
        model = _fake_model(
            slots=[TermSlot("x:y", "interaction", ["x", "y"], 1, 2, ["x:y"])],
            params=[0.0, 1.0],
        )

        artifact = to_rate_tables(model, deployment=False)

        assert artifact["tables"] == []
        assert artifact["warnings"][0]["term"] == "x:y"

    def test_stacked_csv_includes_deployment_warnings(self, tmp_path):
        model = _fake_model(
            slots=[TermSlot("x:y", "interaction", ["x", "y"], 1, 2, ["x:y"])],
            params=[0.0, 1.0],
        )
        path = tmp_path / "warnings.csv"

        to_rate_tables(model, path=path, format="stacked_csv", deployment=False)

        text = path.read_text()
        assert "_warnings" in text
        assert "x:y" in text

    def test_linear_term_without_lookup_transform_fails_closed(self):
        model = _fake_model(
            slots=[TermSlot("raw_x", "linear", ["raw_x"], 1, 2, ["raw_x"])],
            params=[0.0, 1.0],
        )

        with pytest.raises(rs.ValidationError, match="requires a finite lookup transform"):
            to_rate_tables(model)

    def test_nonimplemented_spline_basis_strategy_fails_closed(self):
        slot = TermSlot("age", "bs", ["age"], 1, 3, ["b1", "b2"])
        model = _fake_model(slots=[slot], params=[0.0, 1.0, 1.0])

        with pytest.raises(rs.ValidationError, match="not implemented"):
            to_rate_tables(model, spline_strategy="basis")


class TestLookupTables:
    def test_lookup_linear_table_includes_components_and_default_row(self):
        slot = TermSlot(
            "territory_factor", "linear", ["territory_factor"], 1, 2, ["territory_factor"]
        )
        transform = _lookup_transform(
            output="territory_factor",
            output_dtype="float64",
            values=[2.0, 1.0, 3.0],
            default=2.0,
        )
        model = _fake_model(slots=[slot], params=[0.1, 0.5])
        model._input_transforms = [transform]

        artifact = to_rate_tables(model, include_components=True)
        table = artifact["tables"][0]

        assert table["kind"] == "lookup_linear"
        assert table["columns"] == [
            "raw",
            "territory_factor",
            "eta",
            "rel",
            "coefficient",
            "transform",
        ]
        assert table["rows"][0][0] == "<null>"
        assert table["default"] == [
            "<default>",
            2.0,
            1.0,
            pytest.approx(np.exp(1.0)),
            0.5,
            transform["name"],
        ]

    def test_lookup_categorical_table_resolves_levels_and_default(self):
        slot = TermSlot(
            "group",
            "categorical",
            ["group"],
            1,
            3,
            ["group[T.B]", "group[T.C]"],
            extra={"levels": ["A", "B", "C"]},
        )
        transform = _lookup_transform(
            output="group",
            output_dtype="string",
            values=["B", "C", "A"],
            default="C",
            on_unseen="default",
            on_null="reference",
        )
        model = _fake_model(slots=[slot], params=[0.0, 0.7, -0.2])
        model._input_transforms = [transform]

        table = to_rate_tables(model, include_components=True)["tables"][0]

        assert table["kind"] == "lookup_categorical"
        assert table["rows"][0][:4] == ["<null>", "A", 0.0, 1.0]
        assert table["default"][:4] == ["<unseen>", "C", -0.2, pytest.approx(np.exp(-0.2))]

    def test_lookup_null_only_default_uses_null_marker(self):
        slot = TermSlot(
            "territory_factor", "linear", ["territory_factor"], 1, 2, ["territory_factor"]
        )
        transform = _lookup_transform(
            output="territory_factor",
            output_dtype="float64",
            values=[2.0, 1.0, 3.0],
            default=2.0,
            on_unseen="reference",
            on_null="default",
        )
        model = _fake_model(slots=[slot], params=[0.0, 0.5])
        model._input_transforms = [transform]

        table = to_rate_tables(model)["tables"][0]

        assert table["default"][:4] == ["<null>", 2.0, 1.0, pytest.approx(np.exp(1.0))]

    def test_lookup_without_default_policy_has_no_default_row(self):
        slot = TermSlot(
            "territory_factor", "linear", ["territory_factor"], 1, 2, ["territory_factor"]
        )
        transform = _lookup_transform(
            output="territory_factor",
            output_dtype="float64",
            values=[2.0, 1.0, 3.0],
            default=2.0,
            on_unseen="reference",
            on_null="reference",
        )
        model = _fake_model(slots=[slot], params=[0.0, 0.5])
        model._input_transforms = [transform]

        table = to_rate_tables(model)["tables"][0]

        assert table["default"] is None

    @pytest.mark.parametrize(
        "transform",
        [
            _lookup_transform(output="x", output_dtype="float64", values=[1.0], default=1.0)
            | {"type": "bucket"},
            _lookup_transform(output="x", output_dtype="string", values=["A"], default="A"),
        ],
    )
    def test_linear_lookup_rejects_unsupported_or_nonnumeric_transform(self, transform):
        slot = TermSlot("x", "linear", ["x"], 1, 2, ["x"])
        model = _fake_model(slots=[slot], params=[0.0, 1.0])
        model._input_transforms = [transform]

        with pytest.raises(rs.ValidationError):
            to_rate_tables(model)

    @pytest.mark.parametrize(
        "transform",
        [
            _lookup_transform(output="cat", output_dtype="string", values=["A"], default="A")
            | {"type": "bucket"},
            _lookup_transform(output="cat", output_dtype="float64", values=[1.0], default=1.0),
        ],
    )
    def test_categorical_lookup_rejects_unsupported_or_nonstring_transform(self, transform):
        slot = TermSlot(
            "cat", "categorical", ["cat"], 1, 2, ["cat[T.B]"], extra={"levels": ["A", "B"]}
        )
        model = _fake_model(slots=[slot], params=[0.0, 1.0])
        model._input_transforms = [transform]

        with pytest.raises(rs.ValidationError):
            to_rate_tables(model)


class TestEncodedAndSplineTables:
    def test_plain_categorical_table_includes_component_columns(self):
        slot = TermSlot(
            "grp",
            "categorical",
            ["grp"],
            1,
            3,
            ["grp[T.B]", "grp[T.C]"],
            extra={"levels": ["A", "B", "C"]},
        )
        model = _fake_model(slots=[slot], params=[0.0, 0.2, -0.1])

        table = to_rate_tables(model, include_components=True)["tables"][0]

        assert table["columns"] == ["grp", "eta", "rel", "design_column", "coefficient"]
        assert table["rows"] == [
            ["A", 0.0, 1.0, None, 0.0],
            ["B", 0.2, pytest.approx(np.exp(0.2)), "grp[T.B]", 0.2],
            ["C", -0.1, pytest.approx(np.exp(-0.1)), "grp[T.C]", -0.1],
        ]

    def test_target_and_frequency_encoding_tables_include_defaults_and_components(self):
        te_slot = TermSlot(
            "TE(brand)",
            "target_encoding",
            ["brand"],
            1,
            2,
            ["TE(brand)"],
            extra={"var_name": "brand"},
        )
        fe_slot = TermSlot(
            "FE(region)",
            "frequency_encoding",
            ["region"],
            2,
            3,
            ["FE(region)"],
            extra={"var_name": "region"},
        )
        builder = SimpleNamespace(
            _te_stats={
                "brand": {
                    "stats": {"B": (2.0, 4), "A": (1.0, 2)},
                    "prior": 0.25,
                    "prior_weight": 2.0,
                    "interaction_vars": None,
                }
            },
            _fe_stats={
                "region": {
                    "level_counts": {"North": 2, "South": 4},
                    "max_count": 4,
                    "interaction_vars": None,
                }
            },
        )
        model = _fake_model(slots=[te_slot, fe_slot], params=[0.0, 0.5, 2.0], builder=builder)

        tables = to_rate_tables(model, include_components=True)["tables"]

        assert tables[0]["kind"] == "target_encoding"
        assert tables[0]["default"][0] == "<unseen>"
        assert tables[0]["default"][1] == 0.25
        assert tables[1]["kind"] == "frequency_encoding"
        assert tables[1]["rows"] == [
            ["North", 0.5, 1.0, pytest.approx(np.exp(1.0)), 2.0, 2, 4.0],
            ["South", 1.0, 2.0, pytest.approx(np.exp(2.0)), 2.0, 4, 4.0],
        ]

    @pytest.mark.parametrize("term_type", ["target_encoding", "frequency_encoding"])
    def test_encoded_interaction_rate_tables_fail_closed(self, term_type):
        slot = TermSlot(
            "encoded",
            term_type,
            ["a", "b"],
            1,
            2,
            ["encoded"],
            extra={"var_name": "encoded", "interaction_vars": ["a", "b"]},
        )
        builder = SimpleNamespace(
            _te_stats={
                "encoded": {
                    "stats": {},
                    "prior": 0.0,
                    "prior_weight": 1.0,
                    "interaction_vars": ["a", "b"],
                }
            },
            _fe_stats={
                "encoded": {"level_counts": {}, "max_count": 0, "interaction_vars": ["a", "b"]}
            },
        )
        model = _fake_model(slots=[slot], params=[0.0, 1.0], builder=builder)

        with pytest.raises(rs.ValidationError, match="interaction term"):
            to_rate_tables(model)

    @pytest.mark.parametrize("term_type", ["target_encoding", "frequency_encoding"])
    def test_encoded_tables_fail_when_training_stats_missing(self, term_type):
        slot = TermSlot(
            "encoded",
            term_type,
            ["encoded"],
            1,
            2,
            ["encoded"],
            extra={"var_name": "encoded"},
        )
        model = _fake_model(slots=[slot], params=[0.0, 1.0], builder=SimpleNamespace())

        with pytest.raises(rs.ValidationError, match="stats missing"):
            to_rate_tables(model)

    def test_spline_grid_table_uses_fitted_basis_and_metadata(self):
        slot = TermSlot(
            "age",
            "bs",
            ["age"],
            1,
            3,
            ["bs(age, 1/2)", "bs(age, 2/2)"],
        )
        builder = SimpleNamespace(_fitted_splines={"age": FakeSpline()})
        model = _fake_model(slots=[slot], params=[0.0, 0.5, 0.25], builder=builder)

        artifact = to_rate_tables(
            model,
            spline_strategy="grid",
            spline_grids={"age": [0.0, 5.0, 10.0]},
            include_components=True,
        )
        table = artifact["tables"][0]

        assert table["kind"] == "spline_grid"
        assert table["metadata"]["grid_size"] == 3
        assert table["metadata"]["basis_columns"] == ["bs(age, 1/2)", "bs(age, 2/2)"]
        assert table["rows"][1][:3] == [5.0, 1.75, pytest.approx(np.exp(1.75))]

    @pytest.mark.parametrize(
        ("grid", "message"),
        [
            ([1.0], "at least two"),
            ([0.0, "bad"], "numeric"),
            ([0.0, np.inf], "finite"),
            ([1.0, 1.0], "strictly increasing"),
        ],
    )
    def test_spline_grid_validation_errors(self, grid, message):
        slot = TermSlot("age", "bs", ["age"], 1, 3, ["b1", "b2"])
        builder = SimpleNamespace(_fitted_splines={"age": FakeSpline()})
        model = _fake_model(slots=[slot], params=[0.0, 1.0, 1.0], builder=builder)

        with pytest.raises(rs.ValidationError, match=message):
            to_rate_tables(model, spline_strategy="grid", spline_grids={"age": grid})

    def test_spline_grid_requires_fitted_metadata_and_matching_width(self):
        slot = TermSlot("age", "bs", ["age"], 1, 3, ["b1", "b2"])

        missing_model = _fake_model(
            slots=[slot],
            params=[0.0, 1.0, 1.0],
            builder=SimpleNamespace(_fitted_splines={}),
        )
        with pytest.raises(rs.ValidationError, match="metadata missing"):
            to_rate_tables(
                missing_model,
                spline_strategy="grid",
                spline_grids={"age": [0.0, 1.0]},
            )

        bad_width_slot = TermSlot("age", "bs", ["age"], 1, 4, ["b1", "b2", "b3"])
        bad_width_model = _fake_model(
            slots=[bad_width_slot],
            params=[0.0, 1.0, 1.0, 1.0],
            builder=SimpleNamespace(_fitted_splines={"age": FakeSpline()}),
        )
        with pytest.raises(rs.ValidationError, match="basis width"):
            to_rate_tables(
                bad_width_model,
                spline_strategy="grid",
                spline_grids={"age": [0.0, 1.0]},
            )

    def test_duplicate_safe_table_names_are_made_unique(self):
        slots = [
            TermSlot("a-b", "categorical", ["a"], 1, 2, ["a[T.B]"], extra={"levels": ["A", "B"]}),
            TermSlot("a_b", "categorical", ["b"], 2, 3, ["b[T.B]"], extra={"levels": ["A", "B"]}),
        ]
        model = _fake_model(slots=slots, params=[0.0, 0.1, 0.2])

        names = [table["name"] for table in to_rate_tables(model)["tables"]]

        assert names == ["a_b", "a_b_2"]

    def test_identity_link_omits_relativities_and_no_intercept_defaults_to_zero(self):
        slot = TermSlot(
            "grp", "categorical", ["grp"], 0, 1, ["grp[T.B]"], extra={"levels": ["A", "B"]}
        )
        model = _fake_model(slots=[slot], params=[0.4], link="identity")
        model.feature_names = ["grp[T.B]"]

        artifact = to_rate_tables(model)

        assert artifact["base"] == {"eta": 0.0, "rel": None}
        assert artifact["tables"][0]["rows"][1] == ["B", 0.4, None]


def test_split_encoded_key_width_contracts():
    assert rate_tables_mod._split_encoded_key("A:B", 2) == ["A", "B"]
    assert rate_tables_mod._split_encoded_key("A", 2) == ["A", ""]
