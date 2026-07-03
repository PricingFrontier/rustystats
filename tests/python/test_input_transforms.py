import csv
import decimal

import numpy as np
import polars as pl
import pytest
import rustystats as rs
from rustystats.exceptions import PredictionError, ValidationError
from rustystats.input_transforms import (
    _any_null_mask,
    _check_unique_normalized_keys,
    _dedup_key_value,
    _sample_bad_keys,
    _unique_temp_name,
    _utf8_key_series,
    _validate_key_value,
    _validate_numeric_param,
    _validate_value,
    apply_input_transforms,
    apply_input_transforms_lazy,
    compile_input_transforms,
    input_transform_output_columns,
    input_transform_source_columns,
    validate_input_transforms,
)


def _numeric_lookup_spec():
    return [
        {
            "type": "lookup",
            "name": "brand_region_effect",
            "sources": ["brand", "region"],
            "output": "brand_region_fts",
            "output_dtype": "float64",
            "keys": [["A", "N"], ["B", "S"], ["A|B", "N"]],
            "values": [0.2, -0.1, 0.4],
            "default": 0.0,
            "on_unseen": "default",
            "on_null": "default",
        }
    ]


def _group_lookup_spec():
    return [
        {
            "type": "lookup",
            "name": "postcode_group",
            "sources": ["postcode"],
            "output": "postcode_grp",
            "output_dtype": "string",
            "keys": [["AB1"], ["AB2"], ["ZZ9"]],
            "values": ["grp_1", "grp_2", "grp_1"],
            "default": "other",
            "on_unseen": "default",
            "on_null": "default",
        }
    ]


def _training_data():
    return pl.DataFrame(
        {
            "y": [1, 2, 1, 3, 2, 1, 4, 2],
            "brand": ["A", "B", "A|B", "A", "B", "C", "A|B", "C"],
            "region": ["N", "S", "N", "S", "S", "N", "N", "S"],
            "postcode": ["AB1", "AB2", "ZZ9", "XX1", "AB1", "AB2", "ZZ9", "XX1"],
            "exposure": [1.0, 1.1, 0.8, 1.2, 1.0, 0.9, 1.4, 1.0],
        }
    )


def _parse_stacked_csv(path):
    blocks = {}
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    i = 0
    while i < len(rows):
        if not rows[i]:
            i += 1
            continue
        name = rows[i][0]
        assert len([cell for cell in rows[i] if cell]) == 1
        i += 1
        header = rows[i]
        i += 1
        body = []
        while i < len(rows) and rows[i]:
            body.append(rows[i])
            i += 1
        blocks[name] = {"columns": header, "rows": body}
    return blocks


def test_lookup_transform_maps_multi_column_keys_and_preserves_order():
    df = pl.DataFrame(
        {
            "brand": ["A", "A|B", "B", "missing"],
            "region": ["N", "N", "S", "S"],
        }
    )

    out = apply_input_transforms(df, _numeric_lookup_spec())

    assert out["brand_region_fts"].to_list() == [0.2, 0.4, -0.1, 0.0]
    assert out["brand"].to_list() == df["brand"].to_list()
    assert "brand_region_fts" not in df.columns


def test_compile_column_helpers_and_chained_transforms_are_reusable():
    specs = [
        {
            "type": "center",
            "name": "x_center",
            "sources": ["x"],
            "output": "x_ctr",
            "center": 1.5,
            "metadata": None,
        },
        {
            "type": "lookup",
            "name": "center_brand_lookup",
            "sources": ["x_ctr", "brand"],
            "output": "risk_group",
            "output_dtype": "string",
            "keys": [[-0.5, "A"], [0.5, "B"]],
            "values": ["low", "high"],
            "default": "other",
            "on_unseen": "default",
            "on_null": "default",
        },
    ]
    canonical = validate_input_transforms(specs, data_schema={"x": pl.Float64, "brand": pl.String})
    compiled = compile_input_transforms(canonical, assume_validated=True)

    assert input_transform_source_columns(canonical) == {"x", "x_ctr", "brand"}
    assert input_transform_output_columns(canonical) == {"x_ctr", "risk_group"}
    assert [transform.name for transform in compiled] == ["x_center", "center_brand_lookup"]
    assert compiled[0].sources == ["x"]
    assert compiled[0].output == "x_ctr"

    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "brand": ["A", "B", "A"]})
    out = apply_input_transforms(df.lazy(), compiled)

    assert out["x_ctr"].to_list() == [-0.5, 0.5, 1.5]
    assert out["risk_group"].to_list() == ["low", "high", "other"]
    assert apply_input_transforms(df, None).equals(df)


def test_numeric_center_and_clip_transforms_match_manual_columns():
    df = pl.DataFrame({"x": [-2.0, 0.0, 1.5, 4.0, None]})
    specs = [
        {
            "type": "center",
            "name": "x_center",
            "sources": ["x"],
            "output": "x_ctr",
            "center": 1.5,
        },
        {
            "type": "clip",
            "name": "x_clip",
            "sources": ["x"],
            "output": "x_clip",
            "lower": -1.0,
            "upper": 2.0,
        },
    ]

    out = apply_input_transforms(df, specs)

    assert out["x_ctr"].to_list() == [-3.5, -1.5, 0.0, 2.5, None]
    assert out["x_clip"].to_list() == [-1.0, 0.0, 1.5, 2.0, None]


def test_input_transform_validation_rejects_malformed_specs():
    base_lookup = {
        "type": "lookup",
        "name": "lookup",
        "sources": ["x"],
        "output": "y",
        "output_dtype": "float64",
        "keys": [[1.0]],
        "values": [2.0],
        "default": 0.0,
    }
    missing_default = {k: v for k, v in base_lookup.items() if k != "default"}
    bad_cases = [
        ({"type": "lookup"}, "must be a list"),
        ([1], "must be a dictionary"),
        ([{**base_lookup, "type": "scale"}], "unsupported type"),
        ([{**base_lookup, "name": ""}], "requires a non-empty string 'name'"),
        ([base_lookup, {**base_lookup, "output": "z"}], "duplicate input transform name"),
        ([{**base_lookup, "sources": []}], "requires non-empty string sources"),
        ([{**base_lookup, "output": ""}], "requires a non-empty string output"),
        ([base_lookup, {**base_lookup, "name": "lookup2"}], "duplicate input transform output"),
        ([{**base_lookup, "sources": ["y"]}], "collides with a source column"),
        ([{**base_lookup, "output_dtype": "int64"}], "output_dtype must be"),
        ([{**base_lookup, "metadata": "owner"}], "metadata must be a dictionary"),
        ([{**base_lookup, "keys": "bad"}], "requires list 'keys' and 'values'"),
        ([{**base_lookup, "keys": [[1.0], [2.0]]}], "requires len\\(keys\\) == len\\(values\\)"),
        ([{**base_lookup, "keys": [[1.0, 2.0]]}], "must be a list of length 1"),
        ([{**base_lookup, "on_unseen": "warn"}], "on_unseen must be"),
        ([{**base_lookup, "on_null": "warn"}], "on_null must be"),
        ([missing_default], "requires default"),
        ([{**base_lookup, "source_cast": "native"}], "only 'string' is supported"),
    ]

    for specs, message in bad_cases:
        with pytest.raises(ValidationError, match=message):
            validate_input_transforms(specs)

    with pytest.raises(ValidationError, match="missing source"):
        validate_input_transforms([base_lookup], data_schema={"z": pl.Float64})

    with pytest.raises(ValidationError, match="requires exactly one source"):
        validate_input_transforms(
            [
                {
                    "type": "center",
                    "name": "center",
                    "sources": ["x", "z"],
                    "output": "x_ctr",
                    "center": 1.0,
                }
            ]
        )
    with pytest.raises(ValidationError, match="requires output_dtype='float64'"):
        validate_input_transforms(
            [
                {
                    "type": "clip",
                    "name": "clip",
                    "sources": ["x"],
                    "output": "x_clip",
                    "output_dtype": "string",
                    "lower": 0.0,
                    "upper": 1.0,
                }
            ]
        )


def test_numeric_transform_rejects_bad_bounds():
    with pytest.raises(ValidationError, match="lower must be <= upper"):
        validate_input_transforms(
            [
                {
                    "type": "clip",
                    "name": "bad_clip",
                    "sources": ["x"],
                    "output": "x_clip",
                    "lower": 2.0,
                    "upper": 1.0,
                }
            ]
        )


def test_input_transform_runtime_error_contracts_and_replacement_paths():
    center = [{"type": "center", "name": "center", "sources": ["x"], "output": "y", "center": 1.0}]
    with pytest.raises(PredictionError, match="already exists"):
        apply_input_transforms(pl.DataFrame({"x": [1.0], "y": [99.0]}), center)

    with pytest.raises(PredictionError, match="missing source"):
        apply_input_transforms(pl.DataFrame({"z": [1.0]}), center)

    fast_lookup = [
        {
            "type": "lookup",
            "name": "fast",
            "sources": ["x"],
            "output": "y",
            "output_dtype": "float64",
            "keys": [[1.0]],
            "values": [2.0],
            "default": 0.0,
            "on_unseen": "default",
            "on_null": "default",
        }
    ]
    with pytest.raises(PredictionError, match="already exists"):
        apply_input_transforms(pl.DataFrame({"x": [1.0], "y": [99.0]}), fast_lookup)

    generic_lookup = [
        {
            "type": "lookup",
            "name": "generic",
            "sources": ["x", "segment"],
            "output": "y",
            "output_dtype": "float64",
            "keys": [[1.0, "A"]],
            "values": [5.0],
            "default": -1.0,
            "on_unseen": "default",
            "on_null": "default",
        }
    ]
    with pytest.raises(PredictionError, match="already exists"):
        apply_input_transforms(
            pl.DataFrame({"x": [1.0], "segment": ["A"], "y": [99.0]}), generic_lookup
        )

    replaced = apply_input_transforms(
        pl.DataFrame({"x": [1.0, 2.0], "segment": ["A", "B"], "y": [99.0, 98.0]}),
        [{**generic_lookup[0], "replace_existing": True}],
    )
    assert replaced["y"].to_list() == [5.0, -1.0]

    none_key_lookup = [{**fast_lookup[0], "keys": [[None], [1.0]], "values": [9.0, 2.0]}]
    out = apply_input_transforms(pl.DataFrame({"x": [None, 1.0, 2.0]}), none_key_lookup)
    assert out["y"].to_list() == [0.0, 2.0, 0.0]


def test_input_transform_normalized_key_collision_fails_loud():
    transform = compile_input_transforms(
        [
            {
                "type": "lookup",
                "name": "text_collision",
                "sources": ["x"],
                "output": "y",
                "output_dtype": "float64",
                "keys": [["1"], [1]],
                "values": [2.0, 3.0],
                "default": 0.0,
            }
        ]
    )[0]
    frame = pl.DataFrame({"norm": ["1", "1"], "match": [0, 1]})

    with pytest.raises(ValidationError, match="collide after normalization"):
        _check_unique_normalized_keys(frame, transform, ["norm"], "match")


def test_input_transform_mixed_keys_against_string_source_fail_loud():
    spec = [
        {
            "type": "lookup",
            "name": "text_collision",
            "sources": ["x", "segment"],
            "output": "y",
            "output_dtype": "float64",
            "keys": [["1", "A"], [1, "A"]],
            "values": [2.0, 3.0],
            "default": 0.0,
            "on_unseen": "default",
            "on_null": "default",
        }
    ]

    with pytest.raises(ValidationError, match="cannot be normalized"):
        apply_input_transforms(pl.DataFrame({"x": ["1"], "segment": ["A"]}), spec)


def test_input_transform_low_level_helpers_enforce_type_contracts():
    mask_data = pl.DataFrame({"a": [1, None, 3], "b": [None, "x", "y"]})
    mask = _any_null_mask(mask_data, ["a", "b"])
    assert mask.to_list() == [True, True, False]
    assert _any_null_mask(mask_data, []).len() == 0
    assert _sample_bad_keys(mask_data, ["a", "b"], mask, n=1) == [(1, None)]
    assert _unique_temp_name(["tmp", "tmp_1"], "tmp") == "tmp_2"
    assert _dedup_key_value(10) == _dedup_key_value(10.0)
    assert _dedup_key_value("10") != _dedup_key_value(10)

    with pytest.raises(ValidationError, match="contains boolean"):
        _validate_key_value("lookup", True, 0)
    with pytest.raises(ValidationError, match="not finite"):
        _validate_key_value("lookup", float("nan"), 0)
    with pytest.raises(ValidationError, match="must be string"):
        _validate_key_value("lookup", object(), 0)

    with pytest.raises(ValidationError, match="contains boolean"):
        _validate_value("lookup", "float64", False, "values")
    with pytest.raises(ValidationError, match="not float64-compatible"):
        _validate_value("lookup", "float64", object(), "values")
    with pytest.raises(ValidationError, match="cannot be null"):
        _validate_value("lookup", "string", None, "default")
    assert _validate_value("lookup", "string", 123, "values") == "123"

    with pytest.raises(ValidationError, match="contains boolean"):
        _validate_numeric_param("center", True, "center")
    with pytest.raises(ValidationError, match="not float64-compatible"):
        _validate_numeric_param("center", object(), "center")
    with pytest.raises(ValidationError, match="not finite"):
        _validate_numeric_param("center", float("inf"), "center")

    transform = compile_input_transforms(
        [
            {
                "type": "lookup",
                "name": "bad_utf8",
                "sources": ["x"],
                "output": "y",
                "output_dtype": "float64",
                "keys": [["a"]],
                "values": [1.0],
                "default": 0.0,
            }
        ]
    )[0]
    with pytest.raises(ValidationError, match="cannot be normalized"):
        _utf8_key_series(transform, "x", [object()], "norm")


@pytest.mark.parametrize("source_dtype", [pl.Float64, pl.Int64, pl.Int32])
def test_lookup_numeric_keys_match_numeric_source_columns(source_dtype):
    # Regression: spec keys written as numbers must match numeric source columns,
    # not silently fall through to the default. Previously int key 10 was str()'d
    # to "10" while a Float64 source value 10.0 cast to "10.0" -> no match.
    spec = [
        {
            "type": "lookup",
            "name": "rating_factor",
            "sources": ["age_band"],
            "output": "age_factor",
            "output_dtype": "float64",
            "keys": [[10], [20]],
            "values": [1000.0, 2000.0],
            "default": -1.0,
            "on_unseen": "default",
            "on_null": "default",
        }
    ]
    df = pl.DataFrame({"age_band": pl.Series([10, 20, 30], dtype=source_dtype)})
    out = apply_input_transforms(df, spec)
    assert out["age_factor"].to_list() == [1000.0, 2000.0, -1.0]


def test_lookup_large_float_keys_match_polars_formatting():
    spec = [
        {
            "type": "lookup",
            "name": "big",
            "sources": ["x"],
            "output": "x_factor",
            "output_dtype": "float64",
            "keys": [[1e20]],
            "values": [5.0],
            "default": -1.0,
            "on_unseen": "default",
            "on_null": "default",
        }
    ]
    df = pl.DataFrame({"x": pl.Series([1e20, 1.0], dtype=pl.Float64)})
    assert apply_input_transforms(df, spec)["x_factor"].to_list() == [5.0, -1.0]


def test_lookup_rejects_non_finite_values_and_defaults():
    base = {
        "type": "lookup",
        "name": "t",
        "sources": ["x"],
        "output": "y",
        "output_dtype": "float64",
        "keys": [["a"]],
        "values": [1.0],
        "default": 0.0,
        "on_unseen": "default",
        "on_null": "default",
    }
    with pytest.raises(ValidationError, match="not finite"):
        validate_input_transforms([{**base, "values": [float("nan")]}])
    with pytest.raises(ValidationError, match="not finite"):
        validate_input_transforms([{**base, "values": [float("inf")]}])
    with pytest.raises(ValidationError, match="not finite"):
        validate_input_transforms([{**base, "default": float("inf")}])


def test_lookup_duplicate_numeric_keys_collide_across_int_float():
    with pytest.raises(ValidationError, match="duplicate key"):
        validate_input_transforms(
            [
                {
                    "type": "lookup",
                    "name": "dup",
                    "sources": ["x"],
                    "output": "y",
                    "output_dtype": "float64",
                    "keys": [[10], [10.0]],
                    "values": [1.0, 2.0],
                    "default": 0.0,
                }
            ]
        )


def test_lookup_keys_colliding_after_dtype_cast_fail_loud():
    # Regression: distinct float keys that truncate/collapse to the same value
    # against an integer source must fail loudly, not duplicate join rows (which
    # previously desynced the null mask and panicked).
    spec = [
        {
            "type": "lookup",
            "name": "t",
            "sources": ["x"],
            "output": "y",
            "output_dtype": "float64",
            "keys": [[10.1], [10.2]],
            "values": [1.0, 2.0],
            "default": -1.0,
            "on_unseen": "default",
            "on_null": "default",
        }
    ]
    df = pl.DataFrame({"x": pl.Series([10, 11], dtype=pl.Int64)})
    with pytest.raises(ValidationError):
        apply_input_transforms(df, spec)


def test_lookup_mixed_type_keys_for_one_source_fail_loud():
    # Regression: ["10"] (str) and [10] (int) for the same source must raise a
    # clear ValidationError, not a raw Polars TypeError on Series construction.
    spec = [
        {
            "type": "lookup",
            "name": "t",
            "sources": ["x"],
            "output": "y",
            "output_dtype": "float64",
            "keys": [["10"], [10]],
            "values": [1.0, 2.0],
            "default": -1.0,
            "on_unseen": "default",
            "on_null": "default",
        }
    ]
    df = pl.DataFrame({"x": pl.Series([10], dtype=pl.Int64)})
    with pytest.raises(ValidationError, match="cannot be normalized"):
        apply_input_transforms(df, spec)


def test_lookup_all_null_source_column_uses_null_policy():
    # An all-null source column has the Null dtype; keys can't cast to it, but
    # the lookup must still apply the on_null policy rather than crashing.
    base = {
        "type": "lookup",
        "name": "t",
        "sources": ["x"],
        "output": "y",
        "output_dtype": "float64",
        "keys": [["A"], ["B"]],
        "values": [1.0, 2.0],
        "default": -1.0,
    }
    df = pl.DataFrame({"x": [None, None]})  # inferred Null dtype
    # on_null="default" -> all rows take the default.
    out = apply_input_transforms(df, [{**base, "on_null": "default", "on_unseen": "default"}])
    assert out["y"].to_list() == [-1.0, -1.0]
    # on_null="raise" -> the null-source guard fires.
    with pytest.raises(PredictionError, match="null source"):
        apply_input_transforms(df, [{**base, "on_null": "raise", "on_unseen": "raise"}])


def _one_key_lookup(keys, values, default=-1.0, on_unseen="default"):
    return [
        {
            "type": "lookup",
            "name": "t",
            "sources": ["x"],
            "output": "y",
            "output_dtype": "float64",
            "keys": keys,
            "values": values,
            "default": default,
            "on_unseen": on_unseen,
            "on_null": "default",
        }
    ]


class TestNonStringSourceNormalization:
    """Audit regressions: keys against non-string source columns must either match
    correctly on the native dtype or fail loud — never silently mismatch, change
    row cardinality, leak a raw Polars error, or crash the process."""

    def test_enum_out_of_range_numeric_key_does_not_segfault(self):
        # Previously a numeric key cast to an out-of-range Enum physical code
        # crashed Polars below the FFI boundary (exit 139). It must now be a
        # clean no-match (keys never cast to the categorical dtype).
        df = pl.DataFrame({"x": pl.Series(["A", "B", "C"], dtype=pl.Enum(["A", "B", "C"]))})
        out = apply_input_transforms(df, _one_key_lookup([[99]], [5.0]))
        assert out["y"].to_list() == [-1.0, -1.0, -1.0]
        # A valid string key matches the category text.
        out2 = apply_input_transforms(df, _one_key_lookup([["B"]], [5.0]))
        assert out2["y"].to_list() == [-1.0, 5.0, -1.0]

    def test_categorical_numeric_key_is_not_a_physical_code_match(self):
        df = pl.DataFrame({"x": pl.Series(["A", "B"], dtype=pl.Categorical)})
        # key 1 must NOT match the category at physical code 1 ("B").
        out = apply_input_transforms(df, _one_key_lookup([[1]], [5.0]))
        assert out["y"].to_list() == [-1.0, -1.0]

    def test_signed_zero_matches_via_native_equality(self):
        df = pl.DataFrame({"x": pl.Series([-0.0, 1.0], dtype=pl.Float64)})
        out = apply_input_transforms(df, _one_key_lookup([[0.0]], [5.0]))
        assert out["y"].to_list() == [5.0, -1.0]

    def test_int_key_truncation_against_int_source_fails_loud(self):
        df = pl.DataFrame({"x": pl.Series([10, 11], dtype=pl.Int64)})
        with pytest.raises(ValidationError, match="cannot be normalized"):
            apply_input_transforms(df, _one_key_lookup([[10.1]], [5.0], on_unseen="raise"))

    def test_int_key_precision_loss_against_float32_fails_loud(self):
        df = pl.DataFrame({"x": pl.Series([16777216.0, 1.0], dtype=pl.Float32)})
        with pytest.raises(ValidationError, match="cannot be normalized"):
            apply_input_transforms(df, _one_key_lookup([[16777217]], [5.0]))

    def test_boolean_source_numeric_keys(self):
        df = pl.DataFrame({"x": pl.Series([True, False], dtype=pl.Boolean)})
        # non-0/1 numeric key is not representable as a bool -> fail loud
        with pytest.raises(ValidationError, match="cannot be normalized"):
            apply_input_transforms(df, _one_key_lookup([[2]], [5.0]))
        # 1 round-trips to True and matches
        out = apply_input_transforms(df, _one_key_lookup([[1]], [5.0]))
        assert out["y"].to_list() == [5.0, -1.0]

    def test_decimal_rounding_key_fails_loud(self):
        df = pl.DataFrame({"x": pl.Series([decimal.Decimal("1.51")], dtype=pl.Decimal(scale=2))})
        with pytest.raises(ValidationError, match="cannot be normalized"):
            apply_input_transforms(df, _one_key_lookup([[1.514]], [5.0]))

    def test_duration_source_does_not_leak_raw_error(self):
        df = pl.DataFrame({"x": pl.Series([1000, 2000], dtype=pl.Duration)})
        # native-dtype matching avoids the unsupported Duration->Utf8 cast.
        out = apply_input_transforms(df, _one_key_lookup([[1000]], [5.0]))
        assert out["y"].to_list() == [5.0, -1.0]

    def test_duration_source_degenerate_keys_do_not_leak_raw_error(self):
        # Empty key list and an all-null key + on_null=match previously leaked a
        # raw Duration->Utf8 InvalidOperationError; native matching avoids it.
        df = pl.DataFrame({"x": pl.Series([1000, None], dtype=pl.Duration)})
        out_empty = apply_input_transforms(
            df,
            _one_key_lookup([], []),  # everything -> default
        )
        assert out_empty["y"].to_list() == [-1.0, -1.0]
        spec = {
            "type": "lookup",
            "name": "t",
            "sources": ["x"],
            "output": "y",
            "output_dtype": "float64",
            "keys": [[None]],
            "values": [7.0],
            "default": -1.0,
            "on_null": "match",
        }
        out_match = apply_input_transforms(df, [spec])
        assert out_match["y"].to_list() == [-1.0, 7.0]


def test_lookup_replace_existing_output_equals_source_column():
    # validate_input_transforms permits output==source when replace_existing=True;
    # _apply_lookup must not drop the source column before normalizing it (which
    # previously leaked a raw ColumnNotFoundError).
    spec = [
        {
            "type": "lookup",
            "name": "t",
            "sources": ["x"],
            "output": "x",
            "output_dtype": "float64",
            "keys": [["A"], ["B"]],
            "values": [1.0, 2.0],
            "default": -1.0,
            "on_unseen": "default",
            "on_null": "default",
            "replace_existing": True,
        }
    ]
    df = pl.DataFrame({"x": ["A", "B", "C"]})
    out = apply_input_transforms(df, spec)
    assert out["x"].to_list() == [1.0, 2.0, -1.0]
    assert out.height == df.height


def test_lookup_transform_unseen_and_null_raise():
    unseen_spec = _numeric_lookup_spec()
    unseen_spec[0]["on_unseen"] = "raise"
    with pytest.raises(PredictionError, match="unseen key"):
        apply_input_transforms(
            pl.DataFrame({"brand": ["C"], "region": ["N"]}),
            unseen_spec,
        )

    null_spec = _numeric_lookup_spec()
    null_spec[0]["on_null"] = "raise"
    with pytest.raises(PredictionError, match="null source"):
        apply_input_transforms(
            pl.DataFrame({"brand": [None], "region": ["N"]}),
            null_spec,
        )


def test_lookup_transform_raise_policies_do_not_require_default():
    spec = [
        {
            "type": "lookup",
            "name": "strict",
            "sources": ["brand"],
            "output": "effect",
            "output_dtype": "float64",
            "keys": [["A"], ["B"]],
            "values": [0.1, 0.2],
            "on_unseen": "raise",
            "on_null": "raise",
        }
    ]

    canonical = validate_input_transforms(spec)
    assert canonical[0]["default"] is None
    assert validate_input_transforms(canonical) == canonical
    out = apply_input_transforms(pl.DataFrame({"brand": ["A"]}), canonical)
    assert out["effect"].to_list() == [0.1]


def test_apply_input_transforms_lazy_matches_eager_error_semantics():
    spec = _numeric_lookup_spec()
    spec[0]["on_unseen"] = "raise"

    with pytest.raises(PredictionError, match="unseen key"):
        apply_input_transforms_lazy(
            pl.DataFrame({"brand": ["missing"], "region": ["S"]}).lazy(),
            spec,
        ).collect()


def test_lookup_transform_null_match_is_explicit():
    spec = [
        {
            "type": "lookup",
            "name": "null_key",
            "sources": ["brand"],
            "output": "effect",
            "output_dtype": "float64",
            "keys": [[None], ["A"]],
            "values": [0.5, 0.1],
            "default": 0.0,
            "on_null": "match",
        }
    ]

    out = apply_input_transforms(pl.DataFrame({"brand": [None, "A", "B"]}), spec)

    assert out["effect"].to_list() == [0.5, 0.1, 0.0]


def test_lookup_transform_internal_temp_columns_do_not_clobber_user_columns():
    temp_col = "__rustystats_it_brand_region_effect_0_brand__"
    match_col = "__rustystats_input_transform_match__"
    df = pl.DataFrame(
        {
            "brand": ["A", "B"],
            "region": ["N", "S"],
            temp_col: ["keep_a", "keep_b"],
            match_col: ["m1", "m2"],
        }
    )

    out = apply_input_transforms(df, _numeric_lookup_spec())

    assert out[temp_col].to_list() == ["keep_a", "keep_b"]
    assert out[match_col].to_list() == ["m1", "m2"]
    assert out["brand_region_fts"].to_list() == [0.2, -0.1]


def test_lookup_validation_rejects_duplicate_keys_and_output_collision():
    with pytest.raises(ValidationError, match="duplicate key"):
        validate_input_transforms(
            [
                {
                    "type": "lookup",
                    "name": "bad",
                    "sources": ["x"],
                    "output": "x_fts",
                    "output_dtype": "float64",
                    "keys": [["A"], ["A"]],
                    "values": [1.0, 2.0],
                    "default": 0.0,
                }
            ]
        )

    with pytest.raises(ValidationError, match="already exists"):
        validate_input_transforms(
            _numeric_lookup_spec(),
            data_schema={
                "brand": pl.Utf8,
                "region": pl.Utf8,
                "brand_region_fts": pl.Float64,
            },
        )


def test_glm_dict_input_transforms_match_prepared_data_predictions():
    df = _training_data()
    specs = _numeric_lookup_spec()

    raw_result = rs.glm_dict(
        response="y",
        terms={"brand_region_fts": {"type": "linear"}},
        data=df,
        family="poisson",
        exposure="exposure",
        input_transforms=specs,
    ).fit()
    prepared = raw_result.prepare_input(df)
    prepared_result = rs.glm_dict(
        response="y",
        terms={"brand_region_fts": {"type": "linear"}},
        data=prepared,
        family="poisson",
        exposure="exposure",
    ).fit()

    np.testing.assert_allclose(raw_result.params, prepared_result.params, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(raw_result.predict(df), prepared_result.predict(prepared))
    np.testing.assert_allclose(raw_result.predict(df), raw_result.predict(prepared))
    assert raw_result.required_columns == ["brand", "exposure", "region"]


def test_glm_numeric_input_transform_scores_raw_interaction_data():
    df = pl.DataFrame(
        {
            "y": [0.2, 0.4, 0.8, 1.1, 1.5, 1.9, 2.4, 2.8],
            "x": [0.0, 0.2, 0.4, 0.9, 1.1, 1.4, 1.8, 2.0],
            "brand": ["A", "B", "A", "B", "A", "B", "A", "B"],
        }
    )
    specs = [
        {
            "type": "center",
            "name": "x_center",
            "sources": ["x"],
            "output": "x_ctr",
            "center": 1.0,
            "replace_existing": True,
        }
    ]
    interactions = [
        {
            "x_ctr": {"type": "linear"},
            "brand": {"type": "categorical"},
            "include_main": False,
        }
    ]
    result = rs.glm_dict(
        response="y",
        terms={"x_ctr": {"type": "linear"}, "brand": {"type": "categorical"}},
        interactions=interactions,
        data=df,
        family="gaussian",
        input_transforms=specs,
    ).fit(compute_covariance=False)

    raw_pred = result.predict(df)
    prepared_pred = result.predict(apply_input_transforms(df, specs))

    np.testing.assert_allclose(raw_pred, prepared_pred, rtol=1e-12, atol=1e-12)


def test_glm_dict_input_transforms_lazyframe_and_serialization(tmp_path):
    df = _training_data()
    path = tmp_path / "train.parquet"
    df.write_parquet(path)

    result = rs.glm_dict(
        response="y",
        terms={"brand_region_fts": {"type": "linear"}},
        data=pl.scan_parquet(path),
        family="poisson",
        exposure="exposure",
        input_transforms=_numeric_lookup_spec(),
    ).fit()
    loaded = rs.GLMModel.from_bytes(result.to_bytes())

    np.testing.assert_allclose(result.predict(df), loaded.predict(df))
    assert loaded.input_transforms == result.input_transforms


def test_glm_dict_chained_input_transforms_work_with_lazyframe_pruning(tmp_path):
    df = _training_data()
    path = tmp_path / "train.parquet"
    df.write_parquet(path)
    specs = [
        {
            "type": "lookup",
            "name": "brand_group",
            "sources": ["brand"],
            "output": "brand_grp",
            "output_dtype": "string",
            "keys": [["A"], ["B"], ["A|B"], ["C"]],
            "values": ["g1", "g2", "g3", "g2"],
            "default": "other",
            "on_unseen": "default",
            "on_null": "default",
        },
        {
            "type": "lookup",
            "name": "group_effect",
            "sources": ["brand_grp"],
            "output": "group_effect",
            "output_dtype": "float64",
            "keys": [["g1"], ["g2"], ["g3"], ["other"]],
            "values": [0.2, -0.1, 0.4, 0.0],
            "default": 0.0,
            "on_unseen": "default",
            "on_null": "default",
        },
    ]

    result = rs.glm_dict(
        response="y",
        terms={"group_effect": {"type": "linear"}},
        data=pl.scan_parquet(path),
        family="poisson",
        exposure="exposure",
        input_transforms=specs,
    ).fit()

    assert result.required_columns == ["brand", "exposure"]
    np.testing.assert_allclose(result.predict(df), result.predict(result.prepare_input(df)))


def test_predict_resolves_transform_produced_exposure_from_prepared_data():
    df = _training_data().with_columns(
        pl.when(pl.col("exposure") >= 1.0)
        .then(pl.lit("high"))
        .otherwise(pl.lit("low"))
        .alias("expo_band")
    )
    specs = [
        {
            "type": "lookup",
            "name": "exposure_lookup",
            "sources": ["expo_band"],
            "output": "expo",
            "output_dtype": "float64",
            "keys": [["high"], ["low"]],
            "values": [1.25, 0.75],
            "on_unseen": "raise",
            "on_null": "raise",
        }
    ]

    result = rs.glm_dict(
        response="y",
        terms={"brand": {"type": "categorical"}},
        data=df,
        family="poisson",
        exposure="expo",
        input_transforms=specs,
    ).fit()

    np.testing.assert_allclose(result.predict(df), result.predict(result.prepare_input(df)))
    assert result.diagnostics(train_data=df) is not None


def test_predict_transform_produced_exposure_chunked_matches_unchunked(monkeypatch):
    # When the exposure column is itself produced by a transform, the chunked
    # prediction path resolves it through a projected (memory-bounded) prepared
    # frame; it must match the single-shot path exactly.
    import rustystats.formula as F

    df = _training_data().with_columns(
        pl.when(pl.col("exposure") >= 1.0)
        .then(pl.lit("high"))
        .otherwise(pl.lit("low"))
        .alias("expo_band")
    )
    specs = [
        {
            "type": "lookup",
            "name": "exposure_lookup",
            "sources": ["expo_band"],
            "output": "expo",
            "output_dtype": "float64",
            "keys": [["high"], ["low"]],
            "values": [1.25, 0.75],
            "on_unseen": "raise",
            "on_null": "raise",
        }
    ]
    result = rs.glm_dict(
        response="y",
        terms={"brand": {"type": "categorical"}},
        data=df,
        family="poisson",
        exposure="expo",
        input_transforms=specs,
    ).fit()
    unchunked = result.predict(df)  # small data -> single-shot path
    monkeypatch.setattr(F, "_compute_predict_chunk_size", lambda n_features: 2)
    chunked = result.predict(df)  # forced chunked path -> projected aux frame
    np.testing.assert_allclose(unchunked, chunked, rtol=1e-12, atol=1e-12)


def test_predict_chunked_aux_mixes_transform_produced_and_raw_columns(monkeypatch):
    # A transform-produced exposure alongside a RAW offset column: the projected
    # aux frame must keep the raw offset column too, not just transform sources.
    import rustystats.formula as F

    df = _training_data().with_columns(
        pl.when(pl.col("exposure") >= 1.0)
        .then(pl.lit("high"))
        .otherwise(pl.lit("low"))
        .alias("expo_band"),
        pl.Series("off", [0.1, 0.2, 0.0, 0.3, 0.1, 0.2, 0.0, 0.3]),
    )
    specs = [
        {
            "type": "lookup",
            "name": "exposure_lookup",
            "sources": ["expo_band"],
            "output": "expo",
            "output_dtype": "float64",
            "keys": [["high"], ["low"]],
            "values": [1.25, 0.75],
            "on_unseen": "raise",
            "on_null": "raise",
        }
    ]
    result = rs.glm_dict(
        response="y",
        terms={"brand": {"type": "categorical"}},
        data=df,
        family="poisson",
        exposure="expo",  # transform-produced
        offset="off",  # raw column
        input_transforms=specs,
    ).fit()
    unchunked = result.predict(df)
    monkeypatch.setattr(F, "_compute_predict_chunk_size", lambda n_features: 2)
    chunked = result.predict(df)
    np.testing.assert_allclose(unchunked, chunked, rtol=1e-12, atol=1e-12)


def test_predict_contributions_with_input_transforms_adds_up():
    df = _training_data()
    result = rs.glm_dict(
        response="y",
        terms={"brand_region_fts": {"type": "linear"}},
        data=df,
        family="poisson",
        exposure="exposure",
        input_transforms=_numeric_lookup_spec(),
    ).fit()

    rows = result.predict_contributions(df.head(3), validate=True)

    assert len(rows) == 3
    assert any(c["term"] == "brand_region_fts" for c in rows[0]["contributions"])


def test_rate_tables_inline_numeric_and_grouped_transforms(tmp_path):
    df = _training_data()
    numeric = rs.glm_dict(
        response="y",
        terms={"brand_region_fts": {"type": "linear"}},
        data=df,
        family="poisson",
        exposure="exposure",
        input_transforms=_numeric_lookup_spec(),
    ).fit()

    artifact = numeric.to_rate_tables()
    table = artifact["tables"][0]
    assert table["kind"] == "lookup_linear"
    assert table["columns"] == ["brand", "region", "brand_region_fts", "eta", "rel"]
    assert table["default"][0:2] == ["<default>", "<default>"]
    assert table["metadata"]["on_unseen"] == "default"
    assert table["metadata"]["on_null"] == "default"

    csv_path = tmp_path / "rates.csv"
    numeric.to_rate_tables(path=csv_path, format="stacked_csv")
    blocks = _parse_stacked_csv(csv_path)
    assert "_base" in blocks
    assert "_manifest" in blocks
    assert "brand_region_fts" in blocks
    assert blocks["brand_region_fts"]["columns"] == table["columns"]
    assert len(blocks["brand_region_fts"]["rows"]) == len(table["rows"]) + 1

    grouped = rs.glm_dict(
        response="y",
        terms={"postcode_grp": {"type": "categorical"}},
        data=df,
        family="poisson",
        exposure="exposure",
        input_transforms=_group_lookup_spec(),
    ).fit()
    grouped_table = grouped.to_rate_tables()["tables"][0]
    assert grouped_table["kind"] == "lookup_categorical"
    assert grouped_table["columns"] == ["postcode", "postcode_grp", "eta", "rel"]


def test_rate_tables_export_selected_categorical_levels():
    df = _training_data()
    result = rs.glm_dict(
        response="y",
        terms={"brand": {"type": "categorical", "levels": ["A", "B"]}},
        data=df,
        family="poisson",
    ).fit()

    table = result.to_rate_tables()["tables"][0]
    assert table["kind"] == "categorical"
    assert table["columns"] == ["brand", "eta", "rel"]
    assert [row[0] for row in table["rows"]] == ["A", "B"]


def test_rate_tables_do_not_emit_default_row_for_strict_lookup():
    df = _training_data().filter(
        ((pl.col("brand") == "A") & (pl.col("region") == "N"))
        | ((pl.col("brand") == "B") & (pl.col("region") == "S"))
        | ((pl.col("brand") == "A|B") & (pl.col("region") == "N"))
    )
    specs = _numeric_lookup_spec()
    specs[0].pop("default")
    specs[0]["on_unseen"] = "raise"
    specs[0]["on_null"] = "raise"
    result = rs.glm_dict(
        response="y",
        terms={"brand_region_fts": {"type": "linear"}},
        data=df,
        family="poisson",
        input_transforms=specs,
    ).fit()

    table = result.to_rate_tables()["tables"][0]
    assert table["default"] is None
    assert table["metadata"]["on_unseen"] == "raise"
    assert table["metadata"]["on_null"] == "raise"


def test_rate_tables_render_null_match_keys_explicitly():
    df = pl.DataFrame({"y": [1, 2, 3, 4], "brand": [None, "A", None, "A"]})
    spec = [
        {
            "type": "lookup",
            "name": "null_key",
            "sources": ["brand"],
            "output": "effect",
            "output_dtype": "float64",
            "keys": [[None], ["A"]],
            "values": [0.5, 0.1],
            "default": 0.0,
            "on_null": "match",
        }
    ]
    result = rs.glm_dict(
        response="y",
        terms={"effect": {"type": "linear"}},
        data=df,
        family="poisson",
        input_transforms=spec,
    ).fit()

    table = result.to_rate_tables()["tables"][0]
    assert table["rows"][0][0] == "<null>"


def test_rate_tables_export_target_and_frequency_encoding_terms():
    df = pl.DataFrame(
        {
            "y": [1, 2, 1, 3, 2, 4, 2, 5, 1, 3, 2, 4, 3, 1, 5, 2],
            "brand": [
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
                "B",
                "C",
                "C",
                "C",
                "D",
                "D",
                "E",
                "E",
                "E",
                "E",
            ],
        }
    )
    te = rs.glm_dict(
        response="y",
        terms={"brand": {"type": "target_encoding", "prior_weight": 1.0}},
        data=df,
        family="poisson",
    ).fit()
    te_table = te.to_rate_tables()["tables"][0]

    assert te_table["kind"] == "target_encoding"
    assert te_table["columns"] == ["brand", "TE(brand)", "eta", "rel"]
    assert te_table["default"][0] == "<unseen>"
    assert te_table["default"][1] > 0

    fe = rs.glm_dict(
        response="y",
        terms={"brand": {"type": "frequency_encoding"}},
        data=df,
        family="poisson",
    ).fit()
    fe_table = fe.to_rate_tables()["tables"][0]

    assert fe_table["kind"] == "frequency_encoding"
    assert fe_table["columns"] == ["brand", "FE(brand)", "eta", "rel"]
    assert fe_table["default"] == ["<unseen>", 0.0, 0.0, 1.0]


def test_rate_tables_make_sanitized_names_unique():
    df = pl.DataFrame(
        {
            "y": [1, 2, 1, 3, 2, 4, 2, 5],
            "brand": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "region": ["N", "S", "N", "S", "N", "S", "N", "S"],
        }
    )
    specs = [
        {
            "type": "lookup",
            "name": "linear_te_brand_name_collision",
            "sources": ["region"],
            "output": "TE_brand",
            "output_dtype": "float64",
            "keys": [["N"], ["S"]],
            "values": [0.1, -0.1],
            "default": 0.0,
            "on_unseen": "default",
            "on_null": "default",
        }
    ]
    result = rs.glm_dict(
        response="y",
        terms={
            "brand": {"type": "target_encoding", "prior_weight": 1.0},
            "TE_brand": {"type": "linear"},
        },
        data=df,
        family="poisson",
        input_transforms=specs,
    ).fit()

    names = [table["name"] for table in result.to_rate_tables()["tables"]]
    assert names == ["TE_brand", "TE_brand_2"]


def test_rate_table_names_unique_avoid_collision_with_genuine_suffix():
    # A generated "foo_2" must not collide with a genuine table already named "foo_2".
    from rustystats.rate_tables import _make_table_names_unique

    tables = [{"name": "foo"}, {"name": "foo"}, {"name": "foo_2"}]
    _make_table_names_unique(tables)
    names = [t["name"] for t in tables]
    assert names == ["foo", "foo_2", "foo_2_2"]
    assert len(set(names)) == len(names)


def test_rate_tables_artifact_blocks_helper_removed():
    import rustystats.rate_tables as rt

    assert not hasattr(rt, "_artifact_blocks")


def test_rate_tables_reject_splines_by_default():
    df = _training_data()
    result = rs.glm_dict(
        response="y",
        terms={"brand_region_fts": {"type": "linear"}, "exposure": {"type": "bs", "df": 4}},
        data=df,
        family="poisson",
        input_transforms=_numeric_lookup_spec(),
    ).fit()

    with pytest.raises(ValidationError, match="spline"):
        result.to_rate_tables()


def test_rate_tables_export_spline_grid_when_requested(tmp_path):
    df = pl.DataFrame(
        {
            "y": [1, 1, 2, 2, 3, 4, 3, 5, 4, 6, 5, 7],
            "age": [18, 21, 25, 28, 32, 36, 41, 47, 53, 59, 65, 72],
            "exposure": [1.0] * 12,
        }
    )
    result = rs.glm_dict(
        response="y",
        terms={"age": {"type": "bs", "df": 4}},
        data=df,
        family="poisson",
        exposure="exposure",
    ).fit()

    grid = [18, 25, 35, 45, 55, 65, 72]
    artifact = result.to_rate_tables(
        spline_strategy="grid",
        spline_grids={"age": grid},
        spline_interpolation="linear",
        spline_extrapolation="clip",
    )

    table = artifact["tables"][0]
    assert table["kind"] == "spline_grid"
    assert table["columns"] == ["age", "eta", "rel"]
    assert [row[0] for row in table["rows"]] == [float(v) for v in grid]
    assert table["metadata"]["approximation"] is True
    assert table["metadata"]["interpolation"] == "linear"
    assert table["metadata"]["extrapolation"] == "clip"

    csv_path = tmp_path / "spline_rates.csv"
    result.to_rate_tables(
        path=csv_path,
        format="stacked_csv",
        spline_strategy="grid",
        spline_grids={"age": grid},
    )
    blocks = _parse_stacked_csv(csv_path)
    assert blocks["age"]["columns"] == ["age", "eta", "rel"]
    manifest = blocks["_manifest"]
    assert "approximation" in manifest["columns"]
    spline_row = next(row for row in manifest["rows"] if row[0] == "age")
    assert spline_row[manifest["columns"].index("approximation")] == "True"
    assert spline_row[manifest["columns"].index("interpolation")] == "linear"
    assert spline_row[manifest["columns"].index("extrapolation")] == "clip"


def test_rate_tables_spline_grid_requires_explicit_increasing_grid():
    df = pl.DataFrame(
        {
            "y": [1, 1, 2, 2, 3, 4, 3, 5],
            "age": [18, 21, 25, 28, 32, 36, 41, 47],
        }
    )
    result = rs.glm_dict(
        response="y",
        terms={"age": {"type": "bs", "df": 4}},
        data=df,
        family="poisson",
    ).fit()

    with pytest.raises(ValidationError, match="requires spline_grids"):
        result.to_rate_tables(spline_strategy="grid")
    with pytest.raises(ValidationError, match="strictly increasing"):
        result.to_rate_tables(
            spline_strategy="grid",
            spline_grids={"age": [18, 25, 25, 40]},
        )


def test_raw_export_modes_fail_closed_with_input_transforms():
    df = _training_data()
    result = rs.glm_dict(
        response="y",
        terms={"brand_region_fts": {"type": "linear"}},
        data=df,
        family="poisson",
        input_transforms=_numeric_lookup_spec(),
    ).fit()

    with pytest.raises(ValidationError, match="PMML raw-data export"):
        result.to_pmml()
    with pytest.raises(ValidationError, match="ONNX full raw-data export"):
        result.to_onnx(mode="full")
