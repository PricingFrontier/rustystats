#!/usr/bin/env python3
"""Replay minimized malformed-input corpus cases."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import rustystats as rs

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "tests" / "fuzz_corpus" / "python_malformed.json"


def _base_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "y": [0.0, 1.0, 2.0, 1.0, 3.0, 2.0],
            "x": [0.1, 0.5, -0.3, 1.2, -0.7, 0.0],
            "brand": ["A", "B", "A", "C", "B", "C"],
        }
    )


def _small_model():
    return rs.glm_dict(
        response="y",
        terms={"x": {"type": "linear"}},
        data=_base_data(),
        family="poisson",
    ).fit()


def _glm_missing_response_column() -> None:
    rs.glm_dict(
        response="missing",
        terms={"x": {"type": "linear"}},
        data=_base_data(),
        family="poisson",
    ).fit()


def _glm_unknown_term_type() -> None:
    rs.glm_dict(
        response="y",
        terms={"x": {"type": "not_a_term"}},
        data=_base_data(),
        family="poisson",
    ).fit()


def _glm_weight_length_mismatch() -> None:
    rs.glm_dict(
        response="y",
        terms={"x": {"type": "linear"}},
        data=_base_data(),
        family="poisson",
        weights=np.array([1.0, 2.0]),
    ).fit()


def _glm_corrupt_serialized_model() -> None:
    rs.GLMModel.from_bytes(b"not a rustystats model")


def _onnx_unknown_mode() -> None:
    _small_model().to_onnx(mode="definitely-not-a-mode")


def _rate_table_unknown_format() -> None:
    rs.glm_dict(
        response="y",
        terms={"brand": {"type": "categorical"}},
        data=_base_data(),
        family="poisson",
    ).fit().to_rate_tables(format="json")


def _pmml_rejects_input_transforms() -> None:
    data = _base_data()
    spec = [
        {
            "type": "lookup",
            "name": "brand_lookup",
            "sources": ["brand"],
            "output": "brand_score",
            "output_dtype": "float64",
            "keys": [["A"], ["B"], ["C"]],
            "values": [0.1, 0.2, 0.3],
            "default": 0.0,
            "on_unseen": "default",
            "on_null": "default",
        }
    ]
    rs.glm_dict(
        response="y",
        terms={"brand_score": {"type": "linear"}},
        data=data,
        family="poisson",
        input_transforms=spec,
    ).fit().to_pmml()


def _glm_negative_exposure() -> None:
    data = _base_data().with_columns(pl.Series("exposure", [1.0, 0.5, 0.0, 1.0, 1.5, 2.0]))
    rs.glm_dict(
        response="y",
        terms={"x": {"type": "linear"}},
        data=data,
        family="poisson",
        exposure="exposure",
    ).fit()


def _glm_unknown_family() -> None:
    rs.glm_dict(
        response="y",
        terms={"x": {"type": "linear"}},
        data=_base_data(),
        family="definitely-not-a-family",
    ).fit()


def _predict_missing_required_column() -> None:
    model = _small_model()
    model.predict(_base_data().drop("x"))


def _glm_invalid_expression_term() -> None:
    rs.glm_dict(
        response="y",
        terms={"bad": {"type": "expression", "expr": "x + "}},
        data=_base_data(),
        family="poisson",
    ).fit()


ACTION_MAP: dict[str, Callable[[], None]] = {
    "glm_missing_response_column": _glm_missing_response_column,
    "glm_unknown_term_type": _glm_unknown_term_type,
    "glm_weight_length_mismatch": _glm_weight_length_mismatch,
    "glm_corrupt_serialized_model": _glm_corrupt_serialized_model,
    "onnx_unknown_mode": _onnx_unknown_mode,
    "rate_table_unknown_format": _rate_table_unknown_format,
    "pmml_rejects_input_transforms": _pmml_rejects_input_transforms,
    "glm_negative_exposure": _glm_negative_exposure,
    "glm_unknown_family": _glm_unknown_family,
    "predict_missing_required_column": _predict_missing_required_column,
    "glm_invalid_expression_term": _glm_invalid_expression_term,
}


def _exception_name(exc: BaseException) -> str:
    if isinstance(exc, pickle.UnpicklingError):
        return "UnpicklingError"
    return exc.__class__.__name__


def replay(corpus_path: Path) -> int:
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if corpus.get("schema_version") != 1:
        failures.append("schema_version must be 1")
    seen_ids: set[str] = set()
    for case in corpus.get("cases", []):
        case_id = case["id"]
        if case_id in seen_ids:
            failures.append(f"{case_id}: duplicate case id")
            continue
        seen_ids.add(case_id)
        action_name = case["action"]
        action = ACTION_MAP.get(action_name)
        if action is None:
            failures.append(f"{case_id}: unknown action {action_name!r}")
            continue
        if not isinstance(case.get("expected_exception"), list) or not case["expected_exception"]:
            failures.append(f"{case_id}: expected_exception must be a non-empty list")
            continue

        try:
            action()
        except Exception as exc:
            actual = _exception_name(exc)
            expected = set(case["expected_exception"])
            if actual not in expected:
                failures.append(f"{case_id}: got {actual}, expected one of {sorted(expected)}")
            elif not str(exc):
                failures.append(f"{case_id}: {actual} had an empty error message")
        else:
            failures.append(f"{case_id}: malformed input unexpectedly succeeded")

    if failures:
        print("Fuzz corpus replay failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print(f"Fuzz corpus replay passed for {len(corpus.get('cases', []))} cases.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args()
    return replay(args.corpus)


if __name__ == "__main__":
    sys.exit(main())
