#!/usr/bin/env python3
"""Generate archived statsmodels MNLogit oracle fixtures."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
ORACLE_DIR = ROOT / "tests" / "oracles" / "statsmodels_multinomial"


def _round_array(values: np.ndarray) -> list[float] | list[list[float]]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return [float(f"{v:.16g}") for v in arr]
    return [[float(f"{v:.16g}") for v in row] for row in arr]


def _fixture() -> dict:
    rng = np.random.default_rng(20260701)
    n = 180
    classes = ["bronze", "silver", "gold"]
    reference = "silver"
    x1 = np.linspace(-1.5, 1.8, n)
    x2 = np.sin(np.linspace(0.0, 5.0, n))
    eta_bronze = -0.2 + 0.7 * x1 - 0.25 * x2
    eta_gold = 0.35 - 0.3 * x1 + 0.5 * x2
    logits = np.column_stack([eta_bronze, np.zeros(n), eta_gold])
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    y = np.asarray([classes[rng.choice(len(classes), p=row)] for row in probabilities])

    # statsmodels chooses an internal baseline/category ordering, so fixtures
    # store probabilities aligned back to the RustyStats class order.
    y_categorical = pd.Categorical(y, categories=[reference, "bronze", "gold"])
    x_design = sm.add_constant(np.column_stack([x1, x2]))
    result = sm.MNLogit(y_categorical, x_design).fit(method="newton", maxiter=200, disp=False)
    raw_probabilities = np.asarray(result.predict(x_design), dtype=np.float64)
    statsmodels_order = [result.model._ynames_map[idx] for idx in range(result.model.J)]
    order = [statsmodels_order.index(label) for label in classes]
    aligned_probabilities = raw_probabilities[:, order]

    return {
        "schema_version": 1,
        "case_id": "statsmodels-mnlogit-nondefault-reference",
        "oracle": "statsmodels.MNLogit",
        "oracle_version": sm.__version__,
        "data": {
            "columns": {
                "choice": [str(value) for value in y],
                "x1": _round_array(x1),
                "x2": _round_array(x2),
            }
        },
        "model": {
            "response": "choice",
            "terms": {"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            "classes": classes,
            "reference": reference,
            "fit_kwargs": {
                "max_iter": 200,
                "tol": 1e-10,
                "compute_covariance": False,
                "standardize": False,
            },
        },
        "expected": {
            "statsmodels_probability_order": statsmodels_order,
            "probabilities": _round_array(aligned_probabilities),
            "class_counts": {label: int(np.sum(y == label)) for label in classes},
            "log_likelihood": float(f"{result.llf:.16g}"),
        },
        "tolerances": {
            "probability_atol": 2e-6,
            "probability_rtol": 2e-6,
            "row_sum_atol": 1e-12,
        },
    }


def build_fixtures() -> dict[str, dict]:
    return {"nondefault_reference.json": _fixture()}


def _write_fixtures(directory: Path, fixtures: dict[str, dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, fixture in fixtures.items():
        (directory / name).write_text(
            json.dumps(fixture, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accept", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    fixtures = build_fixtures()
    if args.accept:
        _write_fixtures(ORACLE_DIR, fixtures)
        print(
            f"Wrote {len(fixtures)} multinomial oracle fixtures to {ORACLE_DIR.relative_to(ROOT)}."
        )
        return 0

    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="rustystats-mnlogit-oracles-"))
    if args.output_dir and output_dir.exists():
        shutil.rmtree(output_dir)
    _write_fixtures(output_dir, fixtures)

    if args.check:
        failures = []
        for name in fixtures:
            expected = ORACLE_DIR / name
            generated = output_dir / name
            if not expected.is_file():
                failures.append(f"missing checked-in fixture {expected.relative_to(ROOT)}")
            elif expected.read_text(encoding="utf-8") != generated.read_text(encoding="utf-8"):
                failures.append(f"fixture drift: {expected.relative_to(ROOT)}")
        if failures:
            print("Multinomial oracle fixture generation check failed:")
            for failure in failures:
                print(f" - {failure}")
            print(f"Generated comparison fixtures in {output_dir}")
            return 1
        print(f"Multinomial oracle fixture generation check passed for {len(fixtures)} fixtures.")
        return 0

    print(f"Wrote comparison fixtures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
