#!/usr/bin/env python3
"""Validate checked-in external oracle fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORACLE_ROOT = ROOT / "tests" / "oracles"
REQUIRED_STATS_FAMILIES = {
    "gaussian",
    "poisson",
    "binomial",
    "gamma",
    "tweedie",
    "negbinomial",
}


def _fixture_paths() -> list[Path]:
    statsmodels_root = ORACLE_ROOT / "statsmodels"
    if not statsmodels_root.exists():
        return []
    return sorted(statsmodels_root.glob("*.json"))


def _validate_fixture(path: Path, failures: list[str]) -> str | None:
    rel = path.relative_to(ROOT)
    try:
        fixture = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        failures.append(f"{rel}: invalid JSON: {exc}")
        return None

    if fixture.get("schema_version") != 1:
        failures.append(f"{rel}: schema_version must be 1")
    for field in ("case_id", "oracle", "oracle_version", "data", "model", "expected", "tolerances"):
        if field not in fixture:
            failures.append(f"{rel}: missing {field}")
    case_id = fixture.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        failures.append(f"{rel}: case_id must be non-empty")

    columns = fixture.get("data", {}).get("columns")
    if not isinstance(columns, dict) or not columns:
        failures.append(f"{rel}: data.columns must be a non-empty object")
    else:
        lengths = {len(v) for v in columns.values() if isinstance(v, list)}
        if len(lengths) != 1:
            failures.append(f"{rel}: all data columns must be lists of the same length")
        if "y" not in columns:
            failures.append(f"{rel}: data.columns must include response y")

    model = fixture.get("model", {})
    family = model.get("family")
    if not isinstance(family, str) or not family:
        failures.append(f"{rel}: model.family must be non-empty")
    if not isinstance(model.get("terms"), dict) or not model.get("terms"):
        failures.append(f"{rel}: model.terms must be a non-empty object")

    expected = fixture.get("expected", {})
    params = expected.get("params")
    fitted = expected.get("fittedvalues")
    if not isinstance(params, list) or not params:
        failures.append(f"{rel}: expected.params must be a non-empty list")
    if not isinstance(fitted, list) or not fitted:
        failures.append(f"{rel}: expected.fittedvalues must be a non-empty list")

    tolerances = fixture.get("tolerances", {})
    for key in ("params_atol", "params_rtol", "deviance_atol", "deviance_rtol"):
        value = tolerances.get(key)
        if not isinstance(value, int | float) or value < 0:
            failures.append(f"{rel}: tolerances.{key} must be non-negative")

    return family


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing-family", action="append", default=[])
    args = parser.parse_args()

    failures: list[str] = []
    families: set[str] = set()
    paths = _fixture_paths()
    if not paths:
        failures.append("no oracle fixtures found under tests/oracles")

    for path in paths:
        family = _validate_fixture(path, failures)
        if family:
            families.add(family.lower())

    required = REQUIRED_STATS_FAMILIES - set(args.allow_missing_family)
    missing = sorted(required - families)
    if missing:
        failures.append(f"missing required statsmodels oracle families: {missing}")

    if failures:
        print("Oracle fixture check failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print(f"Oracle fixture check passed for {len(paths)} fixtures.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
