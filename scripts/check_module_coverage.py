#!/usr/bin/env python3
"""Evaluate high-assurance per-module coverage thresholds."""

from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD_FILE = ROOT / "specs" / "module_coverage_thresholds.json"
ID_RE = re.compile(r"^[A-Z]+(?:-[A-Z0-9]+)+$")
COVERAGE_TYPES = {"python-json", "rust-lcov"}
REQUIRED_FIELDS = {
    "id",
    "coverage_type",
    "min_percent",
    "owner",
    "rationale",
}


def _normalize_path(path: str | Path) -> str:
    raw = str(path).replace("\\", "/")
    try:
        return str(Path(raw).resolve().relative_to(ROOT)).replace("\\", "/")
    except (OSError, ValueError):
        return raw.lstrip("./")


def _load_thresholds(path: Path = THRESHOLD_FILE) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_config(spec: dict) -> list[str]:
    failures: list[str] = []
    if spec.get("schema_version") != 1:
        failures.append("schema_version must be 1")

    thresholds = spec.get("thresholds")
    if not isinstance(thresholds, list) or not thresholds:
        failures.append("thresholds must be a non-empty list")
        thresholds = []

    seen: set[str] = set()
    for threshold in thresholds:
        threshold_id = threshold.get("id", "<missing id>")
        missing = sorted(REQUIRED_FIELDS - set(threshold))
        if missing:
            failures.append(f"{threshold_id}: missing {missing}")
            continue

        if not isinstance(threshold_id, str) or not ID_RE.match(threshold_id):
            failures.append(f"{threshold_id}: invalid threshold id")
        if threshold_id in seen:
            failures.append(f"{threshold_id}: duplicate threshold id")
        seen.add(str(threshold_id))

        coverage_type = threshold["coverage_type"]
        if coverage_type not in COVERAGE_TYPES:
            failures.append(
                f"{threshold_id}: coverage_type must be one of {sorted(COVERAGE_TYPES)}"
            )

        has_path = isinstance(threshold.get("path"), str) and bool(threshold.get("path"))
        has_glob = isinstance(threshold.get("path_glob"), str) and bool(threshold.get("path_glob"))
        if has_path == has_glob:
            failures.append(f"{threshold_id}: specify exactly one of path or path_glob")

        for key in ("path", "path_glob"):
            if key in threshold and Path(str(threshold[key])).is_absolute():
                failures.append(f"{threshold_id}: {key} must be relative")

        if has_path and not (ROOT / str(threshold["path"])).is_file():
            failures.append(f"{threshold_id}: path does not exist: {threshold['path']}")
        if has_glob and not glob.glob(str(ROOT / str(threshold["path_glob"])), recursive=True):
            failures.append(f"{threshold_id}: path_glob matched no files: {threshold['path_glob']}")

        try:
            minimum = float(threshold["min_percent"])
        except (TypeError, ValueError):
            failures.append(f"{threshold_id}: min_percent must be numeric")
        else:
            if not 0.0 <= minimum <= 100.0:
                failures.append(f"{threshold_id}: min_percent must be between 0 and 100")

        if not str(threshold["owner"]).strip():
            failures.append(f"{threshold_id}: owner must be non-empty")
        if not str(threshold["rationale"]).strip():
            failures.append(f"{threshold_id}: rationale must be non-empty")

    return failures


def _load_python_coverage(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    coverage: dict[str, float] = {}
    for file_path, record in data.get("files", {}).items():
        summary = record.get("summary", {})
        percent = summary.get("percent_covered")
        if percent is None:
            percent = summary.get("percent_covered_display")
        try:
            coverage[_normalize_path(file_path)] = float(percent)
        except (TypeError, ValueError):
            continue
    return coverage


def _load_lcov(path: Path) -> dict[str, tuple[int, int]]:
    coverage: dict[str, tuple[int, int]] = {}
    current_file: str | None = None
    found_lines = 0
    hit_lines = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("SF:"):
            current_file = _normalize_path(line[3:])
            found_lines = 0
            hit_lines = 0
        elif line.startswith("DA:"):
            parts = line[3:].split(",", 1)
            if len(parts) == 2:
                found_lines += 1
                try:
                    if int(parts[1]) > 0:
                        hit_lines += 1
                except ValueError:
                    pass
        elif line == "end_of_record" and current_file is not None:
            prior_hit, prior_found = coverage.get(current_file, (0, 0))
            coverage[current_file] = (prior_hit + hit_lines, prior_found + found_lines)
            current_file = None
    return coverage


def _matching_python_percent(
    threshold: dict, coverage: dict[str, float]
) -> tuple[float | None, list[str]]:
    if "path" in threshold:
        wanted = _normalize_path(threshold["path"])
        for file_path, percent in coverage.items():
            if file_path == wanted or file_path.endswith(wanted):
                return percent, [file_path]
        return None, []

    pattern = _normalize_path(threshold["path_glob"])
    matches = [
        (path, percent) for path, percent in coverage.items() if fnmatch.fnmatch(path, pattern)
    ]
    if not matches:
        return None, []
    return min(percent for _, percent in matches), [path for path, _ in matches]


def _matching_lcov_percent(
    threshold: dict, coverage: dict[str, tuple[int, int]]
) -> tuple[float | None, list[str]]:
    if "path" in threshold:
        pattern = _normalize_path(threshold["path"])
        matches = [
            (path, totals)
            for path, totals in coverage.items()
            if path == pattern or path.endswith(pattern)
        ]
    else:
        pattern = _normalize_path(threshold["path_glob"])
        matches = [
            (path, totals) for path, totals in coverage.items() if fnmatch.fnmatch(path, pattern)
        ]

    found = sum(found for _, (_, found) in matches)
    hit = sum(hit for _, (hit, _) in matches)
    if found == 0:
        return None, [path for path, _ in matches]
    return hit / found * 100.0, [path for path, _ in matches]


def _evaluate_thresholds(
    spec: dict,
    python_coverage: dict[str, float] | None,
    rust_coverage: dict[str, tuple[int, int]] | None,
) -> tuple[list[dict], list[str]]:
    results: list[dict] = []
    failures: list[str] = []
    for threshold in spec.get("thresholds", []):
        coverage_type = threshold["coverage_type"]
        if coverage_type == "python-json":
            if python_coverage is None:
                status = "not-evaluated"
                observed = None
                matched_paths: list[str] = []
            else:
                observed, matched_paths = _matching_python_percent(threshold, python_coverage)
                status = (
                    "passed"
                    if observed is not None and observed >= float(threshold["min_percent"])
                    else "failed"
                )
        else:
            if rust_coverage is None:
                status = "not-evaluated"
                observed = None
                matched_paths = []
            else:
                observed, matched_paths = _matching_lcov_percent(threshold, rust_coverage)
                status = (
                    "passed"
                    if observed is not None and observed >= float(threshold["min_percent"])
                    else "failed"
                )

        result = {
            "id": threshold["id"],
            "coverage_type": coverage_type,
            "minimum_percent": float(threshold["min_percent"]),
            "observed_percent": None if observed is None else round(observed, 6),
            "matched_paths": matched_paths,
            "status": status,
        }
        results.append(result)
        if status == "failed":
            failures.append(
                f"{threshold['id']}: observed "
                f"{'missing' if observed is None else f'{observed:.2f}%'} below "
                f"{float(threshold['min_percent']):.2f}%"
            )
    return results, failures


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=THRESHOLD_FILE)
    parser.add_argument("--python-json", type=Path)
    parser.add_argument("--rust-lcov", type=Path)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--report-json-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    spec = _load_thresholds(args.config)
    failures = _validate_config(spec)
    if failures:
        print("Module coverage config failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    if args.check_config and not args.python_json and not args.rust_lcov:
        print(f"Module coverage config passed for {len(spec.get('thresholds', []))} thresholds.")
        return 0

    python_coverage = _load_python_coverage(args.python_json) if args.python_json else None
    rust_coverage = _load_lcov(args.rust_lcov) if args.rust_lcov else None
    results, eval_failures = _evaluate_thresholds(spec, python_coverage, rust_coverage)

    report = {
        "schema_version": 1,
        "threshold_count": len(results),
        "results": results,
        "valid": not eval_failures,
    }
    if args.report_json_output:
        args.report_json_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )

    if eval_failures:
        print("Module coverage check failed:")
        for failure in eval_failures:
            print(f" - {failure}")
        return 1

    evaluated = sum(1 for result in results if result["status"] != "not-evaluated")
    print(f"Module coverage check passed for {evaluated} evaluated thresholds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
