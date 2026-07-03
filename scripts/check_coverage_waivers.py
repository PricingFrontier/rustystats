#!/usr/bin/env python3
"""Validate high-assurance coverage waivers."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WAIVER_FILE = ROOT / "specs" / "high_assurance_coverage_waivers.json"
ID_RE = re.compile(r"^[A-Z]+(?:-[A-Z0-9]+)+$")
REQUIRED_FIELDS = {
    "id",
    "scope",
    "current",
    "target",
    "metric",
    "current_value",
    "target_value",
    "file_patterns",
    "last_verified",
    "evidence_command",
    "owner",
    "reason",
    "remediation",
    "expires",
}


def _coverage_percent(path: Path) -> float | None:
    """Read coverage.py JSON total percent when supplied."""
    if not path:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("totals") or data.get("summary") or {}
    value = summary.get("percent_covered")
    if value is None:
        value = summary.get("percent_covered_display")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage-json", type=Path)
    parser.add_argument("--report-json-output", type=Path)
    args = parser.parse_args()

    data = json.loads(WAIVER_FILE.read_text(encoding="utf-8"))
    failures: list[str] = []
    report: dict[str, object] = {"schema_version": 1, "waivers": []}
    coverage_percent = _coverage_percent(args.coverage_json) if args.coverage_json else None
    if data.get("schema_version") != 1:
        failures.append("schema_version must be 1")

    seen: set[str] = set()
    today = date.today()
    for waiver in data.get("waivers", []):
        missing = sorted(REQUIRED_FIELDS - set(waiver))
        if missing:
            failures.append(f"{waiver.get('id', '<missing id>')}: missing {missing}")
            continue

        waiver_id = waiver["id"]
        if not ID_RE.match(waiver_id):
            failures.append(f"{waiver_id}: invalid waiver id format")
        if waiver_id in seen:
            failures.append(f"{waiver_id}: duplicate waiver id")
        seen.add(waiver_id)

        for field in REQUIRED_FIELDS - {"expires"}:
            if not str(waiver[field]).strip():
                failures.append(f"{waiver_id}: {field} must be non-empty")

        if not isinstance(waiver["file_patterns"], list) or not waiver["file_patterns"]:
            failures.append(f"{waiver_id}: file_patterns must be a non-empty list")
        if not all(isinstance(pattern, str) and pattern for pattern in waiver["file_patterns"]):
            failures.append(f"{waiver_id}: file_patterns must contain non-empty strings")

        for date_field in ("expires", "last_verified"):
            try:
                date.fromisoformat(waiver[date_field])
            except ValueError:
                failures.append(f"{waiver_id}: {date_field} must be YYYY-MM-DD")

        try:
            expiry = date.fromisoformat(waiver["expires"])
        except ValueError:
            failures.append(f"{waiver_id}: expires must be YYYY-MM-DD")
            continue
        if expiry < today:
            failures.append(f"{waiver_id}: expired on {expiry.isoformat()}")

        remove_when_met = bool(waiver.get("remove_when_met", False))
        metric = waiver.get("metric")
        if remove_when_met and metric == "python-total-percent" and coverage_percent is not None:
            try:
                target = float(waiver["target_value"])
            except (TypeError, ValueError):
                failures.append(f"{waiver_id}: numeric target_value required for {metric}")
            else:
                if coverage_percent >= target:
                    failures.append(
                        f"{waiver_id}: coverage is {coverage_percent:.2f}%, "
                        f"meeting target {target:.2f}%; remove this waiver"
                    )

        report["waivers"].append(
            {
                "id": waiver_id,
                "scope": waiver["scope"],
                "metric": waiver["metric"],
                "expires": waiver["expires"],
                "owner": waiver["owner"],
            }
        )

    if args.report_json_output:
        args.report_json_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )

    if failures:
        print("Coverage waiver check failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print(f"Coverage waiver check passed for {len(seen)} waivers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
