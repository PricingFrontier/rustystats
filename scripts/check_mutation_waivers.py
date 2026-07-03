#!/usr/bin/env python3
"""Validate high-assurance mutation-survivor waivers."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WAIVER_FILE = ROOT / "specs" / "high_assurance_mutation_waivers.json"
TARGET_FILE = ROOT / "specs" / "high_assurance_mutation_targets.json"
ID_RE = re.compile(r"^[A-Z]+(?:-[A-Z0-9]+)+$")
CLASSIFICATIONS = {"equivalent", "accepted-risk", "test-gap"}
REQUIRED_FIELDS = {
    "id",
    "tool",
    "module",
    "mutant",
    "classification",
    "reason",
    "owner",
    "expires",
    "remediation",
}


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"missing required file: {path.relative_to(ROOT)}")


def _validate_targets(failures: list[str]) -> set[str]:
    data = _load_json(TARGET_FILE)
    if data.get("schema_version") != 1:
        failures.append("mutation target schema_version must be 1")

    seen: set[str] = set()
    for target in data.get("targets", []):
        target_id = target.get("id")
        if not isinstance(target_id, str) or not ID_RE.match(target_id):
            failures.append(f"invalid mutation target id: {target_id!r}")
            continue
        if target_id in seen:
            failures.append(f"{target_id}: duplicate mutation target id")
        seen.add(target_id)

        for field in ("language", "module", "path", "original", "mutated", "test_command"):
            if field not in target:
                failures.append(f"{target_id}: missing {field}")
        if target.get("language") not in {"python", "rust"}:
            failures.append(f"{target_id}: language must be 'python' or 'rust'")
        rel_path = target.get("path")
        if isinstance(rel_path, str) and not (ROOT / rel_path).is_file():
            failures.append(f"{target_id}: target path does not exist: {rel_path}")
        command = target.get("test_command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(x, str) for x in command)
        ):
            failures.append(f"{target_id}: test_command must be a non-empty string list")
    return seen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-test-gap", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    target_ids = _validate_targets(failures)
    data = _load_json(WAIVER_FILE)
    if data.get("schema_version") != 1:
        failures.append("mutation waiver schema_version must be 1")

    seen: set[str] = set()
    today = date.today()
    for waiver in data.get("waivers", []):
        missing = sorted(REQUIRED_FIELDS - set(waiver))
        waiver_id = waiver.get("id", "<missing id>")
        if missing:
            failures.append(f"{waiver_id}: missing {missing}")
            continue

        if not isinstance(waiver_id, str) or not ID_RE.match(waiver_id):
            failures.append(f"{waiver_id}: invalid waiver id format")
        if waiver_id in seen:
            failures.append(f"{waiver_id}: duplicate waiver id")
        seen.add(waiver_id)

        for field in REQUIRED_FIELDS - {"expires"}:
            if not str(waiver[field]).strip():
                failures.append(f"{waiver_id}: {field} must be non-empty")

        if waiver["classification"] not in CLASSIFICATIONS:
            failures.append(f"{waiver_id}: classification must be one of {sorted(CLASSIFICATIONS)}")
        if waiver["classification"] == "test-gap" and not args.allow_test_gap:
            failures.append(f"{waiver_id}: test-gap mutation waivers block release")

        try:
            expiry = date.fromisoformat(str(waiver["expires"]))
        except ValueError:
            failures.append(f"{waiver_id}: expires must be YYYY-MM-DD")
        else:
            if expiry < today:
                failures.append(f"{waiver_id}: expired on {expiry.isoformat()}")

        mutant = str(waiver["mutant"])
        if mutant.startswith("MUT-") and mutant not in target_ids:
            failures.append(f"{waiver_id}: references unknown mutation target {mutant}")

    if failures:
        print("Mutation waiver check failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print(
        f"Mutation waiver check passed for {len(seen)} waivers "
        f"and {len(target_ids)} mutation targets."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
