#!/usr/bin/env python3
"""Validate the high-assurance requirement-to-test evidence map."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACEABILITY_FILE = ROOT / "specs" / "high_assurance_traceability.json"
REQ_ID_RE = re.compile(r"^[A-Z]+(?:-[A-Z0-9]+)+$")


def _load_traceability() -> dict:
    with TRACEABILITY_FILE.open(encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    spec = _load_traceability()
    failures: list[str] = []

    if spec.get("schema_version") != 1:
        failures.append("schema_version must be 1")

    seen_ids: set[str] = set()
    requirements = spec.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        failures.append("requirements must be a non-empty list")
        requirements = []

    for req in requirements:
        req_id = req.get("id")
        if not isinstance(req_id, str) or not REQ_ID_RE.match(req_id):
            failures.append(f"invalid requirement id: {req_id!r}")
            continue
        if req_id in seen_ids:
            failures.append(f"duplicate requirement id: {req_id}")
        seen_ids.add(req_id)

        evidence = req.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            failures.append(f"{req_id}: evidence must be a non-empty list")
            continue

        for item in evidence:
            rel_path = item.get("path")
            if not isinstance(rel_path, str):
                failures.append(f"{req_id}: evidence path must be a string")
                continue
            path = ROOT / rel_path
            if not path.is_file():
                failures.append(f"{req_id}: missing evidence file {rel_path}")
                continue

            text = path.read_text(encoding="utf-8")
            for needle in item.get("must_contain", []):
                if needle not in text:
                    failures.append(f"{req_id}: {rel_path} does not contain {needle!r}")

    if failures:
        print("Traceability check failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print(f"Traceability check passed for {len(seen_ids)} requirements.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
