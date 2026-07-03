#!/usr/bin/env python3
"""Validate the external oracle archive manifest."""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "oracles" / "archive_manifest.json"
ID_RE = re.compile(r"^[A-Z]+(?:-[A-Z0-9]+)+$")
STATUSES = {"implemented", "planned", "external-blocked"}
REQUIRED_FIELDS = {
    "id",
    "area",
    "oracle",
    "status",
    "evidence_paths",
    "owner",
    "rationale",
    "regeneration_command",
    "next_action",
    "expires",
}


def main() -> int:
    failures: list[str] = []
    if not MANIFEST.is_file():
        print(f"Oracle archive manifest check failed:\n - missing {MANIFEST.relative_to(ROOT)}")
        return 1

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        failures.append("schema_version must be 1")

    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        failures.append("entries must be a non-empty list")
        entries = []

    seen: set[str] = set()
    today = date.today()
    implemented = 0
    planned = 0
    for entry in entries:
        entry_id = entry.get("id", "<missing id>")
        missing = sorted(REQUIRED_FIELDS - set(entry))
        if missing:
            failures.append(f"{entry_id}: missing {missing}")
            continue

        if not isinstance(entry_id, str) or not ID_RE.match(entry_id):
            failures.append(f"{entry_id}: invalid archive id")
        if entry_id in seen:
            failures.append(f"{entry_id}: duplicate archive id")
        seen.add(str(entry_id))

        status = entry["status"]
        if status not in STATUSES:
            failures.append(f"{entry_id}: status must be one of {sorted(STATUSES)}")
        elif status == "implemented":
            implemented += 1
        else:
            planned += 1

        for field in REQUIRED_FIELDS - {"evidence_paths", "expires"}:
            if not isinstance(entry[field], str) or not entry[field].strip():
                failures.append(f"{entry_id}: {field} must be a non-empty string")

        evidence_paths = entry["evidence_paths"]
        if not isinstance(evidence_paths, list) or not evidence_paths:
            failures.append(f"{entry_id}: evidence_paths must be a non-empty list")
        else:
            for rel_path in evidence_paths:
                if not isinstance(rel_path, str) or not rel_path:
                    failures.append(f"{entry_id}: evidence path must be a non-empty string")
                    continue
                if not (ROOT / rel_path).is_file():
                    failures.append(f"{entry_id}: missing evidence path {rel_path}")

        try:
            expiry = date.fromisoformat(entry["expires"])
        except (TypeError, ValueError):
            failures.append(f"{entry_id}: expires must be YYYY-MM-DD")
        else:
            if expiry < today:
                failures.append(f"{entry_id}: expired on {expiry.isoformat()}")

        if (
            status != "implemented"
            and "Add archived" not in entry["next_action"]
            and "Provision" not in entry["next_action"]
        ):
            failures.append(
                f"{entry_id}: planned/external entries must name the concrete next action"
            )

    if implemented == 0:
        failures.append("at least one implemented oracle archive entry is required")
    if planned == 0:
        failures.append(
            "at least one planned or external-blocked entry is required until oracle coverage is complete"
        )

    if failures:
        print("Oracle archive manifest check failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print(
        "Oracle archive manifest check passed "
        f"for {len(entries)} entries ({implemented} implemented, {planned} planned/external)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
