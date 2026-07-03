#!/usr/bin/env python3
"""Validate full high-assurance mutation campaign configuration."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_FILE = ROOT / "specs" / "mutation_campaigns.json"
ID_RE = re.compile(r"^[A-Z]+(?:-[A-Z0-9]+)+$")
LANGUAGES = {"rust", "python"}
REQUIRED_FIELDS = {
    "id",
    "language",
    "tool",
    "target_paths",
    "minimum_score",
    "timeout_seconds",
    "ci_cadence",
    "owner",
    "rationale",
    "artifact_dir",
    "command",
}


def load_campaigns(path: Path = CAMPAIGN_FILE) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_campaigns(spec: dict) -> list[str]:
    failures: list[str] = []
    if spec.get("schema_version") != 1:
        failures.append("schema_version must be 1")

    campaigns = spec.get("campaigns")
    if not isinstance(campaigns, list) or not campaigns:
        failures.append("campaigns must be a non-empty list")
        campaigns = []

    seen: set[str] = set()
    for campaign in campaigns:
        missing = sorted(REQUIRED_FIELDS - set(campaign))
        campaign_id = campaign.get("id", "<missing id>")
        if missing:
            failures.append(f"{campaign_id}: missing {missing}")
            continue

        if not isinstance(campaign_id, str) or not ID_RE.match(campaign_id):
            failures.append(f"{campaign_id}: invalid campaign id")
        if campaign_id in seen:
            failures.append(f"{campaign_id}: duplicate campaign id")
        seen.add(str(campaign_id))

        if campaign["language"] not in LANGUAGES:
            failures.append(f"{campaign_id}: language must be one of {sorted(LANGUAGES)}")
        if not str(campaign["tool"]).strip():
            failures.append(f"{campaign_id}: tool must be non-empty")
        if not str(campaign["owner"]).strip():
            failures.append(f"{campaign_id}: owner must be non-empty")
        if not str(campaign["rationale"]).strip():
            failures.append(f"{campaign_id}: rationale must be non-empty")

        target_paths = campaign["target_paths"]
        if not isinstance(target_paths, list) or not target_paths:
            failures.append(f"{campaign_id}: target_paths must be a non-empty list")
        else:
            for rel_path in target_paths:
                if not isinstance(rel_path, str) or not rel_path:
                    failures.append(f"{campaign_id}: target path must be a non-empty string")
                    continue
                if not (ROOT / rel_path).exists():
                    failures.append(f"{campaign_id}: target path does not exist: {rel_path}")

        try:
            minimum_score = float(campaign["minimum_score"])
        except (TypeError, ValueError):
            failures.append(f"{campaign_id}: minimum_score must be numeric")
        else:
            if not 0.0 <= minimum_score <= 100.0:
                failures.append(f"{campaign_id}: minimum_score must be between 0 and 100")

        try:
            timeout_seconds = int(campaign["timeout_seconds"])
        except (TypeError, ValueError):
            failures.append(f"{campaign_id}: timeout_seconds must be an integer")
        else:
            if timeout_seconds <= 0:
                failures.append(f"{campaign_id}: timeout_seconds must be positive")

        artifact_dir = campaign["artifact_dir"]
        if not isinstance(artifact_dir, str) or not artifact_dir:
            failures.append(f"{campaign_id}: artifact_dir must be non-empty")
        elif Path(artifact_dir).is_absolute():
            failures.append(f"{campaign_id}: artifact_dir must be relative")

        command = campaign["command"]
        if not isinstance(command, list) or not command:
            failures.append(f"{campaign_id}: command must be a non-empty list")
        elif not all(isinstance(part, str) and part for part in command):
            failures.append(f"{campaign_id}: command elements must be non-empty strings")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CAMPAIGN_FILE)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    spec = load_campaigns(args.config)
    failures = validate_campaigns(spec)

    if args.json_output:
        report = {
            "schema_version": 1,
            "campaign_count": len(spec.get("campaigns", [])),
            "campaign_ids": [campaign.get("id") for campaign in spec.get("campaigns", [])],
            "valid": not failures,
            "failures": failures,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if failures:
        print("Mutation campaign config failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print(f"Mutation campaign config passed for {len(spec.get('campaigns', []))} campaigns.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
