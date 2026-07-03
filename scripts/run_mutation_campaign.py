#!/usr/bin/env python3
"""Run or dry-run configured high-assurance mutation campaigns."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from check_mutation_campaigns import CAMPAIGN_FILE, load_campaigns, validate_campaigns

ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CAMPAIGN_FILE)
    parser.add_argument("--campaign", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def _select_campaigns(spec: dict, requested: list[str], run_all: bool) -> list[dict]:
    campaigns = list(spec.get("campaigns", []))
    if run_all:
        return campaigns
    if not requested:
        raise SystemExit("--campaign or --all is required unless --list is used")

    by_id = {campaign["id"]: campaign for campaign in campaigns}
    missing = sorted(set(requested) - set(by_id))
    if missing:
        raise SystemExit(f"unknown campaign id(s): {', '.join(missing)}")
    return [by_id[campaign_id] for campaign_id in requested]


def _run_campaign(campaign: dict, dry_run: bool) -> dict:
    command = list(campaign["command"])
    result: dict[str, object] = {
        "id": campaign["id"],
        "language": campaign["language"],
        "tool": campaign["tool"],
        "minimum_score": campaign["minimum_score"],
        "timeout_seconds": campaign["timeout_seconds"],
        "artifact_dir": campaign["artifact_dir"],
        "command": command,
        "dry_run": dry_run,
    }

    print(f"{campaign['id']}: {' '.join(command)}")
    if dry_run:
        result.update({"executed": False, "returncode": None, "elapsed_seconds": 0.0})
        return result

    executable = shutil.which(command[0])
    if executable is None:
        result.update(
            {
                "executed": False,
                "returncode": 127,
                "error": f"executable not found: {command[0]}",
            }
        )
        return result

    artifact_dir = ROOT / str(campaign["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=int(campaign["timeout_seconds"]),
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = perf_counter() - started
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        (artifact_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
        (artifact_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
        result.update(
            {
                "executed": True,
                "timed_out": True,
                "returncode": 124,
                "elapsed_seconds": round(elapsed, 6),
                "stdout_path": str((artifact_dir / "stdout.txt").relative_to(ROOT)),
                "stderr_path": str((artifact_dir / "stderr.txt").relative_to(ROOT)),
            }
        )
        return result

    elapsed = perf_counter() - started
    stdout_path = artifact_dir / "stdout.txt"
    stderr_path = artifact_dir / "stderr.txt"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    result.update(
        {
            "executed": True,
            "timed_out": False,
            "returncode": completed.returncode,
            "elapsed_seconds": round(elapsed, 6),
            "stdout_path": str(stdout_path.relative_to(ROOT)),
            "stderr_path": str(stderr_path.relative_to(ROOT)),
        }
    )
    return result


def main() -> int:
    args = _parse_args()
    spec = load_campaigns(args.config)
    failures = validate_campaigns(spec)
    if failures:
        print("Mutation campaign config failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    if args.list:
        for campaign in spec.get("campaigns", []):
            print(
                f"{campaign['id']} "
                f"language={campaign['language']} "
                f"tool={campaign['tool']} "
                f"minimum_score={float(campaign['minimum_score']):.1f}"
            )
        return 0

    selected = _select_campaigns(spec, args.campaign, args.all)
    results = [_run_campaign(campaign, args.dry_run) for campaign in selected]
    report = {
        "schema_version": 1,
        "dry_run": args.dry_run,
        "campaign_count": len(results),
        "results": results,
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    failed = [
        result
        for result in results
        if result.get("returncode") not in (0, None) or result.get("timed_out")
    ]
    if failed:
        print("Mutation campaign run failed:")
        for result in failed:
            print(f" - {result['id']}: returncode={result.get('returncode')}")
        return 1

    mode = "dry-run" if args.dry_run else "run"
    print(f"Mutation campaign {mode} passed for {len(results)} campaign(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
