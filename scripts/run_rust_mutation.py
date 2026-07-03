#!/usr/bin/env python3
"""Run deterministic scored Rust mutation targets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET_FILE = ROOT / "specs" / "high_assurance_mutation_targets.json"
WAIVER_FILE = ROOT / "specs" / "high_assurance_mutation_waivers.json"


def _load_targets(language: str, modules: list[str]) -> list[dict]:
    data = json.loads(TARGET_FILE.read_text(encoding="utf-8"))
    selected = [t for t in data["targets"] if t["language"] == language]
    if modules:
        selected = [t for t in selected if t["module"] in modules or t["path"] in modules]
    return selected


def _load_waived_mutants() -> set[str]:
    data = json.loads(WAIVER_FILE.read_text(encoding="utf-8"))
    return {
        str(w["mutant"])
        for w in data.get("waivers", [])
        if w.get("classification") in {"equivalent", "accepted-risk"}
    }


def _run(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def _apply_mutation(target: dict, timeout: int) -> dict:
    path = ROOT / target["path"]
    original_text = path.read_text(encoding="utf-8")
    original = target["original"]
    mutated = target["mutated"]
    command = list(target["test_command"])
    started = time.time()

    if original not in original_text:
        return {
            "id": target["id"],
            "status": "invalid",
            "reason": "original snippet not found",
            "duration_seconds": 0.0,
        }

    preflight = _run(command, timeout)
    if preflight.returncode != 0:
        return {
            "id": target["id"],
            "status": "invalid",
            "reason": "preflight test failed",
            "output": preflight.stdout,
            "duration_seconds": round(time.time() - started, 3),
        }

    try:
        path.write_text(original_text.replace(original, mutated, 1), encoding="utf-8")
        mutant = _run(command, timeout)
    finally:
        path.write_text(original_text, encoding="utf-8")

    status = "survived" if mutant.returncode == 0 else "killed"
    return {
        "id": target["id"],
        "status": status,
        "module": target["module"],
        "description": target.get("description", ""),
        "test_command": command,
        "duration_seconds": round(time.time() - started, 3),
        "output_tail": "\n".join(mutant.stdout.splitlines()[-60:]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", action="append", default=[])
    parser.add_argument("--minimum-score", type=float, default=90.0)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    targets = _load_targets("rust", args.module)
    waived = _load_waived_mutants()
    results = [_apply_mutation(target, args.timeout) for target in targets]
    scored = [r for r in results if r["status"] in {"killed", "survived"} and r["id"] not in waived]
    killed = sum(1 for r in scored if r["status"] == "killed")
    score = 100.0 if not scored else 100.0 * killed / len(scored)
    report = {
        "schema_version": 1,
        "language": "rust",
        "minimum_score": args.minimum_score,
        "score": score,
        "killed": killed,
        "scored": len(scored),
        "waived": sorted(waived),
        "results": results,
    }

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Rust mutation score: {score:.2f}% ({killed}/{len(scored)} killed)")
    for result in results:
        print(f" - {result['id']}: {result['status']}")

    invalid = [r for r in results if r["status"] == "invalid"]
    survivors = [r for r in scored if r["status"] == "survived"]
    if invalid:
        print("Invalid mutation targets:")
        for result in invalid:
            print(f" - {result['id']}: {result.get('reason', 'unknown')}")
        return 2
    if survivors or score < args.minimum_score:
        print("Mutation threshold failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
