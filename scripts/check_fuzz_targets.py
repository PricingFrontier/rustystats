#!/usr/bin/env python3
"""Validate Rust cargo-fuzz target scaffolding."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FUZZ_MANIFEST = ROOT / "fuzz" / "Cargo.toml"
TARGET_RE = re.compile(
    r"\[\[bin\]\]\s+name\s*=\s*\"(?P<name>[^\"]+)\"\s+path\s*=\s*\"(?P<path>[^\"]+)\"",
    re.MULTILINE,
)


def _extract_targets(manifest_text: str) -> list[tuple[str, str]]:
    return [
        (match.group("name"), match.group("path")) for match in TARGET_RE.finditer(manifest_text)
    ]


def main() -> int:
    failures: list[str] = []
    report: dict[str, object] = {"schema_version": 1, "targets": []}

    if not FUZZ_MANIFEST.is_file():
        failures.append("missing fuzz/Cargo.toml")
        manifest_text = ""
    else:
        manifest_text = FUZZ_MANIFEST.read_text(encoding="utf-8")

    if "cargo-fuzz = true" not in manifest_text:
        failures.append("fuzz/Cargo.toml must declare cargo-fuzz = true")
    if "libfuzzer-sys" not in manifest_text:
        failures.append("fuzz/Cargo.toml must depend on libfuzzer-sys")

    targets = _extract_targets(manifest_text)
    if not targets:
        failures.append("fuzz/Cargo.toml must declare at least one [[bin]] target")

    seen: set[str] = set()
    for name, rel_path in targets:
        if name in seen:
            failures.append(f"duplicate fuzz target name: {name}")
        seen.add(name)
        path = ROOT / "fuzz" / rel_path
        target_report = {"name": name, "path": f"fuzz/{rel_path}"}
        report["targets"].append(target_report)
        if not path.is_file():
            failures.append(f"{name}: target file missing: fuzz/{rel_path}")
            continue
        text = path.read_text(encoding="utf-8")
        if "fuzz_target!" not in text:
            failures.append(f"{name}: target file must contain fuzz_target!")
        if "#![no_main]" not in text:
            failures.append(f"{name}: target file must use #![no_main]")
        if "rustystats_core" not in text:
            failures.append(f"{name}: target file must exercise rustystats_core")

    report["target_count"] = len(targets)
    report["valid"] = not failures
    report["failures"] = failures
    output = ROOT / "target" / "fuzz-target-check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if failures:
        print("Rust fuzz target check failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print(f"Rust fuzz target check passed for {len(targets)} targets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
