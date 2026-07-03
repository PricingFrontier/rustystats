#!/usr/bin/env python3
"""Validate high-assurance dependency hygiene for the checked-in lockfile."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_FILE = ROOT / "uv.lock"
KNOWN_YANKED = {
    ("numpy", "2.4.0"): "Yanked for a backwards-compatibility bug.",
}
PACKAGE_FIELD_RE = re.compile(r'^(name|version)\s*=\s*"([^"]+)"$', re.MULTILINE)


def main() -> int:
    lock_text = LOCK_FILE.read_text(encoding="utf-8")
    failures: list[str] = []

    for block in lock_text.split("[[package]]")[1:]:
        fields = dict(PACKAGE_FIELD_RE.findall(block))
        name = fields.get("name")
        version = fields.get("version")
        if "yanked = true" in block:
            failures.append(f"{name}=={version} is marked yanked in uv.lock")

        reason = KNOWN_YANKED.get((name, version))
        if reason is not None:
            failures.append(f"{name}=={version} is blocked: {reason}")

    if failures:
        print("Dependency hygiene check failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("Dependency hygiene check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
