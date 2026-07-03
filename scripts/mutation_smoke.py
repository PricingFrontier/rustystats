#!/usr/bin/env python3
"""Run a deterministic mutation smoke test against validation guardrails.

This is intentionally tiny. Full mutation testing belongs in scheduled tooling,
but this smoke proves the test suite kills at least one critical validation
mutant and that the source file is restored afterward.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "python" / "rustystats" / "validation.py"
ORIGINAL_SNIPPET = "if np.sum(weights) == 0:"
MUTATED_SNIPPET = "if np.sum(weights) != 0:"
TEST_COMMAND = [
    sys.executable,
    "-m",
    "pytest",
    "tests/python/test_validation_contracts.py::test_weight_zero_mass_contracts",
    "-q",
]


def _run_target_test() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        TEST_COMMAND,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _print_output(label: str, result: subprocess.CompletedProcess[str]) -> None:
    print(f"\n[{label}] exit={result.returncode}")
    print(result.stdout.strip())


def main() -> int:
    original = TARGET.read_text(encoding="utf-8")
    if ORIGINAL_SNIPPET not in original:
        print(f"Could not find mutation target snippet in {TARGET.relative_to(ROOT)}")
        return 2

    preflight = _run_target_test()
    if preflight.returncode != 0:
        _print_output("preflight failed", preflight)
        return preflight.returncode

    try:
        TARGET.write_text(original.replace(ORIGINAL_SNIPPET, MUTATED_SNIPPET, 1), encoding="utf-8")
        mutant = _run_target_test()
    finally:
        TARGET.write_text(original, encoding="utf-8")

    if mutant.returncode == 0:
        _print_output("mutant survived", mutant)
        print("Known validation mutant survived; strengthen test_weight_zero_mass_contracts.")
        return 1

    postflight = _run_target_test()
    if postflight.returncode != 0:
        _print_output("postflight failed after restore", postflight)
        return postflight.returncode

    print("Mutation smoke passed: known validation mutant was killed and source was restored.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
