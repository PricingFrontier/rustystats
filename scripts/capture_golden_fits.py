"""Record the golden fit outputs that ``tests/python/test_golden_fits.py`` checks against.

Run from the repository root against the build being recorded:

    uv run --extra dev python scripts/capture_golden_fits.py tests/golden/fits_v0_9_0.json

The record is the reference for ``specs/out-of-core-fit.md``: it is captured once from
v0.9.0, before the design build and solver loops change, and is only re-captured for the
cases a spec decision changes on purpose (``GoldenCase.changes_under``):

    python scripts/capture_golden_fits.py tests/golden/fits_v0_9_0.json \
        --cases target_encoding_main ... --note "DEC-1 keyed TE order"

A re-capture records, per case, the date, the commit (``-dirty`` for a working tree)
and the note in the record's ``recaptured`` table.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "python"))

from _golden_cases import CASES, GOLDEN_SCHEMA, encode, record


def _git_commit() -> str:
    """The commit, marked ``-dirty`` when the record comes from uncommitted changes."""
    completed = subprocess.run(
        ["git", "describe", "--always", "--dirty", "--abbrev=40"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--cases",
        nargs="+",
        help="re-capture only these cases into an existing record (a deliberate change; "
        "give the reason with --note)",
    )
    parser.add_argument(
        "--note",
        default="",
        help="why the cases are re-captured (the decision), recorded with each case",
    )
    args = parser.parse_args()
    if args.cases and not args.note:
        raise SystemExit("a re-capture needs --note naming the decision that changes the cases")

    import numpy as np
    import polars as pl
    import rustystats as rs

    selected = set(args.cases or [])
    unknown = selected - {case.name for case in CASES}
    if unknown:
        raise SystemExit(f"unknown cases: {sorted(unknown)}")
    existing = json.loads(args.output.read_text(encoding="utf-8")) if selected else None
    cases = dict(existing["cases"]) if existing else {}
    for case in CASES:
        if selected and case.name not in selected:
            continue
        fields = record(case)
        cases[case.name] = {name: encode(value) for name, value in fields.items()}
        print(f"recorded {case.name}: {len(fields)} fields")

    if existing:
        recaptured = sorted(set(existing.get("recaptured", {})) | selected)
        provenance = dict(existing.get("recaptured", {}))
        stamp = datetime.date.today().isoformat()
        build = f"rustystats {rs.__version__}, numpy {np.__version__}, polars {pl.__version__}"
        for name in selected:
            provenance[name] = f"{stamp} {_git_commit()} ({build}): {args.note}"
        document = dict(existing)
        document["cases"] = cases
        document["recaptured"] = {name: provenance[name] for name in recaptured}
        args.output.write_text(
            json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"updated {args.output}: {sorted(selected)}")
        return

    document = {
        "schema": GOLDEN_SCHEMA,
        "rustystats_version": rs.__version__,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "polars": pl.__version__,
        "machine": platform.machine(),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
