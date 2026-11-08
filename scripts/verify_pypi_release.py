#!/usr/bin/env python3
"""Check that PyPI serves exactly the release files the Release run built.

The Release workflow uploads with already-present files skipped, so re-running a
failed publish completes a partial upload instead of failing on the file that
made it. The release counts as published, and is tagged, only once PyPI's JSON
for the version lists every built file with the same SHA-256 and nothing else.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


def built_digests(dist: Path) -> dict[str, str]:
    """SHA-256 of every file the run built, by file name."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(dist.iterdir())
        if path.is_file()
    }


def published_digests(release_json: Path) -> dict[str, str]:
    """SHA-256 of every file PyPI lists for one version, by file name."""
    document = json.loads(release_json.read_text(encoding="utf-8"))
    return {entry["filename"]: entry["digests"]["sha256"] for entry in document["urls"]}


def release_problems(built: Mapping[str, str], published: Mapping[str, str]) -> list[str]:
    """Every way PyPI's files differ from the built ones; empty when they match."""
    problems = [f"{name} is not on PyPI" for name in sorted(built.keys() - published.keys())]
    problems += [
        f"{name} on PyPI has SHA-256 {published[name]}, not the built {built[name]}"
        for name in sorted(built.keys() & published.keys())
        if published[name] != built[name]
    ]
    problems += [
        f"{name} is on PyPI but was not built by this run"
        for name in sorted(published.keys() - built.keys())
    ]
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True, help="the files the run built")
    parser.add_argument(
        "--pypi-json",
        type=Path,
        required=True,
        help="PyPI's JSON API response for the released version",
    )
    args = parser.parse_args(argv)
    built = built_digests(args.dist)
    if not built:
        print(f"No built files in {args.dist}.")
        return 1
    problems = release_problems(built, published_digests(args.pypi_json))
    for problem in problems:
        print(problem)
    if problems:
        return 1
    print(f"PyPI serves exactly the {len(built)} built files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
