#!/usr/bin/env python3
"""Check the version the code declares, for CI and the Release workflow.

The version is set in a pull request, in three files that must agree; CI runs this
with no arguments to hold them together. The Release workflow also passes PyPI's JSON,
which refuses a version PyPI already has or one not newer than every release there, so
an unbumped ``main`` cannot be released twice. On success it writes the version to the
GitHub output file so later jobs can name and tag the release.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import tomllib

_VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
_VERSION_FILES = {
    "pyproject.toml": "project",
    "crates/rustystats/Cargo.toml": "package",
    "crates/rustystats-core/Cargo.toml": "package",
}
_BUMP_HINT = (
    "Set a new version in pyproject.toml and both crates' Cargo.toml in a pull request, "
    "merge it, then run Release again."
)


class ReleaseVersionError(Exception):
    """The declared version cannot be released."""


def _parts(version: str) -> tuple[int, int, int] | None:
    match = _VERSION.fullmatch(version)
    if match is None:
        return None
    return int(match[1]), int(match[2]), int(match[3])


def declared_versions(root: Path) -> dict[str, str]:
    """The version each file under *root* declares, by file."""
    declared = {}
    for name, table in _VERSION_FILES.items():
        with (root / name).open("rb") as stream:
            declared[name] = str(tomllib.load(stream)[table]["version"])
    return declared


def agreed_version(declared: Mapping[str, str]) -> str:
    """The one ``X.Y.Z`` version every file declares, else raise."""
    versions = set(declared.values())
    if len(versions) != 1:
        found = ", ".join(f"{name} has {version}" for name, version in declared.items())
        raise ReleaseVersionError(f"The version files disagree: {found}.")
    version = versions.pop()
    if _parts(version) is None:
        raise ReleaseVersionError(f"The code declares version {version!r}; a release is X.Y.Z.")
    return version


def released_versions(pypi_json: Path) -> list[str]:
    """The versions PyPI's JSON API lists."""
    document = json.loads(pypi_json.read_text(encoding="utf-8"))
    return sorted(document["releases"])


def check_new_release(version: str, released: Iterable[str]) -> None:
    """Raise unless *version* is newer than every ``X.Y.Z`` release on PyPI.

    Releases PyPI has in another form (a pre-release, say) cannot collide with an
    ``X.Y.Z`` version and are not compared.
    """
    published = set(released)
    if version in published:
        raise ReleaseVersionError(f"rustystats {version} is already on PyPI. {_BUMP_HINT}")
    comparable = {parsed: text for text in published if (parsed := _parts(text)) is not None}
    if comparable and _parts(version) <= max(comparable):
        latest = comparable[max(comparable)]
        raise ReleaseVersionError(
            f"rustystats {version} is not newer than {latest}, the latest on PyPI. {_BUMP_HINT}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="the repository root")
    parser.add_argument(
        "--pypi-json",
        type=Path,
        help="PyPI's JSON API response for the project; checks the version is a new release",
    )
    parser.add_argument(
        "--github-output",
        type=Path,
        help="file the version is appended to as `version=X.Y.Z`",
    )
    args = parser.parse_args(argv)
    try:
        version = agreed_version(declared_versions(args.root))
        if args.pypi_json is not None:
            check_new_release(version, released_versions(args.pypi_json))
    except ReleaseVersionError as exc:
        print(f"::error::{exc}")
        return 1
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as stream:
            stream.write(f"version={version}\n")
    print(f"The code declares rustystats {version}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
