#!/usr/bin/env python3
"""Fail test:/ci:-titled PRs that modify non-test source.

Release-hygiene gate (monotone-solver incident, bug.md Fix 4): a commit titled
"test: expand high assurance coverage" shipped ~1,600 changed non-test solver
lines and regressed the monotone smooth solver in v0.8.14. A PR whose title
declares test/CI-only scope must not touch production source; if it needs to,
retitle it (fix:/refactor:/feat:) so reviewers read it as a behavior change.

Production source is anything under crates/*/src/ or python/rustystats/ (the
shipped package). Tests, docs, CI config, scripts, and fixtures are exempt.
For .rs files the check is hunk-aware: Rust unit tests live in a trailing
`#[cfg(test)] mod` inside src files, so changes confined to that module (in
both the merge-base and HEAD versions) are test-scope, not production-scope.

Usage:
    check_pr_title_scope.py --title "<pr title>" --base <git ref>
"""

from __future__ import annotations

import argparse
import re
import subprocess

SCOPED_TITLE = re.compile(r"^(test|tests|ci)(\([^)]*\))?!?:", re.IGNORECASE)

PRODUCTION_SOURCE = re.compile(r"^(crates/[^/]+/src/|python/rustystats/)")
EXEMPT = re.compile(
    r"^(python/rustystats/_rustystats\.pyi$)"  # type stub tracks source mechanically
)
HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True)


def merge_base(base: str) -> str:
    out = _git("merge-base", base, "HEAD")
    if out.returncode != 0:
        raise SystemExit(f"cannot resolve merge-base of {base} and HEAD: {out.stderr.strip()}")
    return out.stdout.strip()


def changed_files(merge_base_sha: str) -> list[str]:
    out = _git("diff", "--name-only", f"{merge_base_sha}..HEAD")
    if out.returncode != 0:
        raise SystemExit(f"git diff failed: {out.stderr.strip()}")
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def _test_module_start(rev: str, path: str) -> int | None:
    """Line of the last `#[cfg(test)]` immediately followed by a `mod` line."""
    show = _git("show", f"{rev}:{path}")
    if show.returncode != 0:
        return None  # absent at this rev (added/deleted file)
    lines = show.stdout.splitlines()
    start = None
    for lineno, line in enumerate(lines, start=1):
        if line.startswith("#[cfg(test)]") and lineno < len(lines):
            follower = lines[lineno].lstrip()
            if follower.startswith("mod ") or follower.startswith("pub mod "):
                start = lineno
    return start


def rust_changes_confined_to_test_module(path: str, merge_base_sha: str) -> bool:
    """True when every changed line of a .rs file falls inside the trailing
    test module in BOTH the merge-base and HEAD versions."""
    base_start = _test_module_start(merge_base_sha, path)
    head_start = _test_module_start("HEAD", path)
    if base_start is None or head_start is None:
        return False
    diff = _git("diff", "-U0", f"{merge_base_sha}..HEAD", "--", path)
    if diff.returncode != 0:
        return False
    for match in HUNK_HEADER.finditer(diff.stdout):
        old_line = int(match.group(1))
        new_line = int(match.group(3))
        if old_line < base_start or new_line < head_start:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--base", default="origin/main")
    args = parser.parse_args()

    if not SCOPED_TITLE.match(args.title.strip()):
        print(f"PR title {args.title!r} does not declare test/ci scope; check skipped.")
        return 0

    merge_base_sha = merge_base(args.base)
    violations = [
        path
        for path in changed_files(merge_base_sha)
        if PRODUCTION_SOURCE.match(path)
        and not EXEMPT.match(path)
        and not (
            path.endswith(".rs") and rust_changes_confined_to_test_module(path, merge_base_sha)
        )
    ]
    if violations:
        print(f"PR title {args.title!r} declares test/CI-only scope, but the diff")
        print("modifies production source:")
        for path in violations:
            print(f" - {path}")
        print()
        print("Either move these changes to a separate PR, or retitle this PR")
        print("(fix:/refactor:/feat:) so the source changes are reviewed as such.")
        return 1

    print(f"PR title scope check passed ({args.title!r}; no production source touched).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
