#!/usr/bin/env python3
"""Fail test:/ci:-titled PRs that modify non-test source.

Release-hygiene gate (monotone-solver incident, bug.md Fix 4): a commit titled
"test: expand high assurance coverage" shipped ~1,600 changed non-test solver
lines and regressed the monotone smooth solver in v0.8.14. A PR whose title
declares test/CI-only scope must not touch production source; if it needs to,
retitle it (fix:/refactor:/feat:) so reviewers read it as a behavior change.

File-level enforcement: production source is anything under crates/*/src/ or
python/rustystats/ (the shipped package). Tests, docs, CI config, scripts, and
fixtures are exempt.

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


def changed_files(base: str) -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--base", default="origin/main")
    args = parser.parse_args()

    if not SCOPED_TITLE.match(args.title.strip()):
        print(f"PR title {args.title!r} does not declare test/ci scope; check skipped.")
        return 0

    violations = [
        path
        for path in changed_files(args.base)
        if PRODUCTION_SOURCE.match(path) and not EXEMPT.match(path)
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
