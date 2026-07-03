# RustyStats Release Evidence

Release:

Date:

Reviewer:

## Required Gates

| Gate | Command or CI job | Result | Evidence link |
| --- | --- | --- | --- |
| Rust debug tests | `cargo test --workspace` |  |  |
| Rust release tests | `cargo test --workspace --release` |  |  |
| Python matrix | GitHub Actions `python` matrix |  |  |
| Python coverage | `uv run --extra dev pytest tests/python --cov=rustystats --cov-report=term-missing:skip-covered --cov-fail-under=97` |  |  |
| Rust coverage | `cargo llvm-cov --workspace --summary-only` |  |  |
| Diagnostics harness | `uv run --extra dev python benchmarks/verify_diagnostics_correctness.py` |  |  |
| Deployment parity | `uv run --extra dev pytest tests/python/test_export.py tests/python/test_rate_tables_parity.py -q` |  |  |
| Fuzz corpus replay | `uv run --extra dev python scripts/replay_fuzz_corpus.py` |  |  |
| Rust fuzz target validation | `python3 scripts/check_fuzz_targets.py && cargo check --manifest-path fuzz/Cargo.toml` |  |  |
| Mutation smoke | `uv run --extra dev python scripts/mutation_smoke.py` |  |  |
| Scored mutation | deep-assurance CI or `uv run --extra dev python scripts/run_python_mutation.py --minimum-score 100 && uv run --extra dev python scripts/run_rust_mutation.py --minimum-score 100` |  |  |
| Full mutation campaigns | deep-assurance CI dry-run or `python3 scripts/run_mutation_campaign.py --all --dry-run`; manual full run with `python3 scripts/run_mutation_campaign.py --all` |  |  |
| External oracle suite | deep-assurance CI or `uv run --extra dev python scripts/check_oracle_fixtures.py && uv run --extra dev python scripts/generate_oracle_fixtures.py --check && uv run --extra dev pytest tests/python/test_oracle_fixtures.py -q` |  |  |
| Oracle archive manifest | `python3 scripts/check_oracle_archive_manifest.py` |  |  |
| Metamorphic contracts | `uv run --extra dev pytest tests/python/test_metamorphic_contracts.py -q` |  |  |
| Numerical torture suite | deep-assurance CI or `uv run --extra dev python scripts/run_numerical_torture.py --json-output numerical-torture.json` |  |  |
| Module coverage gates | `python3 scripts/check_module_coverage.py --python-json python-coverage.json --rust-lcov rust-core.lcov --report-json-output module-coverage.json` |  |  |
| Performance smoke | `uv run --extra dev python benchmarks/performance_smoke.py --baseline benchmarks/baselines/performance_smoke.json` |  |  |
| Deep performance smoke | `uv run --extra dev python benchmarks/deep_performance_smoke.py --baseline benchmarks/baselines/deep_performance_smoke.json --json-output deep-performance-smoke.json` |  |  |
| Requirement traceability | `python3 scripts/check_traceability.py` |  |  |
| Coverage waivers | `python3 scripts/check_coverage_waivers.py` |  |  |
| Dependency hygiene | `uv lock --check && python3 scripts/check_dependency_hygiene.py` |  |  |

## Coverage Summary

Python total:

Rust total:

Files below final high-assurance threshold:

Module gate report:

Approved waivers:

## Mutation Summary

Smoke result:

Full mutation report:

Mutation campaign config/dry-run:

Surviving mutants:

Waived or equivalent mutants:

Test-gap mutants:

## Oracle and Metamorphic Summary

External oracle report:

Oracle archive manifest:

Oracle library versions:

Fixture hashes changed:

Metamorphic contract result:

Numerical torture result:

Failing or quarantined seeds:

## Fuzz Summary

Corpus replay result:

Rust fuzz target validation:

Manual fuzz run artifact:

New corpus cases added:

Unresolved fuzz failures:

## Deployment Parity Summary

ONNX families covered:

PMML cases covered:

Rate-table cases covered:

Unsupported export structures verified fail-closed:

## Performance Summary

Baseline file:

Deep baseline file:

Observed fit seconds:

Observed predict seconds:

Memory observations:

RSS peak delta MB:

## Residual Risks

| Risk | Owner | Expiry | Remediation |
| --- | --- | --- | --- |

## Approval

Release approved by:

Approval date:
