# High-Assurance Testing Implementation Plan

This plan converts the [High-Assurance Testing Specification](high-assurance-testing-spec.md)
into a staged implementation roadmap. It is written so each item can become a
ticket, pull request, or release gate.

The plan is intentionally incremental. RustyStats already has a substantial
test suite, so the fastest path is to first make current evidence measurable,
then close the highest-risk gaps, then add deeper assurance layers such as
mutation testing, fuzzing, and performance gates.

## Guiding Principles

- Keep every phase releasable. Do not create a long-running branch that changes
  many test systems at once.
- Prefer gates that fail clearly and point at the responsible component.
- Add stricter requirements behind report-only mode first when the existing
  codebase is below threshold.
- Turn every discovered production-like failure into a regression test.
- Separate fast PR gates from scheduled deep-assurance jobs.
- Keep numerical tolerances explicit and reviewed.

## Target End State

The implementation is complete when:

- CI enforces formatting, linting, tests, coverage, deployment parity, and
  diagnostics correctness.
- Supported Python versions and OSes are tested.
- Rust and Python coverage meet the spec thresholds or have approved waivers.
- High-risk modules have mutation testing thresholds.
- GLM and multinomial exports are scored against native predictions.
- Property tests include adversarial boundary generators.
- Fuzz corpora replay known parser, validation, and serialization failures.
- Warning contracts are explicit.
- Requirement IDs map high-risk behaviours to test evidence.
- Release docs show exactly which gates passed.

## Current Branch Implementation Status

This branch implements the first enforceable high-assurance baseline:

- CI now includes Python 3.10 through 3.13 on Linux, plus latest macOS and
  Windows smoke coverage.
- Rust release-mode tests, Python coverage, traceability checks, coverage
  waiver checks, packaging smoke, deployment parity, fuzz corpus replay, and
  diagnostics verification are CI jobs.
- A scheduled deep-assurance workflow runs higher-case Rust property tests,
  Python adversarial validation tests, mutation smoke, fuzz replay, deployment
  parity, diagnostics release verification, and performance smoke.
- Deployment artefacts are executed, not only parsed: ONNX scoring/full exports,
  PMML runtime scoring under the JVM-backed deployment gate, and rate-table
  scoring all compare against native predictions.
- Warning contracts are explicit and unexpected warnings fail pytest.
- Dependency hygiene checks enforce lock freshness and reject known yanked
  dependencies.
- Initial coverage is enforced at the current achievable baseline while final
  high-assurance coverage thresholds are tracked with expiring waivers.
- Requirement traceability, fuzz corpus replay, release evidence, mutation
  smoke, and performance baseline files are present and checked.

Remaining work toward the final target standard is intentionally visible:

- Raise Python and Rust coverage from the initial gate to the final spec
  thresholds.
- Replace mutation smoke with scored mutation testing for every critical
  numerical, validation, and export module.
- Add deterministic oracle archives for every supported family, smooth, solver,
  regularized, and multinomial pathway.
- Expand fuzzing from corpus replay into maintained Rust and Python fuzz
  targets.
- Add broader performance baselines beyond the current fit/predict/RSS smoke.

For the focused specification and rollout plan for these remaining tracks, see
[Deep Assurance Expansion Plan](deep-assurance-expansion-plan.md).

## Phase 0: Baseline and Test Infrastructure

Goal: make the current state measurable and reproducible without changing
library behaviour.

### Tasks

1. Add coverage tooling to dev dependencies.
   - Add `coverage` and `pytest-cov` to the dev dependency group.
   - Standardize Rust coverage on `cargo llvm-cov`.
   - Document local commands.

2. Add coverage commands.
   - Rust: `cargo llvm-cov --workspace --summary-only`.
   - Python: `uv run --extra dev pytest tests/python --cov=rustystats --cov-report=term-missing`.
   - Store generated reports as CI artifacts.

3. Add report-only CI coverage job.
   - Do not fail on thresholds yet.
   - Publish summary in CI logs.
   - Track current baseline by module.

4. Add release-mode Rust tests to CI.
   - Run `cargo test --workspace --release` on Linux.
   - Keep debug-mode tests on all OSes.

5. Add a test-suite dashboard document.
   - Current test counts.
   - Current coverage.
   - Known weak modules.
   - Current warnings.

6. Add dependency hygiene checks.
   - Run `uv lock --check` in CI.
   - Reject known yanked dependencies and lock entries marked yanked.
   - Record dependency warnings in release evidence.

### Acceptance Criteria

- A clean CI run produces Rust and Python coverage reports.
- Release-mode Rust tests run in CI.
- Dependency lock freshness and yanked-package checks run in CI.
- No test behaviour changes are required for this phase.
- Current weak coverage areas are visible in CI output.

### Suggested PRs

- `test: add coverage tooling and report-only coverage job`
- `ci: run rust release-mode tests`
- `docs: record test-suite baseline`

## Phase 1: CI Matrix and Warning Contracts

Goal: ensure the suite runs in the environments the package claims to support
and make warning behaviour explicit.

### Tasks

1. Expand Python CI matrix.
   - Run Python tests on 3.10, 3.11, 3.12, and 3.13.
   - Run at least Linux for every Python version.
   - Add macOS and Windows for latest Python.

2. Expand packaging smoke tests.
   - Build the extension through `maturin develop --release`.
   - Import `rustystats`.
   - Fit and predict a tiny GLM.
   - Serialize and deserialize a small model.

3. Add unexpected-warning failure policy.
   - Add pytest warning filters.
   - Convert existing expected warnings to `pytest.warns`.
   - Remove broad `warnings.filterwarnings("ignore")` from tests.

4. Pin or explicitly handle reference-library warning drift.
   - Statsmodels BIC warning should be asserted or silenced locally with a
     documented compatibility note.
   - Domain warnings such as non-integer Poisson should be tested directly.

### Acceptance Criteria

- CI runs Python tests on every supported Python minor.
- No broad test-module warning suppression remains.
- Full pytest suite passes with unexpected warnings treated as failures.
- Expected warnings are asserted at the call site.

### Suggested PRs

- `ci: expand python and os test matrix`
- `test: make warning contracts explicit`
- `test: add packaging smoke coverage`

## Phase 2: Deployment Scoring Parity

Goal: eliminate silent divergence between native predictions and exported
artefacts.

### Tasks

1. Add GLM ONNX scoring parity tests.
   - Execute ONNX with `onnxruntime`.
   - Compare scoring-mode predictions to `result.predict` from design matrices.
   - Compare full-mode predictions to `result.predict` from raw data.
   - Cover Gaussian identity, Poisson log, binomial logit, gamma log, Tweedie
     log, negative-binomial log, and supported non-canonical links.

2. Add GLM PMML scoring parity tests.
   - Execute PMML with `pypmml` or another approved runtime.
   - Compare predictions to native predictions.
   - Include offset, exposure, categorical factors, and spline cases when
     supported.

3. Add rate-table parity tests.
   - Export resolved rate tables.
   - Score covered rows and edge rows.
   - Compare with native predictions.

4. Expand multinomial export parity.
   - Keep current non-reference ONNX parity.
   - Add PMML runtime parity if runtime support is available.
   - Add fail-closed checks for unsupported structures.

5. Add deployment parity CI job.
   - Install ONNX and PMML runtime dependencies.
   - Fail release if runtime dependencies cannot be installed.

### Acceptance Criteria

- Export tests fail if exported artefacts score differently from native
  predictions beyond declared tolerance.
- Unsupported export structures fail closed with `ValidationError`.
- Deployment parity job is a required CI gate.

### Suggested PRs

- `test: add glm onnx runtime parity`
- `test: add pmml runtime parity`
- `test: add rate-table scoring parity`
- `ci: add deployment parity gate`

## Phase 3: Coverage Gates and Critical Gaps

Goal: move from report-only coverage to enforced coverage, while improving the
weakest high-risk modules.

### Tasks

1. Set initial CI coverage thresholds.
   - Start with current baseline plus a small margin.
   - Use per-package and per-module thresholds where supported.
   - Ratchet thresholds upward after each coverage PR.

2. Add missing Python tests for low-coverage modules.
   - `smooth.py`.
   - `export_pmml.py`.
   - `export_onnx.py`.
   - `links.py`.
   - `validation.py`.
   - `interactions.py`.
   - `regularization_path.py`.

3. Add missing Rust tests for low-coverage modules.
   - `solvers/smooth_glm.rs`.
   - `solvers/nnls.rs`.
   - `solvers/gcv_optimizer.rs`.
   - `target_encoding/mod.rs`.
   - `regularization/mod.rs`.
   - `diagnostics/exploration.rs`.

4. Add negative-path tests.
   - Invalid dimensions.
   - Invalid family/link combinations.
   - Unsupported export structures.
   - Missing columns.
   - Invalid weights, exposure, offsets, and response values.
   - Corrupt serialization inputs.

5. Add branch/region coverage where possible.
   - Especially for validation and export paths.

### Acceptance Criteria

- Coverage is enforced in CI.
- No high-risk file remains below the agreed interim threshold without a waiver.
- Each low-coverage module has a remediation ticket or completed tests.
- Coverage trend cannot regress without approval.

### Suggested PRs

- `ci: enforce initial coverage thresholds`
- `test: raise export and validation coverage`
- `test: raise smooth and spline coverage`
- `test: raise solver edge-path coverage`

## Phase 4: Stronger Oracles and Tolerance Hardening

Goal: pair broad comparison tests with tighter deterministic evidence.

### Tasks

1. Define deterministic golden fixtures.
   - One fixture per supported GLM family.
   - Frequency with exposure.
   - Severity.
   - Overdispersed counts.
   - High-cardinality categorical.
   - Smooth/spline.
   - Multinomial with non-default reference class.

2. Add exact or near-exact oracle tests.
   - Hand-calculated small cases.
   - Finite-difference gradients.
   - Hessian checks.
   - KKT checks.
   - Small-matrix solver references.

3. Add external oracle archives.
   - Statsmodels outputs for supported GLMs.
   - Glum outputs for penalized models.
   - R `glm` and `mgcv` outputs for selected actuarial/smooth cases.

4. Review broad tolerances.
   - Classify each tolerance as exact, deterministic oracle, stochastic oracle,
     deployment parity, or performance.
   - Tighten where deterministic evidence allows.
   - Add comments explaining any intentionally wide tolerance.

5. Add oracle refresh process.
   - Script to regenerate oracle outputs.
   - Review checklist for updating golden values.

### Acceptance Criteria

- Every supported family has at least one tight deterministic oracle.
- Broad stochastic oracle tests are backed by tight golden tests.
- Tolerance policy is reflected in test comments or helper names.
- Oracle regeneration is documented.

### Suggested PRs

- `test: add deterministic family golden fixtures`
- `test: add finite-difference solver oracles`
- `test: add archived r oracle fixtures`
- `test: tighten documented numerical tolerances`

## Phase 5: Adversarial Property Testing

Goal: expand property tests beyond polite valid ranges and exercise realistic
edge cases.

### Tasks

1. Add richer Rust proptest strategies.
   - Near-zero and huge `mu`.
   - Probabilities near 0 and 1.
   - Extreme weights and exposures.
   - Zero-inflated and all-zero responses.
   - Ill-conditioned design matrices.
   - Sparse high-cardinality categorical encodings.

2. Add Python property tests.
   - Use Hypothesis if adopted, or deterministic seeded adversarial sweeps.
   - Cover formula construction, dataframe dtypes, serialization, exports, and
     public API validation.

3. Add fold-safety randomized tests.
   - Random fold assignments.
   - Isolated rare levels.
   - Held-out target-encoding combinations.
   - Extreme validation weights.

4. Add thread and chunk equivalence sweeps.
   - `RAYON_NUM_THREADS=1`.
   - Default thread count.
   - Fixed higher thread count.
   - Chunked and unchunked prediction/diagnostics.

5. Add scheduled deep property job.
   - Higher case count.
   - Preserve failing seeds.
   - Keep PR job bounded and fast.

### Acceptance Criteria

- Boundary generators are present for every critical numerical family.
- Failing seeds are printed and reproducible.
- Scheduled deep property job exists.
- PR property tests remain fast enough for normal development.

### Suggested PRs

- `test: add adversarial rust proptest strategies`
- `test: add python property checks`
- `test: add randomized fold-safety sweeps`
- `ci: add scheduled deep property job`

## Phase 6: Mutation Testing

Goal: prove assertions catch realistic code defects.

### Tasks

1. Select mutation tools.
   - Rust: evaluate `cargo-mutants`.
   - Python: evaluate `mutmut` or `cosmic-ray`.

2. Start with critical small modules.
   - Families.
   - Links.
   - Losses.
   - Residuals.
   - Validation.
   - Export scoring helpers.

3. Add mutation smoke CI.
   - Run a limited critical subset on PRs or nightly.
   - Publish surviving mutants.

4. Add full scheduled mutation job.
   - Run broader mutation tests nightly or weekly.
   - Store reports as artifacts.

5. Triage surviving mutants.
   - Equivalent mutant.
   - Accepted low-risk mutant.
   - Test gap.

### Acceptance Criteria

- Critical small modules meet mutation score targets.
- Surviving mutants are tracked and reviewed.
- Mutation reports are visible in CI artifacts.
- New high-risk modules cannot ship with unreviewed mutation gaps.

### Suggested PRs

- `test: add rust mutation smoke target`
- `test: add python mutation smoke target`
- `ci: add scheduled mutation job`
- `test: kill first mutation survivors in families and validation`

## Phase 7: Fuzzing and Malformed Input Hardening

Goal: catch panics and fail-open validation bugs.

### Tasks

1. Add Rust fuzz targets.
   - Family parameter parsing.
   - Design-matrix helpers.
   - Serialization/deserialization.
   - Export builder inputs.
   - Multinomial validation.

2. Add Python malformed-input sweeps.
   - Formula dictionaries.
   - Mixed dataframe dtypes.
   - Missing columns.
   - Corrupt serialized models.
   - Invalid export paths and modes.

3. Add corpus replay.
   - Store minimized failures.
   - Replay corpus in PR CI.

4. Add scheduled fuzz job.
   - Bounded time.
   - Artifact upload for new failures.

### Acceptance Criteria

- Malformed user input does not panic.
- Invalid input returns RustyStats exceptions with useful messages.
- Fuzz corpus replay is required in CI.
- New fuzz failures become regression tests.

### Suggested PRs

- `test: add fuzz targets for parsers and serialization`
- `test: add malformed python api sweeps`
- `ci: add fuzz corpus replay`

## Phase 8: Requirement Traceability

Goal: make assurance reviewable by linking high-risk behaviours to evidence.

### Tasks

1. Create requirement registry.
   - `RS-QA-FAM-*`.
   - `RS-QA-SOL-*`.
   - `RS-QA-DM-*`.
   - `RS-QA-CV-*`.
   - `RS-QA-DIAG-*`.
   - `RS-QA-MN-*`.
   - `RS-QA-EXP-*`.
   - `RS-QA-SER-*`.
   - `RS-QA-VAL-*`.

2. Add test evidence mapping.
   - Requirement ID.
   - Test files and test names.
   - Oracle or invariant.
   - Tolerance.
   - Known residual risk.

3. Add CI check for stale mapping.
   - Optional lightweight script to ensure referenced test files exist.

4. Add release evidence template.
   - Commands run.
   - Coverage summary.
   - Mutation summary.
   - Known waivers.
   - Approved residual risks.

### Acceptance Criteria

- High-risk behaviours are traceable to tests.
- Release reviewers can see which tests protect each actuarial requirement.
- Waivers have owners and expiry dates.

### Suggested PRs

- `docs: add qa requirement registry`
- `docs: map high-risk behaviours to tests`
- `ci: check qa evidence references`

## Phase 9: Performance and Resource Gates

Goal: prevent correct code from becoming operationally unusable.

### Tasks

1. Define benchmark scenarios.
   - Dense IRLS.
   - Sparse IRLS.
   - Smooth GLM.
   - Multinomial fit and predict.
   - Diagnostics high-cardinality.
   - Export generation.
   - Chunked prediction.

2. Add release benchmark runner.
   - Use release build.
   - Pin deterministic datasets.
   - Emit JSON results.

3. Add baseline comparison.
   - Check in approved baseline.
   - Fail on unapproved regressions.
   - Allow explicit baseline refresh PRs.

4. Add memory smoke checks.
   - Large diagnostics.
   - Wide design matrices.
   - Multinomial alternatives.

### Acceptance Criteria

- Critical benchmark regressions are visible.
- Memory peaks are bounded for known scenarios.
- Baseline updates require review.

### Suggested PRs

- `bench: add release performance baseline`
- `ci: add performance regression gate`
- `bench: add memory smoke scenarios`

## Suggested Milestones

| Milestone | Focus | Expected Outcome |
| --- | --- | --- |
| M0 | Baseline evidence | Coverage and release tests visible, not yet strict. |
| M1 | CI and warnings | Supported environments and warning contracts enforced. |
| M2 | Deployment parity | Exported artefacts proven against native scoring. |
| M3 | Coverage hardening | Low-coverage critical modules improved and gated. |
| M4 | Oracle hardening | Tight golden fixtures and documented tolerances. |
| M5 | Adversarial testing | Boundary and randomized properties catch edge bugs. |
| M6 | Mutation and fuzz | Assertions and validation hardened against synthetic defects. |
| M7 | Traceability and performance | Release evidence is reviewable and operational risks are gated. |

## First Ten Tickets

1. Add `pytest-cov` and coverage commands.
2. Add report-only CI coverage job.
3. Add Linux release-mode Rust CI job.
4. Expand Python CI to 3.10 through 3.13.
5. Remove broad warning suppression and assert expected warnings.
6. Add GLM ONNX runtime parity for scoring mode.
7. Add GLM ONNX runtime parity for full mode.
8. Add PMML runtime parity for one simple GLM.
9. Add coverage tests for `validation.py` negative paths.
10. Add adversarial family/link proptest generators.

## Risk-Based Priority

Highest priority:

- Deployment parity.
- Validation and fail-closed behaviour.
- Solver numerical correctness.
- Exposure, offsets, weights, and rate-ranked diagnostics.
- Serialization compatibility.

Medium priority:

- Coverage ratcheting.
- Warning contracts.
- Smooth/spline oracle expansion.
- Multinomial edge cases.

Scheduled deep-assurance priority:

- Full mutation testing.
- Fuzz expansion.
- Performance baseline refinement.
- Requirement traceability automation.

## Local Command Reference

Fast local checks:

```bash
cargo test --workspace
uv run --extra dev pytest tests/python -q
```

Release and diagnostics checks:

```bash
cargo test --workspace --release
uv run --extra dev maturin develop --release
uv run --extra dev python benchmarks/verify_diagnostics_correctness.py
```

Coverage checks:

```bash
cargo llvm-cov --workspace --summary-only
uv run --extra dev pytest tests/python --cov=rustystats --cov-report=term-missing
```

Deployment parity checks:

```bash
RUSTYSTATS_REQUIRE_EXPORT_RUNTIMES=1 RUSTYSTATS_RUN_PMML_RUNTIME=1 \
  uv run --extra dev pytest tests/python/test_export.py tests/python/test_rate_tables_parity.py -q
```

Malformed-input, mutation, and release evidence checks:

```bash
python3 scripts/check_traceability.py
python3 scripts/check_coverage_waivers.py
python3 scripts/check_dependency_hygiene.py
uv run --extra dev python scripts/replay_fuzz_corpus.py
uv run --extra dev python scripts/mutation_smoke.py
uv run --extra dev python scripts/package_smoke.py
```

Performance smoke:

```bash
uv run --extra dev python benchmarks/performance_smoke.py \
  --baseline benchmarks/baselines/performance_smoke.json \
  --json-output /tmp/rustystats-performance-smoke.json
```

## Rollout Rules

- Start new gates in report-only mode if the current suite cannot pass them.
- Convert report-only gates to required gates once the baseline is fixed.
- Every threshold increase should be a separate PR.
- Waivers must include owner, reason, expiry date, and remediation plan.
- A failed required gate blocks release unless explicitly waived by the release
  owner.

## Done Checklist

- [x] Coverage tooling is installed and documented.
- [x] Coverage is enforced in CI at the initial baseline.
- [x] Python and OS matrix matches package support.
- [x] Unexpected warnings fail tests.
- [x] Dependency hygiene checks run in CI.
- [x] Deployment parity tests execute exported artefacts.
- [ ] Low-coverage high-risk modules meet final thresholds; interim waivers
      track remaining gaps.
- [x] Deterministic oracle fixtures cover every supported scalar GLM family;
      smooth and multinomial archive expansion remains scheduled deep work.
- [x] Property tests include boundary and adversarial generators for validation;
      broader solver/export generators remain to be added.
- [x] Mutation smoke runs on a critical validation module; scored mutation
      testing remains to be added.
- [x] Fuzz corpus replay runs in CI.
- [x] Requirement traceability exists.
- [x] Performance and memory smoke gate exists; broader baselines remain to be
      added.
- [x] Release evidence template exists for releases.
