# Deep Assurance Expansion Plan

This document is the focused specification and implementation plan for the
remaining assurance layers above the current high-assurance baseline.

It extends the
[High-Assurance Testing Specification](high-assurance-testing-spec.md) and
[High-Assurance Testing Implementation Plan](high-assurance-testing-implementation-plan.md).
The goal is to make realistic actuarial-library defects hard to hide, even when
ordinary unit tests, integration tests, and coverage gates are already green.

No test suite can prove the absence of every possible bug. The standard here is
the strongest practical standard: independent oracles, metamorphic invariants,
mutation testing, numerical torture cases, fuzzing, enforced waivers, scheduled
deep CI, and release evidence that makes remaining risk explicit.

## Scope

This plan covers seven remaining assurance tracks:

1. Scored mutation testing.
2. External oracle and differential testing.
3. Metamorphic testing.
4. Numerical torture testing.
5. Coverage-waiver discipline.
6. Long-running deep-assurance CI.
7. Release evidence packs.

The plan applies to Rust core code, PyO3 bindings, Python public APIs, export
paths, serialization, diagnostics, and deployment scoring.

## Final Deep Layers

The first expansion added deterministic scored mutation, statsmodels oracle
fixtures, metamorphic contracts, numerical torture, malformed-input replay,
coverage waivers, and scheduled deep CI. The remaining layers add the evidence
that keeps this from becoming a one-time push:

- Full mutation campaign policy:
  `specs/mutation_campaigns.json` defines external-tool campaigns for
  `cargo-mutants` and `mutmut`; `scripts/check_mutation_campaigns.py` validates
  the policy; `scripts/run_mutation_campaign.py` dry-runs or executes the
  configured commands and emits machine-readable evidence.
- Rust libFuzzer entry points:
  `fuzz/fuzz_targets/family_deviance.rs` and
  `fuzz/fuzz_targets/link_roundtrip.rs` exercise low-level numerical contracts
  around family deviance, variance, clamping, streaming deviance, and link
  round-trips. `scripts/check_fuzz_targets.py` keeps the scaffolding present and
  buildable.
- Oracle archive governance:
  `tests/oracles/archive_manifest.json` records implemented oracle fixtures and
  the still-open external-oracle work for glum regularized paths,
  statsmodels multinomial non-default-reference fixtures, and R/mgcv or exact
  smooth-GAM fixtures. `scripts/check_oracle_archive_manifest.py` fails on
  missing evidence, stale planned items, or undocumented external blockers.
- Module-level coverage gates:
  `specs/module_coverage_thresholds.json` defines per-module thresholds for
  critical Python and Rust areas. `scripts/check_module_coverage.py` evaluates
  `coverage.py` JSON and `cargo llvm-cov` LCOV reports so module regressions do
  not hide behind high package-level coverage.
- Expanded performance and memory smoke:
  `benchmarks/deep_performance_smoke.py` runs exposure GLM, penalized smooth
  GLM, and native multinomial scenarios against
  `benchmarks/baselines/deep_performance_smoke.json`, including RSS growth
  budgets and machine-readable results.

These layers do not remove the caveat that no finite test suite can prove all
possible behavior correct. They do close the remaining governance gap: expensive
or externally provisioned checks are now specified, owned, expiring, wired into
deep CI, and visible in the release evidence pack.

## Target State

The expansion is complete when all of the following are true:

- Critical Rust and Python modules have scored mutation reports with thresholds
  enforced or explicitly waived.
- Every supported modelling path has at least one deterministic oracle fixture
  and one independent external comparison where a credible implementation
  exists.
- Metamorphic invariants cover data transformations that should not change
  fitted values, predictions, diagnostics, or export scoring.
- Numerical torture cases exercise ill-conditioning, boundaries, extreme
  weights, rare categories, separation, and family-parameter extremes.
- Coverage gaps below the final high-assurance thresholds have expiring,
  owner-approved waivers tied to remediation work.
- Deep CI runs slow property, mutation, fuzz, oracle, and performance jobs on a
  schedule and stores machine-readable artifacts.
- Every release has an evidence pack showing commands, versions, seeds,
  reports, waivers, survivors, and approved residual risks.

## Track 1: Scored Mutation Testing

### Specification

Mutation testing must prove that the suite catches realistic implementation
defects, not just that code lines execute.

Required module thresholds:

| Area | Minimum mutation score |
| --- | ---: |
| Rust families, links, losses, residuals | 95% |
| Rust solvers, smooths, splines, regularization | 90% |
| Rust diagnostics and calibration | 90% |
| Python validation, formula, serialization | 90% |
| Python export and deployment scoring helpers | 95% |
| Python diagnostics and rate tables | 90% |

Surviving mutants must be classified in a machine-readable registry:

- `equivalent`: the mutant is semantically equivalent.
- `accepted-risk`: the mutant is real but low risk for documented reasons.
- `test-gap`: the mutant exposes a missing assertion and blocks release until
  fixed.

Suggested registry:

```text
specs/high_assurance_mutation_waivers.json
```

The long-running campaign registry is:

```text
specs/mutation_campaigns.json
scripts/check_mutation_campaigns.py
scripts/run_mutation_campaign.py
```

Required fields:

- `id`
- `tool`
- `module`
- `mutant`
- `classification`
- `reason`
- `owner`
- `expires`
- `remediation`

### Implementation Plan

1. Add tool configuration.
   - Rust: add `cargo-mutants` documentation and a checked command wrapper.
   - Python: evaluate `mutmut` first because it is simple to run locally; use
     `cosmic-ray` only if subprocess isolation becomes necessary.

2. Add deterministic subset scripts.
   - `scripts/run_rust_mutation.py`
   - `scripts/run_python_mutation.py`
   - Both scripts should accept `--module`, `--timeout`, `--json-output`, and
     `--minimum-score`.

3. Start with small critical modules.
   - Rust: families, links, diagnostics/loss, validation-like guard modules.
   - Python: `families.py`, `links.py`, `validation.py`, `export_onnx.py`,
     `export_pmml.py`, `rate_tables.py`.

4. Kill first-order survivors.
   - Add missing assertions before broadening scope.
   - Prefer exact value assertions, error-message assertions, and invariant
     assertions over snapshot-only checks.

5. Add CI.
   - PR CI: keep the existing deterministic mutation smoke.
   - Deep CI: run scored mutation on selected critical modules weekly.
   - Store JSON and HTML reports as artifacts.

6. Enforce thresholds.
   - Start report-only.
   - Enforce thresholds once the registry has triaged equivalent mutants.
   - Fail on expired mutation waivers.

### Acceptance Criteria

- A clean deep-assurance run produces mutation reports for Rust and Python.
- Critical small modules meet the thresholds or have reviewed waivers.
- Any `test-gap` mutant blocks release.
- Mutation report paths are recorded in the release evidence pack.

## Track 2: External Oracle and Differential Testing

### Specification

Independent oracle tests must compare RustyStats against trusted external
implementations and archived deterministic outputs.

Required oracle sources:

| Area | Preferred oracle |
| --- | --- |
| Unpenalized GLM | `statsmodels` |
| Penalized GLM | `glum` and hand KKT checks |
| Smooth GLM and GAM-like paths | archived R `mgcv` outputs where practical |
| Multinomial | `statsmodels.MNLogit` where model shape is supported |
| Linear algebra and solvers | exact small matrices and finite differences |
| Export scoring | native RustyStats predictions plus runtime execution |

Required fixture families:

- Gaussian identity and log where supported.
- Poisson log with offset and exposure.
- Binomial logit, including imbalanced outcomes.
- Gamma log with positive severity response.
- Tweedie log across representative variance powers.
- Negative binomial log across representative theta values.
- Quasi labels and quasi scoring contracts.
- Multinomial with non-default reference class.
- Smooth and monotonic smooth fits.
- Regularized lasso, ridge, and elastic-net paths.

Oracle fixtures must store:

- Input data or deterministic fixture seed.
- RustyStats model specification.
- External library name and version.
- Expected coefficients, predictions, losses, diagnostics, and tolerances.
- Tolerance rationale.
- Regeneration command.

Suggested layout:

```text
tests/oracles/
  statsmodels/
  glum/
  r/
  exact/
scripts/generate_oracle_fixtures.py
scripts/check_oracle_fixtures.py
tests/oracles/archive_manifest.json
scripts/check_oracle_archive_manifest.py
```

External oracle caveat: the checked-in statsmodels fixtures cover supported
unpenalized GLM families. Regularized glum fixtures, statsmodels multinomial
non-default-reference fixtures, and R/mgcv or exact smooth fixtures remain
tracked planned items in the oracle archive manifest until those external
environments and fixtures are provisioned.

### Implementation Plan

1. Define fixture schema.
   - Use JSON for metadata and compact numeric arrays.
   - Use NPZ only for larger arrays.
   - Include exact versions of oracle libraries.

2. Build deterministic data generators.
   - Small exact fixtures for hand calculations.
   - Medium actuarial fixtures for external-library parity.
   - Fixed seeds and reviewed input distributions.

3. Add oracle tests by risk.
   - First: Poisson exposure, binomial logit, gamma severity, negative
     binomial, Tweedie, lasso/ridge, multinomial.
   - Second: smooth, monotonic smooth, high-cardinality categoricals.
   - Third: diagnostics, inference, and calibration outputs.

4. Add oracle refresh workflow.
   - Regeneration script writes outputs to a temp directory by default.
   - Updating checked-in fixtures requires an explicit `--accept` flag.
   - Release evidence records oracle versions and changed fixture hashes.

5. Add CI.
   - PR CI runs small exact and statsmodels/glum fixtures.
   - Deep CI runs archived R/mgcv fixtures and larger oracle sweeps.

### Acceptance Criteria

- Every supported family has at least one tight deterministic oracle test.
- Every high-risk solver path has a finite-difference, KKT, or exact-matrix
  oracle.
- Any broad stochastic oracle comparison is backed by a tighter deterministic
  fixture.
- Oracle regeneration is documented and reproducible.

## Track 3: Metamorphic Testing

### Specification

Metamorphic tests must assert transformations that should preserve results or
change them in a known way. These tests are especially important for actuarial
models because many production defects appear only after data reshaping,
weighting, encoding, or deployment conversion.

Required invariants:

| Invariant | Expected result |
| --- | --- |
| Row permutation | Fit, predictions, and diagnostics unchanged within tolerance. |
| Duplicated rows versus integer weights | Equivalent weighted objectives and predictions. |
| Global weight scaling | Coefficients unchanged for supported objectives. |
| Exposure scaling with offset adjustment | Rate predictions unchanged. |
| Offset plus intercept shift | Linear predictor changes only by the known shift. |
| Dense versus sparse design matrix | Coefficients and predictions match. |
| Formula API versus dict API | Same design matrix semantics and predictions. |
| Standardized versus unstandardized regularization | Back-transformed coefficients and predictions match. |
| Chunked versus unchunked prediction | Predictions and diagnostics match. |
| Single-thread versus multi-thread execution | Predictions match within deterministic tolerance. |
| Serialized then loaded model | Predictions, metadata, and export behavior match. |
| Native versus exported scoring | Runtime output matches native predictions. |
| Monotonic smooth transformations | Monotonicity and sign constraints are preserved. |

### Implementation Plan

1. Create shared metamorphic fixture builders.
   - Python fixture helpers for GLM, smooth, multinomial, diagnostics, exports.
   - Rust helpers for solver and family-level invariants.

2. Add public API tests first.
   - Put Python tests in `tests/python/test_metamorphic_contracts.py`.
   - Keep datasets small enough for PR CI.

3. Add Rust core tests for low-level invariants.
   - Solver standardization and KKT invariants.
   - Dense/sparse equivalence.
   - Monotonic smooth constraint preservation.

4. Add deep sweeps.
   - Use Hypothesis or deterministic seeded sweeps for larger combinations.
   - Preserve failing seeds.

5. Connect to traceability.
   - Map each invariant to a `HAT-META-*` requirement ID.

### Acceptance Criteria

- Each required invariant has at least one direct test.
- Failing metamorphic tests print the seed, model spec, and transformation.
- Any invariant intentionally not supported has a fail-closed test or a
  documented exclusion.

## Track 4: Numerical Torture Testing

### Specification

Numerical torture tests must deliberately exercise the cases most likely to
break actuarial modelling code.

Required scenarios:

- Ill-conditioned, rank-deficient, and nearly collinear designs.
- Perfect and quasi separation in binomial and multinomial models.
- Extreme weights, zero total weight rejection, and tiny positive weights.
- Large exposures and tiny exposures.
- Poisson counts with non-integer warning contracts.
- Gamma and inverse-link cases near the positive-response boundary.
- Tweedie variance powers near supported boundaries.
- Negative-binomial theta extremes.
- Probabilities near 0 and 1.
- High-cardinality categorical factors with rare and unseen levels.
- All-rare, single-level, and missing-level categories.
- Sparse matrices with empty rows or nearly empty columns.
- Monotonic smooths with constraints near zero.
- Export paths with maximum coefficient magnitudes allowed by the format.

Expected behavior must be explicit for each scenario:

- successful fit with finite predictions;
- convergence warning with usable status;
- validation error with a clear message;
- fail-closed unsupported export error.

Silent NaN, infinite prediction, panic, or unrelated low-level exception is a
release-blocking failure.

### Implementation Plan

1. Add scenario catalog.
   - Suggested file: `specs/numerical_torture_scenarios.json`.
   - Each scenario records component, fixture seed, expected outcome, and
     tolerance.

2. Add deterministic torture tests.
   - Rust: solver and family edge cases.
   - Python: public API, validation, diagnostics, export, serialization.

3. Add deep-case runner.
   - Suggested script: `scripts/run_numerical_torture.py`.
   - Runs the full catalog and writes JSON evidence.

4. Add scheduled CI.
   - PR CI runs a small representative subset.
   - Deep CI runs the full catalog weekly.

5. Turn discovered failures into regression tests.
   - Preserve minimal input.
   - Link to scenario ID and requirement ID.

### Acceptance Criteria

- Every listed scenario has a passing expected-outcome test.
- All successful fits produce finite coefficients and predictions.
- All rejected inputs fail with RustyStats errors or documented warnings.
- Deep CI stores the scenario result JSON.

## Track 5: Coverage-Waiver Discipline

### Specification

Coverage below the final high-assurance threshold is allowed only when it is
explicit, reviewed, temporary, and tied to a remediation plan.

Coverage waiver checks must fail when:

- a waiver is expired;
- a waiver has no owner;
- a waiver has no remediation;
- a waiver covers a module that now meets the target and should be removed;
- a file below threshold has no waiver.

Required coverage evidence:

- Rust `cargo llvm-cov` summary and lcov report.
- Python `coverage.py` JSON report with branch data.
- PyO3 integration coverage approach or documented waiver.
- Per-module threshold report for high-risk modules.

Final module-level thresholds live in:

```text
specs/module_coverage_thresholds.json
scripts/check_module_coverage.py
```

Release evidence should include the module coverage JSON reports produced from
`coverage.py` and LCOV artifacts, not only total package coverage.

### Implementation Plan

1. Extend waiver schema.
   - Add `file_patterns`, `metric`, `current_value`, `target_value`,
     `last_verified`, and `evidence_command`.

2. Extend `scripts/check_coverage_waivers.py`.
   - Read coverage JSON or lcov when provided.
   - Detect expired, stale, and unnecessary waivers.
   - Emit machine-readable JSON for release evidence.

3. Add module-level coverage gates.
   - Start with validation, export, families, links, solvers, diagnostics.
   - Ratchet thresholds only after tests exist.

4. Update release evidence.
   - Record every active waiver and expiry.
   - Record why any final threshold is not yet enforced.

### Acceptance Criteria

- Coverage waivers are not just documents; they are executable checks.
- Expired or stale waivers fail CI.
- New high-risk modules cannot silently enter below threshold.

## Track 6: Long-Running Deep-Assurance CI

### Specification

Slow assurance jobs should not make every PR painful, but they must run
regularly and produce artifacts that reviewers can inspect.

Required deep jobs:

| Job | Cadence | Artifact |
| --- | --- | --- |
| Rust deep property tests | weekly and manual | proptest log and seeds |
| Python metamorphic tests | weekly and manual | pytest JUnit and seed log |
| External oracle suite | weekly and manual | oracle diff JSON |
| Scored mutation | weekly and manual | mutation JSON and HTML |
| Fuzz bounded run | weekly and manual | corpus additions and crash inputs |
| Numerical torture suite | weekly and manual | scenario result JSON |
| Coverage waiver audit | weekly and PR | waiver report JSON |
| Performance and memory smoke | weekly and manual | benchmark JSON |

Deep CI should upload artifacts even on failure. A failed scheduled job creates
a release-risk finding until triaged.

### Implementation Plan

1. Extend `.github/workflows/deep-assurance.yml`.
   - Add jobs incrementally as scripts become available.
   - Use `workflow_dispatch` inputs for module subsets and case counts.

2. Split fast and slow gates.
   - PR: fast exact, smoke, and regression checks.
   - Deep: mutation, large fuzz, larger proptest, full oracle catalog.

3. Store artifacts consistently.
   - Suggested path convention:
     `artifacts/deep-assurance/<job>/<date>/`.

4. Add failure triage process.
   - New failing seed becomes a regression test.
   - New oracle drift requires explicit fixture refresh review.
   - New mutation survivor is triaged before release.

### Acceptance Criteria

- Deep CI can be run manually before release.
- Scheduled failures are visible and not ignored.
- Release evidence links to the latest successful deep run.

## Track 7: Release Evidence Pack

### Specification

Every release must have an evidence pack showing exactly why the release is
believed correct enough to ship.

Required sections:

- Release metadata: version, date, commit, reviewer.
- Fast CI gate summary.
- Deep CI gate summary.
- Rust and Python coverage reports.
- Coverage waiver audit.
- Mutation reports and survivor triage.
- Oracle and differential report.
- Metamorphic and numerical torture report.
- Fuzz report and new corpus cases.
- Deployment parity report.
- Performance and memory report.
- Dependency hygiene report.
- Residual risks with owner, expiry, and remediation.

### Implementation Plan

1. Extend `docs/maintenance/release-evidence-template.md`.
2. Ensure every deep job writes an artifact path suitable for the template.
3. Add a release checklist that blocks approval when:
   - required evidence is missing;
   - any mutation survivor is classified as `test-gap`;
   - oracle drift is unapproved;
   - a coverage or mutation waiver is expired;
   - a deep scheduled failure remains untriaged.

### Acceptance Criteria

- A reviewer can reconstruct the full assurance state from the evidence pack.
- Residual risks are explicit and time-limited.
- Release approval is tied to evidence, not memory of local runs.

## Implementation Order

### Phase A: Schemas and Harnesses

1. Add mutation waiver schema and checker.
2. Add oracle fixture schema and generation/checking commands.
3. Add numerical torture scenario schema.
4. Extend release evidence template.
5. Add traceability IDs for deep-assurance tracks.

Exit criteria:

- All new schemas have validation scripts.
- Empty or seed registries pass validation.
- Documentation lists exact commands.

### Phase B: Highest-Risk Metamorphic and Torture Tests

1. Add row permutation, duplicated rows versus weights, weight scaling, and
   serialization round-trip metamorphic tests.
2. Add Poisson exposure, binomial separation, Gamma boundary, Tweedie boundary,
   negative-binomial theta, rare-category, and singular-design torture tests.
3. Preserve any discovered bugs as regression fixtures.

Exit criteria:

- New tests pass in PR time.
- Failures print enough information to reproduce.

### Phase C: Deterministic Oracles

1. Add exact family fixtures.
2. Add statsmodels GLM fixtures.
3. Add glum penalized fixtures.
4. Add finite-difference and KKT solver fixtures.
5. Add archived R/mgcv fixtures for smooths where feasible.

Exit criteria:

- Every supported family has a tight oracle.
- Oracle refresh is reproducible and reviewed.

### Phase D: Scored Mutation

1. Add Rust mutation wrapper and run families/links/losses first.
2. Add Python mutation wrapper and run validation/export first.
3. Triage equivalent mutants.
4. Kill surviving test-gap mutants.
5. Enforce thresholds in deep CI.

Exit criteria:

- Critical small modules meet mutation thresholds.
- Survivors are visible and classified.

### Phase E: Fuzz Expansion

1. Add Rust fuzz targets for parser, serialization, export, and validation.
2. Add Python fuzz/malformed sweeps for formula and model artifacts.
3. Add corpus minimization and replay.
4. Store crashes as regression cases.

Exit criteria:

- No known crash or panic lacks a replay case.
- Deep fuzz run artifacts are uploaded.

### Phase F: CI and Release Evidence Enforcement

1. Add deep CI jobs as each harness becomes available.
2. Add artifact upload and summary output.
3. Require release evidence for release approval.
4. Fail release on missing artifacts, expired waivers, test-gap mutants, or
   unapproved oracle drift.

Exit criteria:

- A release can be audited from checked-in docs plus CI artifacts.
- All remaining gaps have owners, expiries, and remediation plans.

## Command Targets

The final command surface should include:

```bash
# Mutation
python3 scripts/check_mutation_waivers.py
uv run --extra dev python scripts/run_python_mutation.py --minimum-score 90
uv run --extra dev python scripts/run_rust_mutation.py --minimum-score 90
python3 scripts/check_mutation_campaigns.py
python3 scripts/run_mutation_campaign.py --all --dry-run --json-output mutation-campaign-dry-run.json

# Oracles
uv run --extra dev python scripts/check_oracle_fixtures.py
uv run --extra dev python scripts/generate_oracle_fixtures.py --check
python3 scripts/check_oracle_archive_manifest.py

# Metamorphic and torture suites
uv run --extra dev pytest tests/python/test_metamorphic_contracts.py -q
uv run --extra dev python scripts/run_numerical_torture.py --json-output torture.json

# Fuzz
python3 scripts/check_fuzz_targets.py
cargo check --manifest-path fuzz/Cargo.toml

# Waivers and release evidence inputs
python3 scripts/check_coverage_waivers.py --coverage-json coverage.json
python3 scripts/check_module_coverage.py --check-config
python3 scripts/check_traceability.py

# Performance and memory
uv run --extra dev python benchmarks/deep_performance_smoke.py --baseline benchmarks/baselines/deep_performance_smoke.json --json-output deep-performance-smoke.json
```

## Done Checklist

- [x] Mutation waiver schema and checker exist.
- [x] Rust scored deterministic mutation runs in deep CI.
- [x] Python scored deterministic mutation runs in deep CI.
- [x] Full mutation campaign registry and dry-run wrapper exist.
- [x] External oracle fixture schema exists.
- [x] Oracle fixtures cover every supported scalar GLM family with a
      statsmodels-compatible archived fixture.
- [x] Oracle refresh command is documented.
- [x] Oracle archive manifest records implemented, planned, and external-blocked
      oracle layers.
- [x] Metamorphic tests cover the highest-risk public GLM invariants.
- [x] Numerical torture scenario catalog exists.
- [x] Numerical torture runner writes JSON evidence.
- [x] Rust cargo-fuzz target scaffolding exists for family and link numerical
      contracts.
- [x] Coverage waiver checks consume and emit machine-readable evidence.
- [x] Module-level coverage threshold checker consumes Python JSON and Rust LCOV.
- [x] Deep performance and memory scenario matrix exists.
- [x] Deep CI uploads artifacts for mutation, torture, waivers, and performance.
- [x] Release evidence template includes every deep-assurance track.
- [x] Traceability maps deep-assurance requirements to evidence.
- [ ] Add archived R/mgcv fixtures for smooth and monotonic smooth cases where
      practical.
- [ ] Run sustained cargo-fuzz/libFuzzer campaigns and archive crash-free
      artifacts for release evidence.
- [ ] Add archived multinomial oracle fixtures for model shapes not covered by
      existing live statsmodels comparisons.
- [ ] Add archived glum regularized lasso, ridge, and elastic-net oracle
      fixtures.
