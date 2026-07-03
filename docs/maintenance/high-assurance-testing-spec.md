# High-Assurance Testing Specification

This document defines the target testing standard for RustyStats. The goal is
not merely to make the test suite large or green; the goal is to make realistic
library defects hard to introduce and easy to diagnose before release.

RustyStats is an actuarial modelling library. Incorrect results can affect
pricing, reserving, underwriting decisions, compliance evidence, and downstream
deployment systems. The test suite must therefore provide evidence across five
dimensions:

1. Mathematical correctness.
2. Numerical robustness.
3. API and serialization compatibility.
4. Deployment scoring parity.
5. Reproducibility across supported environments.
6. Dependency and release-input hygiene.

No software test suite can provide literal 100% confidence. The standard here is
to make residual risk explicit, measured, and continuously reduced.

For the phased execution roadmap, see [High-Assurance Testing Implementation Plan](high-assurance-testing-implementation-plan.md).

## Release Standard

A release is eligible only when all mandatory gates pass on a clean checkout.

| Gate | Requirement |
| --- | --- |
| Rust unit and integration tests | `cargo test --workspace` passes on all supported OSes. |
| Rust release-mode tests | `cargo test --workspace --release` passes on Linux. |
| Python tests | Full pytest suite passes on each supported Python minor. |
| Diagnostics harness | Diagnostics correctness harness passes all scenarios. |
| Formatting and linting | Rustfmt, clippy with `-D warnings`, ruff check, and ruff format all pass. |
| Coverage | Rust and Python coverage meet the thresholds below. |
| Mutation testing | Critical modules meet mutation score thresholds below. |
| Differential oracle tests | Required external-oracle comparisons pass. |
| Deployment parity | PMML, ONNX, and rate-table exports score identically to native predictions within declared tolerances. |
| Fuzz and adversarial tests | Required fuzz smoke targets and adversarial property suites pass. |
| Reproducibility | Determinism tests pass across repeated runs and thread-count settings. |
| Dependency hygiene | Lockfile is current, resolver output has no unresolved warnings, and no yanked dependency is accepted. |
| Performance guardrails | Release performance checks pass for selected hot paths. |

## Dependency Hygiene Standard

Dependency state is part of the test evidence. A mathematically correct suite
can still become unreliable if it runs against yanked, drifting, or silently
changed dependencies.

Required checks:

- The checked-in lockfile must be current with `pyproject.toml`.
- CI must fail on known yanked dependencies and any lock entry marked yanked.
- Dependency resolver warnings must be treated as release-risk findings.
- Dependency changes must be visible in review and included in release
  evidence.
- Numerical dependency bounds should be conservative when a current release has
  known correctness or compatibility risk.

## Coverage Standard

Coverage is a gate, not a vanity metric. Thresholds should be enforced in CI,
and any exclusion must be explicit and reviewed.

| Area | Minimum line coverage | Minimum branch/region coverage | Notes |
| --- | ---: | ---: | --- |
| `rustystats-core` overall | 95% | 90% | Core numerical implementation. |
| Rust family/link modules | 98% | 95% | Small, mathematical, high-risk. |
| Rust solvers | 95% | 90% | IRLS, coordinate descent, smooth GLM, multinomial, NNLS, GCV. |
| Rust diagnostics | 95% | 90% | Diagnostics support actuarial review evidence. |
| PyO3 binding crate | 90% measured through Python integration | 85% | Cargo-only coverage is insufficient for bindings. |
| Python package overall | 95% | 90% | Includes API orchestration and serialization. |
| Export/deployment modules | 98% | 95% | Low tolerance for silent deployment divergence. |
| Validation/error handling | 98% | 95% | Fail-closed behaviour is critical. |

Coverage reports must include:

- Rust coverage from `cargo llvm-cov` or an equivalent maintained tool.
- Python coverage from `pytest-cov` or `coverage.py`.
- Combined evidence for PyO3 paths exercised by Python tests.
- A list of files below threshold with owner-approved remediation tickets.

## Mutation Standard

Mutation testing is required for high-risk mathematical and deployment code. A
module can have high coverage and still have weak assertions; mutation testing
guards against that.

Required thresholds:

| Area | Mutation score |
| --- | ---: |
| Families, links, losses, residuals | 95% |
| Solvers and regularization | 90% |
| Design matrix and formula parsing | 90% |
| Calibration, diagnostics, lift, Gini, A/E | 90% |
| Export and deployment scoring | 95% |
| Python validation and serialization | 90% |

Surviving mutants must be triaged as one of:

- Equivalent mutant, with written justification.
- Non-critical mutant, with accepted-risk note.
- Test gap, with a blocking remediation issue.

## Test Taxonomy

The suite should be organized around layers. Each layer catches a different
class of defect.

### L0: Exact Unit Tests

Purpose: verify small functions against known identities and hand calculations.

Required examples:

- Poisson, binomial, gamma, Gaussian, Tweedie, negative-binomial unit deviance.
- Link, inverse-link, and derivative identities.
- IRLS weights and working responses.
- KKT conditions for penalized solvers.
- Spline basis construction, penalty matrices, and monotonic basis properties.
- A/E, lift, Gini, AUC, confidence intervals, and residual calculations.
- Formula parser and design-matrix naming conventions.

Expectations:

- Prefer exact hand-calculated fixtures for simple cases.
- Use tight tolerances when values are deterministic and well-conditioned.
- Assert error type and message for invalid input.

### L1: Property-Based Tests

Purpose: explore input space and enforce invariants that should hold for all
valid inputs.

Required properties:

- Link/inverse round-trip across safe and boundary ranges.
- Deviance non-negativity and zero deviance at perfect fit.
- Predictions remain in family support.
- Probabilities sum to one for multinomial and binomial models.
- Weights and exposure aggregation identities.
- Regularization monotonicity where mathematically expected.
- Standardization invariance for penalized models.
- Fold-safe target encoding has no validation leakage.
- Serialization round-trip preserves predictions.
- Chunked and unchunked paths are equivalent.

Adversarial generators must include:

- Exact zero and near-zero values.
- Very large and very small magnitudes.
- Probabilities near 0 and 1.
- Highly imbalanced classes.
- Sparse high-cardinality categorical factors.
- Single-level and all-rare categoricals.
- Near-collinear and singular designs.
- Extreme weights and exposure values.
- Non-contiguous arrays and unusual dataframe dtypes.
- Missing, NaN, infinity, and invalid-domain values.

CI should run a bounded case count. Nightly or scheduled jobs should run deeper
case counts and preserve failing seeds.

### L2: Integration Tests

Purpose: verify public APIs as users exercise them.

Required coverage:

- `glm_dict`, low-level fitting bindings, result objects, prediction, summary,
  diagnostics, calibration, contributions, and export methods.
- Multinomial model construction, fitting, prediction, diagnostics,
  calibration, scenario scoring, and serialization.
- Python wrappers around Rust helpers.
- Interactions between terms: linear, categorical, target encoding, frequency
  encoding, splines, monotonic splines, expression terms, offsets, exposure,
  prior weights, and regularization.

Expectations:

- Tests should operate through public APIs unless verifying a private numerical
  primitive directly.
- Public API tests should assert values, not only shape or non-null output.
- All user-visible error paths should be tested.

### L3: Differential and Oracle Tests

Purpose: compare RustyStats against independent implementations.

Required oracles:

- `statsmodels` for unpenalized GLMs, deviance, log-likelihood, standard errors,
  confidence intervals, and residuals where supported.
- `glum` for penalized GLM comparisons and alpha-grid sanity where supported.
- R `glm`, `mgcv`, or archived R oracle outputs for selected actuarial GLM and
  smooth cases.
- Hand-derived finite-difference checks for gradients and Hessians.
- Exact small-matrix linear algebra references for solvers.

Oracle tests must include:

- At least one deterministic golden fixture per supported family.
- At least one exposure/offset frequency fixture.
- At least one severity fixture.
- At least one overdispersed-count fixture.
- At least one high-cardinality categorical fixture.
- At least one regularized fit per penalty type.
- At least one smooth/spline fit per supported spline family.
- At least one multinomial fixture with non-default reference class.

Broad oracle tolerances may be looser for stochastic synthetic datasets, but the
suite must also include tight deterministic fixtures where the expected values
are stable.

### L4: Deployment Parity Tests

Purpose: prove exported artefacts score the same as native RustyStats.

Mandatory parity checks:

- PMML scores match `result.predict`.
- ONNX scoring mode matches `result.predict` from supplied design matrices.
- ONNX full mode matches `result.predict` from raw data for every supported term
  type in full mode.
- Rate-table exports match native predictions on covered and edge rows.
- Multinomial PMML and ONNX probabilities match `predict_proba`.

Required model shapes:

- Gaussian identity.
- Gaussian log, if supported.
- Poisson log with offset and with exposure.
- Binomial logit.
- Gamma log.
- Tweedie log, including configured variance power.
- Negative binomial log, including configured theta.
- Categorical factors with reference levels.
- Continuous and categorical interactions.
- Splines and monotonic splines.
- Target/frequency encoding where export is supported; otherwise fail-closed
  tests must assert a clear error.

Deployment tests must execute exported artefacts with actual runtime libraries
when available, not merely parse bytes or XML. If an optional runtime is
unavailable, CI must either install it or mark the job as failed for release.

### L5: Fuzz and Malformed-Input Tests

Purpose: catch panics, memory-safety issues, parser bugs, and fail-open
validation gaps.

Required fuzz targets:

- Formula and term-spec parsing.
- Model serialization and deserialization.
- PMML and ONNX export builders.
- Design-matrix construction from mixed dtypes.
- Family parameter parsing.
- Multinomial availability and alternative-specific tensor validation.

Fuzz expectations:

- No panic on malformed user input.
- Invalid input fails with a RustyStats error, not a low-level panic or
  unrelated exception.
- Deserializers reject corrupt or incompatible artefacts.
- Fuzz corpora include minimized regression cases.

### L6: Reproducibility and Determinism Tests

Purpose: prove repeated runs produce stable evidence.

Required checks:

- Repeated fits with identical data and seed produce identical parameters and
  predictions.
- Target encoding and CV are deterministic under explicit seed.
- Default seed behaviour is documented and tested.
- Thread-count changes do not change predictions beyond declared tolerance.
- Chunked, parallel, sparse, and dense paths are equivalent.
- Serialization round-trips preserve predictions exactly or within declared
  tolerance.

Recommended environment sweep:

- `RAYON_NUM_THREADS=1`.
- Default Rayon thread count.
- A higher fixed thread count on Linux CI.

### L7: Performance and Resource Guardrails

Purpose: catch algorithmic regressions that do not change answers but make the
library unusable.

Required benchmarks:

- IRLS dense and sparse kernels.
- Design-matrix construction.
- Diagnostics on wide and high-cardinality data.
- Multinomial fitting and prediction.
- Export generation.
- Chunked prediction and diagnostics.

Release gates should use conservative thresholds:

- No critical benchmark may regress more than an approved percentage relative
  to a checked-in baseline.
- Memory peaks for known scenarios must remain below documented limits.
- Performance tests must run in release mode.

## CI Matrix

Required CI jobs:

| Job | OS | Python | Rust profile | Purpose |
| --- | --- | --- | --- | --- |
| Rust debug | Linux, macOS, Windows | 3.10 and latest | debug | Unit, integration, doctests, clippy. |
| Rust release | Linux | latest | release | Optimized numerical path. |
| Python package | Linux, macOS, Windows | 3.10, 3.11, 3.12, 3.13 | release extension | Public API and packaging. |
| Coverage | Linux | latest | debug/instrumented | Coverage reports and thresholds. |
| Dependency hygiene | Linux | latest | n/a | Lock freshness and yanked dependency checks. |
| Diagnostics harness | Linux | latest | release extension | Actuarial diagnostics invariants. |
| Deployment parity | Linux | latest | release extension | PMML/ONNX/rate-table scoring. |
| Mutation smoke | Linux | latest | selected | Critical module mutation gate. |
| Fuzz smoke | Linux | latest | selected | Short corpus replay and bounded fuzz. |
| Nightly deep | Linux | latest | release | Deep proptest, mutation, fuzz, performance. |

Every job must run from a clean checkout. Tests must not rely on local cached
artefacts or developer environment state.

## Warning Policy

Warnings are part of the user contract.

Required policy:

- CI should treat unexpected warnings as errors.
- Expected warnings must be asserted with `pytest.warns`.
- Broad warning suppression is not allowed in test modules.
- Reference-library deprecation warnings must be reviewed and pinned to an
  issue or compatibility plan.

Critical expected warnings:

- Non-integer Poisson response.
- Offset passed where exposure weighting is required.
- Inference unavailable because covariance was skipped or failed.
- Regularized robust SE limitations.
- Export fail-closed warnings or errors for unsupported model structures.

## Numerical Tolerance Policy

Every tolerance must be justified by test type.

| Test type | Expected tolerance |
| --- | --- |
| Hand-computed unit tests | Exact or near machine precision. |
| Deterministic small oracle tests | Tight, usually `1e-8` to `1e-10` where conditioned. |
| External library comparisons | Tolerance documented per library and scenario. |
| Stochastic synthetic comparisons | Wider tolerance allowed, but not the only evidence. |
| Deployment parity | Native and exported predictions should be near machine precision unless the export format imposes lower precision. |
| Performance tests | Relative regression thresholds, not numerical tolerance. |

Loose broad tolerances must be backed by tighter golden fixtures.

## Component Requirements

### Families and Links

Required tests:

- Unit deviance hand calculations for every family.
- Boundary support checks for valid and invalid `mu` and `y`.
- Link/inverse/derivative identities over broad and boundary ranges.
- Log-likelihood checks where implemented.
- Family parameter parsing for Tweedie and negative binomial.
- Quasi-family labels and behaviour separate from base-family labels.

Required oracles:

- Statsmodels comparisons where supported.
- Hand-derived identities for unsupported or quasi families.

### Solvers

Required tests:

- OLS equivalence for Gaussian identity.
- Poisson, binomial, gamma, Tweedie, negative-binomial convergence fixtures.
- Step-halving and non-convergence status contracts.
- Covariance success and failure modes.
- Dense, sparse, chunked, and cached-kernel equivalence.
- Penalized objective, KKT, alpha max, and intercept-exclusion checks.
- Standardization and back-transform invariance.
- Near-singular and collinear design behaviour.
- Release-mode parity against debug-mode results within tolerance.

### Smooths and Splines

Required tests:

- Basis shape, partition, monotonicity, and boundary behaviour.
- Knot reuse on prediction.
- Penalty matrix symmetry, positive semidefiniteness, and null space.
- GCV and EDF monotonicity.
- Smooth GLM family parameter propagation.
- Smooth solver convergence, covariance, offset, weights, and diagnostics.
- Oracle comparisons against a known implementation or archived fixtures.

### Design Matrix and Formula API

Required tests:

- Every term type and combination of term types.
- Reference-level handling and feature naming.
- Unseen category prediction.
- Missing, NaN, infinity, and dtype validation.
- Formula/dict API parity for equivalent models.
- Train/predict matrix consistency.
- Serialization of formula metadata.
- Extraction of required columns for prediction and export.

### Target and Frequency Encoding

Required tests:

- Ordered target encoding leakage prevention.
- Exposure-weighted target encoding.
- Fold-safe CV target encoding.
- Unseen levels and all-rare levels.
- Multiway interactions.
- Determinism under seed and multi-permutation averaging.
- Serialization and prediction parity.

### Diagnostics and Calibration

Required tests:

- A/E by decile and factor.
- Rate-ranked diagnostics with exposure.
- Lift, Gini, AUC, loss metrics, residual summaries.
- Factor diagnostics for fitted and unfitted factors.
- Score tests and robust SE status.
- Train/test diagnostics with unseen test levels.
- Partial dependence for continuous, categorical, and encoded factors.
- JSON serialization of diagnostics.
- Diagnostics harness scenarios as release gates.

### Multinomial

Required tests:

- Shared and alternative-specific parameter layouts.
- Reference-class invariance.
- Availability masks.
- Alternative generic and class-specific terms.
- Offsets and weights.
- Penalization, KKT, warm starts, CV, and calibration.
- Diagnostics and confusion matrices.
- Serialization and deployment fail-closed cases.
- Statsmodels MNLogit oracle fixtures where supported.

### Inference

Required tests:

- Standard errors and covariance matrices.
- Robust covariance types.
- Confidence intervals and p-values.
- Covariance skipped/failed status.
- Regularized inference honesty.
- Score tests for continuous and categorical factors.
- Singular and near-singular cases.

### Serialization

Required tests:

- Round-trip all supported model families and term types.
- Version metadata.
- Backward-compatible fixture loading.
- Corrupt artefact rejection.
- Cross-version compatibility fixtures when format changes.
- Prediction parity after load.

### Export and Rate Tables

Required tests:

- PMML parse and runtime scoring.
- ONNX runtime scoring.
- Rate-table scoring parity.
- Fail-closed unsupported structures.
- Metadata completeness.
- File path and byte/string return parity.
- Cross-platform runtime execution.

### Validation and Errors

Required tests:

- Every public validation branch.
- Clear exception type and useful message.
- No panic or low-level exception for user input.
- Warning contracts.
- Invalid dimensions and dtype mismatches.
- Invalid family/link combinations.
- Invalid weights, exposure, offset, and response support.

## Requirement Traceability

Every high-risk behaviour should have a requirement ID and test evidence.

Recommended ID format:

- `RS-QA-FAM-*` for families and links.
- `RS-QA-SOL-*` for solvers.
- `RS-QA-DM-*` for design matrix and formula.
- `RS-QA-CV-*` for cross-validation and leakage prevention.
- `RS-QA-DIAG-*` for diagnostics and calibration.
- `RS-QA-MN-*` for multinomial.
- `RS-QA-EXP-*` for exports.
- `RS-QA-SER-*` for serialization.
- `RS-QA-VAL-*` for validation and errors.

Each requirement should record:

- Behaviour being protected.
- Primary tests.
- Oracle or invariant used.
- Required tolerance.
- Known exclusions or residual risk.

## Test Data Standard

Test data should be deliberate, not incidental.

Required fixtures:

- Small exact fixtures for hand calculations.
- Medium deterministic synthetic actuarial fixtures.
- High-cardinality categorical fixtures.
- Rare-level and unseen-level fixtures.
- Frequency data with independent exposure.
- Severity data with strictly positive response.
- Overdispersed count data with known dispersion.
- Imbalanced binary and multinomial classification data.
- Ill-conditioned regression data.
- Deployment fixtures for export parity.

Fixtures should be generated from deterministic code or stored as reviewed
golden artefacts. Random seeds must be visible in the test or fixture.

## Failure Triage Standard

Every test failure should answer:

- What invariant or user contract failed?
- Is the failure numerical tolerance drift, oracle drift, API drift, or a real
  behavioural regression?
- Which component owns the failure?
- Is the failing seed or fixture preserved?
- Does the fix require a new regression test?

Flaky tests are release blockers until proven environmental and quarantined
with an owner and expiry date.

## Documentation Requirements

The testing documentation must include:

- How to run each test layer locally.
- How to regenerate coverage reports.
- How to run mutation tests.
- How to replay fuzz failures.
- How to update golden fixtures.
- Tolerance policy.
- CI matrix and release gates.
- Dependency hygiene and lockfile review expectations.
- Current known gaps.

## Implementation Roadmap

Recommended order:

1. Add coverage tooling and CI thresholds.
2. Add dependency hygiene checks for lock freshness and yanked packages.
3. Add Python-version and OS matrix coverage.
4. Convert warning policy to explicit assertions.
5. Add GLM ONNX, PMML, and rate-table scoring parity tests.
6. Add deep smooth/spline tests and raise coverage.
7. Add adversarial property generators and deeper scheduled proptest runs.
8. Add mutation testing for families, solvers, diagnostics, validation, and export.
9. Add fuzz smoke jobs and preserved corpora.
10. Add requirement traceability tables.
11. Add performance and memory release gates.

## Definition of Done

The test suite reaches the target standard when:

- All mandatory gates are enforced in CI.
- Coverage thresholds are met or explicitly waived.
- Mutation scores meet thresholds for critical modules.
- Exported artefacts are scored against native predictions.
- Supported Python and OS combinations are tested.
- Warning contracts are explicit.
- Dependency hygiene is enforced for the checked-in lockfile.
- Numerical tolerances are documented and justified.
- Every high-risk actuarial behaviour has traceable test evidence.
- Failing seeds and golden fixtures are reproducible.
- Known residual risks are documented with owners.
