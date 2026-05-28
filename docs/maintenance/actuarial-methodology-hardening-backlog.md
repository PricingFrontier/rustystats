# Actuarial Methodology Hardening — Backlog

Status: open
Scope: RustyStats only
Created: 2026-05-28
Companion to: [`actuarial-methodology-hardening.md`](./actuarial-methodology-hardening.md),
[`actuarial-methodology-hardening-implementation-plan.md`](./actuarial-methodology-hardening-implementation-plan.md)
Branch context: `actuarial-hardening`

Deferred follow-ups identified during the multi-agent review remediation (the
fix/test commits on `actuarial-hardening`). **None are correctness bugs** — the
shipped behaviour is correct and all gates (`cargo test`, `clippy`,
`ruff`, `pytest`) are green. These are consistency, feature, and
methodological-enhancement items intentionally left out of scope so the
remediation stayed proportionate and low-risk.

Priority is rough: **P1** = a real feature worth doing; **P2** = niche
consistency; **P3** = methodological enhancement / very low marginal value.

---

## 1. Weighted decile/lift aggregates in diagnostics (P1 — feature)

**Origin:** RS-ACT-004 review finding (PR10).

**Current state.** `DiagnosticsComputer.compute_lift_chart`,
`_compute_ae_by_decile`, and `result.diagnostics()` produce **unweighted**
per-bin aggregates (Σy / Σμ / Σexposure); exposure enters only as the rate
denominator and for rate-ranking, never as a prior-weight (Σw) multiplier. This
is now documented (a Notes block on `compute_lift_chart`), and weighted A/E is
available via `rs.calibration_summary(..., weights=)`.

**Deferred.** Thread an optional `weights=` through
`DiagnosticsComputer.__init__` / `compute_lift_chart` / `_compute_ae_by_decile`
(and propagate the model's fitted prior weights from `result.diagnostics()`) so
the lift/decile tables themselves report Σw·y / Σw·μ / Σw·exposure. Add a test on
the rate-vs-count fixture asserting the weighted totals.

**Files.** `python/rustystats/diagnostics/computer.py`,
`python/rustystats/formula.py` (`diagnostics()`),
`tests/python/test_rate_ranked_diagnostics.py`.

---

## 2. Releveled-intercept inference in the diagnostics coefficient summary (P2 — consistency)

**Origin:** RS-ACT-009 relevel finding (PR11); follow-on to the intercept-shift
commit.

**Current state.** After `relevel()`, the intercept's CI/z/p are recentred on the
shifted estimate (the `log(c)` offset is treated as known; SE unchanged) in
`coef_table()`, `relativities()`, and `summary()`. The shift is applied *locally*
in those three display methods.

**Deferred.** `diagnostics().coefficient_summary` (and the factor-significance
path) read the raw `_result` accessors directly, so for a releveled
`valid_standard` model they still show the **un-shifted** intercept row. Niche
(relevel + diagnostics together). Full consistency would either (a) apply the
same `+log(c)` shift in `compute_coefficient_summary`, or (b) move the shift to
GLMModel-level `conf_int`/`tvalues`/`pvalues`/`significance_codes` overrides —
which then requires fixing the `hasattr(result, "bse"/"pvalues")` capability
guards in `diagnostics/factors.py` so deserialized models (whose
`_DeserializedResult` lacks those methods) still skip correctly. Those guards
were deliberately left intact during remediation precisely to avoid that
breakage.

**Files.** `python/rustystats/diagnostics/computer.py`,
`python/rustystats/diagnostics/factors.py`, `python/rustystats/formula.py`.

---

## 3. Robust-SE intercept inference after relevel (P2 — consistency)

**Origin:** RS-ACT-009 relevel finding (PR11).

**Current state.** The intercept shift covers the **model-based** SE/CI/z/p only.

**Deferred.** The robust (HC1) accessors (`conf_int_robust`, robust p-values)
would also need the same `+log(c)` shift on the intercept row for a releveled
model; they are currently un-shifted (stale) there. Niche.

**Files.** `python/rustystats/formula.py`, `crates/rustystats/src/results_py.rs`.

---

## 4. Propagate calibration uncertainty into the releveled intercept SE (P3 — methodological)

**Origin:** RS-ACT-009 relevel finding (PR11).

**Current state.** `relevel()` treats the factor `c = Σ(w·y) / Σ(w·μ)` as a
*known* additive `log(c)` offset, so the intercept SE is unchanged and its CI is
slightly optimistic (it excludes the variance of `ĉ`). `summary()` discloses
this in a note.

**Deferred.** Optionally estimate `Var(log ĉ)` and add it to the intercept
variance so the recentred CI reflects the calibration step's own uncertainty.
Enhancement, not a defect — and it would couple the inference to the calibration
data, so it needs a deliberate API decision.

**Files.** `python/rustystats/formula.py` (`relevel`, intercept inference).

---

## 5. Hand-computed fold oracle for the weighted Rust CV scorer (P3 — test-only)

**Origin:** RS-ACT-001 review finding (PR3).

**Current state.** The weighted validation score (Σw·dev / Σw) is pinned by: two
Rust unit tests on `validation_deviance_score`, the Python `compute_deviance`
helper test, and a uniform-weight-invariance test that calls `fit_cv_path_py`
directly (`test_weighted_cv_score_is_normalized_by_sum_of_weights`).

**Deferred.** A full hand-computed *per-fold* oracle is blocked because the Rust
fold split uses `DefaultHasher`, which isn't reproducible from Python. A
Rust-side integration test that reuses the internal fold assignment could pin the
exact Σw·dev/Σw per fold. Very low marginal value given the existing coverage.

**Files.** `crates/rustystats/src/fitting_py.rs` (tests),
`tests/python/test_cv_safety.py`.
