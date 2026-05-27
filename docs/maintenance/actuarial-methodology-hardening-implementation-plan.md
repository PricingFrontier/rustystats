# Actuarial Methodology Hardening — Implementation Plan

Status: proposed
Companion to: [`actuarial-methodology-hardening.md`](./actuarial-methodology-hardening.md)
Approach: strict TDD (red → green → refactor), small reviewable PRs, fail-closed first.

This plan turns the 11 RS-ACT items into an ordered, test-first delivery. Every
behaviour change is preceded by a failing test that pins the *intended* result,
and every refactor is fenced by characterization tests that pin *current* correct
behaviour so we can prove we didn't move it.

---

## 0. How to read this plan

- §1 fixes the engineering standard: gates, the TDD loop across the Rust/PyO3
  boundary, branching, and a Definition of Done that applies to **every** item.
- §2 is shared scaffolding (fixtures, characterization safety net, serialization
  versioning, cross-validation oracles) — built **before** any RS-ACT change.
- §3 is the dependency graph and the PR sequence.
- §4 is one focused sub-plan per item, in execution order, each with: Goal ·
  Current state (file:line) · **Tests first** · Implementation · Invariants ·
  Back-compat/serialization · Risks.
- §5/§6 are risks and the PR/milestone checklist.

`(R)` = Rust test (inline `#[cfg(test)]` or `tests/property_tests.rs`).
`(P)` = Python test (`tests/python/test_*.py`).

---

## 1. Engineering standard & workflow

### 1.1 Quality gates (mirror CI — these are non-negotiable per PR)

```bash
# Rust
cargo fmt --all -- --check
cargo clippy --workspace -- -D warnings      # new code must be clippy-clean
cargo test --workspace

# Python (after a rebuild)
uv run maturin develop
uvx ruff check python/
uvx ruff format --check python/              # separate gate from `ruff check`
uv run --extra dev pytest -x -q

# Diagnostics correctness (required for any diagnostics/calibration change)
uv run --extra dev maturin develop --release
uv run --extra dev python benchmarks/verify_diagnostics_correctness.py
```

Install the hooks once so fmt/clippy/ruff run automatically:
`uv run pre-commit install`.

### 1.2 The TDD loop across the Rust/PyO3 boundary

The friction unique to this repo: **a Rust change is invisible to pytest until
`maturin develop` re-runs.** So drive logic from the Rust side where possible.

- **Pure-Rust behaviour** (families, solvers, diagnostics math): red→green→refactor
  entirely in `cargo test -p rustystats-core <module>::tests` — fast, no rebuild.
- **Boundary/Python behaviour** (new kwargs, result fields, summaries):
  1. Write the failing `(P)` test.
  2. Implement Rust + update the PyO3 binding **and** `python/rustystats/_rustystats.pyi`.
  3. `uv run maturin develop` → `uv run --extra dev pytest tests/python/test_x.py::test_y -x`.
  4. Refactor; re-run.
- **Pre-PR**: run the full §1.1 sweep, including the diagnostics harness for
  diagnostics-touching items.

### 1.3 Branching & PRs

One PR per item (or per tightly-coupled pair), merged in the §3 dependency order,
**each independently green on all gates**. Stack them on a long-lived integration
branch `actuarial-hardening` and fast-forward to `main` at each phase boundary, or
merge per-item straight to `main` if the team prefers trunk-based. P0 items
(RS-ACT-001/002) ship first as fail-closed guards, so even a half-built feature is
safe in `main`.

### 1.4 Definition of Done (applies to every item)

- [ ] New tests written **first** and observed to fail before implementation.
- [ ] Both `(R)` and `(P)` coverage where the change crosses the boundary.
- [ ] All §1.1 gates green locally.
- [ ] `_rustystats.pyi` updated for any PyO3 signature change.
- [ ] `python/rustystats/__init__.py` exports updated for any new public symbol.
- [ ] Serialization round-trip test added for any stored-state change, **plus** an
      old-model load test (back-compat).
- [ ] Cross-validated against `statsmodels`/`glum` where a reference exists.
- [ ] Docs updated per the spec's Acceptance Criteria (§ list reproduced in PR14).
- [ ] No public API break without a legacy alias **and** a test proving the alias.
- [ ] PR description states the invariant the new tests prove.

---

## 2. Shared scaffolding (PR1 — build before any RS-ACT change)

No behaviour change. Establishes the safety net the rest of the plan leans on.

### 2.1 Deterministic fixtures

Add `tests/python/_fixtures.py` (and a thin `conftest.py` exposing them as
fixtures) plus a mirror Rust helper in `crates/rustystats-core/src/test_support.rs`
(gated `#[cfg(test)]`).

`make_freq_frame(n=4000, seed=0) -> pl.DataFrame` with:
- `Exposure` — strictly positive, **varying independently of risk** (so `mu` and
  `mu/Exposure` deciles genuinely disagree — needed by RS-ACT-004/009).
- `DrivAge`, `VehAge` — continuous, smooth true effect (splines, alpha path).
- `Region` — low-cardinality categorical.
- `Brand` — high-cardinality, with **one deliberately rare level** placed so it
  lands in a single CV fold (needed by RS-ACT-001 leakage test).
- `ClaimCount` — Poisson draws from a known `log`-rate + `log(Exposure)` offset.
- `ClaimAmt` — Gamma draws (severity, no exposure) for RS-ACT-004 negative cases.

A matching `make_severity_frame` (Gamma, `y>0`) and a `make_overdispersed_counts`
(for NB/quasi). Keep generators seeded and pure.

### 2.2 Characterization tests (regression safety net)

Before the two big refactors (exposure threading, fold-safe CV / fit-transform
split), capture current correct behaviour as golden values:

- `(P)` Plain Poisson + `offset="Exposure"`: fitted values, coefficients, deviance,
  AIC → assert unchanged after RS-ACT-002.
- `(P)` Target encoding **without** CV: encoded column values → assert unchanged
  after the fit/transform split (RS-ACT-001b).
- `(P)` Gaussian lasso path on a fixed matrix → snapshot selected support per alpha
  (RS-ACT-005 must only change the *grid endpoints*, not the solver).
- `(P)` Diagnostics deciles **without** exposure → snapshot ordering (RS-ACT-004
  must not touch the no-exposure path).

Store goldens inline (small) or as `tests/python/_golden/*.json`.

### 2.3 Serialization versioning

Introduce an explicit `schema_version` in the serialized model state now, with a
loader that maps "absent" → v0 (legacy). Every later item that adds stored state
bumps the version and adds: (a) a v_new round-trip test, (b) a v0 (legacy) load
test. This makes RS-ACT-002 (exposure/offset spec) and RS-ACT-011 (inference/solver
status) back-compat work mechanical instead of ad hoc.

### 2.4 Cross-validation oracles

`statsmodels` and `glum` are dev deps. Standardise helper assertions:
`assert_close_to_statsmodels(...)` (Poisson/Gamma/NB GLM coefficients, AIC, SE) and
`assert_close_to_glum(...)` (elastic-net path). Used as golden checks in
RS-ACT-003/005/006/010. Extend `benchmarks/verify_diagnostics_correctness.py`
rather than bypassing it for RS-ACT-004/009.

---

## 3. Dependency graph & PR sequence

```
PR1  Scaffolding (§2)
        │
Phase 1 (fail-closed safety; P0)
PR2  RS-ACT-002a  exposure= kwarg + stop array-offset-as-TE-exposure (fail-closed/warn)
PR3  RS-ACT-001a  CV+TE guard · weighted CV scoring · explicit CV convergence
        │
Phase 2 (correct P0 functionality)
PR4  RS-ACT-002b  full exposure threading: predict + diagnostics + serialization
PR5  RS-ACT-001b  fold-safe TE CV  (needs 002b for fold exposure)
PR6  RS-ACT-003   smooth GLM family params (var_power/theta passthrough)
PR7  RS-ACT-010   NB theta contract            (pairs with 003)
        │
Phase 3 (diagnostics & solver hardening; P1/P2)
PR8  RS-ACT-007   IRLS step acceptance + final clamp + emit solver status
PR9  RS-ACT-011   honest inference + surface solver status   (needs 007's status)
PR10 RS-ACT-004   rate-ranked diagnostics       (needs 002b exposure)
PR11 RS-ACT-009   calibration primitives        (needs 002b + 004)
PR12 RS-ACT-005   GLM-score alpha max           (independent)
PR13 RS-ACT-006   Tweedie support contract      (independent)
PR14 RS-ACT-008   quasi-likelihood labels       (independent)
PR15 Docs + release notes + migration examples (Acceptance Criteria §3/§5)
```

PR12/13/14 are independent and can run in parallel with the 007→011 / 004→011
chains. The only hard ordering is **002b before 001b/004/009** and **007 before
011**.

---

## 4. Per-item plans

### PR2 — RS-ACT-002a: `exposure=` kwarg + fail-closed exposure-as-offset (P0)

**Goal.** Introduce the explicit `exposure` concept and stop the silent
array-offset-as-TE-denominator bug, without yet doing the full prediction/
serialization rework.

**Current state.** `_process_offset` logs a string offset, passes an array offset
through (`formula.py:724-738`). `_get_raw_exposure` returns an array offset
*verbatim* as "raw exposure" (`formula.py:820-830`) → flows into TE denominator
(`interactions.py:1360-1362` → `target_encoding/mod.rs:60-61,335-338`). `glm_dict`
has no `exposure` param (`formula.py:3021-3032`).

**Tests first.**
- `(P)` `glm_dict(..., exposure="Exposure")` and legacy `offset="Exposure"` produce
  **identical** fitted values (spec test 002.1). *Red until `exposure=` exists.*
- `(P)` `offset=np.log(exposure)` **without** `exposure=` + a `target_encoding`
  term → raises in strict mode / warns + sets `used_exposure_weighted=False`
  otherwise (spec test 002.3). *This is the bug, pinned.*
- `(P)` Exposure-weighted TE denominator equals `sum(y)/sum(exposure)`, never
  `sum(y)/sum(offset)` (spec test 002.4) — assert against a hand-computed value.
- `(P)` Validation: non-positive / non-finite / wrong-length exposure raises clearly.

**Implementation.**
1. Add `exposure=None` to `glm_dict`/`FormulaGLMDict.__init__`; normalize legacy
   `offset="Exposure"` (log-link) → `exposure="Exposure", offset=None` and record
   `_offset_is_legacy_exposure_alias=True`.
2. Replace `_get_raw_exposure`: raw exposure comes **only** from the `exposure`
   spec, never from an array offset. If a TE term needs exposure weighting and no
   `exposure` is set: raise (strict) or warn + fall back to unweighted.
3. Validate exposure (finite, `>0`, length) at construction.

**Invariants.** Legacy alias is behaviour-preserving (a string offset was always
logged, so there is no non-logged string path to break). Rule (4) intentionally
**changes results** for `array-offset + exposure-weighted TE` — call this out in
the PR as a deliberate correctness fix.

**Risk.** Users relying (unknowingly) on the buggy denominator get different
numbers. Mitigation: the warning message names the fix and points to `exposure=`.

---

### PR3 — RS-ACT-001a: CV fail-closed guard, weighted scoring, explicit convergence (P0)

**Goal.** Make CV safe *before* the fold-safe refactor lands: block the leaky
combination, fix the scoring objective, and stop hidden relaxed convergence.

**Current state.** TE built from full `y`/exposure (`interactions.py:1352-1362`);
CV slices the prebuilt `X` (`regularization_path.py:396-397,471`; `fitting_py.rs`
copies rows per fold). Validation deviance is an **unweighted** per-row mean
(`fitting_py.rs:556-557`; `regularization_path.py:498`). Relaxed convergence is
injected silently (`regularization_path.py:422-425`:
`cv_max_iter=min(max_iter,10)`, `cv_tol=max(tol,1e-4)`).

**Tests first.**
- `(P)` CV requested **and** any target-encoded term/interaction present → raises a
  clear, actionable error (spec test 001.8). *Red until guard exists.*
- `(P)` CV validation score **changes** when validation weights change (spec test
  001.4) — currently it doesn't, because weights never reach the score.
- `(P)` CV uses the requested `max_iter`/`tol` (or records a named relaxed mode in
  result metadata) — assert no silent `10`/`1e-4` (spec test 001.6/001.7).
- `(R)` `fit_cv_path_py` validation score equals `Σ w_i dev_i / Σ w_i` on a tiny
  fixture with non-trivial weights.

**Implementation.**
1. Add the fail-closed guard in the CV resolution path (raise unless fold-safe is
   available — removed in PR5).
2. Thread `weights_val` into the score: `score = Σ w_i dev_i / Σ w_i` (Rust
   `fitting_py.rs` + Python unregularized branch `regularization_path.py:498`).
   `w_i=1` when no prior weights; offsets/exposure stay in the prediction, not the
   denominator.
3. Make CV convergence settings explicit: default to the requested settings; if a
   fast mode is offered it must be named and recorded in `RegularizationPathInfo`.
4. Record CV metadata scaffold (fold scores, mean, SE, objective, convergence) —
   populated fully in PR5.

**Invariant.** Validation deviance excludes the regularization penalty (spec 001.7).

---

### PR4 — RS-ACT-002b: full exposure threading (predict, diagnostics, serialization) (P0)

**Goal.** Make `exposure` first-class end-to-end.

**Current state.** Predict offset resolution at `formula.py:957-978`; diagnostics
exposure resolution only adopts a **string** offset (`diagnostics/api.py:272-277`),
else unit exposure; serialization stores `offset_spec`/`offset_is_exposure`
(`formula.py:2021-2022`, loaded `:2091-2092`).

**Tests first.**
- `(P)` `eta_total = eta_terms + log(exposure) + offset`; both supplied → both add
  (spec test 002.2: `offset=log(e), exposure=e` == `exposure=e` only when offset
  is otherwise zero; construct the additive case explicitly).
- `(P)` Predict with stored exposure column / with a new exposure array / clear
  error when required-but-absent (spec tests 002.5-7).
- `(P)` Diagnostics & `explore` source exposure from the `exposure` spec, **not** a
  string offset (locks the forward-looking inverse hazard from the spec).
- `(P)` Serialization round-trips `exposure_spec`/`offset_spec`; an old
  `offset_is_exposure=True` model loads as a legacy exposure model (spec 002.8 +
  back-compat). Bump `schema_version`.
- Non-log link + `exposure=` → rejected by default `(P)`.

**Implementation.** Store `_exposure_spec`/`_offset_spec`/
`_offset_is_legacy_exposure_alias` separately; resolve both at fit/predict;
route diagnostics/`explore`/complement-division off `_exposure_spec`; extend
`__getstate__`/`__setstate__` and the Rust serialization with the new fields +
"did `log(exposure)` enter the fit" / "did TE use exposure weighting" flags.

---

### PR5 — RS-ACT-001b: fold-safe target-encoding CV (P0)

**Goal.** Remove the leak; remove the PR3 guard.

**Approach.** **Option A (preferred)** — split `InteractionBuilder` into
`fit_design_matrix_state(parsed, data, y, exposure, seed) -> state` and
`transform_design_matrix(parsed, data, state) -> X, names`. The transform half
largely exists already (`transform_new_data`, `interactions.py:1706,2060-2097`,
which already maps unseen levels to the stored prior). Prediction reuses the same
state. *If schedule pressure hits, Option B (per-fold rebuilt `InteractionBuilder`)
is the documented fallback and satisfies the same tests.*

**Tests first.**
- `(P)` Rare level isolated to one fold: its validation encoding equals the
  **fold-training prior**, not the full-data rate (spec test 001.1). *The leakage
  test — must be red against today's code.*
- `(P)` Fold-safe CV matches a hand-written manual fold calculation (001.2).
- `(P)` Target-encoded **interactions** are fold-safe too (001.3).
- `(P)` Exposure-weighted TE in CV uses fold-training claims & fold-training
  exposure only (spec 001.5 + ties to PR4).
- `(P)` No-TE models still use the fast array path (001.5/regression vs §2.2
  characterization golden).
- `(R)` `fit_design_matrix_state`/`transform_design_matrix` round-trip equals the
  one-shot builder on full data (proves the refactor is behaviour-preserving).
- Remove the PR3 guard; assert the previously-raising call now runs.

**Invariant.** Production prediction still uses the final full-training state.
Populate the CV metadata scaffold from PR3.

---

### PR6 — RS-ACT-003: smooth GLM family parameters (P0/P1)

**Goal.** Smooth and non-smooth fits use the same `var_power`/`theta`.

**Current state.** `fit_smooth_glm_unified_py` hard-codes
`family_from_name(family, 1.5, 1.0)` (`fitting_py.rs:655`, no params in signature
`:629`); the Python smooth path drops `self.var_power`/`self.theta`
(`formula.py:509-526`, `:405-419`). The NB family string is built from the
*requested* theta (`formula.py:2992`) → currently **actively misleading** for
smooth NB.

**Tests first.**
- `(R)` `fit_smooth_glm_unified_py` with `var_power=1.2` constructs a Tweedie with
  `var_power=1.2` (assert via family metadata/variance function, not coefficients).
- `(P)` Smooth fit at the **same** `var_power`/`theta` agrees with the non-smooth
  fit (robust assertion; avoids the flaky "differs from default" — per the spec's
  revised test 003.1).
- `(P)` Returned family metadata records the actual `var_power` **and** `theta`,
  for both Tweedie and NB (spec 003.3) — pins the misleading-string bug.
- `(P)` Smooth NB with `theta=3.0` differs from `theta=1.0` (003.2).

**Implementation.** Add `var_power`/`theta` to the PyO3 signature → pass to
`family_from_name`; pass `self.var_power`/`self.theta` from both smooth dispatch
sites; build the result family string from the **used** value; update `.pyi`.

---

### PR7 — RS-ACT-010: Negative Binomial theta contract (P1)

**Goal.** No silent `theta=1.0`. Wire the existing-but-unused estimator.

**Current state.** Dict path falls back to `DEFAULT_NEGBINOMIAL_THETA=1.0`
(`formula.py:2968`, `constants.py:96`); reg path too (`regularization_path.py:401`).
`fit_negbinomial_py` implements full profile estimation (`fitting_py.rs:229-377`,
offset/weights-aware) but has **zero Python callers**.

**Tests first.**
- `(P)` `family="negbinomial"` with no theta does **not** silently fit `theta=1.0`
  (either estimates or errors per chosen policy — spec 010.1). *Red today.*
- `(P)` `theta="estimate"` estimates and records metadata (estimated-vs-fixed,
  init, iterations, convergence, tolerance) (010.2).
- `(P)` numeric `theta` recorded as fixed (010.3); offset/weights respected
  (010.4); cross-check estimated theta vs `statsmodels` NB.
- `(P)` Unsupported combos (regularized/constrained/**smooth** NB + `estimate`)
  raise clearly unless a fixed numeric theta is given (010.5) — this is the
  RS-ACT-003 cross-constraint (smooth solver has no theta loop).
- `(P)` Reg-path policy is explicit (fixed-only / per-fold / fail-closed) (010.6).

**Implementation.** Accept `theta: float | "estimate" | None`; route the
non-smooth/unregularized/unconstrained estimate case to `fit_negbinomial_py`;
surface theta metadata on the result; build the family string from the used theta.
Start with explicit opt-in (`theta="estimate"`); defaulting is an Open Design
Decision.

---

### PR8 — RS-ACT-007: IRLS step acceptance + final clamp + solver status (P1)

**Goal.** Never accept a worse-than-previous step; clamp the final `mu`; emit
solver status.

**Current state.** Unconstrained step-halving falls through and **unconditionally
accepts the last (worse) step** (`irls.rs:623-638,642-644`); only the *constrained*
path has `best_*` tracking (`:664-671`). Final `mu = link.inverse(eta)` is
**unclamped** (`irls.rs:797-804`), unlike every in-loop computation (`:573,605,631`).
No warning is pushed on non-convergence. `clamp_mu` exists on all families.

**Tests first.**
- `(R)` Constructed case where the full step worsens deviance and a half step
  improves it → the half step is accepted (007.1).
- `(R)` Constructed case where no step improves → previous coefficients retained,
  status = failed/`step-halved-no-improvement` (007.2).
- `(R)` Final fitted `mu` lies in each family's clamp domain (e.g. binomial
  `mu∈(0,1)`) (007.3).
- `(R)` **Plateau guard** (from the spec revision): a fit that converges with the
  full step within tolerance is *accepted and flagged converged*, not failed.
- `(R)` Deviance is monotone non-increasing across accepted iterations
  (property test in `tests/property_tests.rs`).
- `(P)` Python result exposes a useful convergence warning/status (007.4).

**Implementation.** Unify the accept logic: try full step; if deviance worsens
beyond a small relative tolerance or is non-finite, halve; accept best valid step;
on budget exhaustion retain previous coefs + set status. Extend the constrained
`best_*` pattern to the unconstrained path. Final value
`mu = family.clamp_mu(link.inverse(eta))`. Fix the convergence flag so a
worse-but-close step can't be reported as converged. Emit the status fields
(accepted/rejected/retained, step-halving used) consumed by PR9.

---

### PR9 — RS-ACT-011: honest inference + solver status surfacing (P1)

**Goal.** Stop presenting ordinary p-values/SEs/AIC/BIC as valid after
regularization/selection/constraints; expose `inference_status` + solver status.

**Current state.** Summary unconditionally prints SE/p/stars/AIC/BIC
(`glm.py:62-159`); PyO3 inference is penalty-agnostic (`results_py.rs:293-371`;
`inference/mod.rs`); AIC/BIC use raw `n_params` (`model_fit.rs:219-235`). EDF for
smooth terms is **already computed** (`gcv_optimizer.rs:250-267`, surfaced as
`total_edf`). `_DeserializedResult` does **not** round-trip status
(`formula.py:1975-2076`).

**Tests first.**
- `(P)` Unpenalized GLM still shows standard inference (011.1) — guards against
  over-suppression.
- `(P)` Lasso/elastic-net summary does **not** present ordinary p-values as valid;
  stars hidden; `inference_status == naive_after_selection` (011.2).
- `(P)` CV-selected fit (incl. pure ridge) → `naive_after_cv_selection` (or folded
  into `naive_after_selection`), never `valid_standard` (spec revision; 011.3).
- `(P)` Constrained fit marks boundary-active coefficients (011.4).
- `(P)` Penalized-spline summary uses effective df, not basis-column count (011.5).
- `(P)` Solver status records step-halving/convergence/route (011.6; ties to PR8).
- `(P)` Serialization preserves inference + solver-status metadata (011.7) — bump
  `schema_version`, add to both `__getstate__`/`__setstate__` and
  `_DeserializedResult`.

**Implementation.** Add an `inference_status` taxonomy (`valid_standard`,
`valid_robust`, `naive_after_regularization`, `naive_after_selection`,
`naive_after_cv_selection`, `constrained_boundary`, `unavailable`) set at fit time
from penalty/selection/constraint state. Gate the summary's stars/p-values/AIC-BIC
on it; default unknown/smooth fits to a conservative (non-`valid_standard`) status.
Surface solver-status fields and `optimizer_route` (`irls`/`coordinate_descent`/
`nnls`/`gcv_penalized` — all real solvers).

---

### PR10 — RS-ACT-004: rate-ranked diagnostics (P1)

**Goal.** Rank by `mu/exposure` when exposure is present; keep count-scale A/E.

**Current state.** Decile/calibration/Python-lift sort by raw `mu`
(`decile.rs:63-67`, `calibration.rs:96,433`, `computer.py:876`), **but**
discrimination stats (Gini/lift@X/KS) **already** rate-rank
(`calibration.rs:282-311`) — an internal inconsistency the fix must harmonize.

**Tests first.**
- `(P)` Fixture where raw `mu` and `mu/exposure` give **different** decile order
  (004.1) — §2.1 generator guarantees this.
- `(P)` Calibration bins rate-rank but aggregate counts: `actual=Σy`, `expected=Σmu`
  (004.2).
- `(P)` Lorenz/lift rate-rank when exposure exists (004.3); Gini and the decile
  table now agree on ordering (harmonization regression).
- `(P)` No-exposure path preserves old ordering vs §2.2 golden (004.4).
- `(P)` Weighted aggregates `Σw·y / Σw·mu / Σw·exposure` correct (004.5).
- `(R)` `decile.rs`/`calibration.rs` accept a `ranking` mode and an external
  `risk_score`/`sort_idx`.

**Implementation.** Add `ranking="auto"|"mean"|"rate"` (auto = rate iff exposure
present). Switch **both** sort sites (Rust default + Python `np.argsort(self.mu)`)
to a rate argsort. Keep aggregates on the count scale. Extend
`verify_diagnostics_correctness.py` with a rate-ranking invariant.

---

### PR11 — RS-ACT-009: explicit calibration diagnostics & primitives (P1)

**Goal.** Add calibration summaries, `GlobalCalibration`, optional
`IsotonicCalibration`, and a log-link `relevel`.

**Current state.** Overall A/E + problem deciles exist but **unweighted**
(`components.py:81-109`); factor A/E exists (`factor_diagnostics.rs`). No
`calibration_summary`/`relevel`/`GlobalCalibration`/`IsotonicCalibration`/
`fit_calibration` anywhere. `predict` builds `mu=inverse_link(Xβ+offset+complement)`
(`formula.py:1736-1767`); relativities exclude intercept (`formula.py:1329,1394`).

**Tests first.**
- `(P)` Overall A/E `= Σy/Σpred`; weighted `= Σwy/Σwpred` (009.1-2).
- `(P)` Exposure models rate-rank bins under `ranking="auto"` (009.3; reuses PR10).
- `(P)` Calibration by factor aggregates actual/expected/exposure/A/E with
  min-exposure suppression (009.4).
- `(P)` `GlobalCalibration.predict(pred) == factor * pred`; `to_dict`/`from_dict`
  round-trip (009.5).
- `(R/P)` **Relevel invariant** (the load-bearing math): after log-link
  `result.relevel()`, (a) `β[1:]` bit-identical, (b) `Σ(w·μ_new) ≈ Σ(w·y)` within
  `rtol=1e-10`, (c) relativities unchanged (009.6-7). Property test over random
  fixtures.
- `(P)` **Weighted** relevel balances `Σ(w·μ_new)=Σ(w·y)` and reduces to unweighted
  when `w≡1` (spec's new test 009.11 — pins the resolved factor convention).
- `(P)` Relevel raises clearly for non-log links (009.8).
- `(P)` Isotonic is monotone, serialized separately, **not** applied implicitly;
  raw and calibrated predictions both accessible (009.9-10).

**Implementation.** Standalone array fns `rs.calibration_summary`,
`rs.fit_global_calibration`, `rs.fit_isotonic_calibration`; result/model
`calibration_summary`/`fit_calibration`/`relevel`. `relevel` factor
`c = Σ(w·y)/Σ(w·μ)`, `intercept_new = intercept_old + log(c)`; `exposure=` only
feeds `predict`, never `c`. Calibration objects serialize separately with metadata;
never fold into GLM coefficients. Subsume the existing unweighted overall A/E so
there's a single A/E path. Docstrings warn about in-sample calibration optimism.

**Invariant rationale.** Under log link, `μ→c·μ` for every row, so the weighted
balance holds exactly and relativities (ratios/`exp(β_j)`) are untouched.

---

### PR12 — RS-ACT-005: GLM-score-based alpha max (P1)

**Goal.** `alpha_max` from the GLM score at the offset/weight-aware null model.

**Current state.** `compute_alpha_max` uses centered raw `y`
(`regularization_path.py:161-182`), no family/link/offset — empirically wrong by a
factor of `n` (reproduced: 0.799 vs correct 479.6 on 600 rows). The solver uses raw
weighted sums, **no `1/n`** (`coordinate_descent.rs:213-250`). Rust fallback grid is
score-blind (`fitting_py.rs:429-436`).

**Tests first.**
- `(P)` At `alpha = alpha_max`, **all penalized** lasso coefficients are zero
  (005.1) — currently fails (2 nonzero at the old value).
- `(P)` Offset distribution changes `alpha_max` (005.2); weights change it (005.3).
- `(P)` Gaussian identity matches **RustyStats' own** solver scaling — raw weighted
  sums, no implicit `1/n` (spec revision 005.4); cross-check against `glum` at
  matched scaling.
- `(P)` Poisson log-link `alpha_max` matches a finite-difference score check at the
  null model (005.5).
- `(R)` ridge (`l1_ratio==0`) uses a documented grid, not an all-zero-derived max.

**Implementation.** Helper computes the offset/weight-aware intercept-only `mu0`,
then per-feature score = gradient of the unpenalized GLM loss at the null, using the
**same loss scaling as the solver** (raw weighted sums); exclude intercept +
unpenalized columns; `alpha_max = max_j|score_j|/l1_ratio`. Use in both Python path
generation and the Rust fallback.

---

### PR13 — RS-ACT-006: Tweedie support contract (P1)

**Goal.** Default to `1 < p < 2`; gate other powers behind
`allow_extended_tweedie`; reject the genuinely invalid region always.

**Current state.** Only `0<p<1` rejected (`tweedie.rs:81-90`,
`families_py.rs:145-150,410-419`); `validation.py:229-234` only checks `y>=0`. p=2
with zeros silently returns `deviance=inf`; p=2.5 with zeros returns a *negative*
deviance. No `allow_extended_tweedie` param.

**Tests first.**
- `(R/P)` `0<p<1` errors **always**, even with `allow_extended_tweedie=True`
  (mathematically invalid) (006.1).
- `(R/P)` `p<=0, ==1, ==2, >2` error by default, accepted under extended mode
  (006.2).
- `(R)` `1<p<2` accepts zeros and positives (006.3).
- `(R)` `p==2` routes to Gamma / errors clearly and **rejects exact zeros** rather
  than returning infinite deviance (006.4) — pins the `deviance=inf` bug.
- `(R)` extended `p>=2` rejects exact zeros (`y>0` required) (006.5).

**Implementation.** Add `allow_extended_tweedie=False` to the public API; enforce
the regime table in `tweedie.rs` + `validation.py` + `families_py.rs`; fix
`unit_deviance` zero-handling so out-of-support powers error instead of producing
inf/negative deviance.

---

### PR14 — RS-ACT-008: quasi-likelihood labels (P2)

**Goal.** Stop reporting quasi AIC/BIC/loglik as ordinary likelihood values.

**Current state.** Quasi families return ordinary Poisson/binomial loglik
(`quasi.rs:140-148,228-236`) → plain AIC/BIC (`results_py.rs:673-696`,
`model_fit.rs:219-236`); summary prints them unlabeled (`glm.py:117-119`). φ is
computed separately (Pearson χ²/df_resid), so the family loglik can stay for
convergence.

**Tests first.**
- `(P)` Quasi-Poisson summary does not present AIC/BIC as ordinary likelihood
  values; returns `None`/`NA` (008.1).
- `(P)` Quasi-binomial behaves the same (008.2).
- `(P)` Serialization preserves the quasi diagnostic flag (008.3).

**Implementation.** Add a quasi flag to the result; `aic()`/`bic()` return
`None` for quasi families (unless a documented QIC is added later); the summary
labels the loglik-like value as quasi-likelihood.

---

### PR15 — Documentation, release notes, migration

Update per Acceptance Criteria §3: `docs/api/dict-api.md` (`exposure`),
`docs/components/target-encoding.md` (fold-safe CV + exposure),
`docs/components/diagnostics.md` (rate ranking), `docs/api/diagnostics.md`
(calibration), `docs/theory/regularization.md` (alpha max),
`docs/theory/families.md` (Tweedie support; fixed-vs-estimated NB theta),
`docs/api/results.md` (inference caveats + solver status). Release notes carry the
three migration spellings (preferred `exposure=`, link-scale offset, legacy alias)
and call out the two deliberate behaviour changes (RS-ACT-002 rule 4; RS-ACT-005
alpha grid).

---

## 5. Risks & mitigations

| Risk | Item | Mitigation |
| --- | --- | --- |
| Silent result changes for existing users | 002a, 005 | Characterization goldens (§2.2); release-note call-outs; fail-closed/warn before change. |
| Fit/transform refactor regresses non-TE models | 001b | `(R)` round-trip equivalence test + §2.2 goldens; Option B fallback. |
| Serialization back-compat breakage | 002b, 009, 011 | `schema_version` (§2.3); explicit v0-load tests per item. |
| `theta="estimate"` unstable for smooth/regularized NB | 010 | Fail-closed for those combos; opt-in first, default later (Open Design Decision 8). |
| Flaky `var_power` coefficient test | 003 | Assert metadata / same-`var_power` agreement, not "differs from default" (spec revision). |
| clippy `-D warnings` on new structs | 007, 009, 011 | Add `Default` impls; run clippy in the inner loop, not just pre-PR. |
| Diagnostics harness drift | 004, 009 | Extend `verify_diagnostics_correctness.py`; it's a required CI job. |

## 6. Milestone checklist

- [ ] **M0** PR1 merged — fixtures, characterization goldens, `schema_version`.
- [ ] **M1 (P0 safety)** PR2-3 — exposure kwarg, no array-offset-as-TE, CV+TE guard,
      weighted CV scoring, explicit CV convergence.
- [ ] **M2 (P0 correctness)** PR4-7 — full exposure threading, fold-safe CV, smooth
      family params, NB theta.
- [ ] **M3 (solver/inference)** PR8-9 — IRLS hardening + honest inference.
- [ ] **M4 (diagnostics)** PR10-11 — rate ranking + calibration primitives.
- [ ] **M5 (independent P1/P2)** PR12-14 — alpha max, Tweedie support, quasi labels.
- [ ] **M6** PR15 — docs, release notes, migration; full Acceptance Criteria met:
      `cargo test` + `uv run pytest tests/python/ -v` green, harness green, old
      offset-as-exposure models load.
```
