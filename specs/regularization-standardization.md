# Regularization Standardization — Spec (RS-ACT-012)

Regularized GLM fits (`ridge` / `lasso` / `elastic_net`, with or without CV) currently apply a **single global penalty strength to raw-scale design columns**. Because rustystats builds the design matrix itself (spline bases, one-hot expansions, interaction products) and never standardizes it, a single `alpha` cannot shrink columns whose natural scales differ by many orders of magnitude. The fix: **standardize the penalized columns before the penalty acts, fit, then back-transform coefficients and covariance to the original scale** — the glmnet/sklearn default. This spec adds a `standardize` option (default **on**) and the machinery to make it correct and cheap.

This is a **correctness fix**, not a feature: at present a routine penalized Poisson GLM can silently collapse to an intercept-only model (see §2). It changes the coefficients/`alpha` of regularized fits, so it is a documented behaviour change with an escape hatch.

---

## 1. Design Principle

- **Standardize internally, report on the original scale.** Match glmnet/sklearn: the caller never sees standardized coefficients. `params`, `bse`, `conf_int`, robust SEs, `predict`, `relativities`, ONNX/PMML export — all return original-scale quantities, unchanged.
- **One source of truth for the scales.** The Python orchestration layer (which owns the design matrix and computes `alpha_max`) computes the per-column centering/scaling vectors for each design context and passes them to every fit entry point. For the fast CV path this is once on the full-data design; for fold-rebuilt designs such as target-encoding CV it is once per fold-training design, then once on the full-data design for the final refit. No duplicated "weighted column sd" implementations that can drift.
- **Back-transform where the result is built.** The covariance and coefficients are finalised in the Rust core; do the de-standardization there so the ~20 Python accessors that read `self._result.*` need **zero changes**.
- **Scale-only fixes the bug; full standardization is solution-equivalent and better-conditioned.** The defect is heterogeneity of the weighted Gram diagonal `Σ wᵢ xᵢⱼ²`; dividing each penalized column by its scale equalizes it. Centering additionally improves conditioning and is glmnet-exact; both are specified, with exact back-transform formulae.
- **Don't touch the smooth-penalty path.** P-spline bases are O(1) by construction and carry a *structured* difference penalty `S = D'D`; per-column standardization would break it. Standardization applies to the parametric elastic-net/ridge/lasso path only (§6).

---

## 2. The Problem (measured)

Reproduced on a Poisson frequency GLM (freMTPL2, log link, exposure offset, `monotone_constraints`, `selection="1se"`, design 45000×107):

- The recipe emits raw-scale `linear` terms and **raw-product interaction columns** (`BonusMalus:Density:DrivAge`). The weighted Gram diagonal `(XᵀWX)ⱼⱼ` spans **~17 orders of magnitude**: O(1) spline/one-hot bases vs **3.6e17** for the triple-product.
- `compute_alpha_max` (`alpha_max = maxⱼ|Xⱼᵀ f(y,μ₀)| / l1_ratio`) is **anchored to the astronomical column** → `alpha_max ≈ 7e9`. The whole log-spaced path (down to `alpha_max·1e-4 ≈ 7e5`) sits far above what the O(1) columns can tolerate.
- Fractional ridge shrinkage `ρⱼ = λ(1−l1)/(XᵀWX)ⱼⱼ` spans **9.9e-9 … 1.7e10**: raw-scale columns **escape** the penalty (ρ≈1e-9); O(1) bases are **annihilated** (ρ≈1e8–1e10).
- Result at the selected α: **105 of 106 structured coefficients are exactly zero** (all 40 splines, all 25 categoricals, 40/41 interactions). The fitted model predicts a flat `exp(intercept)` across every decile (1.01× spread vs 7.4× in the data) — **a degenerate, intercept-only model**, shipped silently with no warning.

**Confirmation that standardization is the fix:** refitting the identical design with columns scaled to unit weighted-variance normalizes the selected α **5.6e9 → 570** and improves holdout Poisson deviance **−3.72%**; discrimination un-collapses from 1.0× to ≥2.36×. (Lower bound; measured with an unconstrained reconstruction.)

The library's own source already states the unmet contract — `crates/rustystats-core/src/regularization/mod.rs` (the `STANDARDIZATION` comment block): *"For regularization to work fairly across predictors: predictors should be standardized (mean=0, std=1), or use penalty weights inversely proportional to scale."* — and `solvers/irls.rs` emits *"Try standardizing continuous predictors"* on conditioning failure. The library asks the caller to do something the caller **cannot** do, because the library builds the columns.

---

## 3. Root Cause (exact locations)

| Layer | File / symbol | What it does today |
|---|---|---|
| α-grid anchor | `python/rustystats/regularization_path.py` → `compute_alpha_max` (≈L289), `generate_alpha_path` (≈L406) | `scores = X.T @ score_factor` on **raw** X; `alpha_max = max|scores|/l1_ratio`. Path = `logspace(alpha_max, alpha_max·1e-4)`. No scaling. |
| Ridge penalty | `crates/rustystats-core/src/solvers/irls.rs` → `fit_glm_unified` (L373); `l2_penalty`/`penalize_intercept` (≈L400–493) | Adds `l2_penalty` to each penalised `XᵀWX` diagonal — a single global value, raw columns. |
| L1/EN penalty | `crates/rustystats-core/src/solvers/coordinate_descent.rs` → `l1_penalty`/`l2_penalty` (L101–102), `pen_start` (L106), soft-threshold + `denom` (L293–300) | Soft-thresholds with a single global `l1_penalty`; `denom = xwx_jj + l2_penalty`. Raw columns. |
| Penalty defn | `crates/rustystats-core/src/regularization/mod.rs` → `Penalty`, `RegularizationConfig`, `l1_penalty()`/`l2_penalty()` | Documents standardization as the caller's job (unmet). `RegularizationConfig.penalty_weights` exists but is not wired through bindings or consumed by the solvers. |
| Bindings | `crates/rustystats/src/fitting_py.rs` → `fit_glm_py` (L121), `fit_negbinomial_py` (L243), `fit_cv_path_py` (L460) | No `standardize` / scale parameter. |
| Orchestration | `python/rustystats/formula.py` → `_resolve_cv_path` (L1016); CV drivers `fit_cv_regularization_path` (L594) / `fit_cv_te_regularization_path` (L831) in `regularization_path.py` | Fast non-TE CV builds one full-data design and slices folds; fold-safe TE CV rebuilds designs per fold and may have different fold column counts. Neither path scales columns. |

There is **no public** `standardize` or `penalty_factor` parameter anywhere in the API.

---

## 4. Chosen Approach (and the rejected alternative)

**Chosen: physical standardization with scales supplied by Python and back-transform inside the Rust fit.**

- Python computes per-penalized-column `center[j]` (weighted mean) and `scale[j]` (weighted sd) for the design being fit, and passes both vectors to (a) `compute_alpha_max` and (b) the matching Rust fit.
- Rust builds a *working* standardized design `x̃ⱼ = (xⱼ − center[j]) / scale[j]`, runs the existing IRLS / coordinate-descent unchanged (the penalty now acts on equally-scaled columns), then **maps coefficients and covariance back** to the original scale before constructing `GLMResults`. The stored design matrix (`store_design_matrix`) and `fittedvalues`/`linear_predictor` are unaffected (the linear predictor is invariant to the reparameterization).
- Full centering requires an intercept-like degree of freedom to absorb the shift. When `fit_intercept=False`, use scale-only standardization (`center=0`) so the no-intercept model class is preserved.

**Rejected: a single per-column `penalty_factor` applied to both L1 and L2 as the standardization mechanism.** It cannot reproduce standardization for elastic net: expressing glmnet's penalty on original coefficients gives `λ Σⱼ[(1−α)/2 · sⱼ² βⱼ² + α · sⱼ |βⱼ|]` — the L2 term scales with `sⱼ²` and the L1 term with `sⱼ`, so no single multiplier is correct when `0 < l1_ratio < 1` (exactly destyler's case). Physical column scaling is exact for every `l1_ratio`.

`penalty_factor` remains a useful power-user feature, but it is not required to fix RS-ACT-012. If implemented with this change, it must be wired explicitly as per-column L1 and L2 multipliers (or equivalent solver logic), not by multiplying the standardization scale. Factor `0` means "unpenalised" and must also exclude the column from `alpha_max`.

### 4.1 Back-transform (exact)

Let `S = diag(scale)` with `scale[intercept]=1`, `scale[j]=1` for unpenalised or constant columns, and let `m[j]=center[j]` (`m[intercept]=0`; constant columns use `m[j]=0` so they are not collapsed to zero). The fit returns `β̃`, `C̃` (unscaled covariance) on the standardized design. Define the linear map `A` (p×p):

```
βⱼ      = β̃ⱼ / scale[j]                         (j ≠ intercept)
β_inter = β̃_inter − Σⱼ (center[j] / scale[j]) · β̃ⱼ
```

i.e. `β = A β̃` with `A = diag(1/scale)` on the non-intercept block, `A[inter,inter]=1`, `A[inter,j] = −center[j]/scale[j]`. Then:

```
Cov(β) = A · C̃ · Aᵀ
```

The linear predictor `η = X̃β̃ = Xβ` is identical, so `fittedvalues`, `deviance`, `llf`, residuals need no change. Only `params` and `covariance_unscaled` are mapped; every SE/CI/p-value accessor derives from those and inherits the correction. Robust covariance uses both the stored original-scale design matrix and the back-transformed bread matrix, so it must be tested explicitly (§8) rather than assumed. **Scale-only variant:** set all `center[j]=0` ⇒ `A` is diagonal (`1/scale[j]`), no intercept cross-term — simpler, fixes the bug, slightly worse conditioning. Ship full standardization (center+scale) as default when an intercept is present; use scale-only when `fit_intercept=False`.

---

## 5. Public API

`FormulaGLMDict.fit()` / `glm_dict(...).fit()`:

```python
def fit(
    self,
    ...,
    regularization: str | None = None,
    standardize: bool = True,          # NEW — standardize penalised columns before the penalty
    ...,
): ...
```

- `standardize=True` (default): center+scale penalised columns internally, back-transform on return. **No effect when there is no penalty** (`regularization=None` / `alpha=0`) — the MLE is scale-invariant, so this is a guaranteed no-op there.
- `standardize=False`: legacy behaviour (raw-scale penalty). Provided as an escape hatch and for exact reproduction of pre-RS-ACT-012 fits.
- No-intercept fits (`intercept=False` / `fit_intercept=False`) use scale-only standardization even when `standardize=True`; centering without an intercept would change the model class.
- The intercept and any column flagged `nonneg`/`nonpos` keep their constraints; scaling by a **positive** `scale[j]` preserves coefficient sign, so monotonicity constraints remain valid after back-transform (assert this in tests).
- Optional follow-up API: `penalty_factor: np.ndarray | dict[str, float] | None = None`. If included in this ticket, it composes with standardization as a separate per-column penalty multiplier; it does **not** replace or alter `scale[j]`.

Selected-α reporting (`regularization_path`, `summary`) continues to report the α actually used; with `standardize=True` it will be a sane magnitude rather than 1e9. Add a one-line note to `summary()` output that standardization was applied. Direct `_rustystats` binding callers can pass `center`/`scale` explicitly; `None` retains legacy raw-scale behaviour for backward compatibility.

---

## 6. Implementation Plan

**A. Rust core — `fit_glm_unified` (`solvers/irls.rs:373`) and the coordinate-descent EN path (`solvers/coordinate_descent.rs`).**
1. Accept optional `standardization: Option<Standardization>` where `Standardization { center: Vec<f64>, scale: Vec<f64> }` (length p; intercept/unpenalised entries `center=0, scale=1`). Reject length mismatches and non-positive/non-finite scales.
2. When present, form a working matrix `X̃` (do **not** mutate the caller's X / the stored design). Run the existing solver on `X̃` — the penalty code (`l1_penalty`/`l2_penalty`, ridge diagonal, soft-threshold) is unchanged; it simply sees equalized columns.
3. After convergence, map `β̃ → β` and `C̃ → C` via §4.1 **before** building the result struct (`covariance_unscaled`, `params`). Guard `scale[j] > 0`; constant columns use `center=0, scale=1` and otherwise follow the existing penalty rules unless explicit `penalty_factor` support marks them unpenalised.
4. When the core fit is called with `standardization`, the `skip_covariance` path (CV folds) skips the covariance map but still back-transforms coefficients, because validation scoring and warm starts may consume original-scale coefficients. A CV wrapper may instead standardize each fold's train/validation matrices once up front, call the inner solver with `standardization=None`, and keep warm starts in standardized coordinates; this is equivalent for CV scoring because no fold coefficients are exposed publicly.
5. Warm starts entering a standardized fit are accepted on the public/original scale and must be transformed into standardized coordinates before iteration:

```
β̃ⱼ      = βⱼ * scale[j]                         (j ≠ intercept)
β̃_inter = β_inter + Σⱼ center[j] · βⱼ           (intercept fit only)
```

For scale-only/no-intercept fits, this reduces to `β̃ⱼ = βⱼ * scale[j]`.

**B. Bindings — `crates/rustystats/src/fitting_py.rs` + stubs.** Add optional `center`/`scale` to the `#[pyo3(signature=...)]` of `fit_glm_py` (L121), `fit_negbinomial_py` (L243), and `fit_cv_path_py` (L460); construct the `Standardization` and thread it into the core fit. `fit_cv_path_py` should standardize each fold's train/validation matrices once per fold before the alpha loop, then call the core fit without another standardization layer so warm starts stay in the fold's standardized coordinate system. Update `python/rustystats/_rustystats.pyi` in the same change. Default `None` ⇒ current behaviour (keeps existing direct callers working).

**C. Python orchestration — `regularization_path.py` + `formula.py`.**
1. New helper `compute_standardization(X, weights, pen_mask) -> (center, scale)`: weighted mean and weighted population sd per penalised, non-constant column; `center=0, scale=1` for the intercept, unpenalised columns, and any constant column. Weighted by the fit's prior weights (`w`); unweighted if none. Deterministic (no RNG).
2. `compute_alpha_max` (L289): when standardizing, compute `scores` on the **standardized** columns (`(X−center)/scale`), i.e. divide `scores[j]` by `scale[j]` (centering does not affect the score because `y−μ₀` is weighted-orthogonal to the intercept). For the ridge `l1_ratio==0` heuristic, the Gram diagonal becomes ≈ `Σw` for all columns ⇒ the heuristic auto-normalizes.
3. `fit_cv_regularization_path` (L594): compute `(center, scale)` **once** on the full-data design (matching the existing "build once, slice folds" architecture) and pass to `compute_alpha_max`, every fold fit, and the final full-data refit. Because this path slices one design with stable columns, reusing the full-data column scale is acceptable and consistent with its existing full-design alpha-grid architecture.
4. `fit_cv_te_regularization_path` (L831): compute `(center, scale)` separately for each fold-training design and use it for that fold's `compute_alpha_max` and candidate fits. This avoids validation-target leakage through target-encoded full-data columns and handles folds whose training designs have different column counts. Build the global alpha grid from the max fold `alpha_max`, as today. Compute full-data `(center, scale)` separately for the final refit after alpha selection.
5. `_resolve_cv_path` (`formula.py:1016`) and the non-CV regularized refit: plumb `standardize` through. Plumb `penalty_factor` only if the optional follow-up is implemented in this ticket.

**Leakage note (be explicit, don't hide it):** the fast non-TE CV path reuses full-data scales across folds, matching its existing "build once, slice folds" design. The fold-safe target-encoding path must not do that: it rebuilds response-dependent encodings on each fold's training rows, so standardization must follow the same fold-local boundary.

---

## 7. Scope / Non-goals

- **Smooth-penalty path is out of scope and must stay unchanged.** `fit_smooth_glm_full_matrix` (`solvers/smooth_glm.rs:568`) uses structured `S=D'D` penalties on O(1) spline bases. Do **not** standardize within smooth blocks. If `SmoothPenalty.parametric_l2` co-penalises parametric columns alongside smooth blocks, standardize only the parametric columns; otherwise skip standardization entirely when `Penalty::Smooth`. Add a test that GCV/REML results are identical with `standardize` toggled.
- **`penalty_factor` is optional/follow-up.** The correctness bug is fixed by physical standardization. Per-column penalty factors are useful, but they require explicit solver support and should not be approximated by altering `scale`.
- **Summed-vs-mean loss is a separate, optional improvement.** rustystats penalises the *summed* deviance (per the RS-ACT-005 note in `compute_alpha_max`), so even after standardization α scales with `Σw` rather than being O(1) like glmnet. This is interpretability, **not** the correctness bug — standardization fixes the cross-column unfairness regardless. Optionally (separate ticket) normalise the loss by `Σw` so α is comparable to glmnet; that is an additional behaviour change to α magnitude and is not required to fix §2.
- **Centering of the response / link offset:** unchanged.

---

## 8. Acceptance Criteria

1. **Affine invariance (the core property the bug violates).** Fit a Poisson EN GLM; refit with one input column multiplied by 1000. With `standardize=True`, fitted values and deviance are identical to ~1e-8, and the corresponding coefficient scales by exactly 1/1000. *(Today this fails — the fit changes wildly.)*
2. **α sanity.** On a design with one column ~1e4 alongside O(1) columns, the CV-selected α is O(`Σw`), not ≥1e9, and the large column no longer dictates `alpha_max`.
3. **Degeneracy regression (reproduces §2).** Poisson with a 1e4-scale column + O(1) bases: with `standardize=True` the O(1) coefficients are **not** all zero and the fitted-value spread ≫ 1 (the model discriminates). With `standardize=False`, assert the old degeneracy still reproduces (locks in the escape hatch + documents the bug).
4. **No-op without penalty.** `regularization=None` (and `alpha=0`): coefficients, SEs, predictions byte-identical with `standardize` True vs False.
5. **SE/CI back-transform correctness.** Back-transformed `bse`/`conf_int`/`pvalues` (and robust variants) match a reference fit performed on externally pre-standardized inputs (or finite-difference covariance) to ~1e-6.
6. **Predict & export parity.** `predict` on new data, `predict_contributions`, ONNX and PMML export all use original-scale coefficients (round-trip prediction parity to ~1e-8).
7. **Constraints preserved.** A monotone (`nonneg`/`nonpos`) fit keeps the sign constraint after back-transform.
8. **Smooth path unaffected.** P-spline GCV/REML fit identical with `standardize` toggled (§7).
9. **Target-encoding CV remains fold-safe.** Fold-safe TE CV computes standardization from fold-training designs only; validation rows do not influence target-encoded columns, `alpha_max`, or fold fit scales. The final refit uses full-training scales.
10. **Warm-start parity.** CV path warm starts produce the same final predictions as cold-start fits at each selected alpha to tolerance; coefficients are transformed into/out of standardized coordinates correctly.
11. **No-intercept policy.** With `fit_intercept=False`, `standardize=True` performs scale-only standardization and preserves affine scale invariance for rescaling columns without introducing a hidden intercept shift.
12. **Determinism.** Two seeded runs produce byte-identical `to_bytes()` (downstream `destyler` relies on byte-identical fits).

---

## 9. Backward Compatibility & Migration

- `standardize=True` **changes the coefficients and selected α of every regularized fit.** This is the intended fix, but it is a behaviour change: bump the minor version, add a `CHANGELOG.md` entry under RS-ACT-012, and update any golden/contract tests that pin specific α values or coefficients (search tests for hard-coded `alpha`/`params` on regularized fits).
- Provide `standardize=False` for exact reproduction of prior fits; document it in the migration note.
- Direct Rust/binding callers default to `None` (legacy) so they are unaffected until they opt in.

---

## 10. Downstream Note (destyler)

This fix makes the penalty *fair across scales*, but raw-product interaction columns (scale ~1e17) remain numerically hostile even when standardized (heavy-tailed, collinear; centering helps conditioning but not tail behaviour). Downstream libraries that build the design (e.g. `destyler`) should still prefer standardized or target-encoded representations for continuous×continuous interactions rather than raw products. The rustystats fix is necessary and sufficient to stop the silent model collapse; clean interaction construction upstream is complementary. A second observed contributor — the `1se` selection rule landing at an extreme α — is largely a *symptom* of the mis-scaled path and should be re-evaluated after this fix (the CV curve is far better identified once columns are standardized).

---

## 11. Task Checklist

- [x] `Standardization` struct + optional plumb into `fit_glm_unified` and coordinate-descent; working-matrix build; β/covariance back-transform; warm-start transform (§4.1, §6A).
- [x] `center`/`scale` added to `fit_glm_py`, `fit_negbinomial_py`, `fit_cv_path_py` signatures and `_rustystats.pyi` (§6B).
- [x] `compute_standardization` helper; `compute_alpha_max` uses standardized scores; fast CV uses full-design scales; TE CV uses fold-training scales; `fit()` exposes `standardize=True` default (§6C).
- [ ] Optional follow-up: `penalty_factor` public API + explicit per-column penalty support (§5, §7).
- [x] Smooth-penalty path guarded/skipped (§7).
- [x] Core regression coverage for affine invariance, raw-scale escape hatch, no-op-without-penalty, no-intercept scale-only standardization, standardized `alpha_max`, Rust coefficient/covariance transforms, and fold-safe TE CV replay (§8).
- [x] Additional acceptance hardening: SE/CI + robust-covariance back-transform parity against an external-standardization oracle, intercept cross-term covariance reference test (Rust), explicit constraint-preservation, smooth-toggle parity, PMML/ONNX/serialization export parity, and deterministic `to_bytes()` over the warm-started CV path (§8).
- [x] CHANGELOG + migration/docs note; no golden-test re-pin required because the existing suite remained green (§9).
