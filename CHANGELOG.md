# Changelog

All notable user-visible changes to RustyStats.

The format follows [Keep a Changelog](https://keepachangelog.com/) and the
project tracks the
[Actuarial Methodology Hardening Spec](https://github.com/PricingFrontier/rustystats/blob/main/docs/maintenance/actuarial-methodology-hardening.md)
via `RS-ACT-NNN` IDs.

## [Unreleased] — Actuarial Methodology Hardening (RS-ACT-001 to 012)

This release closes the actuarial-correctness items from the hardening
spec and is intended to be the first version safe to use as the statistical
engine inside production insurance pricing workflows. The headline behaviour
changes are listed under *Behaviour changes* below; the new APIs are listed
under *Added*; the internal correctness fixes are listed under *Fixed*.

### Behaviour changes (read first)

Two deliberate behaviour changes can affect existing users' numerical results
even with the same call shape.

* **RS-ACT-002 (rule 4) — array `offset=` no longer feeds exposure-weighted
  target encoding.** Previously, passing `offset=np.log(exposure)` *and* a
  target-encoded term silently used `log(exposure)` as the rate denominator,
  producing degenerate encodings. Set `exposure=` explicitly to recover an
  exposure-weighted denominator; otherwise the encoder falls back to
  unweighted with a warning and the model records
  `used_exposure_weighted=False`.

* **RS-ACT-005 — `alpha_max` is now derived from the GLM score at the null
  model.** The legacy formula divided by `n` and used a centred-`y` proxy,
  under-sizing the grid by roughly a factor of `n`. On a 600-row Poisson fit
  the new value is ~479.6 versus the old ~0.799. If you were tuning
  `alpha_min_ratio` to compensate, you can drop that workaround. `alpha_max`
  is now offset, weight, family, and link aware, and matches the solver's raw
  weighted-sum loss scaling (no implicit `1/n`).

* **RS-ACT-012 — regularized fits now standardize penalized columns by
  default.** Ridge, lasso, and elastic-net fits use weighted internal
  standardization before the penalty acts, then return coefficients and
  covariance on the original data scale. This fixes raw-scale penalty
  collapse when design columns have very different magnitudes. Regularized
  coefficients and selected `alpha` values can change; set
  `standardize=False` to recover the legacy raw-scale penalty.

### Added

* **RS-ACT-002 — explicit `exposure=`.** `glm_dict()` (and the underlying
  `FormulaGLMDict`) accept `exposure=` (str or array). Validated as finite
  and strictly positive. Three migration spellings:

  ```python
  # Preferred new spelling
  rs.glm_dict(..., family="poisson", exposure="Exposure")

  # Link-scale offset alongside exposure
  rs.glm_dict(..., family="poisson", offset=np.log(exposure), exposure=exposure)

  # Legacy spelling — `offset="Exposure"` under a log-link family is still
  # accepted and silently treated as `exposure="Exposure"`.
  rs.glm_dict(..., family="poisson", offset="Exposure")
  ```

  Predict-time `exposure=` is exposed on `result.predict(new_data, exposure=...)`.
  Stored exposure / offset specs survive `to_bytes` / `from_bytes`; old models
  with `offset_is_exposure=True` re-load as legacy exposure models.

* **RS-ACT-004 — rate-ranked diagnostics.** Decile / calibration / lift /
  Lorenz / Gini paths now share one ranking helper (`rank_sort_idx`) and rank
  by `μ / exposure` under `ranking="auto"` when exposure is present. The
  knob is exposed on `result.diagnostics(..., ranking=...)` with values
  `"auto" | "mean" | "rate"`. Bin aggregates stay on the count scale.

* **RS-ACT-006 — explicit Tweedie support contract.** `glm_dict` and
  `rs.TweedieFamily` now accept `allow_extended_tweedie: bool = False`.
  Default behaviour is the actuarial pure-premium interior `1 < p < 2`; the
  extended regimes (`p ≤ 0`, `p == 1`, `p == 2`, `p > 2`) require the opt-in
  and bring per-regime support rules (`y > 0` for `p ≥ 2`; `y ≥ 0`
  otherwise). The `0 < p < 1` band — where no Tweedie distribution exists —
  is rejected always.

* **RS-ACT-007 — solver status surfaces.** `result.solver_status`,
  `result.step_halving_used`, `result.optimizer_route`, and solver warnings
  now expose when budget exhaustion retained the previous coefficients.

* **RS-ACT-008 — quasi-likelihood flag + summary labels.**
  `result.is_quasi_likelihood` is `True` for `quasipoisson` /
  `quasibinomial`. `result.aic()` / `result.bic()` return `None` for these
  families. `summary()` relabels the loglik-like value as
  `Quasi-Log-Likelihood:` and prints AIC/BIC as `NA`.

* **RS-ACT-009 — explicit calibration diagnostics & primitives.**
  - `rs.calibration_summary(y, pred, exposure=, weights=, by=, n_bins=,
    ranking=, min_exposure=)` — array-level overall/per-bin/per-factor A/E.
  - `rs.GlobalCalibration` and `rs.fit_global_calibration(y, pred, weights=)`
    — scalar `c = Σ(w·y) / Σ(w·μ)` map; `predict`, `to_dict`/`from_dict`.
  - `rs.IsotonicCalibration` and `rs.fit_isotonic_calibration(...)` — monotone
    PAV fit, opt-in only, serialised separately, never folded into GLM
    coefficients.
  - `result.calibration_summary(data, ...)`, `result.fit_calibration(...)`.
  - `result.relevel(data, weights=, inplace=False)` — log-link intercept
    shift. Updates only `β[0]` by `+log(c)`; every other coefficient stays
    bit-identical; `exp(β_j)` relativities are preserved. Non-log links raise.
    `intercept_delta` / `relevel_history` round-trip through
    `to_bytes`/`from_bytes`.

* **RS-ACT-010 — explicit Negative Binomial `theta` contract.**
  `theta=` accepts a numeric value (recorded as fixed), the string
  `"estimate"` (profile-likelihood estimation on the plain path), or an
  embedded family parameter such as `family="negbinomial(theta=2.0)"`.
  Leaving `theta` unspecified now raises; regularised / smooth / constrained
  paths also raise for `theta="estimate"` and require a numeric theta.
  `theta_metadata` on the result records estimated-vs-fixed, initial theta,
  iterations, convergence, and tolerance.

* **RS-ACT-011 — honest inference + solver-status surfacing.**
  `result.inference_status` is one of `valid_standard`, `valid_robust`,
  `naive_after_regularization`, `naive_after_selection`,
  `naive_after_cv_selection`, `constrained_boundary`, `unavailable`.
  `summary()` hides stars / p-values / AIC / BIC when the status is not
  valid, and reports the status verbatim near the bottom of the table.

* **RS-ACT-012 — `standardize=` for regularization.** `fit(...,
  standardize=True)` is the default for regularized fits. With an intercept,
  penalized columns are centered and scaled; without an intercept, columns are
  scale-only standardized to preserve the model class. The option is a no-op
  when `alpha=0` / no regularization is used.

### Fixed

* **RS-ACT-001 — fold-safe target-encoding CV** + weighted CV scoring +
  explicit CV convergence. Target encoders are rebuilt per fold from
  fold-training rows only; unseen validation levels map to the fold-training
  prior; validation deviance is `Σ w·dev / Σ w`. CV no longer silently
  relaxes `max_iter` / `tol`.

* **RS-ACT-003 — smooth GLM honours `var_power` / `theta`.**
  `fit_smooth_glm_unified_py` previously hard-coded
  `family_from_name(family, 1.5, 1.0)`, silently ignoring the requested
  Tweedie power and NB theta on penalised-spline fits. The result's family
  string also now reports the parameter actually used.

* **RS-ACT-006 — Tweedie deviance no longer returns silent `inf` or negative
  values for out-of-support combinations.** `unit_deviance` at `y == 0` for
  `p ≥ 2` would previously return `+inf` (`p == 2`) or a negative value
  (`p > 2`); validation now catches these upfront with a clear message, and
  the core function returns `+inf` as a defensive lower-level fallback (no
  negative deviances).

* **RS-ACT-007 — IRLS step acceptance.** The unconstrained path used to
  accept the last attempted step even when deviance had worsened. Step
  halving now keeps the best valid step; budget exhaustion retains the
  previous coefficients and sets `solver_status` accordingly. Final $\mu$ is
  clamped through `family.clamp_mu`, matching the in-loop computations.

* **RS-ACT-009 — single overall-A/E definition.** Legacy `_CalibrationComputer`
  now routes through the same `_overall_ae` primitive as
  `rs.calibration_summary`, so the unweighted diagnostics path and the
  weighted primitive cannot drift.

* **RS-ACT-012 — scale-fair regularization.** `alpha_max`, CV paths, target-
  encoding fold rebuilds, solver warm starts, and covariance back-transforms
  now share the same standardization contract. Target-encoding CV computes
  scales from fold-training designs only.

### Migration

* No public APIs were removed.
* Regularized fits now default to `standardize=True`; pass
  `standardize=False` to reproduce pre-RS-ACT-012 raw-scale penalties.
* `compute_alpha_max` (internal helper) gained kwargs-only `family=`, `link=`,
  `offset=`, `weights=`, `var_power=`, `theta=`, `intercept_col=` and the
  positional `weights=` argument moved into keyword-only. The two internal
  call sites are updated; external callers (rare) must add `family=` / `link=`.
* Direct `fit_cv_path_py` calls without `alphas=` now raise — build the grid
  with `compute_alpha_max` + `generate_alpha_path` first.

### Build / CI

* `.pre-commit-config.yaml` and `.github/workflows/ci.yml` are now both
  pinned to `ruff@0.15.14` and check `python/ tests/ examples/`.
  Pre-commit and CI no longer fight each other over format style.

### Verification

* `cargo fmt --check`, `cargo clippy --workspace -- -D warnings`,
  `cargo test --workspace`.
* `uvx ruff@0.15.14 check / format --check python/ tests/ examples/`.
* `uv run --extra dev pytest tests/python/` — full Python suite green.
* `benchmarks/verify_diagnostics_correctness.py` — all invariants green.
