# Changelog

Notable changes in each RustyStats release. For releases before 0.10.0, see
[GitHub Releases](https://github.com/PricingFrontier/rustystats/releases).

## 0.10.0

### Upgrading

- **Polars 2 is required** (`polars>=2.0,<3`). Polars 1.x is no longer supported.
- **Some fitted results change, even with the same seed:**
  - Target-encoded values differ, because training rows now use a keyed order.
  - Cross-validation folds differ, so the selected penalty can change. Folds with
    target encoding are now about equal in size.
  - Other results move at rounding level, because rows are now reduced in a fixed
    order. This includes the null fit behind `alpha_max`, which now uses the weights
    as given.
- **Some inputs are now refused:** List, Array, Struct, Object and Null columns as
  categorical, frequency-encoded or target-encoded inputs. `cv` below 2 raises
  `ValidationError` before any fit runs.
- **Removed:** the undocumented PCG ridge solver and its `RUSTYSTATS_RIDGE_CV_PCG`
  setting.

### Added

- **Out-of-core fitting.** `rs.glm_dict(data=rs.BatchSource(open, schema), ...)` with
  `.fit(out_of_core=rs.OutOfCoreOptions(...))` fits any model the in-memory fit accepts
  from a re-openable source of batches, within a memory budget you set. Results are
  bitwise identical to the in-memory fit. See
  [Out-of-Core Fitting](https://pricingfrontier.github.io/rustystats/api/out-of-core/).
- **Robust covariances at fit time:** `fit(robust=["HC1", "HC3"])`, in either mode.
- **New on every result:** `fit_mode`, `passes`, `solver_settings` and `solver_trace`.
  Out-of-core results also carry `memory_plan` and `encode_rows`. All of them survive
  `to_bytes` and `from_bytes`.
- **`rs.observed_category_labels`** returns each column's sorted labels, from a
  DataFrame or a `BatchSource`.
- **New exceptions**, importable from `rustystats`: `BatchSourceChangedError`,
  `DesignTooWideError`, `LearnedStateTooLargeError`, `MemoryPlanExceededError`,
  `OutOfCoreOptionError`, `OutOfCoreResultError`, `SpillError` and `UnseenLabelError`.

### Fixed

- Results no longer depend on the thread count or on run-to-run scheduling, and
  cross-validation folds no longer depend on the Rust release.
- A smooth covariance that cannot be inverted now raises, instead of falling back to
  the identity matrix.
- Scale, log-likelihood, null deviance, standard errors and residuals now use the exact
  θ and Tweedie power, not values rounded to four decimals.
- Cross-validation fold designs now honour `knots=`.
- Cross-validation accepts link names in any case, as plain fits already did.
