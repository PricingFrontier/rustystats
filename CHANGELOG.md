# Changelog

Notable changes in each RustyStats release. For releases before 0.10.0, see
[GitHub Releases](https://github.com/PricingFrontier/rustystats/releases).

## Unreleased

### Changed

- **Wheels for Python 3.14.** Each platform now has one wheel built on CPython's stable ABI
  instead of one per Python version. It installs on Python 3.10 and later, so Python 3.14 no
  longer compiles rustystats from source. PyO3 and rust-numpy move to 0.29, which clears
  RUSTSEC-2026-0177.

## 0.11.0

### Added

- **Validation report.** `result.html_report(train, test_data=test)` returns one self-contained
  HTML page for reviewing a fitted GLM: performance in and out of sample, each factor's fitted
  effect and A/E, interactions, coefficients, residual and variance checks, influence, and a
  list of review flags. `result.validation_report(...)` returns the same content as a JSON-safe
  dictionary and `rs.report_html(report)` draws a saved one. The page has no scripts and loads
  nothing from the network. Multinomial models get the same page with a class switch. See
  [Validation Report](https://pricingfrontier.github.io/rustystats/api/validation-report/).

### Fixed

- **Null deviance with an offset.** The null model now places the offset by the fitted link:
  `c + offset` on an identity link and `c * exp(offset)` on a log link, in every fit path, with
  the constant that is optimal for the family's variance function. Null deviance, deviance
  explained and the likelihood-ratio test therefore change for a Gaussian, Gamma or Tweedie fit
  with an offset, and now match statsmodels. So does any fit whose link is not its family's
  default. Fits with no offset, and Poisson-type fits with a log link, do not change. Negative
  binomial and logit fits with an offset have no closed-form constant, so they are unchanged.
- **Continuous factor bands kept four decimal places**, which collapsed the bands of a factor on
  a small scale. Bands now keep their true edges; the JSON output still rounds them.
- **`compute_null_deviance_py`** no longer treats exposure as a weight, which gave the wrong null
  deviance for a rated model. It now takes `weights`, `offset` and `link`.
- **`diagnostics()` with null categorical values.** A null in a categorical factor raised a
  `TypeError`. It is now the level `None`, as in the explorer and the design matrix.

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
