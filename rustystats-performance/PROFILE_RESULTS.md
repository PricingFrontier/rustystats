# RustyStats Fit Profile Results

Profiled on 2026-06-03 from the local editable checkout with:

```bash
./.venv/bin/python rustystats-performance/profile_rustystats_fit.py \
  --output rustystats-performance/profile_rustystats_fit_latest.json
```

The run recreates the packaged 50k freMTPL2 RustyStats student fit from
`glm_kwargs.json`, `fit_config.json`, and
`fremtpl2_50k_rustystats_train.parquet`.

## Baseline 50k Ridge CV=2, 20 Alphas

| Stage | Seconds |
| --- | ---: |
| Full `.fit()` | 65.866 |
| CV path selection | 63.515 |
| Rust `fit_cv_path_py` | 62.904 |
| Final full-data refit at alpha=50 | 2.298 |
| CV/full-fit standardization | 0.651 |
| `glm_dict` construction | 0.440 |
| Full design matrix build | 0.373 |
| `compute_alpha_max` | 0.288 |
| Input transforms | 0.066 |
| Parquet load | 0.015 |

The fitted model has 50,000 observations, 298 parameters, selected
`alpha=50`, CV deviance `0.0010342833134815315`, and converged in 6 final-fit
iterations.

## Alpha Grid Probes

These probes use the same 50k frame and skip the unregularized alpha=0
candidate to isolate the regularized grid cost.

| Fit | Rust CV seconds | Total fit seconds | Selected alpha |
| --- | ---: | ---: | ---: |
| 1 alpha | 1.717 | 4.890 | 500000 |
| 5 alphas | 19.761 | 23.032 | 50 |
| 20 alphas + alpha=0 | 62.904 | 65.866 | 50 |

The 5-alpha grid selected the same alpha and final deviance as the 20-alpha
grid in this repro, cutting total fit wall time by about 65%.

## Optimisation Targets

1. `fit_cv_path_py` is the main bottleneck. It accounts for about 95% of the
   full `.fit()` time.
2. Matrix construction, deterministic input transforms, parquet load, and
   alpha-grid setup are not the current limiting stages on this repro.
3. The Rust CV path copies full train/validation matrices once per fold, then
   iterates all alpha candidates sequentially inside each fold. That is visible
   in `crates/rustystats/src/fitting_py.rs` around the fold loop and alpha loop.
4. Reducing the ridge alpha grid is immediately effective for this model. A
   coarse 5-alpha grid found the same selected alpha as the full grid.
5. Larger algorithmic wins likely need Rust-side changes: avoid repeated IRLS
   fits for ridge candidates where possible, expose per-fold/per-alpha timing,
   reuse fold matrix work more aggressively, or add an explicit coarse-to-fine
   alpha search for ridge CV.

Detailed raw timing reports:

- `profile_rustystats_fit_latest.json`
- `profile_rustystats_fit_50k_alpha1.json`
- `profile_rustystats_fit_50k_alpha5.json`

## Deeper Rust CV Profile

After adding Rust-side CV timing, the default full-grid direct CV call splits as:

| Rust CV component | Summed work seconds |
| --- | ---: |
| Per-alpha `fit_glm_unified` calls | 110.155 |
| Fold split/copy | 0.103 |
| Fold standardization | 0.238 |
| Validation matrix-vector dot | 0.297 |
| Validation inverse-link/deviance scoring | 0.018 |

The direct Rust CV wall time was 55.605s because the two folds run in parallel.
The bottleneck is therefore repeated fold/alpha IRLS fitting, not data loading,
input transforms, design-matrix construction, fold slicing, standardization, or
validation scoring.

The slowest candidates are the middle ridge alphas. On both folds, alphas around
`10346` through `347` hit `max_iter=25`, costing roughly 4.5-5.7s per fold per
alpha. The selected alpha remains `50`.

The extension used for these timings is release-built:

- loaded module: `python/rustystats/_rustystats.cpython-313-x86_64-linux-gnu.so`
- byte-identical to `target/release/librustystats.so`
- not byte-identical to the 62MB debug artifact

## Optimisation Experiments

All rows below use the same 50k repro and report the public `.fit()` wall time.
The final fitted model selected `alpha=50`, converged, and had final deviance
`47.087914` in every row.

| Method | Fit seconds | Rust CV seconds | Notes |
| --- | ---: | ---: | --- |
| Baseline before changes | 65.866 | 62.904 | 20 alphas + alpha=0, strict CV/final `tol=1e-8` |
| Skip final extraction in CV fits | 55.281 | 52.215 | Code change; final refit unchanged |
| Skip extraction + `n_alphas=5` | 21.068 | 17.953 | Same selected alpha and CV deviance |
| Skip extraction + `n_alphas=3` | 16.710 | 13.656 | Same selected alpha; slightly different CV deviance |
| Skip extraction + `cv_tol=1e-4` | 9.496 | 6.853 | CV folds relaxed; final refit still strict |
| Skip extraction + `n_alphas=5`, `cv_tol=1e-4` | 5.086 | 2.620 | Fastest tested strict-final route |
| Skip extraction + `cv_max_iter=5` | 17.289 | 14.835 | CV folds capped; final refit still strict |
| Skip extraction + `n_alphas=5`, `cv_max_iter=5` | 7.029 | 4.438 | Slower than `cv_tol=1e-4` combo |
| Scale-only solver transform + sparse WLS | 11.063 | 9.734 | Code-only strict-default route; selected alpha/final deviance unchanged |
| Scale-only sparse WLS + `cv_tol=1e-4` | 2.839 | 1.568 | Full 20-alpha grid, final refit strict |
| Scale-only sparse WLS + `n_alphas=5`, `cv_tol=1e-4` | 2.043 | 0.708 | Fastest tested strict-final route after code optimisation |

Thread-count sweep on a 5-alpha direct CV run showed parallelism is useful:

| `RAYON_NUM_THREADS` | Direct CV seconds |
| ---: | ---: |
| 1 | 84.601 |
| 2 | 42.947 |
| 4 | 25.926 |
| 8 | 20.215 |
| 16 | 16.013 |
| 22 | 15.446 |

## Follow-Up Probes After Sparse WLS

These were tested after the scale-only sparse WLS optimisation and were not
kept as code changes:

| Probe | Result |
| --- | --- |
| Reverse alpha warm-start order | Much slower. Low-to-high strict CV took 43.300s and made 19/20 regularized alphas hit `max_iter` on each fold. |
| Start near selected alpha (`~50`) and move outward | Much slower: 45.232s strict CV, again with almost every alpha at `max_iter`. |
| Per-fit CSR-style sparse design cache | Slower on the 5-alpha direct CV probe: 5.809s versus ~4.35-4.57s. Cache build and sparse dot overhead outweighed saved row scans. |
| Hybrid one-pass/two-pass Python standardization | Slower on this matrix: median 0.259s per call versus 0.214s for the current stable two-pass helper. |
| Thread tuning after sparse WLS | Small only. On 5-alpha direct CV, `RAYON_NUM_THREADS=8` was ~4.39s versus ~4.49s at 22. Full-grid direct CV at 8 threads was 9.970s versus 10.160s default. |

## Related Hot Paths

These timings use the same 50k parquet data and original terms, producing a
`50,000 x 298` design matrix with density `0.1073`. The harness is
`profile_related_hotpaths.py`; the coordinate-descent timings use `cv=2`,
`n_alphas=3`, `max_iter=6`, `tol=1e-6`, and exclude the unregularized alpha so
the L1/elastic-net solver is exercised directly.

| Path | Before | After | Speedup | Notes |
| --- | ---: | ---: | ---: | --- |
| Lasso CV wall | 6.584s | 0.797s | 8.3x | Coordinate descent now reuses sparse-aware `X'WX` / `X'Wz` construction |
| Lasso fold fit work | 12.534s | 1.147s | 10.9x | Sum across folds/alphas |
| Elastic-net CV wall | 6.645s | 0.917s | 7.3x | Same coordinate-descent path |
| Elastic-net fold fit work | 12.149s | 1.283s | 9.5x | Sum across folds/alphas |
| VIF/correlation, `skip_cols=1` | 0.396s | 0.028s | 14.3x | Public diagnostics path; sparse row-pair Gram kernel |
| VIF/correlation, `skip_cols=0` | 0.252s | 0.016s | 15.3x | Contiguous diagnostics path |
| Robust HC1 `bse_robust` | 0.352s | 0.094s | 3.7x | Sparse sandwich meat kernel |
| Robust HC3 `bse_robust` | 0.560s | 0.091s | 6.1x | Sparse leverage + meat kernels |
| Direct lasso final fit | 18.517s | 0.419s | 44.2x | Final covariance now uses sparse active-set `compute_xtwx`; 58 active coefficients get finite covariance on this run |

The kept code changes are:

- `coordinate_descent.rs`: build `X'WX`/`X'Wz` through the shared sparse-aware
  helper, then keep the row-major flat layout for repeated coordinate updates.
- `diagnostics/vif.rs`: add sparse contiguous and strided Gram/sum kernels with
  density sampling, so public `skip_cols=1` no longer scans all zero pairs.
- `inference/mod.rs`: add sparse sandwich kernels for leverage and meat matrix
  construction.
- `irls.rs` / `coordinate_descent.rs`: route standalone `X'WX` construction
  through the sparse-aware helper so final lasso/elastic-net covariance and
  smooth-model EDF avoid the old dense row-pair loop.

## Agent Scout Follow-Up

Four read-only scout agents reviewed solver, diagnostics, design-building, and
benchmark coverage. Two low-risk findings were patched and timed immediately:

| Path | Before | After | Notes |
| --- | ---: | ---: | --- |
| Final strict `rust.fit_glm_py` | 0.467s | 0.413s | Standard IRLS now solves without covariance during each iteration and computes covariance once at final accepted weights |
| Mixed interaction aggregate | 0.261s | 0.248s | Training now reuses the already-built multi-categorical block |
| Nested `Region:VehBrand` categorical interaction | 0.100s x2 | 0.051s x1 | Duplicate build eliminated inside `Region:VehAge:VehBrand` |
| Full `.fit()` wall | 13.019s | 13.119s | One-run total is noise-flat because the remaining run is dominated by ridge CV |

Both runs used the same 50k parquet repro, selected `alpha=50`, converged in 6
final iterations, and produced final deviance `47.087914189154`.

The next best candidates from the scout pass are:

1. Python/Rust standardization and ridge alpha-max temporaries
   (`regularization_path.py`): about 0.6-0.9s visible on this repro, but a
   medium-risk numerical change.
2. CV fold split/materialization (`fitting_py.rs`): currently small on this
   50k/2-fold repro, worth revisiting for larger `n`, larger `p`, or more
   folds.
3. Smooth/spline paths (`smooth_glm.rs`, `splines/mod.rs`): avoid unconditional
   design copies and per-row basis allocations if GAM workloads are hot.
4. Diagnostics score-test/calibration/factor batching: likely large for
   million-row diagnostics, especially continuous score tests and repeated
   calibration/discrimination sorts.
5. Add repeatable benchmark coverage: profile matrix runner, JSON regression
   checker, memory sampler, and Criterion benches for sparse kernels.

Detailed reports:

- `profile_agent_scout_before.json`
- `profile_agent_scout_after.json`

## Current Best Candidates

1. Keep the code change that skips the final WLS extraction for
   `skip_covariance=true` CV fits. It reduces full-grid public fit time by about
   16-18% and leaves final model metrics unchanged on this repro.
2. Keep the scale-only solver transform for regularized fits with an
   unpenalized intercept, plus the sparse-aware WLS cross-product. The alpha
   grid still uses centered standardization statistics, but the solver can
   absorb centering into the intercept and preserve one-hot sparsity. On this
   repro the default strict public fit dropped from 55.281s to 11.063s after the
   first code change, with `alpha=50`, final deviance `47.087914`, 6 final
   iterations, and `solver_status="converged"` unchanged.
3. Use `cv_tol=1e-4` for CV alpha selection while keeping the final fit at the
   strict default `tol=1e-8`. This now cuts the public fit to 2.839s on the
   full 20-alpha grid.
4. For this recipe, `n_alphas=5` plus `cv_tol=1e-4` now cuts the public fit to
   2.043s and still selects `alpha=50`. This should be treated as a recipe-level
   speed/accuracy trade-off until tested on more models.
5. Remaining deeper optimisation targets are algorithmic: the middle alphas
   still hit the strict iteration cap, so further default-path wins would need
   fewer exact CV iterations/candidates or a ridge-specific path that avoids
   redundant `O(n*p^2)` work.

Additional detailed reports:

- `profile_cv_path_detail_latest.json`
- `profile_cv_path_detail_skip_final_extraction_latest.json`
- `profile_cv_path_detail_sparse_scale_latest.json`
- `profile_rustystats_fit_skip_final_extraction_latest.json`
- `profile_rustystats_fit_sparse_scale_latest.json`
- `profile_rustystats_fit_sparse_scale_cv_tol1e-4.json`
- `profile_rustystats_fit_skip_final_extraction_alpha5_cv_tol1e-4.json`
- `profile_rustystats_fit_sparse_scale_alpha5_cv_tol1e-4.json`
- `profile_related_hotpaths_before.json`
- `profile_related_hotpaths_after_cd.json`
- `profile_related_hotpaths_after_all.json`
- `profile_related_hotpaths_after_final.json`
