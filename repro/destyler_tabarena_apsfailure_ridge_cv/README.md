# APSFailure Ridge CV Repro

This repro isolates the RustyStats part of the destyler TabArena APSFailure run.
It is meant for optimizing runtime without changing the benchmark methodology.

The case is:

- Dataset: APSFailure
- OpenML task: 363616
- TabArena split: r0f0
- Teacher: depth-3 CatBoost with the `tabarena_depth3` settings
- Student recipe: `ebm_like`
- Target: teacher predictions
- RustyStats fit: ridge, 5-fold CV, `selection="1se"`, 20 alphas

The frozen artifacts live in `artifacts/`:

- `train.parquet`: exact RustyStats training frame, including the teacher-fit target and anchor rows.
- `test.parquet`: prepared test frame for prediction timing.
- `glm_kwargs.json`: arguments passed to `rustystats.glm_dict`.
- `fit_config.json`: fit arguments passed to `builder.fit`.
- `recipe.json`: destyler recipe after sparse-interaction guarding and anchor setup.
- `metadata.json`: task, preprocessing, recipe size, and prep timings.

## Run

From the RustyStats repo:

```bash
cd /home/ralph/suite/rustystats
uv run maturin develop
uv run --extra dev python repro/destyler_tabarena_apsfailure_ridge_cv/run_repro.py
```

For a quick smoke test:

```bash
uv run --extra dev python repro/destyler_tabarena_apsfailure_ridge_cv/run_repro.py \
  --max-rows 2000 \
  --n-alphas 3 \
  --cv 3 \
  --skip-covariance
```

To include prediction timing:

```bash
uv run --extra dev python repro/destyler_tabarena_apsfailure_ridge_cv/run_repro.py --predict
```

To save the timing JSON:

```bash
uv run --extra dev python repro/destyler_tabarena_apsfailure_ridge_cv/run_repro.py \
  --predict \
  --output repro/destyler_tabarena_apsfailure_ridge_cv/artifacts/baseline_full_run.json
```

## Rebuild Artifacts

Only rebuild when the destyler methodology changes. This step downloads the
OpenML task, trains the teacher, decomposes it, and writes the frozen RustyStats
inputs.

```bash
cd /home/ralph/suite/rustystats
OPENML_CACHE_DIR=/tmp/openml-cache \
uv run --extra dev --with-editable /home/ralph/suite/destyler \
  --with openml \
  --with catboost \
  --with scikit-learn \
  python repro/destyler_tabarena_apsfailure_ridge_cv/prepare_artifacts.py
```

The full destyler profile that led to this repro had the RustyStats student fit
at roughly 356 seconds on this machine. A standalone full run from this repro is
saved in `artifacts/baseline_full_run.json`; it fitted the same ridge CV path in
364.7 seconds and predicted the test frame in 0.68 seconds. The optimization
goal is to reduce the default `run_repro.py` runtime while keeping the same GLM
terms, interactions, weights, target, CV policy, and final fitted predictions.

Useful optimization targets are the ridge CV path, repeated design-matrix work
across folds and alphas, warm starts, and allocation churn in regularized GLM
fits.

## Future Optimization Potentials

The current exact sparse Gram path appears close to a local optimum for this
repro. The strongest retained full-run profile is around 195.8 seconds total
with a CV wall time around 184.4 seconds; summed WLS Gram assembly remains the
dominant cost at roughly 699 seconds of fold/alpha work. Several direct Gram
micro-optimizations were tested and rejected because they regressed either the
15k sample or the full APSFailure run, including narrower cached indices,
wider unrolling, and dense/sparse column splitting.

Two larger ideas remain plausible future projects:

1. Avoid rebuilding the full weighted Gram every IRLS step.

   In exact IRLS, every binomial WLS solve uses weights that depend on the
   current coefficients, so `X'WX` changes every iteration. Exact incremental
   Gram updates would still need to touch most row-pairs for this design. The
   more practical version is a guarded iterative ridge solver for CV folds:
   solve with repeated `X'W(Xv)` products instead of materializing `X'WX`.
   Earlier PCG experiments reduced Gram construction substantially and produced
   a full run around 161.9 seconds, but looser tolerances can shift selected
   alpha or deviance slightly. Treat this as a high-promise, medium-risk route
   that needs strict tolerance/fallback rules and result-equivalence checks.

2. Build a term-aware Gram assembler.

   The fitted 958-column matrix is generated from structured spline, encoded,
   and interaction terms. A block-aware assembler could exploit that structure
   instead of treating the matrix as generic sparse rows. This may have high
   upside, especially for repeated CV WLS fits, but it is a larger architecture
   project: design metadata would need to reach Rust, block kernels would be
   needed for the relevant term families, and the implementation must remain
   exact across CV folds, standardization, interactions, target/frequency
   encodings, and final refits.

These are deliberately left as future work. The smaller exact sparse Gram-loop
optimizations have largely been exhausted for this workload.
