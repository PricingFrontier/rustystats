# RustyStats Fit Performance Notes

This folder is a self-contained repro for the slow RustyStats stage observed
while generating a destyler report on the French MTPL frequency dataset
(`freMTPL2`).

The goal is to diagnose RustyStats fitting performance without needing to rerun
the CatBoost teacher, the functional decomposition, or the destyler recipe
builder. The parquet and JSON files here let RustyStats build and fit exactly
the same GLM student specification directly.

## What Model This Reproduces

The original destyler workflow was:

1. Load the real French MTPL frequency dataset.
2. Use Groups 1-3 as the training pool.
3. Use Group 4 as `test`.
4. Use Group 5 as `oot`.
5. Train one depth-3 CatBoost teacher with a Poisson objective and `Exposure` as
   the log-link offset/baseline.
6. Decompose that teacher with exact functional ANOVA up to order 3.
7. Build a GLM recipe with:
   - `top_k_main=12`
   - `top_2way=6`
   - `top_3way=5`
   - `cat_encoding="frozen"`
   - `one_hot_max_size=2`
   - `ctr_min_share=0.01`
   - `frozen_column_budget=40`
8. Fit the RustyStats student to the teacher predictions, not the observed
   claim counts.

The successful packaged repro uses a seeded 50k sample from the training pool.
The held-out test/OOT rows are included in `fremtpl2_50k_actual_frame.parquet`
for context, validation, and scoring checks, but the RustyStats performance
repro focuses on the 50k train frame.

## Important Target Detail

`fremtpl2_50k_rustystats_train.parquet` is the exact frame to use for the
RustyStats fit.

In that file, `ClaimCount` has deliberately been replaced by the CatBoost
teacher's predicted mean. This matches destyler's default methodology:

```text
fit_to_teacher=True
```

So the GLM is fitting the teacher's structure and predictions. It is not fitting
the observed claim counts in this repro frame.

If you need to inspect the original observed response, use either:

- `fremtpl2_50k_actual_frame.parquet`, where `ClaimCount` is observed.
- `fremtpl2_50k_rustystats_train_with_actual.parquet`, where
  `ClaimCount_actual` preserves the observed count alongside the teacher target.

## RustyStats Inputs

The notebook and quick-run script use:

- `fremtpl2_50k_rustystats_train.parquet`
- `glm_kwargs.json`
- `fit_config.json`

`glm_kwargs.json` is the direct payload for:

```python
rs.glm_dict(**glm_kwargs, data=train, seed=0)
```

It includes:

- response: `ClaimCount`
- family/link: Poisson/log
- offset: `Exposure`
- 12 main terms
- 11 interaction specs
- 5 deterministic `input_transforms`

Those `input_transforms` correspond to the frozen categorical statistics emitted
by destyler. They are part of the RustyStats model contract in this repro.

## Timing Observed

The successful 50k repro timing is in:

```text
timing_success_50k_ridge_cv2.json
```

Observed stage times on this machine:

```text
load_fremtpl2                                  1.826s
build_preassigned_roles                        0.082s
construct_distiller                            1.148s
train_teacher_catboost                        10.972s
functional_anova_decompose                    14.134s
propose_recipe_and_freeze_categoricals         0.072s
fit_student_rustystats_ridge_cv2              74.721s
diagnose_validation_packet_no_full_diagnostics 5.028s
build_result_report                            0.011s
write_report_artifacts                         2.620s
total                                        110.614s
```

The key performance finding is that CatBoost and decomposition were not the
main bottleneck in the successful 50k run. The RustyStats ridge CV fit was the
largest stage.

## Failed Probe Timings

Two additional timing files are included for context:

```text
timing_failed_150k_projection_auto.json
timing_failed_150k_unregularized.json
```

The 150k `projection_auto` probe reached the RustyStats regularized fit stage
and spent about 252 seconds there before the timing script hit an unrelated
instrumentation bug. The fit had already returned by then, so the timing signal
is still useful.

The 150k unregularized probe failed quickly because the GLM design matrix was
singular:

```text
Linear algebra error: Failed to solve linear system - matrix may be singular.
```

That suggests some regularization is needed for this recipe, but the regularized
CV path is the expensive part.

## How To Reproduce The Slow Stage

Open:

```text
reproduce_rustystats_fit.ipynb
```

The core cells do this:

```python
import json
import polars as pl
import rustystats as rs

train = pl.read_parquet("fremtpl2_50k_rustystats_train.parquet")
glm_kwargs = json.loads(open("glm_kwargs.json").read())

builder = rs.glm_dict(**glm_kwargs, data=train, seed=0)
model = builder.fit(
    regularization="ridge",
    cv=2,
    selection="1se",
    n_alphas=20,
    cv_seed=0,
)
```

The notebook also includes an optional unregularized fit cell. That cell is
expected to fail or be unstable for this design, but it is useful for separating
plain GLM construction/linear algebra from the regularized CV path.

## Suggested RustyStats Diagnosis Angles

Likely places to inspect:

- Time spent constructing the model matrix for each CV fold and alpha.
- Whether `input_transforms` are recomputed more often than necessary during CV.
- Whether spline bases / categorical blocks are rebuilt repeatedly across the
  regularization path.
- Whether the regularized path is doing redundant work for ridge when
  `alpha` changes but the design matrix is unchanged.
- Whether fold splits copy large Polars/Numpy objects repeatedly.
- Whether standardization or rank checks are repeated per alpha rather than
  cached per fold.

The model has only a few hundred parameters, so the wall time looks more like
repeated data/design preparation than unavoidable linear algebra.

## Reference Artifacts

- `reference_student.rsglm` is the fitted RustyStats model from the successful
  50k ridge CV run.
- `reference_teacher.cbm` is the CatBoost teacher from the same destyler run.
- `validation_packet.json` and `golden_rows.json` are copied from the `.dst`
  bundle and can be used for scoring sanity checks.

