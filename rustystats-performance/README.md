# RustyStats Performance Repro

This folder isolates the RustyStats fit from the destyler French MTPL report run.
It is intended for diagnosing RustyStats performance without rebuilding the
CatBoost teacher or functional decomposition.

For the fuller context, timing interpretation, and suggested RustyStats
diagnosis angles, see [`EXPLANATION.md`](EXPLANATION.md).

## Files

- `fremtpl2_50k_rustystats_train.parquet` - exact training frame passed to
  `rustystats.glm_dict`. `ClaimCount` is the CatBoost teacher prediction target.
- `fremtpl2_50k_rustystats_train_with_actual.parquet` - same train frame with
  `ClaimCount_actual` retained for inspection. Do not use this for exact timing.
- `fremtpl2_50k_actual_frame.parquet` - sampled train rows plus all Group 4 test
  and Group 5 OOT rows with observed `ClaimCount` and `__destyler_role__`.
- `glm_kwargs.json` - direct kwargs for `rustystats.glm_dict`, including terms,
  interactions, offset, family/link, and `input_transforms`.
- `fit_config.json` - fit settings from the successful timed repro (`ridge`,
  `cv=2`, `selection="1se"`, `n_alphas=20`, `seed=0`).
- `reference_student.rsglm` - fitted RustyStats model from the successful timed
  destyler run.
- `reference_teacher.cbm` - CatBoost teacher from the same run.
- `timing_success_50k_ridge_cv2.json` - successful stage timings.
- `timing_failed_150k_projection_auto.json` - 150k probe where regularized
  fitting took about 252s before the instrumentation bug aborted.
- `timing_failed_150k_unregularized.json` - 150k unregularized probe; the fit
  failed quickly because the design matrix was singular.
- `reproduce_rustystats_fit.ipynb` - notebook that builds and fits the same
  RustyStats model from the parquet and JSON kwargs.

## Quick run

Open `reproduce_rustystats_fit.ipynb` from this directory, or run the cells from
the repo root. The main fit cell is equivalent to:

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

The previously observed successful timing for the RustyStats ridge CV fit was
about 74.7 seconds on this machine for the 50k train frame. CatBoost and
decomposition were not the bottleneck in that run.
