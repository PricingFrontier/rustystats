# RustyStats Performance Reproduction: Destyler Wide Interactions

This folder captures the slow RustyStats workload observed while refreshing the
Destyler FreMTPL2 model-card report.

## What Is Included

- `fremtpl2_frequency_teacher_target.parquet`
  - Full FreMTPL2 frequency rows used by the Destyler report rerun.
  - Includes a `role` column: `train`, `test`, `oot`.
  - Includes `teacher_mu`, the GBM teacher prediction used as the non-integer
    Poisson distillation target.
- `model_spec.json`
  - The exact RustyStats terms, interactions, offset, and input transforms from
    the widened Destyler student recipe.
  - This is the `top_2way=14` recipe with raw modest-cardinality categorical
    interactions and numeric spike terms.
- `run_performance_case.py`
  - Fits the GLM with the saved terms.
  - Times a single prediction.
  - Times a repeated counterfactual PDP prediction workload similar to the
    Destyler report diagnostics.

## Why This Case Is Slow

The fitted model has a modest number of main terms, but a wider interaction
surface:

- frozen categorical main effects are kept as scalar `*_fts` terms;
- selected interactions with modest-cardinality categorical factors use raw
  categorical levels;
- spline interactions such as `VehAge x VehBrand`, `DrivAge x Region`, and
  `BonusMalus x Region` are present;
- model-owned `input_transforms` produce frozen statistics and exact numeric
  spike columns.

The slow part in the Destyler report was not only fitting. The very slow path was
repeated `model.predict(counterfactual_frame)` calls for PDP diagnostics over a
wide interaction design. The interrupted run was spending time in
`transform_new_data`, especially interaction/spline basis construction.

## Commands

Smoke test:

```bash
uv run python rustystats-performance/run_performance_case.py --quick
```

Fit only:

```bash
uv run python rustystats-performance/run_performance_case.py --skip-pdp
```

Full reproduction of the slow counterfactual prediction workload:

```bash
uv run python rustystats-performance/run_performance_case.py
```

Profile the full run:

```bash
uv run python rustystats-performance/run_performance_case.py \
  --cprofile rustystats-performance/profile_full.pstats
```

Focus on the especially costly PDP factors:

```bash
uv run python rustystats-performance/run_performance_case.py \
  --factors VehAge,VehBrand,Region,BonusMalus,Density
```

## Expected Shape Of The Bottleneck

The script should make it easy to separate:

1. GLM fit time;
2. one-shot prediction time;
3. repeated counterfactual prediction time.

The suspected optimization area is repeated raw-data prediction for frames that
change only one column at a time. Opportunities may include caching compiled
input transforms, avoiding repeated full interaction reconstruction where only
one factor changes, and improving spline interaction transform throughput.
