# TabArena Destyler Student Timing Repro

This repro isolates the slow RustyStats student fits from the first destyler
TabArena benchmark. It has two stages:

1. Build artifacts from the exact TabArena split, depth-3 CatBoost teacher, and
   destyler recipe.
2. Re-run only the RustyStats `glm_dict(...).fit(...)` call from those artifacts.

The default case list is [cases_slowest_first_run.csv](cases_slowest_first_run.csv).
It starts with `kddcup09_appetency` and `Diabetes130US`, the two datasets that
dominated the first run's student-side training time.

## Setup

From the RustyStats repo:

```bash
cd /home/ralph/suite/rustystats
uv run maturin develop
uv pip install -r repro/tabarena_destyler_timings/requirements.txt
```

The preparation stage also needs the sibling destyler checkout and its benchmark
dependencies (`catboost`, `openml`, `pandas`, `scikit-learn`, `polars`). The
script defaults to `/home/ralph/suite/destyler` and the first-run manifest at:

```text
/home/ralph/suite/destyler/reports/tabarena/full_20260617_1/manifest.json
```

Set `OPENML_CACHE_DIR` if you want OpenML data cached somewhere specific.

Destyler checks the installed RustyStats package metadata at import time and
requires `rustystats>=0.8.9`. If the preparation script reports an older local
version, update the package metadata and rerun `uv run maturin develop` before
preparing artifacts.

## Prepare One Slow Case

```bash
cd /home/ralph/suite/rustystats
OPENML_CACHE_DIR=/tmp/openml-cache \
uv run python repro/tabarena_destyler_timings/prepare_case_artifacts.py \
  --limit 1 \
  --targets teacher \
  --repeat-fit 1
```

This writes a timestamped folder under:

```text
repro/tabarena_destyler_timings/artifacts/
```

Each case contains:

- `case_metadata.json`: OpenML split, preprocessing, and phase timings.
- `base_recipe.json`: destyler recipe before per-target mutation.
- `teacher/train.parquet`: the exact RustyStats training frame for teacher-fit distillation.
- `teacher/test.parquet`: raw test features for prediction timing.
- `teacher/glm_kwargs.json`: arguments for `rustystats.glm_dict`.
- `teacher/fit_config.json`: regularization, CV, and seed used for the fit.
- `teacher/recipe.json`: final recipe after sparse-interaction conversion and anchors.

## Time RustyStats Directly

After artifacts are prepared, this command does not retrain CatBoost or run
destyler decomposition:

```bash
uv run python repro/tabarena_destyler_timings/time_rustystats_artifacts.py \
  repro/tabarena_destyler_timings/artifacts/<RUN_ID> \
  --targets teacher \
  --repeats 5 \
  --warmup 1 \
  --predict
```

Outputs:

- `rustystats_direct_timings_<timestamp>.csv`
- `rustystats_direct_timings_<timestamp>_summary.csv`

Use `--skip-covariance` to separate coefficient covariance cost from design
matrix and solver cost. Use `--measure-memory` for a coarse process RSS delta if
`psutil` is installed.

## Prepare More Cases

Prepare both teacher-fit and actual-label students for the representative slow
splits:

```bash
uv run python repro/tabarena_destyler_timings/prepare_case_artifacts.py \
  --targets teacher actual \
  --repeat-fit 1
```

Prepare every split for the two worst datasets:

```bash
uv run python repro/tabarena_destyler_timings/prepare_case_artifacts.py \
  --datasets kddcup09_appetency Diabetes130US \
  --all-splits-for-datasets \
  --targets teacher \
  --repeat-fit 1
```

The preparation script also writes `summary.csv` with teacher, decomposition,
proposal, artifact-prep, direct fit, and prediction timings.
