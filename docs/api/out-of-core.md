# Out-of-Core Fitting

An out-of-core fit reads its training rows in passes from a source you supply and never
holds them, the design matrix, or any per-row array all at once. Resident memory is
bounded by a budget you set, not by the number of rows.

An out-of-core fit and an in-memory fit of the same data are **bitwise identical**: the
same coefficients, deviance, dispersion, standard errors, smoothing parameters,
cross-validated penalty and predictions, to the last bit. This holds whatever the batch
size, block size or thread count. Both modes run the same solvers and reduce rows in fixed
blocks of 4,096 in the same order.

Every model the in-memory fit accepts can be fitted out of core: every family and link,
every term type and interaction, target and frequency encoding, fixed, penalised and
monotone splines, ridge, lasso and elastic net (fixed or chosen by cross-validation),
sign constraints, Negative Binomial θ estimation, complements and robust covariances.

## A batch source

```python
import polars as pl
import rustystats as rs

path = "training.parquet"

def open_batches(columns):
    # Every call must yield the same rows in the same order.
    yield from pl.scan_parquet(path).select(columns).collect_batches()

source = rs.BatchSource(open_batches, pl.read_parquet_schema(path))
```

`BatchSource(open, schema)` takes:

- `open(columns)`: returns an iterable of Polars DataFrames that together hold every row,
  with at least `columns`. It is called once per pass.
- `schema`: the source's column dtypes, known without reading rows.

The fit fingerprints every column it reads on every pass and compares them with the first
pass. A source that gives different rows, a different order or a different dtype raises
`BatchSourceChangedError`, naming the pass and the column.

## Fitting

Pass the source as `data` and the limits as `out_of_core`:

```python
model = rs.glm_dict(
    response="claims",
    terms={"age": {"type": "bs", "df": 5}, "region": {"type": "categorical"}},
    data=source,
    family="poisson",
    exposure="exposure",
).fit(out_of_core=rs.OutOfCoreOptions(max_bytes=2 * 1024**3, scratch_dir="/tmp/fit"))
```

`fit` takes the same arguments as an in-memory fit, plus `out_of_core`. Exposure, offset,
weights and a complement may be column names or arrays with one value per source row; an
array is read a slice at a time, never copied whole.

Options and models that cannot fit (an unknown link, `cv=1`, a nested column as a
category, L1 with a smooth term, and so on) are refused before the source is opened.

### OutOfCoreOptions

| Option | Description |
|---|---|
| `max_bytes` | The memory budget. Every phase's planned peak is admitted against it before it allocates; a phase that would not fit raises `MemoryPlanExceededError`. `None` plans without a limit. The source's own batches are the caller's to budget. |
| `encode_rows` | Rows per encoded block, rounded up to a multiple of 4,096. By default, as many as fit in the smaller of 256 MiB and half of what `max_bytes` leaves after the fixed state. |
| `threads` | The threads the fit's passes run on, and the plan budgets for. By default, the global Rayon pool (`RAYON_NUM_THREADS`, or the CPU count). Results do not depend on the thread count. |
| `max_learned_state_bytes` | A cap on learned state (category keys and labels, counts, target-encoding totals, knots, centres and scales). Learned state is also capped by what `max_bytes` leaves, so a high-cardinality column is refused with `LearnedStateTooLargeError` before it outgrows the budget. |
| `max_design_columns` | A cap on the design's width, judged while category levels are learned: a wider design raises `DesignTooWideError` as soon as its levels cross the limit, naming the term that crossed it. |
| `scratch_dir` | Where spill files go. Required for target encoding and for θ estimation. Every spill file is removed when the fit ends, however it ends. |
| `on_batch` | Called with a `BatchProgress(pass_index, pass_kind, rows_done, n_rows)` after every source batch. An exception it raises stops the fit and reaches the caller unchanged: use it for cancellation and memory checks. |

## Results

Both modes report how the fit ran:

| Attribute | In memory | Out of core |
|---|---|---|
| `fit_mode` | `"in_memory"` | `"out_of_core"` |
| `passes` | `{}` | passes over the source, per pass kind (`"freeze"`, `"candidate"`, `"gram"`, ...) |
| `solver_settings` | the caps the solve used: `max_iter` (25 by default, 2,000 with a monotone penalised spline), `tol`, `max_half_steps`, `gram_block_rows` and `target_encoding_seed`; with an estimated θ also `theta_tol`, `max_theta_iter` and `theta_rounds` | the same |
| `solver_trace` | a tuple of `SolverStep(driver, kind, evaluations, accepted, status)`: one per decision the solvers made (an IRLS iteration, a smoothing update, a cross-validation fold and penalty, a θ round) | the same steps |
| `memory_plan`, `encode_rows` | — | the planned peak of each phase, the persistent bytes, the threads and the limit |

An out-of-core result serves coefficients, covariances, fit statistics, inference,
smoothing and regularisation results, `predict`, `to_bytes` and `from_bytes`. Methods that
need per-row training values raise `OutOfCoreResultError`, naming the method:
`fittedvalues`, `linear_predictor`, the `resid_*` methods, `fit_diagnostics`, `relevel`,
`diagnostics` without data, and a robust covariance type not listed in `robust=`. Predict
on the data instead; `diagnostics(data)` does.

`store_design_matrix=True` is refused before any row is read. List the robust covariances
you need in `robust=` instead: they are computed at fit time, in both modes.

## Robust covariances at fit time

```python
model = rs.glm_dict(...).fit(robust=["HC1", "HC3"])
model.bse_robust("HC1")
```

`robust=` computes the listed sandwich covariances during the fit, in either mode, without
keeping the n × p design. A type not listed at fit time is refused.

## Category labels

`rs.observed_category_labels(data, columns, out_of_core=None)` returns each column's
sorted non-null labels exactly as `levels=` matches them. From a `BatchSource` it reads
only those columns in one bounded pass.
