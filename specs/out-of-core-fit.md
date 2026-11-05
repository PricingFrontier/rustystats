# Out-of-Core GLM Fitting — Spec

Status: proposed, revision 7 (5 November 2026). Revision 1 received NEEDS_REWORK, and
revisions 2 to 6 received REQUEST_CHANGES, in Codex plan review; this revision addresses
every finding.

Consumer: Haute `MOD-M01` (adaptive GLM fitting, exact and chunked). Haute decides when to
engage it; RustyStats never engages it on its own.

## 1. Decisions

1. **Full coverage.** Every model the in-memory fit accepts can also be fitted out of core:
   - every family and link;
   - every term type and interaction;
   - target and frequency encoding;
   - fixed, penalised and monotone splines;
   - ridge, lasso and elastic net, with a fixed penalty or one chosen by cross-validation;
   - cross-validation with target encoding;
   - sign constraints;
   - Negative Binomial θ estimation;
   - complements;
   - robust covariances.

   Only two things are refused, because by definition they hold one value per training row:
   `store_design_matrix=True`, and the per-row result accessors of an out-of-core result
   (§4.5).
2. **Bitwise equality.** On the same build and machine, an out-of-core fit and an
   in-memory fit of the same data are bitwise identical in every reported quantity.
   - The equality holds for every source batch size, `encode_rows` and thread count, and
     for every target-encoding sort-run count. The in-memory fit therefore runs the same
     encoder, drivers and kernels (§3).
   - Results change against v0.9.0, once:
     - at rounding level, from the fixed reduction order. That includes the mean
       response that starts the intercept, and alpha_max's null fit. The null fit now uses
       the prior weights and the offset as given; v0.9.0 replaced weights within
       `allclose` of 1 by exactly 1;
     - materially, only where §17 records an owner decision (DEC-1, DEC-2), or where
       §17 records a deliberate compatibility change (DEC-5, DEC-6).
3. **Memory.** Out-of-core RAM is bounded by three things: the design width, the encode
   slice, and the learned state (level sets, encodings, statistics). None of these grows
   with the row count. Per-row state that the chosen algorithms cannot avoid goes to spill
   files on disk, never to RAM:
   - target-encoding training values;
   - the Negative Binomial μ column during the θ search.

   Each phase's allocations are planned and admitted before that phase starts.

Non-goals: approximate or subsampled fits, distributed fitting, GPU, caching the encoded
design on disk, and equality across machines or builds.

## 2. Current state (v0.9.0, measured and read)

### 2.1 Memory

Haute's family memory probe (`tests/performance/test_training_family_memory_perf.py`) ran
on 5 November 2026 against a local release build of v0.9.0. The workload:

- 30 numeric features and 5 categorical features (4, 8, 16, 32 and 64 levels);
- a Poisson target with an exposure weight;
- a 20% holdout;
- 8 threads.

The design has 150 columns.

| Rows | Frame | Peak RSS | Design build (validation fit) | IRLS (validation fit) |
|---:|---:|---:|---:|---:|
| 100,000 | 27 MiB | 628 MiB | +209 MiB | +39 MiB |
| 400,000 | 109 MiB | 1,706 MiB | +790 MiB | +134 MiB |
| 1,600,000 | 436 MiB | 6,292 MiB | +3,405 MiB | +689 MiB |

At 1.6M rows the validation fit's design (1.28M rows) is 1,465 MiB, and the build raises
the peak by 2.3 times that. The causes:

- term blocks are stacked by `_stack_columns` (`interactions.py:1845`);
- dummy matrices stay cached until after the fit (`formula.py:1058-1060`);
- there is one Python `str` per categorical value (`interactions.py:476`);
- `~np.isfinite(X)` allocates boolean masks (`validation.py:74`);
- the bindings copy y, offset and weights (`fitting_py.rs:398-403`).

Every smooth fit copies X into its result (`smooth_glm.rs:1246`). The monotone path also
copies X into `x_tilde` (`smooth_glm.rs:1523`), and cross-validation holds about K design
copies (`fitting_py.rs:1605-1712`).

### 2.2 Passes and determinism

Today an accepted IRLS iteration costs two X passes, and each rejected candidate costs one
more (`irls.rs:1074-1161`). The Gram's chunking (`irls.rs:1537`), the moments
(`fitting_py.rs:233`) and Xβ dispatch (`irls.rs:2100-2190`) all depend on the thread
count.

The golden record (§14) confirms this. v0.9.0's own results differ between thread counts
at rounding level, and between two runs at the same thread count. On one degenerate
fixture (`halving_exhausted`) the retained coefficients move by up to 47% relative.

## 3. Principles

1. **One encoder.** Training encodes rows through one frozen-design encoder in both modes.
   The in-memory design is that encoder's output, written into a single n×p buffer.
2. **One driver per solver.** Each solver's step logic exists once, as a driver that
   requests passes. In-memory mode serves those passes from X; out-of-core mode serves
   them from batches.
3. **One arithmetic.** Both modes run the same kernels, and kernel choice depends on the
   frozen design and n only. Every reduction over rows has one fixed order (§5). Every
   per-row transform is computed row-wise, as today. The rework introduces no algebraic
   rewriting of sums (such as expanding a sum of squared differences) that could change
   cancellation behaviour.
4. **Admitted memory.** Out-of-core mode allocates nothing proportional to n. Each phase
   is planned and admitted against the caller's limit before it starts.
5. **Fail loud.** Contradictory options are refused before any I/O. Changed sources,
   unseen labels, exceeded limits and failed numerical steps raise errors with messages a
   user can act on. No fallback substitutes a value: the identity-covariance fallback is
   removed (§16, D1).

## 4. Lifecycle and API

### 4.1 `BatchSource`

```python
class BatchSource:
    def __init__(
        self,
        open: Callable[[Sequence[str]], Iterable[pl.DataFrame]],
        schema: Mapping[str, pl.DataType],
    ) -> None: ...
```

- **What `open` yields.** `open(columns)` returns an iterable of Polars DataFrames. Taken
  together they hold every row, in a fixed order, with at least the named columns, whose
  dtypes equal `schema`. RustyStats selects those columns and processes every batch in
  slices of at most E rows (§4.3). Batch sizes are arbitrary and may differ from pass to
  pass. Rows and their order must not change; §9 detects any change.
- **Schema.** `schema` is the source's column dtypes, which the caller knows without
  reading rows (from parquet metadata, for example). Validation and design preparation
  use it before any I/O. A batch whose dtypes differ from it raises
  `BatchSourceChangedError`.
- **One open per pass.** RustyStats calls `open` once per pass. It either consumes the
  iterable fully, or calls the iterable's `close()` (when it has one) on abort, error or
  cancellation. The caller owns the reader, its buffering and its memory, including a
  batch larger than E rows.
- **No bundled reader.** RustyStats ships no reader. Haute supplies one over its training
  parquet.

### 4.2 Construction

`rs.glm_dict(response, terms, data=BatchSource(...), ...)` selects out-of-core mode.
Construction:

- validates the terms dict against `schema`;
- records the source;
- does no I/O and learns nothing.

A DataFrame or LazyFrame keeps today's in-memory construction, which now builds its design
through the encoder (§6.8).

### 4.3 Fit

```python
builder.fit(..., robust: Sequence[str] = (), out_of_core: OutOfCoreOptions | None = None)

@dataclass(frozen=True)
class OutOfCoreOptions:
    max_bytes: int | None = None               # admission limit, checked before every phase
    encode_rows: int | None = None             # E; rounded up to a multiple of GRAM_BLOCK_ROWS
    threads: int | None = None                 # accumulation threads; default: the rayon pool
    max_learned_state_bytes: int | None = None # enforced while learned state grows (§6.4)
    max_design_columns: int | None = None      # enforced while level sets grow
    scratch_dir: str | os.PathLike | None = None  # required when the model needs a spill (§10)
    on_batch: Callable[[BatchProgress], None] | None = None
    te_run_records: int | None = None          # records per TE sort run; default from the plan
```

**Default E.** E is the largest multiple of `GRAM_BLOCK_ROWS` that satisfies both of these:

- E × `per_encode_row_bytes` ≤ 256 MiB;
- when `max_bytes` is set, E × `per_encode_row_bytes` ≤ (`max_bytes` −
  `persistent_bytes` − `fixed_bytes`) / 2.

E is at least `GRAM_BLOCK_ROWS`. If even that does not fit, the fit raises
`MemoryPlanExceededError`.

Learning passes run before p is known. They process each batch in slices bounded both by
rows (`E_learn = 65,536`) and by bytes (`learn_slice_bytes`: 64 MiB, or `max_bytes`/16 when
that is less).

Every pass, learning or solver, cuts each source batch into slices before it touches the
rows: fingerprints (§9), input transforms, validation and encoding run a slice at a time,
never on a whole source batch, so a caller's large batch creates no RustyStats allocation
proportional to it. Solver passes cut slices of at most E rows and, when the columns they
read include String, Binary, Categorical or Enum values, of at most `learn_slice_bytes` of
that payload, so the copies the encoder and the fingerprints make of it stay within the
plan (§11).

- **Row sizes.** A slice's bytes are the fixed-width columns' bytes plus the actual
  payload bytes of each row's variable-width values. String and Binary lengths come from
  Polars as an integer column, computed for at most `E_learn` rows at a time, so the
  length scratch is 8·`E_learn` bytes.
- **Cutting slices.** A slice ends before the row that would take its cumulative bytes
  past `learn_slice_bytes`.
- **Oversized rows.** A single row larger than `learn_slice_bytes` forms its own slice.
  Its bytes are charged against the effective learning budget (§6.4) before it is
  processed: before it is even sliced out of its batch (Polars may copy a lone large
  value when it is), fingerprinted, transformed or learned. The charge is the copies
  of it alive while it is processed, 4 × its bytes plus the per-row scratch. If they do
  not fit, `LearnedStateTooLargeError` names the column and the row's size. Row sizes
  come from expressions that slice the column themselves, for the same reason. The
  solver passes plan the largest such row in place of `learn_slice_bytes` (§11).
- **Scratch.** The working set of `unique` and of label formation for a slice (at most
  the slice's bytes, plus a fixed per-entry overhead per row) is reserved from that
  budget before the slice is processed.

So no learning allocation is made before it has been charged (§11).

**Order of work inside `fit` in out-of-core mode:**

1. **Validate (no I/O).** Every option and the whole model are checked. Refused here:
   - `store_design_matrix=True`;
   - unknown `robust` types;
   - `robust` together with `compute_covariance=False`, in both modes, because robust
     covariances need the bread;
   - a spill-requiring model without `scratch_dir`;
   - any contradiction the in-memory fit would also refuse.

   `out_of_core=` is refused in in-memory mode.
2. **Phases in §6.2 order.** Each phase's plan is computed before the phase starts and
   admitted against `max_bytes` (§11). The learned-state and design-column limits are
   enforced while state grows. `on_batch` runs throughout.
3. **Solver passes** (§7), admitted the same way.
4. **Result.** Spill files are removed on every exit path: success, error, cancellation, or
   an exception raised by `on_batch`.

`robust=` lists the HC types (`"HC0"` to `"HC3"`) to compute at fit time, in the final
pass. `bse_robust` and its siblings serve those types. In-memory `store_design_matrix=True`
keeps working and still serves any type on demand.

`te_run_records` lets tests force the number of target-encoding sort runs (§6.6). It never
changes results.

### 4.4 Progress

`on_batch(BatchProgress)` is called after each source batch, before the next one is
requested.

`BatchProgress` fields:

- `pass_index`;
- `pass_kind`: `"freeze"`, `"knots"`, `"te_stats"`, `"te_sort"`, `"te_merge"`, `"te_walk"`,
  `"te_average"`, `"moments"`, `"start"`, `"candidate"`, `"trials"`, `"gram"`,
  `"reconstruct"`, `"final"`, `"fold"`, `"null_fit"`, `"theta"`, `"refresh"`, `"labels"`;
- `rows_done`;
- `n_rows`: None until the freeze pass ends.

Spill stages read no source batches: `te_merge`, `te_walk` and `te_average`. They call
`on_batch` after every 2²⁰ records they process, and once at the end of each stage. For
these stages `rows_done` counts records and `n_rows` is the stage's total record count.

An exception raised by the callback propagates unchanged. Before it does, the iterable or
open spill files are closed and every spill file is removed. This is Haute's checkpoint
and cancellation hook.

### 4.5 Result

**Both modes gain:**

- `fit_mode` (`"in_memory"` or `"out_of_core"`);
- `passes`: a count of I/O passes per `pass_kind`;
- `solver_trace`: a tuple of `SolverStep(driver, kind, evaluations, accepted, status)`
  recording the drivers' semantic steps. Learning and I/O passes are not included, so the
  trace is identical whatever the target-encoding sort-run count or batching. Its length is
  bounded by the solver caps;
- `solver_settings`: `max_iter`, `tol`, the halving caps, `GRAM_BLOCK_ROWS`, the
  target-encoding seed actually used, and the Negative Binomial θ tolerance and round
  count.

**Solver trace vocabulary.** A driver records a step at each decision it makes, never per
pass, so speculation (every halving trial, bracket score or bisection level in one pass
out of core, one at a time in memory) changes no step. `evaluations` counts what the
driver's sequential rule read, in its order: for halving, the trials up to and including
the first acceptable one, or all of them when none is.

| Driver | Kind | `evaluations` | `accepted` | `status` |
|---|---|---|---|---|
| `irls` | `iteration` | the full step, then the halvings up to the first acceptable one (all 25 when none is) | the iterate moved | `full_step`, `halved`, `exhausted` |
| `irls` | `restore` | 0 | true | `constrained_best` (the best sign-constrained iterate replaces the last) |
| `irls` | `extraction` | 1, pass A's deviance check; 0 when the re-solve is not finite | the re-solved coefficients were kept | `accepted`, `rejected`, `nonfinite` |
| `cd` | `iteration` | 1 | true | `sweeps_converged`, `sweeps_exhausted` (the coordinate sweeps hit their cap) |
| `smooth`, `monotone` | `lambda_update` | 0 (λ is optimised on the Gram) | λ changed | `gcv` (smooth); `reml` (monotone: the start, then each outer update) |
| `smooth`, `monotone` | `iteration` | the candidate, then the trials up to the accepted one (all when none is) | a trial was accepted | `full_step`, `halved`, `retained` (the fraction-0 fallback), `exhausted` |
| `monotone` | `warm_start` | 1 | true | `step`, `converged` |
| `monotone` | `alpha_reset` | 1 | true | `reset` |
| `monotone` | `refine_iteration` | as `iteration` | as `iteration` | as `iteration` |
| `monotone` | `refine` | 0 | the refined block was kept | `applied`, `rejected` |
| `smooth`, `monotone` | `refresh` | the intercept scores read (at 0, each bracket pair, each bisection midpoint), plus the deviance at a non-zero shift | the shift was applied | `applied`, `rejected`, `unbracketed`, `zero` |
| `irls`, `cd`, `smooth`, `monotone` | `result` | the iterations | converged | the solver status: `converged`, `max_iterations`, `step_halving_no_improvement`, `stalled_nonstationary` |
| `alpha_max` | `null_fit` | 1 for the closed form, else the width-1 IRLS's iterations | converged | `closed_form`, or the IRLS's status |
| `alpha_max` | `alpha_max` | 1 for the score or column-norm pass, else 0 | false only when it failed and the fold was floored (§7.9) | `score`, `ridge_norms`, `ridge_diag`, `floor`, `failed` |
| `cv` | `fold_alpha` | the fold fit's iterations (0 on error) | the fit succeeded and scored finite | the fit's status, or `error` |
| `nb_theta` | `initial_theta` | 1 for the moment θ₀, 0 when θ₀ is given | true | `moments`, `given` |
| `nb_theta` | `round` | the profile likelihood's θ evaluations | θ moved by less than the tolerance | `converged`, `continue` |

Order: alpha_max's steps, then the CV steps (one per fold and alpha, folds in order;
cross-validation with target encoding adds them fold by fold), then the fit's. A θ
estimation records the Poisson fit's steps, `initial_theta`, each round's NB fit and
`round`, then the final fit's. The width-1 IRLS inside alpha_max and the folds' own fits
are summarised by their `null_fit` and `fold_alpha` steps, not recorded step by step. Each
driver's steps are bounded by its caps.

**Out-of-core mode only:** `encode_rows` and the `memory_plan` (per-phase peaks).

**An out-of-core result holds no per-row arrays,** its training inputs included: it keeps
column specs, and for an exposure or offset given as an array only the fact that
prediction must be given one for the new data (an array of weights or a complement is
not kept at all), so `to_bytes` does not grow with n. It serves:

- `params` and `feature_names`;
- `link`, `family`, `nobs`, `df_model` and `df_resid`;
- `iterations`, `converged` and the solver status;
- `deviance`, `null_deviance()`, `scale()` and `pearson_chi2`;
- `llf()`, `aic()` and `bic()`;
- `bse()`, `tvalues()`, `pvalues()` and `conf_int()`, plus their robust versions for the
  types listed in `robust=`;
- `inference_status` and `is_quasi_likelihood`;
- smooth-term results (`has_smooth_terms()`, `smooth_terms`, `total_edf`, `gcv`);
- regularisation results (`alpha`, `l1_ratio`, `n_nonzero()`, `n_cv_folds`,
  `cv_selection_method`, the CV path and fold scores);
- `theta`;
- `response_scale`;
- `predict(data)`, `to_bytes()` and `from_bytes()`.

**Refused with `OutOfCoreResultError`:** methods that return or need per-row training
values. These are `fittedvalues`, `linear_predictor`, `resid_*`, `fit_diagnostics`,
`relevel`, `diagnostics` without data, and a robust type not listed at fit. The message
names the method.

### 4.6 Observed labels

`rs.observed_category_labels(data, columns, *, out_of_core=None) -> dict[str, list[str]]`
takes a DataFrame or a `BatchSource`.

- **What it returns.** For each column, the labels of its **non-null** values, under the
  whole-column conversion and the explicit-level renderer of §6.3. This matches Haute's current helper
  (`_rustystats.py:49-58`), which excludes null rows but keeps the null-driven dtype
  coercion.
- **How it runs.** With a `BatchSource` it runs one `labels` pass that reads only those
  columns. It uses the same options (`max_bytes`, `max_learned_state_bytes`, `on_batch`),
  admission and slicing as the freeze pass, so it is bounded and cancellable.
- **Haute's use.** Haute calls it with the job's options before building the terms, then
  hands the result to `prepare_glm_design`. Column dtypes come from `BatchSource.schema`.

### 4.7 Errors

New classes in `rustystats.exceptions`, each with a message a user can act on:

| Error | What the message names |
|---|---|
| `OutOfCoreOptionError` | The contradictory option. |
| `BatchSourceChangedError` | The pass and the column (a hash mismatch), or the dtype or row-count difference. |
| `UnseenLabelError` | The column and the label. |
| `DesignTooWideError` | p, the limit and the term that crossed it. |
| `LearnedStateTooLargeError` | The term, its cardinality and its bytes. |
| `MemoryPlanExceededError` | The phase, its planned peak and the limit. |
| `SpillError` | The path and the OS error. |
| `OutOfCoreResultError` | The unavailable method. |

## 5. Arithmetic contract

### 5.1 Shared kernels

Both modes call the same functions for every per-row computation:

- **Linear predictor.** Xβ + offset per row is a scalar loop over columns in index order.
  Neither `ndarray::dot` nor a dispatch on n, p or thread count is involved. Today's
  dispatch (`irls.rs:2100-2190`) is replaced in both modes, and in `predict` (§6.8).
- **Family and link arithmetic.** Inverse link, clamping, variance, link derivative, IRLS
  weights (with the existing [1e-10, 1e10] clamp, or true-Hessian weights where a family
  uses them) and the working response. Unit deviances use `unit_deviance_at`. The
  Poisson-log in-place weight arithmetic (`irls.rs:1670-1755`) is preserved as one shared
  helper.
- **Block Gram kernel.** It accumulates X'WX (packed upper triangle), X'Wz and z'Wz over a
  block's rows, in row order, as `(x_ki·w_k)·x_kj` (`irls.rs:1563-1582`). The dense
  variant and the sparse variant (per-block CSR from the encoded block) evaluate the same
  expression over the same nonzero products in the same order. Their sums are therefore
  identical, because a skipped product is an exact zero.
- **Row transforms.** The monotone row transform (`smooth_glm.rs:1088-1107`) and the
  adjusted working response (`smooth_glm.rs:1763-1776`) are applied row-wise per block.
- **Step and projection helpers.** `blend_coefficient` (`irls.rs:521`, (1−t)·old + t·new),
  `project_coefficients_to_sign_constraints`, the acceptance tests including finiteness
  (`trial_state_is_acceptable`), and every existing convergence predicate are reused as
  they are.
- **Robust and diagnostic sums.** Robust meat and leverage, Pearson χ², the
  log-likelihood terms, KKT sums and score sums.

### 5.2 Dispatch

Dense or sparse kernel choice depends only on two things:

- the frozen design's *structural density*: the maximum nonzeros per row that each term
  can produce, divided by p;
- n.

It never depends on sampled values (`sampled_density`, `irls.rs:1775`) or on the thread
count. The thresholds stay as they are today (`irls.rs:1798-1820`). Kernel choice cannot
change results (§5.1); it only affects speed.

- **Rule.** Rows are served sparse when p ≥ 16, n·p ≥ 10⁷ and the structural density is
  at most 0.35 (`passes::sparse_rows_for`). The bound reaches the executors as
  `max_row_nonzeros`, on the batched bindings and on the in-memory ones.
- **The bound.** `FrozenDesign.max_row_nonzeros` sums each term's most nonzeros per row,
  capped at p: 1 for the intercept, a numeric column, an `I()` or `pos()`/`neg()` term, a
  frequency- or target-encoded term, a categorical's dummies and a categorical
  interaction's cells; degree + 1 for a B-spline basis (inside and beyond its boundary
  knots); the width of a natural or I-spline basis; the product of the factors' bounds for
  a spline tensor product; the sum of the splines' bounds for a categorical × spline
  interaction; one per target-encoded factor of a continuous interaction; and the most
  listed levels one label matches for `levels=`.
- **No whole-design cap.** Today's further cap, 6·10⁷ estimated nonzeros, bounded a CSR copy
  of the whole design. The executors now form each reduction block's rows as CSR as the
  block is reduced (`passes::BlockCsr`, §11's per-block CSR), so the cap no longer applies.
- **Without a frozen design.** An in-memory fit given X with no structural bound (a direct
  call to a binding) uses X's own maximum nonzeros per row. It scans every row exactly and
  samples nothing, and runs only when n and p allow sparse rows. Out of core, no bound
  means dense rows.
- **Centring.** A centred row standardisation (§6.7) makes every row dense in its centred
  columns, so those fits are served dense rows, as today.
- **Explicit kernel.** The fit bindings also take `row_kernel="dense"|"sparse"`
  (`passes::RowKernel`), which gives the kernel whatever the design.
  - It is an explicit argument of the private `_rustystats` bindings only. The public API
    does not expose it, and no environment variable or hidden flag selects a kernel.
  - Tests use it to show that both kernels give the same bits.

### 5.3 Reductions

`GRAM_BLOCK_ROWS` is a crate constant, proposed at 4,096 (DEC-4).

- **Blocks.** Global row r belongs to block ⌊r / GRAM_BLOCK_ROWS⌋. A block's rows are
  reduced in row order, by one thread, into one partial per output. Partials are added in
  block order.
- **Waves.** Threads take blocks in waves of T. The partials of a wave are added in block
  order before the next wave starts, so at most T + 2 partials of each output exist at
  once: the accumulator, the open block's (below) and one per thread.
- **Open-block carry.** A batch or slice may end inside a block. That block's partials
  stay open, and the next rows continue them in row order. No rows are copied, and block
  boundaries follow global row indices whatever the source's batching or the slicing.
- **Coverage.** Every sum over rows goes through this mechanism in both modes:
  - Gram, X'Wz, z'Wz and the mixed right-hand side;
  - deviance, Pearson χ² and log-likelihood terms;
  - null-model sums and moments;
  - means, KKT sums, score sums and validation sums;
  - target-encoding prior sums and Negative Binomial θ terms.
- **Order-dependent statistics.** Target-encoding level totals are prefix sums, not
  reductions. They follow the processing order of §6.6.
- **Other reductions.** Non-sum reductions are order-independent and exact: AND
  (finiteness, the all-ones intercept test, the all-unit-weights flag), min/max, integer
  counts and exact selection. Python-side NumPy reductions in the fit path move to Rust,
  for example the standardisation fallback in `regularization_path.py:382-421`.

### 5.4 Scope

The bitwise guarantee holds for one compiled binary on one CPU. Both modes then execute
the same instructions. RustyStats does not enable fast-math or rely on implicit FMA
contraction, and explicit `mul_add` appears only where both modes share it. Equality
across machines or builds is not promised.

## 6. Learning: the frozen design

### 6.1 State per term

| Term | Learned state | Learned in |
|---|---|---|
| linear, `pos()`/`neg()`, `I()` | none (`I()` is limited to elementwise arithmetic by `_convert_expression_to_polars`, `interactions.py:1403`) | — |
| categorical (discovered) | label set from whole-column conversion (§6.3); reference = first label | freeze |
| categorical with `levels=` | none. Indicators for the listed labels; any other label encodes as all zeros, as today (`interactions.py:1464-1505`) | — |
| bs / ns / ms (fixed, penalised or monotone) | boundary min/max unless given; interior knots as exact order statistics (§6.5) | freeze, knots |
| interaction-local splines (`formula.py:4548-4555`) | as above | freeze, knots |
| frequency encoding | count per label (integers), `max_count` | freeze |
| target encoding | prior and `prior_weight` "auto" (freeze); per-label totals (te_stats); training values (§6.6) | freeze, te_stats, TE sort |
| interactions | the states of their factors | — |
| input transforms | none (user given) | — |
| exposure, offset, weights | none | — |
| complement | none for a column; a `GLMModel` keeps its own fitted state | — |
| standardisation | centres and scales (§6.7) | moments |

Validation and global aggregates (freeze pass):

- the response checks per family;
- weights ≥ 0, Σw > 0 and the zero-weight fraction;
- exposure > 0;
- the constant-response check and binomial values;
- n_obs < p;
- n, Σw, Σy, Σw·y, Σw·exp(offset) and max |y| over finite y (`response_scale`,
  `results_py.rs:205`);
- the all-unit-weights flag used by `maybe_weights` (`results_py.rs:118`).

Every slice is validated before it contributes to any statistic. This removes today's
order where target-encoding statistics read y before it is validated (`interactions.py:1668`
vs `formula.py:5162`).

### 6.2 Phase order

| Phase | Reads | Produces |
|---|---|---|
| freeze | every model input column | §6.1 freeze state; label tables; baseline fingerprints (§9) |
| knots (exactly 4 when any spline has data-derived interior knots) | spline input columns | interior knots |
| te_stats (when TE is present) | TE inputs | per-label totals, accumulated in row order after labels are fixed |
| te_sort, te_merge, te_walk, te_average (per term × permutation × fold) | TE inputs once (te_sort), then spill files only | target-encoding training values (spilled) |
| moments M1, M2 (when standardisation applies) | encoded design | centres and scales |
| solver | as §7 | the fit |

Cross-validation with target encoding learns fold designs in the same order, per fold:
freeze state for all folds in the one freeze pass, then knots, te_stats, the TE sort
stages and moments for every fold.

### 6.3 Categorical labels

Training labels today are `[str(v) for v in column.to_numpy()]` over the whole column
(`interactions.py:475-476`). So a label depends on the NumPy dtype of the whole column.
For example, an Int64 column with any null becomes float64, and its labels become `"1.0"`
and `"nan"`.

The freeze pass reproduces this without creating a Python object per row:

1. **Keys.** For each categorical, frequency-encoded and target-encoded column, every
   slice yields a set of distinct *training keys*. A key preserves every distinction
   training's `str()` makes:

   | dtype class | key |
   |---|---|
   | String, Categorical, Enum | the UTF-8 string; null as its own key |
   | Boolean | the value; null as its own key |
   | integers | the value; null as its own key |
   | floats | the IEEE bit pattern (so −0.0 and +0.0 differ), with every NaN mapped to one canonical NaN (training renders every NaN as `"nan"`); null as its own key |
   | Date, Datetime, Duration, Time | the physical integer, plus a tag for the unit and time zone; null as its own key |
   | Decimal | the (unscaled integer, scale) pair; null as its own key |
   | Binary | the raw bytes; null as its own key |
   | List, Array, Struct, Object, Null | refused for categorical, frequency-encoded and target-encoded terms in both modes (DEC-5) |
   | Int128, UInt128 | refused: today's conversion cannot label them (Polars `to_numpy` panics on these dtypes) |

   Keys are collected with Polars `unique` over the slice's column, reinterpreted to the
   key form. The work is bounded by the slice size.
2. **Labels.** At the end of the freeze pass, whether the column has a null anywhere is
   known, and so is its whole-column NumPy dtype. Each key is turned back into a
   one-element Polars Series of the column's dtype and converted with that whole-column
   NumPy dtype. Two renderers then apply to the same keys, because today's code uses
   two:
   - **Discovered labels** (categorical level discovery, frequency and target
     encoding) render with `str()` of the NumPy element (`interactions.py:475`).
   - **Explicit-level labels** (matching `levels=`, `interactions.py:1499`, and
     `observed_category_labels`, §4.6) render with NumPy `astype(str)`.

   The two differ in two cases. For Binary, `str()` gives `"b'a'"` and `astype(str)`
   gives `"a"`; a Binary value that is not ASCII has no explicit-level label (today
   `astype(str)` raises `UnicodeDecodeError`) and raises `EncodingError`. For a string
   ending in `"\x00"`, `astype(str)` strips the trailing NUL characters.

   **Label memory.** Labels are formed one key at a time, after the freeze pass, from
   the key tables whose payload sizes are known. Before label formation starts, the
   learning budget (§6.4) reserves two amounts:
   - **Persistent:** each label as a Python `str`, held in the level list and
     returned by `observed_category_labels`. Python may store a string at 4 bytes per
     character, and a character takes at least one UTF-8 byte, so the bound is
     4·L + 80 bytes, where 80 bytes covers the object header and L is:
     - the payload bytes for String and Categorical keys;
     - 4·payload + 3 for Binary keys under `str()` (the `\xNN` escapes and the `b''`
       wrapper);
     - a fixed bound per dtype for other scalar dtypes.
   - **Transient**, for the single largest key: NumPy's fixed-width Unicode array
     (4 bytes per character, at most 4·L), plus that key's Python label (4·L + 80),
     both alive at once.

   If either reservation does not fit, `LearnedStateTooLargeError` names the column and
   its largest key before any label is formed.
   - Several keys can share one label, for example integers above 2⁵³ in a column that
     becomes float64. Those keys map to the same code.
   - Frequency-encoding counts are integers, so merging them is exact.
   - Target-encoding totals are floats, so they are not taken from the freeze pass. They
     are accumulated per label, in row order, in the te_stats pass after the mapping is
     fixed.
3. **Encode.** The encoder maps keys to codes with the frozen key-to-code table
   (Polars `replace_strict` over the key form).

A test pins each dtype class, including signed zeros, NaN payloads and integers above 2⁵³
with nulls.

`levels=` columns use their listed labels, and other labels encode as zero. In out-of-core
mode, a discovered column that meets a key missing from its table raises
`UnseenLabelError`; the source has changed.

### 6.4 Learned-state limits

Freeze work runs per learning slice (§4.3), so a slice's distinct keys are bounded both by
rows and by bytes before they are merged.

**Effective budget.** The learned-state budget is the smaller of two figures:
`max_learned_state_bytes`, and what `max_bytes` leaves after the freeze phase's other
allocations. When neither is set, there is no limit.

**Charged before allocation.** For each slice, the freeze pass first computes the bytes
its new entries would add to the tables: each new key's payload (string and binary
lengths come from Polars without materialising Python objects), a fixed per-entry
overhead, and table growth at the next capacity step. Only if the result fits does it
insert them.

The freeze pass keeps a running byte count of learned state:

- key and label tables, including each entry's key bytes, label bytes and a fixed
  per-entry overhead;
- frequency-encoding counts;
- target-encoding statistics;
- every fold's copies in cross-validation with target encoding.

A slice that would cross the effective budget raises `LearnedStateTooLargeError`, naming the
term and its cardinality, before anything is allocated. Spline knots (a learned boundary
pair and each interior knot) are charged when the model is validated, and centres and
scales (§6.7) before the moments passes, so `max_learned_state_bytes` covers them too.

Crossing `max_design_columns` raises `DesignTooWideError` the same way, naming p, the limit
and the term that crossed it. The width is judged from the frozen design's term layout
(§6.8) and each discovered column's label count:

- before the source is opened, from the columns that need no data;
- after every freeze slice, from the fewest labels the keys so far can render to (keys
  render to distinct labels, except integers beyond ±2⁵³ once a null makes the column
  float64, and a float column's null beside its NaN);
- from the final labels, before any column is laid out.

### 6.5 Knots

The in-memory knot is `sorted[knot_rank(i, m, n)]` over the column sorted by
`f64::total_cmp`, with duplicates kept. Here `knot_rank` is
`floor(i/(m+1) · (n−1))` (`splines/mod.rs:72-84`, `147-167`). `knot_rank` becomes one
shared function.

Out-of-core computes the same order statistic exactly, by MSD radix select:

- **Key.** `key(x) = !bits(x)` when the sign bit is set, else `bits(x) ^ (1 << 63)`.
  Unsigned key order then equals `total_cmp` order, including −0.0 < +0.0. Spline inputs
  must be finite; the freeze pass refuses a non-finite value with a message naming the
  column.
- **Pass j (j = 1..4).** For every rank, the pass counts keys in a 65,536-bucket histogram
  of key bits [64−16j, 80−16j) among values whose higher bits match the rank's resolved
  prefix.
  - Ranks of one column that share a prefix share one histogram.
  - Each rank then moves to the bucket holding it and subtracts the counts below.
- **Termination.** After pass 4 every rank is resolved, and the value is decoded by
  inverting the key. There is no early exit: always four passes, each reading only spline
  columns.
- **Memory.** A histogram is 65,536 × 8 bytes = 512 KiB, charged per distinct prefix. In
  pass 1 that is one per spline column; afterwards at most one per rank.

### 6.6 Target encoding

The training value today is CatBoost's ordered statistic over an LCG Fisher–Yates
permutation of all n rows (`target_encoding/mod.rs:228-285`).

DEC-1 (§17) replaces it in both modes with a keyed order:

- **Processing order.** Permutation m (0 ≤ m < `n_permutations`) processes rows in
  ascending `k_m(r) = (mix64(mix64(seed + m) ^ r), r)`. Here `mix64` is SplitMix64's
  finaliser, r is the global row index, and ties fall back to r. In cross-validation with
  target encoding, only fold-training rows take part.
- **Value.** A row's value is (S_<(r) + prior·pw) / (C_<(r) + pw). S_< and C_< are the
  label's running totals of y (or claims) and counts (or exposure) over rows earlier in
  that order. The totals are added one row at a time in that order, as today. Values are
  averaged over permutations in permutation order, with today's expression.

**Exact computation with bounded RAM.** For each TE term, permutation and (in
cross-validation with target encoding) fold, the values come from an external sort:

1. **te_sort, one source pass.** It reads the TE inputs and forms 40-byte records:
   key (u64), row (u64), code (u32 plus padding), target (f64), exposure (f64). Every
   `R_run` records are sorted by key in RAM and written as a run file.
2. **te_merge.** Runs are merged by key with fan-in F = 64 and 1 MiB buffers, in
   ⌈log_F(runs)⌉ − 1 intermediate stages, until at most F runs remain.
3. **te_walk.** The last merge streams records in ascending key order into the walk. The
   walk keeps per-label running totals (one row at a time, as today) and emits
   (row u64, value f64) records. These are written as runs of `R_run` records, each
   sorted by row.
4. **te_merge by row.** The row runs are merged with the same fan-in. The final stage
   writes the permutation's row-aligned float64 column.
5. **te_average.** One sequential pass averages the permutation columns, in permutation
   order, with today's expression, into the term's row-aligned float64 spill.

**Bounds.**

- Keys are unique (ties fall back to the row), so any external sort yields the same
  order as an in-memory sort.
- Run k of a stage is named by (term, permutation, fold, stage, k), so the planner keeps
  only counts, never a list of runs.
- RAM is `R_run × 40 × 2` (records and sort scratch) during te_sort and te_walk, plus
  (F + 1) × 1 MiB during merges. Neither depends on n or on the key distribution.

**In-memory mode** sorts all records in RAM, walks them and keeps the values, with no
spill. It walks the same key order and adds the same totals, so the two modes are bitwise
identical whatever `R_run` is.

**Settings.**

- `R_run` defaults to `te_sort_bytes / 80`, where `te_sort_bytes` is the plan's TE budget
  (§11). `te_run_records` overrides it for tests.
- `seed=None` draws a seed once per fit, as today, and the seed actually used is
  recorded in `solver_settings`.
- Predict-time encoding is unchanged: full statistics, with the prior for unseen labels.

### 6.7 Standardisation

Standardisation applies when `alpha > 0` and `standardize=True` (`formula.py:938-962`).
The moment arithmetic of `fitting_py.rs:208-352` is kept exactly; only the reduction order
changes (§5.3).

- **Pass M1:** S_j = Σw·x_j and W = Σw (or n); m_j = S_j/W.
- **Pass M2:** V_j = Σw(x_j − m_j)², computed row-wise from the subtracted value;
  var = max(V_j/W, 0); sd = √var.
- **Invalid columns.** Columns with a non-finite moment or sd ≤ 1e-12 keep centre 0 and
  scale 1.
- **Ridge diagonal.** `ridge_xtx_diag` follows `fitting_py.rs:334-346`.
- **Centres.** `solver_standardization` zeroes the centres when an intercept is fitted
  (`regularization_path.py:444-459`).
- **Routes.** The two routes stay as they are (`irls.rs:665-753`):
  - **Scale-only ridge** (ridge, all centres 0): X is not transformed, and the penalty
    factors are s_j².
  - **Every other standardised fit:** the encoder writes (x − c)/s into the block in place,
    with `standardize_matrix`'s expression and its skip rule (`regularization/mod.rs:409-424`).
    Coefficients and covariance go through the existing back-transforms. The n×p copy is
    removed in both modes.

### 6.8 Encoder, complement and predict

`FrozenDesign.encode_into(rows, out)` writes the encoded rows of a slice into a
caller-owned C-order float64 block.

- Each term writes its columns directly into its slot of `out`. There are no term blocks,
  no stacking and no cached dummies.
- Spline bases are evaluated by block-level functions that write into the block, with no
  allocation per row.
- Input transforms are applied per slice.
- Target-encoding columns come from the spill (out of core) or from the in-memory column.
- Encoder scratch per row is declared by the frozen design and appears in the plan.

**In-memory build.** Allocate X (n×p) once, then call `encode_into` over consecutive row
slices. The old `_build_design_matrix_core` is deleted once encoder parity (T1) passes.

**Complement.** Complements keep today's semantics (`formula.py:1150-1218`), evaluated per
slice in both modes:

- **A column or array complement** is read per slice. It is already on the receiving
  model's response or rate scale and is not divided by exposure.

  Exposure, offset and weights may likewise be columns or arrays. An array is read a
  slice at a time by global row, never copied whole, and the freeze pass refuses one that
  does not hold exactly n values.
- **A `GLMModel` complement** is `complement.predict(slice, response_ceiling=None)`. That is
  the public predict pipeline: its own encoding, its stored offset handling and its
  inverse link. The result is then divided by the receiving model's raw exposure when one
  is set (`formula.py:1185`).
- **Then, for both:**
  1. validate (finite, and in the receiving family's domain);
  2. apply the receiving model's link;
  3. add the result to the offset.

  As in memory, finiteness is checked on the combined link-scale offset, so a non-finite
  complement raises the same `ValidationError` ("offset contains …") in both modes. The
  freeze pass validates every slice's offset before it reads any statistic.

**Predict.** The predict pipeline's linear predictor moves to the shared row kernel
(§5.1). Its per-row values then depend on neither slice boundaries nor its internal
chunking (`formula.py:168-172`, `243-254`). Predictions change against v0.9.0 at rounding
level only. Predict-time label conversion is unchanged (D4).

## 7. Driver and passes

### 7.1 Pass model

A pass serves a set of **evaluations**. Each evaluation names:

- the per-row state it needs (§7.2);
- the outputs to reduce;
- for cross-validation, the fold filter.

| Output | What is reduced |
|---|---|
| deviance + finite | penalised objective parts; AND of finite η, μ, D |
| Gram | X'WX (packed), X'Wz and z'Wz at the evaluation's μ and weights, on X or on the row-transformed X̃ for monotone smooths |
| mixed RHS | X'W(weights state)·z(predictor state) |
| Pearson | Σw(y−μ)²/V(μ), with weights per `maybe_weights` |
| llf terms | family-specific per-row terms |
| robust meat | one meat per requested HC type; HC2/HC3 leverages via the supplied bread |
| KKT | Σr·x and Σf·x² over the non-smooth columns |
| score at shifts | intercept score S(d) for a set of shifts d |
| null | null-model unit deviances |
| validation | Σw·d and Σw for a fold's held-out rows |
| θ terms | the Negative Binomial log-likelihood terms at a θ |
| μ spill | μ written row-aligned to the spill |

Every driver below is ordinary sequential Rust that calls a **pass executor**
(`solvers/passes.rs`):

```rust
pub trait PassExecutor {
    fn n_rows(&self) -> usize;
    fn n_params(&self) -> usize;
    fn run(&mut self, request: &PassRequest) -> Result<Vec<EvalSum>>;
}
```

A `PassRequest` lists predictors (§7.2) and evaluations. One function, `process_rows`,
computes every evaluation for a range of rows; both executors call it through
`rows::BlockReducer`.

- **In memory**, the executor serves passes from X.
- **Out of core**, the executor lives in the bindings crate. For each pass it calls back
  into Python to open the source, encode each slice into the block buffer and invoke
  `on_batch`, and it feeds the blocks to `process_rows` with the GIL released.
  - A Python exception raised during a pass is kept by the executor and returned as a
    distinct core error, `RustyStatsError::PassAborted`. It is never a numerical
    failure:
    - Every recovery boundary propagates it unchanged. That covers CV's "failed alpha
      scores INF" (`fitting_py.rs:1644`), CV with target encoding's fold and α_max
      handlers (`regularization_path.py:1174`, `:1237`), and the fallbacks inside
      drivers.
    - On the Python side it travels as an internal `_PassAborted`, which derives from
      `BaseException` so that `except Exception` and `except ValueError` handlers do
      not catch it.
  - **Latched cancellation.** After the first abort, every later `run` on any executor
    of the same fit returns the same error at once, without reading the source.
  - **Outer boundary.** `fit()` catches `_PassAborted`, closes the source, removes spill
    files and re-raises the original exception object, so `on_batch` exceptions reach
    the caller unchanged (§4.4).
  - The core crate stays free of Python.

The drivers therefore keep today's loop structure line for line. Only their row access
changes, which keeps them reviewable against v0.9.0.

### 7.2 Per-row state provenance

A driver keeps no per-row state. Instead, every evaluation names how its per-row state is
derived, as a tagged **provenance**.

**Predictor provenance (η and μ):**

| Tag | Derivation | Example |
|---|---|---|
| `Coef(β)` | η = Xβ + o; μ = clamp(g⁻¹(η)) | IRLS candidate |
| `CoefRoundTrip(β)` | μ = clamp(g⁻¹(Xβ + o)); η = g(μ) | coordinate-descent warm start, `coordinate_descent.rs:142` |
| `Initializer` | μ = `initialize_mu`(y) or the safe initialiser chosen by the freeze-pass validity flag; η = g(μ) | cold start, `coordinate_descent.rs:144-151`, `irls.rs:818-893` |
| `Partitioned(R, β_R, F, β_F)` | η = Σ_{j∈R} x_j·β_j + o_F, with o_F = (((o + x_{F₁}·β_{F₁}) + x_{F₂}·β_{F₂}) + …) over the fixed columns F in ascending order, and R in ascending order; μ = clamp(g⁻¹(η)) | monotone refinement's retained state, `smooth_glm.rs:745-751`, `:822` |
| `Shifted(P, d)` | η = η_P + d for any predictor provenance P; μ = clamp(g⁻¹(η)) | after the intercept refresh, `smooth_glm.rs:2307-2317`, including after refinement |

**Weights provenance.** Raw working weights and solve weights are distinguished, because
routes store and reuse them differently:

| Tag | Weights | Example |
|---|---|---|
| `Raw(predictor, route)` | the route's raw working weights at that predictor's μ, with the route's own expression | ordinary IRLS and CD use `compute_irls_weights` (`solvers/mod.rs:222-347`), including family helpers such as Tweedie-log's direct power form and true-Hessian weights; smooth fits use `smooth_glm.rs:300-329` |
| `Solve(raw)` | raw × prior weights, formed as each route forms them today (`combine_prior_and_irls_weights`, `irls.rs:552`) | every Gram |
| `Prior` | prior weights | the start projection |
| `Ones` | all ones | smooth `final_weights` before any main iteration, `smooth_glm.rs:1489` |
| `Zeros` | all zeros | ordinary IRLS `final_weights` when no iteration runs, `irls.rs:922` |

**Retained state.** A driver retains:

- the predictor provenance its result was actually computed from;
- the weights provenance of the Gram the result's covariance needs;
- the objective and status;
- for constrained fits, the same for the best iterate.

The reconstruct pass R (§7.4) evaluates exactly the retained provenances. The branches:

| Branch | Result predictor | Covariance weights |
|---|---|---|
| IRLS cold start, first iteration's weights | — | `Solve(Raw(Coef(β₀), ordinary))`: the start projection replaces η and μ with β₀'s before the first iteration (`irls.rs:853-893`) |
| IRLS warm start, first iteration's weights | — | `Solve(Raw(CoefRoundTrip(β_init), ordinary))`: η = g(μ) is kept |
| IRLS converged / max_iter | `Coef(last accepted)` | `Solve(Raw(predecessor, ordinary))`, where the predecessor is the predictor the last step was solved from (`Coef(β₀)`, `CoefRoundTrip(β_init)` or `Coef(β_k−1)`) |
| IRLS halving exhausted | the retained iterate's actual predictor (`Coef(β₀)`, `CoefRoundTrip(β_init)` or `Coef(β_k)`) | `Solve(Raw(that predictor, ordinary))` (`irls.rs:1145`) |
| IRLS constrained best restored | `Coef(best)` | best's own covariance weights (`irls.rs:1186`, `1238`) |
| IRLS with no iteration (`max_iter = 0`) | the start predictor | `Zeros` |
| CD cold, `max_iter = 1` | `Coef(β₁)` | `Solve(Raw(Initializer, ordinary))` |
| CD warm start, first iteration | `Coef(β₁)` | `Solve(Raw(CoefRoundTrip(β_warm), ordinary))` |
| smooth fits | `Coef`, `Partitioned` (after refinement), or `Shifted` of either | `Solve(Raw(..., smooth))`, or `Ones` |

### 7.3 Candidates, trials and the Gram

1. **Candidate pass.** Each new candidate gets one pass. It evaluates deviance and
   finiteness, and speculatively accumulates the Gram at the candidate.
2. **Trials pass.** If the driver rejects the candidate, one pass evaluates all remaining
   step trials at once (deviance and finiteness only). The driver then replays its
   sequential acceptance rule over the results. Examples are IRLS's 25 halvings and the
   smooth loop's fractions 0.5¹…0.5¹⁰ then 0.
3. **Gram pass.** One more pass accumulates the Gram at the accepted trial.

So an iteration costs 1 pass when the full step is accepted, and at most 3 otherwise.

### 7.4 IRLS (plain, ridge, sign-constrained)

The driver reproduces `fit_glm_core` (`irls.rs:776-1417`):

- μ₀ from `Initializer`;
- the start projection with `Prior` weights;
- candidates against the acceptance threshold `IRLS_ACCEPT_REL_SLACK`;
- up to 25 halvings;
- `step_halving_no_improvement`;
- best-iterate tracking with the 2-iteration stale window;
- convergence by `irls_step_converged`;
- `max_iter` and `tol` unchanged.

Final extraction (`irls.rs:1263-1379`) runs as follows:

- **Reconstruct pass R**, over the retained provenances. It computes the Gram under the
  retained weights and the mixed right-hand side. Solving that Gram plus the ridge penalty
  (the penalised system) gives the extracted coefficients and the covariance, as today.
  With `compute_covariance=False` there is no extraction and no pass R.
- **Final pass A**, at the extracted coefficients (projected when constrained), with the
  retained weights. It computes the deviance for `final_extraction_deviance_acceptable`,
  and every statistic in §8 marked A.
- **Final pass A'.** Pass A repeated at the retained result predictor, only when the
  check fails.
- **Final pass B.** Gamma only: the log-likelihood with φ from pass A (§8).

### 7.5 Lasso and elastic net (coordinate descent)

The driver reproduces `coordinate_descent.rs:81-374`:

- **Start.** On a cold start, β₀ is the intercept g(mean y) and the predictor is
  `Initializer` (`:137-151`). On a warm start it is `CoefRoundTrip(β_warm)`.
- **Each outer iteration** is one fused pass. It returns the deviance at the current
  predictor and the Gram at its μ, which is the next iteration's normal equations. The
  sweeps then run on the p×p Gram with their existing caps (1,000 sweeps, 1e-7, a full
  sweep every 5th, the active-set refresh). The duplicate row-major Gram (`:211-216`) is
  removed.
- **Covariance.** The covariance (`:551-593`) uses the last iteration's weights
  provenance, through pass R. The active-set inverse is unchanged.

### 7.6 Penalised smooth terms (non-monotone, and fixed penalties)

The driver reproduces `smooth_glm.rs:2077-2246`:

- the initialisation, which costs 2 passes (`:1447-1485`);
- one candidate pass per iteration with a speculative Gram and z'Wz;
- GCV λ updates on the Gram, with the same schedule (`:480-482`);
- halving fractions 0.5¹…0.5¹⁰ and then 0, through §7.3;
- convergence with `max(tol, SMOOTH_CONVERGENCE_TOL)`.

A fixed penalty runs the same driver with λ fixed, at no extra pass.

**Exit steps:**

- **Intercept column.** It is located structurally from the frozen design, plus the AND
  over rows that confirms every value is exactly 1.0 (`:564-579`).
- **Intercept refresh** (`:889-944`). It is replayed exactly, by evaluating score trees:
  - Pass A gives s₀, the 14 bracket ends (spans 1, 2, … 64) and the KKT sums.
  - Each tree pass evaluates the next m = 8 bisection levels (255 shifts per row, each
    midpoint computed as `0.5*(lo+hi)` along its path). The driver replays `signum`, the
    `== 0.0` exit, the non-finite exit, the tolerance and the 200-step cap. That takes
    about 6 tree passes.
  - Pass C computes the deviance at the chosen shift and the KKT sums.

  The result predictor is `Shifted(P, d)`, where P is the predictor before the refresh
  (`Coef`, or `Partitioned` after refinement).
- **KKT stationarity.** Its sums are reduced in passes A and C (`:596-690`).
- **Result assembly** (`:1171-1255`). EDF, `total_edf`, `gcv` and the covariance come
  from the Gram under `final_weights`' provenance, through pass R. The Gram is the
  `compute_xtwx` form (`:1197`); the shared kernel computes the same sums. Today
  `final_weights` holds combined prior×IRLS weights (`:2100`). That convention is kept,
  including its effect on robust covariances (§8, D9).

D1 is fixed here: a covariance that fails to invert raises instead of returning the
identity. `SmoothGLMResult` no longer carries X.

### 7.7 Monotone smooth terms

The n×p `x_tilde` copy is removed. The driver keeps today's row-wise arithmetic and applies
it per block:

- For each block, the encoded rows are copied once into a block-local buffer.
- The monotone column ranges are overwritten with today's transform:
  `target[i,j] = scale_j · Σ_{l≥j} x[i,l]`, with scale from J(α) (`smooth_glm.rs:1088-1107`).
- The adjusted working response is formed row-wise as today (`:1763-1776`).
- The Gram is accumulated on the transformed rows.
- Candidates' η uses the untransformed X (`:1881`).

The block-local buffer adds E·8p bytes to the plan.

The stages:

- **Warm start** (`:1608-1686`, cap min(25, max_iter), raw `tol`) costs one fused pass per
  iteration, plus one after the α reset.
- **Phase-1 REML** (`:1564-1601`) uses the *untransformed* sums: the Gram on X, z'Wz from
  the unadjusted working response, and the original penalties S (`:1580`). These come
  from the initialisation's second pass, at no extra pass.
- **Outer REML** (`:1955-2067`) uses the transformed Gram, z̃'Wz̃ and S̃ = J'SJ (`:2018`)
  from the last pass at the current μ and α, at no extra pass.
- **Inner PIRLS** (`:1703-1943`) uses §7.3. Its 21 trials (t = 0..20; t = 20 keeps the
  previous iterate) are evaluated in one trials pass.
- **`refine_unconstrained_block`** (`:692-860`). It solves the refit columns R with their
  existing penalty `S[R,R]` (`:753-770`, `:810`). The fixed columns enter as offsets per
  row, o_F = o + X_F·β_F (`:745-751`), computed per block with the fixed columns taken
  in ascending column order. The refined state is recorded as `Partitioned` (§7.2), so
  its grouped arithmetic is reproduced exactly. Today they come from a randomly seeded `HashSet` (`:712`),
  which is another run-to-run nondeterminism (D11). There is one candidate pass per
  iteration (§7.3), and its n×|R| `x_sub` copy is removed.

### 7.8 Regularisation path and cross-validation (without target encoding)

The driver reproduces `regularization_path.py:823-1033` and `fit_cv_path_py`
(`fitting_py.rs:1471-1897`), in this order:

1. **Moments.** Passes M1 and M2 over the full design.
2. **Null-intercept fit.** A closed form for Poisson-log and Gaussian-identity uses freeze
   sums. Otherwise a width-1 IRLS runs `null_fit` passes over (y, offset, w), with its
   caps (200, 1e-12).
3. **α_max score.** One pass computes X'f and Σf (`:462-632`). The ridge case needs no pass.
4. **Grid.** `n_alphas`, `alpha_min_ratio` and `include_unregularized` as configured
   (`:635-658`, `:925-929`).
5. **Folds in lockstep.** Each fold k runs its own driver over the grid, with warm starts,
   `skip_covariance` and today's failure handling. A failed alpha scores INF and leaves
   that fold's warm start unchanged (`fitting_py.rs:1790-1799`). Folds are independent
   today, so lockstep preserves their semantics.

   **Coordination.**
   - **Threads.** The K fold drivers run on K dedicated scoped threads
     (`std::thread::scope`), never on the Rayon pool. A driver blocked on a pass
     therefore cannot starve the block reducer, and K may exceed the thread count.
   - **Fold executor.** Each fold's driver holds a fold executor whose `run(request)`
     posts its whole `PassRequest` to a shared rendezvous (a mutex and condition
     variable) and waits. The request may hold any number of evaluations, each with
     its own fold filter (training rows or validation rows). Examples are §7.3's
     batched halving trials, and a validation evaluation sharing a pass with the next
     alpha's warm start.
   - **Coordinator.** The calling thread waits until every still-active fold has
     either posted a request or finished. It then merges the posted requests, tagging
     each evaluation with its fold filter, and runs one pass on the real executor. In
     out-of-core mode that is the only thread that touches Python and the GIL. It then
     hands each fold its own sums and wakes the folds.
   - **Divergence.** Folds at different alphas, iterations or halving stages simply
     post different requests. A finished fold leaves the rendezvous.
   - **Memory.**
     - The merged pass holds every active fold's partials at once:
       Σ_k (T + 2) × (fold k's request size).
     - After a pass, folds expand and factor their p×p systems through a shared pool
       of `solve_workspaces` workspaces (default 1, so solves are serialized). Each is
       2·8p² (the expanded matrix and the factor). A fold waits for a free workspace,
       so at most `solve_workspaces` systems are factored at once.
     - Fold threads are spawned with an explicit 2 MiB stack.
     - The plan charges all three: the merged partials, `solve_workspaces` × 2·8p²,
       and K × 2 MiB.
   - Each pass serves one `PassRequest` per active fold. A row in fold g feeds every
     fold k ≠ g's training evaluations, plus fold g's validation evaluations.
   - Validating (k, a) shares a pass with fold k's warm start of alpha a + 1.
   - Finished folds stop requesting.
   - Validation keeps the per-family kernels (`fitting_py.rs:823-1172`), moved to the
     shared kernels.
   - The K fold design copies are removed in both modes.
6. **Selection.** `min` and `1se` are unchanged.
7. **Final refit.** At the selected α, with the CV path's standardisation (§7.4 or §7.5).

Fold assignment today uses Rust's `DefaultHasher` (`fitting_py.rs:1568-1578`), which is
not stable across Rust releases. DEC-2 replaces it with fold(r) =
`mix64(cv_seed ^ mix64(r))` mod K.

### 7.9 Cross-validation with target encoding

Today this path (`regularization_path.py:1057-1283`) works fold by fold:

- folds come from an n-length NumPy permutation;
- each fold gets a design learned on its training rows;
- α_max is the maximum over folds, which is a barrier;
- folds then run **sequentially**, and an alpha that failed in an earlier fold is dropped
  from later folds' grids, changing their warm-start chains (`:1241-1283`,
  `fitting_py.rs:1370`).

The new path keeps that dependency:

- **Fold assignment** uses DEC-2.
- **Learning** follows §6.2 for all K fold designs: one freeze pass for every fold's
  freeze state, then knots, te_stats, the TE sort stages and moments per fold.
- **α_max** comes from per-fold α_max passes, and the maximum is taken.
- **Folds run sequentially**, in fold order, each through the §7.8 per-fold driver. The
  next fold's grid is built after the previous fold finishes, exactly as today.
- **Encoding.** A pass encodes each row under the running fold's design: training rows
  with that fold's spilled TE values, validation rows with its full statistics.
- **Final refit** at the selected α uses the full-data design. Its standardisation is
  recomputed, as today (`formula.py:941-961`).

### 7.10 Negative Binomial θ estimation

The driver reproduces `fit_negbinomial_py` (`fitting_py.rs:569-645`):

1. **Poisson fit.** A cold start.
2. **Moment θ₀.** Unweighted, over μ_P and y, with today's clamps. Its sums ride on the
   Poisson fit's final pass.
3. **Rounds.** Up to 25 rounds. Each round:
   - an NB fit with covariance. **The first round is a cold start**, because the
     coefficient state is reset after the Poisson fit (`fitting_py.rs:600`). Later rounds
     warm-start from the previous round;
   - its final pass also writes μ to the spill;
   - a Brent search on [0.01, 100] with tol 1e-6. That is one initial evaluation plus at
     most 100 more. Each evaluation is a light `theta` pass that reads y and w from the
     source and μ from the spill. The initial evaluation, at a constant θ, rides on the NB
     fit's final pass.
   - Rounds stop when |Δθ| < 1e-5.
4. **Final NB fit**, warm-started.

`solver_settings` records the θ tolerance, the round cap and the rounds used. In memory,
μ stays in RAM and the same sums are reduced.

## 8. Final statistics

| Statistic | Predictor | Weights | Pass | Notes |
|---|---|---|---|---|
| covariance | — | retained weights provenance, plus the penalty | R | ridge: penalised system (`irls.rs:2835`); smooth: (G + Σλ·S)⁻¹ |
| extracted β | — | retained weights, mixed RHS at the retained predictor | R | IRLS only |
| deviance | final | pw | A | — |
| Pearson χ² | final | `maybe_weights` | A | — |
| scale | — | — | none | 1 for fixed-dispersion families; 1 when df_resid = 0 (`dispersion.rs:108`); else χ²/df_resid, with Rust's df_resid = n − p in every route, including smooth fits (`results_py.rs:285`, `336`) |
| df_resid reported | — | — | none | Python reports n − EDF for smooth fits (`formula.py:2250`), as today |
| llf (Poisson, Binomial, NB, quasi, deviance fallback) | final | `maybe_weights` | A | — |
| llf Gaussian | final | — | A | from the weighted SSE and Σw (`model_fit.rs:56`) |
| llf Gamma | final | — | B | per row with α = 1/φ (`model_fit.rs:185`) |
| aic, bic | — | — | none | from llf and p or EDF |
| null deviance | — | `maybe_weights` | start or first candidate | sums from freeze; family default-link dispatch kept (`model_fit.rs:419-480`) |
| robust HC0–HC3 | Pearson residual at final | the result's stored `irls_weights` times prior weights, once (`inference/mod.rs:283`). Ordinary routes store `Raw(...)`; smooth routes store `Solve(...)`, so prior weights apply twice there (D9), reproduced | A | per-row HC1 scaling, leverage clamp and floors kept (`inference/mod.rs:345-440`); bread is the covariance from R |
| KKT, smooth EDF, gcv | as in §7.6 | as in §7.6 | A, C, R | — |
| response_scale | — | — | freeze | — |

With `compute_covariance=False` there is no pass R. Pass A still runs at the retained
predictor. `robust=` is refused in that combination (§4.3). Fits internal to
cross-validation run neither R nor A.

## 9. Source fingerprint

**Baseline.** The freeze pass computes, for every model input column, a streaming 128-bit
XXH3 hash (from the `xxhash-rust` crate) of a canonical byte encoding of its values in row
order. The encoding is available immediately and does not depend on batching:

The hash covers each column as the source holds it, before encoding, so it sees every
change the encoder could see. Each value starts with a kind byte and a validity byte; a
null contributes nothing more. A valid value then contributes its 8-byte pattern or its
length-prefixed bytes, by dtype (`fingerprint::kind`):

| Dtype | Kind | Value bytes |
|---|---|---|
| Floats | 1 | the float64 bit pattern |
| Integers | 2 | the 64-bit integer, so integers above 2⁵³ stay distinct |
| Booleans | 3 | the 64-bit integer |
| Temporals | 4 | the physical 64-bit integer |
| Strings, categoricals, enums | 5 | the UTF-8 label: an 8-byte length, then the bytes |
| Binary | 6 | an 8-byte length, then the value's own bytes |
| Others (decimals, 128-bit integers) | 7 | the text: an 8-byte length, then the bytes |

The bytes go to the hash directly, never through Polars' 64-bit hashes, so keys that
collide there stay distinct. The response, weights, exposure and offset are hashed the
same way.

**Scratch.** The hash keeps one fixed 8 KiB buffer.
- `fingerprint_update` converts its slice 4,096 rows at a time.
- Fixed-width values take at most 25 bytes per row.
- Byte values are passed as each value's own bytes. Nothing is hex-encoded or joined.
- So a call holds at most 2 × the value bytes of 4,096 rows, plus 256 KiB. String and
  binary values cost one copy, about 1×. Categoricals, enums and other scalars cost at most
  2×, because their labels are formed first.
- The admission of an oversized row (§4.3) charges this.

**Arrays.** An exposure, offset, weights or complement given as an array is read for each
slice by global row and hashed as a float64 column under its own name, in the freeze pass
and in every later pass, so a caller's array changed between passes is caught like a
changed column.

**Checks.** Every later pass hashes the columns it reads, in the same encoding, and
compares the hashes at the end of the pass. A mismatch raises `BatchSourceChangedError`
naming the pass and the column, and no result is produced. A changed value, order, row
count or dtype is detected, except with probability about 2⁻¹²⁸ per column per pass. The
error does not localise the first differing row; one hash per column cannot.

## 10. Spill storage

Spills hold target-encoding values (§6.6, §7.9) and the Negative Binomial μ column
(§7.10). Every file is created exclusively by RustyStats under `scratch_dir`, written and
read sequentially with buffered I/O (never `mmap`, so resident memory stays within the
plan), and removed on every exit path.

| File | Layout | Lifetime |
|---|---|---|
| TE key run | 40-byte records (key, row, code, target, exposure), little-endian, sorted by key | from te_sort until the merge stage or walk that consumes it |
| TE row run | (row u64, value f64) records, sorted by row | from te_walk until the merge stage that consumes it |
| TE permutation column | row-aligned float64 | until te_average consumes it |
| TE term column | row-aligned float64, one per TE term (and per fold) | until the fit (or the fold) ends |
| NB μ | row-aligned float64 | one round |

**Schedule.** TE terms are processed one at a time, permutations one at a time within a
term, and folds one at a time. Every stage deletes its input files as soon as it has
consumed them.

**Disk peak.** With n_t the rows taking part (n, or the fold's training rows), T TE terms
and M permutations, the files alive together are at most:

- the earlier terms' columns, 8·n_t·(T − 1);
- this term's finished permutation columns, 8·n_t·(M − 1);
- the current stage's input and output: at most 80·n_t, during a key merge stage. The walk
  holds 40·n_t + 16·n_t, a row merge 32·n_t, and the final row stage 24·n_t.

So for one design `spill_bytes` = 8·n_t·(T + M − 2) + 80·n_t, plus 8·n for NB μ when θ is
estimated.

**Cross-validation with target encoding.** Every fold's design, including its TE term
columns, is learned before the α_max barrier and kept until that fold's path ends
(§7.9). The peak is therefore reached while fold k is learned, with every earlier fold's
term columns alive:

max over k of [ Σ_{j<k} 8·T·n_j + 8·n_k·(T + M − 2) + 80·n_k ],

compared with the final refit's single-design figure. The larger of the two is
`spill_bytes`. A test (T22) checks the observed peak directory size against it, with
K = 10 folds.

## 11. Memory plan (staged admission)

Notation:

- p: design columns;
- T: accumulation threads;
- E: `encode_rows`;
- K: folds;
- H: the number of requested robust types;
- P = p(p+1)/2·8 + 8p + 16 bytes: one packed Gram, its right-hand side and scalars;
- L = `learn_slice_bytes` (§4.3).

Before each phase, RustyStats computes that phase's peak from what is known by then. It
raises `MemoryPlanExceededError` if `persistent_bytes + phase_peak > max_bytes`.
`MemoryPlan` in the result reports every phase's peak, `persistent_bytes`, `spill_bytes`,
`encode_rows` and T. T is `threads`, or the global Rayon pool's size, and the fit's Rust
work runs on a pool of exactly T threads.

The phases are freeze, knots, te_stats, te_sort and te_merge (te_walk and te_average
included), moments, cv, solve, theta and final (the labels pass of §4.6 is admitted the
same way). The learning phases are admitted as they start; every solver phase is admitted
once the design is frozen, before the first solver pass.

| Allocation | Bytes | Lifetime | Known before |
|---|---|---|---|
| learned state (key and label tables, counts, TE totals, knots, centres and scales, fold designs) | counted while it grows (§6.4), capped by `max_learned_state_bytes` | persistent | — (capped) |
| fingerprints | 64 per input column | persistent | validate |
| learning slice working set | 3·L (copies of a slice's payload alive at once: a fold design's training rows, a column's non-null values and their distinct keys; the fingerprints' copies are freed first), plus E_learn × 96 bytes of per-row scratch; the slice's new keys are charged as learned state before they are inserted (§6.4) | freeze, labels, knots, te_stats, te_sort | validate |
| knot histograms | 512 KiB × the ranks of every design (one per selector in pass 1, at most one per rank afterwards) | knots | freeze |
| TE records and sort scratch | R_run × 80 | te_sort, te_walk | freeze |
| TE merge buffers | (F + 1) × 1 MiB | te_merge, te_walk | freeze |
| encode slice and its finiteness mask | E·9p | solver phases, moments | freeze |
| encode-slice payload copies | 2·L when the source columns include String, Binary, Categorical or Enum (§4.3) | solver phases, moments | validate |
| monotone block-local transformed rows | E·8p (monotone smooth only) | solver phases | freeze |
| per-row work vectors (y, w, offset, η per predictor state, μ, W, z) | E·8·(6 + predictor states) | solver phases | freeze |
| encoder scratch | E × the frozen design's declared per-row bytes | solver phases, moments | freeze |
| per-block CSR (sparse dispatch) | when `sparse_rows_py(n, p, max_row_nonzeros)` says sparse rows are served: per thread, T at once, GRAM_BLOCK_ROWS·12·`max_row_nonzeros` + 8·(GRAM_BLOCK_ROWS+1) (`BlockCsr` sizes its vectors exactly), plus the packed Gram's row offsets, 8·(p+1) | solver phases except moments | freeze |
| Gram accumulators | (T + 2)·P per Gram output in flight (Σ over active folds' requests in lockstep CV, else 1), plus the mixed right-hand side | candidate, gram, R | freeze |
| lockstep CV solve workspaces and fold threads | `solve_workspaces` × 2·8p², plus K × 2 MiB of thread stack | CV | freeze |
| robust meats | H·(T + 2)·P | A | freeze |
| θ blocks | E × 40 (y, w and μ read back; η and μ written) | theta | freeze |
| covariance | 8p² | from R to the end of the fit (the result keeps it) | freeze |
| solve workspace | 8p² (expanded and factorised in place) | between passes | freeze |
| smooth workspace | penalty blocks planned from their sizes k (2·8k² each: the term's and the binding's copy, plus 3·8k² for the largest's construction), GCV/REML factorisations (2·8p²), and S̃ (8p²) on the monotone path; the blocks are built only after the solve phase is admitted | between passes | freeze |
| score trees | 255 × 8 per thread | refresh | — |
| driver state | O(p) per retained provenance; O(K·p) for CV | persistent | freeze |
| allocator allowance | a fixed 64 MiB | persistent | — |

- **Defaults.** The default `te_sort_bytes` equals the encode-slice budget. Under
  `max_bytes` it is reduced so the TE phase's peak fits.
- **Buffers.** All buffers are allocated once per phase and reused across passes.
- **Caller's share.** The caller adds its own source batch to `max_bytes` when it decides
  admission.
- **No dependence on n.** No row of the table depends on n. Spill run files are named by
  index, so their bookkeeping is constant too.

## 12. Pass budget

| Model | Passes (k iterations; h iterations with a rejected candidate) |
|---|---|
| IRLS | freeze 1 + knots (0 or 4) + moments (0, or 2 when standardised) + TE learning + start 1 + candidates (1 + k) + 2h + R (0 or 1) + A 1 + A' (0 or 1, when the extraction check fails) + B (0 or 1, Gamma) |
| CD | learning + moments 2 + 1 + T outer iterations + R 1 + A 1 |
| smooth | learning + init 2 + per iteration (1 or 3) + refresh about 8 + R 1 + A 1 |
| CV | learning + moments 2 + null fit + α_max 1 + max over folds of Σ_a (iterations + 1) + refit |
| CV with TE | learning for K folds + α_max per fold + Σ over folds of that fold's path + refit |
| NB θ | Poisson fit + Σ over rounds (NB fit + up to 100 θ passes) + final fit |
| TE | te_stats 1 + Σ over terms, permutations and folds of one te_sort source pass; the merge, walk and average stages read spill files only |

Under the default caps, without target encoding, the IRLS worst case is
1 + 4 + 2 + 1 + 26 + 50 + 1 + 1 + 1 + 1 = 88 passes: a standardised Gamma fit whose
extraction check fails. T15 includes that combination. Every fit reports its actual
counts.

## 13. Implementation order

Every item ships. The order only manages risk.

1. **Reference capture.** Before any change, record v0.9.0 outputs as hex float64 bit
   patterns, in `tests/golden/fits_v0_9_0.json`. Done: 35 cases, captured at `28e7373`
   (§14).
2. **Arithmetic contract** (§5), in memory only: shared kernels, block reductions, waves,
   structural dispatch, and predict's row kernel.
3. **IRLS driver, provenance and final statistics** (§7.2–7.4, §8), plus `robust=`.
4. **Frozen design and encoder** (§6.1–6.5, §6.8); the in-memory build switches over;
   `observed_category_labels`.
5. **Out-of-core mode:** `BatchSource`, `OutOfCoreOptions`, staged admission, fingerprint,
   and out-of-core IRLS.
6. **Standardisation** (§6.7) and the coordinate-descent driver (§7.5).
7. **Penalised smooth driver** (§7.6) and D1.
8. **Monotone smooth with row transforms per block** (§7.7).
9. **Regularisation path and cross-validation in lockstep** (§7.8).
10. **Spill storage** (§10) and **Negative Binomial θ estimation** (§7.10).
11. **Target encoding** (§6.6) and **cross-validation with target encoding** (§7.9).
12. **Complement pipeline** (§6.8).
13. **Release:** benchmarks, documentation, changelog and version 0.10.0.

## 14. Testing

"Equal" means bitwise equal on float64 bit patterns. The golden record
(`tests/python/_golden_cases.py`, `scripts/capture_golden_fits.py`,
`tests/python/test_golden_fits.py`) is the v0.9.0 oracle for T2.

- **Cases.** It records 35 cases: every family; every term kind; every solver route;
  accepted, first-iteration and exhausted halving; separation; an ill-conditioned design;
  `compute_covariance=False`; HC0–HC3.
- **Tolerances.** Each floating-point field class has a relative tolerance r, applied as
  |a − e| ≤ r·|e| + r·max|e| over the field. The base values are:
  - 1e-7 for coefficients, standard errors, statistics derived from them, and
    predictions;
  - 1e-9 for deviance-type statistics;
  - 1e-6 for smoothing results;
  - 1e-7 for path statistics and θ.

  A case may loosen a class, with a comment saying why.
- **Discrete fields.** Statuses, names, availability and integer fields are compared
  exactly, with two documented exceptions:
  - the IRLS iteration count may differ by one, because a rounding-level change can
    cross the convergence threshold;
  - the degenerate `halving_exhausted` fixture compares only discrete fields and
    finiteness, because v0.9.0's own coefficients there move by up to 47% between thread
    counts.
- **Status.** The record passes against v0.9.0 at 1, 2, 4 and 8 threads. It is the oracle
  for in-memory regression (T2). Mode equivalence (T3) is a separate, strict bitwise
  oracle.
- **Decided changes.** Cases marked with a decision (DEC-1, DEC-2) are re-captured when
  that decision lands. The re-captured case is then compared numerically, as any other,
  except that its iteration counts must match exactly. The record's `recaptured` table
  gives each re-capture's date, build and reason (`capture_golden_fits.py --cases …
  --note …`). A test fails when a decided case has no re-capture.
  - On 2026-11-05, the six decided cases were re-captured from the out-of-core-fit working
    tree. That tree is uncommitted on 28e7373 (v0.9.0), so its build is recorded as
    `28e7373…-dirty`, with rustystats 0.9.0, NumPy 2.2.6 and Polars 1.36.1. They are
    `cv_lasso_min`, `cv_ridge_1se` and `cv_elastic_net` (DEC-2), and
    `target_encoding_main`, `target_encoding_interaction` and `target_encoding_cv`
    (DEC-1, and DEC-2 for CV).
  - The other cases keep the v0.9.0 record.
- **T1 digests.** `tests/golden/design_digests_v0_9_0.json` holds SHA-256 digests of X,
  y and the names from v0.9.0's own builder. They cover every `TERM_KINDS` case
  (`tests/python/_design_cases.py`) and every golden case without target encoding.
  - The digests were captured by `scripts/capture_design_digests.py` from a
    `git archive` of 28e7373 with its extension built.
  - The frozen build must reproduce them with and without slicing.

| ID | Scenario | Oracle |
|---|---|---|
| T1 | Encoder parity: every non-TE term and interaction kind; categoricals that are String, Boolean, Binary (discovered and with `levels=`), integer with and without null, float with NaN, ±0.0, or integers above 2⁵³ with nulls; `levels=`; every spline kind with and without given knots | the v0.9.0 builder's X, then pinned SHA-256 digests |
| T1-TE | Target-encoding training values: one and several permutations; with exposure; TE interactions | a pure-NumPy reference implementation of §6.6 (sort by key, walk, average), for n ≤ 10,000 |
| T2 | In-memory regression: every existing numerical test, with no tolerance loosened, and the golden record | existing tests, statsmodels, `test_golden_fits.py` |
| T3 | Mode equivalence: gaussian, poisson, gamma, tweedie, binomial, negbinomial; weights, exposure and offset. Out of core with batch sizes 1, 1,000, B−1, B, B+1 and n; `encode_rows` B and 4B; 1 and 4 threads | the in-memory fit: every §8 statistic, iterations, status and `solver_trace` equal |
| T4 | Ill-conditioned design (condition number about 1e10) | T3 |
| T5 | Branches: accepted halving; exhausted halving; constrained best restored; final extraction rejected; `compute_covariance=False`; HC0–HC3 together; coordinate descent with `max_iter = 1` cold and warm | T3, plus `solver_trace` showing each branch |
| T6 | Separation (binomial) | equal status and iteration count; passes within §12 |
| T7 | Thread independence: in memory with 1 vs 8 threads; out of core with 1 vs 4 | equal |
| T8 | Knots: n = 1..5; all values equal; two values; duplicates at a rank; ±0.0; subnormals; ±1e300; negatives only; interaction-local splines | the sorted-index knot |
| T9 | Source change: a different value, order, row count or dtype in a later pass; an unseen key | `BatchSourceChangedError` naming the pass and column, or `UnseenLabelError`; no result |
| T10 | Abort: an exception raised by `on_batch` in each pass kind, including freeze, labels and te_merge; a sentinel `ValueError` subclass raised by `on_batch` mid-path in cross-validation and in cross-validation with target encoding | that exception, unchanged (never recorded as a failed alpha); `close()` called; no later batch requested; spill directory empty |
| T11 | Option refusals before I/O (including `robust` with `compute_covariance=False`, and DEC-5's nested dtypes); `max_design_columns` and `max_learned_state_bytes`, including high-cardinality frequency encoding, one oversized string row, a batch with one 16 KiB string among one-byte strings, a 1 MiB ASCII label, a 1 MiB ASCII label ending in a non-BMP character, and a 1 MiB Binary value of `0xff` bytes; `max_bytes` refusing each phase | the named error; `open` call count as specified; peak RSS within the reservation for the long-label cases |
| T12 | Result surface: each refused method raises `OutOfCoreResultError`; `to_bytes` round-trip | predictions equal to the in-memory model's |
| T13 | Memory bound, in a fresh-subprocess `VmHWM`: (a) p ≈ 150 (30 linear, 5 categorical with 4–64 levels, 2 splines) at n and 4n (n ≥ 2M); (b) the same plus one TE term with 4 permutations; (c) CV with TE, K = 3 | peak(4n) − peak(n) ≤ max(16 MiB, 2% of peak(n)); peak − baseline ≤ the plan's maximum phase peak + persistent + source batch |
| T14 | In-memory build peak at n = 2M with the T13(a) design, construction isolated from fitting, at 1 and 8 threads | rise ≤ 1.1 × 8np; the v0.9.0 figure recorded beside it |
| T15 | Coordinate descent (lasso and elastic net, with and without intercept, sign constraints); ridge on both routes | T3 |
| T16 | Smooth GCV, fixed-penalty smooth, a fixture that triggers the intercept refresh, a stalled fit reporting `stalled_nonstationary` | T3, plus the refresh steps in `solver_trace` |
| T17 | Monotone smooth (main term and smooth × categorical), with and without an ordinary penalised smooth beside it (so refine carries `S[R,R]`), including refinement followed by the intercept refresh | T3, including the `Partitioned` and `Shifted(Partitioned)` reconstruction |
| T18 | Cross-validation (min and 1se; ridge, lasso and elastic net; a failed alpha; K = 10 folds on 2 threads with folds converging at different iterations and one fold step-halving) and cross-validation with target encoding with a failed alpha in fold 1; lockstep CV at p = 1,000 under a `max_bytes` that admits exactly the plan | T3, including the CV path table and fold grids; peak RSS within the plan |
| T19 | NB `theta="estimate"` | T3, including θ and the round count |
| T20 | Target encoding with `te_run_records` forcing 1, 2 and 130 runs (130 forces an intermediate merge stage at F = 64) | `solver_trace` and every statistic equal across run counts; the spilled column equal to T1-TE's reference |
| T21 | Complement: a column with and without exposure (the column is not divided by exposure); a `GLMModel` with a different link, with exposure division | T3; and today's combined offset for the column-plus-exposure case |
| T22 | Spill: removed on success, on error and on an `on_batch` abort in te_sort, te_merge, te_walk and te_average; `SpillError` on an unwritable directory; the observed peak directory size never above `spill_bytes`, with 2 TE terms and 4 permutations, alone and in cross-validation with target encoding at K = 10 | directory listing and size sampling |
| T23 | `observed_category_labels`: equals Haute's helper on each dtype class, including Binary; bounded and cancellable on a `BatchSource` | Haute's `_observed_category_labels` |

T13, T14 and T18's p = 1,000 run are in `benchmarks/bench_out_of_core_memory.py`, with a
reduced smoke variant in CI; the CI suite runs T18 at p ≈ 150
(`tests/python/test_out_of_core_contract.py`).

## 15. Consumer checklist (Haute)

- **What Haute reads** (`_rustystats.py:365-558`): `inference_status`, `bse`, `tvalues`,
  `pvalues` and `conf_int` (and the robust versions for the configured type), `params`,
  `feature_names`, `link`, `deviance`, `null_deviance()`, `nobs`, `df_model`, `df_resid`,
  `iterations`, `converged`, `scale()`, `is_quasi_likelihood`, `llf()`, `aic()`, `bic()`,
  `has_smooth_terms()`, `total_edf`, `gcv`, `smooth_terms` (`variable`, `k`, `edf`,
  `lambda_`), `alpha`, `l1_ratio`, `n_nonzero()`, `n_cv_folds`, `cv_selection_method`,
  `predict` and `to_bytes`. All are served in both modes.
- **Adapter changes:**
  - `glm_fit_kwargs` passes `robust=[type]` instead of `store_design_matrix=True`.
  - `prepare_glm_design` takes labels from `rs.observed_category_labels` and dtypes from
    the schema.
  - `GLMAlgorithm.fit` accepts a `BatchSource` and `OutOfCoreOptions`.
- **Haute's own side:**
  - deciding engagement from its estimate;
  - building the batch reader and schema over the sunk training parquet;
  - deriving the options from its budget;
  - connecting `on_batch` to `ExecutionContext.checkpoint`;
  - a scratch directory under the job's artifact directory;
  - stamping `fit_mode`, `solver_settings` and `passes` on the result, model card and
    MLflow run;
  - retiring the bounded-streaming refusal for GLM.

  These are specified in Haute's modelling low-level specification.

## 16. Defects found while specifying

| ID | Defect | Handling |
|---|---|---|
| D1 | Smooth covariance falls back to the identity matrix when inversion fails (`smooth_glm.rs:1227`) | fixed here (§7.6) |
| D2 | Families without their own `log_likelihood` report −½ × the unweighted deviance (`families/mod.rs:197-206`); Tweedie is affected, and so is any profile likelihood over its power | separate fix; both modes reproduce today's value |
| D3 | `seed=None` target encoding is not reproducible (`target_encoding/mod.rs:265-271`) | the seed used is recorded (§6.6); DEC-3 |
| D4 | Categorical labels at predict use a Polars cast, not training's `str()` (`interactions.py:2750` vs `476`) | separate change |
| D5 | Reductions and dispatch depend on thread count and on run-to-run scheduling (golden record, §2.2) | fixed (§5) |
| D6 | Design-build copies and caches | fixed (§6.8) |
| D7 | `solvers/nnls.rs` is unused | remove separately |
| D8 | Cross-validation folds use `DefaultHasher` (`fitting_py.rs:1568`), which is not stable across Rust releases | fixed by DEC-2 |
| D9 | Smooth fits export combined prior×IRLS weights as `irls_weights`, and robust covariance multiplies by prior weights again (`smooth_glm.rs:2100`, `:1244`; `inference/mod.rs:283`), so weighted smooth fits get prior weights twice in their robust covariance | reproduced in both modes; separate fix |
| D10 | Overflowing inputs give a negative Poisson deviance (golden case `halving_exhausted`: −2.0e284) | separate fix |
| D11 | `refine_unconstrained_block` takes its fixed columns from a randomly seeded `HashSet` (`smooth_glm.rs:712`, `:745`), so their summation order changes from run to run | fixed: ascending column order (§7.7) |
| D12 | A result rebuilds its family from `family_name`, which formats θ and the Tweedie power to four decimals (`fitting_py.rs:318`), so scale, llf, null deviance, standard errors and residuals used a rounded θ (an estimated θ of 1.39481… became 1.3948) | fixed: results keep the exact parameters (`exact_theta`, `exact_var_power`); the golden case `negbinomial_theta_estimate` was re-captured for it |
| D13 | A spline's boundary knots are NumPy's `min`/`max` of the column (`splines.py:472`, `:553`), whose choice between −0.0 and +0.0 for a zero extreme depends on which SIMD lanes the zeros fall in | fixed: a zero minimum is −0.0 when the column holds −0.0, and a zero maximum is +0.0 when it holds +0.0, whatever the slicing (`frozen_design.py`, `_ColumnRange`) |
| D14 | A continuous interaction with both a spline factor and a target-encoded factor drops the target-encoded factor at fit (`interactions.py` `_build_continuous_interaction`: the spline branch ignores `te_factors`), but prediction multiplies it in (`_build_continuous_interaction_new`), so predictions use a different design from the fit | fit reproduced in both modes (`frozen_design.py` `_continuous_interaction`); separate fix |
| D15 | A continuous interaction with two or more target-encoded factors writes one column per factor at fit (`x2:TE(a)`, `x2:TE(b)`), but prediction multiplies them into one column, so `predict` fails (`x has 5 columns but beta has length 6`) | fit reproduced in both modes; separate fix |
| D16 | `reset_fold_local_spline_state` (`regularization_path.py`) clears a main-effect spline's user-given interior knots (`knots=`), and leaves interaction-local splines with the full data's knots, so v0.9.0's fold designs ignored given knots and placed interaction-local knots from validation rows | fixed for GLM CV with target encoding: each fold's design is learned from a fresh parse of the model's terms (§7.9); the multinomial fold builder still uses the helper; separate fix |

## 17. Decisions for the owner

- **DEC-1 (recommended):** target-encoding training values use the keyed order of §6.6 in
  both modes. TE values change against v0.9.0 (a different random order, with the same
  statistical construction), and RAM is bounded. Two alternatives keep the v0.9.0
  permutation:
  - hold its index vector (4 to 8 bytes per row) in RAM during the TE passes;
  - keep the permutation on disk with bounded buffering. That keeps RAM bounded but costs
    about n random disk accesses per permutation per TE term.
- **DEC-2 (recommended):** one specified fold hash for both cross-validation paths. Fold
  membership changes against v0.9.0. Folds with target encoding become approximately
  equal in size instead of exactly equal blocks.
- **DEC-3:** keep `seed=None` drawing a seed per fit (recorded), or default to a fixed
  seed. The recommendation is to keep it and have Haute pass an explicit seed.
- **DEC-4:** `GRAM_BLOCK_ROWS = 4096`. It is fixed once released, because it is part of
  every fit's numerical identity. A benchmark at p ∈ {20, 150, 1,000} confirms it first.
- **DEC-6 (recommended):** remove the preconditioned-conjugate-gradient ridge solver
  (`solve_weighted_least_squares_pcg`, enabled only by the undocumented
  `RUSTYSTATS_RIDGE_CV_PCG` environment variable).
  - **What it affects.** Every covariance-free ridge solve: CV ridge fits, but also an
    ordinary ridge fit's start projection and iterations (`irls.rs:857`, `:1001` at
    v0.9.0).
  - **A choice, not a necessity.** PCG could run on the accumulated Gram, or as a
    streamed matrix-vector product that both modes share, and either would keep the
    two modes equal. Removing it is a simplification: one solver (Cholesky on the
    Gram), at the cost of always forming the Gram, instead of one source pass per
    conjugate-gradient iteration out of core.
  - **Compatibility.** Fits with the variable set change to the Cholesky results.
    T2 pins that, with the variable set, ridge and CV ridge fits equal the same fits
    without it.
- **DEC-5 (recommended):** refuse List, Array, Struct, Object and Null columns as
  categorical, frequency-encoded and target-encoded inputs, in both modes. This is an
  API restriction, not a technical necessity. A typed, length-delimited recursive key
  could preserve such values, and Null needs only a singleton key. But nested columns as
  categories are accepted today only by accident, labelled with NumPy's string form of
  each element array, and Haute's GLM already restricts these terms to scalar dtypes.
  T11 covers the refusal for each dtype.
