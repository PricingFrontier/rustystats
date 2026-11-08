s", "monotonicity": "decreasing"},
        "PreviousRenewal": {"type": "categorical"},
        "include_main": True,
    },
]
```

For the reference category (`PreviousRenewal=0`), the fitted price curve is
non-increasing. For `PreviousRenewal=1`, the model uses:

```text
main InvitedPremiumDiff smooth + PreviousRenewal[T.1]:InvitedPremiumDiff smooth
```

The main smooth is constrained, but the interaction block is not registered as a
monotone smooth block. The combined curve can therefore increase.

## Self-Contained Repro

This should reproduce on rustystats 0.8.15 without any private data.

Install:

```bash
pip install "rustystats==0.8.15" "polars" "numpy"
```

Save as `repro_monotone_interaction.py`:

```python
import numpy as np
import polars as pl
import rustystats as rs

rng = np.random.default_rng(7)
n = 4000

x = rng.uniform(0.0, 1.0, n)
g = rng.integers(0, 2, n)

# The baseline effect is decreasing in x.
# Group 1 has an increasing local bump. A correct monotone interaction
# implementation should not be able to learn an increasing group-specific curve.
eta = 2.0 - 4.0 * x + (g == 1) * (3.5 * np.exp(-((x - 0.75) / 0.10) ** 2))
y = eta + rng.normal(0.0, 0.15, n)

df = pl.DataFrame(
    {
        "y": y,
        "x": x,
        "g": g.astype(str),
    }
)

interactions = [
    {
        "x": {"type": "bs", "monotonicity": "decreasing"},
        "g": {"type": "categorical"},
        "include_main": True,
    },
]

model = rs.glm_dict(
    response="y",
    terms={},
    interactions=interactions,
    data=df,
    family="gaussian",
    link="identity",
    seed=42,
).fit()

print("converged", getattr(model, "converged", None))
print("stationary", getattr(model, "stationary", None))

print("\nCoefficient block containing x")
for i, name in enumerate(model.feature_names):
    if "x" in name:
        print(i, name, float(model.params[i]))


def check_group(group, lo=0.01, hi=0.99, n_grid=201):
    grid = pl.DataFrame(
        {
            "x": np.linspace(lo, hi, n_grid),
            "g": [str(group)] * n_grid,
        }
    )
    pred = np.asarray(model.predict(grid), dtype=float)
    diff = np.diff(pred)
    return {
        "group": group,
        "max_increase": float(diff.max()),
        "upward_steps": int((diff > 1e-8).sum()),
        "min_prediction": float(pred.min()),
        "max_prediction": float(pred.max()),
    }


print("\nInterior monotonicity checks, x in [0.01, 0.99]")
for group in (0, 1):
    print(check_group(group))

print("\nExpected invariant after fix:")
print("  max_increase <= 0 and upward_steps == 0 for every group")
```

Observed on rustystats 0.8.15:

```text
converged True
stationary True

Interior monotonicity checks, x in [0.01, 0.99]
{'group': 0, 'max_increase': negative, 'upward_steps': 0, ...}
{'group': 1, 'max_increase': about 0.10, 'upward_steps': about 44, ...}
```

The coefficient block also shows the failure mode. The main `bs(x, ..., k, -)`
coefficients are monotone decreasing, but the interaction coefficients
`g[T.1]:bs(x, ..., k, -)` are unconstrained and non-monotone.

## Production Evidence

The renewal annual model has this interaction:

```python
interactions = [
    {
        "InvitedPremiumDiff": {"type": "bs", "monotonicity": "decreasing"},
        "PreviousRenewal": {"type": "categorical"},
        "include_main": True,
    },
]
```

The model fit was solver-clean:

```text
converged=True
stationary=True
solver_status=converged
train mean gap ~= +0.00074
```

But a function sweep of `InvitedPremiumDiff` over the training rows produced:

```text
PreviousRenewal=0:
  rows_violating = 0 / 2931

PreviousRenewal=1:
  rows_violating = 3448 / 3448
  max increase ~= 0.00238 per grid step
```

So the main constrained smooth is not enough. The category-specific combined
curve is allowed to rise.

In the renewal optimiser grid this appeared as annual-only guard clipping:

```text
Applied renewal monotone grid guard:
  9878 raw model grid points clipped across 604 policies
  immediate raw upward steps before cumulative-min guard: 6657
  max immediate increase ~= 0.03095
  max increase above cumulative guard ~= 0.23075
  split by payment type: annual only
```

The largest apparent impact was a trough-and-recovery shape. A low value at one
commission rate was carried forward by the cumulative-min score-time guard, then
later raw predictions recovered above it.

## Where The Bug Is

Relevant rustystats paths:

```text
python/rustystats/formula.py
python/rustystats/interactions.py
crates/rustystats/src/fitting_py.rs
crates/rustystats-core/src/solvers/smooth_glm.rs
```

Specific flow:

1. `formula.py::_parse_interaction_spec`
   - Creates an `InteractionTerm` for the categorical spline interaction.
   - If `include_main=True`, it separately adds the main effect via
     `_parse_term_spec`.

2. `interactions.py::InteractionBuilder.build_design_matrix_from_parsed`
   - Builds and records smooth term ranges for `parsed.spline_terms`.
   - This happens before interaction columns are appended.

3. `interactions.py::InteractionBuilder._build_mixed_interaction`
   - Builds `categorical x spline` columns.
   - Names look like:

```text
PreviousRenewal[T.1]:bs(InvitedPremiumDiff, 1/9, k, -)
...
PreviousRenewal[T.1]:bs(InvitedPremiumDiff, 9/9, k, -)
```

   - These columns are returned as ordinary interaction columns.
   - They are not registered in `_smooth_terms` / `_smooth_col_indices`.

4. `formula.py::_fit_with_smooth_penalties`
   - Passes only the registered smooth column ranges to Rust.
   - The interaction block is not included in `smooth_col_ranges`.

5. `formula.py::_get_constraint_indices`
   - Explicitly avoids sign-clamping names containing `", k,"`, because smooth
     terms are expected to be handled by the smooth solver.
   - The interaction columns contain `", k,"`, but they were not registered with
     the smooth solver either.

6. `crates/rustystats/src/fitting_py.rs::fit_smooth_glm_unified_py`
   - Only enforces monotonicity for column ranges supplied in
     `smooth_col_ranges`.

7. `crates/rustystats-core/src/solvers/smooth_glm.rs`
   - The alpha-to-beta monotone reparameterization is applied only to those
     supplied ranges.

Net result: monotone spline interaction columns fall between two systems:

```text
not sign-clamped because they look like smooth k terms
not smooth-constrained because they are not registered smooth ranges
```

## Additional Issues Identified

### 1. Function-level monotonicity is not asserted

The library currently trusts coefficient constraints / smooth metadata. The bug
would have been caught by a post-fit function check:

```text
for every monotone term:
  for every relevant categorical interaction level:
    score a dense grid over the training range
    assert non-increasing / non-decreasing within tolerance
```

This should be part of the model fit validation path, at least in debug/test
mode, and ideally available as a public diagnostic.

### 2. Combined monotonicity matters, not just component monotonicity

For `include_main=True`, the non-reference level uses:

```text
f_level(x) = f_main(x) + f_interaction_level(x)
```

The required invariant is that `f_level(x)` is monotone for each level. There
are two possible implementations:

- Conservative: require both `f_main` and every `f_interaction_level` to be
  monotone in the same direction. The sum is then monotone.
- More flexible: enforce monotonicity on the combined coefficients/function for
  each level. This requires coupled constraints between the main block and the
  interaction block.

The first option is simpler and probably acceptable as a first fix. The second
is statistically less restrictive.

### 3. Score-time out-of-training-range values can create extreme artefacts

In the production optimiser, some price-derived features can move far outside
the train range. Example from the annual grid:

```text
RatioToMarket > 13
```

while the train 99th percentile was around:

```text
RatioToMarket ~= 2.5
```

This can produce very large prediction jumps at score time. The renewal
application now clips `InvitedPremiumDiff` and applies a cumulative-min guard,
but rustystats should consider one or more of:

- expose train boundary knots/ranges clearly;
- warn or error when scoring far outside fitted spline boundaries;
- optionally clamp spline inputs to boundary knots;
- provide a monotonicity diagnostic that checks both in-range and out-of-range
  score grids.

### 4. Duplicate row IDs can create false diagnostic jumps

The saved renewal parquet has duplicate `POLICY_DETAILS_ID`s. Diagnostics that
group by policy id can accidentally compare different rows as adjacent price
steps and create artificial jumps. Repro/diagnostic code should use a stable
row id when constructing grids unless the input is known to be one row per
policy.

This is not a rustystats solver bug, but it matters when validating examples.

### 5. Regularization plus monotonicity needs an explicit API decision

Earlier solver work found that some regularized paths can bypass/drop monotone
constraints. The API should not silently accept an unsupported combination.

Recommended behaviour:

```text
if regularization is requested and any monotonicity constraint is present:
    either support the constraints correctly
    or raise a hard error
```

Silent warnings are too easy to miss for production pricing models.

### 6. Boundary / identifiability edge should be tested

The B-spline basis drops a column for identifiability and relies on the
intercept plus coefficient sequencing. Synthetic checks over exact endpoints or
outside fitted boundary knots can show large jumps even when the interior grid
is monotone.

At minimum, add tests covering:

- dense interior grid over the fitted training range;
- exact fitted boundary knots;
- values just outside the fitted boundary knots;
- models with and without an intercept;
- dropped-column B-spline basis with monotonicity enabled.

## Suggested Fix Plan

### Immediate hardening

Add detection in `formula.py::_parse_interaction_spec`:

```text
if an interaction contains a spline term with monotonicity:
    if categorical factors are present:
        raise NotImplementedError / ValidationError
```

This prevents the current silent wrong behaviour while the full fix is built.

### Full support

In `InteractionBuilder._build_mixed_interaction`, when constructing
`categorical x monotone spline` columns:

1. Preserve the spline metadata for each category-specific block:
   - column start;
   - column end;
   - penalty matrix;
   - monotonicity direction;
   - source variable;
   - category level.

2. Include those ranges in `smooth_col_ranges` passed to
   `fit_smooth_glm_unified_py`.

3. Decide whether interaction blocks should be constrained independently or
   whether the combined `main + interaction` level curve should be constrained.

4. Add a post-fit function-level monotonicity check over every constrained
   term and category level.

### Regression tests

Add a test using the synthetic repro above. The fixed invariant should be:

```text
for group in (0, 1):
    max(diff(model.predict(grid_for_group))) <= tolerance
```

Also add tests for:

```text
monotone bs main effect only
monotone bs x categorical with include_main=True
monotone bs x categorical with include_main=False
monotone bs x continuous interaction, or hard error if unsupported
regularization + monotonicity hard error or correct support
score-time values outside fitted boundary knots
```

## Temporary Application Mitigation

The renewal optimiser now applies a score-time cumulative minimum over each
policy's ascending commission grid:

```text
RenewalPrediction_guarded[c] = min(
    RenewalPrediction_raw[c],
    RenewalPrediction_guarded[c - 1],
)
```

This is conservative and guarantees the optimiser cannot prefer a higher price
because the raw retention model spuriously increased. It is not a substitute for
fixing the library, because it can be too punitive after a spurious trough.