# Interaction Diagnostics — Spec

Adds an `interactions=[...]` argument to both `result.diagnostics(...)` and `rs.explore_data(...)`. The argument accepts a list of variable pairs; for each pair the library emits a structured `InteractionDiagnostics` (post-fit) or `InteractionExploration` (pre-fit) record. The output is **raw cell-level data** plus model-state quantities the caller can't cheaply derive themselves (nested-model deviances, block score test, block GVIF, fitted coefficients).

The new argument is independent of the existing `detect_interactions=` flag. `detect_interactions=True` continues to populate `interaction_candidates` with top-K auto-discovered pairs. `interactions=[...]` populates a new field with raw per-pair diagnostic data, whether or not those pairs are in the fitted model.

---

## 1. Design Principle

This spec follows a **library-for-libraries** principle:

- **Emit raw aggregations and model-state outputs only.** A downstream library can compute summary statistics (HHI, monotonicity fractions, sign-flip counts, ranked deviations) from the raw cells in one line.
- **Centerpiece is the cell-level surface grid.** Each cell carries `actual`, `predicted`, `ae_ratio`, `exposure`, and `n`. This is exactly the input a 2D visualization needs: heatmaps, lift surfaces, bubble plots, exposure-shaded surfaces.
- **No refits.** Every emitted field is computable from the fitted model, its design matrix, and one pass through the data. Anything that would require a nested-model refit (`aic_delta`, `bic_delta`, deviance deltas, `nested_predicted` per cell) is **deliberately not emitted** — a downstream library that wants those values can refit themselves; rustystats should not silently triple fit time. Significance is delivered through the Rao score test, which is computed from the full model's design matrix and IRLS weights and is asymptotically equivalent to the LRT under H₀.
- **Bundle only what requires model state.** The library emits the surface grid, the block score test, the block GVIF, the interaction's coefficients, and a representation tag. All cheap.
- **Skip opinionated transforms.** No ranking, no monotonicity checks, no sign-flip detection, no clustering, no "consistency" scalars, no verdicts. Callers choose their own.

Field-naming conventions (`actual` / `predicted` / `ae_ratio`), the plural-noun container pattern (`InteractionDiagnostics` mirroring `FactorDiagnostics`), the optional-field idiom, and reuse of existing sub-types (`ScoreTestResult`, `FactorSignificance`, `FactorCoefficient`) all follow what `types.py` already establishes. JSON serialization works automatically through `_to_dict_recursive`.

---

## 2. Public API

### 2.1 `result.diagnostics(...)`

```python
def diagnostics(
    self,
    train_data: pl.DataFrame,
    test_data: pl.DataFrame | None = None,
    categorical_factors: list[str] | None = None,
    continuous_factors: list[str] | None = None,
    interactions: list[InteractionSpec] | None = None,   # NEW
    ...
) -> ModelDiagnostics: ...
```

`ModelDiagnostics` gains one new field; the existing `interaction_candidates` is unchanged.

```python
@dataclass
class ModelDiagnostics:
    ...
    interaction_candidates: list[InteractionCandidate]                       # unchanged
    interactions: list[InteractionDiagnostics] = field(default_factory=list) # NEW
```

### 2.2 `rs.explore_data(...)`

```python
def explore_data(
    data: pl.DataFrame,
    response: str,
    categorical_factors: list[str] | None = None,
    continuous_factors: list[str] | None = None,
    interactions: list[InteractionSpec] | None = None,   # NEW
    exposure: str | None = None,
    family: str = "gaussian",
    detect_interactions: bool = True,
    ...
) -> DataExploration: ...
```

`DataExploration` gains one new field; the existing `interaction_candidates` is unchanged.

```python
@dataclass
class DataExploration:
    ...
    interaction_candidates: list[InteractionCandidate]                       # unchanged
    interactions: list[InteractionExploration] = field(default_factory=list) # NEW
```

### 2.3 `InteractionSpec` format

Accepted forms:

```python
{"factor1": "VehAge", "factor2": "Region"}    # named keys
("VehAge", "Region")                          # positional tuple
["VehAge", "Region"]                          # positional list
```

The two variables must exist as columns in `data` (or `train_data`). They do not need to appear in the fitted model's `terms=` or `interactions=` at fit time — the diagnostics module computes the surface from the raw columns and (post-fit) the model's predictions.

`pair_type` is determined automatically from column dtype:

| factor1 dtype | factor2 dtype | `pair_type` |
|---|---|---|
| numeric | numeric | `continuous_x_continuous` |
| numeric | categorical / string | `continuous_x_categorical` |
| categorical / string | categorical / string | `categorical_x_categorical` |

Order is normalized: continuous is always `factor1` for the mixed case; for `cat × cat` the higher-cardinality variable is `factor1`.

---

## 3. Data Schema

### 3.1 `InteractionDiagnostics` (post-fit)

```python
@dataclass
class InteractionDiagnostics:
    """Diagnostics for a single interaction pair, post-fit."""

    name: str                                # "VehAge:Region"
    factor1: str
    factor2: str
    pair_type: str                           # see §2.3
    in_model: bool                           # whether a matching interaction term was fitted
    representation: str | None               # "tensor_product" | "target_encoding" | "frequency_encoding" | None

    train_surface_grid: SurfaceGrid
    test_surface_grid: SurfaceGrid | None = None         # populated when test_data is supplied
    coefficients: list[FactorCoefficient] | None = None  # one per interaction design column
    significance: FactorSignificance | None = None       # Wald-style block chi²; populated when in_model is True
    score_test: ScoreTestResult | None = None            # Rao score test; populated when in_model is False
    gvif: float | None = None                            # block GVIF; None when in_model is False
```

Mirrors the existing `FactorDiagnostics` pattern: `significance` for fitted blocks (Wald-style chi² on the coefficient block from the existing covariance matrix — no refit), `score_test` for unfitted blocks (Rao score test using the existing design matrix and IRLS weights — also no refit). Exactly one of the two will be populated, controlled by `in_model`.

### 3.2 `InteractionExploration` (pre-fit)

```python
@dataclass
class InteractionExploration:
    """Pre-fit data summary for a single interaction pair (no model required)."""

    name: str
    factor1: str
    factor2: str
    pair_type: str

    surface_grid: SurfaceGrid                # cells carry actual + exposure + n only
    interaction_strength: float              # partial R² of cells vs. additive null, exposure-weighted (§4.4)
```

### 3.3 Shared sub-types

```python
@dataclass
class SurfaceGrid:
    """Binned 2D surface for an interaction pair."""

    row_axis: str                            # = factor1
    col_axis: str                            # = factor2
    row_type: str                            # "quantile" | "levels"
    col_type: str                            # "quantile" | "levels"
    cells: list[SurfaceCell]
    row_edges: list[float] | None = None     # bin edges when row_type == "quantile"
    row_levels: list[str] | None = None      # level labels when row_type == "levels" (may include "_Other")
    col_edges: list[float] | None = None
    col_levels: list[str] | None = None

@dataclass
class SurfaceCell:
    """One cell of the binned interaction surface."""

    r: int                                   # row index
    c: int                                   # col index
    n: int                                   # row count in this cell
    exposure: float
    actual: float                            # exposure-weighted observed rate
    predicted: float | None = None           # full-model prediction; null in pre-fit
    ae_ratio: float | None = None            # actual / predicted; null in pre-fit
```

`ScoreTestResult`, `FactorSignificance`, and `FactorCoefficient` are existing dataclasses in `types.py:286`, `types.py:276`, and `types.py:298` respectively. No new fields are needed on any of them.

---

## 4. Computation Rules

### 4.1 Pair-type detection
Automatic from column dtype (see §2.3). Caller can override by casting columns before passing.

### 4.2 Binning

| `pair_type` | factor1 binning | factor2 binning |
|---|---|---|
| `continuous_x_continuous` | 6 quantile bins | 6 quantile bins |
| `continuous_x_categorical` | 8 quantile bins | top-15 levels by exposure, rest → `_Other` |
| `categorical_x_categorical` | top-12 levels by exposure, rest → `_Other` | top-12 levels by exposure, rest → `_Other` |

Total cells are capped at `max_total_cells` (default 150). When the cap would be exceeded, categorical-side top-K is reduced before continuous-side bin count is changed.

Quantile bin edges use empirical exposure-weighted quantiles on the rows in `train_data` (post-fit) or `data` (pre-fit). Bin labels are formed as `f"{lo:g}-{hi:g}"`.

All binning parameters are configurable via the kwargs in §7.

### 4.3 `interaction_strength` (pre-fit only)

Partial R² of the cell-level rates against an additive null, exposure-weighted. Reuses the existing `interaction_strength_from_codes` machinery in `crates/rustystats-core/src/diagnostics/exploration.rs:368`:

1. Aggregate `sum(y)` and `sum(exposure)` per cell.
2. Cell rate = `sum(y) / sum(exposure)`.
3. Additive null per cell = `row_rate * col_rate / overall_rate` (multiplicative) or `row_rate + col_rate - overall_rate` (additive), chosen by family.
4. `interaction_strength = SS_residual_from_null / SS_total`, both exposure-weighted.

This is the same scalar that the existing `InteractionCandidate.interaction_strength` returns. Same Rust path, called with an explicit pair list rather than top-K discovery.

### 4.4 `significance` and `score_test` (post-fit only)

Exactly one of these two is populated per interaction; both are no-refit and reuse machinery that already exists for factors.

- **`significance: FactorSignificance`** (populated when `in_model=True`). Wald-style chi² on the interaction's coefficient block, computed from the full model's coefficient covariance matrix. Uses the existing `compute_factor_significance_batch` Rust path (`crates/rustystats-core/src/diagnostics/factor_diagnostics.rs:1570`) with the interaction's `col_start..col_end` from its `TermSlot`. Carries `chi2`, `p`, `dev_contrib`, `dev_pct` — same fields as for a fitted factor.
- **`score_test: ScoreTestResult`** (populated when `in_model=False`). Rao score test: append the would-be interaction columns, compute the score vector at the existing coefficient estimates, evaluate the chi² and p-value. No refit — uses the existing IRLS weights and design matrix from the fitted model.

The Rao score statistic is asymptotically equivalent to the LRT under H₀, so any caller wanting an "AIC delta if I added/removed this interaction" can compute it as `score_statistic - 2 * df` (or `- log(n) * df` for BIC).

### 4.5 `gvif`

Generalized VIF (Fox & Monette) on the interaction's design-column block, computed from the full model's `XtX`. `null` when `in_model=False` or when the model was fitted with `store_design_matrix=False`.

### 4.6 `representation`

Drawn from `TermSlot.extra` for the matching interaction term:

| Value | When |
|---|---|
| `"tensor_product"` | Standard product columns (cat × cat dummies, cat × cont, cont × cont) |
| `"target_encoding"` | Single-column `TE(v1:v2)` encoding |
| `"frequency_encoding"` | Single-column `FE(v1:v2)` encoding |
| `null` | `in_model=False` |

### 4.7 Test-data handling

When `test_data` is supplied, `test_surface_grid` is built using **the same bin edges and level lists** computed from `train_data`. This makes the train and test grids cell-aligned, so the caller can do element-wise comparisons in their own analysis. Cells empty in test have `n=0`, `exposure=0`, and `actual`/`predicted`/`ae_ratio`/`nested_predicted` = `null`.

---

## 5. JSON Output

### Post-fit example

```json
{
  "interactions": [
    {
      "name": "VehAge:Region",
      "factor1": "VehAge", "factor2": "Region",
      "pair_type": "continuous_x_categorical",
      "in_model": true,
      "representation": "tensor_product",

      "train_surface_grid": {
        "row_axis": "VehAge", "row_type": "quantile",
        "row_edges": [0, 2, 5, 10, 20], "row_levels": null,
        "col_axis": "Region", "col_type": "levels",
        "col_edges": null,
        "col_levels": ["R11", "R24", "R52", "_Other"],
        "cells": [
          {"r": 0, "c": 0, "n": 8210, "exposure": 12450.0,
           "actual": 0.083, "predicted": 0.079, "ae_ratio": 1.051},
          {"r": 0, "c": 1, "n": 188, "exposure": 312.0,
           "actual": 0.094, "predicted": 0.081, "ae_ratio": 1.160}
        ]
      },

      "test_surface_grid": null,

      "coefficients": [
        {"term": "VehAge:Region[T.R24]", "estimate": -0.0421,
         "std_error": 0.0188, "z_value": -2.24,
         "p_value": 0.025, "relativity": 0.959}
      ],

      "significance": {"chi2": 47.3, "p": 3.2e-6,
                        "dev_contrib": 18.4, "dev_pct": 0.61},
      "score_test": null,

      "gvif": 2.1
    }
  ]
}
```

### Pre-fit example

```json
{
  "interactions": [
    {
      "name": "VehAge:Region",
      "factor1": "VehAge", "factor2": "Region",
      "pair_type": "continuous_x_categorical",
      "surface_grid": {
        "row_axis": "VehAge", "row_type": "quantile",
        "row_edges": [0, 2, 5, 10, 20], "row_levels": null,
        "col_axis": "Region", "col_type": "levels",
        "col_edges": null,
        "col_levels": ["R11", "R24", "R52", "_Other"],
        "cells": [
          {"r": 0, "c": 0, "n": 8210, "exposure": 12450.0, "actual": 0.083},
          {"r": 0, "c": 1, "n": 188, "exposure": 312.0, "actual": 0.094}
        ]
      },
      "interaction_strength": 0.011
    }
  ]
}
```

### Visualization workflow

For an A/E heatmap: pick `train_surface_grid.cells[*].ae_ratio`, position by `(r, c)`, label axes from `row_edges`/`col_levels`, scale alpha or marker size by `exposure`.

For observed vs fitted bubble plot: x=`actual`, y=`predicted`, size=`exposure`, colored by `(r,c)`.

For credibility masking: filter cells by `exposure >= threshold` of the caller's choice.

For train/test divergence: compute `train.ae_ratio - test.ae_ratio` element-wise (the grids are cell-aligned by construction, §4.7).

For "is this interaction significant": read `significance.p` (fitted) or `score_test.pvalue` (unfitted) — both are no-refit.

---

## 6. Implementation Hooks

| Concern | File / Symbol | Action |
|---|---|---|
| Dataclasses | `python/rustystats/diagnostics/types.py` | Add `InteractionDiagnostics`, `InteractionExploration`, `SurfaceGrid`, `SurfaceCell`. (No new sub-types — significance/score test reuse existing `FactorSignificance` / `ScoreTestResult`.) |
| `__all__` registration | `python/rustystats/diagnostics/types.py:18` | Add the four new type names |
| Post-fit driver | `python/rustystats/diagnostics/api.py:868` `compute_diagnostics` | Add `interactions=` arg; route to new computer |
| Post-fit computer | `python/rustystats/diagnostics/pair_diagnostics.py` (new) | New module: `_PairDiagnosticsComputer` analogous to `_FactorDiagnosticsComputer` |
| Pre-fit driver | `python/rustystats/diagnostics/explorer.py:1063` `explore_data` | Add `interactions=` arg; emit `interactions` field on `DataExploration` |
| Factor matching | `python/rustystats/diagnostics/factors.py:117` `_match_factor` | Existing `:` exclusion stays; the new pair pipeline owns interaction columns |
| Pair cell aggregation (Rust) | `crates/rustystats-core/src/diagnostics/exploration.rs:393` `aggregate_interaction_cells_dense/sparse` | Generalize: accept explicit `(factor1, factor2)` pair list instead of computing top-K |
| Pre-fit interaction strength (Rust) | `crates/rustystats-core/src/diagnostics/exploration.rs:368` `interaction_strength_from_codes` | Already callable per-pair; just iterate over the user list |
| Block score test (Rust) | `crates/rustystats-core/src/diagnostics/factor_diagnostics.rs:1570` `compute_factor_significance_batch` | Already accepts arbitrary `param_indices_per_factor`. Call with the interaction's `col_start..col_end` from `TermSlot` |
| Fitted-interaction discovery | `python/rustystats/diagnostics/interactions.py:275` `TermSlot._term_slots` | Iterate `[s for s in builder._term_slots if s.term_type == "interaction"]` to set `in_model` and `representation` |
| Default constants | `python/rustystats/diagnostics/api.py` (top of file) | Add `DEFAULT_INTERACTION_CONT_CONT_BINS`, `DEFAULT_INTERACTION_CONT_CAT_BINS`, `DEFAULT_INTERACTION_CAT_CAT_TOP_K`, `DEFAULT_INTERACTION_MAX_CELLS` |
| Auto-save target | `python/rustystats/diagnostics/api.py:1157` | New `interactions` field flows into `analysis/diagnostics.json` |
| Auto-save target (pre-fit) | `python/rustystats/diagnostics/explorer.py:1270` | New `interactions` field flows into `analysis/exploration.json` |

TE-encoded single-column interactions (`TE(v1:v2)`) are handled by the pair computer via `TermSlot.extra["categorical_flags"]`: when both sides are categorical and the slot's column count is 1, the surface is computed from raw data (not the design matrix); `gvif` is computed on that one column.

---

## 7. Configuration Defaults

| Argument (on `diagnostics` / `explore_data`) | Default | Notes |
|---|---|---|
| `interactions` | `None` | When `None`, the new field is empty and no surface computation runs |
| `binning_cont_cont` | `(6, 6)` | `(rows, cols)` quantile bins |
| `binning_cont_cat` | `(8, 15)` | `(continuous_bins, top_k_levels)` |
| `binning_cat_cat` | `(12, 12)` | `(top_k_factor1, top_k_factor2)` |
| `max_total_cells` | `150` | Categorical-side top-K shrinks first |

---

## 8. Out of Scope

The following are **not emitted** because they are either trivially derivable from `cells`/`coefficients`/the existing top-level data, or are opinionated transforms a downstream library should choose itself:

**Requires a nested-model refit — intentionally excluded to keep diagnostics cheap:**
- `nested_predicted` per cell (would need a refit or zero-coef approximation per interaction)
- `aic_delta`, `bic_delta` vs. main-effects-only model
- `train_dev_delta_pct`, `test_dev_delta_pct`
- Friedman's H-statistic (needs nested predictions on a PD grid)
- A caller that wants any of these can refit themselves and compute from the fitted model — the library should not silently refit. The Rao score test is emitted as a no-refit equivalent for the significance question.

**Derivable from emitted data:**
- Verdict / keep-drop / recommendation strings
- Cell counts, credible-cell counts, exposure HHI, top-K exposure share
- Min/max/range of cell rates or relativities
- Monotonicity fractions, sign-flip flags, "directional consistency" scalars
- Train/test divergence summary (the two grids are cell-aligned; caller computes)
- Pre-fit `lift_ratio`, `actual_range`, `overall_actual`, `additive_explained_pct`
- Per-cell `credible: bool` (the caller has `exposure` + their own threshold)
- Top-K deviation rankings (caller sorts cells by their own criterion)
- `aic_delta_approx`, `bic_delta_approx` (caller computes from `significance.chi2` / `score_test.statistic` and the block df)

**Opinionated transformations:**
- K-cluster compression of high-cardinality cat × cat (caller chooses clustering method)
- Per-cell warning strings or interpretation fields
- Per-cell p-values and standard errors

**Methodologically not bundled:**
- Wald tests on individual interaction coefficient design columns (`significance` is on the block; `coefficients` carries the per-column z's if a caller wants those)
- LRT vs. saturated model
- 2D partial-dependence raw arrays (the surface grid carries cell-level fitted means instead)
- 3D surface payloads

**Unchanged:**
- The existing `interaction_candidates: list[InteractionCandidate]` shape (including `InteractionCandidate.recommendation` — that's the existing detector's output, not this spec's concern)
