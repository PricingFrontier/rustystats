# Factor Diagnostics — Lean Extensions

Adds two model-state-derived fields to the existing `FactorDiagnostics` output: block GVIF, and a typed train/test bin-pair struct. Both require the design matrix or a second pass through test data, but **neither requires a model refit**. They cannot be cheaply derived by a downstream library from existing diagnostics output.

This spec follows the same **library-for-libraries** and **no-refit** principles as the interaction-diagnostics spec.

---

## 1. Design Principle

The existing `FactorDiagnostics` (in `python/rustystats/diagnostics/types.py:309`) already emits the raw per-bin / per-level data a caller needs:

- `actual_vs_expected: list[ActualExpectedBin]` — per-bin `bin`, `n`, `exposure`, `actual`, `expected`, `ae_ratio`, `ae_ci`
- `coefficients: list[FactorCoefficient]`
- `residual_pattern: ResidualPattern(resid_corr, var_explained)`
- `significance: FactorSignificance(chi2, p, dev_contrib, dev_pct)` — already a no-refit Wald-style block chi² on the factor's coefficient block
- `score_test: ScoreTestResult` — already a no-refit Rao score test for unfitted factors
- `relative_importance: float`

From these, a downstream library can already compute:

- Top-K worst-fit bins, credibility flags per bin, HHI / concentration of exposure
- Monotonicity / sign-flip checks
- High-cardinality clustering / level grouping
- AIC / BIC delta approximations: `significance.chi2 - 2*df` (or `- log(n)*df`) is asymptotically equivalent to the exact LRT-derived delta

What's **missing** from `FactorDiagnostics` and **cannot** be cheaply derived without library help:

1. **Block GVIF per factor.** `ModelDiagnostics.vif: list[VIFResult]` is per design column — a 10-knot spline shows 10 separate VIFs that mostly reflect within-basis collinearity. A block GVIF (Fox & Monette) on the factor's design columns needs `XtX` block determinants the caller doesn't have.
2. **Typed train/test stability per factor.** The existing `TrainTestComparison.factor_divergence` is `dict[str, list[dict[str, Any]]]` — loose-typed and only contains raw per-level numbers. A typed cell-aligned analog (matching what the interaction spec does for 2D surfaces) is a strict win.

Both additions are **no-refit**:
- GVIF is two determinants of `XtX` blocks — milliseconds per factor.
- Train/test bin pairs are a second pass through `test_data` using the train-derived bin edges — O(n_test) per factor.

**Intentionally not added:** any "leave-one-out" or "vs-null-model" deviance / AIC / BIC delta. Each of those requires a refit *per fitted factor*. For a 20-factor model that's 20 refits before diagnostics return. The existing `significance.chi2` already gives the no-refit Wald-style equivalent; the asymptotic AIC delta is `chi2 - 2*df`.

---

## 2. Data Schema

### 2.1 Additions to `FactorDiagnostics`

```python
@dataclass
class FactorDiagnostics:
    """Complete diagnostics for a single factor."""

    # Existing fields (unchanged) ...
    name: str
    factor_type: str
    in_model: bool
    transform: str | None
    coefficients: list[FactorCoefficient] | None
    actual_vs_expected: list[ActualExpectedBin]
    residual_pattern: ResidualPattern
    univariate: ContinuousFactorStats | CategoricalFactorStats | None = None
    significance: FactorSignificance | None = None
    score_test: ScoreTestResult | None = None
    relative_importance: float | None = None

    # NEW
    gvif: float | None = None                            # block GVIF on the factor's design columns
    train_test_bins: list[FactorBinPair] | None = None   # cell-aligned train/test pairs per bin
```

`gvif` is `None` when `in_model=False` or when the model was fitted with `store_design_matrix=False`. `train_test_bins` is `None` when `test_data` is not supplied to `diagnostics(...)`.

### 2.2 New sub-types

```python
@dataclass
class FactorBinPair:
    """Train/test pair for one bin/level of a factor (cell-aligned by bin label)."""

    bin: str                                 # bin label or level (matches ActualExpectedBin.bin)
    train_n: int
    train_exposure: float
    train_actual: float
    train_predicted: float
    train_ae_ratio: float
    test_n: int                              # 0 when bin is empty in test
    test_exposure: float
    test_actual: float | None                # null when test_n == 0
    test_predicted: float | None
    test_ae_ratio: float | None
```

`FactorBinPair` is intentionally flat so that callers iterating the list can pull both sides without dictionary navigation. Cell alignment is by `bin` label, which is computed from the same edges/levels as the train-side `actual_vs_expected` — matching the interaction spec's approach (§4.7 of the interaction spec).

---

## 3. Computation Rules

### 3.1 `gvif`

Generalized VIF (Fox & Monette) on the factor's design-column block, computed from the full model's `XtX` correlation matrix. Reuses whatever path the existing per-column VIF computer uses; the block formula reduces to a determinant of the correlation matrix block. **No refit.** Cost: O(p³) once per factor where p is the total design column count — typically milliseconds.

### 3.2 `train_test_bins`

For each bin/level emitted in `actual_vs_expected`, build the matching test-data aggregate using **the same bin edges / level list** computed from `train_data`. Bins empty in test get `test_n=0` and null rates.

Implementation reuses `_FactorDiagnosticsComputer.compute_factor_diagnostics` (`factors.py:328`) on `test_data` with the train-derived bin edges threaded through. **No refit.** Cost: O(n_test) per factor.

---

## 4. JSON Output

```json
{
  "factors": [
    {
      "name": "VehAge",
      "factor_type": "continuous",
      "in_model": true,
      "transform": "bs(VehAge, df=5)",
      "actual_vs_expected": [ "...existing bins..." ],
      "coefficients": [ "..." ],
      "residual_pattern": {"resid_corr": 0.02, "var_explained": 0.001},
      "significance": {"chi2": 142.3, "p": 1.1e-28, "dev_contrib": 38.2, "dev_pct": 1.8},
      "relative_importance": 0.21,

      "gvif": 1.4,

      "train_test_bins": [
        {"bin": "0-2", "train_n": 8210, "train_exposure": 12450.0,
         "train_actual": 0.083, "train_predicted": 0.079, "train_ae_ratio": 1.051,
         "test_n": 2103, "test_exposure": 3120.0,
         "test_actual": 0.087, "test_predicted": 0.079, "test_ae_ratio": 1.101}
      ]
    }
  ]
}
```

For visualization, `train_test_bins` provides flat, cell-aligned data: a downstream library plots train vs. test A/E by bin without any pre-aggregation logic, computes divergence as one element-wise subtraction, and decides for itself what "divergent" means.

For "how important is this factor": `significance.chi2`, `significance.dev_contrib`, `significance.dev_pct`, `relative_importance` are all already emitted (no change here). An AIC-delta approximation if the factor were removed is `significance.chi2 - 2 * df`, derivable in one line.

---

## 5. Implementation Hooks

| Concern | File / Symbol | Action |
|---|---|---|
| Dataclasses | `python/rustystats/diagnostics/types.py:309` `FactorDiagnostics` | Add `gvif`, `train_test_bins` fields (both optional, default `None`) |
| New sub-type | `python/rustystats/diagnostics/types.py` | Add `FactorBinPair` |
| `__all__` registration | `python/rustystats/diagnostics/types.py:18` | Add `FactorBinPair` |
| Block GVIF computation | `python/rustystats/diagnostics/factors.py` (extend existing VIF path) | Compute block GVIF per factor using `TermSlot.col_start/col_end` (interactions.py:275) to identify the factor's design columns |
| Train/test bin pairs | `python/rustystats/diagnostics/factors.py:328` `compute_factor_diagnostics` | Already accepts `data` arg; call twice (train + test) with shared bin edges, then merge per-bin records |
| `TrainTestComparison.factor_divergence` | `python/rustystats/diagnostics/types.py:516` | Deprecate this loose-typed field in favor of `factors[*].train_test_bins`. Keep populated for one release for backwards compatibility |

---

## 6. Configuration

| Argument | Default | Effect |
|---|---|---|
| `compute_gvif` | `True` | When `False`, skip block GVIF (only useful for pathological designs where determinant cost dominates — usually not necessary) |

No `nested_method` / `compute_vs_null` flags — no refits are ever performed.

---

## 7. Out of Scope

**Requires a nested-model refit — intentionally excluded to keep diagnostics cheap:**
- Per-factor leave-one-out AIC / BIC delta vs. nested model
- Per-factor `train_dev_delta_pct`, `test_dev_delta_pct`
- Any "vs null model" deviance comparison
- The existing `significance.chi2` (no-refit Wald-style block chi²) covers significance; `significance.chi2 - 2*df` is the asymptotic AIC-delta equivalent for callers that want it.

**Derivable from existing per-bin data:**
- Per-factor credibility block (n_credible, exposure HHI, top-K share)
- Per-bin `credible: bool` flag (caller has `exposure` + threshold)
- Top-K worst-fit bins / levels (caller sorts `actual_vs_expected`)
- Monotonicity / sign-flip flags
- Directional consistency scalars
- Range and quantile summaries of bin rates or relativities
- Train/test divergent-bin count, max-divergence, "unstable" booleans (caller computes from `train_test_bins`)

**Opinionated transformations:**
- K-cluster compression for high-cardinality categoricals (caller picks clustering method)
- Per-bin / per-level warning strings or recommendations
- "Verdict" or keep/drop suggestions per factor
- Coefficient stability bootstrap

**Already covered by existing fields:**
- Total in-sample deviance contribution (`significance.dev_contrib`)
- In-sample importance share (`relative_importance`)
- Per-bin A/E confidence interval (`ActualExpectedBin.ae_ci`)
- Univariate stats (`univariate`)
- Per-coefficient Wald z and p (`coefficients[*]`)
