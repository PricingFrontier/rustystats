---
title: destyler — implementation spec
type: spec
status: draft
version: 0.1
created: 2026-05-24
author: Ralph (drafted with Claude)
depends-on: [rustystats >= 0.7.0 (with pure-Python residuals fallback) or >= 0.8.0 (with native working_response_weights), catboost >= 1.2]
related: [[rustystats]], [[haute]], [[competitors/wtw-radar]]
tags: [spec, destyler, glm, gbm, distillation, catboost, rustystats, haute, interpretable-ml]
---

# destyler — implementation spec

> **One-liner.** **destyler** is an open-source **GLM distillation library**. It distils a Layered CatBoost *teacher* into a transparent, filable [[rustystats]] GLM *student*. Insurance pricing teams get the structural insight of a GBM (which variables matter, which interactions are real, what shape each effect has) **automatically expressed as a deployable GLM rate table** — no black-box scoring at runtime, no consultancy lock-in.
>
> **What it competes with.** WTW Radar's "Machine-led GLM" workflow ([[competitors/wtw-radar#27-the-machine-led-glm-workflow-radar-5-oct-2025]]), and the wedge Akur8 leaves open by requiring manual interaction specification. destyler is the open, code-first, no-consultancy answer.
>
> **What it depends on.** [[rustystats]] (≥ 0.8.0) for the GLM fit, diagnostics, and serialisation; CatBoost for the teacher model; nothing else heavy. **Pure orchestrator** — no GLM internals, no math in destyler itself.
>
> **License + repo (proposed).** AGPL-3.0, `github.com/PricingFrontier/destyler`, published on PyPI as `destyler`.

---

## 1. Name and concept

**destyler** = *distiller*, evocatively spelled. The library performs **knowledge distillation** (Hinton, Vinyals & Dean, 2015) on insurance pricing models:

- **Teacher** = a Layered CatBoost model. Accurate, captures non-linearities and interactions, but opaque and not regulator-friendly.
- **Student** = a [[rustystats]] GLM. Filable, transparent, every prediction decomposable, every coefficient inspectable.
- **Distillation** = the proposal step that extracts the teacher's structural knowledge (which variables, which interactions, which shape) into the student's specification, then fits the student to the data (not just to the teacher's outputs).

Knowledge distillation in the classic sense (Hinton 2015) trains a small student to *mimic the soft outputs* of a large teacher. destyler does something subtler: the teacher's job is to **propose the student's architecture**, then the student fits the data directly. This is closer to Lindholm/Lindskog/Palmquist (2024) "Black-Box Guided GLM Building" — the academic prior art that shields destyler from WTW's patent (see [§11](#11-ip-and-prior-art-design-choices)).

> "**Pour out the complexity, keep the structure.**"

---

## 2. Goal and scope

| Property | Target |
|---|---|
| **Inputs** | Polars `DataFrame` with response, optional exposure/offset, optional weights, optional monotone constraints, optional cat-feature list. |
| **Output** | A [[rustystats]] `GLMModel` (saved as `.rsglm`) ready for [[haute]] deployment, plus a self-contained `DistillationReport` (HTML + JSON) explaining how the GLM was built. |
| **End-to-end runtime** | < 10 minutes for 1M rows × 50 features on a modern laptop CPU (CatBoost is the bottleneck). |
| **Interpretability** | Every prediction decomposable via `rustystats.predict_contributions` (additivity validated). Every term tagged with its distillation origin (Layer 1 / 2 / 3). |
| **Filability** | Output is a multiplicative-factor GLM rate table; PMML 4.4 and ONNX export inherited from [[rustystats]]. |
| **Determinism** | Same data + same seed → byte-identical artefact. |
| **Dependencies** | `rustystats >= 0.8.0`, `catboost >= 1.2`, `polars >= 1.0`, `numpy >= 1.20`. **That's it.** |

**Non-goals (v0.1):**

- Replacing the teacher with a non-tree model (EBM, GAMI-Net) — possible later behind a backend abstraction; v0.1 commits to CatBoost.
- Self-contained GLM — destyler always defers to [[rustystats]]. (See [§11](#11-ip-and-prior-art-design-choices) on why this is intentional: it's the cleanest separation of concerns.)
- Multi-response (frequency × severity bundled into Tweedie covers most pricing use cases; explicit two-stage Tweedie deferred to v0.2).
- Online / streaming model updates.
- A graphical UI — that's [[haute]]'s job, destyler is the engine.
- Fairness checks, optimisation, regulatory-filing workflow — out of scope.

---

## 3. Architecture overview

```
┌─────────────────────────────────────────────────────────────────┐
│  User / haute training node                                     │
│  destyler.distill(data, response, family, ...)                  │
│  or destyler.Distiller(...).fit().refine()                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────────┐
│  destyler package (pure Python orchestration)                   │
│                                                                  │
│  Phase 1 — Teacher: LayeredCatBoost                             │
│   ├─ layer_1 = depth-1 GBM  (main effects)                      │
│   ├─ layer_2 = depth-2 GBM  (2-way interactions)                │
│   └─ layer_3 = depth-3 GBM  (3-way interactions)                │
│                                                                  │
│  Phase 2 — Attribution                                          │
│   ├─ per-feature gain by layer                                  │
│   ├─ 2-way pair gain  (TreeSHAP interaction values)             │
│   ├─ 3-way triple gain (path co-occurrence)                     │
│   └─ partial-dependence curves per feature per layer            │
│                                                                  │
│  Phase 3 — Recipe (the distillation step)                       │
│   ├─ pick top-K main effects                                    │
│   ├─ pick top-M 2-way + top-N 3-way  (heredity-filtered)        │
│   ├─ propose spline knots from PD shape                         │
│   ├─ propose categorical grouping from leaf values              │
│   ├─ propose TE/FE for high-cardinality categoricals            │
│   └─ assemble rustystats `terms` + `interactions` dicts         │
│                                                                  │
│  Phase 4 — Student: rustystats GLM fit                          │
│   └─ rs.glm_dict(**recipe).fit(regularization='elastic_net')    │
│                                                                  │
│  Phase 5 — Bundle and report                                    │
│   ├─ predict_contributions trace (per-term + per-layer origin)  │
│   ├─ A/E + lift charts (rustystats diagnostics)                 │
│   ├─ Student-vs-Teacher gap (base_predictions_comparison)       │
│   ├─ .dst bundle (teacher cbms + student rsglm + recipe)        │
│   └─ export .rsglm / PMML / ONNX                                │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ├─ catboost 1.2.x         (teacher)
                           ├─ rustystats >= 0.8.0    (student)
                           ├─ polars >= 1.0
                           └─ numpy >= 1.20
```

**Key architectural commitments:**

- **destyler does not own any math.** No linear algebra in destyler. CatBoost owns tree training; rustystats owns GLM fitting and diagnostics. destyler is pure orchestration: assemble inputs, sequence calls, transform outputs.
- **destyler imports rustystats; rustystats does not know about destyler.** One-way dependency. This means rustystats stays lean and broadly useful; destyler can ship faster without rustystats release coupling.
- **The student is the deployable.** The teacher exists only at training time and during distillation. Production scoring uses only the `.rsglm`. The `.dst` bundle is the full audit trail (teacher + student + recipe + attribution).

---

## 4. Phase 1 — Teacher: Layered CatBoost

### 4.1 Why CatBoost (vs LightGBM / XGBoost)

| Property | CatBoost | Why it matters here |
|---|---|---|
| **Ordered target encoding** for high-cardinality categoricals | Built-in, ordered statistics (arxiv:1706.09516) | Matches [[rustystats]]'s `target_encoding` exactly. No leakage. |
| **Monotone constraints** per feature | Yes | Required for regulatory acceptability. Honoured per layer. |
| **Symmetric (oblivious) trees** | Default | Clean, fast inference; cleaner decomposition for Phase 2. |
| **TreeSHAP / SHAP interaction values** | Native (`get_feature_importance(type='ShapInteractionValues')`) | Powers Phase 2 interaction ranking. |
| **Tweedie loss** | Yes (`Tweedie:variance_power=1.5`) | Pricing covers frequency-severity in one fit. |
| **Poisson, MAE, RMSE, LogLoss, Quantile** | Yes | Family coverage matches rustystats. |
| **Categorical native handling** | Yes (`cat_features=`) | No upstream dummy encoding required. |
| **Already a [[haute]] dependency** | 1.2.8 | Zero new top-level deps when integrating into haute. |

### 4.2 Layer training schedule

```python
# destyler/_teacher.py (sketch)

def train_teacher(
    X: pl.DataFrame,
    y: np.ndarray,
    family: Family,
    offset: np.ndarray | None,
    weights: np.ndarray | None,
    cat_features: list[str],
    monotone_constraints: dict[str, int] | None,
    layers: int = 3,
    layer_params: dict | None = None,
    seed: int = 0,
) -> LayeredTeacher:
    eta_offset = np.log(offset) if offset is not None else 0.0
    eta = np.full_like(y, fill_value=_initial_eta(y, family), dtype=np.float64)
    models = []

    for depth in range(1, layers + 1):
        # Working residuals on link scale, given current eta — uses rustystats
        z, w = rs.working_response_weights(
            y, eta, family=family.name, offset=eta_offset, weights=weights,
        )

        params = dict(
            depth=depth,
            iterations=_resolve_iter(depth, layer_params),
            learning_rate=_resolve_lr(depth, layer_params),
            loss_function=_catboost_loss(family),
            monotone_constraints=monotone_constraints or {},
            random_seed=seed + depth,
            thread_count=1,           # determinism by default
            verbose=False,
            allow_writing_files=False,
        )
        model = CatBoostRegressor(**params).fit(
            X, z, sample_weight=w, cat_features=cat_features,
        )
        models.append(model)
        eta = eta + model.predict(X)  # residual-chained on link scale

    return LayeredTeacher(models=models, family=family, ...)
```

### 4.3 Design choices

- **Working residuals on the link scale.** For Poisson + log-link, `z = log(μ̂) + (y − μ̂)/μ̂`, `w = μ̂`. Boosting on the link scale (not the response) means each layer is composable additively in η-space, which is exactly what we need to project to a GLM at the end. This requires the new `rustystats.working_response_weights` helper — see [§9.1](#91-working-response_weights).
- **Each layer is itself a full boosted ensemble.** Layer 1 is a depth-1 GBM (many depth-1 trees); Layer 2 is a depth-2 GBM on Layer 1 residuals. This matches the McGrath & Milam CAS 2022 layered structure.
- **Per-layer iterations / learning rate auto-tuning.** Layer 1 gets many iterations and small LR (it's stabilising main effects); Layer 2 gets fewer (residual signal is weaker); Layer 3 fewer still. Tune via held-out deviance with early stopping inside each layer.
- **Monotone constraints applied per layer.** A monotone constraint on `BonusMalus` is honoured by every layer's trees. CatBoost supports this natively.
- **Loss function chosen by family**, not by user. `poisson` → `Poisson`, `gamma` → `Tweedie:variance_power=2`, `tweedie` → `Tweedie:variance_power=p`, `gaussian` → `RMSE`, `binomial` → `Logloss`. Negative-binomial is approximated by `Poisson` with offset = exposure; dispersion estimated post-hoc.
- **Seed bumped per layer.** Keeps across-layer permutations independent.

### 4.4 Hyperparameter defaults

| Layer | depth | iterations | learning_rate | l2_leaf_reg |
|---|---|---|---|---|
| 1 (main effects) | 1 | 1000 (early-stop patience 50) | 0.05 | 3.0 |
| 2 (2-way) | 2 | 500 (early-stop 30) | 0.05 | 3.0 |
| 3 (3-way) | 3 | 250 (early-stop 20) | 0.05 | 3.0 |

Override via `distill(..., layer_params={1: {...}, 2: {...}, ...})`. Defaults chosen for n ≈ 1M, p ≈ 30–60, motor-pricing-style data.

### 4.5 Edge cases handled in Phase 1

- **Tweedie boundary cases** (`p ≤ 1`, `p ≥ 2`) — refuse with a clear error.
- **Zero-exposure rows** — drop with a warning. CatBoost cannot honour exposure offsets cleanly; we apply exposure via the working-residual transform.
- **Constant features** — auto-drop before training; surfaced in the report.
- **High-cardinality categoricals (> 1000 levels)** — let CatBoost handle internally via ordered TE; *don't* pre-encode in Polars.
- **NaN handling** — CatBoost's `nan_mode='Min'` for monotonic features; NaN bin counted toward indicator extraction in Phase 2.

---

## 5. Phase 2 — Attribution extraction

### 5.1 Per-feature gain by layer

Each CatBoost layer exposes per-feature gain via `model_l.get_feature_importance()`. We compute, for each feature:

- `gain_main`: total gain from Layer 1 (main effects).
- `gain_2way`: total gain from Layer 2 trees that include this feature.
- `gain_3way`: total gain from Layer 3 trees that include this feature.

### 5.2 Pair-level interaction gain

For 2-way interactions we use TreeSHAP interaction values, which are *exact* for tree models:

```python
shap_int = model_2.get_feature_importance(
    data=Pool(X, label=y, cat_features=cat_features),
    type='ShapInteractionValues',
)
# shape: (n_rows, n_features+1, n_features+1)
# sum of absolute off-diagonal entries gives pair-level importance
```

Cost: `O(n · p² · depth)`. For n=1M, p=50, depth=2 that's ~100M ops — fine. For n=10M+ we sample 100K rows for attribution; the GLM fit in Phase 4 uses the full data.

### 5.3 Triple-level interaction gain

SHAP interaction values are exact only for pairwise effects. For 3-way effects we use **path co-occurrence**: for each Layer-3 tree, the set of features along the root-to-leaf path defines a 3-way candidate. We sum the leaf-weight-weighted gain across all paths that touch a given feature triple.

### 5.4 Output dataclass

```python
@dataclass(frozen=True)
class LayeredAttribution:
    feature_gain_by_layer: dict[str, dict[int, float]]  # feature → {1: main, 2: 2way, 3: 3way}
    pair_gain: dict[tuple[str, str], float]              # 2-way pair → gain
    triple_gain: dict[tuple[str, str, str], float]       # 3-way triple → gain
    pd_curves: dict[str, dict[int, np.ndarray]]          # feature → layer → PD on grid
    leaf_values_categorical: dict[str, dict[str, float]] # cat feature → level → Layer-1 leaf value
    cat_features: list[str]
    cont_features: list[str]
```

The PD curves per layer drive spline-knot proposal in Phase 3.

---

## 6. Phase 3 — Recipe (the distillation step)

This is the heart of destyler. Given `LayeredAttribution`, produce a [[rustystats]] `terms` dict + `interactions` list that we can hand to `glm_dict()`.

### 6.1 Main-effect selection

```python
def propose_main_effects(
    attr: LayeredAttribution,
    top_k: int = 15,
    min_gain_share: float = 0.001,
    cat_high_card_threshold: int = 50,
) -> dict[str, dict]:
    terms = {}
    total_main_gain = sum(g[1] for g in attr.feature_gain_by_layer.values())
    ranked = sorted(
        attr.feature_gain_by_layer.items(),
        key=lambda kv: kv[1][1], reverse=True,
    )
    for feature, gains in ranked[:top_k]:
        if gains[1] / total_main_gain < min_gain_share:
            continue
        if feature in attr.cont_features:
            terms[feature] = propose_spline_term(feature, attr)
        elif feature in attr.cat_features:
            n_levels = len(attr.leaf_values_categorical.get(feature, {}))
            if n_levels > cat_high_card_threshold:
                terms[feature] = {"type": "target_encoding", "prior_weight": 1.0}
            else:
                terms[feature] = {"type": "categorical"}
        else:
            terms[feature] = {"type": "linear"}
    return terms
```

### 6.2 Spline-knot proposal from PD curve

This is a destyler-internal heuristic — knot placement *about how to use* splines, not spline math itself. Lives in `destyler/_knots.py`:

```python
def propose_spline_term(feature: str, attr: LayeredAttribution) -> dict:
    pd = attr.pd_curves[feature][1]      # Layer 1 PD curve
    x_grid = attr.pd_grids[feature]
    monotonicity = _infer_monotonicity(pd)   # Spearman ρ vs feature value
    knots = _propose_knots_from_curve(
        x=x_grid, y=pd,
        max_knots=7,
        monotonicity=monotonicity,
        method="kink",
    )
    return {
        "type": "bs",
        "knots": knots.tolist(),
        "monotonicity": monotonicity,
    }


def _propose_knots_from_curve(x, y, max_knots, monotonicity, method) -> np.ndarray:
    """Locate kink positions in a 1-D curve.
    1. Fit a smooth (scipy UnivariateSpline or numpy polynomial) to (x, y).
    2. Compute |y''| on a fine grid.
    3. Return the x-coordinates of the top-`max_knots` local maxima.
    """
```

The full heuristic:

1. Take the Layer-1 PD curve for the feature (1D, quantile grid of 50 points).
2. Fit a smooth (P-spline with k=20) to the PD curve.
3. Compute second-derivative magnitudes; locate top-K "kink" points (knots).
4. Map back to the original feature scale.

Monotonicity inherited from user-specified constraints if provided; otherwise inferred from PD slope sign (Spearman ρ ≥ 0.95). The output is just a `knots=[...]` array that gets passed to [[rustystats]]'s existing `glm_dict` API — no rustystats changes required.

### 6.3 Interaction selection — heredity-filtered

```python
def propose_interactions(
    attr: LayeredAttribution,
    main_terms: dict[str, dict],
    top_2way: int = 10,
    top_3way: int = 5,
    heredity: bool = True,
    promote_missing_main: bool = True,
) -> tuple[list[dict], list[str]]:
    interactions = []
    main_features = set(main_terms.keys())
    promoted = []

    # 2-way candidates ranked by SHAP pair gain
    for (a, b), gain in sorted(attr.pair_gain.items(), key=lambda kv: kv[1], reverse=True)[:top_2way]:
        missing = {a, b} - main_features
        if heredity and missing:
            if promote_missing_main:
                for f in missing:
                    main_terms[f] = _default_main_term(f, attr)
                    main_features.add(f)
                    promoted.append(f)
            else:
                continue
        interactions.append({
            a: main_terms[a], b: main_terms[b], "include_main": False,
        })

    # 3-way same logic — see destyler/_recipe.py
    return interactions, promoted
```

**The heredity flag is a real engineering decision.** GAMI-Net's stance is that interactions without their main effects are statistically unsound. WTW's Layered GBM is silent on it. destyler defaults to `heredity=True, promote_missing_main=True`: if Layer 2 finds a `Convictions × Tenure` interaction with strong gain but Convictions has 0% main gain (the CAS paper Use Case 2 scenario), destyler **adds Convictions as a main effect** with an explicit annotation in the report.

### 6.4 Categorical grouping for mid-cardinality cats

For low-cardinality categoricals (handled as `{"type": "categorical"}`), no extra step. For mid-cardinality (say 50-200 levels) where we still want `categorical` rather than TE, destyler proposes level grouping and **pre-aggregates the Polars column** before handing to [[rustystats]]:

1. Take Layer-1 leaf values per level (`attr.leaf_values_categorical[feature]`).
2. Hierarchical clustering on the leaf values (`scipy.cluster.hierarchy.linkage` + `fcluster`, ~5 lines).
3. Cut the dendrogram at K clusters (default K = ⌈log₂(n_levels)⌉ + 5).
4. Replace the original column with the cluster label and reference the new column in the recipe:

```python
# destyler does this internally before fit_student()
cluster_map = _suggest_grouping(attr.leaf_values_categorical["Brand"], K=8)
df = df.with_columns(
    pl.col("Brand").replace_strict(cluster_map).alias("Brand_grp")
)
recipe.terms["Brand_grp"] = {"type": "categorical"}
recipe.terms.pop("Brand", None)
recipe.grouping_maps["Brand"] = cluster_map  # kept for audit + scoring
```

This is **cleaner than threading a `grouping` parameter through rustystats**: the grouping is visible in the data column, `predict_contributions` correctly shows the *grouped* level, and the audit trail lives in the recipe's `grouping_maps` field. No rustystats changes required.

At scoring time, destyler applies the same `replace_strict(cluster_map)` to incoming data before calling `student.predict()`. Unseen levels fall back to a designated "other" cluster.

### 6.5 Recipe artefact

```python
@dataclass(frozen=True)
class Recipe:
    """The 'recipe' the teacher distils for the student to follow."""
    terms: dict[str, dict]                   # → rustystats `terms`
    interactions: list[dict]                 # → rustystats `interactions`
    family: str
    link: str | None
    offset: str | None
    weights: str | None
    promoted_main_effects: list[str]         # features added solely for heredity
    skipped_interactions: list[tuple]        # candidates dropped + reason
    notes: list[str]                         # human-readable build log

    def to_glm_dict_kwargs(self) -> dict:
        """Returns kwargs ready for `rs.glm_dict(**kwargs, data=...)`."""
```

---

## 7. Phase 4 — Student: rustystats GLM fit

```python
student = rs.glm_dict(**recipe.to_glm_dict_kwargs(), data=data).fit(
    regularization="elastic_net",
    cv=5,
    selection="1se",
)
```

That's the whole phase. The hard work is done in Phase 3.

**Why elastic-net by default?** Conservative shrinkage protects against overfit interactions promoted from Phase 3.

**Optional: Akur8-killer move — lasso credibility with the teacher as complement.** This is the killer-app feature and the cleanest open answer to Akur8's product:

```python
student = rs.glm_dict(
    **recipe.to_glm_dict_kwargs(),
    data=data,
    complement="teacher_pred",   # Layered CatBoost prediction column
).fit(regularization="lasso", selection="1se")
```

The student is then a **deviation from the teacher**. Lasso shrinks small deviations to zero, giving "use the teacher where data is sparse; deviate where data supports it." Mathematically principled (this *is* what credibility theory says to do; it's also what the CAS Monograph 13 "lasso credibility" framework does). Already supported in [[rustystats]] v0.7.0.

---

## 8. Phase 5 — Compare, trace, bundle

### 8.1 Student-vs-Teacher gap

```python
diag = student.diagnostics(
    data=data,
    base_predictions="teacher_pred",
)
diag.to_json()
```

The `base_predictions_comparison` section tells us the predictive cost of distilling the teacher down to a GLM. Heuristics:

- `loss_improvement_pct` near 0% → the student is as good as the teacher. Ship it.
- < 1% worsening → small loss, acceptable for filable rate manuals.
- 1–5% worsening → revisit Phase 3 knobs (more interactions, more knots, drop heredity).
- > 5% worsening → the GBM has structure the GLM can't capture. Consider deploying the teacher with SHAP explanations instead (or drop destyler — this isn't the right tool for this dataset).

### 8.2 Trace explainability with per-layer origin

```python
trace = student.predict_contributions(new_data)
```

destyler maintains an `origin_map: dict[str, str]` mapping each rustystats term name to a distillation layer (`"layer_1"`, `"layer_2"`, `"layer_3"`, `"layer_2_promoted"`) inside the `Recipe`. After calling `student.predict_contributions(data)`, destyler **joins** the origin tag onto each contribution row by term name:

```python
def annotated_trace(self, data: pl.DataFrame) -> list[dict]:
    rows = self.student.predict_contributions(data)
    for row in rows:
        for contrib in row["contributions"]:
            contrib["origin"] = self.recipe.origin_map.get(contrib["term"], "intercept")
    return rows
```

[[rustystats]]'s contributing guide commits to stable term naming under its dict-first API, so this join is reliable. Aggregating by `origin` gives a per-layer breakdown alongside the per-term one. No rustystats changes required.

### 8.3 Bundle artefact: `.dst`

A destyler artefact is a bundle, not a single file:

```
model.dst/
├── manifest.json              # spec version, family, layers, features, git_sha
├── teacher/
│   ├── layer_1.cbm            # CatBoost binary
│   ├── layer_2.cbm
│   └── layer_3.cbm
├── student.rsglm              # rustystats GLMModel.to_bytes()
├── recipe.json                # Recipe (serialised dataclass)
├── attribution.json           # LayeredAttribution (compact form)
└── report.html                # Self-contained model card (zero external deps)
```

**For production scoring, only `student.rsglm` is the live model.** The bundle is for auditability, re-distillation, and the model card. [[haute]]'s deploy pipeline ships only `student.rsglm`; the `.dst` bundle lives in `models/` or Git LFS.

### 8.4 PMML / ONNX export

Inherited verbatim from [[rustystats]]. The student is a standard GLM; nothing destyler-specific to serialise.

---

## 9. RustyStats upstream contribution destyler needs

Only **one** new helper needs to land upstream in [[rustystats]]. Everything else destyler can do for itself, and arguably should — the cleaner separation of concerns is *rustystats computes the GLM, destyler orchestrates the workflow*. See [§9.2](#92-what-we-considered-upstream-and-rejected) for the items that were initially proposed for rustystats and on reflection don't belong there.

### 9.1 `working_response_weights`

**File:** new `python/rustystats/residuals.py` + PyO3 binding around existing `compute_irls_weights` / `initialize_mu_safe` in `crates/rustystats-core/src/solvers/mod.rs`.

```python
def working_response_weights(
    y: np.ndarray,
    eta: np.ndarray,
    family: str,
    link: str = "default",
    offset: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute IRLS working response z and working weight w given current η.

    z = η + (y − μ) · g'(μ)
    w = (g'(μ))⁻² / V(μ) · prior_weight

    where g is the link, V is the variance function.
    """
```

**Why destyler needs it.** Phase 1's layered residual-chained boosting needs to compute working residuals on the link scale given an arbitrary current η (not a fitted `GLMModel`'s η). For each family/link combination, `z` and `w` have well-known closed forms (Poisson + log: `z = log μ̂ + (y − μ̂)/μ̂`, `w = μ̂`; Gamma + log: similar; Tweedie with arbitrary `p`: `w = μ̂^(2−p)`; etc.).

**Why upstream and not in destyler.** Two reasons.

1. **The Rust logic already exists** in [[rustystats]]'s internal IRLS implementation. We're just exposing it. ~50 lines of PyO3 binding around code that's already shipping.
2. **The alternative is destyler re-implementing IRLS arithmetic in Python.** That's tractable for the common families (Poisson/Gamma/Gaussian/Binomial/Tweedie) but adds drift risk: if rustystats adds a family later (e.g., zero-inflated Poisson, ordinal), destyler silently misbehaves. Keeping the math in one place — where the IRLS solver also lives — eliminates that class of bug.

**Effort.** Small. ~50 lines including PyO3 binding, Python wrapper, type stub, and tests.

**Contingency.** If upstream is slow, destyler can ship a pure-Python implementation covering Poisson/Gamma/Tweedie/Gaussian/Binomial as a fallback (in `destyler/_residuals.py`), and migrate to the rustystats helper when v0.8.0 lands. destyler is **not blocked** on this.

### 9.2 What we considered upstream and rejected

The first draft of this spec proposed six other rustystats additions. On reflection each belongs in destyler (or doesn't need to exist at all):

| Considered | Decision | Why |
|---|---|---|
| `propose_knots_from_curve` (spline-knot heuristic) | **destyler-internal** | This is a heuristic about *how to use* splines, not spline math. Lives in `destyler/_knots.py`. Output is just a `knots=[...]` array passed to the existing `glm_dict` API. |
| `categorical` `grouping` parameter on terms | **destyler pre-aggregates the column** | Cleaner: destyler does `pl.col("Brand").replace_strict(cluster_map)`, the grouping is visible in the data column, `predict_contributions` correctly shows the grouped level, audit trail lives in the recipe's `grouping_maps` field. See [§6.4](#64-categorical-grouping-for-mid-cardinality-cats). |
| `suggest_grouping` helper | **destyler-internal** | ~5 lines of `scipy.cluster.hierarchy.linkage` + `fcluster`. No rustystats internals involved. |
| `validate_proposal` static check | **wrap existing `validate_glm_inputs`** | rustystats already lints terms dicts via `validate_glm_inputs` and `GLMModel.validate()`. destyler-specific checks (heredity, sparse cells) belong in destyler. |
| `filter_by_heredity` helper | **destyler-internal** | 30 lines of dict munging with zero rustystats internals. Putting it in rustystats just claims territory. |
| Per-layer `origin` tag in `predict_contributions` | **destyler joins post-hoc** | destyler maintains its own `recipe.origin_map: dict[term_name, layer]` and joins to the contribution rows after `student.predict_contributions(data)` returns. rustystats's dict-first term naming is stable, so the join is reliable. See [§8.2](#82-trace-explainability-with-per-layer-origin). |
| `rustystats.datasets` (freMTPL2, Belgian MTPL loaders) | **destyler bundles its own** | Benchmarks and docs ship in destyler; can vendor a compressed parquet in the wheel or use `requests`-based fetchers. No rustystats dependency. |

**Net effect.** rustystats v0.8.0 (or even a 0.7.x patch release) only needs `working_response_weights`. ~half a day of upstream work, not a 5-day release theme. destyler **stops being blocked on a rustystats release** — it can ship as soon as the one helper lands, or sooner with its own pure-Python fallback.

**Architectural rationale.** Keeping the surface area small in rustystats matches its existing invariant ([[rustystats]] §13: *Rust computes, Python orchestrates; one code path; minimal deps*). destyler is the orchestrator on top — it can be opinionated, can iterate fast, and doesn't need to live in the rustystats release cycle.

---

## 10. Python API design

### 10.1 Top-level functional API

```python
import polars as pl
import destyler as dst

df = pl.read_parquet("freMTPL2freq.parquet")

result = dst.distill(
    data=df,
    response="ClaimNb",
    family="poisson",
    offset="Exposure",
    cat_features=["VehBrand", "VehGas", "Region", "Area"],
    monotone_constraints={"BonusMalus": +1, "DrivAge": 0},
    layers=3,
    top_k_main=15,
    top_2way=10,
    top_3way=5,
    heredity=True,
    seed=42,
)

# result is a DistillationResult
result.student            # rustystats.GLMModel — the deployable
result.teacher            # LayeredTeacher
result.recipe             # Recipe (what got distilled)
result.attribution        # LayeredAttribution
result.report             # DistillationReport — HTML + JSON
result.save("model.dst")  # writes the bundle
```

### 10.2 Class-based API for fine control

```python
distiller = dst.Distiller(
    data=df,
    response="ClaimNb",
    family="poisson",
    offset="Exposure",
    cat_features=[...],
    monotone_constraints={...},
    seed=42,
)

teacher = distiller.train_teacher(layers=3)        # Phase 1
attribution = distiller.attribute()                 # Phase 2
recipe = distiller.propose(top_k_main=15, ...)      # Phase 3 — inspect before fit
print(recipe.summary())                             # human-readable

# Inspect and modify the recipe if needed
recipe.add_main("BonusMalus", {"type": "linear", "monotonicity": "increasing"})
recipe.drop_interaction(("VehGas", "Region"))

student = distiller.fit_student(                    # Phase 4
    recipe=recipe,
    regularization="elastic_net",
    cv=5,
    complement="teacher_pred",                      # optional Akur8-style
)
diag = distiller.diagnose(student)                  # Phase 5

distiller.save("model.dst")
```

### 10.3 Loading

```python
result = dst.DistillationResult.load("model.dst")
result.student.predict(new_data)
result.student.predict_contributions(new_data)

# Redistill with different settings — reuses cached teacher + attribution
new_result = result.redistill(top_k_main=20, heredity=False)
```

### 10.4 [[haute]] integration

In `src/haute/modelling/_algorithms.py`:

```python
class DistilledGLMAlgorithm(Algorithm):
    """Two-stage: Layered CatBoost teacher distilled into a rustystats GLM student."""

    name: ClassVar[str] = "distilled_glm"

    def fit(self, config, X, y, sample_weight=None, offset=None) -> FitResult:
        result = dst.distill(
            data=X.with_columns(pl.Series(config.response, y)),
            response=config.response,
            family=config.family,
            offset=config.offset_column,
            cat_features=config.cat_features,
            monotone_constraints=config.monotone_constraints,
            **config.distill_kwargs,
        )
        return FitResult(model=result, metadata={"recipe": result.recipe.to_dict()})

    def predict(self, model, X):       return model.student.predict(X)
    def save(self, model, path):       model.save(path)
    def load(self, path):              return dst.DistillationResult.load(path)
    def explain(self, model, X):
        return {
            "predict_contributions": model.student.predict_contributions(X),
            "attribution": model.attribution.to_dict(),
            "recipe": model.recipe.to_dict(),
        }

ALGORITHM_REGISTRY.register("distilled_glm", DistilledGLMAlgorithm())
```

`haute.toml` schema:

```toml
[model]
algorithm = "distilled_glm"
response = "ClaimCount"
family = "poisson"
offset = "Exposure"

[model.distill]
layers = 3
top_k_main = 15
top_2way = 10
top_3way = 5
heredity = true
promote_missing_main = true

[model.fit]
regularization = "elastic_net"
cv = 5
selection = "1se"
complement = "teacher_pred"   # optional
```

---

## 11. IP and prior-art design choices

destyler is designed to stay outside the WTW Layered GBM patent (US 11,853,906) and the SAS family (US 10,977,737 + continuations). The choices below are deliberate.

| WTW / SAS feature | destyler's equivalent | Rationale |
|---|---|---|
| Depth-progressive residual-stacked tree training | Layered CatBoost with the same depth schedule | The depth-progressive idea is the patent's narrow claim. **Needs attorney review.** Defence: CatBoost's symmetric/oblivious trees are a structurally distinct ensemble vs WTW's heterogeneous trees. **Fall-back**: switch Phase 1 to EBM-style round-robin training (Caruana et al. 2012, MIT-licensed, decade older than WTW's patent) — leaves Phases 2-5 intact. |
| GBM trees → indicator variables → GLM rebuild | TreeSHAP attribution → spline-knot + cat-grouping proposal → rustystats GLM fit | **Path is different.** destyler does not extract indicator variables from trees and refit a GLM on the indicators (SAS's claim 1). destyler extracts *structural information* (which variables, which interactions, which knots) and fits a clean independent GLM on the original data. |
| GLM auto-build with grouped parameters, polynomial fits, splines | Same workflow, **implemented per Lindholm/Lindskog/Palmquist (2024)** | Lindholm's "black-box guided GLM building" is published in *Annals of Actuarial Science* and is the prior-art shield. Cite it in the destyler README. |
| Heredity constraint | GAMI-Net (Yang/Zhang/Sudjianto 2020) | Implementing Chipman 1996's heredity is well-established. |
| "Machine-led GLM" branding | **"GLM distillation"** — descriptive, accurate, technically grounded in Hinton 2015 | "Machine-led GLM" is WTW's trade dress; not legally protected but avoid for clarity. "Distillation" is established ML terminology with no IP encumbrance. |

**Bottom line:** the spec is implementable, but **get patent counsel sign-off before publishing destyler.** Specifically on whether the depth-progressive CatBoost layer schedule infringes WTW US 11,853,906 claim 1. The EBM-round-robin fall-back is the contingency.

---

## 12. Edge cases and gotchas

### 12.1 Multicollinearity dominates Layer 1
If two correlated features both fit Layer 1, gain splits arbitrarily between them. Mitigation: destyler reports VIF (via rustystats diagnostics); caps top_k_main; lets elastic-net handle residual collinearity in Phase 4.

### 12.2 Sparse interaction cells
A 2-way `Region × Brand` interaction might have cells with < 100 exposure. Mitigation: post-fit credibility check — if any cell has effective sample size < `N_min` (default 100), auto-convert to a TE interaction (`target_encoding: True`) which provides credibility-weighted shrinkage natively.

### 12.3 Tweedie boundary cases
`p ≤ 1` or `p ≥ 2` fail CatBoost's Tweedie loss. Refuse with a clear error; suggest Poisson (p→1) or Gamma (p→2).

### 12.4 Heredity violation in 3-way interactions
A strong 3-way might surface where one of the three components has 0% main-effect gain. Default: promote it to a main effect (CAS paper Use Case 2). Surfaced prominently in `recipe.promoted_main_effects`.

### 12.5 Unseen levels at scoring time
[[rustystats]] falls back to the TE prior for unseen levels; the student inherits this. For non-TE categoricals where destyler grouped levels in Phase 3, the scoring path applies `pl.col("Brand").replace_strict(cluster_map, default="other_cluster")` — unseen levels map to a designated "other" cluster whose coefficient is the size-weighted mean of the population. Configurable via `Distiller(unseen_level_strategy="other_cluster" | "global_mean" | "raise")`.

### 12.6 Offset / exposure mismatch
Phase 1 working-residual computation embeds the offset; Phase 4 GLM takes the same offset. **Both must use the same column.** Validated in `Distiller.__init__`; fail loud.

### 12.7 Determinism with CatBoost
CatBoost is deterministic given a fixed `random_seed` and `thread_count=1` (multi-threading introduces non-determinism in some operations). Default `thread_count=1`; allow `thread_count=N` with explicit warning that determinism is sacrificed.

### 12.8 Memory blow-up on TreeSHAP interaction values
For n=10M, p=100, depth=2, SHAP interaction values are `O(n · p²)` ≈ 800 GB. Mitigation: chunk the SHAP computation; or sample 100K rows for interaction discovery (statistically equivalent for ranking purposes).

### 12.9 Lasso credibility with teacher complement — interpretability footnote
When `complement="teacher_pred"` is used, the student's coefficients are **deviations** from the teacher, not multiplicative relativities. The student's predictions are still correct, but the coefficient table is no longer a standalone rate manual — it's a correction-factor table to be applied on top of the teacher. destyler surfaces this clearly in the model card. Use only when the teacher is itself going into production (or accept the interpretability cost).

### 12.10 Regulatory filing memo
The output looks like a standard GLM, but the *story* — "we used a GBM to propose structure" — needs to be explainable in the filing memo. destyler ships a template paragraph in the model card.

---

## 13. Implementation plan

| Phase | Deliverable | Est. effort | Depends on |
|---|---|---|---|
| **P-1** (parallel) | rustystats `working_response_weights` upstream PR ([§9.1](#91-working_response_weights)) | 0.5 d | — |
| **P0** | destyler package skeleton (CI, tests, docs, ruff, mypy, mkdocs); `_residuals.py` fallback | 1 d | — |
| **P1** | `LayeredTeacher` + integration tests | 2 d | P0 (uses fallback or P-1 once it lands) |
| **P2** | `AttributionExtractor` (SHAP + path co-occurrence) | 2 d | P1 |
| **P3** | `Recipe` — main effects + `_knots.py` + categorical pre-aggregation helper | 3 d | P2 |
| **P4** | Interaction proposal with heredity (destyler-internal helper) | 1 d | P3 |
| **P5** | `Distiller` top-level + `distill()` + `Recipe.origin_map` + `.dst` bundle | 2 d | P1–P4 |
| **P6** | Trace post-hoc origin join + diagnostics integration | 0.5 d | P5 |
| **P7** | Synthetic benchmark suite (CAS Process 1 / Process 2) | 1 d | P5 |
| **P8** | Real benchmarks (freMTPL2, Belgian MTPL) vs EBM, vs pure GLM, vs pure CatBoost | 2 d | P7 |
| **P9** | Haute algorithm registration + integration test | 1 d | P5, haute |
| **P10** | mkdocs site, README, blog post | 2 d | P8 |
| **TOTAL** | | **~18 days** (17.5 destyler + 0.5 upstream, in parallel) | |

**P-1 runs in parallel** with destyler development — destyler ships its pure-Python `_residuals.py` fallback first, then swaps to the rustystats helper once it lands. No serial dependency.

---

## 14. Testing strategy

### 14.1 Unit tests

- `LayeredTeacher`: deterministic training; recovers main effects on Process-1 synthetic data; recovers interactions on Process-2 synthetic.
- `AttributionExtractor`: SHAP interaction values match TreeSHAP reference; path co-occurrence matches brute-force traversal on small trees.
- `Recipe`: heredity filter is correct; spline-knot proposal matches reference on synthetic curves.
- `Distiller`: round-trip save/load; bundle integrity; deterministic re-fit.

### 14.2 Integration tests

- **Process 1 / Process 2** (CAS paper synthetic with known interactions). Distillation must surface seeded interactions; pure-GLM proposal must not (control).
- **freMTPL2 (French motor TPL frequency):** baseline against pure rustystats GLM with hand-crafted terms. Target: destyler matches hand-crafted within 1% Poisson deviance.
- **Belgian MTPL (Henckaerts dataset):** same.
- **End-to-end [[haute]] pipeline:** smoke test that registers `distilled_glm`, fits, predicts, deploys to a local FastAPI service.

### 14.3 Regression tests

- vs EBM (`interpretml/interpret`): destyler within tolerance on Poisson deviance, with the trade-off that destyler outputs a filable GLM.
- vs pure CatBoost: destyler loses < 2% Poisson deviance, gains full interpretability.
- vs pure GLM (hand-crafted terms): destyler ≥ comparable, with automation.

### 14.4 Property tests

- Heredity invariant: every interaction in the final GLM has all components as main effects.
- Additivity: `predict_contributions` sums to η (validated by rustystats; check end-to-end).
- Determinism: two runs with same seed and `thread_count=1` produce byte-identical `.dst` bundles.

---

## 15. Performance targets

| Operation | n=100K, p=30 | n=1M, p=50 | n=10M, p=100 |
|---|---|---|---|
| Layer 1 fit (CatBoost depth 1) | < 5s | < 30s | < 5 min |
| Layer 2 fit | < 10s | < 1 min | < 10 min |
| Layer 3 fit | < 10s | < 1 min | < 10 min |
| SHAP interaction (full) | < 5s | < 1 min | (sample 100K) |
| Recipe proposal | < 1s | < 5s | < 30s |
| rustystats GLM fit (elastic-net, cv=5) | < 5s | < 1 min | < 10 min |
| **End-to-end** | **< 30s** | **< 5 min** | **< 30 min** |

---

## 16. Open questions

1. **Backend abstraction.** Defer to v0.2 (LightGBM, EBM as alternative teachers behind a `Teacher` protocol). v0.1 commits to CatBoost.
2. **Frequency-severity Tweedie vs two-stage.** Default Tweedie. Two-stage (separate frequency Poisson + severity Gamma, then product) deferred to v0.2.
3. **`.dst` manifest schema.** Needs spec'ing. At minimum: `spec_version`, `destyler_version`, `rustystats_version`, `catboost_version`, `family`, `link`, `n_features`, `n_layers`, `response`, `offset`, `weights`, `cat_features`, `monotone_constraints`, `seed`, `fit_timestamp`, `git_sha` (if available).
4. **License.** AGPL-3.0 to match [[rustystats]] and [[haute]]? Confirm before P0.
5. **Repo URL.** `github.com/PricingFrontier/destyler` — confirm before P0.
6. **Naming inside the library.** `Distiller` / `Recipe` / `Teacher` / `Student` — clean and consistent with the metaphor. Confirm before P0 (these are public-API names).
7. **Should `redistill()` be cheap (reuse teacher + attribution) or always retrain?** Default: cheap unless `force_retrain=True`. Bundle saves the attribution for exactly this reason.

---

## 17. Related work and sources

### Algorithms
- **Knowledge distillation:** Hinton, Vinyals, Dean (2015) "Distilling the Knowledge in a Neural Network." [arXiv](https://arxiv.org/abs/1503.02531). The conceptual namesake — though destyler's distillation works at the *structural* level (architecture proposal) rather than the *output* level (soft-label matching).
- **WTW Layered GBM:** McGrath & Milam, "Introduction to Layered GBM" (CAS RPM 2022). [PDF](https://www.casact.org/sites/default/files/2022-11/CS-22_Introduction_to_Layered_GBM.pdf). Full walkthrough in [[competitors/wtw-radar#24-the-layered-gbm-algorithm--ground-truth-from-the-cas-paper]].
- **Lindholm/Lindskog/Palmquist (2024) "Black-Box Guided GLM Building":** *Annals of Actuarial Science*. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4691626). **The prior-art shield for destyler's approach.**
- **EBM / GA²M:** Lou, Caruana, Gehrke, Hooker (2013). [KDD PDF](https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf). [interpretml/interpret](https://github.com/interpretml/interpret).
- **GAMI-Net:** Yang, Zhang, Sudjianto (2020). [arXiv](https://arxiv.org/abs/2003.07132). Heredity constraint origin.
- **Henckaerts et al. (2021) "Boosting Insights in Insurance Tariff Plans":** [arXiv](https://arxiv.org/abs/1904.10890).

### Libraries
- [[rustystats]] — full reference at `pf-suite/rustystats.md`.
- [[haute]] — `pf-suite/haute.md`.
- [CatBoost docs](https://catboost.ai/docs/).
- [interpretml/interpret](https://github.com/interpretml/interpret).

### Patents (see [[competitors/wtw-radar#25-patent-landscape--verified-resolution]])
- US 11,853,906 (WTW Layered GBM).
- US 10,977,737, US 12,260,460, US 12,380,510 (SAS Institute, parallel family).

### Competitive context
- [[competitors/wtw-radar]] — full WTW Radar dossier including the case for destyler's commercial positioning.
- Akur8 "Derivative Lasso" — competitor's GLM-up approach. destyler differentiates on the interaction-discovery axis.
- hyperexponential — competitor's Python-native approach, no automatic-GLM-build pitch.

---

## 18. Worked example

```python
import polars as pl
import destyler as dst
from rustystats.datasets import load_fremtpl2

df = load_fremtpl2()

result = dst.distill(
    data=df,
    response="ClaimNb",
    family="poisson",
    offset="Exposure",
    cat_features=["VehBrand", "VehGas", "Region", "Area"],
    monotone_constraints={"BonusMalus": +1},
    layers=3,
    seed=42,
)

print(result.recipe.summary())
# Distilled 12 main effects, 7 interactions (5 promoted via heredity)
# Top main effects by gain:
#   DrivAge          (24% main / 11% interaction)
#   VehAge           (18% main /  8% interaction)
#   BonusMalus       (15% main /  3% interaction)
#   ...
# Top 2-way interactions:
#   VehBrand × Region    (4.2% pair gain)
#   DrivAge × VehGas     (2.1%)
#   ...

print(result.student.summary())
# rustystats GLM summary, elastic-net selected alpha = 0.013
# Poisson deviance: train 0.327 / test 0.331
# A/E (test): 1.001

result.diagnose()
# Student vs Teacher gap: 0.8% Poisson deviance worsening — ship it.

result.save("model.dst")
result.student.to_pmml("model.pmml")
result.student.to_onnx("model.onnx", mode="full")

# Audit-trail traces
trace = result.student.predict_contributions(df.head(100))
# trace[0]["contributions"] each carry an `origin` ∈ {layer_1, layer_2, layer_3, layer_2_promoted}
```

---

> **Status:** draft v0.1. Next steps: confirm naming and licence; review with Nick; legal review of [§11](#11-ip-and-prior-art-design-choices); open `PricingFrontier/destyler` repo and a single `working_response_weights` issue against `PricingFrontier/rustystats`; start P0 (destyler skeleton) and P-1 (upstream PR) in parallel.
