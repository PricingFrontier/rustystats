# Working Response & Weights API Reference

`working_response_weights` exposes the IRLS working response **z** and combined
working weight **w** for an arbitrary linear predictor η — without requiring a
fitted GLM. It is the helper that lets external tooling (link-scale boosting
loops, custom iterative reweighters, distillation workflows) reproduce the
local quadratic approximation the IRLS solver uses internally.

If you are fitting a GLM directly, use [`glm_dict`](dict-api.md) instead. This
helper is for code that needs to step IRLS by hand.

## working_response_weights

```python
rustystats.working_response_weights(
    y,
    eta,
    family,
    link="default",
    *,
    offset=None,
    weights=None,
    var_power=1.5,
    theta=1.0,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | array-like, shape `(n,)` | required | Observed response. Family-specific bounds are validated (e.g. Gamma requires `y > 0`). |
| `eta` | array-like, shape `(n,)` | required | Current linear predictor **excluding** offset. See [Offset semantics](#offset-semantics). |
| `family` | str | required | One of `"gaussian"`, `"poisson"`, `"binomial"`, `"gamma"`, `"tweedie"`, `"quasipoisson"`, `"quasibinomial"`, `"negbinomial"`. Embedded parameters are accepted (`"tweedie(p=1.5)"`, `"negativebinomial(theta=2.0)"`). |
| `link` | str | `"default"` | One of `"identity"`, `"log"`, `"logit"`. The literal `"default"` (or `None`) resolves to the family's canonical link. |
| `offset` | array-like, shape `(n,)` | `None` | Offset on the linear predictor scale (e.g. `log(exposure)`). Added to `eta` when computing μ. Defaults to zeros. |
| `weights` | array-like, shape `(n,)` | `None` | Non-negative prior weights. Defaults to ones. |
| `var_power` | float | `1.5` | Tweedie variance power. Ignored for non-Tweedie families. |
| `theta` | float | `1.0` | Negative Binomial dispersion. Ignored for non-NB families. |

### Returns

A `(z, w)` tuple of `numpy.ndarray[float64]` arrays of length `n`.

- `z` — working response, `η + (y − μ) · g'(μ)`
- `w` — combined working weight, `prior_weight × IRLS_weight`

---

## Mathematical contract

For each row,

$$
\mu = g^{-1}(\eta + \text{offset})
$$

$$
z = \eta + (y - \mu)\, g'(\mu)
$$

$$
w = w_\text{prior} \times w_\text{IRLS}
$$

The IRLS weight `w_IRLS` is one of two forms, chosen automatically to match
[the IRLS solver](../theory/irls.md):

| Branch | Used when | Weight |
|---|---|---|
| **Fisher information** | Default | `1 / (V(μ) · g'(μ)²)` |
| **True Hessian** | Tweedie + log link with `1 < p < 2` | `μ^(2 − p)` |

Per-family weight at the canonical link (with `prior_weight = 1`):

| Family | Link | Weight |
|---|---|---|
| Gaussian | identity | `1` |
| Poisson | log | `μ` |
| Gamma | log | `1` (Fisher; matches statsmodels) |
| Tweedie | log | `μ^(2−p)` (true Hessian, `1 < p < 2`) |
| Binomial | logit | `μ(1 − μ)` |
| Negative Binomial | log | `μθ / (θ + μ)` |
| QuasiPoisson | log | `μ` (same as Poisson) |
| QuasiBinomial | logit | `μ(1 − μ)` (same as Binomial) |

For derivation, see [The IRLS Algorithm](../theory/irls.md).

---

## Offset semantics

`eta` is the linear predictor **excluding** the offset — that is, the running
model prediction on the link scale. The offset is added internally to form μ.

This matches the convention of a boosting loop where η is rebuilt each
iteration from new trees and the offset (e.g. log-exposure) is held fixed:

```python
eta = np.full_like(y, fill_value=initial_eta, dtype=np.float64)
log_exposure = np.log(exposure)

for layer in range(n_layers):
    z, w = rs.working_response_weights(
        y, eta, family="poisson", offset=log_exposure
    )
    tree = fit_tree(X, z, sample_weight=w)   # any weighted regressor
    eta = eta + tree.predict(X)              # accumulate on link scale
```

The full linear predictor used to compute μ is `eta + offset`. If you have
already added the offset into `eta`, pass `offset=None` (or zeros) — the
formula still holds, you have just shifted what "η" means.

---

## Worked example

Poisson with a perfect-fit row to show the structure:

```python
import numpy as np
import rustystats as rs

y = np.array([0.0, 1.0, 2.0])
eta = np.zeros(3)                        # μ = exp(0) = 1
z, w = rs.working_response_weights(y, eta, family="poisson")

print(z)   # [-1.0  0.0  1.0]   = eta + (y − μ)/μ
print(w)   # [ 1.0  1.0  1.0]   = μ (Fisher: w = μ for Poisson + log)
```

Tweedie with embedded variance power:

```python
z, w = rs.working_response_weights(
    y, eta,
    family="tweedie(p=1.5)",    # equivalent to family="tweedie", var_power=1.5
    offset=np.log(exposure),
    weights=prior_weights,
)
```

---

## Notes

**Consistency with the IRLS solver.** At a fitted model's converged η, the
`(z, w)` returned here equal what the solver would compute on its next
iteration. Performing one more weighted least-squares step on `(z, w)`
reproduces the converged coefficient vector to within IRLS tolerance.

**Determinism.** The per-row computation has no cross-row state, so the
output is identical regardless of how the underlying parallel implementation
schedules work.

**Why Gamma uses Fisher and Tweedie uses the true Hessian.** Both are choices
in the underlying solver. Gamma + log uses Fisher (`w = 1`) so its covariance
matrix matches statsmodels. Tweedie + log with `1 < p < 2` uses the true
Hessian (`w = μ^(2 − p)`) because it reduces IRLS iterations significantly
without changing the optimum. This helper mirrors the solver, so any
downstream computation built on `(z, w)` stays consistent.

---

## See also

- [`glm_dict`](dict-api.md) — for fitting a GLM directly.
- [The IRLS Algorithm](../theory/irls.md) — derivation of the working
  response and weight.
- [Link Functions](../theory/links.md) — link derivatives `g'(μ)` and how
  they enter the formulas.
- [Distribution Families](../theory/families.md) — variance functions `V(μ)`.
