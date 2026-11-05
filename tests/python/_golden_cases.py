"""Golden fit cases: the v0.9.0 reference the out-of-core rework is checked against.

``specs/out-of-core-fit.md`` replaces the design build, every solver loop and the row
reductions. Before any of that changes, ``scripts/capture_golden_fits.py`` records what
v0.9.0 reports for each case below; ``test_golden_fits.py`` compares the current code
against that record. The cases cover every family, term kind, solver route and solver
branch the rework touches.

Values are recorded as float hex strings so the record is exact; the test decides the
tolerance per field (rounding-level changes are expected from the fixed reduction order,
material changes only where the spec decides them).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

GOLDEN_SCHEMA = 1

#: Result attributes recorded for every case, when the result exposes them.
RESULT_FIELDS: tuple[str, ...] = (
    "params",
    "feature_names",
    "bse",
    "tvalues",
    "pvalues",
    "conf_int",
    "deviance",
    "null_deviance",
    "scale",
    "pearson_chi2",
    "llf",
    "aic",
    "bic",
    "iterations",
    "converged",
    "solver_status",
    "step_halving_used",
    "inference_status",
    "nobs",
    "df_resid",
    "df_model",
    "alpha",
    "l1_ratio",
    "n_nonzero",
    "total_edf",
    "gcv",
    "cv_deviance",
    "cv_deviance_se",
    "n_cv_folds",
    "cv_selection_method",
    "regularization_path",
    "response_scale",
    "family",
)
ROBUST_TYPES: tuple[str, ...] = ("HC0", "HC1", "HC2", "HC3")


def design_frame(n: int, seed: int) -> pl.DataFrame:
    """A pricing-shaped frame with every column kind the cases need."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.uniform(18.0, 80.0, n)
    x3 = np.exp(0.5 * rng.normal(0.0, 1.0, n))
    x4 = np.round(rng.normal(0.0, 1.0, n), 1)
    cat_a = rng.choice(["A", "B", "C", "D"], size=n)
    cat_b = rng.choice([f"b{i}" for i in range(8)], size=n)
    brand = rng.choice([f"brand{i:02d}" for i in range(30)], size=n)
    cat_int = rng.integers(1, 5, n).astype(float)
    cat_int_null = rng.random(n) < 0.05
    flag = rng.random(n) < 0.4
    exposure = rng.uniform(0.1, 1.0, n)
    weight = rng.uniform(0.5, 2.0, n)
    offset = rng.normal(0.0, 0.1, n)

    effect_a = np.select([cat_a == "A", cat_a == "B", cat_a == "C"], [0.3, 0.1, -0.1], 0.0)
    effect_b = np.array([int(v[1:]) for v in cat_b], dtype=float) * 0.03
    effect_brand = np.array([int(v[5:]) for v in brand], dtype=float) * 0.01
    eta = (
        -2.0
        + 0.2 * x1
        + 0.01 * (x2 - 45.0)
        - 0.1 * np.log(x3)
        + 0.05 * x4
        + effect_a
        + effect_b
        + effect_brand
        + 0.15 * flag
    )
    mu = np.exp(eta)
    y_pois = rng.poisson(mu * exposure).astype(float)
    y_gamma = rng.gamma(2.0, 100.0 * mu / 2.0)
    counts = rng.poisson(mu * exposure * 3.0)
    y_tweedie = np.array(
        [rng.gamma(1.5, 50.0, k).sum() if k > 0 else 0.0 for k in counts], dtype=float
    )
    y_bin = (rng.random(n) < 1.0 / (1.0 + np.exp(-(eta + 1.5)))).astype(float)
    theta = 1.5
    p_nb = theta / (theta + mu * 4.0)
    y_nb = rng.negative_binomial(theta, p_nb).astype(float)
    y_gauss = 10.0 + 2.0 * x1 + 0.05 * x2 + effect_a * 5.0 + rng.normal(0.0, 1.0, n)
    comp = mu * 0.9  # a response-scale complement (a prior rate) for log-link fits

    return pl.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "cat_a": cat_a,
            "cat_b": cat_b,
            "brand": brand,
            "cat_int": pl.Series(
                "cat_int",
                [None if missing else int(v) for v, missing in zip(cat_int, cat_int_null)],
                dtype=pl.Int64,
            ),
            "flag": flag,
            "exposure": exposure,
            "weight": weight,
            "offset": offset,
            "comp": comp,
            "y_pois": y_pois,
            "y_gamma": y_gamma,
            "y_tweedie": y_tweedie,
            "y_bin": y_bin,
            "y_nb": y_nb,
            "y_gauss": y_gauss,
        }
    )


@dataclass(frozen=True)
class GoldenCase:
    """One fit: how to build it, how to fit it, and what to record."""

    name: str
    build: Callable[[], tuple[dict[str, Any], dict[str, Any], pl.DataFrame | None]]
    robust: tuple[str, ...] = ()
    #: Spec decisions under which this case's values are expected to change materially.
    changes_under: tuple[str, ...] = field(default=())
    #: Field-class tolerance overrides (relative), e.g. for ill-conditioned designs.
    rtol_overrides: dict[str, float] = field(default_factory=dict)
    #: Compare only discrete fields and finiteness: for degenerate fixtures whose
    #: floating-point values v0.9.0 itself does not reproduce across thread counts.
    discrete_only: bool = False


def _frame() -> pl.DataFrame:
    return design_frame(3000, seed=20261105)


def _holdout() -> pl.DataFrame:
    return design_frame(500, seed=20261106)


def _case(
    response: str,
    terms: dict[str, Any],
    family: str,
    *,
    fit: dict[str, Any] | None = None,
    data: Callable[[], pl.DataFrame] = _frame,
    predict: bool = True,
    **glm: Any,
) -> Callable[[], tuple[dict[str, Any], dict[str, Any], pl.DataFrame | None]]:
    def build() -> tuple[dict[str, Any], dict[str, Any], pl.DataFrame | None]:
        kwargs = {"response": response, "terms": terms, "data": data(), "family": family, **glm}
        return kwargs, dict(fit or {}), (_holdout() if predict else None)

    return build


LINEAR_CAT = {
    "x1": {"type": "linear"},
    "x2": {"type": "linear"},
    "x3": {"type": "linear"},
    "x4": {"type": "linear"},
    "cat_a": {"type": "categorical"},
    "cat_b": {"type": "categorical"},
}


def _halving_first_iteration() -> tuple[dict[str, Any], dict[str, Any], None]:
    rng = np.random.default_rng(123)
    y = rng.poisson(np.exp(rng.normal(0.0, 2.0, 50))).astype(float)
    x = rng.normal(0.0, 10.0, 50)
    data = pl.DataFrame({"y": y, "x": x})
    return (
        {"response": "y", "terms": {"x": {"type": "linear"}}, "data": data, "family": "poisson"},
        {"max_iter": 1},
        None,
    )


def _halving_gamma_overshoot() -> tuple[dict[str, Any], dict[str, Any], None]:
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 200)
    mu = np.exp(0.3 + 3.0 * x)
    y = rng.gamma(0.3, mu / 0.3)
    data = pl.DataFrame({"y": y, "x": x})
    return (
        {"response": "y", "terms": {"x": {"type": "linear"}}, "data": data, "family": "gamma"},
        {"max_iter": 100},
        None,
    )


def _halving_exhausted() -> tuple[dict[str, Any], dict[str, Any], None]:
    x = np.array([0.0, 1.0, 2.0, 1.0e6, 1.0e6, 1.0e6])
    y = np.array([1.0, 1.0, 1.0, 1.0e300, 1.0e300, 1.0e300])
    data = pl.DataFrame({"y": y, "x": x})
    return (
        {"response": "y", "terms": {"x": {"type": "linear"}}, "data": data, "family": "poisson"},
        {"max_iter": 5},
        None,
    )


def _separation() -> tuple[dict[str, Any], dict[str, Any], None]:
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 200)
    y = (x > 0).astype(float)
    data = pl.DataFrame({"y": y, "x": x})
    return (
        {"response": "y", "terms": {"x": {"type": "linear"}}, "data": data, "family": "binomial"},
        {"max_iter": 50},
        None,
    )


def _ill_conditioned() -> tuple[dict[str, Any], dict[str, Any], pl.DataFrame]:
    def with_twin(frame: pl.DataFrame, seed: int) -> pl.DataFrame:
        rng = np.random.default_rng(seed)
        twin = frame["x1"].to_numpy() + rng.normal(0.0, 1e-5, frame.height)
        return frame.with_columns(pl.Series("x1_twin", twin))

    data = with_twin(_frame(), 1)
    holdout = with_twin(_holdout(), 2)
    terms = {"x1": {"type": "linear"}, "x1_twin": {"type": "linear"}, "x2": {"type": "linear"}}
    return (
        {"response": "y_gauss", "terms": terms, "data": data, "family": "gaussian"},
        {},
        holdout,
    )


def _monotone_linear_active() -> tuple[dict[str, Any], dict[str, Any], pl.DataFrame]:
    # x1's true effect on y_gauss is positive; constraining it decreasing makes the
    # sign constraint bind, exercising projection and best-iterate tracking.
    terms = {
        "x1": {"type": "linear", "monotonicity": "decreasing"},
        "x2": {"type": "linear"},
        "cat_a": {"type": "categorical"},
    }
    return (
        {"response": "y_gauss", "terms": terms, "data": _frame(), "family": "gaussian"},
        {},
        _holdout(),
    )


CASES: tuple[GoldenCase, ...] = (
    GoldenCase(
        "gaussian_linear_cat_weights",
        _case("y_gauss", LINEAR_CAT, "gaussian", weights="weight"),
    ),
    GoldenCase(
        "poisson_linear_cat_exposure",
        _case("y_pois", LINEAR_CAT, "poisson", exposure="exposure"),
    ),
    GoldenCase(
        "poisson_offset_weights",
        _case("y_pois", LINEAR_CAT, "poisson", offset="offset", weights="weight"),
    ),
    GoldenCase(
        "quasipoisson_exposure",
        _case("y_pois", LINEAR_CAT, "quasipoisson", exposure="exposure"),
    ),
    GoldenCase(
        "gamma_splines",
        _case(
            "y_gamma",
            {
                "x2": {"type": "bs", "df": 5},
                "x3": {"type": "ns", "df": 4},
                "x1": {"type": "linear"},
                "cat_a": {"type": "categorical"},
            },
            "gamma",
        ),
    ),
    GoldenCase(
        "tweedie_interactions",
        _case(
            "y_tweedie",
            {"x1": {"type": "linear"}, "cat_a": {"type": "categorical"}},
            "tweedie",
            var_power=1.5,
            weights="weight",
            interactions=[
                {"cat_a": {"type": "categorical"}, "cat_b": {"type": "categorical"}},
                {"cat_a": {"type": "categorical"}, "x1": {"type": "linear"}, "include_main": False},
            ],
        ),
    ),
    GoldenCase(
        "binomial_spline_interaction_levels",
        _case(
            "y_bin",
            {
                "x2": {"type": "bs", "df": 4},
                "cat_b": {"type": "categorical", "levels": ["b1", "b3", "b5"]},
                "x1": {"type": "linear"},
            },
            "binomial",
            interactions=[
                {"x2": {"type": "bs", "df": 4}, "x1": {"type": "linear"}, "include_main": False}
            ],
        ),
    ),
    GoldenCase(
        "negbinomial_fixed_theta",
        _case("y_nb", LINEAR_CAT, "negbinomial", theta=1.5),
    ),
    GoldenCase(
        "frequency_encoding_expression",
        _case(
            "y_pois",
            {
                "brand": {"type": "frequency_encoding"},
                "x1_sq": {"type": "expression", "expr": "x1 ** 2"},
                "x1": {"type": "linear"},
            },
            "poisson",
            exposure="exposure",
        ),
    ),
    GoldenCase(
        "ms_spline",
        _case(
            "y_gamma",
            {"x2": {"type": "ms", "df": 5}, "x1": {"type": "linear"}},
            "gamma",
        ),
    ),
    GoldenCase(
        "categorical_nullable_int_and_bool",
        _case(
            "y_pois",
            {
                "cat_int": {"type": "categorical"},
                "flag": {"type": "categorical"},
                "x1": {"type": "linear"},
            },
            "poisson",
            exposure="exposure",
        ),
    ),
    GoldenCase(
        "ridge_fixed",
        _case("y_pois", LINEAR_CAT, "poisson", exposure="exposure", fit={"alpha": 0.1}),
    ),
    GoldenCase(
        "lasso_fixed",
        _case(
            "y_pois",
            LINEAR_CAT,
            "poisson",
            exposure="exposure",
            fit={"alpha": 0.002, "l1_ratio": 1.0},
        ),
    ),
    GoldenCase(
        "elastic_net_fixed",
        _case(
            "y_gauss",
            LINEAR_CAT,
            "gaussian",
            weights="weight",
            fit={"alpha": 0.01, "l1_ratio": 0.5},
        ),
    ),
    GoldenCase(
        "smooth_gcv",
        _case(
            "y_pois",
            {
                "x2": {"type": "bs", "k": 10},
                "x1": {"type": "linear"},
                "cat_a": {"type": "categorical"},
            },
            "poisson",
            exposure="exposure",
        ),
        rtol_overrides={"smooth": 1e-3},
    ),
    GoldenCase(
        "smooth_fixed_penalty",
        _case(
            "y_gauss",
            {"x2": {"type": "bs", "k": 10}, "x1": {"type": "linear"}},
            "gaussian",
            fit={"alpha": 1.0},
        ),
    ),
    GoldenCase(
        "monotone_fixed_df",
        _case(
            "y_pois",
            {"x2": {"type": "bs", "df": 5, "monotonicity": "increasing"}, "x1": {"type": "linear"}},
            "poisson",
            exposure="exposure",
        ),
    ),
    GoldenCase(
        "monotone_linear_active",
        _monotone_linear_active,
    ),
    GoldenCase(
        "monotone_smooth",
        _case(
            "y_pois",
            {"x2": {"type": "bs", "k": 8, "monotonicity": "increasing"}, "x1": {"type": "linear"}},
            "poisson",
            exposure="exposure",
        ),
        rtol_overrides={"smooth": 1e-3, "params": 1e-5},
    ),
    GoldenCase(
        "monotone_smooth_cat_interaction",
        _case(
            "y_pois",
            {"x1": {"type": "linear"}, "cat_a": {"type": "categorical"}},
            "poisson",
            exposure="exposure",
            interactions=[
                {
                    "x2": {"type": "bs", "monotonicity": "increasing"},
                    "cat_a": {"type": "categorical"},
                    "include_main": False,
                }
            ],
        ),
        rtol_overrides={"smooth": 1e-3, "params": 1e-5},
    ),
    GoldenCase(
        "cv_lasso_min",
        _case(
            "y_pois",
            LINEAR_CAT,
            "poisson",
            exposure="exposure",
            fit={"cv": 3, "regularization": "lasso", "n_alphas": 5, "cv_seed": 7},
        ),
        changes_under=("DEC-2",),
    ),
    GoldenCase(
        "cv_ridge_1se",
        _case(
            "y_gauss",
            LINEAR_CAT,
            "gaussian",
            weights="weight",
            fit={
                "cv": 3,
                "regularization": "ridge",
                "n_alphas": 5,
                "cv_seed": 7,
                "selection": "1se",
            },
        ),
        changes_under=("DEC-2",),
    ),
    GoldenCase(
        "cv_elastic_net",
        _case(
            "y_pois",
            LINEAR_CAT,
            "poisson",
            exposure="exposure",
            fit={
                "cv": 3,
                "regularization": "elastic_net",
                "l1_ratio": 0.5,
                "n_alphas": 5,
                "cv_seed": 11,
            },
        ),
        changes_under=("DEC-2",),
    ),
    GoldenCase(
        "target_encoding_main",
        _case(
            "y_pois",
            {"brand": {"type": "target_encoding"}, "x1": {"type": "linear"}},
            "poisson",
            exposure="exposure",
            seed=7,
        ),
        changes_under=("DEC-1",),
    ),
    GoldenCase(
        "target_encoding_interaction",
        _case(
            "y_pois",
            {"x1": {"type": "linear"}},
            "poisson",
            exposure="exposure",
            seed=7,
            interactions=[
                {
                    "brand": {"type": "categorical"},
                    "cat_a": {"type": "categorical"},
                    "target_encoding": True,
                }
            ],
        ),
        changes_under=("DEC-1",),
    ),
    GoldenCase(
        "target_encoding_cv",
        _case(
            "y_pois",
            {"brand": {"type": "target_encoding"}, "x1": {"type": "linear"}},
            "poisson",
            exposure="exposure",
            seed=7,
            fit={"cv": 3, "regularization": "ridge", "n_alphas": 3, "cv_seed": 7},
        ),
        changes_under=("DEC-1", "DEC-2"),
    ),
    GoldenCase(
        "negbinomial_theta_estimate",
        _case("y_nb", LINEAR_CAT, "negbinomial", theta="estimate"),
        # θ comes from a Brent search, so platform libm noise in the inputs and links
        # moves it (hence 1e-4); the statistics that depend on θ move with it. Windows
        # gives pearson_chi2 4.2e-9 relative from the record.
        rtol_overrides={"theta": 1e-4, "params": 1e-6, "statistic": 1e-7},
    ),
    GoldenCase(
        "complement_column",
        _case(
            "y_pois",
            {"x1": {"type": "linear"}, "cat_a": {"type": "categorical"}},
            "poisson",
            exposure="exposure",
            complement="comp",
        ),
    ),
    GoldenCase(
        "robust_hc_all",
        _case(
            "y_gamma",
            {"x1": {"type": "linear"}, "x2": {"type": "linear"}, "cat_a": {"type": "categorical"}},
            "gamma",
            weights="weight",
            fit={"store_design_matrix": True},
        ),
        robust=ROBUST_TYPES,
    ),
    GoldenCase(
        "no_covariance",
        _case(
            "y_pois",
            LINEAR_CAT,
            "poisson",
            exposure="exposure",
            fit={"compute_covariance": False},
        ),
    ),
    GoldenCase("halving_first_iteration", _halving_first_iteration),
    GoldenCase("halving_gamma_overshoot", _halving_gamma_overshoot),
    # Overflowing y (1e300) and x (1e6): v0.9.0's retained iterate moves by up to 47%
    # relative between thread counts, so only the status and finiteness are pinned.
    GoldenCase("halving_exhausted", _halving_exhausted, discrete_only=True),
    GoldenCase("binomial_separation", _separation, rtol_overrides={"params": 1e-4}),
    GoldenCase("ill_conditioned", _ill_conditioned, rtol_overrides={"params": 1e-4}),
)


# ── Encoding ────────────────────────────────────────────────────────────────


def encode(value: Any) -> Any:
    """JSON-safe, exact encoding of a recorded value (floats as hex)."""
    if isinstance(value, (bool, np.bool_)):
        return {"bool": bool(value)}
    if isinstance(value, (int, np.integer)):
        return {"int": int(value)}
    if isinstance(value, (float, np.floating)):
        return {"f64": float(value).hex()}
    if isinstance(value, str):
        return {"str": value}
    if value is None:
        return {"none": True}
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fc":
            flat = [float(v).hex() for v in np.asarray(value, dtype=np.float64).ravel()]
            return {"f64_array": flat, "shape": list(value.shape)}
        return {"list": [encode(v) for v in value.tolist()]}
    if isinstance(value, dict):
        return {"dict": {str(k): encode(v) for k, v in value.items()}}
    if isinstance(value, (list, tuple)):
        return {"list": [encode(v) for v in value]}
    raise TypeError(f"cannot encode {type(value).__name__}")


def decode(value: dict[str, Any]) -> Any:
    """Inverse of :func:`encode`."""
    (kind,) = value.keys() - {"shape"}
    payload = value[kind]
    if kind == "bool":
        return bool(payload)
    if kind == "int":
        return int(payload)
    if kind == "f64":
        return float.fromhex(payload)
    if kind == "str":
        return payload
    if kind == "none":
        return None
    if kind == "f64_array":
        return np.array([float.fromhex(v) for v in payload], dtype=np.float64).reshape(
            value["shape"]
        )
    if kind == "dict":
        return {k: decode(v) for k, v in payload.items()}
    if kind == "list":
        return [decode(v) for v in payload]
    raise ValueError(f"unknown encoding {kind!r}")


# ── Recording ───────────────────────────────────────────────────────────────


def _read(result: Any, name: str) -> Any:
    try:
        value = getattr(result, name)
        if callable(value):
            value = value()
    except Exception as exc:  # recorded: availability is part of the contract
        return {"unavailable": type(exc).__name__}
    if hasattr(value, "tolist") and not isinstance(value, np.ndarray):
        value = np.asarray(value)
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return [
            {k: (np.asarray(v) if isinstance(v, (list, tuple)) else v) for k, v in row.items()}
            for row in value
        ]
    if isinstance(value, list):
        return np.asarray(value) if all(isinstance(v, (int, float)) for v in value) else value
    return value


def record(case: GoldenCase) -> dict[str, Any]:
    """Fit one case and return its recorded fields (unencoded)."""
    import rustystats as rs

    glm_kwargs, fit_kwargs, holdout = case.build()
    result = rs.glm_dict(**glm_kwargs).fit(**fit_kwargs)
    fields: dict[str, Any] = {name: _read(result, name) for name in RESULT_FIELDS}
    family = fields.get("family")
    if isinstance(family, str) and family.lower().startswith("negativebinomial"):
        fields["theta"] = _theta_from_family(family)
    smooth = _read(result, "smooth_terms")
    if isinstance(smooth, list):
        fields["smooth_terms"] = [
            {
                "variable": str(term.variable),
                "k": int(term.k),
                "edf": float(term.edf),
                "lambda": float(term.lambda_),
            }
            for term in smooth
        ]
    for hc_type in case.robust:
        fields[f"bse_robust_{hc_type}"] = _call(result, "bse_robust", hc_type)
    if holdout is not None:
        fields["predict_holdout"] = _call(result, "predict", holdout)
    return fields


def _call(result: Any, name: str, *args: Any) -> Any:
    try:
        return np.asarray(getattr(result, name)(*args), dtype=np.float64)
    except Exception as exc:
        return {"unavailable": type(exc).__name__}


def _theta_from_family(family: str) -> float | None:
    start = family.find("theta=")
    if start < 0:
        return None
    digits = family[start + len("theta=") :].rstrip(")")
    try:
        return float(digits)
    except ValueError:
        return math.nan
