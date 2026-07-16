from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import rustystats as rs
from rustystats.exceptions import ValidationError
from rustystats.formula import _monotone_smooth_endpoint_constraints


def _monotone_interaction_frame(seed: int = 7, n: int = 800) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    g = rng.integers(0, 2, n)
    eta = 2.0 - 4.0 * x + (g == 1) * (3.5 * np.exp(-(((x - 0.75) / 0.10) ** 2)))
    y = eta + rng.normal(0.0, 0.15, n)
    return pl.DataFrame({"y": y, "x": x, "g": g.astype(str)})


def _monotone_interaction_spec(include_main: bool = True) -> list[dict]:
    return [
        {
            "x": {"type": "bs", "monotonicity": "decreasing"},
            "g": {"type": "categorical"},
            "include_main": include_main,
        }
    ]


def test_monotone_bs_categorical_interaction_is_monotone_by_level():
    model = rs.glm_dict(
        response="y",
        terms={},
        interactions=_monotone_interaction_spec(include_main=True),
        data=_monotone_interaction_frame(),
        family="gaussian",
        link="identity",
        seed=42,
    ).fit()

    assert model.converged
    assert model.stationary
    smooth_terms, smooth_ranges = model._builder.get_smooth_terms()
    # Main-effect smooth plus one OWN monotone curve per category (full-k,
    # category-major) — not k-1 sign-constrained deviations, which would force
    # every category to lie at/above the reference everywhere.
    assert smooth_ranges == [(2, 11), (11, 20), (20, 29)]
    assert [
        getattr(term, "_interaction_smooth_endpoint_constraint", False) for term in smooth_terms
    ] == [False, True, True]

    for group in ("0", "1"):
        grid = pl.DataFrame({"x": np.linspace(0.01, 0.99, 201), "g": [group] * 201})
        diff = np.diff(np.asarray(model.predict(grid), dtype=float))
        assert diff.max() <= 1e-8


def test_monotone_bs_interaction_without_main_fails_closed():
    """A monotone smooth x categorical interaction without the categorical main
    effect has no free per-category levels: every category's curve would be
    pinned to the same value at the lower boundary, silently biasing level
    shifts. That configuration is rejected, not quietly fitted."""
    x = np.linspace(0.0, 1.0, 30)
    df = pl.DataFrame(
        {
            "y": 1.0 - x,
            "x": x,
            "g": np.tile(["0", "1"], 15),
        }
    )
    with pytest.raises(ValidationError, match="free per-category levels"):
        rs.glm_dict(
            response="y",
            terms={},
            interactions=_monotone_interaction_spec(include_main=False),
            data=df,
            family="gaussian",
            link="identity",
        )


def test_multilevel_monotone_bs_interaction_blocks_are_category_contiguous_for_prediction():
    x = np.linspace(0.0, 1.0, 60)
    df = pl.DataFrame(
        {
            "y": 1.0 - x,
            "x": x,
            "g": np.tile(["0", "1", "2"], 20),
        }
    )
    model = rs.glm_dict(
        response="y",
        terms={},
        interactions=_monotone_interaction_spec(include_main=True),
        data=df,
        family="gaussian",
        link="identity",
    )

    # Main smooth + one own-curve block per category (all 3 levels).
    assert model._builder.get_smooth_terms()[1] == [(3, 12), (12, 21), (21, 30), (30, 39)]
    assert all(name.startswith("g[0]:bs(") for name in model.feature_names[12:21])
    assert all(name.startswith("g[1]:bs(") for name in model.feature_names[21:30])
    assert all(name.startswith("g[2]:bs(") for name in model.feature_names[30:39])

    grid = pl.DataFrame(
        {
            "x": np.linspace(0.05, 0.95, 12),
            "g": np.tile(["0", "1", "2"], 4),
        }
    )
    params = np.linspace(-0.5, 0.5, len(model.feature_names))
    full_design_eta = model._builder.transform_new_data(grid) @ params
    fast_eta = model._builder.linear_predict_new_data(grid, params)
    np.testing.assert_allclose(fast_eta, full_design_eta)


def test_interaction_endpoint_constraints_are_marker_gated():
    unmarked_decreasing = SimpleNamespace(monotonicity="decreasing")
    marked_empty = SimpleNamespace(
        monotonicity="increasing",
        _interaction_smooth_endpoint_constraint=True,
    )
    marked_increasing = SimpleNamespace(
        monotonicity="increasing",
        _interaction_smooth_endpoint_constraint=True,
    )
    marked_decreasing = SimpleNamespace(
        monotonicity="decreasing",
        _interaction_smooth_endpoint_constraint=True,
    )
    marked_unconstrained = SimpleNamespace(
        monotonicity=None,
        _interaction_smooth_endpoint_constraint=True,
    )

    nonneg, nonpos = _monotone_smooth_endpoint_constraints(
        [
            unmarked_decreasing,
            marked_empty,
            marked_increasing,
            marked_decreasing,
            marked_unconstrained,
        ],
        [(0, 2), (2, 2), (2, 4), (4, 6), (6, 8)],
        [10],
        [11],
    )

    assert nonneg == [2, 3, 10]
    assert nonpos == [4, 5, 11]


def test_interaction_local_spline_specs_do_not_clobber_main_effect():
    """A variable may carry different spline specs as a main effect and inside
    an interaction; predict must use each context's own fitted basis (the
    var_name-keyed registry used to be clobbered, crashing predict for
    different widths and silently mis-scoring for same-width divergent specs)."""
    rng = np.random.default_rng(1)
    n = 400
    x = rng.uniform(0.0, 3.0, n)
    g = rng.integers(0, 3, n).astype(str)
    y = rng.poisson(np.exp(0.1 + 0.2 * np.sin(x) + 0.1 * (g == "1"))).astype(float)
    data = pl.DataFrame({"y": y, "x": x, "g": g})

    # Different widths: predict used to crash with a matmul shape error.
    divergent_df = rs.glm_dict(
        response="y",
        terms={"x": {"type": "bs", "df": 6}, "g": {"type": "categorical"}},
        interactions=[{"x": {"type": "bs", "df": 4}, "g": {"type": "categorical"}}],
        data=data,
        family="poisson",
    ).fit()
    np.testing.assert_allclose(
        np.asarray(divergent_df.predict(data)),
        np.asarray(divergent_df.fittedvalues),
        rtol=1e-10,
    )

    # Same width, different degree: predict used to succeed and be silently wrong.
    divergent_degree = rs.glm_dict(
        response="y",
        terms={"x": {"type": "bs", "df": 5}, "g": {"type": "categorical"}},
        interactions=[{"x": {"type": "bs", "df": 5, "degree": 1}, "g": {"type": "categorical"}}],
        data=data,
        family="poisson",
    ).fit()
    np.testing.assert_allclose(
        np.asarray(divergent_degree.predict(data)),
        np.asarray(divergent_degree.fittedvalues),
        rtol=1e-10,
    )


def test_monotone_interaction_recovers_negative_category_level():
    """Per-category monotone curves must admit below-reference levels: the old
    sign-constrained deviation blocks forced every category at/above the
    reference, fitting a true -1.5 category gap as 0.0."""
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.uniform(0.0, 1.0, n)
    g = rng.integers(0, 2, n).astype(str)
    y = 2.0 + 1.0 * x - 1.5 * (g == "1") + rng.normal(0.0, 0.1, n)
    data = pl.DataFrame({"y": y, "x": x, "g": g})

    model = rs.glm_dict(
        response="y",
        terms={"g": {"type": "categorical"}},
        interactions=[
            {"x": {"type": "bs", "monotonicity": "increasing"}, "g": {"type": "categorical"}}
        ],
        data=data,
        family="gaussian",
    ).fit()
    assert model.converged

    grid = np.linspace(0.01, 0.99, 40)
    p0 = np.asarray(model.predict(pl.DataFrame({"x": grid, "g": ["0"] * 40})))
    p1 = np.asarray(model.predict(pl.DataFrame({"x": grid, "g": ["1"] * 40})))
    assert (p1 - p0).mean() == pytest.approx(-1.5, abs=0.05)
    assert np.all(np.diff(p0) >= -1e-8) and np.all(np.diff(p1) >= -1e-8)
    np.testing.assert_allclose(
        np.asarray(model.predict(data)), np.asarray(model.fittedvalues), rtol=1e-10
    )


def test_monotone_smooth_with_continuous_cofactor_is_enforced():
    """Monotonicity and the smoothing penalty apply to the spline function f
    itself in z*f(x) interactions (previously both were silently dropped)."""
    rng = np.random.default_rng(1)
    n = 800
    x = rng.uniform(0.0, 3.0, n)
    z = rng.uniform(0.2, 2.0, n)
    y = rng.poisson(np.exp(0.1 + 0.2 * np.sin(x) + 0.05 * z)).astype(float)
    data = pl.DataFrame({"y": y, "x": x, "z": z})

    model = rs.glm_dict(
        response="y",
        terms={"z": {"type": "linear"}},
        interactions=[{"x": {"type": "bs", "monotonicity": "increasing"}, "z": {"type": "linear"}}],
        data=data,
        family="poisson",
    ).fit()

    smooth_terms, _smooth_ranges = model._builder.get_smooth_terms()
    assert len(smooth_terms) == 1  # the penalty is registered, not dropped

    grid = np.linspace(0.05, 2.95, 40)
    pz = np.asarray(model.predict(pl.DataFrame({"x": grid, "z": np.full(40, 1.0)})))
    assert np.all(np.diff(pz) >= -1e-10)  # f monotone where z > 0
    np.testing.assert_allclose(
        np.asarray(model.predict(data)), np.asarray(model.fittedvalues), rtol=1e-10
    )


def test_monotone_fixed_df_interaction_fails_closed():
    x = np.linspace(0.0, 1.0, 60)
    df = pl.DataFrame({"y": 1.0 - x, "x": x, "g": np.tile(["0", "1"], 30), "z": x + 0.5})

    with pytest.raises(ValidationError, match="Monotone fixed-df"):
        rs.glm_dict(
            response="y",
            terms={"g": {"type": "categorical"}},
            interactions=[
                {
                    "x": {"type": "bs", "df": 4, "monotonicity": "increasing"},
                    "g": {"type": "categorical"},
                }
            ],
            data=df,
            family="gaussian",
        )

    with pytest.raises(ValidationError, match="Monotone fixed-df"):
        rs.glm_dict(
            response="y",
            terms={"z": {"type": "linear"}},
            interactions=[
                {
                    "x": {"type": "bs", "df": 4, "monotonicity": "increasing"},
                    "z": {"type": "linear"},
                }
            ],
            data=df,
            family="gaussian",
        )


def test_monotone_interaction_with_categorical_and_continuous_cofactor():
    """Full-k monotone blocks compose with a continuous co-factor: the fit and
    both predict paths stay consistent, and each category's curve is monotone
    where the co-factor is positive."""
    rng = np.random.default_rng(3)
    n = 900
    x = rng.uniform(0.0, 1.0, n)
    g = rng.integers(0, 2, n).astype(str)
    z = rng.uniform(0.5, 1.5, n)
    y = 1.0 + z * (x + 0.5 * (g == "1")) + rng.normal(0.0, 0.1, n)
    data = pl.DataFrame({"y": y, "x": x, "g": g, "z": z})

    model = rs.glm_dict(
        response="y",
        terms={"g": {"type": "categorical"}, "z": {"type": "linear"}},
        interactions=[
            {
                "x": {"type": "bs", "monotonicity": "increasing"},
                "g": {"type": "categorical"},
                "z": {"type": "linear"},
            }
        ],
        data=data,
        family="gaussian",
    ).fit()

    np.testing.assert_allclose(
        np.asarray(model.predict(data)), np.asarray(model.fittedvalues), rtol=1e-8
    )
    grid = np.linspace(0.01, 0.99, 50)
    for group in ("0", "1"):
        p = np.asarray(
            model.predict(pl.DataFrame({"x": grid, "g": [group] * 50, "z": np.ones(50)}))
        )
        assert np.all(np.diff(p) >= -1e-8), group


def test_monotone_interaction_multi_categorical_fails_closed():
    x = np.linspace(0.0, 1.0, 60)
    df = pl.DataFrame(
        {
            "y": 1.0 - x,
            "x": x,
            "g1": np.tile(["a", "b"], 30),
            "g2": np.tile(["u", "v", "u", "v"], 15),
        }
    )
    with pytest.raises(ValidationError, match="single categorical"):
        rs.glm_dict(
            response="y",
            terms={"g1": {"type": "categorical"}, "g2": {"type": "categorical"}},
            interactions=[
                {
                    "x": {"type": "bs", "monotonicity": "increasing"},
                    "g1": {"type": "categorical"},
                    "g2": {"type": "categorical"},
                }
            ],
            data=df,
            family="gaussian",
        )


def test_monotone_tensor_product_interaction_fails_closed():
    x = np.linspace(0.0, 1.0, 60)
    df = pl.DataFrame({"y": 1.0 - x, "x": x, "w": x**2})
    with pytest.raises(ValidationError, match="tensor-product"):
        rs.glm_dict(
            response="y",
            terms={},
            interactions=[
                {
                    "x": {"type": "bs", "monotonicity": "increasing"},
                    "w": {"type": "bs"},
                }
            ],
            data=df,
            family="gaussian",
        )
