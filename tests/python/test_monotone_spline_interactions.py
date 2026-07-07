from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl
import rustystats as rs
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
    assert smooth_ranges == [(2, 11), (11, 20)]
    assert [
        getattr(term, "_interaction_smooth_endpoint_constraint", False) for term in smooth_terms
    ] == [False, True]

    for group in ("0", "1"):
        grid = pl.DataFrame({"x": np.linspace(0.01, 0.99, 201), "g": [group] * 201})
        diff = np.diff(np.asarray(model.predict(grid), dtype=float))
        assert diff.max() <= 1e-8


def test_monotone_bs_interaction_without_main_still_uses_spline_basis():
    x = np.linspace(0.0, 1.0, 30)
    df = pl.DataFrame(
        {
            "y": 1.0 - x,
            "x": x,
            "g": np.tile(["0", "1"], 15),
        }
    )
    model = rs.glm_dict(
        response="y",
        terms={},
        interactions=_monotone_interaction_spec(include_main=False),
        data=df,
        family="gaussian",
        link="identity",
    )

    assert "g[T.1]:x" not in model.feature_names
    assert model.feature_names[1:] == [f"g[T.1]:bs(x, {i}/9, k, -)" for i in range(1, 10)]
    assert model._builder.get_smooth_terms()[1] == [(1, 10)]


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

    assert model._builder.get_smooth_terms()[1] == [(3, 12), (12, 21), (21, 30)]
    assert all(name.startswith("g[T.1]:bs(") for name in model.feature_names[12:21])
    assert all(name.startswith("g[T.2]:bs(") for name in model.feature_names[21:30])

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
