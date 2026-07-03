#!/usr/bin/env python3
"""Smoke-test an installed RustyStats package."""

from __future__ import annotations

import sys

import numpy as np
import polars as pl
import rustystats as rs


def main() -> int:
    rng = np.random.default_rng(20260630)
    n = 40
    x = rng.normal(size=n)
    exposure = rng.uniform(0.5, 1.5, size=n)
    y = rng.poisson(np.exp(-0.2 + 0.4 * x) * exposure).astype(float)
    data = pl.DataFrame({"y": y, "x": x, "exposure": exposure})

    model = rs.glm_dict(
        response="y",
        terms={"x": {"type": "linear"}},
        data=data,
        family="poisson",
        exposure="exposure",
    ).fit()
    predictions = np.asarray(model.predict(data.head(5)), dtype=np.float64)
    if predictions.shape != (5,) or not np.all(np.isfinite(predictions)):
        print("prediction smoke failed")
        return 1

    loaded = rs.GLMModel.from_bytes(model.to_bytes())
    loaded_predictions = np.asarray(loaded.predict(data.head(5)), dtype=np.float64)
    np.testing.assert_allclose(loaded_predictions, predictions, rtol=1e-12, atol=1e-12)

    print(f"rustystats package smoke passed; version={getattr(rs, '__version__', 'unknown')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
