"""The null model places an offset by the fitted link; the binding takes it on that scale."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import rustystats as rs
from rustystats._rustystats import compute_null_deviance_py

sm = pytest.importorskip("statsmodels.api")


def _gaussian_frame(n: int = 2000) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    offset = rng.normal(scale=0.5, size=n)
    return pl.DataFrame(
        {
            "x": x,
            "o": offset,
            "w": rng.uniform(0.5, 2.0, n),
            "y": offset + 0.4 * x + rng.normal(size=n),
        }
    )


@pytest.mark.parametrize("weighted", [False, True])
def test_identity_link_null_deviance_matches_statsmodels(weighted):
    df = _gaussian_frame()
    kwargs = {"weights": "w"} if weighted else {}
    model = rs.glm_dict(
        response="y",
        terms={"x": {"type": "linear"}},
        data=df,
        family="gaussian",
        offset="o",
        **kwargs,
    ).fit()
    reference = sm.GLM(
        df["y"].to_numpy(),
        sm.add_constant(df["x"].to_numpy()),
        family=sm.families.Gaussian(),
        offset=df["o"].to_numpy(),
        freq_weights=df["w"].to_numpy() if weighted else None,
    ).fit()
    assert model.null_deviance() == pytest.approx(reference.null_deviance, rel=1e-10)


def test_a_non_default_link_places_the_offset_by_that_link():
    rng = np.random.default_rng(1)
    n = 2000
    x = rng.normal(size=n)
    offset = rng.normal(scale=0.3, size=n)
    y = np.exp(0.5 + 0.2 * x + offset) + rng.normal(scale=0.2, size=n)
    df = pl.DataFrame({"x": x, "o": offset, "y": y})
    model = rs.glm_dict(
        response="y",
        terms={"x": {"type": "linear"}},
        data=df,
        family="gaussian",
        link="log",
        offset="o",
    ).fit()
    a = np.exp(offset)
    mean = a * y.sum() / a.sum()
    assert model.null_deviance() == pytest.approx(np.sum((y - mean) ** 2), rel=1e-10)


def test_out_of_core_null_deviance_matches_in_memory(tmp_path):
    df = _gaussian_frame()

    def open_batches(columns):
        return [df.select(columns)]

    glm = {
        "response": "y",
        "terms": {"x": {"type": "linear"}},
        "family": "gaussian",
        "offset": "o",
    }
    in_memory = rs.glm_dict(data=df, **glm).fit()
    out_of_core = rs.glm_dict(data=rs.BatchSource(open_batches, df.schema), **glm).fit(
        out_of_core=rs.OutOfCoreOptions(encode_rows=4096, scratch_dir=tmp_path)
    )
    assert out_of_core.null_deviance() == in_memory.null_deviance()


def test_null_deviance_binding_treats_exposure_as_an_offset():
    rng = np.random.default_rng(2)
    n = 3000
    exposure = rng.uniform(0.1, 1.0, n)
    y = rng.poisson(0.4 * exposure).astype(float)
    rate = y.sum() / exposure.sum()
    mean = rate * exposure
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mean), 0.0)
    expected = 2.0 * np.sum(term - (y - mean))
    got = compute_null_deviance_py(y, "poisson", offset=np.log(exposure))
    assert got == pytest.approx(expected, rel=1e-10)
    # The old call spread the rate by exposure as a weight, which is a different number.
    as_weight = compute_null_deviance_py(y, "poisson", weights=exposure)
    assert as_weight != pytest.approx(expected, rel=1e-3)


def test_null_deviance_binding_honours_the_link_and_weights():
    y = np.array([1.0, 2.0, 4.0, 9.0])
    offset = np.array([0.5, 1.5, -1.0, 3.0])
    weights = np.array([1.0, 2.0, 0.5, 1.5])
    c = np.sum(weights * (y - offset)) / weights.sum()
    expected = np.sum(weights * (y - (c + offset)) ** 2)
    got = compute_null_deviance_py(y, "gaussian", weights=weights, offset=offset, link="identity")
    assert got == pytest.approx(expected, rel=1e-12)
    # With no link given, the family's own is used.
    assert compute_null_deviance_py(y, "gaussian", weights=weights, offset=offset) == pytest.approx(
        expected, rel=1e-12
    )
