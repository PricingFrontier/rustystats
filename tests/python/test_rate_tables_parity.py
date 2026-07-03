import numpy as np
import polars as pl
import pytest
import rustystats as rs


def _inverse_link(link: str, eta: np.ndarray) -> np.ndarray:
    if link == "log":
        return np.exp(eta)
    if link == "identity":
        return eta
    if link == "logit":
        return 1.0 / (1.0 + np.exp(-eta))
    if link == "inverse":
        return 1.0 / eta
    if link == "sqrt":
        return eta * eta
    raise AssertionError(f"unsupported link in rate-table test scorer: {link}")


def _score_rate_table_artifact(artifact: dict, data: pl.DataFrame) -> np.ndarray:
    eta = np.full(len(data), float(artifact["base"]["eta"]), dtype=np.float64)
    for table in artifact["tables"]:
        columns = table["columns"]
        eta_idx = columns.index("eta")
        source_columns = table["sources"]
        row_eta = {
            tuple(str(value) for value in row[: len(source_columns)]): float(row[eta_idx])
            for row in table["rows"]
        }
        default_eta = float(table["default"][eta_idx]) if table.get("default") is not None else 0.0
        for i, row in enumerate(data.iter_rows(named=True)):
            key = tuple(str(row[source]) for source in source_columns)
            eta[i] += row_eta.get(key, default_eta)
    return _inverse_link(artifact["link"], eta)


@pytest.mark.assurance
def test_plain_categorical_rate_table_scores_like_native_prediction():
    rng = np.random.default_rng(17)
    n = 250
    brand = rng.choice(["A", "B", "C"], size=n)
    eta = -0.3 + 0.4 * (brand == "B") - 0.25 * (brand == "C")
    data = pl.DataFrame({"y": rng.poisson(np.exp(eta)).astype(float), "brand": brand})
    model = rs.glm_dict(
        response="y",
        terms={"brand": {"type": "categorical"}},
        data=data,
        family="poisson",
    ).fit()

    artifact = model.to_rate_tables()
    deployed_pred = _score_rate_table_artifact(artifact, data)
    native_pred = np.asarray(model.predict(data), dtype=np.float64)
    np.testing.assert_allclose(deployed_pred, native_pred, rtol=1e-12, atol=1e-12)


@pytest.mark.assurance
def test_lookup_rate_table_scores_like_native_prediction():
    data = pl.DataFrame(
        {
            "y": [1, 2, 3, 1, 4, 2, 5, 3],
            "brand": ["A", "A", "B", "B", "C", "C", "A", "B"],
            "region": ["N", "S", "N", "S", "N", "S", "N", "S"],
        }
    )
    input_transforms = [
        {
            "type": "lookup",
            "name": "brand_region_lookup",
            "sources": ["brand", "region"],
            "output": "rating_factor",
            "output_dtype": "float64",
            "keys": [["A", "N"], ["A", "S"], ["B", "N"], ["B", "S"], ["C", "N"], ["C", "S"]],
            "values": [0.0, 0.2, -0.1, 0.35, 0.15, -0.2],
            "default": 0.0,
            "on_unseen": "default",
            "on_null": "default",
        }
    ]
    model = rs.glm_dict(
        response="y",
        terms={"rating_factor": {"type": "linear"}},
        data=data,
        family="poisson",
        input_transforms=input_transforms,
    ).fit()

    artifact = model.to_rate_tables()
    deployed_pred = _score_rate_table_artifact(artifact, data)
    native_pred = np.asarray(model.predict(data), dtype=np.float64)
    np.testing.assert_allclose(deployed_pred, native_pred, rtol=1e-12, atol=1e-12)
