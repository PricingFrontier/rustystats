"""Benchmark deterministic input transforms and rate-table export.

Run:

    uv run python benchmarks/bench_input_transforms.py --rows 200000

The benchmark compares lookup-transform application against a direct Polars
left join, then times no-transform prediction, transformed prediction, and
rate-table export. It is intentionally dependency-free beyond RustyStats'
runtime dependencies.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import rustystats as rs
from rustystats.input_transforms import apply_input_transforms


@dataclass
class Timing:
    name: str
    seconds: float


def _time(name: str, fn: Callable[[], object]) -> tuple[Timing, object]:
    t0 = time.perf_counter()
    out = fn()
    return Timing(name, time.perf_counter() - t0), out


def make_data(n_rows: int, n_brands: int, n_regions: int) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    brands = np.array([f"B{i:04d}" for i in range(n_brands)], dtype=object)
    regions = np.array([f"R{i:03d}" for i in range(n_regions)], dtype=object)
    brand = rng.choice(brands, n_rows)
    region = rng.choice(regions, n_rows)
    exposure = rng.uniform(0.5, 1.5, n_rows)
    x = rng.normal(0.0, 1.0, n_rows)
    eta = 0.1 + 0.03 * x + np.log(exposure)
    y = rng.poisson(np.exp(eta))
    return pl.DataFrame(
        {
            "y": y,
            "brand": brand,
            "region": region,
            "x": x,
            "exposure": exposure,
        }
    )


def make_lookup(n_brands: int, n_regions: int) -> list[dict[str, Any]]:
    keys = []
    values = []
    for b in range(n_brands):
        for r in range(n_regions):
            keys.append([f"B{b:04d}", f"R{r:03d}"])
            values.append(((b % 17) - 8) * 0.01 + ((r % 11) - 5) * 0.005)
    return [
        {
            "type": "lookup",
            "name": "brand_region_effect",
            "sources": ["brand", "region"],
            "output": "brand_region_fts",
            "output_dtype": "float64",
            "keys": keys,
            "values": values,
            "default": 0.0,
            "on_unseen": "default",
            "on_null": "default",
        }
    ]


def direct_join(data: pl.DataFrame, spec: dict[str, Any]) -> pl.DataFrame:
    lookup = pl.DataFrame(
        {
            "brand": [row[0] for row in spec["keys"]],
            "region": [row[1] for row in spec["keys"]],
            "brand_region_fts": spec["values"],
        }
    )
    return (
        data.with_row_index("__row__")
        .join(lookup, on=["brand", "region"], how="left", maintain_order="left")
        .sort("__row__")
        .drop("__row__")
        .with_columns(pl.col("brand_region_fts").fill_null(0.0))
    )


def run(rows: int, brands: int, regions: int) -> list[Timing]:
    timings: list[Timing] = []
    data = make_data(rows, brands, regions)
    transforms = make_lookup(brands, regions)

    timing, prepared_direct = _time("direct_polars_join", lambda: direct_join(data, transforms[0]))
    timings.append(timing)
    timing, prepared = _time(
        "apply_input_transforms",
        lambda: apply_input_transforms(data, transforms),
    )
    timings.append(timing)
    np.testing.assert_allclose(
        prepared["brand_region_fts"].to_numpy(),
        prepared_direct["brand_region_fts"].to_numpy(),
    )

    timing, plain = _time(
        "fit_plain",
        lambda: rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}},
            data=data,
            family="poisson",
            exposure="exposure",
        ).fit(),
    )
    timings.append(timing)

    timing, transformed = _time(
        "fit_transformed",
        lambda: rs.glm_dict(
            response="y",
            terms={"brand_region_fts": {"type": "linear"}},
            data=data,
            family="poisson",
            exposure="exposure",
            input_transforms=transforms,
        ).fit(),
    )
    timings.append(timing)

    scoring = data.select(["brand", "region", "x", "exposure"])
    timing, _ = _time("predict_plain", lambda: plain.predict(scoring))
    timings.append(timing)
    timing, _ = _time("predict_transformed", lambda: transformed.predict(scoring))
    timings.append(timing)
    timing, _ = _time("rate_table_export_dict", lambda: transformed.to_rate_tables())
    timings.append(timing)
    return timings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--brands", type=int, default=200)
    parser.add_argument("--regions", type=int, default=20)
    args = parser.parse_args()

    timings = run(args.rows, args.brands, args.regions)
    print(f"rows={args.rows:,} lookup_rows={args.brands * args.regions:,}")
    for timing in timings:
        print(f"{timing.name:28s} {timing.seconds:8.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
