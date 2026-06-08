"""Reproduce a Destyler/RustyStats wide-interaction performance case.

Run from the RustyStats repository root:

    uv run python rustystats-performance/run_performance_case.py --quick

The default command fits the saved GLM recipe and runs a full counterfactual PDP
prediction workload over the training frame. That is intentionally slow: it is
the workload that made the Destyler report refresh expensive.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import rustystats as rs

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "fremtpl2_frequency_teacher_target.parquet"
DEFAULT_SPEC = HERE / "model_spec.json"


@contextmanager
def timed(label: str):
    start = time.perf_counter()
    print(f"[stage] {label}...", flush=True)
    try:
        yield
    finally:
        print(f"[stage] {label}: {time.perf_counter() - start:.3f}s", flush=True)


def load_spec(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_model(
    train: pl.DataFrame,
    spec: dict[str, Any],
    *,
    regularization: str | None,
    cv: int,
    selection: str,
    n_alphas: int,
    seed: int,
) -> Any:
    kwargs = {
        "response": spec["response"],
        "terms": spec["terms"],
        "interactions": spec["interactions"],
        "family": spec["family"],
        "link": spec.get("link"),
        "var_power": spec.get("var_power", 1.5),
        "theta": spec.get("theta"),
        "offset": spec.get("offset"),
        "input_transforms": spec.get("input_transforms") or None,
        "data": train,
        "seed": seed,
    }
    builder = rs.glm_dict(**kwargs)
    if regularization in {None, "none", "None"}:
        return builder.fit()
    return builder.fit(
        regularization=regularization,
        cv=cv,
        selection=selection,
        cv_seed=seed,
        n_alphas=n_alphas,
    )


def numeric_grid(values: pl.Series, *, grid_size: int) -> list[float]:
    clean = values.drop_nulls()
    if clean.is_empty():
        return []
    quantiles = np.linspace(0.02, 0.98, grid_size)
    grid = [float(clean.quantile(float(q), interpolation="nearest")) for q in quantiles]
    out: list[float] = []
    seen: set[float] = set()
    for value in grid:
        if np.isfinite(value) and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def categorical_grid(values: pl.Series, *, max_levels: int) -> list[str]:
    levels = sorted(str(v) for v in values.drop_nulls().unique().to_list())
    if len(levels) > max_levels:
        return []
    return levels


def counterfactual_predict_workload(
    model: Any,
    frame: pl.DataFrame,
    spec: dict[str, Any],
    *,
    pdp_rows: int,
    continuous_grid_size: int,
    max_levels: int,
    factors: list[str] | None,
) -> int:
    work = frame
    if pdp_rows > 0 and frame.height > pdp_rows:
        work = frame.sample(n=pdp_rows, seed=0)

    requested = set(factors or [])
    cat_features = list(spec.get("categorical_factors") or [])
    continuous_features = list(spec.get("continuous_factors") or [])
    if requested:
        cat_features = [f for f in cat_features if f in requested]
        continuous_features = [f for f in continuous_features if f in requested]

    calls = 0
    rows_scored = 0
    for feature in cat_features:
        if feature not in work.columns:
            continue
        for value in categorical_grid(work[feature], max_levels=max_levels):
            cf = work.with_columns(pl.lit(value).alias(feature))
            _ = model.predict(cf)
            calls += 1
            rows_scored += cf.height

    for feature in continuous_features:
        if feature not in work.columns:
            continue
        for value in numeric_grid(work[feature], grid_size=continuous_grid_size):
            cf = work.with_columns(pl.lit(value).alias(feature))
            _ = model.predict(cf)
            calls += 1
            rows_scored += cf.height

    print(
        f"[pdp] calls={calls} frame_rows={work.height} total_rows_scored={rows_scored}",
        flush=True,
    )
    return calls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--role", default="train", help="Role used for fitting.")
    parser.add_argument("--predict-role", default="train", help="Role used for prediction/PDP.")
    parser.add_argument("--regularization", default="ridge")
    parser.add_argument("--cv", type=int, default=2)
    parser.add_argument("--selection", default="1se")
    parser.add_argument("--n-alphas", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pdp-rows",
        type=int,
        default=0,
        help="Rows for counterfactual PDP workload; 0 means all rows.",
    )
    parser.add_argument("--continuous-grid-size", type=int, default=24)
    parser.add_argument("--max-levels", type=int, default=80)
    parser.add_argument("--skip-pdp", action="store_true")
    parser.add_argument(
        "--factors",
        default="",
        help="Optional comma-separated factor subset for PDP workload.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test mode: sample fewer rows and fewer PDP points.",
    )
    parser.add_argument("--cprofile", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.pdp_rows = 5_000 if args.pdp_rows == 0 else args.pdp_rows
        args.continuous_grid_size = min(args.continuous_grid_size, 8)

    spec = load_spec(args.spec)
    factors = [f.strip() for f in args.factors.split(",") if f.strip()] or None

    profiler = cProfile.Profile() if args.cprofile else None
    if profiler is not None:
        profiler.enable()

    with timed("read_parquet"):
        data = pl.read_parquet(args.data)
    with timed("split_roles"):
        train = data.filter(pl.col("role") == args.role)
        predict_frame = data.filter(pl.col("role") == args.predict_role)
    print(
        f"[data] total={data.height} train={train.height} predict={predict_frame.height}",
        flush=True,
    )
    print(
        "[spec] "
        f"terms={len(spec['terms'])} interactions={len(spec['interactions'])} "
        f"transforms={len(spec.get('input_transforms') or [])}",
        flush=True,
    )

    with timed("fit_glm"):
        model = build_model(
            train,
            spec,
            regularization=args.regularization,
            cv=args.cv,
            selection=args.selection,
            n_alphas=args.n_alphas,
            seed=args.seed,
        )

    with timed("single_predict"):
        pred = model.predict(predict_frame)
    print(
        f"[predict] rows={predict_frame.height} mean={float(np.mean(pred)):.8f}",
        flush=True,
    )

    if not args.skip_pdp:
        with timed("counterfactual_pdp_predict"):
            counterfactual_predict_workload(
                model,
                predict_frame,
                spec,
                pdp_rows=args.pdp_rows,
                continuous_grid_size=args.continuous_grid_size,
                max_levels=args.max_levels,
                factors=factors,
            )

    if profiler is not None:
        profiler.disable()
        args.cprofile.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(args.cprofile)
        stats = pstats.Stats(profiler).sort_stats("cumtime")
        stats.print_stats(40)
        print(f"[profile] wrote {args.cprofile}", flush=True)


if __name__ == "__main__":
    main()
