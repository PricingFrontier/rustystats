"""Run the Rust CV path directly and capture Rust-side timing detail.

This complements ``profile_rustystats_fit.py``. The first profiler recreates
the full public ``glm_dict(...).fit(...)`` call; this one stops after the design
matrix is built and calls ``rustystats._rustystats.fit_cv_path_py`` directly so
the returned Rust-internal ``profile`` payload can be saved.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the parquet and JSON repro files.",
    )
    parser.add_argument("--rows", type=int, default=None, help="Optional head-row sample.")
    parser.add_argument("--n-alphas", type=int, default=None, help="Override fit_config n_alphas.")
    parser.add_argument("--cv", type=int, default=None, help="Override fit_config cv.")
    parser.add_argument("--alpha-min-ratio", type=float, default=None)
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--cv-seed", type=int, default=None)
    parser.add_argument(
        "--include-unregularized",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle_dir = args.bundle_dir.resolve()
    output = args.output or bundle_dir / "profile_cv_path_detail_latest.json"

    import rustystats as rs
    import rustystats._rustystats as rust
    from rustystats.formula import _get_constraint_indices
    from rustystats.regularization_path import (
        DEFAULT_ALPHA_MIN_RATIO,
        compute_alpha_max,
        compute_standardization,
        generate_alpha_path,
        penalized_column_mask,
        solver_standardization,
    )

    t0 = time.perf_counter()
    train = pl.read_parquet(bundle_dir / "fremtpl2_50k_rustystats_train.parquet")
    if args.rows is not None:
        train = train.head(args.rows)
    glm_kwargs = json.loads((bundle_dir / "glm_kwargs.json").read_text())
    fit_config = json.loads((bundle_dir / "fit_config.json").read_text())

    build_start = time.perf_counter()
    builder = rs.glm_dict(**glm_kwargs, data=train, seed=fit_config.get("seed", 0))
    build_seconds = time.perf_counter() - build_start

    regularization = fit_config.get("regularization")
    if regularization != "ridge":
        raise ValueError(f"This direct profiler currently expects ridge, got {regularization!r}.")
    l1_ratio = 0.0
    cv = args.cv if args.cv is not None else int(fit_config.get("cv", 5))
    n_alphas = args.n_alphas if args.n_alphas is not None else int(fit_config.get("n_alphas", 20))
    alpha_min_ratio = (
        args.alpha_min_ratio if args.alpha_min_ratio is not None else DEFAULT_ALPHA_MIN_RATIO
    )
    cv_seed = args.cv_seed if args.cv_seed is not None else int(fit_config.get("seed", 0))
    include_unregularized = (
        args.include_unregularized if args.include_unregularized is not None else True
    )

    prep_start = time.perf_counter()
    fit_intercept = bool(builder.intercept)
    pen_mask = penalized_column_mask(builder.X.shape[1], fit_intercept)
    center = scale = None
    if args.standardize:
        center, scale = compute_standardization(
            builder.X,
            builder.weights,
            pen_mask,
            fit_intercept=fit_intercept,
        )
    alpha_max = compute_alpha_max(
        builder.X,
        builder.y,
        l1_ratio,
        family=builder.family,
        link=builder.link,
        offset=builder.offset,
        weights=builder.weights,
        var_power=builder.var_power,
        theta=builder.theta if isinstance(builder.theta, (int, float)) else 1.0,
        intercept_col=0 if fit_intercept else None,
        center=center,
        scale=scale,
        pen_mask=pen_mask,
        allow_extended_tweedie=builder.allow_extended_tweedie,
    )
    alphas = generate_alpha_path(alpha_max, n_alphas, alpha_min_ratio)
    if include_unregularized and (alphas.size == 0 or not np.any(alphas == 0.0)):
        alphas = np.concatenate([alphas, np.array([0.0], dtype=np.float64)])
    solver_center, solver_scale = solver_standardization(
        center,
        scale,
        fit_intercept=fit_intercept,
    )
    nonneg_indices, nonpos_indices = _get_constraint_indices(builder.feature_names)
    prep_seconds = time.perf_counter() - prep_start

    cv_start = time.perf_counter()
    result = rust.fit_cv_path_py(
        builder.y,
        builder.X,
        builder.family,
        builder.link,
        builder.var_power,
        builder.theta if isinstance(builder.theta, (int, float)) else 1.0,
        builder.offset,
        builder.weights,
        list(map(float, alphas)),
        l1_ratio,
        cv,
        args.max_iter,
        args.tol,
        cv_seed,
        nonneg_indices=nonneg_indices if nonneg_indices else None,
        nonpos_indices=nonpos_indices if nonpos_indices else None,
        allow_extended_tweedie=builder.allow_extended_tweedie,
        fit_intercept=fit_intercept,
        center=solver_center,
        scale=solver_scale,
    )
    cv_wall_seconds = time.perf_counter() - cv_start

    report = {
        "command": sys.argv,
        "cwd": os.getcwd(),
        "platform": platform.platform(),
        "python": sys.version,
        "rustystats_version": getattr(rs, "__version__", None),
        "extension_file": str(Path(rust.__file__).resolve()),
        "rayon_num_threads_env": os.environ.get("RAYON_NUM_THREADS"),
        "input_shape": {"rows": train.height, "columns": train.width},
        "design_shape": {"rows": int(builder.X.shape[0]), "columns": int(builder.X.shape[1])},
        "build_seconds": build_seconds,
        "prep_seconds": prep_seconds,
        "cv_wall_seconds_python": cv_wall_seconds,
        "total_seconds": time.perf_counter() - t0,
        "cv_args": {
            "regularization": regularization,
            "cv": cv,
            "n_alphas_requested": n_alphas,
            "n_alphas_actual": len(alphas),
            "include_unregularized": include_unregularized,
            "standardize": args.standardize,
            "alpha_min_ratio": alpha_min_ratio,
            "alpha_max": float(alpha_max),
            "alphas": list(map(float, alphas)),
            "max_iter": args.max_iter,
            "tol": args.tol,
            "cv_seed": cv_seed,
        },
        "result": _jsonable(result),
    }
    output.write_text(json.dumps(report, indent=2))
    print(f"Wrote {output}")
    print(f"Design shape: {builder.X.shape[0]:,} x {builder.X.shape[1]:,}")
    print(f"Python CV wall: {cv_wall_seconds:.3f}s")
    profile = result.get("profile", {})
    print("Rust profile totals:")
    for key, value in profile.get("summed_work_seconds", {}).items():
        print(f"  {key:35s} {float(value):9.3f}s")
    print(
        f"  cv_parallel_wall_seconds           {profile.get('cv_parallel_wall_seconds', 0):9.3f}s"
    )
    print(f"  total_wall_seconds                 {profile.get('total_wall_seconds', 0):9.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
