"""Profile related sparse-design hot paths on the performance repro data.

This script keeps the original parquet/model build but swaps the operation being
timed:

* lasso / elastic-net CV paths, which exercise coordinate descent
* Rust VIF/correlation diagnostics
* robust covariance / robust standard errors

Run from the repository root:

    ./.venv/bin/python rustystats-performance/profile_related_hotpaths.py
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

EPSILON = 1e-10


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.size <= 256:
            return value.tolist()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(np.nanmin(value)),
            "max": float(np.nanmax(value)),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _time_call(fn, *args, repeats: int = 1, **kwargs) -> dict[str, Any]:
    seconds: list[float] = []
    result: Any = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        seconds.append(time.perf_counter() - start)
    return {
        "seconds": seconds[-1],
        "repeats": repeats,
        "seconds_all": seconds,
        "seconds_min": min(seconds),
        "seconds_mean": sum(seconds) / len(seconds),
        "result": result,
    }


def _summarize_cv_result(result: dict[str, Any]) -> dict[str, Any]:
    profile = result.get("profile", {})
    folds = profile.get("folds", [])
    statuses: dict[str, int] = {}
    iterations: list[int] = []
    fit_seconds: list[float] = []
    for fold in folds:
        for status in fold.get("statuses", []):
            statuses[status] = statuses.get(status, 0) + 1
        iterations.extend(int(v) for v in fold.get("iterations", []))
        fit_seconds.extend(float(v) for v in fold.get("fit_seconds", []))

    return {
        "best_alpha": result.get("best_alpha"),
        "best_cv_deviance": result.get("best_cv_deviance"),
        "profile": profile,
        "status_counts": statuses,
        "iterations": {
            "count": len(iterations),
            "min": min(iterations) if iterations else None,
            "max": max(iterations) if iterations else None,
            "mean": (sum(iterations) / len(iterations)) if iterations else None,
        },
        "fit_seconds": {
            "count": len(fit_seconds),
            "min": min(fit_seconds) if fit_seconds else None,
            "max": max(fit_seconds) if fit_seconds else None,
            "mean": (sum(fit_seconds) / len(fit_seconds)) if fit_seconds else None,
            "sum": sum(fit_seconds),
        },
    }


def _regularization_l1_ratio(name: str, elastic_net_l1_ratio: float) -> float:
    if name == "lasso":
        return 1.0
    if name == "elastic_net":
        return elastic_net_l1_ratio
    raise ValueError(f"Unsupported regularization for coordinate descent timing: {name!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing parquet and repro JSON files.",
    )
    parser.add_argument("--rows", type=int, default=None, help="Optional head-row sample.")
    parser.add_argument("--cv", type=int, default=2)
    parser.add_argument("--n-alphas", type=int, default=3)
    parser.add_argument("--alpha-min-ratio", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=6)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--cv-seed", type=int, default=None)
    parser.add_argument(
        "--regularization",
        action="append",
        choices=["lasso", "elastic_net"],
        default=None,
        help="Coordinate-descent regularization to time. Repeatable.",
    )
    parser.add_argument("--elastic-net-l1-ratio", type=float, default=0.5)
    parser.add_argument(
        "--include-unregularized",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--vif-repeats", type=int, default=3)
    parser.add_argument("--vif-skip-cols", type=int, action="append", default=None)
    parser.add_argument("--robust-alpha", type=float, default=50.0)
    parser.add_argument("--robust-fit-max-iter", type=int, default=25)
    parser.add_argument("--robust-fit-tol", type=float, default=1e-8)
    parser.add_argument("--robust-types", action="append", default=None)
    parser.add_argument(
        "--skip-robust-fit",
        action="store_true",
        help="Skip the robust covariance harness fit/timings.",
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
    output = args.output or bundle_dir / "profile_related_hotpaths_latest.json"
    regularizations = args.regularization or ["lasso", "elastic_net"]
    vif_skip_cols = args.vif_skip_cols if args.vif_skip_cols is not None else [1, 0]
    robust_types = args.robust_types or ["HC1", "HC3"]

    import rustystats as rs
    import rustystats._rustystats as rust
    from rustystats.formula import _get_constraint_indices
    from rustystats.regularization_path import (
        compute_alpha_max,
        compute_standardization,
        generate_alpha_path,
        penalized_column_mask,
        solver_standardization,
    )

    total_start = time.perf_counter()
    train = pl.read_parquet(bundle_dir / "fremtpl2_50k_rustystats_train.parquet")
    if args.rows is not None:
        train = train.head(args.rows)
    glm_kwargs = json.loads((bundle_dir / "glm_kwargs.json").read_text())
    fit_config = json.loads((bundle_dir / "fit_config.json").read_text())
    cv_seed = args.cv_seed if args.cv_seed is not None else int(fit_config.get("seed", 0))

    build_start = time.perf_counter()
    builder = rs.glm_dict(**glm_kwargs, data=train, seed=fit_config.get("seed", 0))
    build_seconds = time.perf_counter() - build_start

    x = np.asarray(builder.X, dtype=np.float64)
    y = np.asarray(builder.y, dtype=np.float64)
    offset = None if builder.offset is None else np.asarray(builder.offset, dtype=np.float64)
    weights = None if builder.weights is None else np.asarray(builder.weights, dtype=np.float64)
    theta = builder.theta if isinstance(builder.theta, (int, float)) else 1.0
    fit_intercept = bool(builder.intercept)
    pen_mask = penalized_column_mask(x.shape[1], fit_intercept)
    center, scale = compute_standardization(x, weights, pen_mask, fit_intercept=fit_intercept)
    solver_center, solver_scale = solver_standardization(
        center,
        scale,
        fit_intercept=fit_intercept,
    )
    nonneg_indices, nonpos_indices = _get_constraint_indices(builder.feature_names)

    design_nonzero = int(np.count_nonzero(x))
    report: dict[str, Any] = {
        "command": sys.argv,
        "cwd": os.getcwd(),
        "platform": platform.platform(),
        "python": sys.version,
        "rustystats_version": getattr(rs, "__version__", None),
        "extension_file": str(Path(rust.__file__).resolve()),
        "rayon_num_threads_env": os.environ.get("RAYON_NUM_THREADS"),
        "input_shape": {"rows": train.height, "columns": train.width},
        "design_shape": {"rows": int(x.shape[0]), "columns": int(x.shape[1])},
        "design_density": design_nonzero / float(x.size),
        "design_nonzero": design_nonzero,
        "build_seconds": build_seconds,
        "coordinate_descent": {},
        "vif": {},
        "robust": {},
    }

    for regularization in regularizations:
        l1_ratio = _regularization_l1_ratio(regularization, args.elastic_net_l1_ratio)
        prep_start = time.perf_counter()
        alpha_max = compute_alpha_max(
            x,
            y,
            l1_ratio,
            family=builder.family,
            link=builder.link,
            offset=offset,
            weights=weights,
            var_power=builder.var_power,
            theta=theta,
            intercept_col=0 if fit_intercept else None,
            center=center,
            scale=scale,
            pen_mask=pen_mask,
            allow_extended_tweedie=builder.allow_extended_tweedie,
        )
        alphas = generate_alpha_path(alpha_max, args.n_alphas, args.alpha_min_ratio)
        if args.include_unregularized and (alphas.size == 0 or not np.any(alphas == 0.0)):
            alphas = np.concatenate([alphas, np.array([0.0], dtype=np.float64)])
        prep_seconds = time.perf_counter() - prep_start

        cv_start = time.perf_counter()
        cv_result = rust.fit_cv_path_py(
            y,
            x,
            builder.family,
            builder.link,
            builder.var_power,
            theta,
            offset,
            weights,
            list(map(float, alphas)),
            l1_ratio,
            args.cv,
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
        cv_seconds = time.perf_counter() - cv_start
        report["coordinate_descent"][regularization] = {
            "prep_seconds": prep_seconds,
            "cv_wall_seconds_python": cv_seconds,
            "args": {
                "cv": args.cv,
                "n_alphas_requested": args.n_alphas,
                "n_alphas_actual": len(alphas),
                "include_unregularized": args.include_unregularized,
                "alpha_min_ratio": args.alpha_min_ratio,
                "alpha_max": float(alpha_max),
                "alphas": list(map(float, alphas)),
                "l1_ratio": l1_ratio,
                "max_iter": args.max_iter,
                "tol": args.tol,
                "cv_seed": cv_seed,
            },
            "result": _summarize_cv_result(cv_result),
        }
        print(
            f"{regularization}: CV wall {cv_seconds:.3f}s, "
            f"fit work {report['coordinate_descent'][regularization]['result']['fit_seconds']['sum']:.3f}s"
        )

    for skip_cols in vif_skip_cols:
        if skip_cols >= x.shape[1]:
            continue

        def vif_call(skip_cols: int = skip_cols):
            r, vif = rust.compute_correlation_and_vif_py(x, EPSILON, skip_cols)
            return {
                "corr_shape": list(r.shape),
                "vif_len": int(vif.shape[0]),
                "max_abs_corr": float(np.nanmax(np.abs(r))) if r.size else 0.0,
                "max_vif": float(np.nanmax(vif)) if vif.size else 0.0,
            }

        timed = _time_call(vif_call, repeats=args.vif_repeats)
        report["vif"][f"skip_cols={skip_cols}"] = {
            **{k: v for k, v in timed.items() if k != "result"},
            "result": timed["result"],
        }
        print(
            f"vif skip_cols={skip_cols}: min {timed['seconds_min']:.3f}s "
            f"mean {timed['seconds_mean']:.3f}s"
        )

    if not args.skip_robust_fit:
        robust_fit_start = time.perf_counter()
        robust_result = rust.fit_glm_py(
            y,
            x,
            builder.family,
            builder.link,
            builder.var_power,
            theta,
            offset,
            weights,
            args.robust_alpha,
            0.0,
            args.robust_fit_max_iter,
            args.robust_fit_tol,
            nonneg_indices if nonneg_indices else None,
            nonpos_indices if nonpos_indices else None,
            True,
            builder.allow_extended_tweedie,
            fit_intercept,
            solver_center,
            solver_scale,
        )
        robust_fit_seconds = time.perf_counter() - robust_fit_start
        report["robust"]["fit_harness"] = {
            "seconds": robust_fit_seconds,
            "alpha": args.robust_alpha,
            "max_iter": args.robust_fit_max_iter,
            "tol": args.robust_fit_tol,
            "iterations": int(robust_result.iterations),
            "converged": bool(robust_result.converged),
            "solver_status": str(robust_result.solver_status),
            "deviance": float(robust_result.deviance),
        }
        print(
            f"robust harness fit: {robust_fit_seconds:.3f}s "
            f"({robust_result.solver_status}, {robust_result.iterations} iter)"
        )

        for hc_type in robust_types:
            bse_timed = _time_call(
                lambda hc_type=hc_type: np.asarray(robust_result.bse_robust(hc_type))
            )
            cov_timed = _time_call(
                lambda hc_type=hc_type: np.asarray(robust_result.cov_robust(hc_type))
            )
            bse = bse_timed["result"]
            cov = cov_timed["result"]
            report["robust"][hc_type] = {
                "bse_seconds": bse_timed["seconds"],
                "cov_seconds": cov_timed["seconds"],
                "bse_summary": {
                    "len": int(bse.shape[0]),
                    "min": float(np.nanmin(bse)),
                    "max": float(np.nanmax(bse)),
                },
                "cov_summary": {
                    "shape": list(cov.shape),
                    "diag_min": float(np.nanmin(np.diag(cov))),
                    "diag_max": float(np.nanmax(np.diag(cov))),
                },
            }
            print(
                f"robust {hc_type}: bse {bse_timed['seconds']:.3f}s, "
                f"cov {cov_timed['seconds']:.3f}s"
            )

    report["total_seconds"] = time.perf_counter() - total_start
    output.write_text(json.dumps(_jsonable(report), indent=2))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
