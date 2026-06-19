#!/usr/bin/env python3
"""Profile the frozen APSFailure ridge-CV repro at Python/Rust boundaries."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl


def _rustystats_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_artifacts_dir() -> Path:
    return Path(__file__).resolve().parent / "artifacts"


def _ensure_local_rustystats() -> None:
    path = str(_rustystats_root() / "python")
    if path not in sys.path:
        sys.path.insert(0, path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "shape"):
        return list(value.shape)
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


class Profiler:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._call_counts: defaultdict[str, int] = defaultdict(int)

    def record(self, name: str, seconds: float, **meta: Any) -> None:
        self._call_counts[name] += 1
        self.calls.append(
            {
                "name": name,
                "call_index": self._call_counts[name] - 1,
                "seconds": float(seconds),
                **{key: _jsonable(value) for key, value in meta.items()},
            }
        )

    def timed(self, name: str, **meta: Any):
        profiler = self

        class Timer:
            def __enter__(self) -> None:
                self.started = time.perf_counter()

            def __exit__(self, *_exc: object) -> None:
                profiler.record(name, time.perf_counter() - self.started, **meta)

        return Timer()

    def wrap(self, name: str, fn: Callable[..., Any], meta_fn: Callable[..., dict[str, Any]]):
        profiler = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            meta = meta_fn(*args, **kwargs)
            started = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                profiler.record(name, time.perf_counter() - started, **meta)

        return wrapper


def _shape_of(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(item) for item in shape]


def _patch_runtime(profiler: Profiler) -> None:
    import rustystats._rustystats as rust
    import rustystats.formula as formula
    import rustystats.regularization_path as rp
    import rustystats.validation as validation

    def fit_glm_meta(*args: Any, **kwargs: Any) -> dict[str, Any]:
        alpha = kwargs.get("alpha", args[8] if len(args) > 8 else None)
        l1_ratio = kwargs.get("l1_ratio", args[9] if len(args) > 9 else None)
        max_iter = kwargs.get("max_iter", args[10] if len(args) > 10 else None)
        tol = kwargs.get("tol", args[11] if len(args) > 11 else None)
        skip_covariance = kwargs.get(
            "skip_covariance",
            args[19] if len(args) > 19 else None,
        )
        return {
            "x_shape": _shape_of(args[1] if len(args) > 1 else None),
            "family": args[2] if len(args) > 2 else kwargs.get("family"),
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "max_iter": max_iter,
            "tol": tol,
            "skip_covariance": skip_covariance,
        }

    def fit_cv_meta(*args: Any, **kwargs: Any) -> dict[str, Any]:
        alphas = kwargs.get("alphas", args[8] if len(args) > 8 else [])
        return {
            "x_shape": _shape_of(args[1] if len(args) > 1 else None),
            "family": args[2] if len(args) > 2 else kwargs.get("family"),
            "n_alphas": len(alphas) if alphas is not None else None,
            "l1_ratio": kwargs.get("l1_ratio", args[9] if len(args) > 9 else None),
            "n_folds": kwargs.get("n_folds", args[10] if len(args) > 10 else None),
            "max_iter": kwargs.get("max_iter", args[11] if len(args) > 11 else None),
            "tol": kwargs.get("tol", args[12] if len(args) > 12 else None),
        }

    def fit_core_meta(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "x_shape": _shape_of(args[1] if len(args) > 1 else None),
            "family": args[2] if len(args) > 2 else None,
            "alpha": args[8] if len(args) > 8 else kwargs.get("alpha"),
            "l1_ratio": args[9] if len(args) > 9 else kwargs.get("l1_ratio"),
            "compute_covariance": kwargs.get("compute_covariance", True),
        }

    def cv_path_meta(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "cv": kwargs.get("cv"),
            "selection": kwargs.get("selection"),
            "regularization": kwargs.get("regularization"),
            "n_alphas": kwargs.get("n_alphas"),
            "standardize": kwargs.get("standardize"),
        }

    def alpha_meta(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "x_shape": _shape_of(args[0] if args else None),
            "l1_ratio": args[2] if len(args) > 2 else kwargs.get("l1_ratio"),
            "family": kwargs.get("family"),
            "intercept_col": kwargs.get("intercept_col"),
        }

    def standardization_meta(*args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"x_shape": _shape_of(args[0] if args else None)}

    def validation_meta(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "y_shape": _shape_of(args[0] if args else None),
            "x_shape": _shape_of(args[1] if len(args) > 1 else None),
            "family": args[2] if len(args) > 2 else kwargs.get("family"),
        }

    def build_results_meta(*args: Any, **_kwargs: Any) -> dict[str, Any]:
        feature_names = args[1] if len(args) > 1 else []
        return {"n_feature_names": len(feature_names)}

    rust.fit_glm_py = profiler.wrap("rust_fit_glm_py", rust.fit_glm_py, fit_glm_meta)
    rust.fit_cv_path_py = profiler.wrap("rust_fit_cv_path_py", rust.fit_cv_path_py, fit_cv_meta)
    formula._fit_glm_core = profiler.wrap("_fit_glm_core", formula._fit_glm_core, fit_core_meta)
    formula._build_results = profiler.wrap(
        "_build_results", formula._build_results, build_results_meta
    )
    rp.fit_cv_regularization_path = profiler.wrap(
        "fit_cv_regularization_path",
        rp.fit_cv_regularization_path,
        cv_path_meta,
    )
    rp.compute_standardization_with_ridge_diag = profiler.wrap(
        "compute_standardization_with_ridge_diag",
        rp.compute_standardization_with_ridge_diag,
        standardization_meta,
    )
    rp.compute_alpha_max = profiler.wrap("compute_alpha_max", rp.compute_alpha_max, alpha_meta)
    validation.validate_glm_inputs = profiler.wrap(
        "validate_glm_inputs",
        validation.validate_glm_inputs,
        validation_meta,
    )


def _summarize_cv_profile(profile: dict[str, Any] | None) -> dict[str, Any] | None:
    if profile is None:
        return None
    folds = profile.get("folds") or []
    fold_totals = []
    per_alpha_fit = [0.0 for _ in range(int(profile.get("n_alphas", 0)))]
    per_alpha_wls_gram = [0.0 for _ in range(int(profile.get("n_alphas", 0)))]
    per_alpha_wls_gram_local_init = [0.0 for _ in range(int(profile.get("n_alphas", 0)))]
    per_alpha_wls_gram_row_scan = [0.0 for _ in range(int(profile.get("n_alphas", 0)))]
    per_alpha_wls_gram_pairwise_accum = [0.0 for _ in range(int(profile.get("n_alphas", 0)))]
    per_alpha_wls_gram_reduce = [0.0 for _ in range(int(profile.get("n_alphas", 0)))]
    per_alpha_wls_gram_materialize = [0.0 for _ in range(int(profile.get("n_alphas", 0)))]
    per_alpha_wls_penalty = [0.0 for _ in range(int(profile.get("n_alphas", 0)))]
    per_alpha_wls_solve = [0.0 for _ in range(int(profile.get("n_alphas", 0)))]
    per_alpha_validation_dot = [0.0 for _ in range(int(profile.get("n_alphas", 0)))]
    per_alpha_validation_score = [0.0 for _ in range(int(profile.get("n_alphas", 0)))]
    per_alpha_iterations = [0 for _ in range(int(profile.get("n_alphas", 0)))]
    for fold in folds:
        fit_seconds = list(map(float, fold.get("fit_seconds", [])))
        wls_gram_seconds = list(map(float, fold.get("fit_wls_gram_seconds", [])))
        wls_gram_local_init_seconds = list(
            map(float, fold.get("fit_wls_gram_local_init_seconds", []))
        )
        wls_gram_row_scan_seconds = list(map(float, fold.get("fit_wls_gram_row_scan_seconds", [])))
        wls_gram_pairwise_accum_seconds = list(
            map(float, fold.get("fit_wls_gram_pairwise_accum_seconds", []))
        )
        wls_gram_reduce_seconds = list(map(float, fold.get("fit_wls_gram_reduce_seconds", [])))
        wls_gram_materialize_seconds = list(
            map(float, fold.get("fit_wls_gram_materialize_seconds", []))
        )
        wls_penalty_seconds = list(map(float, fold.get("fit_wls_penalty_seconds", [])))
        wls_solve_seconds = list(map(float, fold.get("fit_wls_solve_seconds", [])))
        dot_seconds = list(map(float, fold.get("validation_dot_seconds", [])))
        score_seconds = list(map(float, fold.get("validation_score_seconds", [])))
        iterations = list(map(int, fold.get("iterations", [])))
        total = (
            float(fold.get("split_copy_seconds", 0.0))
            + float(fold.get("sparse_cache_seconds", 0.0))
            + float(fold.get("validation_sparse_cache_seconds", 0.0))
            + float(fold.get("standardize_seconds", 0.0))
            + float(fold.get("setup_seconds", 0.0))
            + sum(fit_seconds)
            + sum(dot_seconds)
            + sum(score_seconds)
        )
        fold_totals.append(
            {
                "fold": fold.get("fold"),
                "n_train": fold.get("n_train"),
                "n_val": fold.get("n_val"),
                "total_work_seconds": total,
                "fit_seconds": sum(fit_seconds),
                "validation_dot_seconds": sum(dot_seconds),
                "validation_score_seconds": sum(score_seconds),
                "iterations": iterations,
                "statuses": fold.get("statuses", []),
            }
        )
        for i, seconds in enumerate(fit_seconds):
            per_alpha_fit[i] += seconds
        for i, seconds in enumerate(wls_gram_seconds):
            per_alpha_wls_gram[i] += seconds
        for i, seconds in enumerate(wls_gram_local_init_seconds):
            per_alpha_wls_gram_local_init[i] += seconds
        for i, seconds in enumerate(wls_gram_row_scan_seconds):
            per_alpha_wls_gram_row_scan[i] += seconds
        for i, seconds in enumerate(wls_gram_pairwise_accum_seconds):
            per_alpha_wls_gram_pairwise_accum[i] += seconds
        for i, seconds in enumerate(wls_gram_reduce_seconds):
            per_alpha_wls_gram_reduce[i] += seconds
        for i, seconds in enumerate(wls_gram_materialize_seconds):
            per_alpha_wls_gram_materialize[i] += seconds
        for i, seconds in enumerate(wls_penalty_seconds):
            per_alpha_wls_penalty[i] += seconds
        for i, seconds in enumerate(wls_solve_seconds):
            per_alpha_wls_solve[i] += seconds
        for i, seconds in enumerate(dot_seconds):
            per_alpha_validation_dot[i] += seconds
        for i, seconds in enumerate(score_seconds):
            per_alpha_validation_score[i] += seconds
        for i, count in enumerate(iterations):
            per_alpha_iterations[i] += count

    alphas = profile.get("alphas")
    alpha_rows = []
    n = max(len(per_alpha_fit), len(per_alpha_validation_dot), len(per_alpha_validation_score))
    for i in range(n):
        alpha_rows.append(
            {
                "alpha_index": i,
                "alpha": alphas[i] if isinstance(alphas, list) and i < len(alphas) else None,
                "fit_seconds": per_alpha_fit[i],
                "wls_gram_seconds": per_alpha_wls_gram[i],
                "wls_gram_local_init_seconds": per_alpha_wls_gram_local_init[i],
                "wls_gram_row_scan_seconds": per_alpha_wls_gram_row_scan[i],
                "wls_gram_pairwise_accum_seconds": per_alpha_wls_gram_pairwise_accum[i],
                "wls_gram_reduce_seconds": per_alpha_wls_gram_reduce[i],
                "wls_gram_materialize_seconds": per_alpha_wls_gram_materialize[i],
                "wls_penalty_seconds": per_alpha_wls_penalty[i],
                "wls_solve_seconds": per_alpha_wls_solve[i],
                "validation_dot_seconds": per_alpha_validation_dot[i],
                "validation_score_seconds": per_alpha_validation_score[i],
                "iterations": per_alpha_iterations[i],
                "total_seconds": (
                    per_alpha_fit[i] + per_alpha_validation_dot[i] + per_alpha_validation_score[i]
                ),
            }
        )

    return {
        "top_level": {k: v for k, v in profile.items() if k != "folds"},
        "fold_totals": fold_totals,
        "alpha_totals": alpha_rows,
        "max_fold_work_seconds": max(
            (row["total_work_seconds"] for row in fold_totals), default=0.0
        ),
        "sum_fold_work_seconds": sum(row["total_work_seconds"] for row in fold_totals),
        "sum_alpha_fit_seconds": sum(per_alpha_fit),
        "sum_alpha_validation_dot_seconds": sum(per_alpha_validation_dot),
        "sum_alpha_validation_score_seconds": sum(per_alpha_validation_score),
        "sum_alpha_iterations": sum(per_alpha_iterations),
    }


def _summarize_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for call in calls:
        grouped[call["name"]].append(float(call["seconds"]))
    rows = []
    for name, values in grouped.items():
        rows.append(
            {
                "name": name,
                "count": len(values),
                "total_seconds": sum(values),
                "max_seconds": max(values),
                "min_seconds": min(values),
            }
        )
    return sorted(rows, key=lambda row: row["total_seconds"], reverse=True)


def profile_repro(args: argparse.Namespace) -> dict[str, Any]:
    _ensure_local_rustystats()
    profiler = Profiler()
    _patch_runtime(profiler)

    import rustystats as rs

    artifacts_dir = args.artifacts_dir
    with profiler.timed("load_metadata_json"):
        metadata = _load_json(artifacts_dir / "metadata.json")
        glm_kwargs = _load_json(artifacts_dir / "glm_kwargs.json")
        fit_config = _load_json(artifacts_dir / "fit_config.json")

    with profiler.timed("read_train_parquet"):
        train = pl.read_parquet(artifacts_dir / "train.parquet")

    if args.max_rows is not None:
        with profiler.timed("slice_train_head", max_rows=args.max_rows):
            train = train.head(args.max_rows)

    fit_kwargs = dict(fit_config["fit_kwargs"])
    if args.skip_covariance:
        fit_kwargs["compute_covariance"] = False
    else:
        fit_kwargs["compute_covariance"] = True
    if args.cv is not None:
        fit_kwargs["cv"] = args.cv
    if args.n_alphas is not None:
        fit_kwargs["n_alphas"] = args.n_alphas
    if args.cv_max_iter is not None:
        fit_kwargs["cv_max_iter"] = args.cv_max_iter
    if args.cv_tol is not None:
        fit_kwargs["cv_tol"] = args.cv_tol
    if args.selection is not None:
        fit_kwargs["selection"] = args.selection
    if args.exclude_unregularized:
        fit_kwargs["include_unregularized"] = False

    seed = int(args.seed if args.seed is not None else fit_config["seed"])

    gc.collect()
    with profiler.timed("builder_init"):
        builder = rs.glm_dict(**glm_kwargs, data=train, seed=seed)

    with profiler.timed("fit_total"):
        model = builder.fit(**fit_kwargs)

    prediction_summary = None
    if args.predict:
        with profiler.timed("read_test_parquet"):
            test = pl.read_parquet(artifacts_dir / "test.parquet")
        with profiler.timed("predict_total"):
            pred = model.predict(test)
        prediction_summary = {
            "n_predictions": len(pred),
            "min": float(min(pred)),
            "max": float(max(pred)),
        }

    final_no_covariance = None
    if args.compare_final_no_covariance and getattr(model, "alpha", None) is not None:
        gc.collect()
        compare_kwargs = {
            "alpha": float(model.alpha),
            "l1_ratio": float(getattr(model, "l1_ratio", 0.0) or 0.0),
            "compute_covariance": False,
        }
        with profiler.timed("compare_builder_init_no_covariance"):
            compare_builder = rs.glm_dict(**glm_kwargs, data=train, seed=seed)
        with profiler.timed("compare_final_fit_no_covariance"):
            compare_model = compare_builder.fit(**compare_kwargs)
        final_no_covariance = {
            "alpha": float(compare_model.alpha or 0.0),
            "n_params": len(compare_model.params),
        }

    cv_profile = model.cv_profile
    if cv_profile is not None and model.regularization_path is not None:
        cv_profile = dict(cv_profile)
        cv_profile["alphas"] = [row["alpha"] for row in model.regularization_path]

    return {
        "artifact_dir": str(artifacts_dir),
        "dataset": metadata["dataset"],
        "task_id": metadata["task_id"],
        "split": metadata["split"],
        "target": metadata["target"],
        "n_rows": train.height,
        "n_columns": train.width,
        "fit_kwargs": fit_kwargs,
        "model": {
            "n_params": len(model.params),
            "family": str(getattr(model, "family", "")),
            "regularization_type": getattr(model, "regularization_type", None),
            "alpha": float(model.alpha) if getattr(model, "alpha", None) is not None else None,
            "cv_deviance": model.cv_deviance,
            "cv_deviance_se": model.cv_deviance_se,
            "iterations": getattr(model, "iterations", None),
            "solver_status": getattr(model, "solver_status", None),
        },
        "prediction": prediction_summary,
        "final_no_covariance": final_no_covariance,
        "calls": profiler.calls,
        "call_summary": _summarize_calls(profiler.calls),
        "cv_profile": cv_profile,
        "cv_summary": _summarize_cv_profile(cv_profile),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=_default_artifacts_dir())
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--skip-covariance", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--compare-final-no-covariance", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cv", type=int, default=None)
    parser.add_argument("--n-alphas", type=int, default=None)
    parser.add_argument("--selection", choices=["min", "1se"], default=None)
    parser.add_argument("--cv-max-iter", type=int, default=None)
    parser.add_argument("--cv-tol", type=float, default=None)
    parser.add_argument("--exclude-unregularized", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = profile_repro(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
