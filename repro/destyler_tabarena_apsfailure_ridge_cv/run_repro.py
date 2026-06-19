#!/usr/bin/env python3
"""Run the frozen APSFailure RustyStats ridge-CV student fit."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
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


def _model_summary(model: Any) -> dict[str, Any]:
    params = getattr(model, "params", None)
    alpha = getattr(model, "alpha", None)
    return {
        "n_params": len(params) if params is not None else None,
        "family": str(getattr(model, "family", "")),
        "regularization_type": getattr(model, "regularization_type", None),
        "alpha": float(alpha) if alpha is not None else None,
    }


def _fit_once(
    *,
    train: pl.DataFrame,
    glm_kwargs: dict[str, Any],
    fit_kwargs: dict[str, Any],
    seed: int,
) -> Any:
    import rustystats as rs

    builder = rs.glm_dict(**glm_kwargs, data=train, seed=seed)
    return builder.fit(**fit_kwargs)


def run_repro(args: argparse.Namespace) -> dict[str, Any]:
    _ensure_local_rustystats()
    artifacts_dir = args.artifacts_dir
    metadata = _load_json(artifacts_dir / "metadata.json")
    glm_kwargs = _load_json(artifacts_dir / "glm_kwargs.json")
    fit_config = _load_json(artifacts_dir / "fit_config.json")

    timings: dict[str, float] = {}
    started = time.perf_counter()
    train = pl.read_parquet(artifacts_dir / "train.parquet")
    timings["load_train_s"] = time.perf_counter() - started

    if args.max_rows is not None:
        train = train.head(args.max_rows)

    fit_kwargs = dict(fit_config["fit_kwargs"])
    fit_kwargs["compute_covariance"] = not args.skip_covariance
    if args.cv is not None:
        fit_kwargs["cv"] = args.cv
    if args.n_alphas is not None:
        fit_kwargs["n_alphas"] = args.n_alphas
    if args.selection is not None:
        fit_kwargs["selection"] = args.selection
    if args.cv_max_iter is not None:
        fit_kwargs["cv_max_iter"] = args.cv_max_iter
    if args.cv_tol is not None:
        fit_kwargs["cv_tol"] = args.cv_tol

    seed = int(args.seed if args.seed is not None else fit_config["seed"])

    for _ in range(args.warmup):
        _ = _fit_once(train=train, glm_kwargs=glm_kwargs, fit_kwargs=fit_kwargs, seed=seed)
        gc.collect()

    runs: list[dict[str, Any]] = []
    for run_index in range(args.repeats):
        gc.collect()
        started = time.perf_counter()
        import rustystats as rs

        builder = rs.glm_dict(**glm_kwargs, data=train, seed=seed)
        timings_build = time.perf_counter() - started

        started = time.perf_counter()
        model = builder.fit(**fit_kwargs)
        timings_fit = time.perf_counter() - started

        predict_s = None
        if args.predict:
            test = pl.read_parquet(artifacts_dir / "test.parquet")
            started = time.perf_counter()
            _ = model.predict(test)
            predict_s = time.perf_counter() - started

        runs.append(
            {
                "run_index": run_index,
                "build_s": timings_build,
                "fit_s": timings_fit,
                "predict_s": predict_s,
                **_model_summary(model),
            }
        )

    return {
        "artifact_dir": str(artifacts_dir),
        "dataset": metadata["dataset"],
        "task_id": metadata["task_id"],
        "split": metadata["split"],
        "target": metadata["target"],
        "n_rows": train.height,
        "n_columns": train.width,
        "recipe_terms": metadata["recipe_terms"],
        "recipe_interactions": metadata["recipe_interactions"],
        "fit_kwargs": fit_kwargs,
        "load_train_s": timings["load_train_s"],
        "runs": runs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=_default_artifacts_dir())
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--skip-covariance", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cv", type=int, default=None)
    parser.add_argument("--n-alphas", type=int, default=None)
    parser.add_argument("--selection", choices=["min", "1se"], default=None)
    parser.add_argument("--cv-max-iter", type=int, default=None)
    parser.add_argument("--cv-tol", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_repro(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
