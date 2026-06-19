#!/usr/bin/env python3
"""Freeze the APSFailure destyler student fit into RustyStats-only artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.preprocessing import LabelEncoder

TARGET_COL = "__target__"
DEFAULT_METADATA = Path(
    "/tmp/tabarena/packages/tabarena/src/tabarena/benchmark/task/metadata/sources/data/"
    "TabArena-v0.1_tasks_metadata.csv"
)
DEFAULT_TASK_ID = 363616
DEFAULT_SPLIT = "r0f0"
DEFAULT_N_ALPHAS = 20


def _rustystats_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_destyler_root() -> Path:
    return _rustystats_root().parent / "destyler"


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "artifacts"


def _ensure_paths(destyler_root: Path) -> None:
    for path in (_rustystats_root() / "python", destyler_root / "python", destyler_root):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _load_tabarena_runner(destyler_root: Path) -> Any:
    runner_path = destyler_root / "benchmarks" / "tabarena" / "run.py"
    if not runner_path.exists():
        raise FileNotFoundError(f"TabArena runner not found: {runner_path}")
    _ensure_paths(destyler_root)
    spec = importlib.util.spec_from_file_location("destyler_tabarena_run", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {runner_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _benchmark_args(runner: Any, *, metadata: Path, threads: int) -> argparse.Namespace:
    raw_argv = [
        "--metadata",
        str(metadata),
        "--teacher-preset",
        "tabarena_depth3",
        "--student-recipe-preset",
        "ebm_like",
        "--numeric-missing-policy",
        "indicator",
        "--student-modes",
        "teacher",
        "--threads",
        str(threads),
    ]
    parser = runner._build_parser()
    args = parser.parse_args(raw_argv)
    args._provided_flags = runner._provided_flags(raw_argv)
    runner._apply_teacher_preset(args, raw_argv)
    return args


def _find_row(runner: Any, metadata: Path, *, task_id: int, split: str) -> Any:
    rows = runner._read_metadata(metadata)
    for row in rows:
        if row.task_id == task_id and row.seed == split:
            return row
    raise ValueError(f"No TabArena metadata row found for task_id={task_id}, split={split!r}")


def _phase(timings: dict[str, float], name: str):
    class Timer:
        def __enter__(self) -> None:
            self.started = time.perf_counter()

        def __exit__(self, *_exc: object) -> None:
            timings[name] = time.perf_counter() - self.started

    return Timer()


def prepare_artifacts(
    *,
    destyler_root: Path,
    metadata: Path,
    output_dir: Path,
    task_id: int,
    split_name: str,
    threads: int,
    n_alphas: int,
) -> None:
    runner = _load_tabarena_runner(destyler_root)
    args = _benchmark_args(runner, metadata=metadata, threads=threads)
    row = _find_row(runner, metadata, task_id=task_id, split=split_name)
    if row.problem_type != "binary":
        raise ValueError("This repro is pinned to the binary APSFailure task.")

    import destyler as dst
    from destyler import api as dst_api
    from destyler._student import guard_sparse_interactions

    output_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    print(f"loading OpenML task {task_id} ({row.dataset_name}, {row.seed})", flush=True)
    with _phase(timings, "load_openml_s"):
        task, X_raw, y_raw, categorical_indicator = runner._load_openml_task(row.task_id)

    print("preparing TabArena split", flush=True)
    with _phase(timings, "split_prepare_s"):
        split = runner._split_data(
            row,
            task,
            X_raw,
            y_raw,
            categorical_indicator,
            max_categorical_levels=args.max_categorical_levels,
            numeric_missing_policy=args.numeric_missing_policy,
        )

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(split.y_train.astype(str)).astype(float)
    train = runner._to_polars_with_target(split.X_train, y_train)
    test = runner._to_polars_features(split.X_test)

    seed = args.seed + row.repeat * 100 + row.fold
    recipe_settings = runner._recipe_settings(split, args)
    teacher_params = runner._teacher_extra_params(row, args)
    early_stopping_rounds = runner._resolved_early_stopping_rounds(args, len(split.X_train))
    regularization = runner._student_regularization(runner.STUDENT_MODE_TEACHER, args)
    cv = runner._student_cv(args)

    distiller = dst.Distiller(
        train,
        TARGET_COL,
        "binomial",
        cat_features=split.cat_features,
        seed=seed,
    )

    print("training depth-3 CatBoost teacher", flush=True)
    with _phase(timings, "teacher_s"):
        distiller.train_teacher(
            depth=args.teacher_depth,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            l2_leaf_reg=args.l2_leaf_reg,
            one_hot_max_size=args.one_hot_max_size,
            teacher_params=teacher_params,
            early_stopping_rounds=early_stopping_rounds,
            validation_fraction=args.validation_fraction,
            thread_count=args.threads,
            teacher_ensemble=args.teacher_ensemble,
        )

    print("decomposing teacher", flush=True)
    with _phase(timings, "decompose_s"):
        distiller.decompose(max_order=args.max_order, decomposition_measure="row")

    print("proposing EBM-like recipe", flush=True)
    with _phase(timings, "propose_s"):
        distiller.propose(
            top_k_main=recipe_settings.top_k_main,
            top_2way=recipe_settings.top_2way,
            top_3way=recipe_settings.top_3way,
            min_importance_share=recipe_settings.min_importance_share,
            min_interaction_energy_share=recipe_settings.min_interaction_energy_share,
            cat_group_threshold=args.cat_group_threshold,
            cat_te_threshold=args.cat_te_threshold,
            frozen_column_budget=recipe_settings.frozen_column_budget,
            one_hot_max_size=args.one_hot_max_size,
        )
    recipe = distiller.recipe
    if recipe is None:
        raise RuntimeError("destyler did not produce a recipe")

    print("building exact RustyStats training frame", flush=True)
    with _phase(timings, "with_teacher_complement_s"):
        full = dst_api._with_teacher_complement(distiller.teacher, train, recipe)

    with _phase(timings, "sparse_guard_s"):
        converted_sparse = guard_sparse_interactions(recipe, full, n_min=args.sparse_cell_min)

    with _phase(timings, "regularization_resolution_s"):
        effective_regularization = dst_api._resolve_projection_regularization(
            regularization,
            recipe,
            full,
        )

    with _phase(timings, "teacher_target_s"):
        mu_teacher = np.asarray(
            distiller.teacher.predict_mu(
                full,
                exposure=dst_api._exposure_values(recipe, full),
                offset=dst_api._offset_values(recipe, full),
                offset_is_exposure=False,
            ),
            dtype=np.float64,
        )
        fit_train = full.with_columns(pl.Series(TARGET_COL, mu_teacher))

    with _phase(timings, "anchor_rows_s"):
        fit_train, recipe, anchor_summary = dst_api._build_counterfactual_anchor_rows(
            recipe=recipe,
            teacher=distiller.teacher,
            train=fit_train,
            response=TARGET_COL,
            anchor_weight=dst_api._DEFAULT_ANCHOR_WEIGHT,
            max_features=12,
            continuous_grid_size=8,
            max_levels=24,
            min_rows=50,
            max_rows_per_point=250,
            seed=seed,
        )

    print("writing parquet and JSON artifacts", flush=True)
    with _phase(timings, "write_artifacts_s"):
        fit_train.write_parquet(output_dir / "train.parquet")
        test.write_parquet(output_dir / "test.parquet")
        _write_json(output_dir / "glm_kwargs.json", recipe.to_glm_dict_kwargs())
        _write_json(output_dir / "recipe.json", recipe.to_dict())
        _write_json(
            output_dir / "fit_config.json",
            {
                "target": "teacher",
                "regularization": effective_regularization,
                "requested_regularization": regularization,
                "cv": cv,
                "selection": "1se",
                "cv_seed": seed,
                "seed": seed,
                "n_alphas": n_alphas,
                "fit_kwargs": {
                    "regularization": effective_regularization,
                    "cv": cv,
                    "selection": "1se",
                    "cv_seed": seed,
                    "n_alphas": n_alphas,
                },
            },
        )
        _write_json(
            output_dir / "metadata.json",
            {
                "dataset": row.dataset_name,
                "task_id": row.task_id,
                "split": row.seed,
                "row": asdict(row),
                "case_seed": seed,
                "target": "teacher predictions",
                "objective": "RustyStats ridge CV runtime reduction",
                "n_train_observed": len(split.X_train),
                "n_test": len(split.X_test),
                "n_features_raw": row.num_features,
                "n_features_prepared": split.X_train.shape[1],
                "n_features_after_missing_indicators": len(split.X_train.columns),
                "fit_train_rows": fit_train.height,
                "fit_train_columns": fit_train.width,
                "recipe_terms": len(recipe.terms),
                "recipe_interactions": len(recipe.interactions),
                "recipe_skipped_interactions": len(recipe.skipped_interactions),
                "converted_sparse_interactions": [list(item) for item in converted_sparse],
                "anchor_summary": anchor_summary,
                "recipe_settings": asdict(recipe_settings),
                "teacher_settings": {
                    "teacher_preset": args.teacher_preset,
                    "depth": args.teacher_depth,
                    "iterations": args.iterations,
                    "learning_rate": args.learning_rate,
                    "l2_leaf_reg": args.l2_leaf_reg,
                    "teacher_ensemble": args.teacher_ensemble,
                    "one_hot_max_size": args.one_hot_max_size,
                    "early_stopping_rounds": early_stopping_rounds,
                    "validation_fraction": args.validation_fraction,
                    "threads": args.threads,
                    "teacher_params": teacher_params,
                },
                "student_settings": {
                    "student_recipe_preset": args.student_recipe_preset,
                    "regularization": effective_regularization,
                    "requested_regularization": regularization,
                    "cv": cv,
                    "selection": "1se",
                    "n_alphas": n_alphas,
                    "sparse_cell_min": args.sparse_cell_min,
                    "counterfactual_anchor_weight": dst_api._DEFAULT_ANCHOR_WEIGHT,
                },
                "preprocessing": {
                    "numeric_missing_policy": args.numeric_missing_policy,
                    "numeric_missing_train": split.numeric_missing_train,
                    "numeric_missing_test": split.numeric_missing_test,
                    "numeric_missing_indicators": split.numeric_missing_indicators,
                    "max_categorical_levels": args.max_categorical_levels,
                    "categorical_grouped_train": split.categorical_grouped_train,
                    "categorical_grouped_test": split.categorical_grouped_test,
                    "capped_categorical_features": split.capped_categorical_features,
                    "cat_features": split.cat_features,
                    "feature_map": split.feature_map,
                },
                "phase_timings": timings,
            },
        )

    print(f"wrote artifacts to {output_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destyler-root", type=Path, default=_default_destyler_root())
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--task-id", type=int, default=DEFAULT_TASK_ID)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--n-alphas", type=int, default=DEFAULT_N_ALPHAS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepare_artifacts(
        destyler_root=args.destyler_root,
        metadata=args.metadata,
        output_dir=args.output_dir,
        task_id=args.task_id,
        split_name=args.split,
        threads=args.threads,
        n_alphas=args.n_alphas,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
