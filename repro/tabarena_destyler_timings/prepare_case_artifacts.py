#!/usr/bin/env python3
"""Prepare TabArena destyler student artifacts for RustyStats timing.

This script recreates the scalar TabArena split, trains the depth-3 teacher,
builds the destyler recipe, then writes the exact RustyStats training frame and
GLM kwargs needed to time the student fit without retraining CatBoost.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.preprocessing import LabelEncoder

TARGET_COL = "__target__"
DEFAULT_CASES = Path(__file__).with_name("cases_slowest_first_run.csv")
DEFAULT_FIRST_RUN_MANIFEST = (
    Path(__file__).resolve().parents[2].parent
    / "destyler"
    / "reports"
    / "tabarena"
    / "full_20260617_1"
    / "manifest.json"
)


def _rustystats_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_destyler_root() -> Path:
    return _rustystats_root().parent / "destyler"


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


def _load_tabarena_runner(destyler_root: Path):
    runner_path = destyler_root / "benchmarks" / "tabarena" / "run.py"
    if not runner_path.exists():
        raise FileNotFoundError(f"TabArena runner not found: {runner_path}")

    for path in (_rustystats_root() / "python", destyler_root / "python", destyler_root):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)

    spec = importlib.util.spec_from_file_location("destyler_tabarena_run", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {runner_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise ModuleNotFoundError(
            f"Missing dependency {missing!r} while importing the destyler TabArena runner. "
            "Install the repro dependencies with "
            "`uv pip install -r repro/tabarena_destyler_timings/requirements.txt`."
        ) from exc
    return module


def _load_destyler_private_modules(destyler_root: Path):
    for path in (_rustystats_root() / "python", destyler_root / "python", destyler_root):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    from destyler import api as dst_api
    from destyler._student import guard_sparse_interactions

    return dst_api, guard_sparse_interactions


def _load_manifest_args(runner: Any, manifest_path: Path | None) -> argparse.Namespace:
    parser = runner._build_parser()
    args = parser.parse_args([])
    if manifest_path is None:
        return args
    if not manifest_path.exists():
        raise FileNotFoundError(f"Benchmark manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, value in payload.get("args", {}).items():
        if hasattr(args, key):
            setattr(args, key, value)
    if isinstance(args.metadata, str):
        args.metadata = Path(args.metadata)
    if isinstance(args.output_root, str):
        args.output_root = Path(args.output_root)
    return args


def _case_rows_from_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _select_case_rows(
    runner: Any,
    metadata_path: Path,
    *,
    cases_path: Path | None,
    datasets: set[str] | None,
    all_splits_for_datasets: bool,
    limit: int | None,
) -> list[Any]:
    rows = runner._read_metadata(metadata_path)
    if all_splits_for_datasets:
        if not datasets:
            raise ValueError("--all-splits-for-datasets requires --datasets")
        selected = [r for r in rows if r.dataset_name in datasets or r.task in datasets]
    else:
        case_specs = _case_rows_from_csv(cases_path or DEFAULT_CASES)
        selected = []
        for spec in case_specs:
            if datasets and spec["dataset"] not in datasets:
                continue
            matches = [
                r
                for r in rows
                if r.dataset_name == spec["dataset"]
                and r.task_id == int(spec["task_id"])
                and r.repeat == int(spec["repeat"])
                and r.fold == int(spec["fold"])
            ]
            if not matches:
                raise ValueError(f"No metadata row matched case: {spec}")
            selected.append(matches[0])
    if limit is not None:
        selected = selected[:limit]
    return selected


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _fit_config_for(
    *,
    target: str,
    requested_regularization: str | None,
    effective_regularization: str | None,
    cv: int,
    selection: str,
    seed: int,
    n_alphas: int,
) -> dict[str, Any]:
    return {
        "target": target,
        "requested_regularization": requested_regularization,
        "effective_regularization": effective_regularization,
        "cv": cv,
        "selection": selection,
        "cv_seed": seed,
        "seed": seed,
        "n_alphas": n_alphas,
        "fit_kwargs": (
            {}
            if effective_regularization is None
            else {
                "regularization": effective_regularization,
                "cv": cv,
                "selection": selection,
                "cv_seed": seed,
                "n_alphas": n_alphas,
            }
        ),
    }


def _time_direct_fit(
    *,
    recipe: Any,
    train: pl.DataFrame,
    effective_regularization: str | None,
    cv: int,
    selection: str,
    seed: int,
    n_alphas: int,
) -> tuple[Any, float]:
    import rustystats as rs

    started = time.perf_counter()
    builder = rs.glm_dict(**recipe.to_glm_dict_kwargs(), data=train, seed=seed)
    if effective_regularization is None:
        model = builder.fit()
    else:
        model = builder.fit(
            regularization=effective_regularization,
            cv=cv,
            selection=selection,
            cv_seed=seed,
            n_alphas=n_alphas,
        )
    return model, time.perf_counter() - started


def _prepare_target_artifact(
    *,
    target: str,
    target_dir: Path,
    runner: Any,
    dst_api: Any,
    guard_sparse_interactions: Any,
    distiller: Any,
    base_recipe: Any,
    train: pl.DataFrame,
    test: pl.DataFrame,
    y_train_actual: np.ndarray,
    case_seed: int,
    benchmark_args: argparse.Namespace,
    counterfactual_anchor_weight: float,
    n_alphas: int,
    repeat_fit: int,
) -> dict[str, Any]:
    target_dir.mkdir(parents=True, exist_ok=True)
    recipe = copy.deepcopy(base_recipe)
    fit_to_teacher = target == "teacher"
    requested_regularization = (
        benchmark_args.regularization
        if fit_to_teacher
        else getattr(benchmark_args, "actual_regularization", "elastic_net")
    )

    timings: dict[str, float] = {}
    started = time.perf_counter()
    full = dst_api._with_teacher_complement(distiller.teacher, train, recipe)
    timings["with_teacher_complement_s"] = time.perf_counter() - started

    converted_sparse: list[tuple[str, ...]] = []
    if benchmark_args.sparse_cell_min:
        started = time.perf_counter()
        converted_sparse = guard_sparse_interactions(
            recipe, full, n_min=benchmark_args.sparse_cell_min
        )
        timings["sparse_guard_s"] = time.perf_counter() - started
    else:
        timings["sparse_guard_s"] = 0.0

    started = time.perf_counter()
    effective_regularization = dst_api._resolve_projection_regularization(
        requested_regularization, recipe, full
    )
    timings["regularization_resolution_s"] = time.perf_counter() - started

    fit_train = full
    anchor_summary = None
    if fit_to_teacher:
        started = time.perf_counter()
        mu_teacher = np.asarray(
            distiller.teacher.predict_mu(
                fit_train,
                exposure=dst_api._exposure_values(recipe, fit_train),
                offset=dst_api._offset_values(recipe, fit_train),
                offset_is_exposure=False,
            ),
            dtype=np.float64,
        )
        fit_train = fit_train.with_columns(pl.Series(TARGET_COL, mu_teacher))
        timings["teacher_target_s"] = time.perf_counter() - started

        started = time.perf_counter()
        fit_train, recipe, anchor_summary = dst_api._build_counterfactual_anchor_rows(
            recipe=recipe,
            teacher=distiller.teacher,
            train=fit_train,
            response=TARGET_COL,
            anchor_weight=counterfactual_anchor_weight,
            max_features=12,
            continuous_grid_size=8,
            max_levels=24,
            min_rows=50,
            max_rows_per_point=250,
            seed=case_seed,
        )
        timings["anchor_rows_s"] = time.perf_counter() - started
    else:
        fit_train = fit_train.with_columns(pl.Series(TARGET_COL, y_train_actual))
        timings["teacher_target_s"] = 0.0
        timings["anchor_rows_s"] = 0.0

    started = time.perf_counter()
    fit_train.write_parquet(target_dir / "train.parquet")
    test.write_parquet(target_dir / "test.parquet")
    timings["write_parquet_s"] = time.perf_counter() - started

    _write_json(target_dir / "recipe.json", recipe.to_dict())
    _write_json(target_dir / "glm_kwargs.json", recipe.to_glm_dict_kwargs())

    fit_config = _fit_config_for(
        target=target,
        requested_regularization=requested_regularization,
        effective_regularization=effective_regularization,
        cv=benchmark_args.cv,
        selection="1se",
        seed=case_seed,
        n_alphas=n_alphas,
    )
    _write_json(target_dir / "fit_config.json", fit_config)

    direct_fit_times: list[float] = []
    predict_times: list[float] = []
    model = None
    for _ in range(repeat_fit):
        model, fit_s = _time_direct_fit(
            recipe=recipe,
            train=fit_train,
            effective_regularization=effective_regularization,
            cv=benchmark_args.cv,
            selection="1se",
            seed=case_seed,
            n_alphas=n_alphas,
        )
        direct_fit_times.append(fit_s)
        started = time.perf_counter()
        _ = model.predict(test)
        predict_times.append(time.perf_counter() - started)

    return {
        "target": target,
        "target_dir": str(target_dir),
        "fit_train_rows": fit_train.height,
        "fit_train_columns": fit_train.width,
        "recipe_terms": len(recipe.terms),
        "recipe_interactions": len(recipe.interactions),
        "recipe_skipped_interactions": len(recipe.skipped_interactions),
        "converted_sparse_interactions": [list(item) for item in converted_sparse],
        "requested_regularization": requested_regularization,
        "effective_regularization": effective_regularization,
        "anchor_summary": anchor_summary,
        "phase_timings": timings,
        "direct_fit_times_s": direct_fit_times,
        "direct_fit_min_s": min(direct_fit_times) if direct_fit_times else None,
        "direct_fit_median_s": (float(np.median(direct_fit_times)) if direct_fit_times else None),
        "direct_predict_median_s": (float(np.median(predict_times)) if predict_times else None),
        "model_params": len(model.params) if model is not None else None,
    }


def _prepare_case(
    *,
    row: Any,
    runner: Any,
    dst_api: Any,
    guard_sparse_interactions: Any,
    output_dir: Path,
    benchmark_args: argparse.Namespace,
    targets: list[str],
    counterfactual_anchor_weight: float,
    n_alphas: int,
    repeat_fit: int,
) -> dict[str, Any]:
    if row.problem_type not in {"binary", "regression"}:
        raise ValueError(
            f"{row.dataset_name} is {row.problem_type}; this scalar repro handles binary/regression only"
        )

    case_seed = benchmark_args.seed + row.repeat * 100 + row.fold
    case_dir = output_dir / f"{_safe_name(row.dataset_name)}__{row.seed}"
    case_dir.mkdir(parents=True, exist_ok=True)

    phase: dict[str, float] = {}
    started = time.perf_counter()
    task, X_raw, y_raw, categorical_indicator = runner._load_openml_task(row.task_id)
    phase["load_openml_s"] = time.perf_counter() - started

    started = time.perf_counter()
    split = runner._split_data(
        row,
        task,
        X_raw,
        y_raw,
        categorical_indicator,
        max_categorical_levels=benchmark_args.max_categorical_levels,
    )
    phase["split_prepare_s"] = time.perf_counter() - started

    if row.problem_type == "binary":
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(split.y_train.astype(str)).astype(float)
        family = "binomial"
        n_classes = len(encoder.classes_)
        class_labels = [str(item) for item in encoder.classes_]
    else:
        y_train = split.y_train.astype(float).to_numpy(dtype=float)
        family = "gaussian"
        n_classes = -1
        class_labels = []

    train = runner._to_polars_with_target(split.X_train, y_train)
    test = runner._to_polars_features(split.X_test)

    import destyler as dst

    distiller = dst.Distiller(
        train,
        TARGET_COL,
        family,
        cat_features=split.cat_features,
        seed=case_seed,
    )

    started = time.perf_counter()
    distiller.train_teacher(
        depth=benchmark_args.teacher_depth,
        iterations=benchmark_args.iterations,
        learning_rate=benchmark_args.learning_rate,
        l2_leaf_reg=benchmark_args.l2_leaf_reg,
        one_hot_max_size=getattr(benchmark_args, "one_hot_max_size", 2),
        early_stopping_rounds=benchmark_args.early_stopping_rounds,
        validation_fraction=benchmark_args.validation_fraction,
        thread_count=benchmark_args.threads,
        teacher_ensemble=getattr(benchmark_args, "teacher_ensemble", 1),
    )
    phase["teacher_s"] = time.perf_counter() - started

    started = time.perf_counter()
    distiller.decompose(max_order=benchmark_args.max_order, decomposition_measure="row")
    phase["decompose_s"] = time.perf_counter() - started

    started = time.perf_counter()
    distiller.propose(
        top_k_main=benchmark_args.top_k_main,
        top_2way=benchmark_args.top_2way,
        top_3way=benchmark_args.top_3way,
        min_importance_share=benchmark_args.min_importance_share,
        min_interaction_energy_share=benchmark_args.min_interaction_energy_share,
        cat_group_threshold=benchmark_args.cat_group_threshold,
        cat_te_threshold=benchmark_args.cat_te_threshold,
        frozen_column_budget=benchmark_args.frozen_column_budget,
    )
    phase["propose_s"] = time.perf_counter() - started

    base_recipe = distiller.recipe
    if base_recipe is None:
        raise RuntimeError("destyler did not produce a recipe")

    _write_json(case_dir / "base_recipe.json", base_recipe.to_dict())
    _write_json(
        case_dir / "case_metadata.json",
        {
            "row": asdict(row),
            "family": family,
            "case_seed": case_seed,
            "n_classes": n_classes,
            "class_labels": class_labels,
            "n_train": len(split.X_train),
            "n_test": len(split.X_test),
            "n_features": split.X_train.shape[1],
            "cat_features": split.cat_features,
            "feature_map": split.feature_map,
            "numeric_missing_train": split.numeric_missing_train,
            "numeric_missing_test": split.numeric_missing_test,
            "categorical_grouped_train": split.categorical_grouped_train,
            "categorical_grouped_test": split.categorical_grouped_test,
            "capped_categorical_features": split.capped_categorical_features,
            "phase_timings": phase,
            "benchmark_args": vars(benchmark_args),
        },
    )

    target_summaries = []
    for target in targets:
        target_summaries.append(
            _prepare_target_artifact(
                target=target,
                target_dir=case_dir / target,
                runner=runner,
                dst_api=dst_api,
                guard_sparse_interactions=guard_sparse_interactions,
                distiller=distiller,
                base_recipe=base_recipe,
                train=train,
                test=test,
                y_train_actual=y_train,
                case_seed=case_seed,
                benchmark_args=benchmark_args,
                counterfactual_anchor_weight=counterfactual_anchor_weight,
                n_alphas=n_alphas,
                repeat_fit=repeat_fit,
            )
        )

    summary = {
        "dataset": row.dataset_name,
        "task": row.task,
        "task_id": row.task_id,
        "repeat": row.repeat,
        "fold": row.fold,
        "split_index": row.seed,
        "problem_type": row.problem_type,
        "case_dir": str(case_dir),
        "phase_timings": phase,
        "targets": target_summaries,
    }
    _write_json(case_dir / "summary.json", summary)
    return summary


def _write_summary_csv(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    rows = []
    for summary in summaries:
        for target in summary["targets"]:
            rows.append(
                {
                    "dataset": summary["dataset"],
                    "task_id": summary["task_id"],
                    "repeat": summary["repeat"],
                    "fold": summary["fold"],
                    "split_index": summary["split_index"],
                    "problem_type": summary["problem_type"],
                    "target": target["target"],
                    "fit_train_rows": target["fit_train_rows"],
                    "fit_train_columns": target["fit_train_columns"],
                    "recipe_terms": target["recipe_terms"],
                    "recipe_interactions": target["recipe_interactions"],
                    "requested_regularization": target["requested_regularization"],
                    "effective_regularization": target["effective_regularization"],
                    "direct_fit_median_s": target["direct_fit_median_s"],
                    "direct_predict_median_s": target["direct_predict_median_s"],
                    "teacher_s": summary["phase_timings"].get("teacher_s"),
                    "decompose_s": summary["phase_timings"].get("decompose_s"),
                    "propose_s": summary["phase_timings"].get("propose_s"),
                    "with_teacher_complement_s": target["phase_timings"].get(
                        "with_teacher_complement_s"
                    ),
                    "sparse_guard_s": target["phase_timings"].get("sparse_guard_s"),
                    "teacher_target_s": target["phase_timings"].get("teacher_target_s"),
                    "anchor_rows_s": target["phase_timings"].get("anchor_rows_s"),
                    "case_dir": summary["case_dir"],
                    "target_dir": target["target_dir"],
                }
            )
    if not rows:
        return
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destyler-root", type=Path, default=_default_destyler_root())
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--first-run-manifest", type=Path, default=DEFAULT_FIRST_RUN_MANIFEST)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--all-splits-for-datasets", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=["teacher", "actual"],
        default=["teacher"],
        help="teacher reproduces the first benchmark student; actual uses the same recipe on labels.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repeat-fit", type=int, default=1)
    parser.add_argument("--n-alphas", type=int, default=20)
    parser.add_argument(
        "--counterfactual-anchor-weight",
        type=float,
        default=0.02,
        help="Destyler default for teacher-fit students in the first benchmark.",
    )
    parser.add_argument(
        "--no-first-run-manifest",
        action="store_true",
        help="Use current benchmark defaults instead of full_20260617_1 manifest args.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    try:
        runner = _load_tabarena_runner(cli.destyler_root)
        dst_api, guard_sparse_interactions = _load_destyler_private_modules(cli.destyler_root)
    except ModuleNotFoundError as exc:
        raise SystemExit(str(exc)) from None
    manifest_path = None if cli.no_first_run_manifest else cli.first_run_manifest
    benchmark_args = _load_manifest_args(runner, manifest_path)
    metadata_path = cli.metadata or benchmark_args.metadata
    rows = _select_case_rows(
        runner,
        metadata_path,
        cases_path=cli.cases,
        datasets=set(cli.datasets) if cli.datasets else None,
        all_splits_for_datasets=cli.all_splits_for_datasets,
        limit=cli.limit,
    )
    if (
        benchmark_args.early_stopping_rounds is not None
        and benchmark_args.early_stopping_rounds <= 0
    ):
        benchmark_args.early_stopping_rounds = None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = cli.output_dir or Path(__file__).with_name("artifacts") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "run_config.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "destyler_root": cli.destyler_root,
            "metadata": metadata_path,
            "cases": cli.cases,
            "targets": cli.targets,
            "repeat_fit": cli.repeat_fit,
            "n_alphas": cli.n_alphas,
            "counterfactual_anchor_weight": cli.counterfactual_anchor_weight,
            "first_run_manifest": manifest_path,
            "benchmark_args": vars(benchmark_args),
        },
    )

    summaries = []
    for index, row in enumerate(rows, start=1):
        print(
            f"[{index}/{len(rows)}] {row.dataset_name} {row.seed} "
            f"({row.problem_type}, task_id={row.task_id})",
            flush=True,
        )
        summary = _prepare_case(
            row=row,
            runner=runner,
            dst_api=dst_api,
            guard_sparse_interactions=guard_sparse_interactions,
            output_dir=output_dir,
            benchmark_args=benchmark_args,
            targets=cli.targets,
            counterfactual_anchor_weight=cli.counterfactual_anchor_weight,
            n_alphas=cli.n_alphas,
            repeat_fit=cli.repeat_fit,
        )
        summaries.append(summary)
        print(f"  wrote {summary['case_dir']}", flush=True)

    _write_json(output_dir / "summaries.json", summaries)
    _write_summary_csv(output_dir, summaries)
    print(f"output_dir={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
