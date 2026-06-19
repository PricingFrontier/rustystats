#!/usr/bin/env python3
"""Time RustyStats fits from prepared destyler TabArena artifacts."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


def _rustystats_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_local_rustystats() -> None:
    path = str(_rustystats_root() / "python")
    if path not in sys.path:
        sys.path.insert(0, path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_dirs(root: Path, targets: set[str] | None) -> list[Path]:
    if (root / "glm_kwargs.json").exists():
        return [root]
    dirs = []
    for path in sorted(root.rglob("glm_kwargs.json")):
        target_dir = path.parent
        if targets and target_dir.name not in targets:
            continue
        dirs.append(target_dir)
    return dirs


def _fit_once(
    *,
    data: pl.DataFrame,
    glm_kwargs: dict[str, Any],
    fit_config: dict[str, Any],
    compute_covariance: bool,
) -> Any:
    import rustystats as rs

    seed = fit_config.get("seed")
    builder = rs.glm_dict(**glm_kwargs, data=data, seed=seed)
    fit_kwargs = dict(fit_config.get("fit_kwargs") or {})
    if compute_covariance is not None:
        fit_kwargs["compute_covariance"] = compute_covariance
    return builder.fit(**fit_kwargs)


def _measure_peak_mb(func) -> tuple[Any, float]:
    try:
        import psutil
    except ImportError:
        result = func()
        return result, float("nan")

    process = psutil.Process()
    before = process.memory_info().rss
    peak = before
    result = func()
    peak = max(peak, process.memory_info().rss)
    return result, (peak - before) / (1024 * 1024)


def _time_artifact(
    target_dir: Path,
    *,
    repeats: int,
    warmup: int,
    predict: bool,
    compute_covariance: bool,
    measure_memory: bool,
) -> list[dict[str, Any]]:
    glm_kwargs = _load_json(target_dir / "glm_kwargs.json")
    fit_config = _load_json(target_dir / "fit_config.json")
    metadata_path = target_dir.parent / "case_metadata.json"
    metadata = _load_json(metadata_path) if metadata_path.exists() else {}

    started = time.perf_counter()
    train = pl.read_parquet(target_dir / "train.parquet")
    test = (
        pl.read_parquet(target_dir / "test.parquet")
        if (target_dir / "test.parquet").exists()
        else None
    )
    load_s = time.perf_counter() - started

    for _ in range(warmup):
        _ = _fit_once(
            data=train,
            glm_kwargs=glm_kwargs,
            fit_config=fit_config,
            compute_covariance=compute_covariance,
        )
        gc.collect()

    rows = []
    for repeat in range(repeats):
        gc.collect()
        started = time.perf_counter()
        if measure_memory:
            model, peak_mb = _measure_peak_mb(
                lambda: _fit_once(
                    data=train,
                    glm_kwargs=glm_kwargs,
                    fit_config=fit_config,
                    compute_covariance=compute_covariance,
                )
            )
        else:
            model = _fit_once(
                data=train,
                glm_kwargs=glm_kwargs,
                fit_config=fit_config,
                compute_covariance=compute_covariance,
            )
            peak_mb = float("nan")
        fit_s = time.perf_counter() - started

        predict_s = None
        if predict and test is not None:
            started = time.perf_counter()
            _ = model.predict(test)
            predict_s = time.perf_counter() - started

        rows.append(
            {
                "artifact_dir": str(target_dir),
                "dataset": metadata.get("row", {}).get("dataset_name", target_dir.parent.name),
                "task_id": metadata.get("row", {}).get("task_id"),
                "tabarena_repeat": metadata.get("row", {}).get("repeat"),
                "fold": metadata.get("row", {}).get("fold"),
                "split_index": metadata.get("row", {}).get("split_index"),
                "problem_type": metadata.get("row", {}).get("problem_type"),
                "target": fit_config.get("target", target_dir.name),
                "requested_regularization": fit_config.get("requested_regularization"),
                "effective_regularization": fit_config.get("effective_regularization"),
                "n_train": train.height,
                "n_columns": train.width,
                "n_params": len(model.params),
                "load_s": load_s if repeat == 0 else 0.0,
                "fit_s": fit_s,
                "predict_s": predict_s,
                "peak_fit_mb": peak_mb,
                "run_index": repeat,
                "compute_covariance": compute_covariance,
            }
        )
    return rows


def _write_outputs(output: Path, rows: list[dict[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output.with_name(output.stem + "_summary.csv")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["dataset"],
            row["task_id"],
            row["split_index"],
            row["target"],
            row["effective_regularization"],
            row["compute_covariance"],
        )
        groups.setdefault(key, []).append(row)
    summary_rows = []
    for key, values in groups.items():
        fit_times = np.array([float(v["fit_s"]) for v in values], dtype=float)
        predict_values = [v["predict_s"] for v in values if v["predict_s"] is not None]
        predict_times = np.array(predict_values, dtype=float) if predict_values else np.array([])
        summary_rows.append(
            {
                "dataset": key[0],
                "task_id": key[1],
                "split_index": key[2],
                "target": key[3],
                "effective_regularization": key[4],
                "compute_covariance": key[5],
                "runs": len(values),
                "fit_min_s": float(np.min(fit_times)),
                "fit_median_s": float(np.median(fit_times)),
                "fit_max_s": float(np.max(fit_times)),
                "predict_median_s": (
                    float(np.median(predict_times)) if predict_times.size else None
                ),
                "n_train": values[0]["n_train"],
                "n_columns": values[0]["n_columns"],
                "n_params": values[0]["n_params"],
                "artifact_dir": values[0]["artifact_dir"],
            }
        )
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--targets", nargs="*", choices=["teacher", "actual"], default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--skip-covariance", action="store_true")
    parser.add_argument("--measure-memory", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _ensure_local_rustystats()
    target_dirs = _artifact_dirs(args.artifact_root, set(args.targets) if args.targets else None)
    if not target_dirs:
        raise SystemExit(f"No prepared artifacts found under {args.artifact_root}")
    rows = []
    for index, target_dir in enumerate(target_dirs, start=1):
        print(f"[{index}/{len(target_dirs)}] timing {target_dir}", flush=True)
        rows.extend(
            _time_artifact(
                target_dir,
                repeats=args.repeats,
                warmup=args.warmup,
                predict=args.predict,
                compute_covariance=not args.skip_covariance,
                measure_memory=args.measure_memory,
            )
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = args.output or args.artifact_root / f"rustystats_direct_timings_{timestamp}.csv"
    _write_outputs(output, rows)
    print(f"wrote {output}", flush=True)
    print(f"wrote {output.with_name(output.stem + '_summary.csv')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
