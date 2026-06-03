"""Profile the packaged RustyStats performance repro.

Run from the repository root:

    ./.venv/bin/python rustystats-performance/profile_rustystats_fit.py

The script deliberately avoids changing RustyStats internals. It wraps selected
Python and PyO3 entry points with lightweight timers, recreates the model build
from ``glm_kwargs.json`` and ``fit_config.json``, then writes a JSON timing
report beside the repro bundle.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import inspect
import json
import os
import platform
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

DetailFn = Callable[..., dict[str, Any]]


class TimingRecorder:
    """Collect individual timings and aggregate them by name/detail label."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    @contextlib.contextmanager
    def span(self, name: str, **details: Any):
        start = time.perf_counter()
        status = "ok"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            self.events.append(
                {
                    "name": name,
                    "seconds": time.perf_counter() - start,
                    "status": status,
                    "details": _jsonable(details),
                }
            )

    def wrap_attr(
        self,
        owner: Any,
        attr: str,
        name: str | None = None,
        detail_fn: DetailFn | None = None,
    ) -> Callable[[], None]:
        """Wrap ``owner.attr`` and return a restorer."""

        raw_descriptor = inspect.getattr_static(owner, attr)
        is_staticmethod = isinstance(raw_descriptor, staticmethod)
        is_classmethod = isinstance(raw_descriptor, classmethod)
        original = getattr(owner, attr)
        event_name = name or f"{owner}.{attr}"

        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any):
            details: dict[str, Any] = {}
            if detail_fn is not None:
                try:
                    details = detail_fn(*args, **kwargs)
                except Exception as exc:  # pragma: no cover - profiling only
                    details = {"detail_error": repr(exc)}
            with self.span(event_name, **details):
                return original(*args, **kwargs)

        if is_staticmethod:
            replacement: Any = staticmethod(wrapped)
        elif is_classmethod:
            replacement = classmethod(wrapped)
        else:
            replacement = wrapped
        setattr(owner, attr, replacement)

        def restore() -> None:
            setattr(owner, attr, raw_descriptor)

        return restore

    def aggregates(self) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for event in self.events:
            details = event.get("details") or {}
            label = str(details.get("label", ""))
            key = (event["name"], label)
            row = grouped.setdefault(
                key,
                {
                    "name": event["name"],
                    "label": label,
                    "count": 0,
                    "seconds": 0.0,
                    "max_seconds": 0.0,
                    "statuses": defaultdict(int),
                },
            )
            seconds = float(event["seconds"])
            row["count"] += 1
            row["seconds"] += seconds
            row["max_seconds"] = max(row["max_seconds"], seconds)
            row["statuses"][event["status"]] += 1

        out = []
        for row in grouped.values():
            row = dict(row)
            row["statuses"] = dict(row["statuses"])
            row["seconds_per_call"] = row["seconds"] / row["count"]
            out.append(row)
        out.sort(key=lambda r: r["seconds"], reverse=True)
        return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _shape(arr: Any) -> list[int] | None:
    try:
        return list(arr.shape)
    except Exception:
        return None


def _interaction_label(interaction: Any) -> str:
    factors = getattr(interaction, "factors", None)
    if factors:
        return ":".join(map(str, factors))
    return repr(interaction)


def install_instrumentation(recorder: TimingRecorder) -> list[Callable[[], None]]:
    import rustystats._rustystats as rust
    import rustystats.formula as formula
    import rustystats.interactions as interactions
    import rustystats.regularization_path as reg_path

    restores: list[Callable[[], None]] = []
    add = restores.append

    add(recorder.wrap_attr(formula, "_collect_lazyframe", "glm.collect_lazyframe"))
    add(recorder.wrap_attr(formula, "dict_to_parsed_formula", "glm.dict_to_parsed_formula"))
    add(recorder.wrap_attr(formula, "validate_input_transforms", "glm.validate_input_transforms"))
    add(recorder.wrap_attr(formula, "compile_input_transforms", "glm.compile_input_transforms"))
    add(recorder.wrap_attr(formula, "apply_input_transforms", "glm.apply_input_transforms"))
    add(
        recorder.wrap_attr(
            formula,
            "_fit_glm_core",
            "fit._fit_glm_core",
            lambda *a, **_kw: {
                "label": f"alpha={float(a[8]):.6g},l1={float(a[9]):.3g}",
                "n": int(a[1].shape[0]),
                "p": int(a[1].shape[1]),
            },
        )
    )
    add(recorder.wrap_attr(formula, "_build_results", "fit._build_results"))

    glm_base = formula._GLMBase
    add(recorder.wrap_attr(glm_base, "_get_raw_exposure", "glm.process_raw_exposure"))
    add(recorder.wrap_attr(glm_base, "_process_offset", "glm.process_offset"))
    add(recorder.wrap_attr(glm_base, "_process_weights", "glm.process_weights"))
    add(recorder.wrap_attr(glm_base, "_process_complement", "glm.process_complement"))
    add(recorder.wrap_attr(glm_base, "_resolve_cv_path", "fit.resolve_cv_path"))
    add(recorder.wrap_attr(glm_base, "_has_target_encoding", "fit.has_target_encoding"))

    glm_dict_cls = formula.FormulaGLMDict
    add(recorder.wrap_attr(glm_dict_cls, "__init__", "glm.FormulaGLMDict.__init__"))
    add(recorder.wrap_attr(glm_dict_cls, "_build_formula_string", "glm.build_formula_string"))
    add(recorder.wrap_attr(glm_dict_cls, "fit", "fit.FormulaGLMDict.fit"))

    builder = interactions.InteractionBuilder
    add(
        recorder.wrap_attr(
            builder,
            "__init__",
            "design.InteractionBuilder.__init__",
            lambda _self, data, *_a, **_kw: {
                "rows": int(data.height),
                "columns": int(data.width),
            },
        )
    )
    add(recorder.wrap_attr(builder, "build_design_matrix_from_parsed", "design.build_matrix"))
    add(recorder.wrap_attr(builder, "_build_design_matrix_core", "design.build_matrix_core"))
    add(
        recorder.wrap_attr(
            builder,
            "_get_column",
            "design.get_column",
            lambda _self, column, *_a, **_kw: {"label": str(column)},
        )
    )
    add(
        recorder.wrap_attr(
            builder,
            "_get_categorical_encoding",
            "design.categorical_encoding",
            lambda _self, column, drop_first=True, *_a, **_kw: {
                "label": f"{column},drop_first={drop_first}"
            },
        )
    )
    add(
        recorder.wrap_attr(
            builder,
            "_build_spline_columns",
            "design.spline_columns",
            lambda _self, spline, *_a, **_kw: {
                "label": f"{getattr(spline, 'var_name', '?')}:{getattr(spline, 'spline_type', '?')}"
            },
        )
    )
    add(
        recorder.wrap_attr(
            builder,
            "build_interaction_columns",
            "design.interaction_columns",
            lambda _self, interaction, *_a, **_kw: {
                "label": _interaction_label(interaction),
                "kind": (
                    "continuous"
                    if getattr(interaction, "is_pure_continuous", False)
                    else "categorical"
                    if getattr(interaction, "is_pure_categorical", False)
                    else "mixed"
                ),
            },
        )
    )
    add(
        recorder.wrap_attr(
            builder,
            "_build_continuous_interaction",
            "design.interaction_continuous",
            lambda _self, interaction, *_a, **_kw: {"label": _interaction_label(interaction)},
        )
    )
    add(
        recorder.wrap_attr(
            builder,
            "_build_categorical_interaction",
            "design.interaction_categorical",
            lambda _self, interaction, *_a, **_kw: {"label": _interaction_label(interaction)},
        )
    )
    add(
        recorder.wrap_attr(
            builder,
            "_build_mixed_interaction",
            "design.interaction_mixed",
            lambda _self, interaction, *_a, **_kw: {"label": _interaction_label(interaction)},
        )
    )
    add(
        recorder.wrap_attr(
            builder,
            "_build_target_encoding_columns",
            "design.target_encoding_columns",
            lambda _self, term, *_a, **_kw: {"label": getattr(term, "var_name", "?")},
        )
    )
    add(
        recorder.wrap_attr(
            builder,
            "_build_frequency_encoding_columns",
            "design.frequency_encoding_columns",
            lambda _self, term, *_a, **_kw: {"label": getattr(term, "var_name", "?")},
        )
    )
    add(recorder.wrap_attr(builder, "_build_identity_columns", "design.identity_columns"))
    add(recorder.wrap_attr(builder, "_build_constraint_columns", "design.constraint_columns"))
    add(recorder.wrap_attr(builder, "_build_categorical_level_indicators", "design.cat_indicators"))
    add(
        recorder.wrap_attr(
            builder,
            "_stack_columns",
            "design.stack_columns",
            lambda columns, n_rows, dtype, *_a, **_kw: {
                "blocks": len(columns),
                "rows": int(n_rows),
                "dtype": str(dtype),
            },
        )
    )
    add(recorder.wrap_attr(builder, "clear_caches", "design.clear_caches"))

    add(recorder.wrap_attr(reg_path, "compute_standardization", "cv.compute_standardization"))
    add(
        recorder.wrap_attr(
            reg_path,
            "compute_alpha_max",
            "cv.compute_alpha_max",
            lambda X, y, l1_ratio, *_a, **_kw: {
                "shape": _shape(X),
                "l1_ratio": float(l1_ratio),
            },
        )
    )
    add(recorder.wrap_attr(reg_path, "generate_alpha_path", "cv.generate_alpha_path"))
    add(recorder.wrap_attr(reg_path, "select_optimal_alpha", "cv.select_optimal_alpha"))
    add(recorder.wrap_attr(reg_path, "fit_cv_regularization_path", "cv.fit_cv_regularization_path"))
    add(
        recorder.wrap_attr(
            reg_path, "fit_cv_te_regularization_path", "cv.fit_cv_te_regularization_path"
        )
    )

    add(
        recorder.wrap_attr(
            rust,
            "fit_cv_path_py",
            "rust.fit_cv_path_py",
            lambda y, x, family, *a, **kw: {
                "n": int(x.shape[0]),
                "p": int(x.shape[1]),
                "family": str(family),
                "alphas": len(kw.get("alphas") or (a[6] if len(a) > 6 else [])),
                "n_folds": int(kw.get("n_folds", a[8] if len(a) > 8 else -1)),
            },
        )
    )
    add(
        recorder.wrap_attr(
            rust,
            "fit_glm_py",
            "rust.fit_glm_py",
            lambda y, x, family, *a, **kw: {
                "label": f"alpha={float(kw.get('alpha', a[5] if len(a) > 5 else 0.0)):.6g}",
                "n": int(x.shape[0]),
                "p": int(x.shape[1]),
                "family": str(family),
            },
        )
    )

    return restores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the parquet and JSON repro files.",
    )
    parser.add_argument("--rows", type=int, default=None, help="Optional head-row sample.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to profile_rustystats_fit_latest.json in bundle-dir.",
    )
    parser.add_argument("--n-alphas", type=int, default=None, help="Override fit_config n_alphas.")
    parser.add_argument("--cv", type=int, default=None, help="Override fit_config cv.")
    parser.add_argument("--selection", default=None, help="Override fit_config selection.")
    parser.add_argument(
        "--regularization", default=None, help="Override fit_config regularization."
    )
    parser.add_argument("--max-iter", type=int, default=None, help="Override solver max_iter.")
    parser.add_argument("--tol", type=float, default=None, help="Override solver tol.")
    parser.add_argument(
        "--cv-max-iter",
        type=int,
        default=None,
        help="Override CV fold max_iter without changing the final refit.",
    )
    parser.add_argument(
        "--cv-tol",
        type=float,
        default=None,
        help="Override CV fold tol without changing the final refit.",
    )
    parser.add_argument(
        "--include-unregularized",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether CV includes alpha=0.",
    )
    parser.add_argument(
        "--store-design-matrix",
        action="store_true",
        help="Pass store_design_matrix=True to fit().",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle_dir = args.bundle_dir.resolve()
    output = args.output or bundle_dir / "profile_rustystats_fit_latest.json"
    recorder = TimingRecorder()

    import rustystats as rs

    restores = install_instrumentation(recorder)
    try:
        with recorder.span(
            "load.train_parquet", path=bundle_dir / "fremtpl2_50k_rustystats_train.parquet"
        ):
            train = pl.read_parquet(bundle_dir / "fremtpl2_50k_rustystats_train.parquet")
        if args.rows is not None:
            with recorder.span("load.sample_head", rows=args.rows):
                train = train.head(args.rows)

        with recorder.span("load.glm_kwargs_json", path=bundle_dir / "glm_kwargs.json"):
            glm_kwargs = json.loads((bundle_dir / "glm_kwargs.json").read_text())
        with recorder.span("load.fit_config_json", path=bundle_dir / "fit_config.json"):
            fit_config = json.loads((bundle_dir / "fit_config.json").read_text())

        fit_kwargs = {
            "regularization": fit_config.get("regularization"),
            "cv": fit_config.get("cv"),
            "selection": fit_config.get("selection", "min"),
            "n_alphas": fit_config.get("n_alphas"),
            "cv_seed": fit_config.get("seed", 0),
            "store_design_matrix": args.store_design_matrix,
        }
        for key, value in (
            ("regularization", args.regularization),
            ("cv", args.cv),
            ("selection", args.selection),
            ("n_alphas", args.n_alphas),
            ("max_iter", args.max_iter),
            ("tol", args.tol),
            ("cv_max_iter", args.cv_max_iter),
            ("cv_tol", args.cv_tol),
            ("include_unregularized", args.include_unregularized),
        ):
            if value is not None:
                fit_kwargs[key] = value

        with recorder.span("model.construct_glm_dict"):
            glm_builder = rs.glm_dict(**glm_kwargs, data=train, seed=fit_config.get("seed", 0))

        with recorder.span("model.fit"):
            fitted = glm_builder.fit(**fit_kwargs)

        result_summary = {
            "n_obs": int(getattr(fitted, "nobs", glm_builder.n_obs)),
            "n_params": len(fitted.params),
            "deviance": float(fitted.deviance),
            "iterations": int(getattr(fitted, "iterations", -1)),
            "converged": bool(getattr(fitted, "converged", False)),
            "selected_alpha": None
            if fitted.alpha is None
            else float(getattr(fitted, "alpha", np.nan)),
            "cv_deviance": None if fitted.cv_deviance is None else float(fitted.cv_deviance),
            "solver_status": getattr(fitted, "solver_status", None),
            "feature_count": len(getattr(fitted, "feature_names", [])),
        }

        report = {
            "command": sys.argv,
            "cwd": os.getcwd(),
            "platform": platform.platform(),
            "python": sys.version,
            "rustystats_version": getattr(rs, "__version__", None),
            "polars_version": pl.__version__,
            "bundle_dir": str(bundle_dir),
            "input_shape": {"rows": train.height, "columns": train.width},
            "fit_kwargs": _jsonable(fit_kwargs),
            "result": result_summary,
            "aggregates": recorder.aggregates(),
            "events": recorder.events,
        }
        output.write_text(json.dumps(report, indent=2))
    finally:
        for restore in reversed(restores):
            restore()

    print(f"Wrote {output}")
    print(f"Input shape: {train.height:,} rows x {train.width:,} columns")
    print(f"Fitted params: {result_summary['n_params']:,}")
    print("\nTop aggregate timings:")
    for row in report["aggregates"][:18]:
        label = f" [{row['label']}]" if row["label"] else ""
        print(f"{row['seconds']:9.3f}s  x{row['count']:<3d}  {row['name']}{label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
