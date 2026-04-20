"""
Track memory footprint of every section of result.diagnostics() on a synthetic
1M-row, 30-feature dataset. Mirrors bench_diagnostics_timing.py so memory and
timing tables can be compared section-by-section.

Usage:
  uv run python benchmarks/bench_diagnostics_memory.py
  uv run python benchmarks/bench_diagnostics_memory.py --repeat 5
  uv run python benchmarks/bench_diagnostics_memory.py --save before.json
  uv run python benchmarks/bench_diagnostics_memory.py --save after.json
  uv run python benchmarks/bench_diagnostics_memory.py compare before.json after.json

Memory accounting:
  - rss_peak_above_baseline: max VmRSS observed during the call minus VmRSS
    just before the call. Captures transient peaks even if memory comes back
    after the call returns. Includes Rust allocations.
  - rss_delta: VmRSS after - VmRSS before. Captures lasting growth (caches,
    objects retained on the result). Can be negative if memory is freed.
  - py_peak: peak Python heap (tracemalloc) during the call. Python-side only.

With --repeat N (N>=2), the diagnostics call runs N times. The first iteration
is discarded as warmup; the remaining are reduced to median + spread (max-min)
per section. Use this to detect optimizations smaller than the noise floor.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import functools
import json
import statistics
import subprocess
import sys
import threading
import time
import tracemalloc
from pathlib import Path

import numpy as np
import polars as pl
import rustystats as rs
from rustystats.diagnostics import DiagnosticsComputer
from rustystats.diagnostics import api as diag_api
from rustystats.diagnostics.factors import _FactorDiagnosticsComputer


def _read_vm_rss_kb() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0


class _RssSampler(threading.Thread):
    def __init__(self, interval_s: float = 0.005):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.peak_kb = _read_vm_rss_kb()
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            v = _read_vm_rss_kb()
            if v > self.peak_kb:
                self.peak_kb = v
            self._stop.wait(self.interval_s)

    def stop(self) -> int:
        self._stop.set()
        self.join()
        return self.peak_kb


class MemTracker:
    def __init__(self):
        self.records: list[dict] = []

    def wrap(self, name: str, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            rss_before = _read_vm_rss_kb()
            tracemalloc.reset_peak()
            py_before, _ = tracemalloc.get_traced_memory()
            sampler = _RssSampler()
            sampler.start()
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - t0
                rss_peak = sampler.stop()
                rss_after = _read_vm_rss_kb()
                py_current, py_peak = tracemalloc.get_traced_memory()
                self.records.append(
                    {
                        "name": name,
                        "elapsed_s": elapsed,
                        "rss_before_kb": rss_before,
                        "rss_after_kb": rss_after,
                        "rss_peak_kb": rss_peak,
                        "rss_delta_kb": rss_after - rss_before,
                        "rss_peak_above_baseline_kb": max(0, rss_peak - rss_before),
                        "py_peak_kb": (py_peak - py_before) // 1024,
                        "py_delta_kb": (py_current - py_before) // 1024,
                    }
                )

        return wrapper

    def aggregate(self) -> list[dict]:
        agg: dict[str, dict] = {}
        for r in self.records:
            a = agg.setdefault(
                r["name"],
                {
                    "name": r["name"],
                    "calls": 0,
                    "elapsed_s": 0.0,
                    "rss_peak_above_baseline_kb": 0,
                    "rss_delta_kb": 0,
                    "py_peak_kb": 0,
                    "py_delta_kb": 0,
                },
            )
            a["calls"] += 1
            a["elapsed_s"] += r["elapsed_s"]
            # Take the worst peak across calls; sum the deltas.
            a["rss_peak_above_baseline_kb"] = max(
                a["rss_peak_above_baseline_kb"], r["rss_peak_above_baseline_kb"]
            )
            a["rss_delta_kb"] += r["rss_delta_kb"]
            a["py_peak_kb"] = max(a["py_peak_kb"], r["py_peak_kb"])
            a["py_delta_kb"] += r["py_delta_kb"]
        return sorted(agg.values(), key=lambda d: -d["rss_peak_above_baseline_kb"])

    def reset(self):
        self.records.clear()


def reduce_iterations(iter_aggs: list[list[dict]]) -> list[dict]:
    """Reduce per-iteration aggregates → median+spread per section.

    iter_aggs[i] is the result of MemTracker.aggregate() for iteration i.
    Returns a section-keyed list with median values and spread (max-min) for
    the metrics that benefit from it.
    """
    n_iters = len(iter_aggs)
    by_name: dict[str, list[dict]] = {}
    for it in iter_aggs:
        for sec in it:
            by_name.setdefault(sec["name"], []).append(sec)

    out: list[dict] = []
    for name, secs in by_name.items():
        peaks = [s["rss_peak_above_baseline_kb"] for s in secs]
        deltas = [s["rss_delta_kb"] for s in secs]
        py_peaks = [s["py_peak_kb"] for s in secs]
        elapsed = [s["elapsed_s"] for s in secs]
        calls = [s["calls"] for s in secs]
        out.append(
            {
                "name": name,
                "iterations": len(secs),
                "calls": int(statistics.median(calls)),
                "elapsed_s": statistics.median(elapsed),
                "rss_peak_above_baseline_kb": statistics.median(peaks),
                "rss_peak_above_baseline_kb_spread": (max(peaks) - min(peaks))
                if n_iters > 1
                else 0,
                "rss_delta_kb": statistics.median(deltas),
                "rss_delta_kb_spread": (max(deltas) - min(deltas)) if n_iters > 1 else 0,
                "py_peak_kb": statistics.median(py_peaks),
            }
        )
    out.sort(key=lambda d: -d["rss_peak_above_baseline_kb"])
    return out


def print_report(sections: list[dict], n_iterations: int, top_n: int | None = None):
    rows = sections[:top_n] if top_n is not None else sections
    show_spread = n_iterations > 1
    if show_spread:
        hdr = (
            f"{'Section':<55} {'Calls':>5} {'Time(s)':>8} "
            f"{'Peak↑MB':>9} {'±MB':>7} {'Δ MB':>8} {'±MB':>7} {'PyPeak':>8}"
        )
    else:
        hdr = (
            f"{'Section':<55} {'Calls':>5} {'Time(s)':>8} "
            f"{'Peak↑MB':>9} {'Δ MB':>8} {'PyPeak MB':>10}"
        )
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        if show_spread:
            print(
                f"{r['name']:<55} {r['calls']:>5} {r['elapsed_s']:>8.2f} "
                f"{r['rss_peak_above_baseline_kb']/1024:>9.1f} "
                f"{r['rss_peak_above_baseline_kb_spread']/1024:>7.1f} "
                f"{r['rss_delta_kb']/1024:>+8.1f} "
                f"{r['rss_delta_kb_spread']/1024:>7.1f} "
                f"{r['py_peak_kb']/1024:>8.1f}"
            )
        else:
            print(
                f"{r['name']:<55} {r['calls']:>5} {r['elapsed_s']:>8.2f} "
                f"{r['rss_peak_above_baseline_kb']/1024:>9.1f} "
                f"{r['rss_delta_kb']/1024:>+8.1f} "
                f"{r['py_peak_kb']/1024:>10.1f}"
            )
    print("-" * len(hdr))
    print(
        "Peak↑MB = max RSS during the call − RSS just before "
        "(transient, captures Rust allocs).\n"
        "Δ MB    = RSS after − RSS before (lasting growth; negative = freed).\n"
        "PyPeak  = tracemalloc peak Python heap during the call (Python only)."
    )
    if show_spread:
        print(
            f"±MB     = spread (max−min) across {n_iterations} iterations "
            "(after discarding warmup). Treat changes smaller than ± as noise."
        )


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return "unknown"


def install_patches(tracker: MemTracker, result):
    for method_name in [
        "compute_calibration",
        "compute_residual_summary",
        "compute_factor_diagnostics",
        "detect_interactions",
        "compute_model_comparison",
        "compute_dataset_diagnostics",
        "generate_warnings",
        "compute_vif",
        "compute_coefficient_summary",
        "enrich_coefficient_summary_with_robust",
        "compute_factor_deviance",
        "compute_lift_chart",
        "compute_partial_dependence",
        "compute_base_predictions_comparison",
    ]:
        if hasattr(DiagnosticsComputer, method_name):
            original = getattr(DiagnosticsComputer, method_name)
            setattr(
                DiagnosticsComputer,
                method_name,
                tracker.wrap(f"Computer.{method_name}", original),
            )

    if hasattr(diag_api, "_factorize_strings"):
        diag_api._factorize_strings = tracker.wrap(
            "_factorize_strings (cache loop)", diag_api._factorize_strings
        )

    for method_name in [
        "_compute_ae_categorical",
        "_compute_ae_continuous",
        "_compute_residual_pattern_categorical",
        "_compute_residual_pattern_continuous",
        "_compute_score_test_categorical",
        "_compute_score_test_continuous",
        "compute_factor_significance",
        "_get_factor_coefficients",
        "_get_transformation",
    ]:
        if hasattr(_FactorDiagnosticsComputer, method_name):
            original = getattr(_FactorDiagnosticsComputer, method_name)
            setattr(
                _FactorDiagnosticsComputer,
                method_name,
                tracker.wrap(f"_Factors.{method_name}", original),
            )

    if hasattr(result, "predict"):
        result.predict = tracker.wrap("result.predict", result.predict)
    for attr in [
        "bse_robust",
        "tvalues_robust",
        "pvalues_robust",
        "get_design_matrix",
        "get_bread_matrix",
        "get_irls_weights",
        "pearson_chi2",
        "scale",
        "scale_pearson",
        "null_deviance",
    ]:
        if hasattr(result, attr):
            attr_val = getattr(result, attr)
            if callable(attr_val):
                setattr(result, attr, tracker.wrap(f"result.{attr}", attr_val))


def make_dataset(n: int = 1_000_000, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    cat_cards = {"cat0": 10, "cat1": 50, "cat2": 4, "cat3": 15, "cat4": 8, "cat5": 200}
    cat_cols = {name: rng.integers(0, k, n) for name, k in cat_cards.items()}
    cont_cols = {f"cont{i}": rng.standard_normal(n).astype(np.float64) for i in range(24)}
    eta = (
        -2.0
        + 0.10 * cat_cols["cat0"] / 10.0
        - 0.05 * cat_cols["cat1"] / 50.0
        + 0.20 * cont_cols["cont0"]
        - 0.15 * cont_cols["cont1"]
        + 0.05 * cont_cols["cont2"] ** 2
        + 0.05 * cont_cols["cont3"]
    )
    exposure = rng.uniform(0.1, 1.0, n).astype(np.float64)
    mu = np.exp(eta) * exposure
    y = rng.poisson(mu).astype(np.float64)
    return pl.DataFrame(
        {
            "y": y,
            "exposure": exposure,
            **{name: arr.astype(str) for name, arr in cat_cols.items()},
            **cont_cols,
        }
    )


def run_benchmark(n_rows: int, repeat: int = 1) -> tuple[list[dict], dict]:
    tracemalloc.start()
    rss_at_start = _read_vm_rss_kb()

    print(f"Generating synthetic data: {n_rows:,} rows × 30 features (6 cat + 24 cont)...")
    t0 = time.perf_counter()
    data = make_dataset(n=n_rows)
    print(
        f"  built in {time.perf_counter() - t0:.2f}s, shape={data.shape}, "
        f"RSS now {(_read_vm_rss_kb()-rss_at_start)/1024:.1f} MB above start"
    )

    terms: dict[str, dict] = {}
    for i in range(10):
        terms[f"cont{i}"] = {"type": "linear"}
    for i in range(10, 14):
        terms[f"cont{i}"] = {"type": "bs", "df": 4}
    for name in ["cat0", "cat1", "cat2", "cat3", "cat4"]:
        terms[name] = {"type": "categorical"}
    terms["cat5"] = {"type": "target_encoding"}

    print("\nFitting Poisson GLM...")
    t0 = time.perf_counter()
    result = rs.glm_dict(
        response="y",
        terms=terms,
        data=data,
        family="poisson",
        offset="exposure",
    ).fit()
    print(
        f"  fitted in {time.perf_counter() - t0:.2f}s, params={len(result.params)}, "
        f"RSS now {(_read_vm_rss_kb()-rss_at_start)/1024:.1f} MB above start"
    )

    tracker = MemTracker()
    install_patches(tracker, result)

    cat_factors = [f"cat{i}" for i in range(6)]
    cont_factors = [f"cont{i}" for i in range(24)]

    print(
        f"\nRunning diagnostics: {len(cat_factors)} cat factors + "
        f"{len(cont_factors)} cont factors"
        f"{f' × {repeat} iterations' if repeat > 1 else ''}..."
    )

    iter_aggs: list[list[dict]] = []
    iter_totals: list[float] = []
    iter_rss_before: list[int] = []
    iter_rss_after: list[int] = []
    diag = None
    for i in range(repeat):
        tracker.reset()
        rss_before_diag = _read_vm_rss_kb()
        t0 = time.perf_counter()
        diag = result.diagnostics(
            train_data=data,
            categorical_factors=cat_factors,
            continuous_factors=cont_factors,
        )
        total = time.perf_counter() - t0
        rss_after_diag = _read_vm_rss_kb()
        tag = "warmup" if (repeat > 1 and i == 0) else f"iter {i+1}/{repeat}"
        print(f"  [{tag}] {total:.3f}s, " f"RSS Δ {(rss_after_diag-rss_before_diag)/1024:+.1f} MB")
        iter_aggs.append(tracker.aggregate())
        iter_totals.append(total)
        iter_rss_before.append(rss_before_diag)
        iter_rss_after.append(rss_after_diag)

    measured = iter_aggs[1:] if repeat > 1 else iter_aggs
    measured_totals = iter_totals[1:] if repeat > 1 else iter_totals
    measured_before = iter_rss_before[1:] if repeat > 1 else iter_rss_before
    measured_after = iter_rss_after[1:] if repeat > 1 else iter_rss_after
    sections = reduce_iterations(measured)
    print_report(sections, n_iterations=len(measured))

    print(
        f"\nDiagnostics ok: n_obs={diag.model_summary['n_obs']}, "
        f"n_factors_analyzed={len(diag.factors)}"
    )

    summary = {
        "git_sha": _git_sha(),
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "n_rows": n_rows,
        "n_factors_cat": len(cat_factors),
        "n_factors_cont": len(cont_factors),
        "iterations_total": repeat,
        "iterations_measured": len(measured),
        "rss_at_start_kb": rss_at_start,
        "rss_before_diag_kb": int(statistics.median(measured_before)),
        "rss_after_diag_kb": int(statistics.median(measured_after)),
        "diagnostics_elapsed_s": statistics.median(measured_totals),
    }
    return sections, summary


def cmd_run(args: argparse.Namespace) -> int:
    sections, summary = run_benchmark(args.rows, repeat=args.repeat)
    if args.save:
        out = {"summary": summary, "sections": sections}
        Path(args.save).write_text(json.dumps(out, indent=2))
        print(f"\nSaved results to {args.save}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    a = json.loads(Path(args.before).read_text())
    b = json.loads(Path(args.after).read_text())

    print(
        f"\nBEFORE: {args.before}  ({a['summary'].get('git_sha','?')}, "
        f"{a['summary'].get('timestamp','?')})"
    )
    print(
        f"AFTER:  {args.after}  ({b['summary'].get('git_sha','?')}, "
        f"{b['summary'].get('timestamp','?')})"
    )
    print(f"Dataset: {a['summary'].get('n_rows','?')} rows")

    by_name_a = {s["name"]: s for s in a["sections"]}
    by_name_b = {s["name"]: s for s in b["sections"]}
    all_names = sorted(set(by_name_a) | set(by_name_b))

    def _peak_spread(s: dict | None) -> float:
        if not s:
            return 0.0
        return s.get("rss_peak_above_baseline_kb_spread", 0) / 1024

    rows = []
    for name in all_names:
        sa = by_name_a.get(name)
        sb = by_name_b.get(name)
        peak_a = (sa["rss_peak_above_baseline_kb"] / 1024) if sa else 0.0
        peak_b = (sb["rss_peak_above_baseline_kb"] / 1024) if sb else 0.0
        delta_a = (sa["rss_delta_kb"] / 1024) if sa else 0.0
        delta_b = (sb["rss_delta_kb"] / 1024) if sb else 0.0
        time_a = sa["elapsed_s"] if sa else 0.0
        time_b = sb["elapsed_s"] if sb else 0.0
        # Combined noise floor: the larger spread between A and B, doubled
        # (rough rule for "is the difference of two noisy medians significant?").
        noise = max(_peak_spread(sa), _peak_spread(sb)) * 2
        sig = "*" if abs(peak_b - peak_a) > noise and noise > 0 else " "
        rows.append(
            (
                name,
                peak_a,
                peak_b,
                peak_b - peak_a,
                sig,
                delta_a,
                delta_b,
                delta_b - delta_a,
                time_a,
                time_b,
            )
        )

    rows.sort(key=lambda r: r[3])  # smallest peak delta first → biggest savings up top

    hdr = (
        f"{'Section':<55} {'PeakA':>7} {'PeakB':>7} {'ΔPeak':>8} {'sig':>4} "
        f"{'DelA':>7} {'DelB':>7} {'ΔDel':>8} {'tA':>6} {'tB':>6}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        name, pa, pb, dp, sig, da, db, dd, ta, tb = r
        print(
            f"{name:<55} {pa:>7.1f} {pb:>7.1f} {dp:>+8.1f} {sig:>4} "
            f"{da:>+7.1f} {db:>+7.1f} {dd:>+8.1f} {ta:>6.2f} {tb:>6.2f}"
        )
    print("-" * len(hdr))
    print(
        "All MB. ΔPeak/ΔDel < 0 means AFTER used less memory than BEFORE. "
        "Sorted by ΔPeak ascending (biggest wins first).\n"
        "sig='*' means |ΔPeak| exceeds 2× the larger spread (likely real, not noise). "
        "Blank when either run had no spread data (single iteration)."
    )

    total_a = a["summary"].get("diagnostics_elapsed_s", 0.0)
    total_b = b["summary"].get("diagnostics_elapsed_s", 0.0)
    rss_growth_a = (
        a["summary"].get("rss_after_diag_kb", 0) - a["summary"].get("rss_before_diag_kb", 0)
    ) / 1024
    rss_growth_b = (
        b["summary"].get("rss_after_diag_kb", 0) - b["summary"].get("rss_before_diag_kb", 0)
    ) / 1024
    print(
        f"\nTotal diagnostics elapsed:  {total_a:.2f}s → {total_b:.2f}s "
        f"({total_b-total_a:+.2f}s)"
    )
    print(
        f"Total RSS growth across diag: {rss_growth_a:+.1f} MB → "
        f"{rss_growth_b:+.1f} MB ({rss_growth_b-rss_growth_a:+.1f} MB)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="run the benchmark (default)")
    p_run.add_argument("--rows", type=int, default=1_000_000)
    p_run.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="number of diagnostics iterations; first is "
        "discarded as warmup, rest reduced to median",
    )
    p_run.add_argument("--save", type=str, default=None, help="write JSON results to this path")
    p_run.set_defaults(func=cmd_run)

    p_cmp = sub.add_parser("compare", help="diff two saved JSON runs")
    p_cmp.add_argument("before")
    p_cmp.add_argument("after")
    p_cmp.set_defaults(func=cmd_compare)

    parser.add_argument(
        "--rows", type=int, default=1_000_000, help="(when no subcommand) rows to generate"
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="(when no subcommand) iterations to run"
    )
    parser.add_argument(
        "--save", type=str, default=None, help="(when no subcommand) write JSON results to path"
    )

    args = parser.parse_args(argv)
    if args.cmd is None:
        return cmd_run(args)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
