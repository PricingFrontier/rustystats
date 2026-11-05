"""Out-of-core memory benchmarks T13 and T14 (``specs/out-of-core-fit.md`` §14).

T13 bounds an out-of-core fit's resident memory in a fresh process:

- (a) p ≈ 150: 30 linear terms, 5 categoricals with 4 to 64 levels, 2 splines;
- (b) the same plus one target-encoded term with 4 permutations;
- (c) cross-validation with that target-encoded term, K = 3.

Each runs at n and 4n rows. The rows come from a ``BatchSource`` that generates each
batch from its index, so no frame of the rows is ever held. The checks are:

- growth with n: peak(4n) − peak(n) ≤ max(16 MiB, 2% of peak(n));
- the plan holds: peak − baseline ≤ the plan's largest phase + persistent bytes + one
  source batch.

Each pair of runs fixes the source batch (at most 100,000 rows, and at most n / 2) and the
target-encoding sort run (n / 2 records), the same at n and 4n. Neither buffer depends on
n, so both have stopped growing by n.

T14 measures the in-memory design build at n rows (at least 400,000) with T13(a)'s
design, isolated from fitting (the peak is reset after the frame is built), at 1 and 8
threads. The check is rise ≤ 1.1 × 8np.

T18 runs lockstep ten-fold ridge CV at p ≈ 1,000 (a 1,000-level categorical) on 2
threads, first unbounded to read its plan, then in a fresh process under a
``max_bytes`` of exactly that plan (persistent bytes plus the largest phase). The
checks: the bounded fit is admitted, and its peak rise stays within the plan it was
admitted under. ``tests/python/test_out_of_core_contract.py`` runs it at p ≈ 300.

Usage::

    python benchmarks/bench_out_of_core_memory.py            # n = 2,000,000
    python benchmarks/bench_out_of_core_memory.py --smoke    # n = 64,000, for CI
    python benchmarks/bench_out_of_core_memory.py --output report.json

The exit status is non-zero when a check fails.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

MIB = 1024**2
CATEGORICAL_LEVELS = (4, 8, 16, 32, 64)
TE_LEVELS = 2_000
BATCH_ROWS = 100_000
MAX_BYTES = 512 * MIB
#: T18: the categorical's levels (p = levels + 1) and the rows.
T18_LEVELS = 1_000
T18_ROWS = 5_000


def _terms(with_te: bool) -> dict[str, dict[str, Any]]:
    terms: dict[str, dict[str, Any]] = {f"x{i}": {"type": "linear"} for i in range(30)}
    terms.update({f"c{i}": {"type": "categorical"} for i in range(len(CATEGORICAL_LEVELS))})
    terms["s0"] = {"type": "bs", "df": 6}
    terms["s1"] = {"type": "ns", "df": 6}
    if with_te:
        terms["t0"] = {"type": "target_encoding", "n_permutations": 4}
    return terms


def _batch(index: int, rows: int, with_te: bool) -> Any:
    """Batch ``index`` of the generated source: the same rows every time it is asked."""
    import numpy as np
    import polars as pl

    rng = np.random.default_rng(1_000 + index)
    columns: dict[str, Any] = {f"x{i}": rng.normal(size=rows) for i in range(30)}
    eta = 0.05 * columns["x0"] - 0.03 * columns["x1"]
    for i, levels in enumerate(CATEGORICAL_LEVELS):
        codes = rng.integers(0, levels, size=rows)
        columns[f"c{i}"] = pl.Series(codes).cast(pl.String).str.zfill(3)
        eta = eta + 0.01 * (codes % 3)
    columns["s0"] = rng.uniform(0.0, 10.0, size=rows)
    columns["s1"] = rng.uniform(-5.0, 5.0, size=rows)
    if with_te:
        columns["t0"] = pl.Series(rng.integers(0, TE_LEVELS, size=rows)).cast(pl.String)
    columns["y"] = rng.poisson(np.exp(-1.0 + eta)).astype(np.float64)
    return pl.DataFrame(columns)


def _source(n: int, with_te: bool, batch_rows: int) -> tuple[Any, int]:
    import rustystats as rs

    def open_batches(columns: list[str]):
        for index, start in enumerate(range(0, n, batch_rows)):
            yield _batch(index, min(batch_rows, n - start), with_te).select(columns)

    first = _batch(0, min(batch_rows, n), with_te)
    return rs.BatchSource(open_batches, first.schema), int(first.estimated_size())


def _status(field: str) -> int:
    with open("/proc/self/status", encoding="ascii") as status:
        for line in status:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    raise RuntimeError(f"/proc/self/status has no {field}")


def _reset_peak() -> None:
    """Reset VmHWM to the current RSS (Linux ``clear_refs`` value 5)."""
    with open("/proc/self/clear_refs", "w", encoding="ascii") as clear:
        clear.write("5")


def _t18_source(levels: int, n: int, batch_rows: int) -> Any:
    import numpy as np
    import polars as pl
    import rustystats as rs

    def batch(index: int, rows: int) -> pl.DataFrame:
        rng = np.random.default_rng(18_000 + index)
        level = rng.integers(0, levels, size=rows)
        return pl.DataFrame(
            {
                "x": rng.normal(size=rows),
                "level": pl.Series(level, dtype=pl.Int64),
                "y": rng.poisson(np.exp(0.1 + 0.3 / levels * level)).astype(np.float64),
            }
        )

    def open_batches(columns: list[str]):
        for index, start in enumerate(range(0, n, batch_rows)):
            yield batch(index, min(batch_rows, n - start)).select(columns)

    return rs.BatchSource(open_batches, batch(0, 1).schema), int(
        batch(0, min(batch_rows, n)).estimated_size()
    )


def _child(
    scenario: str,
    n: int,
    scratch: str,
    te_run_records: int,
    batch_rows: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import polars as pl
    import rustystats as rs

    if scenario == "t18":
        extra = extra or {}
        source, batch_bytes = _t18_source(extra["levels"], n, batch_rows)
        glm = rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}, "level": {"type": "categorical"}},
            data=source,
            family="poisson",
        )
        options = rs.OutOfCoreOptions(threads=2, max_bytes=extra.get("max_bytes"))
        _reset_peak()
        baseline = _status("VmRSS")
        started = time.perf_counter()
        result = glm.fit(cv=10, regularization="ridge", n_alphas=3, out_of_core=options)
        seconds = time.perf_counter() - started
        plan = result.memory_plan
        return {
            "n": n,
            "p": len(result.feature_names),
            "baseline_rss_bytes": baseline,
            "peak_rss_bytes": _status("VmHWM"),
            "seconds": seconds,
            "plan_phases": plan.phases,
            "plan_max_phase_bytes": max(plan.phases.values()),
            "plan_persistent_bytes": plan.persistent_bytes,
            "source_batch_bytes": batch_bytes,
            "max_bytes": extra.get("max_bytes"),
        }

    if scenario == "t14":
        from rustystats.formula import dict_to_parsed_formula
        from rustystats.frozen_design import build_design_matrix

        frame = pl.concat(
            [
                _batch(index, min(BATCH_ROWS, n - start), False)
                for index, start in enumerate(range(0, n, BATCH_ROWS))
            ]
        )
        parsed = dict_to_parsed_formula(response="y", terms=_terms(False), intercept=True)
        _reset_peak()
        baseline = _status("VmRSS")
        started = time.perf_counter()
        _builder, _y, X, _names = build_design_matrix(frame, parsed)
        seconds = time.perf_counter() - started
        return {
            "n": n,
            "p": int(X.shape[1]),
            "baseline_rss_bytes": baseline,
            "peak_rss_bytes": _status("VmHWM"),
            "seconds": seconds,
        }

    with_te = scenario in ("t13b", "t13c")
    source, batch_bytes = _source(n, with_te, batch_rows)
    glm = rs.glm_dict(response="y", terms=_terms(with_te), data=source, family="poisson", seed=7)
    fit: dict[str, Any] = {}
    if scenario == "t13c":
        fit = {"regularization": "ridge", "cv": 3, "cv_seed": 11}
    # One sort run never holds more than ``te_run_records`` records; it is the same
    # at n and 4n, so the TE buffer has stopped growing before n (§6.6, §11).
    options = rs.OutOfCoreOptions(
        max_bytes=MAX_BYTES, scratch_dir=scratch, te_run_records=te_run_records
    )
    _reset_peak()
    baseline = _status("VmRSS")
    started = time.perf_counter()
    result = glm.fit(out_of_core=options, **fit)
    seconds = time.perf_counter() - started
    plan = result.memory_plan
    return {
        "n": n,
        "p": len(result.feature_names),
        "baseline_rss_bytes": baseline,
        "peak_rss_bytes": _status("VmHWM"),
        "seconds": seconds,
        "plan_max_phase_bytes": max(plan.phases.values()),
        "plan_persistent_bytes": plan.persistent_bytes,
        "source_batch_bytes": batch_bytes,
        "encode_rows": plan.encode_rows,
        "passes": sum(result.passes.values()),
    }


def _run_child(
    scenario: str,
    n: int,
    threads: int,
    *,
    te_run_records: int = 1,
    batch_rows: int = BATCH_ROWS,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="rustystats-bench-") as scratch:
        completed = subprocess.run(
            [
                sys.executable,
                __file__,
                "--child",
                scenario,
                str(n),
                scratch,
                str(te_run_records),
                str(batch_rows),
                json.dumps(extra or {}),
            ],
            capture_output=True,
            text=True,
            check=False,
            env={
                **os.environ,
                "RAYON_NUM_THREADS": str(threads),
                "POLARS_MAX_THREADS": str(threads),
            },
        )
    if completed.returncode != 0:
        raise RuntimeError(f"{scenario} at n={n} failed:\n{completed.stderr[-4000:]}")
    return {"scenario": scenario, "threads": threads, **json.loads(completed.stdout)}


def _checks(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks = []
    for scenario in ("t13a", "t13b", "t13c"):
        small, large = sorted((r for r in runs if r["scenario"] == scenario), key=lambda r: r["n"])
        growth = large["peak_rss_bytes"] - small["peak_rss_bytes"]
        allowed = max(16 * MIB, 0.02 * small["peak_rss_bytes"])
        checks.append(
            {
                "check": f"{scenario} growth n→4n",
                "value": growth,
                "limit": allowed,
                "ok": growth <= allowed,
            }
        )
        for run in (small, large):
            rise = run["peak_rss_bytes"] - run["baseline_rss_bytes"]
            limit = (
                run["plan_max_phase_bytes"]
                + run["plan_persistent_bytes"]
                + run["source_batch_bytes"]
            )
            checks.append(
                {
                    "check": f"{scenario} n={run['n']} within plan",
                    "value": rise,
                    "limit": limit,
                    "ok": rise <= limit,
                }
            )
    for run in (r for r in runs if r["scenario"] == "t18" and r["max_bytes"] is not None):
        rise = run["peak_rss_bytes"] - run["baseline_rss_bytes"]
        admitted = run["plan_max_phase_bytes"] + run["plan_persistent_bytes"]
        checks.append(
            {
                "check": f"t18 p={run['p']} admitted by its plan",
                "value": admitted,
                "limit": run["max_bytes"],
                "ok": admitted <= run["max_bytes"],
            }
        )
        checks.append(
            {
                "check": f"t18 p={run['p']} within plan",
                "value": rise,
                "limit": admitted + run["source_batch_bytes"],
                "ok": rise <= admitted + run["source_batch_bytes"],
            }
        )
    for run in (r for r in runs if r["scenario"] == "t14"):
        rise = run["peak_rss_bytes"] - run["baseline_rss_bytes"]
        limit = 1.1 * 8 * run["n"] * run["p"]
        checks.append(
            {
                "check": f"t14 build at {run['threads']} thread(s)",
                "value": rise,
                "limit": limit,
                "ok": rise <= limit,
            }
        )
    return checks


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        scenario, n, scratch = sys.argv[2], int(sys.argv[3]), sys.argv[4]
        extra = json.loads(sys.argv[7]) if len(sys.argv) > 7 else None
        print(json.dumps(_child(scenario, n, scratch, int(sys.argv[5]), int(sys.argv[6]), extra)))
        return 0
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--smoke", action="store_true", help="n = 64,000: the CI variant")
    parser.add_argument("--rows", type=int, help="n for T13 and T14 (default 2,000,000)")
    parser.add_argument("--output", type=Path, help="write the runs and checks as JSON")
    args = parser.parse_args()
    # n and 4n must fall on the same side of the sparse-rows threshold (n·p ≥ 10⁷, §5.2),
    # or the growth check measures the kernel switch's fixed CSR buffers, not growth
    # with n. At p ≈ 160 that threshold is 62,500 rows.
    n = args.rows or (64_000 if args.smoke else 2_000_000)
    # T14's bound is relative (10% over 8np), so it needs a design large beside the
    # build's fixed costs: at least 400,000 rows.
    t14_rows = max(n, 400_000)

    runs = []
    for scenario in ("t13a", "t13b", "t13c"):
        for rows in (n, 4 * n):
            runs.append(
                _run_child(
                    scenario,
                    rows,
                    threads=4,
                    te_run_records=n // 2,
                    batch_rows=min(BATCH_ROWS, n // 2),
                )
            )
    for threads in (1, 8):
        runs.append(_run_child("t14", t14_rows, threads=threads))
    # T18: the plan of the unbounded fit, then the same fit under exactly that plan.
    levels, t18_rows = (300, 4_000) if args.smoke else (T18_LEVELS, T18_ROWS)
    t18 = {"levels": levels}
    planned = _run_child("t18", t18_rows, threads=2, batch_rows=2_048, extra=t18)
    planned["max_bytes"] = None
    runs.append(planned)
    exact = planned["plan_persistent_bytes"] + planned["plan_max_phase_bytes"]
    runs.append(
        _run_child("t18", t18_rows, threads=2, batch_rows=2_048, extra={**t18, "max_bytes": exact})
    )
    checks = _checks(runs)

    for run in runs:
        rise = (run["peak_rss_bytes"] - run["baseline_rss_bytes"]) / MIB
        print(
            f"{run['scenario']:5} n={run['n']:>10,} threads={run['threads']} p={run['p']:>4} "
            f"rise={rise:9.1f} MiB  {run['seconds']:7.1f} s"
        )
    for check in checks:
        status = "ok  " if check["ok"] else "FAIL"
        print(
            f"{status} {check['check']}: {check['value'] / MIB:.1f} MiB "
            f"(limit {check['limit'] / MIB:.1f} MiB)"
        )
    if args.output is not None:
        args.output.write_text(json.dumps({"runs": runs, "checks": checks}, indent=2))
    return 0 if all(check["ok"] for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
