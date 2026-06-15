"""Benchmark native multinomial solver fit and prediction throughput.

The default ``quick`` preset is intentionally small so it is safe to run as a
smoke benchmark on a development laptop:

    uv run python benchmarks/bench_multinomial.py

Use the full matrix from the implementation plan with:

    uv run python benchmarks/bench_multinomial.py --preset full --repeat 3
"""

from __future__ import annotations

import argparse
import csv
import threading
import time
from pathlib import Path

import numpy as np
from rustystats._rustystats import fit_multinomial_py


def _read_rss_bytes() -> int:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return 0
    return 0


class _RssSampler(threading.Thread):
    def __init__(self, interval_s: float = 0.01):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.peak_bytes = _read_rss_bytes()
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.peak_bytes = max(self.peak_bytes, _read_rss_bytes())
            self._stop_event.wait(self.interval_s)

    def stop(self) -> int:
        self._stop_event.set()
        self.join()
        return self.peak_bytes


def _masked_softmax(logits: np.ndarray, availability: np.ndarray) -> np.ndarray:
    masked = np.where(availability, logits, -np.inf)
    max_eta = masked.max(axis=1, keepdims=True)
    exp_eta = np.where(availability, np.exp(masked - max_eta), 0.0)
    return exp_eta / exp_eta.sum(axis=1, keepdims=True)


def _sample_classes(probabilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cumulative = np.cumsum(probabilities, axis=1)
    draws = rng.random(probabilities.shape[0])
    return (draws[:, None] > cumulative).sum(axis=1).astype(np.int64)


def _generate_case(
    *,
    n_rows: int,
    n_features: int,
    n_classes: int,
    availability_mode: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_rows, n_features)).astype(np.float64)
    x[:, 0] = 1.0

    beta = rng.normal(scale=0.25, size=(n_classes - 1, n_features))
    beta[:, 0] = rng.normal(scale=0.5, size=n_classes - 1)
    logits = np.zeros((n_rows, n_classes), dtype=np.float64)
    logits[:, 1:] = x @ beta.T

    availability = None
    if availability_mode == "masked":
        availability = rng.random((n_rows, n_classes)) > 0.15
        availability[:, 0] = True
    elif availability_mode != "all":
        raise ValueError("availability_mode must be 'all' or 'masked'.")

    probabilities = _masked_softmax(
        logits,
        np.ones_like(logits, dtype=bool) if availability is None else availability,
    )
    y = _sample_classes(probabilities, rng)
    if availability is not None:
        availability[np.arange(n_rows), y] = True
    return x, y, availability


def _predict_probabilities(
    x: np.ndarray,
    params: np.ndarray,
    n_classes: int,
    availability: np.ndarray | None,
) -> np.ndarray:
    logits = np.zeros((x.shape[0], n_classes), dtype=np.float64)
    logits[:, 1:] = x @ params.T
    availability_matrix = np.ones_like(logits, dtype=bool) if availability is None else availability
    return _masked_softmax(logits, availability_matrix)


def _measure_case(
    *,
    n_rows: int,
    n_features: int,
    n_classes: int,
    availability_mode: str,
    alpha: float,
    repeat: int,
    seed: int,
) -> dict[str, float | int | str]:
    fit_times: list[float] = []
    predict_times: list[float] = []
    peak_rss_mb: list[float] = []
    iterations: list[int] = []

    for run_idx in range(repeat):
        x, y, availability = _generate_case(
            n_rows=n_rows,
            n_features=n_features,
            n_classes=n_classes,
            availability_mode=availability_mode,
            seed=seed + run_idx,
        )
        rss_before = _read_rss_bytes()
        sampler = _RssSampler()
        sampler.start()
        t0 = time.perf_counter()
        result = fit_multinomial_py(
            y,
            x,
            n_classes,
            0,
            availability=availability,
            alpha=alpha,
            max_iter=80,
            tol=1e-8,
            skip_covariance=True,
        )
        fit_elapsed = time.perf_counter() - t0
        peak = sampler.stop()

        t1 = time.perf_counter()
        probabilities = _predict_probabilities(x, result.params, n_classes, availability)
        predict_elapsed = time.perf_counter() - t1

        fit_times.append(fit_elapsed)
        predict_times.append(predict_elapsed)
        peak_rss_mb.append(max(0.0, (peak - rss_before) / (1024 * 1024)))
        iterations.append(int(result.iterations))
        if not np.all(np.isfinite(probabilities)):
            raise RuntimeError("non-finite prediction probabilities")

    q = n_features * (n_classes - 1)
    return {
        "n_rows": n_rows,
        "n_features": n_features,
        "n_classes": n_classes,
        "q": q,
        "hessian_mb": q * q * 8 / (1024 * 1024),
        "availability": availability_mode,
        "penalty": "ridge" if alpha > 0.0 else "none",
        "alpha": alpha,
        "fit_s_median": float(np.median(fit_times)),
        "fit_s_min": float(np.min(fit_times)),
        "fit_s_max": float(np.max(fit_times)),
        "peak_rss_mb_median": float(np.median(peak_rss_mb)),
        "iterations_median": float(np.median(iterations)),
        "predict_rows_per_s": float(n_rows / np.median(predict_times)),
    }


def _case_grid(preset: str) -> tuple[list[int], list[int], list[int]]:
    if preset == "quick":
        return [2_000], [20], [3]
    if preset == "full":
        return [10_000, 50_000, 200_000], [20, 100, 500], [3, 4, 8]
    raise ValueError("preset must be 'quick' or 'full'.")


def _print_rows(rows: list[dict[str, float | int | str]]) -> None:
    header = (
        f"{'n':>9} {'p':>5} {'K':>3} {'q':>6} {'H MB':>8} {'Avail':>7} "
        f"{'Penalty':>7} {'Fit s':>9} {'RSS MB':>9} {'Iter':>6} {'Pred rows/s':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['n_rows']:>9} {row['n_features']:>5} {row['n_classes']:>3} "
            f"{row['q']:>6} {row['hessian_mb']:>8.1f} {row['availability']:>7} "
            f"{row['penalty']:>7} {row['fit_s_median']:>9.3f} "
            f"{row['peak_rss_mb_median']:>9.1f} {row['iterations_median']:>6.1f} "
            f"{row['predict_rows_per_s']:>12.0f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--save-csv", type=Path)
    args = parser.parse_args()

    if args.repeat <= 0:
        raise SystemExit("--repeat must be positive")

    n_values, p_values, k_values = _case_grid(args.preset)
    rows: list[dict[str, float | int | str]] = []
    for n_rows in n_values:
        for n_features in p_values:
            for n_classes in k_values:
                for availability_mode in ("all", "masked"):
                    for alpha in (0.0, 1.0):
                        print(
                            "running "
                            f"n={n_rows:,}, p={n_features}, K={n_classes}, "
                            f"availability={availability_mode}, alpha={alpha:g}",
                            flush=True,
                        )
                        rows.append(
                            _measure_case(
                                n_rows=n_rows,
                                n_features=n_features,
                                n_classes=n_classes,
                                availability_mode=availability_mode,
                                alpha=alpha,
                                repeat=args.repeat,
                                seed=args.seed + len(rows) * 1000,
                            )
                        )

    print()
    _print_rows(rows)

    if args.save_csv is not None:
        with args.save_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.save_csv}")


if __name__ == "__main__":
    main()
