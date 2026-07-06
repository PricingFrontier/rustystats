#!/usr/bin/env python3
"""Generate exact fixed-penalty smooth oracle fixtures."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORACLE_DIR = ROOT / "tests" / "oracles" / "exact"


def _round_array(values: np.ndarray) -> list[float] | list[list[float]]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return [float(f"{v:.16g}") for v in arr]
    return [[float(f"{v:.16g}") for v in row] for row in arr]


def _difference_penalty(k: int, order: int = 2) -> np.ndarray:
    d = np.eye(k)
    for _ in range(order):
        d = np.diff(d, axis=0)
    return d.T @ d


def _fixture() -> dict:
    n = 64
    k = 5
    lambda_ = 0.7
    x = np.linspace(0.0, 1.0, n)
    basis = np.column_stack([x**power for power in range(1, k + 1)])
    design = np.column_stack([np.ones(n), basis])
    y = 0.8 + 0.5 * x - 0.15 * x**2 + 0.04 * np.sin(np.arange(n) / 3.0)
    penalty = _difference_penalty(k, order=2)
    full_penalty = np.zeros((k + 1, k + 1), dtype=np.float64)
    full_penalty[1:, 1:] = lambda_ * penalty
    params = np.linalg.solve(design.T @ design + full_penalty, design.T @ y)
    fitted = design @ params
    gradient = design.T @ (fitted - y) + full_penalty @ params
    hat = design @ np.linalg.solve(design.T @ design + full_penalty, design.T)
    edf = float(np.trace(hat))

    return {
        "schema_version": 1,
        "case_id": "exact-gaussian-fixed-smooth-penalty",
        "oracle": "exact-penalized-least-squares",
        "oracle_version": "numpy.linalg.solve",
        "data": {
            "columns": {
                "y": _round_array(y),
                "basis": _round_array(basis),
            }
        },
        "model": {
            "family": "gaussian",
            "link": "identity",
            "smooth_col_ranges": [[1, k + 1]],
            "lambda": lambda_,
            "penalty_order": 2,
        },
        "expected": {
            "penalty_matrix": _round_array(penalty),
            "params": _round_array(params),
            "fittedvalues": _round_array(fitted),
            "edf": float(f"{edf:.16g}"),
            "kkt_max_abs": float(f"{np.max(np.abs(gradient)):.16g}"),
        },
        "tolerances": {
            "params_atol": 2e-7,
            "params_rtol": 2e-7,
            "prediction_atol": 2e-7,
            "prediction_rtol": 2e-7,
            "edf_atol": 1e-8,
            "kkt_atol": 1e-8,
        },
    }


def _binomial_penalized_newton(
    x_full: np.ndarray, y: np.ndarray, full_penalty: np.ndarray
) -> np.ndarray:
    """Exact penalized binomial-logit fit by damped-free Newton iteration.

    Solves X'(y - mu) = S_full @ beta, the stationarity condition of the
    penalized deviance dev + beta' S_full beta (factor 2 cancels). Quadratic
    convergence to machine precision on this well-conditioned fixture.
    """
    beta = np.zeros(x_full.shape[1], dtype=np.float64)
    for _ in range(100):
        eta = x_full @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        weights = mu * (1.0 - mu)
        hessian = x_full.T @ (x_full * weights[:, None]) + full_penalty
        gradient = x_full.T @ (y - mu) - full_penalty @ beta
        delta = np.linalg.solve(hessian, gradient)
        beta = beta + delta
        if float(np.max(np.abs(delta))) < 1e-14:
            break
    return beta


def _monotone_binomial_fixture() -> dict:
    """Monotone-decreasing bs x binomial oracle (bug.md Fix 4).

    The true coefficient sequence is strictly decreasing, so the monotone
    constraint is SLACK at the optimum and the constrained solver must
    reproduce the exact unconstrained penalized fit. A solver that stalls at a
    non-stationary point (the historical failure mode) misses these values by
    orders of magnitude more than the tolerances.
    """
    n = 480
    k = 5
    lambda_ = 0.5
    x = np.linspace(0.0, 1.0, n)
    centers = np.linspace(0.1, 0.9, k)
    # Narrow bumps: adjacent-column overlap ~0.14, keeping the block well
    # conditioned so the unconstrained optimum tracks the (strictly
    # decreasing, penalty-null-space linear) truth with a wide margin.
    basis = np.column_stack([np.exp(-((x - c) ** 2) / (2.0 * 0.10**2)) for c in centers])
    design = np.column_stack([np.ones(n), basis])

    beta_true = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
    rng = np.random.default_rng(20260706)
    p_true = 1.0 / (1.0 + np.exp(-(-0.1 + basis @ beta_true)))
    y = rng.binomial(1, p_true).astype(np.float64)

    penalty = _difference_penalty(k, order=2)
    full_penalty = np.zeros((k + 1, k + 1), dtype=np.float64)
    full_penalty[1:, 1:] = lambda_ * penalty

    params = _binomial_penalized_newton(design, y, full_penalty)
    smooth_coefs = params[1:]
    if not np.all(np.diff(smooth_coefs) < -1e-2):
        raise RuntimeError(
            "monotone-binomial fixture invalid: unconstrained optimum is not strictly "
            f"decreasing with margin (coefs {smooth_coefs}); the constrained fit would "
            "not equal the unconstrained oracle"
        )

    eta = design @ params
    mu = 1.0 / (1.0 + np.exp(-eta))
    gradient = design.T @ (y - mu) - full_penalty @ params
    weights = mu * (1.0 - mu)
    xtwx = design.T @ (design * weights[:, None])
    edf = float(np.trace(np.linalg.solve(xtwx + full_penalty, xtwx)))

    return {
        "schema_version": 1,
        "case_id": "exact-binomial-monotone-decreasing-fixed-smooth-penalty",
        "oracle": "exact-penalized-binomial-newton",
        "oracle_version": "numpy.linalg.solve",
        "data": {
            "columns": {
                "y": _round_array(y),
                "basis": _round_array(basis),
            }
        },
        "model": {
            "family": "binomial",
            "link": "logit",
            "smooth_col_ranges": [[1, k + 1]],
            "monotonicity": ["decreasing"],
            "lambda": lambda_,
            "penalty_order": 2,
        },
        "expected": {
            "penalty_matrix": _round_array(penalty),
            "params": _round_array(params),
            "fittedvalues": _round_array(mu),
            "edf": float(f"{edf:.16g}"),
            "kkt_max_abs": float(f"{np.max(np.abs(gradient)):.16g}"),
        },
        "tolerances": {
            "params_atol": 5e-5,
            "params_rtol": 5e-5,
            "prediction_atol": 1e-5,
            "prediction_rtol": 1e-5,
            "edf_atol": 1e-4,
            "kkt_atol": 1e-4,
        },
    }


def build_fixtures() -> dict[str, dict]:
    return {
        "fixed_penalty_gaussian.json": _fixture(),
        "fixed_penalty_monotone_binomial.json": _monotone_binomial_fixture(),
    }


def _write_fixtures(directory: Path, fixtures: dict[str, dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, fixture in fixtures.items():
        (directory / name).write_text(
            json.dumps(fixture, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accept", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    fixtures = build_fixtures()
    if args.accept:
        _write_fixtures(ORACLE_DIR, fixtures)
        print(
            f"Wrote {len(fixtures)} exact smooth oracle fixtures to {ORACLE_DIR.relative_to(ROOT)}."
        )
        return 0

    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="rustystats-smooth-oracles-"))
    if args.output_dir and output_dir.exists():
        shutil.rmtree(output_dir)
    _write_fixtures(output_dir, fixtures)

    if args.check:
        failures = []
        for name in fixtures:
            expected = ORACLE_DIR / name
            generated = output_dir / name
            if not expected.is_file():
                failures.append(f"missing checked-in fixture {expected.relative_to(ROOT)}")
            elif expected.read_text(encoding="utf-8") != generated.read_text(encoding="utf-8"):
                failures.append(f"fixture drift: {expected.relative_to(ROOT)}")
        if failures:
            print("Smooth oracle fixture generation check failed:")
            for failure in failures:
                print(f" - {failure}")
            print(f"Generated comparison fixtures in {output_dir}")
            return 1
        print(f"Smooth oracle fixture generation check passed for {len(fixtures)} fixtures.")
        return 0

    print(f"Wrote comparison fixtures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
