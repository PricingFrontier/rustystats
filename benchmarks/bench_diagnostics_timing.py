"""
Time every section of result.diagnostics() on a synthetic 1M-row, 30-feature
dataset to identify bottlenecks. Run with: uv run python benchmarks/bench_diagnostics_timing.py
"""

from __future__ import annotations

import functools
import time

import numpy as np
import polars as pl
import rustystats as rs
from rustystats.diagnostics import DiagnosticsComputer
from rustystats.diagnostics import api as diag_api
from rustystats.diagnostics.factors import _FactorDiagnosticsComputer


class Timer:
    def __init__(self):
        self.records: list[tuple[str, float]] = []

    def wrap(self, name: str, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.records.append((name, time.perf_counter() - t0))

        return wrapper

    def report(self, total: float):
        agg: dict[str, tuple[float, int]] = {}
        for name, dur in self.records:
            t, c = agg.get(name, (0.0, 0))
            agg[name] = (t + dur, c + 1)
        rows = sorted(agg.items(), key=lambda kv: -kv[1][0])
        print(f"\n{'Section':<55} {'Calls':>6} {'Total (s)':>12} {'% of diag':>12}")
        print("-" * 90)
        accounted = 0.0
        for name, (t, c) in rows:
            pct = 100 * t / total if total > 0 else 0
            accounted += t
            print(f"{name:<55} {c:>6} {t:>12.3f} {pct:>11.1f}%")
        print("-" * 90)
        unaccounted = total - accounted
        print(
            f"{'(unaccounted / orchestrator overhead)':<55} {'':>6} {unaccounted:>12.3f} {100 * unaccounted / total:>11.1f}%"
        )
        print(f"{'TOTAL':<55} {'':>6} {total:>12.3f} {'100.0%':>12}")


T = Timer()


def install_patches(result):
    # Wrap every method on DiagnosticsComputer
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
                T.wrap(f"Computer.{method_name}", original),
            )

    # Wrap the factorize-strings utility used in the cache loop (if still imported).
    # Post-OPT-13 this symbol is no longer bound on diag_api; skip the wrap cleanly.
    if hasattr(diag_api, "_factorize_strings"):
        diag_api._factorize_strings = T.wrap(
            "_factorize_strings (cache loop)", diag_api._factorize_strings
        )

    # Drill into factor-diagnostics internals
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
                T.wrap(f"_Factors.{method_name}", original),
            )

    # Wrap key result-side calls invoked from compute_diagnostics
    if hasattr(result, "predict"):
        result.predict = T.wrap("result.predict", result.predict)
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
                setattr(result, attr, T.wrap(f"result.{attr}", attr_val))


def make_dataset(n: int = 1_000_000, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)

    # 6 categorical features (cardinalities chosen to span typical use cases)
    cat_cards = {"cat0": 10, "cat1": 50, "cat2": 4, "cat3": 15, "cat4": 8, "cat5": 200}
    cat_cols = {name: rng.integers(0, k, n) for name, k in cat_cards.items()}

    # 24 continuous features (standard normal)
    cont_cols = {f"cont{i}": rng.standard_normal(n).astype(np.float64) for i in range(24)}

    # Build a Poisson response using a handful of effects so the model has signal
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

    df = pl.DataFrame(
        {
            "y": y,
            "exposure": exposure,
            **{name: arr.astype(str) for name, arr in cat_cols.items()},
            **cont_cols,
        }
    )
    return df


def main():
    print("Generating synthetic data: 1,000,000 rows × 30 features (6 cat + 24 cont)...")
    t0 = time.perf_counter()
    data = make_dataset()
    print(f"  built in {time.perf_counter() - t0:.2f}s, shape={data.shape}")

    # Build a representative model: mix of linear, splines, categorical, target encoding
    terms = {}
    for i in range(10):
        terms[f"cont{i}"] = {"type": "linear"}
    for i in range(10, 14):
        terms[f"cont{i}"] = {"type": "bs", "df": 4}
    for name in ["cat0", "cat1", "cat2", "cat3", "cat4"]:
        terms[name] = {"type": "categorical"}
    terms["cat5"] = {"type": "target_encoding"}  # high-cardinality

    print("\nFitting Poisson GLM...")
    t0 = time.perf_counter()
    result = rs.glm_dict(
        response="y",
        terms=terms,
        data=data,
        family="poisson",
        offset="exposure",
    ).fit()
    print(f"  fitted in {time.perf_counter() - t0:.2f}s, params={len(result.params)}")

    install_patches(result)

    cat_factors = [f"cat{i}" for i in range(6)]
    cont_factors = [f"cont{i}" for i in range(24)]

    print(
        f"\nRunning diagnostics: {len(cat_factors)} cat factors + {len(cont_factors)} cont factors..."
    )
    t0 = time.perf_counter()
    diag = result.diagnostics(
        train_data=data,
        categorical_factors=cat_factors,
        continuous_factors=cont_factors,
    )
    total = time.perf_counter() - t0
    print(f"  diagnostics() returned in {total:.3f}s")

    T.report(total)

    # Sanity check
    print(
        f"\nDiagnostics ok: n_obs={diag.model_summary['n_obs']}, "
        f"n_factors_analyzed={len(diag.factors)}"
    )


if __name__ == "__main__":
    main()
