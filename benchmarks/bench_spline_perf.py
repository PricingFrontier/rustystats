"""
Benchmark: Penalized Splines — rustystats vs glum

Scenario: 1,000,000 rows × 15 continuous features, each fit with a cubic
B-spline basis, Poisson distribution, penalized (smoothed) splines.

Benchmark modes:
  A) Fixed alpha: both libraries use D'D penalty with a pre-chosen alpha.
     This isolates the IRLS solver speed (X'WX + Cholesky).
  B) Auto-tuned: rustystats uses GCV-based lambda selection per term.
     This is the typical user experience (no manual alpha needed).

glum does not have built-in GCV for P2, so mode A is its standard workflow.
rustystats' default `{"type": "bs"}` uses mode B.

Methodology:
- Median of N_RUNS for timing
- Peak RSS memory tracking via psutil
- Same synthetic data for both
"""
import time
import threading
import gc
import sys

import numpy as np
import polars as pl
import psutil

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_ROWS = 1_000_000
N_FEATURES = 15
N_KNOTS = 9          # internal knots
DEGREE = 3
N_RUNS = 3
SEED = 42
ALPHA = 1.0          # fixed alpha for mode A

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def generate_data(n_rows, n_features, seed=42):
    """Generate Poisson data with smooth nonlinear effects."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, size=(n_rows, n_features))
    eta = np.zeros(n_rows)
    for j in range(n_features):
        eta += 0.1 * np.sin(X[:, j] * (0.5 + 0.1 * j))
    eta += -1.0
    mu = np.exp(np.clip(eta, -5, 5))
    y = rng.poisson(mu).astype(np.float64)
    data = {f"x{i}": X[:, i] for i in range(n_features)}
    data["y"] = y
    return pl.DataFrame(data), X, y


# ---------------------------------------------------------------------------
# Timing / memory helpers
# ---------------------------------------------------------------------------
def time_fn(func, n_runs=N_RUNS, label=""):
    """Return median wall-clock time over n_runs."""
    times = []
    for i in range(n_runs):
        gc.collect()
        t0 = time.perf_counter()
        result = func()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if label:
            print(f"    {label} run {i+1}: {t1-t0:.3f}s")
        del result
    return np.median(times), times


def measure_peak_memory_mb(func):
    """Measure peak RSS above baseline during func() execution."""
    process = psutil.Process()
    gc.collect()
    baseline = process.memory_info().rss
    peak = baseline
    stop = threading.Event()

    def monitor():
        nonlocal peak
        while not stop.is_set():
            current = process.memory_info().rss
            peak = max(peak, current)
            time.sleep(0.005)

    t = threading.Thread(target=monitor, daemon=True)
    t.start()
    result = func()
    stop.set()
    t.join()
    del result
    return (peak - baseline) / (1024 * 1024)


# ---------------------------------------------------------------------------
# glum: fixed alpha + D'D penalty (standard P-spline workflow)
# ---------------------------------------------------------------------------
def bench_glum(X, y, n_features, alpha=ALPHA):
    """Fit penalized splines in glum using P2 = block_diag(D'D)."""
    from sklearn.preprocessing import SplineTransformer
    from glum import GeneralizedLinearRegressor
    from scipy.linalg import block_diag

    st = SplineTransformer(
        n_knots=N_KNOTS + 2,
        degree=DEGREE,
        include_bias=True,
        extrapolation="continue",
    )
    X_spline = st.fit_transform(X)
    n_basis = X_spline.shape[1] // n_features
    print(f"  Design matrix: {X_spline.shape} ({n_basis} basis/feature)")

    # D'D penalty + tiny ridge for numerical PSD
    penalties = []
    for _ in range(n_features):
        D = np.diff(np.eye(n_basis), n=2, axis=0)
        penalties.append(D.T @ D + 1e-6 * np.eye(n_basis))
    P2 = block_diag(*penalties)

    model = GeneralizedLinearRegressor(
        family="poisson", link="log", alpha=alpha, l1_ratio=0,
        P2=P2, fit_intercept=True, max_iter=100, gradient_tol=1e-8,
    )

    def fit_fn():
        model.fit(X_spline, y)
        return model

    # Warmup
    fit_fn()

    # Time
    median_time, all_times = time_fn(fit_fn, label="glum")

    # Memory
    mem = measure_peak_memory_mb(fit_fn)

    # Final fit for stats
    m = fit_fn()
    pred = m.predict(X_spline)
    deviance = m.family_instance.deviance(y, pred)

    return {
        "library": "glum (fixed α)",
        "median_time": median_time,
        "all_times": all_times,
        "peak_memory_mb": mem,
        "n_iter": m.n_iter_,
        "n_cols": X_spline.shape[1],
        "coef_norm": float(np.linalg.norm(m.coef_)),
        "deviance": deviance,
        "note": f"alpha={alpha}, P2=D'D",
    }


# ---------------------------------------------------------------------------
# rustystats: auto-tuned GCV (default user experience)
# ---------------------------------------------------------------------------
def bench_rustystats_auto(df, n_features):
    """Fit penalized splines with automatic GCV lambda selection."""
    import rustystats as rs

    terms = {f"x{i}": {"type": "bs"} for i in range(n_features)}

    def fit_fn():
        return rs.glm_dict(
            response="y", terms=terms, data=df, family="poisson",
        ).fit()

    # Warmup
    r0 = fit_fn()
    n_cols = len(r0.params)
    print(f"  Design matrix cols: {n_cols} (incl intercept)")

    # Time (single run — GCV is expensive)
    median_time, all_times = time_fn(fit_fn, n_runs=1, label="rs-auto")

    # Memory
    mem = measure_peak_memory_mb(fit_fn)

    # Final fit for stats
    result = fit_fn()
    return {
        "library": "rustystats (auto GCV)",
        "median_time": median_time,
        "all_times": all_times,
        "peak_memory_mb": mem,
        "n_iter": result.iterations,
        "n_cols": n_cols,
        "coef_norm": float(np.linalg.norm(result.params)),
        "deviance": result.deviance,
        "note": "GCV-tuned λ per term, D'D penalty",
    }


# ---------------------------------------------------------------------------
# rustystats: fixed-df splines + ridge (no GCV overhead, comparable to glum)
# ---------------------------------------------------------------------------
def bench_rustystats_fixed(df, n_features, alpha=ALPHA):
    """Fit splines with fixed df and explicit ridge — isolates IRLS speed."""
    import rustystats as rs

    # Fixed df=11 matches ~11 basis cols per feature (like glum's 13)
    # Using k=N_KNOTS+DEGREE+1-1 to match glum's column count
    terms = {f"x{i}": {"type": "bs", "df": 11} for i in range(n_features)}

    def fit_fn():
        return rs.glm_dict(
            response="y", terms=terms, data=df, family="poisson",
        ).fit(alpha=alpha)

    # Warmup
    r0 = fit_fn()
    n_cols = len(r0.params)
    print(f"  Design matrix cols: {n_cols} (incl intercept)")

    # Time
    median_time, all_times = time_fn(fit_fn, label="rs-fixed")

    # Memory
    mem = measure_peak_memory_mb(fit_fn)

    # Final fit for stats
    result = fit_fn()
    return {
        "library": "rustystats (fixed df+ridge)",
        "median_time": median_time,
        "all_times": all_times,
        "peak_memory_mb": mem,
        "n_iter": result.iterations,
        "n_cols": n_cols,
        "coef_norm": float(np.linalg.norm(result.params)),
        "deviance": result.deviance,
        "note": f"fixed df=11, alpha={alpha} (scalar L2)",
    }


# ---------------------------------------------------------------------------
# Report helper
# ---------------------------------------------------------------------------
def print_result(res):
    print(f"  Time (median):  {res['median_time']:.3f}s")
    print(f"  All times:      {[f'{t:.3f}' for t in res['all_times']]}")
    print(f"  Peak memory:    {res['peak_memory_mb']:.0f} MB")
    print(f"  IRLS iters:     {res['n_iter']}")
    print(f"  Design cols:    {res['n_cols']}")
    print(f"  Deviance:       {res['deviance']:.2f}")
    print(f"  Note:           {res['note']}")


def print_comparison(results):
    """Print a summary comparison table."""
    print(f"\n{'':32s}", end="")
    for r in results:
        print(f" {r['library']:>24s}", end="")
    print()
    print("-" * (32 + 25 * len(results)))

    for key, fmt in [
        ("median_time", "{:.3f}s"),
        ("n_iter", "{}"),
        ("peak_memory_mb", "{:.0f} MB"),
        ("n_cols", "{}"),
        ("deviance", "{:.2f}"),
    ]:
        label = {
            "median_time": "Total time (median)",
            "n_iter": "IRLS iterations",
            "peak_memory_mb": "Peak memory",
            "n_cols": "Design matrix cols",
            "deviance": "Deviance",
        }[key]
        print(f"  {label:30s}", end="")
        for r in results:
            val = fmt.format(r[key])
            print(f" {val:>24s}", end="")
        print()

    # Speed ratios vs first result
    base = results[0]
    if len(results) > 1:
        print()
        for r in results[1:]:
            ratio = r['median_time'] / base['median_time']
            if ratio > 1:
                print(f"  {base['library']} is {ratio:.1f}× faster than {r['library']}")
            else:
                print(f"  {r['library']} is {1/ratio:.1f}× faster than {base['library']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("PENALIZED SPLINE BENCHMARK: rustystats vs glum")
    print(f"  Rows:      {N_ROWS:,}")
    print(f"  Features:  {N_FEATURES} continuous")
    print(f"  Family:    Poisson (log link)")
    print(f"  Splines:   cubic B-spline, {N_KNOTS} internal knots")
    print(f"  Penalty:   D'D (second-difference)")
    print(f"  Fixed α:   {ALPHA}")
    print(f"  Runs:      {N_RUNS} (median)")
    print("=" * 70)

    print("\nGenerating data...")
    df, X, y = generate_data(N_ROWS, N_FEATURES, seed=SEED)
    print(f"  Shape: {X.shape}, y mean: {y.mean():.3f}, y>0: {(y>0).mean():.1%}")

    all_results = []

    # --- Mode A: glum with fixed alpha + D'D ---
    print("\n" + "=" * 70)
    print("MODE A: Fixed alpha — isolates IRLS solver speed")
    print("=" * 70)

    print("\n[1] glum (fixed α, P2=D'D)")
    try:
        r = bench_glum(X, y, N_FEATURES)
        print_result(r)
        all_results.append(r)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

    gc.collect()

    print("\n[2] rustystats (fixed df=11, ridge α)")
    try:
        r = bench_rustystats_fixed(df, N_FEATURES)
        print_result(r)
        all_results.append(r)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

    gc.collect()

    # --- Mode B: rustystats auto-tuned ---
    print("\n" + "=" * 70)
    print("MODE B: Auto-tuned — rustystats GCV (typical user workflow)")
    print("=" * 70)

    print("\n[3] rustystats (auto GCV, D'D penalty)")
    try:
        r = bench_rustystats_auto(df, N_FEATURES)
        print_result(r)
        all_results.append(r)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

    # --- Summary ---
    if all_results:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print_comparison(all_results)


if __name__ == "__main__":
    main()
