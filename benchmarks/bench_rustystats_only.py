"""RustyStats-only benchmark to isolate crash source."""
import time
import numpy as np
import polars as pl
import gc

import rustystats as rs

N_CONTINUOUS = 10
N_CATEGORICAL = 10
CAT_LEVELS = 10
SIZES = [10_000, 250_000, 500_000]

FORMULA = "y ~ " + " + ".join(
    [f"x{i}" for i in range(N_CONTINUOUS)] + 
    [f"C(c{i})" for i in range(N_CATEGORICAL)]
)

def generate_data(n_rows, family, seed=42):
    """Generate data for a given family."""
    rng = np.random.default_rng(seed)
    
    X_cont = rng.standard_normal((n_rows, N_CONTINUOUS))
    X_cat = rng.integers(0, CAT_LEVELS, (n_rows, N_CATEGORICAL))
    
    beta = np.array([0.3, -0.2, 0.5, -0.1, 0.4, -0.3, 0.2, -0.4, 0.1, -0.2])
    lp = X_cont @ beta
    
    if family == "gaussian":
        y = lp + rng.standard_normal(n_rows) * 0.5
    elif family == "poisson":
        y = rng.poisson(np.exp(np.clip(lp * 0.3, -3, 3))).astype(np.float64)
    elif family == "binomial":
        y = rng.binomial(1, 1 / (1 + np.exp(-lp))).astype(np.float64)
    elif family == "gamma":
        y = np.maximum(rng.gamma(2.0, np.exp(np.clip(lp * 0.3 + 2, 0.5, 5)) / 2.0), 0.001)
    elif family == "negbinomial":
        mu = np.exp(np.clip(lp * 0.3, -2, 3))
        theta = 2.0
        p = theta / (theta + mu)
        y = rng.negative_binomial(theta, p).astype(np.float64)
    
    data = {f"x{i}": X_cont[:, i] for i in range(N_CONTINUOUS)}
    data.update({f"c{i}": X_cat[:, i] for i in range(N_CATEGORICAL)})
    data["y"] = y
    return pl.DataFrame(data)


def time_fn(func, n_runs=3):
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return np.median(times)


def main():
    families = ["gaussian", "poisson", "binomial", "gamma", "negbinomial"]
    results = {}
    
    print("RustyStats-Only Benchmark")
    print(f"Features: {N_CONTINUOUS} continuous + {N_CATEGORICAL} categorical")
    print("=" * 50)
    
    for family in families:
        print(f"\n{family.upper()}")
        results[family] = []
        
        for n_rows in SIZES:
            print(f"  n={n_rows:,}...", end=" ", flush=True)
            
            df = generate_data(n_rows, family)
            
            if family == "negbinomial":
                t = time_fn(lambda: rs.glm(FORMULA, data=df, family=family, theta=2.0).fit())
            else:
                t = time_fn(lambda: rs.glm(FORMULA, data=df, family=family).fit())
            
            results[family].append((n_rows, t))
            print(f"{t:.3f}s")
            
            del df
            gc.collect()
    
    # Ridge
    print(f"\nRIDGE")
    results["ridge"] = []
    for n_rows in SIZES:
        print(f"  n={n_rows:,}...", end=" ", flush=True)
        df = generate_data(n_rows, "gaussian")
        t = time_fn(lambda: rs.glm(FORMULA, data=df, family="gaussian").fit(alpha=0.1, l1_ratio=0.0))
        results["ridge"].append((n_rows, t))
        print(f"{t:.3f}s")
        del df
        gc.collect()
    
    print("\n" + "=" * 50)
    print("RUSTYSTATS RESULTS")
    print("=" * 50)
    print("\n| Family | 10K | 250K | 500K |")
    print("|--------|-----|------|------|")
    for family in families + ["ridge"]:
        row = f"| {family.capitalize()} |"
        for _, t in results[family]:
            row += f" {t:.3f}s |"
        print(row)
    
    print("\nRustyStats benchmark complete. Now run bench_statsmodels_only.py")


if __name__ == "__main__":
    main()
