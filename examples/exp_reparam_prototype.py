"""
Working Python prototype of exp reparameterization for monotonic B-splines.
This produces dev=28559 on the conversion data (matching unconstrained).
The Rust implementation should produce identical results.
"""

import numpy as np
import polars as pl
from rustystats.splines import bs


def run_monotonic_decreasing(x, y, lam=2.1, max_init_iter=25, max_reparam_iter=20):
    """Fit monotonic decreasing B-spline via exp reparameterization."""
    # Step 1: Build design matrix
    X_sp = bs(x, k=10)  # Standard B-spline basis (first col dropped for identifiability)
    X = np.column_stack([np.ones(len(x)), X_sp])  # Add intercept
    n, p = X.shape
    k = X_sp.shape[1]  # Number of spline columns (9 for k=10)

    # Step 2: Build penalty matrix (second-order difference on spline coefficients)
    D2 = np.diff(np.eye(k), n=2, axis=0)
    S_sp = D2.T @ D2  # k x k penalty on spline coefficients
    S_full = np.zeros((p, p))
    S_full[1:, 1:] = S_sp  # No penalty on intercept

    # Step 3: WLS solver
    def wls_solve(X, z, W, S, lam):
        Xw = X * W[:, None]
        A = Xw.T @ X + lam * S
        b = Xw.T @ z
        return np.linalg.solve(A, b)

    # Step 4: Unconstrained IRLS to convergence
    mu = np.full(n, y.mean())
    eta = np.log(mu / (1 - mu))  # logit link
    for _ in range(max_init_iter):
        W = mu * (1 - mu)
        z = eta + (y - mu) / W
        beta = wls_solve(X, z, W, S_full, lam)
        eta = X @ beta
        mu = np.clip(1 / (1 + np.exp(-eta)), 1e-10, 1 - 1e-10)

    # Step 5: Project to non-increasing (monotonic decreasing)
    beta_sp = beta[1:].copy()
    for j in range(1, len(beta_sp)):
        beta_sp[j] = min(beta_sp[j], beta_sp[j - 1])

    # Step 6: Initialize alpha from projected beta
    alpha = np.zeros(k)
    alpha[0] = beta_sp[0]
    for j in range(1, k):
        alpha[j] = np.log(max(beta_sp[j - 1] - beta_sp[j], 1e-10))

    coef = np.concatenate([[beta[0]], beta_sp])
    eta = X @ coef
    mu = np.clip(1 / (1 + np.exp(-eta)), 1e-10, 1 - 1e-10)

    # Step 7: Exp-reparam PIRLS iterations
    for _it in range(max_reparam_iter):
        # 7a: IRLS weights and working response
        W = mu * (1 - mu)
        z = eta + (y - mu) / W

        # 7b: Build Jacobian J (k x k, lower triangular)
        J = np.zeros((k, k))
        J[:, 0] = 1.0  # d(beta_i)/d(alpha_0) = 1 for all i
        for j in range(1, k):
            J[j:, j] = -np.exp(np.clip(alpha[j], -20, 20))  # decreasing

        # 7c: Build X_tilde = [intercept, X_spline @ J]
        X_tilde = np.column_stack([X[:, 0:1], X[:, 1:] @ J])

        # 7d: Build S_tilde = J' @ S_sp @ J (transformed penalty)
        S_tilde_sp = J.T @ S_sp @ J
        S_tilde_full = np.zeros((p, p))
        S_tilde_full[1:, 1:] = S_tilde_sp
        # NOTE: S_tilde_full[0,0] = 0 (no penalty on intercept)
        # NOTE: S_tilde_full[1,1] = 0 (no penalty on alpha[0] - this is correct!)
        # The solver handles this fine via LU decomposition.

        # 7e: Linearization offset c = X_spline @ (beta_current - J @ alpha)
        beta_current = coef[1:]
        offset_mono = beta_current - J @ alpha
        c = X[:, 1:] @ offset_mono

        # 7f: Adjusted working response
        z_adj = z - c

        # 7g: WLS solve in alpha-space
        alpha_coef = wls_solve(X_tilde, z_adj, W, S_tilde_full, lam)

        # 7h: Accept solution directly (NO step control, NO step halving)
        coef[0] = alpha_coef[0]  # intercept
        alpha = np.clip(alpha_coef[1:], -20, 20)  # alpha values

        # 7i: Convert alpha to beta
        beta_new = np.zeros(k)
        beta_new[0] = alpha[0]
        for j in range(1, k):
            beta_new[j] = beta_new[j - 1] - np.exp(alpha[j])
        coef[1:] = beta_new

        # 7j: Update predictions
        eta = X @ coef
        mu = np.clip(1 / (1 + np.exp(-eta)), 1e-10, 1 - 1e-10)

    dev = -2 * np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))
    return coef, mu, dev, alpha


if __name__ == "__main__":
    data = pl.read_parquet("./examples/conversion_data.parquet").limit(100_000)
    x = data["difference_to_market"].to_numpy()
    y = data["sale_flag"].to_numpy().astype(float)

    coef, mu, dev, alpha = run_monotonic_decreasing(x, y, lam=2.1)

    print(f"Deviance: {dev:.2f}")
    print(f"Intercept: {coef[0]:.6f}")
    print(f"Alpha[0] (beta[0]): {alpha[0]:.6f}")
    print(f"Spline coefs: {coef[1:]}")

    # Binned comparison
    pred = mu
    binned = (
        pl.DataFrame({"x": x, "y": y, "p": pred})
        .with_columns(pl.col("x").qcut(20, allow_duplicates=True).alias("b"))
        .group_by("b")
        .agg(
            pl.col("x").mean().alias("mid"),
            pl.col("y").mean().alias("actual"),
            pl.col("p").mean().alias("pred"),
        )
        .sort("mid")
    )
    print(f"\n{'mid':>6} {'actual':>8} {'pred':>8} {'gap':>8}")
    for row in binned.iter_rows(named=True):
        gap = row["actual"] - row["pred"]
        print(f"{row['mid']:6.3f} {row['actual']:8.5f} {row['pred']:8.5f} {gap:+8.5f}")
