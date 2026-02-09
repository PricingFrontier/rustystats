# Computational Performance: Penalised Splines

A focused analysis of where rustystats loses time to glum when fitting
penalised splines on purely continuous data, based on real benchmarks.

**Scenario**: 1,000,000 rows × 15 continuous features, each fit with a cubic
B-spline basis (9 internal knots), Poisson distribution.

**Benchmark script**: `benchmarks/bench_spline_perf.py`

---

## Benchmark Results

Two comparison modes are tested:

- **Mode A (Fixed α)**: Both libraries use a pre-chosen α=1.0 with an L2-type
  penalty. Isolates the pure IRLS solver speed.
- **Mode B (Auto-tuned)**: rustystats' default `{"type": "bs"}` workflow with
  automatic GCV-based λ selection per smooth term (D'D penalty). This is the
  typical user experience.

### Mode A: Fixed Alpha — IRLS Solver Speed

|                       | glum (D'D penalty)  | rustystats (ridge)  | Ratio              |
|-----------------------|---------------------|---------------------|--------------------|
| **Total runtime**     | **2.61s**           | **7.87s**           | glum **3.0×** faster |
| IRLS iterations       | 3                   | 6                   |                    |
| Per-iteration cost    | ~0.87s              | ~1.31s              | glum 1.5× faster   |
| **Peak memory (pipeline)**| 80 MB               | 2,531 MB            | see Memory section |
| Design matrix cols    | 195                 | 166                 |                    |
| Deviance              | 967,944             | 945,155             |                    |

**Note on comparison fairness**: glum uses a D'D difference penalty (P₂ =
block_diag(D'D)), while rustystats uses scalar ridge (α·I) in this mode
because fixed-df splines go through the standard IRLS path. The D'D penalty
gives glum better convergence (3 vs 6 iterations). See Factor 2 below.

The **3.0× gap** decomposes into two independent factors:

```
3.0× = 1.5× (per-iteration speed) × 2.0× (fewer iterations)
         ↑                              ↑
     BLAS vs hand-written loop      D'D penalty vs scalar ridge
```

### Mode B: Auto-Tuned — Typical User Workflow

|                       | glum (fixed α)      | rustystats (GCV)    | Ratio              |
|-----------------------|---------------------|---------------------|--------------------||
| **Total runtime**     | **2.54s**           | **11.3s**           | glum **4.5×** faster |
| IRLS iterations       | 3                   | 7                   |                    |
| Per-iteration cost    | ~0.85s              | ~1.62s              | glum 1.9× faster   |
| Peak memory (pipeline)| 80 MB               | 2,299 MB            |                    |
| Design matrix cols    | 195                 | 151                 |                    |
| Deviance              | 967,944             | 945,412             |                    |

**This is not an apples-to-apples comparison.** glum requires the user to
pre-select α; rustystats automatically optimises λ per smooth term via GCV.
The remaining 4.5× gap is:
- ~1.5× per-iteration (BLAS vs hand-written loop — same as Mode A)
- ~1.9× from GCV overhead per iteration (cheap O(p³) Brent evaluations)
- ~2.3× from more IRLS iterations (7 vs 3)

### Scaling Summary

| n rows   | glum     | rs (fixed α) | rs (auto GCV) | rs auto (before P1) |
|----------|----------|--------------|---------------|---------------------|
| 50K      | 0.93s    | —            | 2.09s         | 22.8s               |
| 100K     | 0.92s    | —            | 3.20s         | 64.0s               |
| 1M       | 2.54s    | 8.0s         | 11.1s         | ~600s (est.)        |

---

## What Happens Per IRLS Iteration

Both libraries execute the same algorithm (IRLS with Cholesky solve). Each
iteration does four operations on the design matrix X (n × p):

| Step | Operation            | Complexity | What it does                        |
|------|----------------------|-----------|-------------------------------------|
| 1    | η = Xβ              | O(n·p)    | Forward pass: linear predictor      |
| 2    | μ, W, z              | O(n)      | Link function, weights, working response |
| 3    | X'WX                 | O(n·p²)   | **Dominant cost**: weighted cross-product |
| 4    | solve(X'WX + αP, X'Wz) | O(p³)  | Cholesky decomposition + back-sub   |

For n=1M, p≈170:

| Step         | FLOPs           | Relative cost |
|--------------|-----------------|---------------|
| η = Xβ      | ~340M           | ~1%           |
| X'WX         | **~29 billion** | **~97%**      |
| Cholesky     | ~1.6M           | ~0%           |

**X'WX dominates everything.**

---

## Factor 1: Why glum's X'WX is 1.5× Faster Per Iteration

### The operation

Both libraries compute:

```
X'WX[i,j] = Σₖ X[k,i] · W[k] · X[k,j]    for all (i,j) pairs
```

### glum: BLAS (OpenBLAS/MKL)

glum's tabmat `DenseMatrix` dispatches to the system BLAS library for the
core GEMM. BLAS implementations like OpenBLAS and MKL have been optimised
for 30+ years with:

- **Cache-oblivious tiling**: Blocks sized to fit L1/L2/L3 cache, minimising
  misses on the ~1.3 GB design matrix
- **SIMD micro-kernels**: Hand-tuned AVX2/AVX-512 assembly, 8–16 doubles per
  instruction
- **Multi-threaded**: Parallelism at the BLAS level with optimised work
  distribution
- **Memory prefetching**: Hardware prefetch hints overlapping loads with
  computation

### rustystats: hand-written parallel loop + rayon

rustystats computes X'WX in `irls.rs` using a chunked parallel loop with
raw slice access:

```rust
// Simplified from compute_xtwx_xtwz in irls.rs
(0..num_chunks).into_par_iter().map(|chunk_idx| {
    let mut xtx_local = vec![0.0; p * p];
    for k in chunk_start..chunk_end {
        for i in 0..p {
            let xki_w = x_slice[k*p + i] * w_slice[k];
            for j in i..p {
                xtx_local[i*p + j] += xki_w * x_slice[k*p + j];
            }
        }
    }
    xtx_local
}).reduce(sum)
```

This is correct and well-parallelised via rayon, but:

- **No SIMD micro-kernels**: Relies on compiler auto-vectorisation rather
  than hand-tuned assembly
- **Fixed chunk size (8192)**: Chosen for parallelism, not cache optimality.
  Each chunk processes 8192 × 170 × 8 ≈ 11 MB, larger than L2 cache
- **Accumulator overhead**: Each rayon thread accumulates a full p×p matrix
  (170² × 8 = 231 KB), then all are reduced
- **No prefetching**: Relies on hardware auto-prefetch

For small p (< ~50), this is competitive with BLAS because the working set
fits in L2. At p≈170, BLAS's cache-blocking wins.

---

## Factor 2: Why glum Converges in Fewer Iterations (3 vs 6)

### The penalty structure difference

In the fixed-alpha benchmark (Mode A), the two libraries use different
penalty structures:

**glum — D'D difference penalty (P-spline)**:

```
Penalty = α · β' P₂ β    where P₂ = block_diag(D'D, D'D, ..., D'D)

D = 2nd difference matrix:
    ┌  1 -2  1  0  0 ┐
    │  0  1 -2  1  0 │
    └  0  0  1 -2  1 ┘
```

Penalises the **roughness** of the fitted curve — adjacent coefficients are
encouraged to be similar.

**rustystats (fixed-df mode) — scalar L2 (ridge)**:

```
Penalty = α · β'β    (equivalently P₂ = I)
```

Penalises the **magnitude** of all coefficients equally.

### Why this matters for convergence

The D'D penalty produces a better-conditioned system (X'WX + α·D'D) because
the penalty eigenvalues align with the spline roughness modes. The identity
penalty shifts all eigenvalues uniformly, leaving the condition number high
and requiring more iterations.

### Important nuance

**rustystats already implements D'D penalties** in its smooth GLM path
(`smooth_glm.rs`). When using `{"type": "bs"}` (default penalized smooth),
it uses `penalty_matrix(k, 2)` which is exactly D'D. This path is used in
Mode B (auto GCV). The iteration difference in Mode A exists only because
fixed-df splines fall through to the standard IRLS path with scalar ridge.

---

## Factor 3: The GCV Bottleneck (Mode B) — FIXED

The biggest performance issue **was** the GCV lambda optimiser. Before the
fix, it recomputed X'WX from scratch using a single-threaded O(n·p²) loop
(`compute_xtwx_xtwz_nalg` in `gcv_optimizer.rs`), separate from the IRLS
loop's own X'WX computation. This caused:

- **~600s** at 1M rows (before fix)
- Scaling as O(n²) due to redundant recomputation

### The Fix (implemented)

The IRLS loop in `smooth_glm.rs` now computes X'WX **once** per iteration
via the fast parallel `compute_xtwx_xtwz` and passes it to both:
1. The GCV optimizer via `MultiTermGCVOptimizer::new_from_cached()`
2. The WLS solver via `solve_wls_from_precomputed()`

RSS is computed from cached matrices using the identity:
```
RSS = z'Wz - 2·β'(X'Wz) + β'(X'WX)β
```
which is O(p²) instead of O(n·p). The GCV optimizer no longer stores the
full n-dimensional X, z, w arrays, also saving ~1.3 GB of memory.

### Result

| | Before fix | After fix | Speedup |
|---|---|---|---|
| 50K rows | 22.8s | 2.09s | **10.9×** |
| 100K rows | 64.0s | 3.20s | **20.0×** |
| 1M rows | ~600s | 11.3s | **~53×** |

---

## What rustystats Needs to Change

### ~~Priority 1: Fix the GCV Optimizer~~ ✅ DONE

Implemented: X'WX is computed once per IRLS iteration and shared between
the GCV optimizer and WLS solver. See Factor 3 above for details.

Files changed:
- `irls.rs`: Made `compute_xtwx_xtwz` public, added `solve_wls_from_precomputed`
- `gcv_optimizer.rs`: Added `new_from_cached`, cached RSS via z'Wz identity
- `smooth_glm.rs`: Restructured IRLS loop to compute X'WX once

### Priority 2: BLAS Linkage (closes the 1.5× per-iteration gap)

The X'WX computation in the IRLS loop should dispatch to BLAS. The optimal
approach is `dsyrk` (symmetric rank-k update) which exploits the symmetry
of X'WX:

**Option A: ndarray + openblas-system**

```toml
# Cargo.toml
[dependencies]
ndarray = { version = "0.15", features = ["blas"] }
blas-src = { version = "0.8", features = ["openblas"] }
openblas-src = { version = "0.10", features = ["system"] }
```

**Option B: Direct cblas FFI**

```rust
extern "C" {
    fn cblas_dsyrk(
        order: i32, uplo: i32, trans: i32,
        n: i32, k: i32,
        alpha: f64, a: *const f64, lda: i32,
        beta: f64, c: *mut f64, ldc: i32,
    );
}
```

`dsyrk` does half the work of `dgemm` by exploiting symmetry.

**Expected impact**: Per-iteration time from ~1.31s to ~0.87s, matching glum.

### ~~Priority 3: Route Fixed-df Splines Through D'D~~ ❌ Not effective

Tested: routing `{"type": "bs", "df": 11}` with `alpha=1.0` through the
smooth_glm path with D'D penalty instead of scalar ridge. Result:

- **Before (scalar ridge)**: 7.70s, 6 iterations
- **After (D'D via smooth path)**: 9.96s, 6 iterations — **slower**

The smooth_glm IRLS path has inherent overhead (penalty matrix construction,
GCV optimizer creation, nalgebra conversions) that outweighs any convergence
benefit. And iterations didn't decrease — α=1.0 with D'D doesn't necessarily
converge faster than α=1.0 with ridge because the effective regularization
strength differs (D'D has different eigenvalues than I).

To properly implement this, the **standard IRLS path** in `irls.rs` would
need to accept an optional penalty matrix directly, avoiding the smooth_glm
overhead. This is a larger refactor.

---

## Current State and Remaining Gaps

| | Mode A (fixed α) | Mode B (auto GCV) | glum |
|---|---------|-----------|------|
| Per-iteration | 1.31s | ~1.62s | 0.85s |
| Iterations | 6 | 7 | 3 |
| **Total** | **7.70s** | **11.3s** | **2.54s** |

P1 (GCV caching) is done — auto-GCV went from ~600s to 11.3s.
The remaining gap to glum is 4.5× (Mode B) and consists of:
- **~1.5×** from BLAS vs hand-written X'WX (Priority 2)
- **~2.3×** from more iterations (7 vs 3) and GCV overhead per iteration
- Proper D'D routing in the standard IRLS path (future work)

---

## Appendix: Design Matrix Characteristics

The B-spline basis matrix has specific structure worth noting:

```
X (1M × ~170) — cubic B-spline basis, ~11 cols per feature × 15 features

Sparsity: ~93% zeros
  Each row has exactly (degree + 1) = 4 nonzeros per feature
  = 60 nonzeros per row out of ~170 columns

Block structure:
  X = [ B₁ | B₂ | ... | B₁₅ ]
  Each Bᵢ is (1M × 11) with 4 nonzeros per row (band structure)
```

This sparsity is **not exploited by either library** — both treat X as a
dense matrix. Exploiting the banded B-spline structure could give further
gains, but the three priorities above are sufficient to match or beat glum.

## Appendix: Memory

### Investigation Results

The ~2,500 MB peak RSS delta during `glm_dict().fit()` was investigated.
The breakdown:

| Component | Memory | Status |
|---|---|---|
| Python design matrix build (spline bases + hstack) | ~1,300 MB | Inherent |
| Rust solver copy of X (`.to_owned()`) | ~1,300 MB | **FIXED → 0** |
| IRLS working vectors (eta, mu, weights, etc.) | ~120 MB | Inherent |

### Fix: Zero-Copy Design Matrix (implemented)

Changed all solver functions (`fit_glm_unified`, `fit_smooth_glm_full_matrix`,
`compute_xtwx_xtwz`, etc.) to accept `ArrayView2<f64>` instead of `&Array2<f64>`.
The PyO3 entry points now pass `x.as_array()` (a zero-copy view of the numpy
array) instead of `x.as_array().to_owned()` (a full copy).

**Result**: Rust solver memory delta dropped from **2,559 MB → 120 MB** (measured
by calling `fit_glm_py` directly with a pre-built design matrix).

The full pipeline still shows ~2,500 MB because the Python-side
`InteractionBuilder` allocates ~1,300 MB building the spline basis columns
(15 features × 1M rows × 11 basis functions × 8 bytes) before hstacking.
This is inherent to any library that builds a dense design matrix.

glum's 80 MB comparison is misleading — glum's benchmark pre-builds the design
matrix via `SplineTransformer.fit_transform()` *before* the memory measurement
starts, so its 80 MB only reflects the fit overhead.

Files changed:
- `irls.rs`: All solver functions accept `ArrayView2` for design matrix
- `coordinate_descent.rs`: Same
- `smooth_glm.rs`: Same
- `fitting_py.rs`: Pass numpy view instead of copying

### GCV Scheduling (implemented)

The GCV lambda optimizer now tracks lambda stability across iterations.
Once all lambdas change by < 1% for 2 consecutive iterations, GCV is
skipped for the remaining IRLS iterations. This saves ~0.2s at 1M rows.
