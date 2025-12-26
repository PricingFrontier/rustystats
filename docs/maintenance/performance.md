# Performance Guide

This guide covers performance optimization, profiling, and benchmarking for RustyStats.

## Performance Characteristics

### Complexity Analysis

| Operation | Complexity | Notes |
|-----------|------------|-------|
| IRLS iteration | O(np² + p³) | X'WX computation + solve |
| Single coordinate descent step | O(np) | Update one coefficient |
| Full CD iteration (all p) | O(np²) | With Gram matrix |
| Spline basis evaluation | O(n × df × degree) | Per-observation |
| Target encoding | O(n × n_perms) | With permutations |

### Typical Performance

| Dataset | Features | Family | Time (release) |
|---------|----------|--------|----------------|
| 10K rows | 20 | Poisson | ~20ms |
| 100K rows | 50 | Poisson | ~200ms |
| 500K rows | 50 | Poisson | ~800ms |
| 1M rows | 100 | Gaussian | ~2s |

---

## Build Optimization

### Always Use Release Mode

```bash
# Development (fast compile, slow run)
maturin develop

# Release (slow compile, fast run)
maturin develop --release
```

Release mode enables:
- Optimizations (`-O3`)
- LTO (Link-Time Optimization)
- Inlining
- SIMD auto-vectorization

### Cargo Profile

```toml
# Cargo.toml
[profile.release]
lto = true
codegen-units = 1
opt-level = 3
```

---

## Parallelization

### Rayon Configuration

Rayon automatically uses all available cores. Control with:

```rust
// Limit threads
rayon::ThreadPoolBuilder::new()
    .num_threads(4)
    .build_global()
    .unwrap();
```

Or environment variable:
```bash
RAYON_NUM_THREADS=4 python script.py
```

### Parallel Patterns

#### Parallel Map

```rust
let results: Vec<_> = data.par_iter()
    .map(|x| expensive_compute(x))
    .collect();
```

#### Parallel Fold-Reduce

```rust
let sum = (0..n).into_par_iter()
    .fold(|| 0.0, |acc, i| acc + data[i])
    .reduce(|| 0.0, |a, b| a + b);
```

#### When to Parallelize

- **Good**: Independent computations, large n
- **Bad**: Small n (overhead dominates), sequential dependencies

Rule of thumb: Parallelize when n > 1000.

---

## Memory Optimization

### Avoid Unnecessary Copies

```rust
// Bad: clones array
fn process(data: Array1<f64>) { ... }

// Good: borrows
fn process(data: &Array1<f64>) { ... }

// Good: moves ownership when needed
fn consume(data: Array1<f64>) -> Array1<f64> { ... }
```

### Pre-allocate Buffers

```rust
// Bad: allocates in loop
for iter in 0..max_iter {
    let buffer = Array1::zeros(n);
    // ...
}

// Good: allocate once
let mut buffer = Array1::zeros(n);
for iter in 0..max_iter {
    buffer.fill(0.0);
    // ...
}
```

### Use Views

```rust
// No copy - just a view
let column = matrix.column(j);
let slice = array.slice(s![10..20]);
```

### Memory Layout

ndarray uses row-major by default. Access patterns matter:

```rust
// Good: row-wise access (contiguous)
for row in matrix.rows() {
    for &val in row.iter() { ... }
}

// Bad: column-wise access (strided)
for col in matrix.columns() {
    for &val in col.iter() { ... }
}
```

---

## Profiling

### Rust Profiling

#### Using perf (Linux)

```bash
# Build with debug symbols
cargo build --release

# Record profile
perf record -g ./target/release/benchmark

# View report
perf report
```

#### Using flamegraph

```bash
# Install
cargo install flamegraph

# Generate
cargo flamegraph --bin benchmark
```

#### Using cargo-instruments (macOS)

```bash
cargo install cargo-instruments
cargo instruments -t "Time Profiler" --release
```

### Python Profiling

#### cProfile

```python
import cProfile
import pstats

with cProfile.Profile() as pr:
    result = rs.fit_glm(y, X, family="poisson")

stats = pstats.Stats(pr)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

#### line_profiler

```python
# Install: pip install line_profiler

@profile
def benchmark():
    result = rs.fit_glm(y, X, family="poisson")
    return result

# Run: kernprof -l -v script.py
```

---

## Benchmarking

### Rust Benchmarks

```rust
// benches/irls.rs
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_irls(c: &mut Criterion) {
    let (y, x) = generate_data(10000, 20);
    
    c.bench_function("irls_10k_20", |b| {
        b.iter(|| {
            fit_glm(&y, &x, &PoissonFamily, &LogLink, &IRLSConfig::default())
        })
    });
}

criterion_group!(benches, bench_irls);
criterion_main!(benches);
```

Run with:
```bash
cargo bench
```

### Python Benchmarks

```python
import time
import numpy as np
import rustystats as rs

def benchmark(n, p, family="poisson", n_runs=10):
    """Benchmark GLM fitting."""
    y = np.random.poisson(5, n).astype(float)
    X = np.column_stack([np.ones(n), np.random.randn(n, p)])
    
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = rs.fit_glm(y, X, family=family)
        times.append(time.perf_counter() - start)
    
    print(f"n={n}, p={p}, family={family}")
    print(f"  Mean: {np.mean(times)*1000:.1f}ms")
    print(f"  Std:  {np.std(times)*1000:.1f}ms")
    print(f"  Iterations: {result.iterations}")

# Run benchmarks
for n in [10000, 100000, 500000]:
    benchmark(n, 20)
```

---

## Common Bottlenecks

### 1. X'WX Computation

The Gram matrix computation is often the bottleneck:

```rust
// O(np²) - parallelized
let xtwx = compute_gram_matrix(x, w);
```

Optimization: Use flat Vec instead of Array2 for better cache performance.

### 2. Matrix Solve

Solving (X'WX)β = X'Wz is O(p³):

```rust
let beta = cholesky_solve(&xtwx, &xtwz)?;
```

Optimization: For very large p, consider iterative solvers.

### 3. Memory Allocation

Allocations in hot loops hurt performance:

```rust
// Profile with: MALLOC_CONF=prof:true

// Look for patterns like:
for iter in 0..max_iter {
    let temp = compute();  // Allocation every iteration
}
```

### 4. Cache Misses

Random memory access patterns cause cache misses:

```rust
// Bad: random access
for &i in random_indices.iter() {
    result += data[i];
}

// Good: sequential access
for &val in data.iter() {
    result += val;
}
```

---

## Optimization Checklist

- [ ] Build with `--release`
- [ ] Enable LTO in Cargo.toml
- [ ] Parallelize with Rayon where appropriate
- [ ] Pre-allocate buffers outside loops
- [ ] Use views instead of copies
- [ ] Prefer row-major access patterns
- [ ] Profile to find actual bottlenecks
- [ ] Benchmark before and after changes

---

## Comparison with Other Libraries

### vs Statsmodels

| Operation | RustyStats | Statsmodels |
|-----------|------------|-------------|
| Poisson 100K × 20 | ~100ms | ~500ms |
| Lasso GLM 100K × 50 | ~500ms | N/A |

### vs glmnet (R)

| Operation | RustyStats | glmnet |
|-----------|------------|--------|
| Lasso path 100K × 100 | ~2s | ~1.5s |

glmnet is highly optimized; RustyStats is competitive.

### Why RustyStats is Fast

1. **Rust**: No GC, predictable performance
2. **Rayon**: Automatic parallelization
3. **SIMD**: Auto-vectorization by LLVM
4. **Cache-friendly**: Careful memory layout
5. **Zero-copy**: NumPy interop where possible
