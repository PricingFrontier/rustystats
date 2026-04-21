// =============================================================================
// IRLS Solver Benchmarks
// =============================================================================
//
// Measures GLM fitting performance at various data sizes and families.
// Run with: cargo bench --workspace
// =============================================================================

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};

use rustystats_core::families::{GammaFamily, GaussianFamily, PoissonFamily};
use rustystats_core::links::{IdentityLink, LogLink};
use rustystats_core::solvers::{fit_glm_unified, FitConfig};

/// Generate synthetic GLM data for benchmarking.
fn generate_data(n: usize, p: usize) -> (Array1<f64>, Array2<f64>) {
    // Design matrix: intercept + random-ish predictors
    let x = Array2::from_shape_fn((n, p), |(i, j)| {
        if j == 0 {
            1.0
        } else {
            ((i * (j + 1)) as f64 / n as f64).sin() * 2.0
        }
    });

    // Response: based on linear predictor with noise
    let true_coefs: Vec<f64> = (0..p)
        .map(|j| if j == 0 { 1.0 } else { 0.1 / (j as f64) })
        .collect();
    let true_coefs = Array1::from_vec(true_coefs);
    let eta = x.dot(&true_coefs);

    // Poisson-like response
    let y = eta.mapv(|e| e.exp().max(0.1));
    (y, x)
}

fn bench_gaussian(c: &mut Criterion) {
    let mut group = c.benchmark_group("IRLS Gaussian");
    let family = GaussianFamily;
    let link = IdentityLink;
    let config = FitConfig::default();

    for &(n, p) in &[(100, 5), (1000, 10), (5000, 20)] {
        let (y, x) = generate_data(n, p);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n={n}_p={p}")),
            &(n, p),
            |b, _| {
                b.iter(|| {
                    fit_glm_unified(
                        black_box(&y),
                        black_box(x.view()),
                        &family,
                        &link,
                        &config,
                        None,
                        None,
                        None,
                    )
                    .expect("test setup should be valid");
                });
            },
        );
    }
    group.finish();
}

fn bench_poisson(c: &mut Criterion) {
    let mut group = c.benchmark_group("IRLS Poisson");
    let family = PoissonFamily;
    let link = LogLink;
    let config = FitConfig::default();

    for &(n, p) in &[(100, 5), (1000, 10), (5000, 20)] {
        let (y, x) = generate_data(n, p);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n={n}_p={p}")),
            &(n, p),
            |b, _| {
                b.iter(|| {
                    fit_glm_unified(
                        black_box(&y),
                        black_box(x.view()),
                        &family,
                        &link,
                        &config,
                        None,
                        None,
                        None,
                    )
                    .expect("test setup should be valid");
                });
            },
        );
    }
    group.finish();
}

fn bench_gamma(c: &mut Criterion) {
    let mut group = c.benchmark_group("IRLS Gamma");
    let family = GammaFamily;
    let link = LogLink;
    let config = FitConfig::default();

    for &(n, p) in &[(100, 5), (1000, 10), (5000, 20)] {
        let (y, x) = generate_data(n, p);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n={n}_p={p}")),
            &(n, p),
            |b, _| {
                b.iter(|| {
                    fit_glm_unified(
                        black_box(&y),
                        black_box(x.view()),
                        &family,
                        &link,
                        &config,
                        None,
                        None,
                        None,
                    )
                    .expect("test setup should be valid");
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_gaussian, bench_poisson, bench_gamma);
criterion_main!(benches);
