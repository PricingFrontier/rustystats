// =============================================================================
// Per-function allocation profiler for the suspect diagnostic functions.
//
// This is dev-only infrastructure (a cargo example, not part of the lib build)
// and installs a custom global allocator that wraps the System allocator to
// count every alloc/dealloc. It produces, per function:
//
//   - peak resident bytes during the call (max(live) - baseline)
//   - total bytes allocated during the call (every alloc summed, ignoring frees)
//   - number of allocations
//   - wall time
//
// Run it with:
//
//     cargo run --release --example measure_diagnostics_alloc -p rustystats-core
//
// Output: a sorted table to stdout AND a JSON file at
// /tmp/rust_alloc_profile.json with the same data.
//
// Why a custom allocator instead of `peak_alloc`/`dhat`?
//  - peak_alloc only exposes peak/current — no per-call total or count.
//  - dhat is great but adds a heavy dependency and a runtime startup hook.
//  - A 60-line AtomicU64-counter wrapper around System gives us everything
//    we need (peak, total, count, current) with zero extra deps.
// =============================================================================

use ndarray::Array1;
use rustystats_core::diagnostics::{
    compute_ae_by_decile, compute_discrimination_stats, compute_residual_summary,
    gamma_deviance_loss, log_loss, negbinomial_deviance_loss, null_deviance_with_offset,
    poisson_deviance_loss, tweedie_deviance_loss,
};

use std::alloc::{GlobalAlloc, Layout, System};
use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

// -----------------------------------------------------------------------------
// Tracking global allocator
// -----------------------------------------------------------------------------
//
// We keep four counters:
//   live_bytes  : currently allocated (incremented on alloc, decremented on dealloc)
//   peak_bytes  : highest value of live_bytes since the last reset
//   total_bytes : cumulative bytes allocated (only goes up)
//   alloc_count : cumulative allocation count (only goes up)
//
// All updates are Relaxed atomics — we never read from another thread inside
// the alloc fast path, only at measurement boundaries (between function calls
// where there are no other active threads in this single-threaded benchmark).
// -----------------------------------------------------------------------------

static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static PEAK_BYTES: AtomicU64 = AtomicU64::new(0);
static TOTAL_BYTES: AtomicU64 = AtomicU64::new(0);
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let size = layout.size() as u64;
            // fetch_add returns the previous value; live = previous + size
            let prev_live = LIVE_BYTES.fetch_add(size, Ordering::Relaxed);
            let new_live = prev_live + size;
            // Bump the peak if necessary (compare-and-swap loop, lock-free).
            let mut peak = PEAK_BYTES.load(Ordering::Relaxed);
            while new_live > peak {
                match PEAK_BYTES.compare_exchange_weak(
                    peak,
                    new_live,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(observed) => peak = observed,
                }
            }
            TOTAL_BYTES.fetch_add(size, Ordering::Relaxed);
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE_BYTES.fetch_sub(layout.size() as u64, Ordering::Relaxed);
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // Defer to System for the actual zeroing then run the same accounting
        // we do in `alloc`. We don't call self.alloc to avoid double-counting.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            let size = layout.size() as u64;
            let prev_live = LIVE_BYTES.fetch_add(size, Ordering::Relaxed);
            let new_live = prev_live + size;
            let mut peak = PEAK_BYTES.load(Ordering::Relaxed);
            while new_live > peak {
                match PEAK_BYTES.compare_exchange_weak(
                    peak,
                    new_live,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(observed) => peak = observed,
                }
            }
            TOTAL_BYTES.fetch_add(size, Ordering::Relaxed);
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // Treat as a free of the old block + an allocation of the new block,
        // which is what realloc does conceptually. (System.realloc is allowed
        // to reuse the same address — that's fine, we still account both ends.)
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            let old_size = layout.size() as u64;
            let new_size_u64 = new_size as u64;
            // Net change in live bytes
            LIVE_BYTES.fetch_sub(old_size, Ordering::Relaxed);
            let prev_live = LIVE_BYTES.fetch_add(new_size_u64, Ordering::Relaxed);
            let new_live = prev_live + new_size_u64;
            let mut peak = PEAK_BYTES.load(Ordering::Relaxed);
            while new_live > peak {
                match PEAK_BYTES.compare_exchange_weak(
                    peak,
                    new_live,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(observed) => peak = observed,
                }
            }
            // Only count the *additional* allocation in total/count when the
            // block grows; otherwise we'd inflate counts on every Vec push.
            // Charging the full new size is the conservative read of "bytes
            // allocated", matching how dhat reports realloc growth.
            if new_size_u64 > old_size {
                TOTAL_BYTES.fetch_add(new_size_u64 - old_size, Ordering::Relaxed);
            }
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        new_ptr
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

// -----------------------------------------------------------------------------
// Measurement helpers
// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Measurement {
    function: String,
    peak_bytes: u64,
    total_alloc_bytes: u64,
    n_allocations: usize,
    elapsed_ms: f64,
}

/// Reset the per-call counters. Live bytes is left alone — it represents the
/// process baseline we want to measure peaks *above*.
fn reset_counters() {
    let live = LIVE_BYTES.load(Ordering::Relaxed);
    PEAK_BYTES.store(live, Ordering::Relaxed);
    TOTAL_BYTES.store(0, Ordering::Relaxed);
    ALLOC_COUNT.store(0, Ordering::Relaxed);
}

/// Run `f` once with counters reset, returning peak above baseline, total
/// bytes allocated, allocation count, and elapsed wall-time milliseconds.
///
/// `R` is passed through so the compiler can't elide the call. We `black_box`
/// the result before dropping it.
fn measure<R, F: FnOnce() -> R>(label: &str, f: F) -> Measurement {
    let baseline_live = LIVE_BYTES.load(Ordering::Relaxed);
    reset_counters();
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    let peak_live = PEAK_BYTES.load(Ordering::Relaxed);
    let total = TOTAL_BYTES.load(Ordering::Relaxed);
    let count = ALLOC_COUNT.load(Ordering::Relaxed);

    // Make sure the optimiser couldn't elide f().
    std::hint::black_box(&result);
    drop(result);

    Measurement {
        function: label.to_string(),
        peak_bytes: peak_live.saturating_sub(baseline_live),
        total_alloc_bytes: total,
        n_allocations: count,
        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
    }
}

/// Run `f` 1 + REPEATS times and report the median peak-bytes measurement (so
/// that one-off allocator quirks don't dominate). We discard the first run as
/// warmup and keep the run with the median peak.
fn measure_n<R, F: FnMut() -> R>(label: &str, mut f: F) -> Measurement {
    const REPEATS: usize = 2;

    // warmup
    let _ = measure(label, &mut f);

    let mut runs: Vec<Measurement> = (0..REPEATS).map(|_| measure(label, &mut f)).collect();
    // Sort by peak_bytes and take the median (with REPEATS=2 that's the lower
    // of the two — fine for our purposes; we want a representative number, not
    // a worst-case).
    runs.sort_by_key(|m| m.peak_bytes);
    runs.swap_remove(runs.len() / 2)
}

// -----------------------------------------------------------------------------
// Synthetic 1M-row Poisson dataset (mirrors bench_diagnostics_timing.py)
// -----------------------------------------------------------------------------

fn make_dataset(n: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    // A tiny LCG so we don't pull in `rand` / `rand_distr` — we don't need
    // statistical quality, only enough variation to keep the optimiser honest.
    let mut state: u64 = 0xCAFEBABE_DEADBEEF;
    let mut next_u64 = || -> u64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };
    let mut next_uniform = || -> f64 {
        // Top 53 bits → uniform(0,1)
        ((next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    };

    let mut y = Array1::<f64>::zeros(n);
    let mut mu = Array1::<f64>::zeros(n);
    let mut exposure = Array1::<f64>::zeros(n);
    let mut weights = Array1::<f64>::ones(n);

    for i in 0..n {
        let exp_i = 0.1 + 0.9 * next_uniform(); // Uniform(0.1, 1.0)
        let rate = 0.05 + 0.95 * next_uniform(); // mean rate per unit exposure
        let mu_i = rate * exp_i;
        // Approximate a Poisson draw: round(mu + sqrt(mu) * (uniform - 0.5))
        // Not statistically correct but good enough for benchmark inputs.
        let noise = next_uniform() - 0.5;
        let y_i = (mu_i + mu_i.sqrt() * noise).max(0.0).round();
        y[i] = y_i;
        mu[i] = mu_i.max(1e-9);
        exposure[i] = exp_i;
        weights[i] = 1.0 + 0.5 * next_uniform(); // mild weight variation
    }

    (y, mu, exposure, weights)
}

// -----------------------------------------------------------------------------
// Suspect functions
// -----------------------------------------------------------------------------

fn main() {
    const N: usize = 1_000_000;
    println!("Building synthetic dataset (n = {N})...");
    let (y, mu, exposure, weights) = make_dataset(N);

    // Pre-compute a residuals array so compute_residual_summary has realistic input.
    let residuals: Array1<f64> = ndarray::Zip::from(&y)
        .and(&mu)
        .map_collect(|&yi, &mui| yi - mui);

    // Pre-compute a log offset so null_deviance_with_offset has realistic input.
    let offset: Array1<f64> = exposure.mapv(|e| e.ln());

    println!("Running measurements (1 warmup + 2 measured per function)...\n");

    let mut results: Vec<Measurement> = Vec::new();

    // ---- loss.rs functions (all have the Vec<f64> collect-then-sum pattern) ----
    results.push(measure_n("loss::poisson_deviance_loss", || {
        poisson_deviance_loss(&y, &mu, None)
    }));
    results.push(measure_n("loss::poisson_deviance_loss(weighted)", || {
        poisson_deviance_loss(&y, &mu, Some(&weights))
    }));
    results.push(measure_n("loss::gamma_deviance_loss", || {
        gamma_deviance_loss(&y, &mu, None)
    }));
    results.push(measure_n("loss::tweedie_deviance_loss(p=1.5)", || {
        tweedie_deviance_loss(&y, &mu, 1.5, None)
    }));
    results.push(measure_n("loss::negbinomial_deviance_loss", || {
        negbinomial_deviance_loss(&y, &mu, 1.5, None)
    }));
    // Bernoulli-ish y/mu for log_loss (clamp mu in [1e-15, 1-1e-15])
    let mu_p: Array1<f64> = mu.mapv(|m| (m / (m + 1.0)).clamp(1e-9, 1.0 - 1e-9));
    let y_b: Array1<f64> = y.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
    results.push(measure_n("loss::log_loss", || log_loss(&y_b, &mu_p, None)));

    // ---- model_fit.rs ----
    // null_deviance_with_offset allocates an Array1<f64> of unit deviances.
    results.push(measure_n(
        "model_fit::null_deviance_with_offset(poisson)",
        || null_deviance_with_offset(&y, "poisson", None, Some(&offset)).expect("ok"),
    ));
    results.push(measure_n(
        "model_fit::null_deviance_with_offset(gaussian)",
        || null_deviance_with_offset(&y, "gaussian", None, None).expect("ok"),
    ));

    // ---- calibration.rs ----
    // compute_discrimination_stats sorts an indices vector and walks it.
    results.push(measure_n(
        "calibration::compute_discrimination_stats",
        || compute_discrimination_stats(&y, &mu, Some(&exposure)),
    ));

    // ---- decile.rs ----
    // compute_ae_by_decile gathers y/mu/exposure into 3 fresh Vecs after sorting.
    results.push(measure_n("decile::compute_ae_by_decile", || {
        compute_ae_by_decile(&y, &mu, Some(&exposure), 10, None)
    }));

    // ---- residuals.rs ----
    // compute_residual_summary clones residuals into a Vec and sorts it.
    results.push(measure_n("residuals::compute_residual_summary", || {
        compute_residual_summary(&residuals)
    }));

    // -------------------------------------------------------------------------
    // Sort & print
    // -------------------------------------------------------------------------
    results.sort_by(|a, b| b.peak_bytes.cmp(&a.peak_bytes));

    println!(
        "{:<55}  {:>14}  {:>14}  {:>10}  {:>10}",
        "function", "peak_bytes", "total_alloc", "n_allocs", "elapsed_ms"
    );
    println!("{}", "-".repeat(112));
    for r in &results {
        println!(
            "{:<55}  {:>14}  {:>14}  {:>10}  {:>10.2}",
            r.function,
            fmt_bytes(r.peak_bytes),
            fmt_bytes(r.total_alloc_bytes),
            r.n_allocations,
            r.elapsed_ms
        );
    }
    println!();

    // -------------------------------------------------------------------------
    // JSON dump
    // -------------------------------------------------------------------------
    let json = build_json(N, &results);
    let path = "/tmp/rust_alloc_profile.json";
    let mut f = File::create(path).expect("can't open output file");
    f.write_all(json.as_bytes()).expect("write json");
    println!("Wrote JSON profile to {path}");
}

// Format bytes as decimal MB / KB for the human-readable column.
fn fmt_bytes(b: u64) -> String {
    if b >= 1_000_000 {
        format!("{:.1} MB", b as f64 / 1e6)
    } else if b >= 1_000 {
        format!("{:.1} KB", b as f64 / 1e3)
    } else {
        format!("{} B", b)
    }
}

// Hand-rolled JSON to avoid pulling in serde for a dev-only example.
fn build_json(n_rows: usize, results: &[Measurement]) -> String {
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str(&format!("  \"n_rows\": {},\n", n_rows));
    s.push_str("  \"results\": [\n");
    for (i, r) in results.iter().enumerate() {
        let comma = if i + 1 == results.len() { "" } else { "," };
        s.push_str(&format!(
            "    {{\"function\": \"{}\", \"peak_bytes\": {}, \"total_alloc_bytes\": {}, \"n_allocations\": {}, \"elapsed_ms\": {:.3}}}{}\n",
            r.function, r.peak_bytes, r.total_alloc_bytes, r.n_allocations, r.elapsed_ms, comma
        ));
    }
    s.push_str("  ]\n");
    s.push_str("}\n");
    s
}
