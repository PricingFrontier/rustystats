// =============================================================================
// Property-Based Tests for RustyStats Core
// =============================================================================
//
// These tests use proptest to verify fundamental mathematical invariants
// that should hold for ALL valid inputs, not just hand-picked examples.
// =============================================================================

use ndarray::{Array1, Array2};
use proptest::prelude::*;

use rustystats_core::constants::{IRLS_ACCEPT_REL_SLACK, ZERO_TOL};
use rustystats_core::families::{
    BinomialFamily, Family, GammaFamily, GaussianFamily, NegativeBinomialFamily, PoissonFamily,
    QuasiBinomialFamily, QuasiPoissonFamily, TweedieFamily,
};
use rustystats_core::links::{IdentityLink, Link, LogLink, LogitLink};
use rustystats_core::solvers::{fit_glm_unified, FitConfig};

// =============================================================================
// Helper Strategies
// =============================================================================

/// Generate a positive float suitable for mu values
fn positive_mu() -> impl Strategy<Value = f64> {
    0.01f64..1000.0
}

/// Generate a probability in (0, 1)
fn probability() -> impl Strategy<Value = f64> {
    0.01f64..0.99
}

/// Generate a vector of positive mu values
fn positive_mu_vec(n: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(positive_mu(), n)
}

/// Generate a vector of probabilities
fn probability_vec(n: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(probability(), n)
}

// =============================================================================
// Link / Inverse Roundtrip
// =============================================================================

proptest! {
    #[test]
    fn log_link_roundtrip(values in positive_mu_vec(10)) {
        let link = LogLink;
        let mu = Array1::from_vec(values);
        let eta = link.link(&mu);
        let mu_back = link.inverse(&eta);
        for (a, b) in mu.iter().zip(mu_back.iter()) {
            prop_assert!((a - b).abs() < 1e-8 * a.abs().max(1.0),
                "log roundtrip failed: {} vs {}", a, b);
        }
    }

    #[test]
    fn identity_link_roundtrip(values in prop::collection::vec(-100.0f64..100.0, 10)) {
        let link = IdentityLink;
        let mu = Array1::from_vec(values);
        let eta = link.link(&mu);
        let mu_back = link.inverse(&eta);
        for (a, b) in mu.iter().zip(mu_back.iter()) {
            prop_assert!((a - b).abs() < 1e-12,
                "identity roundtrip failed: {} vs {}", a, b);
        }
    }

    #[test]
    fn logit_link_roundtrip(values in probability_vec(10)) {
        let link = LogitLink;
        let mu = Array1::from_vec(values);
        let eta = link.link(&mu);
        let mu_back = link.inverse(&eta);
        for (a, b) in mu.iter().zip(mu_back.iter()) {
            prop_assert!((a - b).abs() < 1e-8,
                "logit roundtrip failed: {} vs {}", a, b);
        }
    }
}

// =============================================================================
// Deviance Non-Negativity
// =============================================================================

proptest! {
    #[test]
    fn gaussian_deviance_nonneg(
        y_vals in prop::collection::vec(0.0f64..100.0, 10),
        mu_vals in prop::collection::vec(0.01f64..100.0, 10)
    ) {
        let family = GaussianFamily;
        let y = Array1::from_vec(y_vals);
        let mu = Array1::from_vec(mu_vals);
        let dev = family.unit_deviance(&y, &mu);
        for &d in dev.iter() {
            prop_assert!(d >= 0.0, "Gaussian unit deviance negative: {}", d);
        }
    }

    #[test]
    fn poisson_deviance_nonneg(
        y_vals in prop::collection::vec(0.0f64..50.0, 10),
        mu_vals in positive_mu_vec(10)
    ) {
        let family = PoissonFamily;
        let y = Array1::from_vec(y_vals);
        let mu = Array1::from_vec(mu_vals);
        let dev = family.unit_deviance(&y, &mu);
        for &d in dev.iter() {
            prop_assert!(d >= -1e-10, "Poisson unit deviance negative: {}", d);
        }
    }

    #[test]
    fn gamma_deviance_nonneg(
        y_vals in positive_mu_vec(10),
        mu_vals in positive_mu_vec(10)
    ) {
        let family = GammaFamily;
        let y = Array1::from_vec(y_vals);
        let mu = Array1::from_vec(mu_vals);
        let dev = family.unit_deviance(&y, &mu);
        for &d in dev.iter() {
            prop_assert!(d >= -1e-10, "Gamma unit deviance negative: {}", d);
        }
    }

    #[test]
    fn binomial_deviance_nonneg(
        y_vals in probability_vec(10),
        mu_vals in probability_vec(10)
    ) {
        let family = BinomialFamily;
        let y = Array1::from_vec(y_vals);
        let mu = Array1::from_vec(mu_vals);
        let dev = family.unit_deviance(&y, &mu);
        for &d in dev.iter() {
            prop_assert!(d >= -1e-10, "Binomial unit deviance negative: {}", d);
        }
    }

    #[test]
    fn negbinomial_deviance_nonneg(
        // Counts including exact zeros (exercises the y == 0 branch) plus
        // positive mu and a representative theta sweep.
        y_vals in prop::collection::vec(0.0f64..50.0, 10),
        mu_vals in positive_mu_vec(10),
        theta in 0.1f64..50.0
    ) {
        let family = NegativeBinomialFamily::new(theta).expect("theta > 0 valid");
        let y = Array1::from_vec(y_vals);
        let mu = Array1::from_vec(mu_vals);
        let dev = family.unit_deviance(&y, &mu);
        for &d in dev.iter() {
            prop_assert!(d >= -1e-10, "NegativeBinomial unit deviance negative: {}", d);
        }
    }

    #[test]
    fn quasipoisson_deviance_nonneg(
        y_vals in prop::collection::vec(0.0f64..50.0, 10),
        mu_vals in positive_mu_vec(10)
    ) {
        let family = QuasiPoissonFamily;
        let y = Array1::from_vec(y_vals);
        let mu = Array1::from_vec(mu_vals);
        let dev = family.unit_deviance(&y, &mu);
        for &d in dev.iter() {
            prop_assert!(d >= -1e-10, "QuasiPoisson unit deviance negative: {}", d);
        }
    }

    #[test]
    fn quasibinomial_deviance_nonneg(
        y_vals in probability_vec(10),
        mu_vals in probability_vec(10)
    ) {
        let family = QuasiBinomialFamily;
        let y = Array1::from_vec(y_vals);
        let mu = Array1::from_vec(mu_vals);
        let dev = family.unit_deviance(&y, &mu);
        for &d in dev.iter() {
            prop_assert!(d >= -1e-10, "QuasiBinomial unit deviance negative: {}", d);
        }
    }
}

// =============================================================================
// Tweedie Deviance Across the p Regimes
// =============================================================================
//
// G1: exercise the full Tweedie p-regime structure {p=0} ∪ [1,2] ∪ (2,3] with
// y/mu vectors INCLUDING exact zeros. The unit deviance must never be a finite
// NEGATIVE value (it is either >= -1e-10 or exactly +inf, the latter only for
// y <= 0 with p >= 2). At a perfect fit (y == mu, finite support) the deviance
// must be ~0. This covers the new `y <= 0 & p >= 2 → +inf` boundary branch.

/// Strategy for a Tweedie power spanning the supported regimes:
/// exactly 0 (Gaussian), the closed interval [1, 2] (Poisson, compound
/// Poisson-Gamma, Gamma), and the open-above interval (2, 3] (positive stable).
fn tweedie_power() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(0.0),
        1.0f64..=2.0,
        (2.0f64..=3.0).prop_map(|p| p.max(2.0 + 1e-6)),
    ]
}

/// y values that include exact zeros (Just(0.0)) plus a positive range.
fn nonneg_with_zeros() -> impl Strategy<Value = f64> {
    prop_oneof![Just(0.0), 0.001f64..100.0]
}

proptest! {
    #[test]
    fn tweedie_deviance_nonneg_or_inf(
        p in tweedie_power(),
        y_vals in prop::collection::vec(nonneg_with_zeros(), 10),
        mu_vals in positive_mu_vec(10),
    ) {
        let family = TweedieFamily::new(p).expect("tweedie_power() yields valid p");
        for (&yi, &mui) in y_vals.iter().zip(mu_vals.iter()) {
            let d = family.unit_deviance_at(yi, mui);
            // Either a clearly non-negative finite value, or exactly +inf.
            let ok = (d.is_finite() && d >= -1e-10) || d == f64::INFINITY;
            prop_assert!(
                ok,
                "Tweedie deviance not (>= -1e-10 or +inf): d={} (p={}, y={}, mu={})",
                d, p, yi, mui
            );
            // The only legitimate +inf is y <= 0 with p >= 2.
            if d == f64::INFINITY {
                prop_assert!(
                    yi <= 0.0 && p >= 2.0,
                    "Unexpected +inf Tweedie deviance at p={}, y={}, mu={}",
                    p, yi, mui
                );
            }
        }
    }

    #[test]
    fn tweedie_deviance_zero_at_perfect_fit(
        p in tweedie_power(),
        // Strictly positive so every regime (including p >= 2) has finite
        // support at this point — perfect fit deviance must vanish.
        mu_vals in positive_mu_vec(10),
    ) {
        let family = TweedieFamily::new(p).expect("tweedie_power() yields valid p");
        for &mui in mu_vals.iter() {
            let d = family.unit_deviance_at(mui, mui);
            prop_assert!(
                d.abs() < 1e-6 * (1.0 + mui.abs()),
                "Tweedie deviance not ~0 at perfect fit: d={} (p={}, mu={})",
                d, p, mui
            );
        }
    }
}

// =============================================================================
// Deviance Zero at Perfect Fit
// =============================================================================

proptest! {
    #[test]
    fn deviance_zero_at_perfect_fit(mu_vals in positive_mu_vec(10)) {
        let families: Vec<Box<dyn Family>> = vec![
            Box::new(GaussianFamily),
            Box::new(PoissonFamily),
            Box::new(GammaFamily),
        ];
        let mu = Array1::from_vec(mu_vals);
        for family in &families {
            let dev = family.unit_deviance(&mu, &mu);
            for &d in dev.iter() {
                prop_assert!(d.abs() < 1e-8,
                    "{} deviance not zero at perfect fit: {}", family.name(), d);
            }
        }
    }
}

// =============================================================================
// Variance Positivity
// =============================================================================

proptest! {
    #[test]
    fn gaussian_variance_is_exactly_one(
        // Span extreme magnitudes and signs: the Gaussian variance function is
        // V(mu) = 1 for ALL mu, so a regression that accidentally made it
        // mu-dependent (e.g. copied the Poisson V(mu) = mu) would fail here.
        mu_vals in prop::collection::vec(-1e12f64..1e12, 10)
    ) {
        let family = GaussianFamily;
        let mu = Array1::from_vec(mu_vals);
        let var = family.variance(&mu);
        for &v in var.iter() {
            prop_assert!((v - 1.0).abs() < 1e-12, "Gaussian variance != 1: {}", v);
        }
    }

    #[test]
    fn gaussian_deviance_is_squared_residual(
        // Deviance must equal (y - mu)^2 exactly for the Gaussian family,
        // independent of the magnitude/sign of y and mu. This would fail if the
        // deviance were ever rewritten using a non-identity transform.
        y_vals in prop::collection::vec(-1e6f64..1e6, 10),
        mu_vals in prop::collection::vec(-1e6f64..1e6, 10)
    ) {
        let family = GaussianFamily;
        let y = Array1::from_vec(y_vals);
        let mu = Array1::from_vec(mu_vals);
        let dev = family.unit_deviance(&y, &mu);
        for ((&yi, &mui), &d) in y.iter().zip(mu.iter()).zip(dev.iter()) {
            let expected = (yi - mui) * (yi - mui);
            let tol = 1e-6 * expected.abs().max(1.0);
            prop_assert!(
                (d - expected).abs() <= tol,
                "Gaussian deviance {} != squared residual {} (y={}, mu={})",
                d, expected, yi, mui
            );
        }
    }

    #[test]
    fn poisson_variance_positive(mu_vals in positive_mu_vec(10)) {
        let family = PoissonFamily;
        let mu = Array1::from_vec(mu_vals);
        let var = family.variance(&mu);
        for &v in var.iter() {
            prop_assert!(v > 0.0, "Poisson variance not positive: {}", v);
        }
    }

    #[test]
    fn gamma_variance_positive(mu_vals in positive_mu_vec(10)) {
        let family = GammaFamily;
        let mu = Array1::from_vec(mu_vals);
        let var = family.variance(&mu);
        for &v in var.iter() {
            prop_assert!(v > 0.0, "Gamma variance not positive: {}", v);
        }
    }

    #[test]
    fn binomial_variance_positive(mu_vals in probability_vec(10)) {
        let family = BinomialFamily;
        let mu = Array1::from_vec(mu_vals);
        let var = family.variance(&mu);
        for &v in var.iter() {
            prop_assert!(v > 0.0, "Binomial variance not positive: {}", v);
        }
    }

    #[test]
    fn tweedie_variance_positive(mu_vals in positive_mu_vec(10)) {
        let family = TweedieFamily::new(1.5).expect("test setup should be valid");
        let mu = Array1::from_vec(mu_vals);
        let var = family.variance(&mu);
        for &v in var.iter() {
            prop_assert!(v > 0.0, "Tweedie variance not positive: {}", v);
        }
    }

    #[test]
    fn negbinomial_variance_positive(
        mu_vals in positive_mu_vec(10),
        theta in 0.1f64..50.0
    ) {
        let family = NegativeBinomialFamily::new(theta).expect("theta > 0 valid");
        let mu = Array1::from_vec(mu_vals);
        let var = family.variance(&mu);
        for &v in var.iter() {
            prop_assert!(v > 0.0, "NegativeBinomial variance not positive: {}", v);
        }
    }

    #[test]
    fn quasipoisson_variance_positive(mu_vals in positive_mu_vec(10)) {
        let family = QuasiPoissonFamily;
        let mu = Array1::from_vec(mu_vals);
        let var = family.variance(&mu);
        for &v in var.iter() {
            prop_assert!(v > 0.0, "QuasiPoisson variance not positive: {}", v);
        }
    }

    #[test]
    fn quasibinomial_variance_positive(mu_vals in probability_vec(10)) {
        let family = QuasiBinomialFamily;
        let mu = Array1::from_vec(mu_vals);
        let var = family.variance(&mu);
        for &v in var.iter() {
            prop_assert!(v > 0.0, "QuasiBinomial variance not positive: {}", v);
        }
    }
}

// =============================================================================
// Link Derivative Finite
// =============================================================================

proptest! {
    #[test]
    fn log_derivative_finite(mu_vals in positive_mu_vec(10)) {
        let link = LogLink;
        let mu = Array1::from_vec(mu_vals);
        let deriv = link.derivative(&mu);
        for &d in deriv.iter() {
            prop_assert!(d.is_finite(), "Log derivative not finite: {}", d);
            prop_assert!(d != 0.0, "Log derivative is zero");
        }
    }

    #[test]
    fn logit_derivative_finite(mu_vals in probability_vec(10)) {
        let link = LogitLink;
        let mu = Array1::from_vec(mu_vals);
        let deriv = link.derivative(&mu);
        for &d in deriv.iter() {
            prop_assert!(d.is_finite(), "Logit derivative not finite: {}", d);
            prop_assert!(d != 0.0, "Logit derivative is zero");
        }
    }
}

// =============================================================================
// IRLS Deviance-Monotone Sweep (RS-ACT-007)
// =============================================================================
//
// Across multiple deterministic fixtures and families, the deviance at the
// accepted terminal iterate may not exceed a smaller-budget fit by more than the
// per-step acceptance slack as the iteration budget grows. The solver accepts a
// trial step when `deviance_new <= deviance_old * IRLS_ACCEPT_REL_SLACK`
// (irls.rs), so a larger-budget fit can take one more accepted step that nudges
// the deviance up by up to `(IRLS_ACCEPT_REL_SLACK - 1)` relative. The earlier
// "cannot exceed" claim (with a ~1e-10 relative slack) was therefore too tight:
// we relax the assertion to the solver's real acceptance contract and exercise
// it by stepping max_iterations up across a small seed × family sweep.

/// Minimal linear congruential generator for deterministic fixture data.
/// We avoid pulling in `rand` as a dev-dep; the sequence is reproducible.
fn lcg_next(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

/// Standard-normal sample via Box-Muller on two LCG draws.
fn standard_normal(state: &mut u64) -> f64 {
    let u1 = lcg_next(state).max(1e-12);
    let u2 = lcg_next(state);
    (-2.0_f64 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn make_design(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut xv = Vec::with_capacity(n * 2);
    let mut x_col = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = standard_normal(&mut state);
        xv.push(1.0);
        xv.push(xi);
        x_col.push(xi);
    }
    let x = Array2::from_shape_vec((n, 2), xv).expect("design shape ok");
    (x, Array1::from_vec(x_col))
}

fn assert_monotone_in_budget(
    y: &Array1<f64>,
    x: &Array2<f64>,
    family: &dyn Family,
    link: &dyn Link,
    budgets: &[usize],
    seed: u64,
    label: &str,
) {
    let mut prev_deviance: Option<f64> = None;
    for &budget in budgets {
        let config = FitConfig {
            max_iterations: budget,
            ..FitConfig::default()
        };
        let result = match fit_glm_unified(y, x.view(), family, link, &config, None, None, None) {
            Ok(r) => r,
            Err(e) => panic!("[{label} seed={seed} budget={budget}] fit_glm_unified failed: {e}"),
        };
        assert!(
            result.deviance.is_finite(),
            "[{label} seed={seed} budget={budget}] deviance is non-finite: {}",
            result.deviance
        );
        if let Some(prev) = prev_deviance {
            // Slack matches the solver's real per-step acceptance contract: a
            // larger-budget fit may take one more accepted step that worsens the
            // deviance by up to `(IRLS_ACCEPT_REL_SLACK - 1)` relative, plus a
            // small absolute floor for float noise near zero deviance.
            let slack = ZERO_TOL + prev.abs() * (IRLS_ACCEPT_REL_SLACK - 1.0);
            assert!(
                result.deviance <= prev + slack,
                "[{label} seed={seed}] deviance increased beyond acceptance slack: \
                 prev={prev} (smaller budget) vs current={} (budget={budget}, slack={slack})",
                result.deviance
            );
        }
        prev_deviance = Some(result.deviance);
    }
}

#[test]
fn irls_deviance_non_increasing_in_budget_across_families() {
    let budgets = [1usize, 2, 3, 5, 10, 25];
    let seeds = [42u64, 7, 2026];
    let n = 80;

    for &seed in &seeds {
        let (x, x_col) = make_design(n, seed);

        // Gaussian / Identity
        let mut state = seed.wrapping_add(1);
        let y_gauss: Array1<f64> = (0..n)
            .map(|i| 1.5 + 0.7 * x_col[i] + 0.5 * standard_normal(&mut state))
            .collect();
        assert_monotone_in_budget(
            &y_gauss,
            &x,
            &GaussianFamily,
            &IdentityLink,
            &budgets,
            seed,
            "gaussian/identity",
        );

        // Poisson / Log (mu kept moderate so synthetic y is sensible)
        let mut state = seed.wrapping_add(2);
        let y_pois: Array1<f64> = (0..n)
            .map(|i| {
                let eta = (0.3 + 0.4 * x_col[i]).clamp(-5.0, 5.0);
                let mu = eta.exp();
                (mu + 0.5 * standard_normal(&mut state)).max(0.0).round()
            })
            .collect();
        assert_monotone_in_budget(
            &y_pois,
            &x,
            &PoissonFamily,
            &LogLink,
            &budgets,
            seed,
            "poisson/log",
        );

        // Gamma / Log (strictly positive response)
        let mut state = seed.wrapping_add(3);
        let y_gamma: Array1<f64> = (0..n)
            .map(|i| {
                let eta = (0.5 + 0.3 * x_col[i]).clamp(-3.0, 3.0);
                let mu = eta.exp();
                let noise = (0.3 * standard_normal(&mut state)).exp();
                (mu * noise).max(1e-3)
            })
            .collect();
        assert_monotone_in_budget(
            &y_gamma,
            &x,
            &GammaFamily,
            &LogLink,
            &budgets,
            seed,
            "gamma/log",
        );

        // Binomial / Logit
        let mut state = seed.wrapping_add(4);
        let y_bin: Array1<f64> = (0..n)
            .map(|i| {
                let eta = (-0.2 + 0.6 * x_col[i]).clamp(-4.0, 4.0);
                let p = 1.0 / (1.0 + (-eta).exp());
                let u = lcg_next(&mut state);
                if u < p {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        assert_monotone_in_budget(
            &y_bin,
            &x,
            &BinomialFamily,
            &LogitLink,
            &budgets,
            seed,
            "binomial/logit",
        );

        // Tweedie p=1.5 / Log: reuse Gamma-like positive response.
        let tweedie = TweedieFamily::new(1.5).expect("p=1.5 valid");
        assert_monotone_in_budget(
            &y_gamma,
            &x,
            &tweedie,
            &LogLink,
            &budgets,
            seed,
            "tweedie p=1.5/log",
        );
    }
}
