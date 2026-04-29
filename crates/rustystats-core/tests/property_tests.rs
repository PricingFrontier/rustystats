// =============================================================================
// Property-Based Tests for RustyStats Core
// =============================================================================
//
// These tests use proptest to verify fundamental mathematical invariants
// that should hold for ALL valid inputs, not just hand-picked examples.
// =============================================================================

use ndarray::Array1;
use proptest::prelude::*;

use rustystats_core::families::{
    BinomialFamily, Family, GammaFamily, GaussianFamily, PoissonFamily, TweedieFamily,
};
use rustystats_core::links::{IdentityLink, Link, LogLink, LogitLink};

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
    fn gaussian_variance_positive(mu_vals in positive_mu_vec(10)) {
        let family = GaussianFamily;
        let mu = Array1::from_vec(mu_vals);
        let var = family.variance(&mu);
        for &v in var.iter() {
            prop_assert!(v > 0.0, "Gaussian variance not positive: {}", v);
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
