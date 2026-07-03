#![no_main]

use libfuzzer_sys::fuzz_target;
use ndarray::array;
use rustystats_core::links::{IdentityLink, Link, LogLink, LogitLink};

fn byte(data: &[u8], index: usize) -> u8 {
    data.get(index).copied().unwrap_or(0)
}

fn scaled(data: &[u8], index: usize, min: f64, max: f64) -> f64 {
    let raw = byte(data, index) as f64 / 255.0;
    min + raw * (max - min)
}

fn assert_roundtrip(link: &dyn Link, mu: f64, tolerance: f64) {
    let mu_array = array![mu];
    let eta = link.link(&mu_array);
    let recovered = link.inverse(&eta);
    let derivative = link.derivative(&mu_array);

    assert!(eta.iter().all(|value| value.is_finite()));
    assert!(recovered.iter().all(|value| value.is_finite()));
    assert!(derivative.iter().all(|value| value.is_finite()));
    assert!((recovered[0] - mu).abs() <= tolerance * (1.0 + mu.abs()));
}

fuzz_target!(|data: &[u8]| {
    match byte(data, 0) % 3 {
        0 => {
            let link = IdentityLink;
            let mu = scaled(data, 1, -1.0e12, 1.0e12);
            assert_roundtrip(&link, mu, 1e-12);
        }
        1 => {
            let link = LogLink;
            let mu = 10.0_f64.powf(scaled(data, 1, -12.0, 12.0));
            assert_roundtrip(&link, mu, 1e-10);
        }
        _ => {
            let link = LogitLink;
            let mu = scaled(data, 1, 1e-12, 1.0 - 1e-12);
            assert_roundtrip(&link, mu, 1e-10);
        }
    }
});
