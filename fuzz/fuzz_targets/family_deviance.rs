#![no_main]

use libfuzzer_sys::fuzz_target;
use ndarray::{array, Array1};
use rustystats_core::families::{
    BinomialFamily, Family, GammaFamily, GaussianFamily, NegativeBinomialFamily, PoissonFamily,
    TweedieFamily,
};

fn byte(data: &[u8], index: usize) -> u8 {
    data.get(index).copied().unwrap_or(0)
}

fn scaled(data: &[u8], index: usize, min: f64, max: f64) -> f64 {
    let raw = byte(data, index) as f64 / 255.0;
    min + raw * (max - min)
}

fn positive(data: &[u8], index: usize) -> f64 {
    10.0_f64.powf(scaled(data, index, -10.0, 6.0))
}

fn assert_family_contract(family: &dyn Family, y: Array1<f64>, mu: Array1<f64>) {
    let deviance = family.unit_deviance(&y, &mu);
    let variance = family.variance(&mu).into_owned();
    let total = family.deviance(&y, &mu, None);
    let clamped = family.clamp_mu(&mu);
    let initialized = family.initialize_mu(&y);

    assert_eq!(deviance.len(), y.len());
    assert_eq!(variance.len(), mu.len());
    assert_eq!(clamped.len(), mu.len());
    assert_eq!(initialized.len(), y.len());
    assert!(total.is_finite());
    assert!(total >= -1e-8);
    assert!(deviance.iter().all(|value| value.is_finite()));
    assert!(deviance.iter().all(|value| *value >= -1e-8));
    assert!(variance.iter().all(|value| value.is_finite()));
    assert!(variance.iter().all(|value| *value > 0.0));

    for (yi, mui, unit) in y
        .iter()
        .zip(mu.iter())
        .zip(deviance.iter())
        .map(|((y, m), d)| (y, m, d))
    {
        let streaming = family.unit_deviance_at(*yi, *mui);
        assert!(streaming.is_finite());
        assert!((streaming - *unit).abs() <= 1e-8 * (1.0 + unit.abs()));
    }
}

fuzz_target!(|data: &[u8]| {
    let family_choice = byte(data, 0) % 6;
    match family_choice {
        0 => {
            let family = GaussianFamily;
            let y = array![
                scaled(data, 1, -1.0e6, 1.0e6),
                scaled(data, 2, -1.0e6, 1.0e6)
            ];
            let mu = array![
                scaled(data, 3, -1.0e6, 1.0e6),
                scaled(data, 4, -1.0e6, 1.0e6)
            ];
            assert_family_contract(&family, y, mu);
        }
        1 => {
            let family = PoissonFamily;
            let y = array![f64::from(byte(data, 1) % 32), f64::from(byte(data, 2) % 64)];
            let mu = array![positive(data, 3), positive(data, 4)];
            assert_family_contract(&family, y, mu);
        }
        2 => {
            let family = BinomialFamily;
            let y = array![scaled(data, 1, 0.0, 1.0), scaled(data, 2, 0.0, 1.0)];
            let mu = array![
                scaled(data, 3, 1e-12, 1.0 - 1e-12),
                scaled(data, 4, 1e-12, 1.0 - 1e-12)
            ];
            assert_family_contract(&family, y, mu);
        }
        3 => {
            let family = GammaFamily;
            let y = array![positive(data, 1), positive(data, 2)];
            let mu = array![positive(data, 3), positive(data, 4)];
            assert_family_contract(&family, y, mu);
        }
        4 => {
            let powers = [0.0, 1.0, 1.2, 1.5, 1.9, 2.0, 3.0];
            let power = powers[usize::from(byte(data, 5)) % powers.len()];
            let Ok(family) = TweedieFamily::new(power) else {
                return;
            };
            let y = array![
                if power >= 2.0 {
                    positive(data, 1)
                } else {
                    scaled(data, 1, 0.0, 1000.0)
                },
                if power >= 2.0 {
                    positive(data, 2)
                } else {
                    scaled(data, 2, 0.0, 1000.0)
                },
            ];
            let mu = array![positive(data, 3), positive(data, 4)];
            assert_family_contract(&family, y, mu);
        }
        _ => {
            let theta = positive(data, 5).clamp(1e-6, 1e6);
            let Ok(family) = NegativeBinomialFamily::new(theta) else {
                return;
            };
            let y = array![f64::from(byte(data, 1) % 32), f64::from(byte(data, 2) % 64)];
            let mu = array![positive(data, 3), positive(data, 4)];
            assert_family_contract(&family, y, mu);
        }
    }
});
