// =============================================================================
// Family Parameter Parsing
// =============================================================================
//
// Extracts the embedded parameters out of family-name strings like
//   "negativebinomial(theta=1.38)"
//   "tweedie(p=1.5)"
//
// The parser is permissive: any family name that doesn't carry an embedded
// parameter falls back to the supplied defaults. This is the canonical
// implementation used by both the diagnostics computations and any other
// caller that has only the family-name string and wants the parameter pair.
// =============================================================================

/// Tweedie variance power and Negative Binomial dispersion as a pair.
///
/// Returned by [`parse_family_params`]. Callers ignore whichever field is
/// irrelevant for their family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FamilyParams {
    /// Tweedie variance power `p` (used for `tweedie` families only).
    pub var_power: f64,
    /// Negative Binomial dispersion `θ` (used for NB families only).
    pub theta: f64,
}

/// Parse `theta` and `var_power` out of a family-name string.
///
/// Recognised forms:
/// - `"negativebinomial(theta=X)"` / `"negbinomial(theta=X)"` (case-insensitive
///   on the family prefix; X parses with `f64::from_str`).
/// - `"tweedie(p=X)"` (case-insensitive on the prefix).
///
/// Any other family string returns the supplied defaults verbatim. The parser
/// only updates the field that matches the family prefix — i.e. a pure
/// "tweedie(p=1.5)" leaves `theta` at the default, and vice versa.
///
/// # Errors
/// Returns `Err(String)` with a descriptive message if a `theta=` or `p=` is
/// present in the relevant family but cannot be parsed as `f64`.
///
/// # Arguments
/// * `family` - Family name (possibly with embedded parameters).
/// * `default_var_power` - Fallback Tweedie variance power.
/// * `default_theta` - Fallback Negative Binomial dispersion.
pub fn parse_family_params(
    family: &str,
    default_var_power: f64,
    default_theta: f64,
) -> Result<FamilyParams, String> {
    let lower = family.to_lowercase();

    let theta = if lower.starts_with("negativebinomial") || lower.starts_with("negbinomial") {
        if let Some(start) = lower.find("theta=") {
            let rest = &lower[start + "theta=".len()..];
            let end = rest.find(')').unwrap_or(rest.len());
            let theta_str = &rest[..end];
            theta_str.parse::<f64>().map_err(|_| {
                format!(
                    "Failed to parse theta value '{}' in family '{}'. Expected a numeric value.",
                    theta_str, family
                )
            })?
        } else {
            default_theta
        }
    } else {
        default_theta
    };

    let var_power = if lower.starts_with("tweedie") {
        if let Some(start) = lower.find("p=") {
            let rest = &lower[start + "p=".len()..];
            let end = rest.find(')').unwrap_or(rest.len());
            let p_str = &rest[..end];
            p_str.parse::<f64>().map_err(|_| {
                format!(
                    "Failed to parse var_power value '{}' in family '{}'. Expected a numeric value.",
                    p_str, family
                )
            })?
        } else {
            default_var_power
        }
    } else {
        default_var_power
    };

    Ok(FamilyParams { var_power, theta })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_parse_plain_families_returns_defaults() {
        let p = parse_family_params("poisson", 1.5, 1.0).unwrap();
        assert_abs_diff_eq!(p.var_power, 1.5, epsilon = 1e-12);
        assert_abs_diff_eq!(p.theta, 1.0, epsilon = 1e-12);

        let p = parse_family_params("Gaussian", 2.0, 3.0).unwrap();
        assert_abs_diff_eq!(p.var_power, 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(p.theta, 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_parse_negbin_with_theta() {
        let p = parse_family_params("negativebinomial(theta=2.5)", 1.5, 1.0).unwrap();
        assert_abs_diff_eq!(p.theta, 2.5, epsilon = 1e-12);
        // var_power stays at default for negbin.
        assert_abs_diff_eq!(p.var_power, 1.5, epsilon = 1e-12);
    }

    #[test]
    fn test_parse_negbinomial_alias() {
        let p = parse_family_params("negbinomial(theta=0.42)", 1.5, 1.0).unwrap();
        assert_abs_diff_eq!(p.theta, 0.42, epsilon = 1e-12);
    }

    #[test]
    fn test_parse_negbin_without_theta_uses_default() {
        let p = parse_family_params("negativebinomial", 1.5, 7.0).unwrap();
        assert_abs_diff_eq!(p.theta, 7.0, epsilon = 1e-12);
    }

    #[test]
    fn test_parse_negbin_case_insensitive() {
        let p = parse_family_params("NegativeBinomial(THETA=4.0)", 1.5, 1.0).unwrap();
        assert_abs_diff_eq!(p.theta, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn test_parse_tweedie_with_p() {
        let p = parse_family_params("tweedie(p=1.7)", 1.5, 1.0).unwrap();
        assert_abs_diff_eq!(p.var_power, 1.7, epsilon = 1e-12);
        assert_abs_diff_eq!(p.theta, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_parse_tweedie_without_p_uses_default() {
        let p = parse_family_params("tweedie", 1.42, 1.0).unwrap();
        assert_abs_diff_eq!(p.var_power, 1.42, epsilon = 1e-12);
    }

    #[test]
    fn test_parse_invalid_theta_errors() {
        let err = parse_family_params("negativebinomial(theta=oops)", 1.5, 1.0).unwrap_err();
        assert!(err.contains("Failed to parse theta value"));
        assert!(err.contains("oops"));
    }

    #[test]
    fn test_parse_invalid_var_power_errors() {
        let err = parse_family_params("tweedie(p=zoom)", 1.5, 1.0).unwrap_err();
        assert!(err.contains("Failed to parse var_power value"));
        assert!(err.contains("zoom"));
    }

    #[test]
    fn test_parse_handles_missing_close_paren() {
        // Be permissive: "theta=1.5" without trailing ')' should still parse.
        let p = parse_family_params("negativebinomial(theta=1.5", 1.5, 1.0).unwrap();
        assert_abs_diff_eq!(p.theta, 1.5, epsilon = 1e-12);
    }
}
