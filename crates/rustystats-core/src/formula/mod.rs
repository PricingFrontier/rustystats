//! Formula parsing for R-style model specifications.
//!
//! This module parses formulas like "y ~ x1*x2 + C(cat) + bs(age, df=5) + TE(brand)"
//! into structured components for design matrix construction.

use std::collections::HashSet;

/// Parsed spline term specification
#[derive(Debug, Clone, PartialEq)]
pub struct SplineTerm {
    pub var_name: String,
    pub spline_type: String,  // "bs" or "ns"
    pub df: usize,
    pub degree: usize,
}

/// Parsed target encoding term specification
#[derive(Debug, Clone, PartialEq)]
pub struct TargetEncodingTermSpec {
    pub var_name: String,
    pub prior_weight: f64,
    pub n_permutations: usize,
}

/// Parsed identity term specification for I() expressions
/// I() protects expressions from formula interpretation, e.g., I(x**2) for polynomial
#[derive(Debug, Clone, PartialEq)]
pub struct IdentityTermSpec {
    pub expression: String,  // The raw expression inside I(), e.g., "x**2" or "x + y"
}

/// Parsed categorical term specification for C() with optional level selection
/// C(var) - all levels (standard treatment coding)
/// C(var, level='Paris') - single level indicator (0/1 for that level)
/// C(var, levels=['Paris', 'Lyon']) - multiple specific levels
#[derive(Debug, Clone, PartialEq)]
pub struct CategoricalTermSpec {
    pub var_name: String,
    pub levels: Option<Vec<String>>,  // None = all levels, Some = specific levels only
}

/// Parsed interaction term
#[derive(Debug, Clone, PartialEq)]
pub struct InteractionTerm {
    pub factors: Vec<String>,
    pub categorical_flags: Vec<bool>,
}

/// Result of parsing a formula
#[derive(Debug, Clone)]
pub struct ParsedFormula {
    pub response: String,
    pub main_effects: Vec<String>,
    pub interactions: Vec<InteractionTerm>,
    pub categorical_vars: HashSet<String>,
    pub categorical_terms: Vec<CategoricalTermSpec>,  // C() terms with optional level selection
    pub spline_terms: Vec<SplineTerm>,
    pub target_encoding_terms: Vec<TargetEncodingTermSpec>,
    pub identity_terms: Vec<IdentityTermSpec>,
    pub has_intercept: bool,
}

/// Find the position of the closing parenthesis that matches the opening one at `start`.
fn find_matching_paren(s: &str, start: usize) -> Option<usize> {
    let bytes = s.as_bytes();
    if start >= bytes.len() || bytes[start] != b'(' {
        return None;
    }
    let mut depth = 0;
    for (i, &b) in bytes[start..].iter().enumerate() {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(start + i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Split a string by a delimiter, respecting parentheses.
/// E.g., "TE(Area):bs(DrivAge, df=4)" splits into ["TE(Area)", "bs(DrivAge, df=4)"]
fn split_respecting_parens(s: &str, delim: char) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut depth = 0;
    
    for c in s.chars() {
        match c {
            '(' => {
                depth += 1;
                current.push(c);
            }
            ')' => {
                depth -= 1;
                current.push(c);
            }
            c if c == delim && depth == 0 => {
                if !current.is_empty() {
                    parts.push(current.trim().to_string());
                    current = String::new();
                }
            }
            _ => current.push(c),
        }
    }
    
    if !current.is_empty() {
        parts.push(current.trim().to_string());
    }
    
    parts
}

/// Parse a target encoding term like "TE(brand)" or "TE(brand, prior_weight=2.0)"
fn parse_target_encoding_term(term: &str) -> Option<TargetEncodingTermSpec> {
    let term = term.trim();
    
    // Check if starts with TE(
    if !term.starts_with("TE(") {
        return None;
    }
    
    // Find matching parenthesis (not rfind - need to match balanced parens)
    let start = term.find('(')?;
    let end = find_matching_paren(term, start)?;
    if end <= start {
        return None;
    }
    
    let inner = &term[start + 1..end];
    let parts: Vec<&str> = inner.split(',').collect();
    
    if parts.is_empty() {
        return None;
    }
    
    let var_name = parts[0].trim().to_string();
    let mut prior_weight = 1.0f64;
    let mut n_permutations = 4usize;
    
    // Parse remaining arguments
    for part in parts.iter().skip(1) {
        let part = part.trim();
        if let Some(eq_pos) = part.find('=') {
            let key = part[..eq_pos].trim();
            let value = part[eq_pos + 1..].trim();
            match key {
                "prior_weight" | "pw" => {
                    prior_weight = value.parse().unwrap_or_else(|_| {
                        panic!("Invalid prior_weight value '{}' in TE() - expected a number", value)
                    });
                }
                "n_permutations" | "nperm" => {
                    n_permutations = value.parse().unwrap_or_else(|_| {
                        panic!("Invalid n_permutations value '{}' in TE() - expected an integer", value)
                    });
                }
                other => {
                    panic!("Unknown argument '{}' in TE(). Supported: prior_weight (pw), n_permutations (nperm)", other)
                }
            }
        }
    }
    
    Some(TargetEncodingTermSpec {
        var_name,
        prior_weight,
        n_permutations,
    })
}

/// Parse an identity term like "I(x**2)" or "I(x + y)"
/// The I() function protects expressions from formula interpretation
fn parse_identity_term(term: &str) -> Option<IdentityTermSpec> {
    let term = term.trim();
    
    // Check if starts with I(
    if !term.starts_with("I(") {
        return None;
    }
    
    // Find matching parenthesis
    let start = term.find('(')?;
    let end = find_matching_paren(term, start)?;
    if end <= start {
        return None;
    }
    
    let expression = term[start + 1..end].trim().to_string();
    if expression.is_empty() {
        return None;
    }
    
    Some(IdentityTermSpec { expression })
}

/// Parse a categorical term like "C(region)" or "C(region, level='Paris')"
/// Returns None if not a C() term, Some with the parsed spec if it is
fn parse_categorical_term(term: &str) -> Option<CategoricalTermSpec> {
    let term = term.trim();
    
    // Check if starts with C(
    if !term.starts_with("C(") {
        return None;
    }
    
    // Find matching parenthesis
    let start = term.find('(')?;
    let end = find_matching_paren(term, start)?;
    if end <= start {
        return None;
    }
    
    let inner = &term[start + 1..end];
    
    // Check if there are any commas (indicating options)
    if !inner.contains(',') {
        // Simple case: C(var)
        return Some(CategoricalTermSpec {
            var_name: inner.trim().to_string(),
            levels: None,
        });
    }
    
    // Parse with options: C(var, level='value') or C(var, levels=['a', 'b'])
    let parts: Vec<&str> = inner.splitn(2, ',').collect();
    if parts.is_empty() {
        return None;
    }
    
    let var_name = parts[0].trim().to_string();
    let mut levels: Option<Vec<String>> = None;
    
    if parts.len() > 1 {
        let options = parts[1].trim();
        
        // Parse level='value' (single level)
        if options.starts_with("level=") || options.starts_with("level =") {
            let value_start = options.find('=')? + 1;
            let value = options[value_start..].trim();
            // Remove quotes
            let level = value.trim_matches(|c| c == '\'' || c == '"').to_string();
            levels = Some(vec![level]);
        }
        // Parse levels=['a', 'b'] (multiple levels)
        else if options.starts_with("levels=") || options.starts_with("levels =") {
            let value_start = options.find('=')? + 1;
            let value = options[value_start..].trim();
            // Parse list format: ['a', 'b'] or ["a", "b"]
            if value.starts_with('[') && value.ends_with(']') {
                let list_inner = &value[1..value.len()-1];
                let level_strs: Vec<String> = list_inner
                    .split(',')
                    .map(|s| s.trim().trim_matches(|c| c == '\'' || c == '"').to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                if !level_strs.is_empty() {
                    levels = Some(level_strs);
                }
            }
        }
    }
    
    Some(CategoricalTermSpec {
        var_name,
        levels,
    })
}

/// Parse a spline term like "bs(age, df=5)" or "ns(income, df=4)"
fn parse_spline_term(term: &str) -> Option<SplineTerm> {
    let term = term.trim();
    
    // Check if starts with bs( or ns(
    let spline_type = if term.starts_with("bs(") {
        "bs"
    } else if term.starts_with("ns(") {
        "ns"
    } else {
        return None;
    };
    
    // Find matching parenthesis
    let start = term.find('(')?;
    let end = find_matching_paren(term, start)?;
    if end <= start {
        return None;
    }
    
    let inner = &term[start + 1..end];
    let parts: Vec<&str> = inner.split(',').collect();
    
    if parts.is_empty() {
        return None;
    }
    
    let var_name = parts[0].trim().to_string();
    let mut df = 5usize;
    let mut degree = 3usize;
    
    // Parse remaining arguments
    for part in parts.iter().skip(1) {
        let part = part.trim();
        if let Some(eq_pos) = part.find('=') {
            let key = part[..eq_pos].trim();
            let value = part[eq_pos + 1..].trim();
            match key {
                "df" => {
                    df = value.parse().unwrap_or_else(|_| {
                        panic!("Invalid df value '{}' in {}() - expected an integer", value, spline_type)
                    });
                }
                "degree" => {
                    degree = value.parse().unwrap_or_else(|_| {
                        panic!("Invalid degree value '{}' in {}() - expected an integer", value, spline_type)
                    });
                }
                other => {
                    panic!("Unknown argument '{}' in {}(). Supported: df, degree", other, spline_type)
                }
            }
        } else if let Ok(v) = part.parse::<usize>() {
            // Positional argument assumed to be df
            df = v;
        } else {
            panic!("Invalid argument '{}' in {}() - expected 'key=value' or a number for df", part, spline_type)
        }
    }
    
    Some(SplineTerm {
        var_name,
        spline_type: spline_type.to_string(),
        df,
        degree,
    })
}

/// Split formula RHS by '+', respecting parentheses
fn split_terms(rhs: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut current = String::new();
    let mut depth = 0;
    
    for c in rhs.chars() {
        match c {
            '(' => {
                depth += 1;
                current.push(c);
            }
            ')' => {
                depth -= 1;
                current.push(c);
            }
            '+' if depth == 0 => {
                let term = current.trim().to_string();
                if !term.is_empty() {
                    terms.push(term);
                }
                current = String::new();
            }
            _ => {
                current.push(c);
            }
        }
    }
    
    let term = current.trim().to_string();
    if !term.is_empty() {
        terms.push(term);
    }
    
    terms
}

/// Clean variable name: "C(var)" -> "var"
fn clean_var_name(term: &str) -> String {
    let term = term.trim();
    if term.starts_with("C(") && term.ends_with(')') {
        term[2..term.len() - 1].to_string()
    } else {
        term.to_string()
    }
}

/// Check if a term looks like an unsupported function call.
/// Returns Some(function_name) if unsupported, None if it's a valid term.
fn check_unsupported_function(term: &str) -> Option<String> {
    let term = term.trim();
    
    // Check if it looks like a function call: name(...)
    if let Some(paren_pos) = term.find('(') {
        if term.ends_with(')') {
            let func_name = term[..paren_pos].trim();
            
            // Skip supported functions
            let supported = ["C", "bs", "ns", "TE", "I"];
            if !supported.contains(&func_name) && !func_name.is_empty() {
                // This looks like an unsupported function call
                return Some(func_name.to_string());
            }
        }
    }
    None
}

/// Check if term is categorical
fn is_categorical(term: &str, categorical_vars: &HashSet<String>) -> bool {
    let term = term.trim();
    if term.starts_with("C(") {
        return true;
    }
    categorical_vars.contains(&clean_var_name(term))
}

/// Parse a formula string into structured components.
///
/// Handles:
/// - Main effects: x1, x2, C(cat)
/// - Two-way interactions: x1:x2, x1*x2, C(cat):x
/// - Higher-order: x1:x2:x3
/// - Intercept removal: 0 + ... or -1
/// - Spline terms: bs(x, df=5), ns(x, df=4)
///
/// # Arguments
/// * `formula` - R-style formula like "y ~ x1*x2 + C(cat) + bs(age, df=5)"
///
/// # Returns
/// Parsed formula structure with all terms identified
pub fn parse_formula(formula: &str) -> Result<ParsedFormula, String> {
    // Split into response and predictors
    let parts: Vec<&str> = formula.split('~').collect();
    if parts.len() != 2 {
        return Err(format!("Formula must contain exactly one '~': {}", formula));
    }
    
    let response = parts[0].trim().to_string();
    let mut rhs = parts[1].trim().to_string();
    
    // Check for intercept removal
    let mut has_intercept = true;
    
    // Handle "0 +" or "0+"
    if rhs.starts_with("0 +") || rhs.starts_with("0+") {
        has_intercept = false;
        rhs = rhs[if rhs.starts_with("0 +") { 3 } else { 2 }..].trim().to_string();
    }
    
    // Handle "- 1" or "-1" at end
    if rhs.ends_with("- 1") || rhs.ends_with("-1") {
        has_intercept = false;
        let len = rhs.len();
        rhs = rhs[..len - if rhs.ends_with("- 1") { 3 } else { 2 }].trim().to_string();
        // Remove trailing +
        if rhs.ends_with('+') {
            rhs = rhs[..rhs.len() - 1].trim().to_string();
        }
    }
    
    // Find all C(...) categorical markers
    let mut categorical_vars = HashSet::new();
    let mut pos = 0;
    while let Some(start) = rhs[pos..].find("C(") {
        let abs_start = pos + start + 2;
        if let Some(end) = rhs[abs_start..].find(')') {
            let var = rhs[abs_start..abs_start + end].trim().to_string();
            categorical_vars.insert(var);
            pos = abs_start + end + 1;
        } else {
            break;
        }
    }
    
    // Split into terms
    let terms = split_terms(&rhs);
    
    let mut main_effects = Vec::new();
    let mut interactions = Vec::new();
    let mut spline_terms = Vec::new();
    let mut target_encoding_terms = Vec::new();
    let mut identity_terms = Vec::new();
    let mut categorical_terms = Vec::new();
    
    for term in terms {
        // Check for unsupported function-like terms FIRST before any other processing
        // This catches things like I(), poly(), log() that look like function calls
        if let Some(func_name) = check_unsupported_function(&term) {
            return Err(format!(
                "Unsupported function '{}()' in formula term '{}'. \
                Supported functions are: C() for categorical, bs() for B-splines, \
                ns() for natural splines, TE() for target encoding.",
                func_name, term
            ));
        }
        
        // Check for identity term (I() for polynomial/transformation expressions)
        if let Some(identity) = parse_identity_term(&term) {
            identity_terms.push(identity);
            continue;
        }
        
        // Check for target encoding term
        if let Some(te_term) = parse_target_encoding_term(&term) {
            target_encoding_terms.push(te_term);
            continue;
        }
        
        // Check for spline term
        if let Some(spline) = parse_spline_term(&term) {
            spline_terms.push(spline);
            continue;
        }
        
        // Check for categorical term with level selection: C(var, level='value')
        if let Some(cat_term) = parse_categorical_term(&term) {
            if cat_term.levels.is_some() {
                // Level-specific categorical - add to categorical_terms, not main_effects
                categorical_terms.push(cat_term);
                continue;
            }
            // Otherwise fall through to normal C(var) handling below
        }
        
        if term.contains('*') {
            // Full interaction: a*b = a + b + a:b
            // Use split_respecting_parens for cases like TE(Area)*bs(Age, df=4)
            let factor_strs = split_respecting_parens(&term, '*');
            
            if factor_strs.len() > 1 {
                // Add main effects
                for f in &factor_strs {
                    let clean = clean_var_name(f);
                    if !main_effects.contains(&clean) {
                        main_effects.push(clean);
                    }
                }
                
                // Add interaction
                let factors: Vec<String> = factor_strs.iter().map(|f| clean_var_name(f)).collect();
                let categorical_flags: Vec<bool> = factor_strs
                    .iter()
                    .map(|f| is_categorical(f, &categorical_vars))
                    .collect();
                
                interactions.push(InteractionTerm {
                    factors,
                    categorical_flags,
                });
            } else {
                // Single factor - treat as main effect
                let clean = clean_var_name(&term);
                if !clean.is_empty() && !main_effects.contains(&clean) {
                    main_effects.push(clean);
                }
            }
        } else if term.contains(':') {
            // Pure interaction: a:b (no main effects)
            // Use split_respecting_parens to handle TE(Area):bs(DrivAge, df=4) correctly
            let factor_strs = split_respecting_parens(&term, ':');
            if factor_strs.len() > 1 {
                let factors: Vec<String> = factor_strs.iter().map(|f| clean_var_name(f)).collect();
                let categorical_flags: Vec<bool> = factor_strs
                    .iter()
                    .map(|f| is_categorical(f, &categorical_vars))
                    .collect();
                
                interactions.push(InteractionTerm {
                    factors,
                    categorical_flags,
                });
            } else {
                // Single factor after split - treat as main effect
                let clean = clean_var_name(&term);
                if !clean.is_empty() && !main_effects.contains(&clean) {
                    main_effects.push(clean);
                }
            }
        } else {
            // Main effect
            let clean = clean_var_name(&term);
            if !clean.is_empty() && !main_effects.contains(&clean) {
                main_effects.push(clean);
            }
        }
    }
    
    Ok(ParsedFormula {
        response,
        main_effects,
        interactions,
        categorical_vars,
        categorical_terms,
        spline_terms,
        target_encoding_terms,
        identity_terms,
        has_intercept,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_formula() {
        let parsed = parse_formula("y ~ x1 + x2").unwrap();
        assert_eq!(parsed.response, "y");
        assert_eq!(parsed.main_effects, vec!["x1", "x2"]);
        assert!(parsed.interactions.is_empty());
        assert!(parsed.has_intercept);
    }

    #[test]
    fn test_parse_categorical() {
        let parsed = parse_formula("y ~ x1 + C(region)").unwrap();
        assert_eq!(parsed.main_effects, vec!["x1", "region"]);
        assert!(parsed.categorical_vars.contains("region"));
    }

    #[test]
    fn test_parse_interaction() {
        let parsed = parse_formula("y ~ x1*x2").unwrap();
        assert_eq!(parsed.main_effects, vec!["x1", "x2"]);
        assert_eq!(parsed.interactions.len(), 1);
        assert_eq!(parsed.interactions[0].factors, vec!["x1", "x2"]);
    }

    #[test]
    fn test_parse_spline() {
        let parsed = parse_formula("y ~ bs(age, df=5) + ns(income, df=4)").unwrap();
        assert_eq!(parsed.spline_terms.len(), 2);
        assert_eq!(parsed.spline_terms[0].var_name, "age");
        assert_eq!(parsed.spline_terms[0].spline_type, "bs");
        assert_eq!(parsed.spline_terms[0].df, 5);
        assert_eq!(parsed.spline_terms[1].var_name, "income");
        assert_eq!(parsed.spline_terms[1].spline_type, "ns");
        assert_eq!(parsed.spline_terms[1].df, 4);
    }

    #[test]
    fn test_no_intercept() {
        let parsed = parse_formula("y ~ 0 + x1 + x2").unwrap();
        assert!(!parsed.has_intercept);
        
        let parsed2 = parse_formula("y ~ x1 + x2 - 1").unwrap();
        assert!(!parsed2.has_intercept);
    }

    #[test]
    fn test_complex_formula() {
        let parsed = parse_formula("y ~ bs(age, df=5) + C(region)*income + x1:x2").unwrap();
        assert_eq!(parsed.response, "y");
        assert_eq!(parsed.spline_terms.len(), 1);
        assert!(parsed.categorical_vars.contains("region"));
        assert_eq!(parsed.main_effects, vec!["region", "income"]);
        assert_eq!(parsed.interactions.len(), 2); // region*income and x1:x2
    }

    #[test]
    fn test_parse_target_encoding() {
        let parsed = parse_formula("y ~ TE(brand) + age").unwrap();
        assert_eq!(parsed.target_encoding_terms.len(), 1);
        assert_eq!(parsed.target_encoding_terms[0].var_name, "brand");
        assert!((parsed.target_encoding_terms[0].prior_weight - 1.0).abs() < 1e-10);
        assert_eq!(parsed.target_encoding_terms[0].n_permutations, 4);
        assert_eq!(parsed.main_effects, vec!["age"]);
    }

    #[test]
    fn test_parse_target_encoding_with_options() {
        let parsed = parse_formula("y ~ TE(brand, prior_weight=2.0, n_permutations=8)").unwrap();
        assert_eq!(parsed.target_encoding_terms.len(), 1);
        assert_eq!(parsed.target_encoding_terms[0].var_name, "brand");
        assert!((parsed.target_encoding_terms[0].prior_weight - 2.0).abs() < 1e-10);
        assert_eq!(parsed.target_encoding_terms[0].n_permutations, 8);
    }

    #[test]
    fn test_parse_formula_with_te_and_splines() {
        let parsed = parse_formula("y ~ TE(brand) + bs(age, df=5) + C(region)").unwrap();
        assert_eq!(parsed.target_encoding_terms.len(), 1);
        assert_eq!(parsed.spline_terms.len(), 1);
        assert!(parsed.categorical_vars.contains("region"));
        assert_eq!(parsed.main_effects, vec!["region"]);
    }

    #[test]
    fn test_unsupported_function_error() {
        // poly() is an R function - not supported
        let result = parse_formula("y ~ poly(age, 2) + x");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Unsupported function 'poly()'"));
        
        // log() transformation - not supported in formula syntax
        let result = parse_formula("y ~ log(income) + age");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Unsupported function 'log()'"));
    }

    #[test]
    fn test_parse_identity_term() {
        // I() for polynomial terms
        let parsed = parse_formula("y ~ I(x ** 2) + age").unwrap();
        assert_eq!(parsed.identity_terms.len(), 1);
        assert_eq!(parsed.identity_terms[0].expression, "x ** 2");
        assert_eq!(parsed.main_effects, vec!["age"]);
        
        // Multiple I() terms
        let parsed = parse_formula("y ~ I(x ** 2) + I(x ** 3) + z").unwrap();
        assert_eq!(parsed.identity_terms.len(), 2);
        assert_eq!(parsed.identity_terms[0].expression, "x ** 2");
        assert_eq!(parsed.identity_terms[1].expression, "x ** 3");
        
        // I() with addition inside (protected from formula interpretation)
        let parsed = parse_formula("y ~ I(a + b)").unwrap();
        assert_eq!(parsed.identity_terms.len(), 1);
        assert_eq!(parsed.identity_terms[0].expression, "a + b");
        
        // I() combined with other term types
        let parsed = parse_formula("y ~ I(DrivAge ** 2) + bs(VehAge, df=4) + C(Region)").unwrap();
        assert_eq!(parsed.identity_terms.len(), 1);
        assert_eq!(parsed.identity_terms[0].expression, "DrivAge ** 2");
        assert_eq!(parsed.spline_terms.len(), 1);
        assert!(parsed.categorical_vars.contains("Region"));
    }

    #[test]
    fn test_parse_categorical_with_level() {
        // Single level indicator: C(Region, level='Paris')
        let parsed = parse_formula("y ~ C(Region, level='Paris') + age").unwrap();
        assert_eq!(parsed.categorical_terms.len(), 1);
        assert_eq!(parsed.categorical_terms[0].var_name, "Region");
        assert_eq!(parsed.categorical_terms[0].levels, Some(vec!["Paris".to_string()]));
        assert_eq!(parsed.main_effects, vec!["age"]);
        // Should NOT be in main_effects
        assert!(!parsed.main_effects.contains(&"Region".to_string()));
        
        // Multiple levels: C(Region, levels=['Paris', 'Lyon'])
        let parsed = parse_formula("y ~ C(Region, levels=['Paris', 'Lyon'])").unwrap();
        assert_eq!(parsed.categorical_terms.len(), 1);
        assert_eq!(parsed.categorical_terms[0].var_name, "Region");
        assert_eq!(parsed.categorical_terms[0].levels, Some(vec!["Paris".to_string(), "Lyon".to_string()]));
        
        // Regular C() should still work and go to main_effects
        let parsed = parse_formula("y ~ C(Region) + age").unwrap();
        assert!(parsed.categorical_terms.is_empty());
        assert!(parsed.main_effects.contains(&"Region".to_string()));
        
        // Double quotes should also work
        let parsed = parse_formula("y ~ C(Region, level=\"Paris\")").unwrap();
        assert_eq!(parsed.categorical_terms[0].levels, Some(vec!["Paris".to_string()]));
    }
}
