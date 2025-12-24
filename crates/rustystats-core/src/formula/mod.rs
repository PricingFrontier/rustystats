//! Formula parsing for R-style model specifications.
//!
//! This module parses formulas like "y ~ x1*x2 + C(cat) + bs(age, df=5)"
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
    pub spline_terms: Vec<SplineTerm>,
    pub has_intercept: bool,
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
    let end = term.rfind(')')?;
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
                    if let Ok(v) = value.parse() {
                        df = v;
                    }
                }
                "degree" => {
                    if let Ok(v) = value.parse() {
                        degree = v;
                    }
                }
                _ => {}
            }
        } else if let Ok(v) = part.parse::<usize>() {
            // Positional argument assumed to be df
            df = v;
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
    
    for term in terms {
        // Check for spline term
        if let Some(spline) = parse_spline_term(&term) {
            spline_terms.push(spline);
            continue;
        }
        
        if term.contains('*') && !term.starts_with("bs(") && !term.starts_with("ns(") {
            // Full interaction: a*b = a + b + a:b
            let factor_strs: Vec<&str> = term.split('*').collect();
            
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
        } else if term.contains(':') && !term.starts_with("bs(") && !term.starts_with("ns(") {
            // Pure interaction: a:b (no main effects)
            let factor_strs: Vec<&str> = term.split(':').collect();
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
        spline_terms,
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
}
