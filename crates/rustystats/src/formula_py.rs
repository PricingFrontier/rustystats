// =============================================================================
// Formula Parsing (Python Bindings)
// =============================================================================
//
// Wraps the Rust formula parser for Python. Returns parsed formula components
// as Python dicts.
// =============================================================================

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use rustystats_core::formula;

/// Parse a formula string into structured components.
///
/// Parameters
/// ----------
/// formula_str : str
///     R-style formula like "y ~ x1*x2 + C(cat) + bs(age, df=5)"
///
/// Returns
/// -------
/// dict
///     Parsed formula with keys:
///     - response: str
///     - main_effects: list[str]
///     - interactions: list[dict] with 'factors' and 'categorical_flags'
///     - categorical_vars: list[str]
///     - spline_terms: list[dict] with 'var_name', 'spline_type', 'df', 'degree', 'increasing'
///     - target_encoding_terms: list[dict] with 'var_name', 'prior_weight', 'n_permutations'
///     - identity_terms: list[dict] with 'expression'
///     - has_intercept: bool
#[pyfunction]
pub fn parse_formula_py(formula_str: &str) -> PyResult<std::collections::HashMap<String, pyo3::PyObject>> {
    use pyo3::types::PyDict;
    
    let parsed = formula::parse_formula(formula_str)
        .map_err(|e| PyValueError::new_err(e))?;
    
    Python::with_gil(|py| {
        let mut result = std::collections::HashMap::new();
        
        result.insert("response".to_string(), parsed.response.into_py(py));
        result.insert("main_effects".to_string(), parsed.main_effects.into_py(py));
        result.insert("has_intercept".to_string(), parsed.has_intercept.into_py(py));
        result.insert("categorical_vars".to_string(), 
            parsed.categorical_vars.into_iter().collect::<Vec<_>>().into_py(py));
        
        // Convert interactions
        let interactions: Vec<_> = parsed.interactions
            .into_iter()
            .map(|i| {
                let dict = PyDict::new_bound(py);
                dict.set_item("factors", i.factors).unwrap();
                dict.set_item("categorical_flags", i.categorical_flags).unwrap();
                dict.into_py(py)
            })
            .collect();
        result.insert("interactions".to_string(), interactions.into_py(py));
        
        // Convert spline terms
        let splines: Vec<_> = parsed.spline_terms
            .into_iter()
            .map(|s| {
                let dict = PyDict::new_bound(py);
                dict.set_item("var_name", s.var_name).unwrap();
                dict.set_item("spline_type", s.spline_type).unwrap();
                dict.set_item("df", s.df).unwrap();
                dict.set_item("degree", s.degree).unwrap();
                dict.set_item("increasing", s.increasing).unwrap();
                dict.set_item("monotonic", s.monotonic).unwrap();
                dict.set_item("is_smooth", s.is_smooth).unwrap();
                if s.monotonic {
                    dict.set_item("monotonicity", if s.increasing { "increasing" } else { "decreasing" }).unwrap();
                } else {
                    dict.set_item("monotonicity", py.None()).unwrap();
                }
                dict.into_py(py)
            })
            .collect();
        result.insert("spline_terms".to_string(), splines.into_py(py));
        
        // Convert target encoding terms
        let te_terms: Vec<_> = parsed.target_encoding_terms
            .into_iter()
            .map(|t| {
                let dict = PyDict::new_bound(py);
                dict.set_item("var_name", t.var_name).unwrap();
                dict.set_item("prior_weight", t.prior_weight).unwrap();
                dict.set_item("n_permutations", t.n_permutations).unwrap();
                dict.set_item("interaction_vars", t.interaction_vars).unwrap();
                dict.into_py(py)
            })
            .collect();
        result.insert("target_encoding_terms".to_string(), te_terms.into_py(py));
        
        // Convert frequency encoding terms
        let fe_terms: Vec<_> = parsed.frequency_encoding_terms
            .into_iter()
            .map(|t| {
                let dict = PyDict::new_bound(py);
                dict.set_item("var_name", t.var_name).unwrap();
                dict.into_py(py)
            })
            .collect();
        result.insert("frequency_encoding_terms".to_string(), fe_terms.into_py(py));
        
        // Convert identity terms (I() expressions)
        let identity_terms: Vec<_> = parsed.identity_terms
            .into_iter()
            .map(|i| {
                let dict = PyDict::new_bound(py);
                dict.set_item("expression", i.expression).unwrap();
                dict.into_py(py)
            })
            .collect();
        result.insert("identity_terms".to_string(), identity_terms.into_py(py));
        
        // Convert categorical terms with level selection (C(var, level='...'))
        let categorical_terms: Vec<_> = parsed.categorical_terms
            .into_iter()
            .map(|c| {
                let dict = PyDict::new_bound(py);
                dict.set_item("var_name", c.var_name).unwrap();
                dict.set_item("levels", c.levels).unwrap();
                dict.into_py(py)
            })
            .collect();
        result.insert("categorical_terms".to_string(), categorical_terms.into_py(py));
        
        // Convert constraint terms (pos() / neg())
        let constraint_terms: Vec<_> = parsed.constraint_terms
            .into_iter()
            .map(|c| {
                let dict = PyDict::new_bound(py);
                dict.set_item("var_name", c.var_name).unwrap();
                dict.set_item("constraint", c.constraint).unwrap();
                dict.into_py(py)
            })
            .collect();
        result.insert("constraint_terms".to_string(), constraint_terms.into_py(py));
        
        Ok(result)
    })
}
